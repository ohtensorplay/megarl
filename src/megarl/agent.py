"""Multi-turn agentic RL loop (token-in-token-out).

The loop is the atomic rollout unit for agentic training. For every turn it
sends the *exact* tokens it has recorded to the rollout service, appends the
generated tokens with their sampled logprobs and per-token versions, executes
tool calls between turns, and appends tool observations with ``loss_mask=0``.
It never re-derives logprobs from text: doing so would train on actions that
were not sampled from the behavior policy (the repeated failure mode in the
pinned AReaL / ROLL / slime / SkyRL / trl references).

If a caller supplies a ``prompt_provider`` that re-renders history (e.g. an
external harness with a chat template), :func:`reconcile_prefix` classifies the
drift as CLEAN / REALIGN / FORK and the loop handles it explicitly instead of
silently inheriting stale logprobs.

The resulting :class:`~megarl.contracts.AgentTrajectory` plugs into the normal
controller/staleness/reward/loss pipeline unchanged.
"""

from __future__ import annotations

import dataclasses
import enum
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from . import contracts
from .batching import AgentRecord
from .contracts import (
    AgentToolCall,
    AgentTrajectory,
    RolloutRequest,
    SamplingParams,
    TrajectoryRecord,
)
from .controller import Controller, ControllerConfig, Prompt, Trainer
from .async_controller import AsyncController, AsyncControllerConfig
from .observability import Telemetry
from .rollout import RolloutError, RolloutService
from .tools import ParsedToolCall, ToolError, ToolRegistry, ToolResult, UnknownTool


class Finish(str, enum.Enum):
    STOP = "stop"
    MAX_TURNS = "max_turns"
    MAX_TOKENS = "max_tokens"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"


_FINISH_CODES = {
    Finish.STOP: 0,
    Finish.MAX_TURNS: 1,
    Finish.MAX_TOKENS: 2,
    Finish.TOOL_ERROR: 3,
    Finish.TIMEOUT: 4,
}


class DriftKind(str, enum.Enum):
    CLEAN = "clean"
    REALIGN = "realign"
    FORK = "fork"


@dataclasses.dataclass(frozen=True, slots=True)
class Reconciliation:
    kind: DriftKind
    common_prefix: int

    @property
    def drifted_tokens(self) -> int:
        return self.common_prefix


def reconcile_prefix(
    recorded: Sequence[int], incoming: Sequence[int], *, fork_threshold: int = 1024
) -> Reconciliation:
    """Classify a re-rendered prompt against the tokens actually sampled.

    ``CLEAN`` — the incoming prompt preserves the recorded prefix; append the
    tail as conditioning tokens. ``REALIGN`` — a short span diverged; the loop
    zeroes loss on the affected assistant tokens and keeps the sampled
    logprobs. ``FORK`` — the divergence is large; the loop drops the old branch
    and continues from the common prefix (no stale logprobs are inherited).
    """
    if fork_threshold < 0:
        raise ValueError("fork_threshold must be >= 0")
    common = 0
    for left, right in zip(recorded, incoming):
        if left != right:
            break
        common += 1
    if common >= len(recorded):
        return Reconciliation(DriftKind.CLEAN, common)
    kind = DriftKind.REALIGN if len(recorded) - common <= fork_threshold else DriftKind.FORK
    return Reconciliation(kind, common)


@dataclasses.dataclass(frozen=True, slots=True)
class TurnRecord:
    """One assistant generation, the atomic unit shared with external agents."""

    turn: int
    prompt_ids: list[int]
    output_ids: list[int]
    output_logprobs: list[float]
    output_versions: list[int]
    finish_reason: int
    tool_calls: tuple[ParsedToolCall, ...] = ()
    sampling_logprobs: list[float] = dataclasses.field(default_factory=list)
    top_logprobs: list[list[tuple[int, float]]] = dataclasses.field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.output_ids)


@dataclasses.dataclass(frozen=True, slots=True)
class AgentConfig:
    max_turns: int = 8
    max_new_tokens_per_turn: int = 128
    max_total_tokens: int = 4096
    wall_clock_secs: float = 120.0
    tool_format: str = "auto"
    strict: bool = True
    max_tool_result_chars: int = 4096
    turn_discount: float = 1.0
    session_affinity: bool = True
    fork_threshold: int = 1024

    def validate(self) -> None:
        if self.max_turns < 1 or self.max_new_tokens_per_turn < 1:
            raise ValueError("max_turns and max_new_tokens_per_turn must be positive")
        if self.max_total_tokens < 1 or self.wall_clock_secs <= 0:
            raise ValueError("max_total_tokens and wall_clock_secs must be positive")
        if not 0.0 < self.turn_discount <= 1.0:
            raise ValueError("turn_discount must be in (0, 1]")
        if self.tool_format not in ("auto", "json", "xml"):
            raise ValueError("tool_format must be auto, json or xml")


def discounted_step_rewards(
    terminal: float, turns: int, discount: float
) -> list[float]:
    """Broadcast a terminal reward to every turn with backward discount.

    ``discount=1`` is the GRPO-style broadcast used by most refs; smaller
    values implement AReaL-style turn discount and rllm-style MC returns.
    """
    return [float(terminal) * discount ** (turns - 1 - i) for i in range(turns)]


class Tokenizer(Protocol):
    """Minimal encode/decode contract; real deployments pass their HF tokenizer."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


@dataclasses.dataclass(frozen=True, slots=True)
class ByteTokenizer:
    """Reversible UTF-8 byte tokenizer for CPU tests and fallback use.

    Production models must pass their own tokenizer: byte ids are not the
    model's vocabulary. It is provided so an agent loop can be exercised
    end-to-end without a GPU or a model download.
    """

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:
        return bytes(int(t) & 0xFF for t in tokens).decode("utf-8", errors="replace")


class AgentLoop:
    """Run a bounded multi-turn episode and emit an :class:`AgentTrajectory`."""

    def __init__(
        self,
        rollout: RolloutService,
        tokenizer: Any,
        registry: ToolRegistry | None = None,
        *,
        config: AgentConfig | None = None,
        telemetry: Telemetry | None = None,
        version_provider: Callable[[], int] | None = None,
        prompt_provider: Callable[[Sequence[TurnRecord], Sequence[int]], Sequence[int]] | None = None,
        observation_renderer: Callable[
            [Sequence[ParsedToolCall], Sequence[Any]], str
        ]
        | None = None,
        validate: bool = True,
    ) -> None:
        config = config or AgentConfig()
        config.validate()
        self.rollout = rollout
        self.tokenizer = tokenizer
        self.registry = registry
        self.config = config
        self.telemetry = telemetry
        self.version_provider = version_provider or (lambda: 0)
        self.prompt_provider = prompt_provider
        self.observation_renderer = observation_renderer or self._default_observation
        self.validate = validate

    @staticmethod
    def _default_observation(
        calls: Sequence[ParsedToolCall], results: Sequence[Any]
    ) -> str:
        parts: list[str] = []
        for call, result in zip(calls, results):
            status = "ok" if result.ok else f"error: {result.error}"
            parts.append(
                f'<tool_result call_id="{call.id}" name="{call.name}" '
                f'status="{status}">\n{result.content}\n</tool_result>'
            )
        return "\n".join(parts)

    def _emit(self, event: str, **fields: Any) -> None:
        if self.telemetry is not None:
            self.telemetry.emit(event, **fields)

    def run(
        self,
        prompt: Prompt | Sequence[int],
        *,
        request_id: int | None = None,
        policy_version: int | None = None,
        reward_fn: Callable[[AgentTrajectory], Any] | None = None,
    ) -> AgentTrajectory:
        """Execute one episode. Reward is optional here; the controller scores
        trajectories when ``reward_fn`` is left to :func:`score_group`."""
        prompt_ids = list(prompt.token_ids) if isinstance(prompt, Prompt) else list(prompt)
        if not prompt_ids:
            raise ValueError("agent episode needs a non-empty prompt")
        rid = (
            request_id
            if request_id is not None
            else (prompt.request_id if isinstance(prompt, Prompt) else 0)
        )
        version = int(
            policy_version if policy_version is not None else self.version_provider()
        )
        started = time.monotonic()

        # Full-sequence provenance arrays.
        context = prompt_ids[:]
        prompt_len = len(prompt_ids)
        roles = [0] * prompt_len
        turns = [0] * prompt_len
        train = [0] * prompt_len
        # Response-aligned arrays: token_ids[prompt_len:] (includes tool tokens).
        logps: list[float] = []
        versions: list[int] = []
        sampling: list[float] = []
        tops: list[list[tuple[int, float]]] = []
        has_sampling = False
        has_top = False

        records: list[TurnRecord] = []
        calls_log: list[AgentToolCall] = []
        finish = Finish.STOP

        for turn in range(self.config.max_turns):
            if len(context) - prompt_len >= self.config.max_total_tokens:
                finish = Finish.MAX_TOKENS
                break
            if time.monotonic() - started > self.config.wall_clock_secs:
                finish = Finish.TIMEOUT
                break

            if self.prompt_provider is not None and turn > 0:
                rendered = [
                    int(t) for t in self.prompt_provider(list(records), list(context))
                ]
                recon = reconcile_prefix(context, rendered, fork_threshold=self.config.fork_threshold)
                if recon.kind is DriftKind.REALIGN and len(rendered) < len(context):
                    # A shortened re-render cannot preserve response alignment.
                    recon = Reconciliation(DriftKind.FORK, recon.common_prefix)
                if recon.kind is DriftKind.REALIGN:
                    for i in range(recon.common_prefix, len(context)):
                        if roles[i] == 1:
                            train[i] = 0
                    # Use the re-rendered tokens (sampled logprobs stay attached
                    # to their positions but those positions are no longer
                    # trainable, so no stale action is trained on).
                    context[:] = rendered[: len(context)]
                    self._emit(
                        "agent_realign",
                        request_id=rid,
                        turn=turn,
                        drifted_tokens=len(context) - recon.common_prefix,
                    )
                elif recon.kind is DriftKind.FORK:
                    keep = max(recon.common_prefix, 0)
                    context, roles, turns, train = (
                        context[:keep],
                        roles[:keep],
                        turns[:keep],
                        train[:keep],
                    )
                    prompt_len = keep
                    logps, versions, sampling, tops = [], [], [], []
                    has_sampling = has_top = False
                    self._emit(
                        "agent_fork",
                        request_id=rid,
                        turn=turn,
                        common_prefix=keep,
                    )
                extra = rendered[len(context) :]
                context.extend(extra)
                roles.extend([0] * len(extra))
                turns.extend([turn] * len(extra))
                train.extend([0] * len(extra))
                logps.extend([0.0] * len(extra))
                versions.extend([0] * len(extra))
                if has_sampling:
                    sampling.extend([0.0] * len(extra))
                if has_top:
                    tops.extend([[] for _ in extra])

            remaining = self.config.max_total_tokens - (len(context) - prompt_len)
            max_new = min(self.config.max_new_tokens_per_turn, remaining)
            if max_new <= 0:
                finish = Finish.MAX_TOKENS
                break
            request = RolloutRequest(
                request_id=rid,
                policy_version=version,
                prompt_token_ids=list(context),
                max_new_tokens=max_new,
                n=1,
                sampling=SamplingParams(1.0, 1.0, 0, None, True, 0),
                session_id=rid if self.config.session_affinity else None,
            )
            generated = list(self.rollout.generate(request))
            if len(generated) != 1:
                raise RolloutError(
                    f"agent turn {turn}: expected 1 trajectory, got {len(generated)}"
                )
            record: TrajectoryRecord = generated[0]
            output = list(record.token_ids)
            if not output:
                finish = Finish.STOP
                break
            if len(record.logprobs) != len(output):
                raise RolloutError(
                    f"agent turn {turn}: logprobs/tokens misaligned "
                    f"({len(record.logprobs)} vs {len(output)})"
                )
            if len(record.policy_versions) != len(output):
                raise RolloutError(
                    f"agent turn {turn}: versions/tokens misaligned"
                )

            prompt_at_turn = len(context)
            context.extend(output)
            roles.extend([1] * len(output))
            turns.extend([turn] * len(output))
            train.extend([1] * len(output))
            logps.extend(float(v) for v in record.logprobs)
            versions.extend(int(v) for v in record.policy_versions)
            if record.sampling_logprobs:
                has_sampling = True
                if len(record.sampling_logprobs) != len(output):
                    raise RolloutError("sampling_logprobs/tokens misaligned")
                sampling.extend(float(v) for v in record.sampling_logprobs)
            elif has_sampling:
                sampling.extend([0.0] * len(output))
            if record.top_logprobs:
                has_top = True
                tops.extend(
                    [[(int(t), float(p)) for t, p in row] for row in record.top_logprobs]
                )
            elif has_top:
                tops.extend([[] for _ in output])

            text = self.tokenizer.decode(output)
            try:
                parsed = ParsedToolCall.parse(text, self.config.tool_format)
            except (ToolError, ValueError) as exc:
                self._emit("agent_tool_parse_error", request_id=rid, turn=turn, error=str(exc))
                finish = Finish.TOOL_ERROR
                break
            parsed = [
                dataclasses.replace(call, id=call.id or f"{rid}:{turn}:{i}")
                for i, call in enumerate(parsed)
            ]
            records.append(
                TurnRecord(
                    turn=turn,
                    prompt_ids=context[:prompt_at_turn],
                    output_ids=output,
                    output_logprobs=list(record.logprobs),
                    output_versions=list(record.policy_versions),
                    finish_reason=int(record.finish_reason),
                    tool_calls=tuple(parsed),
                    sampling_logprobs=list(record.sampling_logprobs),
                    top_logprobs=[
                        [(int(t), float(p)) for t, p in row] for row in record.top_logprobs
                    ],
                )
            )
            self._emit(
                "agent_turn",
                request_id=rid,
                turn=turn,
                tokens=len(output),
                tool_calls=len(parsed),
            )
            if not parsed:
                finish = Finish.STOP
                break
            if self.registry is None:
                self._emit("agent_tool_error", request_id=rid, turn=turn, error="no tool registry")
                if self.config.strict:
                    finish = Finish.TOOL_ERROR
                    break
                results = [
                    ToolResult(call.id, call.name, False, "", "no tool registry", 0)
                    for call in parsed
                ]
            else:
                try:
                    results = self.registry.execute(parsed)
                except (UnknownTool, ToolError, ValueError) as exc:
                    # Validation is all-or-nothing: nothing executed, so no
                    # tokens from an invalid tool turn enter the trajectory.
                    self._emit(
                        "agent_tool_error", request_id=rid, turn=turn, error=str(exc)
                    )
                    if self.config.strict:
                        finish = Finish.TOOL_ERROR
                        break
                    results = [
                        ToolResult(call.id, call.name, False, "", str(exc), 0)
                        for call in parsed
                    ]
            for call, result in zip(parsed, results):
                calls_log.append(
                    AgentToolCall(
                        id=call.id,
                        name=call.name,
                        arguments_json=call.arguments_json,
                        turn=turn,
                        ok=bool(result.ok),
                        elapsed_ms=int(result.elapsed_ms),
                    )
                )
            observation = self.observation_renderer(parsed, results)
            obs_ids = self.tokenizer.encode(observation)
            if len(obs_ids) > self.config.max_tool_result_chars:
                obs_ids = obs_ids[: self.config.max_tool_result_chars]
            if obs_ids:
                context.extend(obs_ids)
                roles.extend([2] * len(obs_ids))
                turns.extend([turn + 1] * len(obs_ids))
                train.extend([0] * len(obs_ids))
                logps.extend([0.0] * len(obs_ids))
                versions.extend([0] * len(obs_ids))
                if has_sampling:
                    sampling.extend([0.0] * len(obs_ids))
                if has_top:
                    tops.extend([[] for _ in obs_ids])
            self._emit(
                "agent_observations",
                request_id=rid,
                turn=turn,
                tokens=len(obs_ids),
                failures=sum(not r.ok for r in results),
            )
        else:
            finish = Finish.MAX_TURNS

        trajectory = AgentTrajectory(
            request_id=rid,
            policy_version=version,
            prompt_len=prompt_len,
            token_ids=context,
            logprobs=logps,
            top_logprobs=tops if has_top else [],
            policy_versions=versions,
            token_roles=roles,
            turn_ids=turns,
            loss_mask=train,
            reward=None,
            finish_reason=_FINISH_CODES[finish],
            payload_handle=None,
            sampling_logprobs=sampling if has_sampling else [],
            tool_calls=calls_log,
            step_rewards=[],
        )
        if reward_fn is not None:
            reward = reward_fn(trajectory)
            if isinstance(reward, (list, tuple)):
                steps = [float(value) for value in reward]
                terminal = steps[-1] if steps else 0.0
            else:
                terminal = float(reward)
                steps = discounted_step_rewards(
                    terminal, len(records), self.config.turn_discount
                )
            trajectory = dataclasses.replace(
                trajectory, reward=terminal, step_rewards=steps
            )
        if self.validate:
            contracts.validate(trajectory)  # Rust-side invariant check
        self._emit(
            "agent_finish",
            request_id=rid,
            finish=finish.value,
            turns=len(records),
            trainable_tokens=sum(train),
            total_tokens=len(context),
            wall_ms=int((time.monotonic() - started) * 1000),
        )
        return trajectory

    def close(self) -> None:
        """Release the tool registry's worker threads."""
        if self.registry is not None:
            self.registry.close()

    def __enter__(self) -> AgentLoop:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AgentRollout:
    """Adapter: an :class:`AgentConfiguration`-driven loop as a rollout service.

    It satisfies the structural ``RolloutService`` interface so the ordinary
    controllers need no agent-specific branch. Each of the ``n`` samples of a
    request is a separate episode sharing the request id (GRPO group). Records
    are wrapped in :class:`~megarl.batching.AgentRecord` so the version ledger
    sees response-aligned versions, exactly as for single-turn trajectories.
    """

    def __init__(self, loop: AgentLoop, *, max_workers: int = 1) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.loop = loop
        self.max_workers = max_workers
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()

    def _executor(self) -> ThreadPoolExecutor | None:
        """Lazily create one shared pool; sequential mode uses none."""
        if self.max_workers <= 1:
            return None
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.max_workers, thread_name_prefix="megarl-agent"
                )
            return self._pool

    def generate(self, request: RolloutRequest) -> Iterator[AgentRecord]:
        n = max(1, request.n)

        def episode() -> AgentRecord:
            return AgentRecord(
                self.loop.run(
                    Prompt(request.request_id, list(request.prompt_token_ids)),
                    request_id=request.request_id,
                    policy_version=request.policy_version,
                )
            )

        pool = self._executor() if n > 1 else None
        if pool is None:
            for _ in range(n):
                yield episode()
        else:
            # Submit all episodes up front, then yield in submission order so
            # GRPO groups stay aligned with the controller's expectation.
            futures = [pool.submit(episode) for _ in range(n)]
            for future in futures:
                yield future.result()

    def reload_weights(self, path: str, expected_digest: str | None = None) -> None:
        reload = getattr(self.loop.rollout, "reload_weights", None)
        if reload is not None:
            reload(path, expected_digest)

    def close(self) -> None:
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            # wait=True: an in-flight episode must finish before we drop it.
            pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> AgentRollout:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AgenticController(Controller):
    """Convenience controller that wires an :class:`AgentLoop` into ``Controller``."""

    def __init__(
        self,
        config: ControllerConfig,
        *,
        rollout: RolloutService,
        tokenizer: Any,
        reward_fn: Callable[[AgentTrajectory], float],
        trainer: Trainer,
        registry: ToolRegistry | None = None,
        agent_config: AgentConfig | None = None,
        telemetry: Telemetry | None = None,
        weight_path: str | None = None,
        max_workers: int = 1,
        **loop_kwargs: Any,
    ) -> None:
        loop = AgentLoop(
            rollout,
            tokenizer,
            registry,
            config=agent_config,
            telemetry=telemetry,
            **loop_kwargs,
        )
        agent_rollout = AgentRollout(loop, max_workers=max_workers)
        super().__init__(
            config,
            rollout=agent_rollout,
            reward_fn=reward_fn,
            trainer=trainer,
            telemetry=telemetry,
            weight_path=weight_path,
        )
        self.agent_loop = loop
        self.agent_rollout = agent_rollout

    def close(self) -> None:
        self.agent_rollout.close()
        self.agent_loop.close()

    def __enter__(self) -> AgenticController:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncAgenticController(AsyncController):
    """Async counterpart; blocking tools already run inside ``asyncio.to_thread``."""

    def __init__(
        self,
        config: AsyncControllerConfig,
        *,
        rollout: RolloutService,
        tokenizer: Any,
        reward_fn: Callable[[AgentTrajectory], float],
        trainer: Trainer,
        registry: ToolRegistry | None = None,
        agent_config: AgentConfig | None = None,
        telemetry: Telemetry | None = None,
        weight_path: str | None = None,
        max_workers: int = 1,
        **loop_kwargs: Any,
    ) -> None:
        loop = AgentLoop(
            rollout,
            tokenizer,
            registry,
            config=agent_config,
            telemetry=telemetry,
            **loop_kwargs,
        )
        agent_rollout = AgentRollout(loop, max_workers=max_workers)
        super().__init__(
            config,
            rollout=agent_rollout,
            reward_fn=reward_fn,
            trainer=trainer,
            telemetry=telemetry,
            weight_path=weight_path,
        )
        self.agent_loop = loop
        self.agent_rollout = agent_rollout

    def close(self) -> None:
        self.agent_rollout.close()
        self.agent_loop.close()

    async def __aenter__(self) -> AsyncAgenticController:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()
