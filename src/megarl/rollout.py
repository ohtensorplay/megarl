"""Rollout adapters.

Contract: a rollout service emits `TrajectoryRecord`s whose `logprobs` are the
*processed* logprobs of the policy that actually generated each token, and whose
`policy_versions` are stamped per token. The latter is what makes partial
rollout and in-flight weight updates mathematically usable.

megaLM's embedded `Engine` is the default backend. We consume `Engine.step()`
events directly: `generate_chat`/`run_until_complete` drop logprobs, and
speculative decoding cannot supply them, so both are deliberately avoided.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol

from .contracts import RolloutRequest, SamplingParams, TrajectoryRecord

VersionProvider = Callable[[], int]


class RolloutService(Protocol):
    def generate(self, request: RolloutRequest) -> Iterator[TrajectoryRecord]: ...

    def reload_weights(self, path: str, expected_digest: str | None = None) -> None: ...


class RolloutError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EmbeddedRolloutConfig:
    model_path: Path
    max_new_tokens: int = 512
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_num_sequences: int = 16
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.9
    enable_prefix_cache: bool = True
    enable_cuda_graph: bool = True


class EmbeddedRollout:
    """In-process rollout on megaLM's `Engine`.

    `version_provider` is queried before each decode step, so a policy update
    that lands mid-generation is reflected in the token versions rather than
    mislabelling the whole trajectory as one version.
    """

    def __init__(
        self,
        config: EmbeddedRolloutConfig,
        *,
        version_provider: VersionProvider | None = None,
    ) -> None:
        self.config = config
        self._version_provider = version_provider or (lambda: 0)
        self._engine = None
        self._megasampling_cls = None

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        from megalm.contracts.request import SamplingParams as MegaSamplingParams
        from megalm.runtime.engine import Engine, EngineConfig

        cfg = EngineConfig(
            model_path=self.config.model_path,
            device=self.config.device,
            dtype=self.config.dtype,
            max_num_sequences=self.config.max_num_sequences,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enable_prefix_cache=self.config.enable_prefix_cache,
            enable_cuda_graph=self.config.enable_cuda_graph,
            speculative_mode="off",  # speculative completion has no logprobs
        )
        self._megasampling_cls = MegaSamplingParams
        self._engine = Engine(cfg)
        return self._engine

    def _sampling(self, sampling: SamplingParams):
        return self._megasampling_cls(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            seed=sampling.seed,
            logprobs=True,  # never optional: importance ratios require them
            top_logprobs=sampling.top_logprobs,
        )

    def generate(self, request: RolloutRequest) -> Iterator[TrajectoryRecord]:
        engine = self._ensure_engine()
        max_new = min(request.max_new_tokens, self.config.max_new_tokens)
        sampling = self._sampling(request.sampling)
        fingerprint = str(request.request_id)

        pending: dict[str, dict] = {}
        for i in range(max(1, request.n)):
            rid = f"{request.request_id}:{i}"
            engine.add_request(
                list(request.prompt_token_ids),
                max_new_tokens=max_new,
                sampling=sampling,
                request_id=rid,
                prefix_fingerprint=fingerprint,  # n samples share the prefix
            )
            pending[rid] = {
                "token_ids": [],
                "logprobs": [],
                "top_logprobs": [],
                "versions": [],
            }

        # Bound the loop so a stuck request cannot hang the controller forever.
        max_steps = max_new * max(1, request.n) * 4 + 64
        for _ in range(max_steps):
            if not pending:
                return
            events = engine.step()
            if not events:
                raise RolloutError("engine.step() produced no events with live requests")
            for event in events:
                rid = event.request_id
                if rid not in pending:
                    continue
                state = pending[rid]
                kind = str(getattr(event.kind, "value", event.kind))
                if kind == "token":
                    tokens = list(event.token_ids)
                    logps = list(event.logprobs)
                    tops = [
                        [tuple(pair) for pair in row] for row in event.top_logprobs
                    ]
                    if not tokens:
                        continue
                    if len(logps) != len(tokens):
                        raise RolloutError(
                            f"{rid}: logprobs/token_ids misaligned "
                            f"({len(logps)} vs {len(tokens)}); refusing to emit "
                            "a record with an invalid importance ratio"
                        )
                    if tops and len(tops) != len(tokens):
                        raise RolloutError(f"{rid}: top_logprobs misaligned")
                    state["token_ids"].extend(tokens)
                    state["logprobs"].extend(logps)
                    state["top_logprobs"].extend(tops)
                    # One version per token: a chunk event may carry several
                    # tokens produced across an in-flight weight update.
                    state["versions"].extend(
                        self._version_provider() for _ in tokens
                    )
                elif kind == "terminal":
                    yield TrajectoryRecord(
                        request_id=request.request_id,
                        prompt_len=len(request.prompt_token_ids),
                        token_ids=state["token_ids"],
                        logprobs=state["logprobs"],
                        top_logprobs=state["top_logprobs"],
                        policy_versions=state["versions"],
                        reward=None,  # filled by the reward layer
                        finish_reason=int(
                            getattr(event.finish_reason, "value", 0)
                            if event.finish_reason is not None
                            else 0
                        ),
                        payload_handle=None,
                    )
                    del pending[rid]
        if pending:
            raise RolloutError(f"rollout did not finish: {sorted(pending)}")

    def reload_weights(self, path: str, expected_digest: str | None = None) -> None:
        """Reload a full safetensors checkpoint in place.

        The delta weight fabric supersedes this for frequent policy updates.
        megaLM currently returns the startup digest from `update_weights`, so
        `expected_digest` cannot be verified here yet (see architecture §3.3).
        """
        engine = self._ensure_engine()
        engine.update_weights(path)
