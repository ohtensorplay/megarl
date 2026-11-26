"""Fully asynchronous controller with a staleness budget and out-of-band queue.

Unlike `Controller`, this never waits on a single slow trajectory to admit the
next batch: generation runs in worker threads, results land in a metadata-only
queue with leases, and the trainer pulls token-budget batches independently.
`pause` implements the abort/wait/keep primitives needed for in-flight weight
updates and partial rollout.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
from collections import deque
from typing import Sequence

from .batching import (
    RLSample,
    collate_preference_pairs,
    from_trajectory,
    pack_with_order,
)
from .control import AsyncPacer, RateLimiter, StalenessBudget, VersionLedger, rate_limit
from .contracts import RolloutRequest, SamplingParams, TrajectoryRecord
from .dataplane import InProcessPayloadStore, LeasedPayloadQueue
from .controller import ControllerConfig, Prompt, StepResult, Trainer
from .controller import compute_advantages
from .dpo import build_preference_pairs
from .observability import Telemetry
from .reward import has_signal, score_group


class PauseMode(str, enum.Enum):
    ABORT = "abort"
    WAIT = "wait"
    KEEP = "keep"


@dataclasses.dataclass(frozen=True, slots=True)
class AsyncControllerConfig(ControllerConfig):
    max_concurrent_requests: int = 8
    # Batched paced wake-up for deferred prompts. The Rust AsyncPacer releases
    # at most `pacer_batch_size` waiters per tick and never accumulates more
    # than one batch of credits, so a burst cannot stampede the trainer.
    pacer_rate_per_sec: float = 1_000.0
    pacer_batch_size: int = 8


class PayloadQueue(LeasedPayloadQueue):
    """Metadata-only queue: records are encoded into a real payload store.

    ``store`` defaults to an in-process store for the embedded fast path; pass a
    ``SharedMemoryPayloadStore`` for a controller living in another process.
    """

    def __init__(self, ttl_secs: float = 300.0, *, store=None) -> None:
        super().__init__(store or InProcessPayloadStore(), ttl_secs=ttl_secs)


class AsyncController:
    def __init__(
        self,
        config: AsyncControllerConfig,
        *,
        rollout,
        reward_fn,
        trainer: Trainer,
        telemetry: Telemetry | None = None,
        weight_path: str | None = None,
        sample_builder=None,
        batch_packer=None,
        queue=None,
    ) -> None:
        self.config = config
        self.rollout = rollout
        self.reward_fn = reward_fn
        self.trainer = trainer
        # Same injection points as the synchronous Controller. dLLM installs a
        # record adapter and a canvas/trajectory collator here; AR defaults are
        # unchanged.
        self.sample_builder = sample_builder or from_trajectory
        self.batch_packer = batch_packer or pack_with_order
        self.telemetry = telemetry or Telemetry()
        self.weight_path = weight_path
        self.offpolicy_budget = (
            config.max_offpolicy_tokens
            if config.max_offpolicy_tokens is not None
            else config.max_staleness
        )
        self.budget = StalenessBudget(
            self.offpolicy_budget, config.consumer_batch_size, config.concurrency
        )
        self.ledger = VersionLedger()
        self.queue = queue if queue is not None else PayloadQueue(config.lease_ttl_secs)
        self.version = 0
        self.accepted_tokens = 0
        self.rate_limiter = (
            RateLimiter(config.trajectories_per_sec, config.rate_burst)
            if config.trajectories_per_sec
            else None
        )
        self._tasks: set[asyncio.Task] = set()
        self._paused: PauseMode | None = None

        # Deferred-admission waiter queue. `waiters` are still waiting for a
        # paced credit; `ready` were released by the pacer and are admitted
        # before newly arrived prompts on the next step.
        self.waiters: deque[Prompt] = deque()
        self._ready: deque[Prompt] = deque()
        self._waiter_cond = asyncio.Condition()
        self._pacer = (
            AsyncPacer(config.pacer_rate_per_sec, config.pacer_batch_size)
            if config.pacer_rate_per_sec > 0
            else None
        )

        # Rollout resume state, checkpointed with the trainer.
        self.prompt_cursor = 0
        self.model_version: str | None = None

    def pause(self, mode: PauseMode) -> None:
        """Signal the controller to stop admitting new work."""
        self._paused = mode
        self.telemetry.emit("pause", mode=mode.value, version=self.version)
        if mode is PauseMode.ABORT:
            for task in self._tasks:
                task.cancel()
            # An aborted in-flight batch must not be replayed after resume.
            self.waiters.clear()
            self._ready.clear()

    def resume(self) -> None:
        self._paused = None
        self.telemetry.emit("resume", version=self.version)

    def _pace_waiters(self) -> int:
        """Release a Rust-paced batch of waiters; returns the number woken."""
        if not self.waiters or self._pacer is None:
            return 0
        wake = self._pacer.wake(len(self.waiters))
        for _ in range(wake):
            self._ready.append(self.waiters.popleft())
        return wake

    async def wait_for_capacity(self, timeout: float | None = None) -> bool:
        """Wait until a paced waiter is ready, a slot frees, or `timeout` ends.

        Lets a driver block on admission instead of spinning on `step()` while
        still letting the Rust pacer decide when a batch may wake.
        """
        async with self._waiter_cond:
            if self._ready or len(self._tasks) < self.config.max_concurrent_requests:
                return True
            if timeout is None:
                await self._waiter_cond.wait()
                return True
            try:
                await asyncio.wait_for(self._waiter_cond.wait(), timeout)
                return True
            except asyncio.TimeoutError:
                return False

    def _notify_waiters(self) -> None:
        async def notify() -> None:
            async with self._waiter_cond:
                self._waiter_cond.notify_all()

        # Called from synchronous admission code; schedule only on a running
        # loop so the controller also works in plain sync contexts.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(notify())

    def restore_checkpoint(self, state: dict) -> None:
        """Restore admission counters after a checkpoint resume (AReaL rebase)."""
        self.version = int(state["version"])
        rebased = self.budget.rebase_after_recovery(self.version)
        self.accepted_tokens = max(int(state.get("accepted_tokens", 0)), rebased)
        self.prompt_cursor = max(int(state.get("prompt_cursor", 0)), 0)
        self.model_version = state.get("model_version")
        # Deferred work belongs to the old run; resume starts from the cursor.
        self.waiters.clear()
        self._ready.clear()
        set_version = getattr(self.trainer, "set_version", None)
        if set_version is not None:
            set_version(self.version)
        self.telemetry.emit(
            "checkpoint_restored",
            version=self.version,
            accepted_tokens=self.accepted_tokens,
            rebased_tokens=rebased,
            prompt_cursor=self.prompt_cursor,
            model_version=self.model_version,
        )

    def _admit(self, prompts: Sequence[Prompt]) -> list[Prompt]:
        if self._paused is not None:
            return []
        capacity = self.budget.capacity(self.version, self.accepted_tokens, 0)
        room = self.config.max_concurrent_requests - len(self._tasks)
        grant = rate_limit(self.rate_limiter, min(capacity, room), self.telemetry.emit)

        # Paced waiters have priority over arrivals so a deferred prompt is not
        # starved by a stream of fresh ones.
        admitted: list[Prompt] = []
        while grant > 0 and self._ready:
            admitted.append(self._ready.popleft())
            grant -= 1
        if grant > 0:
            admitted.extend(prompts[:grant])
            deferred = list(prompts[grant:])
        else:
            deferred = list(prompts)
        if deferred:
            self.waiters.extend(deferred)
            self.telemetry.emit(
                "admission_deferred",
                requested=len(prompts),
                queued=len(self.waiters),
                capacity=capacity,
            )
        if admitted:
            self.prompt_cursor = max(
                self.prompt_cursor, max(p.request_id for p in admitted) + 1
            )
        return admitted

    async def _one(self, prompt: Prompt) -> list[TrajectoryRecord]:
        request = RolloutRequest(
            request_id=prompt.request_id,
            policy_version=self.version,
            prompt_token_ids=prompt.token_ids,
            max_new_tokens=64,
            n=self.config.group_size,
            sampling=SamplingParams(
                self.config.temperature, self.config.top_p, 0, None, True, 0
            ),
            session_id=None,
        )

        def collect() -> list[TrajectoryRecord]:
            return list(self.rollout.generate(request))

        records = await asyncio.to_thread(collect)
        for record in records:
            self.ledger.record(
                record.request_id, len(record.token_ids), record.policy_versions
            )
            scored = score_group([record], self.reward_fn)[0]
            self.queue.put(scored)
        return records

    async def step(self, prompts: Sequence[Prompt]) -> StepResult | None:
        woken = self._pace_waiters()
        if woken:
            self.telemetry.emit("pacer_wake", count=woken, waiting=len(self.waiters))
        admitted = self._admit(prompts)
        if not admitted:
            if self.waiters and self._pacer is not None:
                # Back off by the Rust pacer's estimate instead of spinning.
                delay = min(self._pacer.retry_after_ms() / 1000.0, 0.05)
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)  # yield so in-flight work can progress
            return None
        for prompt in admitted:
            task = asyncio.create_task(self._one(prompt))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        done, _ = await asyncio.wait(
            self._tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                continue
        self._notify_waiters()

        # Drain completed payloads into zero-variance-filtered groups.
        by_request: dict[int, list[TrajectoryRecord]] = {}
        lease_ids = list(self.queue._handle_for)
        for lease_id in lease_ids:
            record = self.queue.take(lease_id)
            if record is not None:
                by_request.setdefault(record.request_id, []).append(record)

        dropped = sum(not has_signal(group) for group in by_request.values())
        self.telemetry.emit(
            "reward_group_summary",
            groups=len(by_request),
            dropped=dropped,
            version=self.version,
        )
        samples: list[RLSample] = []
        rewards: list[list[float]] = []
        group_samples: list[list[RLSample]] = []
        prompt_by_id = {p.request_id: p.token_ids for p in admitted}
        for rid, group in by_request.items():
            if not has_signal(group):
                continue
            rewards.append([float(r.reward) for r in group])
            members: list[RLSample] = []
            for record in group:
                prompt = prompt_by_id.get(rid)
                if prompt is None:
                    continue  # completed from an earlier step; already consumed
                member = self.sample_builder(record, prompt)
                members.append(member)
                samples.append(member)
            if len(members) >= 2:
                group_samples.append(members)

        stale = self.ledger.stale_tokens(self.budget, self.version)
        self.telemetry.emit("staleness", version=self.version, stale_tokens=stale)
        token_count = 0
        metrics: dict = {}
        if samples:
            set_version = getattr(self.trainer, "set_version", None)
            if set_version is not None:
                set_version(self.version)
            if self.config.dpo:
                pairs = build_preference_pairs(group_samples)
                dpo_update = getattr(self.trainer, "dpo_update", None)
                if dpo_update is None:
                    raise ValueError("dpo=True requires a trainer with dpo_update()")
                for start in range(0, len(pairs), self.config.consumer_batch_size):
                    chosen, rejected = collate_preference_pairs(
                        pairs[start : start + self.config.consumer_batch_size]
                    )
                    metrics = await asyncio.to_thread(
                        lambda c=chosen, r=rejected: self.trainer.dpo_update(c, r)
                    )
                    token_count += chosen.token_count + rejected.token_count
                    self.telemetry.emit(
                        "trainer_dpo_update", version=self.version, pairs=chosen.size
                    )
            else:
                advantages = compute_advantages(
                    rewards, self.config.advantage_estimator, self.config.scale_rewards
                )
                batches, orders = self.batch_packer(
                    samples,
                    max_tokens=self.config.max_tokens_per_batch,
                    alignment=self.config.alignment,
                )
                if len(advantages) != len(samples):
                    raise ValueError(
                        f"advantage/sample mismatch: {len(advantages)} vs {len(samples)}"
                    )
                for batch, order in zip(batches, orders):
                    batch_adv = [advantages[index] for index in order]
                    metrics = await asyncio.to_thread(
                        lambda b=batch, a=batch_adv: self.trainer.update(
                            b,
                            a,
                            clip_ratio=self.config.clip_ratio,
                            kl_coef=self.config.kl_coef,
                        )
                    )
                    token_count += batch.token_count
                    self.telemetry.emit(
                        "trainer_update", version=self.version, batch_size=batch.size
                    )
        self.accepted_tokens += token_count
        self.version += 1
        if self.weight_path is not None:
            await asyncio.to_thread(self.rollout.reload_weights, self.weight_path)
            self.model_version = self.weight_path
            self.telemetry.emit("weight_reload", version=self.version)
        # Checkpoint after the reload so the recorded model version is exactly
        # what the engine serves when this checkpoint is resumed.
        save = getattr(self.trainer, "save_checkpoint", None)
        every = self.config.checkpoint_every
        if save is not None and every > 0 and self.version % every == 0:
            save(
                f"version_{self.version}",
                version=self.version,
                accepted_tokens=self.accepted_tokens,
                reward_stats={"groups": len(rewards)},
                prompt_cursor=self.prompt_cursor,
                model_version=self.model_version,
            )
            self.telemetry.emit("checkpoint_saved", version=self.version)
        return StepResult(
            version=self.version,
            groups=len(by_request),
            dropped_groups=dropped,
            stale_tokens=stale,
            token_count=token_count,
            trainer_metrics=metrics,
        )

    async def run(self, prompts: Sequence[Prompt], steps: int) -> list[StepResult]:
        results: list[StepResult] = []
        for _ in range(steps):
            result = await self.step(prompts)
            if result is not None:
                results.append(result)
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        return results
