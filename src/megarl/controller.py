"""Single logical controller: versioned, staleness-budgeted async loop.

The loop is driven synchronously here for deterministic testing. The contracts
and ledger are async-ready (per-token versions, leases, metadata-only frames),
so a concurrent scheduler can replace `step` without changing the
rollout/reward/trainer interfaces.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, Sequence

from .batching import (
    RLBatch,
    RLSample,
    collate_preference_pairs,
    from_trajectory,
    pack_with_order,
)
from .control import LeaseTable, RateLimiter, StalenessBudget, VersionLedger, rate_limit
from .contracts import RolloutRequest, SamplingParams, TrajectoryRecord
from .dpo import build_preference_pairs
from . import advantages as _advantages


def compute_advantages(groups, estimator: str = "grpo", scale: str = "group"):
    """Select an advantage estimator, honoring GRPO reward scaling.

    Both the registry-based and function-based forms of ``megarl.advantages``
    are supported so the controller stays usable while that module evolves.
    """
    if estimator == "grpo":
        return _advantages.group_advantages(groups, scale_rewards=scale)
    if estimator == "rloo":
        # Compute here so singleton groups still emit one (zero) advantage and
        # stay aligned with the samples later fed to the trainer.
        out: list[float] = []
        for rewards in groups:
            if len(rewards) < 2:
                out.extend(0.0 for _ in rewards)
                continue
            total = sum(float(r) for r in rewards)
            n = len(rewards)
            out.extend(float(r) - (total - float(r)) / (n - 1) for r in rewards)
        return out
    registry = _advantages.ADVANTAGE_ESTIMATORS
    if isinstance(registry, dict):
        if estimator not in registry:
            raise ValueError(f"unknown advantage_estimator: {estimator}")
        return registry[estimator](groups)
    return _advantages.estimate_advantages(groups, estimator)
from .observability import Telemetry
from .reward import has_signal, score_group, zero_variance_fraction
from .rollout import RolloutService


@dataclasses.dataclass(frozen=True, slots=True)
class Prompt:
    request_id: int
    token_ids: list[int]


class Trainer(Protocol):
    def update(
        self,
        batch: RLBatch,
        advantages: Sequence[float],
        *,
        clip_ratio: float,
        kl_coef: float,
    ) -> dict: ...


@dataclasses.dataclass(frozen=True, slots=True)
class ControllerConfig:
    max_staleness: int = 4
    """Deprecated alias; kept for compatibility. Prefer max_offpolicy_tokens."""
    max_offpolicy_tokens: int | None = None
    """Single typed staleness budget (policy versions), enforced in the loss."""
    consumer_batch_size: int = 32
    concurrency: int = 64
    trajectories_per_sec: float | None = None
    """Optional token-bucket rate cap, independent of the concurrency cap."""
    rate_burst: int = 64
    lease_ttl_secs: float = 300.0
    group_size: int = 8
    max_tokens_per_batch: int = 8192
    alignment: int = 1
    clip_ratio: float = 0.2
    kl_coef: float = 0.0
    advantage_estimator: str = "grpo"
    """grpo | rloo | reinforce, selected from megarl.advantages."""
    scale_rewards: str = "group"
    """group | batch | none reward scaling (used by GRPO advantages)."""
    dpo: bool = False
    """Train with DPO preference pairs (best/worst of each reward group)."""
    temperature: float = 1.0
    top_p: float = 1.0
    checkpoint_every: int = 0
    """0 disables checkpointing; otherwise save every N completed versions."""


@dataclasses.dataclass(frozen=True, slots=True)
class StepResult:
    version: int
    groups: int
    dropped_groups: int
    stale_tokens: int
    token_count: int
    trainer_metrics: dict


class Controller:
    def __init__(
        self,
        config: ControllerConfig,
        *,
        rollout: RolloutService,
        reward_fn,
        trainer: Trainer,
        telemetry: Telemetry | None = None,
        weight_path: str | None = None,
        sample_builder=None,
        batch_packer=None,
    ) -> None:
        self.config = config
        self.rollout = rollout
        self.reward_fn = reward_fn
        self.trainer = trainer
        # Injection points for non-autoregressive trajectories. Defaults keep
        # the AR path byte-for-byte identical; dLLM installs a record adapter
        # and a canvas collator through the same signatures.
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
        self.leases = LeaseTable()
        self.version = 0
        self.accepted_tokens = 0
        self.run_tokens = 0
        # Rollout resume state: id after the last admitted prompt and the model
        # snapshot the engine currently serves. Both are checkpointed.
        self.prompt_cursor = 0
        self.model_version: str | None = None
        self.rate_limiter = (
            RateLimiter(config.trajectories_per_sec, config.rate_burst)
            if config.trajectories_per_sec
            else None
        )

    def _maybe_checkpoint(self, rewards_by_group: list[list[float]]) -> None:
        save = getattr(self.trainer, "save_checkpoint", None)
        every = self.config.checkpoint_every
        if save is None or every <= 0 or self.version % every != 0:
            return
        reward_stats = {
            "groups": len(rewards_by_group),
            "mean": sum(sum(rewards) for rewards in rewards_by_group)
            / max(1, sum(len(rewards) for rewards in rewards_by_group)),
        }
        save(
            f"version_{self.version}",
            version=self.version,
            accepted_tokens=self.accepted_tokens,
            run_tokens=self.run_tokens,
            reward_stats=reward_stats,
            prompt_cursor=self.prompt_cursor,
            model_version=self.model_version,
        )
        self.telemetry.emit("checkpoint_saved", version=self.version)

    def restore_checkpoint(self, state: dict) -> None:
        """Restore admission counters after a checkpoint resume.

        `state` is the manifest returned by ``Trainer.load_checkpoint``. The
        admitted-token count is floored by the version-rebased value so a
        resumed controller cannot admit a full fresh window on top of work
        that is still in flight (AReaL's `on_version_recovered`).
        """
        self.version = int(state["version"])
        rebased = self.budget.rebase_after_recovery(self.version)
        self.accepted_tokens = max(int(state.get("accepted_tokens", 0)), rebased)
        self.run_tokens = int(state.get("run_tokens", 0))
        self.prompt_cursor = max(int(state.get("prompt_cursor", 0)), 0)
        self.model_version = state.get("model_version")
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
        capacity = self.budget.capacity(
            self.version, self.accepted_tokens, self.run_tokens
        )
        capacity = rate_limit(
            self.rate_limiter, capacity, self.telemetry.emit
        )
        admitted = list(prompts[:capacity])
        if admitted:
            self.prompt_cursor = max(
                self.prompt_cursor, max(p.request_id for p in admitted) + 1
            )
        if len(prompts) > capacity:
            self.telemetry.emit("admission_deferred", requested=len(prompts), capacity=capacity)
        return admitted

    def step(self, prompts: Sequence[Prompt]) -> StepResult | None:
        admitted = self._admit(prompts)
        if not admitted:
            return None
        prompt_by_id = {p.request_id: p.token_ids for p in admitted}
        sampling = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=0,
            seed=None,
            logprobs=True,
            top_logprobs=0,
        )

        groups: list[list[TrajectoryRecord]] = []
        for prompt in admitted:
            request = RolloutRequest(
                request_id=prompt.request_id,
                policy_version=self.version,
                prompt_token_ids=prompt.token_ids,
                max_new_tokens=64,  # the rollout adapter caps this with its own config
                n=self.config.group_size,
                sampling=sampling,
                session_id=None,
            )
            records = list(self.rollout.generate(request))
            scored = score_group(records, self.reward_fn)
            for record in scored:
                self.ledger.record(
                    record.request_id, len(record.token_ids), record.policy_versions
                )
            groups.append(scored)

        dropped = sum(not has_signal(group) for group in groups)
        good_groups = [group for group in groups if has_signal(group)]
        self.telemetry.emit(
            "reward_group_summary",
            groups=len(groups),
            dropped=dropped,
            zero_variance_fraction=zero_variance_fraction(groups),
        )

        stale = self.ledger.stale_tokens(self.budget, self.version)
        self.telemetry.emit(
            "staleness", version=self.version, stale_tokens=stale, budget=self.config.max_staleness
        )

        samples: list[RLSample] = []
        rewards_by_group: list[list[float]] = []
        group_samples: list[list[RLSample]] = []
        for group in good_groups:
            rewards = [float(record.reward) for record in group]
            rewards_by_group.append(rewards)
            members = [
                self.sample_builder(record, prompt_by_id[record.request_id])
                for record in group
            ]
            group_samples.append(members)
            samples.extend(members)
        if not samples:
            self.version += 1
            return StepResult(
                version=self.version,
                groups=len(groups),
                dropped_groups=dropped,
                stale_tokens=stale,
                token_count=0,
                trainer_metrics={},
            )

        metrics: dict = {}
        trained_tokens = 0
        set_version = getattr(self.trainer, "set_version", None)
        if set_version is not None:
            set_version(self.version)
        if self.config.dpo:
            pairs = build_preference_pairs(group_samples)
            dpo_update = getattr(self.trainer, "dpo_update", None)
            if dpo_update is None:
                raise ValueError("dpo=True requires a trainer with dpo_update()")
            if pairs:
                chosen, rejected = collate_preference_pairs(pairs)
                trained_tokens = chosen.token_count + rejected.token_count
                metrics = dpo_update(chosen, rejected)
                self.telemetry.emit(
                    "trainer_dpo_update",
                    version=self.version,
                    pairs=len(pairs),
                    metrics=metrics,
                )
        else:
            advantages = compute_advantages(
                rewards_by_group,
                self.config.advantage_estimator,
                self.config.scale_rewards,
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
            trained_tokens = sum(batch.token_count for batch in batches)
            for batch, order in zip(batches, orders):
                # Packing sorts by length; follow the same permutation for the
                # advantages so each row is trained with its own trajectory's value.
                batch_adv = [advantages[index] for index in order]
                metrics = self.trainer.update(
                    batch,
                    batch_adv,
                    clip_ratio=self.config.clip_ratio,
                    kl_coef=self.config.kl_coef,
                )
                self.telemetry.emit(
                    "trainer_update",
                    version=self.version,
                    batch_size=batch.size,
                    token_count=batch.token_count,
                    metrics=metrics,
                )

        self.accepted_tokens += sum(len(record.token_ids) for group in groups for record in group)
        self.version += 1
        self.run_tokens = 0
        if self.weight_path is not None:
            self.rollout.reload_weights(self.weight_path)
            self.model_version = self.weight_path
            self.telemetry.emit("weight_reload", version=self.version, path=self.weight_path)
        # Capture the model version the engine now serves.
        self._maybe_checkpoint(rewards_by_group)
        return StepResult(
            version=self.version,
            groups=len(groups),
            dropped_groups=dropped,
            stale_tokens=stale,
            token_count=trained_tokens,
            trainer_metrics=metrics,
        )
