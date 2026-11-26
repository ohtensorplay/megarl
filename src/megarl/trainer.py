"""Production RL trainer on TensorPlay, built on the megaTrain core.

megaRL owns the RL semantics — policy objectives, advantages, staleness,
adaptive KL, distillation/DPO state — while :class:`megatrain.BaseTrainer`
owns the generic training step: FSDP/compile integration, autocast +
GradScaler, gradient accumulation with ``no_sync``, gradient clipping,
optimizer/scheduler stepping and atomic checkpoint/resume.

The controller-facing contract is unchanged (``set_version`` / ``update`` /
``dpo_update`` / ``save_checkpoint`` / ``load_checkpoint`` and the duck-typed
``optimizer``/``step_count`` attributes). Checkpoints additionally carry the
rollout prompt cursor and the published model version so a resumed controller
does not replay consumed prompts against a stale engine.

Staleness is *enforced*, not merely observed: tokens produced more than
``max_offpolicy_tokens`` policy versions ago are masked out of the loss. That
makes the single typed knob in ``devdocs/architecture.md`` §1 real instead of a
telemetry-only counter.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

import tensorplay as tp

from megatrain import BaseTrainer
from megatrain import TrainerConfig as CoreTrainerConfig
from megatrain.trainer import MANIFEST

from .advantages import gae_advantages, token_rewards
from .batching import RLBatch
from .dpo import DPO_LOSS_TYPES, dpo_loss, sequence_logps
from .kl import KL_ESTIMATORS, approx_kl
from .losses import LOSS_TYPES, TensorBatch, policy_loss
from .stateless_adam import StatelessAdam

# Losses normalized per sequence/row; a microbatch is weighted by row count.
# Every other objective is a token mean, so weighting by fresh-token share makes
# accumulation reproduce one full-batch step exactly. That is mathematically
# identical to DAPO/CISPO/VESPO's explicit global-token denominator, but keeps
# per-microbatch gradients, memory and diagnostics correct.
_ROW_LOSSES = ("grpo", "sapo", "luspo", "dr_grpo", "diffu_grpo", "wd1")


@dataclasses.dataclass(frozen=True, slots=True)
class TrainerConfig(CoreTrainerConfig):
    """megaTrain's backend config plus the RL objective knobs.

    Inheriting rather than copying means every new core field (device, AMP,
    FSDP, compile, checkpointing, gradient clipping, ...) is available here the
    moment megaTrain adds it, and ``BaseTrainer`` reads the very same object.
    """

    max_offpolicy_tokens: int = 4
    """Single staleness budget, in policy versions, enforced in the loss mask."""
    # Algorithm selection. One implementation per algorithm lives in losses.py;
    # the trainer only supplies data and normalizers.
    loss_type: str = "grpo"
    """grpo | bnpo | dr_grpo | dapo | ppo | cispo | sapo | luspo | vespo |
    diffu_grpo | wd1 | reverse_kl | opd."""
    reduction: str | None = None
    """Legacy alias: seq_mean->grpo, token_mean->bnpo, global_token_mean->dapo.
    When set it overrides ``loss_type``."""
    epsilon_low: float = 0.2
    epsilon_high: float | None = None
    """DAPO clip-higher upper bound; defaults to ``epsilon_low`` (5.0 for cispo)."""
    delta: float | None = None
    """Optional two-sided upper clip on the ratio (INTELLECT-2)."""
    importance_sampling_level: str = "token"
    """token | sequence (GSPO). LUSPO expects sequence-level ratios."""
    max_completion_length: int | None = None
    """Fixed normalizer for Dr.GRPO; defaults to the batch's max length."""
    sapo_temperature_pos: float = 1.0
    sapo_temperature_neg: float = 1.05
    vespo_k_pos: float = 2.0
    vespo_lambda_pos: float = 3.0
    vespo_k_neg: float = 3.0
    vespo_lambda_neg: float = 2.0
    off_policy_threshold: float | None = None
    """DeepSeek-V3.2 off-policy sequence mask threshold (delta)."""
    wd1_temperature: float = 1.0
    """ψ in wd1's w+/w- weights; used only by ``loss_type='wd1'``."""
    clip_ratio: float = 0.2
    clip_ratio_high: float | None = None
    """Deprecated aliases for ``epsilon_low`` / ``epsilon_high``."""
    kl_coef: float = 0.0
    """Reward-penalty KL to the reference model (or behavior policy by default)."""
    kl_type: str = "k3"
    """KL estimator: k1 | k2 | k3 | low_var_kl (Schulman k3 is the default)."""
    adaptive_kl: bool = False
    target_kl: float = 0.06
    kl_horizon: int = 10000
    entropy_coef: float = 0.0
    distill_kl_type: str = "k1"
    """Estimator for loss_type='reverse_kl'/'opd'; k1 is the score function."""
    opd_kl_coef: float = 0.0
    """Orthogonal on-policy distillation: A -= coef * (student - teacher)."""
    opd_estimator: str = "k1"
    """Estimator used by the OPD advantage reward (slime/SkyRL use k1)."""
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    """sigmoid (DPO/cDPO) | ipo."""
    dpo_label_smoothing: float = 0.0
    dpo_length_normalize: bool = False
    """IPO length normalization: divide summed logps by response length."""
    advantage_estimator: str = "grpo"
    """grpo | rloo | reinforce, used by the controller when scoring groups."""
    scale_rewards: str = "group"
    """group (GRPO) | batch (PPO-lite) | none (Dr.GRPO)."""
    gamma: float = 0.99
    lam: float = 0.95
    value_coef: float = 0.0
    clip_value: float = 0.2
    """PPO value-loss coefficient and optional value clipping range."""
    is_mode: str | None = None
    """Train/inference logprob correction: token_truncate | token_mask |
    sequence_truncate | sequence_mask. None disables correction."""
    is_clip_min: float = 0.0
    is_clip_max: float = 3.0
    # megaRL's historical default was 1e-6; keep it while inheriting the
    # core field so BaseTrainer still sees one config object.
    learning_rate: float = 1e-6


class AdaptiveKLController:
    """OpenRLHF/verl-style clamped proportional controller, stateful across steps."""

    def __init__(self, coef: float, target: float, horizon: int) -> None:
        self.coef = coef
        self.target = target
        self.horizon = horizon

    def update(self, kl: float) -> float:
        if kl > self.target * 1.5:
            self.coef *= 1.5
        elif kl < self.target / 1.5:
            self.coef /= 1.5
        # Clamp the multiplicative step so a single spike cannot explode the coef.
        self.coef = max(self.coef / 2.0, min(self.coef * 2.0, self.coef))
        return self.coef

    def state_dict(self) -> dict:
        return {"coef": self.coef, "target": self.target, "horizon": self.horizon}

    def load_state_dict(self, state: dict) -> None:
        self.coef = float(state["coef"])
        self.target = float(state["target"])
        self.horizon = int(state["horizon"])


class TensorPlayTrainer(BaseTrainer):
    """Real policy-gradient trainer: megaTrain's step loop + RL's objective.

    ``policy`` must be a TensorPlay module whose ``__call__(input_ids)`` returns
    logits shaped ``[B, L, vocab]``. ``forward_fn`` can override the call when a
    model needs extra arguments (e.g. vision features); it receives the full
    :class:`megarl.losses.TensorBatch` so packed/varlen models can read
    ``attention_mask``, ``position_ids`` and ``cu_seqlens``.

    Frozen models (``ref_model``, ``teacher_model``) are never optimized or
    checkpointed. ``value_model`` (PPO) shares the optimizer with the policy and
    is stored next to it in every checkpoint.
    """

    tensor_batch_cls = TensorBatch

    def __init__(
        self,
        policy: Any,
        config: TrainerConfig | None = None,
        *,
        optimizer: Any | None = None,
        forward_fn: Callable[[Any, TensorBatch], Any] | None = None,
        value_model: Any | None = None,
        value_fn: Callable[[Any, TensorBatch], Any] | None = None,
        ref_model: Any | None = None,
        ref_fn: Callable[[Any, TensorBatch], Any] | None = None,
        teacher_model: Any | None = None,
        teacher_fn: Callable[[Any, TensorBatch], Any] | None = None,
        scheduler: Any | None = None,
    ) -> None:
        cfg = config or TrainerConfig()
        if cfg.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if cfg.reduction is not None:
            legacy = {
                "seq_mean": "grpo",
                "token_mean": "bnpo",
                "global_token_mean": "dapo",
            }
            if cfg.reduction not in legacy:
                raise ValueError(f"unknown reduction: {cfg.reduction}")
            object.__setattr__(cfg, "loss_type", legacy[cfg.reduction])
        if cfg.loss_type not in LOSS_TYPES:
            raise ValueError(f"unknown loss_type: {cfg.loss_type}")
        if cfg.importance_sampling_level not in ("token", "sequence"):
            raise ValueError("importance_sampling_level must be token or sequence")
        if cfg.advantage_estimator not in ("grpo", "rloo", "reinforce"):
            raise ValueError(f"unknown advantage_estimator: {cfg.advantage_estimator}")
        if cfg.scale_rewards not in ("group", "batch", "none"):
            raise ValueError(f"unknown scale_rewards: {cfg.scale_rewards}")
        for name in ("kl_type", "opd_estimator", "distill_kl_type"):
            if getattr(cfg, name) not in KL_ESTIMATORS:
                raise ValueError(f"unknown {name}: {getattr(cfg, name)}")
        if cfg.dpo_loss_type not in DPO_LOSS_TYPES:
            raise ValueError(f"unknown dpo_loss_type: {cfg.dpo_loss_type}")
        if cfg.loss_type in ("reverse_kl", "opd") and teacher_model is None:
            raise ValueError(f"loss_type='{cfg.loss_type}' requires a teacher_model")
        if cfg.opd_kl_coef > 0 and teacher_model is None:
            raise ValueError("opd_kl_coef > 0 requires a teacher_model")
        if cfg.loss_type == "ppo" and value_model is None:
            raise ValueError("loss_type='ppo' requires a value_model")
        if cfg.compile_policy and forward_fn is not None:
            # megaTrain compiles the ids-only policy path; a custom forward that
            # needs extra batch fields cannot be captured by that signature.
            raise ValueError(
                "compile_policy requires the default policy(input_ids) forward; "
                "custom forward_fn only runs uncompiled"
            )

        # PPO optimizes the critic jointly with the policy; dedupe parameters in
        # case the value head shares a backbone with the policy.
        params: list = []
        seen: set[int] = set()
        candidates = list(policy.parameters()) + (
            list(value_model.parameters()) if value_model is not None else []
        )
        for candidate in candidates:
            if id(candidate) not in seen:
                seen.add(id(candidate))
                params.append(candidate)
        if optimizer is None:
            if cfg.optimizer_kind == "stateless_adam":
                optimizer = StatelessAdam(
                    params,
                    lr=cfg.learning_rate,
                    betas=cfg.adam_betas,
                    eps=cfg.adam_eps,
                    weight_decay=cfg.weight_decay,
                    bias_correction=cfg.stateless_bias_correction,
                )
            else:
                optimizer = tp.optim.AdamW(
                    params,
                    lr=cfg.learning_rate,
                    betas=cfg.adam_betas,
                    eps=cfg.adam_eps,
                    weight_decay=cfg.weight_decay,
                )

        # BaseTrainer owns seeding, device/AMP setup, optimizer/scheduler state,
        # FSDP/compile and checkpoint publication.
        super().__init__(
            policy,
            cfg,
            optimizer=optimizer,
            forward_fn=forward_fn,
            teacher_model=teacher_model,
            teacher_fn=teacher_fn,
            scheduler=scheduler,
        )

        self.value_model = value_model
        self.value_fn = value_fn or (lambda model, tensors: model(tensors.input_ids))
        self.ref_model = ref_model
        self.ref_fn = ref_fn or (lambda model, tensors: model(tensors.input_ids))
        # ``config`` is the same object BaseTrainer reads; ``rl_config`` is the
        # explicit RL-facing alias used by callers and documentation.
        self.rl_config = cfg
        self.version = 0
        self.adaptive_kl = (
            AdaptiveKLController(cfg.kl_coef, cfg.target_kl, cfg.kl_horizon)
            if cfg.adaptive_kl
            else None
        )

    # -- megaTrain hooks ------------------------------------------------
    def _needs_auto_teacher(self) -> bool:
        return (
            self.config.loss_type in ("reverse_kl", "opd")
            or self.config.opd_kl_coef > 0
        )

    def _slice_batch(self, batch: RLBatch, start: int, stop: int) -> RLBatch:
        return _slice_batch(batch, start, stop)

    def _keep_tokens(self, batch: RLBatch) -> int:
        """Accumulation weight unit for one (micro)batch.

        megaTrain normalizes each microbatch by ``keep_i / sum(keep)``. The RL
        objectives need two different units, so this hook encodes them:

        * row-normalized losses (GRPO/SAPO/LUSPO/Dr.GRPO) are weighted by row
          count, because their loss is a mean over rows;
        * every token-normalized objective (BNPO, DAPO, CISPO, VESPO, PPO,
          reverse-KL/OPD) is weighted by its fresh valid token count. Summing
          those shares reproduces the global-token denominator exactly while
          keeping each microbatch's graph private.
        """
        if self.config.loss_type in _ROW_LOSSES:
            return int(batch.size)
        return self._fresh_token_count(batch)

    def _keep_tokens_tensors(self, tensors) -> int:
        """Tensor-path counterpart of :meth:`_keep_tokens` (data-plane batches)."""
        if self.config.loss_type in _ROW_LOSSES:
            return int(tensors.input_ids.shape[0])
        versions = getattr(tensors, "versions", None)
        keep = tensors.loss_mask
        if versions is None:
            return int(keep[:, 1:].sum().flatten()[0])
        fresh = (~((self.version - versions) > self.config.max_offpolicy_tokens)).to(
            keep.dtype
        )
        return int((keep * fresh).sum().flatten()[0])

    # -- RL helpers -----------------------------------------------------
    def set_version(self, version: int) -> None:
        """Called by the controller before each update so staleness is exact."""
        self.version = int(version)

    def _fresh_token_count(self, batch: RLBatch) -> int:
        """Valid, non-stale, non-padding tokens in one batch."""
        return sum(
            1
            for mask_row, version_row in zip(batch.loss_mask, batch.versions)
            for keep, version in zip(mask_row, version_row)
            if keep and self.version - version <= self.config.max_offpolicy_tokens
        )

    def _ref_logprobs(self, tensors: TensorBatch) -> Any:
        return self._logprobs(self.ref_fn(self.ref_model, tensors), tensors.input_ids)

    def _value_preds(self, tensors: TensorBatch) -> Any:
        return self.value_fn(self.value_model, tensors)

    def _rl_aux(
        self, batch: RLBatch, advantages: Any, *, kl_coef: float
    ) -> tuple[dict, int, Any, Any]:
        """Build the per-row tensors and scalars the RL objective needs.

        Returns ``(aux, global_tokens, old_logprobs, teacher_logprobs,
        metric_mask)``. Frozen forwards run once for the whole batch, never per
        microbatch, and every tensor is handed to ``train_step`` through ``aux``
        so megaTrain slices it along with the rows.
        """
        cfg = self.config
        device = cfg.device
        tensors = self.tensor_batch_cls(batch, device=device)
        aux: dict[str, Any] = {}
        teacher_full = None
        if cfg.loss_type in ("reverse_kl", "opd") or cfg.opd_kl_coef > 0:
            with tp.no_grad():
                teacher_full = self._teacher_logprobs(tensors)
            aux["teacher_logprobs"] = teacher_full
        if (kl_coef or cfg.kl_coef) > 0 and self.ref_model is not None:
            with tp.no_grad():
                aux["ref_logprobs"] = self._ref_logprobs(tensors)

        old_full = None
        opd_bonus = None
        if cfg.opd_kl_coef > 0:
            old_full = tensors.old_logprobs
            # OPD is orthogonal to the base estimator: A' = A - λ·KL(student||teacher).
            opd_bonus = -cfg.opd_kl_coef * approx_kl(
                old_full, teacher_full, kl_type=cfg.opd_estimator
            )

        fresh = (~((self.version - tensors.versions) > cfg.max_offpolicy_tokens)).to(
            tensors.loss_mask.dtype
        )
        aux["extra_mask"] = fresh
        metric_mask = tensors.loss_mask * fresh

        if cfg.loss_type == "ppo":
            # GAE uses the staleness-masked reward stream; the loss additionally
            # re-applies ``fresh`` through ``extra_mask``. Values are read once
            # under no_grad for the advantage estimate; the value *loss* is
            # recomputed per microbatch by the injected loss wrapper so gradient
            # accumulation never backpropagates through a shared graph twice.
            gae_mask = metric_mask
            with tp.no_grad():
                values_full = self._value_preds(tensors)
            if tuple(values_full.shape) != tuple(gae_mask.shape):
                raise ValueError("value_fn must return one value per token, shape [B, L]")
            bootstrap = tp.zeros((batch.size, 1), dtype=gae_mask.dtype, device=device)
            reward_tokens = token_rewards(
                tp.tensor(batch.rewards, dtype=tp.float32, device=device), gae_mask
            )
            adv_full, returns_full = gae_advantages(
                reward_tokens,
                tp.cat([values_full, bootstrap], dim=-1),
                gae_mask,
                gamma=cfg.gamma,
                lam=cfg.lam,
            )
            # The helpers materialize host lists; move the results back once.
            adv_full = tp.tensor(adv_full.tolist(), dtype=tp.float32, device=device)
            returns_full = tp.tensor(returns_full.tolist(), dtype=tp.float32, device=device)
            if opd_bonus is not None:
                adv_full = adv_full + opd_bonus
            aux["advantages"] = adv_full
            aux["returns"] = returns_full
        else:
            adv = [float(a) for a in advantages]
            if len(adv) != batch.size:
                raise ValueError(f"advantages/rows mismatch: {len(adv)} vs {batch.size}")
            adv_full = tp.tensor(adv, dtype=tp.float32, device=device)
            if opd_bonus is not None:
                adv_full = adv_full.unsqueeze(-1) + opd_bonus
            aux["advantages"] = adv_full
        return aux, self._fresh_token_count(batch), old_full, teacher_full, metric_mask

    def _loss_kwargs(self, *, clip_ratio: float, kl_coef: float) -> dict[str, Any]:
        cfg = self.config
        low = cfg.epsilon_low if cfg.epsilon_low != 0.2 else (
            cfg.clip_ratio if cfg.clip_ratio != 0.2 else clip_ratio
        )
        high = cfg.epsilon_high or cfg.clip_ratio_high or clip_ratio
        coef = self.adaptive_kl.coef if self.adaptive_kl else (kl_coef or cfg.kl_coef)
        return {
            "loss_type": cfg.loss_type,
            "epsilon_low": low,
            "epsilon_high": high,
            "importance_sampling_level": cfg.importance_sampling_level,
            "delta": cfg.delta,
            "kl_coef": coef,
            "entropy_coef": cfg.entropy_coef,
            "is_mode": cfg.is_mode,
            "is_clip_min": cfg.is_clip_min,
            "is_clip_max": cfg.is_clip_max,
            "max_completion_length": cfg.max_completion_length,
            # The token-normalized losses divide by this microbatch's mask; the
            # core's fresh-token weighting then reconstructs the full-window
            # denominator without a shared graph or double normalization.
            "global_token_count": None,
            "sapo_temperature_pos": cfg.sapo_temperature_pos,
            "sapo_temperature_neg": cfg.sapo_temperature_neg,
            "vespo_k_pos": cfg.vespo_k_pos,
            "vespo_lambda_pos": cfg.vespo_lambda_pos,
            "vespo_k_neg": cfg.vespo_k_neg,
            "vespo_lambda_neg": cfg.vespo_lambda_neg,
            "off_policy_threshold": cfg.off_policy_threshold,
            "wd1_temperature": cfg.wd1_temperature,
            "value_coef": cfg.value_coef,
            "clip_value": cfg.clip_value,
            "kl_type": cfg.kl_type,
            "distill_kl_type": cfg.distill_kl_type,
        }

    def _rl_loss(self, logits, tensors: TensorBatch, *, advantages, **kwargs):
        """Injected objective: :func:`policy_loss` plus the PPO critic forward.

        The critic is evaluated on the current microbatch so its graph is
        private to that microbatch; that keeps gradient accumulation exact and
        memory-bounded instead of backpropagating through one shared full-batch
        graph several times.
        """
        if self.config.loss_type == "ppo":
            if "returns" not in kwargs:
                raise ValueError("loss_type='ppo' requires per-token returns")
            kwargs["value_preds"] = self._value_preds(tensors)
        return policy_loss(logits, tensors, advantages, **kwargs)

    # -- training -------------------------------------------------------
    def update(
        self,
        batch: RLBatch,
        advantages: Any,
        *,
        clip_ratio: float = 0.2,
        kl_coef: float = 0.0,
        dry_run: bool = False,
    ) -> dict:
        """Run one optimizer step over ``batch`` using real TensorPlay tensors.

        ``dry_run`` keeps gradients for auditing and skips the optimizer step;
        production callers never set it. For PPO the token-level advantages and
        returns are computed here with GAE from the batch's sequence rewards and
        the configured value model, so ``advantages`` is ignored in that case.
        """
        aux, global_tokens, old_full, teacher_full, metric_mask = self._rl_aux(
            batch, advantages, kl_coef=kl_coef
        )
        kwargs = self._loss_kwargs(clip_ratio=clip_ratio, kl_coef=kl_coef)
        metrics = self.train_step(
            batch,
            loss_fn=self._rl_loss,
            loss_kwargs=kwargs,
            aux=aux,
            dry_run=dry_run,
        )
        metrics["train/version"] = float(self.version)
        metrics["train/global_tokens"] = float(global_tokens)
        if old_full is not None and teacher_full is not None:
            reverse_kl = (old_full - teacher_full).detach()
            opd_kl = (reverse_kl * metric_mask).sum() / metric_mask.sum().clamp(min=1.0)
            metrics["train/opd_reverse_kl"] = float(opd_kl.flatten()[0])
        if self.adaptive_kl:
            self.adaptive_kl.update(metrics.get("train/kl", 0.0))
            metrics["train/kl_coef"] = self.adaptive_kl.coef
        return metrics

    def dpo_update(
        self,
        chosen: RLBatch,
        rejected: RLBatch,
        *,
        beta: float | None = None,
        dry_run: bool = False,
    ) -> dict:
        """One DPO optimizer step over aligned chosen/rejected batches.

        Pairs are accumulated with the same ``no_sync`` / GradScaler / clipping
        path as :meth:`megatrain.BaseTrainer.train_step`. ``ref_model`` is
        optional: without it the reference log-probabilities are treated as zero
        (the implicit-reference DPO variant). Log-probs are summed over response
        tokens; IPO can optionally length-normalize them.
        """
        cfg = self.config
        device = cfg.device
        if chosen.size != rejected.size:
            raise ValueError(
                f"chosen/rejected row mismatch: {chosen.size} vs {rejected.size}"
            )
        rows = chosen.size
        n = min(cfg.gradient_accumulation_steps, rows)
        size = (rows + n - 1) // n
        spans = [(start, min(start + size, rows)) for start in range(0, rows, size)]

        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        diag_accum: dict[str, float] = {}
        for index, (start, stop) in enumerate(spans):
            chosen_t = TensorBatch(_slice_batch(chosen, start, stop), device=device)
            rejected_t = TensorBatch(_slice_batch(rejected, start, stop), device=device)
            with tp.amp.autocast(
                self._device_type, dtype=self._amp_dtype, enabled=self._amp_enabled
            ):
                policy_chosen_logits = (
                    self._compiled(chosen_t.input_ids)
                    if self._compiled
                    else self._forward(chosen_t)
                )
                policy_rejected_logits = (
                    self._compiled(rejected_t.input_ids)
                    if self._compiled
                    else self._forward(rejected_t)
                )
            pc = sequence_logps(
                policy_chosen_logits, chosen_t.input_ids, chosen_t.loss_mask
            )
            pr = sequence_logps(
                policy_rejected_logits, rejected_t.input_ids, rejected_t.loss_mask
            )
            ref_chosen = ref_rejected = None
            if self.ref_model is not None:
                with tp.no_grad():
                    ref_chosen = sequence_logps(
                        self.ref_fn(self.ref_model, chosen_t),
                        chosen_t.input_ids,
                        chosen_t.loss_mask,
                    )
                    ref_rejected = sequence_logps(
                        self.ref_fn(self.ref_model, rejected_t),
                        rejected_t.input_ids,
                        rejected_t.loss_mask,
                    )
            if cfg.dpo_length_normalize:
                pc = pc / chosen_t.loss_mask.sum(-1).clamp(min=1.0)
                pr = pr / rejected_t.loss_mask.sum(-1).clamp(min=1.0)
                if ref_chosen is not None:
                    ref_chosen = ref_chosen / chosen_t.loss_mask.sum(-1).clamp(min=1.0)
                    ref_rejected = ref_rejected / rejected_t.loss_mask.sum(-1).clamp(min=1.0)
            loss, diag = dpo_loss(
                pc,
                pr,
                ref_chosen_logps=ref_chosen,
                ref_rejected_logps=ref_rejected,
                beta=cfg.dpo_beta if beta is None else beta,
                loss_type=cfg.dpo_loss_type,
                label_smoothing=cfg.dpo_label_smoothing,
            )
            weight = (stop - start) / rows
            accumulation_cm = self._no_sync() if index < len(spans) - 1 else _nullcontext()
            with accumulation_cm:
                scaled = loss * weight
                (self.scaler.scale(scaled) if self._scaler_enabled else scaled).backward()
            loss_sum += float(loss.detach().flatten()[0]) * weight
            for key, value in diag.items():
                diag_accum[key] = diag_accum.get(key, 0.0) + float(
                    value.detach().flatten()[0]
                ) * weight
        if self._scaler_enabled:
            self.scaler.unscale_(self.optimizer)
        if cfg.max_grad_norm is not None:
            tp.nn.utils.clip_grad_norm_(
                [
                    param
                    for group in self.optimizer.param_groups
                    for param in group["params"]
                ],
                cfg.max_grad_norm,
            )
        if not dry_run:
            if self._scaler_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step_count += 1
        metrics = {f"dpo/{key}": value for key, value in diag_accum.items()}
        metrics["train/loss"] = loss_sum
        metrics["train/step"] = float(self.step_count)
        metrics["train/version"] = float(self.version)
        return metrics

    # -- checkpoint / resume --------------------------------------------
    def checkpoint_artifacts(self) -> dict[str, Any]:
        if self.value_model is None:
            return {}
        return {"value.mega": self.value_model.state_dict()}

    def load_checkpoint_artifacts(self, root: Path, files: dict[str, str]) -> None:
        if self.value_model is not None and "value.mega" in files:
            self.value_model.load_state_dict(tp.load(root / "value.mega"))

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        version: int,
        accepted_tokens: int = 0,
        run_tokens: int = 0,
        reward_stats: dict | None = None,
        extra: dict | None = None,
        prompt_cursor: int = 0,
        model_version: str | None = None,
    ) -> Path:
        """Atomically publish a resumable checkpoint with RL state and digests.

        megaTrain writes the policy, optimizer (and critic artifact), RNG and
        SHA-256 manifest; the RL sidecar fields are stored at the manifest top
        level so old readers keep working. ``prompt_cursor`` is the id after the
        last prompt admitted for rollout, and ``model_version`` names the weight
        snapshot the engine is serving; both are restored on resume.
        """
        return super().save_checkpoint(
            path,
            extra=extra,
            version=int(version),
            accepted_tokens=int(accepted_tokens),
            run_tokens=int(run_tokens),
            reward_stats=reward_stats or {},
            kl=self.adaptive_kl.state_dict() if self.adaptive_kl else None,
            prompt_cursor=int(prompt_cursor),
            model_version=model_version,
        )

    def load_checkpoint(self, path: str | Path) -> dict:
        """Restore weights/optimizer/RNG/critic and return the RL metadata."""
        payload = super().load_checkpoint(path)
        if payload.get("kl") and self.adaptive_kl:
            self.adaptive_kl.load_state_dict(payload["kl"])
        return {
            "version": int(payload.get("version", 0)),
            "accepted_tokens": int(payload.get("accepted_tokens", 0)),
            "run_tokens": int(payload.get("run_tokens", 0)),
            "reward_stats": payload.get("reward_stats", {}),
            # Older manifests predate these fields; zero/None means "unknown",
            # which callers treat as "start from the beginning".
            "prompt_cursor": int(payload.get("prompt_cursor", 0)),
            "model_version": payload.get("model_version"),
            "extra": payload.get("extra", {}),
        }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _slice_batch(batch: RLBatch, start: int, stop: int) -> RLBatch:
    return RLBatch(
        input_ids=batch.input_ids[start:stop],
        attention_mask=batch.attention_mask[start:stop],
        loss_mask=batch.loss_mask[start:stop],
        position_ids=batch.position_ids[start:stop],
        old_logprobs=batch.old_logprobs[start:stop],
        versions=batch.versions[start:stop],
        rewards=batch.rewards[start:stop],
        cu_seqlens=[0] + [n - batch.cu_seqlens[start] for n in batch.cu_seqlens[start + 1 : stop + 1]],
        max_len=batch.max_len,
        sampling_logprobs=batch.sampling_logprobs[start:stop],
    )


__all__ = ["AdaptiveKLController", "MANIFEST", "TensorPlayTrainer", "TrainerConfig"]