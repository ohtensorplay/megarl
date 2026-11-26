"""Policy-gradient losses on TensorPlay — one implementation per algorithm.

The public entry point is :func:`policy_loss`, which covers the full family:

===================  =========================================================
``loss_type``        Objective / normalizer
===================  =========================================================
``grpo``             clipped surrogate, per-sequence mean then batch mean
``bnpo``             clipped surrogate, batch-token mean
``dr_grpo``          clipped surrogate, fixed ``max_completion_length`` denom
``dapo``             clipped surrogate (clip-higher), global-token denom
``ppo``              clipped surrogate + GAE value loss (optional value clip)
``cispo``            clipped *ratio* weights times advantage times logprobs
``sapo``             sigmoid soft-adaptive gate with pos/neg temperatures
``luspo``            sequence-level IS ratio, length-scaled per-sequence loss
``vespo``            asymmetric Gamma weighting of sequence-level ratios
``diffu_grpo``       GRPO ratio over a *diffusion* log-prob estimator (d1)
``wd1``              ratio-free weighted log-likelihood (wd1, ICLR 2026)
``reverse_kl``       sampled-token reverse KL to a teacher (KL(student||teacher))
``opd``              alias of ``reverse_kl`` (on-policy distillation)
===================  =========================================================

All variants share: token-level staleness masking, train/inference importance
correction, an estimator-based KL penalty (``kl_type``: k1/k2/k3/low_var_kl),
entropy bonus, optional two-sided upper clip (``delta``, INTELLECT-2), and
DeepSeek-V3.2 off-policy sequence masking. ``reverse_kl``/``opd`` additionally
take ``teacher_logprobs`` and use ``distill_kl_type`` (default k1).
``diffu_grpo`` shares ``grpo``'s objective but is meant to be called with
``old_logprobs``/``ref_logprobs`` recomputed by a diffusion log-prob estimator
(d1's one-step unmasking estimator); ``wd1`` uses softmax-normalised positive
and negative weights instead of an explicit importance ratio. ``grpo_loss`` is
kept as a backwards-compatible wrapper over ``policy_loss``.
"""

from __future__ import annotations

import math
from typing import Sequence

import tensorplay as tp
import tensorplay.nn.functional as F

from megatrain import TensorBatch as CoreTensorBatch

from .advantages import group_advantages
from .batching import RLBatch
from .corrections import importance_correction
from .kl import approx_kl

LOSS_TYPES = (
    "grpo",
    "bnpo",
    "dr_grpo",
    "dapo",
    "ppo",
    "cispo",
    "sapo",
    "luspo",
    "vespo",
    "diffu_grpo",
    "wd1",
    "reverse_kl",
    "opd",
)

_REDUCTION_MAP = {
    "seq_mean": "grpo",
    "token_mean": "bnpo",
    "global_token_mean": "dapo",
}


@tp.no_grad()
def group_advantages_alias(groups: Sequence[Sequence[float]], eps: float = 1e-4) -> list[float]:
    """Backwards-compatible alias for :func:`megarl.advantages.group_advantages`."""
    return group_advantages(groups, eps=eps)


# Re-exported so ``from megarl.losses import group_advantages`` keeps working.
__all__ = [
    "LOSS_TYPES",
    "TensorBatch",
    "diffu_grpo_loss",
    "group_advantages",
    "group_advantages_alias",
    "grpo_loss",
    "policy_loss",
    "token_logprobs",
]


class TensorBatch(CoreTensorBatch):
    """TensorPlay tensors for one RLBatch, ready for a compiled policy step.

    Extends :class:`megatrain.TensorBatch` with the RL-only fields: behavior
    log-probabilities, per-token policy versions, and the optional sampling
    log-probabilities used by the train/inference importance correction.
    """

    def __init__(self, batch: RLBatch, device: str = "cpu") -> None:
        super().__init__(batch, device=device)
        self.old_logprobs = tp.tensor(
            batch.old_logprobs, dtype=tp.float32, device=device
        )
        self.versions = tp.tensor(batch.versions, dtype=tp.long, device=device)
        self.sampling_logprobs = (
            tp.tensor(batch.sampling_logprobs, dtype=tp.float32, device=device)
            if any(batch.sampling_logprobs)
            else None
        )
        # Optional reference-model logprobs for an explicit KL(target) term.
        # When absent, the frozen behavior policy is the reference.
        self.ref_logprobs: object | None = None
        # Teacher logprobs for reverse-KL / OPD, normally supplied by the
        # trainer's frozen teacher model rather than the wire contract.
        self.teacher_logprobs: object | None = None


def token_logprobs(logits, input_ids):
    """Log-probability assigned to each target token, shape [B, L]."""
    return tp.gather(F.log_softmax(logits, dim=-1), -1, input_ids.unsqueeze(-1)).squeeze(-1)


def _masked_mean(x, mask, denom) -> object:
    return (x * mask).sum() / denom


@tp.no_grad()
def _gamma_weights(
    advantages,
    log_ratio,
    mask,
    is_ratio,
    k_pos: float,
    lambda_pos: float,
    lambda_neg: float,
    k_neg: float,
):
    """VESPO gamma weight ``phi(w) = exp(lambda + k log w - lambda w)``, per sequence.

    Computed in log space with the paper's asymmetric (k, lambda) pair chosen
    by the advantage sign, detached so it only scales the gradient.
    """
    lower = math.log(1e-8)
    log_ratio_clamped = log_ratio.clamp(-20.0, 20.0)
    seq_log_ratio = (log_ratio_clamped * mask).sum(-1, keepdim=True)
    if is_ratio is not None:
        safe_ratio = is_ratio.clamp(min=1e-8, max=20.0)
        seq_log_ratio = seq_log_ratio + (tp.log(safe_ratio) * mask).sum(-1, keepdim=True)
    log_w = seq_log_ratio.clamp(lower, 20.0)
    w = log_w.exp()
    nonneg = advantages >= 0
    k = tp.where(nonneg, tp.full_like(log_w, k_pos), tp.full_like(log_w, k_neg))
    lam = tp.where(
        nonneg,
        tp.full_like(log_w, lambda_pos),
        tp.full_like(log_w, lambda_neg),
    ).clamp(min=1e-4)
    log_phi = lam + k * log_w - lam * w
    return tp.nan_to_num(log_phi.exp(), nan=0.0, posinf=0.0, neginf=0.0)


def _off_policy_mask(advantages, new_logprobs, behavior_logprobs, mask, threshold):
    """DeepSeek-V3.2 sequence keep mask: keep positive advantages or low KL."""
    kl = behavior_logprobs - new_logprobs.detach()
    kl = tp.nan_to_num(kl, nan=0.0)
    seq_kl = (kl * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1.0)
    adv_seq = advantages if advantages.shape[-1] == 1 else advantages.mean(-1, keepdim=True)
    keep = (adv_seq >= 0) | (seq_kl <= threshold)
    return keep.to(mask.dtype)


def _as_token_advantages(advantages, like):
    """Broadcast [B] / [B, 1] advantages to [B, L] via normal broadcasting."""
    if len(advantages.shape) == 1:
        return advantages.unsqueeze(-1)
    return advantages


def policy_loss(
    logits,
    tensors: TensorBatch,
    advantages,
    *,
    loss_type: str = "grpo",
    epsilon_low: float = 0.2,
    epsilon_high: float | None = None,
    importance_sampling_level: str = "token",
    delta: float | None = None,
    kl_coef: float = 0.0,
    entropy_coef: float = 0.0,
    extra_mask=None,
    is_mode: str | None = None,
    is_clip_min: float = 0.0,
    is_clip_max: float = 3.0,
    max_completion_length: int | None = None,
    global_token_count: int | None = None,
    sapo_temperature_pos: float = 1.0,
    sapo_temperature_neg: float = 1.05,
    vespo_k_pos: float = 2.0,
    vespo_lambda_pos: float = 3.0,
    vespo_k_neg: float = 3.0,
    vespo_lambda_neg: float = 2.0,
    off_policy_threshold: float | None = None,
    old_logprobs=None,
    wd1_temperature: float = 1.0,
    value_preds=None,
    returns=None,
    value_coef: float = 0.0,
    clip_value: float = 0.2,
    ref_logprobs=None,
    kl_type: str = "k3",
    teacher_logprobs=None,
    distill_kl_type: str = "k1",
    reduction: str | None = None,
) -> tuple:
    """Return ``(scalar loss, diagnostics dict of tensors)``.

    ``advantages`` may be ``[B]``, ``[B, 1]`` or ``[B, L]`` (token-level, needed
    by PPO/GAE). ``extra_mask`` is an optional ``[B, L]`` multiplier used to
    mask stale tokens without touching rewards or advantage normalization
    (AReaL semantics). ``global_token_count`` is the number of valid tokens in
    the whole accumulation window; DAPO/CISPO/VESPO use it as their normalizer.

    ``old_logprobs`` overrides ``tensors.old_logprobs``. Diffusion RL uses
    this: the wire contract carries tokens, while behavior/reference log-probs
    are recomputed with a diffusion log-prob estimator (d1/wd1) rather than
    taken from a next-token sampling engine.
    """
    if reduction is not None:
        if reduction not in _REDUCTION_MAP:
            raise ValueError(f"unknown reduction: {reduction}")
        loss_type = _REDUCTION_MAP[reduction]
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"unknown loss_type: {loss_type}")
    if importance_sampling_level not in ("token", "sequence"):
        raise ValueError("importance_sampling_level must be 'token' or 'sequence'")
    if loss_type == "ppo" and (value_preds is None or returns is None):
        raise ValueError("loss_type='ppo' requires value_preds and returns")
    if loss_type in ("reverse_kl", "opd") and teacher_logprobs is None and tensors.teacher_logprobs is None:
        raise ValueError(f"loss_type='{loss_type}' requires teacher_logprobs")

    high = epsilon_high if epsilon_high is not None else (5.0 if loss_type == "cispo" else epsilon_low)

    log_softmax = F.log_softmax(logits, dim=-1)
    new_logprobs = tp.gather(log_softmax, -1, tensors.input_ids.unsqueeze(-1)).squeeze(-1)
    mask = tensors.loss_mask
    if extra_mask is not None:
        mask = mask * extra_mask.to(mask.dtype)
    denom = mask.sum().clamp(min=1.0)
    old = old_logprobs if old_logprobs is not None else tensors.old_logprobs
    if tuple(old.shape) != tuple(new_logprobs.shape):
        raise ValueError(
            f"old_logprobs shape {tuple(old.shape)} != logits token grid {tuple(new_logprobs.shape)}"
        )
    log_ratio = new_logprobs - old

    # Train/inference importance correction (TRL four modes). The multiplicative
    # ratio is applied to the policy term for every loss except VESPO, which
    # folds it into its sequence weight in log space.
    is_ratio = None
    if is_mode is not None:
        if tensors.sampling_logprobs is None:
            raise ValueError("is_mode requires sampling_logprobs on the batch")
        is_ratio = importance_correction(
            tensors.sampling_logprobs,
            old,
            mode=is_mode,
            clip_min=is_clip_min,
            clip_max=is_clip_max,
            mask=mask,
        )

    if importance_sampling_level == "token":
        log_iw = log_ratio
    else:
        keep = mask
        log_iw = ((log_ratio * keep).sum(-1) / keep.sum(-1).clamp(min=1.0)).unsqueeze(-1)
    coef_1 = log_iw.exp()
    if delta is not None and loss_type in (
        "grpo", "bnpo", "dr_grpo", "dapo", "luspo", "diffu_grpo"
    ):
        coef_1 = coef_1.clamp(max=delta)

    # KL(new || ref) penalty. Without a reference model the frozen behavior
    # policy is the reference (GRPO default). The estimator is selectable; k3
    # keeps the previous low-variance behavior.
    ref = ref_logprobs if ref_logprobs is not None else old
    per_token_kl = approx_kl(new_logprobs, ref, kl_type=kl_type)
    kl_mean = (per_token_kl * mask).sum() / denom

    adv = _as_token_advantages(advantages, new_logprobs)
    soft = None
    phi = None
    value_loss = None
    reverse_kl = None
    if loss_type in ("reverse_kl", "opd"):
        teacher = teacher_logprobs if teacher_logprobs is not None else tensors.teacher_logprobs
        # Sampled-token reverse KL: minimizing d·log π_θ (k1) differentiates
        # to the reverse-KL score function, as in SDPO/slime/SkyRL OPD.
        reverse_kl = approx_kl(
            new_logprobs, teacher, kl_type=distill_kl_type
        )
        per_token_loss = reverse_kl
    elif loss_type in (
        "grpo", "bnpo", "dr_grpo", "dapo", "luspo", "ppo", "diffu_grpo"
    ):
        coef_2 = coef_1.clamp(1.0 - epsilon_low, 1.0 + high)
        per_token_loss = -tp.minimum(coef_1 * adv, coef_2 * adv)
    elif loss_type == "wd1":
        # wd1 (ICLR 2026): L = (1/G) Σ (w- - w+) · log πθ(o|q), with
        # w+ = softmax(ψ·A), w- = softmax(-ψ·A) over the sampled group. No
        # importance ratio, so likelihood approximation error grows linearly
        # instead of being exponentiated by a ratio.
        adv_seq = adv if adv.shape[-1] == 1 else adv.mean(-1, keepdim=True)
        scaled = (adv_seq * wd1_temperature).clamp(-30.0, 30.0)
        w_pos = tp.softmax(scaled, dim=0).detach()
        w_neg = tp.softmax(-scaled, dim=0).detach()
        per_token_loss = (w_neg - w_pos) * new_logprobs
    elif loss_type == "cispo":
        clamped = coef_1.clamp(max=high).detach()
        per_token_loss = -clamped * adv * new_logprobs
    elif loss_type == "sapo":
        temp = tp.where(
            adv > 0,
            tp.full_like(adv, sapo_temperature_pos),
            tp.full_like(adv, sapo_temperature_neg),
        )
        soft = tp.sigmoid(temp * (coef_1 - 1.0)) * (4.0 / temp)
        per_token_loss = -soft * adv
    else:  # vespo
        vespo_mask = mask if is_ratio is None else mask * (is_ratio > 0).to(mask.dtype)
        phi = _gamma_weights(
            adv,
            log_ratio,
            vespo_mask,
            is_ratio,
            vespo_k_pos,
            vespo_lambda_pos,
            vespo_lambda_neg,
            vespo_k_neg,
        )
        per_token_loss = -phi * adv * new_logprobs * vespo_mask

    if off_policy_threshold is not None:
        behavior = tensors.sampling_logprobs if is_ratio is not None else old
        keep = _off_policy_mask(adv, new_logprobs, behavior, mask, off_policy_threshold)
        per_token_loss = per_token_loss * keep
    if is_ratio is not None and loss_type != "vespo":
        per_token_loss = per_token_loss * is_ratio
    per_token_loss = per_token_loss + kl_coef * per_token_kl

    policy_term = per_token_loss * mask
    if loss_type in ("grpo", "sapo", "diffu_grpo", "wd1"):
        loss = (policy_term.sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        loss = policy_term.sum() / denom
    elif loss_type == "dr_grpo":
        width = max_completion_length if max_completion_length is not None else tensors.max_len
        loss = policy_term.sum() / (mask.shape[0] * max(width, 1))
    elif loss_type == "luspo":
        # per_token_loss is [B, 1] under sequence-level IS; masking and summing
        # still handles KL/IS broadcasting into [B, L].
        loss = policy_term.sum(-1).mean()
    else:  # dapo / cispo / vespo / ppo / reverse_kl / opd
        normalizer = float(global_token_count) if global_token_count is not None else float(denom.detach())
        loss = policy_term.sum() / max(normalizer, 1.0)

    # Diagnostics report the *effective* ratio (IS correction included) so the
    # reported mean/clip rate matches the ratio the policy term actually sees.
    diag_ratio = coef_1 * is_ratio if is_ratio is not None else coef_1
    diagnostics = {
        "loss": loss,
        "mean_ratio": _masked_mean(diag_ratio, mask, denom),
        "clip_fraction": (
            ((diag_ratio > 1.0 + high) | (diag_ratio < 1.0 - epsilon_low))
            .to(tp.float32)
            * mask
        ).sum()
        / denom,
        "kl": kl_mean,
        "mean_token_logprob": _masked_mean(new_logprobs, mask, denom),
    }
    if reverse_kl is not None:
        diagnostics["reverse_kl"] = _masked_mean(reverse_kl.detach(), mask, denom)
    if loss_type == "cispo":
        diagnostics["cispo_clip_ratio"] = (
            ((coef_1 > high) & (adv > 0)).to(tp.float32) * mask
        ).sum() / denom
    if loss_type == "sapo":
        diagnostics["sapo_soft_coef_mean"] = _masked_mean(soft, mask, denom)
    if loss_type == "vespo":
        diagnostics["vespo_phi_seq_mean"] = phi.mean()
    if is_ratio is not None:
        diagnostics["is_correction_mean"] = _masked_mean(is_ratio, mask, denom)
        diagnostics["is_affected_fraction"] = (
            ((is_ratio - 1.0).abs() > 1e-6).to(tp.float32) * mask
        ).sum() / denom
    if off_policy_threshold is not None:
        diagnostics["off_policy_keep_fraction"] = (
            _off_policy_mask(adv, new_logprobs, tensors.sampling_logprobs if is_ratio is not None else old, mask, off_policy_threshold)
            * mask
        ).sum() / denom

    if loss_type == "ppo":
        err = returns - value_preds
        if clip_value:
            clipped = value_preds + err.clamp(-clip_value, clip_value)
            vf_err = tp.maximum(err ** 2, (returns - clipped) ** 2)
        else:
            vf_err = err ** 2
        value_loss = 0.5 * _masked_mean(vf_err, mask, denom)
        loss = loss + value_coef * value_loss
        diagnostics["value_loss"] = value_loss.detach()
        diagnostics["returns_mean"] = _masked_mean(returns, mask, denom)

    if entropy_coef:
        entropy = -(log_softmax.exp() * log_softmax).sum(-1)
        entropy_mean = _masked_mean(entropy, mask, denom)
        loss = loss - entropy_coef * entropy_mean
        diagnostics["entropy"] = entropy_mean
    return loss, diagnostics


def diffu_grpo_loss(
    logits,
    tensors: TensorBatch,
    advantages,
    *,
    old_logprobs=None,
    ref_logprobs=None,
    clip_ratio: float = 0.2,
    clip_ratio_high: float | None = None,
    kl_coef: float = 0.0,
    kl_type: str = "k3",
    entropy_coef: float = 0.0,
    extra_mask=None,
    reduction: str | None = None,
) -> tuple:
    """d1 / diffu-GRPO (NeurIPS 2025, arXiv 2504.12216).

    The clipped GRPO surrogate applied to a diffusion log-prob estimator:
    ``r = exp(log pi_theta_est - log pi_old_est)`` where the estimator log-probs
    are recomputed on the trainer by :func:`megarl.dllm.dllm_logprobs` (random
    prompt masking + one unmasking pass over the completion). Pass those values
    through ``old_logprobs`` / ``ref_logprobs``; without them the AR wire values
    are used, which is the ordinary GRPO path.
    """
    kwargs = dict(
        loss_type="diffu_grpo",
        epsilon_low=clip_ratio,
        epsilon_high=clip_ratio_high,
        kl_coef=kl_coef,
        kl_type=kl_type,
        entropy_coef=entropy_coef,
        extra_mask=extra_mask,
        reduction=reduction,
    )
    if old_logprobs is not None:
        kwargs["old_logprobs"] = old_logprobs
    if ref_logprobs is not None:
        kwargs["ref_logprobs"] = ref_logprobs
    return policy_loss(logits, tensors, advantages, **kwargs)


def grpo_loss(
    logits,
    tensors: TensorBatch,
    advantages,
    *,
    clip_ratio: float = 0.2,
    kl_coef: float = 0.0,
    clip_ratio_high: float | None = None,
    reduction: str = "token_mean",
    entropy_coef: float = 0.0,
    extra_mask=None,
    is_mode: str | None = None,
    is_clip_min: float = 0.0,
    is_clip_max: float = 3.0,
) -> tuple:
    """Backwards-compatible GRPO/DAPO entry point over :func:`policy_loss`.

    ``reduction`` selects ``bnpo`` (token_mean), ``grpo`` (seq_mean) or
    ``dapo`` (global_token_mean); ``clip_ratio_high`` enables DAPO clip-higher.
    """
    return policy_loss(
        logits,
        tensors,
        advantages,
        epsilon_low=clip_ratio,
        epsilon_high=clip_ratio_high,
        kl_coef=kl_coef,
        entropy_coef=entropy_coef,
        extra_mask=extra_mask,
        is_mode=is_mode,
        is_clip_min=is_clip_min,
        is_clip_max=is_clip_max,
        reduction=reduction,
    )
