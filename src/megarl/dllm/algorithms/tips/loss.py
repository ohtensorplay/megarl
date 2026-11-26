"""TIPS loss — TIPS: A Tiny Inference-time Policy for Scaling in Diffusion Language Models
(Zhang et al., arXiv:2610.22099; code https://github.com/zhengyz97/TIPS).
"""

from __future__ import annotations

from typing import Any

import tensorplay as tp
import tensorplay.nn.functional as F

from .batching import TIPSTensorBatch
from .policy import calibrate_tips_shift, tips_bernoulli_logprob


def tips_loss(
    logits,
    tensors: TIPSTensorBatch,
    advantages,
    *,
    old_logprobs=None,
    clip_low: float = 0.5,
    clip_high: float = 1.5,
    calibration_iters: int = 12,
) -> tuple:
    """Clipped GRPO objective over factorised Bernoulli visibility actions."""
    if old_logprobs is None:
        raise ValueError("tips_loss requires behaviour log-probabilities")
    adv = (
        advantages.reshape(-1)
        if hasattr(advantages, "reshape")
        else tp.tensor(list(advantages), dtype=tp.float32, device=logits.device)
    )
    delta = calibrate_tips_shift(logits, 0.5, iterations=calibration_iters)
    # The calibration scalar enforces a budget constraint; treat it as a
    # constant so the policy gradient only differentiates local preferences.
    rho = F.sigmoid(logits + delta.detach().unsqueeze(-1)).clamp(1e-6, 1.0 - 1e-6)
    logp = tips_bernoulli_logprob(rho, tensors.tips_actions)
    ratio = (logp - old_logprobs).exp()
    clipped = ratio.clamp(clip_low, clip_high)
    objective = tp.minimum(ratio * adv, clipped * adv)
    loss = -objective.mean()
    diagnostics = {
        "loss": loss,
        "tips_mean_ratio": ratio.mean(),
        "tips_clip_fraction": ((ratio < clip_low) | (ratio > clip_high)).to(tp.float32).mean(),
        "tips_mean_advantage": adv.mean(),
    }
    return loss, diagnostics


def tips_visibility_utility(
    visible_ids,
    reference_ids,
    active_prob,
    alternative_prob,
    is_mask,
) -> Any:
    """Reference-derived visibility utility in ``[-1, 1]`` (paper eq. 10)."""
    matches = (visible_ids == reference_ids).to(tp.float32)
    visible_utility = tp.where(matches > 0.5, tp.ones_like(matches), -tp.ones_like(matches))
    active = active_prob.clamp(1e-6, 1.0 - 1e-6)
    alternative = alternative_prob.clamp(min=1e-6)
    masked_utility = (active - alternative) / (active + alternative)
    return tp.where(is_mask.to(tp.bool), masked_utility, visible_utility)


def tips_warmup_loss(
    logits,
    targets,
    *,
    beta: float = 4.0,
    delta_star: float = 0.0,
) -> tuple:
    """Binary cross-entropy against stop-gradient visibility targets."""
    target = targets.detach().clamp(0.0, 1.0)
    calibrated = F.sigmoid(beta * target.logit() + delta_star).detach()
    loss = F.binary_cross_entropy_with_logits(logits, calibrated)
    return loss, {
        "loss": loss,
        "tips_warmup_target_mean": calibrated.mean(),
    }
