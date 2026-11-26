"""Flow-GRPO arXiv 2505.05470 and DiffNFT (ROLL diffusion pipeline
refs/ROLL/roll/pipeline/diffusion); see package docstring for details."""

from __future__ import annotations

import math

import tensorplay as tp

from ....kl import approx_kl
from .batching import _reshape_per_sample
from .config import FlowConfig

_LOG_2PI = math.log(2.0 * math.pi)


def flow_step_stats(xt, model_output, sigma, sigma_next, config: FlowConfig):
    """Recompute ``(mu, std)`` of the Gaussian reverse transition.

    Matches ``FlowMatchSDEDiscreteScheduler.sample_previous_step`` in ROLL:

    * ``sde``: ``std_t = sqrt(sigma / (1 - sigma)) * noise_level``,
      ``mu = xt * (1 + std_t^2 dt / (2 sigma)) + out * (1 + std_t^2 (1-sigma)
      / (2 sigma)) dt`` and ``std = std_t * sqrt(-dt)``.
    * ``cps``: ``std = sigma_next * sin(noise_level * pi / 2)`` and the mean is
      the interpolated ``x0`` / ``x1`` prediction.

    ``xt`` and ``model_output`` are ``[B, K, D]`` (or broadcastable);
    ``sigma``/``sigma_next`` are ``[B, K]``.
    """
    sigma = sigma.clamp(min=1e-6, max=1.0)
    sigma_next = sigma_next.clamp(min=0.0, max=1.0)
    s = _reshape_per_sample(sigma, xt)
    s_next = _reshape_per_sample(sigma_next, xt)
    dt = s_next - s
    if config.sde_type == "sde":
        denom = tp.where(s == 1.0, tp.full_like(s, 0.999), s)
        std_t = tp.sqrt(s / (1.0 - denom).clamp(min=1e-6)) * config.noise_level
        std = std_t * tp.sqrt((-dt).clamp(min=0.0))
        mu = xt * (1.0 + std_t**2 / (2.0 * s) * dt) + model_output * (
            1.0 + std_t**2 * (1.0 - s) / (2.0 * s)
        ) * dt
    else:
        std = s_next * math.sin(config.noise_level * math.pi / 2.0)
        x0_pred = xt - s * model_output
        noise_pred = xt + model_output * (1.0 - s)
        mu = x0_pred * (1.0 - s_next) + noise_pred * tp.sqrt(
            (s_next**2 - std**2).clamp(min=0.0)
        )
    return mu, std.clamp(min=1e-6)


def flow_step_logprob(
    next_latent,
    xt,
    model_output,
    sigma,
    sigma_next,
    config: FlowConfig,
    *,
    reduction: str = "mean",
) -> tp.Tensor:
    """Exact log-probability of ``x_{t-1}`` under the current denoiser.

    Returns ``[B, K]`` after reducing the latent dimensions (ROLL averages, so
    the log-prob is scale-invariant to the image/video latent size). ``cps``
    drops the Gaussian normalising constants exactly as the reference does.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"unknown flow logprob reduction: {reduction}")
    mu, std = flow_step_stats(xt, model_output, sigma, sigma_next, config)
    residual = next_latent.detach() - mu
    logp = -residual**2 / (2.0 * std**2) - tp.log(std)
    if config.sde_type == "sde":
        logp = logp - 0.5 * _LOG_2PI
    if reduction != "none" and len(logp.shape) > 2:
        dims = tuple(range(2, len(logp.shape)))
        logp = logp.sum(dim=dims) if reduction == "sum" else logp.mean(dim=dims)
    return logp


def flow_grpo_loss(
    new_logp,
    old_logp,
    advantages,
    mask,
    *,
    clip_low: float = 0.2,
    clip_high: float | None = None,
    advantage_clip: float | None = None,
    kl_coef: float = 0.0,
    kl_type: str = "k2",
    ref_logp=None,
) -> tuple[tp.Tensor, dict[str, tp.Tensor]]:
    """Clipped per-step FlowGRPO objective (ROLL's ``_loss_func``)."""
    high = clip_high if clip_high is not None else clip_low
    adv = advantages
    if len(adv.shape) == 1:
        adv = adv.unsqueeze(-1)
    if advantage_clip is not None:
        adv = adv.clamp(min=-advantage_clip, max=advantage_clip)
    mask = mask.to(adv.dtype)
    log_ratio = (new_logp - old_logp).clamp(min=-20.0, max=20.0)
    ratio = log_ratio.exp()
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_low, 1.0 + high) * adv
    per_token = -tp.minimum(unclipped, clipped)
    ref = ref_logp if ref_logp is not None else old_logp
    kl = approx_kl(new_logp, ref, kl_type=kl_type)
    denom = mask.sum().clamp(min=1.0)
    loss = ((per_token + kl_coef * kl) * mask).sum() / denom
    diagnostics = {
        "flow_pg_loss": (per_token * mask).sum() / denom,
        "flow_ratio_mean": (ratio * mask).sum() / denom,
        "flow_clip_fraction": (
            ((ratio > 1.0 + high) | (ratio < 1.0 - clip_low)).to(tp.float32) * mask
        ).sum() / denom,
        "flow_kl": (kl * mask).sum() / denom,
    }
    return loss, diagnostics


def diffnft_loss(
    x0,
    xt,
    sigma,
    new_pred,
    old_pred,
    reward_prob,
    *,
    beta: float,
    adv_clip_max: float,
    ref_pred=None,
    ref_kl_coef: float = 0.0,
    use_adaptive_weight: bool = False,
) -> tuple[tp.Tensor, dict[str, tp.Tensor]]:
    """Reward-weighted x0 reconstruction loss from ROLL's ``ActorNFTWorker``.

    ``positive = beta * new + (1-beta) * old`` and
    ``negative = (1+beta) * old - beta * new`` anchor both targets at the frozen
    behaviour prediction, so the loss is well behaved under off-policy updates.
    """
    if beta <= 0:
        raise ValueError("diffnft beta must be positive")
    dims = tuple(range(1, len(x0.shape)))
    t = _reshape_per_sample(sigma, x0)
    r = _reshape_per_sample(reward_prob, x0).to(x0.dtype)
    positive = beta * new_pred + (1.0 - beta) * old_pred
    negative = (1.0 + beta) * old_pred - beta * new_pred
    x0_pos = xt - t * positive
    x0_neg = xt - t * negative
    pos_sq = (x0_pos - x0) ** 2
    neg_sq = (x0_neg - x0) ** 2
    if use_adaptive_weight:
        pos_scale = pos_sq.detach().mean(dim=dims, keepdim=True).clamp(min=1e-5)
        neg_scale = neg_sq.detach().mean(dim=dims, keepdim=True).clamp(min=1e-5)
        pos_sq = pos_sq / pos_scale
        neg_sq = neg_sq / neg_scale
    pos_loss = (r * pos_sq).mean()
    neg_loss = ((1.0 - r) * neg_sq).mean()
    loss = (pos_loss + neg_loss) / beta * adv_clip_max
    diagnostics = {
        "nft_pos_loss": pos_loss.detach(),
        "nft_neg_loss": neg_loss.detach(),
        "nft_old_deviate": ((new_pred.detach() - old_pred.detach()) ** 2).mean(),
    }
    if ref_pred is not None:
        ref_kl = ((new_pred - ref_pred.to(new_pred.dtype)) ** 2).mean()
        loss = loss + ref_kl_coef * ref_kl
        diagnostics["nft_ref_kl"] = ref_kl.detach()
    return loss, diagnostics
