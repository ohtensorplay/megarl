"""REMARL loss — ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language Models
(Shibaev et al., arXiv:2611.22576; code https://github.com/EgorShibaev/remarl).
"""

from __future__ import annotations

import tensorplay as tp

from .batching import ReMaRLTensorBatch


def remarl_loss(
    scores,
    tensors: ReMaRLTensorBatch,
    advantages,
    *,
    old_logprobs=None,
    tau: float = 1.0,
    clip_eps: float = 0.2,
) -> tuple:
    """Clipped GRPO objective over Plackett--Luce position decisions."""
    if old_logprobs is None:
        raise ValueError("remarl_loss requires behaviour log-probabilities")
    adv = (
        advantages.reshape(-1)
        if hasattr(advantages, "reshape")
        else tp.tensor(list(advantages), dtype=tp.float32, device=scores.device)
    )
    logits = scores / tau
    batch, length = logits.shape
    k = tensors.remarl_actions.shape[-1]
    valid = tensors.remarl_action_valid
    allowed = tp.ones_like(logits)
    total = tp.zeros((batch,), dtype=logits.dtype, device=logits.device)
    for index in range(k):
        candidate = tensors.remarl_actions[:, index : index + 1]
        masked = tp.where(allowed > 0.5, logits, tp.full_like(logits, -1e9))
        peak = masked.max(dim=-1).values
        log_denom = peak + (masked - peak.unsqueeze(-1)).exp().sum(dim=-1).clamp(min=1e-9).log()
        chosen = logits.gather(-1, candidate).squeeze(-1)
        total = total + valid[:, index] * (chosen - log_denom)
        onehot = tp.one_hot(candidate.squeeze(-1), length).to(logits.dtype)
        allowed = allowed * (1.0 - onehot * valid[:, index : index + 1].unsqueeze(-1))
    ratio = (total - old_logprobs).exp()
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    loss = -tp.minimum(ratio * adv, clipped * adv).mean()
    return (
        loss,
        {
            "loss": loss,
            "remarl_mean_ratio": ratio.mean(),
            "remarl_clip_fraction": (
                (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
            ).to(tp.float32).mean(),
            "remarl_mean_logprob": total.mean(),
        },
    )
