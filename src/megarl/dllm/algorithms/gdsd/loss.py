"""GDSD, Xiaohang Tang et al., arXiv 2605.29398; code https://github.com/GaryBall/GDSD;
see package docstring for details."""

from __future__ import annotations

import math
from typing import Any

import tensorplay as tp
import tensorplay.nn.functional as F

from ....dllm import default_adapter
from .batching import GDSDTensorBatch
from .config import GDSDConfig


def gdsd_token_logprobs(
    model: Any,
    tensors: GDSDTensorBatch,
    config: GDSDConfig,
    *,
    adapter: Any | None = None,
) -> Any:
    """Per-token log-prob of the clean token on a corrupted canvas.

    With ``tlc=True`` this returns the centralised logit at the target token
    (``logit - mean_v logit``), which is the paper's normalisation-free log p̄.
    """
    forward = adapter or default_adapter(config)
    logits = forward(model, tensors.dllm_noisy_ids, tensors, config).float()
    if config.tlc:
        logits = logits - logits.mean(dim=-1, keepdim=True)
    log_probs = F.log_softmax(logits, dim=-1)
    return tp.gather(log_probs, -1, tensors.input_ids.unsqueeze(-1)).squeeze(-1)


def gdsd_loss(
    logits,
    tensors: GDSDTensorBatch,
    advantages,
    *,
    old_logprobs,
    ref_logprobs=None,
    psi: float = 1.0,
    beta: float = 0.1,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    tlc: bool = True,
) -> tuple:
    """GDSD's squared self-distillation objective (paper eq. 24).

    ``advantages`` are group-centred rewards (no std division in the paper).
    The sequence log-prob estimate of each corruption view is divided by its
    corruption probability, so averaging the two views is unbiased.
    """
    values = logits.float()
    if tlc:
        values = values - values.mean(dim=-1, keepdim=True)
    log_probs = F.log_softmax(values, dim=-1)
    new_token = tp.gather(log_probs, -1, tensors.input_ids.unsqueeze(-1)).squeeze(-1)
    mask = tensors.loss_mask.to(tp.float32)
    weight = tensors.gdsd_weights.unsqueeze(-1)
    length = tensors.gdsd_lengths.unsqueeze(-1).clamp(min=1.0)

    def sequence(logp):
        return (logp * mask).sum(dim=-1) / weight.squeeze(-1)

    new_seq = sequence(new_token)
    old_seq = sequence(old_logprobs)
    log_ratio = (new_seq - old_seq) / length.squeeze(-1)
    low = math.log(max(1e-6, 1.0 - epsilon_low))
    high = math.log(1.0 + max(0.0, epsilon_high))
    clipped = log_ratio.clamp(min=low, max=high)
    adv = (
        advantages.reshape(-1)
        if hasattr(advantages, "reshape")
        else tp.tensor(list(advantages), dtype=tp.float32, device=logits.device)
    )
    guidance_error = clipped - psi * adv
    mse = (guidance_error * guidance_error).mean()

    ref_term = tp.zeros((), dtype=mse.dtype, device=mse.device)
    if ref_logprobs is not None and beta > 0:
        ref_seq = sequence(ref_logprobs)
        ref_ratio = (new_seq - ref_seq) / length.squeeze(-1)
        ref_term = beta * (ref_ratio * ref_ratio).mean()

    loss = mse + ref_term
    diagnostics = {
        "loss": loss,
        "gdsd_mse": mse.detach(),
        "gdsd_ref_kl": ref_term.detach(),
        "gdsd_clip_fraction": (
            ((log_ratio > high) | (log_ratio < low)).to(tp.float32).mean()
        ),
        "mean_ratio": log_ratio.clamp(-20.0, 20.0).exp().detach().mean(),
    }
    return loss, diagnostics
