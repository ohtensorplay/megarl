"""Train/inference mismatch correction for importance ratios.

vLLM returns the logprob of the token *under the sampling distribution*, while
the trainer computes it under the training distribution. Ignoring the gap biases
every importance ratio. TRL exposes four explicit modes; we implement the same
four with NaN neutralization, symmetric/clipped handling and per-token or
per-sequence normalization.
"""

from __future__ import annotations

import tensorplay as tp

MODES = ("token_truncate", "token_mask", "sequence_truncate", "sequence_mask")


def importance_correction(
    sampling_logprobs,
    behavior_logprobs,
    *,
    mode: str = "sequence_mask",
    clip_min: float = 0.0,
    clip_max: float = 3.0,
    mask=None,
):
    """Return a multiplicative correction ``w`` for the importance ratio.

    ``sampling_logprobs`` come from the rollout engine, ``behavior_logprobs``
    from the trainer's frozen old policy. Missing/NaN entries are neutral (w=1)
    rather than poisoning the ratio. ``mask`` selects response tokens for the
    sequence-level variants.
    """
    if mode not in MODES:
        raise ValueError(f"unknown importance-sampling mode: {mode}")
    if clip_min > clip_max:
        raise ValueError("clip_min must be <= clip_max")
    log_ratio = behavior_logprobs - sampling_logprobs
    finite = tp.isfinite(log_ratio)
    log_ratio = tp.where(finite, log_ratio, tp.zeros_like(log_ratio))
    raw = log_ratio.exp()

    if mode.startswith("token"):
        scores = raw
    else:
        if mask is None:
            raise ValueError("sequence-level correction requires a token mask")
        keep = mask.to(raw.dtype)
        denom = keep.sum(-1, keepdim=True).clamp(min=1.0)
        seq_log = (log_ratio * keep).sum(-1, keepdim=True) / denom
        scores = seq_log.exp()

    low = scores >= clip_min
    high = scores <= clip_max
    valid = low & high
    if mode.endswith("truncate"):
        result = tp.where(valid, scores, tp.where(scores > clip_max, tp.full_like(scores, clip_max), tp.full_like(scores, clip_min)))
    else:
        result = tp.where(valid, scores, tp.zeros_like(scores))
    return tp.where(finite, result, tp.ones_like(result))
