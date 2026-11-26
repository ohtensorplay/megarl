"""T3D, Tunyu Zhang et al., NeurIPS 2026 arXiv 2602.12262; code
https://github.com/Tyrion58/T3D; see package docstring for details."""

from __future__ import annotations

import tensorplay as tp
import tensorplay.nn.functional as F


def _gather_logp(logits, positions, targets):
    """Log-prob of ``targets`` at flattened ``positions`` of ``logits``."""
    vocab = int(logits.shape[-1])
    index = positions.unsqueeze(-1).expand(
        positions.shape[0], positions.shape[1], vocab
    )
    selected = tp.gather(logits.float(), 1, index)
    tokens = tp.gather(targets, 1, positions).unsqueeze(-1)
    return tp.gather(selected, -1, tokens).squeeze(-1)


def t3d_loss(
    student_logits,
    reference_logits,
    input_ids,
    ref_targets,
    mask_positions,
    mask_valid,
    path_positions,
    path_valid,
    path_weights,
    *,
    ddo_temperature: float = 1.0,
    ratio_clip: float = 20.0,
    path_weight: float = 1.0,
) -> tuple:
    """T3D's DDO plus weighted path-consistency objective (paper eq. 10)."""
    student_logp = F.log_softmax(student_logits, dim=-1)
    with tp.no_grad():
        reference_logp = F.log_softmax(reference_logits.float(), dim=-1)
    valid = mask_valid.to(tp.float32)
    pos_student = _gather_logp(student_logp, mask_positions, input_ids)
    pos_reference = _gather_logp(reference_logp, mask_positions, input_ids)
    neg_student = _gather_logp(student_logp, mask_positions, ref_targets)
    neg_reference = _gather_logp(reference_logp, mask_positions, ref_targets)
    positive_ratio = ((pos_student - pos_reference) * valid).sum(dim=-1)
    negative_ratio = ((neg_student - neg_reference) * valid).sum(dim=-1)
    scale = 1.0 / ddo_temperature
    positive = -F.logsigmoid(positive_ratio.clamp(-ratio_clip, ratio_clip) * scale)
    negative = -F.logsigmoid((-negative_ratio).clamp(-ratio_clip, ratio_clip) * scale)
    ddo = (positive + negative).mean()

    path_mask = path_valid.to(tp.float32) * path_weights
    path_logp = _gather_logp(student_logp, path_positions, input_ids)
    path_sum = (-(path_logp * path_mask).sum() / path_mask.sum().clamp(min=1.0))
    loss = ddo + path_weight * path_sum
    diagnostics = {
        "loss": loss,
        "t3d_ddo": ddo.detach(),
        "t3d_path": path_sum.detach(),
        "t3d_positive_ratio": positive_ratio.detach().mean(),
        "t3d_negative_ratio": negative_ratio.detach().mean(),
    }
    return loss, diagnostics
