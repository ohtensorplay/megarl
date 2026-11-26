"""dOPSD, Phuong Tuan Dat, Qi Li, Xinchao Wang, arXiv 2607.04428; code
https://github.com/tuandattt/dOPSD; see package docstring for details."""

from __future__ import annotations

from typing import Any

import tensorplay.nn.functional as F


def generalized_jsd(p_teacher, p_student, weight: float) -> Any:
    """Generalised JS divergence (paper eq. 2), gradients into ``p_student`` only."""
    eps = 1e-8
    teacher = p_teacher.detach().clamp(min=eps)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True)
    student = p_student.clamp(min=eps)
    student = student / student.sum(dim=-1, keepdim=True)
    mixture = weight * teacher + (1.0 - weight) * student
    mixture = mixture.clamp(min=eps)
    kl_teacher = (teacher * (teacher.log() - mixture.log())).sum(dim=-1)
    kl_student = (student * (student.log() - mixture.log())).sum(dim=-1)
    return weight * kl_teacher + (1.0 - weight) * kl_student


def dopsd_loss(
    teacher_probs,
    student_logits,
    position_valid,
    keep,
    *,
    temperature: float = 1.0,
    js_weight: float = 0.5,
) -> tuple:
    """Token-level JSD averaged over masked positions and verified rollouts."""
    logits = student_logits.float() / temperature
    student = F.softmax(logits, dim=-1)
    per_position = generalized_jsd(teacher_probs, student, js_weight)
    valid = position_valid.to(per_position.dtype)
    per_row = (per_position * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1.0)
    loss = (per_row * keep.to(per_row.dtype)).mean()
    diagnostics = {
        "loss": loss,
        "dopsd_jsd": per_row.detach().mean(),
        "dopsd_keep_fraction": keep.detach().mean(),
        "dopsd_student_entropy": (
            -(student * student.clamp(min=1e-8).log()).sum(dim=-1).detach().mean()
        ),
    }
    return loss, diagnostics
