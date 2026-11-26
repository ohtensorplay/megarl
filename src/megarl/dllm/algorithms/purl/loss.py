"""PURL loss — When Should Diffusion Language Models Reconsider Their Commitments? (Zheng
et al., arXiv:2611.11551v1; withdrawn after v1).
"""

from __future__ import annotations

from typing import Any

import tensorplay as tp
import tensorplay.nn.functional as F

from .batching import PURLTensorBatch


def _weighted_bce(logits, targets, candidates, *, beta: float) -> Any:
    """Candidate-averaged BCE with a positive-class weight."""
    loss = -(
        beta * targets * F.logsigmoid(logits)
        + (1.0 - targets) * F.logsigmoid(-logits)
    )
    return (loss * candidates).sum() / candidates.sum().clamp(min=1.0)


def purl_loss(
    token_logits,
    edit_logits,
    unmask_logits,
    tensors: PURLTensorBatch,
    *,
    beta_e: float = 4.0,
    beta_u: float = 4.0,
    lambda_e: float = 1.0,
    lambda_u: float = 1.0,
    lambda_theta: float = 1.0,
    w_mask: float = 1.0,
    w_noisy: float = 1.0,
    w_correct: float = 1.0,
    unmask_budget: int = 1,
) -> tuple:
    """PURL's factorised joint objective (paper eq. 6)."""
    span = tensors.purl_prompt_span
    L = tensors.purl_canvas_len
    logits = token_logits[:, span : span + L, :].float()
    log_probs = F.log_softmax(logits, dim=-1)
    reference = tensors.purl_reference_ids
    nll = -log_probs.gather(-1, reference.unsqueeze(-1)).squeeze(-1)

    def group_ce(weights):
        return (nll * weights).sum() / weights.sum().clamp(min=1.0)

    l_theta = (
        w_mask * group_ce(tensors.purl_mask)
        + w_noisy * group_ce(tensors.purl_noisy)
        + w_correct * group_ce(tensors.purl_correct)
    )
    edit_logits = edit_logits[:, span : span + L]
    unmask_logits = unmask_logits[:, span : span + L]
    l_e = _weighted_bce(
        edit_logits,
        tensors.purl_edit_labels,
        tensors.purl_edit_candidates,
        beta=beta_e,
    )
    # Teacher unmask target: stop-gradient margin of the reference token over
    # its strongest alternative, TopK within the masked candidate set.
    onehot = tp.one_hot(reference, logits.shape[-1]).to(logits.dtype)
    alternative = logits.masked_fill(onehot > 0.5, -1e9).max(dim=-1).values
    active = logits.gather(-1, reference.unsqueeze(-1)).squeeze(-1)
    margin = (active - alternative).detach()
    allowed = tensors.purl_unmask_candidates
    masked_margin = tp.where(allowed > 0.5, margin, tp.full_like(margin, -1e9))
    k = min(unmask_budget, int(allowed.to(tp.long).sum(dim=-1).max().flatten()[0]))
    if k > 0:
        _, top = tp.topk(masked_margin, k=k, dim=-1)
        unmask_labels = tp.zeros_like(allowed).scatter(
            -1, top, tp.ones_like(top).to(allowed.dtype)
        )
    else:
        unmask_labels = tp.zeros_like(allowed)
    l_u = _weighted_bce(
        unmask_logits, unmask_labels, allowed, beta=beta_u
    )
    loss = lambda_e * l_e + lambda_u * l_u + lambda_theta * l_theta
    return (
        loss,
        {
            "loss": loss,
            "purl_edit_loss": l_e,
            "purl_unmask_loss": l_u,
            "purl_token_loss": l_theta,
            "purl_edit_fraction": tensors.purl_edit_labels.mean(),
        },
    )
