"""REMARL policy — ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language
Models (Shibaev et al., arXiv:2611.22576; code https://github.com/EgorShibaev/remarl).
"""

from __future__ import annotations

from typing import Any

import tensorplay as tp

from .config import ReMaRLConfig


class ReMaRLPolicy(tp.nn.Module):
    """Shared position-wise residual policy ``Delta_phi``.

    ``forward(scores, mask, hidden)`` returns the corrected scores ``z`` with the
    P2' confidence added back. ``hidden`` has shape ``[B, L, hidden_dim]``;
    a zero tensor is a valid fallback when the backbone does not expose states.
    """

    def __init__(self, config: ReMaRLConfig, *, max_len: int = 512) -> None:
        super().__init__()
        self.pos = tp.nn.Embedding(max_len + 1, config.pos_dim)
        self.hidden_norm = tp.nn.LayerNorm(config.hidden_dim)
        self.hidden_proj = tp.nn.Linear(config.hidden_dim, config.proj_dim)
        width = 1 + 1 + config.pos_dim + config.proj_dim
        self.mlp = tp.nn.Sequential(
            tp.nn.Linear(width, config.mlp_hidden),
            tp.nn.GELU(),
            tp.nn.Linear(config.mlp_hidden, config.mlp_hidden),
            tp.nn.GELU(),
            tp.nn.Linear(config.mlp_hidden, 1),
        )
        self.skip = tp.nn.Linear(width, 1, bias=False)
        # Zero-initialise the last layer and the skip so z starts as P2' scores.
        tp.nn.init.zeros_(self.mlp[-1].weight)
        tp.nn.init.zeros_(self.mlp[-1].bias)
        tp.nn.init.zeros_(self.skip.weight)

    def forward(self, scores, mask, hidden):
        batch, length = scores.shape
        position = tp.arange(length, device=scores.device).unsqueeze(0).expand(batch, length)
        projected = self.hidden_proj(self.hidden_norm(hidden))
        features = tp.cat(
            [
                scores.unsqueeze(-1),
                mask.to(scores.dtype).unsqueeze(-1),
                self.pos(position.to(tp.long)),
                projected,
            ],
            dim=-1,
        )
        delta = self.mlp(features).squeeze(-1) + self.skip(features).squeeze(-1)
        return scores + delta


def plackett_luce_logprob(scores, order, valid=None) -> Any:
    """Exact ordered-top-k Plackett--Luce log-probability.

    ``scores`` is ``[B, L]`` and ``order`` is ``[B, K]`` with padded indices.
    ``valid`` masks padded decisions. The Gumbel top-k construction makes the
    ordered sample follow this distribution exactly.
    """
    logits = scores / 1.0
    batch, length = logits.shape
    k = order.shape[-1]
    valid = (
        tp.ones((batch, k), dtype=logits.dtype, device=logits.device)
        if valid is None
        else valid.to(logits.dtype)
    )
    allowed = tp.ones_like(logits)
    total = tp.zeros((batch,), dtype=logits.dtype, device=logits.device)
    for index in range(k):
        candidate = order[:, index : index + 1].to(tp.long)
        masked = tp.where(allowed > 0.5, logits, tp.full_like(logits, -1e9))
        peak = masked.max(dim=-1).values
        log_denom = peak + (masked - peak.unsqueeze(-1)).exp().sum(dim=-1).clamp(min=1e-9).log()
        chosen = logits.gather(-1, candidate).squeeze(-1)
        total = total + valid[:, index] * (chosen - log_denom)
        # Remove the chosen index only after scoring it: the Plackett--Luce
        # denominator contains the current candidate and all later ones.
        onehot = tp.one_hot(candidate.squeeze(-1), length).to(logits.dtype)
        allowed = allowed * (1.0 - onehot * valid[:, index : index + 1])
    return total


def gumbel_topk(scores, k: int, *, tau: float) -> Any:
    """Sample an ordered list of ``k`` positions by Gumbel top-k."""
    noise = -tp.log(-tp.log(tp.rand(scores.shape, device=scores.device).clamp(1e-9, 1.0)))
    perturbed = scores / tau + noise
    _, order = tp.topk(perturbed, k=k, dim=-1)
    return order


def _zero_hidden(batch: int, length: int, dim: int, device: str) -> Any:
    return tp.zeros((batch, length, dim), dtype=tp.float32, device=device)
