"""TIPS policy — TIPS: A Tiny Inference-time Policy for Scaling in Diffusion Language
Models (Zhang et al., arXiv:2610.22099; code https://github.com/zhengyz97/TIPS).
"""

from __future__ import annotations

import math
from typing import Any

import tensorplay as tp
import tensorplay.nn.functional as F


def _rotate_half(x):
    half = x.shape[-1] // 2
    return tp.cat([-x[..., half:], x[..., :half]], dim=-1)


class _TIPSBlock(tp.nn.Module):
    """Pre-norm transformer block with rotary attention."""

    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.norm1 = tp.nn.LayerNorm(hidden)
        self.norm2 = tp.nn.LayerNorm(hidden)
        self.q = tp.nn.Linear(hidden, hidden)
        self.k = tp.nn.Linear(hidden, hidden)
        self.v = tp.nn.Linear(hidden, hidden)
        self.o = tp.nn.Linear(hidden, hidden)
        self.ff = tp.nn.Sequential(
            tp.nn.Linear(hidden, 2 * hidden),
            tp.nn.GELU(),
            tp.nn.Linear(2 * hidden, hidden),
        )

    def forward(self, x, cos, sin):
        h = self.norm1(x)
        batch, length, _ = h.shape
        head_dim = h.shape[-1] // self.heads
        q, k, v = self.q(h), self.k(h), self.v(h)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        q = q.view(batch, length, self.heads, head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, head_dim).transpose(1, 2)
        scores = tp.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        weights = F.softmax(scores, dim=-1)
        mixed = tp.matmul(weights, v).transpose(1, 2).reshape(batch, length, -1)
        x = x + self.o(mixed)
        return x + self.ff(self.norm2(x))


class TIPSPolicy(tp.nn.Module):
    """Tiny per-position visibility policy (two RoPE-DiT blocks).

    ``forward(features, mask, time)`` returns visibility logits ``b`` shaped
    ``[B, L]``. ``features`` has shape ``[B, L, 2]``, ``mask`` is the binary
    mask indicator and ``time`` is the normalised decoding time per row.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 2,
        hidden: int = 64,
        layers: int = 2,
        heads: int = 4,
        time_dim: int = 16,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.time_dim = int(time_dim)
        self.hidden = int(hidden)
        self.input = tp.nn.Linear(feature_dim + 1 + time_dim, hidden)
        self.blocks = tp.nn.ModuleList(
            [_TIPSBlock(hidden, heads) for _ in range(layers)]
        )
        self.norm = tp.nn.LayerNorm(hidden)
        self.head = tp.nn.Linear(hidden, 1)
        self.pos = tp.nn.Embedding(max_len + 1, hidden)

    def _rope(self, length: int, dtype, device) -> tuple:
        half = self.hidden // 2
        position = tp.arange(length, device=device, dtype=tp.float32)
        inverse = tp.exp(-math.log(10000.0) * tp.arange(half, device=device, dtype=tp.float32) * (2.0 / self.hidden))
        angles = tp.matmul(
            position.unsqueeze(-1).to(tp.float32), inverse.unsqueeze(0)
        )
        cos = tp.cat([angles.cos(), angles.cos()], dim=-1).to(dtype).unsqueeze(0)
        sin = tp.cat([angles.sin(), angles.sin()], dim=-1).to(dtype).unsqueeze(0)
        return cos, sin

    def forward(self, features, mask, time):
        batch, length, _ = features.shape
        device = features.device
        half = self.time_dim // 2
        idx = tp.arange(half, device=device, dtype=tp.float32)
        inverse = tp.exp(-math.log(10000.0) * idx * (2.0 / self.time_dim))
        angles = tp.matmul(
            tp.as_tensor(time, dtype=tp.float32, device=device).reshape(batch, 1),
            inverse.reshape(1, half),
        )
        time_embed = tp.cat([angles.sin(), angles.cos()], dim=-1)
        time_embed = time_embed.unsqueeze(1).expand(-1, length, -1)
        position = tp.arange(length, device=device).unsqueeze(0).expand(batch, length)
        x = tp.cat([features, mask.to(features.dtype).unsqueeze(-1), time_embed], dim=-1)
        x = self.input(x) + self.pos(position.to(tp.long))
        cos, sin = self._rope(length, features.dtype, device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.norm(x)).squeeze(-1)


def tips_visibility_features(logits, canvas, *, noise_token_id: int) -> tuple:
    """Return ``(features, mask, greedy_ids)`` for a canvas logits tensor.

    Logits are ``[B, L, V]`` and ``canvas`` is ``[B, L]``. The mask token is
    removed from the proposal distribution before the active token and its
    strongest alternative are computed (paper eqs. 4--6).
    """
    vocab = logits.shape[-1]
    if 0 <= noise_token_id < vocab:
        is_noise = tp.arange(vocab, device=logits.device) == noise_token_id
        logits = logits.masked_fill(is_noise.reshape(1, 1, vocab), -1e9)
    probs = F.softmax(logits.float(), dim=-1)
    greedy = probs.argmax(dim=-1)
    is_mask = (canvas == noise_token_id).to(tp.long)
    active = tp.where(is_mask.to(tp.bool), greedy, canvas).to(tp.long)
    active_prob = probs.gather(-1, active.unsqueeze(-1)).squeeze(-1)
    active_onehot = tp.one_hot(active, vocab).to(probs.dtype)
    alternative = (probs * (1.0 - active_onehot)).max(dim=-1).values
    active_prob = active_prob.clamp(1e-6, 1.0 - 1e-6)
    alternative = alternative.clamp(min=1e-6)
    odds = tp.log(active_prob / (1.0 - active_prob))
    margin = tp.log(active_prob / alternative)
    features = tp.stack([odds, margin], dim=-1)
    return features, is_mask, greedy


def calibrate_tips_shift(logits, kappa: float, *, iterations: int = 12) -> Any:
    """Solve ``mean(sigmoid(logits + delta)) == kappa`` by Newton iteration."""
    delta = tp.zeros((logits.shape[0],), dtype=logits.dtype, device=logits.device)
    target = tp.full_like(delta, float(kappa))
    for _ in range(iterations):
        probs = F.sigmoid(logits + delta.unsqueeze(-1))
        error = probs.mean(dim=-1) - target
        slope = (probs * (1.0 - probs)).mean(dim=-1).clamp(min=1e-4)
        delta = delta - error / slope
    return delta.detach()


def tips_bernoulli_logprob(rho, actions) -> Any:
    """Factorised Bernoulli log-probability of the visibility actions."""
    return (
        actions.to(rho.dtype) * F.logsigmoid((rho - 1e-6).clamp(min=1e-6))
        + (1.0 - actions.to(rho.dtype))
        * F.logsigmoid(-(rho.clamp(min=1e-6)))
    ).sum(dim=-1)
