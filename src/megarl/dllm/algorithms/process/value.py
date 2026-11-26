"""Stepwise dLLM credit assignment (TraceRL ICLR 2026 arXiv 2509.06949; Schulman et al.
arXiv 1506.02438); see package docstring for details."""

from __future__ import annotations

from typing import Any

import tensorplay as tp


class DLLMValueHead(tp.nn.Module):
    """Learned step-value model over the (noisy) canvas tokens.

    Pools the embeddings of the valid canvas positions and maps them to one
    scalar per replay row. ``feature_fn`` can substitute richer features (e.g.
    the denoiser's penultimate hidden states) when the backbone exposes them.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        hidden: int = 64,
        feature_fn: Any | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or hidden <= 0:
            raise ValueError("vocab_size and hidden must be positive")
        self.vocab_size = int(vocab_size)
        self.feature_fn = feature_fn
        self.embed = tp.nn.Embedding(self.vocab_size, hidden)
        self.mlp = tp.nn.Sequential(
            tp.nn.Linear(hidden, hidden),
            tp.nn.Tanh(),
            tp.nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, *, noisy_ids=None, mask=None):
        ids = noisy_ids if noisy_ids is not None else input_ids
        features = self.feature_fn(ids) if self.feature_fn is not None else self.embed(ids)
        if mask is not None:
            keep = mask.to(features.dtype).unsqueeze(-1)
            features = features * keep
            denom = keep.sum(dim=1).clamp(min=1.0)
        else:
            denom = tp.full(
                (features.shape[0], 1), float(features.shape[1]),
                dtype=features.dtype, device=features.device,
            )
        pooled = features.sum(dim=1) / denom
        return self.mlp(pooled).squeeze(-1)


def dllm_value_fn(model, tensors) -> Any:
    """Trainer hook: broadcast the row's process value over its canvas."""
    values = model(
        tensors.input_ids,
        noisy_ids=tensors.dllm_noisy_ids,
        mask=tensors.loss_mask,
    )
    width = tensors.loss_mask.shape[1]
    return values.unsqueeze(-1).expand(-1, width)
