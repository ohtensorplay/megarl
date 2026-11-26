"""PURL model — When Should Diffusion Language Models Reconsider Their Commitments? (Zheng
et al., arXiv:2611.11551v1; withdrawn after v1).
"""

from __future__ import annotations

from typing import Any, Callable

import tensorplay as tp

from ....dllm import default_adapter
from .config import PURLConfig


class PURLPlanners(tp.nn.Module):
    """Position-wise Bernoulli logits for edit and unmask selection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = tp.nn.LayerNorm(dim)
        self.edit = tp.nn.Sequential(tp.nn.Linear(dim, dim), tp.nn.GELU(), tp.nn.Linear(dim, 1))
        self.unmask = tp.nn.Sequential(tp.nn.Linear(dim, dim), tp.nn.GELU(), tp.nn.Linear(dim, 1))

    def forward(self, features):
        hidden = self.norm(features)
        return self.edit(hidden).squeeze(-1), self.unmask(hidden).squeeze(-1)


class PURLModel(tp.nn.Module):
    """Wrap a dLLM backbone with PURL's two planner heads.

    ``forward(noisy_ids, tensors, config)`` returns
    ``(token_logits, edit_logits, unmask_logits)``. When ``feature_fn`` is
    omitted the token logits themselves are used as planner features, which
    makes the module work with any backbone exposing ``[B, L, V]`` logits.
    """

    def __init__(
        self,
        backbone: Any,
        config: PURLConfig,
        *,
        adapter: Any | None = None,
        feature_fn: Callable[[Any, Any, Any], Any] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.adapter = adapter or default_adapter(config)
        self.feature_fn = feature_fn
        self.planners = PURLPlanners(config.planner_dim)

    def _backbone_output(self, noisy_ids):
        try:
            return self.backbone(noisy_ids, output_hidden_states=True)
        except TypeError:
            return self.backbone(noisy_ids)

    def _features(self, noisy_ids, logits):
        if self.feature_fn is not None:
            features = self.feature_fn(self.backbone, noisy_ids, logits)
        else:
            features = logits
        if features.shape[-1] != self.config.planner_dim:
            raise ValueError(
                f"planner features have dim {features.shape[-1]}, "
                f"expected {self.config.planner_dim}"
            )
        return features

    def forward(self, noisy_ids, tensors, config: PURLConfig | None = None):
        cfg = config or self.config
        logits = self.adapter(self.backbone, noisy_ids, tensors, cfg)
        features = self._features(noisy_ids, logits)
        edit, unmask = self.planners(features)
        return logits, edit, unmask
