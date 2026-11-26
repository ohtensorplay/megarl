"""PURL trainer — When Should Diffusion Language Models Reconsider Their Commitments?
(Zheng et al., arXiv:2611.11551v1; withdrawn after v1).
"""

from __future__ import annotations

import dataclasses

from ....trainer import TensorPlayTrainer, TrainerConfig
from .batching import PURLBatch, PURLTensorBatch, slice_purl_batch
from .config import PURLConfig
from .loss import purl_loss
from .model import PURLModel


class PURLTrainer(TensorPlayTrainer):
    """Joint trainer for PURL's token predictor and both planners."""

    tensor_batch_cls = PURLTensorBatch

    def __init__(
        self,
        policy: PURLModel,
        config: TrainerConfig | None = None,
        *,
        purl_config: PURLConfig,
        **kwargs,
    ) -> None:
        base = config or TrainerConfig()
        if dataclasses.is_dataclass(base):
            base = dataclasses.replace(base, loss_type="grpo", compile_policy=False)
        if base.loss_type != "grpo":
            raise ValueError("PURLTrainer requires loss_type='grpo'")
        kwargs["ref_model"] = None
        kwargs["ref_fn"] = None
        self.purl_config = purl_config

        def forward(model, tensors):
            return model(tensors.dllm_noisy_ids, tensors, purl_config)

        super().__init__(policy, base, forward_fn=forward, **kwargs)

    def _slice_batch(self, batch, start: int, stop: int):
        if not isinstance(batch, PURLBatch):
            raise TypeError("PURLTrainer requires a PURLBatch")
        return slice_purl_batch(batch, start, stop)

    def _rl_aux(self, batch, advantages, *, kl_coef: float):
        if not isinstance(batch, PURLBatch):
            raise TypeError("PURLTrainer requires a PURLBatch")
        return {}, batch.size, None, None, None

    def _loss_kwargs(self, *, clip_ratio: float, kl_coef: float) -> dict:
        cfg = self.purl_config
        return {
            "beta_e": cfg.beta_e,
            "beta_u": cfg.beta_u,
            "lambda_e": cfg.lambda_e,
            "lambda_u": cfg.lambda_u,
            "lambda_theta": cfg.lambda_theta,
            "w_mask": cfg.w_mask,
            "w_noisy": cfg.w_noisy,
            "w_correct": cfg.w_correct,
            "unmask_budget": cfg.unmask_budget,
        }

    def _rl_loss(self, output, tensors: PURLTensorBatch, **kwargs):
        logits, edit_logits, unmask_logits = output
        return purl_loss(logits, edit_logits, unmask_logits, tensors, **kwargs)
