"""REMARL trainer — ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language
Models (Shibaev et al., arXiv:2611.22576; code https://github.com/EgorShibaev/remarl).
"""

from __future__ import annotations

import dataclasses

import tensorplay as tp

from ....trainer import TensorPlayTrainer, TrainerConfig
from .batching import ReMaRLBatch, ReMaRLTensorBatch, slice_remarl_batch
from .config import ReMaRLConfig
from .loss import remarl_loss
from .policy import ReMaRLPolicy


class ReMaRLTrainer(TensorPlayTrainer):
    """GRPO trainer for the auxiliary ReMaRL position policy."""

    tensor_batch_cls = ReMaRLTensorBatch

    def __init__(
        self,
        policy: ReMaRLPolicy,
        config: TrainerConfig | None = None,
        *,
        remarl_config: ReMaRLConfig,
        **kwargs,
    ) -> None:
        base = config or TrainerConfig()
        if dataclasses.is_dataclass(base):
            base = dataclasses.replace(base, loss_type="grpo", compile_policy=False)
        if base.loss_type != "grpo":
            raise ValueError("ReMaRLTrainer requires loss_type='grpo'")
        kwargs["ref_model"] = None
        kwargs["ref_fn"] = None
        self.remarl_config = remarl_config

        def forward(model, tensors):
            return model(
                tensors.remarl_scores,
                tensors.remarl_mask,
                tensors.remarl_hidden,
            )

        super().__init__(policy, base, forward_fn=forward, **kwargs)

    def _slice_batch(self, batch, start: int, stop: int):
        if not isinstance(batch, ReMaRLBatch):
            raise TypeError("ReMaRLTrainer requires a ReMaRLBatch")
        return slice_remarl_batch(batch, start, stop)

    def _rl_aux(self, batch, advantages, *, kl_coef: float):
        if not isinstance(batch, ReMaRLBatch):
            raise TypeError("ReMaRLTrainer requires a ReMaRLBatch")
        values = [float(a) for a in advantages]
        if len(values) != batch.size:
            raise ValueError(f"advantages/rows mismatch: {len(values)} vs {batch.size}")
        aux = {
            "advantages": tp.tensor(values, dtype=tp.float32, device=self.config.device),
            "old_logprobs": tp.tensor(
                batch.remarl_old_logprobs, dtype=tp.float32, device=self.config.device
            ),
        }
        return aux, batch.size, None, None, None

    def _loss_kwargs(self, *, clip_ratio: float, kl_coef: float) -> dict:
        return {
            "old_logprobs": None,
            "tau": self.remarl_config.tau,
            "clip_eps": self.remarl_config.clip_eps or clip_ratio,
        }

    def _rl_loss(self, scores, tensors: ReMaRLTensorBatch, *, advantages, **kwargs):
        return remarl_loss(
            scores,
            tensors,
            advantages,
            old_logprobs=kwargs["old_logprobs"],
            tau=kwargs["tau"],
            clip_eps=kwargs["clip_eps"],
        )
