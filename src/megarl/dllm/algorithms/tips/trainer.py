"""TIPS trainer — TIPS: A Tiny Inference-time Policy for Scaling in Diffusion Language
Models (Zhang et al., arXiv:2610.22099; code https://github.com/zhengyz97/TIPS).
"""

from __future__ import annotations

import dataclasses

import tensorplay as tp

from ....trainer import TensorPlayTrainer, TrainerConfig
from .batching import TIPSBatch, TIPSTensorBatch, slice_tips_batch
from .config import TIPSConfig
from .loss import tips_loss, tips_warmup_loss
from .policy import TIPSPolicy


class TIPSTrainer(TensorPlayTrainer):
    """GRPO trainer for the auxiliary TIPS visibility policy."""

    tensor_batch_cls = TIPSTensorBatch

    def __init__(
        self,
        policy: TIPSPolicy,
        config: TrainerConfig | None = None,
        *,
        tips_config: TIPSConfig,
        **kwargs,
    ) -> None:
        cfg = tips_config
        base = config or TrainerConfig()
        if dataclasses.is_dataclass(base):
            base = dataclasses.replace(base, loss_type="grpo", compile_policy=False)
        if base.loss_type != "grpo":
            raise ValueError("TIPSTrainer requires loss_type='grpo'")
        kwargs["ref_model"] = None
        kwargs["ref_fn"] = None

        def forward(model, tensors):
            return model(
                tensors.tips_features,
                tensors.tips_mask,
                tensors.tips_time,
            )

        self.tips_config = cfg
        super().__init__(policy, base, forward_fn=forward, **kwargs)

    def _slice_batch(self, batch, start: int, stop: int):
        if not isinstance(batch, TIPSBatch):
            raise TypeError("TIPSTrainer requires a TIPSBatch")
        return slice_tips_batch(batch, start, stop)

    def _rl_aux(self, batch, advantages, *, kl_coef: float):
        if not isinstance(batch, TIPSBatch):
            raise TypeError("TIPSTrainer requires a TIPSBatch")
        values = [float(a) for a in advantages]
        if len(values) != batch.size:
            raise ValueError(f"advantages/rows mismatch: {len(values)} vs {batch.size}")
        aux = {
            "advantages": tp.tensor(values, dtype=tp.float32, device=self.config.device),
            "old_logprobs": tp.tensor(
                batch.tips_old_logprobs, dtype=tp.float32, device=self.config.device
            ),
        }
        return aux, batch.size, None, None, None

    def _loss_kwargs(self, *, clip_ratio: float, kl_coef: float) -> dict:
        return {
            "old_logprobs": None,
            "clip_low": self.tips_config.clip_low,
            "clip_high": self.tips_config.clip_high,
            "calibration_iters": self.tips_config.calibration_iters,
        }

    def _rl_loss(self, logits, tensors: TIPSTensorBatch, *, advantages, **kwargs):
        return tips_loss(
            logits,
            tensors,
            advantages,
            old_logprobs=kwargs["old_logprobs"],
            clip_low=kwargs["clip_low"],
            clip_high=kwargs["clip_high"],
            calibration_iters=kwargs["calibration_iters"],
        )

    def warmup_update(self, batch: TIPSBatch, targets, *, dry_run: bool = False) -> dict:
        """Optional supervised warmup on reference-derived visibility utilities."""
        tensors = self.tensor_batch_cls(batch, device=self.config.device)
        target_tensor = tp.tensor(targets, dtype=tp.float32, device=self.config.device)
        kwargs = {
            "beta": self.tips_config.warmup_beta,
            "delta_star": 0.0,
        }
        return self.train_step(
            batch,
            loss_fn=lambda logits, tensors, **kw: tips_warmup_loss(logits, target_tensor, **kw),
            loss_kwargs=kwargs,
            aux={},
            dry_run=dry_run,
        )
