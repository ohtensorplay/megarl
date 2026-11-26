"""JustGRPO / The Flexibility Trap, Zanlin Ni et al., ICML 2026 arXiv 2601.15165; code
https://github.com/LeapLabTHU/JustGRPO; see package docstring for details."""

from __future__ import annotations

import dataclasses
from typing import Any

import tensorplay as tp
import tensorplay.nn.functional as F

from ....dllm import default_adapter
from ....trainer import TensorPlayTrainer, TrainerConfig
from .batching import JustGRPOBatch, JustGRPOTensorBatch, slice_justgrpo_batch
from .config import JustGRPOConfig


class JustGRPOTrainer(TensorPlayTrainer):
    """Exact-likelihood GRPO trainer for AR-constrained dLLMs.

    Uses ``bnpo``'s token mean with per-row advantages pre-scaled by
    ``selected / L``; summed over microbatches this reproduces the paper's
    ``1 / L`` normaliser exactly, including JustGRPO-Fast subsets.
    """

    tensor_batch_cls = JustGRPOTensorBatch

    def __init__(
        self,
        policy: Any,
        config: TrainerConfig | None = None,
        *,
        justgrpo_config: JustGRPOConfig,
        adapter: Any | None = None,
        ref_model: Any | None = None,
        **kwargs,
    ) -> None:
        cfg = justgrpo_config
        if config is None:
            config = TrainerConfig()
        if config.loss_type == "grpo":
            config = dataclasses.replace(config, loss_type="bnpo")
        if config.loss_type != "bnpo":
            raise ValueError("JustGRPOTrainer requires loss_type='bnpo'")
        if config.compile_policy:
            raise ValueError("compile_policy is not supported for AR dLLM forwards")
        if cfg.kl_coef > 0 and ref_model is None:
            raise ValueError("kl_coef > 0 requires a frozen ref_model")

        self.justgrpo_config = cfg
        self.adapter = adapter or default_adapter(cfg)
        self.justgrpo_ref_model = ref_model
        kwargs["ref_model"] = None
        kwargs["ref_fn"] = None

        def forward(model, tensors):
            return self.adapter(model, tensors.dllm_noisy_ids, tensors, cfg)

        super().__init__(policy, config, forward_fn=forward, **kwargs)

    def _slice_batch(self, batch, start: int, stop: int):
        if not isinstance(batch, JustGRPOBatch):
            raise TypeError("JustGRPOTrainer requires a JustGRPOBatch")
        return slice_justgrpo_batch(batch, start, stop)

    def _ref_logprobs(self, tensors: JustGRPOTensorBatch) -> Any:
        cfg = self.justgrpo_config
        with tp.no_grad():
            logits = self.adapter(
                self.justgrpo_ref_model,
                tensors.dllm_noisy_ids,
                tensors,
                cfg,
            )
            logits = logits / cfg.temperature
            log_probs = F.log_softmax(logits.float(), dim=-1)
            return tp.gather(
                log_probs, -1, tensors.input_ids.unsqueeze(-1)
            ).squeeze(-1)

    def _rl_aux(self, batch, advantages, *, kl_coef: float):
        if not isinstance(batch, JustGRPOBatch):
            raise TypeError("JustGRPOTrainer requires a JustGRPOBatch")
        values = [float(a) * batch.justgrpo_loss_scale for a in advantages]
        aux, global_tokens, old_full, teacher_full, metric_mask = super()._rl_aux(
            batch, values, kl_coef=0.0
        )
        if self.justgrpo_ref_model is not None:
            tensors = self.tensor_batch_cls(batch, device=self.config.device)
            aux["ref_logprobs"] = self._ref_logprobs(tensors)
        return aux, global_tokens, old_full, teacher_full, metric_mask

    def _loss_kwargs(self, *, clip_ratio: float, kl_coef: float) -> dict:
        kwargs = super()._loss_kwargs(clip_ratio=clip_ratio, kl_coef=0.0)
        cfg = self.justgrpo_config
        kwargs["loss_type"] = "bnpo"
        kwargs["kl_coef"] = cfg.kl_coef
        kwargs["kl_type"] = cfg.kl_type
        kwargs["epsilon_low"] = cfg.epsilon_low
        kwargs["epsilon_high"] = cfg.epsilon_high or cfg.epsilon_low
        return kwargs
