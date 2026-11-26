"""AGRPO / Amortized GRPO, Anthony Zhan, ICML 2026 arXiv 2510.04019; code
https://github.com/probablyabot/agrpo; see package docstring for details."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import tensorplay as tp
import tensorplay.nn.functional as F

from ....dllm import default_adapter
from ....trainer import TensorPlayTrainer, TrainerConfig
from .batching import AGRPOBatch, AGRPOTensorBatch, slice_agrpo_batch
from .config import AGRPOConfig


class AGRPOTrainer(TensorPlayTrainer):
    """Unbiased MC step-level GRPO trainer on TensorPlay."""

    tensor_batch_cls = AGRPOTensorBatch

    def __init__(
        self,
        policy: Any,
        config: TrainerConfig | None = None,
        *,
        agrpo_config: AGRPOConfig,
        adapter: Any | None = None,
        ref_model: Any | None = None,
        **kwargs,
    ) -> None:
        cfg = agrpo_config
        if config is None:
            config = TrainerConfig()
        if config.loss_type == "grpo":
            config = dataclasses.replace(config, loss_type="bnpo")
        if config.loss_type != "bnpo":
            raise ValueError("AGRPOTrainer requires loss_type='bnpo'")
        if config.compile_policy:
            raise ValueError("compile_policy is not supported for step-level dLLM forwards")
        if cfg.kl_coef > 0 and ref_model is None:
            raise ValueError("kl_coef > 0 requires a frozen ref_model")

        self.agrpo_config = cfg
        self.adapter = adapter or default_adapter(cfg)
        self.agrpo_ref_model = ref_model
        kwargs["ref_model"] = None
        kwargs["ref_fn"] = None

        def forward(model, tensors):
            return self.adapter(model, tensors.dllm_noisy_ids, tensors, cfg)

        super().__init__(policy, config, forward_fn=forward, **kwargs)

    def _slice_batch(self, batch, start: int, stop: int):
        if not isinstance(batch, AGRPOBatch):
            raise TypeError("AGRPOTrainer requires an AGRPOBatch")
        return slice_agrpo_batch(batch, start, stop)

    def _row_advantages(self, batch: AGRPOBatch) -> list[float]:
        """Group-mean-centred advantages with MC scale and IS correction."""
        by_group: dict[int, list[float]] = {}
        seen: dict[int, float] = {}
        for sample_id, reward in zip(batch.agrpo_sample_ids, batch.agrpo_sample_rewards):
            if sample_id in seen:
                continue
            seen[sample_id] = reward
        for sample_id, reward in seen.items():
            group = batch.agrpo_group_ids[batch.agrpo_sample_ids.index(sample_id)]
            by_group.setdefault(group, []).append(reward)
        mean = {group: sum(values) / len(values) for group, values in by_group.items()}
        std = {
            group: math.sqrt(
                sum((v - mean[group]) ** 2 for v in values) / max(len(values), 1)
            )
            for group, values in by_group.items()
        }
        values: list[float] = []
        for sample_id, group, iw in zip(
            batch.agrpo_sample_ids,
            batch.agrpo_group_ids,
            batch.agrpo_importance_weights,
        ):
            reward = seen[sample_id]
            adv = reward - mean[group]
            if self.agrpo_config.advantage_mode == "standardize":
                adv = adv / (std[group] + 1e-4)
            values.append(adv * batch.agrpo_loss_scale * iw)
        return values

    def _ref_logprobs(self, tensors: AGRPOTensorBatch) -> Any:
        cfg = self.agrpo_config
        with tp.no_grad():
            logits = self.adapter(
                self.agrpo_ref_model, tensors.dllm_noisy_ids, tensors, cfg
            )
            log_probs = F.log_softmax((logits / cfg.temperature).float(), dim=-1)
            return tp.gather(log_probs, -1, tensors.input_ids.unsqueeze(-1)).squeeze(-1)

    def _rl_aux(self, batch, advantages, *, kl_coef: float):
        if not isinstance(batch, AGRPOBatch):
            raise TypeError("AGRPOTrainer requires an AGRPOBatch")
        values = self._row_advantages(batch)
        aux, global_tokens, old_full, teacher_full, metric_mask = super()._rl_aux(
            batch, values, kl_coef=0.0
        )
        if self.agrpo_ref_model is not None:
            tensors = self.tensor_batch_cls(batch, device=self.config.device)
            aux["ref_logprobs"] = self._ref_logprobs(tensors)
        return aux, global_tokens, old_full, teacher_full, metric_mask

    def _loss_kwargs(self, *, clip_ratio: float, kl_coef: float) -> dict:
        kwargs = super()._loss_kwargs(clip_ratio=clip_ratio, kl_coef=0.0)
        cfg = self.agrpo_config
        kwargs["loss_type"] = "bnpo"
        kwargs["kl_coef"] = cfg.kl_coef
        kwargs["kl_type"] = cfg.kl_type
        kwargs["epsilon_low"] = cfg.epsilon_low
        kwargs["epsilon_high"] = cfg.epsilon_high or cfg.epsilon_low
        return kwargs
