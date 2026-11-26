"""Stepwise dLLM credit assignment (TraceRL ICLR 2026 arXiv 2509.06949; Schulman et al.
arXiv 1506.02438); see package docstring for details."""

from __future__ import annotations

import dataclasses
from typing import Any

import tensorplay as tp

from ....dllm import DLLMConfig, DLLMTrainer, dllm_logprobs
from .config import ProcessRewardConfig
from .gae import process_gae
from .value import DLLMValueHead, dllm_value_fn


class DLLMProcessTrainer(DLLMTrainer):
    """Trajectory trainer with a learned process critic and stepwise GAE."""

    def __init__(
        self,
        policy: Any,
        config,
        *,
        dllm_config: DLLMConfig,
        value_model: DLLMValueHead,
        process_config: ProcessRewardConfig | None = None,
        **kwargs,
    ) -> None:
        if dllm_config.estimator != "trajectory":
            raise ValueError("process rewards require estimator='trajectory'")
        if dataclasses.is_dataclass(config) and getattr(config, "loss_type", None) != "ppo":
            config = dataclasses.replace(config, loss_type="ppo")
        if kwargs.get("value_model") is not None:
            raise ValueError("DLLMProcessTrainer takes value_model as a keyword")
        kwargs["value_model"] = value_model
        kwargs.setdefault("value_fn", dllm_value_fn)
        super().__init__(policy, config, dllm_config=dllm_config, **kwargs)
        self.process_config = process_config or ProcessRewardConfig()

    def _loss_kwargs(self, *, clip_ratio: float, kl_coef: float) -> dict:
        kwargs = super()._loss_kwargs(clip_ratio=clip_ratio, kl_coef=kl_coef)
        kwargs["loss_type"] = "ppo"
        kwargs["value_coef"] = self.config.value_coef
        kwargs["clip_value"] = self.config.clip_value
        return kwargs

    def _trajectory_aux(self, batch, advantages):
        tensors = self.tensor_batch_cls(batch, device=self.config.device)
        ids = list(batch.dllm_sample_ids)
        steps = list(batch.dllm_step_ids)
        if not ids or len(ids) != batch.size or len(steps) != batch.size:
            raise ValueError("process rewards need per-row sample/step provenance")
        with tp.no_grad():
            values = self.value_fn(self.value_model, tensors)
            row_values = (values * tensors.loss_mask).sum(-1) / tensors.loss_mask.sum(
                -1
            ).clamp(min=1.0)
        flat_values = [float(v.flatten()[0]) for v in row_values]
        count = max(ids) + 1
        width = max(steps) + 1
        values_2d = [[0.0] * width for _ in range(count)]
        mask_2d = [[0] * width for _ in range(count)]
        rewards = [0.0] * count
        for row, (sample, step) in enumerate(zip(ids, steps)):
            values_2d[sample][step] = flat_values[row]
            mask_2d[sample][step] = 1
            rewards[sample] = float(batch.rewards[row])
        cfg = self.process_config
        adv_2d, ret_2d = process_gae(
            rewards, values_2d, mask_2d, gamma=cfg.gamma, lam=cfg.lam
        )
        device = self.config.device
        flat = [(adv_2d[s][k], ret_2d[s][k]) for s, k in zip(ids, steps)]
        aux: dict[str, Any] = {
            "advantages": tp.tensor(
                [pair[0] for pair in flat], dtype=tp.float32, device=device
            ),
            "returns": tp.tensor(
                [pair[1] for pair in flat], dtype=tp.float32, device=device
            ).unsqueeze(-1).expand(-1, tensors.loss_mask.shape[1]),
            "old_logprobs": tensors.old_logprobs,
        }
        fresh = (~((self.version - tensors.versions) > self.config.max_offpolicy_tokens)).to(
            tensors.loss_mask.dtype
        )
        aux["extra_mask"] = fresh
        if self.dllm_ref_model is not None:
            with tp.no_grad():
                aux["ref_logprobs"] = dllm_logprobs(
                    self.dllm_ref_model, tensors, self.dllm_config,
                    adapter=self.adapter,
                )
        metric_mask = tensors.loss_mask * fresh
        return aux, self._fresh_token_count(batch), None, None, metric_mask
