"""dOPSD, Phuong Tuan Dat, Qi Li, Xinchao Wang, arXiv 2607.04428; code
https://github.com/tuandattt/dOPSD; see package docstring for details."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import tensorplay as tp
import tensorplay.nn.functional as F

from ....dllm import default_adapter
from ....trainer import TensorPlayTrainer, TrainerConfig
from .batching import DOPSDBatch, DOPSDTensorBatch, slice_dopsd_batch
from .config import DOPSDConfig
from .loss import dopsd_loss


class DOPSDTrainer(TensorPlayTrainer):
    """Trajectory-privileged self-distillation trainer.

    Teacher distributions are computed once for the full batch under
    ``no_grad`` from later states of each rollout's own trajectory; student
    logits are the ordinary adapter forward on the genuinely masked state.
    """

    tensor_batch_cls = DOPSDTensorBatch

    def __init__(
        self,
        policy: Any,
        config: TrainerConfig | None = None,
        *,
        dopsd_config: DOPSDConfig,
        adapter: Any | None = None,
        **kwargs,
    ) -> None:
        cfg = dopsd_config
        base = config or TrainerConfig()
        if dataclasses.is_dataclass(base):
            base = dataclasses.replace(base, loss_type="grpo", compile_policy=False)
        if base.loss_type != "grpo":
            raise ValueError("DOPSDTrainer requires loss_type='grpo'")
        kwargs["ref_model"] = None
        kwargs["ref_fn"] = None
        self.dopsd_config = cfg
        self.adapter = adapter or default_adapter(cfg)

        def forward(model, tensors):
            return self.adapter(
                model, tensors.dopsd_student_canvas, tensors, self.dopsd_config
            )

        super().__init__(policy, base, forward_fn=forward, **kwargs)

    def _slice_batch(self, batch, start: int, stop: int):
        if not isinstance(batch, DOPSDBatch):
            raise TypeError("DOPSDTrainer requires a DOPSDBatch")
        return slice_dopsd_batch(batch, start, stop)

    def _teacher_probs(self, tensors: DOPSDTensorBatch) -> Any:
        cfg = self.dopsd_config
        B, T, W = tuple(tensors.dopsd_teacher_canvas.shape)
        K = int(tensors.dopsd_mask_positions.shape[1])
        flat = tensors.dopsd_teacher_canvas.reshape(B * T, W)
        prompt_lens = tensors.dopsd_prompt_lens.unsqueeze(1).expand(B, T).reshape(B * T)
        shim = SimpleNamespace(
            dllm_noisy_ids=flat,
            dllm_prompt_lens=prompt_lens,
            dllm_prompt_span=tensors.dopsd_prompt_span,
            dllm_canvas_len=int(tensors.dopsd_student_canvas.shape[1]) - tensors.dopsd_prompt_span,
            dllm_alpha=tp.ones((B * T,), dtype=tp.float32, device=flat.device),
        )
        with tp.no_grad():
            logits = self.adapter(self.policy, flat, shim, cfg)
        probs = F.softmax(logits.float(), dim=-1).reshape(B * T, W, -1)
        vocab = probs.shape[-1]
        position_index = (
            tensors.dopsd_mask_positions.unsqueeze(1).expand(B, T, K).reshape(B * T, K)
        )
        gather_index = position_index.unsqueeze(-1).expand(B * T, K, vocab)
        gathered = tp.gather(probs, 1, gather_index).reshape(B, T, K, vocab)
        present = tensors.dopsd_teacher_present.reshape(B, T, 1, 1)
        per_position = tensors.dopsd_teacher_mask.reshape(B, T, K, 1)
        numerator = (gathered * present * per_position).sum(dim=1)
        denominator = tensors.dopsd_counts.unsqueeze(-1).clamp(min=1.0)
        return (numerator / denominator).detach()

    def _rl_aux(self, batch, advantages, *, kl_coef: float):
        if not isinstance(batch, DOPSDBatch):
            raise TypeError("DOPSDTrainer requires a DOPSDBatch")
        tensors = self.tensor_batch_cls(batch, device=self.config.device)
        aux = {"dopsd_teacher": self._teacher_probs(tensors)}
        return aux, batch.size, None, None, None

    def _loss_kwargs(self, *, clip_ratio: float, kl_coef: float) -> dict:
        return {
            "temperature": self.dopsd_config.temperature,
            "js_weight": self.dopsd_config.js_weight,
        }

    def _rl_loss(self, logits, tensors: DOPSDTensorBatch, *, advantages=None, **kwargs):
        positions = tensors.dopsd_mask_positions
        vocab = int(logits.shape[-1])
        index = positions.unsqueeze(-1).expand(
            positions.shape[0], positions.shape[1], vocab
        )
        student_logits = tp.gather(logits, 1, index)
        return dopsd_loss(
            kwargs["dopsd_teacher"],
            student_logits,
            tensors.dopsd_position_valid,
            tensors.dopsd_keep,
            temperature=kwargs["temperature"],
            js_weight=kwargs["js_weight"],
        )
