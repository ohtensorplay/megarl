"""T3D, Tunyu Zhang et al., NeurIPS 2026 arXiv 2602.12262; code
https://github.com/Tyrion58/T3D; see package docstring for details."""

from __future__ import annotations

import dataclasses
from typing import Any

import tensorplay as tp

from ....dllm import default_adapter
from ....trainer import TensorPlayTrainer, TrainerConfig
from .batching import T3DBatch, T3DTensorBatch, slice_t3d_batch
from .config import T3DConfig
from .loss import t3d_loss


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class T3DTrainer(TensorPlayTrainer):
    """Few-step self-distillation trainer with an exact DDO log-ratio.

    Unlike the ratio-based dLLM trainers this owns its update loop: both the
    student and frozen reference are forwarded on each intermediate teacher
    state inside a microbatch, so no full-batch [B, L, V] graph is retained.
    """

    tensor_batch_cls = T3DTensorBatch

    def __init__(
        self,
        policy: Any,
        config: TrainerConfig | None = None,
        *,
        t3d_config: T3DConfig,
        adapter: Any | None = None,
        ref_model: Any | None = None,
        **kwargs,
    ) -> None:
        cfg = t3d_config
        base = config or TrainerConfig()
        if dataclasses.is_dataclass(base):
            base = dataclasses.replace(base, loss_type="grpo", compile_policy=False)
        if ref_model is None:
            raise ValueError("T3D requires a frozen reference model")
        kwargs["ref_model"] = None
        kwargs["ref_fn"] = None
        self.t3d_config = cfg
        self.adapter = adapter or default_adapter(cfg)
        self.t3d_ref_model = ref_model
        self._t3d_syncs = 0
        super().__init__(policy, base, forward_fn=None, **kwargs)

    @tp.no_grad()
    def sync_ref_model(self) -> None:
        """Refresh the stop-gradient reference snapshot from the student."""
        for ref, live in zip(self.t3d_ref_model.parameters(), self.policy.parameters()):
            ref.copy_(live.detach())

    def _slice_batch(self, batch, start: int, stop: int):
        if not isinstance(batch, T3DBatch):
            raise TypeError("T3DTrainer requires a T3DBatch")
        return slice_t3d_batch(batch, start, stop)

    def _forward_for(self, model, tensors: T3DTensorBatch):
        shim = type(
            "Shim",
            (),
            {
                "dllm_noisy_ids": tensors.t3d_canvas,
                "dllm_prompt_lens": tensors.t3d_prompt_lens,
                "dllm_prompt_span": tensors.t3d_prompt_span,
                "dllm_canvas_len": int(tensors.t3d_canvas.shape[1]) - tensors.t3d_prompt_span,
                "dllm_alpha": tp.ones(
                    (int(tensors.t3d_canvas.shape[0]),),
                    dtype=tp.float32,
                    device=tensors.t3d_canvas.device,
                ),
            },
        )()
        return self.adapter(model, tensors.t3d_canvas, shim, self.t3d_config)

    def _optimize(self, rows, compute_loss, *, dry_run: bool):
        accum = max(1, self.config.gradient_accumulation_steps)
        micro = max(1, (rows + accum - 1) // accum)
        spans = [(s, min(s + micro, rows)) for s in range(0, rows, micro)]
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        diagnostics: dict[str, float] = {}
        for index, (start, stop) in enumerate(spans):
            with tp.amp.autocast(
                self._device_type, dtype=self._amp_dtype, enabled=self._amp_enabled
            ):
                loss, diag = compute_loss(start, stop)
            weight = (stop - start) / rows
            context = self._no_sync() if index < len(spans) - 1 else _NullContext()
            with context:
                scaled = loss * weight
                (self.scaler.scale(scaled) if self._scaler_enabled else scaled).backward()
            loss_sum += float(loss.detach().flatten()[0]) * weight
            for key, value in diag.items():
                diagnostics[key] = diagnostics.get(key, 0.0) + float(
                    value.detach().flatten()[0]
                ) * weight
        if self._scaler_enabled:
            self.scaler.unscale_(self.optimizer)
        if self.config.max_grad_norm is not None:
            tp.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                self.config.max_grad_norm,
            )
        if not dry_run:
            if self._scaler_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step_count += 1
        metrics = {f"train/{key}": value for key, value in diagnostics.items()}
        metrics["train/loss"] = loss_sum
        metrics["train/step"] = float(self.step_count)
        metrics["train/version"] = float(self.version)
        return metrics

    def update(
        self,
        batch: T3DBatch,
        advantages: Any = None,
        *,
        clip_ratio: float = 0.2,
        kl_coef: float = 0.0,
        dry_run: bool = False,
    ) -> dict[str, float]:
        if not isinstance(batch, T3DBatch):
            raise TypeError("T3DTrainer requires a T3DBatch")
        if (
            not dry_run
            and self.step_count % self.t3d_config.refresh_every == 0
        ):
            self.sync_ref_model()
            self._t3d_syncs += 1
        cfg = self.t3d_config

        def loss_for(start: int, stop: int):
            micro = self.tensor_batch_cls(
                slice_t3d_batch(batch, start, stop), device=self.config.device
            )
            student = self._forward_for(self.policy, micro)
            with tp.no_grad():
                reference = self._forward_for(self.t3d_ref_model, micro)
            return t3d_loss(
                student,
                reference,
                micro.input_ids,
                micro.t3d_ref_targets,
                micro.t3d_mask_positions,
                micro.t3d_mask_valid,
                micro.t3d_path_positions,
                micro.t3d_path_valid,
                micro.t3d_path_weights,
                ddo_temperature=cfg.ddo_temperature,
                ratio_clip=cfg.ratio_clip,
                path_weight=cfg.path_weight,
            )

        metrics = self._optimize(batch.size, loss_for, dry_run=dry_run)
        metrics["train/ref_syncs"] = float(self._t3d_syncs)
        return metrics
