"""GDSD, Xiaohang Tang et al., arXiv 2605.29398; code https://github.com/GaryBall/GDSD;
see package docstring for details."""

from __future__ import annotations

import dataclasses
from typing import Any

import tensorplay as tp

from ....dllm import default_adapter
from ....trainer import TensorPlayTrainer, TrainerConfig
from .batching import GDSDBatch, GDSDTensorBatch, slice_gdsd_batch
from .config import GDSDConfig
from .loss import gdsd_loss, gdsd_token_logprobs


class GDSDTrainer(TensorPlayTrainer):
    """Normalisation-free self-distillation trainer for masked / block dLLMs.

    The inherited accumulation loop is kept: ``loss_type='grpo'`` makes each
    microbatch weight its own rows, and the objective above is already a mean
    over rows. The frozen behaviour and reference forwards happen once for the
    full batch in :meth:`_rl_aux`, exactly like the other dLLM trainers.
    """

    tensor_batch_cls = GDSDTensorBatch

    def __init__(
        self,
        policy: Any,
        config: TrainerConfig | None = None,
        *,
        gdsd_config: GDSDConfig,
        adapter: Any | None = None,
        old_model: Any | None = None,
        ref_model: Any | None = None,
        **kwargs,
    ) -> None:
        cfg = gdsd_config
        base = config or TrainerConfig()
        if dataclasses.is_dataclass(base):
            base = dataclasses.replace(base, loss_type="grpo", compile_policy=False)
        if base.loss_type != "grpo":
            raise ValueError("GDSDTrainer requires loss_type='grpo'")
        if cfg.beta > 0 and ref_model is None:
            raise ValueError("GDSD beta > 0 requires a frozen ref_model")
        if cfg.refresh_every > 1 and old_model is None:
            raise ValueError("refresh_every > 1 requires a frozen old_model snapshot")
        kwargs["ref_model"] = None
        kwargs["ref_fn"] = None

        self.gdsd_config = cfg
        self.adapter = adapter or default_adapter(cfg)
        self.old_model = old_model
        self.gdsd_ref_model = ref_model
        self._gdsd_syncs = 0

        def forward(model, tensors):
            return self.adapter(
                model, tensors.dllm_noisy_ids, tensors, self.gdsd_config
            )

        super().__init__(policy, base, forward_fn=forward, **kwargs)

    @tp.no_grad()
    def sync_old_model(self) -> None:
        """Refresh the frozen behaviour snapshot from the live policy."""
        if self.old_model is None:
            return
        for old, live in zip(self.old_model.parameters(), self.policy.parameters()):
            old.copy_(live.detach())

    def _slice_batch(self, batch, start: int, stop: int):
        if not isinstance(batch, GDSDBatch):
            raise TypeError("GDSDTrainer requires a GDSDBatch")
        return slice_gdsd_batch(batch, start, stop)

    def _rl_aux(self, batch, advantages, *, kl_coef: float):
        if not isinstance(batch, GDSDBatch):
            raise TypeError("GDSDTrainer requires a GDSDBatch")
        tensors = self.tensor_batch_cls(batch, device=self.config.device)
        source = self.old_model if self.old_model is not None else self.policy
        with tp.no_grad():
            old = gdsd_token_logprobs(source, tensors, self.gdsd_config, adapter=self.adapter)
            ref = None
            if self.gdsd_ref_model is not None:
                ref = gdsd_token_logprobs(
                    self.gdsd_ref_model, tensors, self.gdsd_config, adapter=self.adapter
                )
        values = [float(a) for a in advantages]
        if len(values) != batch.size:
            raise ValueError(f"advantages/rows mismatch: {len(values)} vs {batch.size}")
        aux = {
            "advantages": tp.tensor(values, dtype=tp.float32, device=self.config.device),
            "old_logprobs": old,
        }
        if ref is not None:
            aux["ref_logprobs"] = ref
        return aux, batch.size, None, None, None

    def _loss_kwargs(self, *, clip_ratio: float, kl_coef: float) -> dict:
        return {
            "psi": self.gdsd_config.psi,
            "beta": self.gdsd_config.beta,
            "epsilon_low": self.gdsd_config.epsilon_low,
            "epsilon_high": self.gdsd_config.epsilon_high,
            "tlc": self.gdsd_config.tlc,
        }

    def _rl_loss(self, logits, tensors: GDSDTensorBatch, *, advantages, **kwargs):
        return gdsd_loss(
            logits, tensors, advantages,
            old_logprobs=kwargs["old_logprobs"],
            ref_logprobs=kwargs.get("ref_logprobs"),
            psi=kwargs["psi"],
            beta=kwargs["beta"],
            epsilon_low=kwargs["epsilon_low"],
            epsilon_high=kwargs["epsilon_high"],
            tlc=kwargs["tlc"],
        )

    def update(self, batch, advantages, *, clip_ratio=0.2, kl_coef=0.0, dry_run=False):
        metrics = super().update(
            batch, advantages, clip_ratio=clip_ratio, kl_coef=kl_coef, dry_run=dry_run
        )
        if (
            not dry_run
            and self.old_model is not None
            and self.step_count % self.gdsd_config.refresh_every == 0
        ):
            self.sync_old_model()
            self._gdsd_syncs += 1
        metrics["train/old_syncs"] = float(self._gdsd_syncs)
        return metrics
