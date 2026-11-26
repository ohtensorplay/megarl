"""Flow-GRPO arXiv 2505.05470 and DiffNFT (ROLL diffusion pipeline
refs/ROLL/roll/pipeline/diffusion); see package docstring for details."""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Callable

import tensorplay as tp

from ....trainer import TensorPlayTrainer, TrainerConfig
from .batching import FlowBatch, FlowTensorBatch
from .config import FlowConfig
from .loss import diffnft_loss, flow_grpo_loss, flow_step_logprob
from .rollout import _ModelStepFn
from .types import FlowTrajectoryStore


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class FlowTrainer(TensorPlayTrainer):
    """Policy-gradient trainer for continuous flow-matching generators.

    The model-facing hook is ``model_step_fn(model, latents, timesteps)`` and
    must return the velocity/noise prediction ``[B, D]`` for inputs ``[B, D]``
    (timesteps are ``[B]`` after scaling by ``FlowConfig.timestep_scale``).
    For DiffNFT pass a frozen EMA policy as ``old_model``; without one the
    behaviour anchor degenerates to the live policy.
    """

    tensor_batch_cls = FlowTensorBatch

    def __init__(
        self,
        policy: Any,
        config: TrainerConfig | None = None,
        *,
        flow_config: FlowConfig,
        model_step_fn: _ModelStepFn,
        store: FlowTrajectoryStore | None = None,
        old_model: Any | None = None,
        ref_model: Any | None = None,
        **kwargs,
    ) -> None:
        # The inherited loop validates loss_type; our custom update owns the
        # flow objective, so the outer config stays on the standard GRPO enum.
        base = config or TrainerConfig()
        if dataclasses.is_dataclass(base):
            base = dataclasses.replace(base, loss_type="grpo", compile_policy=False)
        if kwargs.get("ref_model") is not None or kwargs.get("ref_fn") is not None:
            raise ValueError("FlowTrainer takes ref_model as an explicit keyword")
        kwargs["ref_model"] = None
        kwargs["ref_fn"] = None
        super().__init__(policy, base, **kwargs)
        self.flow_config = flow_config
        self.model_step_fn = model_step_fn
        self.store = store or FlowTrajectoryStore()
        self.old_model = old_model
        self.flow_ref_model = ref_model

    @tp.no_grad()
    def ema_update(self, decay: float) -> None:
        """Update the frozen behaviour anchor: ``old = decay*old + (1-decay)*live``."""
        if not 0.0 <= decay <= 1.0:
            raise ValueError("ema decay must be in [0, 1]")
        if self.old_model is None:
            raise ValueError("ema_update requires old_model")
        for old, live in zip(self.old_model.parameters(), self.policy.parameters()):
            old.lerp_(live.detach(), 1.0 - decay)

    def _model_step(self, model: Any, latents, sigma) -> Any:
        return self.model_step_fn(model, latents, sigma * self.flow_config.timestep_scale)

    def _optimize(
        self,
        rows: int,
        compute_loss: Callable[[int, int], tuple[tp.Tensor, dict[str, tp.Tensor]]],
        *,
        dry_run: bool,
    ) -> dict[str, float]:
        """Microbatch accumulation with megaTrain's scaler/clip/optimizer path."""
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
            context = self._no_sync() if index < len(spans) - 1 else _nullcontext()
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
        batch: FlowBatch,
        advantages: Any,
        *,
        clip_ratio: float = 0.2,
        kl_coef: float = 0.0,
        dry_run: bool = False,
    ) -> dict[str, float]:
        """One optimizer step over a collated continuous-trajectory batch."""
        if not isinstance(batch, FlowBatch):
            raise TypeError("FlowTrainer requires a FlowBatch")
        tensors = FlowTensorBatch(batch, device=self.config.device)
        rows = batch.size
        adv = tp.tensor([float(a) for a in advantages], dtype=tp.float32, device=self.config.device)
        if adv.shape[0] != rows:
            raise ValueError(f"advantages/rows mismatch: {adv.shape[0]} vs {rows}")
        fresh = (~((self.version - tensors.versions) > self.config.max_offpolicy_tokens)).to(
            tp.float32
        ).unsqueeze(-1)
        cfg = self.flow_config
        if cfg.objective == "flow_grpo":
            return self._update_flow_grpo(tensors, adv, fresh, clip_ratio, kl_coef, dry_run)
        return self._update_diffnft(tensors, adv, fresh, dry_run)

    def _update_flow_grpo(self, tensors, adv, fresh, clip_ratio, kl_coef, dry_run):
        cfg = self.flow_config
        steps = tensors.latents.shape[1] - 1
        xt = tensors.latents[:, :steps].detach()
        x_next = tensors.latents[:, 1 : steps + 1].detach()
        sigma = tensors.timesteps[:, :steps]
        sigma_next = tensors.timesteps[:, 1 : steps + 1]
        old = tensors.old_logprobs[:, :steps]
        mask = tensors.loss_mask[:, :steps] * fresh
        clip_low = clip_ratio if cfg.pg_clip == 0.2 else cfg.pg_clip
        ref_logp = None
        if self.flow_ref_model is not None:
            with tp.no_grad():
                ref_out = self._model_step(self.flow_ref_model, xt, sigma)
                ref_logp = flow_step_logprob(x_next, xt, ref_out, sigma, sigma_next, cfg)

        def loss_for(start: int, stop: int):
            xt_m = xt[start:stop]
            flat = xt_m.reshape(-1, xt_m.shape[-1])
            flat_t = sigma[start:stop].reshape(-1)
            out = self._model_step(self.policy, flat, flat_t).reshape(
                stop - start, steps, xt_m.shape[-1]
            )
            new_logp = flow_step_logprob(
                x_next[start:stop], xt_m, out,
                sigma[start:stop], sigma_next[start:stop], cfg,
            )
            ref_m = ref_logp[start:stop] if ref_logp is not None else None
            return flow_grpo_loss(
                new_logp, old[start:stop], adv[start:stop], mask[start:stop],
                clip_low=clip_low,
                advantage_clip=cfg.advantage_clip,
                kl_coef=kl_coef or cfg.kl_coef,
                kl_type=cfg.kl_type,
                ref_logp=ref_m,
            )

        return self._optimize(tensors.size, loss_for, dry_run=dry_run)

    def _update_diffnft(self, tensors, adv, fresh, dry_run):
        cfg = self.flow_config
        x0 = tensors.latents[:, -1].detach()
        width = x0.shape[1]
        rng = random.Random(cfg.seed + self.version)
        schedule = tensors.timesteps[:, :-1]
        counts = max(1, min(cfg.num_nft_timesteps, schedule.shape[1]))
        columns = [rng.randrange(schedule.shape[1]) for _ in range(counts)]
        sigma = tp.stack(
            [schedule[:, column] for column in columns], dim=1
        ).detach()
        noise = tp.randn(
            (tensors.size, counts, width), dtype=x0.dtype, device=x0.device
        )
        t = sigma.reshape(tensors.size, counts, 1)
        xt = (1.0 - t) * x0.unsqueeze(1) + t * noise
        # ROLL stores the verifier's optimality probability in the advantage
        # field. Negative group advantages are mapped back to [0, 1].
        reward_prob = (adv > 0).to(tp.float32).reshape(tensors.size, 1)
        mask = fresh.unsqueeze(-1) * tp.ones(
            (tensors.size, counts, 1), dtype=tp.float32, device=x0.device
        )

        def loss_for(start: int, stop: int):
            xt_m = xt[start:stop]
            flat = xt_m.reshape(-1, width)
            flat_t = sigma[start:stop].reshape(-1)
            anchor = self.old_model if self.old_model is not None else self.policy
            with tp.no_grad():
                old_out = self._model_step(anchor, flat, flat_t).reshape(
                    stop - start, counts, width
                )
            new_out = self._model_step(self.policy, flat, flat_t).reshape(
                stop - start, counts, width
            )
            ref_out = None
            if self.flow_ref_model is not None:
                with tp.no_grad():
                    ref_out = self._model_step(self.flow_ref_model, flat, flat_t).reshape(
                        stop - start, counts, width
                    )
            loss, diag = diffnft_loss(
                x0[start:stop].unsqueeze(1),
                xt_m,
                sigma[start:stop],
                new_out,
                old_out,
                reward_prob[start:stop],
                beta=cfg.nft_beta,
                adv_clip_max=cfg.adv_clip_max,
                ref_pred=ref_out,
                ref_kl_coef=cfg.ref_kl_coef,
                use_adaptive_weight=cfg.use_adaptive_weight,
            )
            return loss, diag

        return self._optimize(tensors.size, loss_for, dry_run=dry_run)
