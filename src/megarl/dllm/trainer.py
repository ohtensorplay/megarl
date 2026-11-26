"""dLLM reinforcement learning (d1 arXiv 2505.22765, wd1; see devdocs/dllm-rl.md); see
package docstring for details."""

from __future__ import annotations

from typing import Any

import tensorplay as tp

from ..trainer import TensorPlayTrainer
from .adapters import (
    BlockDiffusionAdapter,
    DLLMModelAdapter,
    FullAttentionAdapter,
    default_adapter,
)
from .batching import DLLMBatch, DLLMTensorBatch, slice_dllm_batch
from .config import DLLMConfig
from .estimator import dllm_logprobs
from .trajectory import DLLMTrajectoryStore


class DLLMTrainer(TensorPlayTrainer):
    """Policy-gradient trainer for masked / block diffusion language models.

    The policy is a dLLM (``FullAttentionAdapter``) or a block-diffusion model
    (``BlockDiffusionAdapter``); all RL state — staleness, group advantages,
    KL control, checkpointing — is inherited unchanged from
    :class:`~megarl.trainer.TensorPlayTrainer`.

    ``behavior_model`` is a frozen snapshot of the policy at sampling time.
    It is required when ``num_iterations > 1`` (multiple optimiser updates per
    rollout group) so the importance ratio stays on-policy. ``ref_model`` is
    an optional KL reference; without it the behavior policy is the reference.
    """

    tensor_batch_cls = DLLMTensorBatch

    def __init__(
        self,
        policy: Any,
        config,
        *,
        dllm_config: DLLMConfig,
        adapter: DLLMModelAdapter | None = None,
        behavior_model: Any | None = None,
        ref_model: Any | None = None,
        store: DLLMTrajectoryStore | None = None,
        **kwargs,
    ) -> None:
        if dllm_config.estimator == "trajectory" and store is None:
            raise ValueError("trajectory estimator requires a DLLMTrajectoryStore")
        # Trajectory replay already carries the sampling policy's log-probs, so
        # it stays valid across multiple optimiser updates without a snapshot.
        if (
            dllm_config.estimator == "one_step"
            and dllm_config.num_iterations > 1
            and behavior_model is None
        ):
            raise ValueError("num_iterations > 1 requires a frozen behavior_model")
        if dllm_config.kl_coef > 0 and ref_model is None and behavior_model is None:
            raise ValueError("kl_coef > 0 requires a ref_model or behavior_model")
        if config.loss_type not in (
            "diffu_grpo", "wd1", "grpo", "bnpo", "dapo", "ppo"
        ):
            raise ValueError(
                "dLLM training needs a policy-gradient loss_type "
                "(diffu_grpo, wd1, grpo, bnpo, dapo, ppo)"
            )
        if config.compile_policy:
            raise ValueError("compile_policy is not supported for dLLM forwards")

        self.dllm_config = dllm_config
        self.adapter = adapter or default_adapter(dllm_config)
        self.behavior_model = behavior_model
        self.dllm_ref_model = ref_model
        self.store = store
        # The AR ref forward must not run: dLLM reference logprobs come from a
        # noisy-canvas pass. ``dllm_config.kl_coef`` drives the penalty below.
        kwargs["ref_model"] = None
        kwargs["ref_fn"] = None

        def forward(model, tensors):
            return self.adapter(model, tensors.dllm_noisy_ids, tensors, self.dllm_config)

        super().__init__(policy, config, forward_fn=forward, **kwargs)

    def _slice_batch(self, batch, start: int, stop: int):
        if not isinstance(batch, DLLMBatch):
            raise TypeError("DLLMTrainer requires a DLLMBatch")
        return slice_dllm_batch(batch, start, stop)

    def _dllm_refs(self, batch: DLLMBatch) -> tuple[Any, Any | None]:
        cfg = self.dllm_config
        tensors = self.tensor_batch_cls(batch, device=self.config.device)
        source = self.behavior_model if self.behavior_model is not None else self.policy
        with tp.no_grad():
            old = dllm_logprobs(source, tensors, cfg, adapter=self.adapter)
            ref = None
            if self.dllm_ref_model is not None:
                ref = dllm_logprobs(
                    self.dllm_ref_model, tensors, cfg, adapter=self.adapter
                )
        return old, ref

    def _trajectory_aux(self, batch: DLLMBatch, advantages):
        """Replay path: old logprobs come from the recorded reverse steps."""
        tensors = self.tensor_batch_cls(batch, device=self.config.device)
        old = tensors.old_logprobs
        ref = None
        if self.dllm_ref_model is not None:
            with tp.no_grad():
                ref = dllm_logprobs(
                    self.dllm_ref_model, tensors, self.dllm_config,
                    adapter=self.adapter,
                )
        fresh = (~((self.version - tensors.versions) > self.config.max_offpolicy_tokens)).to(
            tensors.loss_mask.dtype
        )
        aux: dict[str, Any] = {"extra_mask": fresh}
        if ref is not None:
            aux["ref_logprobs"] = ref
        values = [float(a) for a in advantages]
        if len(values) != batch.size:
            raise ValueError(f"advantages/rows mismatch: {len(values)} vs {batch.size}")
        aux["advantages"] = tp.tensor(values, dtype=tp.float32, device=self.config.device)
        aux["old_logprobs"] = old
        metric_mask = tensors.loss_mask * fresh
        return aux, self._fresh_token_count(batch), None, None, metric_mask

    def _rl_aux(self, batch, advantages, *, kl_coef: float):
        if not isinstance(batch, DLLMBatch):
            raise TypeError("DLLMTrainer requires a DLLMBatch")
        if batch.dllm_is_trajectory:
            return self._trajectory_aux(batch, advantages)
        old, ref = self._dllm_refs(batch)
        aux, global_tokens, old_full, teacher_full, metric_mask = super()._rl_aux(
            batch, advantages, kl_coef=0.0
        )
        # train_step slices aux tensors along rows; these are [B, L] so each
        # microbatch gets exactly its own rows' estimator log-probs.
        aux["old_logprobs"] = old
        if ref is not None:
            aux["ref_logprobs"] = ref
        return aux, global_tokens, old_full, teacher_full, metric_mask

    def _loss_kwargs(self, *, clip_ratio: float, kl_coef: float) -> dict:
        kwargs = super()._loss_kwargs(clip_ratio=clip_ratio, kl_coef=kl_coef)
        cfg = self.dllm_config
        kwargs["loss_type"] = cfg.objective
        kwargs["kl_coef"] = cfg.kl_coef
        kwargs["kl_type"] = cfg.kl_type
        kwargs["epsilon_low"] = cfg.epsilon_low
        kwargs["epsilon_high"] = cfg.epsilon_high or cfg.epsilon_low
        kwargs["wd1_temperature"] = cfg.wd1_temperature
        kwargs["max_completion_length"] = cfg.max_completion_length
        return kwargs
