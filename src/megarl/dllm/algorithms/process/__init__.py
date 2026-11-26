"""Stepwise credit assignment for replayed dLLM trajectories.

The plain trajectory path in :mod:`megarl.dllm.trajectory` broadcasts one
outcome advantage to every denoising step (wd1++ / TraceRL style). This module
adds a learned process value head and generalised advantage estimation over the
replay steps, so early denoising decisions receive credit for the final reward.

The critic is a small token/embedding model over the noisy canvas — it reads
``dllm_noisy_ids`` and the replay ``loss_mask`` through the trainer's
``value_fn`` hook, shares the policy's optimizer, and is checkpointed next to
the policy by :class:`megatrain.BaseTrainer`. The policy loss, staleness mask,
KL penalty and out-of-band trajectory transport are unchanged.

References: TraceRL / dLLM-RL (ICLR 2026, arXiv:2509.06949) for trajectory
replay, and Schulman et al. (arXiv:1506.02438) for generalised advantage
estimation.
"""

from __future__ import annotations

from .config import ProcessRewardConfig
from .gae import process_gae
from .trainer import DLLMProcessTrainer
from .value import DLLMValueHead, dllm_value_fn

__all__ = [
    "DLLMProcessTrainer",
    "DLLMValueHead",
    "ProcessRewardConfig",
    "dllm_value_fn",
    "process_gae",
]
