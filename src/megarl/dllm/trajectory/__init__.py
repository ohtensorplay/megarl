"""Trajectory-level dLLM rollouts and replay (TraceRL / CJ-GRPO style).

The one-step estimator in :mod:`megarl.dllm` deliberately avoids storing
denoising trajectories. This module provides the *other* production path:
record every reverse transition (noisy canvas, sampled tokens, behaviour
log-probabilities, accepted positions, timestep, version), ship the bulk array
through the existing out-of-band payload store, and replay those exact
transitions on the trainer with ``policy_loss``.

References: TraceRL (ICLR 2026, arXiv 2509.06949), MaskGRPO (2510.02880),
CJ-GRPO (2509.23924), and the FlowGRPO data contract in
``refs/ROLL/roll/pipeline/diffusion/schema.py``.
"""

from __future__ import annotations

from .generator import TrajectoryDiffusionGenerator
from .rollout import TrajectoryDLLMRollout
from .store import DLLMPayloadQueue, DLLMTrajectoryStore
from .types import (
    DLLMStep,
    DLLMTrajectoryRecord,
    TrajectoryGenerationConfig,
    TrajectoryGenerationResult,
    decode_trajectory,
    encode_trajectory,
)

__all__ = [
    "DLLMPayloadQueue",
    "DLLMStep",
    "DLLMTrajectoryRecord",
    "DLLMTrajectoryStore",
    "TrajectoryDLLMRollout",
    "TrajectoryDiffusionGenerator",
    "TrajectoryGenerationConfig",
    "TrajectoryGenerationResult",
    "decode_trajectory",
    "encode_trajectory",
]
