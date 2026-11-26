"""Continuous latent diffusion RL: Flow-GRPO and DiffNFT.

The discrete dLLM paths in :mod:`megarl.dllm` / :mod:`megarl.dllm.trajectory`
train a token denoiser. Continuous flow-matching generators (image / video and
some multimodal LMs) instead produce a latent trajectory

``x_0 -> x_1 -> ... -> x_K``

where each reverse step is a Gaussian transition. This module adds the second
production family without changing the Rust wire contract: bulk latents live in
the out-of-band payload store, and only the integer ``payload_handle`` travels
in the ``TrajectoryRecord`` metadata frame.

References: Flow-GRPO (arXiv:2505.05470) and the pinned ROLL diffusion pipeline
(``refs/ROLL/roll/pipeline/diffusion``), including its NFT worker.

Two objectives are supported, matching the reference implementation in
``refs/ROLL/roll/pipeline/diffusion``:

``flow_grpo``
    Replay the recorded SDE steps. The mean ``mu`` and std ``sigma`` are
    recomputed from the *current* policy with the exact
    :func:`flow_step_logprob` formula (ROLL's ``FlowMatchSDEDiscreteScheduler``),
    then a clipped policy-gradient loss is applied per step.
``diffnft``
    Take the final latent ``x_0``, re-noise it at sampled timesteps, and
    optimise a reward-weighted positive/negative ``x_0`` reconstruction loss
    against a frozen behaviour (EMA) policy. No rollout log-probs are needed.

Nothing in this module imports a reference repository at runtime.
"""

from __future__ import annotations

from .batching import FlowBatch, FlowTensorBatch, collate_flow, make_flow_packer
from .config import FlowConfig
from .controller import AsyncFlowController, FlowController
from .loss import diffnft_loss, flow_grpo_loss, flow_step_logprob, flow_step_stats
from .rollout import FlowTrajectoryRollout
from .trainer import FlowTrainer
from .types import (
    FlowStep,
    FlowTrajectoryRecord,
    FlowTrajectoryResult,
    FlowTrajectoryStore,
    decode_flow_trajectory,
    encode_flow_trajectory,
)

__all__ = [
    "AsyncFlowController",
    "FlowBatch",
    "FlowConfig",
    "FlowController",
    "FlowStep",
    "FlowTensorBatch",
    "FlowTrajectoryRecord",
    "FlowTrajectoryResult",
    "FlowTrajectoryRollout",
    "FlowTrajectoryStore",
    "FlowTrainer",
    "collate_flow",
    "decode_flow_trajectory",
    "diffnft_loss",
    "encode_flow_trajectory",
    "flow_grpo_loss",
    "flow_step_logprob",
    "flow_step_stats",
    "make_flow_packer",
]
