"""Diffusion language model post-training (TensorPlay / MegaLM stack).

See ``devdocs/dllm-rl.md`` for the full method and paper references.
"""

from __future__ import annotations

from .adapters import (
    BlockDiffusionAdapter,
    DLLMModelAdapter,
    FullAttentionAdapter,
    default_adapter,
)
from .batching import (
    DLLMBatch,
    DLLMTensorBatch,
    collate_dllm,
    collate_trajectory,
    make_dllm_packer,
    make_dllm_trajectory_packer,
    slice_dllm_batch,
)
from .config import DiffuGRPOConfig, DLLMConfig
from .controller import AsyncDLLMController, DLLMController
from .estimator import DLLMSample, dllm_logprobs, from_dllm_record
from .rollout import DLLMRollout, DLLMRolloutConfig
from .trainer import DLLMTrainer
from .trajectory import DLLMPayloadQueue, DLLMTrajectoryStore

from . import algorithms, trajectory

__all__ = [
    "BlockDiffusionAdapter",
    "DiffuGRPOConfig",
    "DLLMBatch",
    "DLLMConfig",
    "DLLMController",
    "AsyncDLLMController",
    "DLLMModelAdapter",
    "DLLMSample",
    "DLLMTensorBatch",
    "DLLMTrainer",
    "FullAttentionAdapter",
    "collate_dllm",
    "collate_trajectory",
    "default_adapter",
    "dllm_logprobs",
    "from_dllm_record",
    "make_dllm_packer",
    "make_dllm_trajectory_packer",
    "slice_dllm_batch",
    "DLLMRollout",
    "DLLMRolloutConfig",
]
