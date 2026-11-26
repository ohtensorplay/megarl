"""AGRPO / Amortized GRPO, Anthony Zhan, ICML 2026 arXiv 2510.04019; code
https://github.com/probablyabot/agrpo; see package docstring for details."""

from __future__ import annotations

import dataclasses
import random


@dataclasses.dataclass(frozen=True, slots=True)
class AGRPOConfig:
    """Knobs for amortized, step-level GRPO on masked dLLMs."""

    family: str = "masked"
    mask_token_id: int = 126336
    pad_token_id: int = 0
    canvas_token_id: int | None = None
    vocab_size: int | None = None
    max_completion_length: int = 384
    """Response length ``n``."""
    sampling_steps: int = 128
    """Number of denoising steps ``m`` used by both rollout and training."""
    block_length: int = 3
    """Semi-AR block size ``B``; ``UM = n/m = B/tokens_per_step``."""
    mc_samples: int = 24
    """Monte Carlo timestep budget ``k`` (paper default: m/4)."""
    temperature: float = 0.6
    unmasking: str = "random"
    """``random`` (paper default) or ``low_confidence`` (confidence remasking)."""
    epsilon_low: float = 0.2
    epsilon_high: float | None = None
    kl_coef: float = 0.0
    kl_type: str = "k3"
    importance_sampling: str = "none"
    """``none`` (uniform t) or ``entropy`` (entropy-proportional t + correction)."""
    low_discrepancy: bool = True
    advantage_mode: str = "mean"
    """``mean`` (AGRPO: r - group mean) or ``standardize`` (GRPO: / std)."""
    group_size: int = 8
    """Group size ``G`` used by the objective's ``1/(k G)`` normaliser."""
    eos_token_ids: tuple[int, ...] = (1,)
    seed: int = 0
    version: int = 0
    max_prompt_len: int | None = None

    def __post_init__(self) -> None:
        if self.family not in ("masked", "block"):
            raise ValueError(f"unknown dLLM family: {self.family}")
        n = self.max_completion_length
        if n <= 0 or self.sampling_steps <= 0 or self.block_length <= 0:
            raise ValueError("completion length, steps and block length must be positive")
        if n % self.block_length:
            raise ValueError("max_completion_length must be divisible by block_length")
        blocks = n // self.block_length
        if self.sampling_steps % blocks:
            raise ValueError("sampling_steps must be divisible by n/block_length")
        if self.tokens_per_step <= 0:
            raise ValueError("sampling_steps must not exceed n/block_length * block_length")
        if self.mc_samples <= 0:
            raise ValueError("mc_samples must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.unmasking not in ("random", "low_confidence"):
            raise ValueError(f"unknown unmasking: {self.unmasking}")
        if self.importance_sampling not in ("none", "entropy"):
            raise ValueError(f"unknown importance_sampling: {self.importance_sampling}")
        if self.advantage_mode not in ("mean", "standardize"):
            raise ValueError(f"unknown advantage_mode: {self.advantage_mode}")
        if self.epsilon_low < 0 or self.kl_coef < 0:
            raise ValueError("epsilon_low and kl_coef cannot be negative")
        if self.kl_type not in ("k1", "k2", "k3", "low_var_kl"):
            raise ValueError(f"unknown kl_type: {self.kl_type}")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.version < 0:
            raise ValueError("version cannot be negative")

    @property
    def noise_token_id(self) -> int:
        return self.canvas_token_id if self.canvas_token_id is not None else self.mask_token_id

    @property
    def tokens_per_step(self) -> int:
        return self.max_completion_length // self.sampling_steps

    def prompt_fill(self) -> int:
        return self.noise_token_id if self.family == "block" else self.pad_token_id
