"""GDSD, Xiaohang Tang et al., arXiv 2605.29398; code https://github.com/GaryBall/GDSD;
see package docstring for details."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class GDSDConfig:
    """Knobs for advantage-guided denoiser self-distillation."""

    family: str = "masked"
    """``masked`` (LLaDA/Dream) or ``block`` (DiffusionGemma/SDAR)."""
    mask_token_id: int = 126336
    canvas_token_id: int | None = None
    vocab_size: int | None = None
    max_completion_length: int = 128
    """Completion length L; the loss divides log-ratios by this value."""
    sampling_steps: int = 32
    """Number of iterative remasking steps used by the rollout generator."""
    mc_timesteps: int = 4
    """K diffusion times sampled uniformly per completion."""
    coupled_views: bool = True
    """Average two complementary corruption views (variance reduction)."""
    psi: float = 1.0
    """Guidance strength on the advantage in the teacher."""
    beta: float = 0.1
    """Weight of the reference regulariser; 0 disables the reference pass."""
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    """Log-ratio clipping bounds, matching PPO's ``[1-e, 1+e]``."""
    tlc: bool = True
    """Token-level logit centralisation (paper eq. 8)."""
    prompt_mask_prob: float = 0.3
    corruption_eps: float = 1e-4
    temperature: float = 1.0
    refresh_every: int = 1
    """Refresh the frozen old policy every mu optimiser steps (paper's mu)."""
    eos_token_ids: tuple[int, ...] = (1,)
    seed: int = 42
    version: int = 0
    max_prompt_len: int | None = None

    def __post_init__(self) -> None:
        if self.family not in ("masked", "block"):
            raise ValueError(f"unknown dLLM family: {self.family}")
        if self.max_completion_length <= 0:
            raise ValueError("max_completion_length must be positive")
        if self.sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if self.mc_timesteps <= 0:
            raise ValueError("mc_timesteps must be positive")
        if not self.coupled_views and self.mc_timesteps < 1:
            raise ValueError("mc_timesteps must be positive")
        if self.psi <= 0:
            raise ValueError("psi must be positive")
        if self.beta < 0:
            raise ValueError("beta cannot be negative")
        if self.epsilon_low < 0 or self.epsilon_high < 0:
            raise ValueError("clipping bounds cannot be negative")
        if not 0.0 <= self.prompt_mask_prob <= 1.0:
            raise ValueError("prompt_mask_prob must be in [0, 1]")
        if not 0.0 < self.corruption_eps < 0.5:
            raise ValueError("corruption_eps must be in (0, 0.5)")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.refresh_every < 1:
            raise ValueError("refresh_every must be >= 1")
        if self.version < 0:
            raise ValueError("version cannot be negative")
        if self.family == "block" and (
            self.vocab_size is None or self.vocab_size <= 0
        ):
            raise ValueError("block diffusion requires a positive vocab_size")

    @property
    def noise_token_id(self) -> int:
        return self.canvas_token_id if self.canvas_token_id is not None else self.mask_token_id

    @property
    def rows_per_sample(self) -> int:
        return self.mc_timesteps * (2 if self.coupled_views else 1)

    def prompt_fill(self) -> int:
        return self.noise_token_id if self.family == "block" else 0
