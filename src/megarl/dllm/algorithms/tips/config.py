"""TIPS config — TIPS: A Tiny Inference-time Policy for Scaling in Diffusion Language
Models (Zhang et al., arXiv:2610.22099; code https://github.com/zhengyz97/TIPS).
"""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True, slots=True)
class TIPSConfig:
    """Configuration for the tiny inference-time visibility policy."""

    family: str = "masked"
    mask_token_id: int = 126336
    canvas_token_id: int | None = None
    vocab_size: int | None = None
    max_completion_length: int = 128
    budget: int = 8
    gamma: float = 0.0
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    time_dim: int = 16
    clip_low: float = 0.5
    clip_high: float = 1.5
    temperature: float = 0.0
    warmup_beta: float = 4.0
    calibration_iters: int = 12
    max_prompt_len: int | None = None
    eos_token_ids: tuple[int, ...] = (1,)
    seed: int = 42
    version: int = 0

    def __post_init__(self) -> None:
        if self.family not in ("masked", "block"):
            raise ValueError("family must be 'masked' or 'block'")
        if self.max_completion_length <= 0 or self.budget < 1:
            raise ValueError("max_completion_length must be > 0 and budget >= 1")
        if self.hidden <= 0 or self.layers < 1 or self.heads < 1:
            raise ValueError("hidden/layers/heads must be positive")
        if self.hidden % self.heads != 0:
            raise ValueError("hidden must be divisible by heads")
        if self.time_dim <= 0 or self.time_dim % 2 != 0:
            raise ValueError("time_dim must be a positive even number")
        if self.calibration_iters < 1:
            raise ValueError("calibration_iters must be >= 1")
        if not (0.0 < self.clip_low < 1.0 < self.clip_high):
            raise ValueError("require 0 < clip_low < 1 < clip_high")
        if self.temperature < 0.0:
            raise ValueError("temperature cannot be negative")
        if self.version < 0:
            raise ValueError("version cannot be negative")
        if self.family == "block" and self.vocab_size is None:
            raise ValueError("block family requires vocab_size")

    @property
    def noise_token_id(self) -> int:
        return self.mask_token_id if self.canvas_token_id is None else self.canvas_token_id

    def prompt_fill(self) -> int:
        return self.noise_token_id

    def schedule(self, tau: float) -> float:
        """Expected visible fraction at normalised decoding time ``tau``."""
        tau = min(1.0, max(0.0, float(tau)))
        return tau ** math.exp(self.gamma)

    @property
    def train_steps(self) -> int:
        """Number of stochastic (non-final) decoding decisions."""
        return max(0, self.budget - 1)
