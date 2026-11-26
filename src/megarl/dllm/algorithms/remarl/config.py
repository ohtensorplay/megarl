"""REMARL config — ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language
Models (Shibaev et al., arXiv:2611.22576; code https://github.com/EgorShibaev/remarl).
"""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True, slots=True)
class ReMaRLConfig:
    family: str = "masked"
    mask_token_id: int = 126336
    canvas_token_id: int | None = None
    vocab_size: int | None = None
    max_completion_length: int = 128
    budget: int = 8
    tau: float = 1.0
    hidden_dim: int = 64
    pos_dim: int = 16
    proj_dim: int = 64
    mlp_hidden: int = 64
    clip_eps: float = 0.2
    seed: int = 42
    max_prompt_len: int | None = None
    eos_token_ids: tuple[int, ...] = (1,)
    version: int = 0

    def __post_init__(self) -> None:
        if self.family not in ("masked", "block"):
            raise ValueError("family must be 'masked' or 'block'")
        if self.max_completion_length <= 0 or self.budget < 1:
            raise ValueError("max_completion_length must be > 0 and budget >= 1")
        if self.tau <= 0.0:
            raise ValueError("tau must be > 0")
        if min(self.hidden_dim, self.pos_dim, self.proj_dim, self.mlp_hidden) <= 0:
            raise ValueError("policy dimensions must be positive")
        if not 0.0 < self.clip_eps < 1.0:
            raise ValueError("clip_eps must be in (0, 1)")
        if self.version < 0:
            raise ValueError("version cannot be negative")
        if self.family == "block" and self.vocab_size is None:
            raise ValueError("block family requires vocab_size")

    @property
    def noise_token_id(self) -> int:
        return self.mask_token_id if self.canvas_token_id is None else self.canvas_token_id

    def prompt_fill(self) -> int:
        return self.noise_token_id

    def keep_count(self, step: int) -> int:
        """P2 schedule: number of visible positions after non-final call ``step``."""
        return min(
            self.max_completion_length,
            max(1, math.ceil(self.max_completion_length * (step + 1) / self.budget)),
        )

    @property
    def train_steps(self) -> int:
        return max(0, self.budget - 1)
