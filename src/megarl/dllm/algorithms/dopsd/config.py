"""dOPSD, Phuong Tuan Dat, Qi Li, Xinchao Wang, arXiv 2607.04428; code
https://github.com/tuandattt/dOPSD; see package docstring for details."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class DOPSDConfig:
    """Knobs for trajectory-derived privileged self-distillation."""

    family: str = "masked"
    mask_token_id: int = 126336
    canvas_token_id: int | None = None
    vocab_size: int | None = None
    max_completion_length: int = 128
    sampling_steps: int = 32
    """Number of decoding steps recorded in the rollout trajectory."""
    mask_threshold: float = 0.3
    """Sample student steps whose masked fraction is above this threshold."""
    js_weight: float = 0.5
    """JSD mixing weight; 0.5 is the ordinary Jensen--Shannon divergence."""
    max_scored_positions: int = 16
    """Cap on scored masked positions per rollout (bounds the teacher batch)."""
    max_teacher_steps: int = 4
    """Cap on future states averaged into the teacher target."""
    temperature: float = 1.0
    verify_rollouts: bool = True
    """Keep only rollouts whose reward clears ``correct_reward_threshold``."""
    correct_reward_threshold: float = 0.0
    prompt_mask_prob: float = 0.0
    """Optional prompt corruption in the forward pass (0 keeps prompts clean)."""
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
        if not 0.0 <= self.mask_threshold < 1.0:
            raise ValueError("mask_threshold must be in [0, 1)")
        if not 0.0 <= self.js_weight <= 1.0:
            raise ValueError("js_weight must be in [0, 1]")
        if self.max_scored_positions <= 0 or self.max_teacher_steps <= 0:
            raise ValueError("position/teacher caps must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.prompt_mask_prob <= 1.0:
            raise ValueError("prompt_mask_prob must be in [0, 1]")
        if self.version < 0:
            raise ValueError("version cannot be negative")
        if self.family == "block" and (
            self.vocab_size is None or self.vocab_size <= 0
        ):
            raise ValueError("block diffusion requires a positive vocab_size")

    @property
    def noise_token_id(self) -> int:
        return self.canvas_token_id if self.canvas_token_id is not None else self.mask_token_id

    def prompt_fill(self) -> int:
        return self.noise_token_id if self.family == "block" else 0
