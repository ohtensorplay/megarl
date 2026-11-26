"""T3D, Tunyu Zhang et al., NeurIPS 2026 arXiv 2602.12262; code
https://github.com/Tyrion58/T3D; see package docstring for details."""

from __future__ import annotations

import dataclasses
from typing import Sequence


@dataclasses.dataclass(frozen=True, slots=True)
class T3DConfig:
    """Knobs for few-step trajectory self-distillation with DDO."""

    family: str = "masked"
    mask_token_id: int = 126336
    canvas_token_id: int | None = None
    vocab_size: int | None = None
    max_completion_length: int = 128
    teacher_steps: int = 32
    """Steps used by the full-step teacher rollout."""
    student_steps: int = 2
    """Decoding-step budget B used by the path-consistency weights."""
    max_train_steps: int = 8
    """Cap on intermediate states used per teacher trajectory."""
    max_masked_positions: int = 16
    """Cap on DDO positions per state."""
    path_weight: float = 1.0
    """Lambda on the path-consistency term."""
    ddo_temperature: float = 1.0
    """Temperature dividing the log-likelihood-ratio before the sigmoid."""
    ratio_clip: float = 20.0
    """Clamp on the log-ratio for numerical stability."""
    refresh_every: int = 1
    """Refresh the frozen reference snapshot every optimiser step."""
    temperature: float = 1.0
    prompt_mask_prob: float = 0.0
    eos_token_ids: tuple[int, ...] = (1,)
    seed: int = 42
    version: int = 0
    max_prompt_len: int | None = None

    def __post_init__(self) -> None:
        if self.family not in ("masked", "block"):
            raise ValueError(f"unknown dLLM family: {self.family}")
        if self.max_completion_length <= 0:
            raise ValueError("max_completion_length must be positive")
        if self.teacher_steps <= 0 or self.student_steps <= 0:
            raise ValueError("teacher/student step budgets must be positive")
        if self.max_train_steps <= 0 or self.max_masked_positions <= 0:
            raise ValueError("training caps must be positive")
        if self.path_weight < 0:
            raise ValueError("path_weight cannot be negative")
        if self.ddo_temperature <= 0 or self.temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.ratio_clip <= 0:
            raise ValueError("ratio_clip must be positive")
        if self.refresh_every < 1:
            raise ValueError("refresh_every must be >= 1")
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

    def path_weights(self, decode_steps: Sequence[int]) -> list[float]:
        """``w_i = (B - pi_i + 1) / B`` (paper eq. 9)."""
        # Decode steps are 0-based here; the paper's pi is 1-based, hence
        # w_i = (B - pi_i + 1)/B = (B - step)/B.
        return [
            max(1, self.student_steps - int(step)) / self.student_steps
            for step in decode_steps
        ]
