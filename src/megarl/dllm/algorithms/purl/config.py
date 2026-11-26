"""PURL config — When Should Diffusion Language Models Reconsider Their Commitments? (Zheng
et al., arXiv:2611.11551v1; withdrawn after v1).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class PURLConfig:
    family: str = "masked"
    mask_token_id: int = 126336
    canvas_token_id: int | None = None
    vocab_size: int = 32000
    max_completion_length: int = 128
    unmask_budget: int = 1
    edit_budget: int = 1
    revision_gate: float = 0.0
    feature_dim: int | None = None
    beta_e: float = 4.0
    beta_u: float = 4.0
    lambda_e: float = 1.0
    lambda_u: float = 1.0
    lambda_theta: float = 1.0
    w_mask: float = 1.0
    w_noisy: float = 1.0
    w_correct: float = 1.0
    corrupt_prob: float = 0.25
    mask_prob: float = 0.5
    seed: int = 42
    max_prompt_len: int | None = None
    eos_token_ids: tuple[int, ...] = (1,)
    version: int = 0

    def __post_init__(self) -> None:
        if self.family not in ("masked", "block"):
            raise ValueError("family must be 'masked' or 'block'")
        if self.vocab_size <= 0 or self.max_completion_length <= 0:
            raise ValueError("vocab_size and max_completion_length must be positive")
        if self.unmask_budget < 1 or self.edit_budget < 0:
            raise ValueError("unmask_budget >= 1 and edit_budget >= 0 required")
        if not 0.0 <= self.revision_gate <= 1.0:
            raise ValueError("revision_gate must be in [0, 1]")
        if not 0.0 <= self.corrupt_prob <= 1.0 or not 0.0 < self.mask_prob <= 1.0:
            raise ValueError("probabilities must be in range")
        if min(self.beta_e, self.beta_u) <= 0.0:
            raise ValueError("positive-class weights must be > 0")
        if min(self.lambda_e, self.lambda_u, self.lambda_theta) < 0.0:
            raise ValueError("block weights cannot be negative")
        if self.version < 0:
            raise ValueError("version cannot be negative")

    @property
    def noise_token_id(self) -> int:
        return self.mask_token_id if self.canvas_token_id is None else self.canvas_token_id

    def prompt_fill(self) -> int:
        return self.noise_token_id

    @property
    def planner_dim(self) -> int:
        return self.feature_dim or self.vocab_size
