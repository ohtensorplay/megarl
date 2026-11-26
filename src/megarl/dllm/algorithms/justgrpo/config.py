"""JustGRPO / The Flexibility Trap, Zanlin Ni et al., ICML 2026 arXiv 2601.15165; code
https://github.com/LeapLabTHU/JustGRPO; see package docstring for details."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class JustGRPOConfig:
    """Knobs for AR-order GRPO on a masked / block dLLM."""

    family: str = "masked"
    """``masked`` (LLaDA/Dream, full attention) or ``block`` (DiffusionGemma)."""
    mask_token_id: int = 126336
    pad_token_id: int = 0
    canvas_token_id: int | None = None
    """Noise token for block diffusion; defaults to ``mask_token_id``."""
    vocab_size: int | None = None
    max_new_tokens: int = 256
    """Completion length ``L``; also the loss normaliser."""
    temperature: float = 1.0
    """Rollout and log-prob temperature (must match between the two)."""
    epsilon_low: float = 0.2
    epsilon_high: float | None = None
    kl_coef: float = 0.0
    """Beta in ``J - beta * KL(new || ref)``; 0 disables the reference pass."""
    kl_type: str = "k3"
    fast_fraction: float = 1.0
    """Fraction of highest-entropy positions used by the loss (1.0 = exact)."""
    eos_token_ids: tuple[int, ...] = (1,)
    seed: int = 0
    version: int = 0
    max_prompt_len: int | None = None
    """Optional fixed prompt span; defaults to the batch maximum."""

    def __post_init__(self) -> None:
        if self.family not in ("masked", "block"):
            raise ValueError(f"unknown dLLM family: {self.family}")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 < self.fast_fraction <= 1.0:
            raise ValueError("fast_fraction must be in (0, 1]")
        if self.epsilon_low < 0:
            raise ValueError("epsilon_low cannot be negative")
        if self.kl_coef < 0:
            raise ValueError("kl_coef cannot be negative")
        if self.kl_type not in ("k1", "k2", "k3", "low_var_kl"):
            raise ValueError(f"unknown kl_type: {self.kl_type}")
        if self.version < 0:
            raise ValueError("version cannot be negative")

    @property
    def noise_token_id(self) -> int:
        return self.canvas_token_id if self.canvas_token_id is not None else self.mask_token_id

    def prompt_fill(self) -> int:
        return self.noise_token_id if self.family == "block" else self.pad_token_id

    def selected_positions(self, L: int) -> int:
        return max(1, min(L, round(self.fast_fraction * L)))
