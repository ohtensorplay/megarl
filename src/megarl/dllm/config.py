"""dLLM reinforcement learning (d1 arXiv 2505.22765, wd1; see devdocs/dllm-rl.md); see
package docstring for details."""

from __future__ import annotations

import dataclasses
import random


@dataclasses.dataclass(frozen=True, slots=True)
class DLLMConfig:
    """Knobs for the dLLM rollout-to-train bridge.

    ``mask_token_id`` is the id the model uses for a fully-noised position
    (LLaDA: 126336; DiffusionGemma uses uniform random tokens instead, so set
    ``mask_token_id`` to its pad/unknown id only for bookkeeping — the canvas
    is built from ``canvas_token_id`` when ``family='block'``).
    """

    family: str = "masked"
    """``masked`` (full attention) or ``block`` (encoder prefix + canvas)."""
    mask_token_id: int = 126336
    canvas_token_id: int | None = None
    """Noise token for block diffusion; defaults to ``mask_token_id``."""
    parameterization: str = "mean"
    """``mean`` (plain denoiser) or ``mean_loo`` (leave-one-out posterior)."""
    estimator: str = "one_step"
    """``one_step`` (d1) or ``trajectory`` (per-step replay)."""
    noise_mode: str = "auto"
    """``mask`` (masked dLLM) or ``uniform`` (block diffusion canvas)."""
    vocab_size: int | None = None
    """Vocabulary size, required by ``noise_mode='uniform'``."""
    corruption_eps: float = 1e-4
    """Bounds for the block-diffusion corruption-rate schedule."""
    prompt_mask_prob: float = 0.3
    """d1's ``p_mask_prompt``: regularisation for repeated policy updates."""
    objective: str = "diffu_grpo"
    """``diffu_grpo`` (clipped ratio) or ``wd1`` (ratio-free weights)."""
    num_iterations: int = 1
    """Optimiser updates per sampled group; >1 needs ``behavior_model``."""
    kl_coef: float = 0.0
    """KL(new || ref) coefficient; needs a reference model when positive."""
    kl_type: str = "k3"
    epsilon_low: float = 0.2
    epsilon_high: float | None = None
    wd1_temperature: float = 1.0
    """ψ in wd1's ``w+ = softmax(ψ·A)`` / ``w- = softmax(-ψ·A)``."""
    max_completion_length: int | None = None
    """Fixed canvas length for block models; otherwise the batch maximum."""
    seed: int = 42
    """Base seed; per-sample seeds are ``seed + request/sample index``."""

    def __post_init__(self) -> None:
        if self.family not in ("masked", "block"):
            raise ValueError(f"unknown dLLM family: {self.family}")
        if self.parameterization not in ("mean", "mean_loo"):
            raise ValueError(f"unknown parameterization: {self.parameterization}")
        if self.estimator not in ("one_step", "trajectory"):
            raise ValueError(f"unknown estimator: {self.estimator}")
        if self.noise_mode not in ("auto", "mask", "uniform"):
            raise ValueError(f"unknown noise_mode: {self.noise_mode}")
        mode = self.noise_mode if self.noise_mode != "auto" else (
            "uniform" if self.family == "block" else "mask"
        )
        if mode == "uniform" and (
            self.vocab_size is None or self.vocab_size <= 0
        ):
            raise ValueError("uniform canvas noise requires a positive vocab_size")
        if not 0.0 < self.corruption_eps < 0.5:
            raise ValueError("corruption_eps must be in (0, 0.5)")
        if self.objective not in ("diffu_grpo", "wd1"):
            raise ValueError(f"unknown dLLM objective: {self.objective}")
        if not 0.0 <= self.prompt_mask_prob <= 1.0:
            raise ValueError("prompt_mask_prob must be in [0, 1]")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        if self.kl_coef < 0:
            raise ValueError("kl_coef cannot be negative")
        if self.wd1_temperature <= 0:
            raise ValueError("wd1_temperature must be positive")
        if self.epsilon_low < 0:
            raise ValueError("epsilon_low cannot be negative")

    @property
    def effective_noise_mode(self) -> str:
        if self.noise_mode != "auto":
            return self.noise_mode
        return "uniform" if self.family == "block" else "mask"

    @property
    def noise_token_id(self) -> int:
        if self.canvas_token_id is not None:
            return self.canvas_token_id
        return self.mask_token_id


@dataclasses.dataclass(frozen=True, slots=True)
class DiffuGRPOConfig(DLLMConfig):
    """Named d1 / diffu-GRPO recipe (NeurIPS 2025, arXiv 2504.12216).

    Identical to :class:`DLLMConfig` with the one-step estimator and the
    clipped-ratio objective pinned. Use this when the algorithm, not just the
    mechanism, is the configuration intent; :class:`DLLMTrainer` accepts it
    unchanged.
    """

    estimator: str = "one_step"
    objective: str = "diffu_grpo"
