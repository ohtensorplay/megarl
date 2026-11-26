"""Flow-GRPO arXiv 2505.05470 and DiffNFT (ROLL diffusion pipeline
refs/ROLL/roll/pipeline/diffusion); see package docstring for details."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class FlowConfig:
    """Knobs for continuous flow-matching RL."""

    objective: str = "flow_grpo"
    """``flow_grpo`` (replay SDE steps) or ``diffnft`` (re-noise x0)."""
    sde_type: str = "sde"
    """``sde`` (exact Gaussian) or ``cps`` (constants-dropped variant)."""
    noise_level: float = 0.7
    """ETA / noise scale in ROLL's ``FlowMatchSDEDiscreteScheduler``."""
    timestep_scale: float = 1.0
    """Multiply normalised sigma by this before calling the model (1000 for ROLL)."""
    num_train_steps: int | None = None
    """Train on the first K steps; ``None`` trains on every recorded step."""
    num_nft_timesteps: int = 4
    """Timesteps sampled per sample for the DiffNFT objective."""
    pg_clip: float = 0.2
    """FlowGRPO importance-ratio clip (two sided)."""
    advantage_clip: float | None = None
    """Optional absolute advantage clip (ROLL's ``advantage_clip``)."""
    kl_coef: float = 0.0
    """Coefficient on a KL(new || old/ref) penalty."""
    kl_type: str = "k2"
    """ KL estimator; k2 is the low-variance Gaussian-friendly default."""
    nft_beta: float = 0.05
    """DiffNFT positive/negative interpolation coefficient (must be > 0)."""
    adv_clip_max: float = 1.0
    """DiffNFT loss scale (ROLL's ``adv_clip_max``)."""
    ref_kl_coef: float = 0.0
    """Optional coefficient on ``(new_pred - ref_pred)^2`` for DiffNFT."""
    use_adaptive_weight: bool = False
    """Normalise DiffNFT error magnitudes per sample."""
    seed: int = 0

    def __post_init__(self) -> None:
        if self.objective not in ("flow_grpo", "diffnft"):
            raise ValueError(f"unknown flow objective: {self.objective}")
        if self.sde_type not in ("sde", "cps"):
            raise ValueError(f"unknown sde_type: {self.sde_type}")
        if not 0.0 < self.noise_level <= 1.0:
            raise ValueError("noise_level must be in (0, 1]")
        if self.timestep_scale <= 0:
            raise ValueError("timestep_scale must be positive")
        if self.num_train_steps is not None and self.num_train_steps < 1:
            raise ValueError("num_train_steps must be >= 1")
        if self.num_nft_timesteps < 1:
            raise ValueError("num_nft_timesteps must be >= 1")
        if self.pg_clip < 0:
            raise ValueError("pg_clip cannot be negative")
        if self.advantage_clip is not None and self.advantage_clip <= 0:
            raise ValueError("advantage_clip must be positive when set")
        if self.kl_coef < 0 or self.ref_kl_coef < 0:
            raise ValueError("KL coefficients cannot be negative")
        if self.nft_beta <= 0:
            raise ValueError("nft_beta must be positive")
