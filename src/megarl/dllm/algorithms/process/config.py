"""Stepwise dLLM credit assignment (TraceRL ICLR 2026 arXiv 2509.06949; Schulman et al.
arXiv 1506.02438); see package docstring for details."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class ProcessRewardConfig:
    """GAE knobs for stepwise dLLM credit assignment."""

    gamma: float = 0.99
    lam: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam must be in [0, 1]")
