"""Stepwise dLLM credit assignment (TraceRL ICLR 2026 arXiv 2509.06949; Schulman et al.
arXiv 1506.02438); see package docstring for details."""

from __future__ import annotations

from typing import Sequence


def process_gae(
    rewards: Sequence[float],
    values: Sequence[Sequence[float]],
    mask: Sequence[Sequence[int]],
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[list[list[float]], list[list[float]]]:
    """GAE over denoising steps; the outcome reward lands on the last valid step.

    Returns ``(advantages, returns)`` with the same ``[sample, step]`` shape as
    ``values``. Steps with ``mask == 0`` are skipped without carrying the GAE
    recursion across them, which mirrors AReaL's staleness masking.
    """
    if len(values) != len(rewards) or len(values) != len(mask):
        raise ValueError("process_gae needs aligned rewards/values/mask rows")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= lam <= 1.0:
        raise ValueError("gamma and lam must be in [0, 1]")
    advantages: list[list[float]] = []
    returns: list[list[float]] = []
    for reward, row_values, row_mask in zip(rewards, values, mask):
        width = len(row_values)
        if len(row_mask) != width:
            raise ValueError("process_gae value/mask rows differ in width")
        valid = [index for index, keep in enumerate(row_mask) if keep]
        adv_row = [0.0] * width
        ret_row = [0.0] * width
        gae = 0.0
        next_value = 0.0
        for index in reversed(valid):
            terminal = float(reward) if index == valid[-1] else 0.0
            delta = terminal + gamma * next_value - float(row_values[index])
            gae = delta + gamma * lam * gae
            adv_row[index] = gae
            ret_row[index] = gae + float(row_values[index])
            next_value = float(row_values[index])
        advantages.append(adv_row)
        returns.append(ret_row)
    return advantages, returns
