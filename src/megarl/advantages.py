"""Advantage estimators for policy-gradient training.

GRPO is the default; RLOO and REINFORCE leave-one-out baselines are provided
for the same group-shaped rewards. Token-level helpers place sparse outcome
rewards on the last valid token and compute GAE with an explicit bootstrap
slot, so padding can never leak a value estimate back into a live episode.
"""

from __future__ import annotations

import math
from typing import Sequence

import tensorplay as tp

ADVANTAGE_ESTIMATORS = ("grpo", "rloo", "reinforce")


def _stats(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def group_advantages(
    groups: Sequence[Sequence[float]],
    eps: float = 1e-4,
    *,
    scale_rewards: str = "group",
) -> list[float]:
    """Per-group centered advantages.

    ``scale_rewards`` selects the standardizer: ``group`` uses each group's
    own std (GRPO), ``batch`` uses the std over all rewards in the batch, and
    ``none`` skips scaling entirely.
    """
    if scale_rewards not in ("group", "batch", "none"):
        raise ValueError(f"unknown reward scaling: {scale_rewards}")
    flat = [float(r) for group in groups for r in group]
    _, batch_std = _stats(flat)
    out: list[float] = []
    for rewards in groups:
        if not rewards:
            continue
        mean, std = _stats(rewards)
        if scale_rewards == "group":
            denom = std + eps
        elif scale_rewards == "batch":
            denom = batch_std + eps
        else:
            denom = 1.0
        out.extend((float(r) - mean) / denom for r in rewards)
    return out


def rloo_advantages(groups: Sequence[Sequence[float]], eps: float = 1e-4) -> list[float]:
    """Leave-one-out baseline: ``r_i - mean(r_others)``."""
    out: list[float] = []
    for rewards in groups:
        if len(rewards) < 2:
            continue
        total = sum(rewards)
        n = len(rewards)
        out.extend(float(r) - (total - float(r)) / (n - 1) for r in rewards)
    return out


def reinforce_advantages(groups: Sequence[Sequence[float]]) -> list[float]:
    """Group-mean baseline without std normalization."""
    out: list[float] = []
    for rewards in groups:
        if not rewards:
            continue
        mean = sum(rewards) / len(rewards)
        out.extend(float(r) - mean for r in rewards)
    return out


def estimate_advantages(
    groups: Sequence[Sequence[float]], estimator: str = "grpo"
) -> list[float]:
    if estimator == "grpo":
        return group_advantages(groups)
    if estimator == "rloo":
        return rloo_advantages(groups)
    if estimator == "reinforce":
        return reinforce_advantages(groups)
    raise ValueError(f"unknown advantage estimator: {estimator}")


def token_rewards(rewards, mask):
    """Place one outcome reward on the last valid token of each sequence."""
    reward_rows = rewards.flatten().tolist()
    mask_rows = mask.tolist()
    if len(reward_rows) != len(mask_rows):
        raise ValueError("rewards and mask must have the same number of rows")
    rows: list[list[float]] = []
    for reward, row in zip(reward_rows, mask_rows):
        values = [0.0] * len(row)
        valid = [i for i, value in enumerate(row) if float(value) > 0.0]
        if valid:
            values[valid[-1]] = float(reward)
        rows.append(values)
    return tp.tensor(rows, dtype=tp.float32)


def gae_advantages(
    rewards,
    values,
    mask,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize: bool = True,
) -> tuple:
    """Generalized advantage estimation using an explicit bootstrap slot.

    ``values`` has shape ``[B, L + 1]``: the extra column is the bootstrap
    value for the final valid token. A bootstrap is only used when the next
    token is itself valid, so padded values cannot contaminate advantages.
    """
    reward_rows = rewards.tolist()
    value_rows = values.tolist()
    mask_rows = mask.tolist()
    width = len(mask_rows[0]) if mask_rows else 0
    if any(len(row) != width + 1 for row in value_rows):
        raise ValueError("values must have shape [B, L + 1]")
    if len(reward_rows) != len(mask_rows) or len(value_rows) != len(mask_rows):
        raise ValueError("rewards, values and mask must have the same batch size")

    advantage_rows: list[list[float]] = []
    return_rows: list[list[float]] = []
    for r_row, v_row, m_row in zip(reward_rows, value_rows, mask_rows):
        adv = [0.0] * width
        ret = [0.0] * width
        next_gae = 0.0
        for t in range(width - 1, -1, -1):
            live = float(m_row[t]) > 0.0
            next_live = (float(m_row[t + 1]) > 0.0) if t + 1 < width else live
            bootstrap = v_row[t + 1] if next_live else 0.0
            delta = float(r_row[t]) + gamma * bootstrap - float(v_row[t])
            gae = delta + gamma * lam * next_gae * (1.0 if next_live else 0.0)
            adv[t] = gae if live else 0.0
            ret[t] = (gae + float(v_row[t])) if live else 0.0
            next_gae = gae
        advantage_rows.append(adv)
        return_rows.append(ret)

    advantages = tp.tensor(advantage_rows, dtype=tp.float32)
    returns = tp.tensor(return_rows, dtype=tp.float32)
    if normalize:
        flat = [v for row in advantage_rows for v, m in zip(row, mask_rows[0]) if float(m) > 0]
        # Normalize with each row's own mask, not the first row's.
        flat = [
            v
            for row, m_row in zip(advantage_rows, mask_rows)
            for v, m in zip(row, m_row)
            if float(m) > 0
        ]
        mean, std = _stats(flat) if flat else (0.0, 1.0)
        advantages = (advantages - mean) / (std + 1e-4)
    return advantages, returns
