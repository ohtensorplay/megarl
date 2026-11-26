"""Direct preference optimization (DPO) on TensorPlay.

Ported from the pinned references (OpenRLHF ``models/loss.py``, ROLL
``pipeline/dpo/actor_worker.py``, AReaL ``functional.py``):

    logits = (log π(chosen) - log π(rejected))
             - (log π_ref(chosen) - log π_ref(rejected))
    L_sigmoid = -(1-ε)·log σ(β·logits) - ε·log σ(-β·logits)
    L_ipo     = (logits - 1/(2β))**2

``ref_*`` log-probabilities may come from a frozen reference model; when
omitted they are treated as zero (the policy itself is the implicit prior).
Log-probabilities are summed over the response tokens only, as in every
reference implementation. The loss is averaged over preference pairs.
"""

from __future__ import annotations

from typing import Sequence

import tensorplay as tp
import tensorplay.nn.functional as F

from .batching import RLSample

DPO_LOSS_TYPES = ("sigmoid", "ipo")

__all__ = ["DPO_LOSS_TYPES", "build_preference_pairs", "dpo_loss", "sequence_logps"]


def build_preference_pairs(
    groups: Sequence[Sequence[RLSample]],
) -> list[tuple[RLSample, RLSample]]:
    """Pick the best/worst trajectory of each group as a chosen/rejected pair.

    Groups without a strict reward spread carry no preference signal and are
    dropped, matching the zero-variance gate used by the RL controller.
    """
    pairs: list[tuple[RLSample, RLSample]] = []
    for group in groups:
        if len(group) < 2:
            continue
        best = max(group, key=lambda sample: float(sample.reward))
        worst = min(group, key=lambda sample: float(sample.reward))
        if float(best.reward) <= float(worst.reward):
            continue
        pairs.append((best, worst))
    return pairs


def sequence_logps(logits, token_ids, loss_mask):
    """Summed response-token log-probabilities, shape ``[B]``."""
    logp = tp.gather(
        F.log_softmax(logits, dim=-1), -1, token_ids.unsqueeze(-1)
    ).squeeze(-1)
    return (logp * loss_mask).sum(-1)


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    *,
    ref_chosen_logps=None,
    ref_rejected_logps=None,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
) -> tuple:
    """Return ``(scalar loss, diagnostics dict of tensors)`` for a pair batch."""
    if loss_type not in DPO_LOSS_TYPES:
        raise ValueError(
            f"unknown dpo loss_type: {loss_type}; choose one of {', '.join(DPO_LOSS_TYPES)}"
        )
    if beta <= 0:
        raise ValueError("dpo beta must be positive")
    if not 0.0 <= label_smoothing <= 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5]")
    if (ref_chosen_logps is None) != (ref_rejected_logps is None):
        raise ValueError("ref_chosen_logps and ref_rejected_logps must be given together")

    if ref_chosen_logps is None:
        policy_logits = policy_chosen_logps - policy_rejected_logps
    else:
        policy_logits = (policy_chosen_logps - policy_rejected_logps) - (
            ref_chosen_logps - ref_rejected_logps
        )

    if loss_type == "sigmoid":
        eps = label_smoothing
        per_pair_loss = -(1.0 - eps) * F.logsigmoid(beta * policy_logits) - eps * F.logsigmoid(
            -beta * policy_logits
        )
    else:  # IPO (length-normalized by the caller when configured)
        per_pair_loss = (policy_logits - 1.0 / (2.0 * beta)) ** 2

    loss = per_pair_loss.mean()
    zero = tp.zeros_like(policy_logits)
    one = tp.ones_like(policy_logits)
    accuracy = tp.where(policy_logits > 0, one, zero).mean()
    diagnostics = {
        "loss": loss,
        "dpo_logits": policy_logits.mean(),
        "reward_accuracy": accuracy,
        "reward_margin": policy_logits.mean(),
    }
    if ref_chosen_logps is None:
        diagnostics["chosen_reward"] = beta * policy_chosen_logps.detach().mean()
        diagnostics["rejected_reward"] = beta * policy_rejected_logps.detach().mean()
    else:
        diagnostics["chosen_reward"] = (
            beta * (policy_chosen_logps - ref_chosen_logps).detach().mean()
        )
        diagnostics["rejected_reward"] = (
            beta * (policy_rejected_logps - ref_rejected_logps).detach().mean()
        )
    return loss, diagnostics
