"""TIPS — Tiny Inference-time Policy for Scaling (arXiv 2610.22099).

A frozen masked diffusion language model separates *what* token to write from
*which* positions become visible. TIPS learns the latter with a roughly 600K
parameter auxiliary controller while the decoder stays frozen. The controller
acts on per-position visibility features

    C[i] = [ log(P_act / (1 - P_act)),  log(P_act / P_alt) ]

where ``P_act`` is the probability of the *active* token (the greedy proposal at
a masked position, the stored token at a visible one) and ``P_alt`` is the
probability of its strongest competitor. Visibility consistency -- measuring
support for the token whose visibility is being decided -- is what lets the
same action cover unmasking, retention and remasking.

The policy emits per-position logits ``b``. A shared scalar ``delta`` is solved
by Newton iteration so that ``mean(sigmoid(b + delta)) == kappa(t)`` with the
power-law schedule ``kappa(t) = ((t+1)/T) ** exp(gamma)``. Because the same shift
is applied everywhere, it controls *how many* positions are visible without
changing the policy's *ordering*. Actions are conditionally independent
Bernoulli draws, whose factorised likelihood makes a GRPO ratio exact:

    log Pr(a|s) = sum_i [ a_i log rho_i + (1-a_i) log(1-rho_i) ]
    L = -1/(G(T-1)) sum_{g,t} min(w A, clip(w, 0.5, 1.5) A)

An optional supervised warmup regresses the logits toward reference-derived
visibility utilities ``u`` in ``[-1, 1]`` with stop-gradient targets.

Wire contract: only tokens, rewards and a ``payload_handle`` cross the Rust
frame; the visibility features, calibration logits and Bernoulli actions stay
out of band in :class:`TIPSStore`.

References
----------
Ya-Shi Zhang, Jian Tang, Pierre-Andre Noel, Oleksiy Ostapenko. "TIPS: A Tiny
Inference-time Policy for Scaling in Diffusion Language Models." arXiv:2610.22099.
Code: https://github.com/zhengyz97/TIPS (paper-reported implementation)
"""

from __future__ import annotations

from .batching import (
    TIPSBatch,
    TIPSTensorBatch,
    collate_tips,
    make_tips_packer,
    slice_tips_batch,
)
from .config import TIPSConfig
from .controller import AsyncTIPSController, TIPSController
from .loss import tips_loss, tips_visibility_utility, tips_warmup_loss
from .policy import (
    TIPSPolicy,
    calibrate_tips_shift,
    tips_bernoulli_logprob,
    tips_visibility_features,
)
from .rollout import TIPSGenerator, TIPSRollout
from .trainer import TIPSTrainer
from .types import (
    TIPSGenerationResult,
    TIPSRecord,
    TIPSStep,
    TIPSStore,
    decode_tips,
    encode_tips,
)

__all__ = [
    "AsyncTIPSController",
    "TIPSBatch",
    "TIPSConfig",
    "TIPSController",
    "TIPSGenerationResult",
    "TIPSGenerator",
    "TIPSPolicy",
    "TIPSRecord",
    "TIPSRollout",
    "TIPSStep",
    "TIPSStore",
    "TIPSTensorBatch",
    "TIPSTrainer",
    "calibrate_tips_shift",
    "collate_tips",
    "decode_tips",
    "encode_tips",
    "make_tips_packer",
    "slice_tips_batch",
    "tips_bernoulli_logprob",
    "tips_loss",
    "tips_visibility_features",
    "tips_visibility_utility",
    "tips_warmup_loss",
]
