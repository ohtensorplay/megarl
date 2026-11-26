"""ReMaRL — Remasking with Reinforcement Learning (arXiv 2611.22576).

Learned unmasking policies cannot revise a token once it is committed. ReMaRL
keeps the diffusion model frozen and trains a small (roughly 0.3M parameter)
position policy that decides which tokens are unmasked, retained or remasked,
optimising those decisions directly for final-answer correctness with GRPO.

The decoding rule is P2' (a stochastic adaptation of P2). At a masked position
the confidence is the probability of the greedy proposal; at a visible position
it is the probability of the *stored* token, so selecting a visible position
retains it. The policy learns a residual correction to these scores:

    z_i = s_i + Delta_phi(f_i),   Delta_phi(f) = MLP(f) + w_skip^T f
    f_i = [s_i; m_i; e_i; u_i],   u_i = W_h LN(h_i) + b_h

The final MLP layer and the skip weights are zero-initialised, so training starts
from stochastic P2'. Positions are drawn by Gumbel top-k on ``z / tau``, i.e. an
ordered Plackett--Luce sample whose likelihood is available in closed form:

    PL(a; z/tau) = prod_j exp(z_{a_j}/tau) / sum_{i not in a_<j} exp(z_i/tau)

GRPO then maximises the clipped surrogate over per-decision likelihood ratios,
with group-normalised correctness advantages. The number of unmasked positions
per step follows the fixed P2 schedule ``k_t = ceil(L t / T)``.

Wire contract: only tokens, rewards and a ``payload_handle`` cross the Rust
frame; confidence features, Gumbel actions and Plackett--Luce log-probabilities
travel out of band through :class:`ReMaRLStore`.

References
----------
Egor Shibaev, Danila Biktimirov, Ivan Klimov, Artem Makoian,
Viacheslav Meshchaninov, Dmitry Vetrov. "ReMaRL: Learning to Unmask and Remask
in Frozen Diffusion Language Models." arXiv:2611.22576.
Code: https://github.com/EgorShibaev/remarl
"""

from __future__ import annotations

from .batching import (
    ReMaRLBatch,
    ReMaRLTensorBatch,
    collate_remarl,
    make_remarl_packer,
    slice_remarl_batch,
)
from .config import ReMaRLConfig
from .controller import AsyncReMaRLController, ReMaRLController
from .loss import remarl_loss
from .policy import ReMaRLPolicy, gumbel_topk, plackett_luce_logprob
from .rollout import ReMaRLGenerator, ReMaRLRollout
from .trainer import ReMaRLTrainer
from .types import (
    ReMaRLGenerationResult,
    ReMaRLRecord,
    ReMaRLStep,
    ReMaRLStore,
    decode_remarl,
    encode_remarl,
)

__all__ = [
    "AsyncReMaRLController",
    "ReMaRLBatch",
    "ReMaRLConfig",
    "ReMaRLController",
    "ReMaRLGenerationResult",
    "ReMaRLGenerator",
    "ReMaRLPolicy",
    "ReMaRLRecord",
    "ReMaRLRollout",
    "ReMaRLStep",
    "ReMaRLStore",
    "ReMaRLTensorBatch",
    "ReMaRLTrainer",
    "collate_remarl",
    "decode_remarl",
    "encode_remarl",
    "gumbel_topk",
    "make_remarl_packer",
    "remarl_loss",
    "slice_remarl_batch",
    "plackett_luce_logprob",
]
