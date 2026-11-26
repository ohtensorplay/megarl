"""AGRPO — Amortized GRPO (ICML 2026, arXiv 2510.04019).

AGRPO formulates masked-dLLM generation as a multi-step MDP: state ``x_t`` is
the partially masked sequence, the action is the ``n/m`` tokens unmasked at
step ``t``, and the verifiable reward arrives at the last step. That gives an
*exact* stepwise likelihood ``pi(x_t | x_{t-1})`` (the product of the per-token
marginals at the unmasked positions) with no ELBO surrogate.

The full PPO-style objective averages over all ``m`` denoising steps. AGRPO
reinterprets that inner sum as an expectation over a uniformly drawn timestep
and estimates it with ``k << m`` Monte Carlo samples. The result is unbiased
and costs ``k`` forwards instead of ``m``. Two variance reducers are provided:

* **low-discrepancy sampling** — anti-correlated uniforms ``(perm+i+u)/S``
  that cover ``{1..m}`` evenly while staying marginally uniform;
* **entropy importance sampling** — draw ``t`` with probability proportional
  to the rollout's per-step entropy mass and correct with ``1/(m p_t)``.

Advantages are group-mean-centred (no std division, per the paper). The KL term
uses the standard Schulman k3 estimator; the paper reports beta=0 is best, so
the reference forward is only taken when ``kl_coef > 0``.

Wire contract: behaviour log-probs/entropies and unmask-step indices live in
an out-of-band :class:`AGRPOStore`; the Rust frame carries only
``payload_handle``. Nothing here imports the reference repository at runtime.

References
----------
Anthony Zhan. "Simple Policy Gradients for Reasoning with Diffusion Language
Models" (Amortized GRPO / AGRPO). ICML 2026. arXiv:2510.04019.
Code: https://github.com/probablyabot/agrpo
"""

from __future__ import annotations

from .batching import (
    AGRPOBatch,
    AGRPOTensorBatch,
    collate_agrpo,
    make_agrpo_packer,
    slice_agrpo_batch,
)
from .config import AGRPOConfig
from .controller import AGRPOController, AsyncAGRPOController
from .rollout import AGRPOGenerator, AGRPORollout, sample_mc_steps
from .trainer import AGRPOTrainer
from .types import (
    AGRPOGenerationResult,
    AGRPORecord,
    AGRPOStore,
    decode_agrpo,
    encode_agrpo,
)

__all__ = [
    "AGRPOBatch",
    "AGRPOConfig",
    "AGRPOController",
    "AsyncAGRPOController",
    "AGRPOGenerationResult",
    "AGRPOGenerator",
    "AGRPORecord",
    "AGRPORollout",
    "AGRPOStore",
    "AGRPOTensorBatch",
    "AGRPOTrainer",
    "collate_agrpo",
    "decode_agrpo",
    "encode_agrpo",
    "make_agrpo_packer",
    "sample_mc_steps",
    "slice_agrpo_batch",
]
