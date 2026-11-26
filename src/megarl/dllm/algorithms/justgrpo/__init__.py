"""JustGRPO — ICML 2026 "The Flexibility Trap" (arXiv 2601.15165).

The paper's finding is that arbitrary-order decoding tends to *bypass* the
high-entropy logical forks that matter for exploration, collapsing Pass@k
relative to strict left-to-right decoding. Its algorithm is deliberately
minimal: during RL, treat the dLLM as an autoregressive policy by observing the
prefix and masking the future, then apply standard GRPO with an *exact*
likelihood. The architecture is untouched (no causal mask is imposed), so the
checkpoint keeps its native parallel decoding at inference time.

Formulation (paper eq. 5-8)
---------------------------
For completion position ``k`` the model input is::

    x~_k = [o_1, ..., o_{k-1}, [MASK], ..., [MASK]]

and the AR policy reads the logits at position ``k``::

    pi_AR(o_k | o_<k, q) = softmax(f_theta,k(x~_k, q))

so ``log pi_AR(o|q) = sum_k log pi_AR(o_k | o_<k, q)`` is exact. The GRPO
objective is the usual clipped surrogate with group-standardised advantages.
This module feeds one ``(sample, position)`` pair per trainer row and reuses
megaRL's existing loss/trainer stack with a fixed ``1/L`` normaliser.

``JustGRPO-Fast`` records per-position entropy at rollout and, for the loss,
keeps only the top ``fast_fraction`` highest-entropy positions. The normaliser
stays ``L`` (not the selected count), so the objective is unchanged; only 75%
of the forwards are skipped when ``fast_fraction=0.25``.

Wire contract: bulk arrays (behaviour log-probs, entropies) live in an
out-of-band :class:`JustGRPOStore`; the Rust frame carries only
``payload_handle``. Nothing here imports the reference repository at runtime.

References
----------
Zanlin Ni, Shenzhi Wang, Yang Yue, Tianyu Yu, Weilin Zhao, Yeguo Hua,
Tianyi Chen, Jun Song, Cheng Yu, Bo Zheng, Gao Huang. "The Flexibility Trap:
Rethinking the Value of Arbitrary Order in Diffusion Language Models." ICML
2026 (Oral). arXiv:2601.15165. Code: https://github.com/LeapLabTHU/JustGRPO
"""

from __future__ import annotations

from .batching import (
    JustGRPOBatch,
    JustGRPOTensorBatch,
    collate_justgrpo,
    make_justgrpo_packer,
    slice_justgrpo_batch,
)
from .config import JustGRPOConfig
from .controller import AsyncJustGRPOController, JustGRPOController
from .rollout import JustGRPOGenerator, JustGRPORollout
from .trainer import JustGRPOTrainer
from .types import (
    JustGRPOGenerationResult,
    JustGRPORecord,
    JustGRPOStore,
    decode_justgrpo,
    encode_justgrpo,
)

__all__ = [
    "AsyncJustGRPOController",
    "JustGRPOBatch",
    "JustGRPOConfig",
    "JustGRPOController",
    "JustGRPOGenerationResult",
    "JustGRPOGenerator",
    "JustGRPORecord",
    "JustGRPORollout",
    "JustGRPOStore",
    "JustGRPOTensorBatch",
    "JustGRPOTrainer",
    "collate_justgrpo",
    "decode_justgrpo",
    "encode_justgrpo",
    "make_justgrpo_packer",
    "slice_justgrpo_batch",
]
