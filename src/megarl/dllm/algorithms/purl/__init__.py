"""PURL — Planned Unmasking-and-Revision Joint Learning (arXiv 2611.11551v1).

Standard absorbing decoding makes the first write to a token irreversible, so
conditionally independent parallel predictions can commit contradictions that
later context cannot repair. PURL replaces the one-way transition with a
*normalised joint kernel* whose likelihood factorises into three decisions:

    K(s+, E, U | s) = pi_E(E|s) * pi_U(U|s) * P_theta(s+ | s, E, U)
    -log K = L_E + L_U + L_theta

``E`` is the set of visible positions to *edit*, ``U`` the set of masked
positions to *unmask*, and ``P_theta`` the shared token predictor. All three are
supervised on partially masked sequences built from the model's own unmasking
policy, with sparse temporary corruptions supplying known error locations:

* edit planner: candidate-averaged weighted BCE against corrupted positions;
* unmask planner: weighted BCE against the stop-gradient teacher ``TopK`` by
  reference-token margin;
* token predictor: reweighted cross-entropy over masked (``w_M``), noisy
  (``w_N``) and correct visible (``w_C``) positions -- the last term is what
  teaches *retention*, without which revision introduces more errors than it
  fixes.

At inference one forward pass supplies both planner score maps and token
logits; two ``TopK`` calls pick ``U`` and ``E`` and the sampled tokens are
written simultaneously, so generation and revision share a single pass.

Wire contract: only tokens, rewards and a ``payload_handle`` cross the Rust
frame; reference completions and corruption masks stay out of band in
:class:`PURLStore`.

Note: arXiv:2611.11551 was withdrawn by its authors after v1; this module
implements the public v1 method as published.

References
----------
Haoyang Zheng, Yidong Ouyang, Blake Jianhang Chen, Tian Tan, Ziqian Lin,
Liqian Peng, Alec Go, Guang Lin. "When Should Diffusion Language Models
Reconsider Their Commitments?" arXiv:2611.11551v1.
"""

from __future__ import annotations

from .batching import (
    PURLBatch,
    PURLTensorBatch,
    collate_purl,
    make_purl_packer,
    slice_purl_batch,
)
from .config import PURLConfig
from .controller import AsyncPURLController, PURLController
from .loss import purl_loss
from .model import PURLModel, PURLPlanners
from .rollout import PURLGenerator, PURLRollout
from .trainer import PURLTrainer
from .types import PURLGenerationResult, PURLRecord, PURLStore, decode_purl, encode_purl

__all__ = [
    "AsyncPURLController",
    "PURLBatch",
    "PURLConfig",
    "PURLController",
    "PURLGenerationResult",
    "PURLGenerator",
    "PURLModel",
    "PURLPlanners",
    "PURLRecord",
    "PURLRollout",
    "PURLStore",
    "PURLTensorBatch",
    "PURLTrainer",
    "collate_purl",
    "decode_purl",
    "encode_purl",
    "make_purl_packer",
    "purl_loss",
    "slice_purl_batch",
]
