"""dOPSD — On-Policy Self-Distillation for Diffusion Language Models (arXiv 2607.04428).

Autoregressive OPSD gives a model a privileged, instance-specific reference
solution and distils its per-token distribution into the same model conditioned
only on the question. On dLLMs that privilege is unavailable at inference, so
the student can only learn a weak PI-marginalised consensus, and the random
re-masking used to build student states is off the model's confidence-ordered
decoding path. dOPSD draws *both* the student's noise and the teacher's
privilege from one object: the student's own decoding trajectory.

For a genuine intermediate state ``s_k`` with still-masked set ``M_k``, the
student scores the masked positions there. The teacher is the same model
evaluated at later states of that trajectory where a position is *still*
masked (otherwise the model would merely copy it); its predictions are
averaged over those steps into a stop-gradient target ``p̄_i``. The student is
trained with a token-level generalised Jensen--Shannon divergence

    JSD_beta(pT || pS) = beta*KL(pT || m) + (1-beta)*KL(pS || m),
    m = beta*pT + (1-beta)*pS,

where ``beta=0.5`` is the ordinary JS divergence. Rollouts are verified
against the short final answer and only correct ones contribute a gradient.

Bulk trajectory arrays (per-step canvases and masks) travel out of band via
:class:`DOPSDStore`; the Rust metadata frame carries only ``payload_handle``.

References
----------
Phuong Tuan Dat, Qi Li, Xinchao Wang. "dOPSD: On-Policy Self-Distillation for
Diffusion Language Models." arXiv:2607.04428.
Code: https://github.com/tuandattt/dOPSD
"""

from __future__ import annotations

from .batching import (
    DOPSDBatch,
    DOPSDTensorBatch,
    collate_dopsd,
    make_dopsd_packer,
    slice_dopsd_batch,
)
from .config import DOPSDConfig
from .controller import AsyncDOPSDController, DOPSDController
from .loss import dopsd_loss, generalized_jsd
from .rollout import DOPSDGenerator, DOPSDRollout
from .trainer import DOPSDTrainer
from .types import (
    DOPSDGenerationResult,
    DOPSDRecord,
    DOPSDStep,
    DOPSDStore,
    decode_dopsd,
    encode_dopsd,
)

__all__ = [
    "AsyncDOPSDController",
    "DOPSDBatch",
    "DOPSDConfig",
    "DOPSDController",
    "DOPSDGenerationResult",
    "DOPSDGenerator",
    "DOPSDRecord",
    "DOPSDRollout",
    "DOPSDStep",
    "DOPSDStore",
    "DOPSDTensorBatch",
    "DOPSDTrainer",
    "collate_dopsd",
    "decode_dopsd",
    "dopsd_loss",
    "encode_dopsd",
    "generalized_jsd",
    "make_dopsd_packer",
    "slice_dopsd_batch",
]
