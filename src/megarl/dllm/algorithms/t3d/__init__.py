"""T3D — Trajectory Self-Distillation via Direct Discriminative Optimization
(arXiv 2602.12262).

Full-step decoding is the quality ceiling for masked diffusion LMs, but cutting
the step budget aggressively over-smooths the student: a forward-KL trajectory
match is mode-covering when the posterior at a coarse intermediate state is
highly multimodal. T3D distils the teacher's *own* denoising trajectories with
Direct Discriminative Optimization, a reverse-KL-style, mode-seeking objective
that contrasts teacher samples against samples from a frozen reference model:

    L_ddo = -log sigma( E_{x0 ~ teacher}[ log p_theta(x0|xt) / p_ref(x0|xt) ] )
            -log(1 - sigma( E_{x0 ~ ref}[ log p_theta(x0|xt) / p_ref(x0|xt) ] ))

A path-consistency term emphasises tokens decoded early (where a mistake would
cascade under few-step decoding), with ``w_i = (B - pi_i + 1) / B``:

    L = L_ddo + lambda * L_path.

The reference model is a stop-gradient snapshot of the student, refreshed
every ``refresh_every`` optimiser steps. The student and reference are scored
on the *same* intermediate states, so the log-ratio is exact under the one-step
denoiser estimator -- no ELBO surrogate is needed.

Bulk trajectory arrays (per-step canvases/masks, decode steps, reference
completions) travel out of band via :class:`T3DStore`; the Rust metadata frame
carries only ``payload_handle``.

References
----------
Tunyu Zhang, Xinxi Zhang, Ligong Han, Haizhou Shi, Xiaoxiao He, Zhuowei Li,
Hao Wang, Kai Xu. "T3D: Few-Step Diffusion Language Models via Trajectory
Self-Distillation with Direct Discriminative Optimization." NeurIPS 2026.
arXiv:2602.12262. Code: https://github.com/Tyrion58/T3D
"""

from __future__ import annotations

from .batching import (
    T3DBatch,
    T3DTensorBatch,
    collate_t3d,
    make_t3d_packer,
    slice_t3d_batch,
)
from .config import T3DConfig
from .controller import AsyncT3DController, T3DController
from .loss import t3d_loss
from .rollout import T3DGenerator, T3DRollout
from .trainer import T3DTrainer
from .types import (
    T3DGenerationResult,
    T3DRecord,
    T3DStep,
    T3DStore,
    decode_t3d,
    encode_t3d,
)

__all__ = [
    "AsyncT3DController",
    "T3DBatch",
    "T3DConfig",
    "T3DController",
    "T3DGenerationResult",
    "T3DGenerator",
    "T3DRecord",
    "T3DRollout",
    "T3DStep",
    "T3DStore",
    "T3DTensorBatch",
    "T3DTrainer",
    "collate_t3d",
    "decode_t3d",
    "encode_t3d",
    "make_t3d_packer",
    "slice_t3d_batch",
    "t3d_loss",
]
