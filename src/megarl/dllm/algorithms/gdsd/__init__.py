"""GDSD — Guided Denoiser Self-Distillation (arXiv 2605.29398).

The paper shows that replacing the intractable dLLM likelihood by its ELBO
inside PPO/GRPO introduces a training--inference mismatch: the random masking
used to estimate the ratio is not the distribution the sampler actually sees,
and exponentiating the resulting error destabilises training. GDSD removes the
ratio entirely. It derives the closed-form optimum of reverse-KL-regularised RL:

    log p*(x0 | xt) = (1-beta) log p_old(x0 | xt)
                    + beta log p_ref(x0 | xt) + psi * A(x0) - b

and distils the denoiser into that advantage-guided *self-teacher* with a
squared logit-matching objective. Because the teacher is built from frozen
policies plus a scalar advantage, no likelihood ratio -- and therefore no
normalisation constant -- is needed. Token-level logit centralisation (TLC)
subtracts the per-position vocabulary mean, which removes the unknown energy
normaliser while leaving the softmax distribution unchanged.

This module implements the practical objective (paper eq. 24):

    E_t [ ( log p_theta(x0|xt)/p_old(x0|xt) - psi*A )^2
        + beta * ( log p_theta(x0|xt)/p_ref(x0|xt) )^2 ]

using a low-variance *coupled* estimator: at each sampled diffusion time the
completion is corrupted once with probability ``t`` and once on the exact
complementary positions, and the two unbiased views are averaged. The old
policy is a frozen snapshot refreshed every ``refresh_every`` optimiser steps.

Wire contract: GDSD transports only tokens and rewards in the Rust frame; the
old/reference log-probs are recomputed on the trainer from deterministic noisy
canvases, so bulk arrays never cross the wire.

References
----------
Xiaohang Tang, Keyue Jiang, Che Liu, Qifang Zhao, Xiaoxiao Xu, Sangwoong Yoon,
Ilija Bogunovic. "GDSD: Reinforcement Learning as Guided Denoiser
Self-Distillation for Diffusion Language Models." arXiv:2605.29398.
Code: https://github.com/GaryBall/GDSD
"""

from __future__ import annotations

from .batching import (
    GDSDBatch,
    GDSDTensorBatch,
    collate_gdsd,
    make_gdsd_packer,
    slice_gdsd_batch,
)
from .config import GDSDConfig
from .controller import AsyncGDSDController, GDSDController
from .loss import gdsd_loss, gdsd_token_logprobs
from .rollout import GDSDGenerator, GDSDRollout
from .trainer import GDSDTrainer
from .types import GDSDGenerationResult, GDSDRecord

__all__ = [
    "AsyncGDSDController",
    "GDSDBatch",
    "GDSDConfig",
    "GDSDController",
    "GDSDGenerationResult",
    "GDSDGenerator",
    "GDSDRecord",
    "GDSDRollout",
    "GDSDTensorBatch",
    "GDSDTrainer",
    "collate_gdsd",
    "gdsd_loss",
    "gdsd_token_logprobs",
    "make_gdsd_packer",
    "slice_gdsd_batch",
]
