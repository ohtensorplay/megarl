#!/usr/bin/env python3
"""GPU smoke: real-checkpoint updates for JustGRPO and AGRPO.

Loads the tiny DiffusionGemma block-diffusion checkpoint through MegaLM's
native adapter, runs AR-order (JustGRPO) and amortized step-level (AGRPO)
generation, then performs a real forward/backward/optimizer step for each and
audits gradients plus the resulting weight change.

Example
-------
    python scripts/smoke_dllm_algorithms_gpu.py --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

import tensorplay as tp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MEGALM_SRC = Path(
    os.environ.get("MEGALM_SRC", ROOT.parent / "megaLM" / "src")
)
if MEGALM_SRC.exists():
    sys.path.insert(0, str(MEGALM_SRC))

from megalm.kernels.ops import KernelPolicy  # noqa: E402
from megalm.models.diffusiongemma import DiffusionGemmaForBlockDiffusion  # noqa: E402

from megarl.dllm import DLLMSample, default_adapter  # noqa: E402
from megarl.dllm.algorithms.agrpo import (  # noqa: E402
    AGRPOConfig,
    AGRPOGenerator,
    AGRPOTrainer,
    collate_agrpo,
)
from megarl.dllm.algorithms.justgrpo import (  # noqa: E402
    JustGRPOConfig,
    JustGRPOGenerator,
    JustGRPOTrainer,
    collate_justgrpo,
)
from megarl.trainer import TrainerConfig  # noqa: E402


def default_checkpoint() -> str:
    env = os.environ.get("MEGALM_TINY_DIFFUSIONGEMMA")
    if env:
        return env
    root = (
        Path.home()
        / ".cache/huggingface/hub"
        / "models--trl-internal-testing--tiny-DiffusionGemmaForBlockDiffusion"
    )
    snapshots = sorted((root / "snapshots").glob("*")) if root.exists() else []
    if not snapshots:
        raise SystemExit("no tiny DiffusionGemma checkpoint found; pass --checkpoint")
    return str(snapshots[0])


def _load(checkpoint: str, device: str):
    model, _ = DiffusionGemmaForBlockDiffusion.from_pretrained(
        checkpoint,
        device=device,
        dtype=tp.float32,
        kernel_policy=KernelPolicy(use_triton=False),
    )
    # MegaLM parameters are inference-first (requires_grad=False).
    model.requires_grad_(True)
    model.train()
    return model


def _audit(model, trainer, batch, *, just: bool) -> dict:
    device = trainer.config.device
    advantages = (
        [batch.rewards[i] for i in range(batch.size)]
        if just
        else [0.0] * batch.size
    )
    probe = next(p for p in model.parameters() if p.requires_grad)
    before = float(probe.detach().abs().sum().flatten()[0])
    audit = trainer.update(batch, advantages, dry_run=True)
    grads = [
        float(p.grad.abs().sum().flatten()[0])
        for p in model.parameters()
        if p.grad is not None
    ]
    if not math.isfinite(float(audit["train/loss"])) or not any(g > 0 for g in grads):
        raise SystemExit(
            f"smoke failed in forward/backward: loss={audit['train/loss']}, "
            f"nonzero_grads={sum(g > 0 for g in grads)}"
        )
    metrics = trainer.update(batch, advantages, dry_run=False)
    after = float(probe.detach().abs().sum().flatten()[0])
    if before == after:
        raise SystemExit("smoke failed: optimizer step did not change the weights")
    return {
        "loss": float(metrics["train/loss"]),
        "params_with_grad": len(grads),
        "grad_norm_sum": sum(grads),
        "step": int(metrics["train/step"]),
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=default_checkpoint())
    parser.add_argument("--device", default="cuda" if tp.cuda.is_available() else "cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if str(args.device).startswith("cuda") and not tp.cuda.is_available():
        raise SystemExit("CUDA requested but tensorplay reports no CUDA device")

    model = _load(args.checkpoint, args.device)
    canvas = int(model.canvas_length)
    vocab = int(model.config.text.vocab_size)
    pad = int(model.config.pad_token_id)
    # A 262k-vocab logits tensor is ~37 MB per row; cap the rows per forward
    # through gradient accumulation so log_softmax fits in GPU memory.
    def config(accumulation: int) -> TrainerConfig:
        return TrainerConfig(
            device=args.device,
            learning_rate=args.learning_rate,
            max_grad_norm=None,
            gradient_accumulation_steps=accumulation,
        )

    # --- JustGRPO (AR order, top-50% entropy positions) -------------------
    tp.manual_seed(17)
    just_cfg = JustGRPOConfig(
        family="block",
        mask_token_id=pad,
        canvas_token_id=pad,
        vocab_size=vocab,
        max_new_tokens=canvas,
        temperature=1.0,
        fast_fraction=0.5,
        seed=17,
    )
    adapter = default_adapter(just_cfg)
    generated = JustGRPOGenerator(model, adapter).generate(
        [[2, 100, 200], [2, 101, 201]], config=just_cfg
    )
    just_samples = [
        DLLMSample([2, 100 + i, 200 + i], list(r.completion_ids), 1.0 if i == 0 else 0.0)
        for i, r in enumerate(generated.records)
    ]
    just_batch = collate_justgrpo(just_samples, generated.records, just_cfg)
    just_trainer = JustGRPOTrainer(
        model, config(8), justgrpo_config=just_cfg, adapter=adapter
    )
    just_result = _audit(model, just_trainer, just_batch, just=True)

    # --- AGRPO (amortized step-level, low-discrepancy MC) -----------------
    tp.manual_seed(19)
    model = _load(args.checkpoint, args.device)
    agrpo_cfg = AGRPOConfig(
        family="block",
        mask_token_id=pad,
        canvas_token_id=pad,
        vocab_size=vocab,
        max_completion_length=canvas,
        sampling_steps=8,
        block_length=4,
        mc_samples=2,
        temperature=1.0,
        group_size=2,
        seed=19,
    )
    adapter = default_adapter(agrpo_cfg)
    generated = AGRPOGenerator(model, adapter).generate(
        [[2, 100, 200], [2, 101, 201]], config=agrpo_cfg
    )
    agrpo_samples = [
        DLLMSample([2, 100 + i, 200 + i], list(r.completion_ids), 1.0 if i == 0 else 0.0)
        for i, r in enumerate(generated.records)
    ]
    agrpo_records = tuple(
        dataclasses.replace(r, group_id=1) for r in generated.records
    )
    agrpo_batch = collate_agrpo(agrpo_samples, agrpo_records, agrpo_cfg)
    agrpo_trainer = AGRPOTrainer(
        model, config(4), agrpo_config=agrpo_cfg, adapter=adapter
    )
    agrpo_result = _audit(model, agrpo_trainer, agrpo_batch, just=False)

    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "device": args.device,
                "canvas_length": canvas,
                "justgrpo": just_result,
                "agrpo": agrpo_result,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
