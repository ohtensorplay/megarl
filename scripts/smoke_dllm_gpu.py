#!/usr/bin/env python3
"""GPU smoke test: one real DiffusionGemma dLLM RL update.

Loads a DiffusionGemma block-diffusion checkpoint through MegaLM's native
adapter, builds a deterministic canvas with megaRL's dLLM collator, and runs a
real forward/backward/optimizer step using ``BlockDiffusionAdapter``.

The default checkpoint is Hugging Face's tiny test model (cached locally on
this workstation); pass ``--checkpoint`` for a production checkpoint. The
script fails loudly on non-finite loss/gradient and prints a JSON summary.

Example
-------
    python scripts/smoke_dllm_gpu.py --device cuda
"""

from __future__ import annotations

import argparse
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

from megarl.dllm import (  # noqa: E402
    BlockDiffusionAdapter,
    DLLMConfig,
    DLLMSample,
    DLLMTrainer,
    collate_dllm,
)
from megarl.trainer import TrainerConfig  # noqa: E402


def default_checkpoint() -> str:
    env = os.environ.get("MEGALM_TINY_DIFFUSIONGEMMA")
    if env:
        return env
    cache = Path.home() / ".cache/huggingface/hub"
    root = cache / "models--trl-internal-testing--tiny-DiffusionGemmaForBlockDiffusion"
    snapshots = sorted((root / "snapshots").glob("*")) if root.exists() else []
    if not snapshots:
        raise SystemExit(
            "no tiny DiffusionGemma checkpoint found; pass --checkpoint or set "
            "MEGALM_TINY_DIFFUSIONGEMMA"
        )
    return str(snapshots[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=default_checkpoint())
    parser.add_argument("--device", default="cuda" if tp.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not tp.cuda.is_available():
        raise SystemExit("CUDA requested but tensorplay reports no CUDA device")

    model, _ = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.checkpoint,
        device=args.device,
        dtype=tp.float32,
        kernel_policy=KernelPolicy(use_triton=False),
    )
    # MegaLM builds inference-first parameters (requires_grad=False). RL must
    # explicitly unfreeze them before the optimizer/trainer sees the model.
    model.requires_grad_(True)
    model.train()
    canvas_length = int(model.canvas_length)
    vocab = int(model.config.text.vocab_size)
    pad = int(model.config.pad_token_id)

    cfg = DLLMConfig(
        family="block",
        mask_token_id=pad,
        canvas_token_id=pad,
        vocab_size=vocab,
        noise_mode="uniform",
        estimator="one_step",
        objective="diffu_grpo",
        max_completion_length=canvas_length,
        seed=17,
    )
    samples = [
        DLLMSample(
            prompt_token_ids=[2, 100 + i, 200 + i],
            completion_token_ids=[
                (300 + i * 31 + position) % (vocab - 4) + 4
                for position in range(canvas_length)
            ],
            reward=1.0 if i == 0 else 0.0,
            version=0,
        )
        for i in range(max(1, args.samples))
    ]
    batch = collate_dllm(samples, cfg)
    trainer = DLLMTrainer(
        model,
        TrainerConfig(
            device=args.device,
            learning_rate=args.learning_rate,
            max_grad_norm=None,
        ),
        dllm_config=cfg,
        adapter=BlockDiffusionAdapter(),
    )
    advantages = [sample.reward for sample in samples]
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
    probe = next(p for p in model.parameters() if p.requires_grad)
    before = probe.detach().flatten()[:16].tolist()
    metrics = trainer.update(batch, advantages, dry_run=False)
    after = probe.detach().flatten()[:16].tolist()
    if before == after:
        raise SystemExit("smoke failed: optimizer step did not change the weights")
    loss = float(metrics["train/loss"])
    if not math.isfinite(loss) or not grads or not any(g > 0 for g in grads):
        raise SystemExit(f"smoke failed: loss={loss}, nonzero_grads={sum(g > 0 for g in grads)}")
    summary = {
        "checkpoint": args.checkpoint,
        "device": str(args.device),
        "canvas_length": canvas_length,
        "rows": batch.size,
        "loss": loss,
        "params_with_grad": len(grads),
        "grad_norm_sum": sum(grads),
        "step": int(metrics["train/step"]),
        "status": "ok",
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
