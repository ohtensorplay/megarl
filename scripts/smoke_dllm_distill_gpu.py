#!/usr/bin/env python3
"""GPU smoke: real-checkpoint updates for GDSD, dOPSD and T3D.

Loads the tiny DiffusionGemma block-diffusion checkpoint through MegaLM's
native adapter, runs each algorithm's generator, then performs a real
forward/backward/optimizer step and audits gradients plus the weight change.

Example
-------
    python scripts/smoke_dllm_distill_gpu.py --device cuda
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
MEGALM_SRC = Path(os.environ.get("MEGALM_SRC", ROOT.parent / "megaLM" / "src"))
if MEGALM_SRC.exists():
    sys.path.insert(0, str(MEGALM_SRC))

from megalm.kernels.ops import KernelPolicy  # noqa: E402
from megalm.models.diffusiongemma import DiffusionGemmaForBlockDiffusion  # noqa: E402

from megarl.contracts import RolloutRequest, SamplingParams, TrajectoryRecord  # noqa: E402
from megarl.dllm import default_adapter, from_dllm_record  # noqa: E402
from megarl.dllm.algorithms.dopsd import (  # noqa: E402
    DOPSDConfig, DOPSDGenerator, DOPSDRollout, DOPSDStore, DOPSDTrainer,
    make_dopsd_packer,
)
from megarl.dllm.algorithms.gdsd import (  # noqa: E402
    GDSDConfig, GDSDGenerator, GDSDRollout, GDSDTrainer, make_gdsd_packer,
)
from megarl.dllm.algorithms.t3d import (  # noqa: E402
    T3DConfig, T3DGenerator, T3DRollout, T3DStore, T3DTrainer, make_t3d_packer,
)
from megarl.trainer import TrainerConfig  # noqa: E402


def default_checkpoint() -> str:
    env = os.environ.get("MEGALM_TINY_DIFFUSIONGEMMA")
    if env:
        return env
    root = (
        Path.home() / ".cache/huggingface/hub"
        / "models--trl-internal-testing--tiny-DiffusionGemmaForBlockDiffusion"
    )
    snapshots = sorted((root / "snapshots").glob("*")) if root.exists() else []
    if not snapshots:
        raise SystemExit("no tiny DiffusionGemma checkpoint found; pass --checkpoint")
    return str(snapshots[0])


def _load(checkpoint: str, device: str, *, train: bool = True):
    model, _ = DiffusionGemmaForBlockDiffusion.from_pretrained(
        checkpoint, device=device, dtype=tp.float32,
        kernel_policy=KernelPolicy(use_triton=False),
    )
    if train:
        # MegaLM parameters are inference-first (requires_grad=False).
        model.requires_grad_(True)
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    return model


def _config(device: str, accum: int) -> TrainerConfig:
    return TrainerConfig(
        device=device, learning_rate=1e-5, max_grad_norm=None,
        gradient_accumulation_steps=accum,
    )


def _audit(model, trainer, batches, *, advantages, label: str) -> dict:
    results = []
    for batch in batches:
        probe = next(p for p in model.parameters() if p.requires_grad)
        before = float(probe.detach().abs().sum().flatten()[0])
        audit = trainer.update(batch, advantages(batch), dry_run=True)
        grads = [
            float(p.grad.abs().sum().flatten()[0])
            for p in model.parameters()
            if p.grad is not None
        ]
        if not math.isfinite(float(audit["train/loss"])) or not any(g > 0 for g in grads):
            raise SystemExit(
                f"{label} failed forward/backward: loss={audit['train/loss']}, "
                f"nonzero_grads={sum(g > 0 for g in grads)}"
            )
        metrics = trainer.update(batch, advantages(batch), dry_run=False)
        after = float(probe.detach().abs().sum().flatten()[0])
        if before == after:
            raise SystemExit(f"{label} failed: optimizer step did not change weights")
        results.append({
            "loss": float(metrics["train/loss"]),
            "params_with_grad": len(grads),
            "grad_norm_sum": sum(grads),
            "step": int(metrics["train/step"]),
        })
    return {"batches": results, "status": "ok"}


def _rollout_samples(records, prompt, *, store=None, rewards=(1.0, 0.0)):
    samples = []
    for i, record in enumerate(records):
        wire = TrajectoryRecord(
            request_id=1, prompt_len=len(prompt), token_ids=list(record.token_ids),
            logprobs=[0.0] * len(record.token_ids), top_logprobs=[],
            policy_versions=[0] * len(record.token_ids), reward=None,
            finish_reason=0, payload_handle=record.payload_handle,
        )
        samples.append(from_dllm_record(wire, prompt, reward=rewards[i % len(rewards)]))
    return samples


def _request(prompt, n: int, max_new_tokens: int) -> RolloutRequest:
    return RolloutRequest(
        1, 0, prompt, max_new_tokens, n,
        SamplingParams(1.0, 1.0, 0, None, False, 0), None,
    )


def gdsd_smoke(checkpoint, device) -> dict:
    tp.manual_seed(31)
    model = _load(checkpoint, device)
    ref = _load(checkpoint, device, train=False)
    canvas = int(model.canvas_length)
    vocab = int(model.config.text.vocab_size)
    pad = int(model.config.pad_token_id)
    L = min(8, canvas)
    cfg = GDSDConfig(
        family="block", mask_token_id=pad, canvas_token_id=pad, vocab_size=vocab,
        max_completion_length=L, sampling_steps=4, mc_timesteps=2,
        psi=1.0, beta=0.1, seed=31, max_prompt_len=4,
    )
    adapter = default_adapter(cfg)
    rollout = GDSDRollout(GDSDGenerator(model, adapter), cfg)
    records = list(rollout.generate(_request([2, 100, 200], 2, L)))
    samples = _rollout_samples(records, [2, 100, 200])
    batches, _ = make_gdsd_packer(cfg)(samples)
    trainer = GDSDTrainer(
        model, _config(device, 8), gdsd_config=cfg, adapter=adapter,
        old_model=None, ref_model=ref,
    )
    def advantages(batch):
        return [0.5 if s == 0 else -0.5 for s in batch.gdsd_sample_ids]
    return _audit(model, trainer, batches, advantages=advantages, label="GDSD")


def dopsd_smoke(checkpoint, device) -> dict:
    tp.manual_seed(37)
    model = _load(checkpoint, device)
    canvas = int(model.canvas_length)
    vocab = int(model.config.text.vocab_size)
    pad = int(model.config.pad_token_id)
    L = min(8, canvas)
    cfg = DOPSDConfig(
        family="block", mask_token_id=pad, canvas_token_id=pad, vocab_size=vocab,
        max_completion_length=L, sampling_steps=4, max_scored_positions=4,
        max_teacher_steps=2, verify_rollouts=True, correct_reward_threshold=0.5,
        seed=37, max_prompt_len=4,
    )
    store = DOPSDStore()
    adapter = default_adapter(cfg)
    rollout = DOPSDRollout(DOPSDGenerator(model, adapter), store, cfg)
    records = list(rollout.generate(_request([2, 100, 200], 2, L)))
    samples = _rollout_samples(records, [2, 100, 200])
    batches, _ = make_dopsd_packer(cfg, store)(samples)
    trainer = DOPSDTrainer(model, _config(device, 2), dopsd_config=cfg, adapter=adapter)
    return _audit(model, trainer, batches, advantages=lambda _: [0.0] * 2, label="dOPSD")


def t3d_smoke(checkpoint, device) -> dict:
    tp.manual_seed(41)
    model = _load(checkpoint, device)
    ref = _load(checkpoint, device, train=False)
    canvas = int(model.canvas_length)
    vocab = int(model.config.text.vocab_size)
    pad = int(model.config.pad_token_id)
    L = min(8, canvas)
    cfg = T3DConfig(
        family="block", mask_token_id=pad, canvas_token_id=pad, vocab_size=vocab,
        max_completion_length=L, teacher_steps=4, student_steps=2,
        max_train_steps=2, max_masked_positions=4, seed=41, max_prompt_len=4,
    )
    store = T3DStore()
    adapter = default_adapter(cfg)
    rollout = T3DRollout(
        T3DGenerator(model, adapter), store, cfg, ref_model=ref
    )
    records = list(rollout.generate(_request([2, 100, 200], 2, L)))
    samples = _rollout_samples(records, [2, 100, 200])
    batches, _ = make_t3d_packer(cfg, store)(samples)
    trainer = T3DTrainer(
        model, _config(device, 2), t3d_config=cfg, adapter=adapter, ref_model=ref
    )
    return _audit(model, trainer, batches, advantages=lambda _: None, label="T3D")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=default_checkpoint())
    parser.add_argument("--device", default="cuda" if tp.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if str(args.device).startswith("cuda") and not tp.cuda.is_available():
        raise SystemExit("CUDA requested but tensorplay reports no CUDA device")

    print(json.dumps({
        "checkpoint": args.checkpoint,
        "device": args.device,
        "gdsd": gdsd_smoke(args.checkpoint, args.device),
        "dopsd": dopsd_smoke(args.checkpoint, args.device),
        "t3d": t3d_smoke(args.checkpoint, args.device),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
