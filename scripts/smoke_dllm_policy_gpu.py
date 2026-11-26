#!/usr/bin/env python3
"""GPU smoke: real-checkpoint updates for TIPS, ReMaRL and PURL.

These three algorithms train decoding policies around a DiffusionGemma dLLM.
TIPS and ReMaRL keep the backbone frozen and train a small auxiliary policy;
PURL trains the predictor jointly with its edit/unmask planners. The script
loads the tiny DiffusionGemma checkpoint through MegaLM's native adapter,
generates trajectories, packs them, then performs a real forward/backward and
optimizer step, auditing gradients and the resulting weight change.

Example
-------
    python scripts/smoke_dllm_policy_gpu.py --device cuda

References
----------
TIPS  — Zhang et al., arXiv:2610.22099
ReMaRL — Shibaev et al., arXiv:2611.22576
PURL  — Zheng et al., arXiv:2611.11551v1
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
from megarl.dllm import from_dllm_record  # noqa: E402
from megarl.dllm.algorithms.purl import (  # noqa: E402
    PURLConfig, PURLGenerator, PURLModel, PURLRollout, PURLStore,
    PURLTrainer, make_purl_packer,
)
from megarl.dllm.algorithms.remarl import (  # noqa: E402
    ReMaRLConfig, ReMaRLGenerator, ReMaRLPolicy, ReMaRLRollout, ReMaRLStore,
    ReMaRLTrainer, make_remarl_packer,
)
from megarl.dllm.algorithms.tips import (  # noqa: E402
    TIPSConfig, TIPSGenerator, TIPSPolicy, TIPSRollout, TIPSStore,
    TIPSTrainer, make_tips_packer,
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


def _load(checkpoint: str, device: str, *, train: bool):
    model, _ = DiffusionGemmaForBlockDiffusion.from_pretrained(
        checkpoint, device=device, dtype=tp.float32,
        kernel_policy=KernelPolicy(use_triton=False),
    )
    model.requires_grad_(train)
    model.train() if train else model.eval()
    return model


def _trainer_config(device: str, accum: int) -> TrainerConfig:
    return TrainerConfig(
        device=device, learning_rate=1e-4, max_grad_norm=None,
        gradient_accumulation_steps=accum,
    )


def _request(prompt, n: int, max_new_tokens: int) -> RolloutRequest:
    return RolloutRequest(
        1, 0, prompt, max_new_tokens, n,
        SamplingParams(1.0, 1.0, 0, None, False, 0), None,
    )


def _samples(records, prompt, *, rewards=(1.0, 0.0)):
    samples = []
    for i, record in enumerate(records):
        completion = getattr(record, "completion_ids", None)
        if completion is None:
            completion = record.token_ids
        wire = TrajectoryRecord(
            request_id=1, prompt_len=len(prompt), token_ids=list(completion),
            logprobs=[0.0] * len(completion), top_logprobs=[],
            policy_versions=[0] * len(completion), reward=None, finish_reason=0,
            payload_handle=record.payload_handle,
        )
        samples.append(from_dllm_record(wire, prompt, reward=rewards[i % len(rewards)]))
    return samples


def _audit(module, trainer, batches, *, advantages, label: str) -> dict:
    results = []
    for batch in batches:
        audit = trainer.update(batch, advantages(batch), dry_run=True)
        grads = [
            (p, float(p.grad.abs().sum().flatten()[0]))
            for p in module.parameters()
            if p.grad is not None
        ]
        loss = float(audit["train/loss"])
        if not math.isfinite(loss) or not any(g > 0 for _, g in grads):
            raise SystemExit(
                f"{label} failed forward/backward: loss={loss}, "
                f"nonzero_grads={sum(g > 0 for _, g in grads)}"
            )
        # ReMaRL starts as a zero-init residual, so only the final layer and
        # skip have gradients. Probe the parameter that actually receives one.
        probe, _ = max(grads, key=lambda item: item[1])
        before = probe.detach().clone()
        metrics = trainer.update(batch, advantages(batch), dry_run=False)
        after = probe.detach()
        if bool(tp.allclose(before, after)):
            raise SystemExit(f"{label} failed: optimizer step did not change weights")
        results.append({
            "loss": loss,
            "params_with_grad": len(grads),
            "grad_norm_sum": sum(g for _, g in grads),
            "step": int(metrics["train/step"]),
        })
    return {"batches": results, "status": "ok"}


def _checkpoint_meta(model):
    return (
        int(model.config.pad_token_id),
        int(model.config.text.vocab_size),
        min(8, int(model.canvas_length)),
    )


def tips_smoke(checkpoint, device) -> dict:
    tp.manual_seed(31)
    model = _load(checkpoint, device, train=False)
    pad, vocab, length = _checkpoint_meta(model)
    prompt = [2, 100, 200]
    cfg = TIPSConfig(
        family="block", mask_token_id=pad, canvas_token_id=pad, vocab_size=vocab,
        max_completion_length=length, budget=4, gamma=0.3, hidden=16, heads=2,
        time_dim=8, layers=2, seed=31, max_prompt_len=4,
    )
    policy = TIPSPolicy(
        hidden=cfg.hidden, heads=cfg.heads, time_dim=cfg.time_dim,
        layers=cfg.layers, max_len=cfg.max_completion_length + 8,
    ).to(device)
    store = TIPSStore()
    rollout = TIPSRollout(TIPSGenerator(model, policy), store, cfg)
    records = list(rollout.generate(_request(prompt, 2, length)))
    samples = _samples(records, prompt)
    batches, _ = make_tips_packer(cfg, store)(samples)
    trainer = TIPSTrainer(policy, _trainer_config(device, 4), tips_config=cfg)
    return _audit(
        policy, trainer, batches,
        advantages=lambda b: [0.5 if s == 0 else -0.5 for s in b.tips_sample_ids],
        label="TIPS",
    )


def remarl_smoke(checkpoint, device) -> dict:
    tp.manual_seed(37)
    model = _load(checkpoint, device, train=False)
    pad, vocab, length = _checkpoint_meta(model)
    prompt = [2, 100, 200]
    cfg = ReMaRLConfig(
        family="block", mask_token_id=pad, canvas_token_id=pad, vocab_size=vocab,
        max_completion_length=length, budget=4, tau=1.0, hidden_dim=16,
        pos_dim=8, proj_dim=16, mlp_hidden=16, seed=37, max_prompt_len=4,
    )
    policy = ReMaRLPolicy(cfg, max_len=length + 8).to(device)
    store = ReMaRLStore()
    rollout = ReMaRLRollout(ReMaRLGenerator(model, policy), store, cfg)
    records = list(rollout.generate(_request(prompt, 2, length)))
    samples = _samples(records, prompt)
    batches, _ = make_remarl_packer(cfg, store)(samples)
    trainer = ReMaRLTrainer(policy, _trainer_config(device, 2), remarl_config=cfg)
    return _audit(
        policy, trainer, batches,
        advantages=lambda b: [0.5 if s == 0 else -0.5 for s in b.remarl_sample_ids],
        label="ReMaRL",
    )


def purl_smoke(checkpoint, device) -> dict:
    tp.manual_seed(41)
    model = _load(checkpoint, device, train=True)
    pad, vocab, length = _checkpoint_meta(model)
    prompt = [2, 100, 200]
    cfg = PURLConfig(
        family="block", mask_token_id=pad, canvas_token_id=pad, vocab_size=vocab,
        max_completion_length=length, unmask_budget=1, edit_budget=1,
        feature_dim=16, seed=41, max_prompt_len=4,
    )

    def features(_backbone, _ids, logits):
        # A deterministic low-rank slice keeps the planner tiny even when the
        # checkpoint's vocabulary is large.
        return logits[..., : cfg.feature_dim]

    wrapped = PURLModel(model, cfg, feature_fn=features).to(device)
    store = PURLStore()
    rollout = PURLRollout(PURLGenerator(wrapped, cfg), store, cfg)
    records = list(rollout.generate(_request(prompt, 2, length)))
    samples = _samples(records, prompt)
    batches, _ = make_purl_packer(cfg, store)(samples)
    trainer = PURLTrainer(wrapped, _trainer_config(device, 2), purl_config=cfg)
    return _audit(
        wrapped, trainer, batches, advantages=lambda _: [0.0, 0.0], label="PURL"
    )


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
        "tips": tips_smoke(args.checkpoint, args.device),
        "remarl": remarl_smoke(args.checkpoint, args.device),
        "purl": purl_smoke(args.checkpoint, args.device),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
