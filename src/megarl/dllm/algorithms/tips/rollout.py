"""TIPS rollout — TIPS: A Tiny Inference-time Policy for Scaling in Diffusion Language
Models (Zhang et al., arXiv:2610.22099; code https://github.com/zhengyz97/TIPS).
"""

from __future__ import annotations

from typing import Any, Sequence

import tensorplay as tp
import tensorplay.nn.functional as F

from ....contracts import RolloutRequest, TrajectoryRecord
from ....dllm import default_adapter
from ....rollout import RolloutService
from .config import TIPSConfig
from .policy import (
    TIPSPolicy,
    calibrate_tips_shift,
    tips_bernoulli_logprob,
    tips_visibility_features,
)
from .types import TIPSGenerationResult, TIPSRecord, TIPSStep, TIPSStore


class _GenerationTensors:
    """Minimal adapter input for a block-diffusion generator."""

    def __init__(self, x, prompt_lens: Sequence[int], span: int) -> None:
        self.dllm_noisy_ids = x
        self.dllm_prompt_lens = tp.tensor(list(prompt_lens), dtype=tp.long, device=x.device)
        self.dllm_prompt_span = int(span)
        self.dllm_canvas_len = int(x.shape[1]) - int(span)
        self.dllm_alpha = tp.ones((int(x.shape[0]),), dtype=tp.float32, device=x.device)


class TIPSGenerator:
    """Fixed-horizon visibility sampler around a frozen dLLM."""

    def __init__(self, model: Any, policy: TIPSPolicy, adapter: Any | None = None) -> None:
        self.model = model
        self.policy = policy
        self.adapter = adapter

    @tp.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        config: TIPSConfig | None = None,
    ) -> TIPSGenerationResult:
        cfg = config or TIPSConfig()
        if not prompts:
            raise ValueError("at least one prompt is required")
        batch = len(prompts)
        width = max(cfg.max_prompt_len or 0, max(len(p) for p in prompts))
        L = cfg.max_completion_length
        device = getattr(self.model, "device", "cpu")
        fill = cfg.prompt_fill()
        padded = [list(p) + [fill] * (width - len(p)) for p in prompts]
        prompt_tensor = tp.tensor(padded, dtype=tp.long, device=device)
        noise = cfg.noise_token_id
        x = tp.cat(
            [prompt_tensor, tp.full((batch, L), noise, dtype=tp.long, device=device)],
            dim=1,
        )
        prompt_lens = [len(p) for p in prompts]
        forward = self.adapter or default_adapter(cfg)
        steps: list[list[TIPSStep]] = [[] for _ in range(batch)]

        for step in range(cfg.budget):
            if cfg.family == "block":
                logits = forward(
                    self.model, x, _GenerationTensors(x, prompt_lens, width), cfg
                )[:, width:, :]
            else:
                logits = forward(
                    self.model, x, _GenerationTensors(x, prompt_lens, 0), cfg
                )[:, width:, :]
            features, is_mask, greedy = tips_visibility_features(
                logits, x[:, width:], noise_token_id=noise
            )
            tau = step / max(1, cfg.budget)
            time = tp.full((batch,), tau, dtype=tp.float32, device=device)
            if step == cfg.budget - 1:
                # Deterministic finalisation: fill every remaining mask.
                x[:, width:] = tp.where(is_mask.to(tp.bool), greedy, x[:, width:])
                break
            base = self.policy(features, is_mask.to(tp.float32), time)
            kappa = cfg.schedule((step + 1) / cfg.budget)
            delta = calibrate_tips_shift(base, kappa, iterations=cfg.calibration_iters)
            rho = F.sigmoid(base + delta.unsqueeze(-1)).clamp(1e-6, 1.0 - 1e-6)
            actions = (tp.rand((batch, L), device=device) < rho).to(tp.long)
            logp = tips_bernoulli_logprob(rho, actions)
            canvas = x[:, width:]
            next_canvas = tp.where(
                actions.to(tp.bool),
                tp.where(is_mask.to(tp.bool), greedy, canvas),
                tp.full_like(canvas, noise),
            )
            x[:, width:] = next_canvas
            for row in range(batch):
                steps[row].append(
                    TIPSStep(
                        canvas_ids=canvas[row].tolist(),
                        mask=is_mask[row].tolist(),
                        features=features[row].tolist(),
                        time=float(tau),
                        rho=rho[row].tolist(),
                        actions=actions[row].tolist(),
                        old_logprob=float(logp[row].flatten()[0]),
                    )
                )

        sequences: list[list[int]] = []
        records: list[TIPSRecord] = []
        eos = set(cfg.eos_token_ids)
        for row, prompt in enumerate(prompts):
            tokens = [int(v) for v in x[row, width:].tolist()]
            cut = next((i for i, tok in enumerate(tokens) if tok in eos), None)
            display = tokens[: cut + 1] if cut is not None else tokens
            sequences.append(list(prompt) + display)
            records.append(
                TIPSRecord(
                    request_id=0,
                    prompt_ids=list(prompt),
                    completion_ids=tokens,
                    steps=tuple(steps[row]),
                    version=cfg.version,
                )
            )
        return TIPSGenerationResult(sequences, tuple(records))


class TIPSRollout(RolloutService):
    """Wire adapter: TIPS trajectories travel out of band via :class:`TIPSStore`."""

    def __init__(
        self,
        generator: TIPSGenerator,
        store: TIPSStore | None = None,
        config: TIPSConfig | None = None,
        *,
        version_provider: Any | None = None,
        weight_reloader: Any | None = None,
    ) -> None:
        self.generator = generator
        self.store = store or TIPSStore()
        self.config = config or TIPSConfig()
        self.version_provider = version_provider
        self.weight_reloader = weight_reloader

    def generate(self, request: RolloutRequest):
        prompts = [list(request.prompt_token_ids) for _ in range(request.n)]
        result = self.generator.generate(prompts, config=self.config)
        for record in result.records:
            handle = self.store.put(record)
            length = len(record.completion_ids)
            yield TrajectoryRecord(
                request_id=request.request_id,
                prompt_len=len(record.prompt_ids),
                token_ids=list(record.completion_ids),
                logprobs=[0.0] * length,
                top_logprobs=[],
                policy_versions=[record.version] * length,
                reward=None,
                finish_reason=0,
                payload_handle=handle,
            )

    def reload_weights(self, path: str, expected_digest: str | None = None) -> None:
        if self.weight_reloader is None:
            raise NotImplementedError("TIPSRollout.reload_weights needs a weight_reloader")
        self.weight_reloader(path, expected_digest)
