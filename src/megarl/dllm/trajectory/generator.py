"""Trajectory-level dLLM rollouts and replay (TraceRL ICLR 2026 arXiv 2509.06949;
MaskGRPO arXiv 2510.02880; CJ-GRPO arXiv 2509.23924); see package docstring for
details."""

from __future__ import annotations

import math
from typing import Any, Sequence

import tensorplay as tp
import tensorplay.nn.functional as F

from .types import (
    DLLMStep,
    DLLMTrajectoryRecord,
    TrajectoryGenerationConfig,
    TrajectoryGenerationResult,
)


class TrajectoryDiffusionGenerator:
    """Batched masked-diffusion decoder that records every reverse step.

    Implements LLaDA's confidence-based remasking (unmask the most confident
    ``ceil(L/steps)`` positions each step) and stores the *exact* sampled token
    and its log-probability at every transition. Because the recorded logits
    come from the generating model, replay on the trainer is exact rather than
    a one-step approximation.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    @tp.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        config: TrajectoryGenerationConfig | None = None,
    ) -> TrajectoryGenerationResult:
        cfg = config or TrajectoryGenerationConfig()
        if not prompts:
            raise ValueError("at least one prompt is required")
        batch = len(prompts)
        prompt_width = max(len(p) for p in prompts)
        eos = set(cfg.eos_token_ids)
        device = getattr(self.model, "device", "cpu")
        padded_prompts = [
            list(p) + [cfg.mask_token_id] * (prompt_width - len(p))
            for p in prompts
        ]
        prompt_tensor = tp.tensor(padded_prompts, dtype=tp.long, device=device)
        canvas = tp.full(
            (batch, cfg.max_new_tokens),
            cfg.mask_token_id,
            dtype=tp.long,
            device=device,
        )
        x = tp.cat([prompt_tensor, canvas], dim=1)
        per_step = max(1, math.ceil(cfg.max_new_tokens / cfg.steps))
        traces: list[list[DLLMStep]] = [[] for _ in range(batch)]

        for step in range(cfg.steps):
            logits = self.model(x)
            logits = getattr(logits, "logits", logits)
            if len(logits.shape) != 3:
                raise ValueError("dLLM generator must return [B, L, V] logits")
            log_probs = F.log_softmax(logits.float(), dim=-1)
            probs = tp.softmax(logits.float() / cfg.temperature, dim=-1)
            sampled = tp.multinomial(
                probs.view(-1, probs.shape[-1]), 1
            ).view(batch, logits.shape[1])
            sampled_logp = tp.gather(
                log_probs, -1, sampled.unsqueeze(-1)
            ).squeeze(-1)
            canvas_now = x[:, prompt_width:]
            masked = canvas_now == cfg.mask_token_id
            confidence = probs[:, prompt_width:].max(dim=-1).values
            # Top-k over masked positions only: push unmasked positions to -inf.
            ranked = tp.where(masked, confidence, tp.full_like(confidence, -1e9))
            _, indices = tp.topk(ranked, k=min(per_step, cfg.max_new_tokens), dim=-1)
            accept = tp.zeros_like(canvas_now, dtype=tp.float32)
            accept = tp.scatter(accept, -1, indices, tp.ones_like(indices, dtype=tp.float32))
            accept = accept * masked.to(accept.dtype)
            t = step / cfg.steps
            for row in range(batch):
                traces[row].append(
                    DLLMStep(
                        canvas_ids=canvas_now[row].tolist(),
                        sampled_ids=sampled[row, prompt_width:].tolist(),
                        logprobs=sampled_logp[row, prompt_width:].tolist(),
                        accept_mask=[int(v) for v in accept[row].tolist()],
                        timestep=t,
                        version=cfg.version,
                    )
                )
            x = tp.cat(
                [prompt_tensor, tp.where(accept > 0, sampled[:, prompt_width:], canvas_now)],
                dim=1,
            )

        final = x[:, prompt_width:].tolist()
        sequences: list[list[int]] = []
        trajectories: list[DLLMTrajectoryRecord] = []
        for row, prompt in enumerate(prompts):
            tokens = [int(v) for v in final[row]]
            eos_at = next((i for i, token in enumerate(tokens) if token in eos), None)
            if eos_at is not None:
                tokens = tokens[: eos_at + 1]
            sequences.append(list(prompt) + tokens)
            trajectories.append(
                DLLMTrajectoryRecord(
                    request_id=0,
                    prompt_ids=list(prompt),
                    completion_ids=tokens,
                    steps=tuple(traces[row]),
                )
            )
        return TrajectoryGenerationResult(sequences, tuple(trajectories))
