"""REMARL rollout — ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language
Models (Shibaev et al., arXiv:2611.22576; code https://github.com/EgorShibaev/remarl).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import tensorplay as tp
import tensorplay.nn.functional as F

from ....contracts import RolloutRequest, TrajectoryRecord
from ....dllm import default_adapter
from ....rollout import RolloutService
from .config import ReMaRLConfig
from .policy import ReMaRLPolicy, _zero_hidden, gumbel_topk, plackett_luce_logprob
from .types import ReMaRLGenerationResult, ReMaRLRecord, ReMaRLStep, ReMaRLStore


class _GenerationTensors:
    def __init__(self, x, prompt_lens: Sequence[int], span: int) -> None:
        self.dllm_noisy_ids = x
        self.dllm_prompt_lens = tp.tensor(list(prompt_lens), dtype=tp.long, device=x.device)
        self.dllm_prompt_span = int(span)
        self.dllm_canvas_len = int(x.shape[1]) - int(span)
        self.dllm_alpha = tp.ones((int(x.shape[0]),), dtype=tp.float32, device=x.device)


class ReMaRLGenerator:
    """Frozen-dLLM sampler with a learned position-selection policy."""

    def __init__(
        self,
        model: Any,
        policy: ReMaRLPolicy,
        adapter: Any | None = None,
        *,
        hidden_fn: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        self.model = model
        self.policy = policy
        self.adapter = adapter
        self.hidden_fn = hidden_fn

    @tp.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        config: ReMaRLConfig | None = None,
    ) -> ReMaRLGenerationResult:
        cfg = config or ReMaRLConfig()
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
        steps: list[list[ReMaRLStep]] = [[] for _ in range(batch)]

        for step in range(cfg.budget):
            shim = _GenerationTensors(x, prompt_lens, width)
            if cfg.family == "block":
                logits = forward(self.model, x, shim, cfg)[:, width:, :]
            else:
                logits = forward(self.model, x, shim, cfg)[:, width:, :]
            probs = F.softmax(logits.float(), dim=-1)
            greedy = probs.argmax(dim=-1)
            is_mask = x[:, width:] == noise
            proposal = tp.where(is_mask, greedy, x[:, width:])
            score = probs.gather(-1, proposal.unsqueeze(-1)).squeeze(-1)
            hidden = (
                self.hidden_fn(proposal, x[:, width:])
                if self.hidden_fn is not None
                else _zero_hidden(batch, L, cfg.hidden_dim, device)
            )
            canvas = proposal
            if step == cfg.budget - 1:
                x[:, width:] = proposal
                break
            z = self.policy(score, is_mask.to(tp.float32), hidden)
            k = cfg.keep_count(step)
            order = gumbel_topk(z, k, tau=cfg.tau)
            logp = plackett_luce_logprob(z, order)
            keep = tp.zeros((batch, L), dtype=tp.bool, device=device)
            rows = tp.arange(batch, device=device).unsqueeze(1).expand(batch, k)
            keep[rows, order] = True
            x[:, width:] = tp.where(keep, proposal, tp.full_like(proposal, noise))
            for row in range(batch):
                steps[row].append(
                    ReMaRLStep(
                        canvas_ids=canvas[row].tolist(),
                        mask=is_mask[row].to(tp.long).tolist(),
                        scores=score[row].tolist(),
                        order=order[row].tolist(),
                        old_logprob=float(logp[row].flatten()[0]),
                    )
                )

        sequences: list[list[int]] = []
        records: list[ReMaRLRecord] = []
        eos = set(cfg.eos_token_ids)
        for row, prompt in enumerate(prompts):
            tokens = [int(v) for v in x[row, width:].tolist()]
            cut = next((i for i, tok in enumerate(tokens) if tok in eos), None)
            display = tokens[: cut + 1] if cut is not None else tokens
            sequences.append(list(prompt) + display)
            records.append(
                ReMaRLRecord(
                    request_id=0,
                    prompt_ids=list(prompt),
                    completion_ids=tokens,
                    steps=tuple(steps[row]),
                    version=cfg.version,
                )
            )
        return ReMaRLGenerationResult(sequences, tuple(records))


class ReMaRLRollout(RolloutService):
    """Wire adapter for ReMaRL position-selection trajectories."""

    def __init__(
        self,
        generator: ReMaRLGenerator,
        store: ReMaRLStore | None = None,
        config: ReMaRLConfig | None = None,
        *,
        version_provider: Any | None = None,
        weight_reloader: Any | None = None,
    ) -> None:
        self.generator = generator
        self.store = store or ReMaRLStore()
        self.config = config or ReMaRLConfig()
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
            raise NotImplementedError("ReMaRLRollout.reload_weights needs a weight_reloader")
        self.weight_reloader(path, expected_digest)
