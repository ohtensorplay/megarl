"""PURL rollout — When Should Diffusion Language Models Reconsider Their Commitments?
(Zheng et al., arXiv:2611.11551v1; withdrawn after v1).
"""

from __future__ import annotations

from typing import Any, Sequence

import tensorplay as tp
import tensorplay.nn.functional as F

from ....contracts import RolloutRequest, TrajectoryRecord
from ....rollout import RolloutService
from .config import PURLConfig
from .model import PURLModel
from .types import PURLGenerationResult, PURLRecord, PURLStore


class _GenerationTensors:
    def __init__(self, x, prompt_lens: Sequence[int], span: int) -> None:
        self.dllm_noisy_ids = x
        self.dllm_prompt_lens = tp.tensor(list(prompt_lens), dtype=tp.long, device=x.device)
        self.dllm_prompt_span = int(span)
        self.dllm_canvas_len = int(x.shape[1]) - int(span)
        self.dllm_alpha = tp.ones((int(x.shape[0]),), dtype=tp.float32, device=x.device)


class PURLGenerator:
    """Revisable finite-step decoder using one shared forward per step."""

    def __init__(self, model: PURLModel, config: PURLConfig | None = None) -> None:
        self.model = model
        self.config = config or model.config

    @tp.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        config: PURLConfig | None = None,
    ) -> PURLGenerationResult:
        cfg = config or self.config
        if not prompts:
            raise ValueError("at least one prompt is required")
        batch = len(prompts)
        width = max(cfg.max_prompt_len or 0, max(len(p) for p in prompts))
        L = cfg.max_completion_length
        device = getattr(self.model.backbone, "device", "cpu")
        fill = cfg.prompt_fill()
        padded = [list(p) + [fill] * (width - len(p)) for p in prompts]
        prompt_tensor = tp.tensor(padded, dtype=tp.long, device=device)
        noise = cfg.noise_token_id
        x = tp.cat(
            [prompt_tensor, tp.full((batch, L), noise, dtype=tp.long, device=device)],
            dim=1,
        )
        prompt_lens = [len(p) for p in prompts]
        guard = 0
        while True:
            shim = _GenerationTensors(x, prompt_lens, width)
            logits, edit_logits, unmask_logits = self.model(x, shim, cfg)
            canvas = x[:, width:]
            canvas_logits = logits[:, width:, :]
            is_mask = canvas == noise
            if not bool(is_mask.any().flatten()[0]) or guard >= 4 * L:
                break
            allowed_unmask = tp.where(
                is_mask,
                unmask_logits[:, width:],
                tp.full_like(unmask_logits[:, width:], -1e9),
            )
            u_count = min(cfg.unmask_budget, int(is_mask.to(tp.long).sum(dim=-1).max().flatten()[0]))
            _, unmask = tp.topk(allowed_unmask, k=u_count, dim=-1)
            revise = (
                float(1.0 - (is_mask.to(tp.float32).mean(dim=-1)).flatten()[0])
                >= cfg.revision_gate
                and cfg.edit_budget > 0
            )
            selected: list = [unmask]
            if revise:
                visible = ~is_mask
                allowed_edit = tp.where(
                    visible,
                    edit_logits[:, width:],
                    tp.full_like(edit_logits[:, width:], -1e9),
                )
                e_count = min(cfg.edit_budget, int(visible.to(tp.long).sum(dim=-1).max().flatten()[0]))
                if e_count > 0:
                    _, edit = tp.topk(allowed_edit, k=e_count, dim=-1)
                    selected.append(edit)
            rows = tp.arange(batch, device=device).unsqueeze(-1)
            for positions in selected:
                probs = F.softmax(canvas_logits.float() / max(cfg.w_mask, 1e-6), dim=-1)
                sampled = tp.multinomial(
                    probs.reshape(-1, probs.shape[-1]), 1
                ).view(batch, L).gather(-1, positions)
                x[:, width:] = canvas.scatter(-1, positions, sampled)
                canvas = x[:, width:]
            guard += 1

        sequences: list[list[int]] = []
        records: list[PURLRecord] = []
        eos = set(cfg.eos_token_ids)
        for row, prompt in enumerate(prompts):
            tokens = [int(v) for v in x[row, width:].tolist()]
            cut = next((i for i, tok in enumerate(tokens) if tok in eos), None)
            display = tokens[: cut + 1] if cut is not None else tokens
            reference = list(tokens)
            sequences.append(list(prompt) + display)
            records.append(
                PURLRecord(
                    request_id=0,
                    prompt_ids=list(prompt),
                    completion_ids=tokens,
                    reference_ids=reference,
                    version=cfg.version,
                )
            )
        return PURLGenerationResult(sequences, tuple(records))


class PURLRollout(RolloutService):
    """Wire adapter for PURL revisable decoding."""

    def __init__(
        self,
        generator: PURLGenerator,
        store: PURLStore | None = None,
        config: PURLConfig | None = None,
        *,
        version_provider: Any | None = None,
        weight_reloader: Any | None = None,
    ) -> None:
        self.generator = generator
        self.store = store or PURLStore()
        self.config = config or PURLConfig()
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
            raise NotImplementedError("PURLRollout.reload_weights needs a weight_reloader")
        self.weight_reloader(path, expected_digest)
