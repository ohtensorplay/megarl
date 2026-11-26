"""GDSD, Xiaohang Tang et al., arXiv 2605.29398; code https://github.com/GaryBall/GDSD;
see package docstring for details."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Iterator, Sequence

import tensorplay as tp
import tensorplay.nn.functional as F

from ....contracts import RolloutRequest, TrajectoryRecord
from ....dllm import default_adapter
from ....rollout import RolloutService
from .config import GDSDConfig
from .types import GDSDGenerationResult, GDSDRecord


class _GenerationTensors:
    """Minimal adapter input for a block-diffusion generator."""

    def __init__(self, x, prompt_lens: Sequence[int], span: int) -> None:
        self.dllm_noisy_ids = x
        self.dllm_prompt_lens = tp.tensor(
            list(prompt_lens), dtype=tp.long, device=x.device
        )
        self.dllm_prompt_span = int(span)
        self.dllm_canvas_len = int(x.shape[1]) - int(span)
        self.dllm_alpha = tp.ones(
            (int(x.shape[0]),), dtype=tp.float32, device=x.device
        )


class GDSDGenerator:
    """Confidence-ordered iterative remasking sampler.

    At each step the whole canvas is forwarded once; masked positions are
    sampled at ``temperature`` and the most confident ``L / steps`` tokens are
    committed. This is the behaviour policy whose completions become the
    self-distillation targets.
    """

    def __init__(self, model: Any, adapter: Any | None = None) -> None:
        self.model = model
        self.adapter = adapter

    @tp.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        config: GDSDConfig | None = None,
    ) -> GDSDGenerationResult:
        cfg = config or GDSDConfig()
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
        committed = tp.zeros((batch, L), dtype=tp.bool, device=device)
        per_step = max(1, math.ceil(L / cfg.sampling_steps))
        rows = tp.arange(batch, device=device).unsqueeze(1)
        positions = tp.arange(L, device=device).unsqueeze(0)

        for _ in range(cfg.sampling_steps):
            if bool(committed.all().flatten()[0]):
                break
            if cfg.family == "block":
                logits = forward(self.model, x, _GenerationTensors(x, prompt_lens, width), cfg)
                logits = logits[:, width:, :]
            else:
                logits = forward(self.model, x, _GenerationTensors(x, prompt_lens, 0), cfg)[
                    :, width:, :
                ]
            probs = F.softmax(logits.float() / cfg.temperature, dim=-1)
            sampled = tp.multinomial(
                probs.reshape(-1, probs.shape[-1]), 1
            ).view(batch, L)
            max_prob = probs.max(dim=-1).values
            score = tp.where(committed, tp.full_like(max_prob, -1e9), max_prob)
            _, take = tp.topk(score, k=min(per_step, L), dim=-1)
            chosen = sampled.gather(-1, take)
            x[rows, width + take] = chosen
            committed[rows, take] = True
        # Any position left masked after the budget is filled with its sample.
        for row in range(batch):
            for pos in range(L):
                if not bool(committed[row, pos].flatten()[0]):
                    x[row, width + pos] = noise

        sequences: list[list[int]] = []
        records: list[GDSDRecord] = []
        eos = set(cfg.eos_token_ids)
        for row, prompt in enumerate(prompts):
            tokens = [int(v) for v in x[row, width:].tolist()]
            cut = next((i for i, tok in enumerate(tokens) if tok in eos), None)
            display = tokens[: cut + 1] if cut is not None else tokens
            sequences.append(list(prompt) + display)
            records.append(
                GDSDRecord(
                    request_id=0,
                    prompt_ids=list(prompt),
                    completion_ids=tokens,
                    version=cfg.version,
                )
            )
        return GDSDGenerationResult(sequences, tuple(records))


class GDSDRollout(RolloutService):
    """Wrap :class:`GDSDGenerator` in megaRL's ``RolloutService`` contract."""

    def __init__(
        self,
        generator: Any,
        config: GDSDConfig | None = None,
        *,
        version_provider: Callable[[], int] | None = None,
        weight_reloader: Callable[[str, str | None], None] | None = None,
    ) -> None:
        self.generator = generator
        self.config = config or GDSDConfig()
        self._version_provider = version_provider or (lambda: 0)
        self._weight_reloader = weight_reloader

    def generate(self, request: RolloutRequest) -> Iterator[TrajectoryRecord]:
        prompt = list(request.prompt_token_ids)
        if not prompt:
            raise ValueError("GDSD rollout needs a non-empty prompt")
        n = max(1, request.n)
        version = int(self._version_provider())
        config = dataclasses.replace(self.config, version=version)
        result = self.generator.generate([prompt] * n, config=config)
        records = list(result.records)
        if len(records) != n:
            raise RuntimeError(f"GDSD generator returned {len(records)} records for n={n}")
        for record in records:
            record = dataclasses.replace(
                record, request_id=request.request_id, group_id=request.request_id
            )
            yield TrajectoryRecord(
                request_id=request.request_id,
                prompt_len=len(prompt),
                token_ids=list(record.completion_ids),
                logprobs=[0.0] * len(record.completion_ids),
                top_logprobs=[],
                policy_versions=[version] * len(record.completion_ids),
                reward=None,
                finish_reason=0,
                payload_handle=None,
            )

    def reload_weights(self, path: str, expected_digest: str | None = None) -> None:
        if self._weight_reloader is None:
            raise NotImplementedError("GDSD weight reload needs weight_reloader")
        self._weight_reloader(path, expected_digest)
