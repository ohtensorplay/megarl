"""T3D, Tunyu Zhang et al., NeurIPS 2026 arXiv 2602.12262; code
https://github.com/Tyrion58/T3D; see package docstring for details."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Iterator, Sequence

import tensorplay as tp
import tensorplay.nn.functional as F

from ....contracts import RolloutRequest, TrajectoryRecord
from ....dllm import default_adapter
from ....rollout import RolloutService
from .config import T3DConfig
from .types import T3DGenerationResult, T3DRecord, T3DStep, T3DStore


class _GenerationTensors:
    def __init__(self, x, prompt_lens: Sequence[int], span: int) -> None:
        self.dllm_noisy_ids = x
        self.dllm_prompt_lens = tp.tensor(list(prompt_lens), dtype=tp.long, device=x.device)
        self.dllm_prompt_span = int(span)
        self.dllm_canvas_len = int(x.shape[1]) - int(span)
        self.dllm_alpha = tp.ones((int(x.shape[0]),), dtype=tp.float32, device=x.device)


class T3DGenerator:
    """Full-step denoiser that records intermediate states and decode steps."""

    def __init__(self, model: Any, adapter: Any | None = None) -> None:
        self.model = model
        self.adapter = adapter

    @tp.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        config: T3DConfig | None = None,
    ) -> T3DGenerationResult:
        cfg = config or T3DConfig()
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
        decode_steps = [[-1] * L for _ in range(batch)]
        trajectories: list[list[T3DStep]] = [[] for _ in range(batch)]
        per_step = max(1, math.ceil(L / cfg.teacher_steps))
        rows = tp.arange(batch, device=device).unsqueeze(1)

        for step in range(cfg.teacher_steps):
            for row in range(batch):
                trajectories[row].append(
                    T3DStep(
                        [int(v) for v in x[row, width:].tolist()],
                        [0 if bool(committed[row, p].flatten()[0]) else 1 for p in range(L)],
                    )
                )
            if bool(committed.all().flatten()[0]):
                break
            if cfg.family == "block":
                logits = forward(self.model, x, _GenerationTensors(x, prompt_lens, width), cfg)[
                    :, width:, :
                ]
            else:
                logits = forward(self.model, x, _GenerationTensors(x, prompt_lens, 0), cfg)[
                    :, width:, :
                ]
            probs = F.softmax(logits.float() / cfg.temperature, dim=-1)
            sampled = tp.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(batch, L)
            confidence = probs.max(dim=-1).values
            score = tp.where(committed, tp.full_like(confidence, -1e9), confidence)
            _, take = tp.topk(score, k=min(per_step, L), dim=-1)
            x[rows, width + take] = sampled.gather(-1, take)
            committed[rows, take] = True
            for row in range(batch):
                for col in take[row].tolist():
                    if decode_steps[row][int(col)] < 0:
                        decode_steps[row][int(col)] = step

        for row in range(batch):
            for pos in range(L):
                if decode_steps[row][pos] < 0:
                    decode_steps[row][pos] = cfg.teacher_steps - 1
            trajectories[row].append(
                T3DStep(
                    [int(v) for v in x[row, width:].tolist()],
                    [0] * L,
                )
            )

        sequences: list[list[int]] = []
        records: list[T3DRecord] = []
        eos = set(cfg.eos_token_ids)
        for row, prompt in enumerate(prompts):
            tokens = [int(v) for v in x[row, width:].tolist()]
            cut = next((i for i, tok in enumerate(tokens) if tok in eos), None)
            display = tokens[: cut + 1] if cut is not None else tokens
            sequences.append(list(prompt) + display)
            records.append(
                T3DRecord(
                    request_id=0,
                    prompt_ids=list(prompt),
                    completion_ids=tokens,
                    steps=tuple(trajectories[row]),
                    decode_steps=list(decode_steps[row]),
                    ref_completion_ids=list(tokens),
                    version=cfg.version,
                )
            )
        return T3DGenerationResult(sequences, tuple(records))


class T3DRollout(RolloutService):
    """Wrap :class:`T3DGenerator` and attach reference ("fake") completions."""

    def __init__(
        self,
        generator: Any,
        store: T3DStore | None = None,
        config: T3DConfig | None = None,
        *,
        ref_model: Any | None = None,
        ref_generator: Any | None = None,
        version_provider: Callable[[], int] | None = None,
        weight_reloader: Callable[[str, str | None], None] | None = None,
    ) -> None:
        self.generator = generator
        self.store = store or T3DStore()
        self.config = config or T3DConfig()
        self.ref_model = ref_model
        self.ref_generator = ref_generator or (
            T3DGenerator(ref_model) if ref_model is not None else None
        )
        self._version_provider = version_provider or (lambda: 0)
        self._weight_reloader = weight_reloader

    def generate(self, request: RolloutRequest) -> Iterator[TrajectoryRecord]:
        prompt = list(request.prompt_token_ids)
        if not prompt:
            raise ValueError("T3D rollout needs a non-empty prompt")
        n = max(1, request.n)
        version = int(self._version_provider())
        config = dataclasses.replace(self.config, version=version)
        result = self.generator.generate([prompt] * n, config=config)
        records = list(result.records)
        if len(records) != n:
            raise RuntimeError(f"T3D generator returned {len(records)} records for n={n}")
        ref_records: tuple[T3DRecord, ...] | None = None
        if self.ref_generator is not None:
            ref_result = self.ref_generator.generate([prompt] * n, config=config)
            ref_records = ref_result.records
        for index, record in enumerate(records):
            fake = (
                list(ref_records[index].completion_ids)
                if ref_records is not None
                else list(record.completion_ids)
            )
            record = dataclasses.replace(
                record,
                request_id=request.request_id,
                group_id=request.request_id,
                ref_completion_ids=fake,
            )
            handle = self.store.put(record)
            yield TrajectoryRecord(
                request_id=request.request_id,
                prompt_len=len(prompt),
                token_ids=list(record.completion_ids),
                logprobs=[0.0] * len(record.completion_ids),
                top_logprobs=[],
                policy_versions=[version] * len(record.completion_ids),
                reward=None,
                finish_reason=0,
                payload_handle=handle,
            )

    def reload_weights(self, path: str, expected_digest: str | None = None) -> None:
        if self._weight_reloader is None:
            raise NotImplementedError("T3D weight reload needs weight_reloader")
        self._weight_reloader(path, expected_digest)
