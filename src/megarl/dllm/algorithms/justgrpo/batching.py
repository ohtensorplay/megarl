"""JustGRPO / The Flexibility Trap, Zanlin Ni et al., ICML 2026 arXiv 2601.15165; code
https://github.com/LeapLabTHU/JustGRPO; see package docstring for details."""

from __future__ import annotations

import dataclasses
import random
from typing import Sequence

import tensorplay as tp

from ....batching import RLBatch
from ....dllm import DLLMSample
from ....losses import TensorBatch
from .config import JustGRPOConfig
from .types import JustGRPORecord, JustGRPOStore


@dataclasses.dataclass(frozen=True, slots=True)
class JustGRPOBatch(RLBatch):
    """One row per ``(sample, selected position)`` pair."""

    dllm_noisy_ids: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    dllm_prompt_lens: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    dllm_prompt_span: int = dataclasses.field(default=0, kw_only=True)
    dllm_canvas_len: int = dataclasses.field(default=0, kw_only=True)
    justgrpo_sample_ids: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    justgrpo_step_ids: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    justgrpo_group_ids: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    justgrpo_loss_scale: float = dataclasses.field(default=1.0, kw_only=True)


class JustGRPOTensorBatch(TensorBatch):
    """Device tensors for a :class:`JustGRPOBatch`."""

    def __init__(self, batch: JustGRPOBatch, device: str = "cpu") -> None:
        super().__init__(batch, device=device)
        self.dllm_noisy_ids = tp.tensor(
            batch.dllm_noisy_ids, dtype=tp.long, device=device
        )
        self.dllm_prompt_lens = tp.tensor(
            batch.dllm_prompt_lens, dtype=tp.long, device=device
        )
        self.dllm_prompt_span = int(batch.dllm_prompt_span)
        self.dllm_canvas_len = int(batch.dllm_canvas_len)


def collate_justgrpo(
    samples: Sequence[DLLMSample],
    records: Sequence[JustGRPORecord],
    config: JustGRPOConfig,
) -> JustGRPOBatch:
    """Expand samples into AR canvases, selecting top-entropy positions."""
    if not samples:
        raise ValueError("cannot collate an empty JustGRPO batch")
    if len(samples) != len(records):
        raise ValueError("JustGRPO samples/records mismatch")
    for record in records:
        record.validate()

    L = config.max_new_tokens
    for record in records:
        if record.length != L:
            raise ValueError(
                f"JustGRPO record length {record.length} != configured {L}"
            )
    max_prompt = max(len(s.prompt_token_ids) for s in samples)
    prompt_span = max(max_prompt, config.max_prompt_len or 0)
    width = prompt_span + L
    fill = config.prompt_fill()
    noise = config.noise_token_id
    k = config.selected_positions(L)
    scale = k / L
    rng = random.Random(config.seed)

    input_ids: list[list[int]] = []
    noisy_ids: list[list[int]] = []
    attention: list[list[int]] = []
    loss_mask: list[list[int]] = []
    position_ids: list[list[int]] = []
    old_logprobs: list[list[float]] = []
    versions: list[list[int]] = []
    sample_ids: list[int] = []
    step_ids: list[int] = []
    group_ids: list[int] = []
    cu_seqlens = [0]

    for index, (sample, record) in enumerate(zip(samples, records)):
        prompt = list(sample.prompt_token_ids)
        completion = list(sample.completion_token_ids)
        padded_prompt = prompt + [fill] * (prompt_span - len(prompt))
        # Deterministic tie-break keeps fast-mode selection reproducible.
        ranked = sorted(
            range(L),
            key=lambda t: (-record.entropies[t], t, rng.random()),
        )[:k]
        selected = sorted(ranked)
        for t in selected:
            canvas = padded_prompt + completion[:t] + [noise] * (L - t)
            target_pos = prompt_span + t
            target_row = [0] * width
            target_row[target_pos] = completion[t]
            mask_row = [0] * width
            mask_row[target_pos] = 1
            old_row = [0.0] * width
            old_row[target_pos] = record.token_logprobs[t]
            input_ids.append(target_row)
            noisy_ids.append(canvas)
            attention.append(
                [1] * len(prompt)
                + [0] * (prompt_span - len(prompt))
                + [1] * L
            )
            loss_mask.append(mask_row)
            position_ids.append(list(range(width)))
            old_logprobs.append(old_row)
            versions.append([record.version] * width)
            sample_ids.append(index)
            step_ids.append(t)
            group_ids.append(record.group_id)
            cu_seqlens.append(cu_seqlens[-1] + width)

    return JustGRPOBatch(
        input_ids=input_ids,
        attention_mask=attention,
        loss_mask=loss_mask,
        position_ids=position_ids,
        old_logprobs=old_logprobs,
        versions=versions,
        rewards=[float(s.reward) for s in samples for _ in selected],
        cu_seqlens=cu_seqlens,
        max_len=width,
        sampling_logprobs=[],
        dllm_noisy_ids=noisy_ids,
        dllm_prompt_lens=[len(s.prompt_token_ids) for s in samples for _ in selected],
        dllm_prompt_span=prompt_span,
        dllm_canvas_len=L,
        justgrpo_sample_ids=sample_ids,
        justgrpo_step_ids=step_ids,
        justgrpo_group_ids=group_ids,
        justgrpo_loss_scale=scale,
    )


def slice_justgrpo_batch(batch: JustGRPOBatch, start: int, stop: int) -> JustGRPOBatch:
    return JustGRPOBatch(
        input_ids=batch.input_ids[start:stop],
        attention_mask=batch.attention_mask[start:stop],
        loss_mask=batch.loss_mask[start:stop],
        position_ids=batch.position_ids[start:stop],
        old_logprobs=batch.old_logprobs[start:stop],
        versions=batch.versions[start:stop],
        rewards=batch.rewards[start:stop],
        cu_seqlens=[0]
        + [n - batch.cu_seqlens[start] for n in batch.cu_seqlens[start + 1 : stop + 1]],
        max_len=batch.max_len,
        sampling_logprobs=batch.sampling_logprobs[start:stop],
        dllm_noisy_ids=batch.dllm_noisy_ids[start:stop],
        dllm_prompt_lens=batch.dllm_prompt_lens[start:stop],
        dllm_prompt_span=batch.dllm_prompt_span,
        dllm_canvas_len=batch.dllm_canvas_len,
        justgrpo_sample_ids=batch.justgrpo_sample_ids[start:stop],
        justgrpo_step_ids=batch.justgrpo_step_ids[start:stop],
        justgrpo_group_ids=batch.justgrpo_group_ids[start:stop],
        justgrpo_loss_scale=batch.justgrpo_loss_scale,
    )


def make_justgrpo_packer(config: JustGRPOConfig, store: JustGRPOStore):
    """Resolve handles, select positions and align advantages with rows."""

    def pack(samples, *, max_tokens=None, alignment=1):
        records = []
        for sample in samples:
            handle = getattr(sample, "payload_handle", None)
            if handle is None:
                raise ValueError("JustGRPO packer needs payload handles on samples")
            record = store.get(handle)
            store.release(handle)
            records.append(dataclasses.replace(record, reward=float(sample.reward)))
        batch = collate_justgrpo(samples, records, config)
        return [batch], [list(batch.justgrpo_sample_ids)]

    return pack
