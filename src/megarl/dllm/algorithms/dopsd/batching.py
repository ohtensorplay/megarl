"""dOPSD, Phuong Tuan Dat, Qi Li, Xinchao Wang, arXiv 2607.04428; code
https://github.com/tuandattt/dOPSD; see package docstring for details."""

from __future__ import annotations

import dataclasses
import random
from typing import Sequence

import tensorplay as tp

from ....batching import RLBatch
from ....dllm import DLLMSample
from ....losses import TensorBatch
from .config import DOPSDConfig
from .types import DOPSDRecord, DOPSDStore


@dataclasses.dataclass(frozen=True, slots=True)
class DOPSDBatch(RLBatch):
    """One row per verified rollout; teacher states are padded into a tensor."""

    dopsd_student_canvas: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    dopsd_teacher_canvas: list[list[list[int]]] = dataclasses.field(default_factory=list, kw_only=True)
    dopsd_teacher_present: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    dopsd_teacher_mask: list[list[list[int]]] = dataclasses.field(default_factory=list, kw_only=True)
    """1 only where a position is still masked at that future state."""
    dopsd_prompt_lens: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    dopsd_prompt_span: int = dataclasses.field(default=0, kw_only=True)
    dopsd_mask_positions: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    dopsd_position_valid: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    dopsd_counts: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    dopsd_keep: list[float] = dataclasses.field(default_factory=list, kw_only=True)


class DOPSDTensorBatch(TensorBatch):
    """Device tensors for one :class:`DOPSDBatch` slice."""

    def __init__(self, batch: DOPSDBatch, device: str = "cpu") -> None:
        super().__init__(batch, device=device)
        self.dopsd_student_canvas = tp.tensor(
            batch.dopsd_student_canvas, dtype=tp.long, device=device
        )
        self.dopsd_teacher_canvas = tp.tensor(
            batch.dopsd_teacher_canvas, dtype=tp.long, device=device
        )
        self.dopsd_teacher_present = tp.tensor(
            batch.dopsd_teacher_present, dtype=tp.float32, device=device
        )
        self.dopsd_teacher_mask = tp.tensor(
            batch.dopsd_teacher_mask, dtype=tp.float32, device=device
        )
        self.dopsd_prompt_lens = tp.tensor(
            batch.dopsd_prompt_lens, dtype=tp.long, device=device
        )
        self.dopsd_prompt_span = int(batch.dopsd_prompt_span)
        # The shared block-diffusion adapter reads the canonical dllm_* names.
        self.dllm_prompt_span = self.dopsd_prompt_span
        self.dllm_prompt_lens = self.dopsd_prompt_lens
        self.dllm_noisy_ids = self.dopsd_student_canvas
        self.dopsd_mask_positions = tp.tensor(
            batch.dopsd_mask_positions, dtype=tp.long, device=device
        )
        self.dopsd_position_valid = tp.tensor(
            batch.dopsd_position_valid, dtype=tp.float32, device=device
        )
        self.dopsd_counts = tp.tensor(batch.dopsd_counts, dtype=tp.float32, device=device)
        self.dopsd_keep = tp.tensor(batch.dopsd_keep, dtype=tp.float32, device=device)


def _full_canvas(prompt: list[int], span: int, canvas: list[int], config: DOPSDConfig):
    """Build a model input from a prompt and a completion-span canvas."""
    if config.family == "block":
        return prompt + [config.noise_token_id] * (span - len(prompt)) + canvas
    return prompt + canvas


def collate_dopsd(
    samples: Sequence[DLLMSample],
    records: Sequence[DOPSDRecord],
    config: DOPSDConfig,
) -> DOPSDBatch:
    """Pick one genuine student step per rollout and build teacher targets."""
    if not samples or len(samples) != len(records):
        raise ValueError("dOPSD collate needs matching samples and records")
    for record in records:
        record.validate()
    L = config.max_completion_length
    for record in records:
        if record.length != L:
            raise ValueError("dOPSD completion length must match the config")
    prompt_span = (
        max(len(s.prompt_token_ids) for s in samples)
        if config.family == "block"
        else 0
    )
    width = (prompt_span if config.family == "block" else max(len(s.prompt_token_ids) for s in samples)) + L
    offset = prompt_span if config.family == "block" else None
    rng = random.Random(config.seed)

    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    loss_mask: list[list[int]] = []
    positions: list[list[int]] = []
    old_logprobs: list[list[float]] = []
    versions: list[list[int]] = []
    student_canvas: list[list[int]] = []
    teacher_canvas: list[list[list[int]]] = []
    teacher_present: list[list[int]] = []
    teacher_position_mask: list[list[list[int]]] = []
    mask_positions: list[list[int]] = []
    position_valid: list[list[int]] = []
    counts: list[list[int]] = []
    keeps: list[float] = []
    cu_seqlens = [0]
    K = config.max_scored_positions
    T = config.max_teacher_steps

    for index, (sample, record) in enumerate(zip(samples, records)):
        prompt = list(sample.prompt_token_ids)
        completion = list(record.completion_ids)
        start = offset if offset is not None else len(prompt)
        eligible = [
            k for k, step in enumerate(record.steps)
            if k < len(record.steps) - 1 and sum(step.mask) / L > config.mask_threshold
        ]
        if not eligible:
            eligible = [0]
        k = eligible[rng.randrange(len(eligible))]
        student_step = record.steps[k]
        flags = [
            pos for pos in range(L) if student_step.mask[pos] and pos < L
        ]
        if len(flags) > K:
            flags = sorted(rng.sample(flags, K))
        valid = [1] * len(flags) + [0] * (K - len(flags))
        flags_padded = flags + [0] * (K - len(flags))

        later = [
            t for t in range(k + 1, len(record.steps))
            if any(record.steps[t].mask[pos] for pos in flags)
        ]
        if len(later) > T:
            later = sorted(rng.sample(later, T))
        if not later:
            later = [len(record.steps) - 1]
        present = [1] * len(later) + [0] * (T - len(later))
        canvases: list[list[int]] = []
        for t in later:
            canvases.append(
                _full_canvas(prompt, prompt_span, record.steps[t].canvas_ids, config)
                + [0] * (width - (prompt_span if config.family == "block" else len(prompt)) - L)
            )
        while len(canvases) < T:
            canvases.append([0] * width)
        position_masks: list[list[int]] = []
        for t in later:
            position_masks.append(
                [int(record.steps[t].mask[pos]) if pos < L else 0 for pos in flags_padded]
            )
        while len(position_masks) < T:
            position_masks.append([0] * K)
        # No genuine future prediction: fall back to the endpoint teacher.
        if later == [len(record.steps) - 1]:
            position_masks[0] = [valid[j] for j in range(K)]
        per_position = [
            max(1, sum(row[pos] for row in position_masks))
            for pos in range(K)
        ]

        full = _full_canvas(prompt, prompt_span, completion, config)
        input_ids.append(full + [0] * (width - len(full)))
        attention.append(
            [1] * len(prompt)
            + [0] * (prompt_span - len(prompt))
            + [1] * L
            + [0] * (width - prompt_span - L)
            if config.family == "block"
            else [1] * (len(prompt) + L) + [0] * (width - len(prompt) - L)
        )
        loss_mask.append([0] * width)
        positions.append(list(range(width)))
        old_logprobs.append([0.0] * width)
        versions.append([record.version] * width)
        student = _full_canvas(prompt, prompt_span, student_step.canvas_ids, config)
        student_canvas.append(student + [0] * (width - len(student)))
        teacher_canvas.append(canvases)
        teacher_present.append(present)
        teacher_position_mask.append(position_masks)
        mask_positions.append([start + pos for pos in flags_padded])
        position_valid.append(valid)
        counts.append(per_position)
        keeps.append(
            1.0
            if not config.verify_rollouts
            or float(sample.reward) >= config.correct_reward_threshold
            else 0.0
        )
        cu_seqlens.append(cu_seqlens[-1] + width)

    return DOPSDBatch(
        input_ids=input_ids,
        attention_mask=attention,
        loss_mask=loss_mask,
        position_ids=positions,
        old_logprobs=old_logprobs,
        versions=versions,
        rewards=[float(s.reward) for s in samples],
        cu_seqlens=cu_seqlens,
        max_len=width,
        sampling_logprobs=[],
        dopsd_student_canvas=student_canvas,
        dopsd_teacher_canvas=teacher_canvas,
        dopsd_teacher_present=teacher_present,
        dopsd_teacher_mask=teacher_position_mask,
        dopsd_prompt_lens=[len(s.prompt_token_ids) for s in samples],
        dopsd_prompt_span=prompt_span,
        dopsd_mask_positions=mask_positions,
        dopsd_position_valid=position_valid,
        dopsd_counts=counts,
        dopsd_keep=keeps,
    )


def slice_dopsd_batch(batch: DOPSDBatch, start: int, stop: int) -> DOPSDBatch:
    return DOPSDBatch(
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
        dopsd_student_canvas=batch.dopsd_student_canvas[start:stop],
        dopsd_teacher_canvas=batch.dopsd_teacher_canvas[start:stop],
        dopsd_teacher_present=batch.dopsd_teacher_present[start:stop],
        dopsd_teacher_mask=batch.dopsd_teacher_mask[start:stop],
        dopsd_prompt_lens=batch.dopsd_prompt_lens[start:stop],
        dopsd_prompt_span=batch.dopsd_prompt_span,
        dopsd_mask_positions=batch.dopsd_mask_positions[start:stop],
        dopsd_position_valid=batch.dopsd_position_valid[start:stop],
        dopsd_counts=batch.dopsd_counts[start:stop],
        dopsd_keep=batch.dopsd_keep[start:stop],
    )


def make_dopsd_packer(config: DOPSDConfig, store: DOPSDStore):
    """Resolve trajectory handles, apply answer verification and collate."""

    def pack(samples, *, max_tokens=None, alignment=1):
        records = []
        for sample in samples:
            handle = getattr(sample, "payload_handle", None)
            if handle is None:
                raise ValueError("dOPSD packer needs payload handles on samples")
            record = store.get(handle)
            store.release(handle)
            records.append(dataclasses.replace(record, reward=float(sample.reward)))
        batch = collate_dopsd(samples, records, config)
        return [batch], [list(range(len(samples)))]

    return pack
