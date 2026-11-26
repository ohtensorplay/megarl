"""T3D, Tunyu Zhang et al., NeurIPS 2026 arXiv 2602.12262; code
https://github.com/Tyrion58/T3D; see package docstring for details."""

from __future__ import annotations

import dataclasses
import random
from typing import Sequence

import tensorplay as tp

from ....batching import RLBatch
from ....dllm import DLLMSample
from ....losses import TensorBatch
from .config import T3DConfig
from .types import T3DRecord, T3DStore


@dataclasses.dataclass(frozen=True, slots=True)
class T3DBatch(RLBatch):
    """Rows are ``(sample, intermediate teacher state)`` pairs."""

    t3d_canvas: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    t3d_prompt_lens: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    t3d_prompt_span: int = dataclasses.field(default=0, kw_only=True)
    t3d_ref_targets: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    t3d_mask_positions: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    t3d_mask_valid: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    t3d_path_positions: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    t3d_path_valid: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    t3d_path_weights: list[list[float]] = dataclasses.field(default_factory=list, kw_only=True)
    t3d_sample_ids: list[int] = dataclasses.field(default_factory=list, kw_only=True)


class T3DTensorBatch(TensorBatch):
    """Device tensors for one :class:`T3DBatch` slice."""

    def __init__(self, batch: T3DBatch, device: str = "cpu") -> None:
        super().__init__(batch, device=device)
        self.t3d_canvas = tp.tensor(batch.t3d_canvas, dtype=tp.long, device=device)
        self.t3d_prompt_lens = tp.tensor(batch.t3d_prompt_lens, dtype=tp.long, device=device)
        self.t3d_prompt_span = int(batch.t3d_prompt_span)
        self.t3d_ref_targets = tp.tensor(batch.t3d_ref_targets, dtype=tp.long, device=device)
        self.t3d_mask_positions = tp.tensor(batch.t3d_mask_positions, dtype=tp.long, device=device)
        self.t3d_mask_valid = tp.tensor(batch.t3d_mask_valid, dtype=tp.float32, device=device)
        self.t3d_path_positions = tp.tensor(batch.t3d_path_positions, dtype=tp.long, device=device)
        self.t3d_path_valid = tp.tensor(batch.t3d_path_valid, dtype=tp.float32, device=device)
        self.t3d_path_weights = tp.tensor(batch.t3d_path_weights, dtype=tp.float32, device=device)
        self.t3d_sample_ids = list(batch.t3d_sample_ids)


def _full_canvas(prompt: list[int], span: int, canvas: list[int], config: T3DConfig):
    if config.family == "block":
        return prompt + [config.noise_token_id] * (span - len(prompt)) + canvas
    return prompt + canvas


def collate_t3d(
    samples: Sequence[DLLMSample], records: Sequence[T3DRecord], config: T3DConfig
) -> T3DBatch:
    """Expand each teacher trajectory into intermediate-state training rows."""
    if not samples or len(samples) != len(records):
        raise ValueError("T3D collate needs matching samples and records")
    for record in records:
        record.validate()
    L = config.max_completion_length
    for record in records:
        if record.length != L:
            raise ValueError("T3D completion length must match the config")
    prompt_span = (
        max(len(s.prompt_token_ids) for s in samples)
        if config.family == "block"
        else 0
    )
    width = (prompt_span if config.family == "block" else max(len(s.prompt_token_ids) for s in samples)) + L
    start = prompt_span if config.family == "block" else None
    rng = random.Random(config.seed)
    K = config.max_masked_positions
    path_weights_by_sample = [config.path_weights(r.decode_steps) for r in records]

    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    loss_mask: list[list[int]] = []
    positions: list[list[int]] = []
    old_logprobs: list[list[float]] = []
    versions: list[list[int]] = []
    canvases: list[list[int]] = []
    ref_targets: list[list[int]] = []
    mask_positions: list[list[int]] = []
    mask_valid: list[list[int]] = []
    path_positions: list[list[int]] = []
    path_valid: list[list[int]] = []
    path_weights: list[list[float]] = []
    sample_ids: list[int] = []
    rewards: list[float] = []
    prompt_lengths: list[int] = []
    cu_seqlens = [0]

    for index, (sample, record) in enumerate(zip(samples, records)):
        prompt = list(sample.prompt_token_ids)
        completion = list(record.completion_ids)
        offset = start if start is not None else len(prompt)
        steps = [
            k for k, step in enumerate(record.steps) if any(step.mask)
        ][: config.max_train_steps]
        for k in steps:
            step = record.steps[k]
            masked = [pos for pos in range(L) if step.mask[pos]]
            if len(masked) > K:
                masked = sorted(rng.sample(masked, K))
            valid = [1] * len(masked) + [0] * (K - len(masked))
            padded_mask = masked + [0] * (K - len(masked))
            decoded = [
                pos for pos in range(L) if record.decode_steps[pos] == k
            ]
            P = max(1, len(decoded))
            padded_path = decoded + [0] * (P - len(decoded))
            path_ok = [1] * len(decoded) + [0] * (P - len(decoded))
            weights = [path_weights_by_sample[index][pos] for pos in decoded] + [
                0.0
            ] * (P - len(decoded))

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
            canvas = _full_canvas(prompt, prompt_span, step.canvas_ids, config)
            canvases.append(canvas + [0] * (width - len(canvas)))
            ref = _full_canvas(prompt, prompt_span, record.ref_completion_ids, config)
            ref_targets.append(ref + [0] * (width - len(ref)))
            mask_positions.append([offset + pos for pos in padded_mask])
            mask_valid.append(valid)
            path_positions.append([offset + pos for pos in padded_path])
            path_valid.append(path_ok)
            path_weights.append(weights)
            sample_ids.append(index)
            rewards.append(float(sample.reward))
            prompt_lengths.append(len(prompt))
            cu_seqlens.append(cu_seqlens[-1] + width)

    return T3DBatch(
        input_ids=input_ids,
        attention_mask=attention,
        loss_mask=loss_mask,
        position_ids=positions,
        old_logprobs=old_logprobs,
        versions=versions,
        rewards=rewards,
        cu_seqlens=cu_seqlens,
        max_len=width,
        sampling_logprobs=[],
        t3d_canvas=canvases,
        t3d_prompt_lens=prompt_lengths,
        t3d_prompt_span=prompt_span,
        t3d_ref_targets=ref_targets,
        t3d_mask_positions=mask_positions,
        t3d_mask_valid=mask_valid,
        t3d_path_positions=path_positions,
        t3d_path_valid=path_valid,
        t3d_path_weights=path_weights,
        t3d_sample_ids=sample_ids,
    )


def slice_t3d_batch(batch: T3DBatch, start: int, stop: int) -> T3DBatch:
    return T3DBatch(
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
        t3d_canvas=batch.t3d_canvas[start:stop],
        t3d_prompt_lens=batch.t3d_prompt_lens[start:stop],
        t3d_prompt_span=batch.t3d_prompt_span,
        t3d_ref_targets=batch.t3d_ref_targets[start:stop],
        t3d_mask_positions=batch.t3d_mask_positions[start:stop],
        t3d_mask_valid=batch.t3d_mask_valid[start:stop],
        t3d_path_positions=batch.t3d_path_positions[start:stop],
        t3d_path_valid=batch.t3d_path_valid[start:stop],
        t3d_path_weights=batch.t3d_path_weights[start:stop],
        t3d_sample_ids=batch.t3d_sample_ids[start:stop],
    )


def make_t3d_packer(config: T3DConfig, store: T3DStore):
    """Resolve handles and align each intermediate state with its sample."""

    def pack(samples, *, max_tokens=None, alignment=1):
        records = []
        for sample in samples:
            handle = getattr(sample, "payload_handle", None)
            if handle is None:
                raise ValueError("T3D packer needs payload handles on samples")
            record = store.get(handle)
            store.release(handle)
            records.append(dataclasses.replace(record, reward=float(sample.reward)))
        batch = collate_t3d(samples, records, config)
        return [batch], [list(batch.t3d_sample_ids)]

    return pack
