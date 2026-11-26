"""TIPS batching — TIPS: A Tiny Inference-time Policy for Scaling in Diffusion Language
Models (Zhang et al., arXiv:2610.22099; code https://github.com/zhengyz97/TIPS).
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import tensorplay as tp

from ....batching import RLBatch
from ....dllm import DLLMSample
from ....losses import TensorBatch
from .config import TIPSConfig
from .types import TIPSRecord, TIPSStore


@dataclasses.dataclass(frozen=True, slots=True)
class TIPSBatch(RLBatch):
    tips_features: list[list[list[float]]] = dataclasses.field(default_factory=list, kw_only=True)
    tips_mask: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    tips_time: list[float] = dataclasses.field(default_factory=list, kw_only=True)
    tips_actions: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    tips_old_logprobs: list[float] = dataclasses.field(default_factory=list, kw_only=True)
    tips_sample_ids: list[int] = dataclasses.field(default_factory=list, kw_only=True)


class TIPSTensorBatch(TensorBatch):
    def __init__(self, batch: TIPSBatch, device: str = "cpu") -> None:
        super().__init__(batch, device=device)
        self.tips_features = tp.tensor(batch.tips_features, dtype=tp.float32, device=device)
        self.tips_mask = tp.tensor(batch.tips_mask, dtype=tp.float32, device=device)
        self.tips_time = tp.tensor(batch.tips_time, dtype=tp.float32, device=device)
        self.tips_actions = tp.tensor(batch.tips_actions, dtype=tp.float32, device=device)
        self.tips_old_logprobs = tp.tensor(batch.tips_old_logprobs, dtype=tp.float32, device=device)
        self.tips_sample_ids = list(batch.tips_sample_ids)


def collate_tips(
    samples: Sequence[DLLMSample],
    records: Sequence[TIPSRecord],
    config: TIPSConfig,
) -> TIPSBatch:
    """Flatten each recorded visibility decision into one trainer row."""
    if not samples or len(samples) != len(records):
        raise ValueError("collate_tips needs aligned samples and records")
    if len({id(r) for r in records}) != len(records):
        raise ValueError("records must be distinct")
    L = config.max_completion_length
    width = max((max(len(s.prompt_token_ids) for s in samples), config.max_prompt_len or 0))
    span = width if config.family == "block" else 0
    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    loss_mask: list[list[int]] = []
    position_ids: list[list[int]] = []
    old_logprobs: list[list[float]] = []
    versions: list[list[int]] = []
    features: list[list[list[float]]] = []
    masks: list[list[int]] = []
    times: list[float] = []
    actions: list[list[int]] = []
    old_step_logprobs: list[float] = []
    sample_ids: list[int] = []
    cu_seqlens = [0]

    for sample_index, (sample, record) in enumerate(zip(samples, records)):
        record.validate()
        if len(record.steps) != config.train_steps:
            raise ValueError("TIPS record step count does not match the budget")
        padded_prompt = list(sample.prompt_token_ids) + [config.noise_token_id] * (
            span - len(sample.prompt_token_ids)
        )
        for step in record.steps:
            if len(step.features[0]) != 2:
                raise ValueError("TIPS features must be two-dimensional")
            row = padded_prompt + list(step.canvas_ids) + [0] * (L - len(step.canvas_ids))
            input_ids.append(row)
            attention.append(
                [1] * len(sample.prompt_token_ids)
                + [0] * (span - len(sample.prompt_token_ids))
                + [1] * len(step.canvas_ids)
                + [0] * (L - len(step.canvas_ids))
                if config.family == "block"
                else [1] * len(row)
            )
            loss_mask.append(
                [0] * span + [1] * len(step.canvas_ids) + [0] * (L - len(step.canvas_ids))
            )
            position_ids.append(list(range(len(row))))
            old_logprobs.append([0.0] * len(row))
            versions.append([sample.version] * len(row))
            features.append(step.features)
            masks.append(step.mask)
            times.append(step.time)
            actions.append(step.actions)
            old_step_logprobs.append(step.old_logprob)
            sample_ids.append(sample_index)
            cu_seqlens.append(cu_seqlens[-1] + len(row))

    return TIPSBatch(
        input_ids=input_ids,
        attention_mask=attention,
        loss_mask=loss_mask,
        position_ids=position_ids,
        old_logprobs=old_logprobs,
        versions=versions,
        rewards=[
            sample.reward
            for sample, record in zip(samples, records)
            for _ in record.steps
        ],
        cu_seqlens=cu_seqlens,
        max_len=span + L,
        sampling_logprobs=[],
        tips_features=features,
        tips_mask=masks,
        tips_time=times,
        tips_actions=actions,
        tips_old_logprobs=old_step_logprobs,
        tips_sample_ids=sample_ids,
    )


def slice_tips_batch(batch: TIPSBatch, start: int, stop: int) -> TIPSBatch:
    return TIPSBatch(
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
        tips_features=batch.tips_features[start:stop],
        tips_mask=batch.tips_mask[start:stop],
        tips_time=batch.tips_time[start:stop],
        tips_actions=batch.tips_actions[start:stop],
        tips_old_logprobs=batch.tips_old_logprobs[start:stop],
        tips_sample_ids=batch.tips_sample_ids[start:stop],
    )


def make_tips_packer(config: TIPSConfig, store: TIPSStore):
    """Return a ``batch_packer`` aligned with the controller's row order."""

    def pack(samples, *, max_tokens=None, alignment=1):
        records = []
        for sample in samples:
            if sample.payload_handle is None:
                raise ValueError("TIPS packer needs a payload handle on every sample")
            record = store.get(sample.payload_handle)
            store.release(sample.payload_handle)
            record = dataclasses.replace(record, reward=sample.reward)
            records.append(record)
        batch = collate_tips(samples, records, config)
        return [batch], [list(batch.tips_sample_ids)]

    return pack
