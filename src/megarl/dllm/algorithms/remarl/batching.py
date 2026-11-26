"""REMARL batching — ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language
Models (Shibaev et al., arXiv:2611.22576; code https://github.com/EgorShibaev/remarl).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

import tensorplay as tp

from ....batching import RLBatch
from ....dllm import DLLMSample
from ....losses import TensorBatch
from .config import ReMaRLConfig
from .types import ReMaRLRecord, ReMaRLStore


@dataclasses.dataclass(frozen=True, slots=True)
class ReMaRLBatch(RLBatch):
    remarl_scores: list[list[float]] = dataclasses.field(default_factory=list, kw_only=True)
    remarl_mask: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    remarl_hidden: list[list[list[float]]] = dataclasses.field(default_factory=list, kw_only=True)
    remarl_actions: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    remarl_action_valid: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    remarl_old_logprobs: list[float] = dataclasses.field(default_factory=list, kw_only=True)
    remarl_sample_ids: list[int] = dataclasses.field(default_factory=list, kw_only=True)


class ReMaRLTensorBatch(TensorBatch):
    def __init__(self, batch: ReMaRLBatch, device: str = "cpu") -> None:
        super().__init__(batch, device=device)
        self.remarl_scores = tp.tensor(batch.remarl_scores, dtype=tp.float32, device=device)
        self.remarl_mask = tp.tensor(batch.remarl_mask, dtype=tp.float32, device=device)
        self.remarl_hidden = tp.tensor(batch.remarl_hidden, dtype=tp.float32, device=device)
        self.remarl_actions = tp.tensor(batch.remarl_actions, dtype=tp.long, device=device)
        self.remarl_action_valid = tp.tensor(batch.remarl_action_valid, dtype=tp.float32, device=device)
        self.remarl_old_logprobs = tp.tensor(batch.remarl_old_logprobs, dtype=tp.float32, device=device)
        self.remarl_sample_ids = list(batch.remarl_sample_ids)


def collate_remarl(
    samples: Sequence[DLLMSample],
    records: Sequence[ReMaRLRecord],
    config: ReMaRLConfig,
    *,
    hidden_fn: Callable[[Any, Any], Any] | None = None,
) -> ReMaRLBatch:
    if not samples or len(samples) != len(records):
        raise ValueError("collate_remarl needs aligned samples and records")
    L = config.max_completion_length
    width = max((max(len(s.prompt_token_ids) for s in samples), config.max_prompt_len or 0))
    span = width if config.family == "block" else 0
    max_k = max((len(step.order) for r in records for step in r.steps), default=1)
    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    loss_mask: list[list[int]] = []
    old_logprobs: list[list[float]] = []
    versions: list[list[int]] = []
    scores: list[list[float]] = []
    masks: list[list[int]] = []
    hidden: list[list[list[float]]] = []
    actions: list[list[int]] = []
    action_valid: list[list[int]] = []
    old_step_logprobs: list[float] = []
    sample_ids: list[int] = []
    cu_seqlens = [0]

    for sample_index, (sample, record) in enumerate(zip(samples, records)):
        record.validate()
        if len(record.steps) != config.train_steps:
            raise ValueError("ReMaRL record step count does not match the budget")
        padded_prompt = list(sample.prompt_token_ids) + [config.noise_token_id] * (
            span - len(sample.prompt_token_ids)
        )
        for step in record.steps:
            order = list(step.order) + [0] * (max_k - len(step.order))
            valid = [1] * len(step.order) + [0] * (max_k - len(step.order))
            ids = tp.tensor(
                [[config.noise_token_id if flag else token
                  for token, flag in zip(step.canvas_ids, step.mask)]],
                dtype=tp.long,
            )
            canvas_tensor = tp.tensor([step.canvas_ids], dtype=tp.long)
            if hidden_fn is not None:
                computed = hidden_fn(ids, canvas_tensor)
                hidden_rows = computed.tolist() if hasattr(computed, "tolist") else computed
            else:
                hidden_rows = [[0.0] * config.hidden_dim for _ in range(L)]
            row = padded_prompt + list(step.canvas_ids) + [0] * (L - len(step.canvas_ids))
            input_ids.append(row)
            attention.append(
                [1] * len(sample.prompt_token_ids)
                + [0] * (span - len(sample.prompt_token_ids))
                + [1] * len(step.canvas_ids)
                if config.family == "block"
                else [1] * len(row)
            )
            loss_mask.append(
                [0] * span + [1] * len(step.canvas_ids) + [0] * (L - len(step.canvas_ids))
            )
            old_logprobs.append([0.0] * len(row))
            versions.append([sample.version] * len(row))
            scores.append(step.scores)
            masks.append(step.mask)
            hidden.append(hidden_rows)
            actions.append(order)
            action_valid.append(valid)
            old_step_logprobs.append(step.old_logprob)
            sample_ids.append(sample_index)
            cu_seqlens.append(cu_seqlens[-1] + len(row))

    return ReMaRLBatch(
        input_ids=input_ids,
        attention_mask=attention,
        loss_mask=loss_mask,
        position_ids=[list(range(len(row))) for row in input_ids],
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
        remarl_scores=scores,
        remarl_mask=masks,
        remarl_hidden=hidden,
        remarl_actions=actions,
        remarl_action_valid=action_valid,
        remarl_old_logprobs=old_step_logprobs,
        remarl_sample_ids=sample_ids,
    )


def slice_remarl_batch(batch: ReMaRLBatch, start: int, stop: int) -> ReMaRLBatch:
    return ReMaRLBatch(
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
        remarl_scores=batch.remarl_scores[start:stop],
        remarl_mask=batch.remarl_mask[start:stop],
        remarl_hidden=batch.remarl_hidden[start:stop],
        remarl_actions=batch.remarl_actions[start:stop],
        remarl_action_valid=batch.remarl_action_valid[start:stop],
        remarl_old_logprobs=batch.remarl_old_logprobs[start:stop],
        remarl_sample_ids=batch.remarl_sample_ids[start:stop],
    )


def make_remarl_packer(
    config: ReMaRLConfig,
    store: ReMaRLStore,
    *,
    hidden_fn: Callable[[Any, Any], Any] | None = None,
):
    def pack(samples, *, max_tokens=None, alignment=1):
        records = []
        for sample in samples:
            if sample.payload_handle is None:
                raise ValueError("ReMaRL packer needs a payload handle on every sample")
            record = store.get(sample.payload_handle)
            store.release(sample.payload_handle)
            records.append(dataclasses.replace(record, reward=sample.reward))
        batch = collate_remarl(samples, records, config, hidden_fn=hidden_fn)
        return [batch], [list(batch.remarl_sample_ids)]

    return pack
