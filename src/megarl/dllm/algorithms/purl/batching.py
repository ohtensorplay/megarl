"""PURL batching — When Should Diffusion Language Models Reconsider Their Commitments?
(Zheng et al., arXiv:2611.11551v1; withdrawn after v1).
"""

from __future__ import annotations

import dataclasses
import random
from typing import Sequence

import tensorplay as tp

from ....batching import RLBatch
from ....dllm import DLLMSample
from ....losses import TensorBatch
from .config import PURLConfig
from .types import PURLRecord, PURLStore


@dataclasses.dataclass(frozen=True, slots=True)
class PURLBatch(RLBatch):
    purl_reference_ids: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    purl_mask: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    purl_noisy: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    purl_correct: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    purl_edit_candidates: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    purl_unmask_candidates: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    purl_edit_labels: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    purl_sample_ids: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    purl_prompt_span: int = dataclasses.field(default=0, kw_only=True)
    purl_prompt_lens: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    purl_canvas_len: int = dataclasses.field(default=0, kw_only=True)


class PURLTensorBatch(TensorBatch):
    def __init__(self, batch: PURLBatch, device: str = "cpu") -> None:
        super().__init__(batch, device=device)
        self.purl_reference_ids = tp.tensor(batch.purl_reference_ids, dtype=tp.long, device=device)
        self.purl_mask = tp.tensor(batch.purl_mask, dtype=tp.float32, device=device)
        self.purl_noisy = tp.tensor(batch.purl_noisy, dtype=tp.float32, device=device)
        self.purl_correct = tp.tensor(batch.purl_correct, dtype=tp.float32, device=device)
        self.purl_edit_candidates = tp.tensor(batch.purl_edit_candidates, dtype=tp.float32, device=device)
        self.purl_unmask_candidates = tp.tensor(batch.purl_unmask_candidates, dtype=tp.float32, device=device)
        self.purl_edit_labels = tp.tensor(batch.purl_edit_labels, dtype=tp.float32, device=device)
        self.purl_sample_ids = list(batch.purl_sample_ids)
        self.purl_prompt_span = int(batch.purl_prompt_span)
        self.purl_prompt_lens = list(batch.purl_prompt_lens)
        self.purl_canvas_len = int(batch.purl_canvas_len)
        # Canonical names used by the shared model adapter.
        self.dllm_prompt_span = self.purl_prompt_span
        self.dllm_prompt_lens = tp.tensor(batch.purl_prompt_lens, dtype=tp.long, device=device)
        self.dllm_canvas_len = self.purl_canvas_len
        self.dllm_alpha = tp.ones((self.input_ids.shape[0],), dtype=tp.float32, device=device)
        self.dllm_noisy_ids = self.input_ids


def _build_corruption(reference, config: PURLConfig, seed: int) -> tuple:
    """Build a clean partially masked state plus a transient noisy view."""
    rng = random.Random(seed)
    L = len(reference)
    mask = [rng.random() < config.mask_prob for _ in range(L)]
    if not any(mask):
        mask[rng.randrange(L)] = True
    clean = [
        config.noise_token_id if flag else token
        for token, flag in zip(reference, mask)
    ]
    eligible = [i for i in range(L) if not mask[i]]
    noisy_positions = [
        i for i in eligible if rng.random() < config.corrupt_prob
    ]
    if not noisy_positions and eligible:
        noisy_positions = [rng.choice(eligible)]
    noisy = list(clean)
    for i in noisy_positions:
        alternatives = [
            token for token in range(min(config.vocab_size, 32)) if token != reference[i]
        ]
        noisy[i] = rng.choice(alternatives) if alternatives else config.noise_token_id
    noisy_flags = [0] * L
    correct_flags = [0] * L
    for i in range(L):
        if i in noisy_positions:
            noisy_flags[i] = 1
        elif not mask[i]:
            correct_flags[i] = 1
    return clean, noisy, [int(v) for v in mask], noisy_flags, correct_flags


def collate_purl(
    samples: Sequence[DLLMSample],
    records: Sequence[PURLRecord],
    config: PURLConfig,
) -> PURLBatch:
    if not samples or len(samples) != len(records):
        raise ValueError("collate_purl needs aligned samples and records")
    L = config.max_completion_length
    width = max((max(len(s.prompt_token_ids) for s in samples), config.max_prompt_len or 0))
    span = width if config.family == "block" else 0
    input_ids: list[list[int]] = []
    references: list[list[int]] = []
    masks: list[list[int]] = []
    noisy: list[list[int]] = []
    correct: list[list[int]] = []
    edit_candidates: list[list[int]] = []
    unmask_candidates: list[list[int]] = []
    edit_labels: list[list[int]] = []
    sample_ids: list[int] = []
    cu_seqlens = [0]

    for sample_index, (sample, record) in enumerate(zip(samples, records)):
        record.validate()
        reference = (list(record.reference_ids) + [config.noise_token_id] * L)[:L]
        _, noisy_canvas, mask, noisy_flags, correct_flags = _build_corruption(
            reference, config, config.seed + sample_index
        )
        padded_prompt = list(sample.prompt_token_ids) + [config.noise_token_id] * (
            span - len(sample.prompt_token_ids)
        )
        row = padded_prompt + noisy_canvas
        input_ids.append(row)
        references.append(reference)
        masks.append(mask)
        noisy.append(noisy_flags)
        correct.append(correct_flags)
        edit_candidates.append(correct_flags)
        unmask_candidates.append(mask)
        edit_labels.append(noisy_flags)
        sample_ids.append(sample_index)
        cu_seqlens.append(cu_seqlens[-1] + len(row))

    return PURLBatch(
        input_ids=input_ids,
        attention_mask=[
            [1] * len(s.prompt_token_ids)
            + [0] * (span - len(s.prompt_token_ids))
            + [1] * L
            for s in samples
        ],
        loss_mask=[[0] * span + [1] * L for _ in samples],
        position_ids=[list(range(span + L)) for _ in samples],
        old_logprobs=[[0.0] * (span + L) for _ in samples],
        versions=[[s.version] * (span + L) for s in samples],
        rewards=[s.reward for s in samples],
        cu_seqlens=cu_seqlens,
        max_len=span + L,
        sampling_logprobs=[],
        purl_reference_ids=references,
        purl_mask=masks,
        purl_noisy=noisy,
        purl_correct=correct,
        purl_edit_candidates=edit_candidates,
        purl_unmask_candidates=unmask_candidates,
        purl_edit_labels=edit_labels,
        purl_sample_ids=sample_ids,
        purl_prompt_span=span,
        purl_prompt_lens=[len(s.prompt_token_ids) for s in samples],
        purl_canvas_len=L,
    )


def slice_purl_batch(batch: PURLBatch, start: int, stop: int) -> PURLBatch:
    return PURLBatch(
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
        purl_reference_ids=batch.purl_reference_ids[start:stop],
        purl_mask=batch.purl_mask[start:stop],
        purl_noisy=batch.purl_noisy[start:stop],
        purl_correct=batch.purl_correct[start:stop],
        purl_edit_candidates=batch.purl_edit_candidates[start:stop],
        purl_unmask_candidates=batch.purl_unmask_candidates[start:stop],
        purl_edit_labels=batch.purl_edit_labels[start:stop],
        purl_sample_ids=batch.purl_sample_ids[start:stop],
        purl_prompt_span=batch.purl_prompt_span,
        purl_prompt_lens=batch.purl_prompt_lens[start:stop],
        purl_canvas_len=batch.purl_canvas_len,
    )


def make_purl_packer(config: PURLConfig, store: PURLStore):
    def pack(samples, *, max_tokens=None, alignment=1):
        records = []
        for sample in samples:
            if sample.payload_handle is None:
                raise ValueError("PURL packer needs a payload handle on every sample")
            record = store.get(sample.payload_handle)
            store.release(sample.payload_handle)
            records.append(dataclasses.replace(record, reward=sample.reward))
        batch = collate_purl(samples, records, config)
        return [batch], [list(batch.purl_sample_ids)]

    return pack
