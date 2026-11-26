"""AGRPO / Amortized GRPO, Anthony Zhan, ICML 2026 arXiv 2510.04019; code
https://github.com/probablyabot/agrpo; see package docstring for details."""

from __future__ import annotations

import dataclasses
from typing import Sequence

import tensorplay as tp

from ....batching import RLBatch
from ....dllm import DLLMSample
from ....losses import TensorBatch
from .config import AGRPOConfig
from .rollout import sample_mc_steps
from .types import AGRPORecord, AGRPOStore


@dataclasses.dataclass(frozen=True, slots=True)
class AGRPOBatch(RLBatch):
    """One row per ``(sample, MC draw, unmasked position)`` triple."""

    dllm_noisy_ids: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    dllm_prompt_lens: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    dllm_prompt_span: int = dataclasses.field(default=0, kw_only=True)
    dllm_canvas_len: int = dataclasses.field(default=0, kw_only=True)
    agrpo_sample_ids: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    agrpo_group_ids: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    agrpo_sample_rewards: list[float] = dataclasses.field(default_factory=list, kw_only=True)
    agrpo_importance_weights: list[float] = dataclasses.field(default_factory=list, kw_only=True)
    agrpo_loss_scale: float = dataclasses.field(default=1.0, kw_only=True)


class AGRPOTensorBatch(TensorBatch):
    def __init__(self, batch: AGRPOBatch, device: str = "cpu") -> None:
        super().__init__(batch, device=device)
        self.dllm_noisy_ids = tp.tensor(batch.dllm_noisy_ids, dtype=tp.long, device=device)
        self.dllm_prompt_lens = tp.tensor(batch.dllm_prompt_lens, dtype=tp.long, device=device)
        self.dllm_prompt_span = int(batch.dllm_prompt_span)
        self.dllm_canvas_len = int(batch.dllm_canvas_len)


def collate_agrpo(
    samples: Sequence[DLLMSample],
    records: Sequence[AGRPORecord],
    config: AGRPOConfig,
) -> AGRPOBatch:
    """Build MC step canvases and align each unmasked token with its log-prob."""
    if not samples or len(samples) != len(records):
        raise ValueError("AGRPO collator needs aligned non-empty samples/records")
    for record in records:
        record.validate()
        if record.length != config.max_completion_length:
            raise ValueError("AGRPO record length does not match max_completion_length")

    C = config.max_completion_length
    max_prompt = max(len(s.prompt_token_ids) for s in samples)
    prompt_span = max(max_prompt, config.max_prompt_len or 0)
    width = prompt_span + C
    fill = config.prompt_fill()
    noise = config.noise_token_id
    mc_steps, importance_weights = sample_mc_steps(records, config)

    input_ids: list[list[int]] = []
    noisy_ids: list[list[int]] = []
    attention: list[list[int]] = []
    loss_mask: list[list[int]] = []
    position_ids: list[list[int]] = []
    old_logprobs: list[list[float]] = []
    versions: list[list[int]] = []
    sample_ids: list[int] = []
    prompt_lens: list[int] = []
    group_ids: list[int] = []
    sample_rewards: list[float] = []
    iw_rows: list[float] = []
    cu_seqlens = [0]

    for index, (sample, record) in enumerate(zip(samples, records)):
        prompt = list(sample.prompt_token_ids)
        completion = list(sample.completion_token_ids)
        padded_prompt = prompt + [fill] * (prompt_span - len(prompt))
        for j in range(config.mc_samples):
            step = mc_steps[index * config.mc_samples + j]
            iw = importance_weights[index * config.mc_samples + j]
            targets = [
                pos for pos, s in enumerate(record.unmask_steps) if s == step
            ]
            if not targets:
                continue
            canvas = padded_prompt + [
                token if s < step else noise
                for token, s in zip(completion, record.unmask_steps)
            ]
            for pos in targets:
                target = prompt_span + pos
                target_row = [0] * width
                target_row[target] = completion[pos]
                mask_row = [0] * width
                mask_row[target] = 1
                old_row = [0.0] * width
                old_row[target] = record.token_logprobs[pos]
                input_ids.append(target_row)
                noisy_ids.append(canvas)
                attention.append(
                    [1] * len(prompt) + [0] * (prompt_span - len(prompt)) + [1] * C
                )
                loss_mask.append(mask_row)
                position_ids.append(list(range(width)))
                old_logprobs.append(old_row)
                versions.append([record.version] * width)
                sample_ids.append(index)
                prompt_lens.append(len(prompt))
                group_ids.append(record.group_id)
                sample_rewards.append(float(sample.reward))
                iw_rows.append(iw)
                cu_seqlens.append(cu_seqlens[-1] + width)

    if not input_ids:
        raise ValueError("AGRPO collator produced no MC training rows")
    scale = len(input_ids) / (config.mc_samples * config.group_size)
    return AGRPOBatch(
        input_ids=input_ids,
        attention_mask=attention,
        loss_mask=loss_mask,
        position_ids=position_ids,
        old_logprobs=old_logprobs,
        versions=versions,
        rewards=sample_rewards,
        cu_seqlens=cu_seqlens,
        max_len=width,
        sampling_logprobs=[],
        dllm_noisy_ids=noisy_ids,
        dllm_prompt_lens=prompt_lens,
        dllm_prompt_span=prompt_span,
        dllm_canvas_len=C,
        agrpo_sample_ids=sample_ids,
        agrpo_group_ids=group_ids,
        agrpo_sample_rewards=sample_rewards,
        agrpo_importance_weights=iw_rows,
        agrpo_loss_scale=scale,
    )


def slice_agrpo_batch(batch: AGRPOBatch, start: int, stop: int) -> AGRPOBatch:
    return AGRPOBatch(
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
        agrpo_sample_ids=batch.agrpo_sample_ids[start:stop],
        agrpo_group_ids=batch.agrpo_group_ids[start:stop],
        agrpo_sample_rewards=batch.agrpo_sample_rewards[start:stop],
        agrpo_importance_weights=batch.agrpo_importance_weights[start:stop],
        agrpo_loss_scale=batch.agrpo_loss_scale,
    )


def make_agrpo_packer(config: AGRPOConfig, store: AGRPOStore):
    def pack(samples, *, max_tokens=None, alignment=1):
        records = []
        for sample in samples:
            handle = getattr(sample, "payload_handle", None)
            if handle is None:
                raise ValueError("AGRPO packer needs payload handles on samples")
            record = store.get(handle)
            store.release(handle)
            records.append(dataclasses.replace(record, reward=float(sample.reward)))
        batch = collate_agrpo(samples, records, config)
        return [batch], [list(batch.agrpo_sample_ids)]

    return pack
