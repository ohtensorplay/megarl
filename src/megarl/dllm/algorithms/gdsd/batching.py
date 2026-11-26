"""GDSD, Xiaohang Tang et al., arXiv 2605.29398; code https://github.com/GaryBall/GDSD;
see package docstring for details."""

from __future__ import annotations

import dataclasses
import random
from typing import Sequence

import tensorplay as tp

from ....batching import RLBatch
from ....dllm import DLLMSample
from ....losses import TensorBatch
from .config import GDSDConfig


class GDSDTensorBatch(TensorBatch):
    """Device tensors for one :class:`GDSDBatch` slice."""

    def __init__(self, batch: GDSDBatch, device: str = "cpu") -> None:
        super().__init__(batch, device=device)
        self.dllm_noisy_ids = tp.tensor(
            batch.dllm_noisy_ids, dtype=tp.long, device=device
        )
        self.dllm_prompt_lens = tp.tensor(
            batch.dllm_prompt_lens, dtype=tp.long, device=device
        )
        self.dllm_prompt_span = int(batch.dllm_prompt_span)
        self.dllm_canvas_len = int(batch.dllm_canvas_len)
        self.gdsd_weights = tp.tensor(
            batch.gdsd_weights, dtype=tp.float32, device=device
        )
        self.gdsd_lengths = tp.tensor(
            batch.gdsd_lengths, dtype=tp.float32, device=device
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GDSDBatch(RLBatch):
    """Rows are ``(sample, timestep, view)`` corruption views."""

    dllm_noisy_ids: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    dllm_prompt_lens: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    dllm_prompt_span: int = dataclasses.field(default=0, kw_only=True)
    dllm_canvas_len: int = dataclasses.field(default=0, kw_only=True)
    gdsd_weights: list[float] = dataclasses.field(default_factory=list, kw_only=True)
    gdsd_lengths: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    gdsd_sample_ids: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    gdsd_timesteps: list[float] = dataclasses.field(default_factory=list, kw_only=True)


def _corrupt_row(
    prompt: list[int],
    completion: list[int],
    config: GDSDConfig,
    rng: random.Random,
    *,
    prompt_span: int,
    mask,
) -> tuple[list[int], list[int], float]:
    """Build one canonical row plus the noisy canvas of a corruption view.

    ``mask[v]`` marks completion positions corrupted in view ``v``; view 1 is
    the exact complement of view 0 so the pair is a coupled estimator.
    """
    uniform = config.family == "block"
    padded_prompt = (
        prompt + [config.noise_token_id] * (prompt_span - len(prompt))
        if uniform
        else list(prompt)
    )
    noisy_prompt = [
        config.noise_token_id if rng.random() < config.prompt_mask_prob else token
        for token in padded_prompt
    ]
    ids = padded_prompt + list(completion)
    noisy_canvas: list[int] = []
    for pos, token in enumerate(completion):
        if mask[pos]:
            if uniform:
                assert config.vocab_size is not None
                noisy_canvas.append(rng.randrange(config.vocab_size))
            else:
                noisy_canvas.append(config.noise_token_id)
        else:
            noisy_canvas.append(token)
    return ids, noisy_prompt + noisy_canvas, float(sum(mask))


def collate_gdsd(
    samples: Sequence[DLLMSample], config: GDSDConfig
) -> GDSDBatch:
    """Expand every completion into K times x 2 coupled corruption views."""
    if not samples:
        raise ValueError("cannot collate an empty GDSD batch")
    for sample in samples:
        sample.validate()
    L = config.max_completion_length
    for sample in samples:
        if len(sample.completion_token_ids) != L:
            raise ValueError(
                f"GDSD completion length {len(sample.completion_token_ids)} != {L}"
            )
    prompt_span = (
        max(len(s.prompt_token_ids) for s in samples)
        if config.family == "block"
        else 0
    )
    width = (prompt_span if config.family == "block" else max(len(s.prompt_token_ids) for s in samples)) + L
    eps = config.corruption_eps
    rng = random.Random(config.seed)

    input_ids: list[list[int]] = []
    noisy_ids: list[list[int]] = []
    attention: list[list[int]] = []
    loss_mask: list[list[int]] = []
    positions: list[list[int]] = []
    old_logprobs: list[list[float]] = []
    versions: list[list[int]] = []
    weights: list[float] = []
    lengths: list[int] = []
    sample_ids: list[int] = []
    timesteps: list[float] = []
    cu_seqlens = [0]

    for index, sample in enumerate(samples):
        prompt = list(sample.prompt_token_ids)
        completion = list(sample.completion_token_ids)
        for k in range(config.mc_timesteps):
            t = eps + (1.0 - 2.0 * eps) * rng.random()
            first = [rng.random() < t for _ in range(L)]
            # Both views must actually predict something: a view with no masked
            # positions contributes a zero (and useless) estimator sample.
            if not any(first):
                first[rng.randrange(L)] = True
            if all(first):
                first[rng.randrange(L)] = False
            views = [first]
            if config.coupled_views:
                views.append([not flag for flag in first])
            for view_index, flags in enumerate(views):
                prob = t if view_index == 0 else 1.0 - t
                prob = min(1.0 - eps, max(eps, prob))
                ids, noisy, _ = _corrupt_row(
                    prompt, completion, config, rng,
                    prompt_span=prompt_span, mask=flags,
                )
                offset = prompt_span if config.family == "block" else len(prompt)
                row_loss = [0] * width
                for pos, flag in enumerate(flags):
                    row_loss[offset + pos] = int(flag)
                input_ids.append(ids + [0] * (width - len(ids)))
                noisy_ids.append(noisy + [0] * (width - len(noisy)))
                attention.append(
                    [1] * len(prompt)
                    + [0] * (prompt_span - len(prompt))
                    + [1] * L
                    + [0] * (width - prompt_span - L)
                    if config.family == "block"
                    else [1] * (len(prompt) + L) + [0] * (width - len(prompt) - L)
                )
                loss_mask.append(row_loss)
                positions.append(list(range(width)))
                old_logprobs.append([0.0] * width)
                versions.append([sample.version] * width)
                weights.append(1.0 / prob)
                lengths.append(L)
                sample_ids.append(index)
                timesteps.append(t)
                cu_seqlens.append(cu_seqlens[-1] + width)

    return GDSDBatch(
        input_ids=input_ids,
        attention_mask=attention,
        loss_mask=loss_mask,
        position_ids=positions,
        old_logprobs=old_logprobs,
        versions=versions,
        rewards=[float(s.reward) for s in samples for _ in range(config.rows_per_sample)],
        cu_seqlens=cu_seqlens,
        max_len=width,
        sampling_logprobs=[],
        dllm_noisy_ids=noisy_ids,
        dllm_prompt_lens=[len(s.prompt_token_ids) for s in samples for _ in range(config.rows_per_sample)],
        dllm_prompt_span=prompt_span,
        dllm_canvas_len=L,
        gdsd_weights=weights,
        gdsd_lengths=lengths,
        gdsd_sample_ids=sample_ids,
        gdsd_timesteps=timesteps,
    )


def slice_gdsd_batch(batch: GDSDBatch, start: int, stop: int) -> GDSDBatch:
    """Row slice used by the trainer's gradient accumulation."""
    return GDSDBatch(
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
        gdsd_weights=batch.gdsd_weights[start:stop],
        gdsd_lengths=batch.gdsd_lengths[start:stop],
        gdsd_sample_ids=batch.gdsd_sample_ids[start:stop],
        gdsd_timesteps=batch.gdsd_timesteps[start:stop],
    )


def make_gdsd_packer(config: GDSDConfig):
    """Return a ``batch_packer`` aligned with the controller's row order."""

    def pack(samples, *, max_tokens=None, alignment=1):
        batch = collate_gdsd(samples, config)
        return [batch], [list(batch.gdsd_sample_ids)]

    return pack
