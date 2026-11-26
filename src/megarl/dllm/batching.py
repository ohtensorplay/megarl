"""dLLM reinforcement learning (d1 arXiv 2505.22765, wd1; see devdocs/dllm-rl.md); see
package docstring for details."""

from __future__ import annotations

import dataclasses
from typing import Sequence

import tensorplay as tp

from ..batching import RLBatch
from ..losses import TensorBatch
from .config import DLLMConfig
from .estimator import DLLMSample, _noise_row
from .trajectory import DLLMTrajectoryStore


@dataclasses.dataclass(frozen=True, slots=True)
class DLLMBatch(RLBatch):
    """AR-compatible batch plus the deterministic noisy canvas per row.

    ``dllm_noisy_ids`` is the exact input fed to the denoiser; it is built once
    at collate time from ``config.seed`` so the policy, behavior and reference
    forwards see *identical* mask realisations. Bulk tensors still travel out
    of band — these are metadata-sized token lists.
    """

    dllm_noisy_ids: list[list[int]] = dataclasses.field(default_factory=list, kw_only=True)
    dllm_prompt_lens: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    dllm_prompt_span: int = dataclasses.field(default=0, kw_only=True)
    dllm_canvas_len: int = dataclasses.field(default=0, kw_only=True)
    dllm_alpha: list[float] = dataclasses.field(default_factory=list, kw_only=True)
    dllm_payload_handles: list[int | None] = dataclasses.field(
        default_factory=list, kw_only=True
    )
    dllm_is_trajectory: bool = dataclasses.field(default=False, kw_only=True)
    dllm_sample_ids: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    """Trajectory replay only: maps each replay row back to its sample."""
    dllm_step_ids: list[int] = dataclasses.field(default_factory=list, kw_only=True)
    """Trajectory replay only: index of the denoising step within its sample."""


class DLLMTensorBatch(TensorBatch):
    """Device tensors for one (micro)batch of :class:`DLLMBatch`."""

    def __init__(self, batch: DLLMBatch, device: str = "cpu") -> None:
        super().__init__(batch, device=device)
        self.dllm_noisy_ids = tp.tensor(
            batch.dllm_noisy_ids, dtype=tp.long, device=device
        )
        self.dllm_prompt_lens = tp.tensor(
            batch.dllm_prompt_lens, dtype=tp.long, device=device
        )
        self.dllm_prompt_span = int(batch.dllm_prompt_span)
        self.dllm_canvas_len = int(batch.dllm_canvas_len)
        self.dllm_alpha = tp.tensor(batch.dllm_alpha, dtype=tp.float32, device=device)
        self.dllm_is_trajectory = bool(batch.dllm_is_trajectory)


def collate_dllm(
    samples: Sequence[DLLMSample], config: DLLMConfig
) -> DLLMBatch:
    """Pad dLLM samples into one trainer-ready batch.

    Block models get a fixed ``[prompt_span, canvas_len]`` layout: prompts are
    right-padded and the completion canvas is sized once for the whole batch,
    matching ``DiffusionGemmaForBlockDiffusion.denoise``.
    """
    if not samples:
        raise ValueError("cannot collate an empty dLLM batch")
    for sample in samples:
        sample.validate()

    max_prompt = max(len(s.prompt_token_ids) for s in samples)
    max_completion = max(len(s.completion_token_ids) for s in samples)
    if config.family == "block" and config.max_completion_length is not None:
        canvas_len = max(max_completion, config.max_completion_length)
    else:
        canvas_len = max_completion
    prompt_span = max_prompt if config.family == "block" else 0
    width = (prompt_span if config.family == "block" else max_prompt) + canvas_len

    input_ids: list[list[int]] = []
    noisy_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    loss_mask: list[list[int]] = []
    position_ids: list[list[int]] = []
    old_logprobs: list[list[float]] = []
    versions: list[list[int]] = []
    prompt_lens: list[int] = []
    seeds: list[int] = []
    alphas: list[float] = []
    handles: list[int | None] = []
    cu_seqlens = [0]

    for index, sample in enumerate(samples):
        ids, noisy, comp_len, alpha = _noise_row(
            sample.prompt_token_ids,
            sample.completion_token_ids,
            config,
            sample.mask_seed
            if sample.mask_seed is not None
            else config.seed + index,
            prompt_span=prompt_span,
            canvas_len=canvas_len,
        )
        pad = width - len(ids)
        prompt_len = len(sample.prompt_token_ids)
        if config.family == "block":
            completion_start = prompt_span
            attention_mask.append(
                [1] * prompt_len
                + [0] * (completion_start - prompt_len)
                + [1] * comp_len
                + [0] * (canvas_len - comp_len)
                + [0] * pad
            )
            loss_mask.append(
                [0] * completion_start
                + [1] * comp_len
                + [0] * (canvas_len - comp_len)
                + [0] * pad
            )
        else:
            completion_start = prompt_len
            attention_mask.append([1] * len(ids) + [0] * pad)
            loss_mask.append(
                [0] * completion_start + [1] * comp_len + [0] * pad
            )
        input_ids.append(ids + [0] * pad)
        noisy_ids.append(noisy + [0] * pad)
        position_ids.append(list(range(len(ids))) + [0] * pad)
        old_logprobs.append([0.0] * width)
        versions.append([sample.version] * len(ids) + [0] * pad)
        prompt_lens.append(prompt_len)
        seeds.append(
            sample.mask_seed if sample.mask_seed is not None else config.seed + index
        )
        alphas.append(alpha)
        handles.append(sample.payload_handle)
        cu_seqlens.append(cu_seqlens[-1] + len(ids))

    return DLLMBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        position_ids=position_ids,
        old_logprobs=old_logprobs,
        versions=versions,
        rewards=[s.reward for s in samples],
        cu_seqlens=cu_seqlens,
        max_len=width,
        sampling_logprobs=[],
        dllm_noisy_ids=noisy_ids,
        dllm_prompt_lens=prompt_lens,
        dllm_prompt_span=prompt_span,
        dllm_canvas_len=canvas_len,
        dllm_alpha=alphas,
        dllm_payload_handles=handles,
    )


def slice_dllm_batch(batch: DLLMBatch, start: int, stop: int) -> DLLMBatch:
    """Row slice used by the trainer's gradient accumulation."""
    return DLLMBatch(
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
        dllm_alpha=batch.dllm_alpha[start:stop],
        dllm_payload_handles=batch.dllm_payload_handles[start:stop],
        dllm_is_trajectory=batch.dllm_is_trajectory,
        dllm_sample_ids=batch.dllm_sample_ids[start:stop],
        dllm_step_ids=batch.dllm_step_ids[start:stop],
    )


def collate_trajectory(
    records: Sequence, config: DLLMConfig
) -> DLLMBatch:
    """Flatten recorded denoising steps into one trainer batch.

    Each ``(sample, step)`` pair becomes one row: ``input_ids`` holds the
    sampled tokens (the replay target), ``dllm_noisy_ids`` holds the exact
    canvas that was fed to the denoiser, and ``old_logprobs`` are the recorded
    behaviour log-probabilities. The loss mask is the recorded accept mask.
    """
    if not records:
        raise ValueError("cannot collate an empty dLLM trajectory batch")
    for record in records:
        record.validate()
    # Replay needs the conditioning prompt in every row, including masked
    # dLLMs: the denoiser predicts the canvas conditioned on the prompt.
    prompt_span = max(len(r.prompt_ids) for r in records)
    canvas_len = max(r.canvas_len for r in records)
    width = prompt_span + canvas_len
    input_ids: list[list[int]] = []
    noisy_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    loss_mask: list[list[int]] = []
    old_logprobs: list[list[float]] = []
    versions: list[list[int]] = []
    alphas: list[float] = []
    rewards: list[float] = []
    sample_ids: list[int] = []
    step_ids: list[int] = []
    cu_seqlens = [0]

    for sample_index, record in enumerate(records):
        for step_index, step in enumerate(record.steps):
            prompt = list(record.prompt_ids)
            padded_prompt = prompt + [config.noise_token_id] * (
                prompt_span - len(prompt)
            )
            sampled = list(step.sampled_ids)
            canvas = list(step.canvas_ids)
            canvas_pad = canvas_len - len(canvas)
            clean_row = padded_prompt + sampled + [0] * canvas_pad
            noisy_row = padded_prompt + canvas + [0] * canvas_pad
            input_ids.append(clean_row)
            noisy_ids.append(noisy_row)
            attention_mask.append(
                [1] * len(prompt)
                + [0] * (prompt_span - len(prompt))
                + [1] * canvas_len
            )
            loss_mask.append(
                [0] * prompt_span
                + step.accept_mask
                + [0] * canvas_pad
            )
            old_logprobs.append(
                [0.0] * prompt_span + step.logprobs + [0.0] * canvas_pad
            )
            versions.append([step.version] * width)
            if config.effective_noise_mode == "uniform":
                alphas.append(1.0 - min(1.0, max(0.0, step.timestep)))
            else:
                alphas.append(1.0 - config.prompt_mask_prob)
            rewards.append(float(record.reward or 0.0))
            sample_ids.append(sample_index)
            step_ids.append(step_index)
            cu_seqlens.append(cu_seqlens[-1] + width)

    return DLLMBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        position_ids=[list(range(width)) for _ in input_ids],
        old_logprobs=old_logprobs,
        versions=versions,
        rewards=rewards,
        cu_seqlens=cu_seqlens,
        max_len=width,
        sampling_logprobs=[],
        dllm_noisy_ids=noisy_ids,
        dllm_prompt_lens=[len(r.prompt_ids) for r in records for _ in r.steps],
        dllm_prompt_span=prompt_span,
        dllm_canvas_len=canvas_len,
        dllm_alpha=alphas,
        dllm_payload_handles=[],
        dllm_is_trajectory=True,
        dllm_sample_ids=sample_ids,
        dllm_step_ids=step_ids,
    )


def make_dllm_packer(config: DLLMConfig):
    """Return a ``batch_packer`` with the controller's calling convention.

    dLLM canvases are fixed-width per batch, so length-budget packing does not
    apply: one scored group becomes one batch and the identity permutation
    keeps advantages aligned with rows.
    """

    def pack(samples, *, max_tokens=None, alignment=1):
        return [collate_dllm(samples, config)], [list(range(len(samples)))]

    return pack


def make_dllm_trajectory_packer(config: DLLMConfig, store: DLLMTrajectoryStore):
    """Resolve out-of-band handles and flatten steps into one replay batch.

    The returned permutation repeats each sample's advantage once per recorded
    denoising step, so the controller's existing advantage/order machinery
    assigns the right outcome advantage to every replay row.
    """

    def pack(samples, *, max_tokens=None, alignment=1):
        records = []
        order: list[int] = []
        for index, sample in enumerate(samples):
            handle = getattr(sample, "payload_handle", None)
            if handle is None:
                raise ValueError("trajectory packer needs payload handles on samples")
            record = store.get(handle)
            store.release(handle)  # consumed exactly once by the packer
            record = dataclasses.replace(record, reward=sample.reward)
            records.append(record)
            order.extend([index] * len(record.steps))
        return [collate_trajectory(records, config)], [order]

    return pack
