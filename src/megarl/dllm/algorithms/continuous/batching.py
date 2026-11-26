"""Flow-GRPO arXiv 2505.05470 and DiffNFT (ROLL diffusion pipeline
refs/ROLL/roll/pipeline/diffusion); see package docstring for details."""

from __future__ import annotations

import dataclasses
from typing import Sequence

import tensorplay as tp

from ....controller import Controller
from .config import FlowConfig
from .types import FlowTrajectoryRecord, FlowTrajectoryStore


@dataclasses.dataclass(frozen=True, slots=True)
class FlowBatch:
    """One collated group of continuous trajectories."""

    latents: list[list[list[float]]]
    timesteps: list[list[float]]
    old_logprobs: list[list[float]]
    loss_mask: list[list[int]]
    rewards: list[float]
    versions: list[int]

    @property
    def size(self) -> int:
        return len(self.latents)

    @property
    def token_count(self) -> int:
        """Step count used by the controller's telemetry/throughput accounting."""
        return sum(sum(row) for row in self.loss_mask)

    def validate(self) -> None:
        if not self.latents:
            raise ValueError("cannot train on an empty flow batch")
        steps = len(self.steps)
        for name in ("timesteps", "old_logprobs", "loss_mask", "rewards", "versions"):
            if len(getattr(self, name)) != self.size:
                raise ValueError(f"flow batch {name} is not row-aligned")
        for index, row in enumerate(self.latents):
            if len(row) != len(self.timesteps[index]) or len(row) < 2:
                raise ValueError("flow latents/timesteps must be [K+1] per row")
            if len(row) - 1 != steps:
                raise ValueError("flow rows have inconsistent step counts")
            if len(self.old_logprobs[index]) != steps:
                raise ValueError("flow logprobs must be [K] per row")

    @property
    def steps(self) -> int:
        return len(self.latents[0]) - 1 if self.latents else 0

    def slice(self, start: int, stop: int) -> "FlowBatch":
        return FlowBatch(
            latents=self.latents[start:stop],
            timesteps=self.timesteps[start:stop],
            old_logprobs=self.old_logprobs[start:stop],
            loss_mask=self.loss_mask[start:stop],
            rewards=self.rewards[start:stop],
            versions=self.versions[start:stop],
        )


class FlowTensorBatch:
    """Device tensors for one (micro)batch of :class:`FlowBatch`."""

    def __init__(self, batch: FlowBatch, device: str = "cpu") -> None:
        self.latents = tp.tensor(batch.latents, dtype=tp.float32, device=device)
        self.timesteps = tp.tensor(batch.timesteps, dtype=tp.float32, device=device)
        self.old_logprobs = tp.tensor(batch.old_logprobs, dtype=tp.float32, device=device)
        self.loss_mask = tp.tensor(batch.loss_mask, dtype=tp.float32, device=device)
        self.versions = tp.tensor(batch.versions, dtype=tp.long, device=device)
        self.rewards = tp.tensor(batch.rewards, dtype=tp.float32, device=device)
        self.size = batch.size


def collate_flow(records: Sequence[FlowTrajectoryRecord], config: FlowConfig) -> FlowBatch:
    """Pad recorded trajectories into one trainer-ready batch."""
    if not records:
        raise ValueError("cannot collate an empty flow batch")
    for record in records:
        record.validate()
    steps = min(len(r.steps) for r in records)
    if config.num_train_steps is not None:
        steps = min(steps, config.num_train_steps)
    width = records[0].latent_dim
    if any(r.latent_dim != width for r in records):
        raise ValueError("flow trajectories in one batch need the same latent width")
    return FlowBatch(
        latents=[list(map(list, r.latents[: steps + 1])) for r in records],
        timesteps=[
            [s.timestep for s in r.steps[:steps]]
            + [
                r.steps[steps].next_timestep
                if steps < len(r.steps)
                else r.steps[-1].next_timestep
            ]
            for r in records
        ],
        old_logprobs=[[s.log_prob for s in r.steps[:steps]] for r in records],
        loss_mask=[[s.keep for s in r.steps[:steps]] for r in records],
        rewards=[float(r.reward or 0.0) for r in records],
        versions=[0] * len(records),
    )


def make_flow_packer(config: FlowConfig, store: FlowTrajectoryStore):
    """Controller-compatible packer that resolves out-of-band handles once."""

    def pack(samples, *, max_tokens=None, alignment=1):
        records = []
        order: list[int] = []
        for index, sample in enumerate(samples):
            handle = getattr(sample, "payload_handle", None)
            if handle is None:
                raise ValueError("flow packer needs payload handles on samples")
            record = store.get(handle)
            store.release(handle)
            record = dataclasses.replace(
                record,
                reward=sample.reward if sample.reward is not None else record.reward,
            )
            records.append(record)
            order.append(index)
        return [collate_flow(records, config)], [order]

    return pack


def _reshape_per_sample(value, like):
    """Broadcast ``[B]`` or ``[B, K]`` scalars over the latent dims of ``like``."""
    if len(value.shape) == 1:
        return value.reshape((value.shape[0],) + (1,) * (len(like.shape) - 1))
    if len(value.shape) == 2:
        return value.reshape(
            (value.shape[0], value.shape[1]) + (1,) * max(0, len(like.shape) - 2)
        )
    return value
