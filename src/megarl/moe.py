"""Mixture-of-experts conversion and expert-parallel placement.

A fused expert tensor (experts stacked on dim 0) is split into contiguous
expert groups, one group per EP rank. Conversion is byte-exact and shard-local:
no rank ever assembles the full fused tensor during a weight update.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

from .weights import TensorState, WeightSnapshot, compute_delta


@dataclasses.dataclass(frozen=True, slots=True)
class ExpertShard:
    rank: int
    expert_start: int
    expert_end: int
    dtype: str
    shape: tuple[int, ...]

    @property
    def experts(self) -> int:
        return self.expert_end - self.expert_start


def plan_expert_shards(
    shape: Sequence[int], num_experts: int, ep_size: int, dtype: str
) -> list[ExpertShard]:
    if num_experts <= 0 or ep_size <= 0:
        raise ValueError("num_experts and ep_size must be positive")
    if ep_size > num_experts or num_experts % ep_size:
        raise ValueError("ep_size must divide num_experts")
    if not shape or shape[0] != num_experts:
        raise ValueError("first dimension must equal num_experts")
    per_rank = num_experts // ep_size
    return [
        ExpertShard(
            rank=rank,
            expert_start=rank * per_rank,
            expert_end=(rank + 1) * per_rank,
            dtype=dtype,
            shape=(per_rank,) + tuple(shape[1:]),
        )
        for rank in range(ep_size)
    ]


def split_expert_tensor(
    state: TensorState, num_experts: int, ep_size: int
) -> dict[int, TensorState]:
    shards = plan_expert_shards(state.shape, num_experts, ep_size, state.dtype)
    if len(state.data) % num_experts:
        raise ValueError("expert tensor bytes must divide evenly by expert count")
    span = len(state.data) // num_experts
    return {
        shard.rank: TensorState(
            name=f"{state.name}.ep{shard.rank}",
            shape=shard.shape,
            dtype=shard.dtype,
            data=b"".join(
                state.data[e * span : (e + 1) * span]
                for e in range(shard.expert_start, shard.expert_end)
            ),
        )
        for shard in shards
    }


def merge_expert_tensors(
    shards: dict[int, TensorState], num_experts: int, name: str
) -> TensorState:
    ordered = [shards[rank] for rank in sorted(shards)]
    if sum(len(shard.data) for shard in ordered) == 0:
        raise ValueError("no expert shards to merge")
    data = b"".join(shard.data for shard in ordered)
    return TensorState(
        name=name,
        shape=(num_experts,) + ordered[0].shape[1:],
        dtype=ordered[0].dtype,
        data=data,
    )


def expert_delta(
    previous: WeightSnapshot,
    current: WeightSnapshot,
    *,
    num_experts: int,
    ep_size: int,
    name: str,
) -> dict[int, object]:
    """Compute one shard-local delta per EP rank, never staging the fused tensor."""
    old_shards = split_expert_tensor(previous.tensors[name], num_experts, ep_size)
    new_shards = split_expert_tensor(current.tensors[name], num_experts, ep_size)
    deltas: dict[int, object] = {}
    for rank in sorted(old_shards):
        old_snapshot = WeightSnapshot.from_tensors({"w": old_shards[rank]})
        new_snapshot = WeightSnapshot.from_tensors({"w": new_shards[rank]})
        rank_delta = compute_delta(old_snapshot, new_snapshot)
        if rank_delta.changes:  # only ranks with actual byte changes
            deltas[rank] = rank_delta
    return deltas
