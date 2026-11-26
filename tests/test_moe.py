"""Expert-parallel conversion is byte-exact and shard-local."""

from megarl.moe import (
    expert_delta,
    merge_expert_tensors,
    plan_expert_shards,
    split_expert_tensor,
)
from megarl.weights import TensorState, WeightSnapshot, apply_delta


def _snapshots():
    experts = 8
    hidden = 4
    base_data = bytes(range(experts * hidden))
    changed = bytearray(base_data)
    changed[9] = 200
    changed[30] = 201
    base = WeightSnapshot.from_tensors(
        {"experts": TensorState("experts", (experts, hidden), "bfloat16", base_data)}
    )
    current = WeightSnapshot.from_tensors(
        {
            "experts": TensorState(
                "experts", (experts, hidden), "bfloat16", bytes(changed)
            )
        }
    )
    return base, current


def test_expert_shards_are_contiguous_and_complete():
    shards = plan_expert_shards((8, 4), 8, 4, "bfloat16")
    assert [(s.expert_start, s.expert_end) for s in shards] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
    ]
    assert all(s.shape == (2, 4) for s in shards)


def test_split_merge_roundtrip_is_byte_exact():
    base, current = _snapshots()
    merged = merge_expert_tensors(
        split_expert_tensor(current.tensors["experts"], 8, 4), 8, "experts"
    )
    assert merged == current.tensors["experts"]


def test_shard_local_expert_delta_updates_only_affected_ranks():
    base, current = _snapshots()
    deltas = expert_delta(base, current, num_experts=8, ep_size=4, name="experts")
    assert set(deltas) == {1, 3}  # byte 9 -> expert 2 (rank 1), byte 30 -> expert 7 (rank 3)

    old_shards = split_expert_tensor(base.tensors["experts"], 8, 4)
    updated = {}
    for rank, delta in deltas.items():
        old = WeightSnapshot.from_tensors({"w": old_shards[rank]})
        new = apply_delta(old, delta)
        updated[rank] = new.tensors["w"]
        assert new.digest == WeightSnapshot.from_tensors(
            {"w": split_expert_tensor(current.tensors["experts"], 8, 4)[rank]}
        ).digest
    for rank in set(deltas) ^ {0, 1, 2, 3}:
        updated[rank] = old_shards[rank]
    assert merge_expert_tensors(updated, 8, "experts") == current.tensors["experts"]
