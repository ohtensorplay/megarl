"""RDMA weight updates across tensor-parallel ranks."""

from megarl import megarl_rs
from megarl.rdma import (
    RdmaTarget,
    RdmaWeightTransport,
    rebuild_from_shards,
    shard_tensor,
)
from megarl.weights import (
    TensorState,
    WeightSnapshot,
    apply_delta,
    compute_delta,
)


def _snapshots():
    base = WeightSnapshot.from_tensors(
        {"w": TensorState("w", (8,), "bfloat16", bytes(8))}, version=1
    )
    changed = bytes([0, 0, 0, 5, 6, 0, 0, 7])
    current = WeightSnapshot.from_tensors(
        {"w": TensorState("w", (8,), "bfloat16", changed)}, version=2
    )
    return base, current, compute_delta(base, current)


def test_tp_layout_maps_global_ranges_to_ranks():
    layout = megarl_rs.TpLayout("w", [8], 4, 0)
    assert layout.map_range(2, 4) == [
        (1, 2, 0, 2),
        (2, 4, 0, 2),
    ]
    assert len(layout.shards_json()) > 0


def test_rdma_buffer_one_sided_write():
    buffer = megarl_rs.RdmaBuffer(8)
    buffer.write(3, b"abc")
    assert buffer.read() == bytes([0, 0, 0, 97, 98, 99, 0, 0])


def test_tp_aware_inplace_delta_application():
    base, current, delta = _snapshots()
    layout, source_buffers = shard_tensor(base.tensors["w"], 4)
    target_layout, target_buffers = shard_tensor(base.tensors["w"], 4)
    transport = RdmaWeightTransport({"w": layout}, {"w": source_buffers})
    writes = transport.apply(delta)

    # Range [2,6) touches ranks 1 and 2; byte 7 touches rank 3 only.
    assert {(w.rank, w.local_offset) for w in writes} == {(1, 1), (2, 0), (3, 1)}
    for write in writes:
        target_buffers[write.rank].write(write.local_offset, write.data)

    rebuilt = rebuild_from_shards("w", (8,), "bfloat16", target_buffers)
    assert rebuilt.tensors == current.tensors
    assert rebuilt.digest == current.digest
    assert apply_delta(base, delta).digest == current.digest


def test_rdma_transport_send_to_registered_target():
    base, current, delta = _snapshots()
    layout, source_buffers = shard_tensor(base.tensors["w"], 2)
    _, target_buffers = shard_tensor(base.tensors["w"], 2)
    target = RdmaTarget({"w": target_buffers})
    address = target.start()
    try:
        transport = RdmaWeightTransport({"w": layout}, {"w": source_buffers})
        transport.send(delta, address)
        rebuilt = rebuild_from_shards("w", (8,), "bfloat16", target_buffers)
        assert rebuilt.tensors == current.tensors
    finally:
        target.close()
