"""Sparse delta transports: shared memory and filesystem."""

import uuid

from megarl.weights import (
    FileSystemTransport,
    SharedMemoryTransport,
    TensorState,
    WeightSnapshot,
    apply_delta,
    compute_delta,
)


def _delta():
    base = WeightSnapshot.from_tensors(
        {"w": TensorState("w", (64,), "bfloat16", bytes(64))}, version=1
    )
    current = WeightSnapshot.from_tensors(
        {
            "w": TensorState(
                "w", (64,), "bfloat16", bytes([0] * 10 + [9] + [0] * 53)
            )
        },
        version=2,
    )
    return base, current, compute_delta(base, current)


def test_shared_memory_transport():
    base, current, delta = _delta()
    name = f"megarl_test_{uuid.uuid4().hex[:12]}"
    transport = SharedMemoryTransport()
    transport.send(delta, name)
    received = transport.receive(name)
    assert received == delta
    assert apply_delta(base, received).digest == current.digest


def test_filesystem_transport(tmp_path):
    base, current, delta = _delta()
    transport = FileSystemTransport(tmp_path)
    transport.send(delta, "v2")
    assert apply_delta(base, transport.receive("v2")).digest == current.digest
