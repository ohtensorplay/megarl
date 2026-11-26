"""Sparse weight deltas, digest verification and transport roundtrip."""

import pytest

from megarl import contracts
from megarl.weights import (
    DigestMismatch,
    FileSystemTransport,
    TensorState,
    WeightSnapshot,
    apply_delta,
    compute_delta,
)


def tensors(changed=False):
    a = bytearray(64)
    b = bytearray([1] * 64)
    if changed:
        a[10] = 9
        a[40] = 7
        b[30] = 3
    return {
        "layer0": TensorState("layer0", (64,), "bfloat16", bytes(a)),
        "layer1": TensorState("layer1", (64,), "bfloat16", bytes(b)),
    }


def test_delta_is_sparse_and_exact():
    previous = WeightSnapshot.from_tensors(tensors(), version=3)
    current = WeightSnapshot.from_tensors(tensors(changed=True), version=4)
    delta = compute_delta(previous, current)
    assert delta.changed_bytes == 3
    assert len(delta.changes) == 2  # only the two tensors that changed
    applied = apply_delta(previous, delta)
    assert applied.digest == current.digest
    assert applied.version == 4
    assert applied.tensors == current.tensors


def test_digest_mismatch_is_detected():
    previous = WeightSnapshot.from_tensors(tensors(), version=1)
    current = WeightSnapshot.from_tensors(tensors(changed=True), version=2)
    delta = compute_delta(previous, current)
    import dataclasses

    bad = dataclasses.replace(delta, digest="0" * 64)
    with pytest.raises(DigestMismatch):
        apply_delta(previous, bad)


def test_contract_descriptors_and_filesystem_transport(tmp_path):
    previous = WeightSnapshot.from_tensors(tensors(), version=1)
    current = WeightSnapshot.from_tensors(tensors(changed=True), version=2)
    delta = compute_delta(previous, current)

    message = delta.to_contract(transport="fs", abort_inflight=True)
    contracts.validate(message)  # drift check against Rust struct
    assert len(message.tensors) == delta.changed_bytes

    transport = FileSystemTransport(tmp_path)
    transport.send(delta, "v2")
    received = transport.receive("v2")
    assert received == delta
    assert apply_delta(previous, received).digest == current.digest
