"""Production weight sync: binary frames, integrity, version handshake."""

import time

import pytest

from megarl.weights import (
    DigestMismatch,
    FileSystemTransport,
    TensorState,
    VersionMismatch,
    WeightSnapshot,
    WeightSyncCoordinator,
    apply_delta,
    compute_delta,
    deserialize_delta,
    serialize_delta,
)


def snapshots(size=256, changes=((10, 9), (200, 7))):
    old = bytearray(size)
    new = bytearray(size)
    for offset, value in changes:
        new[offset] = value
    base = WeightSnapshot.from_tensors(
        {"w": TensorState("w", (size,), "bfloat16", bytes(old))}, version=3
    )
    current = WeightSnapshot.from_tensors(
        {"w": TensorState("w", (size,), "bfloat16", bytes(new))}, version=4
    )
    return base, current, compute_delta(base, current)


def test_binary_frame_roundtrip_and_checksum():
    base, current, delta = snapshots()
    delta = compute_delta(base, current, shard_rank=2)
    frame = serialize_delta(delta)
    assert frame[:4] == b"MGDP"
    assert deserialize_delta(frame) == delta
    corrupted = bytearray(frame)
    corrupted[-1] ^= 0xFF
    with pytest.raises(DigestMismatch):
        deserialize_delta(bytes(corrupted))


def test_filesystem_transport_uses_binary_frames(tmp_path):
    base, current, delta = snapshots()
    transport = FileSystemTransport(tmp_path)
    transport.send(delta, "v4")
    assert (tmp_path / "v4" / "delta.mgdp").exists()
    assert apply_delta(base, transport.receive("v4")).digest == current.digest


def test_coordinator_enforces_monotonic_versions_and_digest():
    base, current, delta = snapshots()
    coord = WeightSyncCoordinator(base)
    assert coord.status()["base_version"] == 3
    with pytest.raises(VersionMismatch):
        coord.publish(compute_delta(current, base, to_version=2))
    coord.publish(delta, frame=serialize_delta(delta))
    assert coord.status()["pending"] == 4
    with pytest.raises(VersionMismatch):
        coord.publish(delta)  # second uncommitted version rejected
    applied = coord.commit()
    assert applied.digest == current.digest
    assert coord.status()["base_version"] == 4
    with pytest.raises(VersionMismatch):
        coord.commit()


def test_coordinator_rejects_a_tampered_payload():
    import dataclasses

    base, current, delta = snapshots()
    coord = WeightSyncCoordinator(base)
    bad = dataclasses.replace(delta, digest="0" * 64)
    with pytest.raises(DigestMismatch):
        coord.publish(bad, frame=serialize_delta(bad))


def test_sparse_diff_on_large_tensor_is_block_indexed():
    size = 8 * 1024 * 1024
    old = bytearray(size)
    new = bytearray(size)
    new[17] = 1
    new[size - 3] = 2
    base = WeightSnapshot.from_tensors({"w": TensorState("w", (size,), "u8", bytes(old))})
    current = WeightSnapshot.from_tensors({"w": TensorState("w", (size,), "u8", bytes(new))})
    start = time.monotonic()
    delta = compute_delta(base, current)
    elapsed = time.monotonic() - start
    assert delta.changed_bytes == 2
    assert elapsed < 2.0, f"block-indexed diff too slow: {elapsed:.2f}s"
    assert apply_delta(base, delta).digest == current.digest
