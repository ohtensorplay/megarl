"""Out-of-band payload store: real handles, cross-process reads, checksums."""

import multiprocessing
import uuid

import pytest

from megarl.async_controller import PayloadQueue
from megarl.contracts import TrajectoryRecord
from megarl.dataplane import InProcessPayloadStore, SharedMemoryPayloadStore


def record(value: int) -> TrajectoryRecord:
    return TrajectoryRecord(
        request_id=value,
        prompt_len=2,
        token_ids=[3, 4],
        logprobs=[-0.2, -0.4],
        top_logprobs=[],
        policy_versions=[1, 1],
        reward=1.0,
        finish_reason=0,
        payload_handle=None,
    )


def _reader(name: str, handle: int, output) -> None:
    store = SharedMemoryPayloadStore(name, create=False, capacity=1 << 20)
    output.put(store.get(handle))
    store.close()


def test_in_process_store_and_leased_queue():
    queue = PayloadQueue()
    lease = queue.put(record(7))
    assert len(queue) == 1
    restored = queue.take(lease)
    assert restored.request_id == 7
    assert restored.logprobs == [-0.2, -0.4]
    assert restored.payload_handle is not None
    assert len(queue) == 0


def test_zero_ttl_leases_are_reclaimed():
    queue = PayloadQueue(ttl_secs=0, store=InProcessPayloadStore())
    lease = queue.put(record(9))
    reclaimed = queue.reclaim()
    assert [r.request_id for r in reclaimed] == [9]
    assert len(queue) == 0
    assert queue.take(lease) is None


def test_shared_memory_payload_crosses_process_boundary():
    name = f"megarl_dp_{uuid.uuid4().hex[:10]}"
    store = SharedMemoryPayloadStore(name, capacity=1 << 20)
    try:
        handle = store.put(b"payload-from-writer")
        output = multiprocessing.Queue()
        proc = multiprocessing.Process(target=_reader, args=(name, handle, output))
        proc.start()
        received = output.get(timeout=10)
        proc.join(timeout=10)
        assert proc.exitcode == 0
        assert received == b"payload-from-writer"
        store.release(handle)
        with pytest.raises(KeyError):
            store.get(handle)
    finally:
        store.close()
        store.unlink()


def test_shared_memory_detects_payload_corruption():
    name = f"megarl_dp_{uuid.uuid4().hex[:10]}"
    store = SharedMemoryPayloadStore(name, capacity=1 << 20)
    try:
        handle = store.put(b"integrity-checked")
        # Flip a byte directly in the bound segment, bypassing the API.
        flags, length, offset, _ = store._slot(handle)
        store._buf()[offset] ^= 0xFF
        with pytest.raises(ValueError, match="checksum"):
            store.get(handle)
    finally:
        store.close()
        store.unlink()
