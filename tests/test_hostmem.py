"""Pinned host-memory pool: exact-size recycling, real page locking, staging."""

import numpy as np
import pytest
import torch

from megarl.async_controller import PayloadQueue
from megarl.contracts import TrajectoryRecord
from megarl.dataplane import PinnedPayloadStore
from megarl.hostmem import HostMemPool, shared_host_mem_pool

_CUDA = torch.cuda.is_available()


def _record(value: int) -> TrajectoryRecord:
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


@pytest.mark.skipif(not _CUDA, reason="requires CUDA/ROCm cudaHostRegister")
def test_exact_size_buffers_are_pinned_and_recycled():
    pool = HostMemPool()
    first = pool.alloc(1024, torch.float32)
    assert first.device.type == "cpu"
    assert first.dtype == torch.float32
    assert first.is_pinned()
    ptr = first.data_ptr()

    assert pool.free(first)
    stats = pool.stats()
    assert stats["host_buffers"] == 1
    assert stats["host_pinned_gb"] > 0
    assert stats["host_pinned_free_gb"] == stats["host_pinned_gb"]

    second = pool.alloc(1024, torch.float32)
    assert second.data_ptr() == ptr  # same registered pages, no re-pin
    assert second.is_pinned()
    assert pool.stats()["host_reuses"] == 1
    assert pool.stats()["host_pinned_free_gb"] == 0

    pool.free(second)
    pool.close()
    assert not second.is_pinned()  # unregistered only at explicit teardown


def test_exact_size_buckets_do_not_share_pages():
    pool = HostMemPool(enable_pinning=False)
    small = pool.alloc(8, torch.float32)
    large = pool.alloc(16, torch.float32)
    assert small.data_ptr() != large.data_ptr()
    pool.free(large)
    pool.free(small)
    assert pool.alloc(8, torch.float32).data_ptr() == small.data_ptr()
    assert pool.alloc(16, torch.float32).data_ptr() == large.data_ptr()
    assert pool.stats()["host_allocations"] == 2
    assert pool.stats()["host_reuses"] == 2


def test_views_recycle_the_owning_base_and_free_is_idempotent():
    pool = HostMemPool(enable_pinning=False)
    base = pool.alloc(8, torch.float32)
    ptr = base.data_ptr()
    view = base[2:4]  # nonzero storage offset: data_ptr() differs from base

    assert pool.free(view)
    assert pool.free(base)  # a late/duplicate free must not corrupt the list
    assert pool.stats()["host_free_buffers"] == 1
    assert pool.alloc(8, torch.float32).data_ptr() == ptr

    foreign = torch.empty(8, dtype=torch.float32)
    assert not pool.free(foreign)
    assert pool.alloc(0).numel() == 0


@pytest.mark.skipif(not _CUDA, reason="requires CUDA/ROCm cudaHostRegister")
def test_close_refuses_outstanding_buffers_and_unregisters_when_idle():
    pool = HostMemPool()
    held = pool.alloc_bytes(64)
    with pytest.raises(RuntimeError, match="buffers in use"):
        pool.close()
    pool.free(held)
    pool.close()
    assert not held.is_pinned()
    with pytest.raises(RuntimeError, match="closed"):
        pool.alloc_bytes(64)


def test_pageable_mode_recycles_without_pinning():
    pool = HostMemPool(enable_pinning=False)
    buf = pool.alloc_bytes(32)
    assert not buf.is_pinned()
    buf.numpy()[:4] = np.frombuffer(b"data", dtype=np.uint8)
    pool.free(buf)
    again = pool.alloc_bytes(32)
    assert again.data_ptr() == buf.data_ptr()
    assert bytes(again.numpy()[:4]) == b"data"
    stats = pool.stats()
    assert stats["host_pinned_gb"] == 0
    assert stats["host_pinned_free_gb"] == 0
    assert stats["host_reuses"] == 1


def test_shared_pool_is_a_singleton():
    assert shared_host_mem_pool() is shared_host_mem_pool()


def test_pinned_payload_store_roundtrip_and_recycle():
    pool = HostMemPool(enable_pinning=False)
    store = PinnedPayloadStore(pool=pool)
    handle = store.put(b"weight-delta")
    assert store.get(handle) == b"weight-delta"
    ptr = store._payloads[handle][0].data_ptr()

    store.write(handle, b"bigger-weight-delta")  # grows into a new bucket
    assert store.get(handle) == b"bigger-weight-delta"
    grown_ptr = store._payloads[handle][0].data_ptr()
    assert grown_ptr != ptr
    store.release(handle)

    reused = store.put(b"x" * 19)  # exactly the grown bucket -> recycled pages
    assert store.get(reused) == b"x" * 19
    assert store._payloads[reused][0].data_ptr() == grown_ptr
    assert store.stats()["host_reuses"] >= 1
    store.close()
    assert store.stats()["handles"] == 0


@pytest.mark.skipif(not _CUDA, reason="verifies real pinning of staged payloads")
def test_payload_store_stages_in_pinned_memory():
    pool = HostMemPool()
    store = PinnedPayloadStore(pool=pool)
    handle = store.put(b"pinned-payload")
    assert store._payloads[handle][0].is_pinned()
    store.release(handle)
    store.close()
    pool.close()


def test_payload_queue_runs_over_pinned_store():
    store = PinnedPayloadStore(pool=HostMemPool(enable_pinning=False))
    queue = PayloadQueue(store=store)
    lease = queue.put(_record(11))
    restored = queue.take(lease)
    assert restored.request_id == 11
    assert restored.logprobs == [-0.2, -0.4]
    assert store.stats()["handles"] == 0
