"""Recycled pinned host-memory pool for offload and weight-transfer staging.

Pinned host buffers are expensive to create: ``cudaHostRegister`` page-locks a
range, and doing it on every offload/weight-sync step dominates the transfer
time. This pool mirrors the production design of the pinned ROLL reference
(``roll/distributed/store/host_mem_pool.py`` at ``601c2442``): exact-size
``(numel, dtype)`` free lists, because offload cycles re-put the same tensor
sizes every step and exact buckets avoid wasting locked pages; strong references
to the registered base tensors by storage identity, because a garbage-collected
registered range is undefined behaviour and a view's pointer must not be
mistaken for a new allocation.

Differences from the reference, all preserving the same contract:

* storage-pointer identity (``untyped_storage().data_ptr()``) instead of
  ``data_ptr()``, so FSDP-style views with nonzero storage offsets recycle the
  owning base buffer;
* an ``RLock`` so a controller thread and an offload thread may share a pool;
* explicit ``close()`` to unregister at process teardown (ROLL deliberately
  keeps the high-water mark resident for the worker's lifetime);
* PyTorch is imported lazily, so importing the data plane does not require it.

Callers that need a byte payload use :meth:`HostMemPool.alloc_bytes`, which
returns a flat ``uint8`` CPU tensor suitable for direct H2D/RDMA writes.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    import torch
    from torch import Tensor

Bucket = Tuple[int, "torch.dtype"]
_GIB = 1 << 30


def _torch():
    """Import PyTorch only when a pinned buffer is actually requested."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "HostMemPool needs PyTorch for pinned host buffers; "
            "install the 'pinned' extra"
        ) from exc
    return torch


def _cuda_register_supported() -> bool:
    """CUDA and ROCm builds both expose ``torch.cuda.is_available()``."""
    try:
        return bool(_torch().cuda.is_available())
    except Exception:
        return False


class HostMemPool:
    """Exact-size pinned CPU buffers, recycled across offload/transfer cycles.

    ``enable_pinning=None`` (default) registers with ``cudaHostRegister`` on
    CUDA/ROCm, falls back to the platform pinned allocator (e.g. NPU) and
    finally to pageable memory. ``True`` requires successful pinning; ``False``
    disables it, which keeps the recycling semantics testable without hardware.
    Registration failure is fatal on CUDA/ROCm: silently falling back would
    turn a DMA into a pageable copy and hide the real error.
    """

    def __init__(self, *, enable_pinning: Optional[bool] = None) -> None:
        self._owned: Dict[int, "Tensor"] = {}
        self._free: Dict[Bucket, List["Tensor"]] = {}
        self._free_ptrs: Set[int] = set()
        self._lock = threading.RLock()
        self._enable_pinning = enable_pinning
        self._use_cuda_register = (
            enable_pinning is not False and _cuda_register_supported()
        )
        self.pinned_bytes = 0
        self.pinned_free_bytes = 0
        self.allocations = 0
        self.reuses = 0
        self._closed = False

    @staticmethod
    def _storage_ptr(tensor: "Tensor") -> int:
        return tensor.untyped_storage().data_ptr()

    def _register(self, buf: "Tensor") -> None:
        nbytes = buf.numel() * buf.element_size()
        error = _torch().cuda.cudart().cudaHostRegister(buf.data_ptr(), nbytes, 0)
        if int(error) != 0:
            raise RuntimeError(
                f"cudaHostRegister failed (cudaError {int(error)}) "
                f"for {nbytes / _GIB:.2f}GB"
            )

    def _pin(self, buf: "Tensor") -> "Tensor":
        """Pin a freshly allocated flat CPU tensor using the best backend."""
        if self._enable_pinning is False:
            return buf
        if self._use_cuda_register:
            self._register(buf)
            return buf
        try:
            pinned = buf.pin_memory()  # platform allocator (NPU / other accelerators)
        except RuntimeError:
            if self._enable_pinning is True:
                raise
            return buf
        if self._enable_pinning is True and not pinned.is_pinned():
            raise RuntimeError("pinning required but no pinned allocator is available")
        return pinned

    def alloc(self, numel: int, dtype: Optional["torch.dtype"] = None) -> "Tensor":
        """Return a flat CPU tensor of exactly ``numel`` elements.

        A free buffer of the same ``(numel, dtype)`` is handed out first, with
        its data pointer unchanged, so repeated cycles never call
        ``cudaHostRegister`` twice for the same size class.
        """
        if numel < 0:
            raise ValueError("numel must be non-negative")
        torch = _torch()
        dtype = dtype if dtype is not None else torch.uint8
        bucket = (numel, dtype)
        with self._lock:
            if self._closed:
                raise RuntimeError("host memory pool is closed")
            free = self._free.get(bucket)
            if free:
                buf = free.pop()
                if not free:
                    del self._free[bucket]
                self._free_ptrs.discard(self._storage_ptr(buf))
                if buf.is_pinned():
                    self.pinned_free_bytes -= buf.numel() * buf.element_size()
                self.reuses += 1
                return buf

            buf = torch.empty(numel, dtype=dtype, device="cpu")
            if numel:
                buf = self._pin(buf)
                self._owned[self._storage_ptr(buf)] = buf
                if buf.is_pinned():
                    self.pinned_bytes += buf.numel() * buf.element_size()
            self.allocations += 1
            return buf

    def alloc_bytes(self, nbytes: int) -> "Tensor":
        """Flat ``uint8`` staging buffer for byte payloads and deltas."""
        return self.alloc(nbytes)

    def free(self, buf: "Tensor") -> bool:
        """Recycle a pool-owned buffer; views resolve to their owning base.

        Foreign tensors are a no-op (``False``) and a repeated free is ignored,
        so a stale reference or an exception path cannot corrupt the free list.
        """
        with self._lock:
            ptr = self._storage_ptr(buf)
            base = self._owned.get(ptr)
            if base is None:
                return False
            if ptr in self._free_ptrs:
                return True
            if buf.is_pinned():
                self.pinned_free_bytes += base.numel() * base.element_size()
            self._free.setdefault((base.numel(), base.dtype), []).append(base)
            self._free_ptrs.add(ptr)
            return True

    def stats(self) -> dict:
        """Pool telemetry, keyed like ROLL's ``HostMemPool.stats``."""
        with self._lock:
            return {
                "host_pinned_gb": self.pinned_bytes / _GIB,
                "host_pinned_free_gb": self.pinned_free_bytes / _GIB,
                "host_buffers": len(self._owned),
                "host_free_buffers": sum(len(v) for v in self._free.values()),
                "host_allocations": self.allocations,
                "host_reuses": self.reuses,
            }

    def close(self) -> None:
        """Unregister and drop idle buffers at teardown.

        Outstanding (non-recycled) buffers are refused: unregistering memory
        still in use by an async copy would be undefined behaviour.
        """
        with self._lock:
            outstanding = len(self._owned) - len(self._free_ptrs)
            if outstanding:
                raise RuntimeError(f"cannot close host pool with {outstanding} buffers in use")
            if self._closed:
                return
            if self._use_cuda_register:
                cudart = _torch().cuda.cudart()
                for base in self._owned.values():
                    if base.is_pinned():
                        error = cudart.cudaHostUnregister(base.data_ptr())
                        if int(error) != 0:
                            raise RuntimeError(f"cudaHostUnregister failed: {int(error)}")
            self._owned.clear()
            self._free.clear()
            self._free_ptrs.clear()
            self.pinned_bytes = 0
            self.pinned_free_bytes = 0
            self._closed = True

    def __enter__(self) -> "HostMemPool":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


_pool: Optional[HostMemPool] = None
_pool_lock = threading.Lock()


def shared_host_mem_pool() -> HostMemPool:
    """Process-wide pool: one high-water mark per worker, as in ROLL."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = HostMemPool()
        return _pool


__all__ = ["HostMemPool", "shared_host_mem_pool"]
