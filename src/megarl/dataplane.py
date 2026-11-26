"""Out-of-band data plane: payload bytes never pass through the controller.

A frame carries a small integer handle. The bulk payload lives in a bound POSIX
shared-memory segment, with a shared slot table, checksums and a cross-process
file lock so a worker and a controller in different processes can safely
exchange handles. The controller only ever sees metadata and leases.
"""

from __future__ import annotations

import dataclasses
import hashlib
import struct
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from .control import LeaseTable
from .hostmem import HostMemPool, shared_host_mem_pool

MAGIC = b"MGRLSHM\x01"
_SLOT = struct.Struct("!BQQQ")  # used, length, offset, checksum64
_HEADER = 16
_MAX_SLOTS = 4096
_HEADER_BYTES = _HEADER + _MAX_SLOTS * _SLOT.size


class PayloadStore(Protocol):
    def put(self, payload: bytes) -> int: ...

    def get(self, handle: int) -> bytes: ...

    def write(self, handle: int, payload: bytes) -> None: ...

    def release(self, handle: int) -> None: ...

    def reclaim(self) -> list[int]: ...


@dataclasses.dataclass(frozen=True, slots=True)
class PayloadStoreStats:
    capacity: int
    used: int
    handles: int


class InProcessPayloadStore:
    """Deterministic store for single-process tests and embedded fast paths."""

    def __init__(self) -> None:
        self._payload: dict[int, bytes] = {}
        self._next = 1

    def put(self, payload: bytes) -> int:
        handle = self._next
        self._next += 1
        self._payload[handle] = payload
        return handle

    def get(self, handle: int) -> bytes:
        return self._payload[handle]

    def write(self, handle: int, payload: bytes) -> None:
        self._payload[handle] = payload

    def release(self, handle: int) -> None:
        self._payload.pop(handle, None)

    def reclaim(self) -> list[int]:
        return []


class PinnedPayloadStore:
    """Payload store that stages bytes in recycled pinned host memory.

    This is the in-process counterpart of ``SharedMemoryPayloadStore``: each
    handle owns a flat ``uint8`` buffer from a :class:`HostMemPool`, so a
    repeated weight-sync snapshot of the same size writes into pages that are
    already page-locked (DMA-stable and free of page faults). Releasing a
    handle recycles the buffer; it never unpins it mid-run.
    """

    def __init__(self, *, pool: HostMemPool | None = None) -> None:
        self.pool = pool if pool is not None else shared_host_mem_pool()
        self._payloads: dict[int, tuple] = {}
        self._next = 1
        self._lock = threading.Lock()

    def put(self, payload: bytes) -> int:
        buffer = self.pool.alloc_bytes(len(payload))
        if payload:
            buffer.numpy()[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        with self._lock:
            handle = self._next
            self._next += 1
            self._payloads[handle] = (buffer, len(payload))
        return handle

    def get(self, handle: int) -> bytes:
        with self._lock:
            buffer, length = self._payloads[handle]
        return bytes(buffer.numpy()[:length])

    def write(self, handle: int, payload: bytes) -> None:
        with self._lock:
            buffer, _ = self._payloads[handle]
        if len(payload) > buffer.numel():
            grown = self.pool.alloc_bytes(len(payload))
            if payload:
                grown.numpy()[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
            self.pool.free(buffer)
            buffer = grown
        elif payload:
            buffer.numpy()[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        with self._lock:
            self._payloads[handle] = (buffer, len(payload))

    def release(self, handle: int) -> None:
        with self._lock:
            entry = self._payloads.pop(handle, None)
        if entry is not None:
            self.pool.free(entry[0])

    def reclaim(self) -> list[int]:
        """Leases are owned by ``LeasedPayloadQueue``; there is no TTL here."""
        return []

    def stats(self) -> dict:
        with self._lock:
            return {**self.pool.stats(), "handles": len(self._payloads)}

    def close(self) -> None:
        for handle in list(self._payloads):
            self.release(handle)


class SharedMemoryPayloadStore:
    """Cross-process shared-memory payload store with a slot table.

    Layout: ``MAGIC(8) | capacity(u64) | slots[4096](BQQQ) | data``. Slot index
    plus one is the handle, so 0 is reserved. A file lock serializes slot
    allocation between processes; payload bytes are protected by a checksum.
    """

    def __init__(
        self,
        name: str,
        *,
        capacity: int = 64 * 1024 * 1024,
        create: bool = True,
        lock_dir: str | Path | None = None,
    ) -> None:
        if capacity <= _HEADER_BYTES:
            raise ValueError("capacity too small for slot table")
        from multiprocessing.shared_memory import SharedMemory

        self.name = name
        self.capacity = capacity
        self._shm = SharedMemory(name=name, create=create, size=capacity)
        self._lock_path = Path(lock_dir or "/dev/shm") / f".{name}.lock"
        if create:
            with self._locked():
                self._buf()[:8] = MAGIC
                struct.pack_into("!Q", self._buf(), 8, capacity)
                self._buf()[_HEADER_BYTES:] = b"\x00" * (capacity - _HEADER_BYTES)
        else:
            if self._buf()[:8] != MAGIC:
                raise ValueError(f"{name} is not a megaRL payload segment")
            self.capacity = struct.unpack_from("!Q", self._buf(), 8)[0]

    def _buf(self):
        return self._shm.buf

    def _locked(self):
        import fcntl

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._lock_path.open("a+b")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return _Lock(handle)

    def _bump(self) -> int:
        used = _HEADER_BYTES  # slot table lives before the payload area
        for i in range(_MAX_SLOTS):
            flags, length, offset, _ = _SLOT.unpack_from(self._buf(), _HEADER + i * _SLOT.size)
            if flags:
                used = max(used, offset + length)
        return used

    def put(self, payload: bytes) -> int:
        checksum = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        with self._locked():
            start = self._bump()
            if start + len(payload) > self.capacity:
                raise MemoryError("payload segment full; release or grow the store")
            for i in range(_MAX_SLOTS):
                flags, _, _, _ = _SLOT.unpack_from(self._buf(), _HEADER + i * _SLOT.size)
                if not flags:
                    _SLOT.pack_into(
                        self._buf(),
                        _HEADER + i * _SLOT.size,
                        1,
                        len(payload),
                        start,
                        checksum,
                    )
                    self._buf()[start : start + len(payload)] = payload
                    return i + 1
        raise MemoryError("no free payload slots")

    def _slot(self, handle: int):
        if not 1 <= handle <= _MAX_SLOTS:
            raise KeyError(handle)
        return _SLOT.unpack_from(self._buf(), _HEADER + (handle - 1) * _SLOT.size)

    def get(self, handle: int) -> bytes:
        flags, length, offset, checksum = self._slot(handle)
        if not flags:
            raise KeyError(handle)
        payload = bytes(self._buf()[offset : offset + length])
        actual = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        if actual != checksum:
            raise ValueError(f"payload handle {handle} failed checksum")
        return payload

    def write(self, handle: int, payload: bytes) -> None:
        if not 1 <= handle <= _MAX_SLOTS:
            raise KeyError(handle)
        flags, length, offset, _ = _SLOT.unpack_from(
            self._buf(), _HEADER + (handle - 1) * _SLOT.size
        )
        if not flags or len(payload) > length:
            raise ValueError(f"handle {handle} has no room for {len(payload)} bytes")
        checksum = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        with self._locked():
            _SLOT.pack_into(
                self._buf(), _HEADER + (handle - 1) * _SLOT.size,
                flags, len(payload), offset, checksum,
            )
            self._buf()[offset : offset + len(payload)] = payload

    def release(self, handle: int) -> None:
        if not 1 <= handle <= _MAX_SLOTS:
            return
        with self._locked():
            _SLOT.pack_into(self._buf(), _HEADER + (handle - 1) * _SLOT.size, 0, 0, 0, 0)

    def reclaim(self) -> list[int]:
        """Return handles whose slots are marked used but out of segment bounds."""
        out: list[int] = []
        for i in range(_MAX_SLOTS):
            flags, length, offset, _ = _SLOT.unpack_from(self._buf(), _HEADER + i * _SLOT.size)
            if flags and (offset + length > self.capacity or length < 0):
                out.append(i + 1)
        return out

    def stats(self) -> PayloadStoreStats:
        used = sum(
            1
            for i in range(_MAX_SLOTS)
            if _SLOT.unpack_from(self._buf(), _HEADER + i * _SLOT.size)[0]
        )
        return PayloadStoreStats(self.capacity, self._bump(), used)

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()
        self._lock_path.unlink(missing_ok=True)


class _Lock:
    def __init__(self, handle) -> None:
        self.handle = handle

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        import fcntl

        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        return False


class LeasedPayloadQueue:
    """Metadata-only queue: records are serialized into the out-of-band store."""

    def __init__(
        self,
        store: PayloadStore,
        *,
        ttl_secs: float = 300.0,
        codec=None,
    ) -> None:
        if codec is None:
            from .contracts import decode as decode_contract, encode as encode_contract

            codec = (encode_contract, decode_contract)
        self.store = store
        self.codec = codec
        self.ttl_secs = ttl_secs
        self.leases = LeaseTable()
        self._handle_for: dict[int, int] = {}
        self._lease_for: dict[int, int] = {}

    def put(self, record) -> int:
        """Store the record out of band; return a lease id for its handle.

        The stamped ``payload_handle`` travels in the metadata frame; the record
        itself (including logprobs) never goes through the controller.
        """
        encode, _ = self.codec
        handle = self.store.put(encode(record))
        if hasattr(record, "payload_handle"):
            self.store.write(handle, encode(dataclasses.replace(record, payload_handle=handle)))
        lease = self.leases.acquire(handle, self.ttl_secs)
        self._handle_for[lease] = handle
        self._lease_for[handle] = lease
        return lease

    def take(self, lease: int):
        handle = self._handle_for.pop(lease, None)
        if handle is None:
            return None
        self._lease_for.pop(handle, None)
        _, decode = self.codec
        kind, record = decode(self.store.get(handle))
        if kind != type(record).__name__:
            raise ValueError(f"unexpected payload kind: {kind}")
        self.store.release(handle)
        self.leases.release(lease)
        return record

    def reclaim(self) -> list:
        out = []
        for handle in self.leases.reclaim_expired():
            lease = self._lease_for.pop(handle, None)
            if lease is None:
                continue
            self._handle_for.pop(lease, None)
            _, decode = self.codec
            out.append(decode(self.store.get(handle))[1])
            self.store.release(handle)
        return out

    def __len__(self) -> int:
        return len(self._lease_for)

    @staticmethod
    def attach_handle(record, handle: int):
        """Stamp a real out-of-band handle into the trajectory contract."""
        return dataclasses.replace(record, payload_handle=handle)
