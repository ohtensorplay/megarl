"""Trajectory-level dLLM rollouts and replay (TraceRL ICLR 2026 arXiv 2509.06949;
MaskGRPO arXiv 2510.02880; CJ-GRPO arXiv 2509.23924); see package docstring for
details."""

from __future__ import annotations

from ...contracts import TrajectoryRecord
from ...dataplane import InProcessPayloadStore, LeasedPayloadQueue, PayloadStore
from .types import DLLMTrajectoryRecord, decode_trajectory, encode_trajectory


class DLLMPayloadQueue(LeasedPayloadQueue):
    """Metadata queue that preserves an existing dLLM trajectory handle.

    The default ``LeasedPayloadQueue`` stamps its own store handle into every
    record. For dLLM rollouts that handle points at the trajectory arrays, so
    overwriting it would make the step data unreachable. This subclass keeps
    the producer's handle and still leases the metadata frame normally.
    """

    def put(self, record) -> int:
        encode, _ = self.codec
        handle = self.store.put(encode(record))
        lease = self.leases.acquire(handle, self.ttl_secs)
        self._handle_for[lease] = handle
        self._lease_for[handle] = lease
        return lease


class DLLMTrajectoryStore:
    """Out-of-band trajectory store keyed by ``TrajectoryRecord.payload_handle``.

    Defaults to an in-process store; pass ``SharedMemoryPayloadStore`` (or a
    pinned store) for cross-process controllers. Only the integer handle ever
    enters the controller frame.
    """

    def __init__(self, store: PayloadStore | None = None) -> None:
        self.store = store or InProcessPayloadStore()

    def put(self, record: DLLMTrajectoryRecord) -> int:
        return self.store.put(encode_trajectory(record))

    def get(self, handle: int) -> DLLMTrajectoryRecord:
        return decode_trajectory(self.store.get(handle))

    def release(self, handle: int) -> None:
        self.store.release(handle)
