"""Python-facing control/data primitives implemented in Rust.

The controller never touches tensors: it holds a version ledger, a staleness
budget and a table of leases for out-of-band payload handles.
"""

from contextlib import contextmanager

from . import megarl_rs as _native

StalenessBudget = _native.StalenessBudget
VersionLedger = _native.VersionLedger
LeaseTable = _native.LeaseTable
RateLimiter = _native.RateLimiter
TokenSlotSemaphore = _native.TokenSlotSemaphore
AsyncPacer = _native.AsyncPacer

__all__ = [
    "AsyncPacer",
    "StalenessBudget",
    "VersionLedger",
    "LeaseTable",
    "RateLimiter",
    "TokenSlotSemaphore",
    "rate_limit",
    "slot_guard",
]


def rate_limit(limiter, count: int, emit=None) -> int:
    """Try to admit `count` under a Rust token bucket, backing off on failure.

    The limiter is authoritative; this helper only decides how much to admit
    when the bucket cannot cover the full capacity.
    """
    if limiter is None or count <= 0:
        return count
    grant = count
    while grant > 0 and not limiter.try_acquire(grant):
        grant //= 2
    if grant < count and emit is not None:
        emit("rate_limited", requested=count, granted=grant)
    return grant


@contextmanager
def slot_guard(pool):
    """Borrow one reusable slot for the block; yields ``None`` when pool is full.

    Releasing in a ``finally`` block means an exception or early return cannot
    leak a slot. The Rust lease token makes a late release harmless anyway.
    """
    lease = pool.try_acquire()
    if lease is None:
        yield None
        return
    try:
        yield lease
    finally:
        pool.release(*lease)
