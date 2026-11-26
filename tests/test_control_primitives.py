"""Python surface of the Rust control-plane primitives."""

from megarl.control import AsyncPacer, RateLimiter, TokenSlotSemaphore, slot_guard


def test_token_slot_semaphore_bounds_leases_and_rejects_stale_release():
    sem = TokenSlotSemaphore(capacity=2, ttl_secs=60.0)
    assert sem.capacity() == 2
    first = sem.try_acquire()
    second = sem.try_acquire()
    assert first is not None and second is not None
    assert first[0] != second[0]
    assert sem.try_acquire() is None
    assert sem.in_use() == 2

    # A stale token must not free the live lease.
    assert sem.release(first[0], first[1] + 100) is False
    assert sem.in_use() == 2
    assert sem.release(*first) is True
    assert sem.release(*first) is False
    assert sem.available() == 1


def test_slot_guard_releases_on_exception():
    sem = TokenSlotSemaphore(capacity=1, ttl_secs=60.0)
    with slot_guard(sem) as lease:
        assert lease is not None
    # A single slot is available again after the context exits.
    assert sem.available() == 1

    try:
        with slot_guard(sem) as lease:
            assert lease is not None
            raise ValueError("fail inside block")
    except ValueError:
        pass
    assert sem.available() == 1


def test_async_pacer_batches_waiter_wakeups():
    pacer = AsyncPacer(rate_per_sec=1.0, batch_size=2)
    # One credit takes a full second, so an immediate tick wakes nobody.
    assert pacer.wake(10) == 0
    assert pacer.retry_after_ms() > 0


def test_rate_limiter_is_available_from_python():
    limiter = RateLimiter(rate_per_sec=10.0, burst=1)
    assert limiter.available() == 1.0
    assert limiter.try_acquire(1) is True
    assert limiter.try_acquire(1) is False
    assert limiter.retry_after_ms(1) > 0
