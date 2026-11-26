"""Colocate engine lifecycle controller (worker side).

A single lock owns the engine state so pause/resume cannot interleave with
generation. `pause` fences admission first (no new generation is accepted),
then drains in-flight requests against a deadline, optionally releases GPU
memory, and reports the resulting phase. `resume` restores memory and continues
generation. The callbacks are supplied by the engine owner (megaLM engine or a
test double); all coordination, draining and epoch guarding live here.

This mirrors AReaL's pause -> pause_generation -> offload and onload ->
continue_generation sequence without putting GPU work in the Rust gateway.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator


@dataclass(frozen=True, slots=True)
class LifecycleSnapshot:
    phase: str
    epoch: int
    in_flight: int
    drained: int
    error: str | None


class EngineLifecycle:
    def __init__(
        self,
        *,
        offload: Callable[[], None] | None = None,
        onload: Callable[[], None] | None = None,
        pause_generation: Callable[[], None] | None = None,
        continue_generation: Callable[[], None] | None = None,
    ) -> None:
        # RLock so callbacks can call snapshot() without self-deadlocking.
        self._cond = threading.Condition(threading.RLock())
        self._phase = "running"
        self._epoch = 0
        self._in_flight = 0
        self._drained = 0
        self._error: str | None = None
        # A failed attempt must not make a retry at the same epoch look like an
        # idempotent success: only successful transitions are idempotent.
        self._failed_epoch: int | None = None
        self._offload = offload
        self._onload = onload
        self._pause_generation = pause_generation
        self._continue_generation = continue_generation

    @property
    def phase(self) -> str:
        with self._cond:
            return self._phase

    @property
    def epoch(self) -> int:
        with self._cond:
            return self._epoch

    def snapshot(self) -> LifecycleSnapshot:
        with self._cond:
            return LifecycleSnapshot(
                phase=self._phase,
                epoch=self._epoch,
                in_flight=self._in_flight,
                drained=self._drained,
                error=self._error,
            )

    @contextmanager
    def generation(self) -> Iterator[None]:
        """Admit one generation. Raises while the engine is fenced/paused.

        The worker calls this around `rollout.generate`; the counter is what
        `pause()` drains, so an in-flight request can finish before GPU memory
        is released instead of being killed mid-decode.
        """
        with self._cond:
            if self._phase != "running":
                raise RuntimeError(f"rollout engine is {self._phase}")
            self._in_flight += 1
        try:
            yield
        finally:
            with self._cond:
                self._in_flight -= 1
                if self._phase in ("pausing", "paused", "offloaded"):
                    self._drained += 1
                if self._in_flight == 0:
                    self._cond.notify_all()

    def wait_drained(self, timeout_ms: int = 30_000) -> bool:
        """Wait until no generation is in flight; report whether it drained.

        Used by the worker to time the fence barrier before a weight reload
        without changing the engine's phase or replaying a lifecycle command.
        """
        deadline = time.monotonic() + max(timeout_ms, 0) / 1000.0
        with self._cond:
            while self._in_flight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def apply(self, action: str, *, epoch: int, offload: bool, drain_ms: int) -> dict:
        """Apply a command and return fields for a `LifecycleAck`.

        A command older than the last applied epoch is rejected instead of
        replayed; an equal epoch is idempotent (the gateway may retry an ack
        that was lost). Failures leave the engine in a recoverable state.
        """
        started = time.monotonic()
        if action not in ("pause", "resume"):
            return self._ack(epoch, action, False, "unknown lifecycle action", started)
        with self._cond:
            if epoch < self._epoch:
                return self._ack(epoch, action, False, "stale lifecycle epoch", started)
            if (
                epoch == self._epoch
                and self._failed_epoch != epoch
                and self._phase in ("paused", "offloaded", "running")
            ):
                return self._ack(epoch, action, True, None, started)
            self._epoch = epoch
            try:
                if action == "pause":
                    self._pause(offload=offload, drain_ms=drain_ms)
                else:
                    self._resume()
            except Exception as exc:  # noqa: BLE001 - reported in the ack
                self._error = str(exc)
                self._failed_epoch = epoch
                return self._ack(epoch, action, False, str(exc), started)
            self._error = None
            self._failed_epoch = None
            return self._ack(epoch, action, True, None, started)

    def _pause(self, *, offload: bool, drain_ms: int) -> None:
        # Fence admission before waiting: `generation()` now rejects new work.
        self._phase = "pausing"
        deadline = time.monotonic() + max(drain_ms, 0) / 1000.0
        while self._in_flight > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._phase = "running"
                raise TimeoutError(
                    f"{self._in_flight} in-flight generation(s) did not drain"
                )
            self._cond.wait(remaining)
        if self._pause_generation is not None:
            self._pause_generation()
        if offload and self._offload is not None:
            self._offload()
        self._phase = "offloaded" if offload else "paused"

    def _resume(self) -> None:
        previous = self._phase
        self._phase = "resuming"
        try:
            if self._onload is not None:
                self._onload()
            if self._continue_generation is not None:
                self._continue_generation()
        except Exception:
            self._phase = previous
            raise
        self._phase = "running"

    def _ack(
        self,
        epoch: int,
        action: str,
        ok: bool,
        error: str | None,
        started: float,
    ) -> dict:
        return {
            "request_id": epoch,
            "epoch": epoch,
            "action": action,
            "ok": ok,
            "phase": self._phase,
            "drained": self._drained,
            "error": error,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
        }
