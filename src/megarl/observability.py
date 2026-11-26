"""Structured, RL-shaped telemetry.

Events are emitted from the controller loop, not bolted on afterwards. Each
event is a JSON line so it can be shipped to Better Stack / file / stdout
without a second metrics stack. The storage and rendering primitives (metric
registry, Prometheus renderer, JSON logging, OTLP hooks, training trackers)
come from the shared ``tensorplay-observability`` package; only the mapping
from RL events to RL metrics lives here.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from tp_observability import (
    MetricRegistry,
    Tracking,
    configure_logging,
    configure_telemetry,
    get_registry,
    render_prometheus_metrics,
)

# Staleness is measured in tokens, but the shared observe() API takes the same
# bucket sequence as latency histograms; these bounds cover 1 .. 10k tokens.
STALE_TOKEN_BUCKETS_MS = (1, 10, 100, 1000, 10000)


def _as_float(value: Any) -> float:
    """Best-effort numeric coercion; missing/broken fields become zero."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _metric_name(name: str) -> str:
    """Keep dynamic phase/event names safe for Prometheus identifiers."""

    cleaned = re.sub(r"\W+", "_", str(name)).strip("_")
    return cleaned or "unknown"


def _record_event(registry: MetricRegistry, event: str, fields: dict[str, Any]) -> None:
    """Map one RL control-plane event into shared-registry metric updates.

    Every event keeps a counter so `rate()` works; gauges carry the latest
    observed value (version, capacity, queue depth). `_histogram` names are
    rendered as real Prometheus histograms with ``_bucket`` / ``_sum`` /
    ``_count`` series by ``render_prometheus_metrics``. Label cardinality is
    bounded by the registry's ``max_series_per_metric`` cap: unknown event names
    and high-cardinality agent/tool names cannot create unbounded series.

    Events not listed below are counted as ``other_events{event=...}``.
    """

    if event == "staleness":
        registry.counter("staleness_events")
        stale_tokens = _as_float(fields.get("stale_tokens", 0))
        registry.gauge("max_stale_tokens", stale_tokens)
        registry.observe(
            "stale_tokens_histogram", stale_tokens, buckets=STALE_TOKEN_BUCKETS_MS
        )
    elif event == "trainer_update":
        registry.counter("trainer_updates")
        batch_size = _as_float(fields.get("batch_size", 0))
        registry.counter("tokens_trained", batch_size)
        registry.observe("batch_size_histogram", batch_size)
    elif event == "reward_group_summary":
        registry.counter("reward_groups")
        registry.counter("groups_dropped", _as_float(fields.get("dropped", 0)))
        registry.gauge("reward_group_count", _as_float(fields.get("groups", 0)))
    elif event == "weight_reload":
        registry.counter("weight_reloads")
        registry.gauge("policy_version", _as_float(fields.get("version", 0)))
    elif event == "checkpoint_saved":
        registry.counter("checkpoint_saved")
        registry.gauge("policy_version", _as_float(fields.get("version", 0)))
    elif event == "admission_deferred":
        registry.counter("admission_deferred")
        registry.gauge("admission_capacity", _as_float(fields.get("capacity", 0)))
    elif event == "rate_limited":
        registry.counter("rate_limited")
    elif event in ("pause", "resume"):
        registry.counter(event)
        registry.gauge("paused", 1 if event == "pause" else 0)
    elif event == "rollout_generate":
        registry.counter("rollout_generate")
    elif event.startswith("agent_"):
        registry.counter("agent_events", event=_metric_name(event))
        if "duration_ms" in fields:
            registry.observe(
                "agent_duration_ms_histogram", _as_float(fields["duration_ms"])
            )
    elif event.startswith("tool_"):
        registry.counter("tool_events", event=_metric_name(event))
    else:
        registry.counter("other_events", event=_metric_name(event))


@dataclass
class Telemetry:
    """RL event recorder shared by controllers, agents and gateway workers."""

    path: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    registry: MetricRegistry | None = None
    max_events: int = 100000
    prefix: str = "megarl"
    # Parallel agent episodes share one Telemetry instance. Guard both the
    # in-memory list and the file append so records stay whole and ordered.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = get_registry()

    def emit(self, event: str, **fields: Any) -> None:
        """Record an event, then mirror it into RL metrics (fail-soft)."""

        record = {"ts_ns": time.time_ns(), "event": event, **fields}
        with self._lock:
            self.events.append(record)
            if self.max_events > 0 and len(self.events) > self.max_events:
                del self.events[: len(self.events) - self.max_events]
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")

        if self.registry is not None:
            try:
                _record_event(self.registry, event, fields)
            except Exception:
                # Observability is a side effect: a broken label, a non-numeric
                # field or a full registry must never break the training loop.
                pass

    def observe_duration(self, name: str, milliseconds: float) -> None:
        """Record a phase duration into ``phase_<name>_ms_histogram``."""

        if self.registry is None:
            return
        try:
            self.registry.observe(
                f"phase_{_metric_name(name)}_ms_histogram", _as_float(milliseconds)
            )
        except Exception:
            pass

    def render_prometheus(self) -> str:
        """Render the shared registry snapshot as Prometheus 0.0.4 text."""

        if self.registry is None:
            return ""
        return render_prometheus_metrics(self.registry.snapshot(), prefix=self.prefix)

    def close(self) -> None:
        """Flush hook for symmetry with ``Tracking``; no buffered state here."""

    def summary(self) -> dict[str, Any]:
        by_event: dict[str, int] = {}
        for record in self.events:
            by_event[record["event"]] = by_event.get(record["event"], 0) + 1
        return {"total": len(self.events), "by_event": by_event}


class Timer:
    """Time a block and record it via ``telemetry.observe_duration``."""

    def __init__(self, telemetry: Telemetry, name: str) -> None:
        self.telemetry = telemetry
        self.name = name
        self.started: float | None = None

    def __enter__(self) -> Timer:
        self.started = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.started is not None:
            self.telemetry.observe_duration(self.name, (time.perf_counter() - self.started) * 1000)
            self.started = None


@dataclass(frozen=True)
class ObservabilityConfig:
    """Small handle returned by :func:`configure`."""

    prefix: str
    service: str
    level: str
    registry: MetricRegistry
    otel_enabled: bool


def configure(
    *,
    level: str = "INFO",
    service: str = "megarl",
    prefix: str = "megarl",
    registry: MetricRegistry | None = None,
    otel: bool = False,
    **otel_options: Any,
) -> ObservabilityConfig:
    """Install JSON logging and, optionally, OTLP export.

    All hooks are fail-soft: without the ``otel`` extra (or an unreachable
    collector) telemetry stays local and training continues unchanged.
    """

    configure_logging(level=level, service=service)
    enabled = False
    if otel:
        try:
            enabled = bool(
                configure_telemetry(service_name=service, prefix=prefix, **otel_options)
            )
        except Exception:
            enabled = False
    return ObservabilityConfig(
        prefix=prefix,
        service=service,
        level=level.upper(),
        registry=registry if registry is not None else get_registry(),
        otel_enabled=enabled,
    )


__all__ = [
    "ObservabilityConfig",
    "STALE_TOKEN_BUCKETS_MS",
    "MetricRegistry",
    "Timer",
    "Tracking",
    "Telemetry",
    "configure",
    "get_registry",
    "render_prometheus_metrics",
]
