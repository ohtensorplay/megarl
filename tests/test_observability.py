"""RL event mapping on top of the shared tensorplay-observability package."""

from __future__ import annotations

import json
import threading

import tp_observability

import megarl.observability as observability
from megarl.observability import (
    MetricRegistry,
    Telemetry,
    Timer,
    Tracking,
    configure,
    get_registry,
    render_prometheus_metrics,
)


def test_reexports_shared_primitives():
    assert Tracking is tp_observability.Tracking
    assert MetricRegistry is tp_observability.MetricRegistry
    assert get_registry() is tp_observability.get_registry()
    assert render_prometheus_metrics is tp_observability.render_prometheus_metrics


def test_event_mapping_and_prometheus_render():
    registry = MetricRegistry()
    telemetry = Telemetry(registry=registry)

    telemetry.emit("staleness", version=1, stale_tokens=7)
    telemetry.emit("trainer_update", batch_size=14)

    text = telemetry.render_prometheus()
    assert "# HELP megarl_max_stale_tokens" in text
    assert "# TYPE megarl_max_stale_tokens gauge" in text
    assert "megarl_max_stale_tokens 7.0" in text
    assert "megarl_staleness_events 1.0" in text
    assert "megarl_trainer_updates 1.0" in text
    assert "megarl_tokens_trained 14" in text
    assert "# TYPE megarl_stale_tokens_histogram histogram" in text
    assert 'megarl_stale_tokens_histogram_bucket{le="10"} 1' in text
    assert "megarl_stale_tokens_histogram_sum 7.0" in text
    assert "megarl_stale_tokens_histogram_count 1" in text
    assert "# TYPE megarl_batch_size_histogram histogram" in text


def test_full_event_catalog():
    registry = MetricRegistry()
    telemetry = Telemetry(registry=registry)

    telemetry.emit("reward_group_summary", groups=8, dropped=2)
    telemetry.emit("weight_reload", version=3)
    telemetry.emit("checkpoint_saved", version=3)
    telemetry.emit("admission_deferred", requested=10, capacity=4)
    telemetry.emit("rate_limited")
    telemetry.emit("pause", mode="wait")
    telemetry.emit("resume")
    telemetry.emit("rollout_generate", tokens=16)
    telemetry.emit("agent_turn", duration_ms=12)
    telemetry.emit("tool_proxy_http", status=200)
    telemetry.emit("something_else", note="unmapped")

    snapshot = registry.snapshot()
    assert snapshot["reward_groups"] == 1.0
    assert snapshot["groups_dropped"] == 2.0
    assert snapshot["reward_group_count"] == 8.0
    assert snapshot["weight_reloads"] == 1.0
    assert snapshot["checkpoint_saved"] == 1.0
    assert snapshot["policy_version"] == 3.0
    assert snapshot["admission_deferred"] == 1.0
    assert snapshot["admission_capacity"] == 4.0
    assert snapshot["rate_limited"] == 1.0
    assert snapshot["pause"] == 1.0
    assert snapshot["resume"] == 1.0
    assert snapshot["paused"] == 0.0
    assert snapshot["rollout_generate"] == 1.0

    text = telemetry.render_prometheus()
    assert 'megarl_agent_events{event="agent_turn"} 1.0' in text
    assert "megarl_agent_duration_ms_histogram_count 1" in text
    assert 'megarl_tool_events{event="tool_proxy_http"} 1.0' in text
    assert 'megarl_other_events{event="something_else"} 1.0' in text


def test_timer_observes_phase_histogram():
    registry = MetricRegistry()
    telemetry = Telemetry(registry=registry)

    with Timer(telemetry, "rollout"):
        pass

    text = telemetry.render_prometheus()
    assert "# TYPE megarl_phase_rollout_ms_histogram histogram" in text
    assert "megarl_phase_rollout_ms_histogram_count 1" in text
    assert "megarl_phase_rollout_ms_histogram_sum " in text


def test_configure_noop_without_otel(monkeypatch):
    calls: list[dict] = []

    def fake_configure_telemetry(**kwargs):
        calls.append(kwargs)
        return False

    monkeypatch.setattr(observability, "configure_telemetry", fake_configure_telemetry)

    config = configure()
    assert config.prefix == "megarl"
    assert config.service == "megarl"
    assert config.level == "INFO"
    assert config.registry is get_registry()
    assert config.otel_enabled is False
    assert calls == []

    config = configure(
        level="DEBUG",
        service="megarl-worker",
        otel=True,
        metrics_endpoint="http://localhost:4318/v1/metrics",
    )
    assert config.otel_enabled is False
    assert calls[0]["prefix"] == "megarl"
    assert calls[0]["service_name"] == "megarl-worker"
    assert calls[0]["metrics_endpoint"] == "http://localhost:4318/v1/metrics"

    def broken_configure_telemetry(**kwargs):
        raise RuntimeError("otel extra unavailable")

    monkeypatch.setattr(observability, "configure_telemetry", broken_configure_telemetry)
    assert configure(otel=True).otel_enabled is False


def test_tracking_fanout_to_jsonl_and_prometheus(tmp_path):
    path = tmp_path / "train.jsonl"
    registry = MetricRegistry()
    tracker = Tracking(
        "megarl",
        "unit-test",
        backends=("jsonl", "prometheus"),
        jsonl_path=path,
        registry=registry,
    )
    tracker.log({"actor/pg_loss@mean": 0.25, "reward": 1.5}, step=3)
    tracker.summary({"best_reward": 1.5})
    tracker.finish(0)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"step": 3, "data": {"actor/pg_loss": 0.25, "reward": 1.5}}
    ]
    text = render_prometheus_metrics(registry.snapshot(), prefix="megarl")
    assert "megarl_actor_pg_loss 0.25" in text
    assert "megarl_reward 1.5" in text


def test_backward_compatible_summary_and_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"
    telemetry = Telemetry(path=path, registry=MetricRegistry())
    telemetry.emit("staleness", version=1, stale_tokens=7)
    telemetry.emit("staleness", version=2, stale_tokens=3)
    telemetry.emit("reward_group_summary", groups=4, dropped=1)

    assert telemetry.summary() == {
        "total": 3,
        "by_event": {"staleness": 2, "reward_group_summary": 1},
    }
    assert set(telemetry.events[0]) == {"ts_ns", "event", "version", "stale_tokens"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == [
        "staleness",
        "staleness",
        "reward_group_summary",
    ]
    telemetry.close()


def test_bounded_events_and_thread_safe_emit():
    telemetry = Telemetry(max_events=5, registry=MetricRegistry())
    threads = [
        threading.Thread(target=lambda i=i: telemetry.emit("tick", i=i))
        for i in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(telemetry.events) == 5
    assert telemetry.summary() == {"total": 5, "by_event": {"tick": 5}}


def test_metric_mapping_errors_never_break_event_logging():
    class BrokenRegistry:
        def counter(self, *args, **kwargs):
            raise RuntimeError("registry down")

        def gauge(self, *args, **kwargs):
            raise RuntimeError("registry down")

        def observe(self, *args, **kwargs):
            raise RuntimeError("registry down")

    telemetry = Telemetry(registry=BrokenRegistry())  # type: ignore[arg-type]
    telemetry.emit("staleness", stale_tokens=7)
    assert telemetry.summary() == {"total": 1, "by_event": {"staleness": 1}}
