"""Production rollout pause/resume: real gateway, real worker, real frames.

The Python side owns only the engine lifecycle callbacks and the HTTP client;
draining, epoch guarding and orchestration are exercised through the Rust
gateway over the versioned MGRL protocol.
"""

import threading
import time
from urllib.error import HTTPError

from megarl.contracts import TrajectoryRecord
from megarl.gateway import RustGateway, get_json, post_json
from megarl.lifecycle import EngineLifecycle


class FakeRollout:
    def __init__(self, gate: threading.Event | None = None) -> None:
        self.gate = gate

    def generate(self, request):
        if self.gate is not None:
            self.gate.wait(5)
        yield TrajectoryRecord(
            request_id=request.request_id,
            prompt_len=len(request.prompt_token_ids),
            token_ids=[10],
            logprobs=[-0.1],
            top_logprobs=[],
            policy_versions=[request.policy_version],
            reward=None,
            finish_reason=0,
            payload_handle=None,
        )

    def reload_weights(self, path, expected_digest=None):
        return None


PAYLOAD = {
    "request_id": 1,
    "policy_version": 1,
    "prompt_token_ids": [1, 2],
    "max_new_tokens": 2,
    "n": 1,
    "sampling": {
        "temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": None,
        "logprobs": True, "top_logprobs": 0,
    },
}


def test_engine_lifecycle_drains_in_flight_before_offload():
    events: list[str] = []
    lifecycle = EngineLifecycle(
        offload=lambda: events.append("offload"),
        onload=lambda: events.append("onload"),
        pause_generation=lambda: events.append("pause_generation"),
        continue_generation=lambda: events.append("continue_generation"),
    )
    with lifecycle.generation():
        try:
            lifecycle.apply(
                "pause", epoch=1, offload=True, drain_ms=120
            )
        except Exception as exc:  # _apply reports failures in the ack, not raises
            raise AssertionError("apply must report failures in the ack") from exc
        # A command that cannot drain reports failure in the ack and stays running.
        assert events == []
    ack = lifecycle.apply("pause", epoch=1, offload=True, drain_ms=1_000)
    assert ack["ok"] is True
    assert ack["phase"] == "offloaded"
    assert events == ["pause_generation", "offload"]
    ack = lifecycle.apply("resume", epoch=2, offload=False, drain_ms=0)
    assert ack["phase"] == "running"
    assert events[-2:] == ["onload", "continue_generation"]


def test_engine_lifecycle_rejects_generation_while_paused():
    lifecycle = EngineLifecycle()
    lifecycle.apply("pause", epoch=1, offload=False, drain_ms=1_000)
    try:
        with lifecycle.generation():
            pass
    except RuntimeError as exc:
        assert "paused" in str(exc)
    else:
        raise AssertionError("generation was admitted while paused")
    # A stale epoch cannot re-apply an older transition.
    stale = lifecycle.apply("resume", epoch=0, offload=False, drain_ms=0)
    assert stale["ok"] is False
    assert stale["error"] == "stale lifecycle epoch"


def test_gateway_pause_resume_over_http():
    events: list[str] = []
    lifecycle = EngineLifecycle(
        offload=lambda: events.append("offload"),
        onload=lambda: events.append("onload"),
    )
    gateway = RustGateway(FakeRollout(), lifecycle=lifecycle)
    port = gateway.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert get_json(base + "/rollout/lifecycle")["paused"] is False
        assert len(post_json(base + "/rollout/generate", PAYLOAD)["records"]) == 1

        paused = gateway.pause_rollout(offload=True, drain_ms=1_000, timeout_ms=5_000)
        assert paused["ok"] is True
        assert paused["state"]["offloaded"] is True
        assert events == ["offload"]
        try:
            post_json(base + "/rollout/generate", PAYLOAD)
        except HTTPError as exc:
            assert exc.code == 503
        else:
            raise AssertionError("generation was admitted while offloaded")

        resumed = gateway.resume_rollout(timeout_ms=5_000)
        assert resumed["ok"] is True
        assert events[-1] == "onload"
        assert len(post_json(base + "/rollout/generate", PAYLOAD)["records"]) == 1

        # /metrics is Prometheus text exposition, not JSON.
        from urllib.request import urlopen

        with urlopen(base + "/metrics", timeout=5) as response:
            text = response.read().decode()
        assert "megarl_lifecycle_pauses 1" in text
        assert "megarl_lifecycle_resumes 1" in text
        assert gateway.lifecycle()["paused"] is False
    finally:
        gateway.close()


def test_gateway_pause_times_out_when_generation_does_not_drain():
    gate = threading.Event()
    lifecycle = EngineLifecycle()
    gateway = RustGateway(FakeRollout(gate), workers=1, lifecycle=lifecycle)
    port = gateway.start()
    try:
        base = f"http://127.0.0.1:{port}"
        def generate_in_thread() -> None:
            try:
                post_json(base + "/rollout/generate", PAYLOAD)
            except Exception:  # gateway may close while the worker is draining
                pass

        result = threading.Thread(target=generate_in_thread, daemon=True)
        result.start()
        deadline = time.monotonic() + 3
        while lifecycle.snapshot().in_flight == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert lifecycle.snapshot().in_flight == 1
        try:
            post_json(
                base + "/rollout/pause",
                {"offload": True, "drain_ms": 150, "timeout_ms": 5_000},
            )
        except HTTPError as exc:
            assert exc.code == 502  # gateway rolls back on drain failure
        else:
            raise AssertionError("pause succeeded despite an in-flight generation")
        assert gateway.lifecycle()["paused"] is False
    finally:
        gate.set()
        gateway.close()
