"""Q6: weight-reload phase timings flow from the worker back to the gateway."""

import json
import time
from urllib.request import urlopen

from megarl.contracts import WeightUpdateAck, encode, kind_ids, validate
from megarl.gateway import RustGateway
from megarl.rollout import RolloutService
from megarl import megarl_rs as native


class SlowReload(RolloutService):
    def __init__(self, delay_s: float = 0.02) -> None:
        self.delay_s = delay_s
        self.paths: list[str] = []

    def generate(self, request):
        return []

    def reload_weights(self, path, expected_digest=None):
        self.paths.append(path)
        time.sleep(self.delay_s)


def test_protocol_mirror_includes_the_weight_update_ack():
    ids = kind_ids()
    assert ids["WeightUpdateAck"] == 10
    ack = WeightUpdateAck(
        request_id=7,
        ok=True,
        pause_ms=1,
        barrier_ms=2,
        transfer_ms=3,
        apply_ms=4,
        total_ms=10,
    )
    frame = encode(ack)
    kind, payload = native.decode(frame)
    assert kind == "WeightUpdateAck"
    restored = WeightUpdateAck(**json.loads(payload))
    assert restored == ack
    validate(restored)  # raises on drift with the authoritative Rust struct


def test_gateway_reports_worker_measured_phases():
    rollout = SlowReload(delay_s=0.02)
    gateway = RustGateway(rollout)
    port = gateway.start()
    try:
        result = gateway.reload_weights("/tmp/policy.mega")
        assert result["ok"] is True
        phases = result["phases"]
        assert phases["apply_ms"] >= 10, phases
        assert phases["wire_ms"] >= phases["apply_ms"]
        assert phases["barrier_ms"] >= 0
        assert phases["transfer_ms"] >= 0
        assert rollout.paths == ["/tmp/policy.mega"]

        with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
            text = response.read().decode()
        assert "megarl_weight_reload_apply_ms " in text
        assert "megarl_weight_reload_phase_samples 1" in text
    finally:
        gateway.close()
