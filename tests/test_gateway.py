"""Rust gateway end-to-end: per-token logprobs survive the HTTP boundary."""

from urllib.error import HTTPError
from urllib.request import Request, urlopen

from megarl.contracts import TrajectoryRecord
from megarl.gateway import RustGateway, get_json, post_json


def get_metrics(url: str, *, token: str | None = None, timeout: float = 10.0) -> dict:
    """Read scalar samples from the gateway's Prometheus text endpoint."""

    request = Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "{" in line:
            continue
        name, _, value = line.partition(" ")
        name = name.removeprefix("megarl_")
        try:
            values[name] = float(value)
        except ValueError:
            continue
    return values


class FakeRollout:
    def generate(self, request):
        for i in range(request.n):
            yield TrajectoryRecord(
                request_id=request.request_id,
                prompt_len=len(request.prompt_token_ids),
                token_ids=[10 + i],
                logprobs=[-0.1 - i],
                top_logprobs=[],
                policy_versions=[request.policy_version],
                reward=None,
                finish_reason=0,
                payload_handle=None,
            )

    def reload_weights(self, path, expected_digest=None):
        self.reloaded = path


def test_rust_gateway_exposes_logprobs():
    gateway = RustGateway(FakeRollout())
    port = gateway.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert get_json(base + "/health")["ok"] is True
        result = post_json(
            base + "/rollout/generate",
            {
                "request_id": 9,
                "policy_version": 3,
                "prompt_token_ids": [1, 2],
                "max_new_tokens": 4,
                "n": 2,
                "sampling": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "seed": None,
                    "logprobs": True,
                    "top_logprobs": 0,
                },
            },
        )
        records = result["records"]
        assert len(records) == 2
        assert all("logprobs" in record and record["logprobs"] for record in records)
        reload = post_json(base + "/rollout/weights", {"path": "/tmp/w"})
        assert reload["ok"] is True
        assert "phases" in reload  # worker-reported timing breakdown (Q6)
        metrics = get_metrics(base + "/metrics")
        assert metrics["generates"] >= 1
        assert metrics["workers"] == 1
        assert "in_flight" in metrics and "retries" in metrics
        assert "weight_reload_ms" in metrics and "weight_reload_max_ms" in metrics
    finally:
        gateway.close()


def test_gateway_rate_limits_with_retry_after(monkeypatch):
    monkeypatch.setenv("MEGARL_GATEWAY_RATE_PER_SEC", "0.1")
    monkeypatch.setenv("MEGARL_GATEWAY_RATE_BURST", "1")
    gateway = RustGateway(FakeRollout(), workers=1)
    port = gateway.start()
    try:
        base = f"http://127.0.0.1:{port}"
        payload = {
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
        assert len(post_json(base + "/rollout/generate", payload)["records"]) == 1
        try:
            post_json(base + "/rollout/generate", payload)
        except HTTPError as exc:
            assert exc.code == 429
            assert exc.headers.get("Retry-After")
        else:
            raise AssertionError("rate limit was not enforced")
        assert get_metrics(base + "/metrics")["rate_rejected"] == 1
    finally:
        gateway.close()


def test_gateway_supports_a_runtime_worker_registry():
    gateway = RustGateway(FakeRollout(), workers=1, token="s3cret")
    port = gateway.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert get_metrics(base + "/metrics", token="s3cret")["workers"] == 1
        added = post_json(
            base + "/workers/register",
            {"address": "127.0.0.1:19999"},
            token="s3cret",
        )
        assert added["ok"] is True
        assert added["workers"] == 2
        assert get_metrics(base + "/metrics", token="s3cret")["workers"] == 2
        removed = post_json(
            base + "/workers/deregister",
            {"address": "127.0.0.1:19999"},
            token="s3cret",
        )
        assert removed["ok"] is True
        assert removed["workers"] == 1
    finally:
        gateway.close()


def test_gateway_enforces_bearer_token_and_supports_a_worker_pool():
    gateway = RustGateway(FakeRollout(), workers=2, token="s3cret")
    port = gateway.start()
    try:
        base = f"http://127.0.0.1:{port}"
        payload = {
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
        try:
            post_json(base + "/rollout/generate", payload)
        except HTTPError as exc:
            assert exc.code == 401
        else:
            raise AssertionError("missing token was accepted")
        assert get_json(base + "/health")["ok"] is True
        assert get_metrics(base + "/metrics", token="s3cret")["workers"] == 2
    finally:
        gateway.close()


def test_gateway_model_version_registry_over_http():
    gateway = RustGateway(FakeRollout(), workers=1, token="s3cret")
    port = gateway.start()
    try:
        digest = "ab" * 32
        staged = gateway.publish_model_version("policy", 1, digest, ["w1"])
        assert staged == {"ok": True, "model": "policy", "version": 1, "state": "staged"}
        committed = gateway.commit_model_version("policy", 1)
        assert committed["committed"] == 1 and committed["retired"] is None

        gateway.publish_model_version("policy", 2, "cd" * 32, ["w2"])
        assert gateway.abort_model_version("policy", 2)["state"] == "aborted"
        snapshot = gateway.get_model("policy")
        assert snapshot["active"] == 1
        assert [v["state"] for v in snapshot["versions"]] == ["committed", "aborted"]

        metrics = get_metrics(f"http://127.0.0.1:{port}/metrics", token="s3cret")
        assert metrics["model_publishes"] == 2
        assert metrics["model_commits"] == 1
        assert metrics["model_aborts"] == 1
        assert any(m["model"] == "policy" for m in gateway.list_models())
    finally:
        gateway.close()


def test_gateway_exposes_prometheus_text_metrics():
    gateway = RustGateway(FakeRollout(), workers=1)
    port = gateway.start()
    try:
        request = Request(f"http://127.0.0.1:{port}/metrics")
        with urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
            assert response.headers["Content-Type"].startswith("text/plain")
        assert "# HELP megarl_generates" in body
        assert "# TYPE megarl_generates counter" in body
        assert "megarl_workers " in body
    finally:
        gateway.close()
