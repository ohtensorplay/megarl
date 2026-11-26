"""Rust gateway manager.

The HTTP server itself is Rust (`megarl-gateway`); this module only spawns it
and the Python frame worker, and offers a small HTTP client for tests/tools.
Keeping HTTP out of Python avoids the GIL on the service path and matches
megaLM's axum-server + Python-GPU-worker split.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

from .lifecycle import EngineLifecycle
from .observability import Telemetry
from .rollout import RolloutService
from .worker import FrameWorker

_DEFAULT_BINARY = (
    Path(__file__).resolve().parents[2] / "rust" / "target" / "debug" / "megarl-gateway"
)


class RustGateway:
    def __init__(
        self,
        rollout: RolloutService,
        *,
        binary: str | Path | None = None,
        workers: int = 1,
        token: str | None = None,
        lifecycle: EngineLifecycle | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.rollout = rollout
        self.binary = Path(binary or os.environ.get("MEGARL_GATEWAY_BIN", _DEFAULT_BINARY))
        self.worker_pool = [
            FrameWorker(rollout, lifecycle=lifecycle) for _ in range(workers)
        ]
        self.token = token
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.telemetry = Telemetry()

    def start(self, host: str = "127.0.0.1") -> int:
        if not self.binary.exists():
            raise FileNotFoundError(
                f"Rust gateway binary missing: {self.binary}. "
                "Build it with `cargo build --manifest-path rust/Cargo.toml -p megarl-rs --bin megarl-gateway`."
            )
        addresses = ",".join(worker.start() for worker in self.worker_pool)
        env = os.environ.copy()
        token = self.token or env.get("MEGARL_GATEWAY_TOKEN")
        if token:
            env["MEGARL_GATEWAY_TOKEN"] = token
        self.proc = subprocess.Popen(
            [str(self.binary), f"{host}:0", addresses],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        line = self.proc.stdout.readline().strip()
        if not line:
            raise RuntimeError(f"gateway failed to start: {self.proc.stderr.read()}")
        self.port = int(line.rsplit(":", 1)[1])
        return self.port

    def register_worker(self, address: str) -> dict:
        """Add a running frame worker to the live gateway pool."""
        if self.port is None:
            raise RuntimeError("gateway is not started")
        return post_json(
            f"http://127.0.0.1:{self.port}/workers/register",
            {"address": address},
            token=self.token,
        )

    def deregister_worker(self, address: str) -> dict:
        """Remove a worker from the live pool (the last one is retained)."""
        if self.port is None:
            raise RuntimeError("gateway is not started")
        return post_json(
            f"http://127.0.0.1:{self.port}/workers/deregister",
            {"address": address},
            token=self.token,
        )

    def publish_model_version(
        self, model: str, version: int, digest: str, workers: list[str]
    ) -> dict:
        """Stage a new model version. `digest` is a SHA-256 hex string."""
        if self.port is None:
            raise RuntimeError("gateway is not started")
        return post_json(
            f"http://127.0.0.1:{self.port}/models/{model}/versions",
            {"version": version, "digest": digest, "workers": workers},
            token=self.token,
        )

    def commit_model_version(self, model: str, version: int) -> dict:
        """Atomically promote a staged version to active (old one retires)."""
        if self.port is None:
            raise RuntimeError("gateway is not started")
        return post_json(
            f"http://127.0.0.1:{self.port}/models/{model}/versions/{version}/commit",
            {},
            token=self.token,
        )

    def abort_model_version(self, model: str, version: int) -> dict:
        """Discard a failed staged version without touching the active one."""
        if self.port is None:
            raise RuntimeError("gateway is not started")
        return post_json(
            f"http://127.0.0.1:{self.port}/models/{model}/versions/{version}/abort",
            {},
            token=self.token,
        )

    def get_model(self, model: str) -> dict:
        """Snapshot of a model: version history, states and active version."""
        if self.port is None:
            raise RuntimeError("gateway is not started")
        return get_json(
            f"http://127.0.0.1:{self.port}/models/{model}", token=self.token
        )

    def list_models(self) -> list:
        if self.port is None:
            raise RuntimeError("gateway is not started")
        return get_json(f"http://127.0.0.1:{self.port}/models", token=self.token)

    def reload_weights(
        self,
        path: str,
        *,
        expected_digest: str | None = None,
        abort_inflight: bool = False,
    ) -> dict:
        """Trigger a rollout-side weight reload and return its phase timings."""
        if self.port is None:
            raise RuntimeError("gateway is not started")
        return post_json(
            f"http://127.0.0.1:{self.port}/rollout/weights",
            {
                "path": path,
                "expected_digest": expected_digest,
                "abort_inflight": abort_inflight,
            },
            token=self.token,
        )

    def pause_rollout(
        self, *, offload: bool = True, drain_ms: int = 30_000, timeout_ms: int = 120_000
    ) -> dict:
        """Fence generation, drain in-flight work and release GPU memory."""
        if self.port is None:
            raise RuntimeError("gateway is not started")
        return post_json(
            f"http://127.0.0.1:{self.port}/rollout/pause",
            {"offload": offload, "drain_ms": drain_ms, "timeout_ms": timeout_ms},
            token=self.token,
        )

    def resume_rollout(self, *, timeout_ms: int = 120_000) -> dict:
        """Restore GPU memory and continue generation."""
        if self.port is None:
            raise RuntimeError("gateway is not started")
        return post_json(
            f"http://127.0.0.1:{self.port}/rollout/resume",
            {"timeout_ms": timeout_ms},
            token=self.token,
        )

    def lifecycle(self) -> dict:
        if self.port is None:
            raise RuntimeError("gateway is not started")
        return get_json(
            f"http://127.0.0.1:{self.port}/rollout/lifecycle", token=self.token
        )

    def close(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        for worker in self.worker_pool:
            worker.close()


def _auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def post_json(url: str, payload: dict, *, token: str | None = None, timeout: float = 10.0) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **_auth_headers(token)},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get_json(url: str, *, token: str | None = None, timeout: float = 10.0) -> dict:
    request = Request(url, headers=_auth_headers(token))
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())
