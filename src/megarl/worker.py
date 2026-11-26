"""Python GPU worker behind the Rust rollout gateway.

It speaks only versioned MGRL frames: metadata in, metadata out. The heavy
generation loop runs here (megaLM `Engine` is Python), while HTTP, validation,
routing and coordination stay in Rust.
"""

from __future__ import annotations

import json
import socketserver
import struct
import threading
import time
from typing import Any

from . import megarl_rs as native
from .contracts import (
    LifecycleAck,
    LifecycleCommand,
    WeightUpdate,
    WeightUpdateAck,
    decode as decode_contract,
    encode,
)
from .lifecycle import EngineLifecycle
from .rollout import RolloutService

ZERO_LENGTH = struct.pack(">I", 0)


def _recv_exact(sock, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise EOFError("worker connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


class FrameHandler(socketserver.BaseRequestHandler):
    def _send_frame(self, frame: bytes) -> None:
        self.request.sendall(struct.pack(">I", len(frame)) + frame)

    def handle(self) -> None:
        try:
            while True:
                head = self.request.recv(4)
                if not head or len(head) < 4:
                    break
                size = struct.unpack(">I", head)[0]
                if size == 0:
                    break
                frame = _recv_exact(self.request, size)
                kind, payload_json = native.decode(frame)
                if kind == "RolloutRequest":
                    _, request = decode_contract(frame)
                    lifecycle = self.server.lifecycle
                    if lifecycle is not None:
                        with lifecycle.generation():
                            for record in self.server.rollout.generate(request):
                                self._send_frame(encode(record))
                    else:
                        for record in self.server.rollout.generate(request):
                            self._send_frame(encode(record))
                elif kind == "LifecycleCommand":
                    command = LifecycleCommand(**json.loads(payload_json))
                    result = self.server.lifecycle.apply(
                        command.action,
                        epoch=command.epoch,
                        offload=command.offload,
                        drain_ms=command.drain_ms,
                    )
                    self._send_frame(
                        encode(
                            LifecycleAck(
                                request_id=command.request_id,
                                epoch=result["epoch"],
                                action=result["action"],
                                ok=result["ok"],
                                phase=result["phase"],
                                drained=result["drained"],
                                error=result["error"],
                                elapsed_ms=result["elapsed_ms"],
                            )
                        )
                    )
                elif kind == "WeightUpdate":
                    data = json.loads(payload_json)
                    path = data.pop("path", None)
                    update = WeightUpdate(**data)
                    started = time.perf_counter()
                    # Only an explicitly fenced engine has a barrier to time.
                    lifecycle = self.server.lifecycle
                    barrier_ms = 0
                    if lifecycle is not None and lifecycle.snapshot().phase != "running":
                        fenced = time.perf_counter()
                        lifecycle.wait_drained(timeout_ms=5_000)
                        barrier_ms = int((time.perf_counter() - fenced) * 1000)
                    error: str | None = None
                    applying = time.perf_counter()
                    try:
                        self.server.rollout.reload_weights(
                            path, expected_digest=update.digest
                        )
                    except Exception as exc:  # reported in the ack, then raised
                        error = str(exc)
                    apply_ms = int((time.perf_counter() - applying) * 1000)
                    total_ms = int((time.perf_counter() - started) * 1000)
                    transfer_ms = max(0, total_ms - apply_ms - barrier_ms)
                    self._send_frame(
                        encode(
                            WeightUpdateAck(
                                request_id=update.to_version,
                                ok=error is None,
                                pause_ms=0,
                                barrier_ms=barrier_ms,
                                transfer_ms=transfer_ms,
                                apply_ms=apply_ms,
                                total_ms=total_ms,
                                error=error,
                            )
                        )
                    )
                    if error is not None:
                        raise RuntimeError(error)
                elif kind == "HealthCheck":
                    # Echo the nonce without touching the rollout engine so the
                    # gateway can prove this worker is responsive right now.
                    probe = json.loads(payload_json)["probe"]
                    self._send_frame(
                        native.encode("HealthCheck", json.dumps({"probe": probe, "ok": True}))
                    )
                else:
                    break
                self.request.sendall(ZERO_LENGTH)
        except (EOFError, ConnectionError, OSError):
            return


class FrameTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address,
        rollout: RolloutService,
        lifecycle: EngineLifecycle | None = None,
    ) -> None:
        self.rollout = rollout
        self.lifecycle = lifecycle
        super().__init__(address, FrameHandler)


class FrameWorker:
    def __init__(
        self,
        rollout: RolloutService,
        host: str = "127.0.0.1",
        port: int = 0,
        lifecycle: EngineLifecycle | None = None,
    ) -> None:
        self.server = FrameTCPServer((host, port), rollout, lifecycle)
        self.thread: threading.Thread | None = None

    @property
    def address(self) -> str:
        host, port = self.server.server_address[:2]
        return f"{host}:{port}"

    def start(self) -> str:
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.address

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
