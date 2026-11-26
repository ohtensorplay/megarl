"""RDMA weight transport with tensor-parallel shard mapping.

Deltas are planned per rank: a global changed byte range is split across the
ranks that own it, then written into registered memory regions with one-sided
write semantics. No rank stages a full tensor. `RdmaBuffer` is backed by the
Rust transport provider; a real NIXL/ibverbs provider replaces it behind the
same interface.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import socket
import socketserver
import struct
import threading
from typing import Callable

from . import megarl_rs as native
from .weights import TensorState, WeightDelta, WeightSnapshot

WriteSink = Callable[["RemoteWrite"], None]


@dataclasses.dataclass(frozen=True, slots=True)
class RemoteWrite:
    tensor: str
    rank: int
    global_offset: int
    local_offset: int
    data: bytes


class RdmaWeightTransport:
    def __init__(
        self,
        layouts: dict[str, object],
        buffers: dict[str, list],
    ) -> None:
        self.layouts = layouts
        self.buffers = buffers

    def plan(self, delta: WeightDelta) -> list[RemoteWrite]:
        writes: list[RemoteWrite] = []
        for change in delta.changes:
            layout = self.layouts.get(change.name)
            if layout is None:
                raise KeyError(f"no tensor-parallel layout for {change.name}")
            for offset, chunk in change.ranges:
                for rank, global_start, local_offset, length in layout.map_range(
                    offset, len(chunk)
                ):
                    start = global_start - offset
                    writes.append(
                        RemoteWrite(
                            tensor=change.name,
                            rank=int(rank),
                            global_offset=int(global_start),
                            local_offset=int(local_offset),
                            data=chunk[start : start + length],
                        )
                    )
        return writes

    def apply(self, delta: WeightDelta) -> list[RemoteWrite]:
        """Write directly into the registered local regions."""
        writes = self.plan(delta)
        for write in writes:
            self.buffers[write.tensor][write.rank].write(
                write.local_offset, write.data
            )
        return writes

    def send(
        self,
        delta: WeightDelta,
        destination: str | WriteSink | None = None,
    ) -> list[RemoteWrite]:
        writes = self.apply(delta)
        if destination is None:
            return writes
        if callable(destination):
            for write in writes:
                destination(write)
            return writes
        host, port = destination.rsplit(":", 1)
        frames = [
            json.dumps(
                {
                    "tensor": w.tensor,
                    "rank": w.rank,
                    "local_offset": w.local_offset,
                    "data": base64.b64encode(w.data).decode(),
                }
            ).encode()
            for w in writes
        ]
        with socket.create_connection((host, int(port)), timeout=10) as sock:
            for frame in frames:
                sock.sendall(struct.pack(">I", len(frame)) + frame)
            sock.sendall(struct.pack(">I", 0))
            for _ in frames:  # wait for every one-sided write to complete
                if sock.recv(1) != b"\x01":
                    raise RuntimeError("RDMA write completion failed")
        return writes


class _WriteHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        while True:
            head = self.request.recv(4)
            if len(head) < 4:
                break
            length = struct.unpack(">I", head)[0]
            if length == 0:
                break
            chunks = bytearray()
            while len(chunks) < length:
                part = self.request.recv(length - len(chunks))
                if not part:
                    return
                chunks.extend(part)
            write = json.loads(chunks)
            self.server.buffers[write["tensor"]][write["rank"]].write(
                int(write["local_offset"]), base64.b64decode(write["data"])
            )
            self.request.sendall(b"\x01")  # write completion


class RdmaTarget:
    """Receiving side: registered regions owned by rollout ranks."""

    def __init__(self, buffers: dict[str, list], host: str = "127.0.0.1", port: int = 0):
        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server((host, port), _WriteHandler)
        self.server.buffers = buffers
        self.thread: threading.Thread | None = None

    def start(self) -> str:
        host, port = self.server.server_address[:2]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


def shard_tensor(state: TensorState, tp_size: int):
    """Create one registered region per rank for a tensor's local shard."""
    elements = 1
    for dim in state.shape:
        elements *= dim
    if elements % tp_size:
        raise ValueError("tensor size must be divisible by tp_size")
    per_rank = elements // tp_size
    layout = native.TpLayout(state.name, list(state.shape), tp_size, 0)
    buffers = [
        native.RdmaBuffer(per_rank) for _ in range(tp_size)
    ]
    return layout, buffers


def rebuild_from_shards(
    name: str, shape: tuple[int, ...], dtype: str, buffers: list
) -> WeightSnapshot:
    payload = b"".join(buffer.read() for buffer in buffers)
    return WeightSnapshot.from_tensors(
        {name: TensorState(name, shape, dtype, payload)}
    )
