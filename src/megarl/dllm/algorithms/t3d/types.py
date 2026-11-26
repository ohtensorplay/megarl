"""T3D, Tunyu Zhang et al., NeurIPS 2026 arXiv 2602.12262; code
https://github.com/Tyrion58/T3D; see package docstring for details."""

from __future__ import annotations

import dataclasses
import io
import json

import numpy as np

from ....contracts import TrajectoryRecord
from ....dataplane import InProcessPayloadStore, PayloadStore


@dataclasses.dataclass(frozen=True, slots=True)
class T3DStep:
    """An intermediate teacher state and its still-masked positions."""

    canvas_ids: list[int]
    mask: list[int]


@dataclasses.dataclass(frozen=True, slots=True)
class T3DRecord:
    """One full-step teacher trajectory plus a reference-model completion."""

    request_id: int
    prompt_ids: list[int]
    completion_ids: list[int]
    steps: tuple[T3DStep, ...]
    decode_steps: list[int]
    ref_completion_ids: list[int]
    reward: float | None = None
    group_id: int = 0
    version: int = 0

    @property
    def length(self) -> int:
        return len(self.completion_ids)

    def validate(self) -> None:
        L = self.length
        if not self.prompt_ids or L <= 0:
            raise ValueError("T3D record needs a prompt and a completion")
        if not self.steps:
            raise ValueError("T3D record needs a teacher trajectory")
        if len(self.decode_steps) != L or len(self.ref_completion_ids) != L:
            raise ValueError("T3D record arrays must align with the completion")
        for step in self.steps:
            if len(step.canvas_ids) != L or len(step.mask) != L:
                raise ValueError("T3D step arrays must align with the completion")
        if self.version < 0:
            raise ValueError("T3D record version cannot be negative")


def encode_t3d(record: T3DRecord) -> bytes:
    record.validate()
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        prompt=np.asarray(record.prompt_ids, dtype=np.int64),
        completion=np.asarray(record.completion_ids, dtype=np.int64),
        canvases=np.asarray([s.canvas_ids for s in record.steps], dtype=np.int64),
        masks=np.asarray([s.mask for s in record.steps], dtype=np.int8),
        decode_steps=np.asarray(record.decode_steps, dtype=np.int64),
        ref_completion=np.asarray(record.ref_completion_ids, dtype=np.int64),
    )
    meta = json.dumps(
        {
            "request_id": record.request_id,
            "reward": record.reward,
            "group_id": record.group_id,
            "version": record.version,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return len(meta).to_bytes(4, "big") + meta + buffer.getvalue()


def decode_t3d(payload: bytes) -> T3DRecord:
    length = int.from_bytes(payload[:4], "big")
    meta = json.loads(payload[4 : 4 + length].decode("utf-8"))
    arrays = np.load(io.BytesIO(payload[4 + length :]))
    steps = tuple(
        T3DStep(canvas_ids=canvas.tolist(), mask=mask.astype(np.int64).tolist())
        for canvas, mask in zip(arrays["canvases"], arrays["masks"])
    )
    return T3DRecord(
        request_id=int(meta["request_id"]),
        prompt_ids=arrays["prompt"].tolist(),
        completion_ids=arrays["completion"].tolist(),
        steps=steps,
        decode_steps=arrays["decode_steps"].tolist(),
        ref_completion_ids=arrays["ref_completion"].tolist(),
        reward=meta["reward"],
        group_id=int(meta["group_id"]),
        version=int(meta["version"]),
    )


class T3DStore:
    """Out-of-band store keyed by ``TrajectoryRecord.payload_handle``."""

    def __init__(self, store: PayloadStore | None = None) -> None:
        self.store = store or InProcessPayloadStore()

    def put(self, record: T3DRecord) -> int:
        return self.store.put(encode_t3d(record))

    def get(self, handle: int) -> T3DRecord:
        return decode_t3d(self.store.get(handle))

    def release(self, handle: int) -> None:
        self.store.release(handle)


@dataclasses.dataclass(frozen=True, slots=True)
class T3DGenerationResult:
    sequences: list[list[int]]
    records: tuple[T3DRecord, ...]
