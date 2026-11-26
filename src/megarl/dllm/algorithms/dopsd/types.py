"""dOPSD, Phuong Tuan Dat, Qi Li, Xinchao Wang, arXiv 2607.04428; code
https://github.com/tuandattt/dOPSD; see package docstring for details."""

from __future__ import annotations

import dataclasses
import io
import json

import numpy as np

from ....contracts import TrajectoryRecord
from ....dataplane import InProcessPayloadStore, PayloadStore


@dataclasses.dataclass(frozen=True, slots=True)
class DOPSDStep:
    """A genuine intermediate decoding state and its still-masked positions."""

    canvas_ids: list[int]
    mask: list[int]
    """1 where the position is still masked (predicted, not copied)."""


@dataclasses.dataclass(frozen=True, slots=True)
class DOPSDRecord:
    """One rollout with its full decoding trajectory."""

    request_id: int
    prompt_ids: list[int]
    completion_ids: list[int]
    steps: tuple[DOPSDStep, ...]
    reward: float | None = None
    group_id: int = 0
    version: int = 0

    @property
    def length(self) -> int:
        return len(self.completion_ids)

    def validate(self) -> None:
        if not self.prompt_ids or not self.completion_ids:
            raise ValueError("dOPSD record needs a prompt and a completion")
        if not self.steps:
            raise ValueError("dOPSD record needs at least one trajectory step")
        L = self.length
        for step in self.steps:
            if len(step.canvas_ids) != L or len(step.mask) != L:
                raise ValueError("dOPSD step arrays must align with the completion")
            if any(flag not in (0, 1) for flag in step.mask):
                raise ValueError("dOPSD mask must be binary")
        if self.version < 0:
            raise ValueError("dOPSD record version cannot be negative")


def encode_dopsd(record: DOPSDRecord) -> bytes:
    """Compress trajectory arrays with numpy; metadata stays a tiny JSON header."""
    record.validate()
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        prompt=np.asarray(record.prompt_ids, dtype=np.int64),
        completion=np.asarray(record.completion_ids, dtype=np.int64),
        canvases=np.asarray([s.canvas_ids for s in record.steps], dtype=np.int64),
        masks=np.asarray([s.mask for s in record.steps], dtype=np.int8),
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


def decode_dopsd(payload: bytes) -> DOPSDRecord:
    length = int.from_bytes(payload[:4], "big")
    meta = json.loads(payload[4 : 4 + length].decode("utf-8"))
    arrays = np.load(io.BytesIO(payload[4 + length :]))
    masks = arrays["masks"].astype(np.int64)
    steps = tuple(
        DOPSDStep(canvas_ids=canvas.tolist(), mask=mask.tolist())
        for canvas, mask in zip(arrays["canvases"], masks)
    )
    return DOPSDRecord(
        request_id=int(meta["request_id"]),
        prompt_ids=arrays["prompt"].tolist(),
        completion_ids=arrays["completion"].tolist(),
        steps=steps,
        reward=meta["reward"],
        group_id=int(meta["group_id"]),
        version=int(meta["version"]),
    )


class DOPSDStore:
    """Out-of-band store keyed by ``TrajectoryRecord.payload_handle``."""

    def __init__(self, store: PayloadStore | None = None) -> None:
        self.store = store or InProcessPayloadStore()

    def put(self, record: DOPSDRecord) -> int:
        return self.store.put(encode_dopsd(record))

    def get(self, handle: int) -> DOPSDRecord:
        return decode_dopsd(self.store.get(handle))

    def release(self, handle: int) -> None:
        self.store.release(handle)


@dataclasses.dataclass(frozen=True, slots=True)
class DOPSDGenerationResult:
    sequences: list[list[int]]
    records: tuple[DOPSDRecord, ...]
