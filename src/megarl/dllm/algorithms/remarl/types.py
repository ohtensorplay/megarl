"""REMARL types — ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language Models
(Shibaev et al., arXiv:2611.22576; code https://github.com/EgorShibaev/remarl).
"""

from __future__ import annotations

import dataclasses
import io
import json

import numpy as np

from ....contracts import TrajectoryRecord
from ....dataplane import InProcessPayloadStore, PayloadStore


@dataclasses.dataclass(frozen=True, slots=True)
class ReMaRLStep:
    """One stochastic Plackett--Luce position-selection decision."""

    canvas_ids: list[int]
    mask: list[int]
    scores: list[float]
    order: list[int]
    old_logprob: float


@dataclasses.dataclass(frozen=True, slots=True)
class ReMaRLRecord:
    request_id: int
    prompt_ids: list[int]
    completion_ids: list[int]
    steps: tuple[ReMaRLStep, ...]
    reward: float | None = None
    group_id: int = 0
    version: int = 0

    @property
    def length(self) -> int:
        return len(self.completion_ids)

    def validate(self) -> None:
        if not self.prompt_ids or not self.completion_ids:
            raise ValueError("ReMaRL record needs a prompt and a completion")
        L = self.length
        for step in self.steps:
            if len(step.canvas_ids) != L or len(step.mask) != L or len(step.scores) != L:
                raise ValueError("ReMaRL step arrays must align with the completion")
            if any(flag not in (0, 1) for flag in step.mask):
                raise ValueError("ReMaRL mask must be binary")
            if len(set(step.order)) != len(step.order):
                raise ValueError("ReMaRL action order contains duplicate positions")
            if any(not 0 <= index < L for index in step.order):
                raise ValueError("ReMaRL action index outside the completion")
        if self.version < 0:
            raise ValueError("ReMaRL version cannot be negative")


def encode_remarl(record: ReMaRLRecord) -> bytes:
    record.validate()
    buffer = io.BytesIO()
    max_order = max((len(s.order) for s in record.steps), default=0)
    orders = np.full((len(record.steps), max_order), -1, dtype=np.int64)
    for index, step in enumerate(record.steps):
        orders[index, : len(step.order)] = np.asarray(step.order, dtype=np.int64)
    np.savez_compressed(
        buffer,
        prompt=np.asarray(record.prompt_ids, dtype=np.int64),
        completion=np.asarray(record.completion_ids, dtype=np.int64),
        canvases=np.asarray([s.canvas_ids for s in record.steps], dtype=np.int64),
        masks=np.asarray([s.mask for s in record.steps], dtype=np.int8),
        scores=np.asarray([s.scores for s in record.steps], dtype=np.float32),
        orders=orders,
        old_logprobs=np.asarray([s.old_logprob for s in record.steps], dtype=np.float32),
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


def decode_remarl(payload: bytes) -> ReMaRLRecord:
    length = int.from_bytes(payload[:4], "big")
    meta = json.loads(payload[4 : 4 + length].decode("utf-8"))
    arrays = np.load(io.BytesIO(payload[4 + length :]))
    steps = tuple(
        ReMaRLStep(
            canvas_ids=canvas.tolist(),
            mask=mask.tolist(),
            scores=score.tolist(),
            order=[int(v) for v in order if int(v) >= 0],
            old_logprob=float(logprob),
        )
        for canvas, mask, score, order, logprob in zip(
            arrays["canvases"], arrays["masks"], arrays["scores"],
            arrays["orders"], arrays["old_logprobs"],
        )
    )
    return ReMaRLRecord(
        request_id=int(meta["request_id"]),
        prompt_ids=arrays["prompt"].tolist(),
        completion_ids=arrays["completion"].tolist(),
        steps=steps,
        reward=meta["reward"],
        group_id=int(meta["group_id"]),
        version=int(meta["version"]),
    )


class ReMaRLStore:
    """Out-of-band store keyed by ``TrajectoryRecord.payload_handle``."""

    def __init__(self, store: PayloadStore | None = None) -> None:
        self.store = store or InProcessPayloadStore()

    def put(self, record: ReMaRLRecord) -> int:
        return self.store.put(encode_remarl(record))

    def get(self, handle: int) -> ReMaRLRecord:
        return decode_remarl(self.store.get(handle))

    def release(self, handle: int) -> None:
        self.store.release(handle)


@dataclasses.dataclass(frozen=True, slots=True)
class ReMaRLGenerationResult:
    sequences: list[list[int]]
    records: tuple[ReMaRLRecord, ...]
