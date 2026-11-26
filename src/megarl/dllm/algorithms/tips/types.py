"""TIPS types — TIPS: A Tiny Inference-time Policy for Scaling in Diffusion Language Models
(Zhang et al., arXiv:2610.22099; code https://github.com/zhengyz97/TIPS).
"""

from __future__ import annotations

import dataclasses
import io
import json

import numpy as np

from ....contracts import TrajectoryRecord
from ....dataplane import InProcessPayloadStore, PayloadStore


@dataclasses.dataclass(frozen=True, slots=True)
class TIPSStep:
    """One stochastic visibility decision at a denoising step."""

    canvas_ids: list[int]
    mask: list[int]
    features: list[list[float]]
    time: float
    rho: list[float]
    actions: list[int]
    old_logprob: float


@dataclasses.dataclass(frozen=True, slots=True)
class TIPSRecord:
    request_id: int
    prompt_ids: list[int]
    completion_ids: list[int]
    steps: tuple[TIPSStep, ...]
    reward: float | None = None
    group_id: int = 0
    version: int = 0

    @property
    def length(self) -> int:
        return len(self.completion_ids)

    def validate(self) -> None:
        if not self.prompt_ids or not self.completion_ids:
            raise ValueError("TIPS record needs a prompt and a completion")
        L = self.length
        for step in self.steps:
            if len(step.canvas_ids) != L or len(step.mask) != L:
                raise ValueError("TIPS step arrays must align with the completion")
            if len(step.features) != L or len(step.rho) != L or len(step.actions) != L:
                raise ValueError("TIPS step feature/action arrays must align")
            if any(flag not in (0, 1) for flag in step.mask):
                raise ValueError("TIPS mask must be binary")
            if any(action not in (0, 1) for action in step.actions):
                raise ValueError("TIPS actions must be binary")
        if self.version < 0:
            raise ValueError("TIPS version cannot be negative")


def encode_tips(record: TIPSRecord) -> bytes:
    record.validate()
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        prompt=np.asarray(record.prompt_ids, dtype=np.int64),
        completion=np.asarray(record.completion_ids, dtype=np.int64),
        canvases=np.asarray([s.canvas_ids for s in record.steps], dtype=np.int64),
        masks=np.asarray([s.mask for s in record.steps], dtype=np.int8),
        features=np.asarray([s.features for s in record.steps], dtype=np.float32),
        times=np.asarray([s.time for s in record.steps], dtype=np.float32),
        rhos=np.asarray([s.rho for s in record.steps], dtype=np.float32),
        actions=np.asarray([s.actions for s in record.steps], dtype=np.int8),
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


def decode_tips(payload: bytes) -> TIPSRecord:
    length = int.from_bytes(payload[:4], "big")
    meta = json.loads(payload[4 : 4 + length].decode("utf-8"))
    arrays = np.load(io.BytesIO(payload[4 + length :]))
    steps = tuple(
        TIPSStep(
            canvas_ids=canvas.tolist(),
            mask=mask.tolist(),
            features=feature.tolist(),
            time=float(time),
            rho=rho.tolist(),
            actions=action.tolist(),
            old_logprob=float(logprob),
        )
        for canvas, mask, feature, time, rho, action, logprob in zip(
            arrays["canvases"], arrays["masks"], arrays["features"], arrays["times"],
            arrays["rhos"], arrays["actions"], arrays["old_logprobs"],
        )
    )
    return TIPSRecord(
        request_id=int(meta["request_id"]),
        prompt_ids=arrays["prompt"].tolist(),
        completion_ids=arrays["completion"].tolist(),
        steps=steps,
        reward=meta["reward"],
        group_id=int(meta["group_id"]),
        version=int(meta["version"]),
    )


class TIPSStore:
    """Out-of-band store keyed by ``TrajectoryRecord.payload_handle``."""

    def __init__(self, store: PayloadStore | None = None) -> None:
        self.store = store or InProcessPayloadStore()

    def put(self, record: TIPSRecord) -> int:
        return self.store.put(encode_tips(record))

    def get(self, handle: int) -> TIPSRecord:
        return decode_tips(self.store.get(handle))

    def release(self, handle: int) -> None:
        self.store.release(handle)


@dataclasses.dataclass(frozen=True, slots=True)
class TIPSGenerationResult:
    sequences: list[list[int]]
    records: tuple[TIPSRecord, ...]
