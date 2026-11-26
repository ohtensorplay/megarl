"""PURL types — When Should Diffusion Language Models Reconsider Their Commitments? (Zheng
et al., arXiv:2611.11551v1; withdrawn after v1).
"""

from __future__ import annotations

import dataclasses
import io
import json

import numpy as np

from ....contracts import TrajectoryRecord
from ....dataplane import InProcessPayloadStore, PayloadStore


@dataclasses.dataclass(frozen=True, slots=True)
class PURLRecord:
    """A rollout plus its clean reference completion for joint supervision."""

    request_id: int
    prompt_ids: list[int]
    completion_ids: list[int]
    reference_ids: list[int]
    reward: float | None = None
    group_id: int = 0
    version: int = 0

    @property
    def length(self) -> int:
        return len(self.completion_ids)

    def validate(self) -> None:
        if not self.prompt_ids or not self.completion_ids or not self.reference_ids:
            raise ValueError("PURL record needs a prompt, completion and reference")
        if len(self.completion_ids) != len(self.reference_ids):
            raise ValueError("PURL completion/reference lengths must match")
        if self.version < 0:
            raise ValueError("PURL version cannot be negative")


def encode_purl(record: PURLRecord) -> bytes:
    record.validate()
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        prompt=np.asarray(record.prompt_ids, dtype=np.int64),
        completion=np.asarray(record.completion_ids, dtype=np.int64),
        reference=np.asarray(record.reference_ids, dtype=np.int64),
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


def decode_purl(payload: bytes) -> PURLRecord:
    length = int.from_bytes(payload[:4], "big")
    meta = json.loads(payload[4 : 4 + length].decode("utf-8"))
    arrays = np.load(io.BytesIO(payload[4 + length :]))
    return PURLRecord(
        request_id=int(meta["request_id"]),
        prompt_ids=arrays["prompt"].tolist(),
        completion_ids=arrays["completion"].tolist(),
        reference_ids=arrays["reference"].tolist(),
        reward=meta["reward"],
        group_id=int(meta["group_id"]),
        version=int(meta["version"]),
    )


class PURLStore:
    """Out-of-band store keyed by ``TrajectoryRecord.payload_handle``."""

    def __init__(self, store: PayloadStore | None = None) -> None:
        self.store = store or InProcessPayloadStore()

    def put(self, record: PURLRecord) -> int:
        return self.store.put(encode_purl(record))

    def get(self, handle: int) -> PURLRecord:
        return decode_purl(self.store.get(handle))

    def release(self, handle: int) -> None:
        self.store.release(handle)


@dataclasses.dataclass(frozen=True, slots=True)
class PURLGenerationResult:
    sequences: list[list[int]]
    records: tuple[PURLRecord, ...]
