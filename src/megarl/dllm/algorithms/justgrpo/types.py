"""JustGRPO / The Flexibility Trap, Zanlin Ni et al., ICML 2026 arXiv 2601.15165; code
https://github.com/LeapLabTHU/JustGRPO; see package docstring for details."""

from __future__ import annotations

import dataclasses
import io
import json
import math

import numpy as np

from ....contracts import TrajectoryRecord
from ....dataplane import InProcessPayloadStore, PayloadStore


@dataclasses.dataclass(frozen=True, slots=True)
class JustGRPORecord:
    """One AR-order completion with the behaviour log-probs and entropies."""

    request_id: int
    prompt_ids: list[int]
    completion_ids: list[int]
    token_logprobs: list[float]
    entropies: list[float]
    reward: float | None = None
    group_id: int = 0
    version: int = 0

    @property
    def length(self) -> int:
        return len(self.completion_ids)

    def validate(self) -> None:
        if not self.prompt_ids:
            raise ValueError("JustGRPO record needs a non-empty prompt")
        if not self.completion_ids:
            raise ValueError("JustGRPO record needs a non-empty completion")
        if len(self.token_logprobs) != self.length or len(self.entropies) != self.length:
            raise ValueError("JustGRPO record arrays must align with the completion")
        if any(not math.isfinite(v) for v in self.token_logprobs):
            raise ValueError("JustGRPO behaviour log-probs must be finite")
        if self.version < 0:
            raise ValueError("JustGRPO record version cannot be negative")


def encode_justgrpo(record: JustGRPORecord) -> bytes:
    """Compress bulk arrays with numpy; metadata stays a tiny JSON header."""
    record.validate()
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        prompt=np.asarray(record.prompt_ids, dtype=np.int64),
        completion=np.asarray(record.completion_ids, dtype=np.int64),
        logprobs=np.asarray(record.token_logprobs, dtype=np.float32),
        entropies=np.asarray(record.entropies, dtype=np.float32),
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


def decode_justgrpo(payload: bytes) -> JustGRPORecord:
    length = int.from_bytes(payload[:4], "big")
    meta = json.loads(payload[4 : 4 + length].decode("utf-8"))
    arrays = np.load(io.BytesIO(payload[4 + length :]))
    return JustGRPORecord(
        request_id=int(meta["request_id"]),
        prompt_ids=arrays["prompt"].tolist(),
        completion_ids=arrays["completion"].tolist(),
        token_logprobs=arrays["logprobs"].tolist(),
        entropies=arrays["entropies"].tolist(),
        reward=meta["reward"],
        group_id=int(meta["group_id"]),
        version=int(meta["version"]),
    )


class JustGRPOStore:
    """Out-of-band store keyed by ``TrajectoryRecord.payload_handle``."""

    def __init__(self, store: PayloadStore | None = None) -> None:
        self.store = store or InProcessPayloadStore()

    def put(self, record: JustGRPORecord) -> int:
        return self.store.put(encode_justgrpo(record))

    def get(self, handle: int) -> JustGRPORecord:
        return decode_justgrpo(self.store.get(handle))

    def release(self, handle: int) -> None:
        self.store.release(handle)


@dataclasses.dataclass(frozen=True, slots=True)
class JustGRPOGenerationResult:
    sequences: list[list[int]]
    records: tuple[JustGRPORecord, ...]
