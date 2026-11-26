"""AGRPO / Amortized GRPO, Anthony Zhan, ICML 2026 arXiv 2510.04019; code
https://github.com/probablyabot/agrpo; see package docstring for details."""

from __future__ import annotations

import dataclasses
import io
import json
import math

import numpy as np

from ....contracts import TrajectoryRecord
from ....dataplane import InProcessPayloadStore, PayloadStore


@dataclasses.dataclass(frozen=True, slots=True)
class AGRPORecord:
    """One denoising trajectory with exact step-level behaviour statistics."""

    request_id: int
    prompt_ids: list[int]
    completion_ids: list[int]
    unmask_steps: list[int]
    token_logprobs: list[float]
    entropies: list[float]
    reward: float | None = None
    group_id: int = 0
    version: int = 0

    @property
    def length(self) -> int:
        return len(self.completion_ids)

    def validate(self) -> None:
        n = self.length
        if not self.prompt_ids or n == 0:
            raise ValueError("AGRPO record needs a prompt and a completion")
        for name in ("unmask_steps", "token_logprobs", "entropies"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"AGRPO record {name} is completion-aligned")
        if any(step < 0 for step in self.unmask_steps):
            raise ValueError("AGRPO record has positions that were never unmasked")
        if any(not math.isfinite(v) for v in self.token_logprobs):
            raise ValueError("AGRPO behaviour log-probs must be finite")
        if self.version < 0:
            raise ValueError("AGRPO record version cannot be negative")


def encode_agrpo(record: AGRPORecord) -> bytes:
    record.validate()
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        prompt=np.asarray(record.prompt_ids, dtype=np.int64),
        completion=np.asarray(record.completion_ids, dtype=np.int64),
        steps=np.asarray(record.unmask_steps, dtype=np.int64),
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


def decode_agrpo(payload: bytes) -> AGRPORecord:
    length = int.from_bytes(payload[:4], "big")
    meta = json.loads(payload[4 : 4 + length].decode("utf-8"))
    arrays = np.load(io.BytesIO(payload[4 + length :]))
    return AGRPORecord(
        request_id=int(meta["request_id"]),
        prompt_ids=arrays["prompt"].tolist(),
        completion_ids=arrays["completion"].tolist(),
        unmask_steps=arrays["steps"].tolist(),
        token_logprobs=arrays["logprobs"].tolist(),
        entropies=arrays["entropies"].tolist(),
        reward=meta["reward"],
        group_id=int(meta["group_id"]),
        version=int(meta["version"]),
    )


class AGRPOStore:
    """Out-of-band store keyed by ``TrajectoryRecord.payload_handle``."""

    def __init__(self, store: PayloadStore | None = None) -> None:
        self.store = store or InProcessPayloadStore()

    def put(self, record: AGRPORecord) -> int:
        return self.store.put(encode_agrpo(record))

    def get(self, handle: int) -> AGRPORecord:
        return decode_agrpo(self.store.get(handle))

    def release(self, handle: int) -> None:
        self.store.release(handle)


@dataclasses.dataclass(frozen=True, slots=True)
class AGRPOGenerationResult:
    sequences: list[list[int]]
    records: tuple[AGRPORecord, ...]
