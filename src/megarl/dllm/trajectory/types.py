"""Trajectory-level dLLM rollouts and replay (TraceRL ICLR 2026 arXiv 2509.06949;
MaskGRPO arXiv 2510.02880; CJ-GRPO arXiv 2509.23924); see package docstring for
details."""

from __future__ import annotations

import dataclasses
import io
import json

import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class DLLMStep:
    """One reverse transition of a discrete dLLM denoising trajectory."""

    canvas_ids: list[int]
    """Noisy canvas fed to the denoiser at this step, shape [C]."""
    sampled_ids: list[int]
    """Token sampled by the denoiser at each canvas position, shape [C]."""
    logprobs: list[float]
    """Behaviour log-prob of ``sampled_ids`` under the generating policy."""
    accept_mask: list[int]
    """1 where the sampled token was committed this step, shape [C]."""
    timestep: float
    """Scheduler value in [0, 1] (or a raw integer step index)."""
    version: int
    """Policy version that produced this transition."""

    def validate(self, canvas_len: int) -> None:
        for name in ("canvas_ids", "sampled_ids", "logprobs", "accept_mask"):
            if len(getattr(self, name)) != canvas_len:
                raise ValueError(f"dLLM step {name} is not canvas-aligned")
        if any(keep not in (0, 1) for keep in self.accept_mask):
            raise ValueError("dLLM step accept_mask must be binary")
        if self.version < 0:
            raise ValueError("dLLM step version cannot be negative")


@dataclasses.dataclass(frozen=True, slots=True)
class DLLMTrajectoryRecord:
    """Full reverse trajectory for one completion, metadata + bulk steps."""

    request_id: int
    prompt_ids: list[int]
    completion_ids: list[int]
    steps: tuple[DLLMStep, ...]
    reward: float | None = None
    group_id: int = 0

    @property
    def canvas_len(self) -> int:
        return len(self.steps[0].canvas_ids) if self.steps else len(self.completion_ids)

    def validate(self) -> None:
        if not self.prompt_ids:
            raise ValueError("dLLM trajectory needs a prompt")
        if not self.steps:
            raise ValueError("dLLM trajectory needs at least one denoising step")
        for step in self.steps:
            step.validate(self.canvas_len)


def encode_trajectory(record: DLLMTrajectoryRecord) -> bytes:
    """Serialise bulk step arrays with numpy; metadata stays tiny JSON."""
    record.validate()
    steps = record.steps
    width = record.canvas_len
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        prompt=np.asarray(record.prompt_ids, dtype=np.int64),
        completion=np.asarray(record.completion_ids, dtype=np.int64),
        canvas=np.asarray([s.canvas_ids for s in steps], dtype=np.int64),
        sampled=np.asarray([s.sampled_ids for s in steps], dtype=np.int64),
        logprobs=np.asarray([s.logprobs for s in steps], dtype=np.float32),
        accept=np.asarray([s.accept_mask for s in steps], dtype=np.uint8),
        timesteps=np.asarray([s.timestep for s in steps], dtype=np.float32),
        versions=np.asarray([s.version for s in steps], dtype=np.int64),
    )
    meta = json.dumps(
        {
            "request_id": record.request_id,
            "reward": record.reward,
            "group_id": record.group_id,
            "steps": len(steps),
            "canvas_len": width,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return len(meta).to_bytes(4, "big") + meta + buffer.getvalue()


def decode_trajectory(payload: bytes) -> DLLMTrajectoryRecord:
    length = int.from_bytes(payload[:4], "big")
    meta = json.loads(payload[4 : 4 + length].decode("utf-8"))
    arrays = np.load(io.BytesIO(payload[4 + length :]))
    steps = tuple(
        DLLMStep(
            canvas_ids=arrays["canvas"][i].tolist(),
            sampled_ids=arrays["sampled"][i].tolist(),
            logprobs=arrays["logprobs"][i].tolist(),
            accept_mask=arrays["accept"][i].tolist(),
            timestep=float(arrays["timesteps"][i]),
            version=int(arrays["versions"][i]),
        )
        for i in range(meta["steps"])
    )
    return DLLMTrajectoryRecord(
        request_id=int(meta["request_id"]),
        prompt_ids=arrays["prompt"].tolist(),
        completion_ids=arrays["completion"].tolist(),
        steps=steps,
        reward=meta["reward"],
        group_id=int(meta["group_id"]),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TrajectoryGenerationConfig:
    max_new_tokens: int = 64
    steps: int = 8
    temperature: float = 1.0
    mask_token_id: int = 126336
    pad_token_id: int = 0
    eos_token_ids: tuple[int, ...] = (1,)
    vocab_size: int = 32000
    seed: int = 0
    version: int = 0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0 or self.steps <= 0:
            raise ValueError("max_new_tokens and steps must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class TrajectoryGenerationResult:
    sequences: list[list[int]]
    trajectories: tuple[DLLMTrajectoryRecord, ...]
