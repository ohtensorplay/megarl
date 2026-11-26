"""Flow-GRPO arXiv 2505.05470 and DiffNFT (ROLL diffusion pipeline
refs/ROLL/roll/pipeline/diffusion); see package docstring for details."""

from __future__ import annotations

import dataclasses
import io
import json

import numpy as np

from ....dataplane import InProcessPayloadStore, PayloadStore


@dataclasses.dataclass(frozen=True, slots=True)
class FlowStep:
    """One recorded Gaussian reverse transition."""

    timestep: float
    """sigma_t at the start of the step, normalised to [0, 1]."""
    next_timestep: float
    """sigma_{t-1} after the step."""
    log_prob: float
    """Behaviour log-probability of the recorded ``next_latent``."""
    keep: int = 1
    """1 when this step participates in the loss."""

    def validate(self) -> None:
        if not 0.0 <= self.timestep <= 1.0:
            raise ValueError("flow step timestep outside [0, 1]")
        if self.next_timestep > self.timestep + 1e-6:
            raise ValueError("flow step timesteps must decrease")
        if self.keep not in (0, 1):
            raise ValueError("flow step keep must be binary")


@dataclasses.dataclass(frozen=True, slots=True)
class FlowTrajectoryRecord:
    """Full latent trajectory for one completion, metadata + bulk arrays."""

    request_id: int
    latents: tuple[tuple[float, ...], ...]
    """``[K+1, D]`` flattened latents ``x_0 ... x_K``."""
    steps: tuple[FlowStep, ...]
    reward: float | None = None
    group_id: int = 0

    @property
    def latent_dim(self) -> int:
        return len(self.latents[0]) if self.latents else 0

    def validate(self) -> None:
        if len(self.latents) < 2:
            raise ValueError("flow trajectory needs at least two latents")
        width = self.latent_dim
        if width <= 0 or any(len(row) != width for row in self.latents):
            raise ValueError("flow latents must be a rectangular [K+1, D] array")
        if len(self.steps) != len(self.latents) - 1:
            raise ValueError("flow steps must match the latent transitions")
        for step in self.steps:
            step.validate()


def encode_flow_trajectory(record: FlowTrajectoryRecord) -> bytes:
    """Serialise latents/steps with numpy; metadata stays tiny JSON."""
    record.validate()
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        latents=np.asarray(record.latents, dtype=np.float32),
        timesteps=np.asarray([s.timestep for s in record.steps], dtype=np.float32),
        next_timesteps=np.asarray(
            [s.next_timestep for s in record.steps], dtype=np.float32
        ),
        logprobs=np.asarray([s.log_prob for s in record.steps], dtype=np.float32),
        keep=np.asarray([s.keep for s in record.steps], dtype=np.uint8),
    )
    meta = json.dumps(
        {
            "request_id": record.request_id,
            "reward": record.reward,
            "group_id": record.group_id,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return len(meta).to_bytes(4, "big") + meta + buffer.getvalue()


def decode_flow_trajectory(payload: bytes) -> FlowTrajectoryRecord:
    length = int.from_bytes(payload[:4], "big")
    meta = json.loads(payload[4 : 4 + length].decode("utf-8"))
    arrays = np.load(io.BytesIO(payload[4 + length :]))
    steps = tuple(
        FlowStep(
            timestep=float(arrays["timesteps"][i]),
            next_timestep=float(arrays["next_timesteps"][i]),
            log_prob=float(arrays["logprobs"][i]),
            keep=int(arrays["keep"][i]),
        )
        for i in range(len(arrays["timesteps"]))
    )
    return FlowTrajectoryRecord(
        request_id=int(meta["request_id"]),
        latents=tuple(tuple(float(v) for v in row) for row in arrays["latents"]),
        steps=steps,
        reward=meta["reward"],
        group_id=int(meta["group_id"]),
    )


class FlowTrajectoryStore:
    """Out-of-band store for continuous trajectories; wire frames carry handles."""

    def __init__(self, store: PayloadStore | None = None) -> None:
        self.store = store or InProcessPayloadStore()

    def put(self, record: FlowTrajectoryRecord) -> int:
        return self.store.put(encode_flow_trajectory(record))

    def get(self, handle: int) -> FlowTrajectoryRecord:
        return decode_flow_trajectory(self.store.get(handle))

    def release(self, handle: int) -> None:
        self.store.release(handle)


@dataclasses.dataclass(frozen=True, slots=True)
class FlowTrajectoryResult:
    """Generator output: one continuous trajectory per prompt."""

    trajectories: tuple[FlowTrajectoryRecord, ...]
    sequences: list[list[int]] | None = None
