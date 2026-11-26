"""Weight fabric: shard-local sparse deltas with a version + digest handshake.

The dominant cost in reference frameworks is full-checkpoint staging and
rank-0 all-gather. Here the trainer diffs its own shard against a pinned
snapshot and exports only changed byte ranges in final rollout coordinates.
Rollout applies them in place and verifies the new digest.

Transport frames are binary (not JSON-hex): a small header, length-prefixed
ranges and a CRC/SHA integrity check. Diffing is block-indexed so the common
case (1-3% of bytes changed) never loops in Python over the untouched bytes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from pathlib import Path
from typing import Protocol, Sequence

from . import megarl_rs as _native
from .contracts import TensorDescriptor, WeightUpdate

FRAME_MAGIC = b"MGDP"
FRAME_VERSION = 1
_DIFF_BLOCK = 4096


@dataclasses.dataclass(frozen=True, slots=True)
class TensorState:
    name: str
    shape: tuple[int, ...]
    dtype: str
    data: bytes


@dataclasses.dataclass(frozen=True, slots=True)
class WeightSnapshot:
    tensors: dict[str, TensorState]
    digest: str
    version: int = 0

    @classmethod
    def from_tensors(
        cls, tensors: dict[str, TensorState], version: int = 0
    ) -> WeightSnapshot:
        digest = hashlib.sha256(
            b"".join(tensors[name].data for name in sorted(tensors))
        ).hexdigest()
        return cls(tensors=tensors, digest=digest, version=version)


@dataclasses.dataclass(frozen=True, slots=True)
class TensorDelta:
    name: str
    ranges: tuple[tuple[int, bytes], ...]
    shard_rank: int = 0
    # Descriptor hint only; excluded from equality so delta comparisons stay clean.
    tensor_hint: TensorState | None = dataclasses.field(
        default=None, repr=False, compare=False
    )


@dataclasses.dataclass(frozen=True, slots=True)
class WeightDelta:
    from_version: int
    to_version: int
    changes: tuple[TensorDelta, ...]
    digest: str

    @property
    def changed_bytes(self) -> int:
        return sum(len(chunk) for change in self.changes for _, chunk in change.ranges)

    def to_contract(self, *, transport: str, abort_inflight: bool) -> WeightUpdate:
        descriptors: list[TensorDescriptor] = []
        for change in self.changes:
            tensor = change.tensor_hint
            for offset, chunk in change.ranges:
                descriptors.append(
                    TensorDescriptor(
                        name=change.name,
                        shape=list(tensor.shape) if tensor else [],
                        dtype=tensor.dtype if tensor else "unknown",
                        shard_rank=change.shard_rank,
                        byte_offset=offset,
                        byte_length=len(chunk),
                    )
                )
        return WeightUpdate(
            from_version=self.from_version,
            to_version=self.to_version,
            transport=transport,
            digest=self.digest,
            tensors=descriptors,
            abort_inflight=abort_inflight,
        )


def _diff_bytes(old: bytes, new: bytes) -> list[tuple[int, bytes]]:
    """Byte-exact diff, skipping equal 4 KiB blocks at C speed."""
    if len(old) != len(new):
        raise ValueError("tensor changed size; layout change needs recapture")
    ranges: list[tuple[int, bytes]] = []
    for block_start in range(0, len(old), _DIFF_BLOCK):
        block_end = min(block_start + _DIFF_BLOCK, len(old))
        if old[block_start:block_end] == new[block_start:block_end]:
            continue
        start: int | None = None
        for i in range(block_start, block_end):
            if old[i] != new[i]:
                if start is None:
                    start = i
            elif start is not None:
                ranges.append((start, new[start:i]))
                start = None
        if start is not None:
            ranges.append((start, new[start:block_end]))
    return ranges


@dataclasses.dataclass(frozen=True, slots=True)
class AdamwStepState:
    """Post-step Adam moments and the LR/hyperparameters of the completed step.

    `lr` must be the LR actually consumed by that step (recorded at step time),
    never a scheduler's next-step LR. The inversion is silent-wrong otherwise.
    """

    exp_avg: bytes
    exp_avg_sq: bytes
    step: int
    lr: float
    weight_decay: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    # Adam moments stay fp32 even when the visible weight payload is bf16.
    moments_dtype: str = "float32"


def invert_pre_step_bytes(current: TensorState, state: AdamwStepState) -> bytes:
    """Reconstruct the pre-step master weights for one tensor (fp32 in Rust)."""
    return bytes(
        _native.invert_adamw_bytes(
            current.data,
            state.exp_avg,
            state.exp_avg_sq,
            state.step,
            state.lr,
            state.weight_decay,
            state.beta1,
            state.beta2,
            state.eps,
            current.dtype,
            state.moments_dtype,
        )
    )


def compute_delta_by_inversion(
    current: WeightSnapshot,
    optimizer_states: dict[str, AdamwStepState],
    *,
    to_version: int | None = None,
    shard_rank: int = 0,
    detector: _native.AdamwInversionDetector | None = None,
) -> WeightDelta:
    """Diff against a *reconstructed* baseline instead of a stored snapshot.

    For every tensor with optimizer state, Rust inverts the completed AdamW
    step to recover its pre-step weights in the same storage dtype; tensors
    without state are treated as unchanged. The result is fed through the same
    byte-exact `compute_delta` path, so the wire format, digest and receiver
    verification are identical to the snapshot-based flow.
    """
    changes: list[TensorDelta] = []
    for name, tensor in current.tensors.items():
        state = optimizer_states.get(name)
        if state is None:
            continue
        if detector is not None and not detector.should_detect(
            name, tensor.data, tensor.dtype, state.step
        ):
            continue
        raw_ranges = _native.invert_and_diff_bytes(
            tensor.data,
            state.exp_avg,
            state.exp_avg_sq,
            state.step,
            state.lr,
            state.weight_decay,
            state.beta1,
            state.beta2,
            state.eps,
            tensor.dtype,
            state.moments_dtype,
        )
        ranges = tuple((offset, bytes(chunk)) for offset, chunk in raw_ranges)
        if ranges:
            changes.append(
                TensorDelta(
                    name=name,
                    ranges=ranges,
                    shard_rank=shard_rank,
                    tensor_hint=tensor,
                )
            )
    return WeightDelta(
        from_version=current.version - 1,
        to_version=current.version if to_version is None else to_version,
        changes=tuple(changes),
        digest=current.digest,
    )


def mark_inversion_synced(
    current: WeightSnapshot,
    optimizer_states: dict[str, AdamwStepState],
    detector: _native.AdamwInversionDetector,
) -> None:
    """Record a successful commit so an update is not delta-detected twice."""
    for name, state in optimizer_states.items():
        tensor = current.tensors[name]
        detector.mark_synced(name, tensor.data, tensor.dtype, state.step)


def compute_delta(
    previous: WeightSnapshot,
    current: WeightSnapshot,
    *,
    to_version: int | None = None,
    shard_rank: int = 0,
) -> WeightDelta:
    if set(previous.tensors) != set(current.tensors):
        raise ValueError("weight snapshots have different tensor sets")
    changes: list[TensorDelta] = []
    for name in sorted(current.tensors):
        ranges = _diff_bytes(previous.tensors[name].data, current.tensors[name].data)
        if ranges:
            changes.append(
                TensorDelta(
                    name=name,
                    ranges=tuple(ranges),
                    shard_rank=shard_rank,
                    tensor_hint=current.tensors[name],
                )
            )
    return WeightDelta(
        from_version=previous.version,
        to_version=current.version if to_version is None else to_version,
        changes=tuple(changes),
        digest=current.digest,
    )


class DigestMismatch(RuntimeError):
    pass


class VersionMismatch(RuntimeError):
    pass


def apply_delta(base: WeightSnapshot, delta: WeightDelta) -> WeightSnapshot:
    tensors = {name: state for name, state in base.tensors.items()}
    for change in delta.changes:
        data = bytearray(tensors[change.name].data)
        for offset, chunk in change.ranges:
            data[offset : offset + len(chunk)] = chunk
        tensors[change.name] = dataclasses.replace(tensors[change.name], data=bytes(data))
    result = WeightSnapshot(
        tensors=tensors, digest=delta.digest, version=delta.to_version
    )
    if WeightSnapshot.from_tensors(tensors, delta.to_version).digest != delta.digest:
        raise DigestMismatch(
            f"weight digest mismatch after applying delta {delta.from_version}->{delta.to_version}"
        )
    return result


# --- binary wire format ---------------------------------------------------
# MAGIC(4) VER(u16) FROM(u64) TO(u64) DIGEST_LEN(u16) DIGEST(ASCII)
# TENSORS(u32) [NAME_LEN(u16) NAME SHARD(u32) RANGES(u32) [OFFSET(u64) LEN(u32) DATA]]
# CHECKSUM(32) = sha256(all bytes before it)


def serialize_delta(delta: WeightDelta) -> bytes:
    digest = delta.digest.encode("ascii")
    out = bytearray(FRAME_MAGIC)
    out += struct.pack("!HQQH", FRAME_VERSION, delta.from_version, delta.to_version, len(digest))
    out += digest
    out += struct.pack("!I", len(delta.changes))
    for change in delta.changes:
        name = change.name.encode("utf-8")
        out += struct.pack("!HI", len(name), change.shard_rank) + name
        out += struct.pack("!I", len(change.ranges))
        for offset, chunk in change.ranges:
            out += struct.pack("!QI", offset, len(chunk)) + chunk
    out += hashlib.sha256(out).digest()
    return bytes(out)


def deserialize_delta(frame: bytes) -> WeightDelta:
    if len(frame) < 4 + 2 + 8 + 8 + 2 + 32 or frame[:4] != FRAME_MAGIC:
        raise ValueError("not a megaRL weight-delta frame")
    body, checksum = frame[:-32], frame[-32:]
    if hashlib.sha256(body).digest() != checksum:
        raise DigestMismatch("weight-delta frame checksum mismatch")
    pos = 4
    version, from_version, to_version, digest_len = struct.unpack_from("!HQQH", frame, pos)
    pos += 20
    if version != FRAME_VERSION:
        raise ValueError(f"unsupported weight frame version: {version}")
    digest = frame[pos : pos + digest_len].decode("ascii")
    pos += digest_len
    (count,) = struct.unpack_from("!I", frame, pos)
    pos += 4
    changes: list[TensorDelta] = []
    for _ in range(count):
        name_len, shard_rank = struct.unpack_from("!HI", frame, pos)
        pos += 6
        name = frame[pos : pos + name_len].decode("utf-8")
        pos += name_len
        (ranges_count,) = struct.unpack_from("!I", frame, pos)
        pos += 4
        ranges: list[tuple[int, bytes]] = []
        for _ in range(ranges_count):
            offset, length = struct.unpack_from("!QI", frame, pos)
            pos += 12
            ranges.append((offset, frame[pos : pos + length]))
            pos += length
        changes.append(TensorDelta(name, tuple(ranges), shard_rank=shard_rank))
    if pos != len(body):
        raise ValueError("weight-delta frame has trailing bytes")
    return WeightDelta(from_version, to_version, tuple(changes), digest)


def _serialize_delta(delta: WeightDelta) -> dict:
    """Legacy JSON form, kept for tooling compatibility."""
    return {
        "from_version": delta.from_version,
        "to_version": delta.to_version,
        "digest": delta.digest,
        "changes": [
            {"name": c.name, "ranges": [[o, ch.hex()] for o, ch in c.ranges]}
            for c in delta.changes
        ],
    }


def _deserialize_delta(raw: dict) -> WeightDelta:
    return WeightDelta(
        raw["from_version"],
        raw["to_version"],
        tuple(
            TensorDelta(
                c["name"], tuple((o, bytes.fromhex(ch)) for o, ch in c["ranges"])
            )
            for c in raw["changes"]
        ),
        raw["digest"],
    )


class WeightTransport(Protocol):
    def send(self, delta: WeightDelta, destination: str) -> None: ...

    def receive(self, source: str) -> WeightDelta: ...


class FileSystemTransport:
    """Cross-node / heterogeneous path: binary deltas on a shared filesystem."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def send(self, delta: WeightDelta, destination: str) -> None:
        path = self.root / destination
        path.mkdir(parents=True, exist_ok=True)
        target = path / "delta.mgdp"
        tmp = path / ".delta.mgdp.tmp"
        tmp.write_bytes(serialize_delta(delta))
        tmp.replace(target)

    def receive(self, source: str) -> WeightDelta:
        return deserialize_delta((self.root / source / "delta.mgdp").read_bytes())


class SharedMemoryTransport:
    """Same-node cross-process path: binary delta in POSIX shared memory."""

    def send(self, delta: WeightDelta, destination: str) -> None:
        from multiprocessing.shared_memory import SharedMemory

        payload = serialize_delta(delta)
        shm = SharedMemory(name=destination, create=True, size=len(payload))
        try:
            shm.buf[: len(payload)] = payload
        finally:
            shm.close()

    def receive(self, source: str) -> WeightDelta:
        from multiprocessing.shared_memory import SharedMemory

        shm = SharedMemory(name=source, create=False)
        try:
            payload = bytes(shm.buf[: shm.size])
        finally:
            shm.close()
            shm.unlink()
        return deserialize_delta(payload)


class WeightSyncCoordinator:
    """Version handshake between trainer publication and rollout commit.

    Mirrors the production pattern used by verl and AReaL: at most one
    uncommitted version, monotonic version numbers, and digest verification at
    both ends. A bad frame cannot silently advance the rollout version.
    """

    def __init__(self, base: WeightSnapshot) -> None:
        self.base = base
        self.version = base.version
        self.published: WeightDelta | None = None
        self.committed = base.version

    def publish(self, delta: WeightDelta, *, frame: bytes | None = None) -> WeightDelta:
        if self.published is not None:
            raise VersionMismatch("another version is already uncommitted")
        if delta.from_version != self.version:
            raise VersionMismatch(
                f"publish from {delta.from_version}, expected {self.version}"
            )
        if delta.to_version <= self.version:
            raise VersionMismatch(f"non-monotonic version {delta.to_version}")
        if frame is not None and deserialize_delta(frame) != delta:
            raise DigestMismatch("published frame does not match the delta")
        # Verify the advertised digest against the actual byte ranges before
        # accepting the version, not only when the rollout eventually commits.
        apply_delta(self.base, delta)
        self.published = delta
        return delta

    def commit(self) -> WeightSnapshot:
        delta = self.published
        if delta is None:
            raise VersionMismatch("no published version to commit")
        self.base = apply_delta(self.base, delta)
        self.version = delta.to_version
        self.committed = self.version
        self.published = None
        return self.base

    def status(self) -> dict:
        return {
            "base_version": self.version,
            "committed": self.committed,
            "pending": self.published.to_version if self.published else None,
        }
