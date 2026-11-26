"""Quantized weight-delta codec: FP8 and NVFP4 stacked-expert wire format.

Training keeps master weights in floating point; rollout engines keep MoE
experts stacked along dim 0 in FP8 (E4M3/E5M2) or NVFP4 (E2M1). Shipping FP16
bytes between them wastes the bandwidth this project exists to save. This module
quantizes a :class:`~megarl.weights.WeightSnapshot` into that final rollout
coordinate system, diffs in quantized space (so unchanged blocks cost nothing),
and streams a versioned, checksummed ``MGDQ`` frame.

Nothing here is a stub: FP8 uses TensorPlay's native blockwise scale kernels;
NVFP4 uses the E2M1 value table with E4M3 block scales and a per-tensor global
scale. Round-trip error is bounded and measured by tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import struct
from typing import Sequence

import tensorplay as tp

from .weights import DigestMismatch, VersionMismatch, _diff_bytes

QUANT_MODES = ("none", "fp8_e4m3", "fp8_e5m2", "nvfp4")
FRAME_MAGIC = b"MGDQ"
FRAME_VERSION = 1

# NVFP4 E2M1 decode table: the high bit is the sign, 3 low bits the magnitude.
_NVFP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
_NVFP4_BLOCK = 16
_NVFP4_MAX = 6.0

_DTYPES = {
    "float32": tp.float32, "f32": tp.float32, "F32": tp.float32,
    "float16": tp.float16, "f16": tp.float16, "F16": tp.float16,
    "bfloat16": tp.bfloat16, "bf16": tp.bfloat16, "BF16": tp.bfloat16,
    "uint8": tp.uint8, "u8": tp.uint8, "U8": tp.uint8,
    "int8": tp.int8, "i8": tp.int8,
    "int16": tp.int16, "i16": tp.int16,
    "int32": tp.int32, "i32": tp.int32,
    "int64": tp.int64, "i64": tp.int64,
}


def _dtype(name: str):
    try:
        return _DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype for quantization: {name}") from exc


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class QuantConfig:
    """Quantization recipe applied to every tensor in a snapshot.

    ``block`` must have one entry per tensor dimension for FP8 (TensorPlay's
    blockwise scale kernel requires divisibility along every dimension). NVFP4
    uses fixed 16-element blocks in flattened rollout order. ``num_experts``
    marks a stacked-expert tensor; its shards keep their expert window in the
    wire metadata so the rollout can place bytes without dequantizing.
    """

    mode: str = "fp8_e4m3"
    block: tuple[int, ...] = (1, 128)
    num_experts: int | None = None
    ep_rank: int = 0

    def __post_init__(self) -> None:
        if self.mode not in QUANT_MODES:
            raise ValueError(f"unknown quantization mode: {self.mode}")
        if any(size <= 0 for size in self.block):
            raise ValueError("block sizes must be positive")
        if self.num_experts is not None and self.num_experts <= 0:
            raise ValueError("num_experts must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class QuantizedTensor:
    name: str
    shape: tuple[int, ...]
    source_dtype: str
    mode: str
    block: tuple[int, ...]
    payload: bytes
    """Quantized bytes followed by scale bytes."""
    q_len: int
    """Length of the quantized-byte prefix (the rest is scale data)."""
    global_scale: float = 1.0
    """Per-tensor scale shared by NVFP4 blocks; 1.0 for other modes."""
    num_experts: int | None = None
    ep_rank: int = 0
    expert_start: int = 0
    expert_end: int = 0

    @property
    def digest(self) -> str:
        return _digest(self.payload)


@dataclasses.dataclass(frozen=True, slots=True)
class QuantizedSnapshot:
    tensors: dict[str, QuantizedTensor]
    digest: str
    version: int = 0

    @classmethod
    def from_tensors(
        cls, tensors: dict[str, QuantizedTensor], version: int = 0
    ) -> QuantizedSnapshot:
        digest = hashlib.sha256(
            "".join(f"{n}:{t.digest}" for n, t in sorted(tensors.items())).encode()
        ).hexdigest()
        return cls(tensors=tensors, digest=digest, version=version)


@dataclasses.dataclass(frozen=True, slots=True)
class QuantizedDelta:
    from_version: int
    to_version: int
    changes: tuple[tuple[str, tuple[tuple[int, bytes], ...]], ...]
    digest: str

    @property
    def changed_bytes(self) -> int:
        return sum(len(chunk) for _, ranges in self.changes for _, chunk in ranges)


# --- quantization primitives ----------------------------------------------


def _encode_fp8_scalar(value: float) -> bytes:
    """Encode a float as one E4M3 byte using the native cast."""
    tensor = tp.tensor([float(value)], dtype=tp.float32).to(tp.float8_e4m3fn)
    return bytes(tensor.view(tp.uint8).numpy().tobytes())


def _decode_fp8_scalar(byte: int) -> float:
    tensor = tp.frombuffer(bytes([byte]), dtype=tp.uint8).view(tp.float8_e4m3fn)
    return float(tensor.to(tp.float32).tolist()[0])


def _nvfp4_index(value: float) -> int:
    """Nearest E2M1 code for a value already divided by its block scale."""
    best = 0
    best_error = math.inf
    for index, candidate in enumerate(_NVFP4_VALUES):
        error = abs(candidate - value)
        if error < best_error:
            best, best_error = index, error
    return best


def _quantize_nvfp4(values: list[float]):
    max_abs = max((abs(v) for v in values), default=0.0)
    global_scale = max_abs / _NVFP4_MAX if max_abs > 0 else 1.0
    qbytes = bytearray()
    scale_bytes = bytearray()
    for start in range(0, len(values), _NVFP4_BLOCK):
        block = values[start : start + _NVFP4_BLOCK]
        block_max = max((abs(v) for v in block), default=0.0)
        ratio = (block_max / _NVFP4_MAX / global_scale) if global_scale > 0 else 0.0
        scale_byte = _encode_fp8_scalar(min(ratio, 448.0))[0]
        scale = _decode_fp8_scalar(scale_byte) * global_scale
        if scale <= 0:
            scale = global_scale or 1.0
        codes = [_nvfp4_index(v / scale) for v in block]
        codes += [0] * (_NVFP4_BLOCK - len(codes))  # pad the final block
        for i in range(0, len(codes), 2):
            qbytes.append(codes[i] | (codes[i + 1] << 4))
        scale_bytes.append(scale_byte)
    return bytes(qbytes), bytes(scale_bytes), global_scale


def _dequantize_nvfp4(qbytes: bytes, scale_bytes: bytes, global_scale: float, shape) -> tp.Tensor:
    values: list[float] = []
    for index, packed in enumerate(qbytes):
        # Two nibbles per byte, 16 values per block => 8 bytes per scale.
        block_scale = _decode_fp8_scalar(scale_bytes[(index * 2) // _NVFP4_BLOCK]) * global_scale
        for code in (packed & 0x0F, packed >> 4):
            values.append(_NVFP4_VALUES[code] * block_scale)
    count = math.prod(shape)
    return tp.tensor(values[:count], dtype=tp.float32).reshape(tuple(shape))


def _source_bytes_to_tensor(state) -> tp.Tensor:
    return tp.frombuffer(state.data, dtype=_dtype(state.dtype)).reshape(tuple(state.shape))


def quantize_tensor(state, config: QuantConfig) -> QuantizedTensor:
    """Quantize one tensor state into rollout-ready bytes."""
    experts = config.num_experts
    ep_rank = config.ep_rank
    start, end = 0, 0
    if experts is not None and state.shape:
        per_rank = state.shape[0]
        start = ep_rank * per_rank
        end = min(experts, start + per_rank)
        if end <= start:  # full stacked tensor on rank 0
            start, end = 0, experts
    if config.mode == "none":
        qbytes, scale_bytes, global_scale = state.data, b"", 1.0
    elif config.mode in ("fp8_e4m3", "fp8_e5m2"):
        source = _source_bytes_to_tensor(state)
        block = list(config.block)
        if len(block) != len(state.shape):
            raise ValueError("block rank must match tensor rank for FP8")
        if any(dim % size for dim, size in zip(state.shape, block)):
            raise ValueError("tensor shape must be divisible by the FP8 block on every dimension")
        fp8_dtype = tp.float8_e4m3fn if config.mode == "fp8_e4m3" else tp.float8_e5m2
        scale = tp.choose_scale_float8(source, block, fp8_dtype)
        quantized = tp.quantize_affine_float8(source, scale, fp8_dtype)
        qbytes = bytes(quantized.view(tp.uint8).numpy().tobytes())
        scale_bytes = bytes(scale.numpy().tobytes())
        global_scale = 1.0
    else:  # nvfp4
        source = _source_bytes_to_tensor(state).to(tp.float32)
        qbytes, scale_bytes, global_scale = _quantize_nvfp4(source.flatten().tolist())
    return QuantizedTensor(
        name=state.name,
        shape=tuple(state.shape),
        source_dtype=state.dtype,
        mode=config.mode,
        block=tuple(config.block),
        payload=qbytes + scale_bytes,
        q_len=len(qbytes),
        global_scale=global_scale,
        num_experts=experts,
        ep_rank=ep_rank,
        expert_start=start,
        expert_end=end,
    )


def quantize_snapshot(snapshot, config: QuantConfig) -> QuantizedSnapshot:
    tensors = {
        name: quantize_tensor(state, config)
        for name, state in snapshot.tensors.items()
    }
    return QuantizedSnapshot.from_tensors(tensors, snapshot.version)


def dequantize_tensor(quantized: QuantizedTensor):
    """Reconstruct a TensorState in the original dtype (lossy by design)."""
    qbytes = quantized.payload[: quantized.q_len]
    scale_bytes = quantized.payload[quantized.q_len :]
    if quantized.mode == "none":
        data = qbytes
    elif quantized.mode in ("fp8_e4m3", "fp8_e5m2"):
        fp8_dtype = tp.float8_e4m3fn if quantized.mode == "fp8_e4m3" else tp.float8_e5m2
        q = tp.frombuffer(qbytes, dtype=tp.uint8).view(fp8_dtype).reshape(tuple(quantized.shape))
        scale_shape = tuple(
            dim // size for dim, size in zip(quantized.shape, quantized.block)
        )
        scale = tp.frombuffer(scale_bytes, dtype=tp.float32).reshape(scale_shape)
        data = bytes(
            tp.dequantize_affine_float8(q, scale, _dtype(quantized.source_dtype))
            .numpy()
            .tobytes()
        )
    else:
        data = bytes(
            _dequantize_nvfp4(
                qbytes, scale_bytes, quantized.global_scale, quantized.shape
            )
            .to(_dtype(quantized.source_dtype))
            .numpy()
            .tobytes()
        )
    from .weights import TensorState

    return TensorState(
        name=quantized.name,
        shape=quantized.shape,
        dtype=quantized.source_dtype,
        data=data,
    )


def dequantize_snapshot(snapshot: QuantizedSnapshot):
    """Return the rollout-visible WeightSnapshot (lossy reconstruction)."""
    from .weights import WeightSnapshot

    return WeightSnapshot.from_tensors(
        {name: dequantize_tensor(t) for name, t in snapshot.tensors.items()},
        snapshot.version,
    )


# --- quantized deltas and wire format -------------------------------------


def quantized_delta(
    previous: QuantizedSnapshot,
    current: QuantizedSnapshot,
    *,
    to_version: int | None = None,
) -> QuantizedDelta:
    if set(previous.tensors) != set(current.tensors):
        raise ValueError("quantized snapshots have different tensor sets")
    changes: list[tuple[str, tuple[tuple[int, bytes], ...]]] = []
    for name in sorted(current.tensors):
        old, new = previous.tensors[name], current.tensors[name]
        if (old.mode, old.shape, old.q_len, old.num_experts, old.ep_rank) != (
            new.mode, new.shape, new.q_len, new.num_experts, new.ep_rank
        ):
            raise ValueError(f"quantization layout changed for {name}; needs a full reload")
        ranges = _diff_bytes(old.payload, new.payload)
        if ranges:
            changes.append((name, tuple(ranges)))
    return QuantizedDelta(
        from_version=previous.version,
        to_version=current.version if to_version is None else to_version,
        changes=tuple(changes),
        digest=current.digest,
    )


def apply_quantized_delta(
    base: QuantizedSnapshot, delta: QuantizedDelta
) -> QuantizedSnapshot:
    if delta.from_version != base.version:
        raise VersionMismatch(
            f"quantized delta from {delta.from_version}, expected {base.version}"
        )
    tensors = dict(base.tensors)
    for name, ranges in delta.changes:
        if name not in tensors:
            raise ValueError(f"unknown tensor in quantized delta: {name}")
        payload = bytearray(tensors[name].payload)
        for offset, chunk in ranges:
            payload[offset : offset + len(chunk)] = chunk
        tensors[name] = dataclasses.replace(tensors[name], payload=bytes(payload))
    result = QuantizedSnapshot.from_tensors(tensors, delta.to_version)
    if result.digest != delta.digest:
        raise DigestMismatch(
            f"quantized digest mismatch after {delta.from_version}->{delta.to_version}"
        )
    return result


def serialize_quantized_delta(delta: QuantizedDelta) -> bytes:
    digest = delta.digest.encode("ascii")
    out = bytearray(FRAME_MAGIC)
    out += struct.pack("!HQQH", FRAME_VERSION, delta.from_version, delta.to_version, len(digest))
    out += digest
    out += struct.pack("!I", len(delta.changes))
    for name, ranges in delta.changes:
        encoded = name.encode("utf-8")
        out += struct.pack("!H", len(encoded)) + encoded
        out += struct.pack("!I", len(ranges))
        for offset, chunk in ranges:
            out += struct.pack("!QI", offset, len(chunk)) + chunk
    out += hashlib.sha256(out).digest()
    return bytes(out)


def deserialize_quantized_delta(frame: bytes) -> QuantizedDelta:
    if len(frame) < 30 or frame[:4] != FRAME_MAGIC:
        raise ValueError("not a megaRL quantized-delta frame")
    body, checksum = frame[:-32], frame[-32:]
    if hashlib.sha256(body).digest() != checksum:
        raise DigestMismatch("quantized-delta frame checksum mismatch")
    version, from_version, to_version, digest_len = struct.unpack_from("!HQQH", frame, 4)
    if version != FRAME_VERSION:
        raise ValueError(f"unsupported quantized frame version: {version}")
    pos = 4 + 20  # magic + header fields
    digest = frame[pos : pos + digest_len].decode("ascii")
    pos += digest_len
    (count,) = struct.unpack_from("!I", frame, pos)
    pos += 4
    changes: list[tuple[str, tuple[tuple[int, bytes], ...]]] = []
    for _ in range(count):
        (name_len,) = struct.unpack_from("!H", frame, pos)
        pos += 2
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
        changes.append((name, tuple(ranges)))
    if pos != len(body):
        raise ValueError("quantized-delta frame has trailing bytes")
    return QuantizedDelta(from_version, to_version, tuple(changes), digest)


def stack_quantized_experts(
    shards: Sequence[QuantizedTensor], num_experts: int, name: str
) -> QuantizedTensor:
    """Byte-concatenate EP shards into one stacked-expert quantized tensor.

    No dequantization or re-quantization happens: the rollout buffer receives
    exactly the bytes its kernel expects, with one scale per source block.
    """
    if not shards:
        raise ValueError("no quantized expert shards to stack")
    ordered = sorted(shards, key=lambda shard: shard.ep_rank)
    first = ordered[0]
    if any(shard.mode != first.mode or shard.q_len != first.q_len for shard in ordered):
        raise ValueError("incompatible quantized expert shards")
    if any(abs(shard.global_scale - first.global_scale) > 1e-12 for shard in ordered):
        raise ValueError("expert shards have different global scales")
    qbytes = b"".join(shard.payload[: shard.q_len] for shard in ordered)
    scales = b"".join(shard.payload[shard.q_len :] for shard in ordered)
    shape = (num_experts,) + tuple(first.shape[1:])
    return dataclasses.replace(
        first,
        name=name,
        shape=shape,
        payload=qbytes + scales,
        q_len=len(qbytes),
        num_experts=num_experts,
        ep_rank=0,
        expert_start=0,
        expert_end=num_experts,
    )


def split_quantized_experts(
    tensor: QuantizedTensor, num_experts: int, ep_size: int
) -> dict[int, QuantizedTensor]:
    """Split a stacked-expert quantized tensor into shard-local byte windows."""
    if ep_size <= 0 or num_experts <= 0 or num_experts % ep_size:
        raise ValueError("ep_size must divide num_experts")
    if tensor.shape[0] != num_experts:
        raise ValueError("tensor's first dimension must equal num_experts")
    per_rank = num_experts // ep_size
    q_per_expert = tensor.q_len // num_experts
    s_per_expert = (len(tensor.payload) - tensor.q_len) // num_experts
    result: dict[int, QuantizedTensor] = {}
    for rank in range(ep_size):
        start, end = rank * per_rank, (rank + 1) * per_rank
        qbytes = tensor.payload[start * q_per_expert : end * q_per_expert]
        scale_start = tensor.q_len + start * s_per_expert
        scale_end = scale_start + per_rank * s_per_expert
        result[rank] = dataclasses.replace(
            tensor,
            name=f"{tensor.name}.ep{rank}",
            shape=(per_rank,) + tuple(tensor.shape[1:]),
            payload=qbytes + tensor.payload[scale_start:scale_end],
            q_len=len(qbytes),
            ep_rank=rank,
            expert_start=start,
            expert_end=end,
        )
    return result
