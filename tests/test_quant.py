"""FP8/NVFP4 quantized weight deltas and stacked-expert wire format."""

import math

import numpy as np
import pytest
import tensorplay as tp

from megarl.quant import (
    QUANT_MODES,
    QuantConfig,
    apply_quantized_delta,
    dequantize_snapshot,
    deserialize_quantized_delta,
    quantize_snapshot,
    quantized_delta,
    serialize_quantized_delta,
    split_quantized_experts,
    stack_quantized_experts,
)
from megarl.weights import DigestMismatch, TensorState, VersionMismatch, WeightSnapshot


def snapshot(values, version=1, dtype="float32"):
    tensor = tp.tensor(values, dtype=tp.float32).to(getattr(tp, dtype))
    data = bytes(tensor.view(tp.uint8).numpy().tobytes()) if dtype == "float8_e4m3fn" else bytes(tensor.numpy().tobytes())
    return WeightSnapshot.from_tensors(
        {"w": TensorState("w", tuple(tensor.shape), dtype, data)}, version
    )


def as_numpy(state: WeightSnapshot) -> np.ndarray:
    return np.frombuffer(state.tensors["w"].data, dtype=np.float32).reshape(state.tensors["w"].shape)


@pytest.mark.parametrize("mode", ["fp8_e4m3", "fp8_e5m2"])
def test_fp8_roundtrip_has_bounded_error(mode):
    values = [[0.1, 1.5, -2.25, 10.0], [0.001, -0.5, 3.3, 7.7]]
    original = as_numpy(snapshot(values))
    quantized = quantize_snapshot(snapshot(values), QuantConfig(mode=mode, block=(1, 4)))
    restored = as_numpy(dequantize_snapshot(quantized))
    assert restored.shape == original.shape
    assert np.isfinite(restored).all()
    assert np.abs(restored - original).max() < 0.5


def test_nvfp4_roundtrip_uses_e2m1_and_block_scales():
    positives = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.5] * 2
    values = [positives, [-v for v in positives]]
    quantized = quantize_snapshot(snapshot(values), QuantConfig(mode="nvfp4"))
    tensor = quantized.tensors["w"]
    assert tensor.mode == "nvfp4"
    # Two E2M1 nibbles per byte (32 values -> 16 bytes), one E4M3 scale per block.
    assert tensor.q_len == 16
    assert len(tensor.payload) - tensor.q_len == 2
    restored = as_numpy(dequantize_snapshot(quantized))
    assert np.isfinite(restored).all()
    assert np.abs(restored - np.asarray(values, dtype=np.float32)).max() < 0.05


def test_none_mode_is_byte_exact():
    state = snapshot([[1.0, 2.0], [3.0, 4.0]])
    quantized = quantize_snapshot(state, QuantConfig(mode="none", block=(1, 2)))
    assert dequantize_snapshot(quantized).tensors["w"].data == state.tensors["w"].data


def test_quantized_delta_diffs_only_changed_bytes_and_verifies_digest():
    first = snapshot([[1.0] * 128], 1)
    second = snapshot([[1.0] * 128], 2)
    changed = tp.tensor([[1.0] * 128])
    changed[0, 64] = 2.0
    second = snapshot(changed.tolist(), 2)
    cfg = QuantConfig(mode="fp8_e4m3", block=(1, 16))
    old_q = quantize_snapshot(first, cfg)
    new_q = quantize_snapshot(second, cfg)
    delta = quantized_delta(old_q, new_q)
    assert delta.from_version == 1 and delta.to_version == 2
    assert delta.changed_bytes > 0
    assert delta.changed_bytes < len(new_q.tensors["w"].payload)

    frame = serialize_quantized_delta(delta)
    restored = deserialize_quantized_delta(frame)
    assert restored == delta
    applied = apply_quantized_delta(old_q, restored)
    assert applied.digest == new_q.digest
    assert applied.version == 2

    corrupted = bytearray(frame)
    corrupted[-1] ^= 0xFF
    with pytest.raises(DigestMismatch):
        deserialize_quantized_delta(bytes(corrupted))
    with pytest.raises(VersionMismatch):
        apply_quantized_delta(new_q, delta)


def test_stacked_expert_shards_roundtrip_without_dequantization():
    experts, hidden = 8, 32
    values = np.arange(experts * hidden, dtype=np.float32).reshape(experts, hidden) / 100.0
    state = WeightSnapshot.from_tensors(
        {"experts": TensorState("experts", (experts, hidden), "float32", values.tobytes())},
        version=1,
    )
    cfg = QuantConfig(mode="fp8_e4m3", block=(1, 16), num_experts=experts)
    whole = quantize_snapshot(state, cfg).tensors["experts"]
    shards = split_quantized_experts(whole, experts, ep_size=4)
    assert sorted(shards) == [0, 1, 2, 3]
    for rank, shard in shards.items():
        assert shard.shape == (2, hidden)
        assert (shard.expert_start, shard.expert_end) == (rank * 2, rank * 2 + 2)
    merged = stack_quantized_experts(list(shards.values()), experts, "experts")
    assert merged.payload == whole.payload
    assert merged.q_len == whole.q_len
    assert merged.num_experts == experts
    assert set(QUANT_MODES) >= {"none", "fp8_e4m3", "fp8_e5m2", "nvfp4"}


def test_quantization_rejects_bad_layouts():
    state = snapshot([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        quantize_snapshot(state, QuantConfig(mode="fp8_e4m3", block=(1, 2)))
    with pytest.raises(ValueError):
        QuantConfig(mode="bogus")
    good = quantize_snapshot(state, QuantConfig(mode="nvfp4"))
    other = quantize_snapshot(snapshot([[1.0, 2.0, 4.0]]), QuantConfig(mode="nvfp4"))
    delta = quantized_delta(good, other)
    frame = serialize_quantized_delta(delta)
    assert deserialize_quantized_delta(frame).to_version == other.version
    assert math.isfinite(float(other.version))
