"""AReaL-style AdamW-inversion delta detection without a stored baseline.

The forward AdamW step here is an independent NumPy reference; the inversion
and the byte-level diff run in Rust. These tests check that reconstructing the
pre-step weights reproduces exactly the delta a pinned snapshot would have
produced — including bf16 visible payloads with fp32 master moments.
"""

import numpy as np
import pytest

from megarl import megarl_rs as native
from megarl.weights import (
    AdamwStepState,
    TensorState,
    WeightSnapshot,
    apply_delta,
    compute_delta_by_inversion,
    mark_inversion_synced,
)

LR, WD, B1, B2, EPS = 1e-3, 0.01, 0.9, 0.999, 1e-8


def _forward(prev, grad, m, v, step, lr=LR, wd=WD, b1=B1, b2=B2, eps=EPS):
    """PyTorch decoupled AdamW forward step, computed independently in NumPy."""
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad * grad
    bc1 = 1 - b1**step
    bc2 = 1 - b2**step
    denom = np.sqrt(v) / np.sqrt(bc2) + eps
    theta = prev * (1 - lr * wd) - (lr / bc1) * m / denom
    return theta.astype(np.float32), m.astype(np.float32), v.astype(np.float32)


def _bf16_bytes(values):
    bits = np.frombuffer(np.asarray(values, dtype=np.float32).tobytes(), dtype=np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16
    return rounded.astype("<u2").tobytes()


def _state(m, v, step):
    return AdamwStepState(
        exp_avg=m.tobytes(),
        exp_avg_sq=v.tobytes(),
        step=step,
        lr=LR,
        weight_decay=WD,
        beta1=B1,
        beta2=B2,
        eps=EPS,
    )


def _assert_delta_reproduces_current(previous, current, delta):
    """The real contract: applying the delta to the actual prior weights yields
    the new weights and digest. Extra ranges are harmless; missing ones are not.
    """
    restored = apply_delta(previous, delta)
    assert restored.digest == current.digest
    for name, tensor in current.tensors.items():
        assert restored.tensors[name].data == tensor.data
    covered = {
        (change.name, offset, offset + len(chunk))
        for change in delta.changes
        for offset, chunk in change.ranges
    }
    for name, tensor in current.tensors.items():
        old = previous.tensors[name].data
        for i, (before, after) in enumerate(zip(old, tensor.data)):
            if before != after and not any(
                n == name and start <= i < end for n, start, end in covered
            ):
                pytest.fail(f"changed byte {i} of {name} is not covered by the delta")


def test_inverted_f32_weights_match_an_independent_adamw_step():
    n = 256
    prev = (np.arange(n, dtype=np.float32) - n / 2) * 0.25
    grad = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    m = np.zeros(n, dtype=np.float32)
    v = np.zeros(n, dtype=np.float32)
    theta, m, v = _forward(prev, grad, m, v, step=1)

    recovered = np.frombuffer(
        bytes(
            native.invert_adamw_bytes(
                theta.tobytes(), m.tobytes(), v.tobytes(), 1, LR, WD, B1, B2, EPS,
                "float32", "float32",
            )
        ),
        dtype=np.float32,
    )
    np.testing.assert_allclose(recovered, prev, rtol=1e-3, atol=1e-4)


def test_inversion_delta_equals_stored_snapshot_delta_fp32():
    n = 256
    prev = (np.arange(n, dtype=np.float32) - n / 2) * 0.25
    grad = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    theta, m, v = _forward(prev, grad, np.zeros(n, np.float32), np.zeros(n, np.float32), 1)

    prev_snap = WeightSnapshot.from_tensors(
        {"w": TensorState("w", (n,), "float32", prev.tobytes())}, version=4
    )
    current = WeightSnapshot.from_tensors(
        {"w": TensorState("w", (n,), "float32", theta.tobytes())}, version=5
    )
    actual = compute_delta_by_inversion(current, {"w": _state(m, v, 1)})

    _assert_delta_reproduces_current(prev_snap, current, actual)
    assert actual.changed_bytes > 0


def test_inversion_delta_equals_stored_snapshot_delta_bf16_with_fp32_moments():
    n = 128
    prev = (np.arange(n, dtype=np.float32) - n / 2) * 0.25
    grad = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    theta, m, v = _forward(prev, grad, np.zeros(n, np.float32), np.zeros(n, np.float32), 7)

    prev_snap = WeightSnapshot.from_tensors(
        {"w": TensorState("w", (n,), "bfloat16", _bf16_bytes(prev))}, version=9
    )
    current = WeightSnapshot.from_tensors(
        {"w": TensorState("w", (n,), "bfloat16", _bf16_bytes(theta))}, version=10
    )
    actual = compute_delta_by_inversion(current, {"w": _state(m, v, 7)})

    _assert_delta_reproduces_current(prev_snap, current, actual)
    assert actual.changed_bytes > 0


def test_detector_prevents_replaying_the_same_update():
    n = 64
    prev = (np.arange(n, dtype=np.float32) - n / 2) * 0.25
    grad = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    theta, m, v = _forward(prev, grad, np.zeros(n, np.float32), np.zeros(n, np.float32), 1)
    current = WeightSnapshot.from_tensors(
        {"w": TensorState("w", (n,), "float32", theta.tobytes())}, version=1
    )
    states = {"w": _state(m, v, 1)}
    detector = native.AdamwInversionDetector()

    first = compute_delta_by_inversion(current, states, detector=detector)
    assert first.changed_bytes > 0
    mark_inversion_synced(current, states, detector)
    second = compute_delta_by_inversion(current, states, detector=detector)
    assert second.changed_bytes == 0
    assert second.changes == ()


def test_invalid_hyperparameters_are_rejected():
    with pytest.raises(ValueError):
        native.invert_adamw_bytes(
            b"\0" * 4, b"\0" * 4, b"\0" * 4,
            0, LR, WD, B1, B2, EPS, "float32", "float32",
        )  # step must be >= 1
