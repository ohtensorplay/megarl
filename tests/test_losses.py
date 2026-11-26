"""GRPO loss on TensorPlay: gradient flows and diagnostics are sane."""

import math

import tensorplay as tp

from megarl.batching import collate, from_trajectory
from megarl.contracts import TrajectoryRecord
from megarl.losses import TensorBatch, grpo_loss, group_advantages, token_logprobs


def batch():
    records = [
        TrajectoryRecord(1, 2, [3, 4], [-0.2, -0.4], [], [0, 0], 1.0, 0, None),
        TrajectoryRecord(2, 2, [5, 6], [-0.1, -0.3], [], [0, 0], 0.0, 0, None),
    ]
    samples = [from_trajectory(r, [1, 2]) for r in records]
    return collate(samples)


def test_group_advantages_are_normalized():
    advantages = group_advantages([[0.0, 1.0]])
    assert len(advantages) == 2
    assert math.isclose(advantages[0], -1.0, abs_tol=1e-3)
    assert math.isclose(advantages[1], 1.0, abs_tol=1e-3)


def test_token_logprobs_shape_and_normalization():
    logits = tp.zeros((2, 4, 8), dtype=tp.float32)
    tensors = TensorBatch(batch())
    logps = token_logprobs(logits, tensors.input_ids)
    assert tuple(logps.shape) == (2, 4)
    assert all(abs(float(v) - math.log(1 / 8)) < 1e-5 for v in logps.flatten().tolist())


def test_grpo_loss_backward_and_clipping_diagnostic():
    tensors = TensorBatch(batch())
    logits = tp.zeros((2, 4, 8), dtype=tp.float32, requires_grad=True)
    advantages = tp.tensor([-1.0, 1.0], dtype=tp.float32)
    loss, diag = grpo_loss(logits, tensors, advantages, clip_ratio=0.2, kl_coef=0.01)
    assert math.isfinite(float(loss))
    loss.backward()
    assert logits.grad is not None
    assert "clip_fraction" in diag and "kl" in diag
    assert float(diag["mean_ratio"]) > 0


def test_old_logprob_gap_changes_ratio():
    tensors = TensorBatch(batch())
    logits = tp.zeros((2, 4, 8), dtype=tp.float32)
    advantages = tp.tensor([1.0, -1.0], dtype=tp.float32)
    _, diag = grpo_loss(logits, tensors, advantages)
    assert float(diag["mean_ratio"]) != 1.0
