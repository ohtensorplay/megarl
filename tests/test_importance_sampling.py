"""Train/inference logprob mismatch correction (TRL-style IS modes)."""

import math

import pytest
import tensorplay as tp

from megarl.batching import collate, from_trajectory
from megarl.contracts import TrajectoryRecord
from megarl.corrections import importance_correction
from megarl.losses import TensorBatch, grpo_loss


def test_token_correction_neutralizes_nan_and_respects_clip():
    sampling = tp.zeros((1, 3), dtype=tp.float32)
    behavior = tp.tensor([[0.1, float("nan"), 0.5]], dtype=tp.float32)
    weights = importance_correction(
        sampling, behavior, mode="token_truncate", clip_min=0.5, clip_max=1.5
    ).flatten().tolist()
    assert abs(weights[0] - math.exp(0.1)) < 1e-4
    assert weights[1] == 1.0  # NaN never poisons the importance ratio
    assert abs(weights[2] - 1.5) < 1e-4  # clipped to the upper bound

    masked = importance_correction(
        sampling, behavior, mode="token_mask", clip_min=0.5, clip_max=1.5
    ).flatten().tolist()
    assert masked[2] == 0.0  # out-of-range tokens are removed instead


def test_sequence_correction_requires_and_uses_mask():
    sampling = tp.zeros((1, 2), dtype=tp.float32)
    behavior = tp.tensor([[0.2, 0.2]], dtype=tp.float32)
    mask = tp.tensor([[1.0, 0.0]], dtype=tp.float32)
    with pytest.raises(ValueError, match="token mask"):
        importance_correction(sampling, behavior, mode="sequence_truncate")
    truncated = importance_correction(
        sampling, behavior, mode="sequence_truncate", mask=mask, clip_max=1.1
    ).flatten().tolist()
    assert abs(truncated[0] - 1.1) < 1e-4


def _batch(with_sampling: bool):
    records = [
        TrajectoryRecord(
            1, 2, [3, 4], [-0.2, -0.4], [], [0, 0], 1.0, 0, None,
            sampling_logprobs=[-0.5, -0.5] if with_sampling else [],
        ),
        TrajectoryRecord(
            2, 2, [5, 6], [-0.1, -0.3], [], [0, 0], 0.0, 0, None,
            sampling_logprobs=[-0.1, -0.3] if with_sampling else [],
        ),
    ]
    return collate([from_trajectory(record, [1, 2]) for record in records])


def test_sampling_logprobs_flow_into_batch_and_loss():
    tensors = TensorBatch(_batch(with_sampling=True))
    assert tensors.sampling_logprobs is not None
    logits = tp.zeros((2, 4, 8), dtype=tp.float32)
    advantages = tp.tensor([-1.0, 1.0], dtype=tp.float32)
    _, plain = grpo_loss(logits, tensors, advantages)
    _, corrected = grpo_loss(logits, tensors, advantages, is_mode="token_truncate")
    assert "is_correction_mean" in corrected
    assert not math.isclose(
        float(plain["mean_ratio"]), float(corrected["mean_ratio"]), abs_tol=1e-6
    )


def test_correction_without_sampling_logprobs_is_rejected():
    tensors = TensorBatch(_batch(with_sampling=False))
    assert tensors.sampling_logprobs is None
    with pytest.raises(ValueError, match="sampling_logprobs"):
        grpo_loss(
            tp.zeros((2, 4, 8), dtype=tp.float32),
            tensors,
            tp.tensor([-1.0, 1.0], dtype=tp.float32),
            is_mode="token_mask",
        )
