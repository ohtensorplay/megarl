"""Full policy-gradient loss family: math, normalization and diagnostics."""

import math

import pytest
import tensorplay as tp
import tensorplay.nn.functional as F

from megarl.batching import collate, from_trajectory
from megarl.contracts import TrajectoryRecord
from megarl.losses import LOSS_TYPES, TensorBatch, policy_loss


def make_batch(with_sampling=False, *, old_logprobs=None, sampling=None):
    # Old logprobs sit just below log(1/8) so ratios start inside the clip
    # window; every algorithm therefore has a live gradient. Tests that need
    # saturated ratios build their own batch.
    first_old = old_logprobs[0] if old_logprobs else [-2.0, -2.0, -2.0]
    second_old = old_logprobs[1] if old_logprobs else [-2.0, -2.0]
    first_sampling = sampling[0] if sampling else [-1.9, -1.9, -1.9]
    second_sampling = sampling[1] if sampling else [-1.9, -1.9]
    records = [
        TrajectoryRecord(
            1,
            3,
            [3, 4, 5],
            first_old,
            [],
            [0, 0, 0],
            1.0,
            0,
            None,
            sampling_logprobs=first_sampling if with_sampling else [],
        ),
        TrajectoryRecord(
            2, 2, [5, 6], second_old, [], [0, 0], 0.0, 0, None,
            sampling_logprobs=second_sampling if with_sampling else [],
        ),
    ]
    return collate([from_trajectory(record, [1, 2]) for record in records])


def tensors_and_logits(with_sampling=False, *, old_logprobs=None, sampling=None):
    batch = make_batch(
        with_sampling=with_sampling, old_logprobs=old_logprobs, sampling=sampling
    )
    tensors = TensorBatch(batch)
    logits = tp.zeros((batch.size, batch.max_len, 8), dtype=tp.float32, requires_grad=True)
    return tensors, logits


ADVANTAGES = tp.tensor([1.0, -1.0], dtype=tp.float32)


def test_every_loss_type_is_finite_differentiable_and_instrumented():
    for loss_type in LOSS_TYPES:
        tensors, logits = tensors_and_logits()
        kwargs = {}
        if loss_type == "ppo":
            kwargs = dict(
                value_preds=tp.zeros_like(tensors.loss_mask, dtype=tp.float32),
                returns=tp.ones_like(tensors.loss_mask, dtype=tp.float32),
                value_coef=0.5,
            )
        if loss_type == "luspo":
            kwargs["importance_sampling_level"] = "sequence"
        if loss_type in ("reverse_kl", "opd"):
            kwargs["teacher_logprobs"] = tp.zeros_like(tensors.loss_mask, dtype=tp.float32)
        loss, diag = policy_loss(logits, tensors, ADVANTAGES, loss_type=loss_type, **kwargs)
        assert math.isfinite(float(loss.detach()))
        loss.backward()
        assert logits.grad is not None
        assert float(logits.grad.abs().sum()) > 0
        assert {"loss", "mean_ratio", "clip_fraction", "kl", "mean_token_logprob"} <= set(diag)
        assert all(hasattr(value, "shape") for value in diag.values()), loss_type


def test_dapo_clip_higher_widens_positive_advantage_update():
    tensors, _ = tensors_and_logits(old_logprobs=[[-5.0] * 3, [-5.0] * 2])
    logits = tp.zeros((tensors.input_ids.shape[0], tensors.input_ids.shape[1], 8), dtype=tp.float32)
    symmetric, _ = policy_loss(logits, tensors, ADVANTAGES, epsilon_high=0.2)
    higher, _ = policy_loss(logits, tensors, ADVANTAGES, epsilon_high=0.6)
    # Ratios are far above 1 because old logprobs are -5; the surrogate is
    # (1 + eps_high) * A, so a wider bound yields a smaller positive loss.
    assert float(higher.detach()) < float(symmetric.detach())


def test_cispo_clips_and_detaches_the_ratio():
    tensors, logits = tensors_and_logits()
    loss, diag = policy_loss(logits, tensors, ADVANTAGES, loss_type="cispo", epsilon_high=5.0)
    logp = tp.gather(F.log_softmax(logits, dim=-1), -1, tensors.input_ids.unsqueeze(-1)).squeeze(-1)
    ratio = tp.exp(logp - tensors.old_logprobs).clamp(max=5.0).detach()
    per_token = -ratio * ADVANTAGES.unsqueeze(-1) * logp
    mask = tensors.loss_mask
    expected = (per_token * mask).sum() / mask.sum()
    assert math.isclose(float(loss.detach()), float(expected.detach()), abs_tol=1e-6)
    assert "cispo_clip_ratio" in diag


def test_sapo_soft_gate_is_temperature_sensitive():
    tensors, _ = tensors_and_logits()
    logits = tp.zeros((tensors.input_ids.shape[0], tensors.input_ids.shape[1], 8), dtype=tp.float32)
    cool, cool_diag = policy_loss(
        logits, tensors, ADVANTAGES, loss_type="sapo", sapo_temperature_pos=0.5
    )
    warm, _ = policy_loss(
        logits, tensors, ADVANTAGES, loss_type="sapo", sapo_temperature_pos=2.0
    )
    assert not math.isclose(float(cool.detach()), float(warm.detach()), abs_tol=1e-6)
    assert "sapo_soft_coef_mean" in cool_diag


def test_luspo_uses_sequence_level_ratios():
    tensors, logits = tensors_and_logits()
    loss, diag = policy_loss(
        logits, tensors, ADVANTAGES, loss_type="luspo", importance_sampling_level="sequence"
    )
    assert math.isfinite(float(loss.detach()))
    assert float(diag["mean_ratio"].detach()) > 0
    with pytest.raises(ValueError):
        policy_loss(logits, tensors, ADVANTAGES, importance_sampling_level="pairwise")


def test_vespo_gamma_weighting_is_reported_and_differentiable():
    tensors, logits = tensors_and_logits()
    loss, diag = policy_loss(logits, tensors, ADVANTAGES, loss_type="vespo")
    loss.backward()
    assert "vespo_phi_seq_mean" in diag
    assert math.isfinite(float(diag["vespo_phi_seq_mean"].detach()))
    assert float(logits.grad.abs().sum()) > 0


def test_importance_correction_is_reported_on_every_effective_ratio():
    tensors, logits = tensors_and_logits(with_sampling=True)
    _, plain = policy_loss(logits, tensors, ADVANTAGES)
    _, corrected = policy_loss(
        logits, tensors, ADVANTAGES, is_mode="token_truncate", is_clip_max=3.0
    )
    assert "is_correction_mean" in corrected and "is_affected_fraction" in corrected
    assert not math.isclose(
        float(plain["mean_ratio"].detach()),
        float(corrected["mean_ratio"].detach()),
        abs_tol=1e-6,
    )


def test_off_policy_mask_drops_high_kl_negative_sequences():
    tensors, _ = tensors_and_logits(old_logprobs=[[-1.0] * 3, [-1.0] * 2])
    logits = tp.zeros((tensors.input_ids.shape[0], tensors.input_ids.shape[1], 8), dtype=tp.float32)
    # Behavior logprobs (-1) far above the current policy (log(1/8) ≈ -2.08)
    # give a high forward KL; the negative-advantage sequence must be dropped.
    full, _ = policy_loss(logits, tensors, ADVANTAGES)
    masked, diag = policy_loss(logits, tensors, ADVANTAGES, off_policy_threshold=1e-4)
    assert float(masked.detach()) != float(full.detach())
    assert 0.0 <= float(diag["off_policy_keep_fraction"].detach()) <= 1.0


def test_entropy_bonus_and_global_normalizer():
    tensors, _ = tensors_and_logits()
    logits = tp.zeros((tensors.input_ids.shape[0], tensors.input_ids.shape[1], 8), dtype=tp.float32)
    plain, _ = policy_loss(logits, tensors, ADVANTAGES)
    entropic, diag = policy_loss(logits, tensors, ADVANTAGES, entropy_coef=0.1)
    assert "entropy" in diag
    assert not math.isclose(float(plain.detach()), float(entropic.detach()), abs_tol=1e-6)

    local, _ = policy_loss(logits, tensors, ADVANTAGES, loss_type="dapo")
    total_tokens = int(float(tensors.loss_mask.sum().detach()))
    global_loss, _ = policy_loss(
        logits, tensors, ADVANTAGES, loss_type="dapo", global_token_count=total_tokens * 2
    )
    assert math.isclose(
        float(global_loss.detach()), float(local.detach()) / 2.0, abs_tol=1e-6
    )


def test_ppo_value_loss_requires_and_uses_critic():
    tensors, logits = tensors_and_logits()
    values = tp.zeros_like(tensors.loss_mask, dtype=tp.float32, requires_grad=True)
    returns = tp.ones_like(tensors.loss_mask, dtype=tp.float32)
    with pytest.raises(ValueError):
        policy_loss(logits, tensors, ADVANTAGES, loss_type="ppo")
    loss, diag = policy_loss(
        logits, tensors, ADVANTAGES, loss_type="ppo", value_preds=values,
        returns=returns, value_coef=0.5, clip_value=0.1,
    )
    loss.backward()
    assert "value_loss" in diag and "returns_mean" in diag
    assert values.grad is not None and float(values.grad.abs().sum()) > 0
