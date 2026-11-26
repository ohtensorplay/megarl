"""Reverse-KL distillation, OPD and DPO: math and trainer integration."""

import math

import pytest
import tensorplay as tp
import tensorplay.nn.functional as F

from megarl.batching import collate, collate_preference_pairs, from_trajectory
from megarl.contracts import TrajectoryRecord
from megarl.dpo import DPO_LOSS_TYPES, build_preference_pairs, dpo_loss, sequence_logps
from megarl.kl import KL_ESTIMATORS, approx_kl
from megarl.losses import LOSS_TYPES, TensorBatch, policy_loss
from megarl.trainer import TensorPlayTrainer, TrainerConfig


class TinyPolicy(tp.nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 8) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(vocab, hidden)
        self.head = tp.nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))


def records(rewards=(1.0, 0.0)):
    return [
        TrajectoryRecord(
            i + 1, 2, [3 + i, 4 + i], [-0.2, -0.4], [], [0, 0], reward, 0, None
        )
        for i, reward in enumerate(rewards)
    ]


def make_batch(rewards=(1.0, 0.0)):
    return collate([from_trajectory(r, [1, 2]) for r in records(rewards)])


def teacher_logprobs(batch):
    return tp.full((batch.size, batch.max_len), -1.0, dtype=tp.float32)


# --- KL estimators -----------------------------------------------------------


@pytest.mark.parametrize("kl_type", KL_ESTIMATORS)
def test_approx_kl_is_nonnegative_and_finite(kl_type):
    logp = tp.tensor([[-2.0, -1.5], [0.3, -0.2]], requires_grad=True)
    ref = tp.tensor([[-1.0, -1.5], [0.1, 0.4]])
    value = approx_kl(logp, ref, kl_type=kl_type)
    assert all(math.isfinite(float(v)) for v in value.flatten().tolist())
    if kl_type != "k1":
        assert all(float(v) >= -1e-6 for v in value.flatten().tolist())
    value.sum().backward()
    assert logp.grad is not None and float(logp.grad.abs().sum()) > 0


def test_approx_kl_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown kl_type"):
        approx_kl(tp.zeros((2,)), tp.zeros((2,)), kl_type="jsd")


def test_kl_penalty_uses_configured_estimator():
    batch = make_batch()
    tensors = TensorBatch(batch)
    logits = tp.zeros((batch.size, batch.max_len, 8), dtype=tp.float32)
    k1, d1 = policy_loss(
        logits, tensors, tp.tensor([1.0, 0.0]), kl_coef=1.0, kl_type="k1"
    )
    k2, d2 = policy_loss(
        logits, tensors, tp.tensor([1.0, 0.0]), kl_coef=1.0, kl_type="k2"
    )
    assert not math.isclose(float(k1.detach()), float(k2.detach()), abs_tol=1e-6)
    assert not math.isclose(float(d1["kl"].detach()), float(d2["kl"].detach()), abs_tol=1e-6)


# --- Reverse KL / OPD loss ---------------------------------------------------


@pytest.mark.parametrize("loss_type", ["reverse_kl", "opd"])
def test_reverse_kl_loss_is_teacher_aware(loss_type):
    assert loss_type in LOSS_TYPES
    batch = make_batch()
    tensors = TensorBatch(batch)
    teacher = teacher_logprobs(batch)
    logits = tp.zeros((batch.size, batch.max_len, 8), dtype=tp.float32, requires_grad=True)
    loss, diag = policy_loss(
        logits,
        tensors,
        tp.tensor([1.0, -1.0]),
        loss_type=loss_type,
        teacher_logprobs=teacher,
    )
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0
    assert "reverse_kl" in diag and math.isfinite(float(diag["reverse_kl"].detach()))
    with pytest.raises(ValueError, match="requires teacher_logprobs"):
        policy_loss(logits, tensors, tp.tensor([1.0, -1.0]), loss_type=loss_type)


def test_reverse_kl_k2_is_a_quadratic_penalty():
    batch = make_batch()
    tensors = TensorBatch(batch)
    teacher = teacher_logprobs(batch)
    logp = tp.gather(
        F.log_softmax(tp.zeros((batch.size, batch.max_len, 8)), dim=-1),
        -1,
        tensors.input_ids.unsqueeze(-1),
    ).squeeze(-1)
    loss, _ = policy_loss(
        tp.zeros((batch.size, batch.max_len, 8)),
        tensors,
        tp.tensor([0.0, 0.0]),
        loss_type="reverse_kl",
        teacher_logprobs=teacher,
        distill_kl_type="k2",
    )
    mask = tensors.loss_mask
    expected = (0.5 * (logp - teacher) ** 2 * mask).sum() / mask.sum()
    assert math.isclose(float(loss.detach()), float(expected.detach()), abs_tol=1e-6)


# --- DPO math ----------------------------------------------------------------


@pytest.mark.parametrize("loss_type", DPO_LOSS_TYPES)
def test_dpo_loss_prefers_the_correct_pair(loss_type):
    chosen = tp.tensor([-1.0], requires_grad=True)
    rejected = tp.tensor([-3.0], requires_grad=True)
    loss, diag = dpo_loss(chosen, rejected, beta=0.1, loss_type=loss_type)
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    assert chosen.grad is not None and rejected.grad is not None
    assert float(chosen.grad[0]) < 0  # increase chosen logp
    assert float(rejected.grad[0]) > 0  # decrease rejected logp
    assert {"loss", "reward_accuracy", "reward_margin"} <= set(diag)
    assert float(diag["reward_accuracy"].detach()) == 1.0


def test_dpo_reference_removes_a_common_offset():
    chosen = tp.tensor([-1.0], requires_grad=True)
    rejected = tp.tensor([-3.0], requires_grad=True)
    ref_chosen = tp.tensor([5.0])
    ref_rejected = tp.tensor([5.0])
    plain, _ = dpo_loss(chosen, rejected, beta=0.1)
    with_ref, _ = dpo_loss(
        chosen, rejected, ref_chosen_logps=ref_chosen,
        ref_rejected_logps=ref_rejected, beta=0.1,
    )
    assert math.isclose(float(plain.detach()), float(with_ref.detach()), abs_tol=1e-6)


def test_dpo_cdpo_label_smoothing_and_validation():
    chosen = tp.tensor([2.0])
    rejected = tp.tensor([-2.0])
    plain, _ = dpo_loss(chosen, rejected, beta=0.1)
    smoothed, _ = dpo_loss(chosen, rejected, beta=0.1, label_smoothing=0.1)
    assert float(smoothed.detach()) > float(plain.detach())
    with pytest.raises(ValueError, match="beta must be positive"):
        dpo_loss(chosen, rejected, beta=0.0)
    with pytest.raises(ValueError, match="must be given together"):
        dpo_loss(chosen, rejected, ref_chosen_logps=chosen)


def test_sequence_logps_sums_only_response_tokens():
    logits = tp.zeros((1, 4, 8), dtype=tp.float32)
    token_ids = tp.tensor([[1, 2, 3, 4]])
    mask = tp.tensor([[0.0, 0.0, 1.0, 1.0]])
    logp = sequence_logps(logits, token_ids, mask)
    assert math.isclose(float(logp[0]), 2 * math.log(1 / 8), abs_tol=1e-6)


def test_build_preference_pairs_drops_uninformative_groups():
    samples = [from_trajectory(r, [1, 2]) for r in records((1.0, 0.0))]
    tie = [from_trajectory(r, [1, 2]) for r in records((0.5, 0.5))]
    pairs = build_preference_pairs([samples, tie, samples[:1]])
    assert len(pairs) == 1
    assert pairs[0][0].reward == 1.0 and pairs[0][1].reward == 0.0
    assert collate_preference_pairs(pairs)[0].size == 1


# --- Trainer integration -----------------------------------------------------


def make_trainer(loss_type="grpo", **kwargs):
    tp.manual_seed(1234)
    config = TrainerConfig(loss_type=loss_type, **kwargs)
    teacher = TinyPolicy()
    return TensorPlayTrainer(TinyPolicy(), config, teacher_model=teacher)


def test_trainer_runs_reverse_kl_and_updates_policy_not_teacher():
    trainer = make_trainer("reverse_kl")
    before = trainer.policy.head.weight.detach().clone()
    teacher_before = trainer.teacher_model.head.weight.detach().clone()
    metrics = trainer.update(make_batch(), [0.0, 0.0])
    assert math.isfinite(metrics["train/loss"])
    assert float((trainer.policy.head.weight.detach() - before).abs().sum()) > 0
    assert float(
        (trainer.teacher_model.head.weight.detach() - teacher_before).abs().sum()
    ) == 0


@pytest.mark.parametrize("loss_type", ["grpo", "ppo"])
def test_opd_advantage_penalty_is_orthogonal(loss_type):
    kwargs = {"opd_kl_coef": 0.5}
    if loss_type == "ppo":
        kwargs["value_coef"] = 0.5
    tp.manual_seed(1234)
    config = TrainerConfig(loss_type=loss_type, **kwargs)
    if loss_type == "ppo":
        trainer = TensorPlayTrainer(
            TinyPolicy(),
            config,
            teacher_model=TinyPolicy(),
            value_model=TinyPolicy(),
            value_fn=lambda model, tensors: model(tensors.input_ids).mean(-1),
        )
    else:
        trainer = TensorPlayTrainer(TinyPolicy(), config, teacher_model=TinyPolicy())
    metrics = trainer.update(make_batch(), [1.0, -1.0])
    assert math.isfinite(metrics["train/loss"])
    assert "train/opd_reverse_kl" in metrics
    assert metrics["train/opd_reverse_kl"] >= -1e-6


def test_dpo_accumulation_matches_single_batch():
    def grads(accum: int):
        tp.manual_seed(1234)
        trainer = TensorPlayTrainer(
            TinyPolicy(),
            TrainerConfig(
                dpo_beta=0.1, learning_rate=1e-2, gradient_accumulation_steps=accum
            ),
        )
        samples = [from_trajectory(r, [1, 2]) for r in records((1.0, 0.0, 1.0, 0.0))]
        pairs = [(samples[0], samples[1]), (samples[2], samples[3])]
        chosen, rejected = collate_preference_pairs(pairs)
        trainer.dpo_update(chosen, rejected, dry_run=True)
        return [float(v) for v in trainer.policy.head.weight.grad.flatten().tolist()]

    single, accumulated = grads(1), grads(2)
    assert len(single) == len(accumulated)
    worst = max(abs(a - b) for a, b in zip(single, accumulated))
    assert worst < 1e-4, f"DPO accumulation mismatch: {worst}"


def test_trainer_dpo_update_moves_policy_and_reports_metrics():
    tp.manual_seed(1234)
    trainer = TensorPlayTrainer(
        TinyPolicy(), TrainerConfig(dpo_beta=0.1, learning_rate=1e-2)
    )
    before = trainer.policy.head.weight.detach().clone()
    samples = [from_trajectory(r, [1, 2]) for r in records((1.0, 0.0))]
    chosen, rejected = collate_preference_pairs([(samples[0], samples[1])])
    metrics = trainer.dpo_update(chosen, rejected)
    assert math.isfinite(metrics["dpo/loss"])
    assert "dpo/reward_accuracy" in metrics
    assert float((trainer.policy.head.weight.detach() - before).abs().sum()) > 0


def test_controller_dpo_step_calls_dpo_update():
    import dataclasses

    from megarl.controller import Controller, ControllerConfig, Prompt
    from megarl.observability import Telemetry
    from megarl.contracts import TrajectoryRecord

    class Rollout:
        def generate(self, request):
            for i in range(request.n):
                yield TrajectoryRecord(
                    request_id=request.request_id,
                    prompt_len=len(request.prompt_token_ids),
                    token_ids=[10 + i],
                    logprobs=[-0.1],
                    top_logprobs=[],
                    policy_versions=[request.policy_version],
                    reward=None,
                    finish_reason=0,
                    payload_handle=None,
                )

        def reload_weights(self, path, expected_digest=None):
            pass

    class DPOTrainer:
        def __init__(self):
            self.calls = []

        def set_version(self, version):
            pass

        def dpo_update(self, chosen, rejected):
            self.calls.append((chosen.size, rejected.size))
            return {"dpo/loss": 0.5}

    config = dataclasses.replace(
        ControllerConfig(group_size=4, max_tokens_per_batch=128), dpo=True
    )
    trainer = DPOTrainer()
    controller = Controller(
        config,
        rollout=Rollout(),
        reward_fn=lambda record: 1.0 if record.token_ids[0] == 10 else 0.0,
        trainer=trainer,
        telemetry=Telemetry(),
    )
    result = controller.step([Prompt(1, [1, 2])])
    assert result is not None
    assert trainer.calls and trainer.calls[0][0] == trainer.calls[0][1]
    assert result.token_count > 0
