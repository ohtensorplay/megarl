"""Trainer integration for every loss algorithm, including PPO/GAE."""

import json
import math

import pytest
import tensorplay as tp

from megatrain import BaseTrainer as CoreTrainer
from megatrain import TrainerConfig as CoreTrainerConfig

from megarl.batching import collate, from_trajectory
from megarl.contracts import TrajectoryRecord
from megarl.losses import LOSS_TYPES
from megarl.trainer import TensorPlayTrainer, TrainerConfig


class TinyPolicy(tp.nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 8) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(vocab, hidden)
        self.head = tp.nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))


def make_batch(n: int = 2):
    records = [
        TrajectoryRecord(
            i + 1, 2, [3, 4], [-0.2, -0.4], [], [0, 0],
            1.0 if i % 2 == 0 else 0.0, 0, None,
        )
        for i in range(n)
    ]
    return collate([from_trajectory(record, [1, 2]) for record in records])


def make_trainer(loss_type: str, **kwargs):
    tp.manual_seed(1234)
    config = TrainerConfig(loss_type=loss_type, **kwargs)
    if loss_type == "ppo":
        value = TinyPolicy()
        return TensorPlayTrainer(
            TinyPolicy(), config, value_model=value,
            value_fn=lambda model, tensors: model(tensors.input_ids).mean(-1),
        )
    if loss_type in ("reverse_kl", "opd"):
        return TensorPlayTrainer(TinyPolicy(), config, teacher_model=TinyPolicy())
    return TensorPlayTrainer(TinyPolicy(), config)


@pytest.mark.parametrize("loss_type", sorted(set(LOSS_TYPES) - {"ppo"}))
def test_trainer_runs_every_policy_loss(loss_type):
    trainer = make_trainer(loss_type)
    before = trainer.policy.head.weight.detach().clone()
    metrics = trainer.update(make_batch(4), [1.0, -1.0, 1.0, -1.0])
    assert math.isfinite(metrics["train/loss"])
    assert trainer.step_count == 1
    delta = float((trainer.policy.head.weight.detach() - before).abs().sum())
    assert delta > 0, f"{loss_type} did not update policy weights"


def test_trainer_ppo_updates_policy_and_critic_and_checkpoints_both(tmp_path):
    trainer = make_trainer("ppo", value_coef=0.5)
    policy_before = trainer.policy.head.weight.detach().clone()
    value_before = trainer.value_model.head.weight.detach().clone()
    metrics = trainer.update(make_batch(), [0.0, 0.0])
    assert math.isfinite(metrics["train/loss"])
    assert "train/value_loss" in metrics
    assert float((trainer.policy.head.weight.detach() - policy_before).abs().sum()) > 0
    assert float((trainer.value_model.head.weight.detach() - value_before).abs().sum()) > 0

    path = trainer.save_checkpoint(tmp_path / "v1", version=3)
    assert (path / "value.mega").exists()

    restored = make_trainer("ppo", value_coef=0.5)
    state = restored.load_checkpoint(path)
    assert state["version"] == 3
    actual = restored.value_model.head.weight.detach()
    after = trainer.value_model.head.weight.detach()  # post-update value model
    assert all(
        math.isclose(a, b, abs_tol=1e-9)
        for a, b in zip(actual.flatten().tolist(), after.flatten().tolist())
    )
    assert float((actual - value_before).abs().sum()) > 0  # resume is not the initial state


def test_global_token_objective_accumulation_matches_single_batch():
    def grads(accum: int):
        trainer = make_trainer("dapo", gradient_accumulation_steps=accum)
        trainer.update(make_batch(4), [1.0, -1.0, 1.0, -1.0], dry_run=True)
        return [float(v) for v in trainer.policy.head.weight.grad.flatten().tolist()]

    single, accumulated = grads(1), grads(2)
    worst = max(abs(a - b) for a, b in zip(single, accumulated))
    assert worst < 1e-4, f"global-token accumulation mismatch: {worst}"


@pytest.mark.parametrize("loss_type", sorted(LOSS_TYPES))
def test_accumulation_matches_single_batch_for_every_algorithm(loss_type):
    """The core's ``keep_i / sum(keep)`` weighting must reproduce one big step.

    This pins the RL hook mapping for all three normalizer families: row
    (grpo/sapo/luspo/dr_grpo), fresh-token (bnpo) and global-window
    (dapo/cispo/vespo/ppo/reverse_kl/opd).
    """

    def grads(accum: int):
        kwargs = {}
        if loss_type == "ppo":
            kwargs["value_coef"] = 0.5
        if loss_type == "opd":
            kwargs["opd_kl_coef"] = 0.5
        trainer = make_trainer(loss_type, gradient_accumulation_steps=accum, **kwargs)
        trainer.set_version(0)
        trainer.update(make_batch(4), [1.0, -1.0, 1.0, -1.0], dry_run=True)
        return {
            "policy": [float(v) for v in trainer.policy.head.weight.grad.flatten().tolist()],
            "value": (
                [float(v) for v in trainer.value_model.head.weight.grad.flatten().tolist()]
                if trainer.value_model is not None
                else []
            ),
        }

    single, accumulated = grads(1), grads(2)
    for key in ("policy", "value"):
        assert len(single[key]) == len(accumulated[key])
        worst = max(
            (abs(a - b) for a, b in zip(single[key], accumulated[key])), default=0.0
        )
        assert worst < 1e-4, f"{loss_type} {key} accumulation mismatch: {worst}"


def test_trainer_reuses_the_megatrain_core_contract():
    trainer = make_trainer("grpo")
    assert isinstance(trainer, CoreTrainer)
    # One config object: BaseTrainer reads the same instance the RL layer does.
    assert isinstance(trainer.config, CoreTrainerConfig)
    assert trainer.config is trainer.rl_config
    assert trainer.tensor_batch_cls.__mro__[1].__module__ == "megatrain.losses"
    assert trainer.optimizer is not None and trainer.step_count == 0


def test_keep_tokens_matches_each_normalizer_family():
    batch = make_batch(2)
    expected = {
        "grpo": batch.size,
        "sapo": batch.size,
        "luspo": batch.size,
        "dr_grpo": batch.size,
        "bnpo": 4,  # two rows x two response tokens
        "dapo": 4,
        "cispo": 4,
        "vespo": 4,
        "ppo": 4,
        "reverse_kl": 4,
        "opd": 4,
    }
    for loss_type, units in expected.items():
        trainer = make_trainer(loss_type)
        assert trainer._keep_tokens(batch) == units, loss_type


class CountingTeacher(TinyPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, input_ids):
        self.calls += 1
        return super().forward(input_ids)


def test_teacher_is_not_forwarded_when_the_objective_does_not_need_it():
    teacher = CountingTeacher()
    trainer = TensorPlayTrainer(
        TinyPolicy(), TrainerConfig(loss_type="grpo"), teacher_model=teacher
    )
    trainer.update(make_batch(), [1.0, -1.0])
    assert teacher.calls == 0

    distiller = TensorPlayTrainer(
        TinyPolicy(), TrainerConfig(loss_type="reverse_kl"), teacher_model=CountingTeacher()
    )
    distiller.update(make_batch(), [1.0, -1.0])
    assert distiller.teacher_model.calls > 0


def test_ppo_checkpoint_artifact_is_checksum_verified(tmp_path):
    trainer = make_trainer("ppo", value_coef=0.5)
    path = trainer.save_checkpoint(tmp_path / "v1", version=1)
    (path / "value.mega").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        make_trainer("ppo", value_coef=0.5).load_checkpoint(path)


def test_checkpoint_stores_optimizer_in_mega_and_resumes_exactly(tmp_path):
    trainer = make_trainer("grpo")
    trainer.update(make_batch(), [1.0, -1.0])
    path = trainer.save_checkpoint(tmp_path / "v1", version=5)
    assert (path / "optimizer.mega").exists()
    assert not (path / "optimizer.pkl").exists()
    manifest = json.loads((path / "manifest.json").read_text())
    assert "optimizer.mega" in manifest["files"]

    restored = make_trainer("grpo")
    assert restored.load_checkpoint(path)["version"] == 5
    before = trainer.optimizer.state_dict()["state"][0]["exp_avg"]
    after = restored.optimizer.state_dict()["state"][0]["exp_avg"]
    assert all(
        math.isclose(a, b, abs_tol=1e-9)
        for a, b in zip(before.flatten().tolist(), after.flatten().tolist())
    )


def test_legacy_reduction_alias_still_selects_loss_type():
    trainer = TensorPlayTrainer(TinyPolicy(), TrainerConfig(reduction="global_token_mean"))
    assert trainer.config.loss_type == "dapo"
    assert math.isfinite(trainer.update(make_batch(), [1.0, -1.0])["train/loss"])
