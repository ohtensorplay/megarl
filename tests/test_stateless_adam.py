"""Stateless Adam: exact reset-every-step update and moment-free checkpoints."""

import math

import tensorplay as tp

from megarl.batching import collate, from_trajectory
from megarl.contracts import TrajectoryRecord
from megarl.stateless_adam import StatelessAdam
from megarl.trainer import TensorPlayTrainer, TrainerConfig


def make_optimizer(**kwargs):
    params = tp.nn.Parameter(tp.tensor([1.0, -2.0, 0.5]))
    optimizer = StatelessAdam([params], **kwargs)
    return params, optimizer


def test_update_matches_zero_moment_adamw_formula():
    params, optimizer = make_optimizer(lr=0.1, weight_decay=0.01)
    params.grad = tp.tensor([0.5, -0.25, 2.0])
    before = params.detach().tolist()
    optimizer.step()
    after = params.detach().tolist()
    # AdamW with zero moments and bias correction: p *= (1 - lr*wd); p -= lr*sign(g)
    expected = [
        before[i] * (1 - 0.1 * 0.01) - 0.1 * math.copysign(1.0, grad)
        for i, grad in enumerate([0.5, -0.25, 2.0])
    ]
    assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(after, expected))


def test_no_moment_tensors_and_no_cross_step_history():
    params, optimizer = make_optimizer(lr=0.1)
    state = optimizer.state_dict()
    assert state["state"] == {}
    assert state["stateless"] is True
    params.grad = tp.tensor([0.5, -0.25, 2.0])
    optimizer.step()
    first = params.detach().tolist()
    params.grad = tp.tensor([0.5, -0.25, 2.0])
    optimizer.step()
    second = params.detach().tolist()
    # Same gradient and no carried moments => an identical second update.
    expected_delta = [0.1, -0.1, 0.1]  # lr * sign(grad)
    assert all(
        math.isclose(first[i] - second[i], expected_delta[i], abs_tol=1e-6)
        for i in range(3)
    )
    assert optimizer.state_dict()["state"] == {}


def test_bias_correction_switch_scales_update_by_betas():
    params, optimizer = make_optimizer(
        lr=0.1, betas=(0.9, 0.999), bias_correction=False
    )
    params.grad = tp.tensor([1.0])
    optimizer.step()
    # Without bias correction the scales are (1-beta1) and sqrt(1-beta2).
    expected = 1.0 - 0.1 * (1 - 0.9) / (math.sqrt(1 - 0.999) + 1e-8)
    assert math.isclose(params.detach().tolist()[0], expected, abs_tol=1e-4)


def test_trainer_checkpoint_has_no_moments_and_restores_lr(tmp_path):
    class Tiny(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = tp.nn.Embedding(8, 8)
            self.head = tp.nn.Linear(8, 8)

        def forward(self, ids):
            return self.head(self.emb(ids))

    def batch():
        records = [
            TrajectoryRecord(i + 1, 2, [3, 4], [-0.2, -0.4], [], [0, 0],
                             1.0 if i == 0 else 0.0, 0, None)
            for i in range(2)
        ]
        return collate([from_trajectory(r, [1, 2]) for r in records])

    tp.manual_seed(4)
    config = TrainerConfig(optimizer_kind="stateless_adam", learning_rate=1e-3)
    trainer = TensorPlayTrainer(Tiny(), config)
    assert isinstance(trainer.optimizer, StatelessAdam)
    trainer.update(batch(), [1.0, -1.0])
    path = trainer.save_checkpoint(tmp_path / "v1", version=1)
    restored = TensorPlayTrainer(Tiny(), TrainerConfig(optimizer_kind="stateless_adam"))
    assert restored.load_checkpoint(path)["version"] == 1
    assert restored.optimizer.param_groups[0]["lr"] == config.learning_rate
    assert restored.optimizer.state_dict()["state"] == {}
