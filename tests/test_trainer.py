"""Production trainer: real TensorPlay forward/backward, grad accumulation,
staleness enforcement and checkpoint/resume. No fake tensors or optimizers."""

import dataclasses
import math

import tensorplay as tp

from megarl.batching import collate, from_trajectory
from megarl.contracts import TrajectoryRecord
from megarl.controller import Controller, ControllerConfig, Prompt
from megarl.trainer import TensorPlayTrainer, TrainerConfig


class TinyPolicy(tp.nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 8) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(vocab, hidden)
        self.head = tp.nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))


def make_batch(n: int = 2, versions=(0, 0)):
    records = [
        TrajectoryRecord(i + 1, 2, [3, 4], [-0.2, -0.4], [], list(versions), 1.0, 0, None)
        for i in range(n)
    ]
    return collate([from_trajectory(r, [1, 2]) for r in records])


def make_trainer(**kwargs):
    tp.manual_seed(1234)
    cfg = TrainerConfig(**kwargs)
    return TensorPlayTrainer(TinyPolicy(), cfg)


def test_real_step_produces_gradients_and_updates_weights():
    trainer = make_trainer()
    before = trainer.policy.head.weight.detach().clone()
    metrics = trainer.update(make_batch(), [1.0, -1.0], dry_run=True)
    assert trainer.policy.head.weight.grad is not None
    assert any(abs(float(v)) > 0 for v in trainer.policy.head.weight.grad.flatten().tolist())
    assert math.isfinite(metrics["train/loss"])
    trainer.update(make_batch(), [1.0, -1.0])
    after = trainer.policy.head.weight.detach()
    delta = float((after - before).abs().sum())
    assert delta > 0, "optimizer step must change real parameters"
    assert trainer.step_count == 1


def test_gradient_accumulation_matches_single_batch():
    def grads(accum: int):
        trainer = make_trainer(gradient_accumulation_steps=accum)
        trainer.set_version(0)
        trainer.update(make_batch(4), [1.0, -1.0, 1.0, -1.0], dry_run=True)
        return [float(v) for v in trainer.policy.head.weight.grad.flatten().tolist()]

    single, accumulated = grads(1), grads(2)
    assert len(single) == len(accumulated)
    worst = max(abs(a - b) for a, b in zip(single, accumulated))
    assert worst < 1e-4, f"grad accumulation mismatch: {worst}"


def test_stale_tokens_are_masked_out_of_the_loss():
    trainer = make_trainer(max_offpolicy_tokens=1)
    trainer.set_version(10)  # every token from version 0 is stale
    batch = dataclasses.replace(
        make_batch(versions=(0, 0)), versions=[[10, 10, 0, 0], [10, 10, 0, 0]]
    )
    metrics = trainer.update(batch, [1.0, -1.0], dry_run=True)
    grad = trainer.policy.head.weight.grad
    worst = max(abs(float(v)) for v in grad.flatten().tolist())
    assert worst < 1e-6, f"stale tokens must not contribute gradients: {worst}"
    assert math.isclose(metrics["train/loss"], 0.0, abs_tol=1e-6)


def test_checkpoint_resume_restores_weights_version_and_rng(tmp_path):
    trainer = make_trainer()
    trainer.update(make_batch(), [1.0, -1.0])
    expected = [float(v) for v in trainer.policy.head.weight.detach().flatten().tolist()]
    path = trainer.save_checkpoint(
        tmp_path / "v1", version=7, accepted_tokens=123, reward_stats={"mean": 0.5}
    )
    assert (path / "manifest.json").exists()
    fresh = make_trainer()
    state = fresh.load_checkpoint(path)
    actual = [float(v) for v in fresh.policy.head.weight.detach().flatten().tolist()]
    assert actual == expected
    assert state["version"] == 7 and state["accepted_tokens"] == 123
    assert state["reward_stats"] == {"mean": 0.5}


def test_checkpoint_detects_corruption(tmp_path):
    trainer = make_trainer()
    path = trainer.save_checkpoint(tmp_path / "v1", version=1)
    (path / "model.mega").write_bytes(b"corrupted")
    try:
        make_trainer().load_checkpoint(path)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("corrupted checkpoint was accepted")


def test_custom_forward_receives_the_full_tensor_batch():
    seen = {}

    def forward(model, tensors):
        seen["batch"] = tensors
        return model(tensors.input_ids)

    trainer = TensorPlayTrainer(TinyPolicy(), TrainerConfig(), forward_fn=forward)
    metrics = trainer.update(make_batch(), [1.0, -1.0], dry_run=True)
    assert math.isfinite(metrics["train/loss"])
    batch = seen["batch"]
    # RL-only fields and the core fields are both available to the forward.
    for field in ("input_ids", "attention_mask", "loss_mask", "cu_seqlens", "versions"):
        assert hasattr(batch, field), field


def test_compile_with_custom_forward_is_rejected():
    try:
        TensorPlayTrainer(
            TinyPolicy(),
            TrainerConfig(compile_policy=True),
            forward_fn=lambda model, tensors: model(tensors.input_ids),
        )
    except ValueError as exc:
        assert "compile_policy" in str(exc)
    else:
        raise AssertionError("a compiled custom forward is not supported")


def test_controller_runs_a_real_trainer(tmp_path):
    class Rollout:
        def generate(self, request):
            for i in range(request.n):
                yield TrajectoryRecord(
                    request.request_id, 2, [3, 4 + i], [-0.2, -0.4], [],
                    [request.policy_version] * 2, None, 0, None,
                )

        def reload_weights(self, path, expected_digest=None):
            pass

    trainer = make_trainer(checkpoint_dir=str(tmp_path))
    controller = Controller(
        ControllerConfig(group_size=2, max_tokens_per_batch=64, checkpoint_every=1),
        rollout=Rollout(),
        reward_fn=lambda record: 1.0 if record.token_ids[-1] % 2 else 0.0,
        trainer=trainer,
    )
    result = controller.step([Prompt(1, [1, 2])])
    assert result is not None and result.version == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])
    assert (tmp_path / "version_1" / "manifest.json").exists()
