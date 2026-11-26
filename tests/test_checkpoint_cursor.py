"""Q9: the rollout prompt cursor and model version survive checkpoint/resume."""

import json

import tensorplay as tp

from megarl.async_controller import AsyncController, AsyncControllerConfig
from megarl.contracts import TrajectoryRecord
from megarl.controller import Controller, ControllerConfig, Prompt
from megarl.observability import Telemetry
from megarl.trainer import TensorPlayTrainer, TrainerConfig


class TinyPolicy(tp.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(16, 8)
        self.head = tp.nn.Linear(8, 16)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))


def make_trainer(**kwargs):
    tp.manual_seed(1234)
    return TensorPlayTrainer(TinyPolicy(), TrainerConfig(**kwargs))


def make_batch():
    from megarl.batching import collate, from_trajectory

    record = TrajectoryRecord(1, 2, [3, 4], [-0.2, -0.4], [], [0, 0], 1.0, 0, None)
    return collate([from_trajectory(record, [1, 2])])


class FakeRollout:
    def generate(self, request):
        # Two members with different rewards make a non-zero-variance group,
        # which is what the controllers require before a trainer update.
        for i in range(request.n):
            yield TrajectoryRecord(
                request_id=request.request_id,
                prompt_len=len(request.prompt_token_ids),
                token_ids=[10 + i],
                logprobs=[-0.1],
                top_logprobs=[],
                policy_versions=[request.policy_version],
                reward=1.0 if i == 0 else 0.0,
                finish_reason=0,
                payload_handle=None,
            )

    def reload_weights(self, path, expected_digest=None):
        return None


def reward_fn(record):
    return 1.0 if record.token_ids[0] == 10 else 0.0


class FakeTrainer:
    def update(self, batch, advantages, *, clip_ratio, kl_coef):
        return {"batch": batch.size}


def test_checkpoint_persists_cursor_and_model_version(tmp_path):
    trainer = make_trainer()
    trainer.update(make_batch(), [1.0])
    path = trainer.save_checkpoint(
        tmp_path / "v1",
        version=3,
        prompt_cursor=17,
        model_version="/models/policy-v3",
    )
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["prompt_cursor"] == 17
    assert manifest["model_version"] == "/models/policy-v3"

    state = make_trainer().load_checkpoint(path)
    assert state["prompt_cursor"] == 17
    assert state["model_version"] == "/models/policy-v3"


def test_legacy_checkpoint_defaults_to_unknown_cursor(tmp_path):
    trainer = make_trainer()
    path = trainer.save_checkpoint(tmp_path / "legacy", version=1)
    state = make_trainer().load_checkpoint(path)
    assert state["prompt_cursor"] == 0
    assert state["model_version"] is None


def test_sync_controller_checkpoints_and_restores_the_cursor(tmp_path):
    config = ControllerConfig(
        group_size=2,
        max_tokens_per_batch=64,
        checkpoint_every=1,
    )
    trainer = make_trainer(checkpoint_dir=str(tmp_path))
    controller = Controller(
        config,
        rollout=FakeRollout(),
        reward_fn=reward_fn,
        trainer=trainer,
    )
    controller.model_version = "/models/policy-v1"
    result = controller.step([Prompt(5, [1, 2])])
    assert result is not None
    assert controller.prompt_cursor == 6

    state = make_trainer().load_checkpoint(tmp_path / "version_1")
    restored = Controller(
        config,
        rollout=FakeRollout(),
        reward_fn=reward_fn,
        trainer=make_trainer(),
    )
    restored.restore_checkpoint(state)
    assert restored.prompt_cursor == 6
    assert restored.model_version == "/models/policy-v1"


def test_async_controller_checkpoints_cursor_after_weight_reload(tmp_path):
    async def scenario():
        config = AsyncControllerConfig(
            group_size=2,
            max_tokens_per_batch=64,
            checkpoint_every=1,
        )
        trainer = make_trainer(checkpoint_dir=str(tmp_path))
        controller = AsyncController(
            config,
            rollout=FakeRollout(),
            reward_fn=reward_fn,
            trainer=trainer,
            telemetry=Telemetry(),
            weight_path="/models/policy-v2",
        )
        assert await controller.step([Prompt(11, [1, 2])]) is not None
        assert controller.prompt_cursor == 12
        assert controller.model_version == "/models/policy-v2"

        state = make_trainer().load_checkpoint(tmp_path / "version_1")
        assert state["prompt_cursor"] == 12
        restored = AsyncController(
            config,
            rollout=FakeRollout(),
            reward_fn=reward_fn,
            trainer=make_trainer(),
        )
        restored.restore_checkpoint(state)
        assert restored.prompt_cursor == 12
        assert restored.model_version == "/models/policy-v2"

    import asyncio

    asyncio.run(scenario())
