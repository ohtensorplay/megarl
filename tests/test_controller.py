"""Controller loop: version bump, zero-variance drop, telemetry."""

from megarl.batching import RLBatch
from megarl.contracts import TrajectoryRecord
from megarl.controller import Controller, ControllerConfig, Prompt
from megarl.observability import Telemetry


class FakeRollout:
    def __init__(self, constant_reward=False):
        self.constant_reward = constant_reward
        self.reloads: list[str] = []

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
        self.reloads.append(path)


class FakeTrainer:
    def __init__(self):
        self.batches: list[RLBatch] = []

    def update(self, batch, advantages, *, clip_ratio, kl_coef):
        self.batches.append(batch)
        return {"loss": 0.5, "adv": list(advantages)}


def make(constant_reward=False, weight_path=None):
    telemetry = Telemetry()
    rollout = FakeRollout(constant_reward)
    trainer = FakeTrainer()
    reward_fn = (lambda record: 1.0) if constant_reward else (
        lambda record: 1.0 if record.token_ids[0] == 10 else 0.0
    )
    controller = Controller(
        ControllerConfig(group_size=2, max_tokens_per_batch=64),
        rollout=rollout,
        reward_fn=reward_fn,
        trainer=trainer,
        telemetry=telemetry,
        weight_path=weight_path,
    )
    return controller, rollout, trainer, telemetry


def test_step_updates_weights_and_bumps_version():
    controller, rollout, trainer, telemetry = make(weight_path="/tmp/weights")
    result = controller.step([Prompt(1, [1, 2])])
    assert result is not None
    assert result.version == 1
    assert result.groups == 1 and result.dropped_groups == 0
    assert result.token_count > 0
    assert len(trainer.batches) == 1
    assert rollout.reloads == ["/tmp/weights"]
    events = {e["event"] for e in telemetry.events}
    assert "admission_deferred" not in events
    assert {"reward_group_summary", "staleness", "trainer_update", "weight_reload"} <= events


def test_rate_limiter_caps_admission_independently_of_concurrency():
    telemetry = Telemetry()
    config = ControllerConfig(
        max_staleness=0,
        consumer_batch_size=1,
        concurrency=64,
        trajectories_per_sec=0.1,  # one token every 10s: no refill mid-test
        rate_burst=1,
        group_size=2,
        max_tokens_per_batch=64,
    )
    controller = Controller(
        config,
        rollout=FakeRollout(),
        reward_fn=lambda record: 1.0,
        trainer=FakeTrainer(),
        telemetry=telemetry,
    )
    assert len(controller._admit([Prompt(1, [1]), Prompt(2, [2])])) == 1
    assert controller._admit([Prompt(3, [3])]) == []
    assert any(
        event["event"] == "rate_limited" and event["granted"] == 0
        for event in telemetry.events
    )


def test_restore_checkpoint_rebases_admission_window():
    versions: list[int] = []
    telemetry = Telemetry()
    trainer = FakeTrainer()
    trainer.set_version = versions.append  # type: ignore[attr-defined]
    controller = Controller(
        ControllerConfig(
            max_staleness=0,
            consumer_batch_size=32,
            concurrency=256,
            group_size=2,
            max_tokens_per_batch=64,
        ),
        rollout=FakeRollout(),
        reward_fn=lambda record: 1.0,
        trainer=trainer,
        telemetry=telemetry,
    )
    cold_capacity = controller.budget.capacity(7, 0, 0)
    controller.restore_checkpoint(
        {"version": 7, "accepted_tokens": 0, "run_tokens": 5}
    )
    assert controller.version == 7
    assert versions == [7]
    assert controller.run_tokens == 5
    assert controller.accepted_tokens == 7 * 32
    assert controller.budget.capacity(7, controller.accepted_tokens, 0) < cold_capacity
    assert any(
        event["event"] == "checkpoint_restored" and event["rebased_tokens"] == 224
        for event in telemetry.events
    )


def test_zero_variance_group_is_dropped():
    controller, rollout, trainer, telemetry = make(constant_reward=True)
    result = controller.step([Prompt(1, [1, 2])])
    assert result is not None
    assert result.dropped_groups == 1
    assert result.token_count == 0
    assert trainer.batches == []  # no wasted forward/backward
    assert controller.version == 1
