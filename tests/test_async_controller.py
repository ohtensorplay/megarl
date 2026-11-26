"""Asynchronous controller, admission control and payload leases."""

import asyncio

from megarl.async_controller import AsyncController, AsyncControllerConfig, PauseMode, PayloadQueue
from megarl.contracts import TrajectoryRecord
from megarl.controller import Prompt
from megarl.observability import Telemetry


class FakeRollout:
    def __init__(self):
        self.reloaded: list[str] = []

    def generate(self, request):
        for i in range(request.n):
            yield TrajectoryRecord(
                request_id=request.request_id,
                prompt_len=len(request.prompt_token_ids),
                token_ids=[10 + i],
                logprobs=[-0.1 - i],
                top_logprobs=[],
                policy_versions=[request.policy_version],
                reward=None,
                finish_reason=0,
                payload_handle=None,
            )

    def reload_weights(self, path, expected_digest=None):
        self.reloaded.append(path)


class FakeTrainer:
    def update(self, batch, advantages, *, clip_ratio, kl_coef):
        return {"loss": 0.25, "batch": batch.size}


def make_controller(weight_path=None):
    telemetry = Telemetry()
    rollout = FakeRollout()
    controller = AsyncController(
        AsyncControllerConfig(group_size=2, max_tokens_per_batch=64),
        rollout=rollout,
        reward_fn=lambda record: 1.0 if record.token_ids[0] == 10 else 0.0,
        trainer=FakeTrainer(),
        telemetry=telemetry,
        weight_path=weight_path,
    )
    return controller, rollout, telemetry


def test_step_admission_pause_and_resume():
    controller, rollout, telemetry = make_controller(weight_path="/tmp/w")

    async def scenario():
        first = await controller.step([Prompt(1, [1, 2])])
        assert first is not None and first.version == 1
        assert first.token_count > 0
        assert rollout.reloaded == ["/tmp/w"]
        controller.pause(PauseMode.WAIT)
        blocked = await controller.step([Prompt(2, [3, 4])])
        assert blocked is None
        controller.resume()
        second = await controller.step([Prompt(3, [5, 6])])
        assert second is not None and second.version == 2

    asyncio.run(scenario())
    assert any(e["event"] == "pause" for e in telemetry.events)


def test_payload_queue_leases_and_reclaim():
    queue = PayloadQueue(ttl_secs=0)
    record = TrajectoryRecord(1, 1, [2], [-0.1], [], [0], None, 0, None)
    lease = queue.put(record)
    assert len(queue) == 1
    restored = queue.take(lease)
    assert restored is not None
    assert restored.request_id == record.request_id
    assert restored.logprobs == record.logprobs
    assert restored.payload_handle is not None  # stamped from the payload store
    assert len(queue) == 0

    queue.put(record)
    reclaimed = queue.reclaim()  # zero-TTL leases are already expired
    assert len(reclaimed) == 1 and reclaimed[0].request_id == record.request_id
    assert queue.reclaim() == []
    assert len(queue) == 0
