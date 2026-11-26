"""Q4: deferred admission is drained by the Rust AsyncPacer in bounded batches."""

import asyncio
import threading

from megarl.async_controller import AsyncController, AsyncControllerConfig, PauseMode
from megarl.contracts import TrajectoryRecord
from megarl.controller import Prompt
from megarl.observability import Telemetry


class GatedRollout:
    """Keeps the first generation in flight until `gate` is released."""

    def __init__(self) -> None:
        self.gate = threading.Event()

    def generate(self, request):
        self.gate.wait(5)
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
        return None


class FakeTrainer:
    def update(self, batch, advantages, *, clip_ratio, kl_coef):
        return {"batch": batch.size}


def make_controller(**kwargs):
    telemetry = Telemetry()
    rollout = GatedRollout()
    controller = AsyncController(
        AsyncControllerConfig(
            group_size=2,
            max_tokens_per_batch=64,
            max_concurrent_requests=2,
            pacer_rate_per_sec=100.0,
            pacer_batch_size=2,
            **kwargs,
        ),
        rollout=rollout,
        reward_fn=lambda record: 1.0,
        trainer=FakeTrainer(),
        telemetry=telemetry,
    )
    return controller, rollout, telemetry


def test_deferred_prompts_are_woken_by_the_pacer_in_bounded_batches():
    controller, rollout, telemetry = make_controller()

    async def scenario():
        first = asyncio.create_task(controller.step([Prompt(1, [1, 2])]))
        second_in_flight = asyncio.create_task(controller.step([Prompt(2, [3, 4])]))
        await asyncio.sleep(0.02)  # both slots are now occupied
        deferred = asyncio.create_task(
            controller.step([Prompt(3, [5, 6]), Prompt(4, [7, 8]), Prompt(5, [9, 10])])
        )
        await asyncio.sleep(0.02)
        assert len(controller.waiters) == 3
        assert controller.prompt_cursor == 3  # only prompts 1 and 2 admitted

        rollout.gate.set()
        await first
        await second_in_flight
        await deferred
        assert len(controller.waiters) == 3  # still waiting for a paced credit

        # Give the Rust pacer time for two credits, capped at one batch.
        await asyncio.sleep(0.06)
        drained = await controller.step([])
        assert drained is not None
        assert controller.prompt_cursor == 5  # prompts 3 and 4 admitted
        assert len(controller.waiters) == 1

        await asyncio.sleep(0.06)
        last = await controller.step([])
        assert last is not None
        assert controller.prompt_cursor == 6  # prompt 5 admitted
        assert not controller.waiters and not controller._ready
        wakes = [e["count"] for e in telemetry.events if e["event"] == "pacer_wake"]
        assert wakes and max(wakes) <= 2

    asyncio.run(scenario())


def test_abort_discards_deferred_work_and_resume_keeps_the_cursor():
    controller, rollout, _ = make_controller()

    async def scenario():
        first = asyncio.create_task(controller.step([Prompt(1, [1, 2])]))
        second_in_flight = asyncio.create_task(controller.step([Prompt(2, [3, 4])]))
        await asyncio.sleep(0.02)
        deferred = asyncio.create_task(controller.step([Prompt(9, [5, 6])]))
        await asyncio.sleep(0.02)
        assert len(controller.waiters) == 1

        controller.pause(PauseMode.ABORT)
        assert not controller.waiters and not controller._ready
        rollout.gate.set()
        await first
        await second_in_flight
        await deferred
        controller.resume()
        await asyncio.sleep(0.02)
        # The aborted prompt is never replayed; the cursor stays at prompt 2.
        assert controller.prompt_cursor == 3
        await controller.step([])

    asyncio.run(scenario())


def test_wait_for_capacity_returns_immediately_when_a_slot_is_free():
    controller, _, _ = make_controller()

    async def scenario():
        assert await controller.wait_for_capacity(timeout=0.1) is True

    asyncio.run(scenario())
