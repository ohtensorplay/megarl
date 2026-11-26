"""Agentic RL: token-in-token-out episodes, masks, budgets and controller glue."""

import asyncio
import dataclasses
import json
import math
import threading
import time

import pytest
import tensorplay as tp

from megarl.agent import (
    AgentConfig,
    AgentLoop,
    AgentRollout,
    AgenticController,
    AsyncAgenticController,
    ByteTokenizer,
    DriftKind,
    reconcile_prefix,
)
from megarl.async_controller import AsyncControllerConfig
from megarl.batching import RLBatch, collate, from_trajectory
from megarl.contracts import AgentTrajectory, RolloutRequest, TrajectoryRecord
from megarl.controller import ControllerConfig, Prompt
from megarl.observability import Telemetry
from megarl.tools import CALCULATOR_TOOL, ToolRegistry
from megarl.trainer import TensorPlayTrainer, TrainerConfig


class BytePolicy(tp.nn.Module):
    def __init__(self, vocab: int = 256, hidden: int = 8) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(vocab, hidden)
        self.head = tp.nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))


class ScriptedRollout:
    """One deterministic assistant reply per turn, with per-token versions."""

    def __init__(self, replies, version=5):
        self.replies = list(replies)
        self.version = version
        self.requests = []
        self.reloads = []

    def generate(self, request: RolloutRequest):
        self.requests.append(request)
        index = sum(1 for r in self.requests if r.request_id == request.request_id) - 1
        text = self.replies[min(index, len(self.replies) - 1)]
        tokens = list(text.encode("utf-8"))
        yield TrajectoryRecord(
            request_id=request.request_id,
            prompt_len=len(request.prompt_token_ids),
            token_ids=tokens,
            logprobs=[-0.2] * len(tokens),
            top_logprobs=[],
            policy_versions=[self.version] * len(tokens),
            reward=None,
            finish_reason=0,
            payload_handle=None,
            sampling_logprobs=[-0.19] * len(tokens),
        )

    def reload_weights(self, path, expected_digest=None):
        self.reloads.append(path)


class FakeTrainer:
    def __init__(self):
        self.batches = []

    def update(self, batch, advantages, *, clip_ratio, kl_coef):
        self.batches.append(batch)
        return {"loss": 0.5, "adv": list(advantages)}


def make_loop(replies, registry=None, **config):
    rollout = ScriptedRollout(replies)
    telemetry = Telemetry()
    if registry is not None:
        registry.telemetry = telemetry
    loop = AgentLoop(
        rollout,
        ByteTokenizer(),
        registry,
        config=AgentConfig(**config),
        telemetry=telemetry,
        version_provider=lambda: 5,
    )
    return loop, rollout, telemetry


def assistant_text(trajectory: AgentTrajectory) -> str:
    ids = [
        token
        for token, role in zip(trajectory.token_ids, trajectory.token_roles)
        if role == 1
    ]
    return bytes(ids).decode("utf-8")


def test_multiturn_tool_episode_preserves_tokens_and_masks():
    replies = [
        '{"tool_calls":[{"id":"call-1","name":"calculator","arguments":{"expr":"2+3"}}]}',
        "The answer is 5.",
    ]
    loop, rollout, telemetry = make_loop(replies, ToolRegistry([CALCULATOR_TOOL]))
    trajectory = loop.run(
        Prompt(1, [10, 11]),
        reward_fn=lambda traj: 1.0,
    )

    assert isinstance(trajectory, AgentTrajectory)
    assert trajectory.finish_reason == 0  # STOP
    assert trajectory.reward == 1.0
    assert assistant_text(trajectory) == "".join(replies)  # token-in-token-out
    assert len(trajectory.tool_calls) == 1
    assert trajectory.tool_calls[0].name == "calculator"
    assert trajectory.tool_calls[0].ok is True
    assert trajectory.step_rewards == [1.0, 1.0]

    response_len = len(trajectory.token_ids) - trajectory.prompt_len
    assert len(trajectory.logprobs) == response_len
    assert len(trajectory.policy_versions) == response_len
    assert len(trajectory.sampling_logprobs) == response_len
    assert all(v == 5 for v in trajectory.policy_versions if v)

    for token, role, mask in zip(
        trajectory.token_ids, trajectory.token_roles, trajectory.loss_mask
    ):
        if role != 1:
            assert mask == 0
    assert any(mask for mask in trajectory.loss_mask)
    assert any(role == 2 for role in trajectory.token_roles)  # tool observation kept

    events = {e["event"] for e in telemetry.events}
    assert {"agent_turn", "agent_observations", "agent_tool", "agent_finish"} <= events
    # Session affinity is set so all turns hit the same gateway worker.
    assert all(r.session_id == 1 for r in rollout.requests)
    assert len(rollout.requests) == 2
    loop.registry.close()


def test_plain_answer_terminates_without_tools():
    loop, _, _ = make_loop(["final answer"])
    trajectory = loop.run(Prompt(2, [1]))
    assert trajectory.finish_reason == 0
    assert trajectory.tool_calls == []
    assert not any(role == 2 for role in trajectory.token_roles)


def test_max_turns_and_max_tokens_are_reported():
    tool_reply = '{"tool_calls":[{"name":"calculator","arguments":{"expr":"1+1"}}]}'
    loop, _, _ = make_loop(
        [tool_reply], ToolRegistry([CALCULATOR_TOOL]), max_turns=2
    )
    traj = loop.run(Prompt(3, [1]))
    assert traj.finish_reason == 1  # MAX_TURNS
    assert len(trajectory_turns(traj)) == 2

    loop2, _, _ = make_loop(
        [tool_reply, "final"],
        ToolRegistry([CALCULATOR_TOOL]),
        max_total_tokens=5,
        max_new_tokens_per_turn=1,
    )
    traj2 = loop2.run(Prompt(4, [1]))
    assert traj2.finish_reason == 2  # MAX_TOKENS
    loop2.registry.close()
    loop.registry.close()


def trajectory_turns(trajectory):
    turns = []
    for turn, role in zip(trajectory.turn_ids, trajectory.token_roles):
        if role == 1 and (not turns or turn != turns[-1]):
            turns.append(turn)
    return turns


def test_timeout_finish_reason():
    loop, rollout, _ = make_loop(["answer"], wall_clock_secs=1e-9)
    traj = loop.run(Prompt(5, [1]))
    assert traj.finish_reason == 4  # TIMEOUT
    assert rollout.requests == []


def test_unknown_tool_aborts_before_any_tool_tokens_are_added():
    reply = '{"tool_calls":[{"name":"not_a_tool","arguments":{}}]}'
    loop, _, telemetry = make_loop([reply], ToolRegistry([CALCULATOR_TOOL]))
    traj = loop.run(Prompt(6, [1]))
    assert traj.finish_reason == 3  # TOOL_ERROR
    assert not any(role == 2 for role in traj.token_roles)
    assert any(e["event"] == "agent_tool_error" for e in telemetry.events)
    loop.registry.close()


def test_non_strict_mode_turns_tool_failure_into_observation():
    reply = '{"tool_calls":[{"name":"not_a_tool","arguments":{}}]}'
    loop, _, telemetry = make_loop(
        [reply, "recovered"], ToolRegistry([CALCULATOR_TOOL]), strict=False
    )
    traj = loop.run(Prompt(12, [1]))
    assert traj.finish_reason == 0  # continued after the failed tool call
    assert any(role == 2 for role in traj.token_roles)  # error observation appended
    assert traj.tool_calls and traj.tool_calls[0].ok is False
    assert any(e["event"] == "agent_tool_error" for e in telemetry.events)
    loop.registry.close()


def test_prefix_reconciliation_classifies_drift():
    assert reconcile_prefix([1, 2, 3], [1, 2, 3, 4]).kind is DriftKind.CLEAN
    assert reconcile_prefix([1, 2, 3, 4], [1, 2, 9, 4]).kind is DriftKind.REALIGN
    assert (
        reconcile_prefix([1, 2, 3, 4], [1, 9, 8, 7, 6], fork_threshold=1).kind
        is DriftKind.FORK
    )
    assert reconcile_prefix([1, 2], [1, 2, 3], fork_threshold=0).kind is DriftKind.CLEAN


def test_prompt_provider_realign_masks_diverged_assistant_tokens():
    tool_reply = '{"tool_calls":[{"name":"calculator","arguments":{"expr":"1+1"}}]}'
    loop, rollout, telemetry = make_loop(
        [tool_reply, "final"], ToolRegistry([CALCULATOR_TOOL]), max_turns=2
    )
    # Re-render the whole history with one changed token in the assistant span.
    def provider(records, context):
        rendered = list(context)
        prompt_len = len(records[0].prompt_ids) if records else 0
        if records and prompt_len < len(rendered):
            rendered[prompt_len] = 999
        return rendered

    loop.prompt_provider = provider
    traj = loop.run(Prompt(7, [1]))
    assert traj.token_roles[1] == 1  # assistant token still present
    assert traj.loss_mask[1] == 0  # but no longer trainable after realign
    assert any(mask == 1 for mask in traj.loss_mask)  # later turns still train
    assert any(e["event"] == "agent_realign" for e in telemetry.events)
    assert len(rollout.requests) == 2
    loop.registry.close()


def test_agent_trajectory_batches_with_assistant_only_loss_mask():
    replies = [
        '{"tool_calls":[{"name":"calculator","arguments":{"expr":"2+2"}}]}',
        "done",
    ]
    loop, _, _ = make_loop(replies, ToolRegistry([CALCULATOR_TOOL]))
    traj = loop.run(Prompt(8, [1]), reward_fn=lambda _: 1.0)
    sample = from_trajectory(traj, Prompt(8, [1]).token_ids)
    batch = collate([sample])
    assert isinstance(batch, RLBatch)
    response = batch.loss_mask[0][traj.prompt_len :]
    assert response == traj.loss_mask[traj.prompt_len :]
    assert all(
        keep == 0
        for keep, role in zip(response, traj.token_roles[traj.prompt_len :])
        if role != 1
    )
    loop.registry.close()


def test_agentic_mask_reaches_real_trainer_gradients():
    replies = [
        '{"tool_calls":[{"name":"calculator","arguments":{"expr":"1+2"}}]}',
        "final",
    ]
    loop, _, _ = make_loop(replies, ToolRegistry([CALCULATOR_TOOL]))
    traj = loop.run(Prompt(11, [1]), reward_fn=lambda _: 1.0)
    batch = collate([from_trajectory(traj, [1])])
    trainer = TensorPlayTrainer(
        BytePolicy(), TrainerConfig(max_offpolicy_tokens=16)
    )
    trainer.set_version(5)
    metrics = trainer.update(batch, [1.0], dry_run=True)
    assert math.isfinite(metrics["train/loss"])
    grad = trainer.policy.head.weight.grad
    assert grad is not None and any(abs(float(v)) > 0 for v in grad.flatten().tolist())

    # If every assistant token is masked out, the very same batch produces no
    # gradient: tool observation tokens must never contribute to the policy.
    masked = dataclasses.replace(traj, loss_mask=[0] * len(traj.loss_mask))
    masked_batch = collate([from_trajectory(masked, [1])])
    trainer.policy.head.weight.grad = None
    trainer.update(masked_batch, [1.0], dry_run=True)
    grad2 = trainer.policy.head.weight.grad
    assert grad2 is None or all(abs(float(v)) < 1e-9 for v in grad2.flatten().tolist())
    loop.registry.close()


def test_agentic_controller_runs_through_normal_pipeline():
    replies = [
        '{"tool_calls":[{"name":"calculator","arguments":{"expr":"6*7"}}]}',
        "42",
    ]
    rollout = ScriptedRollout(replies)
    trainer = FakeTrainer()
    controller = AgenticController(
        ControllerConfig(group_size=2, max_tokens_per_batch=4096),
        rollout=rollout,
        tokenizer=ByteTokenizer(),
        registry=ToolRegistry([CALCULATOR_TOOL]),
        reward_fn=lambda traj: 1.0 if traj.tool_calls else 0.0,
        trainer=trainer,
    )
    result = controller.step([Prompt(9, [1, 2])])
    assert result is not None and result.version == 1
    assert len(trainer.batches) == 1
    masks = [mask for row in trainer.batches[0].loss_mask for mask in row]
    assert any(mask == 1 for mask in masks)  # assistant tokens are trainable
    assert any(mask == 0 for mask in masks)  # prompt/tool tokens are not


def test_async_agentic_controller_runs():
    replies = ["answer A", "answer B"]
    rollout = ScriptedRollout(replies)
    trainer = FakeTrainer()

    async def go():
        controller = AsyncAgenticController(
            AsyncControllerConfig(group_size=2, max_tokens_per_batch=4096),
            rollout=rollout,
            tokenizer=ByteTokenizer(),
            reward_fn=lambda traj: 1.0 if "A" in assistant_text(traj) else 0.0,
            trainer=trainer,
        )
        return await controller.step([Prompt(10, [1])])

    result = asyncio.run(go())
    assert result is not None and result.version == 1
    assert len(trainer.batches) == 1


class _ThreadSafeSleepRollout:
    """Simulates a latency-bound rollout service, safe for parallel episodes."""

    def __init__(self, delay=0.15, text="answer"):
        self.delay = delay
        self.text = text
        self.calls = 0
        self._lock = threading.Lock()

    def generate(self, request):
        with self._lock:
            self.calls += 1
        time.sleep(self.delay)
        tokens = list(self.text.encode("utf-8"))
        yield TrajectoryRecord(
            request_id=request.request_id,
            prompt_len=len(request.prompt_token_ids),
            token_ids=tokens,
            logprobs=[-0.1] * len(tokens),
            top_logprobs=[],
            policy_versions=[1] * len(tokens),
            reward=None,
            finish_reason=0,
            payload_handle=None,
            sampling_logprobs=[],
        )


def test_parallel_agent_rollout_is_ordered_and_faster():
    serial_loop = AgentLoop(
        _ThreadSafeSleepRollout(), ByteTokenizer(), config=AgentConfig(max_turns=1)
    )
    serial = AgentRollout(serial_loop)
    t0 = time.monotonic()
    serial_results = list(serial.generate(RolloutRequest(1, 5, [1], 16, 4, None, None)))
    serial_secs = time.monotonic() - t0
    assert len(serial_results) == 4

    parallel_loop = AgentLoop(
        _ThreadSafeSleepRollout(), ByteTokenizer(), config=AgentConfig(max_turns=1)
    )
    parallel = AgentRollout(parallel_loop, max_workers=4)
    t0 = time.monotonic()
    results = list(parallel.generate(RolloutRequest(2, 5, [1], 16, 4, None, None)))
    parallel_secs = time.monotonic() - t0
    assert len(results) == 4
    assert parallel_secs < serial_secs * 0.6  # real concurrency, not just ordered
    parallel.close()
    assert parallel._pool is None  # close releases the shared pool


def test_agent_context_managers_release_threads():
    with ToolRegistry([CALCULATOR_TOOL]) as registry:
        assert registry.names() == ["calculator"]
    with AgentLoop(_ThreadSafeSleepRollout(delay=0), ByteTokenizer(), registry) as loop:
        loop.run(Prompt(3, [1]))
    # Exiting the loop closed the registry pool; a second close is a no-op.
    registry.close()


def test_telemetry_is_thread_safe():
    telemetry = Telemetry()
    threads = [
        threading.Thread(target=lambda i=i: telemetry.emit("tick", i=i)) for i in range(50)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(telemetry.events) == 50
    assert {e["i"] for e in telemetry.events} == set(range(50))
    assert all(set(e) == {"ts_ns", "event", "i"} for e in telemetry.events)
