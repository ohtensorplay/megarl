"""Trajectory-level dLLM RL: recording, replay, controllers and async path."""

import asyncio
import math

import tensorplay as tp

from megarl.contracts import RolloutRequest, SamplingParams
from megarl.controller import ControllerConfig, Prompt
from megarl.async_controller import AsyncControllerConfig
from megarl.dllm import (
    AsyncDLLMController,
    DLLMConfig,
    DLLMController,
    DLLMTrainer,
    collate_trajectory,
)
from megarl.dllm.trajectory import (
    DLLMPayloadQueue,
    DLLMStep,
    DLLMTrajectoryRecord,
    DLLMTrajectoryStore,
    TrajectoryDLLMRollout,
    TrajectoryDiffusionGenerator,
    TrajectoryGenerationConfig,
    TrajectoryGenerationResult,
    decode_trajectory,
    encode_trajectory,
)
from megarl.trainer import TrainerConfig


class TinyDLLM(tp.nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 8) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(vocab, hidden)
        self.head = tp.nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))


def steps(n: int, canvas=(7, 7, 7)):
    return tuple(
        DLLMStep(
            canvas_ids=list(canvas),
            sampled_ids=[3, 4, 1],
            logprobs=[-0.5] * len(canvas),
            accept_mask=[1, 0, 0],
            timestep=i / n,
            version=0,
        )
        for i in range(n)
    )


def records():
    return (
        DLLMTrajectoryRecord(1, [1, 2], [3, 4, 1], steps(2), 1.0),
        DLLMTrajectoryRecord(2, [1, 2], [5, 6, 1], steps(2), 0.0),
    )


class FakeTrajectoryGenerator:
    def generate(self, prompts, *, config=None):
        traj = records()
        sequences = [list(p) + r.completion_ids for p, r in zip(prompts, traj)]
        return TrajectoryGenerationResult(sequences, traj)


def test_step_validation_and_binary_roundtrip():
    record = records()[0]
    encode_trajectory(record)
    restored = decode_trajectory(encode_trajectory(record))
    assert restored.prompt_ids == record.prompt_ids
    assert restored.reward == record.reward
    assert len(restored.steps) == 2
    assert restored.steps[1].accept_mask == [1, 0, 0]

    store = DLLMTrajectoryStore()
    handle = store.put(record)
    assert store.get(handle).completion_ids == [3, 4, 1]
    store.release(handle)

    bad = DLLMStep([7], [1], [-0.1], [2], 0.0, 0)
    try:
        bad.validate(1)
    except ValueError as exc:
        assert "binary" in str(exc)
    else:
        raise AssertionError("non-binary accept mask accepted")


def test_generator_records_exact_reverse_transitions():
    tp.manual_seed(3)
    model = TinyDLLM()
    generator = TrajectoryDiffusionGenerator(model)
    cfg = TrajectoryGenerationConfig(
        max_new_tokens=4, steps=4, mask_token_id=7, vocab_size=8,
        eos_token_ids=(),
    )
    result = generator.generate([[1, 2], [3, 4, 2]], config=cfg)
    assert len(result.sequences) == 2
    for trajectory in result.trajectories:
        assert len(trajectory.steps) == 4
        assert trajectory.steps[0].canvas_ids == [7] * 4
        assert all(
            math.isfinite(value) for step in trajectory.steps for value in step.logprobs
        )
        assert any(any(step.accept_mask) for step in trajectory.steps)


def test_rollout_emits_handle_and_preserves_provenance():
    store = DLLMTrajectoryStore()
    rollout = TrajectoryDLLMRollout(
        FakeTrajectoryGenerator(), store, version_provider=lambda: 3
    )
    request = RolloutRequest(
        1, 3, [1, 2], 8, 2,
        SamplingParams(1.0, 1.0, 0, None, True, 0), None,
    )
    emitted = list(rollout.generate(request))
    assert len(emitted) == 2
    assert all(record.payload_handle is not None for record in emitted)
    assert all(record.policy_versions == [3, 3, 3] for record in emitted)
    assert store.get(emitted[0].payload_handle).request_id == 1


def test_collate_trajectory_aligns_replay_rows():
    cfg = DLLMConfig(estimator="trajectory", mask_token_id=7)
    batch = collate_trajectory(records(), cfg)
    assert batch.size == 4  # two samples x two denoising steps
    assert batch.dllm_is_trajectory
    assert batch.input_ids[0] == [1, 2, 3, 4, 1]
    assert batch.dllm_noisy_ids[0] == [1, 2, 7, 7, 7]
    assert batch.loss_mask[0] == [0, 0, 1, 0, 0]
    assert batch.old_logprobs[0][2] == -0.5


def test_trajectory_trainer_replays_recorded_logprobs():
    tp.manual_seed(9)
    dllm_cfg = DLLMConfig(estimator="trajectory", mask_token_id=7)
    trainer = DLLMTrainer(
        TinyDLLM(),
        TrainerConfig(loss_type="diffu_grpo"),
        dllm_config=dllm_cfg,
        store=DLLMTrajectoryStore(),
    )
    batch = collate_trajectory(records(), dllm_cfg)
    before = trainer.policy.head.weight.detach().clone()
    metrics = trainer.update(batch, [1.0] * 4, dry_run=True)
    assert math.isfinite(metrics["train/loss"])
    grad = trainer.policy.head.weight.grad
    assert grad is not None and any(abs(float(v)) > 0 for v in grad.flatten().tolist())
    trainer.update(batch, [1.0] * 4)
    assert float((trainer.policy.head.weight.detach() - before).abs().sum()) > 0


def test_trajectory_controller_end_to_end():
    tp.manual_seed(13)
    store = DLLMTrajectoryStore()
    dllm_cfg = DLLMConfig(estimator="trajectory", mask_token_id=7)
    trainer = DLLMTrainer(
        TinyDLLM(),
        TrainerConfig(loss_type="diffu_grpo"),
        dllm_config=dllm_cfg,
        store=store,
    )
    rollout = TrajectoryDLLMRollout(
        FakeTrajectoryGenerator(), store, version_provider=lambda: 0
    )
    controller = DLLMController(
        ControllerConfig(group_size=2, max_tokens_per_batch=128),
        rollout=rollout,
        reward_fn=lambda record: 1.0 if record.token_ids[0] == 3 else 0.0,
        trainer=trainer,
        dllm_config=dllm_cfg,
        store=store,
    )
    result = controller.step([Prompt(1, [1, 2])])
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])


def test_async_trajectory_controller_preserves_handles():
    tp.manual_seed(17)
    store = DLLMTrajectoryStore()
    dllm_cfg = DLLMConfig(estimator="trajectory", mask_token_id=7)
    trainer = DLLMTrainer(
        TinyDLLM(),
        TrainerConfig(loss_type="diffu_grpo"),
        dllm_config=dllm_cfg,
        store=store,
    )
    rollout = TrajectoryDLLMRollout(
        FakeTrajectoryGenerator(), store, version_provider=lambda: 0
    )
    controller = AsyncDLLMController(
        AsyncControllerConfig(group_size=2, max_tokens_per_batch=128),
        rollout=rollout,
        reward_fn=lambda record: 1.0 if record.token_ids[0] == 3 else 0.0,
        trainer=trainer,
        dllm_config=dllm_cfg,
        store=store,
    )
    assert isinstance(controller.queue, DLLMPayloadQueue)

    async def scenario():
        return await controller.step([Prompt(1, [1, 2])])

    result = asyncio.run(scenario())
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])


def test_uniform_canvas_noise_requires_vocab_and_uses_alpha():
    try:
        DLLMConfig(family="block", noise_mode="uniform")
    except ValueError as exc:
        assert "vocab_size" in str(exc)
    else:
        raise AssertionError("uniform noise accepted without vocab_size")
    cfg = DLLMConfig(family="block", noise_mode="uniform", vocab_size=8)
    assert cfg.effective_noise_mode == "uniform"
