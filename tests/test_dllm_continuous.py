"""Continuous latent diffusion RL: Flow-GRPO and DiffNFT."""

import asyncio
import math

import tensorplay as tp

from megarl.async_controller import AsyncControllerConfig
from megarl.contracts import RolloutRequest, SamplingParams
from megarl.controller import ControllerConfig, Prompt
from megarl.dllm.algorithms.continuous import (
    AsyncFlowController,
    FlowBatch,
    FlowConfig,
    FlowController,
    FlowStep,
    FlowTrajectoryRecord,
    FlowTrajectoryResult,
    FlowTrajectoryRollout,
    FlowTrajectoryStore,
    FlowTrainer,
    collate_flow,
    decode_flow_trajectory,
    diffnft_loss,
    encode_flow_trajectory,
    flow_grpo_loss,
    flow_step_logprob,
    flow_step_stats,
    make_flow_packer,
)
from megarl.trainer import TrainerConfig


def make_record(reward=1.0, handle_id=1):
    latents = tuple(
        tuple((0.1 * step + dim * 0.2) for dim in range(3)) for step in range(4)
    )
    steps = tuple(
        FlowStep(
            timestep=1.0 - 0.25 * i,
            next_timestep=1.0 - 0.25 * (i + 1),
            log_prob=-0.5 - 0.1 * i,
        )
        for i in range(3)
    )
    return FlowTrajectoryRecord(handle_id, latents, steps, reward)


class TinyFlow(tp.nn.Module):
    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.head = tp.nn.Linear(width, width)

    def forward(self, latents, timesteps):
        return self.head(latents)


def model_step(model, latents, timesteps):
    return model(latents, timesteps)


def test_flow_gaussian_logprob_matches_closed_form():
    cfg = FlowConfig(noise_level=0.5)
    xt = tp.zeros((1, 1, 2))
    out = tp.zeros((1, 1, 2))
    sigma = tp.tensor([[0.5]])
    sigma_next = tp.tensor([[0.4]])
    mu, std = flow_step_stats(xt, out, sigma, sigma_next, cfg)
    expected_std = math.sqrt(0.5 / 0.5) * 0.5 * math.sqrt(0.1)
    assert abs(float(std.flatten()[0]) - expected_std) < 1e-6
    logp = flow_step_logprob(mu, xt, out, sigma, sigma_next, cfg)
    expected = -math.log(expected_std) - 0.5 * math.log(2.0 * math.pi)
    assert abs(float(logp.flatten()[0]) - expected) < 1e-5


def test_flow_grpo_loss_gradient_increases_good_logprob():
    new = tp.tensor([[0.0]], requires_grad=True)
    old = tp.zeros((1, 1))
    loss, diag = flow_grpo_loss(
        new, old, tp.ones((1,)), tp.ones((1, 1)), clip_low=0.2
    )
    loss.backward()
    assert new.grad is not None and float(new.grad.flatten()[0]) < 0
    assert math.isfinite(float(diag["flow_ratio_mean"].flatten()[0]))


def test_diffnft_loss_backprops_only_to_live_prediction():
    x0 = tp.zeros((2, 3))
    xt = tp.ones((2, 3))
    sigma = tp.tensor([0.5, 0.4])
    new = tp.ones((2, 3), requires_grad=True)
    old = tp.full((2, 3), 0.5)
    reward = tp.tensor([1.0, 0.0])
    loss, diag = diffnft_loss(
        x0, xt, sigma, new, old, reward, beta=0.1, adv_clip_max=1.0
    )
    loss.backward()
    assert new.grad is not None and any(abs(float(v)) > 0 for v in new.grad.flatten().tolist())
    assert math.isfinite(float(diag["nft_old_deviate"].flatten()[0]))


def test_flow_store_and_collate_roundtrip():
    record = make_record()
    store = FlowTrajectoryStore()
    handle = store.put(record)
    restored = store.get(handle)
    assert all(
        abs(a - b) < 1e-6 for row, restored_row in zip(restored.latents, record.latents)
        for a, b in zip(row, restored_row)
    )
    assert len(restored.steps) == 3
    assert decode_flow_trajectory(encode_flow_trajectory(record)).reward == 1.0
    store.release(handle)

    batch = collate_flow([record, make_record(0.0, 2)], FlowConfig())
    assert isinstance(batch, FlowBatch)
    assert batch.size == 2 and batch.steps == 3
    assert len(batch.latents[0]) == 4


def test_flow_trainer_flow_grpo_updates_policy():
    tp.manual_seed(3)
    policy = TinyFlow()
    cfg = FlowConfig(objective="flow_grpo", num_train_steps=3)
    trainer = FlowTrainer(
        policy,
        TrainerConfig(learning_rate=1e-3),
        flow_config=cfg,
        model_step_fn=model_step,
    )
    batch = collate_flow([make_record()], cfg)
    before = policy.head.weight.detach().clone()
    metrics = trainer.update(batch, [1.0], dry_run=True)
    assert math.isfinite(metrics["train/loss"])
    assert policy.head.weight.grad is not None
    trainer.update(batch, [1.0])
    assert float((policy.head.weight.detach() - before).abs().sum()) > 0


def test_flow_trainer_diffnft_updates_policy_and_ema():
    tp.manual_seed(5)
    policy = TinyFlow()
    old = TinyFlow()
    cfg = FlowConfig(objective="diffnft", num_nft_timesteps=2)
    trainer = FlowTrainer(
        policy,
        TrainerConfig(learning_rate=1e-3),
        flow_config=cfg,
        model_step_fn=model_step,
        old_model=old,
    )
    trainer.ema_update(0.9)
    before = policy.head.weight.detach().clone()
    batch = collate_flow([make_record()], cfg)
    metrics = trainer.update(batch, [1.0])
    assert math.isfinite(metrics["train/loss"])
    assert float((policy.head.weight.detach() - before).abs().sum()) > 0


def test_flow_rollout_preserves_out_of_band_handle():
    store = FlowTrajectoryStore()

    class Generator:
        def generate(self, prompts, *, config=None):
            return FlowTrajectoryResult((make_record(1.0, 1), make_record(0.0, 2)))

    rollout = FlowTrajectoryRollout(
        Generator(), store, version_provider=lambda: 7
    )
    request = RolloutRequest(
        11, 2, [1, 2], 3, 2, SamplingParams(1.0, 1.0, 0, None, True, 0), None
    )
    records = list(rollout.generate(request))
    assert len(records) == 2
    assert all(r.payload_handle is not None for r in records)
    assert all(r.policy_versions == [7] for r in records)
    from megarl.dllm import from_dllm_record

    samples = [
        from_dllm_record(record, [1, 2], reward=1.0 if i == 0 else 0.0)
        for i, record in enumerate(records)
    ]
    packer = make_flow_packer(FlowConfig(), store)
    batches, orders = packer(samples)
    assert len(batches) == 1 and orders == [[0, 1]]


def _flow_controller(async_mode: bool):
    store = FlowTrajectoryStore()

    class Generator:
        def generate(self, prompts, *, config=None):
            return FlowTrajectoryResult(
                (make_record(1.0, 1), make_record(0.0, 2))
            )

    cfg = FlowConfig(objective="flow_grpo")
    trainer = FlowTrainer(
        TinyFlow(),
        TrainerConfig(learning_rate=1e-3),
        flow_config=cfg,
        model_step_fn=model_step,
    )
    rollout = FlowTrajectoryRollout(Generator(), store, version_provider=lambda: 0)
    cls = AsyncFlowController if async_mode else FlowController
    config_cls = AsyncControllerConfig if async_mode else ControllerConfig
    scored = {"n": 0}

    def score(record):
        scored["n"] += 1
        return 1.0 if scored["n"] % 2 == 1 else 0.0

    return cls(
        config_cls(group_size=2, max_tokens_per_batch=128),
        rollout=rollout,
        reward_fn=score,
        trainer=trainer,
        flow_config=cfg,
        store=store,
    )


def test_flow_controller_end_to_end():
    controller = _flow_controller(False)
    result = controller.step([Prompt(1, [1, 2])])
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])


def test_async_flow_controller_end_to_end():
    controller = _flow_controller(True)
    result = asyncio.run(controller.step([Prompt(1, [1, 2])]))
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])
