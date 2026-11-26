"""Stepwise process reward / value head for replayed dLLM trajectories."""

import math

import tensorplay as tp

from megarl.dllm import DLLMConfig, collate_trajectory
from megarl.dllm.algorithms.process import (
    DLLMProcessTrainer,
    DLLMValueHead,
    ProcessRewardConfig,
    dllm_value_fn,
    process_gae,
)
from megarl.dllm.trajectory import DLLMTrajectoryStore
from megarl.trainer import TrainerConfig
from tests.test_dllm_trajectory import TinyDLLM, records


def test_process_gae_places_outcome_reward_on_last_step():
    advantages, returns = process_gae(
        [1.0], [[0.5, 0.5]], [[1, 1]], gamma=1.0, lam=1.0
    )
    assert advantages == [[0.5, 0.5]]
    assert returns == [[1.0, 1.0]]
    masked, _ = process_gae([1.0], [[0.5, 0.5]], [[1, 0]], gamma=0.9, lam=0.9)
    assert masked[0][0] == 0.5 and masked[0][1] == 0.0


def test_value_head_and_trainer_hook_broadcast_over_canvas():
    tp.manual_seed(1)
    head = DLLMValueHead(vocab_size=8, hidden=8)
    cfg = DLLMConfig(estimator="trajectory", mask_token_id=7)
    batch = collate_trajectory(records(), cfg)
    from megarl.dllm import DLLMTensorBatch

    tensors = DLLMTensorBatch(batch)
    values = dllm_value_fn(head, tensors)
    assert tuple(values.shape) == (batch.size, batch.max_len)
    assert all(math.isfinite(v) for v in values.flatten().tolist()[:1])


def test_process_trainer_updates_policy_and_value_head():
    tp.manual_seed(11)
    dllm_cfg = DLLMConfig(estimator="trajectory", mask_token_id=7)
    policy = TinyDLLM()
    value_model = DLLMValueHead(vocab_size=8, hidden=8)
    trainer = DLLMProcessTrainer(
        policy,
        TrainerConfig(learning_rate=1e-3, value_coef=0.5, max_grad_norm=None),
        dllm_config=dllm_cfg,
        value_model=value_model,
        process_config=ProcessRewardConfig(gamma=0.9, lam=0.95),
        store=DLLMTrajectoryStore(),
    )
    batch = collate_trajectory(records(), dllm_cfg)
    before_policy = policy.head.weight.detach().clone()
    before_value = value_model.mlp[-1].weight.detach().clone()

    metrics = trainer.update(batch, [1.0] * batch.size, dry_run=True)
    assert math.isfinite(metrics["train/loss"])
    assert policy.head.weight.grad is not None
    assert value_model.mlp[-1].weight.grad is not None

    trainer.update(batch, [1.0] * batch.size)
    assert float((policy.head.weight.detach() - before_policy).abs().sum()) > 0
    assert float((value_model.mlp[-1].weight.detach() - before_value).abs().sum()) > 0


def test_process_trainer_rejects_one_step_estimator():
    cfg = DLLMConfig(estimator="one_step", mask_token_id=7)
    try:
        DLLMProcessTrainer(
            TinyDLLM(),
            TrainerConfig(),
            dllm_config=cfg,
            value_model=DLLMValueHead(vocab_size=8),
        )
    except ValueError as exc:
        assert "trajectory" in str(exc)
    else:
        raise AssertionError("one-step process trainer accepted")
