"""AGRPO (arXiv 2510.04019): step MDP, MC timestep estimation, real steps."""

import asyncio
import dataclasses
import math

import tensorplay as tp

from megarl.async_controller import AsyncControllerConfig
from megarl.contracts import RolloutRequest, SamplingParams
from megarl.controller import ControllerConfig, Prompt
from megarl.dllm import DLLMSample, default_adapter, from_dllm_record
from megarl.dllm.algorithms.agrpo import (
    AGRPOConfig,
    AGRPOController,
    AsyncAGRPOController,
    AGRPOGenerator,
    AGRPORecord,
    AGRPORollout,
    AGRPOStore,
    AGRPOTrainer,
    collate_agrpo,
    decode_agrpo,
    encode_agrpo,
    make_agrpo_packer,
    sample_mc_steps,
)
from megarl.trainer import TrainerConfig


class TinyDLLM(tp.nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 8) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(vocab, hidden)
        self.head = tp.nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))


def _config(**kwargs):
    defaults = dict(
        mask_token_id=7,
        pad_token_id=0,
        vocab_size=8,
        max_completion_length=4,
        sampling_steps=2,
        block_length=2,
        mc_samples=2,
        temperature=1.0,
        group_size=2,
    )
    defaults.update(kwargs)
    return AGRPOConfig(**defaults)


def _generated(model=None, **kwargs):
    tp.manual_seed(21)
    model = model or TinyDLLM()
    cfg = _config(**kwargs)
    result = AGRPOGenerator(model, default_adapter(cfg)).generate(
        [[1, 2], [1, 2]], config=cfg
    )
    return model, cfg, result


def _samples(records):
    return [
        DLLMSample([1, 2], list(r.completion_ids), 1.0 if i == 0 else 0.0, 0)
        for i, r in enumerate(records)
    ]


def test_config_validation_and_tokens_per_step():
    for bad in (
        {"family": "x"},
        {"max_completion_length": 5},
        {"sampling_steps": 3},
        {"mc_samples": 0},
        {"unmasking": "x"},
        {"importance_sampling": "x"},
        {"advantage_mode": "x"},
        {"group_size": 0},
    ):
        try:
            _config(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid config: {bad}")
    cfg = _config()
    assert cfg.tokens_per_step == 2
    assert cfg.noise_token_id == 7 and cfg.prompt_fill() == 0


def test_record_roundtrip():
    record = AGRPORecord(
        4,
        [1, 2],
        [3, 4, 5, 6],
        [0, 0, 1, 1],
        [-1.0, -2.0, -3.0, -4.0],
        [1.0, 2.0, 3.0, 4.0],
        reward=0.5,
        group_id=4,
        version=3,
    )
    assert decode_agrpo(encode_agrpo(record)) == record


def test_generator_records_unmask_steps_and_statistics():
    _, cfg, result = _generated()
    for record in result.records:
        assert len(record.completion_ids) == cfg.max_completion_length
        assert sorted(record.unmask_steps) == [0, 0, 1, 1]
        assert all(math.isfinite(v) for v in record.token_logprobs)
        assert all(v > 0 for v in record.entropies)


def test_mc_step_samplers_are_valid():
    _, cfg, result = _generated()
    records = tuple(
        dataclasses.replace(r, group_id=i)
        for i, r in enumerate(result.records)
    )
    steps, weights = sample_mc_steps(records, cfg)
    assert len(steps) == len(weights) == len(records) * cfg.mc_samples
    assert all(0 <= t <= 1 for t in steps)
    assert all(w == 1.0 for w in weights)

    skewed = dataclasses.replace(
        records[0],
        entropies=[0.1, 0.1, 3.0, 3.0],
    )
    entropy_cfg = _config(importance_sampling="entropy")
    esteps, eweights = sample_mc_steps((skewed, skewed), entropy_cfg)
    assert all(w > 0 for w in eweights)
    assert any(abs(w - 1.0) > 1e-9 for w in eweights)

    iid_cfg = _config(low_discrepancy=False)
    isteps, _ = sample_mc_steps(records, iid_cfg)
    assert all(0 <= t <= 1 for t in isteps)


def test_collate_builds_mc_canvases_and_scale():
    _, cfg, result = _generated()
    records = tuple(
        dataclasses.replace(r, group_id=i)
        for i, r in enumerate(result.records)
    )
    batch = collate_agrpo(_samples(records), records, cfg)
    assert batch.size == len(records) * cfg.mc_samples * cfg.tokens_per_step
    assert len(batch.dllm_prompt_lens) == batch.size
    assert all(sum(row) == 1 for row in batch.loss_mask)
    assert abs(batch.agrpo_loss_scale - batch.size / (cfg.mc_samples * cfg.group_size)) < 1e-12
    assert all(w > 0 for w in batch.agrpo_importance_weights)


def test_trainer_backprops_and_updates_weights():
    model, cfg, result = _generated()
    records = tuple(
        dataclasses.replace(r, group_id=7) for r in result.records
    )
    batch = collate_agrpo(_samples(records), records, cfg)
    trainer = AGRPOTrainer(
        model, TrainerConfig(device="cpu", learning_rate=1e-2), agrpo_config=cfg
    )
    audit = trainer.update(batch, [0.0] * batch.size, dry_run=True)
    assert math.isfinite(float(audit["train/loss"]))
    assert any(
        p.grad is not None and float(p.grad.abs().sum().flatten()[0]) > 0
        for p in model.parameters()
    )
    before = sum(float(p.detach().sum().flatten()[0]) for p in model.parameters())
    trainer.update(batch, [0.0] * batch.size, dry_run=False)
    after = sum(float(p.detach().sum().flatten()[0]) for p in model.parameters())
    assert before != after


def test_group_mean_advantage_mode():
    model, cfg, result = _generated(advantage_mode="mean")
    records = tuple(
        dataclasses.replace(r, group_id=7)
        for r in result.records
    )
    batch = collate_agrpo(_samples(records), records, cfg)
    trainer = AGRPOTrainer(model, TrainerConfig(device="cpu"), agrpo_config=cfg)
    values = trainer._row_advantages(batch)
    # rewards (1, 0), group mean 0.5 -> +/-0.5 times scale and weight.
    assert any(v > 0 for v in values) and any(v < 0 for v in values)


def test_rollout_and_packer_release_handles():
    tp.manual_seed(31)
    model = TinyDLLM()
    store = AGRPOStore()
    cfg = _config()
    rollout = AGRPORollout(
        AGRPOGenerator(model, default_adapter(cfg)), store, cfg
    )
    request = RolloutRequest(
        12,
        0,
        [1, 2],
        cfg.max_completion_length,
        2,
        SamplingParams(1.0, 1.0, 0, None, False, 0),
        None,
    )
    records = list(rollout.generate(request))
    assert len(records) == 2 and all(r.payload_handle is not None for r in records)
    samples = [
        from_dllm_record(r, [1, 2], reward=1.0 if i == 0 else 0.0)
        for i, r in enumerate(records)
    ]
    batches, orders = make_agrpo_packer(cfg, store)(samples)
    assert len(batches) == 1 and len(orders[0]) == batches[0].size
    assert orders[0] == batches[0].agrpo_sample_ids


def _controller(async_mode: bool):
    tp.manual_seed(41)
    model = TinyDLLM()
    store = AGRPOStore()
    cfg = _config()
    rollout = AGRPORollout(
        AGRPOGenerator(model, default_adapter(cfg)), store, cfg
    )
    trainer = AGRPOTrainer(
        model, TrainerConfig(learning_rate=1e-3), agrpo_config=cfg
    )
    scores = {"n": 0}

    def score(record):
        scores["n"] += 1
        return 1.0 if scores["n"] % 2 == 1 else 0.0

    cls = AsyncAGRPOController if async_mode else AGRPOController
    config_cls = AsyncControllerConfig if async_mode else ControllerConfig
    return cls(
        config_cls(group_size=2, max_tokens_per_batch=128),
        rollout=rollout,
        reward_fn=score,
        trainer=trainer,
        agrpo_config=cfg,
        store=store,
    )


def test_agrpo_controller_end_to_end():
    result = _controller(False).step([Prompt(1, [1, 2])])
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])


def test_async_agrpo_controller_end_to_end():
    result = asyncio.run(_controller(True).step([Prompt(1, [1, 2])]))
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])
