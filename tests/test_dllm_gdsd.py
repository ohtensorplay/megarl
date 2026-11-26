"""GDSD (arXiv 2605.29398): coupled corruption views, guidance, real steps."""

import asyncio
import math

import tensorplay as tp

from megarl.async_controller import AsyncControllerConfig
from megarl.contracts import RolloutRequest, SamplingParams
from megarl.controller import ControllerConfig, Prompt
from megarl.dllm import DLLMSample, default_adapter, from_dllm_record
from megarl.dllm.algorithms.gdsd import (
    AsyncGDSDController,
    GDSDBatch,
    GDSDConfig,
    GDSDController,
    GDSDGenerationResult,
    GDSDGenerator,
    GDSDRecord,
    GDSDRollout,
    GDSDTensorBatch,
    GDSDTrainer,
    collate_gdsd,
    gdsd_loss,
    gdsd_token_logprobs,
    make_gdsd_packer,
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
        vocab_size=8,
        max_completion_length=4,
        mc_timesteps=2,
        sampling_steps=4,
        beta=0.0,
        seed=3,
    )
    defaults.update(kwargs)
    return GDSDConfig(**defaults)


def _generated(model=None, **kwargs):
    tp.manual_seed(13)
    model = model or TinyDLLM()
    cfg = _config(**kwargs)
    result = GDSDGenerator(model, default_adapter(cfg)).generate(
        [[1, 2], [1, 3]], config=cfg
    )
    return model, cfg, result


def _samples(result, rewards=(1.0, 0.0)):
    return [
        DLLMSample(r.prompt_ids, list(r.completion_ids), rewards[i])
        for i, r in enumerate(result.records)
    ]


def test_config_validation_and_helpers():
    for bad in (
        {"family": "nope"},
        {"max_completion_length": 0},
        {"mc_timesteps": 0},
        {"beta": -0.1},
        {"psi": 0.0},
        {"refresh_every": 0},
        {"version": -1},
        {"family": "block", "vocab_size": None},
    ):
        try:
            _config(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid config: {bad}")
    cfg = _config()
    assert cfg.noise_token_id == 7 and cfg.prompt_fill() == 0
    assert cfg.rows_per_sample == 4
    assert _config(coupled_views=False).rows_per_sample == 2


def test_record_roundtrip():
    record = GDSDRecord(3, [1, 2], [3, 4], reward=1.0, group_id=3, version=2)
    assert record.length == 2
    assert GDSDRecord(3, [1, 2], [3, 4], reward=1.0, group_id=3, version=2) == record


def test_generation_records_full_completions():
    _, cfg, result = _generated()
    assert isinstance(result, GDSDGenerationResult)
    assert len(result.records) == 2
    for record in result.records:
        assert record.length == cfg.max_completion_length
        assert all(0 <= token < cfg.vocab_size for token in record.completion_ids)
        record.validate()


def test_collate_coupled_views_and_weights():
    _, cfg, result = _generated()
    batch = collate_gdsd(_samples(result), cfg)
    assert isinstance(batch, GDSDBatch)
    assert batch.size == 2 * cfg.rows_per_sample
    assert len(batch.gdsd_weights) == batch.size
    assert all(w >= 1.0 for w in batch.gdsd_weights)
    assert all(len(row) == batch.max_len for row in batch.dllm_noisy_ids)
    assert batch.gdsd_sample_ids.count(0) == cfg.rows_per_sample
    # Coupled views cover complementary position sets.
    first, second = batch.loss_mask[0], batch.loss_mask[1]
    assert all(a + b == 1 for a, b in zip(first[2:], second[2:]))
    assert all(sum(row[2:]) > 0 for row in (first, second))


def test_tlc_is_softmax_invariant():
    model, cfg, result = _generated()
    samples = _samples(result)
    batch = collate_gdsd(samples, cfg)
    tensors = GDSDTensorBatch(batch, device="cpu")
    logits = model(tensors.dllm_noisy_ids).float()
    centered = gdsd_token_logprobs(model, tensors, cfg, adapter=default_adapter(cfg))
    expected = (
        logits - logits.mean(dim=-1, keepdim=True)
    ).log_softmax(dim=-1).gather(-1, tensors.input_ids.unsqueeze(-1)).squeeze(-1)
    assert tp.allclose(centered, expected)


def test_guidance_moves_loss_in_advantage_direction():
    _, cfg, result = _generated(coupled_views=False, mc_timesteps=1)
    batch = collate_gdsd(_samples(result), cfg)
    tensors = GDSDTensorBatch(batch, device="cpu")
    logits = tp.zeros_like(tensors.dllm_noisy_ids, dtype=tp.float32).unsqueeze(-1).expand(
        *tensors.dllm_noisy_ids.shape, cfg.vocab_size
    ).contiguous()
    new_token = tp.gather(
        logits.log_softmax(-1), -1, tensors.input_ids.unsqueeze(-1)
    ).squeeze(-1).detach()
    adv = [1.0] * batch.size
    kwargs = dict(
        old_logprobs=new_token,
        psi=cfg.psi,
        beta=0.0,
        epsilon_low=cfg.epsilon_low,
        epsilon_high=cfg.epsilon_high,
        tlc=cfg.tlc,
    )
    before, _ = gdsd_loss(logits, tensors, adv, **kwargs)
    # Lowering the behaviour log-prob increases the ratio, which must reduce
    # the squared guidance error for a positive advantage.
    kwargs["old_logprobs"] = new_token - 0.05
    after, _ = gdsd_loss(logits, tensors, adv, **kwargs)
    assert float(after) < float(before)


def test_trainer_backprops_and_updates_weights():
    model, cfg, result = _generated(beta=0.0)
    batch = collate_gdsd(_samples(result), cfg)
    trainer = GDSDTrainer(
        model, TrainerConfig(device="cpu", learning_rate=1e-2), gdsd_config=cfg
    )
    adv = [0.5 if i == 0 else -0.5 for i in batch.gdsd_sample_ids]
    metrics = trainer.update(batch, adv, dry_run=True)
    assert math.isfinite(float(metrics["train/loss"]))
    assert any(
        p.grad is not None and float(p.grad.abs().sum().flatten()[0]) > 0
        for p in model.parameters()
    )
    before = sum(float(p.detach().sum().flatten()[0]) for p in model.parameters())
    trainer.update(batch, adv, dry_run=False)
    after = sum(float(p.detach().sum().flatten()[0]) for p in model.parameters())
    assert before != after


def test_old_snapshot_refresh():
    model, cfg, result = _generated(refresh_every=2)
    old = TinyDLLM()
    trainer = GDSDTrainer(
        model, TrainerConfig(device="cpu", learning_rate=1e-2),
        gdsd_config=cfg, old_model=old,
    )
    batch = collate_gdsd(_samples(result), cfg)
    adv = [0.5 if i == 0 else -0.5 for i in batch.gdsd_sample_ids]
    trainer.update(batch, adv, dry_run=False)  # step 1: no refresh
    assert float(trainer.update(batch, adv, dry_run=False)["train/old_syncs"]) == 1.0


def test_rollout_and_packer_release_handles():
    tp.manual_seed(7)
    model = TinyDLLM()
    cfg = _config()
    rollout = GDSDRollout(
        GDSDGenerator(model, default_adapter(cfg)), cfg,
        version_provider=lambda: 4,
    )
    request = RolloutRequest(
        9, 0, [1, 2], cfg.max_completion_length, 2,
        SamplingParams(1.0, 1.0, 0, None, False, 0), None,
    )
    records = list(rollout.generate(request))
    assert len(records) == 2
    assert all(r.payload_handle is None for r in records)  # no bulk arrays
    assert all(r.policy_versions[0] == 4 for r in records)
    samples = [
        from_dllm_record(r, [1, 2], reward=1.0 if i == 0 else 0.0)
        for i, r in enumerate(records)
    ]
    batches, orders = make_gdsd_packer(cfg)(samples)
    assert len(batches) == 1 and len(orders[0]) == batches[0].size
    assert orders[0] == batches[0].gdsd_sample_ids


def _controller(async_mode: bool):
    tp.manual_seed(5)
    model = TinyDLLM()
    cfg = _config()
    rollout = GDSDRollout(GDSDGenerator(model, default_adapter(cfg)), cfg)
    trainer = GDSDTrainer(model, TrainerConfig(learning_rate=1e-3), gdsd_config=cfg)
    scores = {"n": 0}

    def score(_):
        scores["n"] += 1
        return 1.0 if scores["n"] % 2 else 0.0

    cls = AsyncGDSDController if async_mode else GDSDController
    config_cls = AsyncControllerConfig if async_mode else ControllerConfig
    return cls(
        config_cls(group_size=2, max_tokens_per_batch=128, advantage_estimator="reinforce"),
        rollout=rollout,
        reward_fn=score,
        trainer=trainer,
        gdsd_config=cfg,
    )


def test_gdsd_controller_end_to_end():
    result = _controller(False).step([Prompt(1, [1, 2])])
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])


def test_async_gdsd_controller_end_to_end():
    result = asyncio.run(_controller(True).step([Prompt(1, [1, 2])]))
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])
