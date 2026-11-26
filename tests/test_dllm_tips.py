import asyncio
import math

import tensorplay as tp
import tensorplay.nn.functional as F

from megarl.contracts import RolloutRequest, SamplingParams
from megarl.controller import ControllerConfig, Prompt
from megarl.async_controller import AsyncControllerConfig
from megarl.dllm import DLLMSample, from_dllm_record
from megarl.dllm.algorithms.tips import (
    TIPSConfig,
    TIPSGenerator,
    TIPSPolicy,
    TIPSRecord,
    TIPSRollout,
    TIPSStore,
    TIPSTrainer,
    calibrate_tips_shift,
    collate_tips,
    decode_tips,
    encode_tips,
    make_tips_packer,
    tips_loss,
    tips_visibility_features,
)
from megarl.trainer import TrainerConfig


class TinyDLLM(tp.nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 8) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(vocab, hidden)
        self.mix = tp.nn.Linear(hidden, hidden)
        self.head = tp.nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        e = self.emb(input_ids)
        shifted = tp.cat([e[:, -1:], e[:, :-1]], dim=1)
        return self.head(self.mix(e + 0.4 * shifted))


def _config(**kwargs):
    base = dict(
        mask_token_id=7,
        vocab_size=8,
        max_completion_length=4,
        budget=4,
        gamma=0.0,
        hidden=16,
        heads=2,
        time_dim=8,
        layers=2,
        seed=3,
    )
    base.update(kwargs)
    return TIPSConfig(**base)


def _policy(cfg):
    return TIPSPolicy(
        hidden=cfg.hidden, heads=cfg.heads, time_dim=cfg.time_dim,
        layers=cfg.layers, max_len=cfg.max_completion_length + 8,
    )


def _generated(model=None, **kwargs):
    tp.manual_seed(13)
    model = model or TinyDLLM()
    cfg = _config(**kwargs)
    policy = _policy(cfg)
    gen = TIPSGenerator(model, policy)
    result = gen.generate([[1, 2], [1, 3]], config=cfg)
    return model, cfg, policy, result


def _samples(result, rewards=(1.0, 0.0)):
    samples = []
    store = TIPSStore()
    handles = []
    for record in result.records:
        handles.append(store.put(record))
    for i, (record, handle) in enumerate(zip(result.records, handles)):
        wire = type("W", (), {})()
        from megarl.contracts import TrajectoryRecord

        w = TrajectoryRecord(
            request_id=1,
            prompt_len=len(record.prompt_ids),
            token_ids=list(record.completion_ids),
            logprobs=[0.0] * len(record.completion_ids),
            top_logprobs=[],
            policy_versions=[0] * len(record.completion_ids),
            reward=None,
            finish_reason=0,
            payload_handle=handle,
        )
        samples.append(from_dllm_record(w, record.prompt_ids, reward=rewards[i % 2]))
    return samples, store


def _request(n=2):
    return RolloutRequest(
        1, 0, [1, 2], 4, n,
        SamplingParams(1.0, 1.0, 0, None, False, 0), None,
    )


def test_config_validation_and_schedule():
    cfg = _config()
    assert cfg.schedule(0.0) == 0.0
    assert cfg.schedule(1.0) == 1.0
    assert cfg.schedule(0.5) == 0.5
    assert cfg.train_steps == cfg.budget - 1
    assert cfg.noise_token_id == 7
    assert cfg.prompt_fill() == 7
    for bad in ("other",):
        try:
            _config(family=bad)
            raise AssertionError("bad family accepted")
        except ValueError:
            pass
    try:
        _config(budget=0)
        raise AssertionError("bad budget accepted")
    except ValueError:
        pass
    try:
        _config(hidden=6, heads=4)
        raise AssertionError("hidden not divisible by heads accepted")
    except ValueError:
        pass


def test_record_roundtrip():
    _, _, _, result = _generated()
    record = result.records[0]
    assert decode_tips(encode_tips(record)) == record
    assert all(step.time < 1.0 for step in record.steps)


def test_generation_records_visibility_decisions():
    _, cfg, _, result = _generated()
    assert len(result.records) == 2
    for record in result.records:
        record.validate()
        assert len(record.steps) == cfg.train_steps
        for step in record.steps:
            assert len(step.features) == cfg.max_completion_length
            assert all(len(f) == 2 for f in step.features)
            assert len(step.rho) == cfg.max_completion_length
            assert len(step.actions) == cfg.max_completion_length
            assert all(a in (0, 1) for a in step.actions)
            assert math.isfinite(step.old_logprob)


def test_schedule_calibration_hits_target():
    logits = tp.tensor([[0.3, -0.5, 1.1, 0.0]])
    for kappa in (0.25, 0.5, 0.8):
        delta = calibrate_tips_shift(logits, kappa)
        rho = F.sigmoid(logits + delta.unsqueeze(-1))
        assert abs(float(rho.mean().flatten()[0]) - kappa) < 1e-4


def test_visibility_features_are_finite():
    logits = tp.randn((2, 4, 8))
    canvas = tp.tensor([[7, 7, 1, 2], [7, 3, 7, 7]])
    features, mask, greedy = tips_visibility_features(logits, canvas, noise_token_id=7)
    assert tuple(features.shape) == (2, 4, 2)
    assert bool(tp.isfinite(features).all().flatten()[0])
    assert bool((mask.sum(dim=-1) > 0).all().flatten()[0])


def test_collate_aligns_rows_with_samples():
    _, cfg, _, result = _generated()
    samples, store = _samples(result)
    batch = collate_tips(samples, [store.get(s.payload_handle) for s in samples], cfg)
    assert batch.size == 2 * cfg.train_steps
    assert batch.tips_sample_ids == [0, 0, 0, 1, 1, 1]
    assert len(batch.tips_features) == batch.size
    assert len(batch.tips_actions) == batch.size


def test_loss_backprops_towards_actions():
    model, cfg, policy, result = _generated()
    samples, store = _samples(result)
    batch = collate_tips(samples, [store.get(s.payload_handle) for s in samples], cfg)
    tensors = type("T", (), {})()
    from megarl.dllm.algorithms.tips import TIPSTensorBatch

    tensors = TIPSTensorBatch(batch, device="cpu")
    logits = policy(tensors.tips_features, tensors.tips_mask, tensors.tips_time)
    loss, diag = tips_loss(
        logits, tensors, tp.ones((batch.size,)),
        old_logprobs=tensors.tips_old_logprobs,
        clip_low=cfg.clip_low, clip_high=cfg.clip_high,
    )
    assert math.isfinite(float(loss.flatten()[0]))
    assert "tips_mean_ratio" in diag
    loss.backward()
    assert any(p.grad is not None and bool((p.grad != 0).any().flatten()[0]) for p in policy.parameters())


def test_trainer_backprops_and_updates_weights():
    model, cfg, policy, result = _generated()
    samples, store = _samples(result)
    batch = collate_tips(samples, [store.get(s.payload_handle) for s in samples], cfg)
    trainer = TIPSTrainer(
        policy, TrainerConfig(device="cpu", learning_rate=1e-2),
        tips_config=cfg,
    )
    before = float(policy.head.weight.detach().abs().sum().flatten()[0])
    audit = trainer.update(batch, [1.0] * batch.size, dry_run=True)
    assert math.isfinite(float(audit["train/loss"]))
    assert any(p.grad is not None and bool((p.grad != 0).any().flatten()[0]) for p in policy.parameters())
    trainer.update(batch, [1.0] * batch.size, dry_run=False)
    after = float(policy.head.weight.detach().abs().sum().flatten()[0])
    assert before != after


def test_rollout_preserves_handles_and_packer_releases():
    tp.manual_seed(3)
    cfg = _config()
    policy = _policy(cfg)
    store = TIPSStore()
    rollout = TIPSRollout(TIPSGenerator(TinyDLLM(), policy), store, cfg)
    records = list(rollout.generate(_request()))
    assert all(r.payload_handle is not None for r in records)
    samples = [
        from_dllm_record(r, [1, 2], reward=1.0 if i == 0 else 0.0)
        for i, r in enumerate(records)
    ]
    batches, orders = make_tips_packer(cfg, store)(samples)
    assert orders[0] == [0, 0, 0, 1, 1, 1]
    assert batches[0].size == 6


def _controller(async_mode):
    tp.manual_seed(5)
    cfg = _config()
    policy = _policy(cfg)
    store = TIPSStore()
    rollout = TIPSRollout(TIPSGenerator(TinyDLLM(), policy), store, cfg)
    trainer = TIPSTrainer(
        policy, TrainerConfig(device="cpu", learning_rate=1e-3), tips_config=cfg
    )
    cls = type("C", (), {})
    from megarl.dllm.algorithms.tips import AsyncTIPSController, TIPSController

    cls = AsyncTIPSController if async_mode else TIPSController
    config_cls = AsyncControllerConfig if async_mode else ControllerConfig
    score = {"n": 0}

    def reward(_):
        score["n"] += 1
        return 1.0 if score["n"] % 2 else 0.0

    return cls(
        config_cls(group_size=2, max_tokens_per_batch=256),
        rollout=rollout, reward_fn=reward, trainer=trainer,
        tips_config=cfg, store=store,
    )


def test_tips_controller_end_to_end():
    result = _controller(False).step([Prompt(1, [1, 2])])
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])


def test_async_tips_controller_end_to_end():
    result = asyncio.run(_controller(True).step([Prompt(1, [1, 2])]))
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])
