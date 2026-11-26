"""JustGRPO (Flexibility Trap, ICML 2026): AR canvas, fast mode, real steps."""

import asyncio
import math

import tensorplay as tp

from megarl.async_controller import AsyncControllerConfig
from megarl.contracts import RolloutRequest, SamplingParams
from megarl.controller import ControllerConfig, Prompt
from megarl.dllm import DLLMSample, default_adapter, from_dllm_record
from megarl.dllm.algorithms.justgrpo import (
    AsyncJustGRPOController,
    JustGRPOConfig,
    JustGRPOController,
    JustGRPOGenerator,
    JustGRPORecord,
    JustGRPORollout,
    JustGRPOStore,
    JustGRPOTrainer,
    collate_justgrpo,
    decode_justgrpo,
    encode_justgrpo,
    make_justgrpo_packer,
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
        max_new_tokens=4,
        temperature=1.0,
    )
    defaults.update(kwargs)
    return JustGRPOConfig(**defaults)


def _generated(model=None, **kwargs):
    tp.manual_seed(11)
    model = model or TinyDLLM()
    cfg = _config(**kwargs)
    result = JustGRPOGenerator(model, default_adapter(cfg)).generate(
        [[1, 2], [1, 2]], config=cfg
    )
    return model, cfg, result


def _samples(records, rewards=(1.0, 0.0)):
    return [
        DLLMSample([1, 2], list(record.completion_ids), rewards[i], 0)
        for i, record in enumerate(records)
    ]


def test_config_validation_and_helpers():
    for bad in (
        {"family": "nope"},
        {"max_new_tokens": 0},
        {"temperature": 0.0},
        {"fast_fraction": 1.4},
        {"kl_type": "bad"},
        {"version": -1},
    ):
        try:
            _config(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid config: {bad}")
    cfg = _config()
    assert cfg.noise_token_id == 7 and cfg.prompt_fill() == 0
    assert cfg.selected_positions(4) == 4
    assert _config(fast_fraction=0.5).selected_positions(5) == 2


def test_record_roundtrip():
    record = JustGRPORecord(
        3,
        [1, 2],
        [3, 4],
        [-1.0, -2.0],
        [1.5, 0.5],
        reward=1.0,
        group_id=3,
        version=2,
    )
    decoded = decode_justgrpo(encode_justgrpo(record))
    assert decoded == record


def test_ar_generation_records_exact_step_logprobs():
    model, cfg, result = _generated()
    assert len(result.records) == 2
    record = result.records[0]
    assert len(record.completion_ids) == cfg.max_new_tokens
    assert all(math.isfinite(v) for v in record.token_logprobs)
    assert all(v > 0 for v in record.entropies)
    # Recompute the log-prob at position 2 from the same AR canvas.
    prefix = record.completion_ids[:2]
    canvas = tp.tensor(
        [[1, 2] + prefix + [cfg.noise_token_id] * (cfg.max_new_tokens - 2)],
        dtype=tp.long,
    )
    logits = model(canvas)[:, 4, :]
    logp = float(
        tp.gather(
            tp.log_softmax(logits.float(), -1),
            -1,
            tp.tensor([[record.completion_ids[2]]], dtype=tp.long),
        ).flatten()[0]
    )
    assert abs(logp - record.token_logprobs[2]) < 1e-5


def test_collate_full_and_fast_modes():
    _, cfg, result = _generated()
    samples = _samples(result.records)
    full = collate_justgrpo(samples, result.records, cfg)
    assert full.size == 2 * cfg.max_new_tokens
    assert all(sum(row) == 1 for row in full.loss_mask)
    assert full.justgrpo_loss_scale == 1.0
    assert full.justgrpo_sample_ids[:4] == [0, 0, 0, 0]

    fast_cfg = _config(fast_fraction=0.5)
    fast = collate_justgrpo(samples, result.records, fast_cfg)
    assert fast.size == 2 * 2
    assert abs(fast.justgrpo_loss_scale - 0.5) < 1e-12
    assert tuple(fast.justgrpo_step_ids[:2]) == (0, 2) or set(
        fast.justgrpo_step_ids[:2]
    ) <= {0, 1, 2, 3}


def test_trainer_backprops_and_updates_weights():
    model, cfg, result = _generated()
    batch = collate_justgrpo(_samples(result.records), result.records, cfg)
    trainer = JustGRPOTrainer(
        model,
        TrainerConfig(device="cpu", learning_rate=1e-2),
        justgrpo_config=cfg,
    )
    advantages = [batch.rewards[i] for i in range(batch.size)]
    audit = trainer.update(batch, advantages, dry_run=True)
    assert math.isfinite(float(audit["train/loss"]))
    assert any(
        p.grad is not None and float(p.grad.abs().sum().flatten()[0]) > 0
        for p in model.parameters()
    )
    before = sum(float(p.detach().sum().flatten()[0]) for p in model.parameters())
    trainer.update(batch, advantages, dry_run=False)
    after = sum(float(p.detach().sum().flatten()[0]) for p in model.parameters())
    assert before != after


def test_trainer_ref_kl_is_finite():
    model, cfg, result = _generated()
    ref = TinyDLLM()
    ref.load_state_dict(model.state_dict())
    cfg = _config(kl_coef=0.1)
    batch = collate_justgrpo(_samples(result.records), result.records, cfg)
    trainer = JustGRPOTrainer(
        model,
        TrainerConfig(device="cpu", learning_rate=1e-3),
        justgrpo_config=cfg,
        ref_model=ref,
    )
    metrics = trainer.update(
        batch, [batch.rewards[i] for i in range(batch.size)], dry_run=True
    )
    assert math.isfinite(float(metrics["train/kl"]))


def test_rollout_and_packer_release_handles():
    tp.manual_seed(3)
    model = TinyDLLM()
    store = JustGRPOStore()
    cfg = _config()
    rollout = JustGRPORollout(
        JustGRPOGenerator(model, default_adapter(cfg)), store, cfg
    )
    request = RolloutRequest(
        9,
        0,
        [1, 2],
        cfg.max_new_tokens,
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
    batches, orders = make_justgrpo_packer(cfg, store)(samples)
    assert len(batches) == 1 and len(orders[0]) == batches[0].size
    assert orders[0] == batches[0].justgrpo_sample_ids


def _controller(async_mode: bool):
    tp.manual_seed(5)
    model = TinyDLLM()
    store = JustGRPOStore()
    cfg = _config()
    rollout = JustGRPORollout(
        JustGRPOGenerator(model, default_adapter(cfg)), store, cfg
    )
    trainer = JustGRPOTrainer(
        model, TrainerConfig(learning_rate=1e-3), justgrpo_config=cfg
    )
    scores = {"n": 0}

    def score(record):
        scores["n"] += 1
        return 1.0 if scores["n"] % 2 == 1 else 0.0

    cls = AsyncJustGRPOController if async_mode else JustGRPOController
    config_cls = AsyncControllerConfig if async_mode else ControllerConfig
    return cls(
        config_cls(group_size=2, max_tokens_per_batch=128),
        rollout=rollout,
        reward_fn=score,
        trainer=trainer,
        justgrpo_config=cfg,
        store=store,
    )


def test_justgrpo_controller_end_to_end():
    result = _controller(False).step([Prompt(1, [1, 2])])
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])


def test_async_justgrpo_controller_end_to_end():
    result = asyncio.run(_controller(True).step([Prompt(1, [1, 2])]))
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])
