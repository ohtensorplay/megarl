import asyncio
import math

import tensorplay as tp

from megarl.async_controller import AsyncControllerConfig
from megarl.contracts import RolloutRequest, SamplingParams, TrajectoryRecord
from megarl.controller import ControllerConfig, Prompt
from megarl.dllm import from_dllm_record
from megarl.dllm.algorithms.remarl import (
    AsyncReMaRLController,
    ReMaRLConfig,
    ReMaRLController,
    ReMaRLGenerator,
    ReMaRLPolicy,
    ReMaRLRecord,
    ReMaRLRollout,
    ReMaRLStore,
    ReMaRLTensorBatch,
    ReMaRLTrainer,
    collate_remarl,
    decode_remarl,
    encode_remarl,
    gumbel_topk,
    make_remarl_packer,
    plackett_luce_logprob,
    remarl_loss,
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
    base = dict(
        mask_token_id=7,
        vocab_size=8,
        max_completion_length=4,
        budget=4,
        hidden_dim=8,
        pos_dim=4,
        proj_dim=8,
        mlp_hidden=8,
        seed=3,
    )
    base.update(kwargs)
    return ReMaRLConfig(**base)


def _generated(model=None, **kwargs):
    tp.manual_seed(17)
    model = model or TinyDLLM()
    cfg = _config(**kwargs)
    policy = ReMaRLPolicy(cfg, max_len=16)
    result = ReMaRLGenerator(model, policy).generate([[1, 2], [1, 2]], config=cfg)
    return model, cfg, policy, result


def _wire(record, handle=None):
    length = len(record.completion_ids)
    return TrajectoryRecord(
        request_id=1, prompt_len=len(record.prompt_ids),
        token_ids=list(record.completion_ids),
        logprobs=[0.0] * length, top_logprobs=[],
        policy_versions=[0] * length, reward=None, finish_reason=0,
        payload_handle=handle,
    )


def _samples(result):
    store = ReMaRLStore()
    handles = [store.put(r) for r in result.records]
    return [
        from_dllm_record(w, r.prompt_ids, reward=1.0 if i == 0 else 0.0)
        for i, (w, r) in enumerate(zip(
            [_wire(r, h) for r, h in zip(result.records, handles)], result.records
        ))
    ], store


def _request(n=2):
    return RolloutRequest(
        1, 0, [1, 2], 4, n,
        SamplingParams(1.0, 1.0, 0, None, False, 0), None,
    )


def test_config_validation_and_schedule():
    cfg = _config()
    assert cfg.keep_count(0) == 1
    assert cfg.keep_count(1) == 2
    assert cfg.keep_count(3) == 4
    assert cfg.train_steps == 3
    assert cfg.noise_token_id == 7
    for bad in (dict(tau=0.0), dict(budget=0), dict(clip_eps=1.5)):
        try:
            _config(**bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


def test_record_roundtrip_and_validation():
    _, _, _, result = _generated()
    record = result.records[0]
    assert decode_remarl(encode_remarl(record)) == record
    bad = ReMaRLRecord(1, [1], [2, 3], steps=record.steps)
    try:
        bad.validate()
        raise AssertionError("short prompt accepted")
    except ValueError:
        pass


def test_generation_uses_gumbel_topk_and_remasks():
    _, cfg, _, result = _generated()
    for record in result.records:
        record.validate()
        assert len(record.steps) == cfg.train_steps
        for i, step in enumerate(record.steps):
            assert len(step.order) == cfg.keep_count(i)
            assert len(set(step.order)) == len(step.order)
            assert math.isfinite(step.old_logprob)
            assert step.old_logprob < 0.0


def test_zero_init_starts_from_p2_scores():
    cfg = _config()
    policy = ReMaRLPolicy(cfg, max_len=16)
    scores = tp.randn((2, 4))
    mask = tp.ones((2, 4))
    hidden = tp.zeros((2, 4, cfg.hidden_dim))
    out = policy(scores, mask, hidden)
    assert tp.allclose(out, scores)


def test_plackett_luce_is_normalised_for_full_order():
    scores = tp.tensor([[1.0, 2.0, 3.0]])
    order = tp.tensor([[2, 1, 0]])
    logp = plackett_luce_logprob(scores, order)
    # P(2 first) * P(1 second | 2 removed) * P(0 last)
    import math
    expected = math.log(math.exp(3) / sum(math.exp(v) for v in [1, 2, 3]))
    expected += math.log(math.exp(2) / (math.exp(1) + math.exp(2)))
    assert abs(float(logp.flatten()[0]) - expected) < 1e-5


def test_gumbel_topk_returns_distinct_indices():
    scores = tp.randn((3, 6))
    order = gumbel_topk(scores, 4, tau=1.0)
    assert tuple(order.shape) == (3, 4)
    for row in order.tolist():
        assert len(set(row)) == 4


def test_collate_aligns_rows_and_padded_actions():
    _, cfg, _, result = _generated()
    samples, store = _samples(result)
    records = [store.get(s.payload_handle) for s in samples]
    batch = collate_remarl(samples, records, cfg)
    assert batch.size == 2 * cfg.train_steps
    assert batch.remarl_sample_ids == [0, 0, 0, 1, 1, 1]
    max_k = max(cfg.keep_count(i) for i in range(cfg.train_steps))
    assert all(len(row) == max_k for row in batch.remarl_actions)
    assert all(
        sum(valid) == cfg.keep_count(i % cfg.train_steps)
        for i, valid in enumerate(batch.remarl_action_valid)
    )


def test_loss_backprops_and_clips():
    _, cfg, policy, result = _generated()
    samples, store = _samples(result)
    batch = collate_remarl(samples, [store.get(s.payload_handle) for s in samples], cfg)
    tensors = ReMaRLTensorBatch(batch, device="cpu")
    scores = policy(tensors.remarl_scores, tensors.remarl_mask, tensors.remarl_hidden)
    loss, diag = remarl_loss(
        scores, tensors, tp.ones((batch.size,)),
        old_logprobs=tensors.remarl_old_logprobs,
        tau=cfg.tau, clip_eps=cfg.clip_eps,
    )
    assert math.isfinite(float(loss.flatten()[0]))
    assert set(["remarl_mean_ratio", "remarl_clip_fraction"]) <= set(diag)
    loss.backward()
    assert any(p.grad is not None and bool((p.grad != 0).any().flatten()[0]) for p in policy.parameters())


def test_trainer_backprops_and_updates_weights():
    _, cfg, policy, result = _generated()
    samples, store = _samples(result)
    batch = collate_remarl(samples, [store.get(s.payload_handle) for s in samples], cfg)
    trainer = ReMaRLTrainer(
        policy, TrainerConfig(device="cpu", learning_rate=1e-2), remarl_config=cfg,
    )
    before = float(policy.skip.weight.detach().abs().sum().flatten()[0])
    audit = trainer.update(batch, [1.0] * batch.size, dry_run=True)
    assert math.isfinite(float(audit["train/loss"]))
    assert any(p.grad is not None and bool((p.grad != 0).any().flatten()[0]) for p in policy.parameters())
    trainer.update(batch, [1.0] * batch.size, dry_run=False)
    assert before != float(policy.skip.weight.detach().abs().sum().flatten()[0])


def test_rollout_preserves_handles_and_packer_releases():
    tp.manual_seed(9)
    cfg = _config()
    policy = ReMaRLPolicy(cfg, max_len=16)
    store = ReMaRLStore()
    rollout = ReMaRLRollout(ReMaRLGenerator(TinyDLLM(), policy), store, cfg)
    records = list(rollout.generate(_request()))
    assert all(r.payload_handle is not None for r in records)
    samples = [
        from_dllm_record(r, [1, 2], reward=1.0 if i == 0 else 0.0)
        for i, r in enumerate(records)
    ]
    batches, orders = make_remarl_packer(cfg, store)(samples)
    assert orders[0] == [0, 0, 0, 1, 1, 1]
    assert batches[0].size == 6


def _controller(async_mode):
    tp.manual_seed(11)
    cfg = _config()
    policy = ReMaRLPolicy(cfg, max_len=16)
    store = ReMaRLStore()
    rollout = ReMaRLRollout(ReMaRLGenerator(TinyDLLM(), policy), store, cfg)
    trainer = ReMaRLTrainer(
        policy, TrainerConfig(device="cpu", learning_rate=1e-3), remarl_config=cfg,
    )
    cls_type = AsyncReMaRLController if async_mode else ReMaRLController
    config_cls = ControllerConfig if not async_mode else AsyncControllerConfig
    state = {"n": 0}

    def reward(_):
        state["n"] += 1
        return 1.0 if state["n"] % 2 else 0.0

    return cls_type(
        config_cls(group_size=2, max_tokens_per_batch=256),
        rollout=rollout, reward_fn=reward, trainer=trainer,
        remarl_config=cfg, store=store,
    )


def test_remarl_controller_end_to_end():
    result = _controller(False).step([Prompt(1, [1, 2])])
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])


def test_async_remarl_controller_end_to_end():
    result = asyncio.run(_controller(True).step([Prompt(1, [1, 2])]))
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])
