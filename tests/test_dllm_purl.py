import asyncio
import math

import tensorplay as tp

from megarl.async_controller import AsyncControllerConfig
from megarl.contracts import RolloutRequest, SamplingParams, TrajectoryRecord
from megarl.controller import ControllerConfig, Prompt
from megarl.dllm import from_dllm_record
from megarl.dllm.algorithms.purl import (
    AsyncPURLController,
    PURLConfig,
    PURLController,
    PURLGenerator,
    PURLModel,
    PURLRecord,
    PURLRollout,
    PURLStore,
    PURLTensorBatch,
    PURLTrainer,
    collate_purl,
    decode_purl,
    encode_purl,
    make_purl_packer,
    purl_loss,
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
        unmask_budget=1,
        edit_budget=1,
        seed=3,
    )
    base.update(kwargs)
    return PURLConfig(**base)


def _model(model=None, cfg=None):
    model = model or TinyDLLM()
    cfg = cfg or _config()
    return PURLModel(model, cfg)


def _generated(model=None, **kwargs):
    tp.manual_seed(19)
    model = model or TinyDLLM()
    cfg = _config(**kwargs)
    wrapped = _model(model, cfg)
    result = PURLGenerator(wrapped, cfg).generate([[1, 2], [1, 2]], config=cfg)
    return wrapped, cfg, result


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
    store = PURLStore()
    handles = [store.put(r) for r in result.records]
    samples = [
        from_dllm_record(w, r.prompt_ids, reward=1.0 if i == 0 else 0.0)
        for i, (w, r) in enumerate(zip(
            [_wire(r, h) for r, h in zip(result.records, handles)], result.records
        ))
    ]
    return samples, store


def _request(n=2):
    return RolloutRequest(
        1, 0, [1, 2], 4, n,
        SamplingParams(1.0, 1.0, 0, None, False, 0), None,
    )


def test_config_validation():
    cfg = _config()
    assert cfg.planner_dim == cfg.vocab_size
    assert cfg.noise_token_id == 7
    for bad in (dict(unmask_budget=0), dict(revision_gate=1.5), dict(vocab_size=0)):
        try:
            _config(**bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


def test_record_roundtrip_and_validation():
    _, _, result = _generated()
    record = result.records[0]
    assert decode_purl(encode_purl(record)) == record
    bad = PURLRecord(1, [1], [2, 3], [2])
    try:
        bad.validate()
        raise AssertionError("mismatched reference accepted")
    except ValueError:
        pass


def test_generation_completes_with_revision():
    wrapped, cfg, result = _generated()
    assert len(result.records) == 2
    for record in result.records:
        record.validate()
        assert len(record.completion_ids) == cfg.max_completion_length
        assert len(record.reference_ids) == cfg.max_completion_length


def test_collate_builds_transient_corruption():
    _, cfg, result = _generated()
    samples, store = _samples(result)
    records = [store.get(s.payload_handle) for s in samples]
    batch = collate_purl(samples, records, cfg)
    assert batch.size == 2
    assert batch.purl_sample_ids == [0, 1]
    assert batch.purl_prompt_span == 0
    assert batch.purl_canvas_len == cfg.max_completion_length
    for mask, noisy, correct in zip(batch.purl_mask, batch.purl_noisy, batch.purl_correct):
        assert all(a + b + c == 1 for a, b, c in zip(mask, noisy, correct))


def test_tensor_batch_exposes_adapter_fields():
    _, cfg, result = _generated()
    samples, store = _samples(result)
    batch = collate_purl(samples, [store.get(s.payload_handle) for s in samples], cfg)
    tensors = PURLTensorBatch(batch, device="cpu")
    assert tuple(tensors.dllm_noisy_ids.shape) == (2, 2 + cfg.max_completion_length)
    assert tensors.dllm_prompt_span == 0
    assert tuple(tensors.purl_reference_ids.shape) == (2, cfg.max_completion_length)


def test_joint_loss_backprops_into_backbone_and_planners():
    wrapped, cfg, result = _generated()
    samples, store = _samples(result)
    batch = collate_purl(samples, [store.get(s.payload_handle) for s in samples], cfg)
    tensors = PURLTensorBatch(batch, device="cpu")
    output = wrapped(tensors.dllm_noisy_ids, tensors, cfg)
    loss, diag = purl_loss(*output, tensors, unmask_budget=cfg.unmask_budget)
    assert math.isfinite(float(loss.flatten()[0]))
    assert set(["purl_edit_loss", "purl_unmask_loss", "purl_token_loss"]) <= set(diag)
    loss.backward()
    assert any(p.grad is not None and bool((p.grad != 0).any().flatten()[0]) for p in wrapped.backbone.parameters())
    assert any(p.grad is not None and bool((p.grad != 0).any().flatten()[0]) for p in wrapped.planners.parameters())


def test_retention_weight_changes_token_loss():
    _, cfg, result = _generated()
    samples, store = _samples(result)
    batch = collate_purl(samples, [store.get(s.payload_handle) for s in samples], cfg)
    tensors = PURLTensorBatch(batch, device="cpu")
    wrapped = _model(cfg=cfg)
    logits, edit, unmask = wrapped(tensors.dllm_noisy_ids, tensors, cfg)
    loss_a, _ = purl_loss(logits, edit, unmask, tensors, w_correct=0.0)
    loss_b, _ = purl_loss(logits, edit, unmask, tensors, w_correct=4.0)
    assert not math.isclose(float(loss_a.flatten()[0]), float(loss_b.flatten()[0]))


def test_trainer_backprops_and_updates_weights():
    wrapped, cfg, result = _generated()
    samples, store = _samples(result)
    batch = collate_purl(samples, [store.get(s.payload_handle) for s in samples], cfg)
    trainer = PURLTrainer(
        wrapped, TrainerConfig(device="cpu", learning_rate=1e-2), purl_config=cfg,
    )
    before = wrapped.planners.edit[-1].weight.detach().clone()
    audit = trainer.update(batch, [0.0, 0.0], dry_run=True)
    assert math.isfinite(float(audit["train/loss"]))
    assert any(p.grad is not None and bool((p.grad != 0).any().flatten()[0]) for p in wrapped.parameters())
    trainer.update(batch, [0.0, 0.0], dry_run=False)
    assert not bool(tp.allclose(before, wrapped.planners.edit[-1].weight.detach()))


def test_rollout_preserves_handles_and_packer_releases():
    tp.manual_seed(23)
    cfg = _config()
    wrapped = _model(cfg=cfg)
    store = PURLStore()
    rollout = PURLRollout(PURLGenerator(wrapped, cfg), store, cfg)
    records = list(rollout.generate(_request()))
    assert all(r.payload_handle is not None for r in records)
    samples = [
        from_dllm_record(r, [1, 2], reward=1.0 if i == 0 else 0.0)
        for i, r in enumerate(records)
    ]
    batches, orders = make_purl_packer(cfg, store)(samples)
    assert orders[0] == [0, 1]
    assert batches[0].size == 2


def _controller(async_mode):
    tp.manual_seed(29)
    cfg = _config()
    wrapped = _model(cfg=cfg)
    store = PURLStore()
    rollout = PURLRollout(PURLGenerator(wrapped, cfg), store, cfg)
    trainer = PURLTrainer(
        wrapped, TrainerConfig(device="cpu", learning_rate=1e-3), purl_config=cfg,
    )
    cls_type = AsyncPURLController if async_mode else PURLController
    config_cls = AsyncControllerConfig if async_mode else ControllerConfig
    state = {"n": 0}

    def reward(_):
        state["n"] += 1
        return 1.0 if state["n"] % 2 else 0.0

    return cls_type(
        config_cls(group_size=2, max_tokens_per_batch=256),
        rollout=rollout, reward_fn=reward, trainer=trainer,
        purl_config=cfg, store=store,
    )


def test_purl_controller_end_to_end():
    result = _controller(False).step([Prompt(1, [1, 2])])
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])


def test_async_purl_controller_end_to_end():
    result = asyncio.run(_controller(True).step([Prompt(1, [1, 2])]))
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])
