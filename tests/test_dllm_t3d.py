"""T3D (arXiv 2602.12262): DDO, path consistency, reference refresh, steps."""

import asyncio
import math

import tensorplay as tp

from megarl.async_controller import AsyncControllerConfig
from megarl.contracts import RolloutRequest, SamplingParams, TrajectoryRecord
from megarl.controller import ControllerConfig, Prompt
from megarl.dllm import default_adapter, from_dllm_record
from megarl.dllm.algorithms.t3d import (
    AsyncT3DController,
    T3DBatch,
    T3DConfig,
    T3DController,
    T3DGenerationResult,
    T3DGenerator,
    T3DRecord,
    T3DRollout,
    T3DStep,
    T3DStore,
    T3DTrainer,
    collate_t3d,
    decode_t3d,
    encode_t3d,
    make_t3d_packer,
    t3d_loss,
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
        teacher_steps=4,
        student_steps=2,
        max_train_steps=3,
        max_masked_positions=3,
        seed=4,
    )
    defaults.update(kwargs)
    return T3DConfig(**defaults)


def _generated(model=None, **kwargs):
    tp.manual_seed(19)
    model = model or TinyDLLM()
    cfg = _config(**kwargs)
    result = T3DGenerator(model, default_adapter(cfg)).generate(
        [[1, 2], [1, 2]], config=cfg
    )
    return model, cfg, result


def _wire(record, handle=None):
    return TrajectoryRecord(
        request_id=1, prompt_len=2, token_ids=list(record.completion_ids),
        logprobs=[0.0] * record.length, top_logprobs=[],
        policy_versions=[0] * record.length, reward=None, finish_reason=0,
        payload_handle=handle,
    )


def test_config_validation_and_path_weights():
    for bad in (
        {"family": "nope"},
        {"max_completion_length": 0},
        {"teacher_steps": 0},
        {"path_weight": -1.0},
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
    cfg = _config(student_steps=4)
    assert cfg.path_weights([0, 1, 3]) == [1.0, 0.75, 0.25]


def test_record_roundtrip():
    record = T3DRecord(
        1, [1, 2], [3, 4, 5, 6],
        (T3DStep([7, 7, 7, 7], [1, 1, 1, 1]), T3DStep([3, 4, 5, 6], [0, 0, 0, 0])),
        [0, 0, 1, 1], [3, 4, 5, 6], reward=1.0, version=2,
    )
    assert decode_t3d(encode_t3d(record)) == record


def test_generation_records_intermediate_states_and_steps():
    _, cfg, result = _generated()
    assert isinstance(result, T3DGenerationResult) and len(result.records) == 2
    record = result.records[0]
    assert record.length == cfg.max_completion_length
    assert sorted(record.decode_steps) == list(range(cfg.max_completion_length))
    masks = [sum(step.mask) for step in record.steps]
    assert masks[0] == cfg.max_completion_length and masks[-1] == 0
    assert all(a >= b for a, b in zip(masks, masks[1:]))


def test_collate_aligns_rows_with_samples():
    _, cfg, result = _generated()
    store = T3DStore()
    samples = []
    for r in result.records:
        handle = store.put(r)
        samples.append(from_dllm_record(_wire(r, handle), [1, 2], reward=1.0))
    batches, orders = make_t3d_packer(cfg, store)(samples)
    batch = batches[0]
    assert isinstance(batch, T3DBatch)
    assert batch.size == len(orders[0])
    assert max(orders[0]) == 1
    assert all(len(row) == cfg.max_masked_positions for row in batch.t3d_mask_positions)
    assert all(sum(row) >= 1 for row in batch.t3d_mask_valid)
    assert all(sum(row) >= 1 for row in batch.t3d_path_valid)


def test_ddo_rewards_teacher_over_reference_samples():
    # [B=1, L=2, V=3]: masked position 1, teacher token 2, fake token 0.
    positions = tp.tensor([[1]], dtype=tp.long)
    valid = tp.ones((1, 1), dtype=tp.float32)
    teacher_targets = tp.tensor([[0, 2, 0]], dtype=tp.long)
    fake_targets = tp.tensor([[0, 0, 0]], dtype=tp.long)
    path_positions = tp.tensor([[0]], dtype=tp.long)
    path_valid = tp.ones((1, 1), dtype=tp.float32)
    path_weights = tp.ones((1, 1), dtype=tp.float32)
    reference = tp.full((1, 2, 3), 0.0, dtype=tp.float32)
    boosted = reference.clone()
    boosted[:, :, 2] = 5.0  # raise the teacher token only
    kwargs = dict(
        mask_positions=positions, mask_valid=valid,
        path_positions=path_positions, path_valid=path_valid,
        path_weights=path_weights, path_weight=0.0,
    )
    baseline, _ = t3d_loss(reference, reference, teacher_targets, fake_targets, **kwargs)
    improved, diag = t3d_loss(boosted, reference, teacher_targets, fake_targets, **kwargs)
    assert float(improved) < float(baseline)
    assert float(diag["t3d_positive_ratio"]) > 0


def test_trainer_backprops_and_refresh():
    model, cfg, result = _generated(refresh_every=2)
    ref = TinyDLLM()
    ref.load_state_dict(model.state_dict())
    record = result.records[0]
    sample = from_dllm_record(_wire(record), [1, 2], reward=1.0)
    batch = collate_t3d([sample], [record], cfg)
    trainer = T3DTrainer(
        model, TrainerConfig(device="cpu", learning_rate=1e-2),
        t3d_config=cfg, ref_model=ref,
    )
    metrics = trainer.update(batch, dry_run=True)
    assert math.isfinite(float(metrics["train/loss"]))
    assert any(
        p.grad is not None and float(p.grad.abs().sum().flatten()[0]) > 0
        for p in model.parameters()
    )
    before = sum(float(p.detach().sum().flatten()[0]) for p in model.parameters())
    trainer.update(batch, dry_run=False)
    after = sum(float(p.detach().sum().flatten()[0]) for p in model.parameters())
    assert before != after
    assert float(trainer.update(batch, dry_run=False)["train/ref_syncs"]) == 1.0


def test_rollout_uses_reference_generator():
    tp.manual_seed(21)
    model, ref = TinyDLLM(), TinyDLLM()
    cfg = _config()
    store = T3DStore()
    rollout = T3DRollout(
        T3DGenerator(model, default_adapter(cfg)), store, cfg, ref_model=ref
    )
    request = RolloutRequest(
        9, 0, [1, 2], cfg.max_completion_length, 2,
        SamplingParams(1.0, 1.0, 0, None, False, 0), None,
    )
    records = list(rollout.generate(request))
    assert len(records) == 2 and all(r.payload_handle is not None for r in records)
    stored = store.get(records[0].payload_handle)
    assert isinstance(stored, T3DRecord)
    assert len(stored.ref_completion_ids) == cfg.max_completion_length


def _controller(async_mode: bool):
    tp.manual_seed(10)
    model = TinyDLLM()
    ref = TinyDLLM()
    cfg = _config()
    store = T3DStore()
    rollout = T3DRollout(
        T3DGenerator(model, default_adapter(cfg)), store, cfg, ref_model=ref
    )
    trainer = T3DTrainer(
        model, TrainerConfig(learning_rate=1e-3), t3d_config=cfg, ref_model=ref
    )
    cls = AsyncT3DController if async_mode else T3DController
    config_cls = AsyncControllerConfig if async_mode else ControllerConfig
    scores = {"n": 0}

    def score(_):
        scores["n"] += 1
        return 1.0 if scores["n"] % 2 else 0.0

    return cls(
        config_cls(group_size=2, max_tokens_per_batch=256),
        rollout=rollout, reward_fn=score, trainer=trainer,
        t3d_config=cfg, store=store,
    )


def test_t3d_controller_end_to_end():
    result = _controller(False).step([Prompt(1, [1, 2])])
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])


def test_async_t3d_controller_end_to_end():
    result = asyncio.run(_controller(True).step([Prompt(1, [1, 2])]))
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])
