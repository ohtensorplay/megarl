"""dOPSD (arXiv 2607.04428): trajectory privilege, JSD, verification, steps."""

import asyncio
import dataclasses
import math

import tensorplay as tp

from megarl.async_controller import AsyncControllerConfig
from megarl.contracts import RolloutRequest, SamplingParams, TrajectoryRecord
from megarl.controller import ControllerConfig, Prompt
from megarl.dllm import default_adapter, from_dllm_record
from megarl.dllm.algorithms.dopsd import (
    AsyncDOPSDController,
    DOPSDBatch,
    DOPSDConfig,
    DOPSDController,
    DOPSDGenerationResult,
    DOPSDGenerator,
    DOPSDRecord,
    DOPSDRollout,
    DOPSDStep,
    DOPSDStore,
    DOPSDTensorBatch,
    DOPSDTrainer,
    collate_dopsd,
    decode_dopsd,
    dopsd_loss,
    encode_dopsd,
    generalized_jsd,
    make_dopsd_packer,
)
from megarl.trainer import TrainerConfig


class TinyDLLM(tp.nn.Module):
    """Context-sensitive toy: logits depend on the neighbouring token too.

    A token-independent model would make the teacher distribution at a masked
    position identical to the student's, giving a degenerate zero-JSD test.
    """

    def __init__(self, vocab: int = 8, hidden: int = 8) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(vocab, hidden)
        self.mix = tp.nn.Linear(hidden, hidden)
        self.head = tp.nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        embeddings = self.emb(input_ids)
        shifted = tp.cat([embeddings[:, -1:], embeddings[:, :-1]], dim=1)
        return self.head(self.mix(embeddings + 0.5 * shifted))


def _config(**kwargs):
    defaults = dict(
        mask_token_id=7,
        vocab_size=8,
        max_completion_length=4,
        sampling_steps=4,
        max_scored_positions=3,
        max_teacher_steps=2,
        seed=2,
    )
    defaults.update(kwargs)
    return DOPSDConfig(**defaults)


def _generated(model=None, **kwargs):
    tp.manual_seed(17)
    model = model or TinyDLLM()
    cfg = _config(**kwargs)
    result = DOPSDGenerator(model, default_adapter(cfg)).generate(
        [[1, 2], [1, 2]], config=cfg
    )
    return model, cfg, result


def test_config_validation():
    for bad in (
        {"family": "nope"},
        {"max_completion_length": 0},
        {"mask_threshold": 1.0},
        {"js_weight": 1.5},
        {"max_scored_positions": 0},
        {"version": -1},
        {"family": "block", "vocab_size": None},
    ):
        try:
            _config(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid config: {bad}")
    assert _config().noise_token_id == 7


def test_record_roundtrip_and_validation():
    record = DOPSDRecord(
        1, [1, 2], [3, 4, 5, 6],
        (DOPSDStep([7, 7, 7, 7], [1, 1, 1, 1]), DOPSDStep([3, 4, 5, 6], [0, 0, 0, 0])),
        reward=1.0, version=2,
    )
    assert decode_dopsd(encode_dopsd(record)) == record
    try:
        dataclasses.replace(record, steps=(DOPSDStep([1], [1]),)).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a misaligned trajectory step")


def test_generation_records_genuine_states():
    _, cfg, result = _generated()
    assert isinstance(result, DOPSDGenerationResult)
    assert len(result.records) == 2
    record = result.records[0]
    masks = [sum(step.mask) for step in record.steps]
    assert masks[0] == cfg.max_completion_length
    assert masks[-1] == 0
    assert all(a >= b for a, b in zip(masks, masks[1:])), masks
    assert all(len(step.canvas_ids) == cfg.max_completion_length for step in record.steps)


def test_collate_teacher_steps_and_verification():
    _, cfg, result = _generated()
    store = DOPSDStore()
    records = [dataclasses.replace(r, reward=1.0 if i == 0 else 0.0) for i, r in enumerate(result.records)]
    samples = []
    for i, r in enumerate(records):
        wire = TrajectoryRecord(
            request_id=1, prompt_len=2, token_ids=list(r.completion_ids),
            logprobs=[0.0] * r.length, top_logprobs=[],
            policy_versions=[0] * r.length, reward=None, finish_reason=0,
            payload_handle=store.put(r),
        )
        samples.append(from_dllm_record(wire, [1, 2], reward=1.0 if i == 0 else 0.0))
    batches, orders = make_dopsd_packer(_config(verify_rollouts=True, correct_reward_threshold=0.5), store)(samples)
    batch = batches[0]
    assert isinstance(batch, DOPSDBatch) and batch.size == 2
    assert batch.dopsd_keep == [1.0, 0.0]
    assert orders == [[0, 1]]
    assert all(sum(row) == cfg.max_teacher_steps for row in batch.dopsd_teacher_present)
    assert all(all(c > 0 for c in row) for row in batch.dopsd_counts)
    assert len(batch.dopsd_mask_positions[0]) == cfg.max_scored_positions


def test_teacher_probs_are_normalised_and_detached():
    model, cfg, result = _generated()
    record = result.records[0]
    wire = TrajectoryRecord(
        request_id=1, prompt_len=2, token_ids=list(record.completion_ids),
        logprobs=[0.0] * record.length, top_logprobs=[],
        policy_versions=[0] * record.length, reward=None, finish_reason=0,
        payload_handle=None,
    )
    sample = from_dllm_record(wire, [1, 2], reward=1.0)
    batch = collate_dopsd([sample], [record], cfg)
    trainer = DOPSDTrainer(model, TrainerConfig(device="cpu"), dopsd_config=cfg)
    tensors = DOPSDTensorBatch(batch, device="cpu")
    probs = trainer._teacher_probs(tensors)
    assert not probs.requires_grad
    assert tuple(probs.shape) == (1, cfg.max_scored_positions, cfg.vocab_size)
    assert tp.allclose(probs.sum(dim=-1), tp.ones((1, cfg.max_scored_positions)))


def test_generalized_jsd_vanishes_for_matching_distributions():
    probs = tp.tensor([[0.1, 0.2, 0.7]], dtype=tp.float32)
    for weight in (0.0, 0.25, 0.5, 1.0):
        value = float(generalized_jsd(probs, probs.clone().requires_grad_(True), weight))
        assert value < 1e-6
    teacher = tp.tensor([[0.8, 0.1, 0.1]], dtype=tp.float32)
    student = tp.tensor([[0.1, 0.1, 0.8]], dtype=tp.float32, requires_grad=True)
    value = generalized_jsd(teacher, student, 0.5)
    assert float(value) > 0.1
    value.backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0


def test_dopsd_loss_respects_verification():
    student = tp.zeros((2, 3, 8), dtype=tp.float32, requires_grad=True)
    teacher = tp.full((2, 3, 8), 1.0 / 8, dtype=tp.float32)
    valid = tp.ones((2, 3), dtype=tp.float32)
    loss, diag = dopsd_loss(teacher, student, valid, tp.tensor([1.0, 0.0]))
    assert math.isfinite(float(loss)) and float(diag["dopsd_keep_fraction"]) == 0.5


def test_trainer_backprops_and_updates_weights():
    model, cfg, result = _generated()
    record = result.records[0]
    wire = TrajectoryRecord(
        request_id=1, prompt_len=2, token_ids=list(record.completion_ids),
        logprobs=[0.0] * record.length, top_logprobs=[],
        policy_versions=[0] * record.length, reward=None, finish_reason=0,
        payload_handle=None,
    )
    sample = from_dllm_record(wire, [1, 2], reward=1.0)
    batch = collate_dopsd([sample], [record], cfg)
    trainer = DOPSDTrainer(
        model, TrainerConfig(device="cpu", learning_rate=1e-2), dopsd_config=cfg
    )
    metrics = trainer.update(batch, [0.0], dry_run=True)
    assert math.isfinite(float(metrics["train/loss"]))
    assert any(
        p.grad is not None and float(p.grad.abs().sum().flatten()[0]) > 0
        for p in model.parameters()
    )
    before = sum(float(p.detach().sum().flatten()[0]) for p in model.parameters())
    trainer.update(batch, [0.0], dry_run=False)
    after = sum(float(p.detach().sum().flatten()[0]) for p in model.parameters())
    assert before != after


def test_rollout_preserves_handles():
    tp.manual_seed(8)
    model = TinyDLLM()
    cfg = _config()
    store = DOPSDStore()
    rollout = DOPSDRollout(DOPSDGenerator(model, default_adapter(cfg)), store, cfg)
    request = RolloutRequest(
        9, 0, [1, 2], cfg.max_completion_length, 2,
        SamplingParams(1.0, 1.0, 0, None, False, 0), None,
    )
    records = list(rollout.generate(request))
    assert len(records) == 2 and all(r.payload_handle is not None for r in records)
    assert all(store.get(r.payload_handle).length == cfg.max_completion_length for r in records)


def _controller(async_mode: bool):
    tp.manual_seed(6)
    model = TinyDLLM()
    cfg = _config()
    store = DOPSDStore()
    rollout = DOPSDRollout(DOPSDGenerator(model, default_adapter(cfg)), store, cfg)
    trainer = DOPSDTrainer(model, TrainerConfig(learning_rate=1e-3), dopsd_config=cfg)
    cls = AsyncDOPSDController if async_mode else DOPSDController
    config_cls = AsyncControllerConfig if async_mode else ControllerConfig
    scores = {"n": 0}

    def score(_):
        scores["n"] += 1
        return 1.0 if scores["n"] % 2 else 0.0

    return cls(
        config_cls(group_size=2, max_tokens_per_batch=128),
        rollout=rollout, reward_fn=score, trainer=trainer,
        dopsd_config=cfg, store=store,
    )


def test_dopsd_controller_end_to_end():
    result = _controller(False).step([Prompt(1, [1, 2])])
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])


def test_async_dopsd_controller_end_to_end():
    result = asyncio.run(_controller(True).step([Prompt(1, [1, 2])]))
    assert result is not None and result.groups == 1
    assert math.isfinite(result.trainer_metrics["train/loss"])
