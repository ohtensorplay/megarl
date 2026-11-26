"""dLLM RL: deterministic canvas estimation, adapters, losses and real steps."""

import math

import tensorplay as tp

from megarl.contracts import TrajectoryRecord
from megarl.dllm import (
    BlockDiffusionAdapter,
    DiffuGRPOConfig,
    DLLMBatch,
    DLLMConfig,
    DLLMSample,
    DLLMTensorBatch,
    DLLMTrainer,
    FullAttentionAdapter,
    collate_dllm,
    dllm_logprobs,
    from_dllm_record,
)
from megarl.dllm.rollout import DLLMRollout, DLLMRolloutConfig
from megarl.losses import TensorBatch, diffu_grpo_loss, policy_loss
from megarl.trainer import TrainerConfig


class TinyDLLM(tp.nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 8) -> None:
        super().__init__()
        self.emb = tp.nn.Embedding(vocab, hidden)
        self.head = tp.nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))


class TinyBlockDLLM(TinyDLLM):
    """Minimal encode/denoise surface matching MegaLM's diffusion model."""

    def allocate_cache(self, capacity: int):
        return {"capacity": capacity, "length": 0}

    def encode(self, ids, cache) -> None:
        cache["length"] += int(ids.shape[0])

    def denoise(self, canvas, caches, *, self_conditioning_logits=None):
        return self.head(self.emb(canvas))


def make_batch(n: int = 2, family: str = "masked"):
    cfg = DLLMConfig(family=family, mask_token_id=7, vocab_size=8)
    samples = [
        DLLMSample([1, 2], [3, 4] if i % 2 == 0 else [5, 6], float(i == 0))
        for i in range(n)
    ]
    return collate_dllm(samples, cfg), cfg


def test_config_rejects_bad_knobs():
    for kwargs in (
        {"family": "unknown"},
        {"prompt_mask_prob": 1.5},
        {"num_iterations": 0},
        {"parameterization": "nope"},
        {"objective": "nope"},
    ):
        try:
            DLLMConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid config: {kwargs}")


def test_masking_is_deterministic_and_d1_style():
    batch, cfg = make_batch()
    again, _ = make_batch()
    assert batch.dllm_noisy_ids == again.dllm_noisy_ids
    # every completion position is masked; prompt positions are masked with
    # the seeded probability (sample 0 keeps token 1, masks token 2).
    assert batch.dllm_noisy_ids[0][2:] == [7, 7]
    assert batch.loss_mask[0] == [0, 0, 1, 1]
    assert batch.versions[0] == [0, 0, 0, 0]


def test_block_layout_has_fixed_prompt_span_and_canvas():
    cfg = DLLMConfig(
        family="block", mask_token_id=7, canvas_token_id=7, vocab_size=8
    )
    samples = [
        DLLMSample([1, 2], [3], 1.0),
        DLLMSample([1], [4, 5, 6], 0.0),
    ]
    batch = collate_dllm(samples, cfg)
    assert batch.dllm_prompt_span == 2
    assert batch.dllm_canvas_len == 3
    assert batch.input_ids[0] == [1, 2, 3, 7, 7]
    assert all(0 <= token < 8 for token in batch.dllm_noisy_ids[1])
    assert sum(batch.loss_mask[0]) == 1
    assert sum(batch.loss_mask[1]) == 3


def test_from_dllm_record_uses_response_span():
    record = TrajectoryRecord(1, 2, [3, 4], [0.0, 0.0], [], [9, 9], 1.0, 0, None)
    sample = from_dllm_record(record, [1, 2])
    assert sample.prompt_token_ids == [1, 2]
    assert sample.completion_token_ids == [3, 4]
    assert sample.version == 9


def test_full_attention_logprobs_match_manual_estimator():
    tp.manual_seed(7)
    batch, cfg = make_batch()
    tensors = DLLMTensorBatch(batch)
    model = TinyDLLM()
    logps = dllm_logprobs(model, tensors, cfg, adapter=FullAttentionAdapter())
    logits = model(tensors.dllm_noisy_ids)
    expected = tp.gather(
        tp.log_softmax(logits.float(), -1), -1, tensors.input_ids.unsqueeze(-1)
    ).squeeze(-1)
    worst = max(
        abs(float(a) - float(b))
        for a, b in zip(logps.flatten().tolist(), expected.flatten().tolist())
    )
    assert worst < 1e-5
    assert tuple(logps.shape) == (2, 4)


def test_block_adapter_places_canvas_logits_after_the_prompt_span():
    tp.manual_seed(7)
    batch, cfg = make_batch(family="block")
    tensors = DLLMTensorBatch(batch)
    model = TinyBlockDLLM()
    adapter = BlockDiffusionAdapter()
    logits = adapter(model, tensors.dllm_noisy_ids, tensors, cfg)
    assert tuple(logits.shape) == (2, 4, 8)
    canvas_logits = model.denoise(
        tensors.dllm_noisy_ids[:, 2:], [{"length": 2}, {"length": 2}]
    )
    worst = max(
        abs(float(a) - float(b))
        for a, b in zip(
            logits[:, 2:, :].flatten().tolist(), canvas_logits.flatten().tolist()
        )
    )
    assert worst < 1e-5


def test_diffu_grpo_uses_override_old_logprobs_and_backprops():
    batch, _ = make_batch()
    tensors = TensorBatch(batch)
    logits = tp.zeros((2, 4, 8), dtype=tp.float32, requires_grad=True)
    old = tp.full((2, 4), -2.0, dtype=tp.float32)
    adv = tp.tensor([1.0, -1.0], dtype=tp.float32)
    loss, diag = policy_loss(
        logits, tensors, adv, loss_type="diffu_grpo", old_logprobs=old
    )
    assert math.isfinite(float(loss))
    # uniform logits give log(1/8), so exp(-2.079 - (-2.0)) < 1
    assert 0.0 < float(diag["mean_ratio"]) < 1.0
    loss.backward()
    assert logits.grad is not None


def test_diffu_grpo_named_recipe_and_wrapper():
    recipe = DiffuGRPOConfig(mask_token_id=7, vocab_size=8)
    assert recipe.estimator == "one_step" and recipe.objective == "diffu_grpo"
    batch, _ = make_batch()
    tensors = TensorBatch(batch)
    logits = tp.zeros((2, 4, 8), dtype=tp.float32, requires_grad=True)
    old = tp.full((2, 4), -2.0, dtype=tp.float32)
    loss, diag = diffu_grpo_loss(
        logits, tensors, tp.tensor([1.0, -1.0]), old_logprobs=old
    )
    assert math.isfinite(float(loss))
    assert 0.0 < float(diag["mean_ratio"]) < 1.0
    loss.backward()
    assert logits.grad is not None


def test_wd1_weights_are_group_normalised_and_directional():
    batch, _ = make_batch()
    tensors = TensorBatch(batch)
    logits = tp.zeros((2, 4, 8), dtype=tp.float32, requires_grad=True)
    adv = tp.tensor([2.0, -2.0], dtype=tp.float32)
    loss, diag = policy_loss(
        logits, tensors, adv, loss_type="wd1", wd1_temperature=1.0
    )
    assert math.isfinite(float(loss))
    loss.backward()
    assert logits.grad is not None
    # w+ > w- for the positive advantage row, so its logprobs are pushed up.
    import tensorplay.nn.functional as F

    probs = F.softmax(logits, dim=-1)
    grad_pos = float(logits.grad[0, 2, 3].flatten()[0])
    grad_neg = float(logits.grad[1, 2, 5].flatten()[0])
    assert grad_pos < 0 < grad_neg, (grad_pos, grad_neg)
    assert float(probs.sum().flatten()[0]) > 0


def test_trainer_runs_a_real_diffusion_step():
    tp.manual_seed(1234)
    cfg = TrainerConfig(loss_type="diffu_grpo", epsilon_low=0.2)
    dllm_cfg = DLLMConfig(mask_token_id=7)
    trainer = DLLMTrainer(TinyDLLM(), cfg, dllm_config=dllm_cfg)
    batch, _ = make_batch()
    before = trainer.policy.head.weight.detach().clone()
    metrics = trainer.update(batch, [1.0, -1.0], dry_run=True)
    assert math.isfinite(metrics["train/loss"])
    grad = trainer.policy.head.weight.grad
    assert grad is not None and any(abs(float(v)) > 0 for v in grad.flatten().tolist())
    trainer.update(batch, [1.0, -1.0])
    after = trainer.policy.head.weight.detach()
    assert float((after - before).abs().sum()) > 0


def test_trainer_requires_behavior_model_for_multiple_iterations():
    cfg = TrainerConfig(loss_type="diffu_grpo")
    dllm_cfg = DLLMConfig(mask_token_id=7, num_iterations=2)
    try:
        DLLMTrainer(TinyDLLM(), cfg, dllm_config=dllm_cfg)
    except ValueError as exc:
        assert "behavior_model" in str(exc)
    else:
        raise AssertionError("multi-iteration dLLM trainer accepted no behavior model")


class FakeGenerator:
    def generate(self, prompts, *, config=None):
        class Result:
            sequences = [list(p) + [3, 4, 1] for p in prompts]

        return Result()


def test_rollout_adapter_emits_wire_compatible_records():
    from megarl.contracts import RolloutRequest, SamplingParams

    request = RolloutRequest(
        request_id=1,
        policy_version=5,
        prompt_token_ids=[1, 2],
        max_new_tokens=16,
        n=2,
        sampling=SamplingParams(1.0, 1.0, 0, None, True, 0),
        session_id=None,
    )
    service = DLLMRollout(
        FakeGenerator(), DLLMRolloutConfig(), version_provider=lambda: 5
    )
    records = list(service.generate(request))
    assert len(records) == 2
    for record in records:
        assert record.token_ids == [3, 4, 1]
        assert record.policy_versions == [5, 5, 5]
        assert len(record.logprobs) == len(record.token_ids)
        assert record.reward is None


def test_controller_runs_a_dllm_group_end_to_end():
    from megarl.controller import ControllerConfig, Prompt
    from megarl.dllm import DLLMController

    class VariantGenerator:
        def generate(self, prompts, *, config=None):
            class Result:
                sequences = [
                    list(prompts[0]) + [3, 4, 1],
                    list(prompts[0]) + [3, 4, 5, 6, 1],
                ]

            return Result()

    tp.manual_seed(11)
    cfg = TrainerConfig(loss_type="diffu_grpo")
    dllm_cfg = DLLMConfig(mask_token_id=7)
    trainer = DLLMTrainer(TinyDLLM(), cfg, dllm_config=dllm_cfg)
    controller = DLLMController(
        ControllerConfig(group_size=2, max_tokens_per_batch=128),
        rollout=DLLMRollout(VariantGenerator(), version_provider=lambda: 0),
        reward_fn=lambda record: 1.0 if len(record.token_ids) > 3 else 0.0,
        trainer=trainer,
        dllm_config=dllm_cfg,
    )
    result = controller.step([Prompt(1, [1, 2])])
    assert result is not None
    assert result.groups == 1
    assert result.token_count > 0
    assert math.isfinite(result.trainer_metrics["train/loss"])
