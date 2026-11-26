"""Reward, zero-variance filtering and token-budget packing."""

from megarl.batching import collate, from_trajectory, pack
from megarl.contracts import TrajectoryRecord
from megarl.reward import (
    MathRuleReward,
    extract_last_number,
    has_signal,
    score_group,
    zero_variance_fraction,
)


def record(rid: int, tokens, logps=None, reward=None, versions=None):
    logps = logps if logps is not None else [-0.1] * len(tokens)
    versions = versions if versions is not None else [0] * len(tokens)
    return TrajectoryRecord(
        request_id=rid,
        prompt_len=2,
        token_ids=list(tokens),
        logprobs=logps,
        top_logprobs=[],
        policy_versions=versions,
        reward=reward,
        finish_reason=0,
        payload_handle=None,
    )


def test_math_reward_uses_last_number():
    assert extract_last_number("the answer is \\boxed{42}") == "42"
    reward = MathRuleReward({1: "42", 2: "7"}, render=lambda ids: "answer 42")
    assert reward(record(1, [1])) == 1.0
    assert reward(record(2, [2])) == 0.0


def test_zero_variance_group_detection():
    good = score_group([record(1, [1]), record(1, [2])], lambda r: 1.0 if r.token_ids[0] == 1 else 0.0)
    bad = score_group([record(2, [1]), record(2, [2])], lambda r: 1.0)
    assert has_signal(good)
    assert not has_signal(bad)
    assert zero_variance_fraction([good, bad]) == 0.5


def test_collate_masks_and_cu_seqlens():
    samples = [
        from_trajectory(record(1, [11, 12], reward=1.0), [1, 2]),
        from_trajectory(record(2, [21], reward=0.0), [3, 4]),
    ]
    batch = collate(samples)
    assert batch.max_len == 4
    assert batch.cu_seqlens == [0, 4, 7]
    assert batch.loss_mask[0] == [0, 0, 1, 1]
    assert batch.loss_mask[1] == [0, 0, 1, 0]
    assert batch.attention_mask[1] == [1, 1, 1, 0]
    assert batch.token_count == 7


def test_pack_respects_token_budget_and_alignment():
    samples = [
        from_trajectory(record(i, list(range(i + 1)), reward=1.0), [1, 2])
        for i in range(1, 7)
    ]
    batches = pack(samples, max_tokens=16, alignment=4)
    assert len(batches) > 1
    for batch in batches:
        aligned = ((batch.max_len + 3) // 4) * 4
        assert aligned * batch.size <= 16
