"""Advantage estimators: exact math, masking and reward-scaling modes."""

import math

import tensorplay as tp

from megarl.advantages import (
    ADVANTAGE_ESTIMATORS,
    gae_advantages,
    group_advantages,
    reinforce_advantages,
    rloo_advantages,
    token_rewards,
)


def test_group_advantages_scaling_modes():
    groups = [[0.0, 2.0], [1.0, 3.0]]
    grpo = group_advantages(groups, scale_rewards="group")
    none = group_advantages(groups, scale_rewards="none")
    batch = group_advantages(groups, scale_rewards="batch")

    assert all(math.isclose(a, b, abs_tol=1e-3) for a, b in zip(grpo, [-1.0, 1.0, -1.0, 1.0]))
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(none, [-1.0, 1.0, -1.0, 1.0]))
    # Batch std (≈1.118) is larger than each group std (1.0), so scaled
    # advantages are smaller than the group-normalized ones.
    assert max(abs(v) for v in batch) < max(abs(v) for v in grpo) + 1e-9


def test_rloo_and_reinforce_baselines():
    assert all(
        math.isclose(a, b, abs_tol=1e-9)
        for a, b in zip(rloo_advantages([[0.0, 1.0, 2.0]]), [-1.5, 0.0, 1.5])
    )
    assert all(
        math.isclose(a, b, abs_tol=1e-9)
        for a, b in zip(reinforce_advantages([[0.0, 1.0, 2.0]]), [-1.0, 0.0, 1.0])
    )
    assert set(ADVANTAGE_ESTIMATORS) == {"grpo", "rloo", "reinforce"}


def test_token_rewards_land_on_last_valid_token():
    mask = tp.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    rewards = token_rewards(tp.tensor([1.0, -1.0]), mask)
    assert rewards.tolist() == [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]


def test_gae_matches_recursive_definition_and_ignores_padding():
    mask = tp.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    rewards = tp.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    values = tp.tensor([[0.1, 0.2, 0.3, 0.0], [0.1, 0.2, 0.3, 0.0]])
    advantages, returns = gae_advantages(
        rewards, values, mask, gamma=1.0, lam=1.0, normalize=False
    )
    # delta = [0.1, 0.1, 0.7]; A_t = delta_t + A_{t+1} on a live episode.
    assert all(
        math.isclose(a, b, abs_tol=1e-5)
        for a, b in zip(advantages[0].tolist(), [0.9, 0.8, 0.7])
    )
    # The second episode terminates after token 1, so no bootstrap leaks back.
    assert all(
        math.isclose(a, b, abs_tol=1e-5)
        for a, b in zip(advantages[1].tolist()[:2], [-0.1, -0.2])
    )
    assert all(
        math.isclose(a, b, abs_tol=1e-5)
        for a, b in zip(returns[0].tolist(), [1.0, 1.0, 1.0])
    )

    corrupted = values.clone()
    corrupted[1, 2] = 99.0  # padded bootstrap must not change valid advantages
    _, returns2 = gae_advantages(rewards, corrupted, mask, gamma=1.0, lam=1.0, normalize=False)
    assert all(
        math.isclose(a, b, abs_tol=1e-5)
        for a, b in zip(returns2[1].tolist()[:2], returns[1].tolist()[:2])
    )


def test_packed_batches_keep_advantage_alignment():
    from megarl.batching import RLSample, pack_with_order

    rewards = [10.0, 20.0, 30.0]
    samples = [
        RLSample(
            prompt_len=1,
            token_ids=[1] * (length + 1),
            old_logprobs=[0.0] * length,
            versions=[0] * length,
            reward=reward,
        )
        for length, reward in zip((2, 3, 4), rewards)
    ]
    batches, orders = pack_with_order(samples, max_tokens=64)
    for batch, order in zip(batches, orders):
        for row_reward, original_index in zip(batch.rewards, order):
            assert row_reward == rewards[original_index]
    assert sorted(i for order in orders for i in order) == [0, 1, 2]


def test_gae_rejects_bad_value_shape():
    mask = tp.zeros((2, 3))
    try:
        gae_advantages(tp.zeros((2, 3)), tp.zeros((2, 3)), mask)
    except ValueError as exc:
        assert "L + 1" in str(exc)
    else:
        raise AssertionError("value shape was not validated")
