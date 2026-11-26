"""Model-based reward runs under no_grad on a TensorPlay module."""

import pytest
import tensorplay as tp

from megarl.contracts import TrajectoryRecord
from megarl.reward import ModelReward, ModelRewardPool


def test_model_reward_runs_under_no_grad():
    reward = ModelReward(lambda tokens: tokens.to(tp.float32).mean())
    record = TrajectoryRecord(1, 1, [2, 4], [-0.1, -0.2], [], [0, 0], None, 0, None)
    assert reward(record) == pytest.approx(3.0)


def test_model_reward_pool_batches_and_preserves_order():
    records = [
        TrajectoryRecord(i, 1, [i, i + 1], [-0.1] * 2, [], [0, 0], None, 0, None)
        for i in range(5)
    ]
    pool = ModelRewardPool(
        lambda tokens, lengths: tokens.to(tp.float32).mean(dim=1),
        batch_size=2,
        workers=2,
    )
    scores = pool.score(records)
    assert len(scores) == 5
    assert scores[0] == pytest.approx(0.5)
    assert scores[4] == pytest.approx(4.5)
    assert pool.score([]) == []
