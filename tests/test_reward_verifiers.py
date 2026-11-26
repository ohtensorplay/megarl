"""Production reward verification: the Python layer is a thin Rust adapter."""

import json
import subprocess
import time
from pathlib import Path

import pytest

from megarl.contracts import TrajectoryRecord
from megarl import megarl_rs as native
from megarl.reward import CodeRuleReward, MathRuleReward, RemoteReward, verify_rewards


def record(rid: int, token_ids: list[int]) -> TrajectoryRecord:
    return TrajectoryRecord(
        request_id=rid,
        prompt_len=0,
        token_ids=token_ids,
        logprobs=[],
        top_logprobs=[],
        policy_versions=[],
        reward=None,
        finish_reason=0,
        payload_handle=None,
    )


def test_math_reward_accepts_equivalent_forms():
    reward = MathRuleReward({1: "1/2", 2: "8"}, render=lambda ids: text[ids[0]])
    text = ["work ... \\boxed{0.5}", "work ... \\boxed{2^{3}}"]
    assert reward(record(1, [0])) == 1.0
    assert reward(record(2, [1])) == 1.0
    assert native.math_equivalent("\\frac{1}{2}", "0.5", 1e-5)


def test_code_reward_runs_in_sandbox():
    reward = CodeRuleReward(
        render=lambda ids: "pass\n" if ids[0] == 1 else "fail\n",
        command="/bin/sh",
        args=["-c", 'read line; test "$line" = pass'],
    )
    assert reward(record(1, [1])) == 1.0
    assert reward(record(2, [0])) == 0.0


def test_sandbox_kills_timed_out_process_group():
    start = time.monotonic()
    result = json.loads(
        native.run_sandboxed_json(
            json.dumps(
                {
                    "command": "/bin/sh",
                    "args": ["-c", "sleep 30"],
                    "timeout_ms": 100,
                }
            )
        )
    )
    assert result["timed_out"] is True
    assert result["succeeded"] is False
    assert time.monotonic() - start < 3


def test_domain_routed_batch_preserves_order_and_reports_stats():
    requests = [
        {"request_id": 0, "domain": "math", "response": "0.5", "expected": "1/2"},
        {"request_id": 1, "domain": "generic", "response": "yes", "expected": "yes"},
        {
            "request_id": 2,
            "domain": "code",
            "response": "pass\n",
            "expected": "",
            "command": "/bin/sh",
            "args": ["-c", 'read line; test "$line" = pass'],
        },
    ]
    outcomes, stats = verify_rewards(requests, workers=3)
    assert [outcome.request_id for outcome in outcomes] == [0, 1, 2]
    assert [outcome.reward for outcome in outcomes] == [1.0, 1.0, 1.0]
    assert stats["completed"] == 3
    assert stats["failed"] == 0
    assert stats["timeouts"] == 0


def test_bad_batch_shape_is_rejected():
    with pytest.raises(ValueError, match="invalid verify batch"):
        native.verify_batch_json("not-json", 1, 0)


def test_remote_reward_router_service():
    binary = (
        Path(__file__).resolve().parents[1]
        / "rust"
        / "target"
        / "debug"
        / "megarl-reward-router"
    )
    if not binary.exists():
        pytest.skip("build megarl-reward-router to run the service integration test")
    proc = subprocess.Popen(
        [str(binary), "127.0.0.1:0"], stdout=subprocess.PIPE, text=True
    )
    try:
        address = proc.stdout.readline().strip()
        url = f"http://{address}"
        reward = RemoteReward(
            url,
            domain="math",
            answers={1: "1/2"},
            render=lambda ids: "work ... \\boxed{0.5}",
        )
        assert reward.health()["ok"] is True
        assert reward(record(1, [1])) == 1.0
        outcomes, stats = reward.verify(
            [
                {
                    "request_id": 7,
                    "domain": "generic",
                    "response": "yes",
                    "expected": "yes",
                }
            ]
        )
        assert outcomes[0].reward == 1.0
        assert stats["completed"] == 1
        assert reward.metrics()["verified"] >= 2
    finally:
        proc.terminate()
        proc.wait(timeout=5)
