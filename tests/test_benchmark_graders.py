"""Domain-specific code graders (HumanEval / MBPP / APPS / custom).

All grading happens in Rust inside the existing resource-limited sandbox; these
tests only exercise the thin Python adapter surface.
"""

import pytest

from megarl.reward import BenchmarkReward, grade_benchmark
from megarl.contracts import TrajectoryRecord


def _record(text: str) -> TrajectoryRecord:
    return TrajectoryRecord(
        request_id=1,
        prompt_len=0,
        token_ids=list(text.encode()),
        logprobs=[],
        top_logprobs=[],
        policy_versions=[],
        reward=None,
        finish_reason=0,
        payload_handle=None,
        sampling_logprobs=[],
    )


def _spec(benchmark: str, expected: str) -> dict:
    return {
        "benchmark": benchmark,
        "entry_point": "add",
        "tests": [
            {"args": [1, 2], "expected": expected},
            {"args": [10, 5], "expected": expected},
        ],
    }


def test_humaneval_return_mode_full_and_partial_credit():
    correct = "def add(a, b):\n    return a + b\n"
    spec = _spec("human_eval", "3")
    spec["tests"][1]["expected"] = "15"
    result = grade_benchmark(spec, correct)
    assert result["benchmark"] == "human_eval"
    assert result["reward"] == 1.0
    assert result["per_test"] == [True, True]

    wrong = "def add(a, b):\n    return a - b\n"
    assert grade_benchmark(spec, wrong)["reward"] == 0.0


def test_apps_stdout_mode_partial_credit():
    spec = _spec("apps", "2")
    result = grade_benchmark(spec, "def add(a, b):\n    print(a * b)\n")
    assert result["reward"] == pytest.approx(0.5)
    assert result["passed"] == 1


def test_benchmark_reward_is_a_reward_fn():
    spec = _spec("mbpp", "3")
    spec["tests"][1]["expected"] = "15"
    reward = BenchmarkReward(spec, lambda ids: bytes(ids).decode())
    record = _record("def add(a, b):\n    return a + b\n")
    assert reward(record) == 1.0


def test_invalid_spec_returns_zero_with_error():
    spec = _spec("custom", "3")
    spec["tests"] = []
    result = grade_benchmark(spec, "def add(a, b): return a + b")
    assert result["reward"] == 0.0
    assert result["error"]
