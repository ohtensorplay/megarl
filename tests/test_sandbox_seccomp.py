"""Thin Python adapters for the Rust seccomp/Firejail reward sandbox.

All isolation decisions are made in Rust; these tests verify that the adapter
passes the profile through, reports the applied layer and that the profile is
actually enforced on a real subprocess.
"""

import json
import subprocess
from pathlib import Path

import pytest

from megarl.reward import CodeRuleReward, RemoteReward, grade_benchmark, sandbox_profile

SOCKET_PROBE = (
    "import ctypes\n"
    "libc = ctypes.CDLL(None, use_errno=True)\n"
    "print('BLOCKED' if libc.socket(2, 1, 0) < 0 and ctypes.get_errno() == 1 else 'ALLOWED')\n"
)

BENCH_SPEC = {
    "benchmark": "custom",
    # `custom` defaults to stdout comparison; these candidates return values.
    "mode": "return",
    "entry_point": "add",
    "tests": [
        {"args": [1, 2], "expected": "3"},
        {"args": [10, 5], "expected": "15"},
    ],
    "seccomp": "blacklist",
    "firejail": "off",
}


def test_sandbox_profile_is_reported_by_the_native_extension():
    profile = sandbox_profile()
    assert profile["seccomp"] in {"off", "blacklist"}
    assert profile["firejail"] in {"auto", "off", "required"}
    assert isinstance(profile["firejail_available"], bool)
    assert profile["effective_label"] in {"none", "seccomp_blacklist", "firejail+seccomp"}


def test_code_rule_reward_enforces_seccomp_blacklist():
    reward = CodeRuleReward(
        lambda ids: SOCKET_PROBE,
        command="python3",
        args=["-I", "-c", "import sys; exec(sys.stdin.read())"],
        seccomp="blacklist",
        firejail="off",
    )
    result = reward.run(SOCKET_PROBE)
    assert result["succeeded"] is True, result
    assert result["stdout"] == "BLOCKED\n"
    assert result["sandbox"] == "seccomp_blacklist"


def test_code_rule_reward_can_disable_seccomp_for_local_experiments():
    reward = CodeRuleReward(
        lambda ids: SOCKET_PROBE,
        command="python3",
        args=["-I", "-c", "import sys; exec(sys.stdin.read())"],
        seccomp="off",
        firejail="off",
    )
    result = reward.run(SOCKET_PROBE)
    assert result["stdout"] == "ALLOWED\n"
    assert result["sandbox"] == "none"


def test_grade_benchmark_reports_the_applied_layer():
    result = grade_benchmark(BENCH_SPEC, "def add(a, b):\n    return a + b\n")
    assert result["reward"] == 1.0
    assert result["sandbox"] == "seccomp_blacklist"


def test_required_firejail_fails_closed_through_the_grader():
    spec = dict(BENCH_SPEC)
    spec["firejail"] = "required"
    spec["firejail_bin"] = "/nonexistent/megarl-firejail"
    result = grade_benchmark(spec, "def add(a, b):\n    return a + b\n")
    assert result["reward"] == 0.0
    assert result["error"]

    # The same failure is visible through the raw sandbox adapter.
    raw = CodeRuleReward(
        lambda ids: "ok",
        command="/bin/echo",
        firejail="required",
        firejail_bin="/nonexistent/megarl-firejail",
    )
    try:
        raw.run("ok\n")
    except ValueError as exc:
        assert "firejail" in str(exc)
    else:
        raise AssertionError("a missing required firejail must fail closed")


def test_profile_json_round_trips_with_legacy_specs():
    # A spec without sandbox keys keeps working (serde defaults in Rust).
    legacy = {key: value for key, value in BENCH_SPEC.items() if "firejail" not in key and key != "seccomp"}
    result = grade_benchmark(legacy, "def add(a, b):\n    return a + b\n")
    assert result["reward"] == 1.0
    assert "sandbox" in result
    assert json.loads(json.dumps(result))["passed"] == 2


def test_reward_service_exposes_the_profile_over_http():
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
        client = RemoteReward("http://" + proc.stdout.readline().strip())
        profile = client.sandbox_profile()
        assert profile["seccomp"] in {"off", "blacklist"}
        assert profile["effective_label"] in {
            "none",
            "seccomp_blacklist",
            "firejail+seccomp",
        }
        assert "sandbox" in client.metrics()
        graded = client.benchmark(BENCH_SPEC, "def add(a, b):\n    return a + b\n")
        assert graded["reward"] == 1.0
        assert graded["sandbox"] == "seccomp_blacklist"
    finally:
        proc.terminate()
