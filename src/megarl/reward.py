"""Reward layer: pluggable verifiers, group scoring, zero-variance filtering.

Reward and advantage are decoupled from placement: a verifier can be a local
function, a TensorPlay model, or an external service returning `RewardOutcome`.
Zero-variance groups are detected here (before the trainer) so the controller
can drop and refill them.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Callable, Protocol, Sequence
from urllib.request import Request, urlopen

from . import megarl_rs as _native
from .contracts import RewardOutcome, TrajectoryRecord


class RewardFn(Protocol):
    def __call__(self, record: TrajectoryRecord) -> float: ...


def extract_last_number(text: str) -> str | None:
    """Legacy helper; answer extraction now lives in the Rust verifier."""
    return _native.extract_answer(text) or None


class MathRuleReward:
    """Verifiable math reward backed by the Rust symbolic/numeric verifier.

    `render` converts response token ids to text, e.g. `tokenizer.decode`.
    Keeping it injectable makes the layer tokenizer-agnostic and testable.
    Equivalence is semantic: `1/2`, `0.5` and `\\frac{1}{2}` all score 1.0.
    """

    def __init__(
        self,
        answers: dict[int, str],
        render: Callable[[Sequence[int]], str],
        *,
        tolerance: float = 1e-5,
    ) -> None:
        self.answers = answers
        self.render = render
        self.tolerance = tolerance

    def __call__(self, record: TrajectoryRecord) -> float:
        expected = self.answers.get(record.request_id)
        if expected is None:
            raise KeyError(f"no expected answer for request {record.request_id}")
        return float(
            _native.verify_math(self.render(record.token_ids), expected, self.tolerance)
        )


class CodeRuleReward:
    """Code reward: candidate program runs in the Rust resource-limited sandbox.

    The verifier command receives the decoded candidate on stdin and decides
    correctness by exit code. stdout/stderr are bounded and timeouts are hard.
    """

    def __init__(
        self,
        render: Callable[[Sequence[int]], str],
        *,
        command: str,
        args: Sequence[str] = (),
        timeout_ms: int = 5_000,
        memory_mb: int = 512,
        max_output_bytes: int = 1 << 20,
        seccomp: str = "blacklist",
        firejail: str = "auto",
        firejail_profile: str | None = None,
        firejail_bin: str | None = None,
    ) -> None:
        self.render = render
        self.command = command
        self.args = list(args)
        self.timeout_ms = timeout_ms
        self.memory_mb = memory_mb
        self.max_output_bytes = max_output_bytes
        self.seccomp = seccomp
        self.firejail = firejail
        self.firejail_profile = firejail_profile
        self.firejail_bin = firejail_bin

    def run(self, text: str) -> dict:
        config = json.dumps(
            {
                "command": self.command,
                "args": self.args,
                "stdin": text,
                "timeout_ms": self.timeout_ms,
                "memory_mb": self.memory_mb,
                "max_output_bytes": self.max_output_bytes,
                "seccomp": self.seccomp,
                "firejail": self.firejail,
                "firejail_profile": self.firejail_profile,
                "firejail_bin": self.firejail_bin,
            }
        )
        return json.loads(_native.run_sandboxed_json(config))

    def __call__(self, record: TrajectoryRecord) -> float:
        return 1.0 if self.run(self.render(record.token_ids))["succeeded"] else 0.0


def grade_benchmark(spec: dict, candidate: str) -> dict:
    """Grade a candidate against a benchmark spec in the Rust sandbox.

    `spec` matches the Rust ``BenchmarkSpec`` JSON shape (``benchmark``,
    ``entry_point``, ``tests`` of ``{args, expected}``, optional ``mode``,
    ``timeout_ms`` and ``memory_mb``). Optional sandbox keys ``seccomp``
    (``off``/``blacklist``), ``firejail`` (``off``/``auto``/``required``),
    ``firejail_profile`` and ``firejail_bin`` select the syscall-level layer.
    The returned dict carries a fractional ``reward`` (passed / total),
    ``per_test`` outcomes, the applied ``sandbox`` layer and an optional error.
    All grading, sandboxing and per-test timeouts live in Rust.
    """
    return json.loads(_native.grade_benchmark_json(json.dumps(spec), candidate))


def sandbox_profile() -> dict:
    """Report the active seccomp/Firejail profile from the Rust extension."""
    return json.loads(_native.sandbox_profile())


class BenchmarkReward:
    """`RewardFn` that scores trajectories with a domain-specific grader.

    HumanEval / MBPP compare the entry point's return value; APPS compares
    stdout; custom specs choose explicitly. Rewards are fractional, so a
    partially correct solution carries partial signal for GRPO.
    """

    def __init__(self, spec: dict, render: Callable[[Sequence[int]], str]) -> None:
        self.spec = spec
        self.render = render

    def __call__(self, record: TrajectoryRecord) -> float:
        return float(grade_benchmark(self.spec, self.render(record.token_ids))["reward"])


class RemoteReward:
    """Client for the standalone Rust reward-router service.

    The verifier pool can run as its own process (with auth, health and
    metrics) so CPU-heavy grading never competes with rollout or training.
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        domain: str = "math",
        answers: dict[int, str] | None = None,
        render: Callable[[Sequence[int]], str] | None = None,
        command: str = "",
        args: Sequence[str] = (),
        timeout: float = 10.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.domain = domain
        self.answers = answers or {}
        self.render = render
        self.command = command
        self.args = list(args)
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, path: str) -> dict:
        request = Request(self.url + path, headers=self._headers())
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def health(self) -> dict:
        return self._get("/health")

    def metrics(self) -> dict:
        return self._get("/metrics")

    def sandbox_profile(self) -> dict:
        """Active seccomp/Firejail profile of the remote verifier."""
        return self._get("/sandbox/profile")

    def verify(self, requests: Sequence[dict]) -> tuple[list[RewardOutcome], dict]:
        request = Request(
            self.url + "/verify",
            data=json.dumps({"requests": list(requests)}).encode(),
            headers=self._headers(),
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        outcomes = [
            RewardOutcome(
                request_id=item["request_id"],
                reward=float(item["reward"]),
                domain=item["domain"],
                metadata={
                    "verifier": item["verifier"],
                    "retries": str(item.get("retries", 0)),
                    "timed_out": str(item.get("timed_out", False)),
                    **({"sandbox": item["sandbox"]} if item.get("sandbox") else {}),
                    **({"error": item["error"]} if item.get("error") else {}),
                },
            )
            for item in payload["outcomes"]
        ]
        return outcomes, payload["stats"]

    def benchmark(self, spec: dict, candidate: str) -> dict:
        """Grade a candidate via the service's `/bench/verify` endpoint."""
        request = Request(
            self.url + "/bench/verify",
            data=json.dumps({"spec": spec, "candidate": candidate}).encode(),
            headers=self._headers(),
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())["result"]

    def __call__(self, record: TrajectoryRecord) -> float:
        if self.render is None:
            raise ValueError("RemoteReward needs render to score a trajectory")
        expected = self.answers.get(record.request_id)
        if expected is None:
            raise KeyError(f"no expected answer for request {record.request_id}")
        request = {
            "request_id": record.request_id,
            "domain": self.domain,
            "response": self.render(record.token_ids),
            "expected": expected,
        }
        if self.command:
            request.update(command=self.command, args=self.args)
        outcomes, _ = self.verify([request])
        return outcomes[0].reward


def verify_rewards(
    requests: Sequence[dict], *, workers: int = 4, max_retries: int = 1
) -> tuple[list[RewardOutcome], dict]:
    """Run the Rust domain-routed, retrying verification pool.

    Each request is a dict matching the Rust ``VerifyRequest`` JSON shape;
    domain routing, sandboxing and concurrency all happen in Rust.
    """
    outcomes_json, stats_json = _native.verify_batch_json(
        json.dumps(list(requests)), workers, max_retries
    )
    outcomes = [
        RewardOutcome(
            request_id=item["request_id"],
            reward=float(item["reward"]),
            domain=item["domain"],
            metadata={
                "verifier": item["verifier"],
                "retries": str(item.get("retries", 0)),
                "timed_out": str(item.get("timed_out", False)),
                **({"error": item["error"]} if item.get("error") else {}),
            },
        )
        for item in json.loads(outcomes_json)
    ]
    return outcomes, json.loads(stats_json)


class ModelReward:
    """Model-based reward (PRM / judge) on a TensorPlay module.

    `score_fn` receives a [1, L] int64 tensor of response token ids and returns
    one score per row. Inference runs under no_grad; the module can live on any
    TensorPlay backend, including out-of-tree devices.
    """

    def __init__(self, score_fn: Callable, *, device: str = "cpu") -> None:
        self.score_fn = score_fn
        self.device = device

    def __call__(self, record: TrajectoryRecord) -> float:
        import tensorplay as tp

        tokens = tp.tensor([record.token_ids], dtype=tp.long, device=self.device)
        with tp.no_grad():
            score = self.score_fn(tokens)
        return float(score.flatten()[0])


class ModelRewardPool:
    """Batched model-based reward across many trajectories.

    Pads variable-length responses into token-budget microbatches and runs the
    scoring model under no_grad in a worker pool, so a verifier model does not
    serialize the rollout stream.
    """

    def __init__(
        self,
        score_fn: Callable,
        *,
        batch_size: int = 16,
        workers: int = 1,
        device: str = "cpu",
    ) -> None:
        if batch_size <= 0 or workers <= 0:
            raise ValueError("batch_size and workers must be positive")
        self.score_fn = score_fn
        self.batch_size = batch_size
        self.workers = workers
        self.device = device

    def score(self, records: Sequence[TrajectoryRecord]) -> list[float]:
        import concurrent.futures

        import tensorplay as tp

        if not records:
            return []
        groups = [
            records[i : i + self.batch_size]
            for i in range(0, len(records), self.batch_size)
        ]

        def score_group(group: Sequence[TrajectoryRecord]) -> list[float]:
            width = max(len(record.token_ids) for record in group)
            tokens = [
                list(record.token_ids) + [0] * (width - len(record.token_ids))
                for record in group
            ]
            lengths = [len(record.token_ids) for record in group]
            tensor = tp.tensor(tokens, dtype=tp.long, device=self.device)
            with tp.no_grad():
                scores = self.score_fn(tensor, lengths)
            return [float(value) for value in scores.flatten().tolist()]

        if self.workers == 1:
            return [score for group in groups for score in score_group(group)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(score_group, group) for group in groups]
            return [score for future in futures for score in future.result()]


class ExternalReward:
    """Adapter from an external verifier service to `RewardFn`."""

    def __init__(self, call: Callable[[TrajectoryRecord], RewardOutcome]) -> None:
        self.call = call

    def __call__(self, record: TrajectoryRecord) -> float:
        return self.call(record).reward


def score_group(
    records: Sequence[TrajectoryRecord], reward_fn: RewardFn
) -> list[TrajectoryRecord]:
    """Return records with rewards filled, preserving order."""
    return [
        dataclasses.replace(record, reward=float(reward_fn(record)))
        for record in records
    ]


def has_signal(records: Sequence[TrajectoryRecord], eps: float = 1e-6) -> bool:
    """False if every sample in a GRPO group has the same reward."""
    rewards = [r.reward for r in records]
    if not rewards or any(reward is None for reward in rewards):
        return False
    return max(rewards) - min(rewards) > eps


def zero_variance_fraction(groups: Sequence[Sequence[TrajectoryRecord]]) -> float:
    if not groups:
        return 0.0
    return sum(not has_signal(group) for group in groups) / len(groups)
