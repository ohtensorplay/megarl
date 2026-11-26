"""Rollout adapter correctness: logprob alignment and per-token versions."""

from types import SimpleNamespace
from pathlib import Path

import pytest

from megarl.contracts import RolloutRequest, SamplingParams
from megarl.rollout import EmbeddedRollout, EmbeddedRolloutConfig, RolloutError


def request(n: int = 1) -> RolloutRequest:
    return RolloutRequest(
        request_id=1,
        policy_version=0,
        prompt_token_ids=[3, 4],
        max_new_tokens=2,
        n=n,
        sampling=SamplingParams(1.0, 1.0, 0, None, True, 0),
        session_id=None,
    )


class FakeEngine:
    def __init__(self, logprobs=(-0.1, -0.2), with_terminal=True):
        self.logprobs = logprobs
        self.with_terminal = with_terminal
        self.submitted: list[str] = []

    def add_request(self, tokens, *, max_new_tokens, sampling, request_id, **kwargs):
        assert sampling["logprobs"] is True
        self.submitted.append(request_id)
        return request_id

    def step(self):
        events = []
        for rid in self.submitted:
            events.append(
                SimpleNamespace(
                    request_id=rid,
                    kind="token",
                    token_ids=(11, 12),
                    logprobs=self.logprobs,
                    top_logprobs=((), ()),
                    finish_reason=None,
                )
            )
            if self.with_terminal:
                events.append(
                    SimpleNamespace(
                        request_id=rid,
                        kind="terminal",
                        token_ids=(),
                        logprobs=(),
                        top_logprobs=(),
                        finish_reason="stop",
                    )
                )
        self.submitted = []
        return tuple(events)


def adapter(engine, versions):
    cfg = EmbeddedRolloutConfig(model_path=Path("/nonexistent"))
    obj = EmbeddedRollout(cfg, version_provider=lambda: next(versions))
    obj._engine = engine
    obj._megasampling_cls = lambda **kw: kw
    return obj


def test_stamps_each_token_with_current_version():
    engine = FakeEngine()
    obj = adapter(engine, iter([0, 7]))
    records = list(obj.generate(request()))
    assert len(records) == 1
    record = records[0]
    assert record.token_ids == [11, 12]
    assert record.logprobs == [-0.1, -0.2]
    assert record.policy_versions == [0, 7]  # mid-generation update is visible


def test_n_fans_out_with_shared_prefix():
    engine = FakeEngine()
    obj = adapter(engine, iter([0] * 20))
    records = list(obj.generate(request(n=3)))
    assert len(records) == 3
    assert len(engine.submitted) == 0  # all consumed
    assert [r.request_id for r in records] == [1, 1, 1]


def test_refuses_misaligned_logprobs():
    engine = FakeEngine(logprobs=(-0.1,))  # one logprob for two tokens
    obj = adapter(engine, iter([0, 0]))
    with pytest.raises(RolloutError, match="misaligned"):
        list(obj.generate(request()))
