"""Tool registry: schema validation, all-or-nothing dispatch, timeouts, sandbox."""

import json
import time

import pytest

from megarl.observability import Telemetry
from megarl.tools import (
    CALCULATOR_TOOL,
    PYTHON_TOOL,
    ParsedToolCall,
    Tool,
    ToolError,
    ToolRegistry,
    UnknownTool,
    calculator,
)


def call(name, args, call_id="c1"):
    canonical = json.dumps(args, separators=(",", ":"))
    return ParsedToolCall(id=call_id, name=name, arguments=args, arguments_json=canonical)


def test_parser_is_all_or_nothing():
    text = json.dumps(
        {
            "tool_calls": [
                {"id": "a", "name": "calculator", "arguments": {"expr": "1+1"}},
                {"id": "b", "name": "calculator", "arguments": [1, 2]},
            ]
        }
    )
    with pytest.raises(ToolError):
        ParsedToolCall.parse(text)
    assert ParsedToolCall.parse("The answer is 2.") == []


def test_registry_validates_every_call_before_any_execution():
    executed = []

    def spy(args):
        executed.append(args)
        return args

    registry = ToolRegistry(
        [
            Tool(
                name="spy",
                description="counts invocations",
                parameters={
                    "type": "object",
                    "required": ["x"],
                    "properties": {"x": {"type": "integer"}},
                    "additionalProperties": False,
                },
                handler=spy,
            )
        ]
    )
    with pytest.raises(UnknownTool):
        registry.execute([call("spy", {"x": 1}), call("missing", {})])
    assert executed == []  # the valid prefix must never run

    with pytest.raises(ToolError):
        registry.execute([call("spy", {"x": "not an int"})])
    assert executed == []
    registry.close()


def test_execute_returns_structured_results_in_order():
    registry = ToolRegistry([CALCULATOR_TOOL], max_parallel_calls=2)
    results = registry.execute(
        [call("calculator", {"expr": "2*3"}, "a"), call("calculator", {"expr": "10-4"}, "b")]
    )
    assert [r.call_id for r in results] == ["a", "b"]
    assert [r.content for r in results] == ["6.0", "6.0"]
    assert all(r.ok for r in results)
    registry.close()


def test_timeout_and_truncation_are_not_fatal():
    def slow(args):
        time.sleep(0.3)
        return "too late"

    registry = ToolRegistry(
        [
            Tool("slow", "slow tool", {"type": "object"}, slow, timeout_secs=0.01),
            Tool(
                "long",
                "long output",
                {"type": "object"},
                lambda args: "x" * 5000,
                max_result_chars=64,
            ),
        ]
    )
    timed_out, truncated = registry.execute([call("slow", {}), call("long", {})])
    assert not timed_out.ok and "timed out" in timed_out.error
    assert truncated.ok and truncated.content.endswith("[truncated]")
    assert len(truncated.content) <= 64 + len("\n[truncated]")
    registry.close()


def test_retries_are_bounded_and_eventually_fail():
    attempts = []

    def flaky(args):
        attempts.append(1)
        raise RuntimeError("boom")

    registry = ToolRegistry(
        [Tool("flaky", "always fails", {"type": "object"}, flaky, retries=2)]
    )
    result = registry.execute([call("flaky", {})])[0]
    assert len(attempts) == 3
    assert result.ok is False and "boom" in result.error
    registry.close()


def test_schema_json_is_cached_and_canonical():
    schema = CALCULATOR_TOOL.schema_json
    assert isinstance(schema, str)
    assert " " not in schema  # separators=(",", ":") — no whitespace
    assert schema == CALCULATOR_TOOL.schema_json  # stable across calls
    # Validation uses the cached schema, not a fresh serialization each call.
    CALCULATOR_TOOL.validate_json('{"expr":"1+1"}')
    with pytest.raises(ToolError):
        CALCULATOR_TOOL.validate_json('{"expr":123}')


def test_deadline_starts_at_submission_not_collection():
    def slow(args):
        time.sleep(0.4)
        return "done"

    registry = ToolRegistry(
        [Tool("slow", "sleeps", {"type": "object"}, slow, timeout_secs=0.05)]
    )
    started = time.monotonic()
    results = registry.execute(
        [call("slow", {}, "a"), call("slow", {}, "b")]
    )
    elapsed = time.monotonic() - started
    # Both calls share a submission-time deadline; collection must not give the
    # second call a fresh 0.05s window after waiting on the first.
    assert elapsed < 0.3
    assert all(r.ok is False and "timed out" in r.error for r in results)
    assert [r.call_id for r in results] == ["a", "b"]  # call order preserved
    registry.close()


def test_tool_events_are_emitted():
    telemetry = Telemetry()
    with ToolRegistry([CALCULATOR_TOOL], telemetry=telemetry) as registry:
        registry.execute([call("calculator", {"expr": "1+1"})])
    events = [e for e in telemetry.events if e["event"] == "agent_tool"]
    assert len(events) == 1 and events[0]["ok"] is True


def test_calculator_rejects_code_injection():
    with pytest.raises(ValueError):
        calculator({"expr": "__import__('os').system('echo hi')"})
    with pytest.raises(ValueError):
        calculator({"expr": "1/0"})
    assert calculator({"expr": "(2+3)*4"}) == "20.0"


def test_sandboxed_python_tool_is_resource_limited():
    result = PYTHON_TOOL.invoke({"code": "print(2+2)", "timeout_ms": 2000, "memory_mb": 128})
    payload = json.loads(result)
    assert payload["succeeded"] is True
    assert "4" in payload["stdout"]
