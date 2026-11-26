"""Tool registry and execution for agentic RL.

Design (see `devdocs/agentic-rl.md`): tool-call *parsing* and JSON-Schema
argument validation live in the Rust spine (`megarl-agent`); this module owns
registration, dispatch, timeouts, retries and result truncation. Parsing is
all-or-nothing: a malformed later call prevents any call from executing, which
avoids the partial-execution failure mode seen in the pinned references.

Tools are ordinary callables with a JSON-Schema parameter definition. Results
are rendered to text (and truncated) before being tokenized back into the
trajectory, so large observations never enter the control plane as tensors.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import operator
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable, Sequence

from . import megarl_rs as _native
from .observability import Telemetry


class ToolError(RuntimeError):
    """Base class for registry, validation and execution failures."""


class UnknownTool(ToolError):
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class ParsedToolCall:
    """A validated, canonical tool call from the Rust parser."""

    id: str
    name: str
    arguments: dict[str, Any]
    arguments_json: str

    @classmethod
    def parse(cls, text: str, format: str = "auto") -> list[ParsedToolCall]:
        """Parse all calls in one shot; never return a partial prefix."""
        try:
            raw = json.loads(_native.parse_tool_calls(text, format))
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        calls: list[ParsedToolCall] = []
        for item in raw:
            canonical = item["arguments_json"]
            arguments = json.loads(canonical)
            if not isinstance(arguments, dict):
                raise ToolError("tool arguments must be a JSON object")
            calls.append(
                cls(
                    id=item["id"],
                    name=item["name"],
                    arguments=arguments,
                    arguments_json=canonical,
                )
            )
        return calls


@dataclasses.dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    name: str
    ok: bool
    content: str
    error: str = ""
    elapsed_ms: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class Tool:
    """One callable tool with a declared JSON-Schema parameter object."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    timeout_secs: float = 10.0
    max_result_chars: int = 4096
    retries: int = 0
    # Cached once at construction: tool schemas are immutable, and serializing
    # them per call was a hot path in multi-turn episodes.
    _schema_json: str = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_schema_json",
            json.dumps(self.parameters, separators=(",", ":")),
        )

    @property
    def schema_json(self) -> str:
        return self._schema_json

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function definition for the rollout's tool prompt."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate(self, arguments: dict[str, Any]) -> None:
        """Validate against the Rust JSON-Schema subset before execution."""
        self.validate_json(json.dumps(arguments, separators=(",", ":")))

    def validate_json(self, arguments_json: str) -> None:
        """Validate already-canonical arguments (no re-serialization)."""
        try:
            _native.validate_tool_arguments(self._schema_json, arguments_json)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    def invoke(self, arguments: dict[str, Any]) -> str:
        value = self.handler(arguments)
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if len(text) > self.max_result_chars:
            text = text[: self.max_result_chars] + "\n[truncated]"
        return text


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    *,
    timeout_secs: float = 10.0,
    max_result_chars: int = 4096,
    retries: int = 0,
) -> Callable[[Callable], Tool]:
    """Decorator form: register a plain function as a :class:`Tool`."""

    def wrap(handler: Callable) -> Tool:
        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            timeout_secs=timeout_secs,
            max_result_chars=max_result_chars,
            retries=retries,
        )

    return wrap


class ToolRegistry:
    """Thread-safe registry with schema-checked, bounded parallel execution.

    All calls are validated before any is submitted, so a hallucinated tool or
    an invalid argument object cannot let a valid prefix of the batch run.
    Timeouts return structured failures rather than raising into the agent
    loop; a tool failure is a training signal (the policy chose a bad action),
    not a reason to drop the whole episode.
    """

    def __init__(
        self,
        tools: Sequence[Tool] = (),
        *,
        max_parallel_calls: int = 4,
        telemetry: Telemetry | None = None,
    ) -> None:
        if max_parallel_calls < 1:
            raise ValueError("max_parallel_calls must be >= 1")
        self._tools: dict[str, Tool] = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(
            max_workers=max_parallel_calls, thread_name_prefix="megarl-tool"
        )
        self.telemetry = telemetry
        for item in tools:
            self.register(item)

    def register(self, item: Tool) -> None:
        if not item.name or not isinstance(item.parameters, dict):
            raise ValueError("tool needs a name and a JSON-Schema object")
        with self._lock:
            if item.name in self._tools:
                raise ValueError(f"duplicate tool: {item.name}")
            self._tools[item.name] = item

    def unregister(self, name: str) -> None:
        with self._lock:
            self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        with self._lock:
            item = self._tools.get(name)
        if item is None:
            raise UnknownTool(f"unknown tool: {name}")
        return item

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        with self._lock:
            return [item.schema() for item in self._tools.values()]

    def execute(self, calls: Sequence[ParsedToolCall]) -> list[ToolResult]:
        """Execute a batch of calls; results are returned in call order."""
        if not calls:
            return []
        # Validate the *entire* batch before submitting anything.
        resolved: list[Tool] = []
        for call in calls:
            item = self.get(call.name)  # raises UnknownTool
            item.validate_json(call.arguments_json)
            resolved.append(item)

        def run(item: Tool, arguments: dict[str, Any]) -> str:
            for attempt in range(item.retries + 1):
                try:
                    return item.invoke(arguments)
                except Exception:
                    if attempt == item.retries:
                        raise

        # Deadlines start at submission, not at collection: a slow earlier
        # call must not grant extra wall time to a later one.
        futures: list[Future] = []
        deadlines: list[float] = []
        for item, call in zip(resolved, calls):
            futures.append(self._pool.submit(run, item, call.arguments))
            deadlines.append(time.monotonic() + item.timeout_secs)
        results: list[ToolResult] = []
        for call, item, future, deadline in zip(calls, resolved, futures, deadlines):
            started = time.monotonic()
            error = ""
            try:
                content = future.result(timeout=max(0.0, deadline - time.monotonic()))
                ok = True
            except FutureTimeout:
                content, ok, error = "", False, f"tool timed out after {item.timeout_secs}s"
                future.cancel()
            except Exception as exc:  # a failing tool is a policy signal, not a crash
                content, ok, error = "", False, f"{type(exc).__name__}: {exc}"
            elapsed = int((time.monotonic() - started) * 1000)
            results.append(
                ToolResult(call.id, call.name, ok, content, error, elapsed)
            )
            if self.telemetry is not None:
                self.telemetry.emit(
                    "agent_tool",
                    call_id=call.id,
                    name=call.name,
                    ok=ok,
                    elapsed_ms=elapsed,
                    error=error,
                )
        return results

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> ToolRegistry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_arith(node: ast.AST) -> float:
    """Evaluate a numeric expression without names, calls or attributes."""
    if isinstance(node, ast.Expression):
        return _safe_arith(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _safe_arith(node.left), _safe_arith(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return float(_ALLOWED_UNARY[type(node.op)](_safe_arith(node.operand)))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def calculator(arguments: dict[str, Any]) -> str:
    """Deterministic arithmetic tool; no `eval`, no names, no builtins."""
    expression = arguments.get("expr")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expr must be a non-empty string")
    try:
        value = _safe_arith(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"invalid arithmetic expression: {exc}") from exc
    if not math.isfinite(value):
        raise ValueError("arithmetic result is not finite")
    return str(value)


CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Evaluate a numeric arithmetic expression (+, -, *, /, %, **).",
    parameters={
        "type": "object",
        "required": ["expr"],
        "properties": {"expr": {"type": "string", "minLength": 1, "maxLength": 512}},
        "additionalProperties": False,
    },
    handler=calculator,
    timeout_secs=2.0,
    max_result_chars=128,
)


def sandboxed_python(arguments: dict[str, Any]) -> str:
    """Run Python source in the Rust resource-limited sandbox (hard timeout)."""
    code = arguments.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")
    config = json.dumps(
        {
            "command": sys.executable,
            "args": ["-I", "-"],  # read the program from stdin under the sandbox
            "stdin": code,
            "timeout_ms": int(arguments.get("timeout_ms", 5_000)),
            "memory_mb": int(arguments.get("memory_mb", 256)),
            "max_output_bytes": 1 << 16,
        }
    )
    return _native.run_sandboxed_json(config)


PYTHON_TOOL = Tool(
    name="python",
    description="Run Python code in a CPU/memory/time-limited sandbox.",
    parameters={
        "type": "object",
        "required": ["code"],
        "properties": {
            "code": {"type": "string", "minLength": 1},
            "timeout_ms": {"type": "integer", "minimum": 100, "maximum": 60_000},
            "memory_mb": {"type": "integer", "minimum": 64, "maximum": 4096},
        },
        "additionalProperties": False,
    },
    handler=sandboxed_python,
    timeout_secs=10.0,
    max_result_chars=4096,
)


def default_registry(telemetry: Telemetry | None = None) -> ToolRegistry:
    """Registry with the built-in calculator and sandboxed Python tools."""
    return ToolRegistry([CALCULATOR_TOOL, PYTHON_TOOL], telemetry=telemetry)
