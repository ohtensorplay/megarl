"""Python mirror of the authoritative Rust wire contract.

Keep the field sets in sync with `rust/crates/protocol/src/lib.rs`. The
`validate()` helper checks a payload against the Rust definition, so drift is
caught by tests instead of at runtime. Frames carry metadata and handles only;
bulk tensors travel out of band (see devdocs/architecture.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from typing import Any

PROTOCOL_VERSION = 1
MAGIC = b"MGRL"


@dataclass(frozen=True, slots=True)
class SamplingParams:
    temperature: float
    top_p: float
    top_k: int
    seed: int | None
    logprobs: bool
    top_logprobs: int


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    request_id: int
    policy_version: int
    prompt_token_ids: list[int]
    max_new_tokens: int
    n: int
    sampling: SamplingParams
    session_id: int | None
    # Optional model name; the gateway uses it to constrain routing to the
    # workers serving that model's active version (None = no constraint).
    model: str | None = None


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    request_id: int
    prompt_len: int
    token_ids: list[int]
    logprobs: list[float]
    top_logprobs: list[list[tuple[int, float]]]
    policy_versions: list[int]
    reward: float | None
    finish_reason: int
    payload_handle: int | None
    # vLLM sampling logprobs; may differ from the trainer's recomputed
    # logprobs. Empty means "no known mismatch" and disables IS correction.
    sampling_logprobs: list[float] = field(default_factory=list, kw_only=True)


@dataclass(frozen=True, slots=True)
class MultimodalTrajectory:
    request_id: int
    prompt_len: int
    token_ids: list[int]
    logprobs: list[float]
    policy_versions: list[int]
    image_token_ids: list[int]
    image_insert_at: int
    image_grid: tuple[int, int]
    modality_mask: list[int]
    reward: float | None
    finish_reason: int
    payload_handle: int | None


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    """One tool invocation inside an agentic episode (Rust-mirrored)."""

    id: str
    name: str
    arguments_json: str
    turn: int
    ok: bool
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class AgentTrajectory:
    """Multi-turn agent episode with exact policy-token provenance.

    ``token_ids`` is the full flattened sequence. Response-aligned vectors
    (``logprobs``, ``policy_versions``, optional ``sampling_logprobs`` and
    ``top_logprobs``) cover ``token_ids[prompt_len:]``. ``loss_mask``,
    ``token_roles`` and ``turn_ids`` align with the *full* sequence; only
    assistant tokens may carry ``loss_mask=1``. The Rust mirror enforces these
    invariants in ``validate()``.

    Roles: 0 = initial prompt, 1 = assistant, 2 = tool observation.
    Finish reasons: 0 = stop, 1 = max_turns, 2 = max_tokens, 3 = tool_error,
    4 = timeout.
    """

    request_id: int
    policy_version: int
    prompt_len: int
    token_ids: list[int]
    logprobs: list[float]
    top_logprobs: list[list[tuple[int, float]]]
    policy_versions: list[int]
    token_roles: list[int]
    turn_ids: list[int]
    loss_mask: list[int]
    reward: float | None
    finish_reason: int
    payload_handle: int | None
    sampling_logprobs: list[float] = field(default_factory=list, kw_only=True)
    tool_calls: list[AgentToolCall] = field(default_factory=list, kw_only=True)
    step_rewards: list[float] = field(default_factory=list, kw_only=True)


@dataclass(frozen=True, slots=True)
class HealthCheck:
    probe: int
    ok: bool


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    name: str
    shape: list[int]
    dtype: str
    shard_rank: int
    byte_offset: int
    byte_length: int


@dataclass(frozen=True, slots=True)
class WeightUpdate:
    from_version: int
    to_version: int
    transport: str
    digest: str
    tensors: list[TensorDescriptor]
    abort_inflight: bool


@dataclass(frozen=True, slots=True)
class RewardOutcome:
    request_id: int
    reward: float
    domain: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class LifecycleCommand:
    """Colocate lifecycle command. `action` is `pause` or `resume`; `epoch`
    guards acks so a delayed reply from an older transition is rejected."""

    request_id: int
    epoch: int
    action: str
    offload: bool = True
    drain_ms: int = 30_000


@dataclass(frozen=True, slots=True)
class LifecycleAck:
    request_id: int
    epoch: int
    action: str
    ok: bool
    phase: str
    drained: int = 0
    error: str | None = None
    elapsed_ms: int = 0


@dataclass(frozen=True, slots=True)
class WeightUpdateAck:
    """Worker-reported weight-reload phase timings (wire kind 10)."""

    request_id: int
    ok: bool
    pause_ms: int = 0
    barrier_ms: int = 0
    transfer_ms: int = 0
    apply_ms: int = 0
    total_ms: int = 0
    error: str | None = None


KINDS: dict[str, type] = {
    "RolloutRequest": RolloutRequest,
    "TrajectoryRecord": TrajectoryRecord,
    "WeightUpdate": WeightUpdate,
    "RewardOutcome": RewardOutcome,
    "MultimodalTrajectory": MultimodalTrajectory,
    "HealthCheck": HealthCheck,
    "AgentTrajectory": AgentTrajectory,
    "LifecycleCommand": LifecycleCommand,
    "LifecycleAck": LifecycleAck,
    "WeightUpdateAck": WeightUpdateAck,
}


def _native():
    try:
        from . import megarl_rs
    except ImportError as exc:  # pragma: no cover - depends on build state
        raise RuntimeError(
            "megarl native extension is missing; build it with "
            "`maturin develop --manifest-path rust/Cargo.toml`"
        ) from exc
    return megarl_rs


def encode(message: Any) -> bytes:
    """Serialize a contract dataclass into a versioned MGRL frame.

    Controller-facing adapters (e.g. ``batching.AgentRecord``) are transparently
    encoded as the underlying contract message, carrying any reward/handle that
    the controller stamped onto the adapter.
    """
    inner = getattr(message, "trajectory", None)
    if inner is not None and type(inner).__name__ in KINDS:
        overrides: dict[str, Any] = {}
        reward = getattr(message, "reward", None)
        handle = getattr(message, "payload_handle", None)
        if reward is not None:
            overrides["reward"] = reward
        if handle is not None:
            overrides["payload_handle"] = handle
        if overrides:
            inner = replace(inner, **overrides)
        message = inner
    kind = type(message).__name__
    if kind not in KINDS:
        raise TypeError(f"not a megaRL contract message: {kind}")
    return _native().encode(kind, json.dumps(asdict(message), separators=(",", ":")))


def decode(frame: bytes) -> tuple[str, Any]:
    """Decode a frame into (kind name, contract dataclass)."""
    kind, payload_json = _native().decode(frame)
    data = json.loads(payload_json)
    # serde treats a missing Option field as None; mirror that on this side.
    if kind == "RolloutRequest":
        data.setdefault("session_id", None)
        data.setdefault("model", None)
        sampling_data = data["sampling"]
        sampling_data.setdefault("seed", None)
        data["sampling"] = SamplingParams(**sampling_data)
    elif kind == "TrajectoryRecord":
        data.setdefault("reward", None)
        data.setdefault("finish_reason", 0)
        data.setdefault("payload_handle", None)
        data["top_logprobs"] = [
            [tuple(pair) for pair in row] for row in data["top_logprobs"]
        ]
    elif kind == "MultimodalTrajectory":
        data.setdefault("reward", None)
        data.setdefault("finish_reason", 0)
        data.setdefault("payload_handle", None)
        data["image_grid"] = tuple(data["image_grid"])
    elif kind == "WeightUpdate":
        data["tensors"] = [TensorDescriptor(**t) for t in data["tensors"]]
    elif kind == "AgentTrajectory":
        data.setdefault("reward", None)
        data.setdefault("finish_reason", 0)
        data.setdefault("payload_handle", None)
        data.setdefault("sampling_logprobs", [])
        data.setdefault("step_rewards", [])
        data["top_logprobs"] = [
            [tuple(pair) for pair in row] for row in data["top_logprobs"]
        ]
        data["tool_calls"] = [AgentToolCall(**call) for call in data["tool_calls"]]
    return kind, KINDS[kind](**data)


def validate(message: Any) -> None:
    """Verify a Python message against the authoritative Rust struct."""
    kind = type(message).__name__
    if kind not in KINDS:
        raise TypeError(f"not a megaRL contract message: {kind}")
    _native().validate(kind, json.dumps(asdict(message), separators=(",", ":")))


def protocol_version() -> int:
    return _native().protocol_version()


def kind_ids() -> dict[str, int]:
    return _native().kind_ids()
