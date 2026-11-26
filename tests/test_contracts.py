"""Wire contract conformance: Python mirror <-> authoritative Rust structs."""

import dataclasses

import pytest

import megarl
from megarl import contracts


def sampling(**overrides):
    values = dict(
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        seed=None,
        logprobs=True,
        top_logprobs=0,
    )
    values.update(overrides)
    return contracts.SamplingParams(**values)


MESSAGES = [
    contracts.RolloutRequest(
        request_id=42,
        policy_version=7,
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=64,
        n=4,
        sampling=sampling(),
        session_id=None,
    ),
    contracts.TrajectoryRecord(
        request_id=42,
        prompt_len=3,
        token_ids=[11, 12, 13],
        logprobs=[-0.1, -0.2, -0.3],
        top_logprobs=[[(11, -0.1)], [(12, -0.2)], [(13, -0.3)]],
        # Mixed-version prefix: an in-flight weight update landed mid-generation.
        policy_versions=[6, 6, 7],
        reward=1.0,
        finish_reason=0,
        payload_handle=None,
    ),
    contracts.WeightUpdate(
        from_version=6,
        to_version=7,
        transport="shm",
        digest="a" * 64,
        tensors=[
            contracts.TensorDescriptor(
                name="model.layers.0.attn.q.weight",
                shape=[4096, 4096],
                dtype="bfloat16",
                shard_rank=0,
                byte_offset=0,
                byte_length=33_554_432,
            )
        ],
        abort_inflight=True,
    ),
    contracts.RewardOutcome(
        request_id=42,
        reward=1.0,
        domain="math",
        metadata={"verifier": "math-rule-v1"},
    ),
    contracts.MultimodalTrajectory(
        request_id=42,
        prompt_len=4,
        token_ids=[11, 12],
        logprobs=[-0.1, -0.2],
        policy_versions=[6, 7],
        image_token_ids=[100, 101],
        image_insert_at=2,
        image_grid=(2, 1),
        modality_mask=[0, 0, 1, 1, 0, 0],
        reward=1.0,
        finish_reason=0,
        payload_handle=None,
    ),
    contracts.HealthCheck(probe=42, ok=True),
    contracts.AgentTrajectory(
        request_id=42,
        policy_version=7,
        prompt_len=2,
        token_ids=[1, 2, 10, 20, 11, 12],
        logprobs=[-0.1, 0.0, -0.2, -0.3],
        top_logprobs=[],
        policy_versions=[7, 7, 8, 8],
        token_roles=[0, 0, 1, 2, 1, 1],
        turn_ids=[0, 0, 0, 1, 1, 1],
        loss_mask=[0, 0, 1, 0, 1, 1],
        reward=1.0,
        finish_reason=0,
        payload_handle=None,
        sampling_logprobs=[-0.1, 0.0, -0.25, -0.3],
        tool_calls=[
            contracts.AgentToolCall(
                id="c1",
                name="calculator",
                arguments_json='{"expr":"1+2"}',
                turn=0,
                ok=True,
                elapsed_ms=3,
            )
        ],
        step_rewards=[0.0, 1.0],
    ),
]


@pytest.mark.parametrize("message", MESSAGES, ids=lambda m: type(m).__name__)
def test_roundtrip(message):
    contracts.validate(message)  # catches field drift against the Rust struct
    frame = contracts.encode(message)
    assert frame[:4] == contracts.MAGIC
    kind, decoded = contracts.decode(frame)
    assert kind == type(message).__name__
    assert decoded == message


def test_protocol_versions_agree():
    assert contracts.PROTOCOL_VERSION == contracts.protocol_version() == 1
    assert contracts.kind_ids() == {
        "RolloutRequest": 1,
        "TrajectoryRecord": 2,
        "WeightUpdate": 3,
        "RewardOutcome": 4,
        "MultimodalTrajectory": 5,
        "HealthCheck": 6,
        "AgentTrajectory": 7,
        "LifecycleCommand": 8,
        "LifecycleAck": 9,
        "WeightUpdateAck": 10,
    }


def test_rejects_version_mismatch():
    frame = bytearray(contracts.encode(MESSAGES[1]))
    frame[4] = 9  # little-endian u16 low byte
    frame[5] = 0
    with pytest.raises(ValueError, match="unsupported protocol version"):
        contracts.decode(bytes(frame))


def test_rejects_unknown_message_type():
    @dataclasses.dataclass(frozen=True)
    class NotAContract:
        x: int

    with pytest.raises(TypeError, match="not a megaRL contract"):
        contracts.encode(NotAContract(1))


def test_agent_trajectory_rejects_trainable_tool_tokens():
    message = dataclasses.replace(MESSAGES[-1], loss_mask=[0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="tool observation token 3"):
        contracts.validate(message)


def test_agent_trajectory_rejects_misaligned_response_vectors():
    message = dataclasses.replace(MESSAGES[-1], policy_versions=[7])
    with pytest.raises(ValueError, match="agent trajectory invariant"):
        contracts.validate(message)
