"""Length-aware RL batching.

Trajectories have variable prompt/response lengths. This module builds the two
things every policy-gradient loss needs: a loss mask that excludes prompt
tokens, and `cu_seqlens` for varlen attention. Batches are packed by token
budget with optional CP/TP alignment (ROLL notes `2*CP*TP` alignment).
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

from .contracts import AgentTrajectory, MultimodalTrajectory, TrajectoryRecord


@dataclasses.dataclass(frozen=True, slots=True)
class RLSample:
    prompt_len: int
    token_ids: list[int]  # prompt + response
    old_logprobs: list[float]  # response only, from the generating policy
    versions: list[int]  # response only
    reward: float
    # Sampling-engine logprobs (vLLM); empty when no mismatch info is available.
    sampling_logprobs: list[float] = dataclasses.field(default_factory=list, kw_only=True)
    # Optional token-level mask over the response span. When present it replaces
    # the default all-ones mask: agentic RL uses it to exclude tool observations
    # from the policy gradient while keeping them in the model's context.
    loss_mask: list[int] | None = dataclasses.field(default=None, kw_only=True)

    @property
    def response_len(self) -> int:
        return len(self.old_logprobs)

    @property
    def length(self) -> int:
        return len(self.token_ids)

    def validate(self) -> None:
        if self.prompt_len < 0 or self.response_len == 0:
            raise ValueError("sample needs a non-empty response")
        if self.length != self.prompt_len + self.response_len:
            raise ValueError("token_ids must be prompt + response")
        if len(self.versions) != self.response_len:
            raise ValueError("versions must align with response tokens")
        if self.sampling_logprobs and len(self.sampling_logprobs) != self.response_len:
            raise ValueError("sampling_logprobs must align with response tokens")
        if self.loss_mask is not None:
            if len(self.loss_mask) != self.response_len:
                raise ValueError("loss_mask must align with response tokens")
            if any(keep not in (0, 1) for keep in self.loss_mask):
                raise ValueError("loss_mask must be binary")


@dataclasses.dataclass(frozen=True, slots=True)
class RLBatch:
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    loss_mask: list[list[int]]
    position_ids: list[list[int]]
    old_logprobs: list[list[float]]
    versions: list[list[int]]
    rewards: list[float]
    cu_seqlens: list[int]
    max_len: int
    sampling_logprobs: list[list[float]] = dataclasses.field(default_factory=list, kw_only=True)

    @property
    def token_count(self) -> int:
        """Real (non-padding) tokens; cu_seqlens[-1] is their exact count."""
        return self.cu_seqlens[-1]

    @property
    def size(self) -> int:
        return len(self.input_ids)


def from_trajectory(
    record: TrajectoryRecord | AgentTrajectory,
    prompt_token_ids: Sequence[int],
    *,
    reward: float | None = None,
) -> RLSample:
    # Agent episodes are accepted through the same entry point so the
    # controllers need no agent-specific branch.
    if isinstance(record, AgentRecord):
        return from_agent_trajectory(
            record.trajectory, prompt_token_ids, reward=reward if reward is not None else record.reward
        )
    if isinstance(record, AgentTrajectory):
        return from_agent_trajectory(record, prompt_token_ids, reward=reward)
    if record.reward is None and reward is None:
        raise ValueError("trajectory has no reward; score it before batching")
    sample = RLSample(
        prompt_len=len(prompt_token_ids),
        token_ids=list(prompt_token_ids) + list(record.token_ids),
        old_logprobs=list(record.logprobs),
        versions=list(record.policy_versions),
        reward=float(reward if reward is not None else record.reward),
        sampling_logprobs=list(record.sampling_logprobs),
    )
    sample.validate()
    return sample


@dataclasses.dataclass(frozen=True, slots=True)
class AgentRecord:
    """Controller-facing view of an agent episode.

    The controller's version ledger counts ``len(record.token_ids)`` against
    ``record.policy_versions``, and ``score_group`` replaces ``reward`` on a
    copy. This view exposes only the *response* span as ``token_ids`` to keep
    that contract, while retaining the full episode for batching.
    """

    trajectory: AgentTrajectory
    reward: float | None = None
    payload_handle: int | None = None

    @property
    def request_id(self) -> int:
        return self.trajectory.request_id

    @property
    def prompt_len(self) -> int:
        return self.trajectory.prompt_len

    @property
    def token_ids(self) -> list[int]:
        return self.trajectory.token_ids[self.trajectory.prompt_len :]

    @property
    def logprobs(self) -> list[float]:
        return self.trajectory.logprobs

    @property
    def policy_versions(self) -> list[int]:
        return self.trajectory.policy_versions

    @property
    def sampling_logprobs(self) -> list[float]:
        return self.trajectory.sampling_logprobs

    @property
    def finish_reason(self) -> int:
        return self.trajectory.finish_reason

    @property
    def token_roles(self) -> list[int]:
        return self.trajectory.token_roles

    @property
    def loss_mask(self) -> list[int]:
        return self.trajectory.loss_mask

    @property
    def turn_ids(self) -> list[int]:
        return self.trajectory.turn_ids

    @property
    def tool_calls(self):
        return self.trajectory.tool_calls

    @property
    def step_rewards(self) -> list[float]:
        return self.trajectory.step_rewards


def from_agent_trajectory(
    record: AgentTrajectory,
    prompt_token_ids: Sequence[int] | None = None,
    *,
    reward: float | None = None,
) -> RLSample:
    """Flatten an agent episode into the ordinary RL sample shape.

    The response span covers every token after the initial prompt, including
    tool observations. Those tokens carry zero logprobs/versions and are
    excluded through the explicit ``loss_mask``; assistant tokens keep their
    sampled logprobs and per-token versions, so staleness masking still works.
    """
    if record.reward is None and reward is None:
        raise ValueError("agent trajectory has no reward; score it before batching")
    prompt_len = int(record.prompt_len)
    if prompt_token_ids is not None and len(prompt_token_ids) != prompt_len:
        raise ValueError("prompt_token_ids do not match the agent prompt_len")
    if prompt_len > len(record.token_ids):
        raise ValueError("agent prompt_len exceeds the full sequence")
    if len(record.loss_mask) != len(record.token_ids):
        raise ValueError("agent loss_mask is not aligned with token_ids")
    response_mask = list(record.loss_mask[prompt_len:])
    sample = RLSample(
        prompt_len=prompt_len,
        token_ids=list(record.token_ids),
        old_logprobs=list(record.logprobs),
        versions=list(record.policy_versions),
        reward=float(reward if reward is not None else record.reward),
        sampling_logprobs=list(record.sampling_logprobs),
        loss_mask=response_mask,
    )
    sample.validate()
    return sample


def from_record(
    record: TrajectoryRecord | AgentTrajectory,
    prompt_token_ids: Sequence[int],
    *,
    reward: float | None = None,
) -> RLSample:
    """Dispatch on record type; controllers use this single entry point."""
    if isinstance(record, AgentTrajectory):
        return from_agent_trajectory(record, prompt_token_ids, reward=reward)
    return from_trajectory(record, prompt_token_ids, reward=reward)


def collate(samples: Sequence[RLSample]) -> RLBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    for sample in samples:
        sample.validate()
    max_len = max(sample.length for sample in samples)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    loss_mask: list[list[int]] = []
    position_ids: list[list[int]] = []
    old_logprobs: list[list[float]] = []
    versions: list[list[int]] = []
    sampling_logprobs: list[list[float]] = []
    has_sampling = any(sample.sampling_logprobs for sample in samples)
    cu_seqlens = [0]

    for sample in samples:
        pad = max_len - sample.length
        input_ids.append(sample.token_ids + [0] * pad)
        attention_mask.append([1] * sample.length + [0] * pad)
        response_mask = (
            sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_len
        )
        loss_mask.append(
            [0] * sample.prompt_len + response_mask + [0] * pad
        )
        position_ids.append(list(range(sample.length)) + [0] * pad)
        # Align logprobs/versions to full sequence with zeros on prompt+pad;
        # the loss mask is what selects the response span.
        old_logprobs.append(
            [0.0] * sample.prompt_len + sample.old_logprobs + [0.0] * pad
        )
        versions.append([0] * sample.prompt_len + sample.versions + [0] * pad)
        if has_sampling:
            sampling_logprobs.append(
                [0.0] * sample.prompt_len + sample.sampling_logprobs + [0.0] * pad
            )
        cu_seqlens.append(cu_seqlens[-1] + sample.length)

    return RLBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        position_ids=position_ids,
        old_logprobs=old_logprobs,
        versions=versions,
        rewards=[sample.reward for sample in samples],
        cu_seqlens=cu_seqlens,
        max_len=max_len,
        sampling_logprobs=sampling_logprobs,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class MultimodalSample:
    prompt_token_ids: list[int]
    image_token_ids: list[int]
    image_insert_at: int
    response_token_ids: list[int]
    old_logprobs: list[float]
    versions: list[int]
    reward: float

    def compose(self) -> tuple[list[int], list[int], list[int]]:
        """Return (input_ids, modality_mask, loss_mask).

        Image tokens are spliced into the prompt at `image_insert_at`; they are
        part of the conditioning span, so the loss mask excludes them.
        """
        at = max(0, min(self.image_insert_at, len(self.prompt_token_ids)))
        image = list(self.image_token_ids)
        prompt = list(self.prompt_token_ids)
        response = list(self.response_token_ids)
        input_ids = prompt[:at] + image + prompt[at:] + response
        modality = (
            [0] * at
            + [1] * len(image)
            + [0] * (len(prompt) - at)
            + [0] * len(response)
        )
        loss_mask = [0] * (len(prompt) + len(image)) + [1] * len(response)
        if len(loss_mask) != len(input_ids):
            raise ValueError("modality composition produced misaligned masks")
        return input_ids, modality, loss_mask


@dataclasses.dataclass(frozen=True, slots=True)
class MultimodalBatch:
    input_ids: list[list[int]]
    modality_mask: list[list[int]]
    loss_mask: list[list[int]]
    old_logprobs: list[list[float]]
    versions: list[list[int]]
    cu_seqlens: list[int]
    max_len: int

    @property
    def size(self) -> int:
        return len(self.input_ids)


def from_multimodal(
    record: MultimodalTrajectory, prompt_token_ids: Sequence[int]
) -> MultimodalSample:
    if record.reward is None:
        raise ValueError("multimodal trajectory has no reward")
    return MultimodalSample(
        prompt_token_ids=list(prompt_token_ids),
        image_token_ids=list(record.image_token_ids),
        image_insert_at=int(record.image_insert_at),
        response_token_ids=list(record.token_ids),
        old_logprobs=list(record.logprobs),
        versions=list(record.policy_versions),
        reward=float(record.reward),
    )


def collate_multimodal(samples: Sequence[MultimodalSample]) -> MultimodalBatch:
    if not samples:
        raise ValueError("cannot collate an empty multimodal batch")
    max_len = 0
    composed = []
    for sample in samples:
        if len(sample.old_logprobs) != len(sample.response_token_ids):
            raise ValueError("multimodal logprobs misaligned with response")
        parts = sample.compose()
        composed.append(parts)
        max_len = max(max_len, len(parts[0]))
    input_ids, modality_mask, loss_mask, old_logprobs, versions = [], [], [], [], []
    cu_seqlens = [0]
    for (ids, modality, loss), sample in zip(composed, samples):
        pad = max_len - len(ids)
        total_prompt = len(ids) - len(sample.response_token_ids)
        input_ids.append(ids + [0] * pad)
        modality_mask.append(modality + [0] * pad)
        loss_mask.append(loss + [0] * pad)
        old_logprobs.append(
            [0.0] * total_prompt + sample.old_logprobs + [0.0] * pad
        )
        versions.append([0] * total_prompt + sample.versions + [0] * pad)
        cu_seqlens.append(cu_seqlens[-1] + len(ids))
    return MultimodalBatch(
        input_ids=input_ids,
        modality_mask=modality_mask,
        loss_mask=loss_mask,
        old_logprobs=old_logprobs,
        versions=versions,
        cu_seqlens=cu_seqlens,
        max_len=max_len,
    )


def _pack_indexed(
    indexed: Sequence[tuple[int, RLSample]],
    *,
    max_tokens: int,
    alignment: int = 1,
) -> list[list[tuple[int, RLSample]]]:
    """First-fit-decreasing token-budget packing over (original_index, sample)."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")

    ordered = sorted(indexed, key=lambda pair: pair[1].length, reverse=True)
    groups: list[list[tuple[int, RLSample]]] = []
    current: list[tuple[int, RLSample]] = []
    current_len = 0
    for index, sample in ordered:
        candidate_len = max(current_len, sample.length)
        aligned = ((candidate_len + alignment - 1) // alignment) * alignment
        if current and aligned * (len(current) + 1) > max_tokens:
            groups.append(current)
            current, current_len = [], 0
            candidate_len = sample.length
            aligned = ((candidate_len + alignment - 1) // alignment) * alignment
        if aligned * 1 > max_tokens:
            raise ValueError(
                f"sample of length {sample.length} exceeds aligned token budget"
            )
        current.append((index, sample))
        current_len = candidate_len
    if current:
        groups.append(current)
    return groups


def _pack_groups(
    samples: Sequence[RLSample], *, max_tokens: int, alignment: int = 1
) -> list[list[tuple[int, RLSample]]]:
    return _pack_indexed(
        list(enumerate(samples)), max_tokens=max_tokens, alignment=alignment
    )


def pack(
    samples: Sequence[RLSample],
    *,
    max_tokens: int,
    alignment: int = 1,
) -> list[RLBatch]:
    """First-fit-decreasing token-budget packing.

    Each sample occupies its own row (padded to the batch max), so the real
    token budget is `batch_size * max_len`. Alignment is applied to that product
    to satisfy CP/TP constraints. Use :func:`pack_with_order` when per-sample
    metadata (advantages, rewards) must follow the packed row order.
    """
    groups = _pack_groups(samples, max_tokens=max_tokens, alignment=alignment)
    return [collate([sample for _, sample in group]) for group in groups]


def collate_preference_pairs(
    pairs: Sequence[tuple[RLSample, RLSample]],
) -> tuple[RLBatch, RLBatch]:
    """Collate chosen and rejected rows into two aligned batches for DPO."""
    if not pairs:
        raise ValueError("cannot collate an empty preference batch")
    return collate([chosen for chosen, _ in pairs]), collate(
        [rejected for _, rejected in pairs]
    )


def pack_with_order(
    samples: Sequence[RLSample],
    *,
    max_tokens: int,
    alignment: int = 1,
) -> tuple[list[RLBatch], list[list[int]]]:
    """``pack`` plus the original sample index of every row in every batch.

    Packing sorts by length, so advantages computed before packing would
    otherwise be assigned to the wrong trajectories. The returned indices let
    the controller permute advantages identically.
    """
    groups = _pack_groups(samples, max_tokens=max_tokens, alignment=alignment)
    batches = [collate([sample for _, sample in group]) for group in groups]
    orders = [[index for index, _ in group] for group in groups]
    return batches, orders
