"""Trajectory-level dLLM rollouts and replay (TraceRL ICLR 2026 arXiv 2509.06949;
MaskGRPO arXiv 2510.02880; CJ-GRPO arXiv 2509.23924); see package docstring for
details."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterator

from ...contracts import RolloutRequest, TrajectoryRecord
from ...rollout import RolloutService
from .store import DLLMTrajectoryStore
from .types import TrajectoryGenerationConfig


class TrajectoryDLLMRollout(RolloutService):
    """Rollout service that persists trajectories out of band.

    Emits the same ``TrajectoryRecord`` as the one-step path; the only addition
    is ``payload_handle``, which the trajectory-aware trainer resolves through
    the shared store. The Rust wire contract is unchanged.
    """

    def __init__(
        self,
        generator: Any,
        store: DLLMTrajectoryStore | None = None,
        config: TrajectoryGenerationConfig | None = None,
        *,
        version_provider=None,
        weight_reloader=None,
    ) -> None:
        self.generator = generator
        self.store = store or DLLMTrajectoryStore()
        self.config = config or TrajectoryGenerationConfig()
        self._version_provider = version_provider or (lambda: 0)
        self._weight_reloader = weight_reloader

    def generate(self, request: RolloutRequest) -> Iterator[TrajectoryRecord]:
        config = dataclasses.replace(
            self.config, version=int(self._version_provider())
        )
        result = self.generator.generate(
            [list(request.prompt_token_ids)] * max(1, request.n), config=config
        )
        trajectories = list(result.trajectories)
        if len(trajectories) != max(1, request.n):
            raise RuntimeError(
                f"dLLM generator returned {len(trajectories)} trajectories "
                f"for n={request.n}"
            )
        for trajectory in trajectories:
            record = dataclasses.replace(
                trajectory, request_id=request.request_id
            )
            handle = self.store.put(record)
            yield TrajectoryRecord(
                request_id=request.request_id,
                prompt_len=len(record.prompt_ids),
                token_ids=list(record.completion_ids),
                logprobs=[0.0] * len(record.completion_ids),
                top_logprobs=[],
                policy_versions=[config.version] * len(record.completion_ids),
                reward=None,
                finish_reason=0,
                payload_handle=handle,
            )

    def reload_weights(self, path: str, expected_digest: str | None = None) -> None:
        if self._weight_reloader is None:
            raise NotImplementedError("dLLM trajectory reload needs weight_reloader")
        self._weight_reloader(path, expected_digest)
