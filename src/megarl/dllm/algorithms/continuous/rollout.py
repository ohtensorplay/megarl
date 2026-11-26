"""Flow-GRPO arXiv 2505.05470 and DiffNFT (ROLL diffusion pipeline
refs/ROLL/roll/pipeline/diffusion); see package docstring for details."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterator

from ....contracts import RolloutRequest, TrajectoryRecord
from ....rollout import RolloutService
from .config import FlowConfig
from .types import FlowTrajectoryStore

_ModelStepFn = Callable[[Any, Any, Any], Any]


class FlowTrajectoryRollout(RolloutService):
    """Persist continuous trajectories and emit wire-compatible records.

    The metadata frame carries a single sentinel token (the verifier scores the
    out-of-band latent, not decoded tokens) and the trajectory handle. Prompt
    provenance, group id and policy version stay on the record as usual.
    """

    def __init__(
        self,
        generator: Any,
        store: FlowTrajectoryStore | None = None,
        config: FlowConfig | None = None,
        *,
        version_provider: Callable[[], int] | None = None,
        weight_reloader: Callable[..., None] | None = None,
    ) -> None:
        self.generator = generator
        self.store = store or FlowTrajectoryStore()
        self.config = config or FlowConfig()
        self._version_provider = version_provider or (lambda: 0)
        self._weight_reloader = weight_reloader

    def generate(self, request: RolloutRequest) -> Iterator[TrajectoryRecord]:
        n = max(1, request.n)
        result = self.generator.generate(
            [list(request.prompt_token_ids)] * n, config=self.config
        )
        trajectories = list(result.trajectories)
        if len(trajectories) != n:
            raise RuntimeError(
                f"flow generator returned {len(trajectories)} trajectories for n={n}"
            )
        version = int(self._version_provider())
        for trajectory in trajectories:
            record = dataclasses.replace(trajectory, request_id=request.request_id)
            handle = self.store.put(record)
            yield TrajectoryRecord(
                request_id=request.request_id,
                prompt_len=len(request.prompt_token_ids),
                token_ids=[0],  # sentinel: the latent lives out of band
                logprobs=[],
                top_logprobs=[],
                policy_versions=[version],
                reward=None,
                finish_reason=0,
                payload_handle=handle,
            )

    def reload_weights(self, path: str, expected_digest: str | None = None) -> None:
        if self._weight_reloader is None:
            raise NotImplementedError("flow reload needs a weight_reloader")
        self._weight_reloader(path, expected_digest)
