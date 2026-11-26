"""dLLM rollout adapter for the MegaLM generator (keeps the Rust wire contract
unchanged); see package docstring for details."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any, Callable, Iterator

from ..contracts import RolloutRequest, TrajectoryRecord
from ..rollout import RolloutService


@dataclasses.dataclass(frozen=True, slots=True)
class DLLMRolloutConfig:
    """Generation knobs forwarded to the megaLM diffusion sampler."""

    max_new_tokens: int = 256
    max_denoising_steps: int = 48
    entropy_bound: float = 0.1
    t_min: float = 0.4
    t_max: float = 0.8
    stability_threshold: int = 1
    confidence_threshold: float = 0.005
    eos_token_ids: tuple[int, ...] = (1,)
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0 or self.max_denoising_steps <= 0:
            raise ValueError("max_new_tokens and max_denoising_steps must be positive")
        if not 0 < self.t_min <= self.t_max:
            raise ValueError("temperature schedule must satisfy 0 < t_min <= t_max")
        if self.entropy_bound <= 0 or self.confidence_threshold <= 0:
            raise ValueError("entropy_bound and confidence_threshold must be positive")


class DLLMRollout(RolloutService):
    """Adapter from a MegaLM-style dLLM generator to ``RolloutService``.

    ``sampling_config`` may be a ``DiffusionSamplingConfig`` instance from
    megaLM or any object with the same fields; importing megaLM lazily keeps
    the base install (CPU unit tests) free of the rollout extra.
    """

    def __init__(
        self,
        generator: Any,
        config: DLLMRolloutConfig | None = None,
        *,
        sampling_config: Any | None = None,
        version_provider: Callable[[], int] | None = None,
        weight_reloader: Callable[[str, str | None], None] | None = None,
    ) -> None:
        self.generator = generator
        self.config = config or DLLMRolloutConfig()
        self._sampling_config = sampling_config
        self._version_provider = version_provider or (lambda: 0)
        self._weight_reloader = weight_reloader

    def _sampling(self, request: RolloutRequest) -> Any:
        cfg = self.config
        kwargs = dict(
            max_new_tokens=min(request.max_new_tokens, cfg.max_new_tokens),
            max_denoising_steps=cfg.max_denoising_steps,
            entropy_bound=cfg.entropy_bound,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            stability_threshold=cfg.stability_threshold,
            confidence_threshold=cfg.confidence_threshold,
            eos_token_ids=cfg.eos_token_ids,
            pad_token_id=cfg.pad_token_id,
            seed=request.sampling.seed,
        )
        if self._sampling_config is not None:
            return dataclasses.replace(self._sampling_config, **kwargs)
        return _diffusion_sampling_config(**kwargs)

    def generate(self, request: RolloutRequest) -> Iterator[TrajectoryRecord]:
        prompt = list(request.prompt_token_ids)
        if not prompt:
            raise ValueError("dLLM rollout needs a non-empty prompt")
        sampling = self._sampling(request)
        result = self.generator.generate(
            [prompt] * max(1, request.n), config=sampling
        )
        sequences = list(result.sequences)
        if len(sequences) != max(1, request.n):
            raise RuntimeError(
                f"dLLM generator returned {len(sequences)} sequences for "
                f"n={request.n}"
            )
        version = int(self._version_provider())
        for sequence in sequences:
            tokens = [int(token) for token in sequence]
            if len(tokens) < len(prompt) or tokens[: len(prompt)] != prompt:
                raise RuntimeError(
                    "dLLM generator changed the prompt prefix; refusing to "
                    "emit a trajectory with an unknown response span"
                )
            completion = tokens[len(prompt) :]
            if not completion:
                continue
            yield TrajectoryRecord(
                request_id=request.request_id,
                prompt_len=len(prompt),
                token_ids=completion,
                # A dLLM estimator recomputes behavior logprobs; the wire
                # field stays aligned but carries no sampled-token values.
                logprobs=[0.0] * len(completion),
                top_logprobs=[],
                policy_versions=[version] * len(completion),
                reward=None,  # controller's reward plane scores the group
                finish_reason=0,
                payload_handle=None,
            )

    def reload_weights(self, path: str, expected_digest: str | None = None) -> None:
        if self._weight_reloader is None:
            raise NotImplementedError(
                "dLLM weight reload needs a weight_reloader callback; the "
                "embedded generator does not own a weight-update API"
            )
        self._weight_reloader(path, expected_digest)


def _diffusion_sampling_config(**kwargs) -> Any:
    """Build megaLM's sampling config lazily, or a local stand-in on CPU."""
    try:
        from megalm.models.diffusiongemma import DiffusionSamplingConfig
    except Exception:  # pragma: no cover - optional rollout dependency
        return SimpleNamespace(**kwargs)
    return DiffusionSamplingConfig(**kwargs)
