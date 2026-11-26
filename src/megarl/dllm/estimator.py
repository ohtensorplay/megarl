"""dLLM reinforcement learning (d1 arXiv 2505.22765, wd1; see devdocs/dllm-rl.md); see
package docstring for details."""

from __future__ import annotations

import dataclasses
import random
from typing import TYPE_CHECKING, Any, Sequence

import tensorplay as tp
import tensorplay.nn.functional as F

from ..contracts import TrajectoryRecord
from .config import DLLMConfig

if TYPE_CHECKING:
    from .batching import DLLMTensorBatch
from .adapters import DLLMModelAdapter, default_adapter


@dataclasses.dataclass(frozen=True, slots=True)
class DLLMSample:
    """One diffusion completion: prompt/canvas tokens and its group reward."""

    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    reward: float
    version: int = 0
    mask_seed: int | None = None
    payload_handle: int | None = None
    """Out-of-band handle for trajectory replay; None on the one-step path."""

    @property
    def length(self) -> int:
        return len(self.prompt_token_ids) + len(self.completion_token_ids)

    def validate(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("dLLM sample needs a non-empty prompt")
        if not self.completion_token_ids:
            raise ValueError("dLLM sample needs a non-empty completion")
        if self.version < 0:
            raise ValueError("policy version cannot be negative")


def from_dllm_record(
    record: TrajectoryRecord,
    prompt_token_ids: Sequence[int],
    *,
    reward: float | None = None,
    mask_seed: int | None = None,
) -> DLLMSample:
    """Adapt the existing wire record to a dLLM sample.

    ``record.token_ids`` is the response span, exactly as for AR rollouts. Its
    ``logprobs`` are ignored: a dLLM estimator recomputes them from tokens.
    """
    if record.reward is None and reward is None:
        raise ValueError("dLLM trajectory has no reward; score it before batching")
    sample = DLLMSample(
        prompt_token_ids=list(prompt_token_ids),
        completion_token_ids=list(record.token_ids),
        reward=float(reward if reward is not None else record.reward),
        version=int(record.policy_versions[0]) if record.policy_versions else 0,
        mask_seed=mask_seed,
        payload_handle=record.payload_handle,
    )
    sample.validate()
    return sample


def _noise_row(
    prompt: list[int],
    completion: list[int],
    config: DLLMConfig,
    seed: int,
    *,
    prompt_span: int,
    canvas_len: int,
) -> tuple[list[int], list[int], int, float]:
    """Build one row and its noisy canvas.

    Returns ``(ids, noisy_ids, comp_len, alpha)`` where ``alpha`` is the
    forward-process keep probability used by the ``mean_loo`` correction.
    Block diffusion uses uniform random-token corruption with a per-row rate
    ``t ~ U(eps, 1-eps)`` (trl/DiffusionGemma convention); masked dLLMs use
    the d1 mask token.
    """
    rng = random.Random(seed)
    uniform = config.effective_noise_mode == "uniform"
    if uniform:
        assert config.vocab_size is not None
        corruption = config.corruption_eps + (
            1.0 - 2.0 * config.corruption_eps
        ) * rng.random()
        alpha = 1.0 - corruption
    else:
        alpha = 1.0 - config.prompt_mask_prob

    if config.family == "block":
        padded_prompt = list(prompt) + [config.noise_token_id] * (
            prompt_span - len(prompt)
        )
        canvas = list(completion) + [config.noise_token_id] * (
            canvas_len - len(completion)
        )
        ids = padded_prompt + canvas
        noisy_prompt = [
            config.noise_token_id
            if rng.random() < config.prompt_mask_prob
            else token
            for token in padded_prompt
        ]
        if uniform:
            noisy_canvas = [
                token
                if rng.random() < alpha
                else rng.randrange(config.vocab_size)
                for token in canvas
            ]
        else:
            noisy_canvas = [config.noise_token_id] * canvas_len
        return ids, noisy_prompt + noisy_canvas, len(completion), alpha

    ids = list(prompt) + list(completion)
    noisy = [
        config.noise_token_id if rng.random() < config.prompt_mask_prob else token
        for token in prompt
    ]
    if uniform:
        noisy.extend(
            token if rng.random() < alpha else rng.randrange(config.vocab_size)
            for token in completion
        )
    else:
        noisy.extend(config.noise_token_id for _ in completion)
    return ids, noisy, len(completion), alpha


def _loo_correction(logits, noisy_ids, alpha, config: DLLMConfig):
    """Convert a leave-one-out posterior back to the plain denoiser.

    DiffusionGemma's plug-in softmax may parameterise the LOO posterior; the
    released checkpoint defaults to ``mean``. When ``mean_loo`` is selected,
    add ``log(1 + V·α/(1-α))`` to the observed token's logit before gathering
    (paper 2605.22765, as implemented in trl's DiffusionGemma SFT example).
    ``alpha`` is the per-row forward keep probability, so block diffusion's
    corruption-rate schedule is respected instead of the prompt-mask rate.
    """
    if config.parameterization != "mean_loo":
        return logits
    vocab = float(logits.shape[-1])
    keep = alpha.clamp(min=1e-6, max=1.0 - 1e-6)
    bump = tp.log(1.0 + vocab * keep / (1.0 - keep))
    observed = (noisy_ids != config.noise_token_id).unsqueeze(-1)
    additive = tp.where(
        observed,
        bump.unsqueeze(-1).unsqueeze(-1),
        tp.zeros(logits.shape, dtype=logits.dtype, device=logits.device),
    )
    return logits.float() + additive


def dllm_logprobs(
    model: Any,
    tensors: DLLMTensorBatch,
    config: DLLMConfig,
    *,
    adapter: DLLMModelAdapter | None = None,
) -> Any:
    """One-step per-token log-probabilities, aligned with ``input_ids``.

    This is d1's estimator: one forward over ``dllm_noisy_ids`` and a gather of
    the final token at every completion position. Prompt/pad positions are
    ignored by the loss mask, so their values are never trained on.
    """
    forward = adapter or default_adapter(config)
    logits = forward(model, tensors.dllm_noisy_ids, tensors, config)
    logits = _loo_correction(
        logits, tensors.dllm_noisy_ids, tensors.dllm_alpha, config
    )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return tp.gather(
        log_probs, -1, tensors.input_ids.unsqueeze(-1)
    ).squeeze(-1)
