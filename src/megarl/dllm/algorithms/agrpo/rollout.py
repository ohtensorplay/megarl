"""AGRPO / Amortized GRPO, Anthony Zhan, ICML 2026 arXiv 2510.04019; code
https://github.com/probablyabot/agrpo; see package docstring for details."""

from __future__ import annotations

import bisect
import dataclasses
import random
from typing import Any, Callable, Iterator, Sequence

import tensorplay as tp
import tensorplay.nn.functional as F

from ....contracts import RolloutRequest, TrajectoryRecord
from ....dllm import default_adapter
from ....rollout import RolloutService
from .config import AGRPOConfig
from .types import AGRPOGenerationResult, AGRPORecord, AGRPOStore


class _GenerationTensors:
    def __init__(self, x, prompt_lens: Sequence[int], span: int) -> None:
        self.dllm_noisy_ids = x
        self.dllm_prompt_lens = tp.tensor(list(prompt_lens), dtype=tp.long, device=x.device)
        self.dllm_prompt_span = int(span)
        self.dllm_canvas_len = int(x.shape[1]) - int(span)
        self.dllm_alpha = tp.ones((int(x.shape[0]),), dtype=tp.float32, device=x.device)


class AGRPOGenerator:
    """Semi-AR masked diffusion sampler with exact stepwise statistics.

    Mirrors the official generator: Gumbel/multinomial sampling at temperature
    ``T``, random or confidence-based selection of ``n/m`` positions per step,
    and a record of ``unmask_steps``, sampled-token log-probs and entropies.
    """

    def __init__(self, model: Any, adapter: Any | None = None) -> None:
        self.model = model
        self.adapter = adapter

    @tp.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        config: AGRPOConfig | None = None,
    ) -> AGRPOGenerationResult:
        cfg = config or AGRPOConfig()
        if not prompts:
            raise ValueError("at least one prompt is required")
        batch = len(prompts)
        width_prompt = max(
            cfg.max_prompt_len or 0, max(len(p) for p in prompts)
        )
        C = cfg.max_completion_length
        BL = cfg.block_length
        UM = cfg.tokens_per_step
        MASK = cfg.mask_token_id
        eos = set(cfg.eos_token_ids)
        device = getattr(self.model, "device", "cpu")
        fill = cfg.prompt_fill()
        padded = [list(p) + [fill] * (width_prompt - len(p)) for p in prompts]
        prompt_tensor = tp.tensor(padded, dtype=tp.long, device=device)
        x = tp.cat(
            [prompt_tensor, tp.full((batch, C), MASK, dtype=tp.long, device=device)],
            dim=1,
        )
        prompt_lens = [len(p) for p in prompts]
        steps = [[-1] * C for _ in range(batch)]
        logps = [[0.0] * C for _ in range(batch)]
        entropies = [[0.0] * C for _ in range(batch)]
        eos_rows = [[False] * C for _ in range(batch)]
        forward = self.adapter or default_adapter(cfg)
        step = 0
        row_idx = tp.arange(batch, device=device).unsqueeze(1)

        for block in range(C // BL):
            start, stop = block * BL, (block + 1) * BL
            finished = [all(eos_rows[r][start - BL : start]) for r in range(batch)] if block else [
                False
            ] * batch
            for row, done in enumerate(finished):
                if not done:
                    continue
                eos_token = next(iter(eos))
                for pos in range(start, C):
                    x[row, width_prompt + pos] = eos_token
                    steps[row][pos] = step
                    eos_rows[row][pos] = True
            while step < (block + 1) * BL // UM:
                if cfg.family == "block":
                    gen = _GenerationTensors(x, prompt_lens, width_prompt)
                    logits = forward(self.model, x, gen, cfg)[
                        :, width_prompt + start : width_prompt + stop, :
                    ]
                else:
                    logits = forward(
                        self.model, x, _GenerationTensors(x, prompt_lens, 0), cfg
                    )[:, width_prompt + start : width_prompt + stop, :]
                logits_t = logits.float() / cfg.temperature
                log_probs = F.log_softmax(logits_t, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)
                sampled = tp.multinomial(
                    probs.reshape(-1, probs.shape[-1]), 1
                ).view(batch, BL)
                sampled_logp = tp.gather(
                    log_probs, -1, sampled.unsqueeze(-1)
                ).squeeze(-1)
                scores = (
                    tp.rand_like(sampled_logp)
                    if cfg.unmasking == "random"
                    else sampled_logp
                )
                already = tp.tensor(
                    [[steps[r][start + c] != -1 for c in range(BL)] for r in range(batch)],
                    dtype=tp.float32,
                    device=device,
                )
                scores = tp.where(already > 0, tp.full_like(scores, -1e9), scores)
                for row, done in enumerate(finished):
                    if done:
                        scores[row] = tp.full_like(scores[row], -1e9)
                _, indices = tp.topk(scores, k=UM, dim=-1)
                x[row_idx, width_prompt + start + indices] = sampled.gather(-1, indices)
                for row in range(batch):
                    for col in indices[row].tolist():
                        pos = start + int(col)
                        if steps[row][pos] != -1:
                            continue
                        steps[row][pos] = step
                        logps[row][pos] = float(sampled_logp[row, int(col)].flatten()[0])
                        entropies[row][pos] = float(entropy[row, int(col)].flatten()[0])
                        eos_rows[row][pos] = int(sampled[row, int(col)].flatten()[0]) in eos
                step += 1

        sequences: list[list[int]] = []
        records: list[AGRPORecord] = []
        for row, prompt in enumerate(prompts):
            tokens = [int(v) for v in x[row, width_prompt:].tolist()]
            cut = next((i for i, tok in enumerate(tokens) if tok in eos), None)
            display = tokens[: cut + 1] if cut is not None else tokens
            sequences.append(list(prompt) + display)
            records.append(
                AGRPORecord(
                    request_id=0,
                    prompt_ids=list(prompt),
                    completion_ids=tokens,
                    unmask_steps=steps[row],
                    token_logprobs=logps[row],
                    entropies=entropies[row],
                    version=cfg.version,
                )
            )
        return AGRPOGenerationResult(sequences, tuple(records))


class AGRPORollout(RolloutService):
    """Wrap :class:`AGRPOGenerator` in megaRL's ``RolloutService`` contract."""

    def __init__(
        self,
        generator: Any,
        store: AGRPOStore | None = None,
        config: AGRPOConfig | None = None,
        *,
        version_provider: Callable[[], int] | None = None,
        weight_reloader: Callable[[str, str | None], None] | None = None,
    ) -> None:
        self.generator = generator
        self.store = store or AGRPOStore()
        self.config = config or AGRPOConfig()
        self._version_provider = version_provider or (lambda: 0)
        self._weight_reloader = weight_reloader

    def generate(self, request: RolloutRequest) -> Iterator[TrajectoryRecord]:
        prompt = list(request.prompt_token_ids)
        if not prompt:
            raise ValueError("AGRPO rollout needs a non-empty prompt")
        n = max(1, request.n)
        version = int(self._version_provider())
        cfg = dataclasses.replace(self.config, version=version)
        result = self.generator.generate([prompt] * n, config=cfg)
        records = list(result.records)
        if len(records) != n:
            raise RuntimeError(f"AGRPO generator returned {len(records)} records for n={n}")
        for record in records:
            record = dataclasses.replace(
                record, request_id=request.request_id, group_id=request.request_id
            )
            handle = self.store.put(record)
            yield TrajectoryRecord(
                request_id=request.request_id,
                prompt_len=len(prompt),
                token_ids=list(record.completion_ids),
                logprobs=[0.0] * len(record.completion_ids),
                top_logprobs=[],
                policy_versions=[version] * len(record.completion_ids),
                reward=None,
                finish_reason=0,
                payload_handle=handle,
            )

    def reload_weights(self, path: str, expected_digest: str | None = None) -> None:
        if self._weight_reloader is None:
            raise NotImplementedError("AGRPO weight reload needs weight_reloader")
        self._weight_reloader(path, expected_digest)


def sample_mc_steps(
    records: Sequence[AGRPORecord], config: AGRPOConfig
) -> tuple[list[int], list[float]]:
    """Return flattened ``(mc_steps, importance_weights)`` for all samples.

    ``mc_steps`` has ``B * k`` entries. Uniform sampling uses anti-correlated
    uniforms when ``low_discrepancy`` is set; entropy sampling builds a
    per-step categorical from the recorded entropy mass and corrects each draw
    by ``1 / ((last+1) * p_t)`` so the estimator stays unbiased.
    """
    B, k, m = len(records), config.mc_samples, config.sampling_steps
    S = B * k
    rng = random.Random(config.seed + 1000 * config.version)
    permutation = list(range(S))
    rng.shuffle(permutation)
    uniforms = [(permutation[i] + rng.random()) / S for i in range(S)]
    steps: list[int] = []
    weights: list[float] = []
    for index, record in enumerate(records):
        valid = [s for s in record.unmask_steps if s >= 0]
        last = max(valid) if valid else 0
        if config.importance_sampling == "entropy":
            mass = [1e-8] * m
            for position, step in enumerate(record.unmask_steps):
                if step >= 0:
                    mass[step] += max(0.0, record.entropies[position])
            for step in range(last + 1, m):
                mass[step] = 0.0
            total = sum(mass)
            probs = [value / total for value in mass]
            cdf: list[float] = []
            running = 0.0
            for prob in probs:
                running += prob
                cdf.append(running)
        for j in range(k):
            u = uniforms[index * k + j]
            if config.importance_sampling == "entropy":
                step = min(last, bisect.bisect_right(cdf, u))
                p = max(probs[step], 1e-12)
                weight = 1.0 / ((last + 1) * p)
            elif config.low_discrepancy:
                step = min(last, int(u * (last + 1)))
                weight = 1.0
            else:
                draw = rng.random()
                step = min(last, int(draw * (last + 1)))
                weight = 1.0
            steps.append(step)
            weights.append(weight)
    return steps, weights
