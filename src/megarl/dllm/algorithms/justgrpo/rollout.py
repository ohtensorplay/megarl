"""JustGRPO / The Flexibility Trap, Zanlin Ni et al., ICML 2026 arXiv 2601.15165; code
https://github.com/LeapLabTHU/JustGRPO; see package docstring for details."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterator, Sequence

import tensorplay as tp
import tensorplay.nn.functional as F

from ....contracts import RolloutRequest, TrajectoryRecord
from ....dllm import default_adapter
from ....rollout import RolloutService
from .config import JustGRPOConfig
from .types import JustGRPOGenerationResult, JustGRPORecord, JustGRPOStore


class _GenerationTensors:
    """Minimal adapter input for a block-diffusion generator."""

    def __init__(self, x, prompt_lens: Sequence[int], span: int) -> None:
        self.dllm_noisy_ids = x
        self.dllm_prompt_lens = tp.tensor(
            list(prompt_lens), dtype=tp.long, device=x.device
        )
        self.dllm_prompt_span = int(span)
        self.dllm_canvas_len = int(x.shape[1]) - int(span)
        self.dllm_alpha = tp.ones(
            (int(x.shape[0]),), dtype=tp.float32, device=x.device
        )


class JustGRPOGenerator:
    """Batched left-to-right dLLM sampler with exact step log-probs.

    At step ``t`` every row is forwarded with the prefix observed and the
    future masked; the token at ``prompt_width + t`` is sampled and its
    temperature-scaled log-probability and entropy are recorded.
    """

    def __init__(self, model: Any, adapter: Any | None = None) -> None:
        self.model = model
        self.adapter = adapter

    @tp.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        config: JustGRPOConfig | None = None,
    ) -> JustGRPOGenerationResult:
        cfg = config or JustGRPOConfig()
        if not prompts:
            raise ValueError("at least one prompt is required")
        batch = len(prompts)
        if cfg.max_prompt_len is None:
            prompt_width = max(len(p) for p in prompts)
        else:
            prompt_width = max(cfg.max_prompt_len, max(len(p) for p in prompts))
        L = cfg.max_new_tokens
        device = getattr(self.model, "device", "cpu")
        fill = cfg.prompt_fill()
        padded = [list(p) + [fill] * (prompt_width - len(p)) for p in prompts]
        prompt_tensor = tp.tensor(padded, dtype=tp.long, device=device)
        x = tp.cat(
            [
                prompt_tensor,
                tp.full((batch, L), cfg.noise_token_id, dtype=tp.long, device=device),
            ],
            dim=1,
        )
        prompt_lens = [len(p) for p in prompts]

        logps = tp.full((batch, L), 0.0, dtype=tp.float32, device=device)
        entropies = tp.full((batch, L), 0.0, dtype=tp.float32, device=device)
        forward = self.adapter or default_adapter(cfg)
        eos = set(cfg.eos_token_ids)
        for t in range(L):
            if cfg.family == "block":
                gen_tensors = _GenerationTensors(x, prompt_lens, prompt_width)
                logits = forward(self.model, x, gen_tensors, cfg)
            else:
                logits = forward(self.model, x, _GenerationTensors(x, prompt_lens, 0), cfg)
            logits_t = logits[:, prompt_width + t, :].float() / cfg.temperature
            log_probs = F.log_softmax(logits_t, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)
            sampled = tp.multinomial(probs, 1).squeeze(-1)
            sampled_logp = tp.gather(
                log_probs, -1, sampled.unsqueeze(-1)
            ).squeeze(-1)
            logps[:, t] = sampled_logp
            entropies[:, t] = entropy
            x[:, prompt_width + t] = sampled
            if cfg.family == "block":
                # The canvas is in place; cache is rebuilt on the next step.
                pass

        sequences: list[list[int]] = []
        records: list[JustGRPORecord] = []
        for row, prompt in enumerate(prompts):
            tokens = [int(v) for v in x[row, prompt_width:].tolist()]
            cut = next((i for i, tok in enumerate(tokens) if tok in eos), None)
            display = tokens[: cut + 1] if cut is not None else tokens
            sequences.append(list(prompt) + display)
            records.append(
                JustGRPORecord(
                    request_id=0,
                    prompt_ids=list(prompt),
                    completion_ids=tokens,
                    token_logprobs=[float(v) for v in logps[row].tolist()],
                    entropies=[float(v) for v in entropies[row].tolist()],
                    version=cfg.version,
                )
            )
        return JustGRPOGenerationResult(sequences, tuple(records))


class JustGRPORollout(RolloutService):
    """Wrap :class:`JustGRPOGenerator` in megaRL's ``RolloutService`` contract."""

    def __init__(
        self,
        generator: Any,
        store: JustGRPOStore | None = None,
        config: JustGRPOConfig | None = None,
        *,
        version_provider: Callable[[], int] | None = None,
        weight_reloader: Callable[[str, str | None], None] | None = None,
    ) -> None:
        self.generator = generator
        self.store = store or JustGRPOStore()
        self.config = config or JustGRPOConfig()
        self._version_provider = version_provider or (lambda: 0)
        self._weight_reloader = weight_reloader

    def generate(self, request: RolloutRequest) -> Iterator[TrajectoryRecord]:
        prompt = list(request.prompt_token_ids)
        if not prompt:
            raise ValueError("JustGRPO rollout needs a non-empty prompt")
        n = max(1, request.n)
        version = int(self._version_provider())
        cfg = dataclasses.replace(self.config, version=version)
        result = self.generator.generate([prompt] * n, config=cfg)
        records = list(result.records)
        if len(records) != n:
            raise RuntimeError(f"JustGRPO generator returned {len(records)} records for n={n}")
        for record in records:
            record = dataclasses.replace(
                record,
                request_id=request.request_id,
                group_id=request.request_id,
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
            raise NotImplementedError("JustGRPO weight reload needs weight_reloader")
        self._weight_reloader(path, expected_digest)
