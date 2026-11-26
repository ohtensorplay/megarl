"""dLLM reinforcement learning (d1 arXiv 2505.22765, wd1; see devdocs/dllm-rl.md); see
package docstring for details."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import tensorplay as tp

from .config import DLLMConfig

if TYPE_CHECKING:
    from .batching import DLLMTensorBatch


class DLLMModelAdapter(Protocol):
    """Maps ``(model, noisy_ids, tensors, config)`` to ``[B, L, V]`` logits."""

    def __call__(self, model: Any, noisy_ids: Any, tensors: DLLMTensorBatch, config: DLLMConfig) -> Any: ...


class FullAttentionAdapter:
    """LLaDA / Dream / MMaDA: one bidirectional pass over the noisy sequence."""

    def __call__(self, model, noisy_ids, tensors, config):
        output = model(noisy_ids)
        logits = getattr(output, "logits", output)
        if not hasattr(logits, "shape") or len(logits.shape) != 3:
            raise ValueError("full-attention dLLM must return [B, L, V] logits")
        return logits


class BlockDiffusionAdapter:
    """DiffusionGemma / SDAR: encode the prompt, denoise the canvas once.

    Expects a model exposing ``allocate_cache(capacity)``, ``encode(ids, cache)``
    and ``denoise(canvas, caches, self_conditioning_logits=None)`` — the same
    interface as ``megalm.models.diffusiongemma.DiffusionGemmaForBlockDiffusion``.
    """

    def __call__(self, model, noisy_ids, tensors, config):
        batch, width = noisy_ids.shape
        span = tensors.dllm_prompt_span
        if span <= 0 or span >= width:
            raise ValueError("block dLLM batch needs a prompt span and a canvas")
        canvas = noisy_ids[:, span:]
        caches = []
        for row in range(batch):
            prompt_len = int(tensors.dllm_prompt_lens[row].flatten()[0])
            if prompt_len <= 0 or prompt_len > span:
                raise ValueError("block dLLM prompt length outside the prompt span")
            cache = model.allocate_cache(prompt_len + canvas.shape[1] + 1)
            model.encode(noisy_ids[row, :prompt_len], cache)
            caches.append(cache)
        canvas_logits = model.denoise(canvas, caches, self_conditioning_logits=None)
        zeros = tp.zeros(
            (batch, span, canvas_logits.shape[-1]),
            dtype=canvas_logits.dtype,
            device=canvas_logits.device,
        )
        return tp.cat([zeros, canvas_logits], dim=1)


def default_adapter(config: DLLMConfig) -> DLLMModelAdapter:
    return FullAttentionAdapter() if config.family == "masked" else BlockDiffusionAdapter()
