"""PURL controller — When Should Diffusion Language Models Reconsider Their Commitments?
(Zheng et al., arXiv:2611.11551v1; withdrawn after v1).
"""

from __future__ import annotations

from ....async_controller import AsyncController
from ....controller import Controller
from ....dataplane import InProcessPayloadStore
from ....dllm import DLLMPayloadQueue, from_dllm_record
from .batching import make_purl_packer
from .config import PURLConfig
from .types import PURLStore


class PURLController(Controller):
    """Synchronous controller for PURL joint training."""

    def __init__(self, *args, purl_config: PURLConfig, store: PURLStore, **kwargs) -> None:
        controller_config = kwargs.get("config") or (args[0] if args else None)
        if getattr(controller_config, "dpo", False):
            raise ValueError("DPO is not supported for PURL controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_purl_packer(purl_config, store))
        super().__init__(*args, **kwargs)


class AsyncPURLController(AsyncController):
    """Asynchronous controller with a handle-preserving payload queue."""

    def __init__(self, *args, purl_config: PURLConfig, store: PURLStore, **kwargs) -> None:
        config = kwargs.get("config") or (args[0] if args else None)
        if getattr(config, "dpo", False):
            raise ValueError("DPO is not supported for PURL controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_purl_packer(purl_config, store))
        kwargs.setdefault(
            "queue",
            DLLMPayloadQueue(
                InProcessPayloadStore(),
                ttl_secs=float(getattr(config, "lease_ttl_secs", 300.0)),
            ),
        )
        super().__init__(*args, **kwargs)
