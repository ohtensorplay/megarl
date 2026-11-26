"""TIPS controller — TIPS: A Tiny Inference-time Policy for Scaling in Diffusion Language
Models (Zhang et al., arXiv:2610.22099; code https://github.com/zhengyz97/TIPS).
"""

from __future__ import annotations

from ....async_controller import AsyncController
from ....controller import Controller
from ....dataplane import InProcessPayloadStore
from ....dllm import DLLMPayloadQueue, from_dllm_record
from .batching import make_tips_packer
from .config import TIPSConfig
from .types import TIPSStore


class TIPSController(Controller):
    """Synchronous controller for TIPS visibility-policy training."""

    def __init__(self, *args, tips_config: TIPSConfig, store: TIPSStore, **kwargs) -> None:
        controller_config = kwargs.get("config") or (args[0] if args else None)
        if getattr(controller_config, "dpo", False):
            raise ValueError("DPO is not supported for TIPS controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_tips_packer(tips_config, store))
        super().__init__(*args, **kwargs)


class AsyncTIPSController(AsyncController):
    """Asynchronous controller with a handle-preserving payload queue."""

    def __init__(self, *args, tips_config: TIPSConfig, store: TIPSStore, **kwargs) -> None:
        config = kwargs.get("config") or (args[0] if args else None)
        if getattr(config, "dpo", False):
            raise ValueError("DPO is not supported for TIPS controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_tips_packer(tips_config, store))
        kwargs.setdefault(
            "queue",
            DLLMPayloadQueue(
                InProcessPayloadStore(),
                ttl_secs=float(getattr(config, "lease_ttl_secs", 300.0)),
            ),
        )
        super().__init__(*args, **kwargs)
