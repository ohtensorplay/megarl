"""GDSD, Xiaohang Tang et al., arXiv 2605.29398; code https://github.com/GaryBall/GDSD;
see package docstring for details."""

from __future__ import annotations

from ....async_controller import AsyncController
from ....controller import Controller
from ....dataplane import InProcessPayloadStore
from ....dllm import DLLMPayloadQueue, from_dllm_record
from .batching import make_gdsd_packer
from .config import GDSDConfig


class GDSDController(Controller):
    """Synchronous controller for GDSD self-distillation."""

    def __init__(self, *args, gdsd_config: GDSDConfig, **kwargs) -> None:
        controller_config = kwargs.get("config") or (args[0] if args else None)
        if getattr(controller_config, "dpo", False):
            raise ValueError("DPO is not supported for GDSD controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_gdsd_packer(gdsd_config))
        super().__init__(*args, **kwargs)


class AsyncGDSDController(AsyncController):
    """Asynchronous controller with a handle-preserving payload queue."""

    def __init__(self, *args, gdsd_config: GDSDConfig, **kwargs) -> None:
        config = kwargs.get("config") or (args[0] if args else None)
        if getattr(config, "dpo", False):
            raise ValueError("DPO is not supported for GDSD controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_gdsd_packer(gdsd_config))
        kwargs.setdefault(
            "queue",
            DLLMPayloadQueue(
                InProcessPayloadStore(),
                ttl_secs=float(getattr(config, "lease_ttl_secs", 300.0)),
            ),
        )
        super().__init__(*args, **kwargs)
