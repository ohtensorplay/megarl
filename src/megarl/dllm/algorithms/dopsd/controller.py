"""dOPSD, Phuong Tuan Dat, Qi Li, Xinchao Wang, arXiv 2607.04428; code
https://github.com/tuandattt/dOPSD; see package docstring for details."""

from __future__ import annotations

from ....async_controller import AsyncController
from ....controller import Controller
from ....dataplane import InProcessPayloadStore
from ....dllm import DLLMPayloadQueue, from_dllm_record
from .batching import make_dopsd_packer
from .config import DOPSDConfig
from .types import DOPSDStore


class DOPSDController(Controller):
    """Synchronous controller for trajectory self-distillation."""

    def __init__(self, *args, dopsd_config: DOPSDConfig, store: DOPSDStore, **kwargs) -> None:
        controller_config = kwargs.get("config") or (args[0] if args else None)
        if getattr(controller_config, "dpo", False):
            raise ValueError("DPO is not supported for dOPSD controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_dopsd_packer(dopsd_config, store))
        super().__init__(*args, **kwargs)


class AsyncDOPSDController(AsyncController):
    """Asynchronous controller with a handle-preserving payload queue."""

    def __init__(self, *args, dopsd_config: DOPSDConfig, store: DOPSDStore, **kwargs) -> None:
        config = kwargs.get("config") or (args[0] if args else None)
        if getattr(config, "dpo", False):
            raise ValueError("DPO is not supported for dOPSD controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_dopsd_packer(dopsd_config, store))
        kwargs.setdefault(
            "queue",
            DLLMPayloadQueue(
                InProcessPayloadStore(),
                ttl_secs=float(getattr(config, "lease_ttl_secs", 300.0)),
            ),
        )
        super().__init__(*args, **kwargs)
