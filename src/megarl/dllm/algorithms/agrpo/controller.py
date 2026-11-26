"""AGRPO / Amortized GRPO, Anthony Zhan, ICML 2026 arXiv 2510.04019; code
https://github.com/probablyabot/agrpo; see package docstring for details."""

from __future__ import annotations

from ....async_controller import AsyncController
from ....controller import Controller
from ....dataplane import InProcessPayloadStore
from ....dllm import DLLMPayloadQueue, from_dllm_record
from .batching import make_agrpo_packer
from .config import AGRPOConfig
from .types import AGRPOStore


class AGRPOController(Controller):
    def __init__(
        self, *args, agrpo_config: AGRPOConfig, store: AGRPOStore, **kwargs
    ) -> None:
        controller_config = kwargs.get("config") or (args[0] if args else None)
        if getattr(controller_config, "dpo", False):
            raise ValueError("DPO is not supported for AGRPO controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_agrpo_packer(agrpo_config, store))
        super().__init__(*args, **kwargs)


class AsyncAGRPOController(AsyncController):
    def __init__(
        self, *args, agrpo_config: AGRPOConfig, store: AGRPOStore, **kwargs
    ) -> None:
        config = kwargs.get("config") or (args[0] if args else None)
        if getattr(config, "dpo", False):
            raise ValueError("DPO is not supported for AGRPO controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_agrpo_packer(agrpo_config, store))
        kwargs.setdefault(
            "queue",
            DLLMPayloadQueue(
                InProcessPayloadStore(),
                ttl_secs=float(getattr(config, "lease_ttl_secs", 300.0)),
            ),
        )
        super().__init__(*args, **kwargs)
