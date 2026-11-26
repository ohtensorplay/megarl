"""JustGRPO / The Flexibility Trap, Zanlin Ni et al., ICML 2026 arXiv 2601.15165; code
https://github.com/LeapLabTHU/JustGRPO; see package docstring for details."""

from __future__ import annotations

from ....async_controller import AsyncController
from ....controller import Controller
from ....dataplane import InProcessPayloadStore
from ....dllm import DLLMPayloadQueue, from_dllm_record
from .batching import make_justgrpo_packer
from .config import JustGRPOConfig
from .types import JustGRPOStore


class JustGRPOController(Controller):
    """Synchronous controller for AR-order GRPO."""

    def __init__(
        self, *args, justgrpo_config: JustGRPOConfig, store: JustGRPOStore,
        **kwargs,
    ) -> None:
        controller_config = kwargs.get("config") or (args[0] if args else None)
        if getattr(controller_config, "dpo", False):
            raise ValueError("DPO is not supported for JustGRPO controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_justgrpo_packer(justgrpo_config, store))
        super().__init__(*args, **kwargs)


class AsyncJustGRPOController(AsyncController):
    """Asynchronous controller with a handle-preserving payload queue."""

    def __init__(
        self, *args, justgrpo_config: JustGRPOConfig, store: JustGRPOStore,
        **kwargs,
    ) -> None:
        config = kwargs.get("config") or (args[0] if args else None)
        if getattr(config, "dpo", False):
            raise ValueError("DPO is not supported for JustGRPO controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_justgrpo_packer(justgrpo_config, store))
        kwargs.setdefault(
            "queue",
            DLLMPayloadQueue(
                InProcessPayloadStore(),
                ttl_secs=float(getattr(config, "lease_ttl_secs", 300.0)),
            ),
        )
        super().__init__(*args, **kwargs)
