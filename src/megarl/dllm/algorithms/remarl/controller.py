"""REMARL controller — ReMaRL: Learning to Unmask and Remask in Frozen Diffusion Language
Models (Shibaev et al., arXiv:2611.22576; code https://github.com/EgorShibaev/remarl).
"""

from __future__ import annotations

from ....async_controller import AsyncController
from ....controller import Controller
from ....dataplane import InProcessPayloadStore
from ....dllm import DLLMPayloadQueue, from_dllm_record
from .batching import make_remarl_packer
from .config import ReMaRLConfig
from .types import ReMaRLStore


class ReMaRLController(Controller):
    """Synchronous controller for ReMaRL position-policy training."""

    def __init__(self, *args, remarl_config: ReMaRLConfig, store: ReMaRLStore, **kwargs) -> None:
        controller_config = kwargs.get("config") or (args[0] if args else None)
        if getattr(controller_config, "dpo", False):
            raise ValueError("DPO is not supported for ReMaRL controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_remarl_packer(remarl_config, store))
        super().__init__(*args, **kwargs)


class AsyncReMaRLController(AsyncController):
    """Asynchronous controller with a handle-preserving payload queue."""

    def __init__(self, *args, remarl_config: ReMaRLConfig, store: ReMaRLStore, **kwargs) -> None:
        config = kwargs.get("config") or (args[0] if args else None)
        if getattr(config, "dpo", False):
            raise ValueError("DPO is not supported for ReMaRL controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_remarl_packer(remarl_config, store))
        kwargs.setdefault(
            "queue",
            DLLMPayloadQueue(
                InProcessPayloadStore(),
                ttl_secs=float(getattr(config, "lease_ttl_secs", 300.0)),
            ),
        )
        super().__init__(*args, **kwargs)
