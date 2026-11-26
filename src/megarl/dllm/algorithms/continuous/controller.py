"""Flow-GRPO arXiv 2505.05470 and DiffNFT (ROLL diffusion pipeline
refs/ROLL/roll/pipeline/diffusion); see package docstring for details."""

from __future__ import annotations

from ....async_controller import AsyncController
from ....controller import Controller
from ....dataplane import InProcessPayloadStore
from ....dllm import DLLMPayloadQueue, from_dllm_record
from .batching import make_flow_packer
from .config import FlowConfig
from .types import FlowTrajectoryStore


class FlowController(Controller):
    """Synchronous controller wired for continuous flow RL."""

    def __init__(
        self,
        *args,
        flow_config: FlowConfig,
        store: FlowTrajectoryStore,
        **kwargs,
    ) -> None:
        config = kwargs.get("config") or (args[0] if args else None)
        if getattr(config, "dpo", False):
            raise ValueError("DPO is not supported for flow controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_flow_packer(flow_config, store))
        super().__init__(*args, **kwargs)


class AsyncFlowController(AsyncController):
    """Asynchronous flow controller with a handle-preserving payload queue."""

    def __init__(
        self,
        *args,
        flow_config: FlowConfig,
        store: FlowTrajectoryStore,
        **kwargs,
    ) -> None:
        config = kwargs.get("config") or (args[0] if args else None)
        if getattr(config, "dpo", False):
            raise ValueError("DPO is not supported for flow controllers")
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", make_flow_packer(flow_config, store))
        kwargs.setdefault(
            "queue",
            DLLMPayloadQueue(
                InProcessPayloadStore(),
                ttl_secs=float(getattr(config, "lease_ttl_secs", 300.0)),
            ),
        )
        super().__init__(*args, **kwargs)
