"""dLLM reinforcement learning (d1 arXiv 2505.22765, wd1; see devdocs/dllm-rl.md); see
package docstring for details."""

from __future__ import annotations

from ..async_controller import AsyncController
from ..controller import Controller
from ..dataplane import InProcessPayloadStore
from .batching import make_dllm_packer, make_dllm_trajectory_packer
from .config import DLLMConfig
from .estimator import from_dllm_record
from .trajectory import DLLMPayloadQueue, DLLMTrajectoryStore


class DLLMController(Controller):
    """Controller wired for dLLM rollouts without any AR branch in the loop."""

    def __init__(
        self, *args, dllm_config: DLLMConfig, store: DLLMTrajectoryStore | None = None,
        **kwargs,
    ) -> None:
        controller_config = kwargs.get("config") or (args[0] if args else None)
        if getattr(controller_config, "dpo", False):
            raise ValueError("DPO is not supported for dLLM controllers")
        if dllm_config.estimator == "trajectory":
            if store is None:
                raise ValueError("trajectory controller requires a DLLMTrajectoryStore")
            packer = make_dllm_trajectory_packer(dllm_config, store)
        else:
            packer = make_dllm_packer(dllm_config)
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", packer)
        super().__init__(*args, **kwargs)


class AsyncDLLMController(AsyncController):
    """Fully asynchronous controller wired for one-step or trajectory dLLM RL."""

    def __init__(
        self, *args, dllm_config: DLLMConfig, store: DLLMTrajectoryStore | None = None,
        **kwargs,
    ) -> None:
        config = kwargs.get("config") or (args[0] if args else None)
        if getattr(config, "dpo", False):
            raise ValueError("DPO is not supported for dLLM controllers")
        if dllm_config.estimator == "trajectory":
            if store is None:
                raise ValueError("trajectory controller requires a DLLMTrajectoryStore")
            packer = make_dllm_trajectory_packer(dllm_config, store)
            kwargs.setdefault(
                "queue",
                DLLMPayloadQueue(
                    InProcessPayloadStore(),
                    ttl_secs=float(getattr(config, "lease_ttl_secs", 300.0)),
                ),
            )
        else:
            packer = make_dllm_packer(dllm_config)
        kwargs.setdefault("sample_builder", from_dllm_record)
        kwargs.setdefault("batch_packer", packer)
        super().__init__(*args, **kwargs)
