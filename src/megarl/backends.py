"""Heterogeneous backend validation for TensorPlay devices.

TensorPlay exposes devices through its dispatcher, with out-of-tree hardware
registered via `PrivateUse1` plugins (MetaX / MLU / NPU). This module probes a
device with a real tensor op before a rollout or trainer is placed on it, so a
missing kernel fails at startup rather than mid-run.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

KNOWN_BACKENDS = ("cpu", "cuda", "rocm", "vulkan", "mps", "meta", "privateuse1")


@dataclasses.dataclass(frozen=True, slots=True)
class BackendValidation:
    name: str
    available: bool
    reason: str
    smoke_passed: bool


def validate_backend(name: str) -> BackendValidation:
    import tensorplay as tp

    try:
        left = tp.tensor([1, 2, 3], device=name)
        right = tp.tensor([4, 5, 6], device=name)
    except Exception as exc:  # device or plugin not present in this build
        return BackendValidation(name, False, f"{type(exc).__name__}: {exc}", False)
    try:
        result = left + right
        smoke = [float(v) for v in result.flatten().tolist()] == [5.0, 7.0, 9.0]
    except Exception as exc:
        return BackendValidation(name, True, f"kernel missing: {exc}", False)
    return BackendValidation(
        name, True, "device and elementwise kernel available", smoke
    )


def validate_backends(names: Sequence[str] = KNOWN_BACKENDS) -> dict[str, BackendValidation]:
    return {name: validate_backend(name) for name in names}


def usable_backends(names: Sequence[str] = KNOWN_BACKENDS) -> list[str]:
    return [
        name
        for name, result in validate_backends(names).items()
        if result.available and result.smoke_passed
    ]
