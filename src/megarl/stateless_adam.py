"""Re-export of the stateless optimizer owned by :mod:`megatrain.stateless_adam`.

The optimizer resets its first and second moments every step, so checkpoints
contain no moment tensors. It is a generic training-core primitive; megaRL keeps
this module only so the historical ``from megarl.stateless_adam import ...``
import path and ``optimizer_kind="stateless_adam"`` configuration keep working.
"""

from __future__ import annotations

from megatrain.stateless_adam import StatelessAdam

__all__ = ["StatelessAdam"]