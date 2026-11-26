"""Re-export of the KL estimators owned by :mod:`megatrain.kl`.

The estimators are generic (an element-wise function of ``log p_new - log p_ref``)
and are shared by megaTrain's SFT distillation and megaRL's KL penalty / OPD /
reverse-KL objectives. Keeping a module here preserves the historical
``from megarl.kl import ...`` import path while the implementation lives in the
training core.
"""

from __future__ import annotations

from megatrain.kl import KL_ESTIMATORS, approx_kl

__all__ = ["KL_ESTIMATORS", "approx_kl"]