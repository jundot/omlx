"""Bonsai 1-bit / 2-bit QuantizedLinear decode patch.

Intercepts ``QuantizedLinear.__call__`` for layers whose weight tensor is
1-bit or 2-bit affine-quantized and routes them through the Bonsai fast
decode kernels (qmv_fast for 1-bit, qmv_wide for 2-bit small-batch).

Activation condition
--------------------
Only active when:
  * ``bits`` in {1, 2}  and  ``mode == "affine"``
  * The input batch dimension M is in the decode regime (M <= 5)
  * The native bonsai extension is available (falls back silently otherwise)

Usage
-----
Call ``apply_bonsai_qmv_patch()`` once after model load.  It monkey-patches
``mlx.nn.QuantizedLinear`` globally, so all matching layers in the loaded
model are accelerated automatically.

Call ``remove_bonsai_qmv_patch()`` to restore the original implementation.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from omlx.custom_kernels.bonsai.fast import (
    bonsai_q1_affine_qmv,
    bonsai_q2_affine_qmv,
    bonsai_qmv_wide,
    has_native,
    _use_qmv_wide,
)

logger = logging.getLogger(__name__)

_original_quantized_linear_call: Any = None
_patch_active = False

# Maximum input batch size routed through fast decode kernels.
# Above this threshold the model is prefilling — use stock mlx qmm_t instead.
_MAX_DECODE_M = 5


def _get_cached_scales_biases(
    self: nn.QuantizedLinear, dtype: Any
) -> tuple[mx.array, mx.array]:
    """Return scales/biases cast to `dtype`, caching the result on the layer."""
    cache_attr = "_bonsai_sb_cache"
    cache = getattr(self, cache_attr, None)
    if cache is None or cache[0] != dtype:
        sc = self.scales.astype(dtype)
        bi = self.biases.astype(dtype)
        mx.eval(sc, bi)
        object.__setattr__(self, cache_attr, (dtype, sc, bi))
    _, sc, bi = getattr(self, cache_attr)
    return sc, bi


def _bonsai_quantized_linear_call(self: nn.QuantizedLinear, x: mx.array) -> mx.array:
    """Replacement for QuantizedLinear.__call__ for 1-bit and 2-bit layers."""
    bits: int = getattr(self, "bits", 4)
    mode: str = getattr(self, "mode", "affine")

    # Only intercept 1-bit / 2-bit affine layers in decode regime.
    if mode != "affine" or bits not in (1, 2):
        return _original_quantized_linear_call(self, x)

    M = x.shape[-2] if x.ndim >= 2 else 1
    if M > _MAX_DECODE_M:
        return _original_quantized_linear_call(self, x)

    w = self.weight
    # Cache scales/biases cast to x's dtype (Metal kernel reads them as T).
    scales, biases = _get_cached_scales_biases(self, x.dtype)

    if _use_qmv_wide(bits, M):
        # M>=3 on gen-15+: stream weights once across all M vectors
        out = bonsai_qmv_wide(x, w, scales, biases, bits=bits)
    elif bits == 1:
        out = bonsai_q1_affine_qmv(x, w, scales, biases)
    else:
        # 2-bit M=1 or M=2: qmv_fast
        out = bonsai_q2_affine_qmv(x, w, scales, biases)

    # QuantizedLinear may have a bias term (separate from quantization biases).
    linear_bias = getattr(self, "bias", None)
    if linear_bias is not None:
        out = out + linear_bias
    return out


def apply_bonsai_qmv_patch() -> bool:
    """Monkey-patch QuantizedLinear for fast 1-bit / 2-bit decode.

    Returns True if the patch was applied (native extension available),
    False if skipped.
    """
    global _original_quantized_linear_call, _patch_active

    if _patch_active:
        return True

    if not has_native():
        logger.debug(
            "bonsai_qmv: native extension not available, skipping patch."
        )
        return False

    _original_quantized_linear_call = nn.QuantizedLinear.__call__
    nn.QuantizedLinear.__call__ = _bonsai_quantized_linear_call
    _patch_active = True
    logger.info("bonsai_qmv: QuantizedLinear patched for 1-bit / 2-bit decode.")
    return True


def remove_bonsai_qmv_patch() -> None:
    """Restore the original QuantizedLinear.__call__."""
    global _original_quantized_linear_call, _patch_active
    if not _patch_active or _original_quantized_linear_call is None:
        return
    nn.QuantizedLinear.__call__ = _original_quantized_linear_call
    _original_quantized_linear_call = None
    _patch_active = False
    logger.info("bonsai_qmv: QuantizedLinear patch removed.")


def is_patch_active() -> bool:
    return _patch_active
