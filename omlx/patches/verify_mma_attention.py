# SPDX-License-Identifier: Apache-2.0
"""Route speculative-verify attention to the MMA multi-row kernel.

Hooks ``mlx_lm.models.base.scaled_dot_product_attention`` (and every model
module that imported it by name). It acts only while the verify flag is
armed (``qwen35_verify_qmm.set_verify_qmm_armed``). A verify forward carries 3..8 new query
rows over a plain ``KVCache``; those go to ``verify_mma_kernel`` with the
cache's backing buffer, everything else (one row, two rows, prefill, array
masks, quantized or rotating caches, other head widths) stays on stock MLX.
Two rows stay stock because the kernel measured no faster there.
"""

from __future__ import annotations

import logging
import sys

import mlx.core as mx

from . import verify_mma_kernel as _kernel
from .qwen35_verify_qmm import _is_armed as _verify_armed

logger = logging.getLogger(__name__)

_MIN_ROWS = 3
_MAX_ROWS = 8
_HEAD_DIM = 128
_DTYPES = (mx.bfloat16, mx.float16)


def route_ok(queries, keys, values, cache, mask, sinks) -> bool:
    """True when this attention call is a verify block the kernel can serve."""
    from mlx_lm.models.cache import KVCache

    # Only inside a speculative verify forward (the flag the MTP loop arms for
    # the verify-shape qmm kernels). Not armed: prefill chunks and plain decode.
    # Row-exact verify leaves it unarmed too: its output must equal one-row decode.
    if not _verify_armed():
        return False
    if sinks is not None or not isinstance(mask, str) or mask != "causal":
        return False
    if type(cache) is not KVCache:
        return False
    B, hq, rows, dim = queries.shape
    if not _MIN_ROWS <= rows <= _MAX_ROWS or dim != _HEAD_DIM:
        return False
    if queries.dtype not in _DTYPES:
        return False
    hkv, n_keys = keys.shape[1], keys.shape[2]
    if hq % hkv or (hq // hkv) * rows > 32 or n_keys != cache.offset:
        return False
    buf_k, buf_v = cache.keys, cache.values
    return (
        isinstance(buf_k, mx.array)
        and isinstance(buf_v, mx.array)
        and buf_k.ndim == 4
        and buf_k.dtype == queries.dtype
        and buf_v.dtype == queries.dtype
        and buf_k.shape[1] == hkv
        and buf_k.shape[3] == _HEAD_DIM
        and buf_k.shape[2] >= n_keys
        and buf_v.shape[2] >= n_keys
    )


def apply() -> bool:
    """Install the hook. Idempotent; False when Metal is not available."""
    if not mx.metal.is_available():
        return False
    from mlx_lm.models import base as mlx_base

    original = mlx_base.scaled_dot_product_attention
    if getattr(original, "_omlx_verify_mma", False):
        return True

    def patched(queries, keys, values, cache, scale, mask, sinks=None):
        if route_ok(queries, keys, values, cache, mask, sinks):
            try:
                return _kernel.mma_verify_sdpa(
                    queries, cache.keys, cache.values, keys.shape[2], scale
                )
            except Exception:
                logger.warning("MMA verify kernel failed; using stock SDPA", exc_info=True)
        return original(queries, keys, values, cache, scale, mask, sinks)

    patched._omlx_verify_mma = True
    mlx_base.scaled_dot_product_attention = patched
    for name, mod in list(sys.modules.items()):
        if name.startswith("mlx_lm.models.") and (
            getattr(mod, "scaled_dot_product_attention", None) is original
        ):
            mod.scaled_dot_product_attention = patched
    return True
