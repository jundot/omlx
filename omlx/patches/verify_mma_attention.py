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
_engaged = 0
_probe_result = None  # None: not probed yet; a failed probe is not re-run


def engaged_count() -> int:
    """Calls this process served with the MMA kernel (0 means stock only)."""
    return _engaged


def _probe_ok() -> bool:
    """Probe once per process and remember the answer (failures too)."""
    global _probe_result
    if _probe_result is None:
        _probe_result = _run_probe()
    return _probe_result


def _run_probe() -> bool:
    """Run the kernel once on a tiny block and evaluate it.

    ``metal_kernel`` compiles and launches lazily, at ``mx.eval``, so only an
    eval here can show that Metal accepts the kernel on this machine.
    """
    try:
        d = _HEAD_DIM
        q = mx.zeros((1, 1, _MIN_ROWS, d), dtype=mx.bfloat16)
        kv = mx.zeros((1, 1, 16, d), dtype=mx.bfloat16)
        mx.eval(_kernel.mma_verify_sdpa(q, kv, kv, 16, 1.0))
        return True
    except Exception:
        logger.warning("MMA verify kernel probe failed; verify SDPA stays stock", exc_info=True)
        return False


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
    if not _probe_ok():
        return False

    def patched(queries, keys, values, cache, scale, mask, sinks=None):
        if route_ok(queries, keys, values, cache, mask, sinks):
            global _engaged
            out = _kernel.mma_verify_sdpa(
                queries, cache.keys, cache.values, keys.shape[2], scale
            )
            if _engaged == 0:
                logger.info(
                    "MMA verify attention engaged (rows=%d, keys=%d, plain KVCache)",
                    queries.shape[2],
                    keys.shape[2],
                )
            _engaged += 1
            return out
        return original(queries, keys, values, cache, scale, mask, sinks)

    patched._omlx_verify_mma = True
    mlx_base.scaled_dot_product_attention = patched
    for name, mod in list(sys.modules.items()):
        if name.startswith("mlx_lm.models.") and (
            getattr(mod, "scaled_dot_product_attention", None) is original
        ):
            mod.scaled_dot_product_attention = patched
    return True
