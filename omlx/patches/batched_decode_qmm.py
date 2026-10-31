# SPDX-License-Identifier: Apache-2.0
"""Multi-row affine kernels for batched decode projections.

Stock affine ``quantized_matmul`` reads the weights once per ``qmv_wide`` tile
of up to five rows, then switches to a qmm, so a decode step of 8 or 16
requests costs far more than one of four. The verify kernels in
``qwen35_verify_qmm`` (``sg8`` up to 8 rows, ``mma`` up to 24) read each
weight tile once for every row, and they serve one-token decode rows too.

Only (rows, 1, K) inputs with at least ``MIN_ROWS`` rows are routed, and only
while no speculative verify or one-row exactness mode is armed. Those modes
own their (rows, 1, K) calls. Prefill chunks and single-request decode keep
their paths. On GPUs with the NAX tensor unit, mma stops at 16 rows.
"""

from __future__ import annotations

import functools
import logging

import mlx.core as mx
import mlx.nn as nn

from . import qwen35_verify_qmm
from .deepseek_v4 import decode_consistency

logger = logging.getLogger(__name__)

# On M3 Ultra sg8 is slower than stock qmv_wide at four decode rows.
MIN_ROWS = 6
_ROUTES = ("sg8", "mma", "m1_mma")

_PATCHED = False
_ENABLED = False
_ENGAGED_LOGGED = False
# Kernel keys whose first launch succeeded (True) or failed (False).
_CHECKED: dict[tuple, bool] = {}


@functools.lru_cache(maxsize=1024)
def _route(rows: int, K: int, N: int, bits: int, group_size: int, dtype):
    route = qwen35_verify_qmm._verify_route(rows, K, N, bits, group_size, dtype)
    return route if route in _ROUTES else None


def _exactness_armed() -> bool:
    return (
        qwen35_verify_qmm._is_armed()
        or qwen35_verify_qmm.is_row_exact_armed()
        or decode_consistency.is_armed()
    )


def apply_batched_decode_qmm_patch() -> bool:
    global _PATCHED, _ENABLED
    _ENABLED = True
    if _PATCHED:
        return True

    cls = nn.QuantizedLinear
    orig_call = cls.__call__

    def patched_call(self, x):
        global _ENGAGED_LOGGED
        if (
            not _ENABLED
            or x.ndim != 3
            or x.shape[1] != 1
            or x.shape[0] < MIN_ROWS
            or getattr(self, "mode", "affine") != "affine"
            or self.scales.dtype != x.dtype
            or _exactness_armed()
        ):
            return orig_call(self, x)
        rows, _, K = x.shape
        N = self.scales.shape[0]
        route = _route(rows, K, N, self.bits, self.group_size, x.dtype)
        key = (route, rows, K, N, self.bits, self.group_size, x.dtype)
        if route is None or _CHECKED.get(key) is False:
            return orig_call(self, x)
        y = qwen35_verify_qmm.run_route(self, x.reshape(rows, K), route)
        if key not in _CHECKED:
            # Kernel build and launch errors surface at eval time.
            try:
                mx.eval(y)
            except Exception:
                logger.warning(
                    "Batched decode %s kernel failed for %s; using stock qmm",
                    route,
                    key[1:],
                    exc_info=True,
                )
                _CHECKED[key] = False
                return orig_call(self, x)
            _CHECKED[key] = True
        if hasattr(self, "bias"):
            y = y + self.bias
        if not _ENGAGED_LOGGED:
            _ENGAGED_LOGGED = True
            logger.info(
                "Batched decode affine kernels engaged (%d rows, %s)", rows, route
            )
        return y.reshape(rows, 1, N)

    cls.__call__ = patched_call
    _PATCHED = True
    return True


def remove_batched_decode_qmm_patch() -> None:
    """Return every call to the stock path; later patches may wrap this one."""
    global _ENABLED
    _ENABLED = False
