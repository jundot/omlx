# SPDX-License-Identifier: Apache-2.0
"""Multi-row affine kernels for batched decode projections.

Stock affine ``quantized_matmul`` reads the weights once per ``qmv_wide`` tile
of up to five rows (once per row on applegpu_g13), then switches to a qmm, so
a decode step of 8 or 16 requests costs far more than one of four. The verify
kernels in ``qwen35_verify_qmm`` (``vk`` at 6 rows, ``sg8`` up to 8 rows,
``mma`` up to 24) read each weight tile once for every row, and they serve
one-token decode rows too.

Only (rows, 1, K) inputs with at least ``MIN_ROWS`` rows are routed, and only
while no speculative verify or one-row exactness mode is armed. Those modes
own their (rows, 1, K) calls. Prefill chunks and single-request decode keep
their paths. On applegpu_g13 the M1-class kernels also serve 3 to 5 rows of
the larger projections. On GPUs with the NAX tensor unit, mma stops at 16
rows.
"""

from __future__ import annotations

import functools
import logging

import mlx.core as mx
import mlx.nn as nn

from . import m1_verify_mma, m1_verify_qmv, qwen35_verify_qmm
from .deepseek_v4 import decode_consistency

logger = logging.getLogger(__name__)

# On M3 Ultra sg8 is slower than stock qmv_wide at four decode rows.
MIN_ROWS = 6
# vk takes the 6-row layers sg8 cannot read (8-bit), which mma starts above.
_ROUTES = ("m1", "sg8", "mma", "m1_mma", "vk")
# applegpu_g13 kernels; stock reads the weights once per row on that GPU.
_SMALL_ROWS_ROUTES = ("m1", "m1_mma")
# Fewest rows served. Below MIN_ROWS only the M1 kernels run, and only for
# projections with at least _M1_MIN_WEIGHT weights: 512x2048 projections were
# 14-39% slower than stock at every row count measured, and two-row calls of
# the large projections measured 3.7-5.5% slower than stock on 4-bit A3B and
# 2-3% slower on 6-bit with the wrapper kept in place, so two rows stay on
# stock.
_M1_MIN_ROWS = 3
_M1_MIN_WEIGHT = 1 << 21

_PATCHED = False
_ENABLED = False
# Smallest row count the wrapper looks at: _M1_MIN_ROWS on applegpu_g13, else MIN_ROWS.
_min_rows = MIN_ROWS
_ENGAGED_LOGGED = False
# Kernel keys whose first launch succeeded (True) or failed (False).
_CHECKED: dict[tuple, bool] = {}


@functools.lru_cache(maxsize=1024)
def _route(rows: int, K: int, N: int, bits: int, group_size: int, dtype):
    route = qwen35_verify_qmm._verify_route(rows, K, N, bits, group_size, dtype)
    # Decode rows keep mma; the skinny kernel is measured on verify rows only.
    if route == "skinny" and qwen35_verify_qmm.mma_eligible(
        rows, K, N, bits, group_size, dtype
    ):
        return "mma"
    if route not in _ROUTES or (
        rows < MIN_ROWS and (route not in _SMALL_ROWS_ROUTES or K * N < _M1_MIN_WEIGHT)
    ):
        return None
    return route


def _on_m1_gpu() -> bool:
    return m1_verify_qmv.enabled() or m1_verify_mma.enabled()


def _exactness_armed() -> bool:
    return (
        qwen35_verify_qmm._is_armed()
        or qwen35_verify_qmm.is_row_exact_armed()
        or decode_consistency.is_armed()
    )


def apply_batched_decode_qmm_patch() -> bool:
    global _PATCHED, _ENABLED, _min_rows
    _ENABLED = True
    _min_rows = _M1_MIN_ROWS if _on_m1_gpu() else MIN_ROWS
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
            or x.shape[0] < _min_rows
            or getattr(self, "mode", "affine") != "affine"
        ):
            return orig_call(self, x)
        rows, _, K = x.shape
        N = self.scales.shape[0]
        route = _route(rows, K, N, self.bits, self.group_size, x.dtype)
        if route is None or self.scales.dtype != x.dtype or _exactness_armed():
            return orig_call(self, x)
        key = (route, rows, K, N, self.bits, self.group_size, x.dtype)
        if _CHECKED.get(key) is False:
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
