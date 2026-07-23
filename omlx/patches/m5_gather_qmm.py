# SPDX-License-Identifier: Apache-2.0
"""Reroute sorted gather_qmm around defective M5 NAX kernels (issue #2267).

On M5-generation GPUs mlx dispatches ``sorted_indices=True`` quantized
gather matmuls (the MoE sorted-prefill path) to the NAX
``*_gather_qmm_rhs_nax`` kernels. Two independent defects corrupt their
output on mlx <= 0.32.0:

- K remainder: the ``align_K=false`` tail bounds the activation tile
  load by ``BK`` instead of the K remainder
  (``quantized_nax.h::affine_gather_qmm_rhs_nax``), so whenever
  ``K % 64 != 0`` the tail multiplies stale threadgroup weights with
  out-of-bounds activation reads. The result is deterministically wrong
  output plus occasional recycled-buffer garbage (~1e36). This is what
  issue #2267's bit-exactness test caught: with ``inter=32`` the test's
  ``down_proj`` runs at K=32. All dtypes are affected (bf16/fp16/fp32),
  and the mxfp4 variant carries the same tail bug.
- Row offsets: the int16 fix that landed in mlx 0.32.0 missed this
  kernel, so sorted row counts above 32768 overflow the row offset
  (ml-explore/mlx#3856). Reachable in production: a 4097+ token prefill
  chunk of a top-8 MoE crosses the boundary.

``sorted_indices`` is a pure performance hint, so the wrapper simply
drops it for calls that match a defect condition. The unsorted gather
path guards ``K % 64 == 0`` before entering NAX and falls back to the
verified steel kernels, at some prefill-throughput cost for the
affected shapes only.

The wrapper self-arms: the first matching call runs a tiny canary
against an fp32 dequantized reference and only intervenes when the
corruption is actually present on this machine/mlx build. Healthy
setups keep the fast path untouched, and the patch retires itself once
mlx ships a kernel fix. Kill switch: ``OMLX_M5_GATHER_QMM_FIX=0``.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

# ml-explore/mlx#3856: sorted row offsets overflow int16 past this count.
_MAX_SORTED_ROWS = 32768

_original_gather_qmm = None
_defective: bool | None = None


def _sorted_gather_qmm_defective() -> bool:
    """Run the K=96 canary once; True when the NAX rhs kernel corrupts."""
    global _defective
    if _defective is not None:
        return _defective
    if not mx.metal.is_available():
        _defective = False
        return False
    n, e, out_dim, k = 64, 8, 64, 96
    keys = mx.random.split(mx.random.key(0x2267), 3)
    w = mx.random.normal((e, out_dim, k), key=keys[0]).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=32, bits=4)
    x = (mx.random.normal((n, 1, k), key=keys[1]) * 0.5).astype(mx.bfloat16)
    idx = mx.sort(mx.random.randint(0, e, (n,), key=keys[2]).astype(mx.uint32))
    wd = mx.dequantize(wq, scales, biases, group_size=32, bits=4)
    ref = x.astype(mx.float32) @ wd[idx].swapaxes(-1, -2).astype(mx.float32)
    out = _original_gather_qmm(
        x,
        wq,
        scales,
        biases,
        rhs_indices=idx,
        transpose=True,
        group_size=32,
        bits=4,
        sorted_indices=True,
    )
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    # Corruption sits at output magnitude (median row error ~7 at this
    # size); bf16 rounding stays below ~0.1. NaN also counts as corrupt.
    _defective = not (err < 1.0)
    if _defective:
        logger.warning(
            "sorted gather_qmm corrupts on this machine (canary max err "
            "%.3g); rerouting K %% 64 != 0 and >%d-row sorted calls to the "
            "unsorted path (issue #2267)",
            err,
            _MAX_SORTED_ROWS,
        )
    return _defective


def _needs_reroute(x, args, kwargs) -> bool:
    """True when this call would select the defective NAX rhs kernel."""
    if not kwargs.get("sorted_indices"):
        return False
    # Positional layout after (x, w): scales, biases, lhs_indices,
    # rhs_indices, transpose, group_size, bits, mode. sorted_indices is
    # keyword-only.
    lhs = args[2] if len(args) > 2 else kwargs.get("lhs_indices")
    rhs = args[3] if len(args) > 3 else kwargs.get("rhs_indices")
    transpose = args[4] if len(args) > 4 else kwargs.get("transpose", True)
    # The rhs kernel is only selected for the rhs-indices-only sorted
    # path with transposed weights (x @ w.T).
    if lhs is not None or rhs is None or not transpose:
        return False
    if x.shape[-1] % 64:
        return True
    return rhs.size * x.shape[-2] > _MAX_SORTED_ROWS


def _gather_qmm_rerouted(x, w, *args, **kwargs):
    if _needs_reroute(x, args, kwargs) and _sorted_gather_qmm_defective():
        kwargs = dict(kwargs, sorted_indices=False)
    return _original_gather_qmm(x, w, *args, **kwargs)


_gather_qmm_rerouted._omlx_m5_reroute = True


def apply_m5_gather_qmm_workaround() -> bool:
    """Install the reroute wrapper on ``mx.gather_qmm``.

    Idempotent; returns True when the wrapper was installed by this
    call. Disabled entirely via ``OMLX_M5_GATHER_QMM_FIX=0``.
    """
    global _original_gather_qmm
    if os.environ.get("OMLX_M5_GATHER_QMM_FIX", "1") == "0":
        return False
    if getattr(mx.gather_qmm, "_omlx_m5_reroute", False):
        return False
    _original_gather_qmm = mx.gather_qmm
    mx.gather_qmm = _gather_qmm_rerouted
    logger.debug("m5 sorted gather_qmm reroute installed")
    return True
