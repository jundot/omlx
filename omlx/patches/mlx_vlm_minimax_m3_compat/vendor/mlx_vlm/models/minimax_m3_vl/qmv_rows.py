"""Decode-exact projections for calls of a few rows.

MLX's kernels for MXFP8 calls of a few rows can round a row differently from
a one-row decode step, and from three rows on they stream the weights again
for every row. ``exact_rows_qmv`` keeps MLX ``fp_qmv_fast``'s per-row
arithmetic and 32-lane reduction order but reads each weight once for all
rows, so speculative verify rows and small decode batches match one-row decode
bit for bit. One-row MXFP8 calls run ``decode_kernels.mxfp8_one_row``, the
same arithmetic with more weight loads in flight. The base checkpoint's BF16
vocabulary head runs through ``exact_verify_gemv`` for the same reason as the
rows: MLX's one-row GEMV with every row sharing a weight load.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from omlx.patches import row_exact_qmv

from . import decode_kernels
from omlx.patches.deepseek_v4.verify_qmv import (
    dense_eligible,
    exact_rows_qmv,
    exact_verify_gemv,
)
from omlx.patches.qwen35_verify_qmm import is_row_exact_armed

MAX_ROWS = 8
# Up to this many rows keep their inputs in registers. Longer calls read them
# from cache inside the row loop, which avoids register spills.
_HELD_INPUT_MAX_ROWS = 5
# Output columns per simdgroup of the row-exact affine tile.
_ROW_EXACT_RPS = 2
# Projections at least this wide (the vocabulary head) share one weight pass
# between all verify rows.
_WIDE_OUTPUT_COLUMNS = 65536


def mxfp8_rows_matmul(
    x: mx.array, weight: mx.array, scales: mx.array, quant: dict
) -> mx.array | None:
    """``x @ dequantized(weight).T`` for 1-8 rows, every row with one-row
    decode arithmetic, or None when not covered."""
    k = x.shape[-1]
    rows = x.size // k
    n = weight.shape[0]
    if not (
        1 <= rows <= MAX_ROWS
        and quant.get("mode") == "mxfp8"
        and quant.get("bits") == 8
        and quant.get("group_size") == 32
        and x.dtype in (mx.bfloat16, mx.float16)
        and weight.dtype == mx.uint32
        and weight.shape[1] * 4 == k
        and scales.shape == (n, k // 32)
        and k % 256 == 0
        and n % 8 == 0
        and mx.metal.is_available()
    ):
        return None
    if rows == 1:
        return decode_kernels.mxfp8_one_row(x, weight, scales)
    held = rows <= _HELD_INPUT_MAX_ROWS
    y = exact_rows_qmv(
        x.reshape(rows, k),
        weight,
        scales,
        None,
        group_size=32,
        mxfp8=True,
        rps=4 if held else 2,
        held=held,
    )
    return y.reshape(*x.shape[:-1], n)


def _row_exact_tile(rows: int, columns: int) -> tuple[int, int]:
    """(rows padded with zero rows, rows per threadgroup) of the affine tile.

    The vocabulary head reads the most bytes, so all of its rows share one
    weight pass. Narrower projections spill registers past five rows per tile;
    an odd row count above that gets one padding row so it splits evenly.
    """
    if columns >= _WIDE_OUTPUT_COLUMNS or rows <= 5:
        return rows, rows
    padded = rows + rows % 2
    return padded, next(d for d in (4, 3, 2) if padded % d == 0)


def row_exact_linears(linears, x: mx.array) -> tuple | None:
    """Affine projections of ``x`` in a row-exact verify forward (and the
    vocabulary head of a decode batch), or None.

    One launch whose threadgroups apply each weight block to several rows, so
    the weights are read once per row group instead of once per row.
    """
    rows = x.size // x.shape[-1]
    columns = sum(int(linear.weight.shape[0]) for linear in linears)
    # A decode batch runs the vocabulary head here too, so its rows match the
    # requests decoded alone.
    if not (
        2 <= rows <= MAX_ROWS
        and (is_row_exact_armed() or columns >= _WIDE_OUTPUT_COLUMNS)
    ):
        return None
    padded, per_group = _row_exact_tile(rows, columns)
    flat = x.reshape(rows, x.shape[-1])
    if padded > rows:
        pad = mx.zeros((padded - rows, flat.shape[-1]), dtype=flat.dtype)
        flat = mx.concatenate([flat, pad])
    # The unrolled tile is faster where its register envelope allows it.
    for unrolled in (True, False):
        out = row_exact_qmv.quantized_linears_tiled(
            linears, flat, per_group, _ROW_EXACT_RPS, unrolled
        )
        if out is not None:
            return tuple(y[:rows].reshape(*x.shape[:-1], -1) for y in out)
    return None


def rows_linear(proj, x: mx.array) -> mx.array:
    """Apply ``proj``, through a decode-exact kernel when one covers the call."""
    if type(proj) is nn.QuantizedLinear and "bias" not in proj:
        quant = {"group_size": proj.group_size, "bits": proj.bits, "mode": proj.mode}
        y = mxfp8_rows_matmul(x, proj["weight"], proj["scales"], quant)
        if y is not None:
            return y
        out = row_exact_linears((proj,), x)
        if out is not None:
            return out[0]
    elif type(proj) is nn.Linear and dense_eligible(proj, x, max_rows=MAX_ROWS):
        return exact_verify_gemv(proj, x)
    return proj(x)
