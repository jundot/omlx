# SPDX-License-Identifier: Apache-2.0
"""Decode-exact draft verification for Hy4.

Lightning MTP verifies a block of draft rows in one forward. While the
generator arms it, that forward gives every row the bits of a one-token decode
step, so greedy output does not depend on MTP:

* quantized projections run the row-exact kernels (one-row arithmetic per row,
  weights read once for affine formats), unquantized ones one row at a time;
* the indexers, the attention core and the sliding-window caches run one row
  at a time against the cache as that decode step saw it;
* routed experts take decode's unsorted gather, in groups small enough not to
  be sorted.

Only single-sequence blocks qualify, on plain caches or on one-row batch
caches without left padding (mlx-lm's generator batches even one request);
each row then sees the mask its decode step built. Batched verify keeps the
regular kernels. The language model decides once per forward with ``scope``.
"""

from __future__ import annotations

import contextlib
import threading

import mlx.core as mx
import mlx.nn as nn

from ..deepseek_v4 import verify_qmv
from ..qwen35_verify_qmm import set_verify_qmm_armed

MAX_ROWS = 16
# The weight-once MXFP8 kernel covers this many rows per call.
_MXFP8_ROWS = 8
# Rows per call of the fp32 head kernel.
_HEAD_ROWS = 6
_STATE = threading.local()


def set_armed(flag: bool) -> None:
    _STATE.armed = bool(flag)
    set_verify_qmm_armed(flag, row_exact=flag)


def armed() -> bool:
    return getattr(_STATE, "armed", False)


@contextlib.contextmanager
def scope(inputs: mx.array, cache):
    """Run one forward as an exact verify block when it qualifies."""
    rows = inputs.shape[1] if inputs.ndim >= 2 else 0
    on = (
        armed()
        and inputs.shape[0] == 1
        and 1 < rows <= MAX_ROWS
        and cache is not None
        and all(_padding_free(c) for c in cache)
    )
    previous = getattr(_STATE, "active", False)
    _STATE.active = on
    try:
        yield on
    finally:
        _STATE.active = previous


def _padding_free(cache) -> bool:
    for c in getattr(cache, "caches", None) or (cache,):
        left = getattr(c, "left_padding", None)
        if left is not None and bool(mx.any(left > 0).item()):
            return False
    return True


def block(x: mx.array) -> bool:
    """Whether ``x`` holds the rows of the active exact verify block."""
    if not getattr(_STATE, "active", False) or x.ndim < 2:
        return False
    return 1 < x.size // x.shape[-1] <= MAX_ROWS


def row_mask(mask, row: int, keys: int):
    """Row ``row`` of a block mask, as the decode step over ``keys`` slots saw it."""
    if mask is None:
        return None
    return mask[..., row : row + 1, :keys]


def rowwise(fn, x: mx.array) -> mx.array:
    """``fn`` on each row of ``x`` as a one-row [1, D] call."""
    rows = x.reshape(-1, x.shape[-1])
    out = mx.concatenate([fn(rows[i : i + 1]) for i in range(rows.shape[0])])
    return out.reshape(*x.shape[:-1], out.shape[-1])


def linear(module, x: mx.array) -> mx.array:
    """``module(x)``, with decode's bits per row inside a block.

    Unquantized layers run row by row; MXFP8 layers take the weight-once
    kernel; other quantized layers take the row-exact kernels armed with the
    block.
    """
    if not block(x):
        return module(x)
    if isinstance(module, nn.Linear):
        return _dense_rows(module, x)
    if getattr(module, "mode", None) == "mxfp8":
        out = _mxfp8_rows(module, x)
        if out is not None:
            return out
    return module(x)


def _dense_rows(module, x: mx.array) -> mx.array:
    """Unquantized rows: the weight-once GEMV of one-row decode where its
    layout allows (bf16, wide outputs), otherwise one call per row."""
    rows = x.reshape(-1, x.shape[-1])
    chunks = [rows[i : i + _HEAD_ROWS] for i in range(0, rows.shape[0], _HEAD_ROWS)]
    if all(len(c) < 2 or verify_qmv.dense_eligible(module, c) for c in chunks):
        out = mx.concatenate(
            [
                module(c) if len(c) < 2 else verify_qmv.exact_verify_gemv(module, c)
                for c in chunks
            ]
        )
        return out.reshape(*x.shape[:-1], out.shape[-1])
    return rowwise(module, x)


def _mxfp8_rows(module, x: mx.array) -> mx.array | None:
    k = x.shape[-1]
    rows = x.reshape(-1, k)
    if (
        module.bits != 8
        or k % 512
        or rows.shape[0] > _MXFP8_ROWS
        or x.dtype not in (mx.bfloat16, mx.float16)
    ):
        return None
    out = verify_qmv.exact_rows_qmv(
        rows,
        module.weight,
        module.scales,
        None,
        group_size=int(module.group_size),
        mxfp8=True,
    )
    out = out.reshape(*x.shape[:-1], out.shape[-1])
    if "bias" in module:
        out = out + module.bias
    return out


def fp32_head(head, hidden: mx.array) -> mx.array:
    """Logits in fp32 from a bf16 head, the same kernel for decode and verify.

    The weight stays bf16 (half the reads of an fp32 copy); the kernel
    multiplies and accumulates in fp32, so the math matches an fp32 head.
    """
    k = hidden.shape[-1]
    n = head.weight.shape[0]
    rows = hidden.reshape(-1, k)
    eligible = (
        isinstance(head, nn.Linear)
        and head.weight.dtype == hidden.dtype
        and hidden.dtype in (mx.bfloat16, mx.float16)
        and k % 128 == 0
        and n >= 4096
        and n % 32 == 0
    )
    if eligible and (rows.shape[0] <= _HEAD_ROWS or block(hidden)):
        out = mx.concatenate(
            [
                verify_qmv.dspark_head_gemv(head, rows[i : i + _HEAD_ROWS])
                for i in range(0, rows.shape[0], _HEAD_ROWS)
            ]
        )
        return out.reshape(*hidden.shape[:-1], n)
    if block(hidden):
        return rowwise(lambda r: head(r.astype(mx.float32)), hidden)
    return head(hidden.astype(mx.float32))
