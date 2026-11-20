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

from ..qwen35_verify_qmm import set_verify_qmm_armed

MAX_ROWS = 16
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
    """``module(x)``; an unquantized layer runs row by row inside a block."""
    if isinstance(module, nn.Linear) and block(x):
        return rowwise(module, x)
    return module(x)
