# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx
from mlx_lm.models.cache import dynamic_roll

from omlx.custom_kernels.attention import fast as _attention_fast

_NATIVE_ROLL_MAX_SHIFT = 16


def _native_roll(x, shifts, length: int, axis: int, max_shift: int):
    if (
        not 0 < max_shift < length
        or max_shift > _NATIVE_ROLL_MAX_SHIFT
        or x.itemsize not in (1, 2, 4)
        or mx.default_device() != mx.gpu
        or not _attention_fast.has_symbol("roll_rows")
    ):
        return None
    rows = mx.broadcast_to(shifts, x.shape[:axis]).reshape(-1).astype(mx.int32)
    return _attention_fast.roll_rows(x, rows, axis % x.ndim, length, max_shift)


def _roll_logical_prefix(x, shifts, length: int, axis: int, max_shift=None):
    if max_shift == 0:
        return x
    if max_shift is not None:
        rolled = _native_roll(x, shifts, length, axis, max_shift)
        if rolled is not None:
            return rolled
    n = x.shape[axis]
    if n == length:
        return dynamic_roll(x, shifts, axis=axis)
    if length <= 0:
        return x
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    positions = mx.arange(n)[expand_indices]
    rolled = (positions - shifts[expand_shifts]) % length
    idx = mx.where(positions < length, rolled, positions)
    return mx.take_along_axis(x, idx, axis=axis)
