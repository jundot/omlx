# SPDX-License-Identifier: Apache-2.0
"""RotatingKVCache subclass tailored for omlx's restored-cache contract.

mlx-lm v0.31.3's BatchRotatingKVCache.merge() does

    keys[i:i+1, :, p:p+l] = c._temporal_order(c.keys)[..., -l:, :]

where ``l = c.size()`` and the default ``RotatingKVCache.size()`` returns
``min(offset, max_size)`` ignoring the actual buffer length. For
SSD-restored caches with ``keys.shape[2] < max_size`` and
``offset >= max_size`` this overshoots and either (a) shape-mismatches
the RHS slice when ``keys.shape[2] < l``, or (b) (when omlx zero-pads
the buffer up to max_size) exposes zero positions to attention causing
softmax dilution that surfaces as infinite loops or empty content
(issues #934, #903, #900).

This subclass clamps ``size()`` to the actual buffer length so merge is
always well-defined without any zero-padding trick on omlx's side.
"""
from __future__ import annotations

from mlx_lm.models.cache import RotatingKVCache


class PrefillReadyRotatingKVCache(RotatingKVCache):
    """RotatingKVCache that reports actual buffer length from ``size()``.

    The default ``size()`` returns ``min(offset, max_size)`` which is the
    logical token count. For caches restored from SSD whose buffer was
    sliced shorter than ``max_size`` (e.g. extract() stripped left
    padding), the logical count can exceed ``keys.shape[2]``. mlx-lm's
    merge then either over-reads the RHS or, when omlx pre-pads with
    zeros, lets those zeros leak into attention.

    Clamping to ``keys.shape[2]`` keeps merge consistent: the row gets
    exactly ``keys.shape[2]`` real entries, padded on the left by the
    enclosing batch (via ``left_padding``) instead of by phantom zeros.
    """

    def size(self):
        if self.keys is None:
            return 0
        buffer_len = self.keys.shape[2]
        if buffer_len == 0:
            return 0
        return min(super().size(), buffer_len)
