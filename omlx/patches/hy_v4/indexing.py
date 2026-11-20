# SPDX-License-Identifier: Apache-2.0
"""Native DSA scoring and top-k selection for Hy4's indexers.

Both indexers score keys with a head-weighted sum of ReLU dot products. The
sparse_index extension computes that sum in one pass (fp32 accumulation,
scores stored in the query dtype) and selects the top 512 or 2048 columns
without a full sort. Every helper returns None when the extension cannot
serve the call, and the caller keeps its MLX path.
"""

from __future__ import annotations

import logging

import mlx.core as mx

from ...custom_kernels.sparse_index import fast as index_fast

logger = logging.getLogger(__name__)

# Columns the native top-k selects.
NATIVE_TOPK = (512, 2048)
_DIM = 128
# The score kernel's tile wants at least this many keys.
_MIN_KEYS = 64
# The score kernel computes 64-row tiles; on M3 Ultra MLX's matmul is faster
# below about 32 rows (verify blocks), except one row on the decode kernel.
_MIN_ROWS = 32
# The decode kernel is built for 32 heads and pays off from this many keys.
_DECODE_HEADS = 32
_DECODE_MIN_KEYS = 1024

_heads_ok: dict[int, bool] = {}


def _supports_heads(heads: int, dtype) -> bool:
    """Whether this build's score kernel takes ``heads`` query heads.

    Older builds take only 32 or 64; probe once with a tiny call.
    """
    if heads not in _heads_ok:
        try:
            q = mx.zeros((1, heads, 1, _DIM), dtype=dtype)
            k = mx.zeros((1, 1, _MIN_KEYS, _DIM), dtype=dtype)
            w = mx.zeros((1, 1, heads), dtype=dtype)
            mx.eval(index_fast.dsa_indexer_scores(q, k, w, causal=False))
            _heads_ok[heads] = True
        except (RuntimeError, ValueError, TypeError):
            _heads_ok[heads] = False
    return _heads_ok[heads]


def available() -> bool:
    return (
        index_fast.is_native_available()
        and index_fast.has_symbol("dsa_indexer_scores")
        and index_fast.has_symbol("dsa_topk_indices")
    )


def scores(
    q: mx.array, keys: mx.array, weights: mx.array, causal: bool
) -> mx.array | None:
    """Head-summed ReLU scores [B, 1, L, T] in q's dtype, or None.

    ``q`` is [B, H, L, 128], ``keys`` [B, 1, T, 128] and ``weights`` [B, L, H]
    with any scale folded in. With ``causal`` the rows are the last L of the
    T keys and later keys score -inf.
    """
    b, heads, length, dim = q.shape
    if (
        not available()
        or dim != _DIM
        or keys.shape[2] < _MIN_KEYS
        or q.dtype not in (mx.bfloat16, mx.float16)
        or not _supports_heads(heads, q.dtype)
    ):
        return None
    decode = (
        length == 1
        and heads == _DECODE_HEADS
        and keys.shape[2] >= _DECODE_MIN_KEYS
        and index_fast.has_symbol("dsa_decode_scores")
    )
    if length < _MIN_ROWS and not decode:
        return None
    keys = mx.contiguous(keys.astype(q.dtype))
    if decode:
        return index_fast.dsa_decode_scores(
            mx.contiguous(q), keys, mx.contiguous(weights.astype(q.dtype))[:, 0]
        )
    return index_fast.dsa_indexer_scores(
        mx.contiguous(q),
        keys,
        mx.contiguous(weights.astype(q.dtype)),
        causal=causal,
    )


def topk(values: mx.array, k: int) -> mx.array | None:
    """Indices [B, 1, L, k] of the k largest of [B, 1, L, T] scores, or None."""
    if k not in NATIVE_TOPK or values.shape[-1] < k or not available():
        return None
    return index_fast.dsa_topk_indices(values, k).astype(mx.int32)
