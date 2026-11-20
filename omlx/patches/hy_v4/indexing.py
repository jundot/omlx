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
from ..glm_moe_dsa.indexer_nax import (
    indexer_scores_nax,
    max_rows_per_call,
    nax_indexer_available,
)

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

# Below the kernel's masked value, bf16(-1e30).
_NAX_MASKED = -1e29

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
    if not decode and b == 1 and q.dtype == mx.bfloat16 and nax_indexer_available():
        out = _nax_scores(q, keys, weights, causal)
        if out is not None:
            return out
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


def _nax_scores(q, keys, weights, causal):
    """The same scores on M5 tensor units (one sequence), or None.

    The kernel masks key ``p`` for row ``s`` unless ``p <= before + s``; a
    non-causal call places every key before the first row.
    """
    length, total = q.shape[2], keys.shape[2]
    rows = q[0].swapaxes(0, 1)
    pool = mx.contiguous(keys[0, 0].astype(mx.bfloat16))
    w = weights[0].astype(mx.bfloat16)
    before = total - length if causal else total
    step = max_rows_per_call(total)
    parts = []
    for lo in range(0, length, step):
        hi = min(length, lo + step)
        part = indexer_scores_nax(
            mx.contiguous(rows[lo:hi]),
            pool,
            mx.contiguous(w[lo:hi]),
            before + lo,
            total,
            1,
        )
        if part is None:
            return None
        parts.append(part)
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts)
    # The kernel writes bf16(-1e30) where the causal rule masks a key.
    out = mx.where(out > _NAX_MASKED, out, -mx.inf)
    return out[None, None]


def decode_rows_ready(heads: int, keys: int, dtype) -> bool:
    """Whether one-row selections over ``keys`` keys run the decode kernel."""
    return (
        heads == _DECODE_HEADS
        and keys >= _DECODE_MIN_KEYS
        and dtype in (mx.bfloat16, mx.float16)
        and available()
        and index_fast.has_symbol("dsa_decode_scores")
    )


def decode_rows_select(q, keys, weights, totals: list[int], mask, k: int) -> list:
    """Each row's one-row decode selection, from one score and top-k call.

    Row ``i`` of ``q`` [1, H, L, 128] sees the first ``totals[i]`` of the
    [1, 1, T, 128] keys; the decode kernels score every key on its own and
    the top-k ignores -inf columns, so the rows get the bits of separate
    one-row calls. Masked picks point at the row's own key count, as decode's.
    """
    length, total = q.shape[2], keys.shape[2]
    rows = mx.contiguous(q[0].swapaxes(0, 1)[:, :, None])
    keys = mx.broadcast_to(
        mx.contiguous(keys.astype(q.dtype)), (length, 1, total, keys.shape[-1])
    )
    scores = index_fast.dsa_decode_scores(
        rows, keys, mx.contiguous(weights[0].astype(q.dtype))
    )
    counts = mx.array(totals)
    visible = mx.arange(total)[None] < counts[:, None]
    if mask is not None:
        visible = visible & mask[0, 0]
    scores = mx.where(visible[:, None, None], scores, -mx.inf)
    chosen = index_fast.dsa_topk_indices(scores, k).astype(mx.int32)
    picked = mx.take_along_axis(scores, chosen, axis=-1)
    out = mx.where(picked > -mx.inf, chosen, counts[:, None, None, None])
    return [out[i : i + 1].astype(mx.int32) for i in range(length)]


def topk(values: mx.array, k: int) -> mx.array | None:
    """Indices [B, 1, L, k] of the k largest of [B, 1, L, T] scores, or None."""
    if k not in NATIVE_TOPK or values.shape[-1] < k or not available():
        return None
    return index_fast.dsa_topk_indices(values, k).astype(mx.int32)
