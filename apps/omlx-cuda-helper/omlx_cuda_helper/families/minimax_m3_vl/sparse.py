# SPDX-License-Identifier: Apache-2.0
"""MiniMax sparse attention for one prefill chunk.

The portable functions here are the reference the CUDA kernels must match.
"""

from __future__ import annotations

import mlx.core as mx

from . import Config, kernels

# Bytes of float32 scores one slab of query rows may hold.
_SLAB_BYTES = 1 << 29


def select_blocks(
    index_queries: mx.array, index_keys: mx.array, start: int, cfg: Config
) -> mx.array:
    """Block ids [index heads, L, k] each query keeps.

    ``index_queries`` is [index heads, L, dim] for the queries at positions
    ``start`` on and ``index_keys`` [T, dim] for every key so far. A block
    scores the highest product among its keys the query may see; the local
    (and initial) blocks are always kept. A query that sees fewer than k
    blocks also lists blocks ahead of it, whose keys attention masks out.
    """
    # A close port of oMLX's MiniMax block selection (mlx-vlm lineage, MIT).
    heads, length, _ = index_queries.shape
    total = index_keys.shape[0]
    blocks = -(-total // cfg.block)
    keep = min(cfg.top_blocks, blocks)
    keys = index_keys.astype(mx.float32).T
    block_ids, positions = mx.arange(blocks), mx.arange(total)
    rows = max(1, _SLAB_BYTES // (4 * heads * total))
    picked = []
    for begin in range(0, length, rows):
        queries = index_queries[:, begin : begin + rows].astype(mx.float32)
        at = mx.arange(start + begin, start + begin + queries.shape[1])[:, None]
        scores = mx.where(positions <= at, queries @ keys, -mx.inf)
        scores = mx.pad(
            scores,
            [(0, 0), (0, 0), (0, blocks * cfg.block - total)],
            constant_values=-mx.inf,
        )
        scores = scores.reshape(heads, -1, blocks, cfg.block).max(axis=-1)
        current = at // cfg.block
        visible = block_ids <= current
        scores = mx.where(visible, scores, -mx.inf)
        if cfg.init_blocks > 0:
            scores = mx.where((block_ids < cfg.init_blocks) & visible, 1e30, scores)
        if cfg.local_blocks > 0:
            local = (block_ids > current - cfg.local_blocks) & visible
            scores = mx.where(local, 1e29, scores)
        picked.append(mx.argpartition(-scores, kth=keep - 1, axis=-1)[..., :keep])
        # Evaluated here, so only one slab of scores is alive at a time.
        mx.eval(picked[-1])
    return mx.concatenate(picked, axis=1).astype(mx.int32)


def host_partials(cfg: Config, rows: int, total: int) -> bool:
    """Whether oMLX attends a sparse chunk with its MSA prefill kernels, which
    round each block's own softmax output before the blocks combine."""
    shapes = cfg.head_dim == 128 and cfg.block == 128
    shapes = shapes and cfg.heads == 16 * cfg.kv_heads == 16 * cfg.index_heads
    return shapes and rows > 16 and total > 2 * cfg.block * cfg.top_blocks


def attend(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    start: int,
    scale: float,
    cfg: Config,
    partial: bool = False,
) -> mx.array:
    """Attention of ``queries`` [1, heads, L, dim] over the kept blocks.

    Index head ``h`` selects for KV head ``h``; a single index head selects
    for all of them. Scores, softmax and the value sum run in float32.
    ``partial`` rounds each block's own softmax output to the query dtype
    before the blocks combine by their log-sum-exp (see ``host_partials``).
    """
    _, heads, length, dim = queries.shape
    kv_heads, total = keys.shape[1], keys.shape[2]
    group = heads // kv_heads
    blocks = -(-total // cfg.block)
    all_blocks, positions = mx.arange(blocks), mx.arange(total)
    pad = [(0, 0), (0, 0), (0, 0), (0, blocks * cfg.block - total)]
    keys_t = keys[0].astype(mx.float32)[:, None].swapaxes(-1, -2)
    values = values[0].astype(mx.float32)[:, None]
    if partial:
        values = mx.pad(values, [(0, 0), (0, 0), pad[-1], (0, 0)])
        values = values.reshape(kv_heads, blocks, cfg.block, dim)
    rows = max(1, _SLAB_BYTES // (4 * heads * total * (4 if partial else 1)))
    outputs = []
    for begin in range(0, length, rows):
        ids = block_ids[:, begin : begin + rows]
        at = mx.arange(start + begin, start + begin + ids.shape[1])[:, None]
        kept = (ids[..., None] == all_blocks).any(axis=-2)
        kept = mx.repeat(kept, cfg.block, axis=-1)[..., :total] & (positions <= at)
        q = queries[0, :, begin : begin + rows].astype(mx.float32) * scale
        scores = q.reshape(kv_heads, group, -1, dim) @ keys_t
        scores = mx.where(kept[:, None], scores, -mx.inf)
        if partial:
            out = _combine_blocks(scores, values, pad, cfg.block, queries.dtype)
        else:
            out = mx.softmax(scores, axis=-1) @ values
        outputs.append(out.reshape(heads, -1, dim))
        mx.eval(outputs[-1])
    return mx.concatenate(outputs, axis=1).astype(queries.dtype)[None]


def _combine_blocks(scores, values, pad, block: int, dtype: mx.Dtype) -> mx.array:
    """Softmax attention as a weighted mean of each block's own attention.

    ``scores`` [kv, group, rows, T] holds -inf at the keys a row skips and
    ``values`` is [kv, blocks, block, dim]. A block's output rounds to
    ``dtype`` and weighs the sum of its exponentials.
    """
    kv, group, rows, _ = scores.shape
    scores = mx.pad(scores, pad, constant_values=-mx.inf)
    scores = scores.reshape(kv, group, rows, -1, block)
    top = scores.max(axis=-1, keepdims=True)
    weights = mx.where(scores == -mx.inf, 0.0, mx.exp(scores - top))
    mass = weights.sum(axis=-1, keepdims=True)
    # Blocks lead the product so each one multiplies its own values.
    part = weights.transpose(0, 3, 1, 2, 4).reshape(kv, -1, group * rows, block)
    part = (part @ values).reshape(kv, -1, group, rows, values.shape[-1])
    part = part.transpose(0, 2, 3, 1, 4) / mass
    # A block without a kept key has no mass; it gets weight zero.
    part = mx.where(mass > 0, part.astype(dtype).astype(mx.float32), 0.0)
    lse = mx.where(mass > 0, top + mx.log(mass), -mx.inf)
    share = mx.exp(lse - lse.max(axis=-2, keepdims=True))
    return (share * part).sum(axis=-2) / share.sum(axis=-2)


def attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    index_queries: mx.array,
    index_keys: mx.array,
    start: int,
    scale: float,
    cfg: Config,
    kernel: bool = False,
) -> mx.array:
    """Sparse attention of a chunk whose first query sits at ``start``."""
    partial = host_partials(cfg, queries.shape[2], keys.shape[2])
    if kernel:
        _, bits = kernels.select_blocks(
            index_queries[0], index_keys[0, 0], start, cfg.top_blocks
        )
        return kernels.attend(queries, keys, values, bits, start, scale, partial)
    block_ids = select_blocks(index_queries[0], index_keys[0, 0], start, cfg)
    return attend(queries, keys, values, block_ids, start, scale, cfg, partial)
