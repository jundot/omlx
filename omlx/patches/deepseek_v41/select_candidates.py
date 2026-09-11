# SPDX-License-Identifier: Apache-2.0
"""Level-1 candidate block prefilter for the V4.1 hierarchical indexer."""
from __future__ import annotations

import mlx.core as mx


def select_candidate_blocks(
    logits: mx.array,
    compress_lens,
    topk_blocks: int,
    block_size: int,
) -> mx.array:
    """Keep the ``topk_blocks`` highest-scoring blocks per query.

    ``logits`` is ``[..., n_positions]`` with unreachable positions at ``-inf``.
    ``compress_lens`` is an int (decode) or an array broadcastable against the
    leading dims of ``logits`` (prefill). Returns a bool mask shaped like
    ``logits``.

    Mirrors official ``select_candidate_blocks`` in inference/model.py.
    """
    width = logits.shape[-1]
    pad = (-width) % block_size
    if pad:
        logits_p = mx.pad(
            logits,
            [(0, 0)] * (logits.ndim - 1) + [(0, pad)],
            constant_values=-mx.inf,
        )
    else:
        logits_p = logits

    # [..., num_blocks, block_size] -> amax over block
    scores = mx.unflatten(logits_p, -1, (-1, block_size)).max(axis=-1)
    num_blocks = scores.shape[-1]

    # Pin the block that holds this query's newest compressed position.
    if isinstance(compress_lens, mx.array):
        last = (compress_lens - 1) // block_size
        while last.ndim < scores.ndim:
            last = last[..., None]
        block_ids = mx.arange(num_blocks)
        while block_ids.ndim < scores.ndim:
            block_ids = block_ids.reshape((1,) * (scores.ndim - 1) + (num_blocks,))
        scores = mx.where(block_ids == last, mx.array(mx.inf, dtype=scores.dtype), scores)
    else:
        last = (int(compress_lens) - 1) // block_size
        if 0 <= last < num_blocks:
            # Set that block's score to +inf via scatter-like where
            block_ids = mx.arange(num_blocks)
            scores = mx.where(block_ids == last, mx.array(mx.inf, dtype=scores.dtype), scores)

    k = min(int(topk_blocks), int(num_blocks))
    # top-k indices; drop -inf leftovers
    part = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    selected = mx.take_along_axis(scores, part, axis=-1)
    # Vectorized keep mask: [..., num_blocks, k] membership without a Python loop.
    block_ids = mx.arange(num_blocks)
    while block_ids.ndim < scores.ndim:
        block_ids = block_ids.reshape((1,) * (scores.ndim - 1) + (num_blocks,))
    hit = block_ids[..., None] == part[..., None, :]
    valid = selected > -mx.inf
    keep = mx.any(mx.logical_and(hit, valid[..., None, :]), axis=-1)

    # Expand blocks to positions and trim pad
    keep = mx.repeat(keep, block_size, axis=-1)[..., :width]
    return keep
