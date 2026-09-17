# SPDX-License-Identifier: Apache-2.0
"""GLM Lightning MTP rollback without splitting the target batch cache."""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList

from ..deepseek_v4.cache_extras import BatchPoolingCache


def rollback_rows(language, caches, captures, accepted, block_size):
    """Restore KDA and sparse caches with per-row accepted lengths."""

    accepted = [int(a) for a in accepted]
    if len(accepted) < 2 or any(a < 0 or a >= block_size for a in accepted):
        raise ValueError("Invalid GLM accepted-length vector")
    kept = [a + 1 for a in accepted]
    trims = [block_size - k for k in kept]
    recurrent = [cache for cache in caches if type(cache) is ArraysCache]
    if len(recurrent) != len(captures):
        raise ValueError("GLM recurrent capture count mismatch")
    for cache, entry in zip(recurrent, captures):
        if len(entry) < 11 or entry[0].shape[:2] != (len(accepted), block_size):
            raise ValueError("GLM recurrent capture shape mismatch")
        if cache[0] is None or cache[1] is None:
            raise ValueError("Missing GLM recurrent state")
    for cache in caches:
        if type(cache) is ArraysCache:
            continue
        if (
            type(cache) is not CacheList
            or len(cache.caches) != 2
            or type(cache[0]) is not BatchKVCache
            or type(cache[1]) is not BatchPoolingCache
        ):
            raise TypeError("Unsupported GLM vector rollback cache")
        if not cache[1]._can_trim_rows(trims):
            raise ValueError("Pooling cache cannot restore every accepted row")
        if cache[0]._idx < block_size or cache[0].keys.shape[0] != len(accepted):
            raise ValueError("GLM KV rollback shape mismatch")

    rewind = mx.array(trims, dtype=mx.int32)
    ends = mx.array(kept, dtype=mx.int32)
    max_end = max(kept)
    mask = mx.arange(max_end)[None] < ends[:, None]
    for cache, entry in zip(recurrent, captures):
        q, k, v, a, b, a_log, dt_bias, initial, conv_input, kernel_size, lower = entry[
            :11
        ]
        old_mask = entry[11] if len(entry) > 11 else None
        active = mask if old_mask is None else mask & old_mask[:, :max_end]
        _, state = language.gated_delta_update(
            q[:, :max_end],
            k[:, :max_end],
            v[:, :max_end],
            a[:, :max_end],
            b[:, :max_end],
            a_log,
            dt_bias,
            state=initial,
            lower_bound=lower,
            mask=active,
        )
        indices = ends[:, None, None] + mx.arange(kernel_size - 1)[None, :, None]
        cache[1] = state
        cache[0] = mx.take_along_axis(conv_input, indices, axis=1)
        cache.advance(-rewind)
    uniform_trim = block_size - max_end
    extra = [max_end - end for end in kept]
    for cache in caches:
        if type(cache) is ArraysCache:
            continue
        kv, pool = cache.caches
        if uniform_trim:
            kv.trim(uniform_trim)
        if any(extra):
            kv.prepare(right_padding=extra)
            kv.finalize()
        pool.trim_rows(trims)
    return max(accepted)
