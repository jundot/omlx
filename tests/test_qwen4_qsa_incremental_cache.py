"""Exactness and lifecycle tests for incremental Qwen4 QSA block caching."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import math

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
qsa_fast = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")


def _identity_rope(x, position_ids):
    del position_ids
    return x


def _append(cache, raw_keys, start, stop):
    length = stop - start
    keys = raw_keys[:, start:stop, :4].reshape(1, 1, length, 4)
    values = (keys + 1).astype(keys.dtype)
    cache.update_and_fetch(keys, values)
    cache.update_indexer(
        raw_keys[:, start:stop],
        mx.arange(start, stop, dtype=mx.int32)[None],
    )


def test_qsa_kv_and_raw_index_buffers_grow_geometrically_with_logical_views():
    cache = language.QSAKVCache()
    raw = mx.arange(8194 * 8, dtype=mx.float32).reshape(1, 8194, 8)

    _append(cache, raw, 0, 2050)
    kv_backing = cache.keys
    index_backing = cache._index_keys
    assert cache.keys.shape[2] == 8192
    assert cache._index_keys.shape[1] == 8192
    assert cache.index_keys.shape == (1, 2050, 8)

    _append(cache, raw, 2050, 4098)
    assert cache.keys is kv_backing
    assert cache._index_keys is index_backing
    assert cache.state[0].shape[2] == 4098
    assert cache.state[2].shape[1] == 4098

    _append(cache, raw, 4098, 8194)
    assert cache.keys.shape[2] == 16384
    assert cache._index_keys.shape[1] == 16384
    assert cache.state[0].shape[2] == 8194
    assert cache.state[2].shape[1] == 8194
    assert language.QSAQuantizedKVCache.step == 8192
    assert language.QSAQuantizedKVCache.geometric_growth is True


@pytest.mark.parametrize("chunks", [(2048, 2048, 2048), (2050, 2048, 2047)])
def test_completed_qsa_blocks_match_one_shot_and_only_compute_new_suffix(chunks):
    total = sum(chunks)
    raw = mx.sin(mx.arange(total * 8, dtype=mx.float32)).reshape(1, total, 8)
    incremental = language.QSAKVCache()
    block_calls = []

    def tracked_norm(x):
        block_calls.append(int(x.shape[1]))
        return x * mx.array(1.25, dtype=x.dtype)

    start = 0
    for length in chunks:
        stop = start + length
        incremental.update_indexer(
            raw[:, start:stop],
            mx.arange(start, stop, dtype=mx.int32)[None],
        )
        actual = incremental.pooled_indexer_keys(
            4,
            tracked_norm,
            _identity_rope,
            cache_tag=tracked_norm,
        )
        start = stop

    calls_before_noop = list(block_calls)
    cached_again = incremental.pooled_indexer_keys(
        4,
        tracked_norm,
        _identity_rope,
        cache_tag=tracked_norm,
    )
    assert block_calls == calls_before_noop
    assert block_calls == [512, 512, 512]

    one_shot = qsa_fast.pool_completed_index_keys(
        raw,
        mx.arange(total, dtype=mx.int32)[None],
        compress_ratio=4,
        index_key_norm=lambda x: x * mx.array(1.25, dtype=x.dtype),
        apply_index_rope=_identity_rope,
    )
    mx.eval(actual, cached_again, one_shot)
    assert mx.array_equal(actual, one_shot).item()
    assert mx.array_equal(cached_again, one_shot).item()


@pytest.mark.parametrize("chunks", [(8, 8, 8), (6, 7, 12)])
def test_gathered_qsa_one_shot_and_incremental_appends_are_exact(chunks, monkeypatch):
    total = sum(chunks)
    mx.random.seed(431)
    queries = mx.random.normal((1, 4, total, 8)).astype(mx.float16)
    keys = mx.random.normal((1, 2, total, 8)).astype(mx.float16)
    values = mx.random.normal((1, 2, total, 8)).astype(mx.float16)
    index_queries = mx.random.normal((1, total, 3, 8)).astype(mx.float16)
    index_keys = mx.random.normal((1, total, 8)).astype(mx.float16)
    positions = mx.arange(total, dtype=mx.int32)[None]
    kwargs = dict(
        num_query_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        indexer_head_dim=8,
        compress_ratio=4,
        token_budget=8,
        index_key_norm=lambda x: x,
        apply_index_rope=_identity_rope,
        query_chunk=3,
    )
    monkeypatch.setattr(qsa_fast, "_native_indexer_scores", lambda *a, **k: None)

    expected = qsa_fast.contiguous_causal_gathered_qsa(
        queries,
        keys,
        values,
        index_queries,
        index_keys,
        positions,
        **kwargs,
    )

    cache = language.QSAKVCache()
    outputs = []
    start = 0
    for length in chunks:
        stop = start + length
        cache.update_indexer(index_keys[:, start:stop], positions[:, start:stop])
        pooled = cache.pooled_indexer_keys(
            4,
            kwargs["index_key_norm"],
            kwargs["apply_index_rope"],
            cache_tag=kwargs["index_key_norm"],
        )
        outputs.append(
            qsa_fast.contiguous_causal_gathered_qsa(
                queries[:, :, start:stop],
                keys[:, :, :stop],
                values[:, :, :stop],
                index_queries[:, start:stop],
                cache.index_keys,
                cache.index_position_ids,
                pooled_index_keys=pooled,
                **kwargs,
            )
        )
        start = stop
    actual = mx.concatenate(outputs, axis=1)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_qsa_ephemeral_pool_rebuilds_after_restore_extract_and_trim():
    cache = language.QSAKVCache()
    raw = mx.sin(mx.arange(13 * 8, dtype=mx.float32)).reshape(1, 13, 8)
    _append(cache, raw, 0, 13)
    pooled = cache.pooled_indexer_keys(
        4, lambda x: x, _identity_rope, cache_tag=cache
    )
    mx.eval(pooled)
    assert cache._pooled_index_offset == 3
    assert len(cache.state) == 4

    restored = language.QSAKVCache()
    restored.prefix_cache_restore(cache.prefix_cache_snapshot())
    assert restored._pooled_index_keys is None
    restored_pool = restored.pooled_indexer_keys(
        4, lambda x: x, _identity_rope, cache_tag=restored
    )

    extracted = cache.extract(0)
    assert extracted._pooled_index_keys is None
    extracted_pool = extracted.pooled_indexer_keys(
        4, lambda x: x, _identity_rope, cache_tag=extracted
    )
    mx.eval(restored_pool, extracted_pool, pooled)
    assert mx.array_equal(restored_pool, pooled).item()
    assert mx.array_equal(extracted_pool, pooled).item()

    assert cache.trim(3) == 3
    assert cache._pooled_index_keys is None
    replacement = mx.cos(mx.arange(3 * 8, dtype=mx.float32)).reshape(1, 3, 8)
    _append(cache, mx.concatenate([raw[:, :10], replacement], axis=1), 10, 13)
    rebuilt = cache.pooled_indexer_keys(
        4, lambda x: x, _identity_rope, cache_tag=cache
    )
    expected = qsa_fast.pool_completed_index_keys(
        cache.index_keys,
        cache.index_position_ids,
        compress_ratio=4,
        index_key_norm=lambda x: x,
        apply_index_rope=_identity_rope,
    )
    mx.eval(rebuilt, expected)
    assert mx.array_equal(rebuilt, expected).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_portable_qsa_flattened_gemm_is_exactly_the_broadcast_reference(dtype):
    mx.random.seed(909)
    queries = mx.random.normal((1, 7, 4, 128)).astype(dtype)
    pooled = mx.random.normal((1, 19, 128)).astype(dtype)
    broadcast = queries.astype(mx.float32) @ pooled[:, None].astype(
        mx.float32
    ).swapaxes(-1, -2)
    expected = mx.sum(mx.maximum(broadcast, 0), axis=-2) / math.sqrt(128)
    actual = qsa_fast._portable_indexer_scores(queries, pooled, 128)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()
