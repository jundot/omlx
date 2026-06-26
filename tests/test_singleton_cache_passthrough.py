"""Tests for singleton cache pass-through in mlx-lm BatchGenerator patches."""

import importlib

import mlx.core as mx
from mlx_lm.generate import PromptProcessingBatch, SequenceStateMachine
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_vlm.turboquant import TurboQuantKVCache

import omlx.scheduler  # noqa: F401  (applies BatchGenerator cache patches)
from omlx.turboquant_kv import BatchTurboQuantKVCache


def _kv_cache(length: int) -> KVCache:
    cache = KVCache()
    cache.update_and_fetch(
        mx.ones((1, 1, length, 4)),
        mx.ones((1, 1, length, 4)),
    )
    mx.eval(cache.keys, cache.values)
    return cache


def _arrays_cache(value: float = 1.0) -> ArraysCache:
    cache = ArraysCache(1)
    cache[0] = mx.full((1, 2, 3), value)
    mx.eval(cache[0])
    return cache


def _tq_cache(length: int) -> TurboQuantKVCache:
    fp_cache = _kv_cache(length)
    cache = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    mx.eval(cache.keys, cache.values)
    return cache


def test_singleton_merge_preserves_regular_cache_objects():
    gen = importlib.import_module("mlx_lm.generate")
    arrays = _arrays_cache()
    kv = _kv_cache(4)

    merged = gen._merge_caches([[arrays, kv]])

    assert merged[0] is arrays
    assert merged[1] is kv


def test_extend_converts_singleton_kv_to_batched_cache():
    gen = importlib.import_module("mlx_lm.generate")
    kv_a = _kv_cache(4)
    kv_b = _kv_cache(2)

    extended = gen._extend_cache([kv_a], [kv_b])
    batch_kv = extended[0]
    mx.eval(batch_kv.offset, batch_kv.left_padding)

    assert isinstance(batch_kv, BatchKVCache)
    assert batch_kv.offset.tolist() == [4, 2]
    assert batch_kv.left_padding.tolist() == [0, 2]


def test_singleton_merge_preserves_plain_turboquant_cache():
    gen = importlib.import_module("mlx_lm.generate")
    tq = _tq_cache(4)

    merged = gen._merge_caches([[tq]])

    assert merged[0] is tq


def test_extend_converts_plain_turboquant_to_batched_cache():
    gen = importlib.import_module("mlx_lm.generate")
    tq_a = _tq_cache(4)
    tq_b = _tq_cache(2)

    extended = gen._extend_cache([tq_a], [tq_b])
    batch_tq = extended[0]
    mx.eval(batch_tq.offset, batch_tq.left_padding)

    assert isinstance(batch_tq, BatchTurboQuantKVCache)
    assert batch_tq.offset.tolist() == [4, 2]
    assert batch_tq.left_padding.tolist() == [0, 2]


def test_extend_keeps_arrays_cache_in_place():
    gen = importlib.import_module("mlx_lm.generate")
    arrays_a = _arrays_cache(1.0)
    arrays_b = _arrays_cache(2.0)

    extended = gen._extend_cache([arrays_a], [arrays_b])

    assert extended[0] is arrays_a
    assert arrays_a[0].shape[0] == 2


def test_prompt_batch_full_split_moves_cache_without_copy():
    arrays = _arrays_cache()
    kv = _kv_cache(3)
    batch = PromptProcessingBatch(
        model=object(),
        uids=[42],
        caches=[[arrays, kv]],
        tokens=[[1, 2, 3]],
        prefill_step_size=4,
        samplers=[None],
        fallback_sampler=lambda logits: logits,
        logits_processors=[[]],
        state_machines=[SequenceStateMachine()],
        max_tokens=[8],
    )

    moved = batch.split([0])

    assert batch.uids == []
    assert batch.prompt_cache == []
    assert moved.uids == [42]
    assert moved.prompt_cache[0] is arrays
    assert moved.prompt_cache[1] is kv
