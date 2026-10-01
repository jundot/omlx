"""Tests for singleton cache pass-through in mlx-lm BatchGenerator patches."""

import importlib

import mlx.core as mx
from mlx_lm.generate import PromptProcessingBatch, StopSequences
from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache
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


def test_join_finds_nested_model_owned_batch_conversion():

    class CustomCache:
        def to_batch(self, left_padding):
            return ("custom-batch", tuple(left_padding))

    class Model:
        layers = (object(),)

        def make_cache(self):
            return [CacheList(CacheList(CustomCache()))]

    caches = [omlx.scheduler._to_batched_cache_layer(c) for c in Model().make_cache()]

    nested = caches[0].caches[0].caches[0]
    assert nested == ("custom-batch", (0,))


def test_join_converts_vendored_qwen4_exp_linear_cache():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.cache import ArraysCache as Qwen4ArraysCache

    class Model:
        def make_cache(self):
            return [Qwen4ArraysCache(size=2)]

    caches = [omlx.scheduler._to_batched_cache_layer(c) for c in Model().make_cache()]

    assert isinstance(caches[0], Qwen4ArraysCache)
    assert caches[0].left_padding.tolist() == [0]


def test_join_leaves_running_qwen4_exp_linear_cache_unpadded():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.cache import ArraysCache as Qwen4ArraysCache

    warm = Qwen4ArraysCache(size=2)
    warm[0] = mx.ones((1, 2, 3))
    mx.eval(warm[0])

    class Model:
        def make_cache(self):
            return [warm]

    caches = [omlx.scheduler._to_batched_cache_layer(c) for c in Model().make_cache()]

    assert caches[0] is warm
    assert warm.left_padding is None


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
        stop_sequences=[StopSequences()],
        max_tokens=[8],
    )

    moved = batch.split([0])

    assert batch.uids == []
    assert batch.prompt_cache == []
    assert moved.uids == [42]
    assert moved.prompt_cache[0] is arrays
    assert moved.prompt_cache[1] is kv


def _mixed_layers(seed: int, length: int):
    """One row of a hybrid model: linear, attention and nested layers."""
    mx.random.seed(seed)
    arrays = ArraysCache(1)
    arrays[0] = mx.random.normal((1, 2, 3))
    kv = KVCache()
    kv.update_and_fetch(
        mx.random.normal((1, 2, length, 4)), mx.random.normal((1, 2, length, 4))
    )
    nested_kv = KVCache()
    nested_kv.update_and_fetch(
        mx.random.normal((1, 1, length, 4)), mx.random.normal((1, 1, length, 4))
    )
    nested = CacheList(nested_kv, _arrays_cache(float(seed)))
    layers = [arrays, BatchKVCache.merge([kv]), nested]
    mx.eval([c.state for c in layers])
    return layers


def _flat_state(layers):
    out = []
    for layer in layers:
        state = layer.state
        stack = [state]
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple)):
                stack.extend(reversed(item))
            elif isinstance(item, mx.array):
                out.append(item)
    return out


def test_join_evaluates_each_layer_and_matches_the_lazy_join():
    """The layer-by-layer join releases the donor and is bit-identical to
    joining every layer lazily, for non-Qwen4 cache types too."""
    gen = importlib.import_module("mlx_lm.generate")

    expected = [
        omlx.scheduler._extend_cache_layer(a, b)
        for a, b in zip(_mixed_layers(1, 5), _mixed_layers(2, 3))
    ]
    donor = _mixed_layers(2, 3)
    joined = gen._extend_cache(_mixed_layers(1, 5), donor)

    assert donor == [None, None, None]
    assert [type(c) for c in joined] == [type(c) for c in expected]
    got, want = _flat_state(joined), _flat_state(expected)
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert mx.array_equal(a, b).item()


def test_join_of_fresh_prompt_batches_keeps_empty_layers():
    """A prompt batch joins freshly made (empty) caches; state access on
    them must not break the per-layer evaluation."""
    gen = importlib.import_module("mlx_lm.generate")
    left = [ArraysCache(1), BatchKVCache([0]), CacheList(KVCache(), ArraysCache(1))]
    right = [ArraysCache(1), BatchKVCache([0]), CacheList(KVCache(), ArraysCache(1))]

    joined = gen._extend_cache(left, right)

    assert len(joined) == 3
    assert joined[1].offset.tolist() == [0, 0]
