# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the materialized fixed-cache allocation lifecycle."""

from __future__ import annotations

import importlib
import json

import mlx.core as mx
import pytest
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    RotatingKVCache,
)
from mlx_lm.sample_utils import make_sampler

import omlx.scheduler  # noqa: F401 - installs OMLX's mlx-lm batch safety patches
from omlx.fixed_kv_memory import estimate_model_memory
from omlx.fixed_kv_pool import (
    FixedKVCacheCapacityError,
    FixedKVCacheError,
    FixedKVCachePool,
)
from omlx.patches.deepseek_v4.cache_extras import BatchPoolingCache, PoolingCache
from omlx.patches.mlx_vlm_minimax_m3_compat import (
    apply_mlx_vlm_minimax_m3_compat_patch,
)

apply_mlx_vlm_minimax_m3_compat_patch()

from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3KVCache  # noqa: E402

from omlx.patches.mlx_vlm_unlimited_ocr_compat import (  # noqa: E402
    apply_mlx_vlm_unlimited_ocr_compat_patch,
)

apply_mlx_vlm_unlimited_ocr_compat_patch()

from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache  # noqa: E402


class _DenseModel:
    def make_cache(self):
        return [KVCache(), KVCache()]

    def __call__(self, tokens, cache):
        batch, length = tokens.shape
        for layer in cache:
            layer.update_and_fetch(
                mx.ones((batch, 2, length, 8), dtype=mx.float16),
                mx.ones((batch, 2, length, 4), dtype=mx.float16),
            )
        return mx.zeros((batch, length, 16))


class _HybridModel:
    def make_cache(self):
        return [
            RotatingKVCache(max_size=8),
            CacheList(KVCache(), KVCache()),
            ArraysCache(2),
        ]

    def __call__(self, tokens, cache):
        batch, length = tokens.shape
        for layer in cache:
            children = layer.caches if isinstance(layer, CacheList) else (layer,)
            for child in children:
                if isinstance(child, ArraysCache):
                    child[0] = mx.ones((batch, 3), dtype=mx.float16)
                    child[1] = mx.ones((batch, 2, 2), dtype=mx.float16)
                else:
                    child.update_and_fetch(
                        mx.ones((batch, 1, length, 4), dtype=mx.float16),
                        mx.ones((batch, 1, length, 2), dtype=mx.float16),
                    )
        return mx.zeros((batch, length, 16))


class _DeepSeekMLAModel:
    def make_cache(self):
        return [KVCache(), KVCache()]

    def __call__(self, tokens, cache):
        batch, length = tokens.shape
        for layer in cache:
            layer.update_and_fetch(
                mx.ones((batch, 1, length, 576), dtype=mx.float16),
                mx.ones((batch, 1, length, 0), dtype=mx.float16),
            )
        return mx.zeros((batch, length, 16))


class _DeepSeekDSAModel:
    def make_cache(self):
        return [CacheList(KVCache(), KVCache()), CacheList(KVCache(), KVCache())]

    def __call__(self, tokens, cache):
        batch, length = tokens.shape
        for layer in cache:
            layer[0].update_and_fetch(
                mx.ones((batch, 1, length, 16), dtype=mx.float16),
                mx.ones((batch, 1, length, 4), dtype=mx.float16),
            )
            layer[1].update_and_fetch(
                mx.ones((batch, 1, length, 8), dtype=mx.float16),
                mx.ones((batch, 1, length, 0), dtype=mx.float16),
            )
        return mx.zeros((batch, length, 16))


class _DeepSeekV4PoolingModel:
    def make_cache(self):
        return [
            CacheList(
                RotatingKVCache(max_size=8),
                PoolingCache(4),
                PoolingCache(4),
            )
        ]

    def __call__(self, tokens, cache):
        batch, length = tokens.shape
        local_cache, main_pool, index_pool = cache[0].caches
        local_cache.update_and_fetch(
            mx.ones((batch, 1, length, 4), dtype=mx.float16),
            mx.ones((batch, 1, length, 0), dtype=mx.float16),
        )
        for pool_cache, pooled_dim in ((main_pool, 4), (index_pool, 2)):
            projected = mx.ones(
                (batch, length, pooled_dim * 2), dtype=mx.float16
            )
            gate = mx.ones((batch, length, pooled_dim * 2), dtype=mx.float16)
            ready, ready_gate, _ = pool_cache.accumulate_windows(
                projected,
                gate,
                local_cache.offset,
            )
            if ready.shape[1]:
                windowed = mx.unflatten(ready, 1, (-1, pool_cache.ratio))
                gate_windowed = mx.unflatten(
                    ready_gate, 1, (-1, pool_cache.ratio)
                )
                pool_cache.store_prev(windowed, gate_windowed, dropped=0)
                pooled = windowed.mean(axis=2)[..., :pooled_dim]
            else:
                pooled = mx.zeros((batch, 0, pooled_dim), dtype=mx.float16)
            pool_cache.update_and_fetch(pooled)
        return mx.zeros((batch, length, 16))


class _QwenGatedDeltaModel:
    def make_cache(self):
        return [ArraysCache(2), KVCache(), ArraysCache(2), KVCache()]

    def __call__(self, tokens, cache):
        batch, length = tokens.shape
        for layer in cache:
            if isinstance(layer, ArraysCache):
                layer[0] = mx.ones((batch, 3, 20), dtype=mx.float16)
                layer[1] = mx.ones((batch, 4, 2, 3), dtype=mx.float32)
            else:
                layer.update_and_fetch(
                    mx.ones((batch, 2, length, 4), dtype=mx.float16),
                    mx.ones((batch, 2, length, 4), dtype=mx.float16),
                )
        return mx.zeros((batch, length, 16))


class _MiniMaxM3Model:
    def make_cache(self):
        return [KVCache(), MiniMaxM3KVCache()]

    def __call__(self, tokens, cache):
        batch, length = tokens.shape
        for layer in cache:
            layer.update_and_fetch(
                mx.ones((batch, 2, length, 8), dtype=mx.float16),
                mx.ones((batch, 2, length, 8), dtype=mx.float16),
            )
            if hasattr(layer, "update_index_and_fetch"):
                layer.update_index_and_fetch(
                    mx.ones((batch, 1, length, 6), dtype=mx.float16)
                )
        return mx.zeros((batch, length, 16))


class _Llama4ChunkedModel:
    def make_cache(self):
        return [
            ChunkedKVCache(256),
            ChunkedKVCache(256),
            ChunkedKVCache(256),
            KVCache(),
        ]

    def __call__(self, tokens, cache):
        batch, length = tokens.shape
        for layer in cache:
            if isinstance(layer, ChunkedKVCache):
                layer.maybe_trim_front()
            layer.update_and_fetch(
                mx.ones((batch, 2, length, 8), dtype=mx.float16),
                mx.ones((batch, 2, length, 8), dtype=mx.float16),
            )
        return mx.zeros((batch, length, 16))


class _UnlimitedOCRRingModel:
    def make_cache(self):
        return [RingSlidingKVCache(4)]

    def __call__(self, tokens, cache):
        batch, length = tokens.shape
        layer = cache[0]
        layer.update_and_fetch(
            mx.ones((batch, 1, length, 4), dtype=mx.float16),
            mx.ones((batch, 1, length, 4), dtype=mx.float16),
        )
        return mx.zeros((batch, length, 16))


class _NativeMTPModel:
    _omlx_mtp_decode_enabled = True

    def make_cache(self):
        return [KVCache()]

    def make_mtp_cache(self):
        return [KVCache()]

    def __call__(self, tokens, cache, return_hidden=False):
        batch, length = tokens.shape
        cache[0].update_and_fetch(
            mx.ones((batch, 2, length, 8), dtype=mx.float16),
            mx.ones((batch, 2, length, 8), dtype=mx.float16),
        )
        logits = mx.zeros((batch, length, 16), dtype=mx.float16)
        hidden = mx.ones((batch, length, 16), dtype=mx.float16)
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden, next_ids, cache):
        batch, length = next_ids.shape
        cache[0].update_and_fetch(
            mx.ones((batch, 2, length, 8), dtype=mx.float16),
            mx.ones((batch, 2, length, 8), dtype=mx.float16),
        )
        return mx.zeros((batch, length, 16), dtype=mx.float16)


class DSparkContextCache:
    def __init__(self, max_size):
        self.max_size = int(max_size)
        self.offset = 0
        self.keys = None

    def append(self, keys, *, start_offset=None):
        if start_offset is not None:
            self.offset = int(start_offset)
        self.keys = keys if self.keys is None else mx.concatenate([self.keys, keys], axis=2)
        self.keys = self.keys[:, :, -self.max_size :]
        self.offset += int(keys.shape[2])

class _DSparkMTPModel:
    _omlx_mtp_decode_enabled = True
    _omlx_mtp_chain = True
    _omlx_mtp_head_clone = False

    def make_cache(self):
        return [RotatingKVCache(max_size=8)]

    def make_mtp_cache(self):
        return [DSparkContextCache(8) for _ in range(3)]

    def __call__(self, tokens, cache, return_hidden=False):
        batch, length = tokens.shape
        cache[0].update_and_fetch(
            mx.ones((batch, 1, length, 4), dtype=mx.float16),
            mx.ones((batch, 1, length, 0), dtype=mx.float16),
        )
        logits = mx.zeros((batch, length, 16), dtype=mx.float16)
        hidden = mx.ones((batch, length, 12), dtype=mx.float16)
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden, next_ids, cache):
        batch, length = next_ids.shape
        for stage in cache:
            stage.append(
                mx.ones((batch, 1, length, 4), dtype=mx.float16)
            )
        return mx.zeros((batch, length, 16), dtype=mx.float16)


def test_pool_materializes_once_and_batch_operations_reuse_backing_rows():
    model = _DenseModel()
    pool = FixedKVCachePool.create(model, context_window=257, slots=2)
    array_ids = [id(array) for array in pool._arrays]

    first = pool.make_cache(0)
    second = pool.make_cache(1)
    model(mx.array([[1, 2, 3]]), cache=first)
    model(mx.array([[1, 2]]), cache=second)
    batch = [left.merge([left, right]) for left, right in zip(first, second)]
    model(mx.array([[4], [4]]), cache=batch)
    for layer in batch:
        layer.filter([1])
    mx.eval([layer.state for layer in batch])

    assert [id(array) for array in pool._arrays] == array_ids
    assert pool.committed_bytes == sum(array.nbytes for array in pool._arrays)
    assert pool.to_dict()["lifecycle"] == "committed"
    assert batch[0].offset.tolist() == [3]


def test_native_mtp_uses_precommitted_auxiliary_rows():
    from omlx.patches.mlx_lm_mtp import make_mtp_cache

    model = _NativeMTPModel()
    pool = FixedKVCachePool.create(
        model,
        context_window=16,
        slots=2,
        native_mtp_enabled=True,
    )
    array_ids = [id(array) for array in pool._arrays]
    target = pool.make_cache(1)
    mtp = make_mtp_cache(model, target)

    for _ in range(4):
        model.mtp_forward(
            mx.ones((1, 1, 16), dtype=mx.float16),
            mx.ones((1, 1), dtype=mx.int32),
            mtp,
        )
    mx.eval(*mtp[0].state)

    assert mtp[0].slot == 1
    assert mtp[0].offset == 4
    assert [id(array) for array in pool._arrays] == array_ids
    assert pool.native_mtp_bytes_per_session == 2 * 2 * 256 * 8 * 2


def test_deepseek_dspark_rings_are_fixed_and_match_the_plan(tmp_path):
    from omlx.patches.mlx_lm_mtp import make_mtp_cache
    from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import (
        DSparkContextCache as DynamicDSparkContextCache,
    )

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "num_hidden_layers": 1,
                "head_dim": 4,
                "index_head_dim": 2,
                "sliding_window": 8,
                "compress_ratios": [0],
                "dspark_block_size": 5,
                "dspark_target_layer_ids": [40, 41, 42],
                "num_nextn_predict_layers": 1,
                "torch_dtype": "float16",
                "max_position_embeddings": 4096,
            }
        )
    )
    plan = estimate_model_memory(
        tmp_path,
        16,
        weights_bytes=0,
        requested_session_slots=2,
        available_memory_bytes=1 << 30,
        native_mtp_enabled=True,
    )
    model = _DSparkMTPModel()
    pool = FixedKVCachePool.create(
        model,
        context_window=16,
        slots=2,
        native_mtp_enabled=True,
    )
    array_ids = [id(array) for array in pool._arrays]
    target = pool.make_cache(0)
    mtp = make_mtp_cache(model, target)

    for _ in range(12):
        model.mtp_forward(
            mx.ones((1, 1, 12), dtype=mx.float16),
            mx.ones((1, 1), dtype=mx.int32),
            mtp,
        )

    assert all(cache.offset == 12 for cache in mtp)
    assert all(cache.keys.shape == (1, 1, 8, 4) for cache in mtp)
    assert [id(array) for array in pool._arrays] == array_ids
    assert pool.serving_bytes == plan.fixed_kv_cache_bytes
    assert pool.pool_scratch_bytes == plan.pool_scratch_bytes

    fixed = mtp[0]
    fixed.reset()
    dynamic = DynamicDSparkContextCache(8)
    for start_offset, values in (
        (5, range(3)),
        (None, range(3, 8)),
        (None, range(8, 10)),
    ):
        keys = mx.repeat(
            mx.array(list(values), dtype=mx.float16).reshape(1, 1, -1, 1),
            4,
            axis=3,
        )
        fixed.append(keys, start_offset=start_offset)
        dynamic.append(keys, start_offset=start_offset)
        mx.eval(fixed.keys, dynamic.keys)
        assert fixed.offset == dynamic.offset
    assert bool(mx.allclose(fixed.keys, dynamic.keys))

    with pytest.raises(FixedKVCacheCapacityError, match="9 tokens"):
        fixed.keys = mx.zeros((1, 1, 9, 4), dtype=mx.float16)


def test_batch_finalize_rolls_padding_through_reserved_scratch(monkeypatch):
    model = _DenseModel()
    pool = FixedKVCachePool.create(model, context_window=16, slots=2)
    first = pool.make_cache(0)
    second = pool.make_cache(1)
    batch = [left.merge([left, right]) for left, right in zip(first, second)]
    for layer in batch:
        layer.prepare(lengths=[3, 1], right_padding=[0, 2])
    model(mx.ones((2, 3), dtype=mx.int32), cache=batch)

    def forbidden_dynamic_roll(*_args, **_kwargs):
        raise AssertionError("finalize allocated a second active-batch cache")

    monkeypatch.setattr(mx, "take_along_axis", forbidden_dynamic_roll)
    for layer in batch:
        layer.finalize()

    assert batch[0].offset.tolist() == [3, 1]
    assert batch[0].left_padding.tolist() == [0, 2]
    assert batch[0]._idx == 3


def test_rotating_batch_finalize_reuses_reserved_scratch(monkeypatch):
    model = _HybridModel()
    pool = FixedKVCachePool.create(model, context_window=16, slots=2)
    first = pool.make_cache(0)
    second = pool.make_cache(1)
    model(mx.array([[1, 2, 3]]), cache=first)
    model(mx.array([[4]]), cache=second)
    batch = [left.merge([left, right]) for left, right in zip(first, second)]
    rotating = batch[0]
    rotating.prepare(lengths=[3, 1], right_padding=[0, 2])
    rotating.update_and_fetch(
        mx.ones((2, 1, 3, 4), dtype=mx.float16),
        mx.ones((2, 1, 3, 2), dtype=mx.float16),
    )

    def forbidden_dynamic_roll(*_args, **_kwargs):
        raise AssertionError("finalize allocated a second rotating batch cache")

    monkeypatch.setattr(mx, "take_along_axis", forbidden_dynamic_roll)
    monkeypatch.setattr(mx, "roll", forbidden_dynamic_roll)
    rotating.finalize()

    assert rotating.offset.tolist() == [6, 2]
    assert rotating.left_padding.tolist() == [0, 4]
    assert rotating._lengths is None


def test_minimax_sparse_index_cache_uses_fixed_rows_through_batch_lifecycle():
    model = _MiniMaxM3Model()
    pool = FixedKVCachePool.create(model, context_window=17, slots=2)
    array_ids = [id(array) for array in pool._arrays]

    first = pool.make_cache(0)
    second = pool.make_cache(1)
    model(mx.array([[1, 2, 3]]), cache=first)
    model(mx.array([[4]]), cache=second)
    batch = [left.merge([left, right]) for left, right in zip(first, second)]
    assert batch[1].merge([batch[1]]) is batch[1]
    model(mx.array([[5], [6]]), cache=batch)
    for layer in batch:
        layer.filter([1])
    sparse = batch[1].extract(0)
    state_arrays = [
        array
        for layer in batch
        for array in layer.state
        if isinstance(array, mx.array)
    ]
    mx.eval(*state_arrays)

    assert sparse.index_offset == 2
    assert sparse.index_keys.shape == (1, 1, 256, 6)
    assert [id(array) for array in pool._arrays] == array_ids
    assert pool.committed_bytes == sum(array.nbytes for array in pool._arrays)


def test_llama4_chunked_cache_reuses_fixed_resident_window():
    model = _Llama4ChunkedModel()
    pool = FixedKVCachePool.create(
        model,
        context_window=600,
        slots=1,
        prefill_step_size=128,
    )
    array_ids = [id(array) for array in pool._arrays]
    cache = pool.make_cache(0)

    for _ in range(4):
        model(mx.ones((1, 128), dtype=mx.int32), cache=cache)
        mx.eval(*[array for layer in cache for array in layer.state])
        assert [id(array) for array in pool._arrays] == array_ids

    assert cache[0].offset == 512
    assert cache[0].start_position == 128
    assert cache[0].state[0].shape[2] == 384
    with pytest.raises(FixedKVCacheCapacityError, match="600"):
        model(mx.ones((1, 128), dtype=mx.int32), cache=cache)


def test_unlimited_ocr_ring_overwrites_inside_fixed_context_row():
    model = _UnlimitedOCRRingModel()
    pool = FixedKVCachePool.create(model, context_window=12, slots=1)
    array_ids = [id(array) for array in pool._arrays]
    cache = pool.make_cache(0)

    model(mx.ones((1, 6), dtype=mx.int32), cache=cache)
    for _ in range(4):
        model(mx.ones((1, 1), dtype=mx.int32), cache=cache)
    mx.eval(*cache[0].state)

    assert cache[0].prefill_length == 6
    assert cache[0].offset == 10
    assert cache[0].state[0].shape[2] == 10
    assert [id(array) for array in pool._arrays] == array_ids
    model(mx.ones((1, 1), dtype=mx.int32), cache=cache)
    assert cache[0].offset == 11
    assert cache[0].state[0].shape[2] == 10


def test_unlimited_ocr_batch_generator_keeps_serialized_fixed_ring():
    model = _UnlimitedOCRRingModel()
    pool = FixedKVCachePool.create(model, context_window=12, slots=1)
    cache = pool.make_cache(0)
    array_ids = [id(array) for array in pool._arrays]
    generator = BatchGenerator(
        model,
        max_tokens=5,
        sampler=make_sampler(temp=0),
        prefill_batch_size=1,
        completion_batch_size=1,
    )

    assert cache[0].merge([cache[0]]) is cache[0]
    uid = generator.insert(
        [[2, 3, 4, 5, 6, 7]],
        max_tokens=[5],
        caches=[cache],
    )[0]
    finished = False
    for _ in range(10):
        _, responses = generator.next()
        finished = finished or any(
            response.uid == uid and response.finish_reason is not None
            for response in responses
        )
        if finished:
            break

    assert finished
    assert [id(array) for array in pool._arrays] == array_ids


def test_mlx_batch_generator_admits_filters_and_reuses_two_pool_rows():
    model = _DenseModel()
    pool = FixedKVCachePool.create(model, context_window=8, slots=2)
    array_ids = [id(array) for array in pool._arrays]
    generator = BatchGenerator(
        model,
        max_tokens=2,
        sampler=make_sampler(temp=0),
        prefill_batch_size=1,
        completion_batch_size=2,
    )
    uids = generator.insert(
        [[2, 3], [4]],
        max_tokens=[2, 2],
        caches=[pool.make_cache(0), pool.make_cache(1)],
    )
    finished = set()

    for _ in range(8):
        _, responses = generator.next()
        finished.update(
            response.uid for response in responses if response.finish_reason is not None
        )
        assert [id(array) for array in pool._arrays] == array_ids

    assert finished == set(uids)

    reused_uid = generator.insert(
        [[5, 6]],
        max_tokens=[1],
        caches=[pool.make_cache(0)],
    )[0]
    reused_finished = False
    for _ in range(4):
        _, responses = generator.next()
        reused_finished = reused_finished or any(
            response.uid == reused_uid and response.finish_reason is not None
            for response in responses
        )
        assert [id(array) for array in pool._arrays] == array_ids

    assert reused_finished


def test_staggered_hybrid_admission_reuses_existing_fixed_batch():
    """A late hybrid request must join a surviving fixed-cache row in place."""
    model = _HybridModel()
    pool = FixedKVCachePool.create(model, context_window=16, slots=2)
    generator = BatchGenerator(
        model,
        max_tokens=8,
        sampler=make_sampler(temp=0),
        prefill_batch_size=1,
        completion_batch_size=2,
    )
    first_uid, short_uid = generator.insert(
        [[2], [3]],
        max_tokens=[8, 1],
        caches=[pool.make_cache(0), pool.make_cache(1)],
    )
    finished = set()
    for _ in range(8):
        _, responses = generator.next()
        finished.update(
            response.uid for response in responses if response.finish_reason is not None
        )
        if short_uid in finished:
            break

    assert short_uid in finished
    assert first_uid not in finished

    late_uid = generator.insert(
        [[4]],
        max_tokens=[2],
        caches=[pool.make_cache(1)],
    )[0]
    for _ in range(16):
        _, responses = generator.next()
        finished.update(
            response.uid for response in responses if response.finish_reason is not None
        )
        if {first_uid, late_uid}.issubset(finished):
            break

    assert {first_uid, short_uid, late_uid}.issubset(finished)


def test_remerging_an_existing_fixed_batch_is_allocation_free():
    """Upstream cache helpers may merge an already batched cache again."""
    model = _DenseModel()
    pool = FixedKVCachePool.create(model, context_window=8, slots=2)
    rows = [pool.make_cache(slot) for slot in range(2)]
    model(mx.array([[2]]), cache=rows[0])
    model(mx.array([[3]]), cache=rows[1])
    batch = rows[0][0].merge([rows[0][0], rows[1][0]])
    array_ids = [id(array) for array in pool._arrays]

    remerged = batch.merge([batch])

    assert remerged is batch
    assert [id(array) for array in pool._arrays] == array_ids


def test_remerging_hybrid_fixed_batches_preserves_every_row():
    model = _HybridModel()
    pool = FixedKVCachePool.create(model, context_window=16, slots=2)
    rows = [pool.make_cache(slot) for slot in range(2)]
    model(mx.array([[2, 3]]), cache=rows[0])
    model(mx.array([[4]]), cache=rows[1])
    batch = [left.merge([left, right]) for left, right in zip(*rows)]
    array_ids = [id(array) for array in pool._arrays]

    rotating = batch[0].merge([batch[0]])
    nested = batch[1].merge([batch[1]])
    recurrent = batch[2].merge([batch[2]])

    assert rotating is batch[0]
    assert nested.caches == batch[1].caches
    assert recurrent is batch[2]
    assert rotating.slots == [0, 1]
    assert recurrent.slots == [0, 1]
    assert [id(array) for array in pool._arrays] == array_ids


def test_remerging_deepseek_pooled_fixed_batch_preserves_every_row():
    model = _DeepSeekV4PoolingModel()
    pool = FixedKVCachePool.create(model, context_window=17, slots=2)
    rows = [pool.make_cache(slot) for slot in range(2)]
    model(mx.array([[2, 3, 4, 5]]), cache=rows[0])
    model(mx.array([[6, 7, 8, 9]]), cache=rows[1])
    batch = rows[0][0].merge([rows[0][0], rows[1][0]])
    array_ids = [id(array) for array in pool._arrays]

    remerged = batch.merge([batch])

    assert remerged.caches == batch.caches
    assert all(child.slots == [0, 1] for child in remerged.caches)
    assert [id(array) for array in pool._arrays] == array_ids


def test_staggered_eight_session_completion_compacts_inside_fixed_pool():
    model = _DenseModel()
    pool = FixedKVCachePool.create(model, context_window=32, slots=8)
    array_ids = [id(array) for array in pool._arrays]
    generator = BatchGenerator(
        model,
        max_tokens=8,
        sampler=make_sampler(temp=0),
        prefill_batch_size=1,
        completion_batch_size=8,
    )
    uids = generator.insert(
        [[2], [3], [4], [5], [6], [7], [8], [9]],
        max_tokens=[1, 8, 2, 7, 3, 6, 4, 5],
        caches=[pool.make_cache(slot) for slot in range(8)],
    )
    finished = set()

    for _ in range(24):
        _, responses = generator.next()
        finished.update(
            response.uid for response in responses if response.finish_reason is not None
        )
        assert [id(array) for array in pool._arrays] == array_ids
        if finished == set(uids):
            break

    assert finished == set(uids)
    assert pool.pool_scratch_bytes * 2 == pool.serving_bytes // pool.slots
    assert pool.committed_bytes == pool.serving_bytes + pool.pool_scratch_bytes


def test_empty_decode_batch_rebases_a_late_prefill_before_next_admission():
    model = _DenseModel()
    pool = FixedKVCachePool.create(model, context_window=32, slots=5)
    generator = BatchGenerator(
        model,
        max_tokens=4,
        sampler=make_sampler(temp=0),
        prefill_batch_size=1,
        completion_batch_size=5,
    )

    first_uid = generator.insert(
        [[2]],
        max_tokens=[4],
        caches=[pool.make_cache(4)],
    )[0]
    generator.next()
    assert generator._generation_batch.prompt_cache[0].slot == 0

    second_uid = generator.insert(
        [[3]],
        max_tokens=[1],
        caches=[pool.make_cache(1)],
    )[0]
    finished = set()
    for _ in range(8):
        _, responses = generator.next()
        finished.update(
            response.uid for response in responses if response.finish_reason is not None
        )
        if {first_uid, second_uid}.issubset(finished):
            break

    assert {first_uid, second_uid}.issubset(finished)


def test_slot_capacity_is_a_hard_limit():
    pool = FixedKVCachePool.create(_DenseModel(), context_window=4, slots=1)
    cache = pool.make_cache(0)

    with pytest.raises(FixedKVCacheCapacityError, match="capacity"):
        _DenseModel()(mx.array([[1, 2, 3, 4, 5]]), cache=cache)

    with pytest.raises(FixedKVCacheCapacityError, match="in use"):
        pool.make_cache(1)


def test_restored_full_attention_state_cannot_be_truncated_or_misbatched():
    pool = FixedKVCachePool.create(_DenseModel(), context_window=4, slots=2)
    first = pool.make_cache(0)[0]
    second = pool.make_cache(1)[0]

    oversized = mx.zeros((1, 2, 5, 4))
    with pytest.raises(FixedKVCacheCapacityError, match="5"):
        first.keys = oversized

    batch = first.merge([first, second])
    wrong_rows = mx.zeros((1, 2, 1, 4))
    with pytest.raises(FixedKVCacheError, match="1 rows"):
        batch.values = wrong_rows


def test_full_attention_merge_rejects_duplicate_or_noncompact_pool_rows():
    pool = FixedKVCachePool.create(_DenseModel(), context_window=4, slots=3)
    first = pool.make_cache(0)[0]
    duplicate = pool.make_cache(0)[0]
    third = pool.make_cache(2)[0]

    with pytest.raises(FixedKVCacheError, match="duplicate"):
        first.merge([first, duplicate])

    with pytest.raises(FixedKVCacheError, match="compactly"):
        first.merge([first, third])


def test_materialization_accounting_deficit_fails_before_pool_is_returned(
    monkeypatch,
):
    readings = iter((1_000, 1_000))
    monkeypatch.setattr(mx, "get_active_memory", lambda: next(readings))

    with pytest.raises(FixedKVCacheError, match="did not fully materialize"):
        FixedKVCachePool.create(_DenseModel(), context_window=64, slots=4)


def test_hybrid_rotating_cachelist_and_recurrent_state_use_fixed_rows():
    model = _HybridModel()
    pool = FixedKVCachePool.create(model, context_window=16, slots=2)
    first = pool.make_cache(0)
    second = pool.make_cache(1)

    model(mx.array([[1, 2, 3]]), cache=first)
    model(mx.array([[1, 2]]), cache=second)
    batch = [left.merge([left, right]) for left, right in zip(first, second)]
    model(mx.array([[4], [4]]), cache=batch)
    mx.eval([layer.state for layer in batch])

    assert batch[0].keys.shape[2] == 8
    assert batch[1][0].keys.shape[2] == 256
    assert batch[2][0].shape == (2, 3)


def test_close_releases_the_pool_owners():
    pool = FixedKVCachePool.create(_DenseModel(), context_window=16, slots=1)
    assert pool._arrays
    assert pool._blueprints

    pool.close()

    assert pool._arrays == []
    assert pool._blueprints == []


def test_unknown_live_cache_layout_fails_closed():
    class UnknownCache:
        @property
        def state(self):
            return (mx.ones((1, 1)),)

    class UnknownModel:
        def make_cache(self):
            return [UnknownCache()]

        def __call__(self, tokens, cache):
            return mx.zeros((*tokens.shape, 4))

    with pytest.raises(FixedKVCacheError, match="no fixed-pool adapter"):
        FixedKVCachePool.create(UnknownModel(), context_window=16, slots=1)


def test_deepseek_mla_plan_matches_the_materialized_live_pool(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v3",
                "num_hidden_layers": 2,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "torch_dtype": "float16",
                "max_position_embeddings": 4096,
            }
        )
    )
    plan = estimate_model_memory(
        tmp_path,
        257,
        weights_bytes=0,
        requested_session_slots=3,
        available_memory_bytes=1 << 40,
    )
    pool = FixedKVCachePool.create(
        _DeepSeekMLAModel(),
        context_window=257,
        slots=3,
    )

    assert pool.serving_bytes == plan.fixed_kv_cache_bytes
    assert pool.pool_scratch_bytes == plan.pool_scratch_bytes
    assert pool.committed_bytes == (
        plan.fixed_kv_cache_bytes + plan.pool_scratch_bytes
    )
    assert pool.serving_bytes == 3 * 2 * 512 * 576 * 2


def test_deepseek_dsa_cachelist_plan_matches_the_materialized_live_pool(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v32",
                "num_hidden_layers": 2,
                "kv_lora_rank": 16,
                "qk_rope_head_dim": 4,
                "index_head_dim": 8,
                "torch_dtype": "float16",
                "max_position_embeddings": 4096,
            }
        )
    )
    plan = estimate_model_memory(
        tmp_path,
        257,
        weights_bytes=0,
        requested_session_slots=2,
        available_memory_bytes=1 << 40,
    )
    pool = FixedKVCachePool.create(
        _DeepSeekDSAModel(),
        context_window=257,
        slots=2,
    )

    assert pool.serving_bytes == plan.fixed_kv_cache_bytes
    assert pool.pool_scratch_bytes == plan.pool_scratch_bytes
    assert pool.committed_bytes == (
        plan.fixed_kv_cache_bytes + plan.pool_scratch_bytes
    )
    assert pool.serving_bytes == 2 * 2 * 512 * (16 + 4 + 8) * 2


def test_deepseek_v4_pooling_plan_matches_materialized_batch_lifecycle(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "num_hidden_layers": 1,
                "head_dim": 4,
                "index_head_dim": 2,
                "sliding_window": 8,
                "compress_ratios": [4],
                "torch_dtype": "float16",
                "max_position_embeddings": 4096,
            }
        )
    )
    plan = estimate_model_memory(
        tmp_path,
        17,
        weights_bytes=0,
        requested_session_slots=2,
        available_memory_bytes=1 << 40,
    )
    model = _DeepSeekV4PoolingModel()
    pool = FixedKVCachePool.create(model, context_window=17, slots=2)
    array_ids = [id(array) for array in pool._arrays]
    first = pool.make_cache(0)
    second = pool.make_cache(1)

    model(mx.array([[1, 2, 3, 4, 5]]), cache=first)
    model(mx.array([[1, 2, 3, 4]]), cache=second)
    batch = [left.merge([left, right]) for left, right in zip(first, second)]
    model(mx.array([[6], [5]]), cache=batch)
    for layer in batch:
        layer.filter([1])
    extracted = batch[0].extract(0)
    mx.eval(extracted.state)

    assert [id(array) for array in pool._arrays] == array_ids
    assert pool.serving_bytes == plan.fixed_kv_cache_bytes
    assert pool.pool_scratch_bytes == plan.pool_scratch_bytes
    assert pool.committed_bytes == plan.fixed_kv_cache_bytes + plan.pool_scratch_bytes
    assert extracted[1].size() == 1
    assert extracted[1].remainder == 1


def test_deepseek_v4_single_slot_commits_rollback_scratch(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "num_hidden_layers": 1,
                "head_dim": 4,
                "index_head_dim": 2,
                "sliding_window": 8,
                "compress_ratios": [4],
                "torch_dtype": "float16",
                "max_position_embeddings": 4096,
            }
        )
    )
    plan = estimate_model_memory(
        tmp_path,
        17,
        weights_bytes=0,
        requested_session_slots=1,
        available_memory_bytes=1 << 40,
    )
    pool = FixedKVCachePool.create(
        _DeepSeekV4PoolingModel(), context_window=17, slots=1
    )

    assert pool.pool_scratch_bytes > 0
    assert pool.pool_scratch_bytes == plan.pool_scratch_bytes
    assert pool.committed_bytes == (
        plan.fixed_kv_cache_bytes + plan.pool_scratch_bytes
    )


def test_deepseek_v4_fixed_pool_rollback_preserves_aliased_remainder():
    from omlx.patches.mlx_lm_mtp import cache_rollback

    def push(cache, values, offset):
        raw = mx.array(values, dtype=mx.float16)[None, :, None]
        kv = mx.broadcast_to(raw, (1, len(values), 8))
        gate = kv * mx.array(0.5, dtype=mx.float16)
        ready, _ready_gate, _ = cache.accumulate_windows(kv, gate, offset)
        if ready.shape[1]:
            pooled = mx.unflatten(ready, 1, (-1, cache.ratio)).mean(axis=2)
            pooled = pooled[..., :4]
        else:
            pooled = mx.zeros((1, 0, 4), dtype=mx.float16)
        cache.update_and_fetch(pooled)

    pool = FixedKVCachePool.create(
        _DeepSeekV4PoolingModel(), context_window=17, slots=1
    )
    fixed = pool.make_cache(0)[0][1]
    dynamic = PoolingCache(4)
    push(fixed, [1, 2, 3], 0)
    push(dynamic, [1, 2, 3], 0)

    cache_rollback.set_undo_armed(True)
    try:
        push(fixed, [4], 3)
        push(dynamic, [4], 3)
        assert fixed.trim(1) == 1
        assert dynamic.trim(1) == 1
        fixed_state = fixed.state
        dynamic_state = dynamic.state
        mx.eval(*[value for value in fixed_state if value is not None])
        mx.eval(*[value for value in dynamic_state if value is not None])
    finally:
        cache_rollback.set_undo_armed(False)

    assert fixed.remainder == dynamic.remainder == 3
    for fixed_value, dynamic_value in zip(fixed_state, dynamic_state):
        if fixed_value is None or dynamic_value is None:
            assert fixed_value is dynamic_value
        else:
            assert bool(mx.allclose(fixed_value, dynamic_value))
    assert bool(mx.allclose(fixed_state[0][:, :, 0], mx.array([[1, 2, 3]])))


def test_deepseek_v4_fixed_batch_pooling_matches_dynamic_storage():
    def prime(cache, values):
        raw = mx.array(values, dtype=mx.float16)[None, :, None]
        kv = mx.broadcast_to(raw, (1, len(values), 8))
        gate = kv * mx.array(0.5, dtype=mx.float16)
        ready, _, _ = cache.accumulate_windows(kv, gate, 0)
        assert int(ready.shape[1]) == 0

    pool = FixedKVCachePool.create(
        _DeepSeekV4PoolingModel(), context_window=17, slots=2
    )
    fixed_rows = [pool.make_cache(index)[0][1] for index in range(2)]
    dynamic_rows = [PoolingCache(4), PoolingCache(4)]
    for fixed, dynamic, values in zip(
        fixed_rows, dynamic_rows, ([1, 2, 3], [11, 12, 13])
    ):
        prime(fixed, values)
        prime(dynamic, values)

    fixed = fixed_rows[0].merge(fixed_rows)
    dynamic = BatchPoolingCache.merge(dynamic_rows)
    for values in ([4, 14], [5, 15]):
        raw = mx.array(values, dtype=mx.float16)[:, None, None]
        kv = mx.broadcast_to(raw, (2, 1, 8))
        gate = kv * mx.array(0.5, dtype=mx.float16)
        fixed_ready, _, _ = fixed.accumulate_windows(kv, gate, mx.array([3, 3]))
        dynamic_ready, _, _ = dynamic.accumulate_windows(
            kv, gate, mx.array([3, 3])
        )
        assert bool(mx.allclose(fixed_ready, dynamic_ready))
        if fixed_ready.shape[1]:
            fixed_px = mx.unflatten(fixed_ready, 1, (-1, 4)).mean(axis=2)[
                ..., :4
            ]
            dynamic_px = mx.unflatten(dynamic_ready, 1, (-1, 4)).mean(axis=2)[
                ..., :4
            ]
        else:
            fixed_px = dynamic_px = mx.zeros((2, 0, 4), dtype=mx.float16)
        fixed.update_and_fetch(fixed_px)
        dynamic.update_and_fetch(dynamic_px)

    fixed.filter([1])
    dynamic.filter([1])
    fixed_state = fixed.state
    dynamic_state = dynamic.state
    mx.eval(*fixed_state, *dynamic_state)
    for fixed_value, dynamic_value in zip(fixed_state, dynamic_state):
        assert bool(mx.allclose(fixed_value, dynamic_value))


def test_fixed_batch_rejects_reordered_or_duplicate_survivors():
    pool = FixedKVCachePool.create(_DenseModel(), context_window=16, slots=3)
    rows = [pool.make_cache(index) for index in range(3)]
    batch = [
        first.merge([first, second, third])
        for first, second, third in zip(*rows)
    ]

    with pytest.raises(FixedKVCacheError, match="unique ascending"):
        batch[0].filter([2, 0])
    with pytest.raises(FixedKVCacheError, match="unique ascending"):
        batch[0].filter([0, 0])


def test_qwen_gated_delta_hybrid_plan_matches_live_arrays_and_kv(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 4,
                "full_attention_interval": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_size": 16,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 3,
                "linear_value_head_dim": 2,
                "linear_conv_kernel_dim": 4,
                "torch_dtype": "float16",
                "max_position_embeddings": 4096,
            }
        )
    )
    plan = estimate_model_memory(
        tmp_path,
        257,
        weights_bytes=0,
        requested_session_slots=3,
        available_memory_bytes=1 << 40,
    )
    pool = FixedKVCachePool.create(
        _QwenGatedDeltaModel(),
        context_window=257,
        slots=3,
    )

    assert {tensor.cache_kind for tensor in plan.cache_tensors} == {
        "ArraysCache",
        "KVCache",
    }
    assert pool.serving_bytes == plan.fixed_kv_cache_bytes
    assert pool.pool_scratch_bytes == plan.pool_scratch_bytes
    assert pool.committed_bytes == (
        plan.fixed_kv_cache_bytes + plan.pool_scratch_bytes
    )


def test_nemotron_h_plan_matches_live_mamba_and_attention_cache(tmp_path):
    from mlx_lm.models import nemotron_h

    config = {
        "model_type": "nemotron_h",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "max_position_embeddings": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "attention_bias": False,
        "mamba_num_heads": 4,
        "mamba_head_dim": 16,
        "mamba_proj_bias": False,
        "ssm_state_size": 8,
        "conv_kernel": 4,
        "n_groups": 2,
        "mlp_bias": False,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "hybrid_override_pattern": ["M", "*"],
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "moe_shared_expert_intermediate_size": 32,
        "n_shared_experts": 1,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "mamba_ssm_cache_dtype": "float32",
        "dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    plan = estimate_model_memory(
        tmp_path,
        17,
        weights_bytes=0,
        requested_session_slots=2,
        available_memory_bytes=1 << 40,
    )
    model = nemotron_h.Model(nemotron_h.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    pool = FixedKVCachePool.create(model, context_window=17, slots=2)

    assert {tensor.cache_kind for tensor in plan.cache_tensors} == {
        "ArraysCache",
        "KVCache",
    }
    assert pool.serving_bytes == plan.fixed_kv_cache_bytes
    assert pool.pool_scratch_bytes == plan.pool_scratch_bytes
    assert pool.committed_bytes == plan.fixed_kv_cache_bytes + plan.pool_scratch_bytes


@pytest.mark.parametrize(
    ("module_name", "config"),
    [
        (
            "mamba",
            {
                "model_type": "mamba",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 12,
                "state_size": 3,
                "num_hidden_layers": 2,
                "conv_kernel": 4,
                "use_bias": False,
                "use_conv_bias": False,
                "time_step_rank": 1,
                "tie_word_embeddings": True,
            },
        ),
        (
            "mamba2",
            {
                "model_type": "mamba2",
                "num_heads": 2,
                "head_dim": 4,
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 8,
                "state_size": 3,
                "num_hidden_layers": 2,
                "layer_norm_epsilon": 1e-5,
                "conv_kernel": 4,
                "n_groups": 1,
                "use_bias": False,
                "use_conv_bias": False,
                "tie_word_embeddings": True,
                "time_step_limit": [0.001, 100.0],
                "time_step_rank": 1,
            },
        ),
        (
            "jamba",
            {
                "model_type": "jamba",
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "attn_layer_offset": 0,
                "attn_layer_period": 2,
                "expert_layer_offset": 0,
                "expert_layer_period": 1,
                "mamba_d_conv": 4,
                "mamba_d_state": 3,
                "mamba_expand": 2,
                "num_experts": 1,
                "num_experts_per_tok": 1,
                "rms_norm_eps": 1e-5,
                "max_position_embeddings": 64,
                "vocab_size": 32,
                "layers_block_type": ["attention", "mamba"],
            },
        ),
        (
            "lfm2",
            {
                "model_type": "lfm2",
                "vocab_size": 32,
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 64,
                "norm_eps": 1e-5,
                "conv_bias": False,
                "conv_L_cache": 4,
                "block_dim": 8,
                "block_ff_dim": 12,
                "block_multiple_of": 2,
                "block_ffn_dim_multiplier": 1.0,
                "block_auto_adjust_ff_dim": False,
                "full_attn_idxs": [1],
                "layer_types": ["conv", "full_attention"],
            },
        ),
        (
            "recurrent_gemma",
            {
                "model_type": "recurrent_gemma",
                "attention_bias": False,
                "conv1d_width": 4,
                "hidden_size": 8,
                "intermediate_size": 12,
                "logits_soft_cap": 0,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000,
                "attention_window_size": 8,
                "vocab_size": 32,
                "block_types": ["recurrent", "attention"],
            },
        ),
        (
            "rwkv7",
            {
                "model_type": "rwkv7",
                "vocab_size": 64,
                "hidden_size": 64,
                "intermediate_size": 96,
                "norm_eps": 1e-5,
                "head_dim": 32,
                "num_hidden_layers": 2,
                "a_low_rank_dim": 4,
                "v_low_rank_dim": 4,
                "gate_low_rank_dim": 4,
                "decay_low_rank_dim": 4,
                "tie_word_embeddings": True,
            },
        ),
    ],
)
def test_recurrent_family_plan_matches_live_pool(
    tmp_path, module_name, config
):
    module = importlib.import_module(f"mlx_lm.models.{module_name}")
    config = {**config, "torch_dtype": "float16"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = module.Model(module.ModelArgs.from_dict(config))
        model.set_dtype(mx.float16)
        plan = estimate_model_memory(
            tmp_path,
            9,
            weights_bytes=0,
            requested_session_slots=2,
            available_memory_bytes=1 << 40,
        )
        pool = FixedKVCachePool.create(model, context_window=9, slots=2)
        assert pool.serving_bytes == plan.fixed_kv_cache_bytes
        assert pool.pool_scratch_bytes == plan.pool_scratch_bytes
        assert pool.committed_bytes == (
            plan.fixed_kv_cache_bytes + plan.pool_scratch_bytes
        )
        pool.close()
    finally:
        mx.set_default_device(previous_device)


def test_longcat_ngram_history_and_nemotron_linear_placeholder_are_fixed(tmp_path):
    longcat = importlib.import_module("mlx_lm.models.longcat_flash_ngram")
    config = {
        "model_type": "longcat_flash_ngram",
        "hidden_size": 16,
        "ffn_hidden_size": 24,
        "moe_topk": 1,
        "expert_ffn_hidden_size": 8,
        "n_routed_experts": 1,
        "zero_expert_num": 0,
        "num_layers": 1,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "num_attention_heads": 2,
        "kv_lora_rank": 8,
        "q_lora_rank": 4,
        "qk_rope_head_dim": 4,
        "qk_nope_head_dim": 8,
        "v_head_dim": 8,
        "routed_scaling_factor": 1.0,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "mla_scale_q_lora": True,
        "mla_scale_kv_lora": True,
        "attention_bias": False,
        "ngram_vocab_size_ratio": 2,
        "emb_neighbor_num": 4,
        "emb_split_num": 2,
        "torch_dtype": "float16",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = longcat.Model(longcat.ModelArgs.from_dict(config))
        model.set_dtype(mx.float16)
        plan = estimate_model_memory(
            tmp_path,
            9,
            weights_bytes=0,
            requested_session_slots=2,
            available_memory_bytes=1 << 40,
        )
        pool = FixedKVCachePool.create(model, context_window=9, slots=2)
        assert plan.cache_tensors[0].shape == (1, 3)
        assert pool.committed_bytes == (
            plan.fixed_kv_cache_bytes + plan.pool_scratch_bytes
        )
        pool.close()

        nemotron = importlib.import_module("mlx_lm.models.nemotron-nas")
        nas_config = {
            "model_type": "nemotron-nas",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "rms_norm_eps": 1e-5,
            "vocab_size": 32,
            "block_configs": [
                {
                    "attention": {"replace_with_linear": True},
                    "ffn": {"replace_with_linear": True},
                },
                {
                    "attention": {"replace_with_linear": True},
                    "ffn": {"replace_with_linear": True},
                },
            ],
            "max_position_embeddings": 64,
            "tie_word_embeddings": True,
            "torch_dtype": "float16",
        }
        (tmp_path / "config.json").write_text(json.dumps(nas_config))
        nas_model = nemotron.Model(nemotron.ModelArgs.from_dict(nas_config))
        nas_model.set_dtype(mx.float16)
        nas_plan = estimate_model_memory(
            tmp_path,
            9,
            weights_bytes=0,
            requested_session_slots=2,
            available_memory_bytes=1 << 40,
        )
        nas_pool = FixedKVCachePool.create(nas_model, context_window=9, slots=2)
        assert type(nas_pool.make_cache(0)[0]).__name__ == "FixedNullCache"
        assert nas_plan.fixed_kv_cache_bytes == 0
        assert nas_pool.committed_bytes == 0
        assert nas_pool.committed_bytes == (
            nas_plan.fixed_kv_cache_bytes + nas_plan.pool_scratch_bytes
        )
        nas_pool.close()
    finally:
        mx.set_default_device(previous_device)
