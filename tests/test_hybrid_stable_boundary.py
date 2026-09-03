# SPDX-License-Identifier: Apache-2.0
"""Lossless exact boundary for finalized ArraysCache + KVCache graphs."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

from omlx.cache import exact_boundary
from omlx.cache.exact_boundary import (
    materialize_hybrid_arrays_kv_boundary,
    plan_hybrid_arrays_kv_boundary,
)
from omlx.cache.exact_resident import ExactResidentPrefixCache
from omlx.cache.type_handlers import SizedArraysCache
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler


def _arrays_cache(*, batch=1):
    cache = ArraysCache(size=2)
    cache.cache = [
        mx.arange(batch * 3 * 12, dtype=mx.float32).reshape(batch, 3, 12),
        mx.arange(batch * 2 * 4 * 4, dtype=mx.float32).reshape(batch, 2, 4, 4),
    ]
    return cache


def _sized_arrays_cache(*, token_count=4, batch=1, inner=None):
    return SizedArraysCache(
        inner or _arrays_cache(batch=batch),
        token_count=token_count,
    )


def _kv_cache(token_count=4):
    cache = KVCache()
    cache.update_and_fetch(
        mx.arange(2 * token_count * 4, dtype=mx.float32).reshape(
            1, 2, token_count, 4
        ),
        mx.arange(2 * token_count * 5, dtype=mx.float32).reshape(
            1, 2, token_count, 5
        ),
    )
    mx.eval(cache.keys, cache.values)
    return cache


def _graph(token_count=4):
    return [_arrays_cache(), _arrays_cache(), _kv_cache(token_count)]


def _restored_graph(token_count=4):
    return [
        _sized_arrays_cache(token_count=token_count),
        _sized_arrays_cache(token_count=token_count),
        _kv_cache(token_count),
    ]


def _qwen38_shape_graph(token_count=4):
    return [
        *[_arrays_cache() for _ in range(48)],
        *[_kv_cache(token_count) for _ in range(16)],
    ]


def _scheduler(*, block_size=4):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = SimpleNamespace()
    scheduler._vlm_mtp_drafter = None
    scheduler._exact_resident_cache = ExactResidentPrefixCache(
        max_entries=2,
        max_bytes=1 << 30,
    )
    scheduler.running = {}
    scheduler.waiting = deque()
    scheduler.prefilling = deque()
    scheduler.config = SimpleNamespace(paged_cache_block_size=block_size)
    scheduler._stream = mx.default_stream(mx.default_device())
    scheduler._memory_abort_limit_bytes = 1 << 40
    scheduler._memory_metal_cap_bytes = 1 << 40
    scheduler._memory_static_ceiling_bytes = 1 << 40
    scheduler._current_usage_bytes = MagicMock(return_value=0)
    scheduler._exact_resident_pool_invalidations = 0
    scheduler._exact_resident_pool_invalidated_bytes = 0
    scheduler._exact_resident_pool_invalidation_ms = 0.0
    return scheduler


def _request(tokens=None):
    tokens = list(tokens or [1, 2, 3, 4, 99])
    request = Request("hybrid-stable", tokens, SamplingParams())
    request.prompt_token_ids = tokens
    request.num_prompt_tokens = len(tokens)
    request._exact_resident_durable_fallback_tokens = 4
    return request


def test_hybrid_provider_detaches_every_recurrent_and_kv_array():
    source = _graph()
    source_arrays = [
        array
        for cache in source
        for array in (
            cache.cache
            if type(cache) is ArraysCache
            else (cache.keys, cache.values)
        )
    ]
    source_values = [mx.array(array) for array in source_arrays]
    mx.eval(*source_values)

    plan = plan_hybrid_arrays_kv_boundary(
        source,
        source_tokens=5,
        target_tokens=4,
    )
    assert plan is not None
    detached = materialize_hybrid_arrays_kv_boundary(
        plan,
        stream=mx.default_stream(mx.default_device()),
    )

    assert detached is not None
    assert detached.token_count == 4
    assert detached.nbytes == plan.estimated_nbytes
    assert [type(cache) for cache in detached.cache] == [
        ArraysCache,
        ArraysCache,
        KVCache,
    ]
    detached_arrays = [
        array
        for cache in detached.cache
        for array in (
            cache.cache
            if type(cache) is ArraysCache
            else (cache.keys, cache.values)
        )
    ]
    assert len(detached_arrays) == len(source_arrays)
    for source_array, original, clone in zip(
        source_arrays,
        source_values,
        detached_arrays,
    ):
        assert clone is not source_array
        assert mx.array_equal(source_array, original).item()
        expected = original
        if clone.shape != original.shape and clone.ndim == 4:
            expected = original[:, :, : clone.shape[2], :]
        assert mx.array_equal(clone, expected).item()
    assert detached.cache[-1].offset == 4
    assert detached.cache[-1].keys.shape[2] == 4

    detached.cache[0].cache[0][0, 0, 0] = 9_999
    detached.cache[-1].keys[0, 0, 0, 0] = 8_888
    mx.eval(detached.cache[0].cache[0], detached.cache[-1].keys)
    assert source[0].cache[0][0, 0, 0].item() != 9_999
    assert source[-1].keys[0, 0, 0, 0].item() != 8_888


def test_qwen38_48_arrays_16_kv_surrogate_graph_is_fully_detached():
    source = _qwen38_shape_graph()
    plan = plan_hybrid_arrays_kv_boundary(
        source,
        source_tokens=5,
        target_tokens=4,
    )
    detached = materialize_hybrid_arrays_kv_boundary(
        plan,
        stream=mx.default_stream(mx.default_device()),
    )

    assert detached is not None
    assert sum(type(cache) is ArraysCache for cache in detached.cache) == 48
    assert sum(type(cache) is KVCache for cache in detached.cache) == 16
    assert len(detached.arrays) == 48 * 2 + 16 * 2


def test_restored_sized_arrays_graph_preserves_wrapper_and_inner_ownership():
    source = _restored_graph()
    plan = plan_hybrid_arrays_kv_boundary(
        source,
        source_tokens=5,
        target_tokens=4,
    )
    detached = materialize_hybrid_arrays_kv_boundary(
        plan,
        stream=mx.default_stream(mx.default_device()),
    )

    assert detached is not None
    for original, clone in zip(source[:2], detached.cache[:2]):
        assert type(clone) is SizedArraysCache
        assert clone is not original
        assert clone._inner is not original._inner
        assert clone._token_count == original._token_count == 4
        assert all(
            cloned is not source_array
            and mx.array_equal(cloned, source_array).item()
            for cloned, source_array in zip(clone.cache, original.cache)
        )
    detached.cache[0].cache[0][0, 0, 0] = 9_999
    mx.eval(detached.cache[0].cache[0])
    assert source[0].cache[0][0, 0, 0].item() != 9_999


def test_hybrid_provider_requires_both_exact_finalized_b1_cache_classes():
    assert (
        plan_hybrid_arrays_kv_boundary(
            [_arrays_cache()],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )

    wrong_count = _sized_arrays_cache(token_count=3)
    assert (
        plan_hybrid_arrays_kv_boundary(
            [wrong_count, _kv_cache()],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )

    class SizedSubclass(SizedArraysCache):
        pass

    assert (
        plan_hybrid_arrays_kv_boundary(
            [SizedSubclass(_arrays_cache(), token_count=4), _kv_cache()],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )

    class InnerSubclass(ArraysCache):
        pass

    inner_subclass = InnerSubclass(size=2)
    inner_subclass.cache = list(_arrays_cache().cache)
    assert (
        plan_hybrid_arrays_kv_boundary(
            [
                _sized_arrays_cache(token_count=4, inner=inner_subclass),
                _kv_cache(),
            ],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )

    assert (
        plan_hybrid_arrays_kv_boundary(
            [_sized_arrays_cache(token_count=4, batch=2), _kv_cache()],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )

    first = _sized_arrays_cache(token_count=4)
    second = _sized_arrays_cache(token_count=4)
    second._inner.cache[0] = first._inner.cache[0]
    assert (
        plan_hybrid_arrays_kv_boundary(
            [first, second, _kv_cache()],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )

    with_none = _arrays_cache()
    with_none.cache[1] = None
    assert (
        plan_hybrid_arrays_kv_boundary(
            [with_none, _kv_cache()],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )
    assert (
        plan_hybrid_arrays_kv_boundary(
            [_kv_cache()],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )

    class ArraysSubclass(ArraysCache):
        pass

    subclass = ArraysSubclass(size=2)
    subclass.cache = list(_arrays_cache().cache)
    assert (
        plan_hybrid_arrays_kv_boundary(
            [subclass, _kv_cache()],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )

    active = _arrays_cache()
    active.lengths = mx.array([1])
    assert (
        plan_hybrid_arrays_kv_boundary(
            [active, _kv_cache()],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )
    padded = _arrays_cache()
    padded.left_padding = mx.array([1])
    assert (
        plan_hybrid_arrays_kv_boundary(
            [padded, _kv_cache()],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )
    assert (
        plan_hybrid_arrays_kv_boundary(
            [_arrays_cache(batch=2), _kv_cache()],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )
    assert (
        plan_hybrid_arrays_kv_boundary(
            [CacheList(_arrays_cache(), _kv_cache())],
            source_tokens=5,
            target_tokens=4,
        )
        is None
    )


def test_hybrid_provider_fails_closed_on_alias_and_source_race(monkeypatch):
    source = _graph()
    plan = plan_hybrid_arrays_kv_boundary(
        source,
        source_tokens=5,
        target_tokens=4,
    )
    assert plan is not None
    monkeypatch.setattr(
        exact_boundary,
        "_copy_exact_array",
        lambda array: array,
    )
    assert (
        materialize_hybrid_arrays_kv_boundary(
            plan,
            stream=mx.default_stream(mx.default_device()),
        )
        is None
    )

    source = _graph()
    plan = plan_hybrid_arrays_kv_boundary(
        source,
        source_tokens=5,
        target_tokens=4,
    )
    source[0].lengths = mx.array([1])
    copy_spy = MagicMock(wraps=exact_boundary._copy_exact_array)
    monkeypatch.setattr(exact_boundary, "_copy_exact_array", copy_spy)
    assert (
        materialize_hybrid_arrays_kv_boundary(
            plan,
            stream=mx.default_stream(mx.default_device()),
        )
        is None
    )
    copy_spy.assert_not_called()

    source = _graph()
    plan = plan_hybrid_arrays_kv_boundary(
        source,
        source_tokens=5,
        target_tokens=4,
    )
    monkeypatch.setattr(
        exact_boundary,
        "_copy_exact_array",
        MagicMock(side_effect=RuntimeError("synthetic allocation failure")),
    )
    assert (
        materialize_hybrid_arrays_kv_boundary(
            plan,
            stream=mx.default_stream(mx.default_device()),
        )
        is None
    )


def test_scheduler_stages_hybrid_n_minus_one_and_exact_next_prompt_claims_it(caplog):
    scheduler = _scheduler()
    request = _request()
    caplog.set_level("INFO")

    assert scheduler._stage_stable_prompt_boundary(request, _graph())
    candidate = request._stable_prompt_resident_candidate
    assert candidate[0] == [1, 2, 3, 4]
    assert "provider=hybrid-arrays-kv-v1" in caplog.text
    assert scheduler._publish_stable_prompt_boundary(request)

    # Qwen3.8's observed template appends to the prior rendered input. The
    # exact N-1 candidate is therefore a prefix; an incompatible template is
    # still only a safe miss because acquisition compares every token.
    hit = scheduler._exact_resident_cache.acquire_prefix(
        [1, 2, 3, 4, 99, 42, 43]
    )
    assert hit is not None
    assert hit.cached_tokens == 4


def test_scheduler_stages_durable_restored_sized_arrays_topology():
    scheduler = _scheduler()
    request = _request()

    assert scheduler._stage_stable_prompt_boundary(request, _restored_graph())
    candidate = request._stable_prompt_resident_candidate
    assert candidate[0] == [1, 2, 3, 4]
    assert all(type(cache) is SizedArraysCache for cache in candidate[1][:2])
    assert scheduler._publish_stable_prompt_boundary(request)


def test_scheduler_hybrid_fails_before_provider_for_spec_media_and_b2(monkeypatch):
    import omlx.cache.exact_boundary as provider

    scheduler = _scheduler()
    request = _request()
    plan_spy = MagicMock(wraps=provider.plan_hybrid_arrays_kv_boundary)
    monkeypatch.setattr(provider, "plan_hybrid_arrays_kv_boundary", plan_spy)

    scheduler.running["other"] = object()
    assert not scheduler._stage_stable_prompt_boundary(request, _graph())
    scheduler.running.clear()
    request.images = [object()]
    assert not scheduler._stage_stable_prompt_boundary(request, _graph())
    request.images = None
    scheduler.model._omlx_mtp_decode_enabled = True
    assert not scheduler._stage_stable_prompt_boundary(request, _graph())
    plan_spy.assert_not_called()


def test_hidden_prewarm_skips_when_hybrid_n_minus_one_is_resident():
    scheduler = _scheduler()
    request = _request()
    assert scheduler._stage_stable_prompt_boundary(request, _graph())
    assert scheduler._publish_stable_prompt_boundary(request)
    scheduler.has_requests = MagicMock(return_value=False)
    scheduler._others_decoding = MagicMock(return_value=False)
    scheduler.tokenizer = SimpleNamespace(encode=lambda _prompt: request.prompt_token_ids)

    result = scheduler.prewarm_prompt_tail(request.prompt_token_ids, min_tokens=2)

    assert result["reason"] == "stable-boundary-already-resident"
    assert scheduler._exact_resident_cache.stats()["entries"] == 1


def test_tiny_qwen35_hybrid_clone_continuation_matches_canonical_exactly():
    language = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    config_mod = pytest.importorskip("mlx_vlm.models.qwen3_5.config")
    config = config_mod.TextConfig(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=1,
        linear_num_key_heads=1,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=128,
        head_dim=32,
        rope_parameters={
            "type": "default",
            "mrope_section": [3, 3, 2],
            "rope_theta": 10_000,
            "partial_rotary_factor": 0.25,
        },
        full_attention_interval=4,
    )
    model = language.LanguageModel(config)
    source = model.make_cache()
    model.model(mx.array([[1, 2, 3, 4, 5]]), cache=source)
    mx.eval(*[array for cache in source for array in cache.state if array is not None])
    plan = plan_hybrid_arrays_kv_boundary(
        source,
        source_tokens=6,
        target_tokens=5,
    )
    detached = materialize_hybrid_arrays_kv_boundary(
        plan,
        stream=mx.default_stream(mx.default_device()),
    )
    assert detached is not None

    continuation = mx.array([[6]])
    canonical = model.model(continuation, cache=source)
    restored = model.model(continuation, cache=detached.cache)
    all_arrays = [canonical, restored]
    all_arrays.extend(
        array
        for graph in (source, detached.cache)
        for cache in graph
        for array in cache.state
        if array is not None
    )
    mx.eval(*all_arrays)
    assert mx.array_equal(canonical, restored).item()
    for canonical_cache, restored_cache in zip(source, detached.cache):
        assert type(canonical_cache) is type(restored_cache)
        for canonical_state, restored_state in zip(
            canonical_cache.state,
            restored_cache.state,
        ):
            assert mx.array_equal(canonical_state, restored_state).item()


def test_tiny_qwen35_restored_wrapper_continuation_matches_canonical_exactly():
    language = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    config_mod = pytest.importorskip("mlx_vlm.models.qwen3_5.config")
    config = config_mod.TextConfig(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=1,
        linear_num_key_heads=1,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=128,
        head_dim=32,
        rope_parameters={
            "type": "default",
            "mrope_section": [3, 3, 2],
            "rope_theta": 10_000,
            "partial_rotary_factor": 0.25,
        },
        full_attention_interval=4,
    )
    model = language.LanguageModel(config)
    raw = model.make_cache()
    model.model(mx.array([[1, 2, 3, 4, 5]]), cache=raw)
    source = [
        SizedArraysCache(cache, token_count=5)
        if type(cache) is ArraysCache
        else cache
        for cache in raw
    ]
    mx.eval(
        *[
            array
            for cache in source
            for array in cache.state
            if array is not None
        ]
    )
    plan = plan_hybrid_arrays_kv_boundary(
        source,
        source_tokens=6,
        target_tokens=5,
    )
    detached = materialize_hybrid_arrays_kv_boundary(
        plan,
        stream=mx.default_stream(mx.default_device()),
    )
    assert detached is not None

    continuation = mx.array([[6]])
    canonical = model.model(continuation, cache=source)
    restored = model.model(continuation, cache=detached.cache)
    all_arrays = [canonical, restored]
    all_arrays.extend(
        array
        for graph in (source, detached.cache)
        for cache in graph
        for array in cache.state
        if array is not None
    )
    mx.eval(*all_arrays)
    assert mx.array_equal(canonical, restored).item()
    for canonical_cache, restored_cache in zip(source, detached.cache):
        assert type(canonical_cache) is type(restored_cache)
        if type(canonical_cache) is SizedArraysCache:
            assert canonical_cache._token_count == restored_cache._token_count == 6
            assert canonical_cache._inner is not restored_cache._inner
        for canonical_state, restored_state in zip(
            canonical_cache.state,
            restored_cache.state,
        ):
            assert mx.array_equal(canonical_state, restored_state).item()
