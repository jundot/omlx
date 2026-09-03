# SPDX-License-Identifier: Apache-2.0
"""V1 generic stable-boundary provider for exact plain KVCache graphs."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)

from omlx.cache import exact_boundary
from omlx.cache.exact_boundary import (
    materialize_plain_kv_boundary,
    plan_plain_kv_boundary,
)
from omlx.cache.exact_resident import ExactResidentPrefixCache
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler


def _plain_kv(token_count=4, *, heads=2, key_dim=3, value_dim=4):
    cache = KVCache()
    keys = mx.arange(
        heads * token_count * key_dim,
        dtype=mx.float32,
    ).reshape(1, heads, token_count, key_dim)
    values = mx.arange(
        heads * token_count * value_dim,
        dtype=mx.float32,
    ).reshape(1, heads, token_count, value_dim) + 100
    cache.update_and_fetch(keys, values)
    mx.eval(cache.keys, cache.values)
    return cache


def _scheduler(*, slots=2, max_bytes=1 << 30, block_size=4):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = SimpleNamespace()
    scheduler._vlm_mtp_drafter = None
    scheduler._exact_resident_cache = ExactResidentPrefixCache(
        max_entries=slots,
        max_bytes=max_bytes,
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
    request = Request("plain-stable", tokens, SamplingParams())
    request.prompt_token_ids = tokens
    request.num_prompt_tokens = len(tokens)
    request._exact_resident_durable_fallback_tokens = 4
    return request


def test_plain_kv_provider_copies_only_logical_boundary_and_preserves_source():
    source = [_plain_kv(), _plain_kv(key_dim=5, value_dim=6)]
    source_arrays = [(cache.keys, cache.values) for cache in source]
    source_states = [
        (mx.array(cache.state[0]), mx.array(cache.state[1])) for cache in source
    ]
    mx.eval(*(array for pair in source_states for array in pair))

    plan = plan_plain_kv_boundary(
        source,
        source_tokens=5,
        source_cache_tokens=4,
        target_tokens=4,
    )
    assert plan is not None
    # The source has 256-token allocation slabs; the plan charges only N-1.
    assert plan.estimated_nbytes < sum(cache.nbytes for cache in source)
    detached = materialize_plain_kv_boundary(
        plan,
        stream=mx.default_stream(mx.default_device()),
    )

    assert detached is not None
    assert detached.token_count == 4
    assert detached.nbytes == plan.estimated_nbytes
    for index, clone in enumerate(detached.cache):
        original = source[index]
        assert type(clone) is KVCache
        assert clone.offset == 4
        assert clone.keys.shape[2] == clone.values.shape[2] == 4
        assert clone.keys is not original.keys
        assert clone.values is not original.values
        assert original.keys is source_arrays[index][0]
        assert original.values is source_arrays[index][1]
        assert original.offset == 4
        assert mx.array_equal(original.state[0], source_states[index][0]).item()
        assert mx.array_equal(original.state[1], source_states[index][1]).item()
        assert mx.array_equal(clone.keys, source_states[index][0]).item()
        assert mx.array_equal(clone.values, source_states[index][1]).item()


def test_plain_kv_detached_boundary_has_canonical_next_token_kickoff():
    prefix = _plain_kv(13)
    plan = plan_plain_kv_boundary(
        [prefix],
        source_tokens=14,
        source_cache_tokens=13,
        target_tokens=8,
    )
    detached = materialize_plain_kv_boundary(
        plan,
        stream=mx.default_stream(mx.default_device()),
    )
    assert detached is not None

    next_keys = mx.full((1, 2, 1, 3), 777, dtype=mx.float32)
    next_values = mx.full((1, 2, 1, 4), 888, dtype=mx.float32)
    clone = detached.cache[0]
    clone.update_and_fetch(
        mx.concatenate([prefix.state[0][:, :, 8:, :], next_keys], axis=2),
        mx.concatenate([prefix.state[1][:, :, 8:, :], next_values], axis=2),
    )

    canonical = KVCache()
    canonical.update_and_fetch(
        mx.concatenate([prefix.state[0], next_keys], axis=2),
        mx.concatenate([prefix.state[1], next_values], axis=2),
    )
    mx.eval(clone.state[0], clone.state[1], canonical.state[0], canonical.state[1])
    assert clone.offset == canonical.offset == 14
    assert mx.array_equal(clone.state[0], canonical.state[0]).item()
    assert mx.array_equal(clone.state[1], canonical.state[1]).item()


def test_plain_kv_whole_graph_preflight_rejects_subclass_before_allocation(
    monkeypatch,
):
    class UnknownKVSubclass(KVCache):
        pass

    subclass = UnknownKVSubclass()
    ordinary = _plain_kv()
    subclass.keys = ordinary.keys
    subclass.values = ordinary.values
    subclass.offset = ordinary.offset
    copy_spy = MagicMock(wraps=exact_boundary._copy_prefix_array)
    monkeypatch.setattr(exact_boundary, "_copy_prefix_array", copy_spy)

    assert (
        plan_plain_kv_boundary(
            [_plain_kv(), subclass],
            source_tokens=5,
            source_cache_tokens=4,
            target_tokens=4,
        )
        is None
    )
    copy_spy.assert_not_called()


def test_plain_kv_provider_refuses_qwen36_arrays_gemma_rotating_and_quantized():
    arrays = ArraysCache(size=2)
    arrays.cache = [mx.zeros((1, 1, 4)), mx.zeros((1, 1, 4))]
    rotating = RotatingKVCache(max_size=16)
    quantized = QuantizedKVCache(group_size=64, bits=4)

    for unsupported in (arrays, rotating, quantized):
        assert (
            plan_plain_kv_boundary(
                [unsupported],
                source_tokens=5,
                source_cache_tokens=4,
                target_tokens=4,
            )
            is None
        )


def test_plain_kv_provider_refuses_batch_dimension_two():
    cache = KVCache()
    cache.keys = mx.zeros((2, 2, 4, 3))
    cache.values = mx.zeros((2, 2, 4, 4))
    cache.offset = 4

    assert (
        plan_plain_kv_boundary(
            [cache],
            source_tokens=5,
            source_cache_tokens=4,
            target_tokens=4,
        )
        is None
    )


def test_plain_kv_materialization_fails_closed_on_source_change_and_copy_error(
    monkeypatch,
):
    source = _plain_kv()
    plan = plan_plain_kv_boundary(
        [source],
        source_tokens=5,
        source_cache_tokens=4,
        target_tokens=4,
    )
    assert plan is not None

    source.offset = 3
    copy_spy = MagicMock(wraps=exact_boundary._copy_prefix_array)
    monkeypatch.setattr(exact_boundary, "_copy_prefix_array", copy_spy)
    assert (
        materialize_plain_kv_boundary(
            plan,
            stream=mx.default_stream(mx.default_device()),
        )
        is None
    )
    copy_spy.assert_not_called()

    source.offset = 4
    monkeypatch.setattr(
        exact_boundary,
        "_copy_prefix_array",
        MagicMock(side_effect=RuntimeError("synthetic allocation failure")),
    )
    assert (
        materialize_plain_kv_boundary(
            plan,
            stream=mx.default_stream(mx.default_device()),
        )
        is None
    )


def test_plain_kv_materialization_rejects_source_array_alias(monkeypatch):
    source = _plain_kv()
    plan = plan_plain_kv_boundary(
        [source],
        source_tokens=5,
        source_cache_tokens=4,
        target_tokens=4,
    )
    assert plan is not None
    monkeypatch.setattr(
        exact_boundary,
        "_copy_prefix_array",
        lambda array, _target: array,
    )

    assert (
        materialize_plain_kv_boundary(
            plan,
            stream=mx.default_stream(mx.default_device()),
        )
        is None
    )


def test_scheduler_stages_plain_kv_b1_and_fails_closed_for_b2_and_speculation():
    scheduler = _scheduler()
    request = _request()
    assert scheduler._stage_stable_prompt_boundary(request, [_plain_kv()])
    assert scheduler._publish_stable_prompt_boundary(request)
    hit = scheduler._exact_resident_cache.acquire_prefix([1, 2, 3, 4, 42])
    assert hit is not None and hit.cached_tokens == 4

    request = _request()
    scheduler.running["other"] = object()
    assert not scheduler._stage_stable_prompt_boundary(request, [_plain_kv()])
    scheduler.running.clear()

    request.images = [object()]
    assert not scheduler._stage_stable_prompt_boundary(request, [_plain_kv()])
    request.images = None

    scheduler.model._omlx_mtp_decode_enabled = True
    assert not scheduler._stage_stable_prompt_boundary(request, [_plain_kv()])


def test_qwen3_style_four_token_marker_rewrite_hits_durable_boundary():
    scheduler = _scheduler(block_size=8)
    source_tokens = list(range(14))
    request = _request(source_tokens)
    request._exact_resident_durable_fallback_tokens = 8

    assert scheduler._stage_stable_prompt_boundary(request, [_plain_kv(13)])
    assert scheduler._publish_stable_prompt_boundary(request)

    # The client preserves N-4 tokens and rewrites the final four-token
    # generation marker. N-1 would miss; the preceding durable block does not.
    rewritten = source_tokens[:10] + [90, 91, 92, 93]
    hit = scheduler._exact_resident_cache.acquire_prefix(rewritten)
    assert hit is not None
    assert hit.cached_tokens == 8


def test_hidden_prewarm_preserves_plain_boundary_and_longer_terminal():
    scheduler = _scheduler(slots=2, block_size=8)
    source_tokens = list(range(14))
    request = _request(source_tokens)
    request._exact_resident_durable_fallback_tokens = 8
    assert scheduler._stage_stable_prompt_boundary(request, [_plain_kv(13)])
    assert scheduler._publish_stable_prompt_boundary(request)
    terminal_cache = [_plain_kv(14)]
    assert scheduler._exact_resident_cache.put(
        source_tokens,
        terminal_cache,
        cache_nbytes=Scheduler._resident_cache_nbytes(terminal_cache),
        durable_tokens=8,
    )

    scheduler.has_requests = MagicMock(return_value=False)
    scheduler.tokenizer = SimpleNamespace(encode=lambda _prompt: source_tokens)
    result = scheduler.prewarm_prompt_tail(source_tokens, min_tokens=2)

    assert result["reason"] == "stable-boundary-already-resident"
    assert scheduler._exact_resident_cache.stats()["entries"] == 2
    assert scheduler._exact_resident_cache.contains_exact(source_tokens[:8])
    assert scheduler._exact_resident_cache.contains_exact(source_tokens)
