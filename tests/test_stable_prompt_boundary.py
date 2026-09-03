# SPDX-License-Identifier: Apache-2.0
"""Immediate, lossless Qwen4 prompt N-1 resident handoff."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx

from omlx.cache.exact_resident import ExactResidentPrefixCache
from omlx.cache.type_handlers import SizedArraysCache
from omlx.patches.mlx_vlm_qwen4_exp_compat import (
    apply_mlx_vlm_qwen4_exp_compat_patch,
)
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler


def _qwen_cache(token_count=4):
    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_lm.models.cache import ArraysCache
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache

    qsa = QSAKVCache()
    qsa.state = (
        mx.arange(2 * token_count * 4, dtype=mx.float32).reshape(
            1, 2, token_count, 4
        ),
        mx.arange(2 * token_count * 4, dtype=mx.float32).reshape(
            1, 2, token_count, 4
        )
        + 100,
        mx.arange(token_count * 3, dtype=mx.float32).reshape(1, token_count, 3),
        mx.arange(token_count, dtype=mx.int32)[None],
    )
    arrays = ArraysCache(size=2)
    arrays.cache = [
        mx.arange(8, dtype=mx.float32).reshape(1, 2, 4),
        mx.arange(4, dtype=mx.float32).reshape(1, 1, 4),
    ]
    return [qsa, SizedArraysCache(arrays, token_count=token_count)]


def _scheduler(*, slots=2, max_bytes=1 << 30):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = SimpleNamespace(
        _omlx_mtp_terminal_commit_v1=True,
        _omlx_mtp_suffix_local_capability="qwen4-verified-text-v1",
    )
    scheduler._vlm_mtp_drafter = None
    scheduler._exact_resident_cache = ExactResidentPrefixCache(
        max_entries=slots,
        max_bytes=max_bytes,
    )
    scheduler.running = {}
    scheduler.waiting = deque()
    scheduler.prefilling = deque()
    scheduler.config = SimpleNamespace(paged_cache_block_size=4)
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
    request = Request("stable", tokens, SamplingParams())
    request.prompt_token_ids = tokens
    request.num_prompt_tokens = len(tokens)
    request._exact_resident_durable_fallback_tokens = 4
    return request


def test_capture_detaches_qsa_sized_arrays_and_ple_owned_state():
    scheduler = _scheduler()
    request = _request()
    source = _qwen_cache()
    mtp_ctx = object()
    scheduler.model._omlx_mtp_prime_ctx = mtp_ctx

    assert scheduler._stage_stable_prompt_boundary(request, source)
    assert scheduler.model._omlx_mtp_prime_ctx is mtp_ctx

    tokens, cloned, nbytes, durable, generation, capture_ms = (
        request._stable_prompt_resident_candidate
    )
    assert tokens == [1, 2, 3, 4]
    assert nbytes > 0
    assert durable == 4
    assert generation == scheduler._exact_resident_cache.generation()
    assert capture_ms >= 0
    assert cloned[0] is not source[0]
    assert cloned[0].keys is not source[0].keys
    assert cloned[0]._index_keys is not source[0]._index_keys
    assert cloned[1] is not source[1]
    assert cloned[1]._inner is not source[1]._inner
    assert all(
        cloned_value is not source_value
        for cloned_value, source_value in zip(
            cloned[1].cache,
            source[1].cache,
        )
    )
    assert scheduler._resident_cache_matches_token_count(cloned, 4)


def test_terminal_and_stable_candidates_coexist_and_choose_longest_exact():
    scheduler = _scheduler()
    request = _request()
    assert scheduler._stage_stable_prompt_boundary(request, _qwen_cache())
    terminal = _qwen_cache(5)
    assert scheduler._exact_resident_cache.put(
        [1, 2, 3, 4, 99],
        terminal,
        cache_nbytes=Scheduler._resident_cache_nbytes(terminal),
        durable_tokens=4,
    )

    assert scheduler._publish_stable_prompt_boundary(request)

    rewritten = scheduler._exact_resident_cache.acquire_prefix([1, 2, 3, 4, 42])
    assert rewritten is not None and rewritten.cached_tokens == 4
    appended = scheduler._exact_resident_cache.acquire_prefix(
        [1, 2, 3, 4, 99, 7]
    )
    assert appended is not None and appended.cached_tokens == 5


def test_clear_generation_prevents_staged_candidate_repopulation():
    scheduler = _scheduler()
    request = _request()
    assert scheduler._stage_stable_prompt_boundary(request, _qwen_cache())

    scheduler._exact_resident_cache.clear()

    assert not scheduler._publish_stable_prompt_boundary(request)
    assert scheduler._exact_resident_cache.stats()["entries"] == 0


def test_capture_fails_closed_for_b2_media_missing_durability_and_byte_cap():
    scheduler = _scheduler()
    request = _request()
    cache = _qwen_cache()

    scheduler.running["other"] = object()
    assert not scheduler._stage_stable_prompt_boundary(request, cache)
    scheduler.running.clear()

    request.images = [object()]
    assert not scheduler._stage_stable_prompt_boundary(request, cache)
    request.images = None

    request._exact_resident_durable_fallback_tokens = 0
    assert not scheduler._stage_stable_prompt_boundary(request, cache)
    request._exact_resident_durable_fallback_tokens = 4

    scheduler._memory_abort_limit_bytes = 1
    scheduler._memory_metal_cap_bytes = 1
    scheduler._memory_static_ceiling_bytes = 1
    assert not scheduler._stage_stable_prompt_boundary(request, cache)


def test_capture_fails_closed_on_clone_exception_without_touching_mtp_ctx(
    monkeypatch,
):
    from omlx.patches.mlx_lm_mtp import prompt_priming

    scheduler = _scheduler()
    request = _request()
    mtp_ctx = object()
    scheduler.model._omlx_mtp_prime_ctx = mtp_ctx

    def fail_clone(*_args, **_kwargs):
        raise RuntimeError("synthetic clone failure")

    monkeypatch.setattr(prompt_priming, "_cache_at_offset", fail_clone)

    assert not scheduler._stage_stable_prompt_boundary(request, _qwen_cache())
    assert request._stable_prompt_resident_candidate is None
    assert scheduler.model._omlx_mtp_prime_ctx is mtp_ctx


def test_capture_rejects_clone_that_shares_source_arrays(monkeypatch):
    from omlx.patches.mlx_lm_mtp import prompt_priming

    scheduler = _scheduler()
    request = _request()
    source = _qwen_cache()
    monkeypatch.setattr(
        prompt_priming,
        "_cache_at_offset",
        lambda _cache, _target: source,
    )

    assert not scheduler._stage_stable_prompt_boundary(request, source)
    assert request._stable_prompt_resident_candidate is None


def test_qwen4_immediate_path_fails_closed_without_terminal_dependency():
    scheduler = _scheduler()
    del scheduler.model._omlx_mtp_terminal_commit_v1
    del scheduler.model._omlx_mtp_suffix_local_capability
    scheduler.model._omlx_mtp_decode_enabled = True
    source = _qwen_cache()
    request = _request()

    assert not scheduler._resident_cache_qwen4_target_only_enabled()
    assert not scheduler._stage_stable_prompt_boundary(request, source)
    assert request._stable_prompt_resident_candidate is None

    assert scheduler._exact_resident_cache.put(
        [1, 2, 3, 4],
        source,
        cache_nbytes=Scheduler._resident_cache_nbytes(source),
        durable_tokens=4,
    )
    assert not scheduler._restore_exact_resident_cache(
        _request([1, 2, 3, 4, 42])
    )
    assert scheduler._exact_resident_cache.stats()["entries"] == 1
