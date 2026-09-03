# SPDX-License-Identifier: Apache-2.0
"""Lossless, bounded prompt-tail prewarm contracts."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

from omlx.patches.mlx_lm_mtp import prompt_priming
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler


def _scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.has_requests = MagicMock(return_value=False)
    scheduler.tokenizer = SimpleNamespace(encode=lambda prompt: list(range(10)))
    scheduler._stream = object()
    scheduler.model = object()
    scheduler._exact_resident_cache = SimpleNamespace(
        max_entries=1,
        max_bytes=8 * 1024**3,
        put=MagicMock(return_value=True),
    )
    scheduler._resident_cache_spec_decode_active = MagicMock(return_value=False)
    scheduler._resident_cache_qwen4_target_only_enabled = MagicMock(
        return_value=False
    )
    scheduler._current_usage_bytes = MagicMock(return_value=1024)
    scheduler._admission_estimate = MagicMock(
        return_value=SimpleNamespace(
            estimated=4096,
            kv_exact=2048,
            transient=1024,
        )
    )
    scheduler._validate_cache = MagicMock(return_value=True)
    scheduler._preflight_memory_check = MagicMock(return_value=None)
    scheduler._invalidate_resident_pool_with_telemetry = MagicMock(return_value=True)
    scheduler._resident_cache_matches_token_count = MagicMock(return_value=True)
    scheduler._resident_cache_nbytes = MagicMock(return_value=1024)
    scheduler._release_paged_cache_for_request = MagicMock()
    scheduler._clear_request_admission_bookkeeping = MagicMock()
    scheduler._finalize_chunked_prefill_cache_for_insert = MagicMock()
    scheduler._schedule_deferred_metal_clear = MagicMock()
    scheduler._prepare_prefix_cache_for_request = MagicMock()
    scheduler._step_prefill_chunk = MagicMock(return_value=True)
    cache = [object()]
    state = SimpleNamespace(cache=cache, boundary_enabled=True)
    scheduler._begin_prefill = MagicMock(return_value=state)

    def prepare(request):
        request.prompt_cache = cache
        request.cached_tokens = min(8, len(request.prompt_token_ids))
        request.remaining_tokens = request.prompt_token_ids[request.cached_tokens :]
        request._exact_resident_durable_fallback_tokens = request.cached_tokens

    scheduler._prepare_prefix_cache_for_request.side_effect = prepare
    return scheduler, cache, state


def test_prompt_tail_prewarm_processes_all_suffix_and_publishes_exact_l0(monkeypatch):
    scheduler, cache, state = _scheduler()
    tracker = MagicMock()
    monkeypatch.setattr("omlx.scheduler.get_prefill_tracker", lambda: tracker)
    monkeypatch.setattr("omlx.scheduler.mx.stream", lambda _stream: MagicMock())

    # MagicMock is not a context manager by default on every Python/mock
    # combination; provide the exact protocol the scheduler uses.
    scheduler._stream = MagicMock()
    stream_context = MagicMock()
    stream_context.__enter__.return_value = None
    stream_context.__exit__.return_value = False
    monkeypatch.setattr("omlx.scheduler.mx.stream", lambda _stream: stream_context)
    sync_clear = MagicMock()
    monkeypatch.setattr("omlx.scheduler._sync_and_clear_cache", sync_clear)

    capture_states = []

    def finish_hidden_prefill(_state):
        capture_states.append(prompt_priming._suppressed())
        return True

    scheduler._step_prefill_chunk.side_effect = finish_hidden_prefill
    drop_ctx = MagicMock(wraps=prompt_priming.drop_ctx)
    monkeypatch.setattr(prompt_priming, "drop_ctx", drop_ctx)

    result = scheduler.prewarm_prompt_tail(
        "prompt",
        min_tokens=2,
        max_suffix_tokens=4,
        chunk_size=256,
    )

    assert result["status"] == "published"
    assert result["source_prompt_tokens"] == 10
    assert result["prompt_tokens"] == 9
    assert result["stable_boundary_trimmed_tokens"] == 1
    assert result["cached_tokens"] == 8
    assert result["suffix_tokens"] == 1
    scheduler._begin_prefill.assert_called_once()
    args, kwargs = scheduler._begin_prefill.call_args
    assert args[1] == [8]
    assert args[2] == cache
    assert kwargs["process_all_tokens"] is True
    assert state.boundary_enabled is False
    assert capture_states == [True]
    assert drop_ctx.call_count == 2
    assert all(call.args == (scheduler.model,) for call in drop_ctx.call_args_list)
    scheduler._exact_resident_cache.put.assert_called_once_with(
        list(range(9)),
        cache,
        cache_nbytes=1024,
        durable_tokens=8,
        protect_longer_prefix=True,
    )
    scheduler._release_paged_cache_for_request.assert_called_once()
    tracker.update.assert_not_called()
    tracker.remove.assert_called_once()
    sync_clear.assert_not_called()


def test_prompt_tail_stable_boundary_respects_minimum_after_trim():
    scheduler, _cache, _state = _scheduler()
    scheduler.tokenizer = SimpleNamespace(encode=lambda _prompt: [1, 2])

    result = scheduler.prewarm_prompt_tail("short", min_tokens=2)

    assert result == {
        "status": "skipped",
        "reason": "prompt-too-short",
        "source_prompt_tokens": 2,
        "stable_boundary_trimmed_tokens": 1,
        "prompt_tokens": 1,
    }
    scheduler._prepare_prefix_cache_for_request.assert_not_called()


def test_prompt_tail_skips_hidden_recompute_when_stable_boundary_is_resident():
    scheduler, _cache, _state = _scheduler()
    scheduler._exact_resident_cache.contains_exact = MagicMock(return_value=True)

    result = scheduler.prewarm_prompt_tail(list(range(10)), min_tokens=2)

    assert result == {
        "status": "skipped",
        "reason": "stable-boundary-already-resident",
        "source_prompt_tokens": 10,
        "stable_boundary_trimmed_tokens": 1,
        "prompt_tokens": 9,
    }
    scheduler._prepare_prefix_cache_for_request.assert_not_called()


def test_prompt_tail_skips_for_generic_durable_resident_prefix():
    scheduler, _cache, _state = _scheduler()
    scheduler.config = SimpleNamespace(paged_cache_block_size=4)
    scheduler._exact_resident_cache.contains_prefix = MagicMock(return_value=True)

    result = scheduler.prewarm_prompt_tail(list(range(10)), min_tokens=2)

    assert result["reason"] == "stable-boundary-already-resident"
    scheduler._exact_resident_cache.contains_prefix.assert_called_once_with(
        list(range(9)),
        minimum_tokens=8,
    )
    scheduler._prepare_prefix_cache_for_request.assert_not_called()


def test_prompt_tail_prefix_restore_does_not_change_user_cache_metrics(monkeypatch):
    scheduler, _cache, _state = _scheduler()
    tracker = MagicMock()
    monkeypatch.setattr("omlx.scheduler.get_prefill_tracker", lambda: tracker)
    stream_context = MagicMock()
    stream_context.__enter__.return_value = None
    stream_context.__exit__.return_value = False
    monkeypatch.setattr("omlx.scheduler.mx.stream", lambda _stream: stream_context)

    prefix_cache = SimpleNamespace(
        _hits=7,
        _misses=3,
        _tokens_saved=4096,
        _tokens_matched_total=4096,
        _tokens_requested_total=5000,
    )
    ssd_manager = SimpleNamespace(
        _stats={"hits": 11, "misses": 2, "loads": 5, "hot_cache_hits": 4}
    )
    scheduler.block_aware_cache = prefix_cache
    scheduler.paged_ssd_cache_manager = ssd_manager
    scheduler._phase_total_ms = {"prefix_cache_lookup": 1.5}
    scheduler._phase_count = {"prefix_cache_lookup": 2}
    original_prepare = scheduler._prepare_prefix_cache_for_request.side_effect

    def prepare_and_count(request):
        prefix_cache._hits += 1
        prefix_cache._tokens_saved += 8
        prefix_cache._tokens_matched_total += 8
        prefix_cache._tokens_requested_total += 10
        ssd_manager._stats["hits"] += 1
        ssd_manager._stats["loads"] += 1
        scheduler._phase_total_ms["prefix_cache_lookup"] += 2.0
        scheduler._phase_count["prefix_cache_lookup"] += 1
        original_prepare(request)

    scheduler._prepare_prefix_cache_for_request.side_effect = prepare_and_count

    result = scheduler.prewarm_prompt_tail(list(range(10)), min_tokens=2)

    assert result["status"] == "published"
    assert prefix_cache._hits == 7
    assert prefix_cache._tokens_saved == 4096
    assert prefix_cache._tokens_matched_total == 4096
    assert prefix_cache._tokens_requested_total == 5000
    assert ssd_manager._stats == {
        "hits": 11,
        "misses": 2,
        "loads": 5,
        "hot_cache_hits": 4,
    }
    assert scheduler._phase_total_ms == {"prefix_cache_lookup": 1.5}
    assert scheduler._phase_count == {"prefix_cache_lookup": 2}


def test_prompt_tail_prewarm_skips_without_reusable_prefix(monkeypatch):
    scheduler, _cache, _state = _scheduler()
    tracker = MagicMock()
    monkeypatch.setattr("omlx.scheduler.get_prefill_tracker", lambda: tracker)
    stream_context = MagicMock()
    stream_context.__enter__.return_value = None
    stream_context.__exit__.return_value = False
    monkeypatch.setattr("omlx.scheduler.mx.stream", lambda _stream: stream_context)
    monkeypatch.setattr("omlx.scheduler._sync_and_clear_cache", lambda _stream: None)

    def prepare(request):
        request.prompt_cache = None
        request.cached_tokens = 0
        request.remaining_tokens = list(range(10))

    scheduler._prepare_prefix_cache_for_request.side_effect = prepare
    result = scheduler.prewarm_prompt_tail("prompt", min_tokens=2)

    assert result["status"] == "skipped"
    assert result["reason"] == "no-reusable-prefix"
    scheduler._begin_prefill.assert_not_called()
    scheduler._exact_resident_cache.put.assert_not_called()


def test_prompt_tail_prewarm_aborts_before_model_work(monkeypatch):
    scheduler, _cache, _state = _scheduler()
    tracker = MagicMock()
    monkeypatch.setattr("omlx.scheduler.get_prefill_tracker", lambda: tracker)

    result = scheduler.prewarm_prompt_tail(
        list(range(10)),
        min_tokens=2,
        abort_requested=lambda: True,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "admission-pending"
    scheduler._prepare_prefix_cache_for_request.assert_not_called()


def test_prompt_tail_prewarm_rejects_unproved_speculative_model():
    scheduler, _cache, _state = _scheduler()
    scheduler._resident_cache_spec_decode_active.return_value = True

    result = scheduler.prewarm_prompt_tail(list(range(10)), min_tokens=2)

    assert result["status"] == "skipped"
    assert result["reason"] == "speculative-terminal-unproved"
    scheduler._prepare_prefix_cache_for_request.assert_not_called()
    scheduler._exact_resident_cache.put.assert_not_called()


def test_prompt_tail_prewarm_allows_verified_qwen4_speculative_model(monkeypatch):
    scheduler, _cache, _state = _scheduler()
    scheduler._resident_cache_spec_decode_active.return_value = True
    scheduler._resident_cache_qwen4_target_only_enabled.return_value = True
    tracker = MagicMock()
    monkeypatch.setattr("omlx.scheduler.get_prefill_tracker", lambda: tracker)
    stream_context = MagicMock()
    stream_context.__enter__.return_value = None
    stream_context.__exit__.return_value = False
    monkeypatch.setattr("omlx.scheduler.mx.stream", lambda _stream: stream_context)

    result = scheduler.prewarm_prompt_tail(list(range(10)), min_tokens=2)

    assert result["status"] == "published"
    scheduler._exact_resident_cache.put.assert_called_once()


def test_prompt_tail_prewarm_rejects_memory_before_prefix_reconstruction(
    monkeypatch,
):
    scheduler, _cache, _state = _scheduler()
    scheduler._admission_estimate.return_value = SimpleNamespace(
        estimated=32 * 1024**3,
        kv_exact=16 * 1024**3,
        transient=1024,
    )
    monkeypatch.setattr(
        "omlx.scheduler.get_max_working_set_bytes",
        lambda: 24 * 1024**3,
    )

    result = scheduler.prewarm_prompt_tail(list(range(10)), min_tokens=2)

    assert result["status"] == "skipped"
    assert result["reason"] == "memory-headroom-rejected"
    scheduler._prepare_prefix_cache_for_request.assert_not_called()
    scheduler._exact_resident_cache.put.assert_not_called()


def test_prompt_tail_prewarm_aborts_between_chunks_without_publication(monkeypatch):
    scheduler, _cache, _state = _scheduler()
    tracker = MagicMock()
    monkeypatch.setattr("omlx.scheduler.get_prefill_tracker", lambda: tracker)
    monkeypatch.setattr("omlx.scheduler.mx.stream", lambda _stream: nullcontext())

    admission_arrived = False

    def first_chunk_then_admit(_state):
        nonlocal admission_arrived
        admission_arrived = True
        return False

    scheduler._step_prefill_chunk.side_effect = first_chunk_then_admit

    result = scheduler.prewarm_prompt_tail(
        list(range(10)),
        min_tokens=2,
        max_suffix_tokens=4,
        abort_requested=lambda: admission_arrived,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "admission-arrived"
    scheduler._step_prefill_chunk.assert_called_once()
    scheduler._exact_resident_cache.put.assert_not_called()
    scheduler._release_paged_cache_for_request.assert_called_once()
    scheduler._schedule_deferred_metal_clear.assert_called_once()
    tracker.remove.assert_called_once()


def test_prompt_tail_prewarm_timeline_mismatch_fails_closed(monkeypatch):
    scheduler, _cache, _state = _scheduler()
    tracker = MagicMock()
    monkeypatch.setattr("omlx.scheduler.get_prefill_tracker", lambda: tracker)
    monkeypatch.setattr("omlx.scheduler.mx.stream", lambda _stream: nullcontext())
    scheduler._resident_cache_matches_token_count.return_value = False

    result = scheduler.prewarm_prompt_tail(
        list(range(10)),
        min_tokens=2,
        max_suffix_tokens=4,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "timeline-mismatch"
    scheduler._finalize_chunked_prefill_cache_for_insert.assert_called_once()
    scheduler._exact_resident_cache.put.assert_not_called()
    scheduler._release_paged_cache_for_request.assert_called_once()
    tracker.remove.assert_called_once()


def test_prompt_tail_publish_rejection_never_falls_back_to_direct_put(monkeypatch):
    scheduler, _cache, _state = _scheduler()
    tracker = MagicMock()
    monkeypatch.setattr("omlx.scheduler.get_prefill_tracker", lambda: tracker)
    monkeypatch.setattr("omlx.scheduler.mx.stream", lambda _stream: nullcontext())
    publish_if_current = MagicMock(return_value=False)

    result = scheduler.prewarm_prompt_tail(
        list(range(10)),
        min_tokens=2,
        max_suffix_tokens=4,
        publish_if_current=publish_if_current,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "resident-budget-rejected"
    publish_if_current.assert_called_once()
    scheduler._exact_resident_cache.put.assert_not_called()


def test_prewarm_prefix_lookup_preserves_existing_exact_resident_entry():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._prefix_cache_prepared = set()
    scheduler.block_aware_cache = None
    scheduler._restore_exact_resident_cache = MagicMock(return_value=True)
    scheduler._log_prefix_divergence = MagicMock()
    scheduler._try_specprefill_scoring = MagicMock()
    request = Request(
        request_id="prewarm-no-l0-take",
        prompt=[1, 2, 3],
        sampling_params=SamplingParams(),
    )
    request.prompt_token_ids = [1, 2, 3]
    request._cache_prewarm_only = True

    scheduler._prepare_prefix_cache_for_request(request)

    scheduler._restore_exact_resident_cache.assert_not_called()
    assert request.remaining_tokens == [1, 2, 3]
    scheduler._log_prefix_divergence.assert_not_called()
    scheduler._try_specprefill_scoring.assert_not_called()
