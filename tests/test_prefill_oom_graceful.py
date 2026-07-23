# SPDX-License-Identifier: Apache-2.0
"""Tests for graceful prefill memory management (predictive throttle +
bounded requeue) added to keep coding-agent workloads from hard-failing
mid-prefill under memory pressure.

Covers:
  - MemoryMonitor.estimate_chunk_transient_bytes math (MLX SDPA dispatch)
  - Scheduler._adaptive_chunk_size predictive sizing (EWMA + static first chunk,
    early-return below the soft watermark, min-chunk floor, bucket clamp)
  - Scheduler._requeue_or_fail_prefill budget behavior + error-type gating

All tests are unit-level: the throttle/requeue logic is exercised on a light
fake object so no model load or GPU is required.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omlx import scheduler as sched_mod
from omlx.exceptions import PrefillMemoryExceededError
from omlx.memory_monitor import (
    _SDPA_FALLBACK_SCORE_DTYPE_SIZE,
    MemoryMonitor,
)
from omlx.prefill_transient_tracker import PrefillTransientTracker
from omlx.scheduler import Scheduler, _PrefillEvictionNeeded, _PrefillState

_GB = 1024**3


# --------------------------------------------------------------------------
# MemoryMonitor.estimate_chunk_transient_bytes
# --------------------------------------------------------------------------


def _monitor(head_dim):
    m = MemoryMonitor(max_kv_cache_memory=_GB)
    m.set_model_info(
        num_layers=32,
        num_kv_heads=8,
        head_dim=head_dim,
        dtype_size=2,
        num_attention_heads=32,
    )
    return m


def test_chunk_transient_unsupported_vector_head_dim_scales_with_kv_len():
    """head_dim=192 is unsupported by vector/full MLX SDPA and falls back."""
    m = _monitor(head_dim=192)
    n_q, hd = 32, 192
    n_tokens, kv_len = 4, 10_000
    expected = n_q * n_tokens * kv_len * _SDPA_FALLBACK_SCORE_DTYPE_SIZE
    expected += n_q * n_tokens * hd * 4
    assert m.estimate_chunk_transient_bytes(n_tokens, kv_len) == expected
    # Doubling kv_len roughly doubles the transient (kv term dominates).
    bigger = m.estimate_chunk_transient_bytes(n_tokens, kv_len * 2)
    assert bigger > expected


def test_chunk_transient_supported_vector_head_dim_is_kv_independent():
    """head_dim=128 short queries use the fused vector kernel."""
    m = _monitor(head_dim=128)
    n_q, hd, n_tokens = 32, 128, 4
    expected = n_q * n_tokens * hd * 4
    assert m.estimate_chunk_transient_bytes(n_tokens, 10_000) == expected
    # kv_len must not change the estimate for the fused path.
    assert m.estimate_chunk_transient_bytes(n_tokens, n_tokens) == expected


def test_chunk_transient_zero_when_model_info_missing():
    m = MemoryMonitor(max_kv_cache_memory=_GB)  # no set_model_info
    assert m.estimate_chunk_transient_bytes(4, 1000) == 0


# --------------------------------------------------------------------------
# Scheduler._adaptive_chunk_size
# --------------------------------------------------------------------------


def _throttle_ctx(
    *,
    current,
    hard,
    soft_ratio=0.80,
    samples_bpt=None,
    monitor=None,
    min_chunk=32,
    abort=None,
    reclaim_to=None,
    abort_margin=Scheduler._PREFILL_ABORT_MARGIN,
):
    """Build a minimal stand-in carrying the attributes / bound methods that
    _adaptive_chunk_size and _guard_prefill_chunk read. `_fake_current` is the
    value the patched memory probes report; `reclaim_to` (if set) is what a
    reclaim drops `current` to."""
    tracker = PrefillTransientTracker()
    if samples_bpt is not None:
        # Seed with one observation: sets last_delta/last_n AND the EWMA.
        tracker.update(1, int(samples_bpt))
    ns = SimpleNamespace(
        _memory_limit_bytes=int(hard * 0.85),  # soft = ceiling*0.85
        _memory_hard_limit_bytes=int(hard),
        _memory_abort_limit_bytes=int(abort if abort is not None else hard),
        _prefill_safe_zone_ratio=soft_ratio,
        _prefill_min_chunk_tokens=min_chunk,
        _prefill_abort_margin=abort_margin,
        _prefill_headroom_safety=Scheduler._PREFILL_HEADROOM_SAFETY,
        _prefill_transient_tracker=tracker,
        memory_monitor=monitor,
        _PREFILL_STEP_TIERS=Scheduler._PREFILL_STEP_TIERS,
        _PREFILL_HEADROOM_SAFETY=Scheduler._PREFILL_HEADROOM_SAFETY,
        _PREFILL_ABORT_MARGIN=Scheduler._PREFILL_ABORT_MARGIN,
        _PREFILL_TRANSIENT_SAFETY=Scheduler._PREFILL_TRANSIENT_SAFETY,
        _last_mlx_active_memory_bytes=0,
    )
    # Bind the real helper methods so the stand-in behaves like a Scheduler.
    ns._current_usage_bytes = Scheduler._current_usage_bytes.__get__(ns, Scheduler)
    ns._predicted_chunk_transient = Scheduler._predicted_chunk_transient.__get__(
        ns, Scheduler
    )
    ns._prefill_abort_cap = Scheduler._prefill_abort_cap.__get__(ns, Scheduler)
    ns._prefill_abort_description = Scheduler._prefill_abort_description.__get__(
        ns, Scheduler
    )
    ns._reclaim_to = reclaim_to

    def _reclaim():
        if ns._reclaim_to is not None:
            ns._fake_current = ns._reclaim_to
        return ns._fake_current

    ns._reclaim_prefill_headroom = _reclaim
    return ns


def _call(ns, requested, kv_len=0):
    with (
        patch.object(sched_mod.mx, "get_active_memory", return_value=0),
        patch.object(sched_mod, "get_phys_footprint", return_value=ns._fake_current),
    ):
        return Scheduler._adaptive_chunk_size(
            ns, requested, request_id="r", loop_label="test", kv_len=kv_len
        )


def test_adaptive_throttle_requests_eviction_before_shrinking():
    ns = _throttle_ctx(
        current=50 * _GB,
        hard=58 * _GB,
        samples_bpt=2 * 1024**2,
    )
    ns._fake_current = 50 * _GB
    request = SimpleNamespace(prefill_eviction_retries=0)
    ns.requests = {"r": request}
    ns.config = SimpleNamespace(model_name="model-b")
    ns._raise_prefill_eviction_if_available = (
        Scheduler._raise_prefill_eviction_if_available.__get__(ns, Scheduler)
    )

    with pytest.raises(_PrefillEvictionNeeded) as exc:
        _call(ns, 2048)

    assert request.prefill_eviction_retries == 1
    assert exc.value.request.request_id == "r"
    assert exc.value.request.model_id == "model-b"
    assert exc.value.request.requested_tokens == 2048
    assert exc.value.request.reason == "adaptive_prefill_throttle"

    # The same request does not loop on eviction; it falls back to throttling.
    result = _call(ns, 2048)
    assert result < 2048


def _guard_call(ns, n, kv_len=0):
    with (
        patch.object(sched_mod.mx, "get_active_memory", return_value=0),
        patch.object(sched_mod, "get_phys_footprint", return_value=ns._fake_current),
    ):
        return Scheduler._guard_prefill_chunk(
            ns, n, kv_len=kv_len, progress=0, loop_label="test"
        )


def _per_token(samples_bpt):
    """The throttle's effective per-token estimate for a seeded EWMA/last."""
    return samples_bpt * Scheduler._PREFILL_TRANSIENT_SAFETY


def test_throttle_noop_when_full_chunk_fits():
    """If the full requested chunk's predicted peak fits, it runs unchanged —
    even at a low baseline (gate is on predicted peak, not the watermark)."""
    hard = 40 * _GB
    # Small per-token transient (~1MB/tok): 2048 tokens ≈ 2.7GB, easily fits.
    ns = _throttle_ctx(current=int(hard * 0.5), hard=hard, samples_bpt=1024 * 1024)
    ns._fake_current = int(hard * 0.5)
    assert _call(ns, 2048, kv_len=5000) == 2048


def test_throttle_shrinks_big_chunk_from_low_baseline():
    """The regression that mattered: a huge per-token transient (MoE-like)
    must shrink the chunk even when current is well BELOW the soft watermark,
    and the result's predicted peak must fit the sizing target."""
    hard = 40 * _GB
    current = int(hard * 0.5)  # 20GB — below soft watermark (0.85*0.80*40=27.2GB)
    bpt = 18 * 1024 * 1024  # ~18 MB/token, matching the observed MoE prefill
    ns = _throttle_ctx(current=current, hard=hard, samples_bpt=bpt)
    ns._fake_current = current
    target = min(
        int(hard * Scheduler._PREFILL_HEADROOM_SAFETY),
        int(hard * Scheduler._PREFILL_ABORT_MARGIN),
    )
    n = _call(ns, 2048, kv_len=5000)
    assert n < 2048  # throttled despite low baseline
    assert n >= ns._prefill_min_chunk_tokens
    # The chosen chunk's predicted peak must fit under the sizing target.
    assert current + _per_token(bpt) * n <= target + _per_token(bpt)


def test_throttle_floors_at_min_chunk_when_over_ceiling():
    """At/over the cap, the smallest step is returned (the guard handles the
    rest)."""
    hard = 40 * _GB
    ns = _throttle_ctx(
        current=hard + _GB, hard=hard, samples_bpt=1_000_000, min_chunk=32
    )
    ns._fake_current = hard + _GB
    assert _call(ns, 2048, kv_len=5000) == 32


def test_throttle_predictor_anchors_on_recent_measurement():
    """At large kv_len the per-token estimate must reflect the most RECENT
    measured transient (not a lagging long-run average) so chunks shrink
    enough to avoid the Metal-cap overshoot that crashed the server."""
    hard = 42 * _GB
    # Resident ~32GB (model + 122k-token KV), last chunk measured ~27MB/token.
    current = 32 * _GB
    bpt = 27 * 1024 * 1024
    ns = _throttle_ctx(current=current, hard=hard, samples_bpt=bpt)
    ns._fake_current = current
    n = _call(ns, 2048, kv_len=122_000)
    # Must shrink hard: the full 2048 chunk's transient (~54GB) is impossible.
    assert n < 2048
    assert n >= ns._prefill_min_chunk_tokens
    cap = int(hard * Scheduler._PREFILL_ABORT_MARGIN)
    # The chosen chunk's predicted peak stays under the margined physical cap.
    assert current + _per_token(bpt) * n <= cap + _per_token(bpt)


# --------------------------------------------------------------------------
# Scheduler._guard_prefill_chunk (the crash preventer)
# --------------------------------------------------------------------------


def test_guard_passes_through_when_chunk_fits():
    hard = 42 * _GB
    ns = _throttle_ctx(current=10 * _GB, hard=hard, samples_bpt=1024 * 1024)
    ns._fake_current = 10 * _GB
    assert _guard_call(ns, 512, kv_len=5000) == 512


def test_guard_shrinks_when_chunk_would_breach_cap():
    """A chunk predicted to breach the margined cap is shrunk to the largest
    safe size (after a reclaim), never raising while the floor still fits."""
    hard = 42 * _GB
    current = 30 * _GB
    bpt = 27 * 1024 * 1024
    # Reclaim doesn't free anything here (transient already cleared).
    ns = _throttle_ctx(current=current, hard=hard, samples_bpt=bpt, reclaim_to=current)
    ns._fake_current = current
    n = _guard_call(ns, 2048, kv_len=122_000)
    cap = int(hard * Scheduler._PREFILL_ABORT_MARGIN)
    assert n >= ns._prefill_min_chunk_tokens
    assert n < 2048
    assert current + _per_token(bpt) * n <= cap


def test_guard_raises_clean_error_when_even_floor_cannot_fit():
    """When resident alone is so high that even a min-chunk transient would
    breach the cap, the guard raises a CLEAN error that is NOT a 'Memory limit
    exceeded' string — so it fails fast instead of looping a doomed retry."""
    hard = 42 * _GB
    current = 41 * _GB  # resident already above the margined cap
    bpt = 27 * 1024 * 1024
    ns = _throttle_ctx(
        current=current, hard=hard, samples_bpt=bpt, reclaim_to=current
    )  # reclaim can't help
    ns._fake_current = current
    with pytest.raises(PrefillMemoryExceededError) as exc:
        _guard_call(ns, 256, kv_len=122_000)
    assert "too large for available memory" in str(exc.value)
    assert "Memory limit exceeded" not in str(exc.value)  # → fails fast, no requeue
    assert "prefill safety cap" in str(exc.value)
    assert "90% of effective ceiling 42.0GB" in str(exc.value)
    assert exc.value.estimated_bytes is not None
    assert exc.value.limit_bytes == int(hard * Scheduler._PREFILL_ABORT_MARGIN)


def test_guard_requests_eviction_before_capacity_rejection():
    hard = 42 * _GB
    current = 41 * _GB
    bpt = 27 * 1024 * 1024
    ns = _throttle_ctx(current=current, hard=hard, samples_bpt=bpt, reclaim_to=current)
    ns._fake_current = current
    request = SimpleNamespace(prefill_eviction_retries=0)
    ns.requests = {"r": request}
    ns.config = SimpleNamespace(model_name="model-b")
    ns._raise_prefill_eviction_if_available = (
        Scheduler._raise_prefill_eviction_if_available.__get__(ns, Scheduler)
    )

    with pytest.raises(_PrefillEvictionNeeded) as exc:
        with (
            patch.object(sched_mod.mx, "get_active_memory", return_value=0),
            patch.object(sched_mod, "get_phys_footprint", return_value=current),
        ):
            Scheduler._guard_prefill_chunk(
                ns,
                256,
                kv_len=122_000,
                progress=0,
                loop_label="test",
                request_id="r",
            )

    assert request.prefill_eviction_retries == 1
    assert exc.value.request.reason == "prefill_safety_cap"


def test_guard_custom_margin_allows_95_percent_of_ceiling():
    """Custom tier propagates a looser prefill safety margin."""
    hard = 30 * _GB
    ns = _throttle_ctx(
        current=0,
        hard=hard,
        samples_bpt=1024 * 1024,
        abort_margin=0.95,
    )
    assert ns._prefill_abort_cap() == int(30 * _GB * 0.95)


def test_guard_recovers_after_reclaim_frees_memory():
    """If a reclaim drops resident back under the cap, the guard proceeds."""
    hard = 42 * _GB
    bpt = 1024 * 1024  # small per-token
    ns = _throttle_ctx(
        current=41 * _GB, hard=hard, samples_bpt=bpt, reclaim_to=20 * _GB
    )
    ns._fake_current = 41 * _GB
    n = _guard_call(ns, 512, kv_len=5000)
    assert n >= ns._prefill_min_chunk_tokens


# --------------------------------------------------------------------------
# Scheduler._predicted_chunk_transient
# --------------------------------------------------------------------------


def test_predicted_transient_takes_max_and_applies_safety():
    """The predictor takes the MAX of measured-last / EWMA / static and applies
    the safety factor — so it can't underestimate at growing kv_len."""
    monitor = _monitor(head_dim=192)
    ns = _throttle_ctx(
        current=0, hard=40 * _GB, samples_bpt=5 * 1024 * 1024, monitor=monitor
    )
    # static per-token at this kv_len: SDPA transient plus the chunk's KV growth.
    static = monitor.estimate_chunk_transient_bytes(1, 100_001)
    static += monitor.estimate_prompt_kv_bytes(1)
    measured = 5 * 1024 * 1024
    expected_per_token = max(measured, static) * Scheduler._PREFILL_TRANSIENT_SAFETY
    got = ns._predicted_chunk_transient(1, 100_000)
    assert got == pytest.approx(expected_per_token, rel=1e-6)


def test_predicted_transient_static_uses_candidate_chunk_size():
    """Static fallback must classify the actual prefill chunk, not query=1."""
    monitor = _monitor(head_dim=256)
    ns = _throttle_ctx(current=0, hard=40 * _GB, samples_bpt=None, monitor=monitor)
    n_tokens = 512
    kv_len = 100_000

    expected_static = monitor.estimate_chunk_transient_bytes(
        n_tokens, kv_len + n_tokens
    )
    expected_static += monitor.estimate_prompt_kv_bytes(n_tokens)
    expected = expected_static * Scheduler._PREFILL_TRANSIENT_SAFETY
    old_query_one_style = (
        monitor.estimate_chunk_transient_bytes(1, kv_len + 1)
        * n_tokens
        * Scheduler._PREFILL_TRANSIENT_SAFETY
    )

    got = ns._predicted_chunk_transient(n_tokens, kv_len)
    assert got == pytest.approx(expected, rel=1e-6)
    assert got > old_query_one_style


def test_predicted_transient_zero_without_signals():
    ns = _throttle_ctx(current=0, hard=40 * _GB, samples_bpt=None, monitor=None)
    assert ns._predicted_chunk_transient(4, 1000) == 0.0


def test_record_chunk_transient_skips_tail_samples():
    tracker = PrefillTransientTracker()
    ns = SimpleNamespace(
        _prefill_min_chunk_tokens=256,
        _prefill_transient_tracker=tracker,
    )
    ns._record_chunk_transient = Scheduler._record_chunk_transient.__get__(
        ns, Scheduler
    )

    ns._record_chunk_transient(
        64,
        0,
        32 * 1024**2,
        request_id="req-tail",
        loop_label="unit",
    )
    assert tracker.samples == 0

    ns._record_chunk_transient(
        256,
        0,
        32 * 1024**2,
        request_id="req-full",
        loop_label="unit",
    )
    assert tracker.samples == 1
    assert tracker.last_delta_bytes == 32 * 1024**2


def test_step_prefill_reclaims_before_first_guard():
    events = []
    request = SimpleNamespace(request_id="req-prefill")
    state = _PrefillState(
        request=request,
        cache=[],
        tokens_remaining=sched_mod.mx.array([[1, 2, 3]]),
        last_token=[4],
        tokens_processed=0,
        base_size=0,
        emitted_boundaries={},
        boundary_enabled=False,
        block_size=0,
        total_length=4,
    )
    ns = SimpleNamespace(
        config=SimpleNamespace(prefill_step_size=2, model_name=""),
        _stream="stream",
        _memory_limit_bytes=0,
        _glm_dsa_adaptive_prefill=None,
        model=lambda *args, **kwargs: events.append("model"),
        _adaptive_chunk_size=lambda n, **kwargs: events.append("adaptive") or n,
        _guard_prefill_chunk=lambda n, **kwargs: events.append("guard") or n,
        _record_chunk_transient=MagicMock(),
    )
    ns._prefill_step_size_for_progress = (
        Scheduler._prefill_step_size_for_progress.__get__(ns, Scheduler)
    )
    ns._step_prefill_chunk = Scheduler._step_prefill_chunk.__get__(ns, Scheduler)

    with (
        patch.object(
            sched_mod,
            "_sync_and_clear_cache",
            side_effect=lambda stream=None: events.append("sync"),
        ),
        patch.object(sched_mod.mx, "stream"),
        patch.object(sched_mod.mx, "eval", lambda *args: events.append("eval")),
        patch.object(sched_mod, "get_phys_footprint", side_effect=[100, 300]),
    ):
        done = ns._step_prefill_chunk(state)

    assert done is False
    assert events[:3] == ["sync", "adaptive", "guard"]
    ns._record_chunk_transient.assert_called_once_with(
        2,
        100,
        300,
        request_id="req-prefill",
        loop_label="chunked_step",
    )


# --------------------------------------------------------------------------
# Scheduler._requeue_or_fail_prefill
# --------------------------------------------------------------------------


def _requeue_ctx():
    """Minimal stand-in for the requeue helper's scheduler state."""
    from collections import deque

    ns = SimpleNamespace(
        requests={},
        waiting=deque(),
        _specprefill_active_request_id=None,
        model=SimpleNamespace(),  # no _language_model attr → rope restore skipped
        _MAX_PREFILL_OOM_RETRIES=2,
        _reclaim_prefill_headroom=lambda: 0,
    )
    return ns


def _fake_request(rid="req-1"):
    return SimpleNamespace(
        request_id=rid,
        prefill_oom_retries=0,
        prompt_token_ids=[1, 2, 3, 4],
        status=None,
        batch_uid="u",
        prompt_cache=object(),
        cached_tokens=128,
        remaining_tokens=None,
        block_table=object(),
        shared_prefix_blocks=2,
        output_token_ids=[9],
        output_text="x",
        num_computed_tokens=10,
        _extracted_cache=object(),
        _model_cache_config=object(),
        think_prefix_sent=True,
        _prefill_saved_rope_deltas=None,
    )


def test_requeue_non_memory_error_fails_immediately():
    ns = _requeue_ctx()
    req = _fake_request()
    out = Scheduler._requeue_or_fail_prefill(ns, req, RuntimeError("boom: bad weights"))
    assert out is False
    assert len(ns.waiting) == 0


def test_requeue_memory_error_requeues_then_resets_state():
    ns = _requeue_ctx()
    req = _fake_request()
    out = Scheduler._requeue_or_fail_prefill(
        ns, req, RuntimeError("Memory limit exceeded during prefill")
    )
    assert out is True
    assert req.prefill_oom_retries == 1
    # Re-registered + requeued, with cache state reset for a cold re-prefill.
    assert ns.requests[req.request_id] is req
    assert list(ns.waiting) == [req]
    assert req.prompt_cache is None
    assert req.cached_tokens == 0
    assert req.block_table is None
    assert req.remaining_tokens == req.prompt_token_ids
    assert req.output_token_ids == []


def test_requeue_budget_exhausts_to_clean_error():
    ns = _requeue_ctx()
    req = _fake_request()
    err = RuntimeError("Memory limit exceeded during prefill")
    # Two retries succeed (1, 2); the third is denied.
    assert Scheduler._requeue_or_fail_prefill(ns, req, err) is True
    assert Scheduler._requeue_or_fail_prefill(ns, req, err) is True
    assert Scheduler._requeue_or_fail_prefill(ns, req, err) is False
    assert req.prefill_oom_retries == 2
