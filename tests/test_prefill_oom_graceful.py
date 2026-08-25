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

import logging
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
        # Component breakdown the enforcer propagates. Left at 0 here so the
        # guard's diagnostic falls back to "effective ceiling" + generic
        # advice; the binding-aware variants are covered in
        # tests/test_engine_preflight.py.
        _memory_static_ceiling_bytes=0,
        _memory_dynamic_ceiling_bytes=0,
        _memory_metal_cap_bytes=0,
        _memory_guard_tier="balanced",
        _prefill_safe_zone_ratio=soft_ratio,
        _prefill_min_chunk_tokens=min_chunk,
        _prefill_abort_margin=abort_margin,
        _prefill_headroom_safety=Scheduler._PREFILL_HEADROOM_SAFETY,
        _prefill_speed_priority=False,
        # Dedupes the once-per-request INFO throttle notice.
        _throttle_notified_requests=set(),
        _prefill_transient_tracker=tracker,
        memory_monitor=monitor,
        _PREFILL_STEP_TIERS=Scheduler._PREFILL_STEP_TIERS,
        _PREFILL_HEADROOM_SAFETY=Scheduler._PREFILL_HEADROOM_SAFETY,
        _PREFILL_ABORT_MARGIN=Scheduler._PREFILL_ABORT_MARGIN,
        _PREFILL_TRANSIENT_SAFETY=Scheduler._PREFILL_TRANSIENT_SAFETY,
        _last_mlx_active_memory_bytes=0,
    )
    # Bind the real helper methods so the stand-in behaves like a Scheduler.
    ns._snap_chunk_size = Scheduler._snap_chunk_size.__get__(ns, Scheduler)
    ns._current_usage_bytes = Scheduler._current_usage_bytes.__get__(ns, Scheduler)
    ns._predicted_chunk_transient = Scheduler._predicted_chunk_transient.__get__(
        ns, Scheduler
    )
    ns._admission_transient_bound = Scheduler._admission_transient_bound.__get__(
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

    # A second pause is allowed: the first pass can be satisfied by a
    # marginal transient reclaim without ever reaching the durable rungs
    # (ANE bank release), so recurring pressure earns one more shot at the
    # ladder before the guard falls back to throttling for good.
    with pytest.raises(_PrefillEvictionNeeded):
        _call(ns, 2048)
    assert request.prefill_eviction_retries == 2

    # The third time the request does not loop on eviction; it throttles.
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


def test_throttle_emits_one_info_notice_per_request(caplog):
    """A throttled prefill has to be visible at the default log level.

    The per-chunk shrink line is DEBUG, so a request running an order of
    magnitude slower used to leave no trace in a normal server log — the
    gap that made a measured 2.6x slowdown undiagnosable. One INFO line
    per request, not per chunk: a 6k-token prompt floored at 32 tokens
    submits ~190 chunks.
    """
    hard = 40 * _GB
    ns = _throttle_ctx(
        current=hard + _GB, hard=hard, samples_bpt=1_000_000, min_chunk=32
    )
    ns._fake_current = hard + _GB

    with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
        for _ in range(5):
            _call(ns, 2048, kv_len=5000)

    notices = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and "Prefill throttled" in r.getMessage()
    ]
    assert len(notices) == 1
    message = notices[0].getMessage()
    assert "chunk 2048 -> 32" in message
    assert "floor=32" in message


def test_throttle_notice_repeats_for_a_new_request():
    """The dedupe is per request id, not process-wide."""
    hard = 40 * _GB
    ns = _throttle_ctx(
        current=hard + _GB, hard=hard, samples_bpt=1_000_000, min_chunk=32
    )
    ns._fake_current = hard + _GB
    with (
        patch.object(sched_mod.mx, "get_active_memory", return_value=0),
        patch.object(sched_mod, "get_phys_footprint", return_value=ns._fake_current),
    ):
        for rid in ("r1", "r2"):
            Scheduler._adaptive_chunk_size(
                ns, 2048, request_id=rid, loop_label="test", kv_len=5000
            )
    assert ns._throttle_notified_requests == {"r1", "r2"}


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
    # The abort has to leave the user a knob, and it must point up the tier
    # ladder — "lower memory_guard_tier" shrinks the ceiling that just
    # rejected them.
    assert "Raise memory_guard_tier (safe → balanced → aggressive)" in str(exc.value)
    assert "lower memory_guard_tier" not in str(exc.value)
    assert exc.value.estimated_bytes is not None
    assert exc.value.limit_bytes == int(hard * Scheduler._PREFILL_ABORT_MARGIN)


def test_guard_rejection_logs_admission_terms_breakdown(caplog):
    """The Phase 0.1 diagnostic line (docs/qwen35-hardening-and-optimization.md)
    must break the admission bound down into its separate contributors on
    every rejection, so a rejection is diagnosable from one log line without
    re-deriving which term actually bound."""
    hard = 42 * _GB
    current = 41 * _GB
    bpt = 27 * 1024 * 1024
    ns = _throttle_ctx(current=current, hard=hard, samples_bpt=bpt, reclaim_to=current)
    ns._fake_current = current

    with caplog.at_level(logging.WARNING, logger="omlx.scheduler"):
        with pytest.raises(PrefillMemoryExceededError):
            _guard_call(ns, 256, kv_len=122_000)

    terms_records = [
        r for r in caplog.records if "admission terms" in r.getMessage()
    ]
    assert len(terms_records) == 1
    msg = terms_records[0].getMessage()
    assert "current=" in msg
    assert "predicted_transient=" in msg
    assert "observed_max_bytes=" in msg
    assert "ane_prefill_transient_bytes=0.00GB" in msg  # ns.memory_monitor is None


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


def test_adaptive_throttle_charges_recently_reclaimed_footprint():
    """A pool drop must remain priced until the next chunk reallocates it."""
    static_prediction = 11.18 * _GB
    released = 6.34 * _GB
    monitor = SimpleNamespace(
        estimate_chunk_transient_bytes=lambda _n, _kv: (
            static_prediction / Scheduler._PREFILL_TRANSIENT_SAFETY
        ),
        estimate_prompt_kv_bytes=lambda _n: 0,
    )
    ns = _throttle_ctx(
        current=97.23 * _GB,
        hard=119.17 * _GB,
        soft_ratio=110.23 / 119.17,
        monitor=monitor,
        abort=200 * _GB,
    )
    ns._prefill_headroom_safety = 110.23 / 119.17
    ns._fake_current = 97.23 * _GB
    ns.requests = {}
    ns.config = SimpleNamespace(model_name="model-b")
    ns._raise_prefill_eviction_if_available = (
        Scheduler._raise_prefill_eviction_if_available.__get__(ns, Scheduler)
    )
    ns._record_chunk_transient = Scheduler._record_chunk_transient.__get__(
        ns, Scheduler
    )

    assert _call(ns, 2048, kv_len=147_680) == 2048

    ns._record_chunk_transient(
        512,
        100 * _GB,
        100 * _GB - released,
        request_id="r",
        loop_label="test",
        requested_step=512,
    )

    assert _call(ns, 2048, kv_len=147_680) < 2048


def test_predicted_transient_does_not_double_count_reclaim_covered_by_raw():
    """A conservative raw-last sample may already cover pool reallocation."""
    raw_prediction = 11.83 * _GB
    static_prediction = 4.11 * _GB
    released = 6.86 * _GB
    raw_per_token = raw_prediction / (
        512 * Scheduler._PREFILL_TRANSIENT_SAFETY
    )
    monitor = SimpleNamespace(
        estimate_chunk_transient_bytes=lambda _n, _kv: (
            static_prediction / Scheduler._PREFILL_TRANSIENT_SAFETY
        ),
        estimate_prompt_kv_bytes=lambda _n: 0,
    )
    ns = _throttle_ctx(
        current=99.12 * _GB,
        hard=118.71 * _GB,
        samples_bpt=raw_per_token,
        monitor=monitor,
    )
    ns._record_chunk_transient = Scheduler._record_chunk_transient.__get__(
        ns, Scheduler
    )
    ns._record_chunk_transient(
        512,
        100 * _GB,
        100 * _GB - released,
        request_id="r",
        loop_label="test",
        requested_step=512,
    )

    predicted = ns._predicted_chunk_transient(512, 186_368)

    assert predicted == pytest.approx(raw_prediction)


def test_sub_floor_tail_release_is_charged():
    """A release on a tail below min_chunk still feeds the reclaim ledger."""
    released = 6 * _GB
    ns = _throttle_ctx(current=97 * _GB, hard=119 * _GB)
    ns._record_chunk_transient = Scheduler._record_chunk_transient.__get__(
        ns, Scheduler
    )

    ns._record_chunk_transient(
        17,
        100 * _GB,
        100 * _GB - released,
        request_id="r",
        loop_label="test",
        requested_step=2048,
    )

    assert ns._prefill_transient_tracker.recent_reclaim_bytes == released


def test_skipped_positive_sample_clears_reclaim_charge():
    """Any positive delta drops the charge, even on EWMA-skipped samples."""
    released = 6 * _GB
    ns = _throttle_ctx(current=97 * _GB, hard=119 * _GB)
    ns._record_chunk_transient = Scheduler._record_chunk_transient.__get__(
        ns, Scheduler
    )
    ns._record_chunk_transient(
        512,
        100 * _GB,
        100 * _GB - released,
        request_id="r",
        loop_label="test",
        requested_step=512,
    )
    assert ns._prefill_transient_tracker.recent_reclaim_bytes == released

    # Positive growth on a sub-floor tail is excluded from the EWMA but the
    # footprint recovered, so the one-shot charge must not stay armed.
    ns._record_chunk_transient(
        17,
        94 * _GB,
        99 * _GB,
        request_id="r",
        loop_label="test",
        requested_step=2048,
    )
    assert ns._prefill_transient_tracker.recent_reclaim_bytes == 0


def test_speed_partial_positive_clears_reclaim_charge():
    """Speed-priority partial chunks also confirm reallocation."""
    released = 6 * _GB
    ns = _throttle_ctx(current=97 * _GB, hard=119 * _GB)
    ns._prefill_speed_priority = True
    ns._record_chunk_transient = Scheduler._record_chunk_transient.__get__(
        ns, Scheduler
    )
    ns._record_chunk_transient(
        512,
        100 * _GB,
        100 * _GB - released,
        request_id="r",
        loop_label="test",
        requested_step=512,
    )
    assert ns._prefill_transient_tracker.recent_reclaim_bytes == released

    ns._record_chunk_transient(
        256,
        94 * _GB,
        99 * _GB,
        request_id="r",
        loop_label="test",
        requested_step=512,
    )
    assert ns._prefill_transient_tracker.recent_reclaim_bytes == 0


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


def test_record_chunk_transient_marks_floor_samples_only():
    """Only floor-size chunks may feed the observed max the admission
    charge uses; big-chunk transients stay EWMA-only."""
    tracker = PrefillTransientTracker()
    ns = SimpleNamespace(
        _prefill_min_chunk_tokens=32,
        _prefill_transient_tracker=tracker,
    )
    ns._record_chunk_transient = Scheduler._record_chunk_transient.__get__(
        ns, Scheduler
    )

    # First sample is always excluded from the max (seed noise).
    ns._record_chunk_transient(32, 0, 100, request_id="r", loop_label="unit")
    # Big chunk: EWMA only, never the max.
    ns._record_chunk_transient(
        2048, 0, 3 * 1024**3, request_id="r", loop_label="unit"
    )
    assert tracker.observed_max_bytes == 0
    # Floor chunk: enters the max.
    ns._record_chunk_transient(
        32, 0, 200 * 1024**2, request_id="r", loop_label="unit"
    )
    assert tracker.observed_max_bytes == 200 * 1024**2


def test_record_chunk_transient_skips_partial_speed_sample():
    """A speed tail must not replace the last representative full step."""
    tracker = PrefillTransientTracker()
    ns = SimpleNamespace(
        _prefill_min_chunk_tokens=32,
        _prefill_speed_priority=True,
        _prefill_transient_tracker=tracker,
        _PREFILL_TRANSIENT_SAFETY=Scheduler._PREFILL_TRANSIENT_SAFETY,
        memory_monitor=None,
    )
    ns._record_chunk_transient = Scheduler._record_chunk_transient.__get__(
        ns, Scheduler
    )

    full_delta = 512 * 1024**2
    ns._record_chunk_transient(
        2048,
        0,
        full_delta,
        request_id="req-full",
        loop_label="unit",
        requested_step=2048,
    )
    baseline_ewma = tracker.bytes_per_token

    ns._record_chunk_transient(
        185,
        0,
        int(10497.1 * 1024 * 185),
        request_id="req-tail",
        loop_label="unit",
        requested_step=2048,
    )

    assert tracker.samples == 1
    assert tracker.bytes_per_token == baseline_ewma
    assert tracker.last_n_tokens == 2048
    assert tracker.last_delta_bytes == full_delta
    predicted = Scheduler._predicted_chunk_transient(ns, 2048, 65_000)
    assert predicted == pytest.approx(
        full_delta * Scheduler._PREFILL_TRANSIENT_SAFETY
    )


def test_record_chunk_transient_keeps_full_speed_spike_as_last_sample():
    """A same-size spike remains available to protect the next full step."""
    tracker = PrefillTransientTracker()
    ns = SimpleNamespace(
        _prefill_min_chunk_tokens=32,
        _prefill_speed_priority=True,
        _prefill_transient_tracker=tracker,
    )
    ns._record_chunk_transient = Scheduler._record_chunk_transient.__get__(
        ns, Scheduler
    )

    ns._record_chunk_transient(
        2048,
        0,
        256 * 1024**2,
        request_id="req-baseline",
        loop_label="unit",
        requested_step=2048,
    )
    spike = 3 * 1024**3
    ns._record_chunk_transient(
        2048,
        0,
        spike,
        request_id="req-spike",
        loop_label="unit",
        requested_step=2048,
    )

    assert tracker.samples == 2
    assert tracker.last_n_tokens == 2048
    assert tracker.last_delta_bytes == spike


def test_record_chunk_transient_keeps_partial_context_sample():
    """Context priority still learns from adaptively reduced chunks."""
    tracker = PrefillTransientTracker()
    ns = SimpleNamespace(
        _prefill_min_chunk_tokens=32,
        _prefill_speed_priority=False,
        _prefill_transient_tracker=tracker,
    )
    ns._record_chunk_transient = Scheduler._record_chunk_transient.__get__(
        ns, Scheduler
    )

    partial_delta = 128 * 1024**2
    ns._record_chunk_transient(
        512,
        0,
        partial_delta,
        request_id="req-context",
        loop_label="unit",
        requested_step=2048,
    )

    assert tracker.samples == 1
    assert tracker.last_n_tokens == 512
    assert tracker.last_delta_bytes == partial_delta


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
        _supports_skip_lm_head=lambda: False,
        _adaptive_chunk_size=lambda n, **kwargs: events.append("adaptive") or n,
        _guard_prefill_chunk=lambda n, **kwargs: events.append("guard") or n,
        _record_chunk_transient=MagicMock(),
        _maybe_record_fixed_state_bytes=MagicMock(),
    )
    ns.running = {}
    ns._decode_fairness = True
    ns._decode_time_owed_s = 0.0
    ns._decode_activity_key = "test-engine"
    ns._prefill_tps_best = None
    for _name in (
        "_prefill_step_size_for_progress",
        "_base_prefill_step_size",
        "_contended_prefill_cap",
        "_decode_contention",
        "_others_decoding",
        "_should_clear_after_chunk",
        "_accrue_decode_debt",
    ):
        setattr(ns, _name, getattr(Scheduler, _name).__get__(ns, Scheduler))
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
        kv_len=0,
        requested_step=2,
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


# --------------------------------------------------------------------------
# Scheduler._snap_chunk_size
# --------------------------------------------------------------------------


class TestSnapChunkSize:
    def _ns(self, min_chunk=32):
        ns = SimpleNamespace(_prefill_min_chunk_tokens=min_chunk)
        ns._snap_chunk_size = Scheduler._snap_chunk_size.__get__(ns, Scheduler)
        return ns

    def test_off_grid_sizes_snap_down_to_multiple(self):
        ns = self._ns()
        assert ns._snap_chunk_size(33, 2048) == 32
        assert ns._snap_chunk_size(63, 2048) == 32
        assert ns._snap_chunk_size(100, 2048) == 96
        assert ns._snap_chunk_size(1023, 2048) == 992

    def test_on_grid_sizes_unchanged(self):
        ns = self._ns()
        assert ns._snap_chunk_size(64, 2048) == 64
        assert ns._snap_chunk_size(512, 2048) == 512

    def test_at_or_below_floor_unchanged(self):
        """A short tail before a block boundary keeps its exact size."""
        ns = self._ns()
        assert ns._snap_chunk_size(32, 2048) == 32
        assert ns._snap_chunk_size(17, 2048) == 17

    def test_unthrottled_chunk_untouched(self):
        ns = self._ns()
        assert ns._snap_chunk_size(2048, 2048) == 2048
        assert ns._snap_chunk_size(2049, 2048) == 2049

    def test_env_toggle_disables_snapping(self, monkeypatch):
        monkeypatch.setenv("OMLX_CHUNK_SNAP", "0")
        ns = self._ns()
        assert ns._snap_chunk_size(33, 2048) == 33

    def test_respects_min_chunk_grid(self):
        ns = self._ns(min_chunk=256)
        assert ns._snap_chunk_size(300, 2048) == 256
        assert ns._snap_chunk_size(700, 2048) == 512


# --------------------------------------------------------------------------
# observed_max transient bound: guard/admission only, never chunk sizing
# --------------------------------------------------------------------------


def test_adaptive_chunk_size_ignores_observed_max():
    """FROZEN policy pin: the throttle's sizing must not move when the
    session records a large observed max transient."""
    hard = 40 * _GB
    ns = _throttle_ctx(current=int(hard * 0.9), hard=hard, samples_bpt=2 * 1024**2)
    ns._fake_current = int(hard * 0.9)
    before = _call(ns, 2048, kv_len=5000)
    assert before < 2048, "precondition: the throttle must actually shrink"

    ns._prefill_transient_tracker._observed_max_bytes = 8 * _GB
    after = _call(ns, 2048, kv_len=5000)
    assert after == before


def test_guard_abort_gate_charges_observed_max():
    """The pre-chunk guard's abort gate prices the flat observed max, so a
    doomed prefill stops deterministically before the dangerous chunk."""
    hard = 40 * _GB  # cap = 36GB (0.9 margin)
    current = 35 * _GB
    ns = _throttle_ctx(current=current, hard=hard, samples_bpt=1024 * 1024)
    ns._fake_current = current

    # Without the observed max: min-chunk transient (~42MB) fits, so the
    # guard shrinks instead of aborting.
    assert _guard_call(ns, 2048, kv_len=50_000) < 2048

    # 2GB observed max no longer fits under the 1GB headroom: abort.
    ns._prefill_transient_tracker._observed_max_bytes = 2 * _GB
    with pytest.raises(PrefillMemoryExceededError):
        _guard_call(ns, 2048, kv_len=50_000)


def test_guard_shrink_math_unchanged_by_observed_max():
    """When the observed max still fits, the shrink arithmetic stays on the
    frozen per-token predictor and returns the same chunk."""
    hard = 40 * _GB
    current = 35 * _GB
    ns = _throttle_ctx(current=current, hard=hard, samples_bpt=1024 * 1024)
    ns._fake_current = current
    before = _guard_call(ns, 2048, kv_len=50_000)
    assert before < 2048

    ns._prefill_transient_tracker._observed_max_bytes = 512 * 1024**2
    after = _guard_call(ns, 2048, kv_len=50_000)
    assert after == before


# --------------------------------------------------------------------------
# Fixed recurrent-state one-shot probe
# --------------------------------------------------------------------------


class TestMaybeRecordFixedStateBytes:
    def _ns(self, armed=True):
        ns = SimpleNamespace(
            memory_monitor=MagicMock(),
            _fixed_state_measure_armed=armed,
            _fixed_state_recorded=False,
        )
        ns._maybe_record_fixed_state_bytes = (
            Scheduler._maybe_record_fixed_state_bytes.__get__(ns, Scheduler)
        )
        return ns

    def _arrays_cache(self, nbytes_list):
        cls = type(
            "ArraysCache",
            (),
            {"state": [SimpleNamespace(nbytes=n) for n in nbytes_list]},
        )
        return cls()

    def test_measures_once_and_sums_state_nbytes(self):
        ns = self._ns()
        caches = [self._arrays_cache([100, 200]), self._arrays_cache([300])]
        ns._maybe_record_fixed_state_bytes(caches)
        ns.memory_monitor.set_fixed_state_bytes.assert_called_once_with(600)
        assert ns._fixed_state_recorded is True
        # Second call is a no-op flag check.
        ns._maybe_record_fixed_state_bytes(caches)
        ns.memory_monitor.set_fixed_state_bytes.assert_called_once()

    def test_unarmed_never_measures(self):
        ns = self._ns(armed=False)
        ns._maybe_record_fixed_state_bytes([self._arrays_cache([100])])
        ns.memory_monitor.set_fixed_state_bytes.assert_not_called()
        assert ns._fixed_state_recorded is False

    def test_non_arrays_caches_contribute_zero(self):
        ns = self._ns()
        plain = type("KVCache", (), {"state": [SimpleNamespace(nbytes=999)]})()
        ns._maybe_record_fixed_state_bytes([plain, self._arrays_cache([50])])
        ns.memory_monitor.set_fixed_state_bytes.assert_called_once_with(50)

    def test_zero_total_marks_recorded_without_setting(self):
        ns = self._ns()
        ns._maybe_record_fixed_state_bytes(
            [type("KVCache", (), {"state": []})()]
        )
        ns.memory_monitor.set_fixed_state_bytes.assert_not_called()
        assert ns._fixed_state_recorded is True



# --------------------------------------------------------------------------
# Prefill speed priority (never shrink)
# --------------------------------------------------------------------------


def test_speed_priority_throttle_keeps_full_chunk_under_pressure():
    """Speed mode returns the requested chunk even when the predicted peak
    misses the sizing target — the guard/abort path handles overruns."""
    hard = 40 * _GB
    current = int(hard * 0.5)
    bpt = 18 * 1024 * 1024  # shrinks hard in context mode
    ns = _throttle_ctx(current=current, hard=hard, samples_bpt=bpt)
    ns._fake_current = current
    ns._prefill_speed_priority = True
    assert _call(ns, 2048, kv_len=5000) == 2048


def test_speed_priority_context_mode_shrink_unchanged():
    """Control: the same pressure with the default flag still shrinks."""
    hard = 40 * _GB
    current = int(hard * 0.5)
    bpt = 18 * 1024 * 1024
    ns = _throttle_ctx(current=current, hard=hard, samples_bpt=bpt)
    ns._fake_current = current
    assert _call(ns, 2048, kv_len=5000) < 2048


def test_speed_priority_guard_aborts_at_full_step_instead_of_shrinking():
    """The guard's abort gate charges the full chunk in speed mode: a chunk
    that context mode would shrink aborts upfront instead."""
    hard = 42 * _GB
    current = 30 * _GB
    bpt = 27 * 1024 * 1024
    ns = _throttle_ctx(current=current, hard=hard, samples_bpt=bpt, reclaim_to=current)
    ns._fake_current = current
    # Context-mode control on the identical setup shrinks (guard test above).
    assert _guard_call(ns, 2048, kv_len=122_000) < 2048
    ns._prefill_speed_priority = True
    with pytest.raises(PrefillMemoryExceededError):
        _guard_call(ns, 2048, kv_len=122_000)


def test_speed_priority_guard_passes_full_chunk_that_fits():
    hard = 42 * _GB
    ns = _throttle_ctx(current=10 * _GB, hard=hard, samples_bpt=1024 * 1024)
    ns._fake_current = 10 * _GB
    ns._prefill_speed_priority = True
    assert _guard_call(ns, 2048, kv_len=5000) == 2048
