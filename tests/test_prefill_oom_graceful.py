# SPDX-License-Identifier: Apache-2.0
"""Tests for graceful prefill memory management (predictive throttle +
bounded requeue) added to keep coding-agent workloads from hard-failing
mid-prefill under memory pressure.

Covers:
  - MemoryMonitor.estimate_chunk_transient_bytes math (head_dim > 128 vs <= 128)
  - Scheduler._adaptive_chunk_size predictive sizing (EWMA + static first chunk,
    early-return below the soft watermark, min-chunk floor, bucket clamp)
  - Scheduler._requeue_or_fail_prefill budget behavior + error-type gating

All tests are unit-level: the throttle/requeue logic is exercised on a light
fake object so no model load or GPU is required.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from omlx import scheduler as sched_mod
from omlx.memory_monitor import MemoryMonitor
from omlx.prefill_transient_tracker import PrefillTransientTracker
from omlx.scheduler import Scheduler

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


def test_chunk_transient_head_dim_gt_128_scales_with_kv_len():
    """head_dim > 128 → SDPA fallback materializes n_q*n*kv_len*4 (+ output)."""
    m = _monitor(head_dim=192)
    n_q, hd = 32, 192
    n_tokens, kv_len = 4, 10_000
    expected = n_q * n_tokens * kv_len * 4 + n_q * n_tokens * hd * 4
    assert m.estimate_chunk_transient_bytes(n_tokens, kv_len) == expected
    # Doubling kv_len roughly doubles the transient (kv term dominates).
    bigger = m.estimate_chunk_transient_bytes(n_tokens, kv_len * 2)
    assert bigger > expected


def test_chunk_transient_head_dim_le_128_is_kv_independent():
    """head_dim <= 128 → fused kernel, O(n): only the output buffer."""
    m = _monitor(head_dim=128)
    n_q, hd, n_tokens = 32, 128, 4
    expected = n_q * n_tokens * hd * 4
    assert m.estimate_chunk_transient_bytes(n_tokens, 10_000) == expected
    # kv_len must not change the estimate for the fused path.
    assert m.estimate_chunk_transient_bytes(n_tokens, 1) == expected


def test_chunk_transient_zero_when_model_info_missing():
    m = MemoryMonitor(max_kv_cache_memory=_GB)  # no set_model_info
    assert m.estimate_chunk_transient_bytes(4, 1000) == 0


# --------------------------------------------------------------------------
# Scheduler._adaptive_chunk_size
# --------------------------------------------------------------------------


def _throttle_ctx(*, current, hard, soft_ratio=0.80, samples_bpt=None,
                  monitor=None, min_chunk=32):
    """Build a minimal stand-in carrying the attributes _adaptive_chunk_size
    reads, plus a patch of the module memory probes to a fixed `current`."""
    tracker = PrefillTransientTracker()
    if samples_bpt is not None:
        # Seed the EWMA with one observation of the given bytes/token.
        tracker.update(1, int(samples_bpt))
    ns = SimpleNamespace(
        _memory_limit_bytes=int(hard * 0.85),       # soft = ceiling*0.85
        _memory_hard_limit_bytes=int(hard),
        _prefill_safe_zone_ratio=soft_ratio,
        _prefill_min_chunk_tokens=min_chunk,
        _prefill_transient_tracker=tracker,
        memory_monitor=monitor,
        _PREFILL_STEP_TIERS=Scheduler._PREFILL_STEP_TIERS,
        _PREFILL_HEADROOM_SAFETY=Scheduler._PREFILL_HEADROOM_SAFETY,
    )
    return ns


def _call(ns, requested, kv_len=0):
    with patch.object(sched_mod.mx, "get_active_memory", return_value=0), \
         patch.object(sched_mod, "get_phys_footprint",
                      return_value=ns._fake_current):
        return Scheduler._adaptive_chunk_size(
            ns, requested, request_id="r", loop_label="test", kv_len=kv_len
        )


def test_throttle_noop_when_full_chunk_fits():
    """If the full requested chunk's predicted peak fits, it runs unchanged —
    even at a low baseline (gate is on predicted peak, not the watermark)."""
    hard = 40 * _GB
    # Small per-token transient (~1MB/tok): 2048 tokens ≈ 2GB, easily fits.
    ns = _throttle_ctx(current=int(hard * 0.5), hard=hard,
                       samples_bpt=1024 * 1024)
    ns._fake_current = int(hard * 0.5)
    assert _call(ns, 2048, kv_len=5000) == 2048


def test_throttle_shrinks_big_chunk_from_low_baseline():
    """The regression that mattered: a huge per-token transient (MoE-like)
    must shrink the chunk even when current is well BELOW the soft watermark."""
    hard = 40 * _GB
    current = int(hard * 0.5)  # 20GB — below soft watermark (0.85*0.80*40=27.2GB)
    bpt = 18 * 1024 * 1024  # ~18 MB/token, matching the observed MoE prefill
    ns = _throttle_ctx(current=current, hard=hard, samples_bpt=bpt)
    ns._fake_current = current
    safe_target = int(hard * Scheduler._PREFILL_HEADROOM_SAFETY)
    expected = int((safe_target - current) / (bpt * 1.2))
    n = _call(ns, 2048, kv_len=5000)
    assert n < 2048                      # throttled despite low baseline
    assert n >= ns._prefill_min_chunk_tokens
    assert n <= expected + 1             # sized to the predicted-peak headroom


def test_throttle_floors_at_min_chunk_when_over_ceiling():
    """At/over the cap, the smallest step is returned (loop handles the rest)."""
    hard = 40 * _GB
    ns = _throttle_ctx(current=hard + _GB, hard=hard, samples_bpt=1_000_000,
                       min_chunk=32)
    ns._fake_current = hard + _GB
    assert _call(ns, 2048, kv_len=5000) == 32


def test_throttle_first_chunk_uses_static_estimate():
    """No EWMA samples yet → fall back to the static SDPA per-token estimate."""
    hard = 40 * _GB
    current = int(hard * 0.5)
    monitor = _monitor(head_dim=192)  # head_dim>128 so estimate is non-trivial
    ns = _throttle_ctx(current=current, hard=hard, samples_bpt=None,
                       monitor=monitor)
    ns._fake_current = current
    # Very large context → static SDPA per-token estimate is big enough that a
    # full 2048-token chunk's predicted peak exceeds the target → shrink.
    n = _call(ns, 2048, kv_len=2_000_000)
    assert n < 2048
    assert n >= ns._prefill_min_chunk_tokens


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
