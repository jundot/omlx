# tests/test_cache_observability.py
# SPDX-License-Identifier: Apache-2.0
"""Tests for cache observability module."""

import threading
import time
from unittest.mock import patch

import pytest

from omlx.cache.observability import CacheRateTracker


def _make_counters(
    prefix_hits=0,
    prefix_misses=0,
    prefix_tokens_matched=0,
    prefix_tokens_requested=0,
    prefix_tokens_saved=0,
    evictions=0,
    ssd_hot_hits=0,
    ssd_disk_loads=0,
    ssd_saves=0,
    ssd_errors=0,
    hot_cache_evictions=0,
    hot_cache_promotions=0,
):
    return {
        "prefix_hits": prefix_hits,
        "prefix_misses": prefix_misses,
        "prefix_tokens_matched": prefix_tokens_matched,
        "prefix_tokens_requested": prefix_tokens_requested,
        "prefix_tokens_saved": prefix_tokens_saved,
        "evictions": evictions,
        "ssd_hot_hits": ssd_hot_hits,
        "ssd_disk_loads": ssd_disk_loads,
        "ssd_saves": ssd_saves,
        "ssd_errors": ssd_errors,
        "hot_cache_evictions": hot_cache_evictions,
        "hot_cache_promotions": hot_cache_promotions,
    }


class TestCacheRateTrackerSnapshot:

    def test_empty_tracker_returns_empty_rates(self):
        tracker = CacheRateTracker()
        result = tracker.get_rates()
        assert result == {"windows": {}, "cumulative": {}}

    def test_first_snapshot_always_accepted(self):
        tracker = CacheRateTracker(min_interval=10.0)
        assert tracker.maybe_snapshot(_make_counters()) is True

    def test_snapshot_rejected_within_min_interval(self):
        tracker = CacheRateTracker(min_interval=10.0)
        tracker.maybe_snapshot(_make_counters())
        assert tracker.maybe_snapshot(_make_counters()) is False

    def test_snapshot_accepted_after_min_interval(self):
        tracker = CacheRateTracker(min_interval=0.0)
        tracker.maybe_snapshot(_make_counters())
        assert tracker.maybe_snapshot(_make_counters()) is True

    def test_deque_overflow_evicts_oldest(self):
        tracker = CacheRateTracker(max_snapshots=3, min_interval=0.0)
        for i in range(5):
            tracker.maybe_snapshot(_make_counters(prefix_hits=i))
        result = tracker.get_rates()
        assert result["cumulative"]["prefix_hits"] == 4


class TestCacheRateTrackerRates:

    def _tracker_with_two_snapshots(self, old_counters, new_counters, elapsed=60.0):
        tracker = CacheRateTracker(min_interval=0.0)
        fake_time = [1000.0]

        def mock_monotonic():
            return fake_time[0]

        with patch("omlx.cache.observability.time.monotonic", side_effect=mock_monotonic):
            tracker.maybe_snapshot(old_counters)

        fake_time[0] = 1000.0 + elapsed
        with patch("omlx.cache.observability.time.monotonic", side_effect=mock_monotonic):
            tracker.maybe_snapshot(new_counters)

        with patch("omlx.cache.observability.time.monotonic", return_value=fake_time[0]):
            return tracker.get_rates(windows=(60, 300, 900))

    def test_steady_state_prefix_hit_rate(self):
        old = _make_counters(prefix_hits=100, prefix_misses=50)
        new = _make_counters(prefix_hits=200, prefix_misses=75)
        result = self._tracker_with_two_snapshots(old, new, elapsed=60.0)
        assert result["windows"]["1m"]["prefix_hit_rate"] == 0.8

    def test_zero_activity_window_no_nan(self):
        counters = _make_counters(prefix_hits=50, prefix_misses=10)
        result = self._tracker_with_two_snapshots(counters, counters, elapsed=60.0)
        assert result["windows"]["1m"]["prefix_hit_rate"] == 0.0
        assert result["windows"]["1m"]["prefix_match_efficiency"] == 0.0
        assert result["windows"]["1m"]["eviction_rate_per_min"] == 0.0

    def test_eviction_rate_per_min(self):
        old = _make_counters(evictions=10)
        new = _make_counters(evictions=40)
        result = self._tracker_with_two_snapshots(old, new, elapsed=300.0)
        assert result["windows"]["5m"]["eviction_rate_per_min"] == 6.0

    def test_prefix_match_efficiency(self):
        old = _make_counters(prefix_tokens_matched=0, prefix_tokens_requested=0)
        new = _make_counters(prefix_tokens_matched=600, prefix_tokens_requested=1000)
        result = self._tracker_with_two_snapshots(old, new, elapsed=60.0)
        assert result["windows"]["1m"]["prefix_match_efficiency"] == 0.6

    def test_ssd_hot_rate(self):
        old = _make_counters(ssd_hot_hits=0, ssd_disk_loads=0)
        new = _make_counters(ssd_hot_hits=80, ssd_disk_loads=20)
        result = self._tracker_with_two_snapshots(old, new, elapsed=60.0)
        assert result["windows"]["1m"]["ssd_hot_rate"] == 0.8

    def test_insufficient_data_returns_empty_window(self):
        tracker = CacheRateTracker(min_interval=0.0)

        with patch("omlx.cache.observability.time.monotonic", return_value=1000.0):
            tracker.maybe_snapshot(_make_counters(prefix_hits=10))

        with patch("omlx.cache.observability.time.monotonic", return_value=1000.5):
            tracker.maybe_snapshot(_make_counters(prefix_hits=20))

        with patch("omlx.cache.observability.time.monotonic", return_value=1000.5):
            result = tracker.get_rates(windows=(60,))
        assert result["windows"]["1m"] == {}

    def test_cumulative_uses_latest_snapshot(self):
        old = _make_counters(prefix_hits=10, prefix_misses=5)
        new = _make_counters(prefix_hits=100, prefix_misses=20)
        result = self._tracker_with_two_snapshots(old, new, elapsed=60.0)
        assert result["cumulative"]["prefix_hits"] == 100
        assert result["cumulative"]["prefix_misses"] == 20
        assert abs(result["cumulative"]["prefix_hit_rate"] - 0.8333) < 0.001


class TestCacheRateTrackerSnapshotAndGetRates:

    def test_combines_snapshot_and_rates(self):
        tracker = CacheRateTracker(min_interval=0.0)

        with patch("omlx.cache.observability.time.monotonic", return_value=1000.0):
            tracker.maybe_snapshot(_make_counters(prefix_hits=0))

        with patch("omlx.cache.observability.time.monotonic", return_value=1060.0):
            result = tracker.snapshot_and_get_rates(
                _make_counters(prefix_hits=80, prefix_misses=20)
            )

        assert result["windows"]["1m"]["prefix_hit_rate"] == 0.8
        assert result["cumulative"]["prefix_hits"] == 80


class TestCacheRateTrackerThreadSafety:

    def test_concurrent_snapshot_and_read(self):
        tracker = CacheRateTracker(min_interval=0.0)
        errors = []
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                try:
                    tracker.maybe_snapshot(_make_counters(prefix_hits=i))
                    i += 1
                except Exception as e:
                    errors.append(e)

        def reader():
            while not stop.is_set():
                try:
                    tracker.get_rates()
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        time.sleep(0.2)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)

        assert errors == [], f"Thread errors: {errors}"


class TestCacheRateTrackerClear:

    def test_clear_resets_state(self):
        tracker = CacheRateTracker(min_interval=0.0)
        tracker.maybe_snapshot(_make_counters(prefix_hits=100))
        tracker.clear()
        assert tracker.get_rates() == {"windows": {}, "cumulative": {}}
