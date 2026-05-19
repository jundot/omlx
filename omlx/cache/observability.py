# SPDX-License-Identifier: Apache-2.0
import threading
import time
from collections import deque
from typing import Any


_DEFAULT_WINDOWS = (60, 300, 900)
_MAX_SNAPSHOTS = 90
_MIN_INTERVAL = 10.0


class CacheRateTracker:

    def __init__(
        self,
        max_snapshots: int = _MAX_SNAPSHOTS,
        min_interval: float = _MIN_INTERVAL,
    ):
        self._snapshots: deque[tuple[float, dict[str, int]]] = deque(
            maxlen=max_snapshots
        )
        self._min_interval = min_interval
        self._lock = threading.Lock()

    def maybe_snapshot(self, counters: dict[str, int]) -> bool:
        with self._lock:
            now = time.monotonic()
            if self._snapshots and (now - self._snapshots[-1][0]) < self._min_interval:
                return False
            self._snapshots.append((now, dict(counters)))
            return True

    def get_rates(
        self, windows: tuple[int, ...] = _DEFAULT_WINDOWS
    ) -> dict[str, Any]:
        with self._lock:
            if not self._snapshots:
                return {"windows": {}, "cumulative": {}}

            now = self._snapshots[-1][0]
            newest = self._snapshots[-1][1]

            window_rates = {}
            for w in windows:
                label = _window_label(w)
                baseline_ts = None
                baseline_counters = None
                for ts, counters in self._snapshots:
                    if (now - ts) <= w:
                        baseline_ts, baseline_counters = ts, counters
                        break
                if baseline_ts is None:
                    baseline_ts, baseline_counters = self._snapshots[0]
                elapsed = now - baseline_ts
                if elapsed < 1.0:
                    window_rates[label] = {}
                    continue
                window_rates[label] = _compute_window(
                    baseline_counters, newest, elapsed
                )

            cumulative = _compute_cumulative(newest)
            return {"windows": window_rates, "cumulative": cumulative}

    def snapshot_and_get_rates(
        self,
        counters: dict[str, int],
        windows: tuple[int, ...] = _DEFAULT_WINDOWS,
    ) -> dict[str, Any]:
        self.maybe_snapshot(counters)
        return self.get_rates(windows)

    def clear(self) -> None:
        with self._lock:
            self._snapshots.clear()


def _window_label(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m"


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _compute_window(
    old: dict[str, int], new: dict[str, int], elapsed: float
) -> dict[str, Any]:
    def delta(key: str) -> int:
        return max(0, new.get(key, 0) - old.get(key, 0))

    d_prefix_hits = delta("prefix_hits")
    d_prefix_misses = delta("prefix_misses")
    d_evictions = delta("evictions")
    d_ssd_hot = delta("ssd_hot_hits")
    d_ssd_disk = delta("ssd_disk_loads")
    d_tokens_matched = delta("prefix_tokens_matched")
    d_tokens_requested = delta("prefix_tokens_requested")

    minutes = elapsed / 60.0

    return {
        "prefix_hit_rate": round(
            _safe_ratio(d_prefix_hits, d_prefix_hits + d_prefix_misses), 4
        ),
        "prefix_hits": d_prefix_hits,
        "prefix_misses": d_prefix_misses,
        "prefix_match_efficiency": round(
            _safe_ratio(d_tokens_matched, d_tokens_requested), 4
        ),
        "evictions": d_evictions,
        "eviction_rate_per_min": round(d_evictions / minutes, 2) if minutes > 0 else 0.0,
        "ssd_hot_hits": d_ssd_hot,
        "ssd_disk_loads": d_ssd_disk,
        "ssd_hot_rate": round(
            _safe_ratio(d_ssd_hot, d_ssd_hot + d_ssd_disk), 4
        ),
    }


def _compute_cumulative(counters: dict[str, int]) -> dict[str, Any]:
    prefix_hits = counters.get("prefix_hits", 0)
    prefix_misses = counters.get("prefix_misses", 0)
    ssd_hot = counters.get("ssd_hot_hits", 0)
    ssd_disk = counters.get("ssd_disk_loads", 0)
    tokens_matched = counters.get("prefix_tokens_matched", 0)
    tokens_requested = counters.get("prefix_tokens_requested", 0)

    return {
        "prefix_hits": prefix_hits,
        "prefix_misses": prefix_misses,
        "prefix_hit_rate": round(_safe_ratio(prefix_hits, prefix_hits + prefix_misses), 4),
        "prefix_tokens_saved": counters.get("prefix_tokens_saved", 0),
        "prefix_match_efficiency": round(
            _safe_ratio(tokens_matched, tokens_requested), 4
        ),
        "evictions": counters.get("evictions", 0),
        "ssd_hot_hits": ssd_hot,
        "ssd_disk_loads": ssd_disk,
        "ssd_saves": counters.get("ssd_saves", 0),
        "hot_cache_evictions": counters.get("hot_cache_evictions", 0),
        "hot_cache_promotions": counters.get("hot_cache_promotions", 0),
        "ssd_hot_rate": round(_safe_ratio(ssd_hot, ssd_hot + ssd_disk), 4),
    }
