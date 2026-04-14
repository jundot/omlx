# SPDX-License-Identifier: Apache-2.0
"""
Server-level metrics for the oMLX admin dashboard.

Provides a thread-safe singleton that aggregates serving metrics
across all engines/models. Session metrics reset on server start,
while all-time metrics persist across restarts via JSON file.

Enhanced with comprehensive stats:
- Request counts and rates
- Latency metrics (TTFT, percentiles)
- Throughput and batching speedup
- Token statistics
"""

import json
import logging
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

# Interval between periodic saves of all-time stats (seconds)
_SAVE_INTERVAL = 300

# Sliding window size for rate calculations (seconds)
_RATE_WINDOW = 60

# Sliding window size for latency tracking (number of requests)
_LATENCY_WINDOW = 1000


class ServerMetrics:
    """
    Global server-level metrics for the Status dashboard.

    Thread-safe: uses threading.Lock since scheduler runs in ThreadPoolExecutor.
    Tracks cumulative totals and average speeds across all requests,
    with optional per-model breakdown.

    Supports two scopes:
    - "session": resets on server restart (default, backward-compatible)
    - "alltime": persisted across restarts via stats_path JSON file
    """

    def __init__(self, stats_path=None):
        self._lock = threading.Lock()
        self._stats_path = stats_path

        # Session totals (reset on server restart or clear)
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_requests: int = 0
        self.total_prefill_duration: float = 0.0
        self.total_generation_duration: float = 0.0
        self._per_model: dict = {}

        # All-time totals (persisted across restarts)
        self._alltime_prompt_tokens: int = 0
        self._alltime_completion_tokens: int = 0
        self._alltime_cached_tokens: int = 0
        self._alltime_requests: int = 0
        self._alltime_prefill_duration: float = 0.0
        self._alltime_generation_duration: float = 0.0
        self._alltime_per_model: dict = {}

        # Sliding window for rate calculations (timestamps of requests)
        self._request_timestamps: deque = deque()

        # Sliding window for latency tracking
        self._latencies: deque = deque()
        self._prefill_latencies: deque = deque()
        self._generation_latencies: deque = deque()

        self._start_time = time.time()
        self._last_save_time = time.time()

        # Load persisted all-time stats
        if stats_path:
            self._load_alltime()

    @staticmethod
    def _new_model_counters() -> dict:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "requests": 0,
            "prefill_duration": 0.0,
            "generation_duration": 0.0,
            "ttft_samples": [],
            # DFlash speculative decoding metrics
            "dflash_cycles": 0,
            "dflash_accepted_from_draft": 0,
            "dflash_generated_tokens": 0,
            "dflash_draft_time_us": 0,
            "dflash_verify_time_us": 0,
            "queue_times": [],
        }

    def _load_alltime(self) -> None:
        """Load all-time stats from disk. Called once during __init__."""
        if not self._stats_path or not self._stats_path.exists():
            return
        try:
            with open(self._stats_path) as f:
                data = json.load(f)
            self._alltime_prompt_tokens = int(data.get("total_prompt_tokens", 0))
            self._alltime_completion_tokens = int(
                data.get("total_completion_tokens", 0)
            )
            self._alltime_cached_tokens = int(data.get("total_cached_tokens", 0))
            self._alltime_requests = int(data.get("total_requests", 0))
            self._alltime_prefill_duration = float(
                data.get("total_prefill_duration", 0.0)
            )
            self._alltime_generation_duration = float(
                data.get("total_generation_duration", 0.0)
            )
            per_model = data.get("per_model", {})
            for model_id, counters in per_model.items():
                self._alltime_per_model[model_id] = {
                    "prompt_tokens": int(counters.get("prompt_tokens", 0)),
                    "completion_tokens": int(counters.get("completion_tokens", 0)),
                    "cached_tokens": int(counters.get("cached_tokens", 0)),
                    "requests": int(counters.get("requests", 0)),
                    "prefill_duration": float(counters.get("prefill_duration", 0.0)),
                    "generation_duration": float(
                        counters.get("generation_duration", 0.0)
                    ),
                    "ttft_samples": counters.get("ttft_samples", []),
                    "queue_times": counters.get("queue_times", []),
                }
            logger.info("Loaded all-time stats from %s", self._stats_path)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError, OSError) as e:
            logger.warning("Failed to load all-time stats from %s: %s", self._stats_path, e)

    def save_alltime(self) -> None:
        """Save all-time stats to disk. Thread-safe."""
        if not self._stats_path:
            return
        with self._lock:
            data = {
                "total_prompt_tokens": self._alltime_prompt_tokens,
                "total_completion_tokens": self._alltime_completion_tokens,
                "total_cached_tokens": self._alltime_cached_tokens,
                "total_requests": self._alltime_requests,
                "total_prefill_duration": self._alltime_prefill_duration,
                "total_generation_duration": self._alltime_generation_duration,
                "per_model": dict(self._alltime_per_model),
            }
            self._last_save_time = time.time()
        try:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._stats_path.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(self._stats_path)
        except OSError as e:
            logger.warning("Failed to save all-time stats to %s: %s", self._stats_path, e)

    def _maybe_save_alltime(self) -> None:
        """Save all-time stats if enough time has passed. Called within lock."""
        if not self._stats_path:
            return
        now = time.time()
        if now - self._last_save_time >= _SAVE_INTERVAL:
            # Release lock before I/O
            self._lock.release()
            try:
                self.save_alltime()
            finally:
                self._lock.acquire()

    def record_request_start(
        self,
        model_id: str = "",
        queue_time: float = 0.0,
    ) -> None:
        """Record a request starting (for rate calculations and queue time)."""
        with self._lock:
            now = time.time()
            self._request_timestamps.append(now)

            # Trim old timestamps
            cutoff = now - _RATE_WINDOW
            while self._request_timestamps and self._request_timestamps[0] < cutoff:
                self._request_timestamps.popleft()

            # Record queue time
            if queue_time > 0:
                self._latencies.append(("queue", queue_time))
                self._trim_latencies()

    def record_request_complete(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        prefill_duration: float = 0.0,
        generation_duration: float = 0.0,
        ttft: float = 0.0,
        model_id: str = "",
    ) -> None:
        """Record a completed request. Thread-safe."""
        with self._lock:
            # Session counters
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cached_tokens += cached_tokens
            self.total_requests += 1
            self.total_prefill_duration += prefill_duration
            self.total_generation_duration += generation_duration

            # All-time counters
            self._alltime_prompt_tokens += prompt_tokens
            self._alltime_completion_tokens += completion_tokens
            self._alltime_cached_tokens += cached_tokens
            self._alltime_requests += 1
            self._alltime_prefill_duration += prefill_duration
            self._alltime_generation_duration += generation_duration

            # Record latencies
            if prefill_duration > 0:
                self._prefill_latencies.append(prefill_duration)
                self._trim_latencies()
            if generation_duration > 0:
                self._generation_latencies.append(generation_duration)
                self._trim_latencies()
            if ttft > 0:
                self._latencies.append(("ttft", ttft))
                self._trim_latencies()

            # Per-model counters (session)
            if model_id:
                if model_id not in self._per_model:
                    self._per_model[model_id] = self._new_model_counters()
                m = self._per_model[model_id]
                m["prompt_tokens"] += prompt_tokens
                m["completion_tokens"] += completion_tokens
                m["cached_tokens"] += cached_tokens
                m["requests"] += 1
                m["prefill_duration"] += prefill_duration
                m["generation_duration"] += generation_duration
                if ttft > 0:
                    m["ttft_samples"].append(ttft)
                    # Keep only last _LATENCY_WINDOW samples
                    if len(m["ttft_samples"]) > _LATENCY_WINDOW:
                        m["ttft_samples"] = m["ttft_samples"][-_LATENCY_WINDOW:]

                # Per-model counters (all-time)
                if model_id not in self._alltime_per_model:
                    self._alltime_per_model[model_id] = self._new_model_counters()
                am = self._alltime_per_model[model_id]
                am["prompt_tokens"] += prompt_tokens
                am["completion_tokens"] += completion_tokens
                am["cached_tokens"] += cached_tokens
                am["requests"] += 1
                am["prefill_duration"] += prefill_duration
                am["generation_duration"] += generation_duration
                if ttft > 0:
                    am["ttft_samples"].append(ttft)
                    if len(am["ttft_samples"]) > _LATENCY_WINDOW:
                        am["ttft_samples"] = am["ttft_samples"][-_LATENCY_WINDOW:]

            # Periodic save
            self._maybe_save_alltime()

    def record_dflash_cycle(
        self,
        model_id: str,
        accepted_from_draft: int,
        generated_tokens: int,
        draft_time_us: float = 0.0,
        verify_time_us: float = 0.0,
    ) -> None:
        """Record DFlash speculative decode cycle metrics. Thread-safe."""
        with self._lock:
            if model_id not in self._per_model:
                self._per_model[model_id] = self._new_model_counters()
            m = self._per_model[model_id]
            m["dflash_cycles"] += 1
            m["dflash_accepted_from_draft"] += accepted_from_draft
            m["dflash_generated_tokens"] += generated_tokens
            m["dflash_draft_time_us"] += draft_time_us
            m["dflash_verify_time_us"] += verify_time_us

    def _trim_latencies(self) -> None:
        """Trim latency windows to _LATENCY_WINDOW size."""
        while len(self._latencies) > _LATENCY_WINDOW:
            self._latencies.popleft()
        while len(self._prefill_latencies) > _LATENCY_WINDOW:
            self._prefill_latencies.popleft()
        while len(self._generation_latencies) > _LATENCY_WINDOW:
            self._generation_latencies.popleft()

    def _calculate_percentile(self, values: list, percentile: float) -> float:
        """Calculate percentile from a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def _build_snapshot(
        self,
        prompt: int,
        completion: int,
        cached: int,
        requests: int,
        prefill_dur: float,
        gen_dur: float,
        uptime: float,
    ) -> dict:
        """Build a metrics snapshot dict from raw values."""
        actual_processed = prompt - cached
        avg_prefill_tps = (
            actual_processed / prefill_dur if prefill_dur > 0 else 0.0
        )
        avg_generation_tps = completion / gen_dur if gen_dur > 0 else 0.0
        cache_efficiency = (cached / prompt * 100) if prompt > 0 else 0.0

        # Calculate rates
        requests_per_minute = 0.0
        if len(self._request_timestamps) >= 2:
            time_span = _RATE_WINDOW
            if time_span > 0:
                requests_per_minute = (len(self._request_timestamps) / time_span) * 60

        # Calculate latency percentiles
        ttft_values = [v for t, v in self._latencies if t == "ttft"]
        prefill_values = list(self._prefill_latencies)
        generation_values = list(self._generation_latencies)

        return {
            "total_tokens_served": prompt + completion,
            "total_cached_tokens": cached,
            "cache_efficiency": round(cache_efficiency, 1),
            "total_prompt_tokens": prompt,
            "total_completion_tokens": completion,
            "total_requests": requests,
            "avg_prefill_tps": round(avg_prefill_tps, 1),
            "avg_generation_tps": round(avg_generation_tps, 1),
            "uptime_seconds": round(uptime, 1),
            # New stats
            "requests_per_minute": round(requests_per_minute, 2),
            "total_tokens_in": prompt,
            "total_tokens_out": completion,
            "tokens_per_second_total": round((prompt + completion) / uptime if uptime > 0 else 0, 1),
            "ttft_p50": round(self._calculate_percentile(ttft_values, 50), 3),
            "ttft_p95": round(self._calculate_percentile(ttft_values, 95), 3),
            "ttft_p99": round(self._calculate_percentile(ttft_values, 99), 3),
            "prefill_p50": round(self._calculate_percentile(prefill_values, 50), 3),
            "prefill_p95": round(self._calculate_percentile(prefill_values, 95), 3),
            "generation_p50": round(self._calculate_percentile(generation_values, 50), 3),
            "generation_p95": round(self._calculate_percentile(generation_values, 95), 3),
            "avg_prefill_duration": round(prefill_dur / requests if requests > 0 else 0, 3),
            "avg_generation_duration": round(gen_dur / requests if requests > 0 else 0, 3),
        }

    def get_snapshot(self, model_id: str = "", scope: str = "session") -> dict:
        """Get current metrics snapshot. Thread-safe.

        Args:
            model_id: If provided and tracked, return per-model metrics.
                      Otherwise return global aggregate.
            scope: "session" for current session, "alltime" for persisted totals.
        """
        with self._lock:
            now = time.time()
            uptime = now - self._start_time

            if scope == "alltime":
                if model_id:
                    m = self._alltime_per_model.get(model_id)
                    if m:
                        return self._build_snapshot(
                            m["prompt_tokens"],
                            m["completion_tokens"],
                            m["cached_tokens"],
                            m["requests"],
                            m["prefill_duration"],
                            m["generation_duration"],
                            uptime,
                        )
                    return self._build_snapshot(0, 0, 0, 0, 0.0, 0.0, uptime)
                return self._build_snapshot(
                    self._alltime_prompt_tokens,
                    self._alltime_completion_tokens,
                    self._alltime_cached_tokens,
                    self._alltime_requests,
                    self._alltime_prefill_duration,
                    self._alltime_generation_duration,
                    uptime,
                )

            # scope == "session" (default)
            if model_id:
                m = self._per_model.get(model_id)
                if m:
                    return self._build_snapshot(
                        m["prompt_tokens"],
                        m["completion_tokens"],
                        m["cached_tokens"],
                        m["requests"],
                        m["prefill_duration"],
                        m["generation_duration"],
                        uptime,
                    )
                return self._build_snapshot(0, 0, 0, 0, 0.0, 0.0, uptime)

            return self._build_snapshot(
                self.total_prompt_tokens,
                self.total_completion_tokens,
                self.total_cached_tokens,
                self.total_requests,
                self.total_prefill_duration,
                self.total_generation_duration,
                uptime,
            )

    def get_request_counts(self) -> dict:
        """Get request count statistics. Thread-safe."""
        with self._lock:
            return {
                "total": self.total_requests,
                "per_model": {
                    model_id: data["requests"]
                    for model_id, data in self._per_model.items()
                },
                "current_rate": len(self._request_timestamps) / _RATE_WINDOW
                if self._request_timestamps
                else 0,
            }

    def get_latency_stats(self) -> dict:
        """Get latency statistics. Thread-safe."""
        with self._lock:
            ttft_values = [v for t, v in self._latencies if t == "ttft"]
            return {
                "ttft": {
                    "count": len(ttft_values),
                    "avg": sum(ttft_values) / len(ttft_values) if ttft_values else 0,
                    "min": min(ttft_values) if ttft_values else 0,
                    "max": max(ttft_values) if ttft_values else 0,
                    "p50": self._calculate_percentile(ttft_values, 50),
                    "p95": self._calculate_percentile(ttft_values, 95),
                    "p99": self._calculate_percentile(ttft_values, 99),
                },
                "prefill": {
                    "count": len(self._prefill_latencies),
                    "avg": sum(self._prefill_latencies) / len(self._prefill_latencies)
                    if self._prefill_latencies
                    else 0,
                },
                "generation": {
                    "count": len(self._generation_latencies),
                    "avg": sum(self._generation_latencies)
                    / len(self._generation_latencies)
                    if self._generation_latencies
                    else 0,
                },
            }

    def clear_metrics(self) -> None:
        """Clear session metrics. Thread-safe."""
        with self._lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_cached_tokens = 0
            self.total_requests = 0
            self.total_prefill_duration = 0.0
            self.total_generation_duration = 0.0
            self._per_model.clear()
            self._request_timestamps.clear()
            self._latencies.clear()
            self._prefill_latencies.clear()
            self._generation_latencies.clear()

    def clear_alltime_metrics(self) -> None:
        """Clear all-time metrics and delete the persisted file. Thread-safe."""
        with self._lock:
            self._alltime_prompt_tokens = 0
            self._alltime_completion_tokens = 0
            self._alltime_cached_tokens = 0
            self._alltime_requests = 0
            self._alltime_prefill_duration = 0.0
            self._alltime_generation_duration = 0.0
            self._alltime_per_model.clear()
        if self._stats_path and self._stats_path.exists():
            try:
                self._stats_path.unlink()
            except OSError as e:
                logger.warning(
                    "Failed to delete stats file %s: %s", self._stats_path, e
                )

    def get_comprehensive_stats(self) -> dict:
        """Get comprehensive stats for API endpoint. Thread-safe."""
        with self._lock:
            now = time.time()
            uptime = now - self._start_time
            snapshot = self._build_snapshot(
                self.total_prompt_tokens,
                self.total_completion_tokens,
                self.total_cached_tokens,
                self.total_requests,
                self.total_prefill_duration,
                self.total_generation_duration,
                uptime,
            )

            # Add per-model stats
            model_stats = {}
            for model_id, data in self._per_model.items():
                dflash_acceptance = (
                    data["dflash_accepted_from_draft"]
                    / data["dflash_generated_tokens"]
                    if data.get("dflash_generated_tokens", 0) > 0
                    else 0.0
                )
                model_stats[model_id] = {
                    "requests": data["requests"],
                    "prompt_tokens": data["prompt_tokens"],
                    "completion_tokens": data["completion_tokens"],
                    "cached_tokens": data["cached_tokens"],
                    "ttft_avg": (
                        sum(data["ttft_samples"]) / len(data["ttft_samples"])
                        if data["ttft_samples"]
                        else 0
                    ),
                    "ttft_p95": (
                        self._calculate_percentile(data["ttft_samples"], 95)
                        if data["ttft_samples"]
                        else 0
                    ),
                    "dflash_acceptance_ratio": round(dflash_acceptance, 4),
                    "dflash_cycles": data.get("dflash_cycles", 0),
                }

            # Calculate batching speedup
            # Speedup = (sequential time) / (actual time)
            # sequential time = prefill_time + gen_time (if no batching)
            # With batching, multiple requests share prefill
            sequential_prefill_estimate = self.total_prompt_tokens / max(
                self.total_requests, 1
            ) * self.total_requests  # Assume all prompts need full prefill
            actual_prefill = self.total_prefill_duration
            batching_speedup = (
                sequential_prefill_estimate / actual_prefill
                if actual_prefill > 0 and self.total_requests > 1
                else 1.0
            )

            return {
                **snapshot,
                "request_counts": {
                    "total": self.total_requests,
                    "per_model": {m: d["requests"] for m, d in self._per_model.items()},
                },
                "latency_stats": self.get_latency_stats(),
                "batching_speedup": round(batching_speedup, 2),
                "per_model_stats": model_stats,
            }


# Global singleton
_server_metrics = None


def get_server_metrics() -> ServerMetrics:
    """Get the global ServerMetrics singleton."""
    global _server_metrics
    if _server_metrics is None:
        _server_metrics = ServerMetrics()
    return _server_metrics


def reset_server_metrics(stats_path=None) -> None:
    """Reset metrics (called on server start).

    If a previous instance exists and has a stats_path, save before resetting.
    """
    global _server_metrics
    if _server_metrics is not None:
        _server_metrics.save_alltime()
    _server_metrics = ServerMetrics(stats_path=stats_path)
