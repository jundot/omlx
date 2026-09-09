# SPDX-License-Identifier: Apache-2.0
"""Tests for SLO definitions and the SLO tracker."""

from __future__ import annotations

import time

import pytest

from omlx.slo_definitions import DEFAULT_SLOS, SLO, SLOMetricKind, SLOStatus, slo_by_name
from omlx.cluster.slo_tracker import SLOTracker, SLOEvaluation, _overall_status

# ---------------------------------------------------------------------------
# SLO definitions
# ---------------------------------------------------------------------------


class TestSLODefinitions:
    def test_slo_definitions_exist(self) -> None:
        assert isinstance(DEFAULT_SLOS, list)
        assert len(DEFAULT_SLOS) > 0

    def test_default_slos_have_required_fields(self) -> None:
        for slo in DEFAULT_SLOS:
            assert isinstance(slo, SLO)
            assert slo.name
            assert isinstance(slo.metric_kind, SLOMetricKind)
            assert slo.target > 0
            assert slo.window_seconds > 0
            assert slo.metric_key

    def test_slo_by_name_returns_slo(self) -> None:
        first = DEFAULT_SLOS[0]
        result = slo_by_name(first.name)
        assert result is first

    def test_slo_by_name_returns_none_for_unknown(self) -> None:
        assert slo_by_name("nonexistent_slo_xyz") is None

    def test_slo_is_frozen(self) -> None:
        slo = DEFAULT_SLOS[0]
        with pytest.raises(AttributeError):
            slo.name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SLO tracker — evaluation
# ---------------------------------------------------------------------------


class TestSLOTrackerEvaluation:
    def _make_tracker_with_samples(
        self, metric_key: str, values: list[float]
    ) -> SLOTracker:
        tracker = SLOTracker()
        for v in values:
            tracker.record(metric_key, v)
        return tracker

    def test_slo_evaluation_healthy(self) -> None:
        """Compliance above target yields HEALTHY status."""
        # cache_hit_rate SLO target is 70.0
        tracker = SLOTracker()
        for v in [72.0, 75.0, 71.0, 73.0, 74.0]:
            tracker.record("cache_hit_rate", v)

        ev = tracker.evaluate(DEFAULT_SLOS[-1])  # cache_hit_rate SLO
        assert ev.status == SLOStatus.HEALTHY
        assert ev.compliance_pct == 100.0
        assert ev.sample_count == 5

    def test_slo_evaluation_breached(self) -> None:
        """Compliance below target yields BREACHED status."""
        tracker = SLOTracker()
        for v in [40.0, 45.0, 50.0, 38.0, 42.0]:
            tracker.record("cache_hit_rate", v)

        ev = tracker.evaluate(DEFAULT_SLOS[-1])  # cache_hit_rate SLO
        assert ev.status == SLOStatus.BREACHED
        assert ev.compliance_pct == 0.0

    def test_slo_evaluation_degraded(self) -> None:
        """Compliance between degraded_target and target yields DEGRADED."""
        tracker = SLOTracker()
        # All above degraded_target (55.0) but below target (70.0)
        for v in [62.0, 63.0, 64.0, 61.0, 65.0]:
            tracker.record("cache_hit_rate", v)

        ev = tracker.evaluate(DEFAULT_SLOS[-1])  # cache_hit_rate: target=70, degraded=55
        assert ev.status == SLOStatus.DEGRADED
        # Compliance measures against the *target* (70), so 0% here
        assert ev.compliance_pct == 0.0
        assert ev.current_value is not None
        assert ev.current_value > 55.0  # above degraded threshold

    def test_slo_evaluation_empty_window(self) -> None:
        """Empty window returns HEALTHY with 100% compliance."""
        tracker = SLOTracker()
        ev = tracker.evaluate(DEFAULT_SLOS[-1])
        assert ev.status == SLOStatus.HEALTHY
        assert ev.compliance_pct == 100.0
        assert ev.sample_count == 0

    def test_status_returns_all_slos(self) -> None:
        tracker = SLOTracker()
        evaluations = tracker.status()
        assert len(evaluations) == len(DEFAULT_SLOS)
        assert all(isinstance(ev, SLOEvaluation) for ev in evaluations)


# ---------------------------------------------------------------------------
# Burn rate
# ---------------------------------------------------------------------------


class TestBurnRate:
    def test_burn_rate_calculation(self) -> None:
        """Burn rate is computed correctly for a fully compliant window."""
        tracker = SLOTracker()
        for v in [75.0, 80.0, 72.0, 78.0, 74.0]:
            tracker.record("cache_hit_rate", v)

        ev = tracker.evaluate(DEFAULT_SLOS[-1])
        assert ev.burn_rate == 0.0

    def test_burn_rate_exceeds_one_on_breach(self) -> None:
        """Burn rate exceeds 1.0 when error budget is being consumed faster."""
        tracker = SLOTracker()
        # All below 70% target → 100% error rate, budget = 30%, burn ≈ 3.33
        for v in [30.0, 35.0, 25.0, 32.0, 28.0]:
            tracker.record("cache_hit_rate", v)

        ev = tracker.evaluate(DEFAULT_SLOS[-1])
        assert ev.burn_rate > 1.0

    def test_burn_rate_time_to_breach(self) -> None:
        """time_to_breach_seconds is set when burn rate > 1."""
        tracker = SLOTracker()
        for v in [30.0, 35.0, 25.0]:
            tracker.record("cache_hit_rate", v)

        ev = tracker.evaluate(DEFAULT_SLOS[-1])
        assert ev.burn_rate > 1.0
        assert ev.time_to_breach_seconds is not None
        assert ev.time_to_breach_seconds > 0

    def test_burn_rate_no_breach_when_healthy(self) -> None:
        """No breach time estimated when compliant."""
        tracker = SLOTracker()
        for v in [80.0, 85.0, 90.0]:
            tracker.record("cache_hit_rate", v)

        ev = tracker.evaluate(DEFAULT_SLOS[-1])
        assert ev.time_to_breach_seconds is None


# ---------------------------------------------------------------------------
# Tracker integration
# ---------------------------------------------------------------------------


class TestSLOTrackerIntegration:
    def test_slo_tracker_uses_sample_store(self) -> None:
        """Tracker can accept samples via record() and evaluate them."""
        tracker = SLOTracker()
        tracker.record("tokens_per_second", 45.0)
        tracker.record("tokens_per_second", 42.0)
        tracker.record("tokens_per_second", 48.0)

        throughput_slo = slo_by_name("throughput_min_small")
        assert throughput_slo is not None
        ev = tracker.evaluate(throughput_slo)
        assert ev.status == SLOStatus.HEALTHY
        assert ev.sample_count == 3

    def test_record_many_bulk_ingest(self) -> None:
        tracker = SLOTracker()
        tracker.record_many("cache_hit_rate", [60.0, 65.0, 72.0, 78.0])

        ev = tracker.evaluate(DEFAULT_SLOS[-1])
        assert ev.sample_count == 4

    def test_clear_resets_window(self) -> None:
        tracker = SLOTracker()
        tracker.record("cache_hit_rate", 80.0)
        assert tracker.evaluate(DEFAULT_SLOS[-1]).sample_count == 1

        tracker.clear("cache_hit_rate")
        assert tracker.evaluate(DEFAULT_SLOS[-1]).sample_count == 0

    def test_clear_all_windows(self) -> None:
        tracker = SLOTracker()
        tracker.record("cache_hit_rate", 80.0)
        tracker.record("tokens_per_second", 40.0)
        tracker.clear()
        assert tracker.evaluate(DEFAULT_SLOS[-1]).sample_count == 0

    def test_status_dict_structure(self) -> None:
        tracker = SLOTracker()
        result = tracker.status_dict()
        assert "slos" in result
        assert "overall_status" in result
        assert isinstance(result["slos"], list)
        assert result["overall_status"] in ("healthy", "degraded", "breached")

        for slo_entry in result["slos"]:
            for key in ("name", "metric_kind", "status", "compliance_pct", "burn_rate", "target", "window_seconds"):
                assert key in slo_entry, f"missing key '{key}' in SLO entry"


# ---------------------------------------------------------------------------
# Overall status
# ---------------------------------------------------------------------------


class TestOverallStatus:
    def test_overall_healthy_when_all_healthy(self) -> None:
        evals = [
            SLOEvaluation(
                slo=DEFAULT_SLOS[0],
                status=SLOStatus.HEALTHY,
                compliance_pct=100.0,
                burn_rate=0.0,
                window_seconds=3600,
                sample_count=10,
            ),
        ]
        assert _overall_status(evals) == "healthy"

    def test_overall_breached_when_any_breached(self) -> None:
        evals = [
            SLOEvaluation(
                slo=DEFAULT_SLOS[0],
                status=SLOStatus.HEALTHY,
                compliance_pct=100.0,
                burn_rate=0.0,
                window_seconds=3600,
                sample_count=10,
            ),
            SLOEvaluation(
                slo=DEFAULT_SLOS[1],
                status=SLOStatus.BREACHED,
                compliance_pct=50.0,
                burn_rate=2.0,
                window_seconds=3600,
                sample_count=10,
            ),
        ]
        assert _overall_status(evals) == "breached"

    def test_overall_degraded_when_no_breach_but_degraded(self) -> None:
        evals = [
            SLOEvaluation(
                slo=DEFAULT_SLOS[0],
                status=SLOStatus.HEALTHY,
                compliance_pct=100.0,
                burn_rate=0.0,
                window_seconds=3600,
                sample_count=10,
            ),
            SLOEvaluation(
                slo=DEFAULT_SLOS[1],
                status=SLOStatus.DEGRADED,
                compliance_pct=80.0,
                burn_rate=0.5,
                window_seconds=3600,
                sample_count=10,
            ),
        ]
        assert _overall_status(evals) == "degraded"


# ---------------------------------------------------------------------------
# API response schema
# ---------------------------------------------------------------------------


class TestSLOAPISchema:
    def test_slo_api_response_schema(self) -> None:
        """GET /cluster/slos returns the expected structure."""
        tracker = SLOTracker()
        result = tracker.status_dict()

        assert isinstance(result, dict)
        assert "slos" in result
        assert "overall_status" in result
        assert result["overall_status"] in ("healthy", "degraded", "breached")

        for slo_entry in result["slos"]:
            assert isinstance(slo_entry["name"], str)
            assert isinstance(slo_entry["metric_kind"], str)
            assert isinstance(slo_entry["status"], str)
            assert slo_entry["status"] in ("healthy", "degraded", "breached")
            assert isinstance(slo_entry["compliance_pct"], (int, float))
            assert isinstance(slo_entry["burn_rate"], (int, float))
            assert isinstance(slo_entry["target"], (int, float))
            assert isinstance(slo_entry["window_seconds"], int)
            assert isinstance(slo_entry["sample_count"], int)

    def test_latency_percentile_evaluation(self) -> None:
        """Latency SLO evaluates using the correct percentile."""
        tracker = SLOTracker()
        # Feed latency samples; p95 target is 2.0s for small models
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                   1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.5]:
            tracker.record("latency_p95", v)

        slo = slo_by_name("latency_p95_small")
        assert slo is not None
        ev = tracker.evaluate(slo)
        # Sorted: [0.1 .. 1.9, 1.5] → p95 index at 95% of 20 = 19 → value 1.9
        assert ev.current_value is not None
        assert ev.current_value <= slo.target  # 1.9 <= 2.0 → healthy
        assert ev.status == SLOStatus.HEALTHY

    def test_throughput_evaluation(self) -> None:
        """Throughput SLO evaluates higher-is-better correctly."""
        tracker = SLOTracker()
        for v in [35.0, 38.0, 40.0, 42.0]:
            tracker.record("tokens_per_second", v)

        slo = slo_by_name("throughput_min_small")
        assert slo is not None
        ev = tracker.evaluate(slo)
        assert ev.status == SLOStatus.HEALTHY
        assert ev.current_value is not None
        assert ev.current_value >= slo.target
