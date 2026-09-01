# SPDX-License-Identifier: Apache-2.0
"""Tests for the capacity planner and its API endpoint."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from omlx.cluster.capacity_planner import (
    CapacityPlanner,
    CapacityReport,
    NodeCapacity,
    ScalingRecommendation,
    TrendPoint,
    _read_proc_cpu,
    _read_proc_mem,
)
from omlx.cluster.capacity_api import get_capacity_planner, set_capacity_planner
from omlx.cluster.error_budget import ErrorBudgetTracker
from omlx.cluster.slo_tracker import SLOTracker
from omlx.slo_definitions import DEFAULT_SLOS, SLO, SLOMetricKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    node_id: str = "node-0",
    cpu: float = 50.0,
    mem_pct: float = 50.0,
    mem_used: int = 8 * 1024**3,
    mem_total: int = 16 * 1024**3,
    gpu: float | None = None,
    connections: int = 10,
    accelerator: str | None = None,
) -> NodeCapacity:
    return NodeCapacity(
        node_id=node_id,
        hostname=f"{node_id}.local",
        cpu_usage_pct=cpu,
        memory_usage_pct=mem_pct,
        memory_used_bytes=mem_used,
        memory_total_bytes=mem_total,
        gpu_utilization_pct=gpu,
        active_connections=connections,
        accelerator=accelerator,
    )


class _StubPlanner(CapacityPlanner):
    """Planner that returns synthetic node data instead of reading the system."""

    def __init__(self, nodes: list[NodeCapacity], **kwargs):
        super().__init__(**kwargs)
        self._stub_nodes = nodes

    def _snapshot_nodes(self) -> list[NodeCapacity]:
        return list(self._stub_nodes)


# ---------------------------------------------------------------------------
# CapacityPlanner.collect_metrics
# ---------------------------------------------------------------------------


class TestCapacityPlannerCollectsMetrics:
    def test_collects_metrics_returns_report(self) -> None:
        nodes = [_make_node(cpu=60.0, mem_pct=70.0)]
        planner = _StubPlanner(nodes)
        report = planner.collect_metrics(now=1_000_000.0)

        assert isinstance(report, CapacityReport)
        assert report.total_nodes == 1
        assert report.avg_cpu_pct == 60.0
        assert report.avg_memory_pct == 70.0
        assert len(report.nodes) == 1
        assert report.nodes[0].node_id == "node-0"

    def test_collects_metrics_populates_trend(self) -> None:
        nodes = [_make_node(cpu=40.0, mem_pct=55.0)]
        planner = _StubPlanner(nodes)
        planner.collect_metrics(now=100.0)
        planner.collect_metrics(now=1900.0)

        trend = planner.get_trend()
        assert len(trend) == 2
        assert trend[0].timestamp == 100.0
        assert trend[1].avg_cpu_pct == 40.0

    def test_headroom_and_saturation(self) -> None:
        nodes = [
            _make_node(mem_used=8 * 1024**3, mem_total=16 * 1024**3),
        ]
        planner = _StubPlanner(nodes)
        report = planner.collect_metrics()

        assert report.total_headroom_pct == pytest.approx(50.0, abs=0.1)
        assert report.saturation_pct == pytest.approx(50.0, abs=0.1)

    def test_multiple_nodes_averaged(self) -> None:
        nodes = [
            _make_node(node_id="a", cpu=40.0, mem_pct=60.0),
            _make_node(node_id="b", cpu=80.0, mem_pct=40.0),
        ]
        planner = _StubPlanner(nodes)
        report = planner.collect_metrics()

        assert report.avg_cpu_pct == 60.0
        assert report.avg_memory_pct == 50.0


# ---------------------------------------------------------------------------
# Scaling recommendations
# ---------------------------------------------------------------------------


class TestScalingRecommendationScaleUp:
    def test_high_cpu_triggers_scale_up(self) -> None:
        nodes = [_make_node(cpu=90.0, mem_pct=50.0)]
        planner = _StubPlanner(nodes)
        report = planner.collect_metrics()

        assert report.recommendation.action == "scale_up"
        assert any("High utilization" in r for r in report.recommendation.reasons)

    def test_high_memory_triggers_scale_up(self) -> None:
        nodes = [_make_node(cpu=50.0, mem_pct=95.0)]
        planner = _StubPlanner(nodes)
        report = planner.collect_metrics()

        assert report.recommendation.action == "scale_up"

    def test_node_hotspot_triggers_scale_up(self) -> None:
        nodes = [_make_node(node_id="hot", cpu=95.0, mem_pct=98.0)]
        planner = _StubPlanner(nodes)
        report = planner.collect_metrics()

        assert report.recommendation.action == "scale_up"
        assert any("hotspot" in r for r in report.recommendation.reasons)


class TestScalingRecommendationStable:
    def test_moderate_utilization_returns_stable(self) -> None:
        nodes = [_make_node(cpu=50.0, mem_pct=60.0)]
        planner = _StubPlanner(nodes)
        report = planner.collect_metrics()

        assert report.recommendation.action == "stable"
        assert any("Moderate" in r for r in report.recommendation.reasons)


class TestScalingRecommendationScaleDown:
    def test_low_utilization_returns_scale_down(self) -> None:
        nodes = [_make_node(cpu=10.0, mem_pct=20.0)]
        planner = _StubPlanner(nodes)
        report = planner.collect_metrics()

        assert report.recommendation.action == "scale_down"
        assert any("Low utilization" in r for r in report.recommendation.reasons)


# ---------------------------------------------------------------------------
# Error budget integration
# ---------------------------------------------------------------------------


class TestScalingRecommendationConsidersErrorBudget:
    def test_depleted_budget_forces_scale_up(self) -> None:
        nodes = [_make_node(cpu=50.0, mem_pct=60.0)]
        now = time.time()
        slo_tracker = SLOTracker()
        # Record failures within the rolling window so they are not evicted.
        for i in range(100):
            slo_tracker.record("uptime_percentage", 0.0, now=now - (100 - i) * 10)
        error_budget = ErrorBudgetTracker(slo_tracker=slo_tracker)
        planner = _StubPlanner(nodes, error_budget=error_budget, slo_tracker=slo_tracker)
        report = planner.collect_metrics(now=now)

        assert report.recommendation.action == "scale_up"
        assert any(
            "Error budget" in r for r in report.recommendation.reasons
        )

    def test_low_budget_adds_reason(self) -> None:
        now = time.time()
        slo_tracker = SLOTracker()
        # Burn ~50% of the availability budget (target 99.9, budget = 0.1%).
        for i in range(50):
            slo_tracker.record("uptime_percentage", 0.0, now=now - (50 - i) * 10)
        error_budget = ErrorBudgetTracker(slo_tracker=slo_tracker)

        nodes = [_make_node(cpu=50.0, mem_pct=60.0)]
        planner = _StubPlanner(
            nodes, error_budget=error_budget, slo_tracker=slo_tracker
        )
        report = planner.collect_metrics(now=now)

        assert any(
            "Low error budget" in r for r in report.recommendation.reasons
        )


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------


class TestCapacityTrendData:
    def test_trend_structure(self) -> None:
        nodes = [_make_node(cpu=50.0, mem_pct=55.0)]
        planner = _StubPlanner(nodes)
        planner.collect_metrics(now=100.0)
        planner.collect_metrics(now=1000.0)
        planner.collect_metrics(now=2000.0)

        trend = planner.get_trend()
        assert len(trend) == 3
        for point in trend:
            assert isinstance(point, TrendPoint)
            assert isinstance(point.timestamp, float)
            assert isinstance(point.avg_cpu_pct, float)
            assert isinstance(point.avg_memory_pct, float)

    def test_sustained_memory_growth_triggers_recommendation(self) -> None:
        """Four consecutive rising memory readings above 70% -> upward trend."""
        # The stub node reports 82% memory, which continues the injected upward trend.
        nodes = [_make_node(cpu=50.0, mem_pct=82.0)]
        planner = _StubPlanner(nodes)

        # Manually inject four trend points with rising memory.
        for i, ts in enumerate([100.0, 1000.0, 2000.0, 3000.0]):
            with planner._lock:
                planner._trend.append(
                    TrendPoint(
                        timestamp=ts,
                        avg_cpu_pct=50.0,
                        avg_memory_pct=70.0 + i * 3.0,
                    )
                )

        # collect_metrics appends its own point (mem=82%), then reads the last
        # four: [73, 76, 79, 82] — all increasing, last > 70%.
        report = planner.collect_metrics(now=4000.0)
        assert any(
            "trending upward" in r for r in report.recommendation.reasons
        )

    def test_clear_trend(self) -> None:
        nodes = [_make_node()]
        planner = _StubPlanner(nodes)
        planner.collect_metrics(now=100.0)
        assert len(planner.get_trend()) == 1
        planner.clear_trend()
        assert len(planner.get_trend()) == 0


# ---------------------------------------------------------------------------
# /proc fallback helpers
# ---------------------------------------------------------------------------


class TestProcFallbacks:
    def test_read_proc_cpu_returns_float(self) -> None:
        result = _read_proc_cpu()
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0

    @pytest.mark.skipif(
        not Path("/proc/meminfo").exists(),
        reason="/proc/meminfo is Linux-only; macOS reports 0 from this fallback",
    )
    def test_read_proc_mem_returns_memory(self) -> None:
        mem = _read_proc_mem()
        assert mem.total > 0


# ---------------------------------------------------------------------------
# API schema validation
# ---------------------------------------------------------------------------


class TestCapacityAPISchema:
    def test_capacity_api_returns_required_fields(self) -> None:
        """The endpoint must return every field the frontend consumes."""
        nodes = [
            _make_node(
                cpu=65.0, mem_pct=72.0, gpu=88.0, connections=42, accelerator="cuda"
            )
        ]
        planner = _StubPlanner(nodes)
        set_capacity_planner(planner)
        try:
            import asyncio

            # get_event_loop() raises "no current event loop" once nothing has
            # installed one for this thread, which is every Python from 3.12.
            response = asyncio.run(_call_capacity_endpoint())
        finally:
            set_capacity_planner(None)

        # Top-level keys.
        assert "nodes" in response
        assert "aggregate" in response
        assert "trend" in response
        assert "recommendation" in response
        assert "collected_at" in response

        # Node fields.
        node = response["nodes"][0]
        for key in (
            "node_id",
            "hostname",
            "cpu_usage_pct",
            "memory_usage_pct",
            "memory_used_bytes",
            "memory_total_bytes",
            "gpu_utilization_pct",
            "active_connections",
            "accelerator",
        ):
            assert key in node, f"missing node field: {key}"
        assert node["gpu_utilization_pct"] == 88.0
        assert node["accelerator"] == "cuda"

        # Aggregate fields.
        agg = response["aggregate"]
        for key in (
            "total_nodes",
            "avg_cpu_pct",
            "avg_memory_pct",
            "total_headroom_pct",
            "saturation_pct",
        ):
            assert key in agg, f"missing aggregate field: {key}"

        # Trend point fields.
        assert len(response["trend"]) >= 1
        tp = response["trend"][0]
        for key in ("timestamp", "avg_cpu_pct", "avg_memory_pct"):
            assert key in tp, f"missing trend field: {key}"

        # Recommendation fields.
        rec = response["recommendation"]
        for key in ("action", "reasons", "confidence"):
            assert key in rec, f"missing recommendation field: {key}"
        assert rec["action"] in ("scale_up", "scale_down", "stable")
        assert isinstance(rec["reasons"], list)
        assert isinstance(rec["confidence"], float)


async def _call_capacity_endpoint() -> dict:
    """Import and call the FastAPI handler directly."""
    from omlx.cluster.capacity_api import cluster_capacity  # noqa: PLC0415

    return await cluster_capacity()
