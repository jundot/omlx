# SPDX-License-Identifier: Apache-2.0
"""Capacity planning API — GET /admin/api/cluster/capacity."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from .capacity_planner import CapacityPlanner

router = APIRouter(prefix="/admin/api/cluster", tags=["cluster"])

_planner: CapacityPlanner | None = None


def get_capacity_planner() -> CapacityPlanner:
    """Return the module-level planner, creating one on first call."""
    global _planner  # noqa: PLW0603
    if _planner is None:
        _planner = CapacityPlanner()
    return _planner


def set_capacity_planner(planner: CapacityPlanner) -> None:
    """Inject a planner instance (for tests or server wiring)."""
    global _planner  # noqa: PLW0603
    _planner = planner


@router.get("/capacity")
async def cluster_capacity() -> dict[str, Any]:
    """Cluster capacity overview with per-node utilization and scaling advice.

    Returns:
        - nodes: per-node CPU, memory, GPU, connections
        - aggregate: total nodes, avg utilization, headroom, saturation
        - trend: utilization at 15-min resolution over last 24h
        - recommendation: scale_up / scale_down / stable with reasoning
    """
    planner = get_capacity_planner()
    report = planner.collect_metrics()

    return {
        "nodes": [
            {
                "node_id": n.node_id,
                "hostname": n.hostname,
                "cpu_usage_pct": n.cpu_usage_pct,
                "memory_usage_pct": n.memory_usage_pct,
                "memory_used_bytes": n.memory_used_bytes,
                "memory_total_bytes": n.memory_total_bytes,
                "gpu_utilization_pct": n.gpu_utilization_pct,
                "active_connections": n.active_connections,
                "accelerator": n.accelerator,
            }
            for n in report.nodes
        ],
        "aggregate": {
            "total_nodes": report.total_nodes,
            "avg_cpu_pct": report.avg_cpu_pct,
            "avg_memory_pct": report.avg_memory_pct,
            "total_headroom_pct": report.total_headroom_pct,
            "saturation_pct": report.saturation_pct,
        },
        "trend": [
            {
                "timestamp": p.timestamp,
                "avg_cpu_pct": p.avg_cpu_pct,
                "avg_memory_pct": p.avg_memory_pct,
            }
            for p in report.trend
        ],
        "recommendation": {
            "action": report.recommendation.action,
            "reasons": report.recommendation.reasons,
            "confidence": report.recommendation.confidence,
        },
        "collected_at": report.collected_at,
    }
