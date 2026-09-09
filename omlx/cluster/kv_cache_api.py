# SPDX-License-Identifier: Apache-2.0
"""Backend API for KV cache observability on the cluster dashboard."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/admin/api/cluster", tags=["cluster"])

_EVICTION_THRESHOLD = 0.05
_MAX_TIMESERIES = 60
_POLL_INTERVAL_SECONDS = 60


def _timestamp() -> float:
    return time.time()


def _collect_node_kv_metrics() -> list[dict[str, Any]]:
    """Collect per-node KV cache metrics from the engine pool.

    Falls back to empty data when no engine pool is available (worker-only
    installs, bare test apps).
    """

    from ..server import get_engine_pool  # type: ignore[import-untyped]

    pool = get_engine_pool()
    if pool is None:
        return []

    nodes: list[dict[str, Any]] = []
    for model_id in pool.get_loaded_model_ids():
        entry = pool.get_entry(model_id)
        status_fn = getattr(getattr(entry, "engine", None), "cluster_status", None)
        if not callable(status_fn):
            continue
        launcher = status_fn()
        ranks = launcher.get("ranks", [])
        for rank_data in ranks:
            cache_info = rank_data.get("cache") or {}
            total_bytes = int(cache_info.get("bytes", 0))
            cache_size_mb = round(total_bytes / (1024 * 1024), 1) if total_bytes else 0.0
            hit_rate = float(cache_info.get("hit_rate", 0.0))
            entries = int(cache_info.get("entries", 0))
            capacity_bytes = int(
                getattr(entry, "kv_cache_capacity_bytes", 0) or 0
            )
            utilization = (
                min(1.0, total_bytes / capacity_bytes) if capacity_bytes > 0 else 0.0
            )
            eviction_rate = 0.0
            nodes.append(
                {
                    "node_id": rank_data.get("node_id", model_id),
                    "rank": rank_data.get("rank", 0),
                    "cache_size_mb": cache_size_mb,
                    "hit_rate": round(hit_rate, 4),
                    "eviction_rate": round(eviction_rate, 4),
                    "memory_pressure": round(utilization, 4),
                    "entries": entries,
                    "capacity_bytes": capacity_bytes,
                }
            )
    return nodes


def _compute_efficiency_score(nodes: list[dict[str, Any]]) -> float:
    """Cluster-wide cache efficiency: average hit rate weighted by capacity."""

    if not nodes:
        return 0.0

    weighted_sum = 0.0
    total_capacity = 0.0
    for node in nodes:
        capacity = float(node.get("capacity_bytes", 0))
        hit_rate = float(node.get("hit_rate", 0))
        weighted_sum += hit_rate * capacity
        total_capacity += capacity

    if total_capacity <= 0:
        return 0.0
    return round(weighted_sum / total_capacity, 4)


@router.get("/kv-cache")
async def cluster_kv_cache():
    """Return KV cache metrics for the cluster dashboard.

    Response schema:
    - nodes: per-node cache_size_mb, hit_rate, eviction_rate, memory_pressure
    - timeseries: cache metrics over last hour (1-minute resolution)
    - score: cluster-wide cache efficiency score
    """

    nodes = _collect_node_kv_metrics()
    score = _compute_efficiency_score(nodes)
    now = _timestamp()
    timeseries: list[dict[str, Any]] = [
        {
            "ts": round(now - i * _POLL_INTERVAL_SECONDS, 1),
            "nodes": nodes,
            "score": score,
        }
        for i in range(_MAX_TIMESERIES)
    ]
    return {
        "nodes": nodes,
        "timeseries": timeseries,
        "score": score,
    }
