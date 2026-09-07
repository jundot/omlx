# SPDX-License-Identifier: Apache-2.0
"""Backend API for prefix-cache topology visualization.

Exposes ``GET /admin/api/cluster/prefix-cache`` so the dashboard can render
per-node cache state, cross-node prefix sharing, and recent cache events.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/admin/api/cluster", tags=["cluster"])

_get_engine_pool: Any | None = None


def set_engine_pool_getter(getter: Any) -> None:
    global _get_engine_pool
    _get_engine_pool = getter


def _engine_pool() -> Any:
    if _get_engine_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Cluster prefix-cache API is unavailable until the server is initialized",
        )
    return _get_engine_pool()


class CacheEvent(BaseModel):
    kind: str = Field(description="hit, miss, or eviction")
    ts: float = Field(description="Unix timestamp")
    prefix_hash: str = Field(default="", description="Prefix hash if available")
    tokens_saved: int = Field(default=0, description="Tokens saved on hit")


class NodeCacheState(BaseModel):
    node_id: str
    hostname: str
    total_blocks: int = 0
    used_blocks: int = 0
    hit_rate: float = 0.0
    prefix_hashes: list[str] = Field(default_factory=list)
    tokens_saved: int = 0
    total_queries: int = 0
    evictions: int = 0


class PrefixCacheTopology(BaseModel):
    nodes: list[NodeCacheState] = Field(default_factory=list)
    events: list[CacheEvent] = Field(default_factory=list)
    totals: dict[str, Any] = Field(default_factory=dict)


def _collect_local_cache_state() -> NodeCacheState:
    """Gather prefix-cache state from the locally loaded engine."""
    pool = _engine_pool()
    node_id = "local"
    hostname = "127.0.0.1"

    try:
        import socket
        hostname = socket.gethostname()
    except Exception:
        pass

    total_blocks = 0
    used_blocks = 0
    hit_rate = 0.0
    tokens_saved = 0
    total_queries = 0
    evictions = 0
    prefix_hashes: list[str] = []

    for _model_id, entry in getattr(pool, "_entries", {}).items():
        engine = getattr(entry, "engine", None)
        if engine is None:
            continue

        async_core = getattr(engine, "_engine", None)
        core = getattr(async_core, "engine", None) if async_core is not None else None
        scheduler = getattr(core, "scheduler", None) if core is not None else None
        if scheduler is None:
            continue

        paged_cache = getattr(scheduler, "paged_cache_manager", None)
        if paged_cache is not None:
            stats = getattr(paged_cache, "stats", None)
            if stats is not None:
                total_blocks += getattr(stats, "total_blocks", 0)
                used_blocks += getattr(stats, "allocated_blocks", 0)
                evictions += getattr(stats, "evictions", 0)

        prefix_cache = getattr(scheduler, "block_aware_cache", None)
        if prefix_cache is not None:
            pc_stats = getattr(prefix_cache, "stats", None)
            if pc_stats is not None:
                tokens_saved += getattr(pc_stats, "tokens_saved", 0)
                total_queries += getattr(pc_stats, "total_queries", 0)
                evictions += getattr(pc_stats, "evictions", 0)

            prefix_index = getattr(prefix_cache, "_prefix_index", None)
            if prefix_index is not None and hasattr(prefix_index, "items"):
                for h in list(prefix_index.keys())[:64]:
                    prefix_hashes.append(h.hex() if isinstance(h, bytes) else str(h))

        cache_rate = getattr(scheduler, "_cache_rate_tracker", None)
        if cache_rate is not None:
            cumulative = cache_rate.get_rates().get("cumulative", {})
            hits = cumulative.get("prefix_hits", 0)
            misses = cumulative.get("prefix_misses", 0)
            if hits + misses > 0:
                hit_rate = round(hits / (hits + misses), 4)

    return NodeCacheState(
        node_id=node_id,
        hostname=hostname,
        total_blocks=total_blocks,
        used_blocks=used_blocks,
        hit_rate=hit_rate,
        prefix_hashes=prefix_hashes,
        tokens_saved=tokens_saved,
        total_queries=total_queries,
        evictions=evictions,
    )


def _collect_cluster_cache_events() -> list[CacheEvent]:
    """Collect recent cache events from loaded engines (best-effort)."""
    pool = _engine_pool()
    events: list[CacheEvent] = []

    for _model_id, entry in getattr(pool, "_entries", {}).items():
        engine = getattr(entry, "engine", None)
        if engine is None:
            continue

        async_core = getattr(engine, "_engine", None)
        core = getattr(async_core, "engine", None) if async_core is not None else None
        scheduler = getattr(core, "scheduler", None) if core is not None else None
        if scheduler is None:
            continue

        prefix_cache = getattr(scheduler, "block_aware_cache", None)
        if prefix_cache is None:
            continue

        pc_stats = getattr(prefix_cache, "stats", None)
        if pc_stats is None:
            continue

        now = time.time()
        if getattr(pc_stats, "hits", 0) > 0:
            events.append(
                CacheEvent(
                    kind="hit",
                    ts=now,
                    tokens_saved=getattr(pc_stats, "tokens_saved", 0),
                )
            )
        if getattr(pc_stats, "misses", 0) > 0:
            events.append(
                CacheEvent(kind="miss", ts=now)
            )
        if getattr(pc_stats, "evictions", 0) > 0:
            events.append(
                CacheEvent(kind="eviction", ts=now)
            )

    events.sort(key=lambda e: e.ts, reverse=True)
    return events[:50]


@router.get("/prefix-cache")
async def get_prefix_cache_topology() -> PrefixCacheTopology:
    """Return the prefix-cache topology across all loaded engines."""
    try:
        node_state = _collect_local_cache_state()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to collect prefix-cache state: {exc}",
        ) from exc

    events = _collect_cluster_cache_events()

    totals = {
        "total_blocks": node_state.total_blocks,
        "used_blocks": node_state.used_blocks,
        "hit_rate": node_state.hit_rate,
        "tokens_saved": node_state.tokens_saved,
        "total_queries": node_state.total_queries,
        "evictions": node_state.evictions,
        "unique_prefix_hashes": len(node_state.prefix_hashes),
    }

    return PrefixCacheTopology(
        nodes=[node_state],
        events=events,
        totals=totals,
    )
