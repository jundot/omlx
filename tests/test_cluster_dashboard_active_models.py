# SPDX-License-Identifier: Apache-2.0
"""Cluster deployments on the main dashboard: live stats, badge, cache row.

The Active Models card and runtime-cache observability are fed by
``GET /admin/api/stats``; distributed engines own no local scheduler, so the
rows are adapted from rank zero's telemetry marker instead.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omlx.admin import routes as admin_routes
from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.planner import PipelineAssignment
from omlx.engine.distributed import DistributedBatchedEngine
from omlx.engine_pool import EngineEntry, EnginePool

ROOT = Path(__file__).resolve().parents[1]


def _deployment(
    model_path: str = "/models/nemotron",
    *,
    host_count: int = 2,
    tensor_parallel_size: int = 1,
) -> ClusterDeployment:
    node_ids = ["local"] + [f"peer-{i}" for i in range(1, host_count)]
    return ClusterDeployment(
        deployment_id="dash-test",
        model=model_path,
        backend="ring",
        hosts=tuple(
            ClusterHost(
                node_id,
                "127.0.0.1" if rank == 0 else f"{node_id}.local",
                (f"10.0.0.{rank + 1}",),
            )
            for rank, node_id in enumerate(node_ids)
        ),
        assignments=tuple(
            PipelineAssignment(
                node_id, rank, rank * 2, rank * 2 + 2, 10, 10, 8, 128
            )
            for rank, node_id in enumerate(node_ids)
        ),
        plan_hash="d" * 64,
        tensor_parallel_size=tensor_parallel_size,
    )


def _write_marker(state_dir: Path, deployment_id: str, payload: dict) -> None:
    (state_dir / f"{deployment_id}-rank-0.json").write_text(json.dumps(payload))


def _engine_with_marker(tmp_path: Path) -> DistributedBatchedEngine:
    engine = DistributedBatchedEngine(_deployment())
    engine._supervisor = SimpleNamespace(state_dir=str(tmp_path))
    return engine


def _metrics_payload(**overrides) -> dict:
    metrics = {
        "scope": "end_to_end_pipeline",
        "active_requests": 0,
        "requests_completed": 3,
        "requests_failed": 0,
        "requests_cancelled": 0,
        "prompt_tokens_total": 900,
        "completion_tokens_total": 120,
        "cached_tokens_total": 300,
        "aggregate_decode_tps": 11.5,
        "cache": {
            "affinity": "deployment",
            "lookups": 4,
            "hits": 3,
            "misses": 1,
            "hit_rate": 0.75,
            "tokens_reused": 300,
            "entries": 2,
            "bytes": 4096,
        },
        "pipeline": {
            "batch_steps": 9,
            "busy_seconds": 4.0,
            "idle_seconds": 6.0,
            "utilization": 0.4,
            "microbatch_target": 1,
            "async_overlap": False,
            "last_batch": None,
        },
        "last_request": None,
    }
    metrics.update(overrides)
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# DistributedBatchedEngine.get_live_metrics
# ---------------------------------------------------------------------------


def test_get_live_metrics_reads_rank_zero_marker(tmp_path):
    engine = _engine_with_marker(tmp_path)
    _write_marker(tmp_path, "dash-test", _metrics_payload())

    live = engine.get_live_metrics()

    assert live is not None
    assert live["metrics"]["cache"]["hit_rate"] == 0.75
    assert isinstance(live["updated_at"], str)
    assert live["age_seconds"] is not None and live["age_seconds"] < 5
    assert live["stale"] is False


def test_get_live_metrics_marks_old_heartbeat_stale(tmp_path):
    engine = _engine_with_marker(tmp_path)
    payload = _metrics_payload()
    payload["updated_at"] = (
        datetime.now(UTC) - timedelta(seconds=120)
    ).isoformat()
    _write_marker(tmp_path, "dash-test", payload)

    live = engine.get_live_metrics()

    assert live is not None
    assert live["stale"] is True
    assert live["age_seconds"] >= 120


def test_get_live_metrics_returns_none_without_marker_or_metrics(tmp_path):
    engine = _engine_with_marker(tmp_path)

    assert engine.get_live_metrics() is None

    _write_marker(tmp_path, "dash-test", {"updated_at": "now"})
    assert engine.get_live_metrics() is None

    (tmp_path / "dash-test-rank-0.json").write_text("not json")
    assert engine.get_live_metrics() is None


# ---------------------------------------------------------------------------
# EnginePool.get_status cluster badge payload
# ---------------------------------------------------------------------------


def _pool_with_cluster_entry(tmp_path, deployment) -> EnginePool:
    pool = EnginePool()
    pool._cluster_registry = SimpleNamespace(
        get_for_model=lambda model: deployment
        if model == deployment.model
        else None
    )
    pool._entries["nemotron"] = EngineEntry(
        model_id="nemotron",
        model_path=deployment.model,
        model_type="llm",
        engine_type="batched",
        estimated_size=300,
    )
    return pool


def test_pool_status_cluster_payload_pipeline_strategy(tmp_path):
    deployment = _deployment(host_count=2, tensor_parallel_size=1)
    model = _pool_with_cluster_entry(tmp_path, deployment).get_status()["models"][0]

    assert model["distributed"] is True
    assert model["cluster"] == {
        "deployment_id": "dash-test",
        "world_size": 2,
        "tensor_parallel_size": 1,
        "pipeline_stages": 2,
        "strategy": "pipeline",
        "backend": "ring",
        "target_context_tokens": 8192,
        "profile": "balanced",
    }


def test_pool_status_cluster_payload_tensor_strategy(tmp_path):
    deployment = _deployment(host_count=2, tensor_parallel_size=2)
    cluster = _pool_with_cluster_entry(tmp_path, deployment).get_status()[
        "models"
    ][0]["cluster"]

    assert cluster["strategy"] == "tensor"
    assert cluster["tensor_parallel_size"] == 2
    assert cluster["pipeline_stages"] == 1


def test_pool_status_cluster_payload_hybrid_strategy(tmp_path):
    deployment = _deployment(host_count=4, tensor_parallel_size=2)
    cluster = _pool_with_cluster_entry(tmp_path, deployment).get_status()[
        "models"
    ][0]["cluster"]

    assert cluster["strategy"] == "hybrid"
    assert cluster["world_size"] == 4
    assert cluster["tensor_parallel_size"] == 2
    assert cluster["pipeline_stages"] == 2


def test_pool_status_local_model_has_no_cluster_payload(tmp_path):
    pool = EnginePool()
    pool._cluster_registry = SimpleNamespace(get_for_model=lambda model: None)
    pool._entries["local-model"] = EngineEntry(
        model_id="local-model",
        model_path="/models/local",
        model_type="llm",
        engine_type="batched",
        estimated_size=100,
    )

    model = pool.get_status()["models"][0]

    assert model["distributed"] is False
    assert model["cluster"] is None


# ---------------------------------------------------------------------------
# _build_active_models_data distributed rows
# ---------------------------------------------------------------------------


class _ClusterPool:
    def __init__(self, live):
        engine = SimpleNamespace(
            _engine=None,
            get_live_metrics=lambda: live,
        )
        self._entries = {"cluster-model": SimpleNamespace(engine=engine)}

    def get_status(self):
        return {
            "current_model_memory": 0,
            "final_ceiling": 0,
            "models": [
                {
                    "id": "cluster-model",
                    "loaded": True,
                    "is_loading": False,
                    "loading_started_at": None,
                    "estimated_size": 1024,
                    "actual_size": 0,
                    "pinned": False,
                    "distributed": True,
                    "cluster": {
                        "deployment_id": "dash-test",
                        "world_size": 2,
                        "tensor_parallel_size": 1,
                        "pipeline_stages": 2,
                        "strategy": "pipeline",
                        "backend": "ring",
                        "target_context_tokens": 8192,
                        "profile": "balanced",
                    },
                    "last_access": None,
                }
            ],
        }


class _EmptyPrefillTracker:
    def get_model_progress(self, model_id):
        return []


def _build_active_models(pool):
    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(admin_routes, "_get_server_state", return_value=None),
        patch.object(admin_routes, "_get_settings_manager", return_value=None),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
        patch(
            "omlx.prefill_progress.get_prefill_tracker",
            return_value=_EmptyPrefillTracker(),
        ),
    ):
        return admin_routes._build_active_models_data()


def test_active_models_distributed_row_shows_decode_rate():
    live = {
        "metrics": _metrics_payload(
            active_requests=1,
            last_request={
                "status": "running",
                "prompt_tokens": 128,
                "cached_tokens": 64,
                "completion_tokens": 30,
                "elapsed_seconds": 2.0,
                "ttft_seconds": 0.5,
                "prefill_tps": 128.0,
                "decode_tps": 21.5,
                "end_to_end_tps": 15.0,
                "prefill_progress": {
                    "active": False,
                    "processed": 64,
                    "total": 64,
                    "speed": 0.0,
                    "average_speed": 0.0,
                    "eta": None,
                    "elapsed": 0.5,
                },
            },
        )["metrics"],
        "updated_at": datetime.now(UTC).isoformat(),
        "age_seconds": 0.4,
        "stale": False,
    }

    model = _build_active_models(_ClusterPool(live))["models"][0]

    assert model["active_requests"] == 1
    assert model["prefilling"] == []
    assert model["generating"] == [
        {
            "request_id": "rank0",
            "elapsed_seconds": 2.0,
            "generated_tokens": 30,
            "tokens_per_second": 21.5,
            "last_activity_age_seconds": 0.4,
            "prompt_tokens": 128,
            "max_tokens": None,
        }
    ]
    assert model["cluster"]["strategy"] == "pipeline"
    assert model["cluster"]["live"]["metrics"]["cache"]["hit_rate"] == 0.75


def test_active_models_distributed_row_shows_prefill_progress():
    live = {
        "metrics": _metrics_payload(
            active_requests=1,
            last_request={
                "status": "running",
                "prompt_tokens": 1000,
                "cached_tokens": 0,
                "completion_tokens": 0,
                "elapsed_seconds": 1.5,
                "ttft_seconds": None,
                "prefill_tps": 400.0,
                "decode_tps": 0.0,
                "end_to_end_tps": 0.0,
                "prefill_progress": {
                    "active": True,
                    "processed": 600,
                    "total": 1000,
                    "speed": 420.0,
                    "average_speed": 400.0,
                    "eta": 0.95,
                    "elapsed": 1.5,
                },
            },
        )["metrics"],
        "updated_at": datetime.now(UTC).isoformat(),
        "age_seconds": 0.2,
        "stale": False,
    }

    model = _build_active_models(_ClusterPool(live))["models"][0]

    assert model["active_requests"] == 1
    assert model["generating"] == []
    assert model["prefilling"] == [
        {
            "request_id": "rank0",
            "processed": 600,
            "total": 1000,
            "speed": 420.0,
            "eta": 0.95,
            "elapsed": 1.5,
            "detail": "cluster prefill",
        }
    ]


def test_active_models_distributed_stale_marker_renders_idle():
    live = {
        "metrics": _metrics_payload(
            active_requests=2,
            last_request={
                "status": "running",
                "decode_tps": 42.0,
                "prefill_progress": {"active": False},
            },
        )["metrics"],
        "updated_at": (datetime.now(UTC) - timedelta(seconds=120)).isoformat(),
        "age_seconds": 121.0,
        "stale": True,
    }

    model = _build_active_models(_ClusterPool(live))["models"][0]

    # Stale telemetry must never surface as live rates.
    assert model["active_requests"] == 0
    assert model["generating"] == []
    assert model["prefilling"] == []
    # …but the badge still tells the user the data is stale.
    assert model["cluster"]["live"]["stale"] is True


def test_active_models_distributed_without_marker_renders_idle():
    model = _build_active_models(_ClusterPool(None))["models"][0]

    assert model["active_requests"] == 0
    assert model["generating"] == []
    assert model["prefilling"] == []
    assert model["cluster"]["live"] is None
    assert model["cluster"]["world_size"] == 2


# ---------------------------------------------------------------------------
# _build_runtime_cache_observability distributed fallback
# ---------------------------------------------------------------------------


def _global_settings(tmp_path):
    cache_dir = tmp_path / "ssd_cache"
    return SimpleNamespace(
        base_path=tmp_path,
        cache=SimpleNamespace(
            get_ssd_cache_dir=lambda base_path: cache_dir,
            get_ssd_cache_max_size_bytes=lambda base_path: 0,
        ),
    )


class _CachePool:
    def __init__(self, live):
        engine = SimpleNamespace(
            _engine=None,
            get_live_metrics=lambda: live,
        )
        self._entries = {"cluster-model": SimpleNamespace(engine=engine)}

    def get_status(self):
        return {
            "models": [
                {
                    "id": "cluster-model",
                    "loaded": True,
                    "distributed": True,
                    "cluster": {"deployment_id": "dash-test"},
                }
            ]
        }


def _build_runtime_cache(pool, tmp_path):
    with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
        return admin_routes._build_runtime_cache_observability(
            _global_settings(tmp_path)
        )


def test_runtime_cache_distributed_row_uses_rank_prompt_cache(tmp_path):
    live = {
        "metrics": _metrics_payload()["metrics"],
        "updated_at": datetime.now(UTC).isoformat(),
        "age_seconds": 0.3,
        "stale": False,
    }

    payload = _build_runtime_cache(_CachePool(live), tmp_path)

    assert len(payload["models"]) == 1
    row = payload["models"][0]
    assert row["cache_tier"] == "rank-prompt-snapshot"
    assert row["rank_prompt_cache"] == {
        "entries": 2,
        "bytes": 4096,
        "lookups": 4,
        "hits": 3,
        "misses": 1,
        "hit_rate": 0.75,
        "tokens_reused": 300,
        "affinity": "deployment",
    }
    # Rank prompt-cache/snapshot stats are NOT the tiered hot/SSD cache and
    # must not leak into its columns or aggregates.
    assert row["num_files"] == 0
    assert row["total_size_bytes"] == 0
    assert row["hot_cache_size_bytes"] == 0
    assert row["hot_cache_entries"] == 0
    assert payload["hot_cache_size_bytes"] == 0
    assert payload["hot_cache_entries"] == 0
    assert payload["total_num_files"] == 0
    assert payload["total_size_bytes"] == 0


def test_runtime_cache_distributed_stale_marker_contributes_no_row(tmp_path):
    live = {
        "metrics": _metrics_payload()["metrics"],
        "updated_at": (datetime.now(UTC) - timedelta(seconds=120)).isoformat(),
        "age_seconds": 121.0,
        "stale": True,
    }

    payload = _build_runtime_cache(_CachePool(live), tmp_path)

    assert payload["models"] == []


# ---------------------------------------------------------------------------
# Template / i18n contract
# ---------------------------------------------------------------------------


def test_status_template_renders_cluster_badge_and_rank_cache_row():
    status = (ROOT / "omlx/admin/templates/dashboard/_status.html").read_text()
    javascript = (ROOT / "omlx/admin/static/js/dashboard.js").read_text()
    en = json.loads((ROOT / "omlx/admin/i18n/en.json").read_text())

    assert status.count("clusterBadgeLabel(m.cluster)") == 2  # mobile + desktop
    assert "clusterBadgeLabel(cluster)" in javascript
    assert "m.cluster.live && m.cluster.live.stale" in status
    assert "m.cache_tier === 'rank-prompt-snapshot'" in status
    assert "m.rank_prompt_cache" in status
    for key in (
        "cluster.badge.label",
        "cluster.badge.tensor",
        "cluster.badge.pipeline",
        "cluster.badge.stale",
        "cluster.badge.rank_cache",
        "cluster.badge.rank_cache_entries",
    ):
        assert en.get(key), f"en.json missing {key}"
