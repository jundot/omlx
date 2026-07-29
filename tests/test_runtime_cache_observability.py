# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin runtime cache observability builder (#2396)."""

from types import SimpleNamespace
from unittest.mock import patch

from omlx.admin import routes as admin_routes


def _global_settings(tmp_path):
    cache_dir = tmp_path / "ssd_cache"
    return SimpleNamespace(
        base_path=tmp_path,
        cache=SimpleNamespace(
            get_ssd_cache_dir=lambda base_path: cache_dir,
            get_ssd_cache_max_size_bytes=lambda base_path: 0,
        ),
    )


class _Pool:
    def __init__(self, engine, model_id="model-a"):
        self._entries = {model_id: SimpleNamespace(engine=engine)}
        self._model_id = model_id

    def get_status(self):
        return {"models": [{"id": self._model_id, "loaded": True}]}


def _build(pool, tmp_path):
    with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
        return admin_routes._build_runtime_cache_observability(
            _global_settings(tmp_path)
        )


def test_dflash_engine_stats_populate_model_row(tmp_path):
    """An engine without an AsyncEngineCore contributes its own runtime
    cache stats through get_runtime_cache_stats()."""

    class Engine:
        scheduler = None

        def get_runtime_cache_stats(self):
            return {
                "ssd_cache": {
                    "num_files": 3,
                    "total_size_bytes": 300,
                    "max_size_bytes": 1000,
                    "hot_cache_max_bytes": 2048,
                    "hot_cache_size_bytes": 512,
                    "hot_cache_entries": 2,
                },
                "cache_rates": {"cumulative": {"prefix_hits": 5}},
            }

    payload = _build(_Pool(Engine()), tmp_path)

    assert len(payload["models"]) == 1
    row = payload["models"][0]
    assert row["num_files"] == 3
    assert row["total_size_bytes"] == 300
    assert row["hot_cache_max_bytes"] == 2048
    assert row["hot_cache_size_bytes"] == 512
    assert row["hot_cache_entries"] == 2
    assert row["cache_rates"] == {"cumulative": {"prefix_hits": 5}}
    # Aggregates drive the RAM chip visibility in the dashboard.
    assert payload["hot_cache_max_bytes"] == 2048
    assert payload["hot_cache_size_bytes"] == 512
    assert payload["hot_cache_entries"] == 2
    assert payload["total_num_files"] == 3
    assert payload["disk_max_bytes"] == 1000


def test_scheduler_property_resolved_without_async_core(tmp_path):
    """DFlash fallback mode: the fallback engine's scheduler is reachable
    through the engine's ``scheduler`` property."""

    class Engine:
        scheduler = SimpleNamespace(
            get_ssd_cache_stats=lambda: {
                "ssd_cache": {
                    "num_files": 1,
                    "total_size_bytes": 100,
                    "max_size_bytes": 0,
                    "hot_cache_max_bytes": 0,
                    "hot_cache_size_bytes": 0,
                    "hot_cache_entries": 0,
                },
            },
        )

    payload = _build(_Pool(Engine()), tmp_path)

    assert len(payload["models"]) == 1
    assert payload["models"][0]["num_files"] == 1


def test_engine_returning_none_stats_is_skipped(tmp_path):
    class Engine:
        scheduler = None

        def get_runtime_cache_stats(self):
            return None

    payload = _build(_Pool(Engine()), tmp_path)

    assert payload["models"] == []
    assert payload["hot_cache_max_bytes"] == 0
