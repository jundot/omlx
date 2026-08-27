# SPDX-License-Identifier: Apache-2.0
"""Admin cache maintenance must reach every rank of a loaded cluster."""

import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.admin.routes as admin_routes


def _pool(clear):
    core = SimpleNamespace(scheduler=None, clear_prompt_caches=clear)
    entry = SimpleNamespace(engine=core)
    pool = SimpleNamespace()
    pool.get_status = MagicMock(return_value={
        "models": [{"id": "cluster-model", "loaded": True}]
    })
    pool._entries = {"cluster-model": entry}
    return pool


@pytest.mark.asyncio
async def test_ssd_clear_aggregates_distributed_rank_results():
    clear = AsyncMock(
        return_value={
            "status": "ok",
            "ssd_deleted": 9,
            "hot_cleared": 0,
            "ranks": [{"rank": 0}, {"rank": 1}],
        }
    )
    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=_pool(clear)),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
    ):
        result = await admin_routes.clear_ssd_cache(is_admin=True)

    clear.assert_awaited_once_with(ssd=True)
    assert result == {
        "status": "ok",
        "total_deleted": 9,
        "distributed_ranks": 2,
    }


@pytest.mark.asyncio
async def test_ssd_clear_surfaces_partial_cluster_failure():
    clear = AsyncMock(side_effect=RuntimeError("rank 1 unreachable"))
    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=_pool(clear)),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
    ):
        with pytest.raises(HTTPException) as raised:
            await admin_routes.clear_ssd_cache(is_admin=True)

    assert raised.value.status_code == 503
    assert "rank 1 unreachable" in raised.value.detail


@pytest.mark.asyncio
async def test_ssd_clear_removes_cold_cluster_and_legacy_roots(tmp_path):
    cache_root = tmp_path / "cache"
    cluster_root = cache_root / "cluster-prompt-snapshots"
    legacy_root = tmp_path / "state" / "cluster/runtime/prompt-cache-ssd"
    for root in (cluster_root, legacy_root):
        target = root / "deployment" / "plan" / "rank-0" / "snapshot.safetensors"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"cache")

    cache = SimpleNamespace(get_ssd_cache_dir=lambda _base: cache_root)
    settings = SimpleNamespace(cache=cache, base_path=tmp_path / "state")
    pool = SimpleNamespace(
        get_status=MagicMock(return_value={"models": []}),
        _entries={},
    )
    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(admin_routes, "_get_global_settings", return_value=settings),
        patch.object(
            admin_routes,
            "_clear_cold_remote_cluster_cache_roots",
            return_value=(0, 0),
        ),
    ):
        result = await admin_routes.clear_ssd_cache(is_admin=True)

    assert result["distributed_ranks"] == 0
    assert result["total_deleted"] == 2
    assert not cluster_root.exists()
    assert not legacy_root.exists()


def test_cold_ssd_clear_reaches_every_unique_configured_peer(tmp_path, monkeypatch):
    from omlx.cluster.deployment import ClusterHost

    roots = (
        tmp_path / "cache/cluster-prompt-snapshots",
        tmp_path / "state/cluster/runtime/prompt-cache-ssd",
    )
    hosts = (
        ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
        ClusterHost("peer", "peer.local", ("10.0.0.2",)),
    )
    registry = SimpleNamespace(
        list=lambda: (
            SimpleNamespace(hosts=hosts),
            SimpleNamespace(hosts=hosts),
        )
    )
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "7\n", "")

    monkeypatch.setattr(
        "omlx.cluster.registry.get_cluster_registry",
        lambda: registry,
    )

    deleted, ranks = admin_routes._clear_cold_remote_cluster_cache_roots(
        roots,
        runner=run,
    )

    assert deleted == 7
    assert ranks == 2
    assert len(calls) == 1
    assert calls[0][0][-2] == "peer.local"
    assert "cluster-prompt-snapshots" in calls[0][0][-1]
