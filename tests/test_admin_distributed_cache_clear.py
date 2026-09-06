# SPDX-License-Identifier: Apache-2.0
"""Admin cache maintenance must reach every rank of a loaded cluster."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.admin.routes as admin_routes


def _pool(clear):
    core = SimpleNamespace(scheduler=None, clear_prompt_caches=clear)
    entry = SimpleNamespace(engine=core)
    pool = SimpleNamespace()
    pool.get_status = MagicMock(
        return_value={"models": [{"id": "cluster-model", "loaded": True}]}
    )
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
        pytest.raises(HTTPException) as raised,
    ):
        await admin_routes.clear_ssd_cache(is_admin=True)

    assert raised.value.status_code == 503
    assert "rank 1 unreachable" in raised.value.detail
