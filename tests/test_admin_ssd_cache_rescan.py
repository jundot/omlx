"""Tests for POST /admin/api/ssd-cache/rescan.

The rescan endpoint re-runs the startup scan on the live SSD managers of
loaded models so out-of-band block files (e.g. copied from another node for
a KV handoff) become discoverable WITHOUT unloading models.

Mocks follow the test_admin_api_key.py style: monkeypatch _server_state /
SimpleNamespace engine pool, MagicMock SSD manager. The scan itself is
exercised end-to-end in tests/test_paged_ssd_cache.py
(test_scan_existing_files_*), so here we only assert the plumbing.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import omlx.server
import pytest

from omlx.admin import routes as admin_routes


def _make_pool(ssd=None, *, model_id="test-model"):
    """Engine pool whose single loaded model exposes the given SSD manager.

    Engine carries `scheduler` directly — the first path the pool's own
    `_resolve_scheduler_from_engine` helper checks; we wire that helper into
    the fake pool so the test exercises the real resolution order.
    """
    scheduler = SimpleNamespace(paged_ssd_cache_manager=ssd)
    engine = SimpleNamespace(scheduler=scheduler)
    entry = SimpleNamespace(engine=engine)
    pool = SimpleNamespace(
        get_loaded_model_ids=lambda: [model_id],
        _entries={model_id: entry},
        _resolve_scheduler_from_engine=lambda engine: getattr(engine, "scheduler", None),
    )
    return pool


@pytest.fixture
def server_state(monkeypatch):
    """Handler does `from ..server import _server_state` at call time, so the
    correct patch target is the module attribute (see test_admin_reload.py)."""
    state = SimpleNamespace(engine_pool=None)
    monkeypatch.setattr(omlx.server, "_server_state", state)
    return state


@pytest.mark.asyncio
async def test_rescan_no_engine_pool(server_state):
    server_state.engine_pool = None
    with pytest.raises(admin_routes.HTTPException) as exc:
        await admin_routes.rescan_ssd_cache(is_admin=True)
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_rescan_calls_scan_on_live_manager(server_state):
    ssd = MagicMock()
    ssd._tracked_ssd_count.side_effect = [10, 13]   # before, after
    ssd._tracked_ssd_size.side_effect = [1000, 1300]
    pool = _make_pool(ssd)
    server_state.engine_pool = pool

    result = await admin_routes.rescan_ssd_cache(is_admin=True)

    assert ssd._scan_existing_files.call_count == 1
    per_model = result["rescan"]["test-model"]
    assert per_model["status"] == "rescanned"
    assert per_model["blocks_before"] == 10
    assert per_model["blocks_after"] == 13
    assert per_model["bytes_before"] == 1000
    assert per_model["bytes_after"] == 1300


@pytest.mark.asyncio
async def test_rescan_reports_missing_manager(server_state):
    pool = _make_pool(ssd=None)
    server_state.engine_pool = pool

    result = await admin_routes.rescan_ssd_cache(is_admin=True)
    assert result["rescan"]["test-model"] == {"status": "no_ssd_manager"}


@pytest.mark.asyncio
async def test_rescan_dedupes_shared_manager(server_state):
    """Two loaded models sharing one SSD manager are scanned once."""
    ssd = MagicMock()
    ssd._tracked_ssd_count.side_effect = [5, 8]
    ssd._tracked_ssd_size.side_effect = [500, 800]
    scheduler = SimpleNamespace(paged_ssd_cache_manager=ssd)
    engine = SimpleNamespace(scheduler=scheduler)
    pool = SimpleNamespace(
        get_loaded_model_ids=lambda: ["model-a", "model-b"],
        _entries={
            "model-a": SimpleNamespace(engine=engine),
            "model-b": SimpleNamespace(engine=engine),
        },
        _resolve_scheduler_from_engine=lambda engine: getattr(engine, "scheduler", None),
    )
    server_state.engine_pool = pool

    result = await admin_routes.rescan_ssd_cache(is_admin=True)
    assert ssd._scan_existing_files.call_count == 1, "общий менеджер сканируется один раз"
    # first model carries the scan result, second is skipped silently
    assert result["rescan"]["model-a"]["status"] == "rescanned"
    assert "model-b" not in result["rescan"]


@pytest.mark.asyncio
async def test_rescan_idempotent_when_nothing_new(server_state):
    ssd = MagicMock()
    ssd._tracked_ssd_count.side_effect = [13, 13]
    ssd._tracked_ssd_size.side_effect = [1300, 1300]
    pool = _make_pool(ssd)
    server_state.engine_pool = pool

    result = await admin_routes.rescan_ssd_cache(is_admin=True)
    per_model = result["rescan"]["test-model"]
    assert per_model["blocks_before"] == per_model["blocks_after"]
    assert per_model["bytes_before"] == per_model["bytes_after"]
