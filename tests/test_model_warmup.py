# SPDX-License-Identifier: Apache-2.0
"""Regression tests for nonblocking integration model warm-up."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import omlx.server as server
from omlx.exceptions import EnginePoolError
from omlx.server import ServerState, load_model_public


@pytest.fixture
async def state(monkeypatch):
    pool = MagicMock()
    pool.get_entry.return_value = SimpleNamespace(engine=None, is_loading=False)
    pool.get_engine = AsyncMock()
    state = ServerState(engine_pool=pool)
    monkeypatch.setattr(server, "_server_state", state)
    yield state
    tasks = list(state.model_load_tasks.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_background_load_acknowledges_before_completion_and_deduplicates(state):
    started = asyncio.Event()
    finish = asyncio.Event()

    async def load(model_id):
        started.set()
        await finish.wait()
        state.engine_pool.get_entry.return_value.engine = object()

    state.engine_pool.get_engine.side_effect = load
    response = await asyncio.wait_for(load_model_public("model", wait=False), timeout=1)
    assert response.status_code == 202
    assert json.loads(response.body) == {
        "status": "loading",
        "model_id": "model",
        "message": "Loading model",
    }
    await asyncio.wait_for(started.wait(), timeout=1)
    task = state.model_load_tasks["model"]
    assert not task.done()

    repeated = await load_model_public("model", wait=False)
    assert repeated.status_code == 202
    assert state.model_load_tasks["model"] is task
    state.engine_pool.get_engine.assert_awaited_once_with("model")

    finish.set()
    await task
    await asyncio.sleep(0)
    assert state.model_load_tasks == {}
    assert (await load_model_public("model", wait=False))["status"] == "ok"
    state.engine_pool.get_engine.assert_awaited_once()


@pytest.mark.asyncio
async def test_waiting_load_retains_blocking_behavior(state):
    finish = asyncio.Event()

    async def load(model_id):
        await finish.wait()

    state.engine_pool.get_engine.side_effect = load
    request = asyncio.create_task(load_model_public("model"))
    try:
        await asyncio.sleep(0)
        assert not request.done()
        assert state.model_load_tasks == {}
        finish.set()
        assert await request == {
            "status": "ok",
            "model_id": "model",
            "message": "Loaded model",
        }
    finally:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("wait", [False, True])
async def test_loaded_model_does_not_schedule_work(state, wait):
    state.engine_pool.get_entry.return_value.engine = object()
    assert (await load_model_public("model", wait=wait))["status"] == "ok"
    assert state.model_load_tasks == {}
    state.engine_pool.get_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_http_background_load_returns_while_engine_is_loading(state):
    started = asyncio.Event()
    finish = asyncio.Event()

    async def load(model_id):
        started.set()
        await finish.wait()

    state.api_key = "secret"
    state.engine_pool.get_engine.side_effect = load
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://omlx"
    ) as client:
        response = await asyncio.wait_for(
            client.post(
                "/v1/models/model/load",
                params={"wait": "false"},
                headers={"Authorization": "Bearer secret"},
            ),
            timeout=1,
        )
        assert response.status_code == 202
        assert response.json()["model_id"] == "model"
        assert response.json()["status"] == "loading"
        await asyncio.wait_for(started.wait(), timeout=1)
        task = state.model_load_tasks["model"]
        assert not task.done()
        finish.set()
        await task


@pytest.mark.asyncio
async def test_existing_load_does_not_schedule_duplicate(state):
    state.engine_pool.get_entry.return_value.is_loading = True
    response = await load_model_public("model", wait=False)
    assert response.status_code == 202
    assert state.model_load_tasks == {}
    state.engine_pool.get_engine.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("wait", [False, True])
async def test_missing_model_is_rejected_before_scheduling(state, wait):
    state.engine_pool.get_entry.return_value = None
    with pytest.raises(HTTPException) as exc:
        await load_model_public("missing", wait=wait)
    assert exc.value.status_code == 404
    assert state.model_load_tasks == {}
    state.engine_pool.get_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_failure_is_logged_and_task_can_be_rescheduled(state, caplog):
    state.engine_pool.get_engine.side_effect = EnginePoolError("load failed")
    response = await load_model_public("model", wait=False)
    assert response.status_code == 202
    await state.model_load_tasks["model"]
    await asyncio.sleep(0)
    assert "Background model load failed for model" in caplog.text
    assert "load failed" in caplog.text
    assert state.model_load_tasks == {}

    state.engine_pool.get_engine.side_effect = None
    await load_model_public("model", wait=False)
    await state.model_load_tasks["model"]
    await asyncio.sleep(0)
    assert state.engine_pool.get_engine.await_count == 2
    assert state.model_load_tasks == {}


@pytest.mark.asyncio
async def test_waiting_load_retains_error_mapping(state):
    state.engine_pool.get_engine.side_effect = EnginePoolError("load failed")
    with pytest.raises(HTTPException) as exc:
        await load_model_public("model")
    assert exc.value.status_code == 500
    assert exc.value.detail == "load failed"


def test_background_load_requires_management_auth(monkeypatch):
    monkeypatch.setattr(server, "_server_state", ServerState(api_key="secret"))
    response = TestClient(server.app).post("/v1/models/model/load?wait=false")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_shutdown_cancels_background_load_before_pool_shutdown(
    state, monkeypatch
):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def load(model_id):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def shutdown():
        assert cancelled.is_set()
        assert state.model_load_tasks == {}

    state.engine_pool.get_engine.side_effect = load
    state.engine_pool.preload_pinned_models = AsyncMock()
    state.engine_pool.shutdown = AsyncMock(side_effect=shutdown)
    monkeypatch.delenv("OMLX_MCP_CONFIG", raising=False)
    monkeypatch.setattr(server, "_reset_boundary_snapshots_for_server", lambda: None)
    monkeypatch.setattr(server, "get_server_metrics", MagicMock())
    monkeypatch.setattr(
        "omlx.cluster.launch.reap_orphaned_launches",
        lambda: {"reaped": [], "failures": []},
    )
    monkeypatch.setattr(
        "omlx.cluster.worker_shim.ensure_cluster_python_shim", lambda: None
    )

    async with server.lifespan(server.app):
        await load_model_public("model", wait=False)
        await asyncio.wait_for(started.wait(), timeout=1)
        task = state.model_load_tasks["model"]

    assert task.cancelled()
    state.engine_pool.shutdown.assert_awaited_once()
