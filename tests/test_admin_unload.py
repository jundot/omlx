# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe admin-triggered model unload."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from omlx import server
from omlx.admin import routes as admin_routes


@pytest.mark.asyncio
async def test_active_model_unload_returns_accepted_until_quiescent():
    entry = MagicMock()
    entry.engine = object()
    entry.is_loading = False
    pool = MagicMock()
    pool.get_entry.return_value = entry
    pool.request_unload = AsyncMock(return_value=False)

    with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
        response = await admin_routes.unload_model("model-a", is_admin=True)

    assert response.status_code == 202
    assert json.loads(response.body) == {
        "status": "unloading",
        "model_id": "model-a",
        "message": "Aborting active requests before unloading model-a",
    }
    pool.request_unload.assert_awaited_once_with(
        "model-a", reason="manual admin unload"
    )


@pytest.mark.asyncio
async def test_idle_model_unload_returns_completed():
    entry = MagicMock()
    entry.engine = object()
    entry.is_loading = False
    pool = MagicMock()
    pool.get_entry.return_value = entry
    pool.request_unload = AsyncMock(return_value=True)

    with patch.object(admin_routes, "_get_engine_pool", return_value=pool):
        response = await admin_routes.unload_model("model-a", is_admin=True)

    assert response == {
        "status": "ok",
        "model_id": "model-a",
        "message": "Unloaded model-a",
    }


def _idle_only_client():
    app = FastAPI()
    app.include_router(admin_routes.router)
    settings = MagicMock()
    settings.auth.skip_api_key_verification = False
    settings.auth.api_key = "test-admin-key"
    settings.auth.sub_keys = []
    return app, settings


def test_idle_only_unload_accepts_valid_bearer_and_returns_exact_model():
    pool = MagicMock()
    pool.get_entry.return_value = MagicMock()
    pool.unload_if_idle_unpinned = AsyncMock(return_value=True)
    app, settings = _idle_only_client()

    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(admin_routes, "_get_global_settings", return_value=settings),
        patch.object(admin_routes, "verify_session", return_value=False),
        TestClient(app) as client,
    ):
        response = client.post(
            "/admin/api/models/model-a/unload-if-idle",
            headers={"Authorization": "Bearer test-admin-key"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_id": "model-a",
        "message": "Unloaded idle model model-a",
    }
    pool.unload_if_idle_unpinned.assert_awaited_once_with("model-a")


def test_idle_only_unload_accepts_valid_admin_session():
    pool = MagicMock()
    pool.get_entry.return_value = MagicMock()
    pool.unload_if_idle_unpinned = AsyncMock(return_value=True)
    app, settings = _idle_only_client()

    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(admin_routes, "_get_global_settings", return_value=settings),
        patch.object(admin_routes, "verify_session", return_value=True),
        TestClient(app) as client,
    ):
        response = client.post("/admin/api/models/model-a/unload-if-idle")

    assert response.status_code == 200
    assert response.json()["model_id"] == "model-a"
    pool.unload_if_idle_unpinned.assert_awaited_once_with("model-a")


@pytest.mark.parametrize("authorization", [None, "Bearer wrong-key"])
def test_idle_only_unload_rejects_missing_or_invalid_bearer(authorization):
    pool = MagicMock()
    pool.unload_if_idle_unpinned = AsyncMock(return_value=False)
    app, settings = _idle_only_client()
    headers = {} if authorization is None else {"Authorization": authorization}

    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(admin_routes, "_get_global_settings", return_value=settings),
        patch.object(admin_routes, "verify_session", return_value=False),
        TestClient(app) as client,
    ):
        response = client.post(
            "/admin/api/models/model-a/unload-if-idle", headers=headers
        )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    pool.get_entry.assert_not_called()
    pool.unload_if_idle_unpinned.assert_not_awaited()


@pytest.mark.parametrize(
    "state, expected_status", [("unavailable", 503), ("missing", 404)]
)
def test_idle_only_unload_reports_unavailable_or_missing(state, expected_status):
    pool = None if state == "unavailable" else MagicMock()
    if state == "missing":
        pool.get_entry.return_value = None
    app, settings = _idle_only_client()

    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(admin_routes, "_get_global_settings", return_value=settings),
        patch.object(admin_routes, "verify_session", return_value=False),
        TestClient(app) as client,
    ):
        response = client.post(
            "/admin/api/models/model-a/unload-if-idle",
            headers={"Authorization": "Bearer test-admin-key"},
        )

    assert response.status_code == expected_status


def test_idle_only_unload_reports_conflict_without_unloading():
    pool = MagicMock()
    pool.get_entry.return_value = MagicMock()
    pool.unload_if_idle_unpinned = AsyncMock(return_value=False)
    app, settings = _idle_only_client()

    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(admin_routes, "_get_global_settings", return_value=settings),
        patch.object(admin_routes, "verify_session", return_value=False),
        TestClient(app) as client,
    ):
        response = client.post(
            "/admin/api/models/model-a/unload-if-idle",
            headers={"Authorization": "Bearer test-admin-key"},
        )

    assert response.status_code == 409
    pool.unload_if_idle_unpinned.assert_awaited_once_with("model-a")


@pytest.mark.asyncio
async def test_lease_rejected_during_manual_unload_uses_unload_error():
    pool = MagicMock()
    pool.get_abort_requested_reason.return_value = "manual admin unload"
    lease = server._LLMEngineLease(model_id="model-a")

    with (
        patch.object(server._server_state, "engine_pool", pool),
        pytest.raises(HTTPException) as exc_info,
    ):
        await server._raise_if_llm_lease_abort_requested(lease)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == (
        "Request aborted because this model is being unloaded."
    )
