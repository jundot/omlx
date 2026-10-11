# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe admin-triggered model unload."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from omlx import server
from omlx.admin import routes as admin_routes
from omlx.engine_core import _raise_request_output_error
from omlx.exceptions import RequestAbortedError
from omlx.request import RequestOutput

UNLOAD_ABORT_MESSAGE = "Request aborted because model 'model-a' is being unloaded"


def _unload_abort_error() -> Exception:
    output = RequestOutput(
        request_id="req-1",
        finished=True,
        finish_reason="error",
        error=UNLOAD_ABORT_MESSAGE,
        error_code="model_unloading",
    )
    try:
        _raise_request_output_error(output)
    except Exception as e:
        return e
    raise AssertionError("error output did not raise")


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


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["manual admin unload", "manual unload"])
async def test_lease_rejected_during_manual_unload_uses_unload_error(reason):
    pool = MagicMock()
    pool.get_abort_requested_reason.return_value = reason
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


@pytest.mark.asyncio
async def test_public_unload_of_active_model_aborts_requests_first():
    entry = MagicMock()
    entry.engine = object()
    entry.is_loading = False
    pool = MagicMock()
    pool.get_entry.return_value = entry
    pool.request_unload = AsyncMock(return_value=False)

    with patch.object(server._server_state, "engine_pool", pool):
        response = await server.unload_model("model-a", _=True)

    assert response.status_code == 202
    assert json.loads(response.body) == {
        "status": "unloading",
        "model_id": "model-a",
        "message": "Aborting active requests before unloading model-a",
    }
    pool.request_unload.assert_awaited_once_with("model-a", reason="manual unload")
    pool._unload_engine.assert_not_called()


@pytest.mark.asyncio
async def test_public_unload_rejects_a_model_that_is_still_loading():
    entry = MagicMock()
    entry.engine = object()
    entry.is_loading = True
    pool = MagicMock()
    pool.get_entry.return_value = entry
    pool.request_unload = AsyncMock()

    with (
        patch.object(server._server_state, "engine_pool", pool),
        pytest.raises(HTTPException) as exc_info,
    ):
        await server.unload_model("model-a", _=True)

    assert exc_info.value.status_code == 409
    pool.request_unload.assert_not_awaited()


def test_unload_abort_of_scheduled_request_returns_409():
    app = FastAPI()
    app.add_exception_handler(RequestAbortedError, server.request_aborted_handler)

    @app.post("/v1/chat/completions")
    async def chat():
        raise _unload_abort_error()

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions")

    assert response.status_code == 409
    assert response.json()["error"]["message"] == UNLOAD_ABORT_MESSAGE


@pytest.mark.asyncio
async def test_unload_abort_after_json_keepalive_started_keeps_409_body():
    result = asyncio.get_running_loop().create_future()
    stream = server._with_json_keepalive(None, result)
    assert await anext(stream) == " "
    result.set_exception(_unload_abort_error())

    chunks = [chunk async for chunk in stream]
    assert json.loads("".join(chunks)) == {
        "error": {
            "message": UNLOAD_ABORT_MESSAGE,
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    }
