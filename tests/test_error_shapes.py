# SPDX-License-Identifier: Apache-2.0
"""Error-body shapes for the Anthropic endpoints and unmatched routes."""

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

import omlx.server as server
from omlx.exceptions import InvalidRequestError, SchedulerQueueFullError


def _request(path: str, method: str = "POST"):
    return SimpleNamespace(url=SimpleNamespace(path=path), method=method)


@pytest.mark.asyncio
async def test_v1_messages_http_error_uses_anthropic_shape():
    response = await server.http_exception_handler(
        _request("/v1/messages"), HTTPException(401, "Invalid API key")
    )

    assert response.status_code == 401
    body = json.loads(response.body)
    assert body == {
        "type": "error",
        "error": {
            "type": "authentication_error",
            "message": "Invalid API key",
        },
    }


@pytest.mark.asyncio
async def test_v1_messages_validation_error_uses_anthropic_shape():
    exc = RequestValidationError(
        [
            {
                "loc": ("body", "max_tokens"),
                "msg": "Field required",
                "type": "missing",
            }
        ]
    )
    response = await server.validation_exception_handler(
        _request("/v1/messages"), exc
    )

    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in body["error"]["message"]


@pytest.mark.asyncio
async def test_v1_messages_routing_error_uses_anthropic_shape():
    response = await server.starlette_http_exception_handler(
        _request("/v1/messages/count_tokens", "POST"),
        StarletteHTTPException(404, "Not Found"),
    )

    assert response.status_code == 404
    body = json.loads(response.body)
    assert body["type"] == "error"
    assert body["error"]["type"] == "not_found_error"


@pytest.mark.asyncio
async def test_openai_routing_error_uses_openai_shape():
    response = await server.starlette_http_exception_handler(
        _request("/v1/does-not-exist", "GET"),
        StarletteHTTPException(404, "Not Found"),
    )

    assert response.status_code == 404
    body = json.loads(response.body)
    assert "type" not in body
    assert body["error"]["message"] == "Not Found"


@pytest.mark.asyncio
async def test_admin_routing_error_keeps_detail_shape():
    response = await server.starlette_http_exception_handler(
        _request("/admin/api/does-not-exist", "GET"),
        StarletteHTTPException(404, "Not Found"),
    )

    assert response.status_code == 404
    assert json.loads(response.body) == {"detail": "Not Found"}


def test_unmatched_v1_route_returns_openai_error_body():
    client = TestClient(server.app)
    response = client.get("/v1/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["message"] == "Not Found"


@pytest.mark.asyncio
async def test_v1_messages_invalid_request_uses_anthropic_shape():
    response = await server.invalid_request_error_handler(
        _request("/v1/messages"),
        InvalidRequestError("Tool calling is not supported", field="tools"),
    )

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Tool calling is not supported",
        },
    }


@pytest.mark.asyncio
async def test_v1_messages_queue_full_uses_anthropic_shape():
    response = await server.scheduler_queue_full_handler(
        _request("/v1/messages"),
        SchedulerQueueFullError(current_depth=32, max_depth=32),
    )

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    body = json.loads(response.body)
    assert body["type"] == "error"
    assert body["error"]["type"] == "overloaded_error"


@pytest.mark.asyncio
async def test_v1_messages_unhandled_error_uses_anthropic_shape():
    response = await server.unhandled_exception_handler(
        _request("/v1/messages"), RuntimeError("internal detail")
    )

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body == {
        "type": "error",
        "error": {"type": "api_error", "message": "Internal server error"},
    }


@pytest.mark.asyncio
async def test_other_api_route_invalid_request_keeps_openai_shape():
    response = await server.invalid_request_error_handler(
        _request("/v1/chat/completions"),
        InvalidRequestError("logprobs is not supported", field="logprobs"),
    )

    body = json.loads(response.body)
    assert "type" not in body
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "logprobs"


def test_unmatched_anthropic_route_returns_anthropic_error_body():
    client = TestClient(server.app)
    response = client.post("/v1/messages/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["type"] == "error"
    assert body["error"] == {"type": "not_found_error", "message": "Not Found"}


def test_path_that_only_starts_with_messages_keeps_openai_shape():
    """A longer unmatched path is not an Anthropic route."""
    client = TestClient(server.app)
    response = client.post("/v1/messages-extra")

    assert response.status_code == 404
    body = response.json()
    assert "type" not in body
    assert body["error"]["type"] == "not_found_error"


def test_v1_messages_errors_are_anthropic_shaped_end_to_end(monkeypatch, tmp_path):
    """Routing, auth and validation all reach the Anthropic envelope."""
    from omlx.admin import auth
    from omlx.settings import GlobalSettings

    settings = GlobalSettings(base_path=tmp_path)
    settings.server.host = "0.0.0.0"
    settings.auth.api_key = "management-key"
    monkeypatch.setattr(server._server_state, "global_settings", settings)
    monkeypatch.setattr(server._server_state, "api_key", "management-key")
    monkeypatch.setattr(server._server_state, "bind_host", "0.0.0.0")
    monkeypatch.setattr(auth, "_get_global_settings", lambda: settings)

    client = TestClient(server.app)

    unauthorized = client.post("/v1/messages", json={})
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["type"] == "authentication_error"

    invalid = client.post(
        "/v1/messages",
        json={},
        headers={"Authorization": "Bearer management-key"},
    )
    assert invalid.status_code == 400
    body = invalid.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "model" in body["error"]["message"]
