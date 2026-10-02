# SPDX-License-Identifier: Apache-2.0
"""Tests for the request body size limiting middleware."""

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.api.body_limit import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    RequestBodySizeLimitMiddleware,
)


@pytest.fixture
def small_app():
    """FastAPI app wrapped with a 1 KiB limit; echoes the parsed body."""
    app = FastAPI()
    app.add_middleware(RequestBodySizeLimitMiddleware, max_bytes=1024)

    @app.post("/echo")
    async def echo(payload: dict):
        return {"keys": len(payload)}

    return app


def test_content_length_over_limit_rejected_before_app(small_app):
    """A Content-Length above the cap gets a 413 without reaching handlers."""
    client = TestClient(small_app, raise_server_exceptions=True)
    response = client.post(
        "/echo",
        content=b"x" * 4096,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "request_too_large"


def test_content_length_under_limit_reaches_handler(small_app):
    client = TestClient(small_app)
    response = client.post(
        "/echo",
        content=json.dumps({"a": 1}).encode(),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200
    assert response.json() == {"keys": 1}


def test_no_body_request_passes_through(small_app):
    """GET requests without a body are not limited."""
    client = TestClient(small_app)
    assert client.get("/docs").status_code in (200, 404)


def test_chunked_overflow_stops_accumulation():
    """Without Content-Length, the wrapped receive reports disconnect at the cap.

    The consumer sees at most one chunk past the cap before the disconnect;
    accumulation over the limit is impossible.
    """
    middleware = RequestBodySizeLimitMiddleware(None, max_bytes=64)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/echo",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
    }
    chunks = [b"x" * 40 for _ in range(5)]  # 200 bytes total vs 64 cap

    async def receive():
        if not chunks:
            return {"type": "http.disconnect"}
        body = chunks.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(chunks)}

    seen = []

    async def inner_app(scope, receive, send):
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            total += len(message.get("body", b""))
            if not message.get("more_body"):
                break
        seen.append(total)

    async def passthrough_send(message):
        return None

    async def main():
        wrapped = RequestBodySizeLimitMiddleware(inner_app, max_bytes=64)
        await wrapped(scope, receive, passthrough_send)

    asyncio.run(main())
    assert seen, "downstream consumer must still run"
    assert seen[0] <= 64 + 40  # one chunk may land before the cap trips


def test_zero_limit_disables_middleware():
    """max_bytes <= 0 short-circuits: the app is called unchanged."""
    seen = []

    async def inner_app(scope, receive, send):
        seen.append(scope["path"])

    middleware = RequestBodySizeLimitMiddleware(inner_app, max_bytes=0)

    async def main():
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/x",
                "headers": [(b"content-length", b"999999999")],
            },
            None,
            None,
        )

    asyncio.run(main())
    assert seen == ["/x"]


def test_non_http_scope_passthrough():
    seen = []

    async def inner_app(scope, receive, send):
        seen.append(scope["type"])

    middleware = RequestBodySizeLimitMiddleware(inner_app, max_bytes=16)

    async def main():
        await middleware({"type": "lifespan"}, None, None)

    asyncio.run(main())
    assert seen == ["lifespan"]


def test_default_limit_is_generous():
    """The module default must fit several max-size images plus overhead."""
    assert DEFAULT_MAX_REQUEST_BODY_BYTES >= 200 * 1024 * 1024
