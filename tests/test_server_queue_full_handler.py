# SPDX-License-Identifier: Apache-2.0
"""Verify SchedulerQueueFullError maps to HTTP 503 + Retry-After in server.py."""

import asyncio
import json

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from omlx.exceptions import SchedulerQueueFullError


def _build_test_app():
    """Build a minimal FastAPI app that re-uses the same exception handler.

    Importing omlx.server would pull in heavy server-state init. We pluck the
    handler function out of the module and register it against a fresh app
    so the test stays fast and free of state.
    """
    import omlx.server as srv

    app = FastAPI()
    app.add_exception_handler(
        SchedulerQueueFullError, srv.scheduler_queue_full_handler
    )

    @app.get("/v1/raise")
    def raise_queue_full():
        raise SchedulerQueueFullError(current_depth=32, max_depth=32)

    @app.get("/health/raise")
    def raise_queue_full_health():
        raise SchedulerQueueFullError(current_depth=33, max_depth=32)

    return app


class TestQueueFullHandler:
    def test_returns_503(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        assert resp.status_code == 503

    def test_has_retry_after_header(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        assert resp.headers.get("Retry-After") == "1"

    def test_api_route_uses_openai_error_body(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        body = resp.json()
        # _openai_error_body wraps in {"error": {...}}
        assert "error" in body
        assert "queue full" in body["error"]["message"].lower()
        # Depth numbers surface to the client
        assert "32/32" in body["error"]["message"]

    def test_non_api_route_uses_plain_detail(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/health/raise")
        body = resp.json()
        assert "detail" in body
        assert "queue full" in body["detail"].lower()


class TestQueueFullAfterResponseStarted:
    """#5305: admission can be rejected after the JSON keepalive stream already
    committed the status line to 200, e.g. a cold model load outlasting the 2 s
    grace period. The rejection must be serialized into the body instead of
    escaping the generator (which Starlette turns into a 500 + truncated
    response -- ``IncompleteRead`` client-side).
    """

    class _Request:
        async def is_disconnected(self):
            return False

    @pytest.mark.asyncio
    async def test_late_queue_full_is_serialized_into_the_body(self):
        import omlx.server as srv

        async def _raise_late():
            await asyncio.sleep(0.01)
            raise SchedulerQueueFullError(current_depth=32, max_depth=32)

        chunks = [
            chunk
            async for chunk in srv._with_json_keepalive(
                self._Request(), _raise_late(), disconnect_poll=0.001
            )
        ]
        body = json.loads("".join(chunks))
        assert body["error"]["type"] == "server_error"
        assert "32/32" in body["error"]["message"]
        assert "Try again shortly" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_slow_queue_full_returns_parseable_body(self, monkeypatch):
        import omlx.server as srv

        monkeypatch.setattr(srv, "_JSON_KEEPALIVE_GRACE_S", 0.01)

        async def _raise_slow():
            await asyncio.sleep(0.05)
            raise SchedulerQueueFullError(current_depth=32, max_depth=32)

        resp = await srv._json_response_or_keepalive(self._Request(), _raise_slow())
        assert isinstance(resp, srv.StreamingResponse)
        chunks = [chunk async for chunk in resp.body_iterator]
        body = json.loads("".join(chunks))
        assert "32/32" in body["error"]["message"]

    def test_client_sees_json_error_not_truncated_body(self, monkeypatch):
        import omlx.server as srv

        app = FastAPI()
        app.add_exception_handler(
            SchedulerQueueFullError, srv.scheduler_queue_full_handler
        )

        @app.get("/v1/probe")
        async def probe(request: Request):
            async def _raise_slow():
                await asyncio.sleep(0.05)
                raise SchedulerQueueFullError(current_depth=32, max_depth=32)

            return await srv._json_response_or_keepalive(request, _raise_slow())

        monkeypatch.setattr(srv, "_JSON_KEEPALIVE_GRACE_S", 0.01)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/probe")

        # The status line cannot be revised once the keepalive byte shipped.
        assert resp.status_code == 200
        body = resp.json()  # prior to the fix this raised JSONDecodeError
        assert "32/32" in body["error"]["message"]

    def test_fast_queue_full_still_returns_503_with_retry_after(self):
        import omlx.server as srv

        app = FastAPI()
        app.add_exception_handler(
            SchedulerQueueFullError, srv.scheduler_queue_full_handler
        )

        @app.get("/v1/probe")
        async def probe(request: Request):
            async def _raise_now():
                raise SchedulerQueueFullError(current_depth=32, max_depth=32)

            return await srv._json_response_or_keepalive(request, _raise_now())

        with TestClient(app) as client:
            resp = client.get("/v1/probe")

        assert resp.status_code == 503
        assert resp.headers.get("Retry-After") == "1"

    @pytest.mark.asyncio
    async def test_late_distributed_error_is_serialized_into_the_body(self):
        import omlx.server as srv
        from omlx.engine.distributed import DistributedInferenceError

        async def _raise_late():
            await asyncio.sleep(0.01)
            raise DistributedInferenceError("cluster preflight failed")

        chunks = [
            chunk
            async for chunk in srv._with_json_keepalive(
                self._Request(), _raise_late(), disconnect_poll=0.001
            )
        ]
        body = json.loads("".join(chunks))
        assert body["error"]["type"] == "server_error"
        assert "cluster preflight failed" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_late_unexpected_error_still_yields_parseable_body(self):
        import omlx.server as srv

        async def _raise_late():
            await asyncio.sleep(0.01)
            raise ValueError("boom")

        chunks = [
            chunk
            async for chunk in srv._with_json_keepalive(
                self._Request(), _raise_late(), disconnect_poll=0.001
            )
        ]
        assert json.loads("".join(chunks)) == {
            "error": {"message": "boom", "type": "server_error"}
        }
