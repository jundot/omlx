# SPDX-License-Identifier: Apache-2.0
"""Tests for per-request correlation ids (``x-request-id`` handling).

Covers the validator, the middleware that publishes the id and echoes it in
the response header, and the helpers the generation endpoints use to hand the
id to an engine.
"""

import asyncio
import uuid
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from omlx.server import (
    RequestIdMiddleware,
    _request_abort_id,
    _request_correlation_id,
)
from omlx.utils.request_id import (
    MAX_REQUEST_ID_BYTES,
    REQUEST_ID_STATE_KEY,
    new_request_id,
    resolve_request_id,
    valid_request_id,
)


class TestValidRequestId:
    """The charset/length rule that keeps ids safe in logs and dict keys."""

    def test_accepts_uuid_hex(self):
        value = uuid.uuid4().hex
        assert valid_request_id(value) == value

    def test_accepts_common_trace_shapes(self):
        for value in ("cmpl-1.2:3_4", "trace_id.7", "a", "0" * 32):
            assert valid_request_id(value) == value

    def test_accepts_max_length(self):
        value = "a" * MAX_REQUEST_ID_BYTES
        assert valid_request_id(value) == value

    def test_rejects_empty_and_non_string(self):
        for value in (None, "", 123, b"abc", ["a"]):
            assert valid_request_id(value) is None

    def test_rejects_over_max_length(self):
        assert valid_request_id("a" * (MAX_REQUEST_ID_BYTES + 1)) is None

    def test_rejects_non_ascii(self):
        assert valid_request_id("请求-1") is None

    def test_rejects_separators_and_control_characters(self):
        for value in ("a b", "a/b", "a\\b", "a\nb", 'a"b', "a,b", "a#b"):
            assert valid_request_id(value) is None


class TestResolveRequestId:
    def test_keeps_a_usable_client_value(self):
        assert resolve_request_id("client-trace-1") == "client-trace-1"

    def test_mints_when_missing_or_unusable(self):
        for value in (None, "", "bad id", "x" * (MAX_REQUEST_ID_BYTES + 1)):
            minted = resolve_request_id(value)
            assert valid_request_id(minted) == minted
            assert len(minted) == 32

    def test_mints_a_fresh_value_each_time(self):
        assert new_request_id() != new_request_id()


class _EchoRequestIdApp:
    """Minimal ASGI app that reports the id published on its scope."""

    async def __call__(self, scope, receive, send):
        request_id = (scope.get("state") or {}).get(REQUEST_ID_STATE_KEY)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send(
            {"type": "http.response.body", "body": str(request_id).encode("ascii")}
        )


def _client() -> TestClient:
    return TestClient(RequestIdMiddleware(_EchoRequestIdApp()))


class TestRequestIdMiddleware:
    def test_echoes_the_client_id_in_header_and_state(self):
        response = _client().get("/", headers={"x-request-id": "client-trace-1"})
        assert response.status_code == 200
        assert response.headers["x-request-id"] == "client-trace-1"
        assert response.text == "client-trace-1"

    def test_mints_when_the_header_is_absent(self):
        response = _client().get("/")
        request_id = response.headers["x-request-id"]
        assert valid_request_id(request_id) == request_id
        assert response.text == request_id

    def test_replaces_an_unusable_client_id(self):
        for value in ("bad id", "x" * (MAX_REQUEST_ID_BYTES + 1), "a/b"):
            response = _client().get("/", headers={"x-request-id": value})
            request_id = response.headers["x-request-id"]
            assert request_id != value
            assert valid_request_id(request_id) == request_id

    def test_replaces_a_non_ascii_client_id(self):
        """Header bytes outside ASCII must not survive into a request id."""
        response = _client().get("/", headers={"x-request-id": b"caf\xe9"})
        request_id = response.headers["x-request-id"]
        assert request_id != "café"
        assert valid_request_id(request_id) == request_id

    def test_each_request_gets_its_own_minted_id(self):
        client = _client()
        first = client.get("/").headers["x-request-id"]
        second = client.get("/").headers["x-request-id"]
        assert first != second


class _BlockingEchoApp:
    """Report the published id, but only once ``expected`` requests are inside.

    Holding every request until they have all arrived is what makes the
    overlap in the uniqueness test deterministic.
    """

    def __init__(self, expected: int):
        self.expected = expected
        self.seen: list[str] = []
        self._all_inside = asyncio.Event()

    async def __call__(self, scope, receive, send):
        request_id = (scope.get("state") or {}).get(REQUEST_ID_STATE_KEY)
        self.seen.append(request_id)
        if len(self.seen) >= self.expected:
            self._all_inside.set()
        await self._all_inside.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {"type": "http.response.body", "body": str(request_id).encode("ascii")}
        )


async def _call_middleware(middleware, inbound_id: str) -> tuple[str, str]:
    """Drive the middleware directly; returns (response header, scope state)."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"x-request-id", inbound_id.encode("ascii"))],
        "state": {},
    }
    messages: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await middleware(scope, receive, send)
    sent_headers = dict(messages[0]["headers"])
    return (
        sent_headers[b"x-request-id"].decode("ascii"),
        scope["state"][REQUEST_ID_STATE_KEY],
    )


class TestRequestIdUniqueness:
    """Two in-flight requests must never share an id."""

    async def test_in_flight_requests_sharing_an_id_get_distinct_ids(self):
        app = _BlockingEchoApp(expected=2)
        middleware = RequestIdMiddleware(app)

        first, second = await asyncio.gather(
            _call_middleware(middleware, "shared-id"),
            _call_middleware(middleware, "shared-id"),
        )
        header_a, state_a = first
        header_b, state_b = second

        assert state_a == header_a
        assert state_b == header_b
        assert len({header_a, header_b}) == 2
        # The first arrival keeps the client's id; the second is replaced.
        assert "shared-id" in {header_a, header_b}

    async def test_an_id_is_reusable_once_the_response_finished(self):
        middleware = RequestIdMiddleware(_EchoRequestIdApp())

        first, _ = await _call_middleware(middleware, "reused-id")
        second, _ = await _call_middleware(middleware, "reused-id")

        assert first == second == "reused-id"


class TestRequestCorrelationId:
    def test_reads_the_published_id(self):
        request = MagicMock()
        request.scope = {"state": {REQUEST_ID_STATE_KEY: "published-id"}}
        assert _request_correlation_id(request) == "published-id"

    def test_mints_when_the_middleware_did_not_run(self):
        request = MagicMock()
        request.scope = {}
        assert valid_request_id(_request_correlation_id(request)) is not None


class TestRequestAbortId:
    def test_none_without_targeted_abort_support(self):
        engine = MagicMock()
        engine.supports_request_scoped_abort = False
        assert _request_abort_id(engine, "client-trace-1") is None

    def test_none_when_abort_request_is_not_callable(self):
        engine = MagicMock()
        engine.supports_request_scoped_abort = True
        engine.abort_request = None
        assert _request_abort_id(engine, "client-trace-1") is None

    def test_prefers_the_correlation_id(self):
        engine = MagicMock()
        engine.supports_request_scoped_abort = True
        assert _request_abort_id(engine, "client-trace-1") == "client-trace-1"

    def test_mints_a_transport_id_without_one(self):
        engine = MagicMock()
        engine.supports_request_scoped_abort = True
        assert _request_abort_id(engine).startswith("transport-")
