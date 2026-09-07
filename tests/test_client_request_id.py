# SPDX-License-Identifier: Apache-2.0
"""Tests for caller request-id correlation on the activity API."""

import asyncio

from omlx.request import (
    CLIENT_REQUEST_ID_MAX_LEN,
    Request,
    client_request_id_from_headers,
    current_client_request_id,
)


def test_header_is_extracted():
    headers = [(b"host", b"localhost"), (b"x-request-id", b"caller-abc")]
    assert client_request_id_from_headers(headers) == "caller-abc"


def test_missing_header_is_none():
    assert client_request_id_from_headers([(b"host", b"localhost")]) is None
    assert client_request_id_from_headers(()) is None


def test_blank_header_counts_as_absent():
    assert client_request_id_from_headers([(b"x-request-id", b"   ")]) is None
    assert client_request_id_from_headers([(b"x-request-id", b"")]) is None


def test_header_is_trimmed_and_bounded():
    value = b"  " + b"a" * (CLIENT_REQUEST_ID_MAX_LEN + 50) + b"  "
    result = client_request_id_from_headers([(b"x-request-id", value)])
    assert result == "a" * CLIENT_REQUEST_ID_MAX_LEN


def test_request_has_no_caller_id_by_default():
    """A request that sent no header is untracked, exactly as before."""
    assert Request.__dataclass_fields__["client_request_id"].default is None


def test_context_var_defaults_to_none():
    assert current_client_request_id.get() is None


def test_context_var_is_isolated_between_tasks():
    """Concurrent requests must not observe each other's caller id."""
    observed: dict[str, str | None] = {}

    async def handler(name: str, value: str) -> None:
        current_client_request_id.set(value)
        await asyncio.sleep(0)  # force interleaving
        observed[name] = current_client_request_id.get()

    async def main() -> None:
        await asyncio.gather(handler("a", "caller-a"), handler("b", "caller-b"))

    asyncio.run(main())
    assert observed == {"a": "caller-a", "b": "caller-b"}


async def test_middleware_exposes_the_header_while_the_body_is_produced():
    """The id must still be readable when a streaming body is iterated.

    That is when the engine registers the request, and it happens after the
    endpoint has returned; a middleware that restores the context on return
    would lose the value before it is needed.
    """
    from omlx.server import ClientRequestIdMiddleware

    seen: list[str | None] = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        seen.append(current_client_request_id.get())
        await send({"type": "http.response.body", "body": b"x", "more_body": False})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    middleware = ClientRequestIdMiddleware(app)
    await middleware(
        {"type": "http", "headers": [(b"host", b"x"), (b"x-request-id", b"caller-1")]},
        receive,
        send,
    )
    await middleware({"type": "http", "headers": [(b"host", b"x")]}, receive, send)
    assert seen == ["caller-1", None]


async def test_middleware_ignores_non_http_scopes():
    from omlx.server import ClientRequestIdMiddleware

    called: list[str] = []

    async def app(scope, receive, send):
        called.append(scope["type"])

    await ClientRequestIdMiddleware(app)({"type": "lifespan"}, None, None)
    assert called == ["lifespan"]
    assert current_client_request_id.get() is None
