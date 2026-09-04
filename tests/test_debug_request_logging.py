# SPDX-License-Identifier: Apache-2.0
"""Tests for the trace-level request body logging middleware."""

import pytest

from omlx.server import DebugRequestLoggingMiddleware, _is_textual_body

TRACE = 5


class TestIsTextualBody:
    def test_textual_types(self):
        assert _is_textual_body("application/json")
        assert _is_textual_body("application/json; charset=utf-8")
        assert _is_textual_body("application/problem+json")
        assert _is_textual_body("application/x-www-form-urlencoded")
        assert _is_textual_body("text/plain")

    def test_binary_types(self):
        assert not _is_textual_body("multipart/form-data; boundary=xyz")
        assert not _is_textual_body("audio/wav")
        assert not _is_textual_body("video/mp4")
        assert not _is_textual_body("application/octet-stream")
        assert not _is_textual_body("")


def _make_scope(content_type: str) -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/audio/transcriptions",
        "headers": [
            (b"content-type", content_type.encode("latin-1")),
            (b"content-length", b"12345"),
        ],
    }


def _make_receive(body: bytes):
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return messages.pop(0)

    return receive


async def _noop_send(message):
    pass


class TestDebugRequestLoggingMiddleware:
    @pytest.mark.asyncio
    async def test_binary_body_not_dumped(self, caplog):
        received = []

        async def inner_app(scope, receive, send):
            message = await receive()
            received.append(message["body"])

        binary = b"\x00\x01RIFFBINARYJUNK\xff\xfe" * 4
        middleware = DebugRequestLoggingMiddleware(inner_app)
        caplog.set_level(TRACE, logger="omlx.server")

        await middleware(
            _make_scope("multipart/form-data; boundary=xyz"),
            _make_receive(binary),
            _noop_send,
        )

        # Body reaches the app untouched (streamed, not re-buffered)
        assert received == [binary]
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert "bytes omitted" in logged
        assert "multipart/form-data" in logged
        assert "RIFFBINARYJUNK" not in logged

    @pytest.mark.asyncio
    async def test_json_body_still_logged(self, caplog):
        received = []

        async def inner_app(scope, receive, send):
            message = await receive()
            received.append(message["body"])

        body = b'{"model": "whisper", "stream": true}'
        middleware = DebugRequestLoggingMiddleware(inner_app)
        caplog.set_level(TRACE, logger="omlx.server")

        await middleware(
            _make_scope("application/json"),
            _make_receive(body),
            _noop_send,
        )

        assert received == [body]
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert '"model": "whisper"' in logged
