# SPDX-License-Identifier: Apache-2.0
"""Tests for _with_sse_keepalive SSE wrapper."""

import asyncio
import json

import pytest

from omlx.server import _with_sse_keepalive


async def _collect(gen):
    """Collect all items from an async generator."""
    items = []
    async for item in gen:
        items.append(item)
    return items


class TestSSEKeepaliveExceptionHandling:
    """Tests for exception handling in _with_sse_keepalive."""

    @pytest.mark.asyncio
    async def test_normal_generator_passes_through(self):
        """Normal generator items should pass through unchanged."""

        async def gen():
            yield "data: chunk1\n\n"
            yield "data: chunk2\n\n"

        items = await _collect(_with_sse_keepalive(gen()))
        # First item is always the initial keepalive
        assert items[0] == ": keep-alive\n\n"
        assert "data: chunk1\n\n" in items
        assert "data: chunk2\n\n" in items

    @pytest.mark.asyncio
    async def test_generator_exception_yields_error_sse(self):
        """When inner generator raises, keepalive wrapper should yield
        error SSE data and [DONE] instead of propagating the exception."""

        async def gen():
            yield "data: first_chunk\n\n"
            raise RuntimeError("Memory limit exceeded during prefill")

        items = await _collect(_with_sse_keepalive(gen()))

        # Should contain initial keepalive + first chunk + error + done
        assert items[0] == ": keep-alive\n\n"
        assert "data: first_chunk\n\n" in items

        # Find the error SSE event
        error_items = [i for i in items if i.startswith("data: {")]
        assert len(error_items) == 1
        error_data = json.loads(error_items[0].removeprefix("data: ").strip())
        assert "error" in error_data
        assert "Memory limit exceeded during prefill" in error_data["error"]["message"]
        assert error_data["error"]["type"] == "server_error"

        # Must end with [DONE]
        assert "data: [DONE]\n\n" in items

    @pytest.mark.asyncio
    async def test_generator_exception_before_any_yield(self):
        """Exception on first iteration should still produce error SSE."""

        async def gen():
            if True:
                raise ValueError("Block allocation failed")
            yield  # unreachable, but makes this an async generator

        items = await _collect(_with_sse_keepalive(gen()))

        assert items[0] == ": keep-alive\n\n"

        error_items = [i for i in items if i.startswith("data: {")]
        assert len(error_items) == 1
        error_data = json.loads(error_items[0].removeprefix("data: ").strip())
        assert "Block allocation failed" in error_data["error"]["message"]
        assert "data: [DONE]\n\n" in items

    @pytest.mark.asyncio
    async def test_empty_generator_completes_cleanly(self):
        """Empty generator should complete without errors."""

        async def gen():
            return
            yield  # make it an async generator

        items = await _collect(_with_sse_keepalive(gen()))
        assert items[0] == ": keep-alive\n\n"
        # No error items
        error_items = [i for i in items if i.startswith("data: {")]
        assert len(error_items) == 0


class TestKeepaliveChunkFormats:
    """Tests for protocol-aware keepalive chunk emission."""

    @pytest.mark.asyncio
    async def test_chat_chunk_format_is_valid_chat_completion_chunk(self):
        from omlx.server import _KEEPALIVE_CHAT_CHUNK

        async def gen():
            yield "data: real\n\n"

        items = await _collect(
            _with_sse_keepalive(gen(), keepalive_chunk=_KEEPALIVE_CHAT_CHUNK)
        )
        assert items[0] == _KEEPALIVE_CHAT_CHUNK
        body = items[0].removeprefix("data: ").strip()
        payload = json.loads(body)
        assert payload["object"] == "chat.completion.chunk"
        assert payload["choices"][0]["delta"]["role"] == "assistant"
        assert payload["choices"][0]["delta"]["content"] == ""
        assert payload["choices"][0]["finish_reason"] is None

    @pytest.mark.asyncio
    async def test_completion_chunk_format_is_valid_text_completion(self):
        from omlx.server import _KEEPALIVE_COMPLETION_CHUNK

        async def gen():
            yield "data: real\n\n"

        items = await _collect(
            _with_sse_keepalive(gen(), keepalive_chunk=_KEEPALIVE_COMPLETION_CHUNK)
        )
        body = items[0].removeprefix("data: ").strip()
        payload = json.loads(body)
        assert payload["object"] == "text_completion"
        assert payload["choices"][0]["text"] == ""
        assert payload["choices"][0]["finish_reason"] is None

    @pytest.mark.asyncio
    async def test_anthropic_ping_event_format(self):
        from omlx.server import _KEEPALIVE_ANTHROPIC_PING

        async def gen():
            yield "event: message_start\ndata: {}\n\n"

        items = await _collect(
            _with_sse_keepalive(gen(), keepalive_chunk=_KEEPALIVE_ANTHROPIC_PING)
        )
        assert items[0].startswith("event: ping\n")
        assert 'data: {"type":"ping"}' in items[0]

    @pytest.mark.asyncio
    async def test_keepalive_off_skips_emission(self):
        async def gen():
            yield "data: real\n\n"

        items = await _collect(_with_sse_keepalive(gen(), keepalive_chunk=None))
        # No keepalive frame, just the real chunk passed through
        assert items == ["data: real\n\n"]


class TestChatKeepaliveSharesStreamId:
    """The chunk-form chat keepalive must reuse the stream's completion id.

    Strict OpenAI stream accumulators key on a single per-stream ``id`` and
    drop chunks whose id differs from the first. A keepalive carrying the
    sentinel ``chatcmpl-keepalive`` id therefore causes them to discard the
    real tool_calls/usage chunks. _chat_keepalive_chunk reuses the stream id so
    the frame is a true no-op for those clients.
    """

    def test_frame_uses_given_response_id(self):
        from omlx.server import _chat_keepalive_chunk

        frame = _chat_keepalive_chunk("chatcmpl-abc123")
        assert frame.startswith("data: ")
        assert frame.endswith("\n\n")
        payload = json.loads(frame.removeprefix("data: ").strip())
        assert payload["id"] == "chatcmpl-abc123"
        assert payload["object"] == "chat.completion.chunk"
        assert payload["choices"][0]["delta"]["role"] == "assistant"
        assert payload["choices"][0]["delta"]["content"] == ""
        assert payload["choices"][0]["finish_reason"] is None

    def test_frame_does_not_use_sentinel_id(self):
        from omlx.server import _chat_keepalive_chunk

        payload = json.loads(
            _chat_keepalive_chunk("chatcmpl-real").removeprefix("data: ").strip()
        )
        assert payload["id"] != "chatcmpl-keepalive"


class TestChatKeepaliveCarriesRole:
    """Every chat keepalive delta must carry ``role: assistant``.

    The chunk-form keepalive is the first SSE event of every stream, and some
    accumulators type the whole stream from the first chunk's role.
    LangChain.js builds a generic ChatMessageChunk when the role is absent and
    then discards all tool_call_chunks when the real AI chunks merge into it,
    so streamed tool calls are silently lost (#2074, n8n AI Agent workflows).
    """

    def _first_chunk_role(self, frame: str):
        # Mirror the accumulator rule: the stream's type is decided by the
        # first chunk's delta.role alone.
        payload = json.loads(frame.removeprefix("data: ").strip())
        return payload["choices"][0]["delta"].get("role")

    def test_static_sentinel_frame_carries_assistant_role(self):
        from omlx.server import _KEEPALIVE_CHAT_CHUNK

        assert self._first_chunk_role(_KEEPALIVE_CHAT_CHUNK) == "assistant"

    def test_id_sharing_frame_carries_assistant_role(self):
        from omlx.server import _chat_keepalive_chunk

        assert self._first_chunk_role(_chat_keepalive_chunk("chatcmpl-x")) == "assistant"


class TestResolveKeepalive:
    """Tests for _resolve_keepalive helper that maps settings to wire format."""

    def _set_mode(self, mode: str):
        from omlx.server import _server_state

        if _server_state.global_settings is None:
            pytest.skip("global_settings not initialized")
        _server_state.global_settings.server.sse_keepalive_mode = mode

    def test_chunk_mode_returns_protocol_specific_frames(self):
        from omlx.server import (
            _KEEPALIVE_ANTHROPIC_PING,
            _KEEPALIVE_CHAT_CHUNK,
            _KEEPALIVE_COMPLETION_CHUNK,
            _resolve_keepalive,
            _server_state,
        )

        if _server_state.global_settings is None:
            pytest.skip("global_settings not initialized")
        original = _server_state.global_settings.server.sse_keepalive_mode
        try:
            self._set_mode("chunk")
            assert _resolve_keepalive("openai_chat") == _KEEPALIVE_CHAT_CHUNK
            assert _resolve_keepalive("openai_completion") == _KEEPALIVE_COMPLETION_CHUNK
            assert _resolve_keepalive("anthropic") == _KEEPALIVE_ANTHROPIC_PING
            # Responses API has no official ping; chunk mode disables keepalive
            assert _resolve_keepalive("openai_responses") is None
        finally:
            _server_state.global_settings.server.sse_keepalive_mode = original

    def test_comment_mode_returns_legacy_comment(self):
        from omlx.server import _KEEPALIVE_COMMENT, _resolve_keepalive, _server_state

        if _server_state.global_settings is None:
            pytest.skip("global_settings not initialized")
        original = _server_state.global_settings.server.sse_keepalive_mode
        try:
            self._set_mode("comment")
            for protocol in ("openai_chat", "openai_completion", "anthropic", "openai_responses"):
                assert _resolve_keepalive(protocol) == _KEEPALIVE_COMMENT
        finally:
            _server_state.global_settings.server.sse_keepalive_mode = original

    def test_off_mode_returns_none(self):
        from omlx.server import _resolve_keepalive, _server_state

        if _server_state.global_settings is None:
            pytest.skip("global_settings not initialized")
        original = _server_state.global_settings.server.sse_keepalive_mode
        try:
            self._set_mode("off")
            for protocol in ("openai_chat", "openai_completion", "anthropic", "openai_responses"):
                assert _resolve_keepalive(protocol) is None
        finally:
            _server_state.global_settings.server.sse_keepalive_mode = original
