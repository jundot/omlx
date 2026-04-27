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
