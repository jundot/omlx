# SPDX-License-Identifier: Apache-2.0
"""Regression contract for browser-to-inference stream cancellation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from omlx import server

CHAT_TEMPLATE = (
    Path(__file__).parents[1] / "omlx" / "admin" / "templates" / "chat.html"
)


def test_stop_cancels_the_open_response_reader_before_aborting_fetch():
    source = CHAT_TEMPLATE.read_text(encoding="utf-8")
    helper_start = source.index("cancelStreamTransport(stream)")
    helper_end = source.index("stopChatStreaming(chatId)", helper_start)
    helper = source[helper_start:helper_end]

    assert "reader.cancel('Stopped by user')" in helper
    assert helper.index("reader.cancel('Stopped by user')") < helper.index(
        "stream.abortController?.abort()"
    )


def test_stream_exposes_its_response_reader_to_every_stop_path():
    source = CHAT_TEMPLATE.read_text(encoding="utf-8")

    assert "stream.responseReader = reader;" in source
    assert "this.cancelStreamTransport(stream);" in source
    assert (
        "this.cancelStreamTransport(this.getStreamSession(chatId, false));" in source
    )


@pytest.mark.asyncio
async def test_closing_public_chat_stream_awaits_engine_request_close():
    """The ASGI body close must reach the engine before ``aclose`` returns.

    An ``async for`` nested inside another async generator is otherwise
    finalized later by the event loop. For a distributed engine that nested
    generator owns the private rank-zero response, so deferred finalization
    leaves decoding live after the browser's Stop button has returned.
    """

    closed = False

    class Engine:
        tokenizer = None

        async def stream_chat(self, **_kwargs):
            nonlocal closed
            try:
                while True:
                    yield SimpleNamespace(
                        new_text="token",
                        tool_calls=None,
                        finish_reason=None,
                        finished=False,
                        prompt_tokens=1,
                        completion_tokens=1,
                        cached_tokens=0,
                    )
            finally:
                closed = True

    request = SimpleNamespace(model="cluster-model", response_format=None)
    response = server.stream_chat_completion(
        Engine(),
        [{"role": "user", "content": "hello"}],
        request,
    )
    keepalive = server._with_sse_keepalive(response, keepalive_chunk=None)
    lease = server._LLMEngineLease()
    body = server._release_after_stream(keepalive, lease)

    await body.__anext__()  # assistant role
    await body.__anext__()  # first generated token
    await body.aclose()

    assert closed is True
    assert lease.released is True


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ("anthropic", "responses"))
async def test_every_public_chat_protocol_awaits_engine_close(protocol):
    """Anthropic and Responses streams share the same cancellation contract."""

    from omlx.api.anthropic_models import MessagesRequest
    from omlx.api.responses_models import ResponsesRequest

    started = False
    closed = False

    class Engine:
        tokenizer = None

        async def stream_chat(self, **_kwargs):
            nonlocal started, closed
            started = True
            try:
                while True:
                    yield SimpleNamespace(
                        new_text="token",
                        tool_calls=None,
                        finish_reason=None,
                        finished=False,
                        prompt_tokens=1,
                        completion_tokens=1,
                        cached_tokens=0,
                    )
            finally:
                closed = True

    engine = Engine()
    messages = [{"role": "user", "content": "hello"}]
    if protocol == "anthropic":
        request = MessagesRequest(
            model="cluster-model",
            max_tokens=16,
            messages=messages,
            stream=True,
        )
        response = server.stream_anthropic_messages(engine, messages, request)
    else:
        request = ResponsesRequest(
            model="cluster-model",
            input="hello",
            stream=True,
        )
        response = server.stream_responses_api(
            engine,
            messages,
            request,
            store_response=False,
        )

    keepalive = server._with_sse_keepalive(response, keepalive_chunk=None)
    lease = server._LLMEngineLease()
    body = server._release_after_stream(keepalive, lease)
    for _ in range(8):
        await body.__anext__()
        if started:
            break

    assert started is True
    await body.aclose()
    assert closed is True
    assert lease.released is True
