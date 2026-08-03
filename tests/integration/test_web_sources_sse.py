# SPDX-License-Identifier: Apache-2.0
"""Tests that stream_chat_completion emits a web_sources SSE event.

When web search is enabled the chat backend injects retrieved source links
into the stream as a custom `event: web_sources` frame (captured by the chat
UI to render a citations footer). These tests verify the backend actually
emits that frame with the correct payload -- independent of any GUI, model,
or API key.

Written as synchronous tests that drive the async generator via asyncio.run
so they run under the bare pytest available in the oMLX app environment
(pytest-asyncio is not bundled there).
"""

import asyncio
import json

from omlx.api.openai_models import ChatCompletionRequest, Message


class _MockTokenizer:
    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if tokenize:
            return [1, 2, 3]
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def encode(self, text):
        return text.split()


class _MockEngine:
    """Minimal engine stub sufficient for stream_chat_completion."""

    def __init__(self, model_name="test-model"):
        self._model_name = model_name
        self._tokenizer = _MockTokenizer()
        self._model_type = "llama"

    @property
    def model_name(self):
        return self._model_name

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_type(self):
        return self._model_type

    @property
    def prefix_cache_enabled(self):
        return False

    async def start(self):
        pass

    async def stop(self):
        pass

    def count_chat_tokens(self, messages, tools=None, chat_template_kwargs=None, **kwargs):
        return 3

    async def chat(self, messages, **kwargs):
        raise NotImplementedError

    async def stream_chat(self, messages, **kwargs):
        from omlx.engine.base import GenerationOutput

        yield GenerationOutput(
            text="Hi there", new_text=" there", completion_tokens=2,
            finished=True, finish_reason="stop", tool_calls=None,
        )

    def get_stats(self):
        return {}

    def get_cache_stats(self):
        return None


SAMPLE_SOURCES = [
    {"title": "Tokyo Weather - Adachi", "url": "https://example.com/adachi", "snippet": "..."},
    {"title": "Tenki.jp Adachi", "url": "https://tenki.jp/adachi", "snippet": "..."},
]


def test_web_sources_event_emitted_when_sources_present():
    from omlx.server import stream_chat_completion

    async def run():
        engine = _MockEngine()
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
        )
        messages = [{"role": "user", "content": "Hi"}]
        events = []
        async for event in stream_chat_completion(
            engine, messages, request,
            web_sources=SAMPLE_SOURCES,
            max_tokens=256, temperature=0.7, top_p=0.9, top_k=40,
        ):
            events.append(event)
        return events

    events = asyncio.run(run())

    web_frames = [e for e in events if e.startswith("event: web_sources")]
    assert web_frames, "expected an `event: web_sources` frame in the stream"

    frame = web_frames[0]
    data_line = [l for l in frame.splitlines() if l.startswith("data: ")][0]
    payload = json.loads(data_line[6:])
    assert payload["sources"] == SAMPLE_SOURCES

    done_idx = events.index("data: [DONE]\n\n")
    assert events.index(web_frames[0]) < done_idx


def test_no_web_sources_event_when_sources_empty():
    from omlx.server import stream_chat_completion

    async def run():
        engine = _MockEngine()
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
        )
        messages = [{"role": "user", "content": "Hi"}]
        events = []
        async for event in stream_chat_completion(
            engine, messages, request,
            web_sources=None,
            max_tokens=256, temperature=0.7, top_p=0.9, top_k=40,
        ):
            events.append(event)
        return events

    events = asyncio.run(run())
    assert not any(e.startswith("event: web_sources") for e in events)
