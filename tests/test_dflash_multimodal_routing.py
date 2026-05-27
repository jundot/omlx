# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dflash + embedded-VLM multimodal request routing.

Path A keeps a VLMBatchedEngine permanently spun up next to the dflash
drafter. Multimodal requests (image_url / audio content parts) need to
land on that embedded VLM rather than dflash's text-only template flow,
or the image is silently dropped. This is flyto's answer to upstream
#1344 — see docs/upstream-sync.md for the design contrast (upstream uses
a lazy _fallback_engine swap; flyto routes at the dflash.chat boundary).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from omlx.engine.dflash import DFlashEngine


def _make_dflash(embedded_vlm: Any = None) -> DFlashEngine:
    """Build a DFlashEngine without running __init__ (the constructor pulls
    in dflash_mlx + model loading, which is way more than these tests need).
    """
    engine = DFlashEngine.__new__(DFlashEngine)
    engine._embedded_vlm = embedded_vlm
    engine._loaded = True
    return engine


class TestSupportsMultimodalFallback:
    def test_false_when_no_embedded_engine(self):
        engine = _make_dflash(embedded_vlm=None)
        assert engine.supports_multimodal_fallback is False

    def test_true_when_embedded_engine_is_vlm(self):
        from omlx.engine.vlm import VLMBatchedEngine

        vlm = VLMBatchedEngine.__new__(VLMBatchedEngine)
        engine = _make_dflash(embedded_vlm=vlm)
        assert engine.supports_multimodal_fallback is True

    def test_false_when_embedded_engine_is_text_only(self):
        from omlx.engine.batched import BatchedEngine

        text_engine = BatchedEngine.__new__(BatchedEngine)
        engine = _make_dflash(embedded_vlm=text_engine)
        assert engine.supports_multimodal_fallback is False


class TestHasMultimodalContent:
    def test_string_content_is_text_only(self):
        messages = [{"role": "user", "content": "hello"}]
        assert DFlashEngine._has_multimodal_content(messages) is False

    def test_array_of_text_parts_is_text_only(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ]
        assert DFlashEngine._has_multimodal_content(messages) is False

    def test_image_url_detected(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what do you see"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
            }
        ]
        assert DFlashEngine._has_multimodal_content(messages) is True

    def test_anthropic_image_type_detected(self):
        # Anthropic uses {"type": "image", "source": {...}}.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "..."}},
                ],
            }
        ]
        assert DFlashEngine._has_multimodal_content(messages) is True

    def test_audio_part_detected(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}},
                ],
            }
        ]
        assert DFlashEngine._has_multimodal_content(messages) is True

    def test_mixed_string_and_array_messages(self):
        messages = [
            {"role": "system", "content": "be helpful"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            },
        ]
        assert DFlashEngine._has_multimodal_content(messages) is True

    def test_empty_messages(self):
        assert DFlashEngine._has_multimodal_content([]) is False


class TestChatRoutesMultimodalToEmbeddedVLM:
    @pytest.mark.asyncio
    async def test_image_request_forwarded_to_embedded_vlm(self):
        from omlx.engine.vlm import VLMBatchedEngine

        vlm = VLMBatchedEngine.__new__(VLMBatchedEngine)
        sentinel_output = MagicMock(name="GenerationOutput")
        vlm.chat = AsyncMock(return_value=sentinel_output)

        engine = _make_dflash(embedded_vlm=vlm)
        # Sentinel: if chat() falls through to the text path, _apply_chat_template
        # gets called. Setting it to a Mock that explodes lets us catch that.
        engine._apply_chat_template = MagicMock(
            side_effect=AssertionError("text path must not be taken for image input")
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
            }
        ]
        result = await engine.chat(messages=messages, max_tokens=32)

        assert result is sentinel_output
        vlm.chat.assert_awaited_once()
        # Verify messages passed through as-is (image parts preserved).
        call_kwargs = vlm.chat.await_args.kwargs
        assert call_kwargs["messages"] == messages
        assert call_kwargs["max_tokens"] == 32

    @pytest.mark.asyncio
    async def test_text_request_uses_dflash_path(self):
        from omlx.engine.vlm import VLMBatchedEngine

        vlm = VLMBatchedEngine.__new__(VLMBatchedEngine)
        vlm.chat = AsyncMock(
            side_effect=AssertionError("VLM path must not be taken for text-only input")
        )

        engine = _make_dflash(embedded_vlm=vlm)
        engine._apply_chat_template = MagicMock(return_value="prompt-token-text")
        sentinel_output = MagicMock(name="GenerationOutput")
        engine.generate = AsyncMock(return_value=sentinel_output)

        messages = [{"role": "user", "content": "hi"}]
        result = await engine.chat(messages=messages, max_tokens=16)

        assert result is sentinel_output
        engine._apply_chat_template.assert_called_once()
        engine.generate.assert_awaited_once()
        vlm.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_image_request_without_fallback_uses_dflash_path(self):
        """If there's no embedded VLM (or it's a text-only fallback), the
        multimodal branch must not fire — we have nothing to forward to."""
        engine = _make_dflash(embedded_vlm=None)
        engine._apply_chat_template = MagicMock(return_value="prompt-token-text")
        sentinel_output = MagicMock(name="GenerationOutput")
        engine.generate = AsyncMock(return_value=sentinel_output)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
        result = await engine.chat(messages=messages, max_tokens=16)
        assert result is sentinel_output
        engine._apply_chat_template.assert_called_once()


class TestStreamChatRoutesMultimodalToEmbeddedVLM:
    @pytest.mark.asyncio
    async def test_image_request_streamed_through_embedded_vlm(self):
        from omlx.engine.vlm import VLMBatchedEngine

        vlm = VLMBatchedEngine.__new__(VLMBatchedEngine)
        chunks = [MagicMock(name=f"chunk-{i}") for i in range(3)]

        async def fake_stream(*_args, **_kwargs):
            for c in chunks:
                yield c

        vlm.stream_chat = MagicMock(side_effect=lambda **kw: fake_stream(**kw))

        engine = _make_dflash(embedded_vlm=vlm)
        engine._apply_chat_template = MagicMock(
            side_effect=AssertionError("text path must not be taken for image input")
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
        received = []
        async for chunk in engine.stream_chat(messages=messages, max_tokens=8):
            received.append(chunk)

        assert received == chunks
        vlm.stream_chat.assert_called_once()
        assert vlm.stream_chat.call_args.kwargs["messages"] == messages

    @pytest.mark.asyncio
    async def test_text_request_streams_dflash(self):
        from omlx.engine.vlm import VLMBatchedEngine

        vlm = VLMBatchedEngine.__new__(VLMBatchedEngine)
        vlm.stream_chat = MagicMock(
            side_effect=AssertionError("VLM path must not be taken for text-only input")
        )

        engine = _make_dflash(embedded_vlm=vlm)
        engine._apply_chat_template = MagicMock(return_value="prompt-token-text")
        chunks = [MagicMock(name=f"chunk-{i}") for i in range(2)]

        async def fake_stream_generate(*_args, **_kwargs):
            for c in chunks:
                yield c

        engine.stream_generate = MagicMock(
            side_effect=lambda **kw: fake_stream_generate(**kw)
        )

        messages = [{"role": "user", "content": "hi"}]
        received = []
        async for chunk in engine.stream_chat(messages=messages, max_tokens=8):
            received.append(chunk)

        assert received == chunks
        engine.stream_generate.assert_called_once()
        vlm.stream_chat.assert_not_called()


class TestGrammarCompilerForward:
    """DFlashEngine must forward grammar_compiler to its embedded target.

    Regression: server.py reads ``getattr(engine, 'grammar_compiler', None)``
    when handling structured-output requests. Before this forward, a
    DFlashEngine-loaded model would always fall through to BaseEngine's
    None default and the request would be rejected with a misleading
    'xgrammar required' message even when xgrammar was installed and
    a plain BatchedEngine load of the same model would have worked.
    """

    def test_none_when_no_embedded_engine(self):
        engine = _make_dflash(embedded_vlm=None)
        assert engine.grammar_compiler is None

    def test_forwards_to_embedded_engine_when_set(self):
        sentinel = object()
        vlm = MagicMock()
        vlm.grammar_compiler = sentinel
        engine = _make_dflash(embedded_vlm=vlm)
        assert engine.grammar_compiler is sentinel

    def test_forwards_none_from_embedded_engine(self):
        vlm = MagicMock()
        vlm.grammar_compiler = None
        engine = _make_dflash(embedded_vlm=vlm)
        assert engine.grammar_compiler is None


class TestPrefixCacheEnabledForward:
    """DFlashEngine must forward prefix_cache_enabled to its embedded target.

    Regression: server.py threads engine.prefix_cache_enabled into the
    Anthropic-style usage accounting (api/anthropic_utils.py splits
    prompt tokens into cache_creation / cache_read on it). Before the
    forward, DFlash-wrapped models silently reported zero cache hits
    even when the embedded engine was hitting its prefix cache.
    """

    def test_false_when_no_embedded_engine(self):
        engine = _make_dflash(embedded_vlm=None)
        assert engine.prefix_cache_enabled is False

    def test_true_when_embedded_engine_reports_true(self):
        vlm = MagicMock()
        vlm.prefix_cache_enabled = True
        engine = _make_dflash(embedded_vlm=vlm)
        assert engine.prefix_cache_enabled is True

    def test_false_when_embedded_engine_reports_false(self):
        vlm = MagicMock()
        vlm.prefix_cache_enabled = False
        engine = _make_dflash(embedded_vlm=vlm)
        assert engine.prefix_cache_enabled is False


class TestCachedTokensSurfacedFromPrefixFlow:
    """DFlash decode path must surface dflash_mlx PrefixCacheFlow hit count
    into GenerationOutput.cached_tokens.

    Regression: upstream #1441. Before this fix the dflash decode path
    constructed GenerationOutput without cached_tokens, so even when
    dflash_mlx's RuntimeCacheManager.lookup matched a prefix the OpenAI
    usage.prompt_tokens_details.cached_tokens reported 0. Multi-turn
    chats showed full-cold prefill on every turn while DFlash was on.

    The fix wires ``prefix_flow.hit_tokens`` through to GenerationOutput
    on both the sync ``generate()`` and the streaming ``stream_generate``
    paths.
    """

    @pytest.mark.asyncio
    async def test_stream_generate_yields_cached_tokens_from_metrics(self):
        """The streaming path threads metrics['cached_tokens'] into every
        yielded GenerationOutput so server.py's `last_output.cached_tokens`
        sees the right value regardless of which chunk it samples.
        """
        import asyncio as _asyncio

        from omlx.engine.base import GenerationOutput

        # Build the minimum DFlashEngine state the streaming generator
        # touches before its first ``await queue.get()`` --- everything
        # else can be patched. We're testing the queue-to-yield wiring,
        # not the dflash_mlx executor.
        engine = DFlashEngine.__new__(DFlashEngine)
        engine._loaded = True
        engine._embedded_vlm = MagicMock()
        engine._tokenizer_obj = MagicMock()
        engine._tokenizer_obj.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        engine._concurrent_sem = None
        engine._active_count = 0
        engine._detect_needs_think_prefix = MagicMock(return_value=False)

        # _route returns "dflash" so the call doesn't bounce to
        # _embedded_vlm.stream_generate.
        engine._route = MagicMock(return_value="dflash")
        engine._ensure_drafter_loaded = AsyncMock()

        # Replace the executor + run function so the queue gets fed
        # deterministically: 2 token chunks then a summary chunk with
        # cached_tokens=42 in metrics.
        async def _fake_run_streaming_via_queue(
            prompt_tokens, max_tokens, temperature, queue, loop, stop_event,
        ):
            await queue.put(("hello", [10], False, None))
            await queue.put((" world", [20], False, None))
            await queue.put(
                ("", [], True, {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "acceptance_ratio": 0.9,
                    "cycles_completed": 1,
                    "cached_tokens": 42,
                }),
            )

        def _fake_run_in_executor(_executor, fn, *args):
            # Drive the fake feeder on the event loop instead of a thread,
            # so the queue.put calls actually land.
            return _asyncio.ensure_future(
                _fake_run_streaming_via_queue(*args)
            )

        loop = _asyncio.get_event_loop()
        loop.run_in_executor = _fake_run_in_executor

        outputs: list[GenerationOutput] = []
        async for out in engine.stream_generate(prompt="hi", max_tokens=4):
            outputs.append(out)

        # 3 yields (2 token chunks + 1 final). Every chunk after the final
        # SummaryEvent metric carries cached_tokens=42. The first chunk
        # comes before metrics so it stays 0; once the metric arrives
        # subsequent chunks pick it up. We require the final yielded
        # chunk to carry the value, since that is what server.py reads
        # for usage accounting.
        assert len(outputs) == 3
        assert outputs[-1].cached_tokens == 42
        assert outputs[-1].finished is True
