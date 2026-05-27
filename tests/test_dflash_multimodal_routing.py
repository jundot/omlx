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
