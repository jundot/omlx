# SPDX-License-Identifier: Apache-2.0
"""Tests for DFlash multimodal VLM fallback (issue #1342).

Before this fix:
- DFlash with VLM fallback was treated as a plain text engine by server.py
- Images were silently stripped by extract_text_content() before reaching the engine
- chat()/stream_chat() had no multimodal detection — images that survived
  extraction would still be flattened by _apply_chat_template()

After this fix:
- server.py detects DFlash engines with VLM fallback via supports_multimodal_fallback
- Image content is preserved through extract_multimodal_content()
- chat()/stream_chat() detect multimodal messages and trigger VLM fallback
  BEFORE applying text-only chat template
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine.dflash import DFlashEngine

# -- Helpers ------------------------------------------------------------------

def _text_only_messages():
    return [{"role": "user", "content": "What is 2+2?"}]


def _image_url_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    ]


def _image_type_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see?"},
                {"type": "image", "source": {"type": "base64", "data": "abc"}},
            ],
        }
    ]


def _input_image_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze"},
                {"type": "input_image", "image_url": {"url": "data:image/jpeg;base64,xyz"}},
            ],
        }
    ]


def _mixed_history_messages():
    """Image in earlier turn, text-only in latest — still multimodal."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        },
        {"role": "assistant", "content": "I see a cat."},
        {"role": "user", "content": "What breed?"},
    ]


# -- supports_multimodal_fallback property ------------------------------------

class TestSupportsMultimodalFallback:
    def test_vlm_fallback_returns_true(self):
        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            fallback_engine_type="vlm",
        )
        assert engine.supports_multimodal_fallback is True

    def test_batched_fallback_returns_false(self):
        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            fallback_engine_type="batched",
        )
        assert engine.supports_multimodal_fallback is False

    def test_default_fallback_returns_false(self):
        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        assert engine.supports_multimodal_fallback is False


# -- _has_multimodal_content detection ----------------------------------------

class TestHasMultimodalContent:
    """Before: DFlash had no way to detect image content in messages.
    After: _has_multimodal_content() scans for all three image part types."""

    def test_text_only_returns_false(self):
        assert DFlashEngine._has_multimodal_content(_text_only_messages()) is False

    def test_image_url_detected(self):
        assert DFlashEngine._has_multimodal_content(_image_url_messages()) is True

    def test_image_type_detected(self):
        assert DFlashEngine._has_multimodal_content(_image_type_messages()) is True

    def test_input_image_detected(self):
        assert DFlashEngine._has_multimodal_content(_input_image_messages()) is True

    def test_mixed_history_detected(self):
        assert DFlashEngine._has_multimodal_content(_mixed_history_messages()) is True

    def test_string_content_ignored(self):
        msgs = [{"role": "user", "content": "plain string"}]
        assert DFlashEngine._has_multimodal_content(msgs) is False

    def test_empty_messages(self):
        assert DFlashEngine._has_multimodal_content([]) is False

    def test_no_content_key(self):
        msgs = [{"role": "system"}]
        assert DFlashEngine._has_multimodal_content(msgs) is False


# -- chat()/stream_chat() multimodal fallback ---------------------------------

class TestChatMultimodalFallback:
    """Before: chat() always applied text-only _apply_chat_template(), which
    flattened multimodal content to plain text. Images were lost.
    After: chat() detects multimodal messages in VLM-fallback DFlash engines
    and delegates to the VLM fallback engine, which handles images natively."""

    @pytest.fixture
    def vlm_dflash_engine(self):
        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            fallback_engine_type="vlm",
        )
        engine._loaded = True
        engine._tokenizer_obj = MagicMock()
        return engine

    @pytest.fixture
    def batched_dflash_engine(self):
        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            fallback_engine_type="batched",
        )
        engine._loaded = True
        engine._tokenizer_obj = MagicMock()
        return engine

    @pytest.mark.asyncio
    async def test_chat_triggers_vlm_fallback_on_images(self, vlm_dflash_engine):
        mock_output = MagicMock()
        mock_fallback = AsyncMock()
        mock_fallback.chat = AsyncMock(return_value=mock_output)

        with patch.object(vlm_dflash_engine, "_evict_dflash_and_start_fallback") as mock_evict:
            mock_evict.side_effect = lambda: setattr(vlm_dflash_engine, "_fallback_engine", mock_fallback) or setattr(vlm_dflash_engine, "_in_fallback_mode", True)

            result = await vlm_dflash_engine.chat(_image_url_messages())

        mock_evict.assert_called_once()
        mock_fallback.chat.assert_called_once()
        call_msgs = mock_fallback.chat.call_args[0][0]
        assert any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for msg in call_msgs
            if isinstance(msg.get("content"), list)
            for part in msg["content"]
        )
        assert result is mock_output

    @pytest.mark.asyncio
    async def test_chat_text_only_uses_normal_dflash_path(self, vlm_dflash_engine):
        vlm_dflash_engine._apply_chat_template = MagicMock(return_value="formatted prompt")
        vlm_dflash_engine.generate = AsyncMock(return_value=MagicMock())

        await vlm_dflash_engine.chat(_text_only_messages())

        vlm_dflash_engine._apply_chat_template.assert_called_once()
        vlm_dflash_engine.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_batched_fallback_ignores_images(self, batched_dflash_engine):
        """DFlash with batched (non-VLM) fallback has no multimodal support.
        Images in messages proceed through the normal text path (existing behavior)."""
        batched_dflash_engine._apply_chat_template = MagicMock(return_value="formatted")
        batched_dflash_engine.generate = AsyncMock(return_value=MagicMock())

        await batched_dflash_engine.chat(_image_url_messages())

        batched_dflash_engine._apply_chat_template.assert_called_once()
        batched_dflash_engine.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_already_in_fallback_forwards_directly(self, vlm_dflash_engine):
        mock_fallback = AsyncMock()
        mock_fallback.chat = AsyncMock(return_value=MagicMock())
        vlm_dflash_engine._in_fallback_mode = True
        vlm_dflash_engine._fallback_engine = mock_fallback

        await vlm_dflash_engine.chat(_image_url_messages())

        mock_fallback.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_already_in_fallback_text_still_forwards(self, vlm_dflash_engine):
        """Once in fallback mode, even text-only messages go through the
        fallback engine (sticky fallback — no reload)."""
        mock_fallback = AsyncMock()
        mock_fallback.chat = AsyncMock(return_value=MagicMock())
        vlm_dflash_engine._in_fallback_mode = True
        vlm_dflash_engine._fallback_engine = mock_fallback

        await vlm_dflash_engine.chat(_text_only_messages())

        mock_fallback.chat.assert_called_once()


class TestStreamChatMultimodalFallback:
    """Same before/after as chat(), but for the streaming path."""

    @pytest.fixture
    def vlm_dflash_engine(self):
        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            fallback_engine_type="vlm",
        )
        engine._loaded = True
        engine._tokenizer_obj = MagicMock()
        return engine

    @pytest.mark.asyncio
    async def test_stream_chat_triggers_vlm_fallback_on_images(self, vlm_dflash_engine):
        mock_output = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield mock_output

        mock_fallback = AsyncMock()
        mock_fallback.stream_chat = mock_stream

        with patch.object(vlm_dflash_engine, "_evict_dflash_and_start_fallback") as mock_evict:
            mock_evict.side_effect = lambda: setattr(vlm_dflash_engine, "_fallback_engine", mock_fallback) or setattr(vlm_dflash_engine, "_in_fallback_mode", True)

            outputs = []
            async for out in vlm_dflash_engine.stream_chat(_image_url_messages()):
                outputs.append(out)

        mock_evict.assert_called_once()
        assert len(outputs) == 1
        assert outputs[0] is mock_output

    @pytest.mark.asyncio
    async def test_stream_chat_already_in_fallback_forwards(self, vlm_dflash_engine):
        mock_output = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield mock_output

        mock_fallback = AsyncMock()
        mock_fallback.stream_chat = mock_stream
        vlm_dflash_engine._in_fallback_mode = True
        vlm_dflash_engine._fallback_engine = mock_fallback

        outputs = []
        async for out in vlm_dflash_engine.stream_chat(_image_url_messages()):
            outputs.append(out)

        assert len(outputs) == 1


# -- Concurrent fallback (lock correctness) -----------------------------------

class TestFallbackLockSafety:
    """Verify _fallback_lock prevents double eviction from concurrent requests."""

    @pytest.mark.asyncio
    async def test_concurrent_image_requests_evict_once(self):
        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            fallback_engine_type="vlm",
        )
        engine._loaded = True
        engine._tokenizer_obj = MagicMock()

        evict_count = 0

        async def mock_evict():
            nonlocal evict_count
            evict_count += 1
            await asyncio.sleep(0.05)
            engine._fallback_engine = AsyncMock()
            engine._fallback_engine.chat = AsyncMock(return_value=MagicMock())
            engine._in_fallback_mode = True

        with patch.object(engine, "_evict_dflash_and_start_fallback", side_effect=mock_evict):
            results = await asyncio.gather(
                engine.chat(_image_url_messages()),
                engine.chat(_image_url_messages()),
                engine.chat(_image_url_messages()),
            )

        assert evict_count == 1
        assert len(results) == 3


# -- Server-side extraction routing (before/after) ----------------------------

class TestServerExtractionRouting:
    """Before: server.py checked `isinstance(engine, VLMBatchedEngine)` only.
    DFlash engines always took the text-only extraction path, stripping images.

    After: server.py also checks `engine.supports_multimodal_fallback` for
    DFlash engines, routing them through multimodal extraction when True."""

    def test_extract_text_content_drops_images(self):
        """BEFORE behavior: extract_text_content silently drops image parts."""
        from omlx.api.utils import extract_text_content

        messages = [
            MagicMock(
                role="user",
                content=[
                    MagicMock(
                        model_dump=lambda: {"type": "text", "text": "Describe this"},
                    ),
                    MagicMock(
                        model_dump=lambda: {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                    ),
                ],
                tool_call_id=None,
            )
        ]
        result = extract_text_content(messages, None, None)
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, str):
                assert "image" not in content.lower()
            elif isinstance(content, list):
                for part in content:
                    assert part.get("type") != "image_url"

    def test_extract_multimodal_content_preserves_images(self):
        """AFTER behavior: extract_multimodal_content keeps image_url parts."""
        from omlx.api.utils import extract_multimodal_content

        messages = [
            MagicMock(
                role="user",
                content=[
                    MagicMock(
                        model_dump=lambda: {"type": "text", "text": "Describe this"},
                    ),
                    MagicMock(
                        model_dump=lambda: {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                    ),
                ],
                tool_call_id=None,
            )
        ]
        result = extract_multimodal_content(messages, None, None)
        has_image = False
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        has_image = True
        assert has_image, "extract_multimodal_content must preserve image_url parts"

    def test_dflash_vlm_detected_via_getattr(self):
        """server.py uses getattr(engine, 'supports_multimodal_fallback', False)
        to avoid importing DFlashEngine directly."""
        engine_vlm = DFlashEngine(
            model_name="test", draft_model_path="test", fallback_engine_type="vlm",
        )
        engine_batched = DFlashEngine(
            model_name="test", draft_model_path="test", fallback_engine_type="batched",
        )
        assert getattr(engine_vlm, "supports_multimodal_fallback", False) is True
        assert getattr(engine_batched, "supports_multimodal_fallback", False) is False

        plain_engine = MagicMock(spec=[])
        assert getattr(plain_engine, "supports_multimodal_fallback", False) is False
