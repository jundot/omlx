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
import time
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


# -- Auto-revert after fallback idle (issue: sticky fallback regression) -------

class TestAutoRevert:
    """After an image request puts the engine in VLM fallback mode, text
    requests should reload dflash once fallback has been idle past the
    cooldown. While requests keep arriving, fallback stays (no churn)."""

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

    def test_should_auto_revert_false_when_not_in_fallback(self, vlm_dflash_engine):
        vlm_dflash_engine._in_fallback_mode = False
        assert vlm_dflash_engine._should_auto_revert() is False

    def test_should_auto_revert_false_for_non_vlm_fallback(self):
        """Context-length fallback (BatchedEngine) must never auto-revert —
        reloading dflash would just re-evict on the next long request."""
        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            fallback_engine_type="batched",
        )
        engine._in_fallback_mode = True
        engine._last_fallback_activity_ts = (
            time.monotonic() - engine._fallback_cooldown_secs - 1
        )
        assert engine._should_auto_revert() is False

    def test_should_auto_revert_false_within_cooldown(self, vlm_dflash_engine):
        vlm_dflash_engine._in_fallback_mode = True
        vlm_dflash_engine._last_fallback_activity_ts = time.monotonic()
        assert vlm_dflash_engine._should_auto_revert() is False

    def test_should_auto_revert_true_after_cooldown(self, vlm_dflash_engine):
        vlm_dflash_engine._in_fallback_mode = True
        vlm_dflash_engine._last_fallback_activity_ts = (
            time.monotonic() - vlm_dflash_engine._fallback_cooldown_secs - 1
        )
        assert vlm_dflash_engine._should_auto_revert() is True

    def test_should_auto_revert_false_when_cooldown_disabled(self):
        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            fallback_engine_type="vlm",
            model_settings=MagicMock(dflash_fallback_cooldown_secs=0),
        )
        engine._in_fallback_mode = True
        engine._last_fallback_activity_ts = time.monotonic() - 999
        assert engine._should_auto_revert() is False

    def test_mark_fallback_activity_resets_cooldown(self, vlm_dflash_engine):
        vlm_dflash_engine._in_fallback_mode = True
        vlm_dflash_engine._last_fallback_activity_ts = (
            time.monotonic() - vlm_dflash_engine._fallback_cooldown_secs - 1
        )
        vlm_dflash_engine._mark_fallback_activity()
        assert vlm_dflash_engine._should_auto_revert() is False

    @pytest.mark.asyncio
    async def test_chat_text_only_auto_reverts_after_idle(self, vlm_dflash_engine):
        """A text-only chat in fallback, idle past cooldown, reloads dflash
        and runs on the dflash path instead of the fallback engine."""
        mock_fallback = AsyncMock()
        vlm_dflash_engine._fallback_engine = mock_fallback
        vlm_dflash_engine._in_fallback_mode = True
        vlm_dflash_engine._last_fallback_activity_ts = (
            time.monotonic() - vlm_dflash_engine._fallback_cooldown_secs - 1
        )
        vlm_dflash_engine._apply_chat_template = MagicMock(return_value="formatted")
        vlm_dflash_engine.generate = AsyncMock(return_value=MagicMock())

        with patch.object(
            vlm_dflash_engine, "_reload_dflash_from_fallback"
        ) as mock_reload:
            async def _fake_reload():
                vlm_dflash_engine._in_fallback_mode = False
                return True

            mock_reload.side_effect = _fake_reload
            await vlm_dflash_engine.chat(_text_only_messages())

        mock_reload.assert_called_once()
        vlm_dflash_engine._apply_chat_template.assert_called_once()
        vlm_dflash_engine.generate.assert_called_once()
        mock_fallback.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_text_reload_failure_serves_fallback(self, vlm_dflash_engine):
        """If the dflash reload fails, the request is served on the fallback
        engine (no 500) and the cooldown resets so reload isn't retried."""
        mock_fallback = AsyncMock()
        mock_fallback.chat = AsyncMock(return_value=MagicMock())
        vlm_dflash_engine._fallback_engine = mock_fallback
        vlm_dflash_engine._in_fallback_mode = True
        vlm_dflash_engine._last_fallback_activity_ts = (
            time.monotonic() - vlm_dflash_engine._fallback_cooldown_secs - 1
        )
        vlm_dflash_engine._apply_chat_template = MagicMock(return_value="formatted")

        with patch.object(
            vlm_dflash_engine, "_reload_dflash_from_fallback", new=AsyncMock(return_value=False)
        ):
            await vlm_dflash_engine.chat(_text_only_messages())

        mock_fallback.chat.assert_called_once()
        assert vlm_dflash_engine._should_auto_revert() is False

    @pytest.mark.asyncio
    async def test_chat_image_stays_in_fallback_even_when_idle(self, vlm_dflash_engine):
        """Image chats never trigger auto-revert — they need the VLM engine."""
        mock_fallback = AsyncMock()
        mock_fallback.chat = AsyncMock(return_value=MagicMock())
        vlm_dflash_engine._fallback_engine = mock_fallback
        vlm_dflash_engine._in_fallback_mode = True
        vlm_dflash_engine._last_fallback_activity_ts = (
            time.monotonic() - vlm_dflash_engine._fallback_cooldown_secs - 1
        )

        await vlm_dflash_engine.chat(_image_url_messages())

        mock_fallback.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_text_within_cooldown_stays_in_fallback(self, vlm_dflash_engine):
        """Text arriving inside the cooldown window keeps using fallback."""
        mock_fallback = AsyncMock()
        mock_fallback.chat = AsyncMock(return_value=MagicMock())
        vlm_dflash_engine._fallback_engine = mock_fallback
        vlm_dflash_engine._in_fallback_mode = True
        vlm_dflash_engine._last_fallback_activity_ts = time.monotonic()

        await vlm_dflash_engine.chat(_text_only_messages())

        mock_fallback.chat.assert_called_once()


# -- Auto-revert lifecycle (concurrency + failure states) -----------------------

class TestAutoRevertLifecycle:
    """Lifecycle review fixes:
    - in-flight fallback requests block auto-revert (no mid-stream teardown)
    - the cooldown is measured from completed serving, not fallback startup
    - a failed dflash reload leaves the engine usable in fallback mode
    - dflash_fallback_cooldown_secs=None does not crash
    """

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

    def test_should_auto_revert_false_while_request_in_flight(self, vlm_dflash_engine):
        """A long image stream still being served must not be torn down by a
        concurrent text request — even when the cooldown has elapsed."""
        engine = vlm_dflash_engine
        engine._in_fallback_mode = True
        engine._fallback_active_requests = 1
        engine._last_fallback_activity_ts = (
            time.monotonic() - engine._fallback_cooldown_secs - 1
        )
        assert engine._should_auto_revert() is False

    def test_exit_fallback_request_stamps_and_releases_slot(self, vlm_dflash_engine):
        engine = vlm_dflash_engine
        engine._in_fallback_mode = True
        engine._enter_fallback_request()
        # Age the timestamp while the request is "in flight", then complete it.
        engine._last_fallback_activity_ts = time.monotonic() - 999
        engine._exit_fallback_request()
        assert engine._fallback_active_requests == 0
        # Activity re-stamped on completion → cooldown restarts from serving.
        assert engine._should_auto_revert() is False

    def test_exit_fallback_request_needs_no_matching_enter(self, vlm_dflash_engine):
        """Defensive: an unbalanced exit must not underflow the counter."""
        engine = vlm_dflash_engine
        engine._exit_fallback_request()
        assert engine._fallback_active_requests == 0

    @pytest.mark.asyncio
    async def test_evict_fallback_does_not_stamp_activity(self, vlm_dflash_engine):
        """The cooldown is measured from actual serving, not from the moment
        the fallback engine started."""
        engine = vlm_dflash_engine
        engine._last_fallback_activity_ts = None

        async def fake_evict():
            engine._fallback_engine = AsyncMock()
            engine._in_fallback_mode = True

        with patch.object(engine, "_evict_dflash_and_start_fallback", side_effect=fake_evict):
            await engine.chat(_image_url_messages())

        # No serving has completed before the first request finishes, and the
        # request has now completed → activity stamped exactly on serving.
        assert engine._fallback_active_requests == 0
        assert engine._last_fallback_activity_ts is not None
        assert engine._should_auto_revert() is False

    def test_cooldown_none_disables_auto_revert(self):
        """Explicit None (settings JSON null) must not crash float() and must
        disable auto-revert, per the documented 0/None-disables contract."""
        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            fallback_engine_type="vlm",
            model_settings=MagicMock(dflash_fallback_cooldown_secs=None),
        )
        engine._in_fallback_mode = True
        engine._last_fallback_activity_ts = time.monotonic() - 999
        assert engine._fallback_cooldown_secs == 0.0
        assert engine._should_auto_revert() is False

    def test_cooldown_invalid_value_falls_back_to_default(self):
        engine = DFlashEngine(
            model_name="test-model",
            draft_model_path="test-draft",
            fallback_engine_type="vlm",
            model_settings=MagicMock(dflash_fallback_cooldown_secs="not-a-number"),
        )
        assert engine._fallback_cooldown_secs == 30.0

    @pytest.mark.asyncio
    async def test_reload_failure_keeps_engine_loaded_and_in_fallback(
        self, vlm_dflash_engine
    ):
        """After a failed reload the engine must remain usable: _loaded stays
        True so the next request routes via fallback mode instead of calling
        start() again (which would fail with the same error and 500)."""
        engine = vlm_dflash_engine
        engine._in_fallback_mode = True
        engine._fallback_engine = None  # stop() cleared it

        async def fake_stop():
            engine._loaded = False

        async def fake_start():
            raise RuntimeError("dflash reload failed")

        async def fake_evict():
            engine._fallback_engine = AsyncMock()

        with patch.object(engine, "stop", side_effect=fake_stop), \
             patch.object(engine, "start", side_effect=fake_start), \
             patch.object(engine, "_evict_dflash_and_start_fallback", side_effect=fake_evict):
            result = await engine._reload_dflash_from_fallback()

        assert result is False
        assert engine._in_fallback_mode is True
        assert engine._loaded is True
        assert engine._fallback_engine is not None

    @pytest.mark.asyncio
    async def test_text_chat_after_reload_failure_stays_on_fallback(
        self, vlm_dflash_engine
    ):
        """End-to-end: a text request after a failed reload is served by the
        fallback engine without calling start() again."""
        engine = vlm_dflash_engine
        mock_fallback = AsyncMock()
        mock_fallback.chat = AsyncMock(return_value=MagicMock())
        engine._fallback_engine = mock_fallback
        engine._in_fallback_mode = True
        engine._loaded = True
        engine._last_fallback_activity_ts = (
            time.monotonic() - engine._fallback_cooldown_secs - 1
        )
        engine.start = AsyncMock(side_effect=RuntimeError("reload broken"))

        async def fake_reload():
            engine._in_fallback_mode = True
            engine._loaded = True
            engine._mark_fallback_activity()
            return False

        with patch.object(engine, "_reload_dflash_from_fallback", side_effect=fake_reload):
            await engine.chat(_text_only_messages())

        mock_fallback.chat.assert_called_once()
        engine.start.assert_not_called()
