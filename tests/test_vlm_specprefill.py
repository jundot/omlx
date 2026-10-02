"""Regression tests for SpecPrefill parameter forwarding in VLM engine."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine.vlm import VLMBatchedEngine


@pytest.mark.asyncio
async def test_vlm_chat_forwards_specprefill_threshold_and_keep_pct():
    """VLM chat must pass both SpecPrefill overrides through to add_request()."""
    engine = VLMBatchedEngine(model_name="test-vlm")
    engine._loaded = True
    engine._vlm_model = MagicMock()
    engine._vlm_model.config.model_type = "test"
    engine._tokenizer = MagicMock()
    engine._tokenizer.apply_chat_template.return_value = "<prompt>"
    engine._tokenizer.encode.side_effect = lambda text, **kwargs: list(range(max(1, len(text.split()))))
    engine._engine = MagicMock()
    engine._engine._mlx_executor = ThreadPoolExecutor(max_workers=1)
    engine._engine.add_request = AsyncMock(return_value="req-1")
    engine._engine.abort_request = AsyncMock(return_value=True)

    async def _one_output_stream(_request_id):
        yield MagicMock(
            output_text="ok",
            new_text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
        )

    engine._engine.stream_outputs = _one_output_stream

    # Mock _process_chat_messages to skip mlx-vlm template processing
    def _mock_process(messages, tools, kwargs):
        return "<prompt>", None, {}, None, None, []

    with patch.object(engine, "_process_chat_messages", side_effect=_mock_process):
        async for _ in engine.stream_chat(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
            specprefill=True,
            specprefill_keep_pct=0.2,
            specprefill_threshold=1024,
        ):
            pass

    try:
        _, kwargs = engine._engine.add_request.call_args
        assert kwargs["specprefill"] is True
        assert kwargs["specprefill_keep_pct"] == 0.2
        assert kwargs["specprefill_threshold"] == 1024
    finally:
        engine._engine._mlx_executor.shutdown(wait=False)


class TestVLMEngineSpecPrefillForwarding:
    """Non-streaming path must forward SpecPrefill overrides (issue #2274/#2281 parity).

    ``generate()``/``chat()`` previously dropped SpecPrefill kwargs on the VLM
    engine, so a configured keep_pct silently fell back to the engine default.
    """

    @staticmethod
    def _fake_output():
        return SimpleNamespace(
            output_text="hi",
            prompt_tokens=5,
            completion_tokens=2,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
            first_token_at=None,
        )

    def test_pop_specprefill_kwargs_extracts_and_pops(self):
        kwargs = {
            "specprefill_keep_pct": 0.25,
            "specprefill_threshold": 100,
            "specprefill_system_end": 12,
            "specprefill": True,
            "temperature": 0.7,
        }
        extracted = VLMBatchedEngine._pop_specprefill_kwargs(kwargs)

        assert extracted == {
            "specprefill_keep_pct": 0.25,
            "specprefill_threshold": 100,
            "specprefill_system_end": 12,
            "specprefill": True,
        }
        # Popped out of the original dict; unrelated kwargs are untouched.
        assert kwargs == {"temperature": 0.7}

    def test_pop_specprefill_kwargs_ignores_none_values(self):
        kwargs = {"specprefill_keep_pct": None, "specprefill": None}
        assert VLMBatchedEngine._pop_specprefill_kwargs(kwargs) == {}

    @pytest.mark.asyncio
    async def test_generate_forwards_specprefill_kwargs(self):
        engine = VLMBatchedEngine(model_name="test-vlm")
        engine._loaded = True
        engine._engine = SimpleNamespace(
            generate=AsyncMock(return_value=self._fake_output())
        )

        await engine.generate(
            "a prompt",
            specprefill_keep_pct=0.25,
            specprefill_threshold=100,
        )

        call_kwargs = engine._engine.generate.call_args.kwargs
        assert call_kwargs["specprefill_keep_pct"] == 0.25
        assert call_kwargs["specprefill_threshold"] == 100

    @pytest.mark.asyncio
    async def test_generate_omits_specprefill_when_absent(self):
        engine = VLMBatchedEngine(model_name="test-vlm")
        engine._loaded = True
        engine._engine = SimpleNamespace(
            generate=AsyncMock(return_value=self._fake_output())
        )

        await engine.generate("a prompt")

        call_kwargs = engine._engine.generate.call_args.kwargs
        assert "specprefill_keep_pct" not in call_kwargs
        assert "specprefill_threshold" not in call_kwargs

    @pytest.mark.asyncio
    async def test_chat_injects_specprefill_system_end(self):
        engine = VLMBatchedEngine(model_name="test-vlm")
        engine._loaded = True
        engine._model_settings = SimpleNamespace(specprefill_enabled=True)
        engine._engine = MagicMock()
        engine._engine._mlx_executor = ThreadPoolExecutor(max_workers=1)
        engine._engine.generate = AsyncMock(return_value=self._fake_output())

        # VLM prompts are pre-tokenized (list[int]) by _process_chat_messages;
        # full_tokens = len(prompt) = 10, non_system_tokens = 4, so
        # system_end = 10 - 4 = 6.
        engine._tokenizer = MagicMock()
        engine._tokenizer.apply_chat_template.return_value = "USER_ONLY"
        engine._tokenizer.encode.side_effect = lambda text, **kwargs: [0] * 4

        def _mock_process(messages, tools, kwargs):
            return list(range(10)), None, None, None, 0, []

        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]
        try:
            with patch.object(engine, "_process_chat_messages", side_effect=_mock_process):
                await engine.chat(messages)
        finally:
            engine._engine._mlx_executor.shutdown(wait=False)

        call_kwargs = engine._engine.generate.call_args.kwargs
        assert call_kwargs["specprefill_system_end"] == 6

    @pytest.mark.asyncio
    async def test_chat_skips_system_end_when_specprefill_disabled(self):
        engine = VLMBatchedEngine(model_name="test-vlm")
        engine._loaded = True
        engine._model_settings = SimpleNamespace(specprefill_enabled=False)
        engine._engine = MagicMock()
        engine._engine._mlx_executor = ThreadPoolExecutor(max_workers=1)
        engine._engine.generate = AsyncMock(return_value=self._fake_output())

        engine._tokenizer = MagicMock()
        engine._tokenizer.apply_chat_template.return_value = "USER_ONLY"
        engine._tokenizer.encode.side_effect = lambda text, **kwargs: [0] * 4

        def _mock_process(messages, tools, kwargs):
            return list(range(8)), None, None, None, 0, []

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        try:
            with patch.object(engine, "_process_chat_messages", side_effect=_mock_process):
                await engine.chat(messages)
        finally:
            engine._engine._mlx_executor.shutdown(wait=False)

        call_kwargs = engine._engine.generate.call_args.kwargs
        assert "specprefill_system_end" not in call_kwargs
