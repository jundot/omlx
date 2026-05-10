# SPDX-License-Identifier: Apache-2.0
"""
Tests for BatchedEngine and BaseEngine modules.

Tests cover:
- GenerationOutput: dataclass behavior
- BaseEngine ABC: interface verification
- BatchedEngine initialization
- _apply_chat_template(): chat template application
- _preprocess_messages(): Harmony preprocessing
- get_stats(), get_cache_stats()

Note: mlx_lm.load() is mocked to avoid loading real models.
"""

from abc import ABC
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from omlx.engine.base import BaseEngine, BaseNonStreamingEngine, GenerationOutput


class FakeStreamingCore:
    """Minimal async engine core for stream cleanup tests."""

    def __init__(self):
        self.aborted_request_id = None

    async def add_request(self, **kwargs):
        return "request-1"

    async def stream_outputs(self, request_id):
        yield SimpleNamespace(
            output_text="partial",
            new_text="partial",
            prompt_tokens=1,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
            tool_calls=None,
            cached_tokens=0,
        )

    async def abort_request(self, request_id):
        self.aborted_request_id = request_id


class TestGenerationOutput:
    """Tests for GenerationOutput dataclass."""

    def test_default_values(self):
        """Test GenerationOutput has correct defaults."""
        output = GenerationOutput(text="Hello, world!")

        assert output.text == "Hello, world!"
        assert output.tokens == []
        assert output.prompt_tokens == 0
        assert output.completion_tokens == 0
        assert output.finish_reason == "stop"
        assert output.new_text == ""
        assert output.finished is True
        assert output.tool_calls is None

    def test_custom_values(self):
        """Test GenerationOutput with custom values."""
        output = GenerationOutput(
            text="Generated text",
            tokens=[100, 101, 102],
            prompt_tokens=10,
            completion_tokens=3,
            finish_reason="length",
            new_text="partial",
            finished=False,
            tool_calls=[{"name": "test_tool", "arguments": "{}"}],
        )

        assert output.text == "Generated text"
        assert output.tokens == [100, 101, 102]
        assert output.prompt_tokens == 10
        assert output.completion_tokens == 3
        assert output.finish_reason == "length"
        assert output.new_text == "partial"
        assert output.finished is False
        assert output.tool_calls == [{"name": "test_tool", "arguments": "{}"}]

    def test_streaming_output(self):
        """Test GenerationOutput for streaming use case."""
        output = GenerationOutput(
            text="",
            new_text="Hello",
            prompt_tokens=5,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
        )

        assert output.text == ""
        assert output.new_text == "Hello"
        assert output.finished is False
        assert output.finish_reason is None


class TestBaseEngine:
    """Tests for BaseEngine ABC."""

    def test_is_abstract(self):
        """Test BaseEngine is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseEngine()

    def test_abstract_methods(self):
        """Test BaseEngine defines required abstract methods."""
        abstract_methods = {
            "model_name",
            "tokenizer",
            "start",
            "stop",
            "generate",
            "stream_generate",
            "chat",
            "stream_chat",
            "model_type",
            "get_stats",
            "get_cache_stats",
        }

        # Check abstractmethods
        actual_methods = set()
        for name in dir(BaseEngine):
            if not name.startswith("_"):
                attr = getattr(BaseEngine, name)
                if getattr(attr, "__isabstractmethod__", False):
                    actual_methods.add(name)

        assert actual_methods == abstract_methods

    def test_concrete_implementation(self):
        """Test a concrete implementation can be created."""

        class ConcreteEngine(BaseEngine):
            @property
            def model_name(self) -> str:
                return "test-model"

            @property
            def tokenizer(self) -> Any:
                return None

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def generate(
                self,
                prompt: str,
                max_tokens: int = 256,
                temperature: float = 0.7,
                top_p: float = 0.9,
                stop: Optional[List[str]] = None,
                **kwargs,
            ) -> GenerationOutput:
                return GenerationOutput(text="test")

            async def stream_generate(
                self,
                prompt: str,
                max_tokens: int = 256,
                temperature: float = 0.7,
                top_p: float = 0.9,
                stop: Optional[List[str]] = None,
                **kwargs,
            ):
                yield GenerationOutput(text="test")

            async def chat(
                self,
                messages: List[Dict[str, Any]],
                max_tokens: int = 256,
                temperature: float = 0.7,
                top_p: float = 0.9,
                tools: Optional[List[dict]] = None,
                **kwargs,
            ) -> GenerationOutput:
                return GenerationOutput(text="test")

            async def stream_chat(
                self,
                messages: List[Dict[str, Any]],
                max_tokens: int = 256,
                temperature: float = 0.7,
                top_p: float = 0.9,
                tools: Optional[List[dict]] = None,
                **kwargs,
            ):
                yield GenerationOutput(text="test")

            @property
            def model_type(self) -> Optional[str]:
                return "test"

            def get_stats(self) -> Dict[str, Any]:
                return {}

            def get_cache_stats(self) -> Optional[Dict[str, Any]]:
                return None

        engine = ConcreteEngine()
        assert engine.model_name == "test-model"


class TestBaseNonStreamingEngine:
    """Tests for BaseNonStreamingEngine ABC."""

    def test_is_abstract(self):
        """Test BaseNonStreamingEngine is abstract."""
        with pytest.raises(TypeError):
            BaseNonStreamingEngine()

    def test_abstract_methods(self):
        """Test BaseNonStreamingEngine defines required abstract methods."""
        abstract_methods = {"model_name", "start", "stop", "get_stats"}

        actual_methods = set()
        for name in dir(BaseNonStreamingEngine):
            if not name.startswith("_"):
                attr = getattr(BaseNonStreamingEngine, name)
                if getattr(attr, "__isabstractmethod__", False):
                    actual_methods.add(name)

        assert actual_methods == abstract_methods


class TestBatchedEngineInitialization:
    """Tests for BatchedEngine initialization."""

    def test_init_stores_parameters(self):
        """Test BatchedEngine stores initialization parameters."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(
            model_name="test-model",
            trust_remote_code=False,
            stream_interval=5,
            enable_thinking=True,
        )

        assert engine._model_name == "test-model"
        assert engine._trust_remote_code is False
        assert engine._stream_interval == 5
        assert engine._enable_thinking is True
        assert engine._loaded is False
        assert engine._model is None
        assert engine._tokenizer is None
        assert engine._engine is None

    def test_init_default_values(self):
        """Test BatchedEngine default values."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        # Issue #926: default flipped to False so HF repos can't auto-execute custom Python.
        assert engine._trust_remote_code is False
        assert engine._scheduler_config is None
        assert engine._stream_interval == 1
        assert engine._enable_thinking is None

    def test_model_name_property(self):
        """Test model_name property."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="my-model")

        assert engine.model_name == "my-model"

    def test_tokenizer_property_before_load(self):
        """Test tokenizer property returns None before loading."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        assert engine.tokenizer is None

    def test_model_type_property_before_load(self):
        """Test model_type property returns None before loading."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        assert engine.model_type is None


class TestBatchedEngineStreamingCleanup:
    """Tests for streaming generator cleanup paths."""

    @pytest.mark.asyncio
    async def test_stream_abort_uses_captured_engine_if_engine_cleared(self):
        """Generator finalization aborts on the original engine reference."""
        from omlx.engine.batched import BatchedEngine

        fake_engine = FakeStreamingCore()
        engine = BatchedEngine(model_name="test-model")
        engine._loaded = True
        engine._engine = fake_engine

        stream = engine.stream_generate("hello")
        first = await stream.__anext__()
        assert first.text == "partial"

        engine._engine = None
        await stream.aclose()

        assert fake_engine.aborted_request_id == "request-1"


class TestBatchedEngineApplyChatTemplate:
    """Tests for BatchedEngine._apply_chat_template()."""

    def test_apply_chat_template_with_tokenizer(self):
        """Test _apply_chat_template when tokenizer has apply_chat_template."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        # Mock tokenizer with apply_chat_template
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted prompt>"
        engine._tokenizer = mock_tokenizer

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = engine._apply_chat_template(messages)

        assert result == "<formatted prompt>"
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_apply_chat_template_with_tools(self):
        """Test _apply_chat_template passes tools to tokenizer."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted>"
        engine._tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test"}}]

        engine._apply_chat_template(messages, tools=tools)

        # Verify tools were passed
        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools

    def test_apply_chat_template_with_enable_thinking(self):
        """Test _apply_chat_template passes enable_thinking."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model", enable_thinking=True)

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted>"
        engine._tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hello"}]

        engine._apply_chat_template(messages)

        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs.get("enable_thinking") is True

    def test_apply_chat_template_fallback(self):
        """Test _apply_chat_template fallback when tokenizer lacks method."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        # Tokenizer without apply_chat_template
        mock_tokenizer = MagicMock(spec=[])
        del mock_tokenizer.apply_chat_template  # Explicitly remove
        engine._tokenizer = mock_tokenizer

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        result = engine._apply_chat_template(messages)

        assert "user: Hello" in result
        assert "assistant: Hi" in result
        assert result.endswith("assistant:")

    def test_apply_chat_template_handles_type_error(self):
        """Test _apply_chat_template handles TypeError from tokenizer."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model", enable_thinking=True)

        # Tokenizer that raises TypeError for enable_thinking
        mock_tokenizer = MagicMock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and "enable_thinking" in kwargs:
                raise TypeError("enable_thinking not supported")
            return "<formatted>"

        mock_tokenizer.apply_chat_template.side_effect = side_effect
        engine._tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hello"}]

        result = engine._apply_chat_template(messages)

        assert result == "<formatted>"
        assert call_count == 2  # Called twice (first fails, second succeeds)


class TestBatchedEnginePreprocessMessages:
    """Tests for BatchedEngine._preprocess_messages()."""

    def test_preprocess_messages_non_harmony(self):
        """Test _preprocess_messages returns unchanged for non-Harmony models."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")
        engine._model = MagicMock()
        engine._model.config = MagicMock()
        engine._model.config.model_type = "llama"  # Not gpt_oss

        messages = [{"role": "user", "content": "Hello"}]

        result = engine._preprocess_messages(messages)

        assert result == messages

    def test_preprocess_messages_model_type_none(self):
        """Test _preprocess_messages when model_type is None."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")
        # No model loaded

        messages = [{"role": "user", "content": "Hello"}]

        result = engine._preprocess_messages(messages)

        assert result == messages


class TestBatchedEngineStats:
    """Tests for BatchedEngine statistics methods."""

    def test_get_stats_before_load(self):
        """Test get_stats() before model is loaded."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model", stream_interval=3)

        stats = engine.get_stats()

        assert stats["engine_type"] == "batched"
        assert stats["model_name"] == "test-model"
        assert stats["loaded"] is False
        assert stats["stream_interval"] == 3

    def test_get_stats_includes_engine_stats(self):
        """Test get_stats() includes engine stats when loaded."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        # Mock loaded engine
        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {
            "running": True,
            "steps_executed": 100,
        }
        engine._engine = mock_engine
        engine._loaded = True

        stats = engine.get_stats()

        assert stats["loaded"] is True
        assert stats["running"] is True
        assert stats["steps_executed"] == 100

    def test_get_cache_stats_before_load(self):
        """Test get_cache_stats() before model is loaded."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        stats = engine.get_cache_stats()

        assert stats is None

    def test_get_cache_stats_after_load(self):
        """Test get_cache_stats() when engine is loaded."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        mock_engine = MagicMock()
        mock_engine.get_cache_stats.return_value = {"entries": 10}
        engine._engine = mock_engine

        stats = engine.get_cache_stats()

        assert stats == {"entries": 10}


class TestBatchedEngineModelType:
    """Tests for BatchedEngine.model_type property."""

    def test_model_type_from_config(self):
        """Test model_type from model.config."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.model_type = "llama"
        engine._model = mock_model

        assert engine.model_type == "llama"

    def test_model_type_from_config_dict(self):
        """Test model_type from dict-style config."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        # Create a mock model where config is a dict
        mock_model = MagicMock(spec=["config"])
        # Use a real dict for config
        mock_model.config = {"model_type": "qwen2"}
        engine._model = mock_model

        # The code checks hasattr(config, 'model_type') first,
        # dicts don't have model_type as attribute, so it checks isinstance(config, dict)
        assert engine.model_type == "qwen2"

    def test_model_type_from_args(self):
        """Test model_type from model.args."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        mock_model = MagicMock(spec=["args"])
        mock_model.args = MagicMock()
        mock_model.args.model_type = "gpt_oss"
        engine._model = mock_model

        assert engine.model_type == "gpt_oss"

    def test_model_type_none_when_not_available(self):
        """Test model_type returns None when not available."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        mock_model = MagicMock(spec=[])
        engine._model = mock_model

        assert engine.model_type is None


class TestApplyChatTemplatePartialMode:
    """Tests for partial mode support in _apply_chat_template()."""

    def test_partial_mode_sets_continue_final_message(self):
        """Final assistant message with partial=True sets continue_final_message."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted>"
        engine._tokenizer = mock_tokenizer

        messages = [
            {"role": "user", "content": "Generate JSON"},
            {"role": "assistant", "content": "{", "partial": True},
        ]

        engine._apply_chat_template(messages)

        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["add_generation_prompt"] is False
        assert call_kwargs["continue_final_message"] is True

    def test_partial_non_assistant_ignored(self):
        """partial=True on a non-assistant message does not trigger partial mode."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted>"
        engine._tokenizer = mock_tokenizer

        messages = [
            {"role": "user", "content": "Hello", "partial": True},
        ]

        engine._apply_chat_template(messages)

        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["add_generation_prompt"] is True
        assert "continue_final_message" not in call_kwargs

    def test_partial_field_stripped_before_template(self):
        """partial field is removed from messages before calling apply_chat_template."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted>"
        engine._tokenizer = mock_tokenizer

        messages = [
            {"role": "user", "content": "Hello", "partial": False},
            {"role": "assistant", "content": "{", "partial": True},
        ]

        engine._apply_chat_template(messages)

        # Check the messages passed to apply_chat_template
        call_args = mock_tokenizer.apply_chat_template.call_args[0][0]
        for msg in call_args:
            assert "partial" not in msg

    def test_partial_true_continues_vs_new_turn(self):
        """Verify the core partial toggle: partial=True continues, absent starts new turn.

        With partial=True on the final assistant message, the engine must pass
        add_generation_prompt=False and continue_final_message=True so the
        model continues from the assistant's content rather than starting a
        new turn.  Without partial, the default add_generation_prompt=True
        appends the generation prompt (e.g. <|im_start|>assistant) for a
        fresh response.

        NOTE: The `name` field is not tested here because its rendering is
        model-template-specific — many templates silently ignore it, so
        assertions on template output would be fragile and model-dependent.
        """
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        mock_tokenizer = MagicMock()
        engine._tokenizer = mock_tokenizer

        base_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Return JSON with keys: name, age"},
        ]

        # --- partial=True: continue from prefill ---
        mock_tokenizer.apply_chat_template.return_value = "...assistant\n{"
        partial_messages = base_messages + [
            {"role": "assistant", "content": "{", "partial": True},
        ]
        engine._apply_chat_template(partial_messages)

        partial_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert partial_kwargs["add_generation_prompt"] is False
        assert partial_kwargs["continue_final_message"] is True

        # --- no partial: new turn ---
        mock_tokenizer.apply_chat_template.reset_mock()
        mock_tokenizer.apply_chat_template.return_value = "...<|im_start|>assistant\n"
        normal_messages = list(base_messages)  # no trailing assistant
        engine._apply_chat_template(normal_messages)

        normal_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert normal_kwargs["add_generation_prompt"] is True
        assert "continue_final_message" not in normal_kwargs

    def test_partial_with_streaming(self):
        """partial mode kwargs are the same regardless of downstream streaming.

        The engine's _apply_chat_template is called identically for streaming
        and non-streaming — the partial toggle affects template kwargs only,
        not the generation path.  This test confirms the kwargs are set
        correctly when the messages would be used in a streaming context.
        """
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "...1."
        engine._tokenizer = mock_tokenizer

        messages = [
            {"role": "user", "content": "List 3 colors"},
            {"role": "assistant", "content": "1.", "partial": True},
        ]

        engine._apply_chat_template(messages)

        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["add_generation_prompt"] is False
        assert call_kwargs["continue_final_message"] is True
        # partial stripped from message dicts
        call_msgs = mock_tokenizer.apply_chat_template.call_args[0][0]
        assert "partial" not in call_msgs[-1]

    def test_count_then_apply_chat_template_idempotent_under_partial_mode(self):
        """Server flow: count_chat_tokens then _apply_chat_template on the
        same messages list must render with identical partial-mode flags.

        Mimics the post-fix server contract: detect_and_strip_partial once
        at the API boundary, forward the resolved value to engine methods
        via an explicit is_partial parameter, and assert both phases pass
        the same partial-mode flags to apply_chat_template.
        """
        from omlx.api.utils import detect_and_strip_partial
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine(model_name="test-model")

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted>"
        mock_tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer = mock_tokenizer

        messages = [
            {"role": "user", "content": "Generate JSON"},
            {"role": "assistant", "content": "{", "partial": True},
        ]

        # Server flow: detect_and_strip_partial once at the API boundary,
        # forward the resolved value to all engine methods.
        is_partial = detect_and_strip_partial(messages)
        assert is_partial is True

        # Phase 1: count.
        engine.count_chat_tokens(messages, is_partial=is_partial)
        count_kwargs = dict(mock_tokenizer.apply_chat_template.call_args.kwargs)

        # Phase 2: chat.  Operates on the same (now-stripped) messages list.
        engine._apply_chat_template(messages, is_partial=is_partial)
        chat_kwargs = dict(mock_tokenizer.apply_chat_template.call_args.kwargs)

        # Both phases must render with identical partial-mode flags.
        assert count_kwargs.get("continue_final_message") == chat_kwargs.get(
            "continue_final_message"
        ), (
            "continue_final_message diverged across phases: "
            f"count={count_kwargs.get('continue_final_message')}, "
            f"chat={chat_kwargs.get('continue_final_message')}"
        )
        assert (
            count_kwargs["add_generation_prompt"]
            == chat_kwargs["add_generation_prompt"]
        ), (
            "add_generation_prompt diverged across phases: "
            f"count={count_kwargs['add_generation_prompt']}, "
            f"chat={chat_kwargs['add_generation_prompt']}"
        )

        # Specific contract: with partial=True forwarded, both phases use
        # continue_final_message=True (not add_generation_prompt=True).
        assert count_kwargs["continue_final_message"] is True
        assert count_kwargs["add_generation_prompt"] is False
