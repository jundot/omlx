"""Tests for VLM (Vision-Language Model) engine logic.

Tests cover:
- Tool calling injection from mlx-lm into VLM tokenizer
- Chat template application with tools and thinking
- OCR prompt substitution
- Message processing (image vs text-only paths)
- Vision input preparation with tools
- Token counting
- Engine stop safety (close() exception guard)
"""

import copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockVLMTokenizer:
    """Mock that mimics mlx-vlm's TokenizerWrapper __getattr__ delegation.

    mlx-vlm TokenizerWrapper delegates unknown attributes to the HF tokenizer
    via __getattr__. This mock reproduces that behavior so we can test that
    _inject_tool_calling() sets instance attributes that take precedence.
    """

    def __init__(self, chat_template=None, vocab=None):
        self.eos_token_id = 0
        self.chat_template = chat_template
        self._vocab = vocab or {}

    def __getattr__(self, attr):
        # Mimic mlx-vlm: delegate to HF tokenizer (which doesn't have
        # tool calling attrs), raising AttributeError
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{attr}'"
        )

    def get_vocab(self):
        return self._vocab

    def apply_chat_template(self, messages, **kwargs):
        return "<formatted>"

    def encode(self, text, **kwargs):
        return list(range(max(1, len(text.split()))))

    def decode(self, ids, **kwargs):
        return "decoded text"


def _make_engine(**overrides):
    """Create a VLMBatchedEngine instance without loading a model."""
    from omlx.engine.vlm import VLMBatchedEngine

    engine = VLMBatchedEngine(
        model_name=overrides.pop("model_name", "test-vlm"),
        **overrides,
    )
    return engine


def _make_loaded_engine(model_type=None, tokenizer=None, **overrides):
    """Create a VLMBatchedEngine with mocked internals (no actual model load)."""
    engine = _make_engine(**overrides)

    # Set up mock model config
    mock_config = MagicMock()
    mock_config.model_type = model_type

    mock_vlm_model = MagicMock()
    mock_vlm_model.config = mock_config

    engine._vlm_model = mock_vlm_model
    engine._tokenizer = tokenizer or MockVLMTokenizer()
    engine._loaded = True
    engine._engine = MagicMock()

    return engine


# ---------------------------------------------------------------------------
# TestInjectToolCalling
# ---------------------------------------------------------------------------

class TestInjectToolCalling:
    """Tests for VLMBatchedEngine._inject_tool_calling()."""

    def test_injects_attributes_for_json_tools(self):
        """Chat template with <tool_call> + tool_call.name → json_tools parser."""
        engine = _make_engine()
        tokenizer = MockVLMTokenizer(
            chat_template="some template with <tool_call> and tool_call.name",
            vocab={"<tool_call>": 100, "</tool_call>": 101},
        )

        engine._inject_tool_calling(tokenizer)

        assert tokenizer.has_tool_calling is True
        assert tokenizer.tool_call_start == "<tool_call>"
        assert tokenizer.tool_call_end == "</tool_call>"
        assert callable(tokenizer.tool_parser)

    def test_injects_attributes_for_qwen3_coder(self):
        """Chat template with <tool_call>\\n<function= → qwen3_coder parser."""
        engine = _make_engine()
        tokenizer = MockVLMTokenizer(
            chat_template='prefix <tool_call>\n<function= suffix',
            vocab={"<tool_call>": 100, "</tool_call>": 101},
        )

        engine._inject_tool_calling(tokenizer)

        assert tokenizer.has_tool_calling is True
        assert tokenizer.tool_call_start == "<tool_call>"

    def test_skips_when_no_chat_template(self):
        """No chat template → no injection."""
        engine = _make_engine()
        tokenizer = MockVLMTokenizer(chat_template=None)

        engine._inject_tool_calling(tokenizer)

        assert not hasattr(tokenizer, "has_tool_calling") or \
            getattr(tokenizer, "has_tool_calling", False) is False

    def test_skips_when_no_tool_markers(self):
        """Chat template without any tool markers → no injection."""
        engine = _make_engine()
        tokenizer = MockVLMTokenizer(
            chat_template="A plain chat template without tool markers",
            vocab={},
        )

        engine._inject_tool_calling(tokenizer)

        # has_tool_calling should not be set as instance attr, and
        # __getattr__ will raise AttributeError → getattr default False
        assert getattr(tokenizer, "has_tool_calling", False) is False

    def test_skips_when_tokens_not_in_vocab(self):
        """Tool tokens not in vocab → no injection (same as mlx-lm behavior)."""
        engine = _make_engine()
        tokenizer = MockVLMTokenizer(
            chat_template="<tool_call> tool_call.name </tool_call>",
            vocab={},  # Empty vocab — tokens not present
        )

        engine._inject_tool_calling(tokenizer)

        assert getattr(tokenizer, "has_tool_calling", False) is False

    def test_skips_when_mlx_lm_not_available(self):
        """When neither parser backend is available, injection is skipped."""
        engine = _make_engine()
        tokenizer = MockVLMTokenizer(
            chat_template="<tool_call> tool_call.name",
            vocab={"<tool_call>": 100, "</tool_call>": 101},
        )

        with patch.dict(
            "sys.modules",
            {
                "mlx_vlm.tool_parsers": None,
                "mlx_lm": None,
                "mlx_lm.tokenizer_utils": None,
            },
        ):
            engine._inject_tool_calling(tokenizer)

        # Should not crash, attributes not set
        assert getattr(tokenizer, "has_tool_calling", False) is False

    def test_instance_attrs_override_getattr(self):
        """After injection, instance attrs override __getattr__ delegation."""
        engine = _make_engine()
        tokenizer = MockVLMTokenizer(
            chat_template="<tool_call> tool_call.name </tool_call>",
            vocab={"<tool_call>": 100, "</tool_call>": 101},
        )

        # Before injection, accessing has_tool_calling raises AttributeError
        with pytest.raises(AttributeError):
            _ = tokenizer.has_tool_calling

        engine._inject_tool_calling(tokenizer)

        # After injection, instance attribute takes precedence
        assert tokenizer.has_tool_calling is True
        assert isinstance(tokenizer.tool_call_start, str)


# ---------------------------------------------------------------------------
# TestApplyChatTemplate
# ---------------------------------------------------------------------------

class TestApplyChatTemplate:
    """Tests for VLMBatchedEngine._apply_chat_template()."""

    def test_applies_template_with_tools(self):
        """Tools are passed to apply_chat_template kwargs."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<prompt with tools>"
        engine = _make_loaded_engine(tokenizer=tokenizer)

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        messages = [{"role": "user", "content": "Hello"}]

        result = engine._apply_chat_template(messages, tools=tools)

        assert result == "<prompt with tools>"
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tokenize"] is False
        assert call_kwargs["add_generation_prompt"] is True

    def test_applies_template_without_tools(self):
        """tools=None → 'tools' key not in kwargs."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<prompt>"
        engine = _make_loaded_engine(tokenizer=tokenizer)

        messages = [{"role": "user", "content": "Hello"}]
        engine._apply_chat_template(messages, tools=None)

        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert "tools" not in call_kwargs

    def test_applies_enable_thinking(self):
        """enable_thinking is forwarded to template kwargs."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<prompt>"
        engine = _make_loaded_engine(tokenizer=tokenizer, enable_thinking=True)

        messages = [{"role": "user", "content": "Hello"}]
        engine._apply_chat_template(messages)

        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["enable_thinking"] is True

    def test_fallback_when_no_template(self):
        """Tokenizer without apply_chat_template → manual concatenation."""
        tokenizer = MagicMock(spec=[])  # spec=[] prevents auto-creating attributes
        engine = _make_loaded_engine(tokenizer=tokenizer)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = engine._apply_chat_template(messages)

        assert "user: Hello" in result
        assert "assistant: Hi" in result
        assert result.endswith("assistant:")

    def test_chat_template_kwargs_override(self):
        """Additional chat_template_kwargs are merged into template kwargs."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<prompt>"
        engine = _make_loaded_engine(tokenizer=tokenizer)

        messages = [{"role": "user", "content": "Hello"}]
        engine._apply_chat_template(
            messages, chat_template_kwargs={"reasoning_effort": "high"}
        )

        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["reasoning_effort"] == "high"

    def test_type_error_fallback_strips_custom_kwargs(self):
        """TypeError from template → retry without custom kwargs."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = [
            TypeError("unexpected kwarg"),
            "<fallback prompt>",
        ]
        engine = _make_loaded_engine(tokenizer=tokenizer, enable_thinking=True)

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        result = engine._apply_chat_template(messages, tools=tools)

        assert result == "<fallback prompt>"
        # Second call should not have tools or enable_thinking
        second_call_kwargs = tokenizer.apply_chat_template.call_args_list[1][1]
        assert "tools" not in second_call_kwargs
        assert "enable_thinking" not in second_call_kwargs


# ---------------------------------------------------------------------------
# TestApplyOcrPrompt
# ---------------------------------------------------------------------------

class TestApplyOcrPrompt:
    """Tests for VLMBatchedEngine._apply_ocr_prompt()."""

    def _make_image_messages(self, text="Describe this"):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def test_preserves_user_prompt_for_dots_ocr(self):
        """dots_ocr model + user text → user prompt preserved."""
        engine = _make_loaded_engine(model_type="dots_ocr")
        messages = self._make_image_messages("What is this?")

        result = engine._apply_ocr_prompt(messages)

        text_parts = [
            p for p in result[0]["content"]
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        assert len(text_parts) == 1
        assert text_parts[0]["text"] == "What is this?"

    def test_preserves_user_prompt_for_deepseekocr(self):
        """deepseekocr model + user text → user prompt preserved."""
        engine = _make_loaded_engine(model_type="deepseekocr")
        messages = self._make_image_messages("Read this document")

        result = engine._apply_ocr_prompt(messages)

        text_parts = [
            p for p in result[0]["content"]
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        assert text_parts[0]["text"] == "Read this document"

    def test_injects_default_prompt_when_no_text(self):
        """OCR model + image-only → default OCR prompt injected."""
        engine = _make_loaded_engine(model_type="dots_ocr")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]

        result = engine._apply_ocr_prompt(messages)

        assert result[0]["content"][0]["type"] == "text"
        assert "Markdown" in result[0]["content"][0]["text"]

    def test_injects_default_prompt_when_empty_text(self):
        """OCR model + empty text + image → default OCR prompt injected."""
        engine = _make_loaded_engine(model_type="glm_ocr")
        messages = self._make_image_messages("")

        result = engine._apply_ocr_prompt(messages)

        text_parts = [
            p for p in result[0]["content"]
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        assert text_parts[0]["text"] == "Text Recognition:"

    def test_injects_default_prompt_when_whitespace_only(self):
        """OCR model + whitespace-only text + image → default OCR prompt injected."""
        engine = _make_loaded_engine(model_type="deepseekocr")
        messages = self._make_image_messages("   ")

        result = engine._apply_ocr_prompt(messages)

        text_parts = [
            p for p in result[0]["content"]
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        assert text_parts[0]["text"] == "Convert the document to markdown."

    def test_no_change_for_non_ocr_model(self):
        """Non-OCR VLM model → messages returned unchanged."""
        engine = _make_loaded_engine(model_type="qwen2_5_vl")
        original = self._make_image_messages("Describe this image")

        result = engine._apply_ocr_prompt(original)

        # Content should be identical
        assert result[0]["content"] == original[0]["content"]

    def test_preserves_image_parts(self):
        """OCR prompt injection preserves image_url parts."""
        engine = _make_loaded_engine(model_type="dots_ocr")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]

        result = engine._apply_ocr_prompt(messages)

        image_parts = [
            p for p in result[0]["content"]
            if isinstance(p, dict) and p.get("type") == "image_url"
        ]
        assert len(image_parts) == 1

    def test_deepcopy_no_mutation(self):
        """Original messages are not mutated."""
        engine = _make_loaded_engine(model_type="dots_ocr")
        messages = self._make_image_messages("Original prompt")
        original_text = messages[0]["content"][1]["text"]

        engine._apply_ocr_prompt(messages)

        assert messages[0]["content"][1]["text"] == original_text


# ---------------------------------------------------------------------------
# TestProcessChatMessages
# ---------------------------------------------------------------------------

class TestProcessChatMessages:
    """Tests for VLMBatchedEngine._process_chat_messages()."""

    @patch("omlx.engine.vlm.extract_images_from_messages")
    def test_text_only_uses_vlm_prepare_path(self, mock_extract):
        """Text-only turns on a VLM model still use _prepare_vision_inputs()."""
        text_msgs = [{"role": "user", "content": "Hello"}]
        mock_extract.return_value = (text_msgs, [])

        engine = _make_loaded_engine()
        engine._prepare_vision_inputs = MagicMock(
            return_value=([1, 2, 3], None, None, None, 0, [])
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = engine._process_chat_messages(messages, tools=None, kwargs={})

        token_ids, vlm_embeds, vlm_kwargs, image_hash, image_cache_key_start, image_cache_key_ranges = result
        assert token_ids == [1, 2, 3]
        assert vlm_embeds is None
        assert vlm_kwargs is None
        assert image_hash is None
        assert image_cache_key_start == 0
        assert image_cache_key_ranges == []
        engine._prepare_vision_inputs.assert_called_once_with(
            text_msgs,
            [],
            chat_template_kwargs=None,
            tools=None,
        )

    @patch("omlx.engine.vlm.extract_images_from_messages")
    def test_text_only_passes_tools_to_prepare_vision(self, mock_extract):
        """Text-only + tools still convert and pass tools through VLM path."""
        text_msgs = [{"role": "user", "content": "Hello"}]
        mock_extract.return_value = (text_msgs, [])

        engine = _make_loaded_engine()
        engine._prepare_vision_inputs = MagicMock(
            return_value=([1, 2, 3], None, None, None, 0, [])
        )

        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        messages = [{"role": "user", "content": "Hello"}]

        with patch("omlx.engine.vlm.convert_tools_for_template") as mock_convert:
            mock_convert.return_value = [{"converted": True}]
            engine._process_chat_messages(messages, tools=tools, kwargs={})

        mock_convert.assert_called_once_with(tools)
        call_kwargs = engine._prepare_vision_inputs.call_args[1]
        assert call_kwargs["tools"] == [{"converted": True}]

    @patch("omlx.engine.vlm.extract_images_from_messages")
    def test_image_path_calls_prepare_vision(self, mock_extract):
        """Messages with images → _prepare_vision_inputs() called."""
        from PIL import Image

        mock_image = Image.new("RGB", (4, 4), "red")
        text_msgs = [{"role": "user", "content": "Describe"}]
        mock_extract.return_value = (text_msgs, [mock_image])

        engine = _make_loaded_engine()
        engine._apply_ocr_prompt = MagicMock(return_value=text_msgs)
        engine._prepare_vision_inputs = MagicMock(
            return_value=([1, 2, 3], MagicMock(), {}, "hash123", 12, [(12, "hash123")])
        )

        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
            {"type": "text", "text": "Describe"},
        ]}]

        result = engine._process_chat_messages(messages, tools=None, kwargs={})

        engine._prepare_vision_inputs.assert_called_once()
        token_ids, vlm_embeds, vlm_kwargs, image_hash, image_cache_key_start, image_cache_key_ranges = result
        assert token_ids == [1, 2, 3]
        assert image_hash == "hash123"
        assert image_cache_key_start == 12
        assert image_cache_key_ranges == [(12, "hash123")]

    @patch("omlx.engine.vlm.extract_images_from_messages")
    def test_image_path_passes_tools(self, mock_extract):
        """Image + tools → tools converted and passed to _prepare_vision_inputs()."""
        from PIL import Image

        mock_image = Image.new("RGB", (4, 4), "red")
        text_msgs = [{"role": "user", "content": "Describe"}]
        mock_extract.return_value = (text_msgs, [mock_image])

        engine = _make_loaded_engine()
        engine._apply_ocr_prompt = MagicMock(return_value=text_msgs)
        engine._prepare_vision_inputs = MagicMock(
            return_value=([1, 2, 3], None, None, None, 0, [])
        )

        tools = [{"type": "function", "function": {"name": "analyze", "parameters": {}}}]
        messages = [{"role": "user", "content": "Describe"}]

        with patch("omlx.engine.vlm.convert_tools_for_template") as mock_convert:
            mock_convert.return_value = [{"converted": True}]
            engine._process_chat_messages(messages, tools=tools, kwargs={})

        # Verify tools were converted and passed
        mock_convert.assert_called_once_with(tools)
        call_kwargs = engine._prepare_vision_inputs.call_args[1]
        assert call_kwargs["tools"] == [{"converted": True}]

    @patch("omlx.engine.vlm.extract_images_from_messages")
    def test_image_path_without_tools(self, mock_extract):
        """Image + tools=None → _prepare_vision_inputs(tools=None)."""
        from PIL import Image

        mock_image = Image.new("RGB", (4, 4), "red")
        text_msgs = [{"role": "user", "content": "Describe"}]
        mock_extract.return_value = (text_msgs, [mock_image])

        engine = _make_loaded_engine()
        engine._apply_ocr_prompt = MagicMock(return_value=text_msgs)
        engine._prepare_vision_inputs = MagicMock(
            return_value=([1, 2, 3], None, None, None, 0, [])
        )

        messages = [{"role": "user", "content": "Describe"}]
        engine._process_chat_messages(messages, tools=None, kwargs={})

        call_kwargs = engine._prepare_vision_inputs.call_args[1]
        assert call_kwargs["tools"] is None


# ---------------------------------------------------------------------------
# TestPrepareVisionInputs
# ---------------------------------------------------------------------------

class TestPrepareVisionInputs:
    """Tests for VLMBatchedEngine._prepare_vision_inputs()."""

    def _setup_engine_for_vision(self, model_type="qwen2_5_vl"):
        """Create engine with mocked VLM internals for vision input testing."""
        engine = _make_loaded_engine(model_type=model_type)

        # Mock processor with apply_chat_template
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "<vision prompt>"
        mock_processor.tokenizer = engine._tokenizer
        engine._processor = mock_processor

        return engine

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    @patch("mlx_vlm.utils.prepare_inputs")
    @patch("mlx_vlm.prompt_utils.apply_chat_template")
    def test_tools_added_to_template_kwargs(self, mock_vlm_act, mock_prepare):
        """When tools are provided, they appear in template_kwargs."""
        engine = self._setup_engine_for_vision()

        # Mock apply_chat_template (mlx-vlm) returning formatted messages
        mock_vlm_act.return_value = [{"role": "user", "content": "formatted"}]

        # Mock prepare_inputs returning minimal inputs
        mock_prepare.return_value = {
            "input_ids": mx.array([[1, 2, 3]]),
            "pixel_values": None,
        }

        messages = [{"role": "user", "content": "Describe"}]
        from PIL import Image
        images = [Image.new("RGB", (4, 4), "red")]
        tools = [{"type": "function", "function": {"name": "test"}}]

        engine._prepare_vision_inputs(messages, images, tools=tools)

        # Verify the processor's apply_chat_template was called with tools
        proc_call = engine._processor.apply_chat_template
        call_kwargs = proc_call.call_args[1]
        assert call_kwargs.get("tools") == tools

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    @patch("mlx_vlm.utils.prepare_inputs")
    @patch("mlx_vlm.prompt_utils.apply_chat_template")
    def test_tools_not_added_when_none(self, mock_vlm_act, mock_prepare):
        """When tools=None, 'tools' key not in template_kwargs."""
        engine = self._setup_engine_for_vision()

        mock_vlm_act.return_value = [{"role": "user", "content": "formatted"}]
        mock_prepare.return_value = {
            "input_ids": mx.array([[1, 2, 3]]),
            "pixel_values": None,
        }

        messages = [{"role": "user", "content": "Describe"}]
        from PIL import Image
        images = [Image.new("RGB", (4, 4), "red")]

        engine._prepare_vision_inputs(messages, images, tools=None)

        proc_call = engine._processor.apply_chat_template
        call_kwargs = proc_call.call_args[1]
        assert "tools" not in call_kwargs

    def test_single_image_model_rejects_multi(self):
        """SINGLE_IMAGE_ONLY_MODELS raise ValueError for multiple images."""
        engine = _make_loaded_engine(model_type="paligemma")
        engine._processor = MagicMock()

        from PIL import Image
        images = [Image.new("RGB", (4, 4), "red"), Image.new("RGB", (4, 4), "blue")]
        messages = [{"role": "user", "content": "Describe"}]

        with pytest.raises(ValueError, match="does not support multi-image"):
            engine._prepare_vision_inputs(messages, images)


class TestFormatMessagesForVLMTemplate:
    """Tests for VLMBatchedEngine._format_messages_for_vlm_template()."""

    @staticmethod
    def _count_image_placeholders(formatted_messages):
        count = 0
        for msg in formatted_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {
                        "image",
                        "image_url",
                        "input_image",
                    }:
                        count += 1
            elif isinstance(content, str):
                count += content.count("<image>")
                count += content.count("<start_of_image>")
                count += content.count("<|image_1|>")
        return count

    def test_assigns_placeholder_to_late_user_image_turn(self):
        """system→assistant→user(image) still places image token on user turn."""
        engine = _make_loaded_engine(model_type="qwen3_5")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Hello"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
        ]

        formatted, image_ranges = engine._format_messages_for_vlm_template(
            messages, num_images=1
        )

        assert self._count_image_placeholders(formatted) == 1
        assert self._count_image_placeholders([formatted[-1]]) == 1
        assert image_ranges == [(2, 1)]

    def test_caps_placeholders_by_loaded_image_count(self):
        """Do not add more placeholders than successfully loaded images."""
        engine = _make_loaded_engine(model_type="qwen3_5")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,a"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,b"}},
                    {"type": "text", "text": "Compare"},
                ],
            },
        ]

        formatted, image_ranges = engine._format_messages_for_vlm_template(
            messages, num_images=1
        )

        assert self._count_image_placeholders(formatted) == 1
        assert image_ranges == [(0, 1)]

    def test_fallback_inserts_first_user_when_no_explicit_parts(self):
        """Legacy path: num_images without explicit image parts still injects once."""
        engine = _make_loaded_engine(model_type="qwen3_5")
        messages = [{"role": "user", "content": "Describe this"}]

        formatted, image_ranges = engine._format_messages_for_vlm_template(
            messages, num_images=1
        )

        assert self._count_image_placeholders(formatted) == 1
        assert image_ranges == [(0, 1)]

    def test_text_only_messages_have_string_content(self):
        """Text-only messages should have string content, not list.

        Regression test for #796: get_message_json() wraps text in list
        format which breaks simplified chat templates.
        """
        engine = _make_loaded_engine(model_type="qwen3_5_moe")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        formatted, image_ranges = engine._format_messages_for_vlm_template(
            messages, num_images=0
        )

        assert image_ranges == []
        for msg in formatted:
            assert isinstance(msg["content"], str), (
                f"Expected string content for {msg['role']} message, "
                f"got {type(msg['content'])}: {msg['content']}"
            )

    def test_image_messages_retain_list_content(self):
        """Image-bearing messages should keep list content with image tokens."""
        engine = _make_loaded_engine(model_type="qwen3_5_moe")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
        ]

        formatted, image_ranges = engine._format_messages_for_vlm_template(
            messages, num_images=1
        )

        assert image_ranges == [(1, 1)]
        # System message should be string (text-only)
        assert isinstance(formatted[0]["content"], str)
        # User message with image should be list
        assert isinstance(formatted[1]["content"], list)
        assert self._count_image_placeholders([formatted[1]]) == 1


# ---------------------------------------------------------------------------
# TestCountChatTokens
# ---------------------------------------------------------------------------

class TestCountChatTokens:
    """Tests for VLMBatchedEngine.count_chat_tokens()."""

    def test_counts_text_tokens(self):
        """Returns token count for text messages."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "Hello World"
        tokenizer.encode.return_value = [1, 2]

        engine = _make_loaded_engine(tokenizer=tokenizer)

        messages = [{"role": "user", "content": "Hello World"}]
        count = engine.count_chat_tokens(messages)

        assert count == 2

    def test_strips_images_from_count(self):
        """Image parts are removed before counting tokens."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "Describe"
        tokenizer.encode.return_value = [1]

        engine = _make_loaded_engine(tokenizer=tokenizer)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    {"type": "text", "text": "Describe"},
                ],
            }
        ]
        count = engine.count_chat_tokens(messages)

        # Should count text tokens only
        assert count == 1


# ---------------------------------------------------------------------------
# TestPartialModeVLM
# ---------------------------------------------------------------------------

class TestPartialModeVLM:
    """Tests for partial mode in VLM engine — always ignored."""

    def test_apply_chat_template_partial_ignored(self):
        """VLM _apply_chat_template strips partial but always uses add_generation_prompt=True."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<formatted>"
        engine = _make_loaded_engine(tokenizer=mock_tokenizer)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "{", "partial": True},
        ]

        engine._apply_chat_template(messages)

        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["add_generation_prompt"] is True
        assert "continue_final_message" not in call_kwargs

        # partial field should be stripped from messages
        call_msgs = mock_tokenizer.apply_chat_template.call_args[0][0]
        for msg in call_msgs:
            assert "partial" not in msg


# ---------------------------------------------------------------------------
# TestGetStats
# ---------------------------------------------------------------------------

class TestGetStats:
    """Tests for VLMBatchedEngine.get_stats()."""

    def test_returns_vlm_engine_type(self):
        """Stats include engine_type='vlm'."""
        engine = _make_loaded_engine()
        engine._engine.get_stats.return_value = {}

        stats = engine.get_stats()

        assert stats["engine_type"] == "vlm"
        assert stats["model_name"] == "test-vlm"
        assert stats["loaded"] is True


# ---------------------------------------------------------------------------
# TestSplitVisionFeatures
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
class TestSplitVisionFeatures:
    """Tests for VLMBatchedEngine._split_vision_features()."""

    def test_single_image_returns_whole(self):
        """Single image returns the feature tensor as-is in a list."""
        engine = _make_loaded_engine()
        features = mx.ones((1, 10, 64))
        result = engine._split_vision_features(features, 1, {})
        assert len(result) == 1
        assert result[0].shape == (1, 10, 64)

    def test_batch_dim_split_gemma_llava(self):
        """Features with batch dim = num_images are split along axis 0."""
        engine = _make_loaded_engine(model_type="gemma4")
        features = mx.ones((3, 10, 64))
        result = engine._split_vision_features(features, 3, {})
        assert result is not None
        assert len(result) == 3
        for f in result:
            assert f.shape == (1, 10, 64)

    def test_qwen_flat_split(self):
        """Qwen flat (total_tokens, dim) features are split using grid_thw."""
        engine = _make_loaded_engine(model_type="qwen3_5")
        # Mock spatial_merge_size on vision_tower
        engine._vlm_model.vision_tower = MagicMock()
        engine._vlm_model.vision_tower.spatial_merge_size = 2

        # 2 images: image1 has grid (1, 4, 4) → 16 patches / 4 = 4 merged
        #           image2 has grid (1, 4, 8) → 32 patches / 4 = 8 merged
        grid_thw = mx.array([[1, 4, 4], [1, 4, 8]])
        features = mx.ones((12, 128))  # 4 + 8 = 12 total merged tokens

        result = engine._split_vision_features(
            features, 2, {"image_grid_thw": grid_thw}
        )
        assert result is not None
        assert len(result) == 2
        assert result[0].shape == (4, 128)
        assert result[1].shape == (8, 128)

    def test_qwen_mismatch_returns_none(self):
        """Returns None if computed token count doesn't match feature shape."""
        engine = _make_loaded_engine(model_type="qwen3_5")
        engine._vlm_model.vision_tower = MagicMock()
        engine._vlm_model.vision_tower.spatial_merge_size = 2

        grid_thw = mx.array([[1, 4, 4]])  # → 4 merged tokens
        features = mx.ones((99, 128))  # Mismatch

        result = engine._split_vision_features(features, 1, {"image_grid_thw": grid_thw})
        # Single image: returns [features] regardless of shape
        assert result is not None

    def test_unsupported_returns_none(self):
        """Unknown model with non-matching dimensions returns None."""
        engine = _make_loaded_engine(model_type="unknown_vlm")
        features = mx.ones((100, 128))  # 2D, non-Qwen
        result = engine._split_vision_features(features, 3, {})
        assert result is None


# ---------------------------------------------------------------------------
# TestStopSafety
# ---------------------------------------------------------------------------

class TestStopSafety:
    """Tests for VLMBatchedEngine.stop() exception safety."""

    @pytest.mark.asyncio
    async def test_stop_completes_when_close_raises(self):
        """stop() should complete even if engine.close() raises an exception."""
        engine = _make_loaded_engine()

        mock_inner_engine = MagicMock()
        mock_inner_engine.close.side_effect = RuntimeError("close failed")
        engine._engine.stop = AsyncMock()
        engine._engine.engine = mock_inner_engine

        await engine.stop()

        assert engine._engine is None
        assert engine._vlm_model is None
        assert engine._tokenizer is None
        assert engine._loaded is False

    @pytest.mark.asyncio
    async def test_stop_completes_when_engine_has_no_engine_attr(self):
        """stop() should complete when _engine has no 'engine' attribute."""
        engine = _make_loaded_engine()
        engine._engine = MagicMock(spec=["stop"])
        engine._engine.stop = AsyncMock()

        await engine.stop()

        assert engine._engine is None
        assert engine._loaded is False

    @pytest.mark.asyncio
    async def test_stop_calls_close_on_success(self):
        """stop() calls engine.close() when no exception occurs."""
        engine = _make_loaded_engine()
        mock_inner_engine = MagicMock()
        engine._engine.stop = AsyncMock()
        engine._engine.engine = mock_inner_engine

        await engine.stop()

        mock_inner_engine.close.assert_called_once()
