# SPDX-License-Identifier: Apache-2.0
"""
Tests for API utility functions.

Tests utility functions from api/utils.py and api/anthropic_utils.py for
text processing, content extraction, and format conversion.
"""

import logging

import pytest

from omlx.api.utils import (
    SPECIAL_TOKENS_PATTERN,
    _consolidate_system_messages,
    _drop_void_assistant_messages,
    _extract_multimodal_content_list,
    _merge_consecutive_roles,
    clean_output_text,
    detect_and_strip_partial,
    extract_harmony_messages,
    extract_multimodal_content,
    extract_text_content,
)
from omlx.api.anthropic_utils import (
    convert_anthropic_to_internal,
    convert_anthropic_tools_to_internal,
    convert_internal_to_anthropic_response,
    create_content_block_start_event,
    create_content_block_stop_event,
    create_error_event,
    create_input_json_delta_event,
    create_message_delta_event,
    create_message_start_event,
    create_message_stop_event,
    create_ping_event,
    create_text_delta_event,
    format_sse_event,
    map_finish_reason_to_stop_reason,
)
from omlx.api.openai_models import ContentPart, FunctionCall, Message, ToolCall
from omlx.api.anthropic_models import (
    AnthropicMessage,
    AnthropicTool,
    ContentBlockDocument,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolResult,
    ContentBlockToolUse,
    MessagesRequest,
    SystemContent,
)


class TestCleanOutputText:
    """Tests for clean_output_text function."""

    def test_clean_empty_text(self):
        """Test cleaning empty text."""
        result = clean_output_text("")
        assert result == ""

    def test_clean_none_text(self):
        """Test cleaning None text."""
        result = clean_output_text(None)
        assert result is None

    def test_clean_text_no_special_tokens(self):
        """Test cleaning text without special tokens."""
        result = clean_output_text("Hello, world!")
        assert result == "Hello, world!"

    def test_clean_im_end_token(self):
        """Test removing <|im_end|> token."""
        result = clean_output_text("Hello<|im_end|>")
        assert result == "Hello"

    def test_clean_im_start_token(self):
        """Test removing <|im_start|> token."""
        result = clean_output_text("<|im_start|>Hello")
        assert result == "Hello"

    def test_clean_endoftext_token(self):
        """Test removing <|endoftext|> token."""
        result = clean_output_text("Response<|endoftext|>")
        assert result == "Response"

    def test_clean_eot_id_token(self):
        """Test removing <|eot_id|> token."""
        result = clean_output_text("Text<|eot_id|>")
        assert result == "Text"

    def test_clean_end_token(self):
        """Test removing <|end|> token."""
        result = clean_output_text("Content<|end|>")
        assert result == "Content"

    def test_clean_header_tokens(self):
        """Test removing header tokens."""
        result = clean_output_text("<|start_header_id|>assistant<|end_header_id|>Hello")
        assert result == "assistantHello"

    def test_clean_eos_bos_tokens(self):
        """Test removing </s>, <s>, <pad> tokens."""
        result = clean_output_text("<s>Hello</s>")
        assert result == "Hello"

    def test_clean_pad_token(self):
        """Test removing <pad> token."""
        result = clean_output_text("Hello<pad>World")
        assert result == "HelloWorld"

    def test_clean_bracket_tokens(self):
        """Test removing [PAD], [SEP], [CLS] tokens."""
        result = clean_output_text("[CLS]Hello[SEP]World[PAD]")
        assert result == "HelloWorld"

    def test_clean_multiple_tokens(self):
        """Test removing multiple special tokens."""
        result = clean_output_text("<|im_start|>Hello<|im_end|><|endoftext|>")
        assert result == "Hello"

    def test_removes_think_tags(self):
        """Test that <think>...</think> tags are removed."""
        result = clean_output_text("<think>reasoning</think>Answer")
        assert "<think>" not in result
        assert "</think>" not in result
        assert "reasoning" not in result
        assert result == "Answer"

    def test_removes_multiple_think_blocks(self):
        """Test removing multiple consecutive think blocks."""
        result = clean_output_text("<think>a</think><think>b</think>Text")
        assert "<think>" not in result
        assert result == "Text"

    def test_removes_partial_think_closing(self):
        """Test removing partial </think> without opening tag."""
        result = clean_output_text("thinking content</think>Answer")
        assert "</think>" not in result
        assert result == "Answer"

    def test_removes_empty_think_blocks(self):
        """Test removing empty think blocks."""
        result = clean_output_text("<think></think>Text")
        assert result == "Text"

    def test_preserves_text_without_think_tags(self):
        """Test that normal text is unaffected."""
        result = clean_output_text("Normal response text")
        assert result == "Normal response text"

    def test_removes_think_with_newlines(self):
        """Test removing think blocks containing newlines."""
        result = clean_output_text("<think>\nreasoning\nprocess\n</think>Answer")
        assert "<think>" not in result
        assert "reasoning" not in result
        assert result == "Answer"

    def test_clean_whitespace(self):
        """Test that result is stripped."""
        result = clean_output_text("  Hello<|im_end|>  ")
        assert result == "Hello"


class TestSpecialTokensPattern:
    """Tests for SPECIAL_TOKENS_PATTERN regex."""

    def test_pattern_matches_im_tokens(self):
        """Test pattern matches <|im_*|> tokens."""
        assert SPECIAL_TOKENS_PATTERN.search("<|im_end|>")
        assert SPECIAL_TOKENS_PATTERN.search("<|im_start|>")

    def test_pattern_matches_endoftext(self):
        """Test pattern matches <|endoftext|>."""
        assert SPECIAL_TOKENS_PATTERN.search("<|endoftext|>")

    def test_pattern_matches_llama_tokens(self):
        """Test pattern matches Llama tokens."""
        assert SPECIAL_TOKENS_PATTERN.search("<|eot_id|>")
        assert SPECIAL_TOKENS_PATTERN.search("<|end|>")
        assert SPECIAL_TOKENS_PATTERN.search("<|start_header_id|>")
        assert SPECIAL_TOKENS_PATTERN.search("<|end_header_id|>")

    def test_pattern_matches_legacy_tokens(self):
        """Test pattern matches legacy tokens."""
        assert SPECIAL_TOKENS_PATTERN.search("</s>")
        assert SPECIAL_TOKENS_PATTERN.search("<s>")
        assert SPECIAL_TOKENS_PATTERN.search("<pad>")
        assert SPECIAL_TOKENS_PATTERN.search("[PAD]")
        assert SPECIAL_TOKENS_PATTERN.search("[SEP]")
        assert SPECIAL_TOKENS_PATTERN.search("[CLS]")


class TestExtractTextContent:
    """Tests for extract_text_content function."""

    def test_simple_text_message(self):
        """Test extracting simple text message."""
        messages = [Message(role="user", content="Hello")]

        result = extract_text_content(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_multiple_messages(self):
        """Test extracting multiple messages."""
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]

        result = extract_text_content(messages)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_content_array_message(self):
        """Test extracting message with content array."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            )
        ]

        result = extract_text_content(messages)

        assert len(result) == 1
        assert "Hello" in result[0]["content"]
        assert "World" in result[0]["content"]

    def test_content_array_with_pydantic(self):
        """Test extracting message with ContentPart objects."""
        messages = [
            Message(
                role="user",
                content=[
                    ContentPart(type="text", text="Hello"),
                ],
            )
        ]

        result = extract_text_content(messages)

        assert "Hello" in result[0]["content"]
        # Ensure content is a string, not a list
        assert isinstance(result[0]["content"], str)

    def test_none_content(self):
        """Test that assistant with None content and no tool_calls is dropped (void message)."""
        messages = [Message(role="assistant", content=None)]

        result = extract_text_content(messages)

        assert len(result) == 0

    def test_none_content_non_assistant_preserved(self):
        """Test that non-assistant messages with None content are preserved."""
        messages = [Message(role="user", content=None)]

        result = extract_text_content(messages)

        assert len(result) == 1
        assert result[0]["content"] == ""

    def test_tool_response_message(self):
        """Test extracting tool response message."""
        messages = [
            Message(
                role="tool",
                content='{"result": "success"}',
                tool_call_id="call_123",
            )
        ]

        result = extract_text_content(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"  # Converted to user
        assert "call_123" in result[0]["content"]
        assert "success" in result[0]["content"]

    def test_tool_response_message_with_content_part_list(self):
        """Test extracting tool response with ContentPart list content."""
        messages = [
            Message(
                role="tool",
                content=[ContentPart(type="text", text='{"result": "success"}')],
                tool_call_id="call_123",
            )
        ]

        result = extract_text_content(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"  # Converted to user
        assert "call_123" in result[0]["content"]
        assert "success" in result[0]["content"]
        # Ensure content is a string, not a list
        assert isinstance(result[0]["content"], str)

    def test_tool_response_fallback_preserves_role_boundary(self):
        """Fallback tool history must not merge into adjacent user turns."""
        messages = [
            Message(role="user", content="Before"),
            Message(
                role="tool",
                content='{"result": "success"}',
                tool_call_id="call_123",
            ),
            Message(role="user", content="After"),
        ]

        result = extract_text_content(messages)

        assert len(result) == 3
        assert result[0]["content"] == "Before"
        assert "Tool Result" in result[1]["content"]
        assert result[2]["content"] == "After"

    def test_assistant_with_tool_calls(self):
        """Test extracting assistant message with tool calls."""
        messages = [
            Message(
                role="assistant",
                content="Let me check.",
                tool_calls=[
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}',
                        }
                    }
                ],
            )
        ]

        result = extract_text_content(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "Let me check." in result[0]["content"]
        assert "get_weather" in result[0]["content"]

    def test_assistant_tool_call_fallback_preserves_role_boundary(self):
        """Fallback assistant tool turns must stay separate from later assistant text."""
        messages = [
            Message(
                role="assistant",
                content="Let me check.",
                tool_calls=[
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}',
                        }
                    }
                ],
            ),
            Message(role="assistant", content="Done."),
        ]

        result = extract_text_content(messages)

        assert len(result) == 2
        assert "get_weather" in result[0]["content"]
        assert result[1]["content"] == "Done."

    def test_developer_role_normalized_to_system(self):
        """Test that 'developer' role is normalized to 'system'."""
        messages = [
            Message(role="developer", content="You are a coding assistant."),
            Message(role="user", content="Hello"),
        ]

        result = extract_text_content(messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a coding assistant."
        assert result[1]["role"] == "user"

    def test_assistant_tool_calls_with_content_array(self):
        """Content array in assistant+tool_calls should be converted to string."""
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock(spec=[])
        mock_tokenizer.has_tool_calling = True

        messages = [
            Message(
                role="assistant",
                content=[
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "x", "name": "f", "input": {}},
                ],
                tool_calls=[{"function": {"name": "f", "arguments": "{}"}}],
            )
        ]

        result = extract_text_content(messages, tokenizer=mock_tokenizer)

        assert result[0]["content"] == "Let me check."
        assert "tool_use" not in str(result[0]["content"])

    def test_developer_role_in_harmony(self):
        """Test that 'developer' role is normalized in extract_harmony_messages."""
        messages = [
            Message(role="developer", content="You are a coding assistant."),
            Message(role="user", content="Hello"),
        ]

        result = extract_harmony_messages(messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a coding assistant."


class TestExtractTextContentReasoningReconstruction:
    """Tests that extract_text_content reassembles <think> from reasoning_content.

    External clients (e.g. Pi) receive reasoning in the OpenAI reasoning_content
    field but echo it back alongside normal content on subsequent turns.  For
    models whose chat template exposes preserve_thinking=True (Qwen 3.6+), we
    must inject <think>…</think> back into the assistant message so the
    template has something to preserve — otherwise thinking is silently dropped
    from conversation history.
    """

    def test_reasoning_and_content_merged_on_assistant(self):
        """reasoning_content + content string should produce a <think>…</think> prefix."""
        messages = [
            Message(role="assistant", reasoning_content="R", content="A"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "<think>\nR\n</think>\n\nA"

    def test_reasoning_with_none_content(self):
        """reasoning_content with content=None should still emit the <think> block."""
        messages = [
            Message(role="assistant", reasoning_content="R", content=None),
        ]
        result = extract_text_content(messages)
        # Non-empty content after reconstruction keeps the message alive.
        assert len(result) == 1
        assert result[0]["content"] == "<think>\nR\n</think>\n\n"

    def test_reasoning_with_content_list(self):
        """reasoning_content + list content should extract text parts and prefix <think>."""
        messages = [
            Message(
                role="assistant",
                reasoning_content="R",
                content=[{"type": "text", "text": "A"}],
            ),
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["content"] == "<think>\nR\n</think>\n\nA"

    def test_reasoning_on_non_assistant_passthrough(self):
        """reasoning_content on a user message must NOT trigger reconstruction."""
        messages = [
            Message(role="user", reasoning_content="R", content="A"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        # User content left untouched — no <think> wrapper.
        assert result[0]["content"] == "A"

    def test_no_reasoning_content_passthrough(self):
        """Without reasoning_content the assistant message should pass through unchanged."""
        messages = [
            Message(role="assistant", content="A"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["content"] == "A"


class TestExtractTextContentNativeReasoningContent:
    """Tests that extract_text_content forwards reasoning_content as a field
    when the caller opts into native mode.

    Qwen 3.6+ chat templates read ``message.reasoning_content`` directly.
    Passing reasoning as a separate field avoids the whitespace round-trip
    that the fallback ``<think>`` reconstruction introduces, which improves
    KV prefix cache reuse.
    """

    def test_native_mode_passes_reasoning_as_field(self):
        """Content stays clean; reasoning rides as a top-level field."""
        messages = [
            Message(role="assistant", reasoning_content="R", content="A"),
        ]
        result = extract_text_content(messages, native_reasoning_content=True)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "A"
        assert result[0]["reasoning_content"] == "R"
        # No <think> tag in content
        assert "<think>" not in result[0]["content"]

    def test_native_mode_with_none_content(self):
        """None content + reasoning_content still emits the field (and empty content)."""
        messages = [
            Message(role="assistant", reasoning_content="R", content=None),
        ]
        result = extract_text_content(messages, native_reasoning_content=True)
        assert len(result) == 1
        # Empty content but message survives because reasoning_content exists.
        # Note: _drop_void_assistant_messages may still drop this; verify it's
        # retained via the reasoning_content presence.
        assert result[0]["reasoning_content"] == "R"

    def test_native_mode_with_list_content(self):
        """List content gets flattened to text; reasoning kept separate."""
        messages = [
            Message(
                role="assistant",
                reasoning_content="R",
                content=[{"type": "text", "text": "A"}],
            ),
        ]
        result = extract_text_content(messages, native_reasoning_content=True)
        assert len(result) == 1
        assert result[0]["content"] == "A"
        assert result[0]["reasoning_content"] == "R"

    def test_native_mode_with_tool_calls(self):
        """Assistant with tool_calls + reasoning_content: field survives alongside tool_calls."""
        messages = [
            Message(
                role="assistant",
                reasoning_content="R",
                content="calling",
                tool_calls=[{"id": "c1", "function": {"name": "fn", "arguments": "{}"}}],
            ),
        ]

        class NativeToolTokenizer:
            has_tool_calling = True

        result = extract_text_content(
            messages,
            tokenizer=NativeToolTokenizer(),
            native_reasoning_content=True,
        )
        assert len(result) == 1
        assert result[0]["content"] == "calling"
        assert result[0]["reasoning_content"] == "R"
        assert result[0]["tool_calls"][0]["function"]["name"] == "fn"

    def test_native_mode_non_assistant_does_not_emit_field(self):
        """reasoning_content on a user message must not produce a field."""
        messages = [
            Message(role="user", reasoning_content="R", content="A"),
        ]
        result = extract_text_content(messages, native_reasoning_content=True)
        assert len(result) == 1
        assert result[0]["content"] == "A"
        assert "reasoning_content" not in result[0]


class TestConvertAnthropicToInternal:
    """Tests for convert_anthropic_to_internal function."""

    def test_simple_message(self):
        """Test converting simple Anthropic message."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        result = convert_anthropic_to_internal(request)

        # With no system message, should have 1 message
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_with_system_string(self):
        """Test converting request with system string."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            system="Be helpful",
        )

        result = convert_anthropic_to_internal(request)

        # Should have system message first, then user message
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"
        assert result[1]["role"] == "user"

    def test_content_blocks(self):
        """Test converting message with content blocks."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockText(text="Hello"),
                        ContentBlockText(text="World"),
                    ],
                )
            ],
        )

        result = convert_anthropic_to_internal(request)

        # First message (no system)
        assert "Hello" in result[0]["content"]
        assert "World" in result[0]["content"]

    def test_tool_use_block(self):
        """Test converting message with tool use block."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockToolUse(
                            id="toolu_123",
                            name="get_weather",
                            input={"location": "Tokyo"},
                        )
                    ],
                )
            ],
        )

        result = convert_anthropic_to_internal(request)

        assert "get_weather" in result[0]["content"]

    def test_system_billing_header_filtered(self):
        """Test that x-anthropic-billing-header system blocks are filtered out."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            system=[
                SystemContent(
                    text="x-anthropic-billing-header: cc_version=2.1.37.3a3; cc_entrypoint=cli; cch=3217b;"
                ),
                SystemContent(
                    text="You are Claude Code.",
                    cache_control={"type": "ephemeral"},
                ),
                SystemContent(
                    text="Be helpful.",
                    cache_control={"type": "ephemeral"},
                ),
            ],
        )

        result = convert_anthropic_to_internal(request)

        assert len(result) == 2  # system + user
        assert result[0]["role"] == "system"
        assert "x-anthropic-billing-header" not in result[0]["content"]
        assert "You are Claude Code." in result[0]["content"]
        assert "Be helpful." in result[0]["content"]

    def test_system_billing_header_only(self):
        """Test that system with only billing header produces no system message."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            system=[
                SystemContent(
                    text="x-anthropic-billing-header: cc_version=2.1.37.3a3; cc_entrypoint=cli; cch=abc12;"
                ),
            ],
        )

        result = convert_anthropic_to_internal(request)

        # Only user message, no system (billing header was the only block)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_tool_result_block(self):
        """Test converting message with tool result block."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockToolResult(
                            tool_use_id="toolu_123",
                            content="The weather is sunny",
                        )
                    ],
                )
            ],
        )

        result = convert_anthropic_to_internal(request)

        assert "toolu_123" in result[0]["content"]
        assert "sunny" in result[0]["content"]

    def test_native_tool_calling_preserves_structured_tool_history(self):
        """Tool use/result blocks should stay structured when tokenizer supports tools."""

        class NativeToolTokenizer:
            has_tool_calling = True

        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockText(text="Checking"),
                        ContentBlockToolUse(
                            id="toolu_123",
                            name="get_weather",
                            input={"location": "Tokyo"},
                        ),
                    ],
                ),
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockToolResult(
                            tool_use_id="toolu_123",
                            content="The weather is sunny",
                        )
                    ],
                ),
            ],
        )

        result = convert_anthropic_to_internal(
            request,
            tokenizer=NativeToolTokenizer(),
        )

        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "toolu_123"
        assert result[1]["content"] == "The weather is sunny"

    def test_tool_result_with_image_preserve_images_nonnative(self):
        """Images in tool_result content are preserved when preserve_images=True (non-native path)."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_img",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "iVBOR",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "screenshot.png",
                                },
                            ],
                        }
                    ],
                )
            ],
        )

        result = convert_anthropic_to_internal(request, preserve_images=True)

        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        image_parts = [p for p in content if p.get("type") == "image_url"]
        text_parts = [p for p in content if p.get("type") == "text"]
        assert len(image_parts) == 1
        assert "iVBOR" in image_parts[0]["image_url"]["url"]
        assert len(text_parts) == 1
        assert "toolu_img" in text_parts[0]["text"]

    def test_tool_result_with_image_no_preserve(self):
        """Images in tool_result content are NOT preserved when preserve_images=False."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_img",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "iVBOR",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "screenshot.png",
                                },
                            ],
                        }
                    ],
                )
            ],
        )

        result = convert_anthropic_to_internal(request, preserve_images=False)

        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, str)
        assert "screenshot.png" in content
        assert "iVBOR" not in content

    def test_tool_result_with_image_native_path(self):
        """Images in tool_result are preserved in native tool calling path."""

        class NativeToolTokenizer:
            has_tool_calling = True

        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockToolUse(
                            id="toolu_img",
                            name="read_file",
                            input={"path": "/tmp/screenshot.png"},
                        ),
                    ],
                ),
                AnthropicMessage(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_img",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "iVBOR",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "screenshot.png",
                                },
                            ],
                        }
                    ],
                ),
            ],
        )

        result = convert_anthropic_to_internal(
            request,
            tokenizer=NativeToolTokenizer(),
            preserve_images=True,
        )

        # assistant message with tool_calls
        assert result[0]["role"] == "assistant"
        # tool result (text only)
        assert result[1]["role"] == "tool"
        assert result[1]["content"] == "screenshot.png"
        # user message with extracted image
        assert result[2]["role"] == "user"
        content = result[2]["content"]
        assert isinstance(content, list)
        image_parts = [p for p in content if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert "iVBOR" in image_parts[0]["image_url"]["url"]

    def test_document_block_text_plain(self):
        """Test converting text/plain document block decodes content."""
        import base64

        text_data = base64.b64encode(b"Hello from document").decode()
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockDocument(
                            source={
                                "type": "base64",
                                "media_type": "text/plain",
                                "data": text_data,
                            },
                            title="notes.txt",
                        ),
                    ],
                ),
            ],
        )

        result = convert_anthropic_to_internal(request)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Hello from document" in result[0]["content"]
        assert "[Document: notes.txt]" in result[0]["content"]

    def test_document_block_pdf_placeholder(self):
        """Test converting PDF document block returns placeholder."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockDocument(
                            source={
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjQ=",
                            },
                            title="manual.pdf",
                        ),
                    ],
                ),
            ],
        )

        result = convert_anthropic_to_internal(request)

        assert len(result) == 1
        content = result[0]["content"]
        assert "manual.pdf" in content
        assert "oMLX does not provide PDF parsing" in content

    def test_thinking_block_reconstructed_as_think_tag(self):
        """Single Anthropic thinking block should be reassembled into a <think> wrapper."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockThinking(
                            type="thinking",
                            thinking="step by step",
                            signature="",
                        ),
                        ContentBlockText(text="Answer"),
                    ],
                ),
            ],
        )

        result = convert_anthropic_to_internal(request)

        assert len(result) == 1
        content = result[0]["content"]
        assert "<think>\nstep by step\n</think>" in content
        assert "Answer" in content
        # <think> must come before the answer text
        assert content.index("<think>") < content.index("Answer")

    def test_multiple_thinking_blocks_preserve_source_order(self):
        """Multiple thinking blocks must appear in Anthropic source order (regression guard).

        Earlier drafts inserted at position 0, which reversed the order of
        consecutive thinking blocks.  Appending preserves natural ordering.
        """
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockThinking(
                            type="thinking",
                            thinking="FIRST",
                            signature="",
                        ),
                        ContentBlockThinking(
                            type="thinking",
                            thinking="SECOND",
                            signature="",
                        ),
                        ContentBlockText(text="Answer"),
                    ],
                ),
            ],
        )

        result = convert_anthropic_to_internal(request)

        content = result[0]["content"]
        assert content.index("FIRST") < content.index("SECOND")
        assert content.index("SECOND") < content.index("Answer")

    def test_thinking_block_native_tool_calling_assistant(self):
        """Native-tool-calling assistant path must also reconstruct thinking blocks.

        Most Qwen 3.6+ models hit this branch (has_tool_calling=True).  Before
        the fix, the branch silently dropped thinking content, so
        preserve_thinking=True in the chat template had nothing to preserve.
        """

        class NativeToolTokenizer:
            has_tool_calling = True

        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockThinking(
                            type="thinking",
                            thinking="deliberating",
                            signature="",
                        ),
                        ContentBlockText(text="Let me check."),
                        ContentBlockToolUse(
                            id="toolu_1",
                            name="get_weather",
                            input={"location": "Tokyo"},
                        ),
                    ],
                ),
            ],
        )

        result = convert_anthropic_to_internal(
            request,
            tokenizer=NativeToolTokenizer(),
        )

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        # tool_calls still structured for native rendering
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        # <think> wrapper present in the text content
        content = result[0]["content"]
        assert "<think>\ndeliberating\n</think>" in content
        assert "Let me check." in content

    def test_document_block_mixed_with_text(self):
        """Test document block alongside text blocks."""
        import base64

        text_data = base64.b64encode(b"Doc content here").decode()
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockText(text="Please read this:"),
                        ContentBlockDocument(
                            source={
                                "type": "base64",
                                "media_type": "text/plain",
                                "data": text_data,
                            },
                        ),
                    ],
                ),
            ],
        )

        result = convert_anthropic_to_internal(request)

        assert len(result) == 1
        content = result[0]["content"]
        assert "Please read this:" in content
        assert "Doc content here" in content


class TestConvertAnthropicToInternalNativeReasoning:
    """Tests that convert_anthropic_to_internal forwards Anthropic thinking
    blocks as ``reasoning_content`` when ``native_reasoning_content=True``.

    Matches Qwen 3.6+ chat template expectations (first-class field over
    fallback ``<think>`` parsing in content).
    """

    def test_native_mode_thinking_becomes_reasoning_field(self):
        """Single thinking block surfaces as reasoning_content, not in content."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockThinking(
                            type="thinking",
                            thinking="step by step",
                            signature="",
                        ),
                        ContentBlockText(text="Answer"),
                    ],
                ),
            ],
        )

        result = convert_anthropic_to_internal(request, native_reasoning_content=True)

        assert len(result) == 1
        assert result[0]["content"] == "Answer"
        assert result[0]["reasoning_content"] == "step by step"
        assert "<think>" not in result[0]["content"]

    def test_native_mode_multiple_thinking_blocks_joined(self):
        """Multiple thinking blocks concatenate with newline into one field."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockThinking(
                            type="thinking", thinking="FIRST", signature=""
                        ),
                        ContentBlockThinking(
                            type="thinking", thinking="SECOND", signature=""
                        ),
                        ContentBlockText(text="Answer"),
                    ],
                ),
            ],
        )

        result = convert_anthropic_to_internal(request, native_reasoning_content=True)

        assert result[0]["content"] == "Answer"
        assert result[0]["reasoning_content"] == "FIRST\nSECOND"

    def test_native_mode_tool_calling_assistant(self):
        """Native-tool-calling path: tool_calls structure + reasoning_content field."""

        class NativeToolTokenizer:
            has_tool_calling = True

        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockThinking(
                            type="thinking",
                            thinking="deliberating",
                            signature="",
                        ),
                        ContentBlockText(text="Let me check."),
                        ContentBlockToolUse(
                            id="toolu_1",
                            name="get_weather",
                            input={"location": "Tokyo"},
                        ),
                    ],
                ),
            ],
        )

        result = convert_anthropic_to_internal(
            request,
            tokenizer=NativeToolTokenizer(),
            native_reasoning_content=True,
        )

        assert len(result) == 1
        assert result[0]["content"] == "Let me check."
        assert result[0]["reasoning_content"] == "deliberating"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert "<think>" not in result[0]["content"]

    def test_native_mode_no_thinking_no_field(self):
        """Assistant without thinking blocks gets no reasoning_content field."""
        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=[
                AnthropicMessage(
                    role="assistant",
                    content=[ContentBlockText(text="Just a reply")],
                ),
            ],
        )

        result = convert_anthropic_to_internal(request, native_reasoning_content=True)

        assert result[0]["content"] == "Just a reply"
        assert "reasoning_content" not in result[0]


class TestConvertAnthropicToolsToInternal:
    """Tests for convert_anthropic_tools_to_internal function."""

    def test_none_tools(self):
        """Test converting None tools."""
        result = convert_anthropic_tools_to_internal(None)
        assert result is None

    def test_empty_tools(self):
        """Test converting empty tools list."""
        result = convert_anthropic_tools_to_internal([])
        assert result is None

    def test_single_tool(self):
        """Test converting single tool."""
        tools = [
            AnthropicTool(
                name="get_weather",
                description="Get weather info",
                input_schema={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            )
        ]

        result = convert_anthropic_tools_to_internal(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather info"
        assert "parameters" in result[0]["function"]

    def test_multiple_tools(self):
        """Test converting multiple tools."""
        tools = [
            AnthropicTool(name="tool1", input_schema={}),
            AnthropicTool(name="tool2", input_schema={}),
        ]

        result = convert_anthropic_tools_to_internal(tools)

        assert len(result) == 2

    def test_tool_as_dict(self):
        """Test converting tool as dict."""
        tools = [
            {
                "name": "search",
                "description": "Search for info",
                "input_schema": {"type": "object"},
            }
        ]

        result = convert_anthropic_tools_to_internal(tools)

        assert result[0]["function"]["name"] == "search"

    def test_drops_server_side_web_search(self):
        """Anthropic web_search server-side tool is dropped (not executable)."""
        tools = [AnthropicTool(type="web_search_20250305", name="web_search")]

        result = convert_anthropic_tools_to_internal(tools)

        assert result is None

    def test_drops_server_side_code_execution(self):
        """Anthropic code_execution server-side tool is dropped."""
        tools = [
            AnthropicTool(type="code_execution_20250825", name="code_execution"),
        ]

        result = convert_anthropic_tools_to_internal(tools)

        assert result is None

    @pytest.mark.parametrize(
        "tool_type,name",
        [
            ("bash_20250124", "bash"),
            ("text_editor_20250728", "str_replace_editor"),
            ("computer_20250124", "computer"),
        ],
    )
    def test_drops_bash_text_editor_computer(self, tool_type, name):
        """Computer-use tool family (bash/text_editor/computer) is dropped."""
        tools = [AnthropicTool(type=tool_type, name=name)]

        result = convert_anthropic_tools_to_internal(tools)

        assert result is None

    def test_keeps_user_tools_drops_server_side(self):
        """Mixed: user tool is forwarded, server-side tool is dropped."""
        tools = [
            AnthropicTool(name="get_weather", input_schema={"type": "object"}),
            AnthropicTool(type="web_search_20250305", name="web_search"),
        ]

        result = convert_anthropic_tools_to_internal(tools)

        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"

    def test_drop_logs_at_info(self, caplog):
        """Dropping server-side tools emits an INFO log naming each one."""
        tools = [
            AnthropicTool(type="web_search_20250305", name="web_search"),
            AnthropicTool(type="code_execution_20250825", name="code_execution"),
        ]

        with caplog.at_level(logging.INFO, logger="omlx.api.anthropic_utils"):
            convert_anthropic_tools_to_internal(tools)

        joined = "\n".join(caplog.messages)
        assert "Dropped 2" in joined
        assert "web_search_20250305:web_search" in joined
        assert "code_execution_20250825:code_execution" in joined

    def test_unknown_type_prefix_is_treated_as_user_tool(self):
        """Unknown type with input_schema is forwarded as a user tool."""
        tools = [
            AnthropicTool(
                name="custom",
                type="unknown_kind_v1",
                input_schema={"type": "object"},
            ),
        ]

        result = convert_anthropic_tools_to_internal(tools)

        assert len(result) == 1
        assert result[0]["function"]["name"] == "custom"
        assert result[0]["function"]["parameters"] == {"type": "object"}


class TestConvertInternalToAnthropicResponse:
    """Tests for convert_internal_to_anthropic_response function."""

    def test_basic_response(self):
        """Test converting basic response."""
        result = convert_internal_to_anthropic_response(
            text="Hello!",
            model="claude-3",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
        )

        assert result.type == "message"
        assert result.role == "assistant"
        assert result.model == "claude-3"
        assert len(result.content) == 1
        assert result.content[0].text == "Hello!"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_response_with_tool_calls(self):
        """Test converting response with tool calls."""
        tool_calls = [
            ToolCall(
                id="toolu_123",
                type="function",
                function=FunctionCall(
                    name="get_weather",
                    arguments='{"location": "Tokyo"}',
                ),
            )
        ]

        result = convert_internal_to_anthropic_response(
            text="",
            model="claude-3",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="tool_calls",
            tool_calls=tool_calls,
        )

        assert result.stop_reason == "tool_use"
        tool_use_blocks = [c for c in result.content if c.type == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0].name == "get_weather"

    def test_response_empty_text(self):
        """Test converting response with empty text."""
        result = convert_internal_to_anthropic_response(
            text="",
            model="claude-3",
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="stop",
        )

        # Should have at least one content block
        assert len(result.content) >= 1

    def test_prefix_cache_disabled_legacy_shape(self):
        """Caching off: usage keeps the legacy shape (input=prompt, cache=0)."""
        result = convert_internal_to_anthropic_response(
            text="hi",
            model="claude-3",
            prompt_tokens=100,
            completion_tokens=5,
            finish_reason="stop",
            cached_tokens=40,
            prefix_cache_enabled=False,
        )

        assert result.usage.input_tokens == 100
        assert result.usage.cache_creation_input_tokens == 0
        assert result.usage.cache_read_input_tokens == 0

    def test_prefix_cache_enabled_splits_prompt(self):
        """Caching on: prompt splits into input(0) + creation + read."""
        result = convert_internal_to_anthropic_response(
            text="hi",
            model="claude-3",
            prompt_tokens=100,
            completion_tokens=5,
            finish_reason="stop",
            cached_tokens=20,
            prefix_cache_enabled=True,
        )

        assert result.usage.input_tokens == 0
        assert result.usage.cache_creation_input_tokens == 80
        assert result.usage.cache_read_input_tokens == 20


class TestMapFinishReasonToStopReason:
    """Tests for map_finish_reason_to_stop_reason function."""

    def test_stop_to_end_turn(self):
        """Test mapping stop -> end_turn."""
        result = map_finish_reason_to_stop_reason("stop", False)
        assert result == "end_turn"

    def test_length_to_max_tokens(self):
        """Test mapping length -> max_tokens."""
        result = map_finish_reason_to_stop_reason("length", False)
        assert result == "max_tokens"

    def test_tool_calls_to_tool_use(self):
        """Test mapping tool_calls -> tool_use."""
        result = map_finish_reason_to_stop_reason("tool_calls", False)
        assert result == "tool_use"

    def test_has_tool_calls_overrides(self):
        """Test that has_tool_calls overrides to tool_use."""
        result = map_finish_reason_to_stop_reason("stop", True)
        assert result == "tool_use"

    def test_none_reason(self):
        """Test mapping None reason."""
        result = map_finish_reason_to_stop_reason(None, False)
        assert result is None

    def test_unknown_reason(self):
        """Test mapping unknown reason defaults to end_turn."""
        result = map_finish_reason_to_stop_reason("unknown", False)
        assert result == "end_turn"


class TestSSEEventFormatters:
    """Tests for SSE event formatting functions."""

    def test_format_sse_event(self):
        """Test basic SSE event formatting."""
        result = format_sse_event("message_start", {"type": "message_start"})

        assert result.startswith("event: message_start\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_create_message_start_event(self):
        """Test creating message_start event."""
        result = create_message_start_event("msg_123", "claude-3", input_tokens=10)

        assert "event: message_start" in result
        assert "msg_123" in result
        assert "claude-3" in result

    def test_create_content_block_start_event_text(self):
        """Test creating content_block_start event for text."""
        result = create_content_block_start_event(0, "text")

        assert "event: content_block_start" in result
        assert '"index": 0' in result

    def test_create_content_block_start_event_tool_use(self):
        """Test creating content_block_start event for tool_use."""
        result = create_content_block_start_event(
            0, "tool_use", id="toolu_123", name="get_weather"
        )

        assert "event: content_block_start" in result
        assert "tool_use" in result

    def test_create_text_delta_event(self):
        """Test creating text delta event."""
        result = create_text_delta_event(0, "Hello")

        assert "event: content_block_delta" in result
        assert "text_delta" in result
        assert "Hello" in result

    def test_create_input_json_delta_event(self):
        """Test creating input_json_delta event."""
        result = create_input_json_delta_event(0, '{"location":')

        assert "event: content_block_delta" in result
        assert "input_json_delta" in result

    def test_create_content_block_stop_event(self):
        """Test creating content_block_stop event."""
        result = create_content_block_stop_event(0)

        assert "event: content_block_stop" in result
        assert '"index": 0' in result

    def test_create_message_delta_event(self):
        """Test creating message_delta event."""
        result = create_message_delta_event("end_turn", 10)

        assert "event: message_delta" in result
        assert "end_turn" in result
        assert '"output_tokens": 10' in result

    def test_create_message_delta_event_with_input_tokens(self):
        """Test creating message_delta event with input tokens."""
        result = create_message_delta_event("end_turn", 10, input_tokens=100)

        assert '"input_tokens": 100' in result

    def test_create_message_delta_event_prefix_cache_enabled(self):
        """With caching active, usage splits into disjoint cache fields."""
        result = create_message_delta_event(
            "end_turn",
            10,
            input_tokens=100,
            cached_tokens=30,
            prefix_cache_enabled=True,
        )

        assert '"input_tokens": 0' in result
        assert '"cache_creation_input_tokens": 70' in result
        assert '"cache_read_input_tokens": 30' in result

    def test_create_message_stop_event(self):
        """Test creating message_stop event."""
        result = create_message_stop_event()

        assert "event: message_stop" in result

    def test_create_ping_event(self):
        """Test creating ping event."""
        result = create_ping_event()

        assert "event: ping" in result

    def test_create_error_event(self):
        """Test creating error event."""
        result = create_error_event("api_error", "Something went wrong")

        assert "event: error" in result
        assert "api_error" in result
        assert "Something went wrong" in result


class TestExtractHarmonyMessages:
    """Tests for extract_harmony_messages function."""

    def test_simple_message(self):
        """Test extracting simple message."""
        messages = [Message(role="user", content="Hello")]

        result = extract_harmony_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_tool_message_preserved(self):
        """Test that tool messages preserve role and tool_call_id."""
        messages = [
            Message(
                role="tool",
                content='{"result": "success"}',
                tool_call_id="call_123",
            )
        ]

        result = extract_harmony_messages(messages)

        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"

    def test_assistant_tool_calls_preserved(self):
        """Test that assistant tool_calls are preserved."""
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}',
                        },
                    }
                ],
            )
        ]

        result = extract_harmony_messages(messages)

        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_message_with_content_part_list(self):
        """Test that tool messages with ContentPart list content are extracted properly."""
        messages = [
            Message(
                role="tool",
                content=[ContentPart(type="text", text='{"result": "success"}')],
                tool_call_id="call_123",
            )
        ]

        result = extract_harmony_messages(messages)

        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        # Harmony parses JSON content via _try_parse_json for |tojson compatibility
        assert not isinstance(result[0]["content"], list)

    def test_json_arguments_parsed(self):
        """Test that JSON arguments are parsed to dict."""
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "function": {
                            "name": "test",
                            "arguments": '{"key": "value"}',
                        },
                    }
                ],
            )
        ]

        result = extract_harmony_messages(messages)

        # Arguments should be parsed as dict for chat_template
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args["key"] == "value"

    def test_tool_content_json_parsed(self):
        """Test that tool content JSON is parsed."""
        messages = [
            Message(
                role="tool",
                content='{"result": "success"}',
                tool_call_id="call_123",
            )
        ]

        result = extract_harmony_messages(messages)

        # Content should be parsed as dict
        content = result[0]["content"]
        assert isinstance(content, dict)
        assert content["result"] == "success"


    # -- dict input tests (issue #683) --

    def test_simple_dict_message(self):
        """Plain dict messages should work (Anthropic endpoint path)."""
        messages = [{"role": "user", "content": "Hello"}]

        result = extract_harmony_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_tool_dict_message(self):
        """Tool messages as dicts should preserve role and tool_call_id."""
        messages = [
            {
                "role": "tool",
                "content": '{"result": "ok"}',
                "tool_call_id": "call_abc",
            }
        ]

        result = extract_harmony_messages(messages)

        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_abc"
        assert isinstance(result[0]["content"], dict)

    def test_assistant_tool_calls_dict(self):
        """Assistant messages with tool_calls as dicts should work."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    }
                ],
            }
        ]

        result = extract_harmony_messages(messages)

        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert isinstance(result[0]["tool_calls"][0]["function"]["arguments"], dict)

    def test_mixed_pydantic_and_dict_messages(self):
        """Mixed Pydantic Message and dict inputs should both work."""
        messages = [
            Message(role="system", content="You are helpful."),
            {"role": "user", "content": "Hi"},
        ]

        result = extract_harmony_messages(messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"


class TestConsolidateSystemMessages:
    """Tests for system message consolidation."""

    def test_no_system_messages(self):
        """No system messages: return as-is."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = _consolidate_system_messages(msgs)
        assert result == msgs

    def test_system_already_first(self):
        """System message already at position 0: no change."""
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = _consolidate_system_messages(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"

    def test_system_mid_conversation(self):
        """System message in the middle should move to front."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _consolidate_system_messages(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello"

    def test_multiple_system_messages_merged(self):
        """Multiple system messages should merge into one at position 0."""
        msgs = [
            {"role": "system", "content": "Instruction 1"},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Instruction 2"},
        ]
        result = _consolidate_system_messages(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Instruction 1\n\nInstruction 2"
        assert result[1]["role"] == "user"

    def test_empty_system_content_skipped(self):
        """System messages with empty content should be skipped."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": ""},
            {"role": "system", "content": "Real instruction"},
        ]
        result = _consolidate_system_messages(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Real instruction"

    def test_all_empty_system_returns_original(self):
        """All system messages empty: treated as no system messages."""
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ]
        result = _consolidate_system_messages(msgs)
        assert result == msgs

    def test_extract_text_content_developer_mid_conversation(self):
        """Developer role mid-conversation should consolidate to front."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="developer", content="New instructions"),
            Message(role="user", content="What now?"),
        ]
        result = extract_text_content(messages)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "New instructions"
        # user messages should be merged (consecutive after system removal)
        assert all(m["role"] != "system" for m in result[1:])

    def test_extract_text_content_preserves_tool_order(self):
        """Tool messages should keep relative order after consolidation."""
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Call tool"),
            Message(role="assistant", content="OK"),
            Message(role="system", content="Extra instruction"),
            Message(role="user", content="Continue"),
        ]
        result = extract_text_content(messages)
        assert result[0]["role"] == "system"
        assert "Be helpful" in result[0]["content"]
        assert "Extra instruction" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_system_message_with_list_content(self):
        """System message with list content should extract text without crashing."""
        msgs = [
            {"role": "system", "content": [
                {"type": "text", "text": "Be helpful"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]},
            {"role": "user", "content": "Hello"},
        ]
        result = _consolidate_system_messages(msgs)
        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], str)
        assert "Be helpful" in result[0]["content"]


class TestMergeConsecutiveRoles:
    """Tests for consecutive same-role message merging."""

    # --- Unit tests for _merge_consecutive_roles ---

    def test_empty_list(self):
        """Empty list should be returned as-is."""
        assert _merge_consecutive_roles([]) == []

    def test_single_message(self):
        """Single message should be returned unchanged."""
        msgs = [{"role": "user", "content": "Hello"}]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_consecutive_user_merged(self):
        """Two consecutive user messages should be merged."""
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "First\n\nSecond"

    def test_preserve_role_boundary_skips_merge(self):
        """Messages marked as tool-history boundaries must not merge."""
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Tool", "_preserve_role_boundary": True},
            {"role": "user", "content": "Third"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 3

    def test_three_consecutive_user_merged(self):
        """Three consecutive user messages should all merge into one."""
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "First\n\nSecond\n\nThird"

    def test_consecutive_assistant_merged(self):
        """Two consecutive assistant messages should be merged."""
        msgs = [
            {"role": "assistant", "content": "Part 1"},
            {"role": "assistant", "content": "Part 2"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "Part 1\n\nPart 2"

    def test_system_messages_not_merged(self):
        """Consecutive system messages should NOT be merged."""
        msgs = [
            {"role": "system", "content": "Instruction 1"},
            {"role": "system", "content": "Instruction 2"},
            {"role": "user", "content": "Hello"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 3

    def test_tool_messages_not_merged(self):
        """Consecutive tool messages should NOT be merged."""
        msgs = [
            {"role": "tool", "content": "Result 1", "tool_call_id": "a"},
            {"role": "tool", "content": "Result 2", "tool_call_id": "b"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 2

    def test_alternating_roles_unchanged(self):
        """Properly alternating messages should not be affected."""
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 4

    def test_empty_content_merge(self):
        """Merging with empty content should not add separators."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": ""},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_both_empty_content(self):
        """Merging two empty-content messages."""
        msgs = [
            {"role": "user", "content": ""},
            {"role": "user", "content": ""},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        assert result[0]["content"] == ""

    def test_does_not_mutate_input(self):
        """Merging should not mutate the input list."""
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        original_first = msgs[0]["content"]
        _merge_consecutive_roles(msgs)
        assert msgs[0]["content"] == original_first
        assert len(msgs) == 2

    def test_merge_list_content_with_string(self):
        """Merging list content (image) with string content should not crash."""
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]},
            {"role": "user", "content": "What do you think?"},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        types = [p["type"] for p in content]
        assert "image_url" in types
        assert "text" in types
        texts = [p["text"] for p in content if p["type"] == "text"]
        assert "Look at this" in texts
        assert "What do you think?" in texts

    def test_merge_string_with_list_content(self):
        """String content followed by list content should merge correctly."""
        msgs = [
            {"role": "user", "content": "Context text"},
            {"role": "user", "content": [
                {"type": "text", "text": "See image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,def"}},
            ]},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 3  # text + text + image_url

    def test_merge_two_list_contents(self):
        """Two list contents should concatenate."""
        msgs = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,def"}},
            ]},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2

    def test_merge_empty_string_with_list_content(self):
        """Empty string + list content should take the list content."""
        msgs = [
            {"role": "user", "content": ""},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]},
        ]
        result = _merge_consecutive_roles(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)

    # --- Integration tests through extract_text_content ---

    def test_extract_text_content_merges_consecutive_user(self):
        """extract_text_content should merge consecutive user messages."""
        messages = [
            Message(role="user", content="Page content here"),
            Message(role="user", content="What is this about?"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Page content here" in result[0]["content"]
        assert "What is this about?" in result[0]["content"]

    def test_brave_leo_pattern(self):
        """Simulate Brave Leo BYOM: system + two consecutive user messages."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Here is some context: blah blah"),
            Message(role="user", content="What does this mean?"),
        ]
        result = extract_text_content(messages)
        assert len(result) == 2  # system + merged user
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert "blah blah" in result[1]["content"]
        assert "What does this mean?" in result[1]["content"]

    def test_extract_harmony_merges_consecutive_user(self):
        """extract_harmony_messages should merge consecutive user messages."""
        messages = [
            Message(role="user", content="First"),
            Message(role="user", content="Second"),
        ]
        result = extract_harmony_messages(messages)
        assert len(result) == 1
        assert result[0]["content"] == "First\n\nSecond"


class TestExtractMultimodalContent:
    """Tests for extract_multimodal_content normalization."""

    def test_tool_message_with_content_part_list(self):
        """Test that tool messages with ContentPart list content are converted to string."""
        messages = [
            Message(
                role="tool",
                content=[ContentPart(type="text", text='{"result": "success"}')],
                tool_call_id="call_123",
            )
        ]

        result = extract_multimodal_content(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"  # Converted to user (no has_tool_calling)
        assert "call_123" in result[0]["content"]
        assert "success" in result[0]["content"]
        assert isinstance(result[0]["content"], str)

    def test_converts_input_text_and_input_image(self):
        """Responses-style input_text/input_image should normalize for VLM."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "input_text", "text": "Describe this image"},
                    {"type": "input_image", "image_url": "/tmp/example.png"},
                ],
            )
        ]

        result = extract_multimodal_content(messages)

        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Describe this image"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "/tmp/example.png"

    def test_converts_input_image_dict_shape(self):
        """input_image with image_url object should normalize to image_url."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Analyze"},
                    {
                        "type": "input_image",
                        "image_url": {"url": "https://example.com/a.png"},
                    },
                ],
            )
        ]

        result = extract_multimodal_content(messages)

        content = result[0]["content"]
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/a.png"

    def test_normalizes_image_url_from_model_dump(self):
        """image_url items from model_dump should be normalized (strip extra keys)."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "text": None,
                        "image_url": {"url": "data:image/png;base64,abc", "detail": "auto"},
                    },
                ],
            )
        ]
        result = extract_multimodal_content(messages)
        content = result[0]["content"]
        assert isinstance(content, list)
        img_part = content[1]
        assert img_part == {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        assert "text" not in img_part
        assert "detail" not in img_part.get("image_url", {})

    def test_normalizes_image_url_string_form(self):
        """image_url with string value (not nested dict) should be normalized."""
        parts = _extract_multimodal_content_list([
            {"type": "image_url", "image_url": "data:image/png;base64,abc"},
        ])
        assert len(parts) == 1
        assert parts[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}

    def test_image_url_missing_url_dropped(self):
        """image_url item with no extractable URL should be dropped."""
        parts = _extract_multimodal_content_list([
            {"type": "image_url", "image_url": None},
            {"type": "image_url"},
        ])
        assert len(parts) == 0


# =============================================================================
# Partial Mode & Name Preservation
# =============================================================================


class TestDetectAndStripPartial:
    """Tests for detect_and_strip_partial() helper."""

    def test_detects_partial_assistant(self):
        """Detects partial=True on final assistant message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "{", "partial": True},
        ]
        assert detect_and_strip_partial(messages) is True

    def test_ignores_partial_non_assistant(self):
        """partial=True on non-assistant final message returns False."""
        messages = [
            {"role": "user", "content": "Hello", "partial": True},
        ]
        assert detect_and_strip_partial(messages) is False

    def test_strips_partial_from_all_messages(self):
        """partial field is removed from every message."""
        messages = [
            {"role": "user", "content": "Hello", "partial": False},
            {"role": "assistant", "content": "{", "partial": True},
        ]
        detect_and_strip_partial(messages)
        for msg in messages:
            assert "partial" not in msg

    def test_empty_messages(self):
        """Empty message list returns False without error."""
        assert detect_and_strip_partial([]) is False

    def test_no_partial_field(self):
        """Messages without partial field return False."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert detect_and_strip_partial(messages) is False


class TestExtractTextContentPreservesNamePartial:
    """Tests that extract_text_content preserves name and partial fields."""

    def test_preserves_name_on_text_message(self):
        """name field survives extraction for text messages."""
        messages = [
            Message(role="assistant", content="Hello", name="Kimi"),
        ]
        result = extract_text_content(messages)
        assert result[0]["name"] == "Kimi"

    def test_preserves_partial_on_assistant(self):
        """partial field survives extraction for assistant messages."""
        messages = [
            Message(role="assistant", content="{", partial=True),
        ]
        result = extract_text_content(messages)
        assert result[0].get("partial") is True

    def test_no_name_when_absent(self):
        """name key is absent when not set on source message."""
        messages = [
            Message(role="user", content="Hello"),
        ]
        result = extract_text_content(messages)
        assert "name" not in result[0]

    def test_no_partial_when_false(self):
        """partial key is absent when False on source message."""
        messages = [
            Message(role="user", content="Hello"),
        ]
        result = extract_text_content(messages)
        assert "partial" not in result[0]

    def test_preserves_name_on_tool_call_message(self):
        """name field preserved on assistant message with tool_calls."""
        messages = [
            Message(
                role="assistant",
                content="Let me call a tool",
                name="Kimi",
                tool_calls=[
                    {"id": "1", "function": {"name": "search", "arguments": "{}"}}
                ],
            ),
        ]
        result = extract_text_content(messages)
        assert result[0].get("name") == "Kimi"

    def test_preserves_name_in_multimodal_extraction(self):
        """name field survives multimodal extraction."""
        messages = [
            Message(role="assistant", content="Hello", name="Kimi"),
        ]
        result = extract_multimodal_content(messages)
        assert result[0]["name"] == "Kimi"


class TestNameFieldSchemaAcceptance:
    """Tests that the `name` field is accepted by the Message schema.

    The `name` field is part of the OpenAI chat completion spec and used by
    models like Kimi K2/K2.5 for named-assistant persona rendering.  Many
    templates silently ignore it, so we cannot reliably assert on template
    output — but we CAN verify that the schema accepts it without error
    and that it survives message extraction for templates that do use it.

    On models that support it, the assistant `name` field acts as a
    probability space constraint — the same prompt produces distinctly
    different character voices depending on the name.  Models that ignore
    it simply drop the field harmlessly.

    Validated on Kimi-K2-Instruct-0905-mlx-3bit with a HHGTTG roleplay
    scenario (system: turn-based RP, user: Arthur banging on bathroom
    door, assistant: partial prefill "*" with name set).  Same prompt,
    three names, three distinct voices:

        name="Marvin the Paranoid Android":
            *door creaks open* ... "A damp towel is flung over the
            shower rail like a limp flag of surrender."

        name="Ford Prefect":
            *door slides open* ... "I seem to have mistaken this door
            for the entry to the relaxation chamber of the Starship
            Heart of Gold."

        name="Zaphod Beeblebrox":
            "Yes, yes, an hour is precisely how long it takes to
            negotiate a cease-fire between the fungal colonies behind
            the soap dish and the mildew syndicate under the sink."
    """

    def test_name_field_accepted_on_all_roles(self):
        """Message schema accepts name on user, assistant, and system roles."""
        msgs = [
            Message(
                role="system",
                content="This is a turn-based roleplaying session set in the "
                "Hitchhiker's Guide to the Galaxy universe.",
            ),
            Message(
                role="user",
                content="*bangs on the bathroom door* Oi! It's been an hour! "
                "Some of us need to use the facilities too, you know!",
                name="Arthur Dent",
            ),
            Message(
                role="assistant",
                content="*",
                name="Marvin the Paranoid Android",
                partial=True,
            ),
        ]
        # No ValidationError raised — schema accepts name on all roles
        assert msgs[1].name == "Arthur Dent"
        assert msgs[2].name == "Marvin the Paranoid Android"

    def test_name_field_survives_extraction_for_template(self):
        """name is carried through extract_text_content so templates can render it."""
        msgs = [
            Message(
                role="system",
                content="This is a turn-based roleplaying session set in the "
                "Hitchhiker's Guide to the Galaxy universe.",
            ),
            Message(
                role="user",
                content="*bangs on the bathroom door* Oi! It's been an hour! "
                "Some of us need to use the facilities too, you know!",
                name="Arthur Dent",
            ),
            Message(
                role="assistant",
                content="*",
                name="Marvin the Paranoid Android",
                partial=True,
            ),
        ]
        result = extract_text_content(msgs)

        # system message is consolidated to front; user and assistant follow
        assert result[1]["name"] == "Arthur Dent"
        assert result[2]["name"] == "Marvin the Paranoid Android"
        # partial also survives for the engine to consume
        assert result[2]["partial"] is True

    def test_name_absent_when_not_provided(self):
        """name key does not leak into message dicts when not set."""
        msgs = [Message(role="user", content="Hello")]
        result = extract_text_content(msgs)
        assert "name" not in result[0]


class TestDropVoidAssistantMessages:
    """Tests for _drop_void_assistant_messages."""

    def test_drops_empty_content_no_tool_calls(self):
        """Assistant message with empty content and no tool_calls should be dropped."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Again"},
        ]
        result = _drop_void_assistant_messages(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "user"

    def test_drops_none_content_no_tool_calls(self):
        """Assistant message with None content and no tool_calls should be dropped."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Again"},
        ]
        result = _drop_void_assistant_messages(msgs)
        assert len(result) == 2

    def test_keeps_assistant_with_content(self):
        """Assistant message with non-empty content should be kept."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Thanks"},
        ]
        result = _drop_void_assistant_messages(msgs)
        assert len(result) == 3

    def test_keeps_assistant_with_tool_calls(self):
        """Assistant message with tool_calls should be kept even if content is empty."""
        msgs = [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "function": {"name": "ls"}}],
            },
            {"role": "user", "content": "Thanks"},
        ]
        result = _drop_void_assistant_messages(msgs)
        assert len(result) == 3

    def test_preserves_other_roles(self):
        """Non-assistant messages should never be dropped."""
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
            {"role": "tool", "content": ""},
        ]
        result = _drop_void_assistant_messages(msgs)
        assert len(result) == 3

    def test_extract_text_content_drops_void_assistant(self):
        """Integration: extract_text_content should drop void assistant messages."""
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content=None),
            Message(role="user", content="Tell me about this repo"),
        ]
        result = extract_text_content(msgs)
        # The void assistant message should be dropped, and the two user
        # messages merged by _merge_consecutive_roles
        assert all(m["role"] != "assistant" or m.get("content") for m in result)

    def test_void_drop_then_merge_consecutive_users(self):
        """Dropping a void assistant between two users should merge them."""
        msgs = [
            Message(role="user", content="hello"),
            Message(role="assistant", content=None),
            Message(role="user", content="world"),
        ]
        result = extract_text_content(msgs)
        # void assistant dropped, then consecutive users merged
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "hello" in result[0]["content"]
        assert "world" in result[0]["content"]

    def test_multiple_void_assistants_merge_surrounding_users(self):
        """Multiple void assistants should be dropped and adjacent users merged."""
        msgs = [
            Message(role="user", content="a"),
            Message(role="assistant", content=None),
            Message(role="user", content="b"),
            Message(role="assistant", content="reply"),
            Message(role="user", content="c"),
            Message(role="assistant", content=None),
            Message(role="user", content="d"),
        ]
        result = extract_text_content(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert "a" in result[0]["content"] and "b" in result[0]["content"]
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "reply"
        assert result[2]["role"] == "user"
        assert "c" in result[2]["content"] and "d" in result[2]["content"]
