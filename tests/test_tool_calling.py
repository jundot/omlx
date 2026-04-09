# SPDX-License-Identifier: Apache-2.0
"""
Tests for tool calling parsing and conversion utilities.

Tests JSON schema validation, JSON extraction, and tool conversion functions.
"""

import json
import pytest

from unittest.mock import MagicMock

from omlx.api.tool_calling import (
    ToolCallStreamFilter,
    _gemma4_args_to_json_robust,
    _parse_gemma4_tool_call_fallback,
    build_json_system_prompt,
    convert_tools_for_template,
    extract_json_from_text,
    extract_tool_calls_with_thinking,
    format_tool_call_for_message,
    parse_json_output,
    parse_tool_calls,
    parse_tool_calls_with_thinking_fallback,
    validate_json_schema,
)
from omlx.api.openai_models import (
    FunctionCall,
    ResponseFormat,
    ResponseFormatJsonSchema,
    ToolCall,
    ToolDefinition,
)


class TestValidateJsonSchema:
    """Tests for validate_json_schema function."""

    def test_valid_simple_object(self):
        """Test validation of simple valid object."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        data = {"name": "John"}

        is_valid, error = validate_json_schema(data, schema)

        assert is_valid is True
        assert error is None

    def test_invalid_missing_required(self):
        """Test validation fails for missing required field."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        data = {}

        is_valid, error = validate_json_schema(data, schema)

        assert is_valid is False
        assert error is not None
        assert "name" in error.lower() or "required" in error.lower()

    def test_invalid_wrong_type(self):
        """Test validation fails for wrong type."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
            },
        }
        data = {"age": "not a number"}

        is_valid, error = validate_json_schema(data, schema)

        assert is_valid is False
        assert error is not None

    def test_valid_nested_object(self):
        """Test validation of nested object."""
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
            },
        }
        data = {"person": {"name": "John"}}

        is_valid, error = validate_json_schema(data, schema)

        assert is_valid is True

    def test_valid_array(self):
        """Test validation of array."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        data = ["a", "b", "c"]

        is_valid, error = validate_json_schema(data, schema)

        assert is_valid is True

    def test_invalid_array_item_type(self):
        """Test validation fails for wrong array item type."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        data = ["a", 123, "c"]

        is_valid, error = validate_json_schema(data, schema)

        assert is_valid is False

    def test_valid_with_additional_properties(self):
        """Test validation with additional properties."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        data = {"name": "John", "extra": "field"}

        is_valid, error = validate_json_schema(data, schema)

        # By default, additional properties are allowed
        assert is_valid is True

    def test_empty_schema(self):
        """Test validation with empty schema."""
        schema = {}
        data = {"anything": "goes"}

        is_valid, error = validate_json_schema(data, schema)

        # Empty schema allows anything
        assert is_valid is True


class TestExtractJsonFromText:
    """Tests for extract_json_from_text function."""

    def test_pure_json_object(self):
        """Test extracting pure JSON object."""
        text = '{"name": "John", "age": 30}'

        result = extract_json_from_text(text)

        assert result == {"name": "John", "age": 30}

    def test_pure_json_array(self):
        """Test extracting pure JSON array."""
        text = "[1, 2, 3]"

        result = extract_json_from_text(text)

        assert result == [1, 2, 3]

    def test_json_with_whitespace(self):
        """Test extracting JSON with leading/trailing whitespace."""
        text = '   {"name": "John"}   '

        result = extract_json_from_text(text)

        assert result == {"name": "John"}

    def test_json_in_markdown_code_block(self):
        """Test extracting JSON from markdown code block."""
        text = """Here is the result:
```json
{"name": "John", "age": 30}
```
"""

        result = extract_json_from_text(text)

        assert result == {"name": "John", "age": 30}

    def test_json_in_plain_code_block(self):
        """Test extracting JSON from plain code block."""
        text = """Result:
```
{"status": "ok"}
```
"""

        result = extract_json_from_text(text)

        assert result == {"status": "ok"}

    def test_json_embedded_in_text(self):
        """Test extracting JSON embedded in text."""
        text = 'The response is {"result": true} and that is all.'

        result = extract_json_from_text(text)

        assert result == {"result": True}

    def test_no_json_found(self):
        """Test when no valid JSON is found."""
        text = "This is just plain text without any JSON."

        result = extract_json_from_text(text)

        assert result is None

    def test_invalid_json(self):
        """Test when JSON is malformed."""
        text = '{"name": "John", age: 30}'  # Missing quotes on key

        result = extract_json_from_text(text)

        # Should return None for invalid JSON
        assert result is None

    def test_nested_json(self):
        """Test extracting nested JSON."""
        text = '{"outer": {"inner": {"deep": "value"}}}'

        result = extract_json_from_text(text)

        assert result["outer"]["inner"]["deep"] == "value"

    def test_json_with_array(self):
        """Test extracting JSON with arrays."""
        text = '{"items": [1, 2, 3]}'

        result = extract_json_from_text(text)

        assert result["items"] == [1, 2, 3]

    def test_json_with_unicode(self):
        """Test extracting JSON with Unicode."""
        text = '{"message": "Hello, 世界!"}'

        result = extract_json_from_text(text)

        assert result["message"] == "Hello, 世界!"


class TestParseJsonOutput:
    """Tests for parse_json_output function."""

    def test_no_response_format(self):
        """Test with no response format."""
        text = "Just some text"

        cleaned, parsed, is_valid, error = parse_json_output(text, None)

        assert cleaned == text
        assert parsed is None
        assert is_valid is True
        assert error is None

    def test_text_format(self):
        """Test with text response format."""
        text = "Just some text"
        response_format = {"type": "text"}

        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)

        assert cleaned == text
        assert parsed is None
        assert is_valid is True

    def test_json_object_format_valid(self):
        """Test with json_object format and valid JSON."""
        text = '{"name": "John"}'
        response_format = {"type": "json_object"}

        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)

        assert is_valid is True
        assert parsed == {"name": "John"}
        assert error is None

    def test_json_object_format_invalid(self):
        """Test with json_object format and invalid JSON."""
        text = "This is not JSON"
        response_format = {"type": "json_object"}

        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)

        assert is_valid is False
        assert parsed is None
        assert error is not None

    def test_json_schema_format_valid(self):
        """Test with json_schema format and valid JSON."""
        text = '{"name": "John"}'
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
        }

        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)

        assert is_valid is True
        assert parsed == {"name": "John"}
        assert error is None

    def test_json_schema_format_invalid_schema(self):
        """Test with json_schema format and schema validation failure."""
        text = '{"age": 30}'  # Missing required "name" field
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
        }

        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)

        assert is_valid is False
        assert parsed == {"age": 30}  # Parsed but invalid
        assert error is not None
        assert "validation failed" in error.lower()

    def test_json_schema_with_pydantic_model(self):
        """Test with ResponseFormat Pydantic model."""
        text = '{"message": "hello"}'
        response_format = ResponseFormat(
            type="json_schema",
            json_schema=ResponseFormatJsonSchema(
                name="greeting",
                schema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                    },
                },
            ),
        )

        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)

        assert is_valid is True
        assert parsed == {"message": "hello"}

    def test_json_from_code_block(self):
        """Test extracting JSON from code block."""
        text = """```json
{"result": true}
```"""
        response_format = {"type": "json_object"}

        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)

        assert is_valid is True
        assert parsed == {"result": True}


class TestBuildJsonSystemPrompt:
    """Tests for build_json_system_prompt function."""

    def test_no_response_format(self):
        """Test with no response format."""
        result = build_json_system_prompt(None)

        assert result is None

    def test_text_format(self):
        """Test with text format."""
        result = build_json_system_prompt({"type": "text"})

        assert result is None

    def test_json_object_format(self):
        """Test with json_object format."""
        result = build_json_system_prompt({"type": "json_object"})

        assert result is not None
        assert "JSON" in result
        assert "valid" in result.lower()

    def test_json_schema_format(self):
        """Test with json_schema format."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "description": "A person object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
            },
        }

        result = build_json_system_prompt(response_format)

        assert result is not None
        assert "person" in result
        assert "A person object" in result

    def test_json_schema_format_with_pydantic(self):
        """Test with ResponseFormat Pydantic model."""
        response_format = ResponseFormat(
            type="json_schema",
            json_schema=ResponseFormatJsonSchema(
                name="output",
                description="Output format",
                schema={"type": "object"},
            ),
        )

        result = build_json_system_prompt(response_format)

        assert result is not None
        assert "output" in result


class TestConvertToolsForTemplate:
    """Tests for convert_tools_for_template function."""

    def test_none_tools(self):
        """Test with None tools."""
        result = convert_tools_for_template(None)

        assert result is None

    def test_empty_tools(self):
        """Test with empty tools list."""
        result = convert_tools_for_template([])

        assert result is None

    def test_dict_tools(self):
        """Test converting tools from dict format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                },
            }
        ]

        result = convert_tools_for_template(tools)

        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather info"

    def test_pydantic_tools(self):
        """Test converting tools from Pydantic models."""
        tools = [
            ToolDefinition(
                type="function",
                function={
                    "name": "search",
                    "description": "Search for info",
                    "parameters": {"type": "object"},
                },
            )
        ]

        result = convert_tools_for_template(tools)

        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "search"

    def test_multiple_tools(self):
        """Test converting multiple tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool1",
                    "description": "First tool",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool2",
                    "description": "Second tool",
                    "parameters": {},
                },
            },
        ]

        result = convert_tools_for_template(tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool1"
        assert result[1]["function"]["name"] == "tool2"

    def test_non_function_tools_ignored(self):
        """Test that non-function tools are ignored."""
        tools = [
            {"type": "other", "data": "something"},
            {
                "type": "function",
                "function": {"name": "valid", "parameters": {}},
            },
        ]

        result = convert_tools_for_template(tools)

        assert len(result) == 1
        assert result[0]["function"]["name"] == "valid"

    def test_tool_without_function_ignored(self):
        """Test that tools without function are ignored."""
        tools = [
            {"type": "function"},  # Missing function field
        ]

        result = convert_tools_for_template(tools)

        assert result is None

    def test_default_parameters(self):
        """Test that missing parameters get default value."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "no_params",
                },
            },
        ]

        result = convert_tools_for_template(tools)

        assert result is not None
        assert result[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }


class TestFormatToolCallForMessage:
    """Tests for format_tool_call_for_message function."""

    def test_format_tool_call(self):
        """Test formatting a tool call for message."""
        tool_call = ToolCall(
            id="call_abc123",
            type="function",
            function=FunctionCall(
                name="get_weather",
                arguments='{"location": "Tokyo"}',
            ),
        )

        result = format_tool_call_for_message(tool_call)

        assert result["id"] == "call_abc123"
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["arguments"] == '{"location": "Tokyo"}'

    def test_format_tool_call_empty_arguments(self):
        """Test formatting tool call with empty arguments."""
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(
                name="no_args",
                arguments="{}",
            ),
        )

        result = format_tool_call_for_message(tool_call)

        assert result["function"]["arguments"] == "{}"


def _make_tokenizer(tool_call_start=""):
    """Create a mock tokenizer with optional tool_call_start."""
    tok = MagicMock(spec=[])
    if tool_call_start:
        tok.tool_call_start = tool_call_start
    return tok


class TestToolCallStreamFilter:
    """Tests for ToolCallStreamFilter."""

    def test_no_marker_passthrough(self):
        """Without tokenizer marker, fallback envelopes are still active."""
        f = ToolCallStreamFilter(_make_tokenizer())
        assert f.active
        assert f.feed("hello world") == "hello world"
        assert f.finish() == ""

    def test_active_property(self):
        """Filter is active when marker is non-empty."""
        f = ToolCallStreamFilter(_make_tokenizer("<tool_call>"))
        assert f.active

    def test_text_without_marker(self):
        """Marker exists but text has none -> all text passes through."""
        f = ToolCallStreamFilter(_make_tokenizer("<tool_call>"))
        result = f.feed("Hello world!")
        result += f.finish()
        assert result == "Hello world!"

    def test_marker_in_middle(self):
        """Text before marker passes, text after is suppressed."""
        f = ToolCallStreamFilter(_make_tokenizer("<tool_call>"))
        result = f.feed('Answer<tool_call>{"name":"func"}')
        assert result == "Answer"
        assert f.feed("more text") == ""
        assert f.finish() == ""

    def test_marker_split_across_feeds(self):
        """Marker split across two feed() calls."""
        f = ToolCallStreamFilter(_make_tokenizer("<tool_call>"))
        r1 = f.feed("Hello <tool_")
        r2 = f.feed("call>JSON data")
        assert r1 + r2 == "Hello "

    def test_false_partial_match(self):
        """Text that starts like marker but doesn't match."""
        f = ToolCallStreamFilter(_make_tokenizer("<tool_call>"))
        result = f.feed("Use <tool_tip> for help")
        result += f.finish()
        assert result == "Use <tool_tip> for help"

    def test_marker_at_start(self):
        """Marker at the very start of text."""
        f = ToolCallStreamFilter(_make_tokenizer("<tool_call>"))
        assert f.feed('<tool_call>{"name":"x"}') == ""
        assert f.finish() == ""

    def test_empty_feed(self):
        """Empty string input."""
        f = ToolCallStreamFilter(_make_tokenizer("<tool_call>"))
        assert f.feed("") == ""

    def test_multiple_small_feeds(self):
        """Character-by-character feeding."""
        f = ToolCallStreamFilter(_make_tokenizer("<tool_call>"))
        text = "Hi<tool_call>data"
        result = ""
        for ch in text:
            result += f.feed(ch)
        result += f.finish()
        assert result == "Hi"

    def test_finish_drops_partial_marker_suffix_under_strict_mode(self):
        """finish() suppresses unresolved control-marker suffixes."""
        f = ToolCallStreamFilter(_make_tokenizer("<tool_call>"))
        # Feed text shorter than marker - all buffered
        r1 = f.feed("<tool")
        r2 = f.finish()
        assert r1 + r2 == ""

    def test_suppressing_blocks_finish(self):
        """An unresolved open envelope keeps buffered control text suppressed at finish()."""
        f = ToolCallStreamFilter(_make_tokenizer())
        f.feed("text<tool_call>rest")
        assert f.finish() == ""

    def test_bracket_literal_passthrough(self):
        """Bracket-style literal text should pass through unchanged."""
        f = ToolCallStreamFilter(_make_tokenizer())
        result = f.feed("Heads up: [Calling tool:")
        result += f.feed(" maybe later]")
        result += f.finish()
        assert result == "Heads up: [Calling tool: maybe later]"

    def test_bracket_tool_call_suppresses_when_complete(self):
        """A complete parseable bracket envelope should be suppressed."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("Lead in [Calling tool:")
        r2 = f.feed(' get_weather({"city":"SF"})]')
        assert r1 == "Lead in "
        assert r2 == ""
        assert f.finish() == ""

    def test_bracket_tool_call_suppresses_envelope_but_preserves_trailing_text(self):
        """Suppression must not truncate prose that follows a complete bracket envelope."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("Before [Calling tool:")
        r2 = f.feed(' get_weather({"city":"SF"})] After text')
        r3 = f.finish()
        assert r1 + r2 + r3 == "Before  After text"

    def test_xml_tool_call_suppresses_envelope_but_preserves_trailing_text(self):
        """Raw XML envelope suppression should resume normal text after close tag."""
        f = ToolCallStreamFilter(_make_tokenizer("<tool_call>"))
        result = f.feed(
            'Before <tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call> After'
        )
        result += f.finish()
        assert result == "Before  After"

    def test_bracket_tool_call_with_hyphen_name_suppresses_when_complete(self):
        """Bracket detector should treat hyphenated tool names as valid calls."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("Lead in [Calling tool:")
        r2 = f.feed(' get-weather({"city":"SF"})] tail')
        r3 = f.finish()
        assert r1 + r2 + r3 == "Lead in  tail"

    def test_long_unresolved_bracket_envelope_does_not_leak_control_markup(self):
        """Long unresolved bracket calls should stay buffered until envelope is complete."""
        f = ToolCallStreamFilter(_make_tokenizer())
        long_note = "x" * 320
        prefix = 'Before [Calling tool: get_weather({"note":"'
        chunk1 = prefix + long_note
        chunk2 = '"})] After'

        r1 = f.feed(chunk1)
        r2 = f.feed(chunk2)
        r3 = f.finish()
        result = r1 + r2 + r3

        assert "[Calling tool:" not in result
        assert result == "Before  After"

    def test_finish_drops_unresolved_bracket_control_fragment(self):
        """Unresolved bracket control fragments should be suppressed at finish()."""
        f = ToolCallStreamFilter(_make_tokenizer())
        result = f.feed('Before [Calling tool: get_weather({"city":"SF"}')
        result += f.finish()
        assert result == "Before "

    def test_later_parseable_bracket_envelope_is_detected_after_literal_bracket(self):
        """A literal early bracket marker must not mask a later parseable envelope."""
        f = ToolCallStreamFilter(_make_tokenizer())
        text = (
            "literal [Calling tool: maybe later] and then "
            '[Calling tool: get_weather({"city":"SF"})] done'
        )
        result = f.feed(text)
        result += f.finish()
        assert result == "literal [Calling tool: maybe later] and then  done"

    def test_unresolved_bracket_prefix_before_parseable_envelope_does_not_leak_marker(
        self,
    ):
        """An unresolved early bracket prefix must not leak when a later call is parseable."""
        f = ToolCallStreamFilter(_make_tokenizer())
        text = (
            "Before [Calling tool: unfinished and then "
            '[Calling tool: get_weather({"city":"NY"})] done'
        )
        result = f.feed(text)
        result += f.finish()
        assert "[Calling tool:" not in result
        assert result == "Before  unfinished and then  done"

    def test_incremental_feeding_unresolved_bracket_split_across_chunks(self):
        """Bracket prefix split across feed() chunks must still be detected."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("Before [Calling tool: unfin")
        r2 = f.feed('ished then [Calling tool: get_weather({"city":"NY"})] done')
        r3 = f.finish()
        result = r1 + r2 + r3
        assert "[Calling tool:" not in result
        assert "done" in result

    def test_tool_call_prefix_variant_later_parseable_envelope(self):
        """[Tool call:] prefix variant must also detect later parseable envelope."""
        f = ToolCallStreamFilter(_make_tokenizer())
        text = (
            "Before [Tool call: unfinished and then "
            '[Tool call: get_weather({"city":"NY"})] done'
        )
        result = f.feed(text)
        result += f.finish()
        assert "[Tool call:" not in result
        assert result == "Before  unfinished and then  done"

    def test_finish_preserves_non_tool_angle_identifier_suffix_literal(self):
        """Non-tool literal tails like '<alpha' should not be dropped at stream end."""
        f = ToolCallStreamFilter(_make_tokenizer())
        result = f.feed("Use <alpha")
        result += f.finish()
        assert result == "Use <alpha"

    def test_partial_non_tool_namespaced_literal_is_preserved(self):
        """Namespaced-looking suffixes that are not :tool_call remain visible."""
        f = ToolCallStreamFilter(_make_tokenizer())
        result = f.feed("Keep literal <alpha:beta")
        result += f.finish()
        assert result == "Keep literal <alpha:beta"

    def test_hyphen_namespaced_tool_call_open_suppresses_markup(self):
        """Hyphenated namespace tool-call open tag should trigger suppression."""
        f = ToolCallStreamFilter(_make_tokenizer())
        result = f.feed('Before <foo-bar:tool_call><invoke name="x">')
        assert result == "Before "
        assert f.finish() == ""

    # --- [Tool call: ...] format tests (issue #159) ---

    def test_tool_call_prefix_literal_passthrough(self):
        """[Tool call: ...] literal text that is not a valid call passes through."""
        f = ToolCallStreamFilter(_make_tokenizer())
        result = f.feed("Heads up: [Tool call:")
        result += f.feed(" maybe later]")
        result += f.finish()
        assert result == "Heads up: [Tool call: maybe later]"

    def test_tool_call_prefix_suppresses_with_args(self):
        """A complete [Tool call: name(args)] envelope should be suppressed."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("Lead in [Tool call:")
        r2 = f.feed(' get_weather({"city":"SF"})]')
        assert r1 == "Lead in "
        assert r2 == ""
        assert f.finish() == ""

    def test_tool_call_prefix_suppresses_without_args(self):
        """A complete [Tool call: name] envelope (no args) should be suppressed."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("Next: [Tool call:")
        r2 = f.feed(" mcp__notebooklm__chat_configure]")
        assert r1 == "Next: "
        assert r2 == ""
        assert f.finish() == ""

    def test_tool_call_prefix_preserves_trailing_text(self):
        """Suppression must preserve prose after a closed [Tool call: ...] envelope."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("Before [Tool call:")
        r2 = f.feed(' get_weather({"city":"SF"})] After text')
        r3 = f.finish()
        assert r1 + r2 + r3 == "Before  After text"

    def test_tool_call_prefix_unresolved_dropped_at_finish(self):
        """Unresolved [Tool call: prefix at stream end should be dropped."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("Text [Tool call:")
        r2 = f.feed(" some_tool")
        r3 = f.finish()
        assert r1 + r2 + r3 == "Text "

    def test_calling_tool_prefix_suppresses_without_args(self):
        """[Calling tool: name] without args should also be suppressed."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("Next: [Calling tool:")
        r2 = f.feed(" mcp__notebooklm__chat_configure]")
        assert r1 == "Next: "
        assert r2 == ""
        assert f.finish() == ""


class TestToolCallStreamFilterBracketPartialPrefix:
    """Tests for bracket partial prefix detection at token boundaries."""

    def test_bracket_partial_prefix_single_char(self):
        """'[' as separate token should be buffered, not emitted."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("Hello [")
        r2 = f.feed('Calling tool: Bash({"cmd":"ls"})]')
        result = r1 + r2 + f.finish()
        assert result == "Hello "

    def test_bracket_partial_prefix_multi_char(self):
        """'[Cal' as partial prefix should be buffered."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("Hello [Cal")
        r2 = f.feed('ling tool: Bash({"cmd":"ls"})]')
        result = r1 + r2 + f.finish()
        assert result == "Hello "

    def test_bracket_partial_prefix_tool_call_variant(self):
        """'[' followed by 'Tool call:' should be buffered and suppressed."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("Result [")
        r2 = f.feed('Tool call: search({"q":"test"})]')
        result = r1 + r2 + f.finish()
        assert result == "Result "

    def test_bracket_partial_prefix_false_alarm(self):
        """'[' followed by non-tool text should be released."""
        f = ToolCallStreamFilter(_make_tokenizer())
        r1 = f.feed("array [")
        r2 = f.feed("1, 2, 3]")
        result = r1 + r2 + f.finish()
        assert result == "array [1, 2, 3]"

    def test_bracket_char_by_char(self):
        """Character-by-character feeding should still suppress tool calls."""
        f = ToolCallStreamFilter(_make_tokenizer())
        text = '[Calling tool: x({"a":1})]'
        result = ""
        for ch in text:
            result += f.feed(ch)
        result += f.finish()
        assert result == ""


def _make_tokenizer_with_end(tool_call_start="", tool_call_end=""):
    """Create a mock tokenizer with start and end markers."""
    tok = MagicMock(spec=[])
    if tool_call_start is not None:
        tok.tool_call_start = tool_call_start
    if tool_call_end is not None:
        tok.tool_call_end = tool_call_end
    return tok


class TestToolCallStreamFilterSuppressAfterMarker:
    """Tests for one-sided markers (e.g. Mistral [TOOL_CALLS] with no end marker)."""

    def test_suppress_after_marker_basic(self):
        """Everything after a one-sided marker should be suppressed."""
        f = ToolCallStreamFilter(_make_tokenizer_with_end("[TOOL_CALLS]", ""))
        result = f.feed('[TOOL_CALLS]func_name[ARGS]{"key":"val"}')
        result += f.finish()
        assert result == ""

    def test_suppress_after_marker_with_preceding_text(self):
        """Text before one-sided marker should pass through."""
        f = ToolCallStreamFilter(_make_tokenizer_with_end("[TOOL_CALLS]", ""))
        r1 = f.feed("Hello ")
        r2 = f.feed('[TOOL_CALLS]func_name[ARGS]{"key":"val"}')
        result = r1 + r2 + f.finish()
        assert result == "Hello "

    def test_suppress_after_marker_partial_prefix(self):
        """Partial one-sided marker prefix should be buffered."""
        f = ToolCallStreamFilter(_make_tokenizer_with_end("[TOOL_CALLS]", ""))
        r1 = f.feed("[TOOL")
        r2 = f.feed('_CALLS]func_name[ARGS]{"key":"val"}')
        result = r1 + r2 + f.finish()
        assert result == ""

    def test_suppress_after_marker_multi_feed(self):
        """Permanent suppression persists across multiple feeds."""
        f = ToolCallStreamFilter(_make_tokenizer_with_end("[TOOL_CALLS]", ""))
        r1 = f.feed("Hi [TOOL_CALLS]start")
        r2 = f.feed(" more data")
        r3 = f.feed(" even more")
        result = r1 + r2 + r3 + f.finish()
        assert result == "Hi "


class TestParseToolCallsEmptyEndMarker:
    """Tests for parse_tool_calls with empty end marker (Mistral)."""

    def test_empty_end_marker_reaches_native_parser(self):
        """Empty tool_call_end should not block native parser invocation."""
        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "[TOOL_CALLS]"
        tok.tool_call_end = ""
        tok.tool_parser = lambda text, tools: {
            "name": "test_func",
            "arguments": {"key": "value"},
        }

        text = "[TOOL_CALLS]ignored"
        cleaned, tool_calls = parse_tool_calls(text, tok)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "test_func"

    def test_empty_end_marker_parses_content_after_marker(self):
        """One-sided marker should pass everything after it to the parser."""
        received_inputs = []

        def mock_parser(text, tools):
            received_inputs.append(text)
            return {"name": "list_files", "arguments": {"path": "."}}

        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "[TOOL_CALLS]"
        tok.tool_call_end = ""
        tok.tool_parser = mock_parser

        text = '[TOOL_CALLS]list_files[ARGS]{"path": "."}'
        cleaned, tool_calls = parse_tool_calls(text, tok)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "list_files"
        # Parser should receive the content after [TOOL_CALLS], not empty string
        assert len(received_inputs) == 1
        assert received_inputs[0] == 'list_files[ARGS]{"path": "."}'

    def test_empty_end_marker_cleans_text_before_marker(self):
        """Text before a one-sided marker should be preserved as cleaned_text."""
        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "[TOOL_CALLS]"
        tok.tool_call_end = ""
        tok.tool_parser = lambda text, tools: {
            "name": "read_file",
            "arguments": {"path": "README.md"},
        }

        text = 'Let me check that file.[TOOL_CALLS]read_file[ARGS]{"path": "README.md"}'
        cleaned, tool_calls = parse_tool_calls(text, tok)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert cleaned == "Let me check that file."

    def test_empty_end_marker_multiple_tool_calls(self):
        """Multiple one-sided tool calls should each be parsed separately."""
        call_count = [0]

        def mock_parser(text, tools):
            call_count[0] += 1
            if "list_files" in text:
                return {"name": "list_files", "arguments": {"path": "."}}
            elif "read_file" in text:
                return {"name": "read_file", "arguments": {"path": "README.md"}}
            raise ValueError(f"Unexpected: {text}")

        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "[TOOL_CALLS]"
        tok.tool_call_end = ""
        tok.tool_parser = mock_parser

        text = '[TOOL_CALLS]list_files[ARGS]{"path": "."}[TOOL_CALLS]read_file[ARGS]{"path": "README.md"}'
        cleaned, tool_calls = parse_tool_calls(text, tok)
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0].function.name == "list_files"
        assert tool_calls[1].function.name == "read_file"
        assert call_count[0] == 2

    def test_empty_end_marker_parser_failure_skips(self):
        """If the parser fails on a segment, it should be skipped gracefully."""
        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "[TOOL_CALLS]"
        tok.tool_call_end = ""

        def failing_parser(text, tools):
            raise ValueError("parse error")

        tok.tool_parser = failing_parser

        text = "[TOOL_CALLS]bad_input"
        cleaned, tool_calls = parse_tool_calls(text, tok)
        # Should fall through to other fallback parsers, not crash
        assert tool_calls is None or len(tool_calls) == 0


class TestParseBracketToolCalls:
    """Tests for bracket-style tool call parsing (issue #159)."""

    def test_tool_call_prefix_with_args(self):
        """[Tool call: name(args)] should be parsed as a tool call."""
        from omlx.api.tool_calling import _parse_bracket_tool_calls

        text = 'Hello [Tool call: get_weather({"city":"Tokyo"})] done'
        cleaned, tool_calls = _parse_bracket_tool_calls(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments) == {"city": "Tokyo"}
        assert "done" in cleaned
        assert "[Tool call:" not in cleaned

    def test_tool_call_prefix_without_args(self):
        """[Tool call: name] without args should be parsed with empty arguments."""
        from omlx.api.tool_calling import _parse_bracket_tool_calls

        text = "Next [Tool call: mcp__notebooklm__chat_configure] done"
        cleaned, tool_calls = _parse_bracket_tool_calls(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "mcp__notebooklm__chat_configure"
        assert tool_calls[0].function.arguments == "{}"
        assert "[Tool call:" not in cleaned

    def test_calling_tool_prefix_without_args(self):
        """[Calling tool: name] without args should also be parsed."""
        from omlx.api.tool_calling import _parse_bracket_tool_calls

        text = "Next [Calling tool: do_thing] done"
        cleaned, tool_calls = _parse_bracket_tool_calls(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "do_thing"
        assert tool_calls[0].function.arguments == "{}"

    def test_calling_tool_prefix_with_args_still_works(self):
        """Existing [Calling tool: name(args)] format must still parse correctly."""
        from omlx.api.tool_calling import _parse_bracket_tool_calls

        text = '[Calling tool: get_weather({"city":"SF"})]'
        cleaned, tool_calls = _parse_bracket_tool_calls(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments) == {"city": "SF"}

    def test_mixed_formats_parsed(self):
        """Both [Tool call:] and [Calling tool:] in same text should parse."""
        from omlx.api.tool_calling import _parse_bracket_tool_calls

        text = '[Tool call: tool_a({"x":1})] middle [Calling tool: tool_b({"y":2})]'
        cleaned, tool_calls = _parse_bracket_tool_calls(text)
        assert tool_calls is not None
        assert len(tool_calls) == 2
        names = {tc.function.name for tc in tool_calls}
        assert names == {"tool_a", "tool_b"}

    def test_no_match_returns_none(self):
        """Plain text without bracket patterns returns None tool_calls."""
        from omlx.api.tool_calling import _parse_bracket_tool_calls

        text = "Just some regular text"
        cleaned, tool_calls = _parse_bracket_tool_calls(text)
        assert tool_calls is None
        assert cleaned == text


class TestParseToolCallsWithThinkingFallback:
    """Tests for parse_tool_calls_with_thinking_fallback.

    Verifies that tool calls inside <think> blocks are recovered
    when small models emit them as reasoning instead of content.
    """

    def _make_tokenizer(self):
        tok = MagicMock(spec=[])
        return tok

    def test_thinking_fallback_xml_tool_call(self):
        """Tool call only in thinking content is recovered via fallback."""
        thinking = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/a.py"}}</tool_call>'
        regular = ""
        tok = self._make_tokenizer()

        cleaned, tool_calls = parse_tool_calls_with_thinking_fallback(
            thinking,
            regular,
            tokenizer=tok,
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "read_file"
        assert cleaned == ""

    def test_regular_content_takes_priority(self):
        """When regular content has tool calls, thinking fallback is skipped."""
        thinking = '<tool_call>{"name": "wrong_tool", "arguments": {}}</tool_call>'
        regular = '<tool_call>{"name": "correct_tool", "arguments": {}}</tool_call>'
        tok = self._make_tokenizer()

        cleaned, tool_calls = parse_tool_calls_with_thinking_fallback(
            thinking,
            regular,
            tokenizer=tok,
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "correct_tool"

    def test_no_tool_calls_anywhere(self):
        """No tool calls in either thinking or regular returns None."""
        thinking = "Let me think about this..."
        regular = "Here is my answer."
        tok = self._make_tokenizer()

        cleaned, tool_calls = parse_tool_calls_with_thinking_fallback(
            thinking,
            regular,
            tokenizer=tok,
        )
        assert tool_calls is None
        assert cleaned == "Here is my answer."

    def test_empty_thinking_no_fallback(self):
        """Empty thinking content skips fallback gracefully."""
        thinking = ""
        regular = "Just a regular response."
        tok = self._make_tokenizer()

        cleaned, tool_calls = parse_tool_calls_with_thinking_fallback(
            thinking,
            regular,
            tokenizer=tok,
        )
        assert tool_calls is None
        assert cleaned == "Just a regular response."

    def test_thinking_fallback_qwen_format(self):
        """Qwen/Llama XML format inside thinking is recovered."""
        thinking = (
            "<tool_call>"
            "<function=read><parameter=filePath>/src/main.py</parameter></function>"
            "</tool_call>"
        )
        regular = ""
        tok = self._make_tokenizer()

        cleaned, tool_calls = parse_tool_calls_with_thinking_fallback(
            thinking,
            regular,
            tokenizer=tok,
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "read"

    def test_cleaned_text_from_regular_not_thinking(self):
        """When regular content has text, thinking tool calls are discarded."""
        thinking = (
            'reasoning here <tool_call>{"name": "func", "arguments": {}}</tool_call>'
        )
        regular = "visible response text"
        tok = self._make_tokenizer()

        cleaned, tool_calls = parse_tool_calls_with_thinking_fallback(
            thinking,
            regular,
            tokenizer=tok,
        )
        assert tool_calls is None
        assert cleaned == "visible response text"

    def test_extract_tool_calls_with_thinking_sanitizes_reasoning_markup(self):
        """Sanitized reasoning should keep prose but drop tool-call control text."""
        thinking = (
            "Need to inspect first."
            '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/a.py"}}</tool_call>'
            "Then continue."
        )
        tok = self._make_tokenizer()

        result = extract_tool_calls_with_thinking(thinking, "", tokenizer=tok)

        assert result.tool_calls is not None
        assert result.tool_calls[0].function.name == "read_file"
        assert "<tool_call>" not in result.cleaned_thinking
        assert "</tool_call>" not in result.cleaned_thinking
        assert "Need to inspect first." in result.cleaned_thinking
        assert "Then continue." in result.cleaned_thinking

    def test_extract_tool_calls_with_thinking_sanitizes_reasoning_even_when_regular_wins(
        self,
    ):
        """Thinking cleanup should still run when regular content provides tool calls."""
        thinking = (
            "Reason about it."
            '<tool_call>{"name": "wrong_tool", "arguments": {}}</tool_call>'
        )
        regular = (
            "Visible text"
            '<tool_call>{"name": "correct_tool", "arguments": {}}</tool_call>'
        )
        tok = self._make_tokenizer()

        result = extract_tool_calls_with_thinking(thinking, regular, tokenizer=tok)

        assert result.tool_calls is not None
        assert result.tool_calls[0].function.name == "correct_tool"
        assert result.cleaned_text == "Visible text"
        assert result.cleaned_thinking == "Reason about it."

    # --- Thinking fallback guard tests (Issue #484) ---

    def test_thinking_fallback_blocked_when_regular_content_exists(self):
        """Tool calls in thinking are discarded when model produced regular text."""
        thinking = '<tool_call>{"name": "search", "arguments": {"q": "weather"}}</tool_call>'
        regular = "The weather is sunny today."
        tok = self._make_tokenizer()

        result = extract_tool_calls_with_thinking(thinking, regular, tokenizer=tok)

        assert result.tool_calls is None
        assert result.cleaned_text == "The weather is sunny today."
        assert result.tool_calls_from_thinking is False

    def test_thinking_fallback_filters_unknown_tools(self):
        """Tool calls with names not in provided tools list are discarded."""
        thinking = '<tool_call>{"name": "hallucinated_tool", "arguments": {}}</tool_call>'
        regular = ""
        tok = self._make_tokenizer()
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

        result = extract_tool_calls_with_thinking(
            thinking, regular, tokenizer=tok, tools=tools,
        )

        assert result.tool_calls is None
        assert result.tool_calls_from_thinking is False

    def test_thinking_fallback_keeps_known_tools_no_regular(self):
        """Tool calls matching provided tools are kept when regular is empty."""
        thinking = '<tool_call>{"name": "get_weather", "arguments": {"city": "Seoul"}}</tool_call>'
        regular = ""
        tok = self._make_tokenizer()
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

        result = extract_tool_calls_with_thinking(
            thinking, regular, tokenizer=tok, tools=tools,
        )

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls_from_thinking is True

    def test_thinking_fallback_mixed_known_unknown(self):
        """Only tool calls matching provided tools survive filtering."""
        thinking = (
            '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "fake_tool", "arguments": {}}</tool_call>'
        )
        regular = ""
        tok = self._make_tokenizer()
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

        result = extract_tool_calls_with_thinking(
            thinking, regular, tokenizer=tok, tools=tools,
        )

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"


# ---------------------------------------------------------------------------
# Gemma 4 robust fallback parser tests
# ---------------------------------------------------------------------------


class TestGemma4ArgsToJsonRobust:
    """Tests for _gemma4_args_to_json_robust()."""

    def test_gemma4_delimiters(self):
        result = _gemma4_args_to_json_robust('{query: <|"|>test search<|"|>}')
        assert result == {"query": "test search"}

    def test_bare_string_value(self):
        result = _gemma4_args_to_json_robust("{location: Tokyo}")
        assert result == {"location": "Tokyo"}

    def test_bare_multiword_value(self):
        result = _gemma4_args_to_json_robust("{city: New York}")
        assert result == {"city": "New York"}

    def test_numeric_value(self):
        result = _gemma4_args_to_json_robust("{count: 5}")
        assert result == {"count": 5}

    def test_boolean_value(self):
        result = _gemma4_args_to_json_robust("{verbose: true}")
        assert result == {"verbose": True}

    def test_null_value(self):
        result = _gemma4_args_to_json_robust("{data: null}")
        assert result == {"data": None}

    def test_mixed_types(self):
        result = _gemma4_args_to_json_robust(
            '{query: <|"|>hello<|"|>, count: 5}'
        )
        assert result == {"query": "hello", "count": 5}

    def test_standard_json_passthrough(self):
        result = _gemma4_args_to_json_robust('{"query": "hello"}')
        assert result == {"query": "hello"}

    def test_empty_object(self):
        result = _gemma4_args_to_json_robust("{}")
        assert result == {}


class TestParseGemma4ToolCallFallback:
    """Tests for _parse_gemma4_tool_call_fallback()."""

    def test_bare_string_args(self):
        result = _parse_gemma4_tool_call_fallback(
            "call:get_weather{location: Tokyo}"
        )
        assert result["name"] == "get_weather"
        assert result["arguments"] == {"location": "Tokyo"}

    def test_gemma4_delimiters(self):
        result = _parse_gemma4_tool_call_fallback(
            'call:search{query: <|"|>test<|"|>}'
        )
        assert result["name"] == "search"
        assert result["arguments"] == {"query": "test"}

    def test_colon_in_function_name(self):
        result = _parse_gemma4_tool_call_fallback(
            'call:tavily:search{query: <|"|>test<|"|>}'
        )
        assert result["name"] == "tavily:search"
        assert result["arguments"] == {"query": "test"}

    def test_standard_json_args(self):
        result = _parse_gemma4_tool_call_fallback(
            'call:search{"query": "hello world"}'
        )
        assert result["name"] == "search"
        assert result["arguments"] == {"query": "hello world"}

    def test_empty_args(self):
        result = _parse_gemma4_tool_call_fallback("call:get_time{}")
        assert result["name"] == "get_time"
        assert result["arguments"] == {}

    def test_multiple_calls(self):
        result = _parse_gemma4_tool_call_fallback(
            "call:a{x: 1}\ncall:b{y: 2}"
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "a"
        assert result[1]["name"] == "b"

    def test_no_match_raises(self):
        with pytest.raises(ValueError):
            _parse_gemma4_tool_call_fallback("not a tool call")


class TestParseToolCallsGemma4Integration:
    """Integration tests for parse_tool_calls() with Gemma 4 tokenizer."""

    @staticmethod
    def _make_gemma4_tokenizer():
        """Create a mock tokenizer that mimics Gemma 4 configuration."""
        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "<|tool_call>"
        tok.tool_call_end = "<tool_call|>"
        tok.tool_parser = MagicMock(
            side_effect=ValueError("mlx-lm parser failed")
        )
        return tok

    def test_fallback_parses_bare_strings(self):
        """Gemma 4 fallback succeeds when mlx-lm parser fails on bare strings."""
        tok = self._make_gemma4_tokenizer()
        text = "<|tool_call>\ncall:get_weather{location: Tokyo}\n<tool_call|>"

        cleaned, tool_calls = parse_tool_calls(text, tok, None)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        args = json.loads(tool_calls[0].function.arguments)
        assert args["location"] == "Tokyo"
        # Markers should be stripped from cleaned_text
        assert "<|tool_call>" not in cleaned
        assert "<tool_call|>" not in cleaned

    def test_markers_stripped_on_total_failure(self):
        """Even when fallback fails, markers are stripped from cleaned_text."""
        tok = self._make_gemma4_tokenizer()
        # Completely unparseable content between markers
        text = "<|tool_call>garbage that matches no format<tool_call|>"

        cleaned, tool_calls = parse_tool_calls(text, tok, None)

        assert tool_calls is None
        assert "<|tool_call>" not in cleaned
        assert "<tool_call|>" not in cleaned

    def test_function_gemma_fallback_not_triggered(self):
        """Fallback is NOT triggered for function_gemma (different markers)."""
        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "<start_function_call>"
        tok.tool_call_end = "<end_function_call>"
        tok.tool_parser = MagicMock(
            side_effect=ValueError("parser failed")
        )
        text = (
            "<start_function_call>"
            "call:func{key:<escape>value<escape>}"
            "<end_function_call>"
        )

        cleaned, tool_calls = parse_tool_calls(text, tok, None)

        # Should NOT have parsed via Gemma4 fallback (gate check fails)
        assert tool_calls is None

    def test_xml_fallback_still_works(self):
        """Models with <tool_call> markers still fall through to XML parser."""
        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "<tool_call>"
        tok.tool_call_end = "</tool_call>"
        tok.tool_parser = MagicMock(
            side_effect=ValueError("parser failed")
        )
        text = '<tool_call>{"name": "search", "arguments": {"q": "hi"}}</tool_call>'

        cleaned, tool_calls = parse_tool_calls(text, tok, None)

        # Should be parsed by _parse_xml_tool_calls fallback (Branch 2)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "search"
