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
    build_json_system_prompt,
    convert_tools_for_template,
    extract_json_from_text,
    format_tool_call_for_message,
    parse_json_output,
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
        text = '[1, 2, 3]'

        result = extract_json_from_text(text)

        assert result == [1, 2, 3]

    def test_json_with_whitespace(self):
        """Test extracting JSON with leading/trailing whitespace."""
        text = '   {"name": "John"}   '

        result = extract_json_from_text(text)

        assert result == {"name": "John"}

    def test_json_in_markdown_code_block(self):
        """Test extracting JSON from markdown code block."""
        text = '''Here is the result:
```json
{"name": "John", "age": 30}
```
'''

        result = extract_json_from_text(text)

        assert result == {"name": "John", "age": 30}

    def test_json_in_plain_code_block(self):
        """Test extracting JSON from plain code block."""
        text = '''Result:
```
{"status": "ok"}
```
'''

        result = extract_json_from_text(text)

        assert result == {"status": "ok"}

    def test_json_embedded_in_text(self):
        """Test extracting JSON embedded in text."""
        text = 'The response is {"result": true} and that is all.'

        result = extract_json_from_text(text)

        assert result == {"result": True}

    def test_no_json_found(self):
        """Test when no valid JSON is found."""
        text = 'This is just plain text without any JSON.'

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
        text = '''```json
{"result": true}
```'''
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
        assert result[0]["function"]["parameters"] == {"type": "object", "properties": {}}


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
        """Once suppressing, finish() returns nothing."""
        f = ToolCallStreamFilter(_make_tokenizer("<tc>"))
        f.feed("text<tc>rest")
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

    def test_partial_non_tool_namespaced_literal_is_preserved(self):
        """Namespaced-looking suffixes that are not :tool_call remain visible."""
        f = ToolCallStreamFilter(_make_tokenizer())
        result = f.feed("Keep literal <alpha:beta")
        result += f.finish()
        assert result == "Keep literal <alpha:beta"

    def test_hyphen_namespaced_tool_call_open_suppresses_markup(self):
        """Hyphenated namespace tool-call open tag should trigger suppression."""
        f = ToolCallStreamFilter(_make_tokenizer())
        result = f.feed("Before <foo-bar:tool_call><invoke name=\"x\">")
        assert result == "Before "
        assert f.finish() == ""
