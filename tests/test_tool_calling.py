# SPDX-License-Identifier: Apache-2.0
"""
Tests for tool calling parsing and conversion utilities.

Tests JSON schema validation, JSON extraction, and tool conversion functions.
"""

import ast
import json
import logging
import re
import time
from unittest.mock import MagicMock

import pytest

from omlx.api.openai_models import (
    FunctionCall,
    ResponseFormat,
    ResponseFormatJsonSchema,
    ToolCall,
    ToolDefinition,
)
from omlx.api.tool_calling import (
    ToolCallStreamFilter,
    _coerce_param_value,
    _gemma4_args_to_json_robust,
    _json_value_end,
    _marker_payloads,
    _parse_gemma4_tool_call_fallback,
    _parse_hermes_tool_calls,
    _parse_namespaced_tool_calls,
    _parse_xml_tool_calls,
    _strip_marker_spans,
    _remap_tool_call_names,
    _repair_json_value,
    _serialize_tool_call_arguments,
    build_json_system_prompt,
    convert_tools_for_template,
    enrich_tool_params_for_gemma4,
    extract_json_from_text,
    extract_tool_calls_with_thinking,
    format_tool_call_for_message,
    parse_json_output,
    parse_tool_calls,
    parse_tool_calls_with_thinking_fallback,
    restore_gemma4_param_names,
    sanitize_tool_call_markup,
    validate_json_schema,
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

    def test_json_schema_format_explicit_none_schema(self):
        """Explicit json_schema=None must behave like a missing key.

        Regression test: clients may send
        response_format={"type": "json_schema", "json_schema": null}; the
        explicit null bypasses dict.get()'s default and previously raised
        AttributeError ('NoneType' has no attribute 'get'). With no schema
        available, the output is treated like json_object: extracted but
        not validated.
        """
        text = '{"name": "John"}'
        response_format = {"type": "json_schema", "json_schema": None}

        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)

        assert cleaned == text
        assert parsed == {"name": "John"}
        assert is_valid is True
        assert error is None

    def test_json_schema_format_explicit_none_schema_wire_path(self):
        """Explicit json_schema=None via the ResponseFormat object (real wire path).

        A real client request arrives as a ResponseFormat, not a raw dict, so it
        takes the isinstance branch, which sets rf_dict["json_schema"] to None.
        The raw-dict test above skips that normalization, so this pins the
        production path directly: it must degrade like a missing schema, not raise.
        """
        text = '{"name": "John"}'
        response_format = ResponseFormat(type="json_schema", json_schema=None)

        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)

        assert cleaned == text
        assert parsed == {"name": "John"}
        assert is_valid is True
        assert error is None

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

    def test_json_schema_format_explicit_none_schema(self):
        """Explicit json_schema=None must behave like a missing key.

        Regression test: the explicit null bypasses dict.get()'s default
        and previously raised AttributeError. The prompt falls back to the
        default schema name 'response' with an empty schema.
        """
        response_format = {"type": "json_schema", "json_schema": None}

        result = build_json_system_prompt(response_format)

        assert result is not None
        assert "response" in result

    def test_json_schema_format_explicit_none_schema_wire_path(self):
        """Explicit json_schema=None via the ResponseFormat object (real wire path).

        A real client request arrives as a ResponseFormat whose isinstance branch
        sets rf_dict["json_schema"] to None; the raw-dict test above skips that
        normalization. Falls back to the default schema name 'response'.
        """
        response_format = ResponseFormat(type="json_schema", json_schema=None)

        result = build_json_system_prompt(response_format)

        assert result is not None
        assert "response" in result

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

    def test_missing_descriptions_are_template_safe(self):
        """Missing function and parameter descriptions render under strict Jinja."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "filters": {
                                "type": "object",
                                "properties": {
                                    "limit": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
            }
        ]

        result = convert_tools_for_template(tools)

        assert result is not None
        func = result[0]["function"]
        assert func["description"] == ""
        props = func["parameters"]["properties"]
        assert props["query"]["description"] == ""
        assert props["filters"]["description"] == ""
        assert props["filters"]["properties"]["limit"]["description"] == ""

        from jinja2 import Environment, StrictUndefined

        template = Environment(undefined=StrictUndefined).from_string(
            "{% for tool in tools %}"
            "{% set tool = tool.function %}"
            "{{ '// ' + tool.description }}"
            "{% for param_name, param_spec in tool.parameters.properties.items() %}"
            "{{ '// ' + param_spec.description }}"
            "{% endfor %}"
            "{% endfor %}"
        )
        template.render(tools=result)

    def test_schema_defaults_do_not_mutate_input(self):
        """Template safety normalization copies the input schema."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                    },
                },
            }
        ]

        result = convert_tools_for_template(tools)

        assert result is not None
        assert (
            "description"
            not in tools[0]["function"]["parameters"]["properties"]["query"]
        )
        assert (
            result[0]["function"]["parameters"]["properties"]["query"]["description"]
            == ""
        )


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
        assert f.take_recovery_candidate() == ""

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

    def test_minimax_tool_call_suppresses_envelope_but_preserves_trailing_text(self):
        """MiniMax M3 namespaced envelope suppression should not eat prose."""
        f = ToolCallStreamFilter(_make_tokenizer())
        result = f.feed('Before ]<]minimax[>[<tool_call><invoke name="x">')
        result += f.feed("]<]minimax[>[</invoke>]<]minimax[>[</tool_call> After")
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

    def test_hermes_marker_pair_suppressed_without_tokenizer_metadata(self):
        """Hermes markers should not leak in streams when tokenizer lacks marker attrs."""
        f = ToolCallStreamFilter(_make_tokenizer())
        chunks = [
            "Before ",
            "<|tool_call_start|>",
            "[execute_code(command='x', timeout=1)]",
            "<|tool_call_end|>",
            " After",
        ]
        result = "".join(f.feed(chunk) for chunk in chunks)
        result += f.finish()
        assert "<|tool_call_start|>" not in result
        assert "execute_code" not in result
        assert result == "Before  After"

    def test_orphan_hermes_end_marker_is_suppressed(self):
        """A closing Hermes marker without a visible open marker must not leak."""
        f = ToolCallStreamFilter(_make_tokenizer())
        result = f.feed("Before <|tool_call_end|> After")
        result += f.finish()
        assert result == "Before  After"

    def test_split_orphan_hermes_end_marker_is_suppressed(self):
        """Split closing Hermes markers must be buffered until classified."""
        f = ToolCallStreamFilter(_make_tokenizer())
        chunks = ["Before ", "<|tool_call_en", "d|>", " After"]
        result = "".join(f.feed(chunk) for chunk in chunks)
        result += f.finish()
        assert result == "Before  After"

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


_DSML_START = "<｜DSML｜tool_calls>"
_DSML_END = "</｜DSML｜tool_calls>"


def _make_dsml_filter():
    return ToolCallStreamFilter(_make_tokenizer_with_end(_DSML_START, _DSML_END))


class TestToolCallStreamFilterDsmlSeparator:
    """DeepSeek V4's "\\n\\n" separator belongs to the DSML envelope.

    The reference decoder's stop token is the literal string
    "\\n\\n<｜DSML｜tool_calls", so the separator before a tool-call block
    is control markup, not content. The filter consumes it with the
    envelope; content deltas never carry it.
    """

    def test_separator_consumed_before_tool_call(self):
        f = _make_dsml_filter()
        r = f.feed("I'll run it.\n\n" + _DSML_START + '{"name":"x"}')
        assert r == "I'll run it."
        assert f.feed(_DSML_END) == ""
        assert f.finish() == ""

    def test_separator_split_across_feeds(self):
        f = _make_dsml_filter()
        out = f.feed("answer.")
        out += f.feed("\n")
        out += f.feed("\n")
        out += f.feed("<｜DSML｜tool_")
        out += f.feed('calls>{"name":"x"}')
        out += f.feed(_DSML_END)
        out += f.finish()
        assert out == "answer."

    def test_marker_without_separator_still_suppressed(self):
        f = _make_dsml_filter()
        r = f.feed("answer." + _DSML_START + '{"name":"x"}' + _DSML_END)
        r += f.finish()
        assert r == "answer."

    def test_only_final_two_newlines_belong_to_envelope(self):
        # The reference decoder's str.find match consumes exactly the last
        # two newlines of a longer run; the rest is content.
        f = _make_dsml_filter()
        r = f.feed("a.\n\n\n" + _DSML_START + '{"name":"x"}' + _DSML_END)
        r += f.finish()
        assert r == "a.\n"

    def test_trailing_newlines_without_tool_call_survive_finish(self):
        # Newlines held back as a potential envelope prefix are literal
        # content when the stream ends without a tool call.
        f = _make_dsml_filter()
        r1 = f.feed("done.\n\n")
        r2 = f.finish()
        assert r1 == "done."
        assert r1 + r2 == "done.\n\n"

    def test_newlines_before_prose_pass_through(self):
        f = _make_dsml_filter()
        r1 = f.feed("a\n\n")
        r2 = f.feed("b")
        r3 = f.finish()
        assert r1 + r2 + r3 == "a\n\nb"

    def test_separator_hold_disabled_for_thinking_channel(self):
        # Filters watching the reasoning channel opt out: trailing newlines
        # flush in place, never as a late delta after the channel closed.
        f = ToolCallStreamFilter(
            _make_tokenizer_with_end(_DSML_START, _DSML_END),
            consume_dsml_separator=False,
        )
        assert f.feed("thinking...\n\n") == "thinking...\n\n"
        assert f.finish() == ""

    def test_opted_out_filter_keeps_separator_but_still_suppresses(self):
        # With the opt-out, a separator-preceded envelope is suppressed via
        # the bare pair and the separator stays visible -- byte-identical
        # to pre-change behavior for the thinking channel.
        f = ToolCallStreamFilter(
            _make_tokenizer_with_end(_DSML_START, _DSML_END),
            consume_dsml_separator=False,
        )
        r = f.feed("think\n\n" + _DSML_START + '{"name":"x"}' + _DSML_END)
        r += f.finish()
        assert r == "think\n\n"

    def test_multiple_envelopes_with_prose_between(self):
        f = _make_dsml_filter()
        r = f.feed(
            "a\n\n"
            + _DSML_START
            + "x"
            + _DSML_END
            + "b\n\n"
            + _DSML_START
            + "y"
            + _DSML_END
            + "c"
        )
        r += f.finish()
        assert r == "abc"

    def test_single_newline_held_then_released(self):
        f = _make_dsml_filter()
        r1 = f.feed("a\n")
        r2 = f.feed("b")
        r3 = f.finish()
        assert r1 + r2 + r3 == "a\nb"

    def test_truncated_open_marker_drops_held_separator_at_finish(self):
        # A stream cut mid-marker is malformed; strict mode drops the
        # partial-marker tail, and the held separator goes with it --
        # intended: the separator belongs to the (broken) envelope.
        f = _make_dsml_filter()
        r = f.feed("a\n\n<")
        r += f.finish()
        assert r == "a"

    def test_unclosed_envelope_recovery_includes_separator(self):
        # The recovery candidate carries the exact withheld bytes,
        # separator included -- nothing streamed is duplicated, nothing
        # withheld is lost.
        f = _make_dsml_filter()
        assert f.feed("a\n\n" + _DSML_START + "payload") == "a"
        assert f.finish() == ""
        assert (
            f.take_recovery_candidate() == "\n\n" + _DSML_START + "payload"
        )

    def test_sanitize_markup_preserves_thinking_separator(self):
        # sanitize_tool_call_markup cleans thinking-channel text at every
        # call site; it must match the streamed reasoning deltas, which
        # keep the separator (the opt-out above).
        tok = _make_tokenizer_with_end(_DSML_START, _DSML_END)
        cleaned = sanitize_tool_call_markup(
            "a\n\n" + _DSML_START + '{"name":"x"}' + _DSML_END + "b", tok
        )
        assert cleaned == "a\n\nb"


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


def _feed_chunked(f, text, chunk_size):
    """Feed text whole (chunk_size 0) or in fixed-size chunks."""
    if chunk_size == 0:
        return f.feed(text)
    return "".join(
        f.feed(text[i : i + chunk_size]) for i in range(0, len(text), chunk_size)
    )


def _paired_envelope_cases():
    return [
        ("<tool_call>", "</tool_call>", _make_tokenizer()),
        (
            "<|tool_call_start|>",
            "<|tool_call_end|>",
            _make_tokenizer(),
        ),
        (
            "<|tool_call>",
            "<tool_call|>",
            _make_tokenizer_with_end("<|tool_call>", "<tool_call|>"),
        ),
        (
            "]<]minimax[>[<tool_call>",
            "]<]minimax[>[</tool_call>",
            _make_tokenizer(),
        ),
        (
            "<vendor-x:tool_call>",
            "</vendor-x:tool_call>",
            _make_tokenizer(),
        ),
        (
            "<custom-call>",
            "</custom-call>",
            _make_tokenizer_with_end("<custom-call>", "</custom-call>"),
        ),
    ]


@pytest.mark.parametrize(
    ("start_marker", "end_marker", "tokenizer"),
    _paired_envelope_cases(),
)
def test_unclosed_paired_envelope_is_available_for_conditional_recovery(
    start_marker, end_marker, tokenizer, caplog
):
    """Every recognized paired format preserves its exact withheld EOF suffix."""
    f = ToolCallStreamFilter(tokenizer)
    suffix = start_marker + "payload" + end_marker[:-1]
    text = "Before " + suffix

    with caplog.at_level(logging.WARNING, logger="omlx.api.tool_calling"):
        visible = "".join(f.feed(ch) for ch in text)
        visible += f.finish()

    assert visible == "Before "
    assert f.take_recovery_candidate() == suffix
    assert f.take_recovery_candidate() == ""
    assert caplog.text.count("Unclosed tool-call envelope at end of stream") == 1


@pytest.mark.parametrize(
    ("start_marker", "end_marker", "tokenizer"),
    _paired_envelope_cases(),
)
def test_closed_paired_envelope_never_creates_recovery_candidate(
    start_marker, end_marker, tokenizer, caplog
):
    """Completed envelopes retain the existing clean streaming behavior."""
    f = ToolCallStreamFilter(tokenizer)
    text = "Before " + start_marker + "payload" + end_marker + " After"

    with caplog.at_level(logging.WARNING, logger="omlx.api.tool_calling"):
        visible = "".join(f.feed(ch) for ch in text)
        visible += f.finish()

    assert visible == "Before  After"
    assert f.take_recovery_candidate() == ""
    assert "Unclosed tool-call envelope" not in caplog.text


@pytest.mark.parametrize(
    ("start_marker", "end_marker", "tokenizer"),
    _paired_envelope_cases(),
)
@pytest.mark.parametrize("chunked", [False, True])
def test_malformed_payload_keeps_prose_after_a_real_close_marker(
    start_marker, end_marker, tokenizer, chunked
):
    """A payload that never parses must not swallow the prose after its close.

    The payload scan cannot confirm where a malformed value ends, but the
    literal close marker still bounds the envelope, so trailing text stays
    visible instead of being withheld to EOF.
    """
    f = ToolCallStreamFilter(tokenizer)
    text = "Before " + start_marker + '{"name":"f"' + end_marker + " After"

    if chunked:
        visible = "".join(f.feed(ch) for ch in text)
    else:
        visible = f.feed(text)
    visible += f.finish()

    assert visible == "Before  After"
    assert f.take_recovery_candidate() == ""


def test_embedded_close_marker_still_bounds_the_envelope():
    """The #2507 case must not regress: a marker inside JSON is not the close."""
    f = ToolCallStreamFilter(_make_tokenizer())
    text = (
        'Before <tool_call>{"name":"f","arguments":'
        '{"x":"</tool_call>"}}</tool_call> After'
    )

    visible = "".join(f.feed(ch) for ch in text) + f.finish()

    assert visible == "Before  After"


@pytest.mark.parametrize("chunk_size", [0, 1, 7])
def test_array_payload_with_embedded_close_marker_is_suppressed(chunk_size):
    """A ``[{...}]`` array payload gets the same #2507 protection as an object.

    Classifying every leading ``[`` as the Hermes bracket dialect made the
    filter fall back to first-marker splitting, so an embedded close marker
    leaked the array's tail as visible content while the non-streaming parser
    recovered the call.
    """
    f = ToolCallStreamFilter(_make_tokenizer())
    text = (
        'A <tool_call>[{"name":"f","arguments":'
        '{"s":"a </tool_call> b"}}]</tool_call> AFTER'
    )

    visible = _feed_chunked(f, text, chunk_size) + f.finish()

    assert visible == "A  AFTER"
    assert f.take_recovery_candidate() == ""


def test_array_payload_with_leading_whitespace_is_suppressed():
    """Whitespace between ``[`` and ``{`` still classifies as a JSON array."""
    f = ToolCallStreamFilter(_make_tokenizer())
    text = (
        'A <tool_call>[ {"name":"f","arguments":'
        '{"s":"</tool_call>"}} ]</tool_call> AFTER'
    )

    visible = "".join(f.feed(ch) for ch in text) + f.finish()

    assert visible == "A  AFTER"
    assert f.take_recovery_candidate() == ""


def test_unterminated_array_payload_becomes_recovery_candidate():
    """An array payload that never closes stays withheld like an object."""
    f = ToolCallStreamFilter(_make_tokenizer())

    visible = "".join(f.feed(ch) for ch in 'A <tool_call>[{"x"') + f.finish()

    assert visible == "A "
    assert f.take_recovery_candidate() == '<tool_call>[{"x"'


def test_object_payload_with_nested_array_and_embedded_marker():
    """Array depth tracking must not break object payloads with inner arrays."""
    f = ToolCallStreamFilter(_make_tokenizer())
    text = (
        'A <tool_call>{"name":"f","arguments":'
        '{"a":[1,2],"s":"</tool_call>"}}</tool_call> AFTER'
    )

    visible = "".join(f.feed(ch) for ch in text) + f.finish()

    assert visible == "A  AFTER"
    assert f.take_recovery_candidate() == ""


def test_payload_without_any_close_marker_is_still_withheld():
    """No close marker at all means the envelope tail stays hidden."""
    f = ToolCallStreamFilter(_make_tokenizer())

    visible = f.feed('Before <tool_call>{"name":"f" After') + f.finish()

    assert visible == "Before "
    assert f.take_recovery_candidate() == '<tool_call>{"name":"f" After'


def test_close_marker_fallback_rescans_the_recovered_tail():
    """Prose recovered after a close marker is re-filtered, not emitted raw."""
    f = ToolCallStreamFilter(_make_tokenizer())
    text = (
        'A <tool_call>{"name":"f"</tool_call> B '
        "<|tool_call_start|>second<|tool_call_end|> C"
    )

    visible = "".join(f.feed(ch) for ch in text) + f.finish()

    assert visible == "A  B  C"
    assert f.take_recovery_candidate() == ""


def test_only_last_unclosed_envelope_becomes_recovery_candidate():
    """Completed earlier calls stay hidden when a later call is unterminated."""
    f = ToolCallStreamFilter(_make_tokenizer())
    text = (
        "Before <tool_call>first</tool_call> middle "
        "<tool_call>second</tool_"
    )

    visible = "".join(f.feed(ch) for ch in text)
    visible += f.finish()

    assert visible == "Before  middle "
    assert f.take_recovery_candidate() == "<tool_call>second</tool_"


@pytest.mark.parametrize("chunk_size", [0, 1, 3, 7, 16])
def test_prose_between_two_malformed_envelopes_is_preserved(chunk_size):
    """The EOF unwind must split each envelope at its own close marker.

    Splitting at the LAST close marker in the withheld text deletes the
    prose sitting between two malformed envelopes of the same pair.
    """
    f = ToolCallStreamFilter(_make_tokenizer())
    text = (
        'A <tool_call>{"name":"f"</tool_call> B '
        '<tool_call>{"name":"g"</tool_call> C'
    )

    visible = _feed_chunked(f, text, chunk_size) + f.finish()

    assert visible == "A  B  C"
    assert f.take_recovery_candidate() == ""


@pytest.mark.parametrize("chunk_size", [0, 1, 7])
def test_literal_close_marker_in_trailing_prose_is_preserved(chunk_size):
    """A literal XML close marker in prose is not a malformed envelope's end.

    XML-style closers may appear as ordinary prose, so the unwind must not
    treat a later literal occurrence as the envelope boundary and delete
    everything before it.
    """
    f = ToolCallStreamFilter(_make_tokenizer())
    text = 'A <tool_call>{"name":"f"</tool_call> B literal </tool_call> C'

    visible = _feed_chunked(f, text, chunk_size) + f.finish()

    assert visible == "A  B literal </tool_call> C"
    assert f.take_recovery_candidate() == ""


@pytest.mark.parametrize("chunk_size", [0, 1, 7])
def test_valid_call_with_embedded_marker_after_malformed_envelope(chunk_size):
    """A valid call after a malformed one keeps its #2507 protection at EOF.

    The unwind uses the same span primitive as the non-streaming parser, so
    the valid payload ends at its structural boundary and its embedded close
    marker neither splits it nor leaks its tail as content.
    """
    f = ToolCallStreamFilter(_make_tokenizer())
    text = (
        'A <tool_call>{"x"</tool_call> B <tool_call>'
        '{"name":"g","arguments":{"x":"</tool_call>"}}</tool_call> C'
    )

    visible = _feed_chunked(f, text, chunk_size) + f.finish()

    assert visible == "A  B  C"
    assert f.take_recovery_candidate() == ""


def test_unwind_preserves_prose_between_malformed_namespaced_envelopes():
    """The unwind resolves dynamic namespaced close markers per envelope."""
    f = ToolCallStreamFilter(_make_tokenizer())
    text = (
        'A <foo:tool_call>{"x"</foo:tool_call> B '
        '<foo:tool_call>{"y"</foo:tool_call> C'
    )

    visible = "".join(f.feed(ch) for ch in text) + f.finish()

    assert visible == "A  B  C"
    assert f.take_recovery_candidate() == ""


def test_unwind_swallows_bracket_call_in_recovered_tail():
    """A self-contained bracket call inside the unwound tail stays hidden."""
    f = ToolCallStreamFilter(_make_tokenizer())
    text = 'A <tool_call>{"x"</tool_call> B [Calling tool: foo] C'

    visible = "".join(f.feed(ch) for ch in text) + f.finish()

    assert visible == "A  B  C"
    assert f.take_recovery_candidate() == ""


def test_unwind_reopened_envelope_becomes_recovery_candidate():
    """A second unterminated envelope in the unwound tail is withheld whole."""
    f = ToolCallStreamFilter(_make_tokenizer())
    text = 'A <tool_call>{"x"</tool_call> B <tool_call>{"y"'

    visible = "".join(f.feed(ch) for ch in text) + f.finish()

    assert visible == "A  B "
    assert f.take_recovery_candidate() == '<tool_call>{"y"'


def test_unwind_drops_partial_open_marker_in_trailing_prose():
    """Strict-mode tail rules still apply to prose recovered by the unwind."""
    f = ToolCallStreamFilter(_make_tokenizer())
    text = 'A <tool_call>{"x"</tool_call> B <tool_ca'

    visible = "".join(f.feed(ch) for ch in text) + f.finish()

    assert visible == "A  B "
    assert f.take_recovery_candidate() == ""


def test_unwind_stays_linear_on_repeated_malformed_envelopes():
    """The EOF unwind must not do quadratic work on marker-repeating output.

    A re-feed loop that copies the remaining text once per malformed envelope
    took ~9 s on this input; the single-pass unwind takes well under a
    second, so the generous bound only trips on a complexity regression.
    """
    f = ToolCallStreamFilter(_make_tokenizer())
    text = "PRE " + '<tool_call>{"x"</tool_call> ' * 4000 + "POST"

    start = time.perf_counter()
    visible = f.feed(text) + f.finish()
    elapsed = time.perf_counter() - start

    assert visible.startswith("PRE ")
    assert visible.endswith("POST")
    assert elapsed < 5.0


class TestToolCallStreamFilterSuppressAfterMarker:
    """Tests for one-sided markers (e.g. Mistral [TOOL_CALLS] with no end marker)."""

    def test_suppress_after_marker_basic(self):
        """Everything after a one-sided marker should be suppressed."""
        f = ToolCallStreamFilter(_make_tokenizer_with_end("[TOOL_CALLS]", ""))
        result = f.feed('[TOOL_CALLS]func_name[ARGS]{"key":"val"}')
        result += f.finish()
        assert result == ""
        assert f.take_recovery_candidate() == ""

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


class TestParseToolCallsSyntaxError:
    """Regression tests for issue #882.

    mlx-lm's qwen3_coder parser calls ast.literal_eval on parameter
    values, which raises SyntaxError on non-Python-literal strings
    (e.g. "python3 test.py" for an array-typed parameter). The
    exception used to escape parse_tool_calls and turned into a
    server_error SSE chunk, silently dropping the tool call.
    """

    def _qwen_tok(self, failing_parser):
        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "<tool_call>"
        tok.tool_call_end = "</tool_call>"
        tok.tool_parser = failing_parser
        return tok

    def test_syntax_error_does_not_escape(self):
        """SyntaxError from native parser must not crash parse_tool_calls."""

        def failing_parser(text, tools):
            raise SyntaxError("invalid syntax (<unknown>, line 1)")

        tok = self._qwen_tok(failing_parser)
        text = (
            "pre\n<tool_call>\n<function=shell>\n"
            "<parameter=command>python3 test.py</parameter>\n"
            "</function>\n</tool_call>\npost"
        )
        # Must not raise.
        cleaned, tool_calls = parse_tool_calls(text, tok)
        # XML fallback should have recovered the call.
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "shell"
        args = json.loads(tool_calls[0].function.arguments)
        assert args == {"command": "python3 test.py"}

    def test_qwen_xml_fallback_recovers_call(self):
        """Qwen-style XML body recovers via _parse_xml_tool_calls on native failure."""

        def failing_parser(text, tools):
            raise SyntaxError("invalid syntax (<unknown>, line 1)")

        tok = self._qwen_tok(failing_parser)
        text = (
            "<tool_call>\n<function=read>\n"
            "<parameter=path>/etc/hosts</parameter>\n"
            "<parameter=lines>10</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, tool_calls = parse_tool_calls(text, tok)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "read"
        args = json.loads(tool_calls[0].function.arguments)
        assert args["path"] == "/etc/hosts"
        assert args["lines"] == 10  # json.loads converts numeric string

    def test_json_fallback_recovers_raw_control_chars(self):
        """Model-generated JSON tool calls may contain raw tabs or newlines."""

        def failing_parser(text, tools):
            raise json.JSONDecodeError("Invalid control character at", text, 90)

        tok = self._qwen_tok(failing_parser)
        old_string = (
            "\t\t// Check if second word is a subcommand.\n"
            "\t\tif len(ce.Args) > 1 && isSubcommand(ce.Args[1]) {"
        )
        text = (
            '<tool_call>{"name": "edit", "arguments": {'
            '"file_path": "/Users/user/project/file.go", '
            f'"old_string": "{old_string}", '
            '"new_string": "x"}}</tool_call>'
        )

        cleaned, tool_calls = parse_tool_calls(text, tok)

        assert cleaned == ""
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "edit"
        args = json.loads(tool_calls[0].function.arguments)
        assert args["old_string"] == old_string

    def test_generic_xml_json_fallback_recovers_raw_control_chars(self):
        """The non-native XML JSON fallback should recover the same malformed JSON."""

        tok = _make_tokenizer()
        old_string = "\tindent\nnext line"
        text = (
            '<tool_call>{"name": "edit", "arguments": {'
            f'"old_string": "{old_string}", '
            '"new_string": "x"}}</tool_call>'
        )

        cleaned, tool_calls = parse_tool_calls(text, tok)

        assert cleaned == ""
        assert tool_calls is not None
        args = json.loads(tool_calls[0].function.arguments)
        assert args == {"old_string": old_string, "new_string": "x"}

    def test_unparseable_body_logs_and_drops(self, caplog):
        """Fully unparseable body drops gracefully and logs a warning."""

        def failing_parser(text, tools):
            raise SyntaxError("invalid syntax (<unknown>, line 1)")

        tok = self._qwen_tok(failing_parser)
        text = "<tool_call>not a function at all, just text</tool_call>"

        with caplog.at_level(logging.WARNING, logger="omlx.api.tool_calling"):
            cleaned, tool_calls = parse_tool_calls(text, tok)

        assert tool_calls is None or len(tool_calls) == 0
        # Warning emitted so failures are visible rather than silent.
        assert any(
            "Native tool parser failed" in r.message
            and "SyntaxError" in r.message
            for r in caplog.records
        )

    def test_type_error_also_caught(self):
        """TypeError from native parser also must not escape."""

        def failing_parser(text, tools):
            raise TypeError("unexpected type during parse")

        tok = self._qwen_tok(failing_parser)
        text = (
            "<tool_call>\n<function=patch>\n"
            "<parameter=path>src/a.py</parameter>\n"
            "</function>\n</tool_call>"
        )
        # Must not raise.
        _, tool_calls = parse_tool_calls(text, tok)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "patch"

    def test_gemma4_path_syntax_error_does_not_escape(self):
        """Gemma 4 fallback branch also must not propagate SyntaxError."""

        def failing_parser(text, tools):
            raise SyntaxError("invalid syntax (<unknown>, line 1)")

        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "<|tool_call>"
        tok.tool_call_end = "<tool_call|>"
        tok.tool_parser = failing_parser

        text = "<|tool_call>garbage body<tool_call|>"
        # Must not raise. Gemma 4 fallback will also fail on this body,
        # but the outer code must still complete gracefully.
        _, tool_calls = parse_tool_calls(text, tok)
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

    def test_hermes_multi_call_block_parses_python_keyword_arguments(self):
        """Hermes blocks may contain multiple Python-style calls in one list."""
        text = (
            "┊ 🐍 preparing execute_code…\n"
            "<|tool_call_start|>"
            "[execute_code(command='python3 diversify_hermes.py --model "
            "\"Qwen3.6-35B-A3B-ConfigI-MLX\" --timeout 180 && echo "
            "\"Hermes mode completed\"', timeout=400), "
            "execute_code(command='python3 diversify_v2.py --runs 50 --per-run 54 "
            "--timeout 300 && echo \"v2 dynamic completed\"', timeout=400)]"
            "<|tool_call_end|>"
        )
        result = extract_tool_calls_with_thinking(
            "",
            text,
            tokenizer=_make_tokenizer(),
            tools=[{"type": "function", "function": {"name": "execute_code"}}],
        )
        assert result.cleaned_text == "┊ 🐍 preparing execute_code…"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert [tc.function.name for tc in result.tool_calls] == [
            "execute_code",
            "execute_code",
        ]
        first_args = json.loads(result.tool_calls[0].function.arguments)
        second_args = json.loads(result.tool_calls[1].function.arguments)
        assert first_args["timeout"] == 400
        assert second_args["timeout"] == 400
        assert "diversify_hermes.py" in first_args["command"]
        assert "diversify_v2.py" in second_args["command"]

    def test_hermes_fallback_runs_when_native_parser_rejects_bracket_payload(self):
        """Tokenizer native parser failures should fall through to Hermes fallback."""

        class NativeRejectingTokenizer:
            has_tool_calling = True
            tool_call_start = "<|tool_call_start|>"
            tool_call_end = "<|tool_call_end|>"

            @staticmethod
            def tool_parser(text, tools):
                raise ValueError("native parser rejected Hermes bracket payload")

        text = (
            "Before <|tool_call_start|>"
            "[execute_code(command='python3 script.py', timeout=400)]"
            "<|tool_call_end|>"
        )
        result = extract_tool_calls_with_thinking(
            "",
            text,
            tokenizer=NativeRejectingTokenizer(),
            tools=[{"type": "function", "function": {"name": "execute_code"}}],
        )
        assert result.cleaned_text == "Before"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "execute_code"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"command": "python3 script.py", "timeout": 400}


class TestParseToolCallsWithThinkingFallback:
    """Tests for parse_tool_calls_with_thinking_fallback.

    Verifies that tool calls inside <think> blocks are recovered
    when small models emit them as reasoning instead of content.
    """

    def test_thinking_fallback_xml_tool_call(self):
        """Tool call only in thinking content is recovered via fallback."""
        thinking = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/a.py"}}</tool_call>'
        regular = ""
        tok = _make_tokenizer()

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
        tok = _make_tokenizer()

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
        tok = _make_tokenizer()

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
        tok = _make_tokenizer()

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
        tok = _make_tokenizer()

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
        tok = _make_tokenizer()

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
        tok = _make_tokenizer()

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
        tok = _make_tokenizer()

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
        tok = _make_tokenizer()

        result = extract_tool_calls_with_thinking(thinking, regular, tokenizer=tok)

        assert result.tool_calls is None
        assert result.cleaned_text == "The weather is sunny today."
        assert result.tool_calls_from_thinking is False

    def test_thinking_fallback_filters_unknown_tools(self):
        """Tool calls with names not in provided tools list are discarded."""
        thinking = '<tool_call>{"name": "hallucinated_tool", "arguments": {}}</tool_call>'
        regular = ""
        tok = _make_tokenizer()
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
        tok = _make_tokenizer()
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
        tok = _make_tokenizer()
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

        result = extract_tool_calls_with_thinking(
            thinking, regular, tokenizer=tok, tools=tools,
        )

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"


# ---------------------------------------------------------------------------
# Guard 1 regression: valid tool calls dropped with preamble (#1392)

class TestThinkingFallbackGuardRegression:
    """Guard 1 drops valid tool calls when the model emits a preamble.

    Qwen3-Coder places real tool invocations inside thinking and adds a
    short narrative preamble as regular content.  Guard 1 assumed regular
    text means the thinking tool call is "just reasoning", but for these
    models it's a genuine invocation.  Guard 2 (name matching) is the
    correct discriminator.

    See https://github.com/jundot/omlx/issues/1392
    """

    def test_known_tool_in_thinking_kept_with_preamble(self):
        """A tool call matching a provided tool should survive even when
        regular content is non-empty (short preamble).

        Qwen3-Coder with thinking enabled places real tool calls inside
        thinking and emits a preamble like "Let me create the file:" as
        regular content. Guard 1 drops these; Guard 2 (name matching)
        should preserve them.
        """
        thinking = (
            'I need to write a file. '
            '<tool_call>{"name": "write_file", "arguments": {"path": "/tmp/test.txt", "content": "hello"}}</tool_call>'
        )
        regular = "Let me create the file:"
        tok = _make_tokenizer()
        tools = [{
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        }]

        result = extract_tool_calls_with_thinking(
            thinking, regular, tokenizer=tok, tools=tools,
        )

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "write_file"
        assert result.tool_calls_from_thinking is True

    def test_empty_tools_list_drops_thinking_calls_no_regular(self):
        """tools=[] means 'no tools allowed' — drop thinking-embedded calls
        even when regular content is empty."""
        thinking = (
            'I need to write a file. '
            '<tool_call' + '>'
            '{"name": "write_file", "arguments": {"path": "/tmp/test.txt", "content": "hello"}}'
            '</tool_call' + '>'
        )
        regular = ""
        tok = _make_tokenizer()
        tools = []

        result = extract_tool_calls_with_thinking(
            thinking, regular, tokenizer=tok, tools=tools,
        )

        assert result.tool_calls is None
        assert result.tool_calls_from_thinking is False

    def test_empty_tools_list_drops_thinking_calls_with_regular(self):
        """tools=[] means 'no tools allowed' — drop thinking-embedded calls
        when regular content is also present."""
        thinking = (
            'I need to write a file. '
            '<tool_call' + '>'
            '{"name": "write_file", "arguments": {"path": "/tmp/test.txt", "content": "hello"}}'
            '</tool_call' + '>'
        )
        regular = "Let me create the file:"
        tok = _make_tokenizer()
        tools = []

        result = extract_tool_calls_with_thinking(
            thinking, regular, tokenizer=tok, tools=tools,
        )

        assert result.tool_calls is None
        assert result.tool_calls_from_thinking is False


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

    def test_unbalanced_open_brace_in_json_string(self):
        """A lone ``{`` inside a JSON string must not unbalance the span
        (#1854). Before the string-aware scan this drove brace depth above
        zero so the span never closed and the whole call was dropped."""
        result = _parse_gemma4_tool_call_fallback(
            'call:ns:create{"content": "open { brace"}'
        )
        assert result["name"] == "ns:create"
        assert result["arguments"] == {"content": "open { brace"}

    def test_multi_call_with_brace_in_first_args(self):
        """A ``}`` inside the first call's JSON string must not corrupt the
        consumed-span bookkeeping that separates sibling calls (#1854).
        Pre-fix the first span truncated early, so the second call's head
        landed inside the supposed-consumed region."""
        result = _parse_gemma4_tool_call_fallback(
            'call:ns:create{"a": "x } y"}call:foo{"b": 1}'
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "ns:create"
        assert result[0]["arguments"] == {"a": "x } y"}
        assert result[1]["name"] == "foo"
        assert result[1]["arguments"] == {"b": 1}

    def test_escaped_backslash_before_closing_quote(self):
        """A value ending in an escaped backslash (Windows path) closes on
        the following quote, not on the backslash-escaped one (#1854)."""
        result = _parse_gemma4_tool_call_fallback(
            r'call:ns:create{"path": "C:\\tmp\\"}'
        )
        assert result["name"] == "ns:create"
        assert result["arguments"] == {"path": "C:\\tmp\\"}

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

    def test_degenerate_prefix_missing_colon(self):
        """Diffusion lane can drop the colon: ``calldone{...}``."""
        result = _parse_gemma4_tool_call_fallback('calldone{answer: ok}')
        assert result["name"] == "done"
        assert result["arguments"] == {"answer": "ok"}

    def test_degenerate_prefix_missing_call(self):
        """Diffusion lane can drop ``call``: ``:done{...}``."""
        result = _parse_gemma4_tool_call_fallback(':done{answer: ok}')
        assert result["name"] == "done"
        assert result["arguments"] == {"answer": "ok"}

    def test_long_bare_value_with_commas_and_newlines(self):
        """Key-anchored capture recovers markdown-laden bare values
        (observed live: hindsight reflect ``done`` calls whose ``answer``
        contains tables, commas, and newlines)."""
        text = (
            "calldone{answer:# Title\n\n"
            "| A | B |\n| :--- | :--- |\n| x, y | z |\n"
            ",directive_compliance:1. ok. 2. fine."
            ",memory_ids:[mm-abc123]"
            ",mental_model_ids:[]"
            ",observation_ids:[]}"
        )
        result = _parse_gemma4_tool_call_fallback(text)
        assert result["name"] == "done"
        args = result["arguments"]
        assert set(args.keys()) == {
            "answer",
            "directive_compliance",
            "memory_ids",
            "mental_model_ids",
            "observation_ids",
        }
        assert args["answer"].startswith("# Title")
        assert "x, y" in args["answer"]
        assert args["observation_ids"] == []

    def test_prose_without_braces_still_raises(self):
        with pytest.raises(ValueError):
            _parse_gemma4_tool_call_fallback("just words, no payload")


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

    def test_fallback_parses_parenthesized_variant(self):
        """End-to-end: the paren kwargs variant (#1846) reaches the Gemma 4
        fallback (native parser raises) and is extracted instead of stripped.

        Before the fix the block was deleted: native parser fails, the
        curly-only fallback also fails, and the client gets empty content.
        """
        tok = self._make_gemma4_tokenizer()
        text = (
            '<|tool_call>\n'
            'call:todo(todos=[{content:<|"|>Draft the plan<|"|>,'
            'id:<|"|>todo-1<|"|>,status:<|"|>pending<|"|>}])\n'
            '<tool_call|>'
        )

        cleaned, tool_calls = parse_tool_calls(text, tok, None)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "todo"
        args = json.loads(tool_calls[0].function.arguments)
        assert args["todos"][0]["content"] == "Draft the plan"
        assert args["todos"][0]["status"] == "pending"
        assert "<|tool_call>" not in cleaned
        assert "<tool_call|>" not in cleaned

    def test_brace_in_json_string_survives_remap(self):
        """A ``}`` inside a JSON double-quoted value must not truncate the
        args span, even when the parse then remaps onto a registered tool
        (#1854). Before the fix the span scanner stopped at the brace inside
        the string, the legacy recovery produced a corrupted ``{"content":
        "has }`` parse, and the suffix remap turned ``ns:create`` into an
        executable ``create`` call carrying silently mangled arguments. The
        full content must round-trip intact."""
        tok = self._make_gemma4_tokenizer()
        tools = [{"type": "function", "function": {"name": "create"}}]
        text = (
            '<|tool_call>\n'
            'call:ns:create{"content": "has } brace"}\n'
            '<tool_call|>'
        )

        cleaned, tool_calls = parse_tool_calls(text, tok, tools)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        # Remap fired: ns:create -> registered create.
        assert tool_calls[0].function.name == "create"
        # ...and the brace inside the string did not corrupt the value.
        args = json.loads(tool_calls[0].function.arguments)
        assert args["content"] == "has } brace"
        assert "<|tool_call>" not in cleaned
        assert "<tool_call|>" not in cleaned

    def test_escaped_quote_in_json_string_does_not_close_early(self):
        """An escaped ``\\"`` inside a JSON value must not be read as the
        closing quote, so a following ``}`` stays string content (#1854)."""
        tok = self._make_gemma4_tokenizer()
        tools = [{"type": "function", "function": {"name": "create"}}]
        text = (
            '<|tool_call>\n'
            'call:ns:create{"content": "a \\" } b"}\n'
            '<tool_call|>'
        )

        cleaned, tool_calls = parse_tool_calls(text, tok, tools)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "create"
        args = json.loads(tool_calls[0].function.arguments)
        assert args["content"] == 'a " } b'

    def test_deep_standard_json_failure_strips_markers(self):
        """Deep valid JSON args are dropped cleanly, not surfaced as 500s."""
        tok = self._make_gemma4_tokenizer()
        args = '{"a": ' * 80 + "1" + "}" * 80
        text = f"<|tool_call>\ncall:ns:create{args}\n<tool_call|>"

        cleaned, tool_calls = parse_tool_calls(text, tok, None)

        assert tool_calls is None
        assert "<|tool_call>" not in cleaned
        assert "<tool_call|>" not in cleaned

    def test_markers_stripped_on_total_failure(self, caplog):
        """Even when fallback fails, markers are stripped and warning is logged."""
        tok = self._make_gemma4_tokenizer()
        # Completely unparseable content between markers
        text = "<|tool_call>garbage that matches no format<tool_call|>"

        with caplog.at_level(logging.WARNING, logger="omlx.api.tool_calling"):
            cleaned, tool_calls = parse_tool_calls(text, tok, None)

        assert tool_calls is None
        assert "<|tool_call>" not in cleaned
        assert "<tool_call|>" not in cleaned
        assert any("parsing failed" in msg for msg in caplog.messages)

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


class TestGemma4SingleQuotedArgs:
    """Single-quoted and structurally hard Gemma 4 args (#1830).

    Table-driven round-trips: each row is (rendered args, expected dict).
    """

    @pytest.mark.parametrize(
        "rendered,expected",
        [
            # Issue #1830's reported shape: single-quoted values
            (
                "{filename: 'output.pdf', title: 'My Doc'}",
                {"filename": "output.pdf", "title": "My Doc"},
            ),
            # Commas and colons inside a quoted value must not shred keys
            (
                "{content: 'a, b: and more', x: 1}",
                {"content": "a, b: and more", "x": 1},
            ),
            # Braces inside a quoted value must not truncate or nest
            (
                "{content: 'has a { brace and } close'}",
                {"content": "has a { brace and } close"},
            ),
            # Apostrophes inside a quoted value (anchored close)
            (
                "{msg: 'don't worry, be happy'}",
                {"msg": "don't worry, be happy"},
            ),
            # Apostrophes in BARE values must not pair across values
            (
                "{a: it's ok, b: don't}",
                {"a": "it's ok", "b": "don't"},
            ),
            # Brace inside a <|"|> string (gap vs mlx-lm's own regex)
            (
                '{a: <|"|>has a { brace<|"|>}',
                {"a": "has a { brace"},
            ),
            # Arrays of quoted strings and of numbers
            (
                "{files: ['a.txt', 'b.txt'], counts: [1, 2]}",
                {"files": ["a.txt", "b.txt"], "counts": [1, 2]},
            ),
            # Nested object with a quoted value containing a comma
            (
                "{opts: {size: 'a4, landscape', deep: {x: 1}}}",
                {"opts": {"size": "a4, landscape", "deep": {"x": 1}}},
            ),
            # Mixed delimiter styles in one call
            (
                '{a: <|"|>x<|"|>, b: \'y\', c: bare, d: 5, e: true}',
                {"a": "x", "b": "y", "c": "bare", "d": 5, "e": True},
            ),
            # Capitalized booleans normalize (models emit True/False)
            ("{flag: True}", {"flag": True}),
        ],
    )
    def test_round_trip(self, rendered, expected):
        assert _gemma4_args_to_json_robust(rendered) == expected

    def test_nul_bytes_cannot_forge_references(self):
        """Literal NUL bytes in model output are data, not placeholders.

        The previous implementation substituted \\x00N\\x00 placeholders for
        captured strings; bare NULs in output forged those references and
        cross-contaminated argument values.
        """
        result = _gemma4_args_to_json_robust(
            '{a: <|"|>captured<|"|>, b: \x000\x00}'
        )
        assert result["a"] == "captured"
        assert result["b"] == "\x000\x00"  # literal, NOT a copy of a

    def test_deep_nesting_fails_cleanly(self):
        """Depth bound surfaces as ValueError, never RecursionError.

        RecursionError is a RuntimeError subclass that no except tuple in
        the parse chain catches; it would escape as a 500.
        """
        deep = "call:f" + "{a: " * 80 + "1" + "}" * 80
        with pytest.raises(ValueError):
            _parse_gemma4_tool_call_fallback(deep)

    def test_deep_standard_json_nesting_fails_cleanly(self):
        """Valid JSON args must not bypass the Gemma 4 depth bound."""
        deep = "call:f" + '{"a": ' * 80 + "1" + "}" * 80
        with pytest.raises(ValueError):
            _parse_gemma4_tool_call_fallback(deep)

    def test_oversized_args_fail_cleanly(self):
        """Args beyond the length cap are a clean no-match, not a hang."""
        huge = "call:f{a: " + "x" * 300_000 + "}"
        with pytest.raises(ValueError):
            _parse_gemma4_tool_call_fallback(huge)

    def test_issue_1830_exact_format(self):
        """The reporter's namespaced-name + single-quoted-args emission."""
        result = _parse_gemma4_tool_call_fallback(
            "call:google:mcp:text_generation:create-pdf-file"
            "{filename: 'output.pdf', title: 'Quarterly Report', "
            "content: 'Revenue grew 12%, costs fell: margins improved. "
            "Don't forget the appendix {tables}.'}"
        )
        assert result["name"] == "google:mcp:text_generation:create-pdf-file"
        assert result["arguments"]["filename"] == "output.pdf"
        assert result["arguments"]["title"] == "Quarterly Report"
        assert result["arguments"]["content"] == (
            "Revenue grew 12%, costs fell: margins improved. "
            "Don't forget the appendix {tables}."
        )

    def test_call_inside_quoted_value_not_double_parsed(self):
        result = _parse_gemma4_tool_call_fallback(
            "call:a{x: 1}\ncall:b{note: 'use call:c{y: 2} later'}"
        )
        assert isinstance(result, list)
        assert [r["name"] for r in result] == ["a", "b"]
        assert result[1]["arguments"]["note"] == "use call:c{y: 2} later"

    def test_one_malformed_call_does_not_drop_siblings(self):
        result = _parse_gemma4_tool_call_fallback(
            "call:bad{:::}\ncall:good{x: 1}"
        )
        assert result == {"name": "good", "arguments": {"x": 1}}


class TestGemma4ParenthesizedArgs:
    """Tests for the parenthesized ``call:name(key=value, ...)`` variant.

    Gemma 4 26B reproducibly degrades to this Python-kwargs form under
    instruction-dense agentic load (#1846).  The nested grammar is identical
    to the canonical curly form, so these tests focus on the new outer shell
    (``()`` instead of ``{}``, ``=`` instead of ``:``) and on confirming the
    #1854 security invariants hold on the new path.
    """

    def test_issue_1846_exact_format(self):
        """The reporter's verbatim emission (3/3 captured samples)."""
        result = _parse_gemma4_tool_call_fallback(
            'call:todo(todos=[{content:<|"|>Clarify goal and draft labor '
            'graph for t_77d3100f<|"|>,id:<|"|>todo-1<|"|>,'
            'status:<|"|>pending<|"|>}])'
        )
        assert result == {
            "name": "todo",
            "arguments": {
                "todos": [
                    {
                        "content": (
                            "Clarify goal and draft labor graph "
                            "for t_77d3100f"
                        ),
                        "id": "todo-1",
                        "status": "pending",
                    }
                ]
            },
        }

    def test_simple_kwargs_with_trailing_bare_value(self):
        # The trailing bare value must not swallow the closing ``)``.
        result = _parse_gemma4_tool_call_fallback(
            'call:get_weather(location=<|"|>Tokyo<|"|>, units=metric)'
        )
        assert result == {
            "name": "get_weather",
            "arguments": {"location": "Tokyo", "units": "metric"},
        }

    def test_empty_paren_call(self):
        result = _parse_gemma4_tool_call_fallback("call:ping()")
        assert result == {"name": "ping", "arguments": {}}

    def test_scalar_value_mix(self):
        result = _parse_gemma4_tool_call_fallback(
            'call:f(a=1, b=<|"|>two<|"|>, c=true, d=null, e=3.5)'
        )
        assert result == {
            "name": "f",
            "arguments": {
                "a": 1, "b": "two", "c": True, "d": None, "e": 3.5
            },
        }

    def test_array_of_scalars(self):
        result = _parse_gemma4_tool_call_fallback("call:f(items=[1, 2, 3])")
        assert result == {"name": "f", "arguments": {"items": [1, 2, 3]}}

    def test_namespaced_name_with_paren(self):
        # Name grammar is shared with the curly head; remap is a later step.
        result = _parse_gemma4_tool_call_fallback("call:google:mcp:todo(x=1)")
        assert result == {"name": "google:mcp:todo", "arguments": {"x": 1}}

    def test_mixed_curly_and_paren_siblings(self):
        result = _parse_gemma4_tool_call_fallback("call:a{x: 1}\ncall:b(y=2)")
        assert result == [
            {"name": "a", "arguments": {"x": 1}},
            {"name": "b", "arguments": {"y": 2}},
        ]

    def test_close_paren_inside_string_does_not_close_span(self):
        result = _parse_gemma4_tool_call_fallback(
            'call:note(text=<|"|>smile :) and (parens)<|"|>)'
        )
        assert result == {
            "name": "note",
            "arguments": {"text": "smile :) and (parens)"},
        }

    def test_equals_inside_string_value_preserved(self):
        result = _parse_gemma4_tool_call_fallback(
            'call:f(expr=<|"|>a=b=c<|"|>)'
        )
        assert result == {"name": "f", "arguments": {"expr": "a=b=c"}}

    def test_curly_value_with_parens_unaffected(self):
        # Regression guard: ``)`` is ordinary content in the curly form, so a
        # value may contain parentheses.  This is why ``)`` is a bare-value
        # terminator only when a paren container is open.
        result = _gemma4_args_to_json_robust("{expr: f(x)}")
        assert result == {"expr": "f(x)"}

    def test_single_quoted_value_before_close_paren(self):
        # The degeneration is a Python-kwargs shell, so single-quoted strings
        # are its native string form (observed live on gemma-4-26b).  When such
        # a value is the last argument its closing quote sits right before the
        # call's ``)``, so ``)`` must anchor a single-quote close exactly as
        # ``,``/``}``/``]`` do; otherwise the quotes leak into the value.
        result = _parse_gemma4_tool_call_fallback(
            "call:create_todo(content='clarify the goal')"
        )
        assert result == {
            "name": "create_todo",
            "arguments": {"content": "clarify the goal"},
        }

    def test_single_quoted_values_parity_curly_and_paren(self):
        # Both shells normalize single-quoted values identically; the paren
        # form must not retain the quotes the curly form strips.
        curly = _gemma4_args_to_json_robust("{a: 'x', b: 'y'}")
        paren = _parse_gemma4_tool_call_fallback("call:f(a='x', b='y')")
        assert curly == {"a": "x", "b": "y"}
        assert paren == {"name": "f", "arguments": {"a": "x", "b": "y"}}

    def test_oversized_paren_args_fail_cleanly(self):
        """Args beyond the length cap are a clean no-match, not a hang.

        Guards the #1854 ReDoS: the paren head's optional/tolerant prefix on
        the ``re`` module would backtrack O(n^2) hunting an opening ``(``.
        """
        huge = "call:f(a=" + "x" * 300_000 + ")"
        with pytest.raises(ValueError):
            _parse_gemma4_tool_call_fallback(huge)

    def test_deep_paren_nesting_fails_cleanly(self):
        """Depth bound surfaces as ValueError, never RecursionError."""
        deep = "call:f(" + "a={" * 80 + "1" + "}" * 80 + ")"
        with pytest.raises(ValueError):
            _parse_gemma4_tool_call_fallback(deep)

    def test_nul_bytes_cannot_forge_references_paren(self):
        """The NUL-placeholder forge vector stays closed on the paren path."""
        result = _gemma4_args_to_json_robust(
            '(a=<|"|>captured<|"|>, b=\x000\x00)'
        )
        assert result["a"] == "captured"
        assert result["b"] == "\x000\x00"  # literal, NOT a copy of a

    def test_paren_call_inside_quoted_value_not_double_parsed(self):
        result = _parse_gemma4_tool_call_fallback(
            'call:a(x=1)\ncall:b(note=<|"|>use call:c(y=2) later<|"|>)'
        )
        assert isinstance(result, list)
        assert [r["name"] for r in result] == ["a", "b"]
        assert result[1]["arguments"]["note"] == "use call:c(y=2) later"


class TestRemapToolCallNames:
    """Tests for _remap_tool_call_names() (#1830)."""

    @staticmethod
    def _call(name):
        return ToolCall(
            id="call_test",
            type="function",
            function=FunctionCall(name=name, arguments="{}"),
        )

    @staticmethod
    def _tools(*names):
        return [{"type": "function", "function": {"name": n}} for n in names]

    def test_namespaced_name_remaps_to_unique_suffix(self, caplog):
        calls = [self._call("google:mcp:text_generation:create-pdf-file")]
        with caplog.at_level(logging.INFO, logger="omlx.api.tool_calling"):
            _remap_tool_call_names(calls, self._tools("create-pdf-file"))
        assert calls[0].function.name == "create-pdf-file"
        assert any("Remapped" in msg for msg in caplog.messages)

    def test_exact_match_is_untouched(self):
        calls = [self._call("tavily:search")]
        _remap_tool_call_names(calls, self._tools("tavily:search", "search"))
        assert calls[0].function.name == "tavily:search"

    def test_ambiguous_suffixes_keep_verbatim(self):
        """Two registered suffix candidates: refuse to guess."""
        calls = [self._call("ns:text_generation:create-pdf-file")]
        _remap_tool_call_names(
            calls,
            self._tools("text_generation:create-pdf-file", "create-pdf-file"),
        )
        assert calls[0].function.name == "ns:text_generation:create-pdf-file"

    def test_endswith_attack_does_not_remap(self):
        """Boundary-aligned matching: 'evilcreate-pdf-file' must not coerce
        into 'create-pdf-file' (no ':' boundary), and neither must a
        namespaced name whose last segment merely ENDS with a registered
        name."""
        calls = [self._call("evilcreate-pdf-file")]
        _remap_tool_call_names(calls, self._tools("create-pdf-file"))
        assert calls[0].function.name == "evilcreate-pdf-file"

        calls = [self._call("evil:xcreate-pdf-file")]
        _remap_tool_call_names(calls, self._tools("create-pdf-file"))
        assert calls[0].function.name == "evil:xcreate-pdf-file"

    def test_no_suffix_match_keeps_verbatim(self):
        calls = [self._call("ns:unknown-tool")]
        _remap_tool_call_names(calls, self._tools("create-pdf-file"))
        assert calls[0].function.name == "ns:unknown-tool"

    def test_no_tools_is_noop(self):
        calls = [self._call("ns:create-pdf-file")]
        _remap_tool_call_names(calls, None)
        assert calls[0].function.name == "ns:create-pdf-file"
        _remap_tool_call_names(calls, [])
        assert calls[0].function.name == "ns:create-pdf-file"


class TestParseToolCallsGemma4RealParser:
    """E2e through parse_tool_calls with mlx-lm's REAL Gemma 4 parser.

    Uses the real parser and real PAIRED markers (tool_call_end is
    "<tool_call|>"; parse_tool_calls takes a different extraction branch
    for paired vs one-sided markers, so a mock without the end marker
    would test the wrong branch).  This also pins the dispatch contract:
    if a future mlx-lm bump makes the native parser accept colon names,
    test_native_parser_rejects_namespaced_names fails and tells us the
    fallback is no longer exercised.
    """

    ISSUE_PAYLOAD = (
        "call:google:mcp:text_generation:create-pdf-file"
        "{filename: 'output.pdf', content: 'Revenue grew 12%, costs "
        "fell: margins improved.'}"
    )

    @staticmethod
    def _make_real_gemma4_tokenizer():
        from mlx_lm.tool_parsers import gemma4 as mlx_gemma4

        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = mlx_gemma4.tool_call_start
        tok.tool_call_end = mlx_gemma4.tool_call_end
        tok.tool_parser = mlx_gemma4.parse_tool_call
        return tok

    @staticmethod
    def _pdf_tool():
        return [
            {
                "type": "function",
                "function": {
                    "name": "create-pdf-file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
            }
        ]

    def test_native_parser_rejects_namespaced_names(self):
        """Dispatch contract: mlx-lm's parser raises on colon names, which
        is what routes #1830's emission into the oMLX fallback."""
        from mlx_lm.tool_parsers import gemma4 as mlx_gemma4

        with pytest.raises(ValueError):
            mlx_gemma4.parse_tool_call(self.ISSUE_PAYLOAD)

    def test_issue_1830_end_to_end(self):
        """Marker-wrapped issue payload parses and remaps end to end."""
        tok = self._make_real_gemma4_tokenizer()
        text = (
            f"{tok.tool_call_start}{self.ISSUE_PAYLOAD}{tok.tool_call_end}"
        )

        cleaned, tool_calls = parse_tool_calls(text, tok, self._pdf_tool())

        assert tool_calls is not None and len(tool_calls) == 1
        assert tool_calls[0].function.name == "create-pdf-file"
        args = json.loads(tool_calls[0].function.arguments)
        assert args["filename"] == "output.pdf"
        assert args["content"] == (
            "Revenue grew 12%, costs fell: margins improved."
        )
        assert tok.tool_call_start not in cleaned
        assert tok.tool_call_end not in cleaned

    def test_thinking_path_promotes_remapped_call(self):
        """Defect #4: a namespaced call in THINKING content must survive
        extract_tool_calls_with_thinking's exact-name validity filter.
        Without post-parse remapping, the filter silently drops the
        cleanly parsed call because the emitted name matches no tool."""
        tok = self._make_real_gemma4_tokenizer()
        thinking = (
            f"{tok.tool_call_start}"
            "call:google:mcp:text_generation:create-pdf-file"
            "{filename: 'output.pdf'}"
            f"{tok.tool_call_end}"
        )

        extraction = extract_tool_calls_with_thinking(
            thinking_content=thinking,
            regular_content="Some unrelated prose.",
            tokenizer=tok,
            tools=self._pdf_tool(),
        )

        assert extraction.tool_calls is not None
        assert extraction.tool_calls[0].function.name == "create-pdf-file"
        assert extraction.tool_calls_from_thinking


class TestEnrichToolParamsForGemma4:
    """Tests for enrich_tool_params_for_gemma4()."""

    def test_renames_description_param(self):
        """Parameter named 'description' gets renamed to 'param_description'."""
        tools = [{"function": {"name": "delegate", "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["description", "prompt"],
        }}}]
        result = enrich_tool_params_for_gemma4(tools)
        props = result[0]["function"]["parameters"]["properties"]
        assert "param_description" in props
        assert "description" not in props
        required = result[0]["function"]["parameters"]["required"]
        assert "param_description" in required
        assert "description" not in required

    def test_does_not_rename_non_colliding_params(self):
        """Parameters like 'name' and 'type' are NOT renamed (not in colliding set)."""
        tools = [{"function": {"name": "create", "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name", "type", "count"],
        }}}]
        result = enrich_tool_params_for_gemma4(tools)
        props = result[0]["function"]["parameters"]["properties"]
        assert "name" in props
        assert "type" in props
        assert "count" in props

    def test_adds_description_to_required_params(self):
        """Required params without descriptions get auto-generated ones."""
        tools = [{"function": {"name": "search", "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }}}]
        result = enrich_tool_params_for_gemma4(tools)
        prop = result[0]["function"]["parameters"]["properties"]["query"]
        assert "description" in prop
        assert "REQUIRED" in prop["description"]
        assert "'query'" in prop["description"]

    def test_preserves_existing_descriptions(self):
        """Params that already have descriptions are left unchanged."""
        tools = [{"function": {"name": "search", "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
            },
            "required": ["query"],
        }}}]
        result = enrich_tool_params_for_gemma4(tools)
        prop = result[0]["function"]["parameters"]["properties"]["query"]
        assert prop["description"] == "Search query text"

    def test_does_not_mutate_input(self):
        """Original tool definitions are not modified."""
        tools = [{"function": {"name": "delegate", "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
            },
            "required": ["description"],
        }}}]
        original_props = list(tools[0]["function"]["parameters"]["properties"].keys())
        enrich_tool_params_for_gemma4(tools)
        assert list(tools[0]["function"]["parameters"]["properties"].keys()) == original_props

    def test_empty_tools_list(self):
        """Empty tools list returns empty list."""
        assert enrich_tool_params_for_gemma4([]) == []

    def test_tool_without_parameters(self):
        """Tools without parameters are passed through unchanged."""
        tools = [{"function": {"name": "get_time"}}]
        result = enrich_tool_params_for_gemma4(tools)
        assert result[0]["function"]["name"] == "get_time"


class TestRestoreGemma4ParamNames:
    """Tests for restore_gemma4_param_names()."""

    def test_restores_renamed_description(self):
        """param_description is restored to description."""
        args = {"param_description": "audit the code", "prompt": "check for bugs"}
        result = restore_gemma4_param_names(args)
        assert result == {"description": "audit the code", "prompt": "check for bugs"}

    def test_does_not_strip_non_colliding_prefix(self):
        """param_count should NOT be renamed to count (not a colliding param)."""
        args = {"param_count": 5, "query": "test"}
        result = restore_gemma4_param_names(args)
        assert result == {"param_count": 5, "query": "test"}

    def test_leaves_regular_params_unchanged(self):
        """Regular params pass through unchanged."""
        args = {"prompt": "hello", "count": 3}
        result = restore_gemma4_param_names(args)
        assert result == {"prompt": "hello", "count": 3}

    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        assert restore_gemma4_param_names({}) == {}

    def test_round_trip(self):
        """Enrich then restore produces original param names."""
        tools = [{"function": {"name": "delegate", "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["description", "prompt"],
        }}}]
        enriched = enrich_tool_params_for_gemma4(tools)
        # Simulate model output using enriched param names
        enriched_props = enriched[0]["function"]["parameters"]["properties"]
        model_args = {k: "test" for k in enriched_props}
        restored = restore_gemma4_param_names(model_args)
        assert set(restored.keys()) == {"description", "prompt"}


class TestParseToolCallsNativeParserListReturn:
    """MiniMax M2 parser returns a list when a single <minimax:tool_call>
    block contains multiple <invoke>s. parse_tool_calls() must flatten that
    into one ToolCall per invoke, not drop the whole block.
    """

    def test_single_block_multiple_invokes(self):
        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "<minimax:tool_call>"
        tok.tool_call_end = "</minimax:tool_call>"
        tok.tool_parser = lambda text, tools: [
            {"name": "list_files", "arguments": {"path": "."}},
            {"name": "read_file", "arguments": {"path": "README.md"}},
        ]

        text = (
            "<minimax:tool_call>"
            "<invoke name=\"list_files\"><parameter name=\"path\">.</parameter></invoke>"
            "<invoke name=\"read_file\"><parameter name=\"path\">README.md</parameter></invoke>"
            "</minimax:tool_call>"
        )
        cleaned, tool_calls = parse_tool_calls(text, tok)
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0].function.name == "list_files"
        assert tool_calls[1].function.name == "read_file"
        assert json.loads(tool_calls[1].function.arguments) == {"path": "README.md"}

    def test_single_block_single_invoke_returns_dict(self):
        """Regression guard: single-invoke case still returns a dict."""
        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "<minimax:tool_call>"
        tok.tool_call_end = "</minimax:tool_call>"
        tok.tool_parser = lambda text, tools: {
            "name": "list_files",
            "arguments": {"path": "."},
        }

        text = (
            "<minimax:tool_call>"
            "<invoke name=\"list_files\"><parameter name=\"path\">.</parameter></invoke>"
            "</minimax:tool_call>"
        )
        cleaned, tool_calls = parse_tool_calls(text, tok)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "list_files"

    def test_multiple_blocks_each_with_multiple_invokes(self):
        """Two blocks, each returning a list — total 4 tool calls."""
        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "<minimax:tool_call>"
        tok.tool_call_end = "</minimax:tool_call>"

        def parser(text, tools):
            if "first" in text:
                return [
                    {"name": "first_a", "arguments": {}},
                    {"name": "first_b", "arguments": {}},
                ]
            return [
                {"name": "second_a", "arguments": {}},
                {"name": "second_b", "arguments": {}},
            ]

        tok.tool_parser = parser
        text = (
            "<minimax:tool_call>first</minimax:tool_call>"
            "<minimax:tool_call>second</minimax:tool_call>"
        )
        cleaned, tool_calls = parse_tool_calls(text, tok)
        assert tool_calls is not None
        assert [tc.function.name for tc in tool_calls] == [
            "first_a",
            "first_b",
            "second_a",
            "second_b",
        ]


class TestSerializeToolCallArguments:
    """Tests for `_serialize_tool_call_arguments`.

    Guards the server-side exit: whatever the parser returns must leave
    omlx as a valid JSON-object string so a subsequent turn's chat template
    (which iterates `arguments.items()`) never crashes on the echo.
    """

    def test_dict_roundtrip(self):
        result = _serialize_tool_call_arguments({"location": "Tokyo", "unit": "c"})
        assert json.loads(result) == {"location": "Tokyo", "unit": "c"}

    def test_empty_dict(self):
        assert _serialize_tool_call_arguments({}) == "{}"

    def test_non_ascii_preserved(self):
        """ensure_ascii=False is applied so CJK/emoji stay readable."""
        result = _serialize_tool_call_arguments({"city": "서울"})
        assert "서울" in result

    def test_non_dict_bare_string_coerced_to_empty(self, caplog):
        with caplog.at_level(logging.WARNING, logger="omlx.api.tool_calling"):
            result = _serialize_tool_call_arguments("Tokyo")
        assert result == "{}"
        assert any("non-dict" in r.message for r in caplog.records)

    def test_non_dict_list_coerced_to_empty(self, caplog):
        with caplog.at_level(logging.WARNING, logger="omlx.api.tool_calling"):
            result = _serialize_tool_call_arguments([1, 2])
        assert result == "{}"

    def test_non_dict_none_coerced_to_empty(self):
        assert _serialize_tool_call_arguments(None) == "{}"

    def test_json_object_string_preserved(self, caplog):
        """mlx-vlm/mlx-lm gemma4 parser hands back a JSON-object string per
        the OpenAI spec; the validator must accept it instead of dropping it."""
        with caplog.at_level(logging.WARNING, logger="omlx.api.tool_calling"):
            result = _serialize_tool_call_arguments('{"command": "ls /tmp\\n"}')
        assert json.loads(result) == {"command": "ls /tmp\n"}
        assert not any("non-dict" in r.message for r in caplog.records)

    def test_json_array_string_coerced_to_empty(self, caplog):
        """JSON arrays/scalars do not satisfy ``arguments.items()`` so they
        must still be coerced."""
        with caplog.at_level(logging.WARNING, logger="omlx.api.tool_calling"):
            result = _serialize_tool_call_arguments("[1, 2]")
        assert result == "{}"
        assert any("non-dict" in r.message for r in caplog.records)


class TestToolCallStreamFilterGemma4StrayClose:
    """Stray closing-marker suppression for Gemma 4 <|tool_call>/<tool_call|>."""

    def _make_filter(self):
        return ToolCallStreamFilter(
            _make_tokenizer_with_end("<|tool_call>", "<tool_call|>")
        )

    def test_stray_close_marker_alone_dropped(self):
        """Bare <tool_call|> with no preceding open is suppressed."""
        f = self._make_filter()
        result = f.feed("<tool_call|>")
        result += f.finish()
        assert result == ""

    def test_stray_close_after_text_dropped(self):
        """Text before a stray close passes through; the close itself is dropped."""
        f = self._make_filter()
        result = f.feed("hello<tool_call|>")
        result += f.finish()
        assert result == "hello"

    def test_stray_close_split_across_feeds_dropped(self):
        """Stray close split across two feed() calls is dropped."""
        f = self._make_filter()
        r1 = f.feed("<tool_call")
        r2 = f.feed("|>")
        result = r1 + r2 + f.finish()
        assert result == ""

    def test_normal_open_close_pair_still_suppressed(self):
        """Regression: a valid open/close pair is still fully suppressed."""
        f = self._make_filter()
        result = f.feed('<|tool_call>call:search{"q":"test"}<tool_call|>')
        result += f.finish()
        assert result == ""

    def test_multiple_stray_closes_in_one_delta_all_dropped(self):
        """Multiple stray close tokens in a single delta are all removed."""
        f = self._make_filter()
        result = f.feed("a<tool_call|>b<tool_call|>c")
        result += f.finish()
        assert result == "abc"

    def test_default_xml_close_in_prose_passes_through(self):
        """Prose containing </tool_call> (hardcoded fallback pair) is not stripped.

        The stray-close strip is scoped to the tokenizer-configured marker only.
        A model discussing XML tag syntax must not have its output corrupted.
        """
        f = self._make_filter()
        result = f.feed("The closing tag is </tool_call> here.")
        result += f.finish()
        assert result == "The closing tag is </tool_call> here."

    def test_configured_xml_close_in_prose_passes_through(self):
        """Configured XML close markers are not treated like Gemma 4 stray closes."""
        f = ToolCallStreamFilter(
            _make_tokenizer_with_end("<tool_call>", "</tool_call>")
        )
        result = f.feed("The closing tag is </tool_call> here.")
        result += f.finish()
        assert result == "The closing tag is </tool_call> here."

    def test_configured_namespaced_close_in_prose_passes_through(self):
        """Configured namespaced close markers are preserved in prose."""
        f = ToolCallStreamFilter(
            _make_tokenizer_with_end(
                "<minimax:tool_call>",
                "</minimax:tool_call>",
            )
        )
        result = f.feed("The marker </minimax:tool_call> is a close marker.")
        result += f.finish()
        assert result == "The marker </minimax:tool_call> is a close marker."

    def test_stray_close_split_after_pipe_dropped(self):
        """Stray close split at the | boundary (<tool_call| + >) is reassembled and dropped."""
        f = self._make_filter()
        r1 = f.feed("<tool_call|")
        r2 = f.feed(">")
        result = r1 + r2 + f.finish()
        assert result == ""

    def test_stray_close_split_after_pipe_with_prose_dropped(self):
        """Prose-wrapped stray close split at the | boundary: prose passes, marker dropped."""
        f = self._make_filter()
        r1 = f.feed("hello<tool_call|")
        r2 = f.feed("> world")
        result = r1 + r2 + f.finish()
        assert result == "hello world"

    def test_valid_pair_plus_stray_close_in_same_delta(self):
        """Valid open/close pair suppressed; trailing stray close in same delta dropped."""
        f = self._make_filter()
        result = f.feed('<|tool_call>call()<tool_call|> extra<tool_call|>')
        result += f.finish()
        assert result == " extra"


class TestSchemaAwareFallbackCoercion:
    """Regression tests for issue #2332.

    The XML fallback parsers used to parse each parameter value with a
    bare json.loads and keep the raw string on failure, so a slightly
    malformed array/object param silently degraded to a string. With the
    tool schema threaded through, declared container params get a
    bracket-repair pass and declared string params are no longer
    json-coerced.
    """

    MALFORMED_EDITS = '[{"range": ["3#8C", "3#8C"], "lines": ["    value: 100,"}]}'
    REPAIRED_EDITS = [{"range": ["3#8C", "3#8C"], "lines": ["    value: 100,"]}]

    EDIT_TOOL = {
        "type": "function",
        "function": {
            "name": "edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {"type": "array"},
                },
            },
        },
    }

    def _qwen_text(self, edits_val):
        return (
            "<tool_call>\n<function=edit>\n"
            "<parameter=path>\nconfig.py\n</parameter>\n"
            f"<parameter=edits>\n{edits_val}\n</parameter>\n"
            "</function>\n</tool_call>"
        )

    def test_qwen_branch_repairs_malformed_array(self):
        _, calls = _parse_xml_tool_calls(
            self._qwen_text(self.MALFORMED_EDITS), [self.EDIT_TOOL]
        )
        args = json.loads(calls[0].function.arguments)
        assert args["edits"] == self.REPAIRED_EDITS
        assert args["path"] == "config.py"

    def test_glm_branch_repairs_malformed_array(self):
        text = (
            "<tool_call>edit<arg_key>edits</arg_key>"
            f"<arg_value>{self.MALFORMED_EDITS}</arg_value></tool_call>"
        )
        _, calls = _parse_xml_tool_calls(text, [self.EDIT_TOOL])
        args = json.loads(calls[0].function.arguments)
        assert args["edits"] == self.REPAIRED_EDITS

    def test_namespaced_branch_repairs_malformed_array(self):
        text = (
            '<minimax:tool_call><invoke name="edit">'
            f'<parameter name="edits">{self.MALFORMED_EDITS}</parameter>'
            "</invoke></minimax:tool_call>"
        )
        _, calls = _parse_namespaced_tool_calls(text, "minimax", [self.EDIT_TOOL])
        args = json.loads(calls[0].function.arguments)
        assert args["edits"] == self.REPAIRED_EDITS

    def test_declared_string_param_not_json_coerced(self):
        """A numeric-looking value for a declared string param stays a string."""
        tool = {
            "type": "function",
            "function": {
                "name": "edit",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
        text = (
            "<tool_call>\n<function=edit>\n"
            "<parameter=path>123</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, calls = _parse_xml_tool_calls(text, [tool])
        args = json.loads(calls[0].function.arguments)
        assert args["path"] == "123"

    def test_no_tools_keeps_legacy_behavior(self):
        """Without tools, malformed values still fall back to the raw string."""
        _, calls = _parse_xml_tool_calls(self._qwen_text(self.MALFORMED_EDITS))
        args = json.loads(calls[0].function.arguments)
        assert args["edits"] == self.MALFORMED_EDITS
        # And valid JSON values still get best-effort parsed.
        _, calls = _parse_xml_tool_calls(self._qwen_text('[{"a": 1}]'))
        args = json.loads(calls[0].function.arguments)
        assert args["edits"] == [{"a": 1}]

    def test_undeclared_param_keeps_legacy_behavior(self):
        """Params missing from the schema keep the best-effort JSON parse."""
        text = (
            "<tool_call>\n<function=edit>\n"
            "<parameter=extra>42</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, calls = _parse_xml_tool_calls(text, [self.EDIT_TOOL])
        args = json.loads(calls[0].function.arguments)
        assert args["extra"] == 42

    def test_unrepairable_container_keeps_raw_string(self):
        """When repair fails the raw string survives so the call is not lost."""
        text = self._qwen_text('[{"a": 1 &&& oops')
        _, calls = _parse_xml_tool_calls(text, [self.EDIT_TOOL])
        args = json.loads(calls[0].function.arguments)
        assert args["edits"] == '[{"a": 1 &&& oops'

    def test_python_literal_container_accepted(self):
        """A Python-literal dict/list (single quotes) parses via literal_eval."""
        text = self._qwen_text("[{'a': True, 'b': None}]")
        _, calls = _parse_xml_tool_calls(text, [self.EDIT_TOOL])
        args = json.loads(calls[0].function.arguments)
        assert args["edits"] == [{"a": True, "b": None}]

    def test_native_failure_fallback_gets_tools(self):
        """parse_tool_calls threads tools into the per-match XML fallback."""

        def failing_parser(text, tools):
            raise SyntaxError("invalid syntax (<unknown>, line 1)")

        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "<tool_call>"
        tok.tool_call_end = "</tool_call>"
        tok.tool_parser = failing_parser
        _, calls = parse_tool_calls(
            self._qwen_text(self.MALFORMED_EDITS), tok, [self.EDIT_TOOL]
        )
        assert calls is not None
        args = json.loads(calls[0].function.arguments)
        assert args["edits"] == self.REPAIRED_EDITS

    def test_int_and_number_coercion(self):
        props = {
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
        }
        assert _coerce_param_value("7", "count", props, "t") == 7
        assert _coerce_param_value("2.5", "ratio", props, "t") == 2.5
        assert _coerce_param_value("3.0", "ratio", props, "t") == 3
        assert _coerce_param_value("true", "flag", props, "t") is True
        assert _coerce_param_value("null", "count", props, "t") is None

    def test_string_type_decodes_json_quoted_literal(self):
        """A JSON-quoted value for a string param decodes to the bare string."""
        props = {"city": {"type": "string"}}
        # Quoted literals (e.g. MiniMax) drop their JSON encoding.
        assert _coerce_param_value('"SF"', "city", props, "t") == "SF"
        assert _coerce_param_value('""', "city", props, "t") == ""
        assert _coerce_param_value('"he said \\"hi\\""', "city", props, "t") == 'he said "hi"'
        # Plain values that merely look like JSON stay verbatim as strings.
        assert _coerce_param_value("SF", "city", props, "t") == "SF"
        assert _coerce_param_value("42", "city", props, "t") == "42"
        assert _coerce_param_value('{"a": 1}', "city", props, "t") == '{"a": 1}'
        # An unbalanced quote is not a JSON literal; keep it raw.
        assert _coerce_param_value('"unterminated', "city", props, "t") == '"unterminated'

    def test_union_type_list_keeps_legacy_behavior(self):
        """A JSON Schema union type list falls back to best-effort parsing."""
        props = {"v": {"type": ["string", "null"]}}
        assert _coerce_param_value("hello", "v", props, "t") == "hello"
        assert _coerce_param_value("123", "v", props, "t") == 123

    def test_repair_json_value(self):
        assert _repair_json_value('[{"a": [1, 2}]}') == [{"a": [1, 2]}]
        assert _repair_json_value('{"a": "unterminated') == {"a": "unterminated"}
        assert _repair_json_value("not json at all") is None


class TestToolCallMarkerInArguments:
    """Literal tool-call markers inside argument strings (#2507).

    A non-greedy ``start(.*?)end`` match stops at the first close marker, so a
    call whose argument embeds that marker was truncated mid-JSON and dropped
    silently. Payload boundaries are now found via JSON decoding.
    """

    PAYLOAD = "text with a literal </tool_call> inside"

    @staticmethod
    def _raw(body):
        inner = json.dumps(
            {"name": "note_write", "arguments": {"title": "t", "body": body}},
            ensure_ascii=False,
        )
        return f"<tool_call>\n{inner}\n</tool_call>"

    @staticmethod
    def _tokenizer():
        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "<tool_call>"
        tok.tool_call_end = "</tool_call>"
        tok.tool_parser = lambda text, tools: json.loads(text)
        return tok

    def test_marker_span_helpers_span_the_embedded_marker(self):
        text = self._raw(self.PAYLOAD)
        payloads = _marker_payloads(text, "<tool_call>", "</tool_call>")
        assert len(payloads) == 1
        assert json.loads(payloads[0])["arguments"]["body"] == self.PAYLOAD
        assert _strip_marker_spans(text, "<tool_call>", "</tool_call>").strip() == ""

    def test_native_path_preserves_argument_with_close_marker(self):
        cleaned, calls = parse_tool_calls(self._raw(self.PAYLOAD), self._tokenizer())
        assert calls is not None and len(calls) == 1
        assert json.loads(calls[0].function.arguments)["body"] == self.PAYLOAD
        # The whole envelope is consumed, so no markup leaks into content.
        assert cleaned == ""

    def test_xml_fallback_preserves_argument_with_close_marker(self):
        cleaned, calls = _parse_xml_tool_calls(self._raw(self.PAYLOAD))
        assert calls is not None and len(calls) == 1
        assert json.loads(calls[0].function.arguments)["body"] == self.PAYLOAD
        assert cleaned == ""

    def test_hermes_payload_with_close_marker(self):
        inner = json.dumps(
            {"name": "note_write", "arguments": {"body": "a <|tool_call_end|> b"}}
        )
        text = f"<|tool_call_start|>{inner}<|tool_call_end|>"
        cleaned, calls = _parse_hermes_tool_calls(text)
        assert calls is not None and len(calls) == 1
        assert json.loads(calls[0].function.arguments)["body"] == "a <|tool_call_end|> b"
        assert cleaned == ""

    def test_streaming_does_not_leak_tail_as_content(self):
        """The stream filter must not end the envelope on the embedded marker."""
        raw = self._raw(self.PAYLOAD)
        for chunk in (1, 3, 7, 64):
            filt = ToolCallStreamFilter(self._tokenizer())
            emitted = "".join(
                filt.feed(raw[i : i + chunk]) for i in range(0, len(raw), chunk)
            )
            emitted += filt.finish()
            assert emitted == "", f"leaked at chunk size {chunk}: {emitted!r}"

    def test_multiple_calls_still_split(self):
        a = json.dumps({"name": "f", "arguments": {"s": "has </tool_call> inside"}})
        b = json.dumps({"name": "g", "arguments": {"y": 2}})
        text = f"pre <tool_call>{a}</tool_call> mid <tool_call>{b}</tool_call> post"
        payloads = _marker_payloads(text, "<tool_call>", "</tool_call>")
        assert [json.loads(p)["name"] for p in payloads] == ["f", "g"]
        assert (
            _strip_marker_spans(text, "<tool_call>", "</tool_call>")
            == "pre  mid  post"
        )

    def test_non_json_dialect_keeps_first_match_behaviour(self):
        """GLM-style payloads are not JSON, so boundary detection must not change them."""
        text = (
            "<tool_call>myfunc<arg_key>k</arg_key>"
            "<arg_value>v</arg_value></tool_call>"
        )
        assert _marker_payloads(text, "<tool_call>", "</tool_call>") == [
            "myfunc<arg_key>k</arg_key><arg_value>v</arg_value>"
        ]

    def test_unterminated_envelope_yields_no_span(self):
        text = '<tool_call>{"name": "f", "arguments": {}}'
        assert _marker_payloads(text, "<tool_call>", "</tool_call>") == []
        assert _strip_marker_spans(text, "<tool_call>", "</tool_call>") == text

    def test_incomplete_json_falls_back_to_first_marker(self):
        """Never-completing JSON must not swallow the rest of the message."""
        text = '<tool_call>{"name": "f", "arguments": {"s": "oops </tool_call>'
        assert _marker_payloads(text, "<tool_call>", "</tool_call>") == [
            '{"name": "f", "arguments": {"s": "oops '
        ]

    def test_boundary_scan_never_raises_on_deep_nesting(self):
        """Boundary detection must degrade, not explode, on hostile nesting.

        ``raw_decode`` recurses per nesting level, so deeply nested output
        raises RecursionError rather than a decode error. A boundary hint must
        never turn into an exception escaping the parse chain.
        """
        deep = "[" * 100_000
        assert _json_value_end(deep, 0) is None
        text = f"<tool_call>{deep}</tool_call>"
        assert _marker_payloads(text, "<tool_call>", "</tool_call>") == [deep]

    def test_boundary_scan_stays_linear_on_adversarial_output(self):
        """Guard the linear-time rule for untrusted model output.

        Locating the payload end must not re-scan the accumulated buffer once
        per chunk: an unterminated JSON object stuffed with close markers
        would then cost quadratic time. Model output is attacker-influenceable
        via prompt injection, so this is a DoS surface, not just a slowdown.
        """
        import time

        def elapsed(markers):
            evil = (
                "<tool_call>"
                + '{"name":"f","arguments":{"s":"'
                + "</tool_call>" * markers
            )
            filt = ToolCallStreamFilter(self._tokenizer())
            start = time.perf_counter()
            for i in range(0, len(evil), 64):
                filt.feed(evil[i : i + 64])
            filt.finish()
            return time.perf_counter() - start

        elapsed(400)  # warm up the interpreter before timing
        small = max(elapsed(400), 1e-4)
        large = elapsed(3200)
        # 8x the input: linear would be ~8x, quadratic ~64x. Allow generous
        # headroom for a loaded CI box while still catching quadratic growth.
        assert large / small < 24, f"superlinear growth: {small=} {large=}"


class TestQwen3CoderMarkerInArguments:
    """qwen3_coder XML dialect with literal markers in a value (#2507).

    Qwen3.5/3.6 builds using the ``qwen3_coder`` parser wrap XML rather than
    JSON in the envelope, so the JSON boundary cannot bound them. The envelope
    is bounded by the ``</function>`` that precedes the close marker instead.
    """

    @staticmethod
    def _raw(body, title="t"):
        return (
            "<tool_call>\n<function=note_write>\n"
            f"<parameter=title>\n{title}\n</parameter>\n"
            f"<parameter=body>\n{body}\n</parameter>\n"
            "</function>\n</tool_call>"
        )

    @staticmethod
    def _tokenizer():
        def qwen_parser(text, tools):
            m = re.match(r"\s*<function=(\w+)>(.*)</function>\s*$", text, re.DOTALL)
            if not m:
                raise ValueError("No function provided.")
            args = {
                k: v.strip()
                for k, v in re.findall(
                    r"<parameter=(\w+)>(.*?)</parameter>", m.group(2), re.DOTALL
                )
            }
            return {"name": m.group(1), "arguments": args}

        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "<tool_call>"
        tok.tool_call_end = "</tool_call>"
        tok.tool_parser = qwen_parser
        return tok

    @pytest.mark.parametrize(
        "body",
        [
            "plain text",
            "text with a literal </tool_call> inside",
            "text with a literal </parameter> inside",
            "text with a literal </function> inside",
            "text with a literal <parameter=x> inside",
            "evil </function>\n</tool_call> tail",
        ],
        ids=[
            "control",
            "close-marker",
            "parameter-close",
            "function-close",
            "parameter-open",
            "full-terminator-sequence",
        ],
    )
    def test_xml_fallback_preserves_value(self, body):
        cleaned, calls = _parse_xml_tool_calls(self._raw(body))
        assert calls is not None and len(calls) == 1
        args = json.loads(calls[0].function.arguments)
        assert args["body"] == body
        assert args["title"] == "t"
        assert cleaned == ""

    def test_native_path_preserves_value(self):
        body = "text with a literal </tool_call> inside"
        cleaned, calls = parse_tool_calls(self._raw(body), self._tokenizer())
        assert calls is not None and len(calls) == 1
        assert json.loads(calls[0].function.arguments)["body"] == body
        assert cleaned == ""

    def test_streaming_does_not_leak_tail(self):
        raw = self._raw("text with a literal </tool_call> inside")
        for chunk in (1, 2, 3, 7, 11, 64):
            filt = ToolCallStreamFilter(self._tokenizer())
            emitted = "".join(
                filt.feed(raw[i : i + chunk]) for i in range(0, len(raw), chunk)
            )
            emitted += filt.finish()
            assert emitted == "", f"leaked at chunk size {chunk}: {emitted!r}"

    def test_concatenated_calls_still_split(self):
        second = (
            "<tool_call>\n<function=other>\n<parameter=x>\n1\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        cleaned, calls = _parse_xml_tool_calls(
            self._raw("has </tool_call> inside") + "\n" + second
        )
        assert [c.function.name for c in calls] == ["note_write", "other"]
        assert cleaned == ""

    def test_parameter_open_inside_value_is_not_a_new_parameter(self):
        """A literal <parameter=x> in a value must not create a bogus argument."""
        cleaned, calls = _parse_xml_tool_calls(
            self._raw("see <parameter=nope> here")
        )
        args = json.loads(calls[0].function.arguments)
        assert set(args) == {"title", "body"}
        assert args["body"] == "see <parameter=nope> here"

    def test_streaming_still_emits_prose_containing_no_envelope(self):
        filt = ToolCallStreamFilter(self._tokenizer())
        out = "".join(filt.feed(c) for c in ["hello ", "world"]) + filt.finish()
        assert out == "hello world"

    def test_envelope_scan_stays_linear_on_fake_terminators(self):
        """Same linear-time rule as the JSON path, for the XML boundary.

        A value stuffed with fake ``</function></tool_call>`` sequences is the
        worst case: every one is a candidate envelope end that has to be
        rejected. That must not become quadratic.
        """
        import time

        def elapsed(count):
            evil = (
                "<tool_call>\n<function=f>\n<parameter=b>\n"
                + "</function>\n</tool_call> " * count
            )
            filt = ToolCallStreamFilter(self._tokenizer())
            start = time.perf_counter()
            for i in range(0, len(evil), 64):
                filt.feed(evil[i : i + 64])
            filt.finish()
            return time.perf_counter() - start

        elapsed(400)  # warm up before timing
        small = max(elapsed(400), 1e-4)
        large = elapsed(3200)
        # 8x the input: linear is ~8x, quadratic ~64x. Generous headroom for a
        # loaded CI box while still catching quadratic growth.
        assert large / small < 24, f"superlinear growth: {small=} {large=}"


class TestDeepNestingNeverEscapesParseChain:
    """Deep nesting must degrade to a parse failure, not raise (#2545).

    The interpreter decides *which* error deep input produces, and it varies
    by version: on 3.14 ``json.loads`` already returns a JSONDecodeError that
    the old excepts caught, while ``ast`` raises SyntaxError. On earlier
    versions RecursionError is the mode. Testing with genuinely deep input
    therefore proves nothing portable, so these tests inject the error at the
    decoder instead and assert the chain still degrades cleanly on every
    version.
    """

    ERRORS = [RecursionError, SyntaxError]

    @staticmethod
    def _tokenizer(parser):
        tok = MagicMock(spec=[])
        tok.has_tool_calling = True
        tok.tool_call_start = "<tool_call>"
        tok.tool_call_end = "</tool_call>"
        tok.tool_parser = parser
        return tok

    @pytest.mark.parametrize("error", ERRORS)
    def test_native_parser_raising_does_not_escape(self, error):
        """A real parser hitting the limit internally must not escape.

        This is the path #2545 is really about: glm47, kimi_k2 and
        qwen3_coder call ``json.loads`` inside the parser, so the error
        surfaces at the parser call rather than at a decode site in this
        module. Recovering via the XML fallback is a fine outcome; raising
        is not.
        """
        def boom(text, tools):
            raise error("nested too deeply")

        # Recoverable payload: the fallback picks it up, nothing escapes.
        _, calls = parse_tool_calls(
            '<tool_call>{"name": "f"}</tool_call>', self._tokenizer(boom)
        )
        assert calls is not None and calls[0].function.name == "f"

        # Unrecoverable payload: drops to no tool calls, still no raise.
        _, calls = parse_tool_calls(
            "<tool_call>" + "[" * 500 + "</tool_call>", self._tokenizer(boom)
        )
        assert calls is None

    @pytest.mark.parametrize("error", ERRORS)
    def test_json_loads_raising_does_not_escape(self, monkeypatch, error):
        """Every json.loads site in the chain sits behind a guard."""
        import omlx.api.tool_calling as mod

        real = json.loads

        def boom(s, *a, **k):
            raise error("nested too deeply")

        monkeypatch.setattr(mod.json, "loads", boom)
        try:
            # Each of these reaches a different decode site. The third runs a
            # real parser so a decoder is actually exercised: an earlier
            # version of this test passed plain text with tool_parser=None,
            # which reached no decoder at all (caught by DiscoStew6082).
            mod._parse_xml_tool_calls("<tool_call>{}</tool_call>")
            mod.extract_json_from_text('{"a": 1}')
            parse_tool_calls(
                '<tool_call>{"name": "f", "arguments": {}}</tool_call>',
                self._tokenizer(lambda text, tools: real(text)),
            )
        finally:
            monkeypatch.setattr(mod.json, "loads", real)

    @pytest.mark.parametrize("error", ERRORS)
    def test_gemma4_args_reject_hard_instead_of_retrying_legacy(
        self, monkeypatch, error
    ):
        """Deep nesting must not fall through to the unbounded legacy parser.

        The legacy path ignores the length/depth bounds on purpose, so routing
        a payload that broke a decoder into it would hand the exact input the
        bounds exist to stop to the parser that does not apply them.
        """
        import omlx.api.tool_calling as mod

        def boom(_args):
            raise error("nested too deeply")

        called = []
        monkeypatch.setattr(mod, "_gemma4_transcode_to_json", boom)
        monkeypatch.setattr(
            mod,
            "_gemma4_args_to_json_legacy",
            lambda a: called.append(a) or {},
        )

        with pytest.raises(mod._Gemma4ArgsTooComplexError):
            mod._gemma4_args_to_json_robust('{"a": 1}')
        assert called == [], "legacy parser must not see deep-nested args"

    @pytest.mark.parametrize("error", ERRORS)
    def test_gemma4_legacy_raising_is_converted(self, monkeypatch, error):
        """The legacy parser is unbounded, so guard its decoders too."""
        import omlx.api.tool_calling as mod

        def transcode_fails(_args):
            raise ValueError("ambiguous, retry with legacy")

        def legacy_boom(_args):
            raise error("nested too deeply")

        monkeypatch.setattr(mod, "_gemma4_transcode_to_json", transcode_fails)
        monkeypatch.setattr(mod, "_gemma4_args_to_json_legacy", legacy_boom)

        with pytest.raises(mod._Gemma4ArgsTooComplexError):
            mod._gemma4_args_to_json_robust('{"a": 1}')

    def test_reported_repro_does_not_raise(self):
        """The issue's original repro, kept as a smoke test.

        It no longer raises on 3.14 because json.loads returns a decode error
        there, so it is a regression guard rather than the proof.
        """
        tok = self._tokenizer(lambda text, tools: json.loads(text))
        cleaned, calls = parse_tool_calls(
            "<tool_call>" + "[" * 100000 + "</tool_call>", tok
        )
        assert calls is None


@pytest.mark.parametrize("error", [RecursionError, SyntaxError])
def test_deep_nesting_does_not_take_down_a_neighboring_tool_call(error):
    """One unparseable call must not lose the valid call beside it (#2545).

    The parse loop runs per match, so a payload that breaks the decoder has
    to fail that match alone and leave the rest of the batch intact.
    """
    def parser(text, tools):
        if "[" in text:
            raise error("nested too deeply")
        return json.loads(text)

    tok = MagicMock(spec=[])
    tok.has_tool_calling = True
    tok.tool_call_start = "<tool_call>"
    tok.tool_call_end = "</tool_call>"
    tok.tool_parser = parser

    cleaned, calls = parse_tool_calls(
        "<tool_call>" + "[" * 500 + "</tool_call>"
        '<tool_call>{"name": "good", "arguments": {"a": 1}}</tool_call>',
        tok,
    )

    assert [c.function.name for c in calls] == ["good"]
    # The broken envelope's markup must not leak into content either.
    assert cleaned == ""


@pytest.mark.parametrize("error", [RecursionError, SyntaxError])
@pytest.mark.parametrize(
    "text",
    [
        "<tool_call><function=f><parameter=x>{}</parameter></function></tool_call>",
        "<tool_call>\n<function=f>\n<parameter=x>\n{}\n</parameter>\n"
        "</function>\n</tool_call>",
        "<tool_call>f\n<arg_key>x</arg_key>\n<arg_value>{}</arg_value>\n</tool_call>",
        # Real namespaced grammar. An earlier version of this fixture used
        # <function=..><parameter=..>, which _parse_namespaced_tool_calls does
        # not accept, so it parsed zero calls and exercised nothing
        # (caught by DiscoStew6082).
        '<ns:tool_call><invoke name="f"><parameter name="x">{}</parameter>'
        "</invoke></ns:tool_call>",
    ],
    ids=["xml-fallback", "qwen-xml", "glm-xml", "namespaced-xml"],
)
def test_serialization_after_a_successful_decode_does_not_escape(
    monkeypatch, error, text
):
    """The re-serialize step is a second decode from a deeper frame (#2545).

    ``json.dumps`` recurses per nesting level just as the decoders do, and it
    runs *after* a value has already parsed, further down the stack. A value
    nested just under the limit when it decoded can therefore breach it here.
    DiscoStew6082 hit this at depth ~987 on 3.11, past the first guard.

    Injecting at ``json.dumps`` rather than nesting for real keeps this
    meaningful on 3.14, where the interpreter does not raise at these depths.
    """
    import omlx.api.tool_calling as mod

    nested = "[" * 400 + "0" + "]" * 400
    real_dumps = json.dumps

    def boom(*args, **kwargs):
        raise error("nested too deeply")

    payload = text.replace("{}", nested)
    monkeypatch.setattr(mod.json, "dumps", boom)
    try:
        # Must not raise. Dropping the call is fine; escaping is not.
        if "ns:tool_call" in payload:
            mod._parse_namespaced_tool_calls(payload, "ns")
        else:
            mod._parse_xml_tool_calls(payload)
    finally:
        monkeypatch.setattr(mod.json, "dumps", real_dumps)


@pytest.mark.parametrize("error", [RecursionError, SyntaxError])
def test_serialize_arguments_raises_so_the_caller_can_drop(monkeypatch, error):
    """``_serialize_tool_call_arguments`` propagates rather than emptying.

    An earlier version of this test asserted it returned "{}" here. That was
    wrong: it produced a runnable tool call with its arguments silently
    removed. The failure has to reach ``_build_tool_call`` so the call is
    dropped (jundot's review on #2593).
    """
    import omlx.api.tool_calling as mod

    real_dumps = json.dumps

    def boom(*args, **kwargs):
        raise error("nested too deeply")

    monkeypatch.setattr(mod.json, "dumps", boom)
    try:
        with pytest.raises(error):
            mod._serialize_tool_call_arguments({"x": 1})
        with pytest.raises(error):
            mod._serialize_tool_call_arguments('{"x": 1}')
    finally:
        monkeypatch.setattr(mod.json, "dumps", real_dumps)


@pytest.mark.parametrize(
    "template",
    [
        "<tool_call><function=f><parameter=x>{}</parameter></function></tool_call>",
        "<tool_call>\n<function=f>\n<parameter=x>\n{}\n</parameter>\n"
        "</function>\n</tool_call>",
        '<ns:tool_call><invoke name="f"><parameter name="x">{}</parameter>'
        "</invoke></ns:tool_call>",
    ],
    ids=["compact-xml", "qwen-xml", "namespaced-xml"],
)
def test_balanced_depth_sweep_through_public_entry(template):
    """Real nesting swept across the recursion boundary (#2545).

    Unlike the injected tests, this uses genuinely deep input and so only
    bites on Python 3.11 through 3.13, where ``json.loads`` still raises
    RecursionError. CI covers exactly those versions. It sweeps rather than
    picking a depth because the boundary moves with how much stack the caller
    has already used, which is why a single hand-picked depth reproduced for
    DiscoStew6082 and not for me.

    Any outcome except an escaping exception is acceptable: parsing the call,
    or dropping it with a warning.
    """
    tok = MagicMock(spec=[])
    tok.has_tool_calling = True
    tok.tool_call_start = "<tool_call>"
    tok.tool_call_end = "</tool_call>"
    tok.tool_parser = None

    for depth in range(940, 1041):
        nested = "[" * depth + "0" + "]" * depth
        try:
            parse_tool_calls(template.replace("{}", nested), tok)
        except (RecursionError, SyntaxError) as exc:
            pytest.fail(f"escaped at depth {depth}: {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("error", [RecursionError, SyntaxError, ValueError])
@pytest.mark.parametrize(
    "template",
    [
        "<tool_call><function=f><parameter=x>{}</parameter></function></tool_call>",
        "<tool_call>\n<function=f>\n<parameter=x>\n{}\n</parameter>\n"
        "</function>\n</tool_call>",
        "<tool_call>f\n<arg_key>x</arg_key>\n<arg_value>{}</arg_value>\n</tool_call>",
        '<ns:tool_call><invoke name="f"><parameter name="x">{}</parameter>'
        "</invoke></ns:tool_call>",
        '<|tool_call_start|>{"name": "f", "arguments": {"x": 1}}<|tool_call_end|>',
    ],
    ids=["compact-xml", "qwen-xml", "glm-xml", "namespaced-xml", "hermes"],
)
def test_functioncall_validation_failure_drops_one_call(
    monkeypatch, error, template
):
    """The third decode lives in openai_models, not this module (#2545).

    ``FunctionCall`` re-parses the arguments string while validating it, from
    a deeper frame than either the parse or the serialize that preceded it, so
    a value fine at both can still breach the limit there. DiscoStew6082 hit
    this on 3.11 at depth ~989 after the serialize guard was already in place.

    Injected here because the real-depth version only bites on 3.11 to 3.13.
    ValueError is included because that is what the validator raises for
    ordinary malformed arguments, and it escaped the parse chain too.
    """
    import omlx.api.openai_models as om

    def boom(_v):
        raise error("nested too deeply")

    monkeypatch.setattr(om, "_coerce_tool_call_arguments", boom)

    tok = MagicMock(spec=[])
    tok.has_tool_calling = True
    tok.tool_call_start = "<tool_call>"
    tok.tool_call_end = "</tool_call>"
    tok.tool_parser = None

    # Must not raise. Dropping the call with a warning is the contract.
    _, calls = parse_tool_calls(template.replace("{}", "1"), tok)
    assert not calls


@pytest.mark.parametrize("chain", [200, 400, 800])
def test_hermes_chained_expression_does_not_escape(chain):
    """`ast.unparse` recurses too, and runs after `ast.parse` succeeded (#2545).

    A long chained expression parses fine, fails `ast.literal_eval` because it
    is not a literal, then breaches the limit in the `ast.unparse` fallback
    that renders it back to source. jundot found this on the Hermes path after
    the decoder, serializer and validator layers were all guarded.

    Real nesting rather than injection, so it only bites on 3.11 to 3.13,
    which is what CI runs.
    """
    expr = "+".join(["1"] * chain)
    text = f"<|tool_call_start|>f(x={expr})<|tool_call_end|>"

    tok = MagicMock(spec=[])
    tok.has_tool_calling = True
    tok.tool_call_start = "<tool_call>"
    tok.tool_call_end = "</tool_call>"
    tok.tool_parser = None

    # Must not raise through the public entry point.
    parse_tool_calls(text, tok)


@pytest.mark.parametrize("error", [RecursionError, SyntaxError])
def test_unrepresentable_argument_drops_the_call(monkeypatch, error):
    """An argument we cannot render drops the call, not just the argument."""
    import omlx.api.tool_calling as mod

    real_unparse = ast.unparse

    def boom(node):
        raise error("nested too deeply")

    monkeypatch.setattr(mod.ast, "unparse", boom)
    # `1+1` is not a literal, so literal_eval fails and unparse is the fallback.
    cleaned, calls = mod._parse_hermes_tool_calls(
        "<|tool_call_start|>f(x=1+1)<|tool_call_end|>"
    )
    monkeypatch.setattr(mod.ast, "unparse", real_unparse)

    assert not calls, "a call missing an argument must not be emitted"


@pytest.mark.parametrize("error", [RecursionError, SyntaxError])
def test_serialization_failure_drops_the_call_rather_than_emptying_it(
    monkeypatch, error
):
    """A serialize failure must not yield a runnable call with no arguments.

    Coercing to "{}" here would hand back a tool call that still executes with
    its arguments silently removed, so a `write_file` would fire with nothing
    to write. That is worse than dropping it. Distinct from the non-object
    coercion below, which is a benign parser quirk (jundot's review on #2593).
    """
    import omlx.api.tool_calling as mod

    real_dumps = json.dumps

    def boom(*args, **kwargs):
        raise error("nested too deeply")

    monkeypatch.setattr(mod.json, "dumps", boom)
    try:
        built = mod._build_tool_call("write_file", {"path": "a", "content": "b"})
    finally:
        monkeypatch.setattr(mod.json, "dumps", real_dumps)

    assert built is None, "must drop, not emit a call with emptied arguments"


def test_non_object_arguments_still_coerce_to_empty_object():
    """The benign coercion is unchanged: a parser quirk, not lost data."""
    import omlx.api.tool_calling as mod

    assert mod._serialize_tool_call_arguments([1, 2]) == "{}"
    assert mod._serialize_tool_call_arguments("not json") == "{}"
    assert mod._serialize_tool_call_arguments({"a": 1}) == '{"a": 1}'


@pytest.mark.parametrize("error", [RecursionError, SyntaxError])
def test_string_parser_output_that_cannot_decode_drops_the_call(
    monkeypatch, error
):
    """The string branch loses arguments the same way the dict branch did.

    Parsers that follow the OpenAI spec hand back a JSON-object *string*
    (mlx-vlm and mlx-lm's gemma4 do). If decoding that string breaches the
    limit, catching the error here would fall through to the "{}" coercion and
    emit a runnable call without its arguments. Caught by DiscoStew6082 on
    #2593 after the dict branch was already fixed.

    The decode is made to fail only for this exact payload, so the downstream
    validator still works and the drop is attributable to this branch.
    """
    import omlx.api.tool_calling as mod

    payload = '{"path": "notes.xml", "content": "IMPORTANT"}'
    real_loads = json.loads

    def selective(s, *args, **kwargs):
        if s == payload:
            raise error("nested too deeply")
        return real_loads(s, *args, **kwargs)

    monkeypatch.setattr(mod.json, "loads", selective)
    try:
        built = mod._build_tool_call("write_file", payload)
    finally:
        monkeypatch.setattr(mod.json, "loads", real_loads)

    assert built is None, "must drop, not emit a call with emptied arguments"


def test_malformed_string_arguments_still_coerce():
    """Undecodable-but-shallow input stays a benign coercion, not a drop."""
    import omlx.api.tool_calling as mod

    assert mod._serialize_tool_call_arguments("not json at all") == "{}"
    assert mod._serialize_tool_call_arguments('{"a": 1}') == '{"a": 1}'


@pytest.mark.parametrize(
    "literal",
    ["{1, 2}", "b'abc'", "1+2j"],
    ids=["set", "bytes", "complex"],
)
def test_hermes_non_json_literal_drops_only_bad_call(literal):
    """Python literals that JSON cannot represent must not escape the parser."""
    tok = MagicMock(spec=[])
    tok.has_tool_calling = False
    text = "<|tool_call_start|>[bad(x=" + literal + "), good(x=1)]<|tool_call_end|>"

    cleaned, calls = parse_tool_calls(text, tok)

    assert cleaned == ""
    assert [call.function.name for call in calls] == ["good"]


def test_bracket_deep_decode_never_runs_raw_arguments():
    """A recursion-bound breach must drop the call, not run its raw payload."""
    tok = MagicMock(spec=[])
    tok.has_tool_calling = False

    for depth in range(900, 1051):
        nested = "[" * depth + "0" + "]" * depth
        text = '[Tool call: bad({"x":' + nested + '})][Tool call: good({"x":1})]'

        cleaned, calls = parse_tool_calls(text, tok)
        names = [call.function.name for call in calls or []]

        assert cleaned == ""
        assert names.count("good") == 1
        for call in calls or []:
            if call.function.name == "bad":
                assert not call.function.arguments.startswith('{"raw":')
