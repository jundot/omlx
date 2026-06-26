# SPDX-License-Identifier: Apache-2.0
"""
Tests for tool calling parsing and conversion utilities.

Tests JSON schema validation, JSON extraction, and tool conversion functions.
"""

import json
import logging
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
    _gemma4_args_to_json_robust,
    _parse_gemma4_tool_call_fallback,
    _remap_tool_call_names,
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
