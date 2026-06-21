# SPDX-License-Identifier: Apache-2.0
"""Unit tests for rescue parsers: rehearsal syntax + improved Mistral bracket-tag."""
from omlx.api.tool_calling import _parse_rehearsal_tool_calls, _parse_mistral_bracket_tool_calls


class TestRehearsalParser:
    def test_single_rehearsal_call(self):
        text = 'search[ARGS]{"query": "hello world"}'
        result = _parse_rehearsal_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert result[0].function.name == "search"
        assert '"query"' in result[0].function.arguments

    def test_multiple_rehearsal_calls(self):
        text = (
            'search[ARGS]{"query": "first"}\n'
            'read[ARGS]{"path": "/tmp/file"}'
        )
        result = _parse_rehearsal_tool_calls(text)
        assert result is not None
        assert len(result) == 2
        assert result[0].function.name == "search"
        assert result[1].function.name == "read"

    def test_invalid_json_skipped(self):
        text = 'search[ARGS]{not valid json}'
        result = _parse_rehearsal_tool_calls(text)
        # Invalid JSON in one expression — should skip, not crash.
        # May return empty list or None.
        assert result is None or len(result) == 0

    def test_no_match_returns_none(self):
        text = "just regular text, no tool calls here"
        result = _parse_rehearsal_tool_calls(text)
        assert result is None

    def test_multiline_json_body(self):
        text = 'search[ARGS]{\n  "query": "multi\nline"\n}'
        result = _parse_rehearsal_tool_calls(text)
        assert result is not None
        assert len(result) == 1


class TestMistralBracketParser:
    def test_simple_mistral_format(self):
        text = '[TOOL_CALLS]search{"query": "hello"}'
        result = _parse_mistral_bracket_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert result[0].function.name == "search"

    def test_nested_json_objects(self):
        text = '[TOOL_CALLS]search{"config": {"depth": 3, "opts": {"a": 1}}}'
        result = _parse_mistral_bracket_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert "config" in result[0].function.arguments

    def test_literal_braces_in_strings(self):
        text = '[TOOL_CALLS]format{"pattern": "use {placeholder} here"}'
        result = _parse_mistral_bracket_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert "placeholder" in result[0].function.arguments

    def test_escaped_quotes_in_strings(self):
        text = r'[TOOL_CALLS]echo{"text": "say \"hello\""}'
        result = _parse_mistral_bracket_tool_calls(text)
        assert result is not None
        assert len(result) == 1

    def test_no_marker_returns_none(self):
        text = "regular output without mistral marker"
        result = _parse_mistral_bracket_tool_calls(text)
        assert result is None

    def test_multiple_tool_calls(self):
        text = (
            '[TOOL_CALLS]search{"query": "a"}\n'
            'read{"path": "/tmp"}'
        )
        result = _parse_mistral_bracket_tool_calls(text)
        assert result is not None
        assert len(result) >= 2
