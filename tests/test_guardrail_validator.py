# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GuardrailValidator — 4 checks, ordering, edge cases."""
import json

from omlx.api.openai_models import FunctionCall, ToolCall
from omlx.api.tool_calling import ToolCallExtraction
from omlx.api.guardrails.validator import GuardrailValidator


def _make_extraction(text="", tool_calls=None):
    """Helper: build a ToolCallExtraction for testing."""
    return ToolCallExtraction(
        cleaned_text=text,
        tool_calls=tool_calls,
        cleaned_thinking="",
    )


def _make_tool_call(name="search", arguments='{"query": "test"}', call_id="call_1"):
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name=name, arguments=arguments),
    )


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
}

NO_PARAMS_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "parameters": {},
    },
}


class TestBareTextCheck:
    def test_text_when_tools_expected_fails(self):
        v = GuardrailValidator([SEARCH_TOOL])
        ext = _make_extraction(text="I cannot help with that.", tool_calls=None)
        result = v.validate(ext, has_tools=True)
        assert not result.passed
        bare_checks = [c for c in result.checks if c.check == "bare_text"]
        assert len(bare_checks) == 1
        assert not bare_checks[0].passed

    def test_text_when_tool_choice_none_passes(self):
        v = GuardrailValidator([SEARCH_TOOL])
        ext = _make_extraction(text="just text", tool_calls=None)
        result = v.validate(ext, tool_choice="none", has_tools=True)
        bare_checks = [c for c in result.checks if c.check == "bare_text"]
        assert bare_checks[0].passed

    def test_tool_calls_present_passes(self):
        v = GuardrailValidator([SEARCH_TOOL])
        tc = _make_tool_call("search", '{"query": "x"}')
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        bare_checks = [c for c in result.checks if c.check == "bare_text"]
        assert bare_checks[0].passed

    def test_no_tools_provided_passes(self):
        v = GuardrailValidator(None)
        ext = _make_extraction(text="text", tool_calls=None)
        result = v.validate(ext, has_tools=False)
        bare_checks = [c for c in result.checks if c.check == "bare_text"]
        assert bare_checks[0].passed


class TestUnknownToolCheck:
    def test_known_tool_passes(self):
        v = GuardrailValidator([SEARCH_TOOL])
        tc = _make_tool_call("search", '{"query": "x"}')
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        unknown_checks = [c for c in result.checks if c.check == "unknown_tool"]
        assert all(c.passed for c in unknown_checks)

    def test_unknown_tool_fails(self):
        v = GuardrailValidator([SEARCH_TOOL])
        tc = _make_tool_call("nonexistent", "{}")
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        unknown_checks = [c for c in result.checks if c.check == "unknown_tool"]
        assert any(not c.passed for c in unknown_checks)
        assert result.nudge is not None
        assert result.nudge.kind == "unknown_tool"

    def test_mixed_known_unknown_reports_unknown(self):
        v = GuardrailValidator([SEARCH_TOOL])
        tc1 = _make_tool_call("search", '{"query": "x"}', "c1")
        tc2 = _make_tool_call("bad", "{}", "c2")
        ext = _make_extraction(tool_calls=[tc1, tc2])
        result = v.validate(ext, has_tools=True)
        unknown_checks = [c for c in result.checks if c.check == "unknown_tool"]
        assert not all(c.passed for c in unknown_checks)


class TestMalformedArgsCheck:
    def test_dict_args_passes(self):
        v = GuardrailValidator([SEARCH_TOOL])
        tc = _make_tool_call("search", '{"query": "x"}')
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        malformed_checks = [c for c in result.checks if c.check == "malformed_args"]
        assert all(c.passed for c in malformed_checks)

    def test_string_args_fails(self):
        v = GuardrailValidator([SEARCH_TOOL])
        # FunctionCall rejects non-dict args at construction; use model_construct
        # to simulate malformed args arriving from rescue parsing.
        fc = FunctionCall.model_construct(name="search", arguments='"just a string"')
        tc = ToolCall(id="c1", type="function", function=fc)
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        malformed_checks = [c for c in result.checks if c.check == "malformed_args"]
        assert any(not c.passed for c in malformed_checks)

    def test_array_args_fails(self):
        v = GuardrailValidator([SEARCH_TOOL])
        fc = FunctionCall.model_construct(name="search", arguments="[1, 2, 3]")
        tc = ToolCall(id="c1", type="function", function=fc)
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        malformed_checks = [c for c in result.checks if c.check == "malformed_args"]
        assert any(not c.passed for c in malformed_checks)


class TestMissingRequiredParams:
    def test_all_required_present_passes(self):
        v = GuardrailValidator([SEARCH_TOOL])
        tc = _make_tool_call("search", '{"query": "hello"}')
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        missing_checks = [c for c in result.checks if c.check == "missing_required_params"]
        assert all(c.passed for c in missing_checks)

    def test_missing_required_fails(self):
        v = GuardrailValidator([SEARCH_TOOL])
        tc = _make_tool_call("search", '{"limit": 5}')
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        missing_checks = [c for c in result.checks if c.check == "missing_required_params"]
        assert any(not c.passed for c in missing_checks)

    def test_no_required_field_passes(self):
        v = GuardrailValidator([NO_PARAMS_TOOL])
        tc = _make_tool_call("ping", "{}")
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        missing_checks = [c for c in result.checks if c.check == "missing_required_params"]
        assert all(c.passed for c in missing_checks)

    def test_empty_parameters_schema_passes(self):
        tool = {"type": "function", "function": {"name": "ping", "parameters": {}}}
        v = GuardrailValidator([tool])
        tc = _make_tool_call("ping", "{}")
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        missing_checks = [c for c in result.checks if c.check == "missing_required_params"]
        assert all(c.passed for c in missing_checks)

    def test_missing_params_in_schema_not_list(self):
        """If 'required' is not a list, check passes defensively."""
        tool = {
            "type": "function",
            "function": {"name": "weird", "parameters": {"required": "not_a_list"}},
        }
        v = GuardrailValidator([tool])
        tc = _make_tool_call("weird", "{}")
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        missing_checks = [c for c in result.checks if c.check == "missing_required_params"]
        assert all(c.passed for c in missing_checks)


class TestCheckOrdering:
    def test_bare_text_takes_priority_over_unknown_tool(self):
        """When bare_text fails, it should be the selected nudge."""
        v = GuardrailValidator([SEARCH_TOOL])
        ext = _make_extraction(text="text", tool_calls=None)
        result = v.validate(ext, has_tools=True)
        assert not result.passed
        assert result.nudge is not None
        assert result.nudge.kind == "retry"

    def test_unknown_tool_before_malformed_args(self):
        """Unknown tool failure produces unknown_tool nudge even if args are also bad."""
        v = GuardrailValidator([SEARCH_TOOL])
        fc = FunctionCall.model_construct(name="bad_tool", arguments='"bad_args"')
        tc = ToolCall(id="c1", type="function", function=fc)
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        assert result.nudge is not None
        assert result.nudge.kind == "unknown_tool"

    def test_valid_response_passes_all(self):
        v = GuardrailValidator([SEARCH_TOOL])
        tc = _make_tool_call("search", '{"query": "hello world"}')
        ext = _make_extraction(tool_calls=[tc])
        result = v.validate(ext, has_tools=True)
        assert result.passed
        assert result.nudge is None


class TestValidatorConstruction:
    def test_no_tools(self):
        v = GuardrailValidator(None)
        assert len(v._tool_names) == 0

    def test_empty_tools(self):
        v = GuardrailValidator([])
        assert len(v._tool_names) == 0

    def test_tools_without_function_key(self):
        """Tools might be bare function dicts (no wrapper)."""
        tool = {"name": "direct", "parameters": {}}
        v = GuardrailValidator([tool])
        assert "direct" in v._tool_names
