# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tool_choice enforcement."""
from omlx.api.openai_models import FunctionCall, ToolCall
from omlx.api.tool_choice import enforce_tool_choice


def _tc(name="search", args='{"q":"x"}', cid="c1"):
    return ToolCall(id=cid, type="function", function=FunctionCall(name=name, arguments=args))


class TestNoneMode:
    def test_suppresses_tool_calls(self):
        tcs = [_tc("search")]
        result, check = enforce_tool_choice(tcs, "none", has_text=True, tools=[])
        assert result is None or len(result) == 0
        assert check is not None
        assert check.check == "tool_choice_enforcement"
        assert check.passed is True  # "none" mode was satisfied

    def test_none_with_no_tool_calls(self):
        result, check = enforce_tool_choice(None, "none", has_text=True, tools=[])
        assert check is not None
        assert check.passed is True


class TestAutoMode:
    def test_passes_through_tool_calls(self):
        tcs = [_tc("search")]
        result, check = enforce_tool_choice(tcs, "auto", has_text=False, tools=[])
        assert result == tcs
        assert check is not None
        assert check.passed is True

    def test_auto_omitted_passes_through(self):
        tcs = [_tc("search")]
        result, check = enforce_tool_choice(tcs, None, has_text=False, tools=[])
        assert result == tcs
        assert check is not None
        assert check.passed is True


class TestRequiredMode:
    def test_bare_text_fails(self):
        result, check = enforce_tool_choice(None, "required", has_text=True, tools=[])
        assert check is not None
        assert not check.passed

    def test_tool_calls_present_passes(self):
        tcs = [_tc("search")]
        result, check = enforce_tool_choice(tcs, "required", has_text=False, tools=[])
        assert check.passed is True


class TestNamedToolMode:
    def test_filters_to_named_tool(self):
        tools = [{"type": "function", "function": {"name": "search"}}]
        tcs = [_tc("search", cid="c1"), _tc("read", cid="c2")]
        choice = {"type": "function", "function": {"name": "search"}}
        result, check = enforce_tool_choice(tcs, choice, has_text=False, tools=tools)
        assert result is not None
        assert len(result) == 1
        assert result[0].function.name == "search"

    def test_all_wrong_tool_flags_failure(self):
        tools = [{"type": "function", "function": {"name": "search"}}]
        tcs = [_tc("read")]
        choice = {"type": "function", "function": {"name": "search"}}
        result, check = enforce_tool_choice(tcs, choice, has_text=False, tools=tools)
        assert check is not None
        assert not check.passed


class TestInvalidToolChoice:
    def test_random_string_returns_pass(self):
        """Invalid values are rejected at request time (HTTP 400), not here.
        At enforcement time, unknown values are treated as pass-through."""
        result, check = enforce_tool_choice(None, "weird", has_text=True, tools=[])
        assert check is not None
        assert check.passed is True
