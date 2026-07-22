# SPDX-License-Identifier: Apache-2.0
"""Unit tests for synthetic respond tool inject + strip."""
from __future__ import annotations

import pytest

from omlx.api.openai_models import FunctionCall, ToolCall
from omlx.api.respond import (
    RESPOND_TOOL_NAME,
    inject_respond_tool,
    strip_respond_calls,
)


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tc(name: str, args: str = "{}") -> ToolCall:
    return ToolCall(
        id=f"call_{name}",
        type="function",
        function=FunctionCall(name=name, arguments=args),
    )


class TestRespondToolName:
    def test_name_is_respond(self):
        assert RESPOND_TOOL_NAME == "respond"


class TestInjectRespondTool:
    def test_appends_to_non_empty_list(self):
        tools = [_tool("search"), _tool("read")]
        result = inject_respond_tool(tools)
        assert len(result) == 3
        assert result[-1]["function"]["name"] == RESPOND_TOOL_NAME

    def test_returns_none_when_none(self):
        assert inject_respond_tool(None) is None

    def test_returns_empty_when_empty(self):
        result = inject_respond_tool([])
        assert result == []

    def test_does_not_duplicate(self):
        from omlx.api.respond import RESPOND_TOOL_SPEC

        tools = [_tool("search"), RESPOND_TOOL_SPEC]
        result = inject_respond_tool(tools)
        assert len(result) == 2  # unchanged
        names = [t["function"]["name"] for t in result]
        assert names.count(RESPOND_TOOL_NAME) == 1

    def test_injected_tool_has_message_param(self):
        result = inject_respond_tool([_tool("search")])
        params = result[-1]["function"]["parameters"]
        assert "message" in params["properties"]
        assert "message" in params["required"]

    def test_injected_tool_has_description(self):
        result = inject_respond_tool([_tool("search")])
        desc = result[-1]["function"]["description"]
        assert len(desc) > 20  # non-trivial description

    def test_does_not_mutate_original_list(self):
        tools = [_tool("search")]
        inject_respond_tool(tools)
        assert len(tools) == 1  # original unchanged


class TestStripRespondCalls:
    def test_pure_respond_becomes_text(self):
        calls = [_tc("respond", '{"message": "Hello world"}')]
        real, text = strip_respond_calls(calls)
        assert real == []
        assert text == "Hello world"

    def test_mixed_drops_respond_silently(self):
        calls = [
            _tc("respond", '{"message": "thinking..."}'),
            _tc("search", '{"query": "test"}'),
        ]
        real, text = strip_respond_calls(calls)
        assert len(real) == 1
        assert real[0].function.name == "search"
        assert text is None  # respond silently dropped

    def test_no_respond_passthrough(self):
        calls = [_tc("search"), _tc("read")]
        real, text = strip_respond_calls(calls)
        assert len(real) == 2
        assert text is None

    def test_empty_list(self):
        real, text = strip_respond_calls([])
        assert real == []
        assert text is None

    def test_none_input(self):
        real, text = strip_respond_calls(None)
        assert real == []
        assert text is None

    def test_pure_respond_extracts_message_from_dict_args(self):
        """When args are already a dict (some parsers do this)."""
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(
                name="respond",
                arguments={"message": "Dict args"},
            ),
        )
        real, text = strip_respond_calls([tc])
        assert real == []
        assert text == "Dict args"

    def test_malformed_respond_args_returns_empty_message(self):
        # Pydantic FunctionCall rejects non-JSON, so simulate a raw dict
        # tool call with malformed string arguments directly.
        tc = {
            "id": "call_x",
            "type": "function",
            "function": {"name": "respond", "arguments": "not-json"},
        }
        real, text = strip_respond_calls([tc])
        assert real == []
        assert text == ""  # graceful degradation

    def test_multiple_real_calls_preserved(self):
        calls = [
            _tc("search", '{"q": "a"}'),
            _tc("respond", '{"message": "x"}'),
            _tc("read", '{"path": "/y"}'),
        ]
        real, text = strip_respond_calls(calls)
        assert len(real) == 2
        assert real[0].function.name == "search"
        assert real[1].function.name == "read"
