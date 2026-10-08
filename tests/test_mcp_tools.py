# SPDX-License-Identifier: Apache-2.0
"""
Tests for MCP tool utilities (omlx/mcp/tools.py).
"""

import json

import pytest

from omlx.mcp.tools import (
    extract_tool_calls,
    format_tool_result,
    format_tool_results,
    has_tool_calls,
    mcp_tool_to_openai,
    mcp_tools_to_openai,
    merge_tools,
    openai_call_to_mcp,
)
from omlx.mcp.types import MCPTool, MCPToolResult


class TestMCPToolToOpenAI:
    """Tests for mcp_tool_to_openai conversion."""

    def test_basic_conversion(self):
        """Test basic tool conversion."""
        tool = MCPTool(
            server_name="weather",
            name="get_forecast",
            description="Get weather forecast",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        )

        result = mcp_tool_to_openai(tool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "weather__get_forecast"
        assert result["function"]["description"] == "Get weather forecast"
        assert result["function"]["parameters"]["type"] == "object"
        assert "location" in result["function"]["parameters"]["properties"]

    def test_empty_schema(self):
        """Test conversion with empty schema."""
        tool = MCPTool(
            server_name="server",
            name="no_params",
            description="Tool without parameters",
        )

        result = mcp_tool_to_openai(tool)

        # Should have default empty schema
        assert result["function"]["parameters"]["type"] == "object"
        assert result["function"]["parameters"]["properties"] == {}

    def test_none_schema(self):
        """Test conversion with None schema."""
        tool = MCPTool(
            server_name="server",
            name="none_schema",
            description="Tool with None schema",
            input_schema=None,  # type: ignore
        )

        result = mcp_tool_to_openai(tool)

        # Should fall back to empty schema
        assert "parameters" in result["function"]


class TestMCPToolsToOpenAI:
    """Tests for mcp_tools_to_openai conversion."""

    def test_empty_list(self):
        """Test conversion of empty list."""
        result = mcp_tools_to_openai([])
        assert result == []

    def test_multiple_tools(self):
        """Test conversion of multiple tools."""
        tools = [
            MCPTool(server_name="s1", name="t1", description="Tool 1"),
            MCPTool(server_name="s2", name="t2", description="Tool 2"),
            MCPTool(server_name="s1", name="t3", description="Tool 3"),
        ]

        result = mcp_tools_to_openai(tools)

        assert len(result) == 3
        names = [t["function"]["name"] for t in result]
        assert "s1__t1" in names
        assert "s2__t2" in names
        assert "s1__t3" in names


class TestOpenAICallToMCP:
    """Tests for openai_call_to_mcp parsing."""

    def test_parse_with_server_prefix(self):
        """Test parsing tool call with server prefix."""
        tool_call = {
            "id": "call_123",
            "function": {
                "name": "weather__get_forecast",
                "arguments": '{"location": "Tokyo"}',
            },
        }

        server_name, tool_name, arguments = openai_call_to_mcp(tool_call)

        assert server_name == "weather"
        assert tool_name == "get_forecast"
        assert arguments == {"location": "Tokyo"}

    def test_parse_without_server_prefix(self):
        """Test parsing tool call without server prefix."""
        tool_call = {
            "function": {
                "name": "simple_tool",
                "arguments": '{"key": "value"}',
            },
        }

        server_name, tool_name, arguments = openai_call_to_mcp(tool_call)

        assert server_name == ""
        assert tool_name == "simple_tool"
        assert arguments == {"key": "value"}

    def test_parse_empty_arguments(self):
        """Test parsing with empty arguments."""
        tool_call = {
            "function": {
                "name": "no_args",
                "arguments": "{}",
            },
        }

        _, _, arguments = openai_call_to_mcp(tool_call)

        assert arguments == {}

    def test_parse_invalid_json_arguments(self):
        """Invalid JSON arguments raise instead of silently becoming {}.

        Executing with empty arguments ran the tool's defaults (which can
        have side effects) and hid the model's formatting error.
        """
        tool_call = {
            "function": {
                "name": "test",
                "arguments": "not valid json",
            },
        }

        with pytest.raises(ValueError, match="not valid JSON"):
            openai_call_to_mcp(tool_call)

    def test_parse_dict_arguments(self):
        """Test parsing with already-parsed dict arguments."""
        tool_call = {
            "function": {
                "name": "test",
                "arguments": {"already": "parsed"},
            },
        }

        _, _, arguments = openai_call_to_mcp(tool_call)

        assert arguments == {"already": "parsed"}

    def test_parse_multiple_underscores(self):
        """Test parsing with multiple underscores in name."""
        tool_call = {
            "function": {
                "name": "server__tool__with__underscores",
                "arguments": "{}",
            },
        }

        server_name, tool_name, _ = openai_call_to_mcp(tool_call)

        assert server_name == "server"
        assert tool_name == "tool__with__underscores"

    def test_parse_missing_function(self):
        """Test parsing with missing function key."""
        tool_call = {}

        server_name, tool_name, arguments = openai_call_to_mcp(tool_call)

        assert server_name == ""
        assert tool_name == ""
        assert arguments == {}


class TestFormatToolResult:
    """Tests for format_tool_result."""

    def test_format_success_string(self):
        """Test formatting successful string result."""
        result = MCPToolResult(
            tool_name="test_tool",
            content="Result text",
        )

        message = format_tool_result(result, "call_123")

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_123"
        assert message["content"] == "Result text"

    def test_format_success_dict(self):
        """Test formatting successful dict result."""
        result = MCPToolResult(
            tool_name="test_tool",
            content={"data": [1, 2, 3]},
        )

        message = format_tool_result(result, "call_456")

        assert message["role"] == "tool"
        assert json.loads(message["content"]) == {"data": [1, 2, 3]}

    def test_format_error(self):
        """Test formatting error result."""
        result = MCPToolResult(
            tool_name="test_tool",
            content=None,
            is_error=True,
            error_message="Something went wrong",
        )

        message = format_tool_result(result, "call_789")

        assert message["content"] == "Error: Something went wrong"


class TestFormatToolResults:
    """Tests for format_tool_results."""

    def test_format_multiple_results(self):
        """Test formatting multiple results."""
        results = [
            (
                MCPToolResult(tool_name="t1", content="Result 1"),
                "call_1",
            ),
            (
                MCPToolResult(tool_name="t2", content="Result 2"),
                "call_2",
            ),
        ]

        messages = format_tool_results(results)

        assert len(messages) == 2
        assert messages[0]["tool_call_id"] == "call_1"
        assert messages[1]["tool_call_id"] == "call_2"

    def test_format_empty_list(self):
        """Test formatting empty list."""
        messages = format_tool_results([])
        assert messages == []


class TestMergeTools:
    """Tests for merge_tools function."""

    def test_mcp_tools_only(self):
        """Test merging with only MCP tools."""
        mcp_tools = [
            MCPTool(server_name="s", name="t1", description="Tool 1"),
            MCPTool(server_name="s", name="t2", description="Tool 2"),
        ]

        result = merge_tools(mcp_tools)

        assert len(result) == 2

    def test_user_tools_only(self):
        """Test merging with only user tools."""
        user_tools = [
            {
                "type": "function",
                "function": {
                    "name": "user_tool",
                    "description": "User defined",
                    "parameters": {},
                },
            }
        ]

        result = merge_tools([], user_tools)

        assert len(result) == 1
        assert result[0]["function"]["name"] == "user_tool"

    def test_merge_both(self):
        """Test merging both MCP and user tools."""
        mcp_tools = [
            MCPTool(server_name="s", name="mcp_tool", description="MCP"),
        ]
        user_tools = [
            {
                "type": "function",
                "function": {
                    "name": "user_tool",
                    "description": "User",
                    "parameters": {},
                },
            }
        ]

        result = merge_tools(mcp_tools, user_tools)

        assert len(result) == 2
        names = [t["function"]["name"] for t in result]
        assert "s__mcp_tool" in names
        assert "user_tool" in names

    def test_user_overrides_mcp(self):
        """Test user tool overrides MCP tool with same name."""
        mcp_tools = [
            MCPTool(server_name="s", name="tool", description="MCP version"),
        ]
        user_tools = [
            {
                "type": "function",
                "function": {
                    "name": "s__tool",  # Same full name
                    "description": "User version",
                    "parameters": {},
                },
            }
        ]

        result = merge_tools(mcp_tools, user_tools)

        assert len(result) == 1
        assert result[0]["function"]["description"] == "User version"

    def test_none_user_tools(self):
        """Test with None user_tools."""
        mcp_tools = [
            MCPTool(server_name="s", name="tool", description="Test"),
        ]

        result = merge_tools(mcp_tools, None)

        assert len(result) == 1


class TestExtractToolCalls:
    """Tests for extract_tool_calls function."""

    def test_extract_from_response(self):
        """Test extracting tool calls from response."""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "tool1", "arguments": "{}"},
                            },
                            {
                                "id": "call_2",
                                "function": {"name": "tool2", "arguments": "{}"},
                            },
                        ]
                    }
                }
            ]
        }

        calls = extract_tool_calls(response)

        assert len(calls) == 2
        assert calls[0]["id"] == "call_1"
        assert calls[1]["id"] == "call_2"

    def test_extract_empty_choices(self):
        """Test extracting from empty choices."""
        response = {"choices": []}

        calls = extract_tool_calls(response)

        assert calls == []

    def test_extract_no_tool_calls(self):
        """Test extracting when no tool_calls present."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Regular response",
                    }
                }
            ]
        }

        calls = extract_tool_calls(response)

        assert calls == []

    def test_extract_missing_message(self):
        """Test extracting when message is missing."""
        response = {"choices": [{}]}

        calls = extract_tool_calls(response)

        assert calls == []


class TestHasToolCalls:
    """Tests for has_tool_calls function."""

    def test_has_tool_calls_true(self):
        """Test returns True when tool calls present."""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "call_1", "function": {"name": "t", "arguments": "{}"}}
                        ]
                    }
                }
            ]
        }

        assert has_tool_calls(response) is True

    def test_has_tool_calls_false(self):
        """Test returns False when no tool calls."""
        response = {
            "choices": [{"message": {"content": "No tools"}}]
        }

        assert has_tool_calls(response) is False

    def test_has_tool_calls_empty(self):
        """Test returns False for empty response."""
        response = {}

        assert has_tool_calls(response) is False
