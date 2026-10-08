# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Tool schema conversion utilities for MCP <-> OpenAI formats.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from .types import MCPTool, MCPToolResult


def mcp_tool_to_openai(tool: MCPTool) -> Dict[str, Any]:
    """
    Convert MCP tool schema to OpenAI function calling format.

    Args:
        tool: MCPTool instance

    Returns:
        OpenAI-compatible tool definition
    """
    return {
        "type": "function",
        "function": {
            "name": tool.full_name,
            "description": tool.description,
            "parameters": tool.input_schema or {
                "type": "object",
                "properties": {},
            },
        }
    }


def mcp_tools_to_openai(tools: List[MCPTool]) -> List[Dict[str, Any]]:
    """
    Convert list of MCP tools to OpenAI format.

    Args:
        tools: List of MCPTool instances

    Returns:
        List of OpenAI-compatible tool definitions
    """
    return [mcp_tool_to_openai(tool) for tool in tools]


def openai_call_to_mcp(tool_call: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """
    Parse OpenAI tool call back to MCP format.

    Args:
        tool_call: OpenAI tool call from model response

    Returns:
        Tuple of (server_name, tool_name, arguments)

    Raises:
        ValueError: If tool call format is invalid
    """
    # Extract function info
    function = tool_call.get("function", {})
    full_name = function.get("name", "")
    arguments_str = function.get("arguments", "{}")

    # Parse arguments
    if isinstance(arguments_str, str):
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            # Executing with silently-empty arguments runs the tool's
            # defaults (which can have side effects) and hides the model's
            # formatting error from it; surface it instead so callers can
            # return an error result the model can self-correct on.
            raise ValueError(
                f"tool arguments are not valid JSON: {e}"
            ) from e
    else:
        arguments = arguments_str or {}

    # Split namespaced name (server__tool)
    if "__" in full_name:
        server_name, tool_name = full_name.split("__", 1)
    else:
        # No namespace, use as-is (will need server lookup)
        server_name = ""
        tool_name = full_name

    return server_name, tool_name, arguments


def format_tool_result(result: MCPToolResult, tool_call_id: str) -> Dict[str, Any]:
    """
    Format tool result for inclusion in conversation messages.

    Args:
        result: MCPToolResult from tool execution
        tool_call_id: ID of the tool call this is responding to

    Returns:
        OpenAI-compatible tool result message
    """
    return result.to_message(tool_call_id)


def format_tool_results(
    results: List[Tuple[MCPToolResult, str]]
) -> List[Dict[str, Any]]:
    """
    Format multiple tool results as messages.

    Args:
        results: List of (MCPToolResult, tool_call_id) tuples

    Returns:
        List of OpenAI-compatible tool result messages
    """
    return [format_tool_result(result, call_id) for result, call_id in results]


def merge_tools(
    mcp_tools: List[MCPTool],
    user_tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Merge MCP tools with user-provided tools.

    User tools take precedence if there are name conflicts.

    Args:
        mcp_tools: Tools discovered from MCP servers
        user_tools: User-provided tools in OpenAI format

    Returns:
        Combined list of tools in OpenAI format
    """
    # Convert MCP tools to OpenAI format
    all_tools = {tool.full_name: mcp_tool_to_openai(tool) for tool in mcp_tools}

    # Add/override with user tools
    if user_tools:
        for tool in user_tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            if name:
                all_tools[name] = tool

    return list(all_tools.values())


def extract_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract tool calls from model response.

    Args:
        response: OpenAI-format model response

    Returns:
        List of tool calls
    """
    choices = response.get("choices", [])
    if not choices:
        return []

    message = choices[0].get("message", {})
    return message.get("tool_calls", [])


def has_tool_calls(response: Dict[str, Any]) -> bool:
    """
    Check if response contains tool calls.

    Args:
        response: OpenAI-format model response

    Returns:
        True if response contains tool calls
    """
    return len(extract_tool_calls(response)) > 0
