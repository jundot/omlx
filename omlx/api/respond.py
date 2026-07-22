# SPDX-License-Identifier: Apache-2.0
"""Synthetic respond tool — structured alternative to bare text responses.

The model calls respond(message="...") instead of producing bare text.
This keeps the model in tool-calling mode where oMLX's guardrail stack
applies. The respond tool is injected server-side and stripped before
the response returns — the client never sees it.

Adapted from Forge's forge/tools/respond.py.

Usage:
    tools = inject_respond_tool(tools)   # before generation
    ...
    real_calls, text = strip_respond_calls(parsed_calls)  # after parsing
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

RESPOND_TOOL_NAME = "respond"

RESPOND_DESCRIPTION = (
    "Respond to the user with a message. Use this when the user is chatting, "
    "asking a question, when you need to ask a clarifying question before "
    "proceeding, or when no other tool action is needed. Use this "
    "after completing the user's request to report the result."
)

RESPOND_TOOL_SPEC: dict = {
    "type": "function",
    "function": {
        "name": RESPOND_TOOL_NAME,
        "description": RESPOND_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send to the user.",
                }
            },
            "required": ["message"],
        },
    },
}


def inject_respond_tool(tools: list | None) -> list:
    """Append the respond tool to the tools list if not already present.

    Args:
        tools: Existing tools list (OpenAI function format). May be None
            or empty — in that case, returns the input unchanged.

    Returns:
        A new list with the respond tool appended, or the original list
        if it was empty/None or already contains a respond tool.
    """
    if not tools:
        return tools  # type: ignore[return-value]

    for tool in tools:
        func = tool.get("function", {}) if isinstance(tool, dict) else {}
        if func.get("name") == RESPOND_TOOL_NAME:
            return tools

    return tools + [RESPOND_TOOL_SPEC]


def strip_respond_calls(
    tool_calls: list[Any] | None,
) -> tuple[list[Any], str | None]:
    """Strip respond calls from parsed tool call output.

    Three cases:
      - Pure respond (only call is respond): return ([], message_text)
      - Mixed (respond + real calls): return (real_calls, None)
      - No respond calls: return (original_calls, None) — passthrough

    Args:
        tool_calls: Parsed ToolCall objects or None.

    Returns:
        Tuple of (real_tool_calls, respond_message). respond_message is
        the extracted message string when the only call was respond,
        otherwise None.
    """
    if not tool_calls:
        return ([], None)

    respond_calls: list[Any] = []
    real_calls: list[Any] = []

    for tc in tool_calls:
        name = _get_name(tc)
        if name == RESPOND_TOOL_NAME:
            respond_calls.append(tc)
        else:
            real_calls.append(tc)

    if respond_calls and not real_calls:
        message = _extract_message(respond_calls[0])
        return ([], message)

    return (real_calls, None)


def _get_name(tc: Any) -> str:
    """Extract function name from a ToolCall or dict."""
    func = getattr(getattr(tc, "function", None), "name", None)
    if func:
        return func
    if isinstance(tc, dict):
        f = tc.get("function", {})
        return f.get("name", "") if isinstance(f, dict) else ""
    return ""


def _extract_message(tc: Any) -> str:
    """Extract the 'message' argument from a respond ToolCall."""
    raw_args: Any = None
    func = getattr(tc, "function", None)
    if func is not None:
        raw_args = getattr(func, "arguments", None)
    elif isinstance(tc, dict):
        f = tc.get("function", {})
        raw_args = f.get("arguments") if isinstance(f, dict) else None

    if isinstance(raw_args, dict):
        return str(raw_args.get("message", ""))
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return str(parsed.get("message", "")) if isinstance(parsed, dict) else ""
        except (json.JSONDecodeError, ValueError):
            return ""
    return ""
