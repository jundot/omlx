# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: Apache-2.0
"""Laguna XML-like tool-call parser from MLX-LM PR #1223.

Laguna emits an XML-style function name and argument envelope. Some clients
also observed JSON call bodies inside the same envelope, so both forms are
accepted.
"""

from __future__ import annotations

import ast
import json
from typing import Any

import regex as re

tool_call_start = "<tool_call>"
tool_call_end = "</tool_call>"

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_NAME_PATTERN = re.compile(r"^(.*?)<arg_key>", re.DOTALL)
_ARGUMENT_PAIR_PATTERN = re.compile(
    r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)


def _is_string_argument(
    tool_name: str,
    argument_name: str,
    tools: list[Any] | None,
) -> bool:
    if tools is None:
        return False

    for tool_definition in tools:
        function_definition = tool_definition.get("function", {})
        if function_definition.get("name") != tool_name:
            continue
        parameters = function_definition.get("parameters") or {}
        argument_type = (
            parameters.get("properties", {}).get(argument_name, {}).get("type")
        )
        # ``true`` and numeric-looking values are otherwise decoded below; an
        # explicit string schema must preserve the model's literal text.
        return isinstance(argument_type, str) and argument_type == "string"
    return False


def _deserialize_argument(argument_value: str) -> Any:
    # XML-style calls encode values as text. Recover JSON/Python scalar types
    # when possible, then preserve free-form values unchanged.
    try:
        return json.loads(argument_value)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(argument_value)
    except (ValueError, SyntaxError):
        return argument_value


def _parse_json_call(tool_call_text: str) -> dict[str, Any] | None:
    try:
        parsed_call = json.loads(tool_call_text.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed_call, dict):
        return None

    tool_name = parsed_call.get("name")
    arguments = parsed_call.get("arguments", {})
    if not isinstance(tool_name, str) or not isinstance(arguments, dict):
        return None
    return {"name": tool_name, "arguments": arguments}


def _parse_name_line_json_arguments(tool_call_text: str) -> dict[str, Any] | None:
    tool_name, separator, arguments_text = tool_call_text.partition("\n")
    if not separator or not arguments_text.lstrip().startswith("{"):
        return None

    try:
        arguments = json.loads(arguments_text.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(arguments, dict):
        return None
    return {"name": tool_name.strip(), "arguments": arguments}


def _parse_single_call(
    tool_call_text: str,
    tools: list[Any] | None,
) -> dict[str, Any]:
    normalized_text = tool_call_text.strip()
    json_call = _parse_json_call(normalized_text)
    if json_call is not None:
        return json_call

    name_line_json_call = _parse_name_line_json_arguments(normalized_text)
    if name_line_json_call is not None:
        return name_line_json_call

    function_name_match = _FUNCTION_NAME_PATTERN.search(normalized_text)
    if function_name_match is None:
        return {
            "name": normalized_text.split("\n", 1)[0].strip(),
            "arguments": {},
        }

    tool_name = function_name_match.group(1).strip()
    arguments: dict[str, Any] = {}
    for argument_name, argument_value in _ARGUMENT_PAIR_PATTERN.findall(
        normalized_text
    ):
        normalized_argument_name = argument_name.strip()
        normalized_argument_value = argument_value.strip()
        if not _is_string_argument(tool_name, normalized_argument_name, tools):
            normalized_argument_value = _deserialize_argument(normalized_argument_value)
        arguments[normalized_argument_name] = normalized_argument_value
    return {"name": tool_name, "arguments": arguments}


def parse_tool_call(
    tool_call_text: str,
    tools: list[Any] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Parse one or more Laguna tool calls from raw model output.

    MLX-LM normally strips the outer markers before invoking a parser, while
    direct callers may pass complete ``<tool_call>`` blocks; support both.
    """
    matched_calls = _TOOL_CALL_PATTERN.findall(tool_call_text)
    if matched_calls:
        parsed_calls = [_parse_single_call(match, tools) for match in matched_calls]
        return parsed_calls[0] if len(parsed_calls) == 1 else parsed_calls
    return _parse_single_call(tool_call_text, tools)
