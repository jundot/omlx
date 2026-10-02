# SPDX-License-Identifier: Apache-2.0
"""Parse IFM K2 Horizon XML and JSON tool-call envelopes for mlx-lm."""

from __future__ import annotations

import json
from typing import Any

import regex as re
from mlx_lm.tool_parsers.glm47 import _get_string_arg_names, _normalize_arguments

tool_call_start = "<ifm|tool_calls>"
tool_call_end = "</ifm|tool_calls>"

_CALL_START = "<ifm|tool_call>"
_CALL_END = "</ifm|tool_call>"
_ARGUMENT_PATTERN = re.compile(
    r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*"
    r"(?:<ifm\|arg_type>(.*?)</ifm\|arg_type>\s*)?"
    r"<ifm\|arg_value>(.*?)</ifm\|arg_value>",
    re.DOTALL,
)


def _parse_json_call(payload: Any, tools: list[Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("K2 Horizon JSON tool call must be an object")
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not name:
        raise ValueError("K2 Horizon JSON tool call is missing a function name")
    if not isinstance(arguments, dict):
        raise ValueError(f"K2 Horizon tool call {name!r} arguments must be an object")
    return {"name": name, "arguments": _normalize_arguments(name, arguments, tools)}


def _parse_xml_call(body: str, tools: list[Any] | None) -> dict[str, Any]:
    matches = list(_ARGUMENT_PATTERN.finditer(body))
    name = (body[: matches[0].start()] if matches else body).strip()
    if not name or "<ifm|" in name:
        raise ValueError("K2 Horizon XML tool call is missing a function name")
    if matches and _ARGUMENT_PATTERN.sub("", body[matches[0].start() :]).strip():
        raise ValueError("K2 Horizon XML tool call contains an incomplete argument")

    string_args = _get_string_arg_names(name, tools)
    arguments: dict[str, Any] = {}
    for match in matches:
        key = match.group(1).strip()
        declared_type = match.group(2)
        if declared_type is not None:
            if declared_type.strip() == "string":
                string_args.add(key)
            else:
                string_args.discard(key)
        arguments[key] = match.group(3)
    return {
        "name": name,
        "arguments": _normalize_arguments(name, arguments, tools, string_args),
    }


def _parse_group_body(text: str, start: int, tools: list[Any] | None):
    calls = []
    while True:
        while start < len(text) and text[start].isspace():
            start += 1
        if not text.startswith(_CALL_START, start):
            break
        body_start = start + len(_CALL_START)
        while body_start < len(text) and text[body_start].isspace():
            body_start += 1
        if text.startswith("{", body_start):
            payload, end = json.JSONDecoder().raw_decode(text, body_start)
            call = _parse_json_call(payload, tools)
            while end < len(text) and text[end].isspace():
                end += 1
        else:
            end = text.find(_CALL_END, body_start)
            if end < 0:
                raise ValueError("K2 Horizon tool group contains an incomplete call")
            call = _parse_xml_call(text[body_start:end].strip(), tools)
        if not text.startswith(_CALL_END, end):
            raise ValueError("K2 Horizon tool group contains an incomplete call")
        calls.append(call)
        start = end + len(_CALL_END)
    if not calls:
        raise ValueError("K2 Horizon tool group contains no complete <ifm|tool_call>")
    return calls, start


def parse_tool_call(text: str, tools: list[Any] | None = None) -> list[dict[str, Any]]:
    """Parse one ``<ifm|tool_calls>`` group body into OpenAI-style call dicts."""
    calls, end = _parse_group_body(text, 0, tools)
    if text[end:].strip():
        raise ValueError("K2 Horizon tool group contains an incomplete call")
    return calls


def parse_tool_groups(text: str, tools: list[Any] | None = None):
    """Parse complete groups, retaining prose outside their structural boundaries."""
    visible, calls = [], []
    position = 0
    while True:
        start = text.find(tool_call_start, position)
        if start < 0:
            visible.append(text[position:])
            break
        visible.append(text[position:start])
        group, end = _parse_group_body(text, start + len(tool_call_start), tools)
        if not text.startswith(tool_call_end, end):
            raise ValueError("Incomplete or malformed K2 tool-call envelope")
        calls.extend(group)
        position = end + len(tool_call_end)
    return "".join(visible), calls
