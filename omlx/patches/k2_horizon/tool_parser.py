# SPDX-License-Identifier: Apache-2.0
"""Parse IFM K2 Horizon XML and JSON tool-call envelopes for mlx-lm."""

from __future__ import annotations

import json
from typing import Any

import regex as re
from mlx_lm.tool_parsers.glm47 import _get_string_arg_names, _normalize_arguments

tool_call_start = "<ifm|tool_calls>"
tool_call_end = "</ifm|tool_calls>"

_CALL_PATTERN = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", re.DOTALL)
_ARGUMENT_PATTERN = re.compile(
    r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*"
    r"(?:<ifm\|arg_type>(.*?)</ifm\|arg_type>\s*)?"
    r"<ifm\|arg_value>(.*?)</ifm\|arg_value>",
    re.DOTALL,
)


def _parse_json_call(body: str, tools: list[Any] | None) -> dict[str, Any]:
    payload = json.loads(body)
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
        arguments[key] = match.group(3).strip()
    return {
        "name": name,
        "arguments": _normalize_arguments(name, arguments, tools, string_args),
    }


def parse_tool_call(text: str, tools: list[Any] | None = None) -> list[dict[str, Any]]:
    """Parse one ``<ifm|tool_calls>`` group body into OpenAI-style call dicts."""
    bodies = [body.strip() for body in _CALL_PATTERN.findall(text)]
    if not bodies:
        raise ValueError("K2 Horizon tool group contains no complete <ifm|tool_call>")
    return [
        (
            _parse_json_call(body, tools)
            if body.startswith("{")
            else _parse_xml_call(body, tools)
        )
        for body in bodies
    ]
