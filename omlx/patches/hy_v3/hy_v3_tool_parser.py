# Copyright © 2026 Apple Inc.

"""
Tool parser for Tencent Hy3-preview (HYV3).

Reference:
https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/hy_v3_tool_parser.py

Format:
    <tool_calls>
    <tool_call>function_name<tool_sep>
    <arg_key>key1</arg_key>
    <arg_value>value1</arg_value>
    ...
    </tool_call>
    ...
    </tool_calls>

The full Hy3 release renames every token with a ":opensource" suffix
(``<tool_call:opensource>`` etc.), which the regexes below also accept;
see ``hy_v3_opensource`` for the suffixed start/end sentinels.
"""

import ast
import json
from typing import Any

import regex as re

tool_call_start = "<tool_calls>"
tool_call_end = "</tool_calls>"

_SUFFIX = r"(?::[\w-]+)?"

_tool_call_regex = re.compile(
    rf"<tool_call{_SUFFIX}>(.*?)<tool_sep{_SUFFIX}>(.*?)</tool_call{_SUFFIX}>",
    re.DOTALL,
)
_arg_pair_regex = re.compile(
    rf"<arg_key{_SUFFIX}>(.*?)</arg_key{_SUFFIX}>"
    rf"(?:\\n|\s)*"
    rf"<arg_value{_SUFFIX}>(.*?)</arg_value{_SUFFIX}>",
    re.DOTALL,
)
_tool_sep_regex = re.compile(
    rf"(?:\s*<tool_call{_SUFFIX}>)?(.*?)<tool_sep{_SUFFIX}>", re.DOTALL
)
_arg_key_regex = re.compile(rf"<arg_key{_SUFFIX}>")


def _is_string_type(
    tool_name: str,
    arg_name: str,
    tools: list[Any] | None,
) -> bool:
    if tools is None:
        return False
    for tool in tools:
        func = tool.get("function", {})
        if func.get("name") != tool_name:
            continue
        params = func.get("parameters") or {}
        arg_type = params.get("properties", {}).get(arg_name, {}).get("type")
        return arg_type == "string"
    return False


def _deserialize(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    return value


def _parse_single_call(text: str, tools: list[Any] | None):
    name_match = _tool_sep_regex.match(text)
    if name_match:
        func_name = name_match.group(1).strip()
        body = text[name_match.end() :]
    else:
        func_name = _arg_key_regex.split(text, 1)[0].strip()
        body = text

    arg_dct: dict[str, Any] = {}
    for key, value in _arg_pair_regex.findall(body):
        arg_key = key.strip()
        arg_val = value.strip()
        if not _is_string_type(func_name, arg_key, tools):
            arg_val = _deserialize(arg_val)
        arg_dct[arg_key] = arg_val
    return dict(name=func_name, arguments=arg_dct)


def parse_tool_call(text: str, tools: list[Any] | None = None):
    matches = _tool_call_regex.findall(text)
    if matches:
        calls = [
            _parse_single_call(f"{name}<tool_sep>{body}", tools)
            for name, body in matches
        ]
        return calls[0] if len(calls) == 1 else calls

    return _parse_single_call(text.strip(), tools)
