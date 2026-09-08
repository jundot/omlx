# SPDX-License-Identifier: Apache-2.0
"""Request-owned K2 tool names, compiled by oMLX's existing grammar backend."""

from ...exceptions import InvalidRequestError


def validate_tool_prefix(messages, tools, is_partial):
    if tools and is_partial and messages:
        prefix = str(messages[-1].get("content") or "")
        if prefix.rfind("<ifm|tool_calls>") > prefix.rfind("</ifm|tool_calls>"):
            raise InvalidRequestError(
                "K2 tool-name constraints require a complete assistant tool-call prefix."
            )


def _token(value):
    return {"type": "token", "token": value}


def _sequence(*elements):
    return {"type": "sequence", "elements": list(elements)}


def compile_tool_grammar(compiler, tools, existing=None):
    """Constrain names in native XML/JSON envelopes without inspecting descriptions."""
    if not tools:
        return existing
    if existing is not None:
        raise InvalidRequestError(
            "K2 tool names and structured output cannot be constrained together."
        )
    if compiler is None:
        raise InvalidRequestError(
            "K2 tool-name constraints require xgrammar. Install omlx[grammar]."
        )
    names = sorted({tool["function"]["name"] for tool in tools})
    whitespace = {"type": "grammar", "grammar": r"root ::= [ \t\r\n]*"}
    xml = _sequence(
        whitespace,
        {
            "type": "or",
            "elements": [{"type": "const_string", "value": name} for name in names],
        },
        whitespace,
        {
            "type": "optional",
            "content": _sequence(
                _token("<ifm|arg_key>"),
                {"type": "any_tokens", "exclude_tokens": ["</ifm|tool_call>"]},
            ),
        },
    )
    properties = {
        "name": {"type": "string", "enum": names},
        "arguments": {"type": "object"},
    }
    json_formats = [
        {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {key: properties[key] for key in order},
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
        }
        for order in (("name", "arguments"), ("arguments", "name"))
    ]
    call = {
        "type": "tag",
        "begin": _token("<ifm|tool_call>"),
        "content": {"type": "or", "elements": [xml, *json_formats]},
        "end": _token("</ifm|tool_call>"),
    }
    group = {
        "type": "tag",
        "begin": _token("<ifm|tool_calls>"),
        "content": _sequence(
            whitespace, {"type": "plus", "content": _sequence(call, whitespace)}
        ),
        "end": _token("</ifm|tool_calls>"),
    }
    return compiler.compile_structural_tag(
        {
            "type": "structural_tag",
            "format": {
                "type": "token_triggered_tags",
                "trigger_tokens": ["<ifm|tool_calls>"],
                "tags": [group],
            },
        }
    )
