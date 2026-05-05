# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 DSML tool-call parser for mlx-lm.

Implements the parser-side counterpart to ``chat_template_v4.py``. The
DSML output grammar emitted by DeepSeek V4 looks like::

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="city" string="true">Seoul</｜DSML｜parameter>
    <｜DSML｜parameter name="unit" string="false">"celsius"</｜DSML｜parameter>
    </｜DSML｜invoke>
    <｜DSML｜invoke name="another_fn">
    ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

Outer ``<｜DSML｜tool_calls>`` marker handling is delegated to mlx-lm's
``TokenizerWrapper`` (it is recognized via ``tool_call_start`` /
``tool_call_end`` and stripped before ``parse_tool_call`` runs). This
module sees only the body — one or more ``<｜DSML｜invoke>`` blocks.

``string="true"`` parameters are kept as raw strings; ``string="false"``
parameters are JSON-decoded so types (numbers, bools, arrays, objects)
survive the round-trip. The ``tools`` argument is accepted for
interface compatibility with other mlx-lm parsers but currently
unused — type recovery from the ``string="..."`` attribute alone is
sufficient.

omlx registers this module as ``mlx_lm.tool_parsers.deepseek_v4``. Once
mlx-lm or upstream mlx-community's tokenizer_config sets
``"tool_parser_type": "deepseek_v4"``, the standard mlx-lm parser
loader picks this up. The omlx tokenizer patch additionally injects the
type into the in-memory wrapper so the published minimal
``chat_template.jinja`` does not need editing.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

# Outer markers — must match what TokenizerWrapper strips.
tool_call_start = "<｜DSML｜tool_calls>"
tool_call_end = "</｜DSML｜tool_calls>"

# Inner grammar.
_INVOKE_RE = re.compile(
    r"<｜DSML｜invoke\s+name=\"(?P<name>[^\"]+)\"\s*>"
    r"(?P<body>.*?)"
    r"</｜DSML｜invoke>",
    re.DOTALL,
)

_PARAM_RE = re.compile(
    r"<｜DSML｜parameter\s+name=\"(?P<key>[^\"]+)\""
    r"\s+string=\"(?P<is_str>true|false)\"\s*>"
    r"(?P<value>.*?)"
    r"</｜DSML｜parameter>",
    re.DOTALL,
)


def _decode_value(raw: str, is_str: bool) -> Any:
    """Decode a single parameter value.

    String parameters arrive verbatim. Non-string parameters arrive as
    JSON literals (e.g. ``42``, ``true``, ``[1, 2]``). Fall back to
    ``ast.literal_eval`` if JSON decoding fails — covers Python-style
    literals models occasionally emit.
    """
    # Models often pad values with a trailing newline before the closing
    # tag. Trim that and only that — leading/trailing intentional
    # whitespace inside string values must be preserved.
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]

    if is_str:
        return raw

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            # Last-resort: pass through as a string so the caller still
            # gets something usable rather than an exception.
            return raw


def _parse_single_invoke(name: str, body: str) -> dict:
    arguments: dict[str, Any] = {}
    for m in _PARAM_RE.finditer(body):
        key = m.group("key")
        is_str = m.group("is_str") == "true"
        arguments[key] = _decode_value(m.group("value"), is_str)
    return {"name": name, "arguments": arguments}


def parse_tool_call(text: str, tools: Any | None = None):
    """Parse the body of a ``<｜DSML｜tool_calls>`` block.

    Returns a single ``{"name": ..., "arguments": ...}`` dict if the
    text contains exactly one ``<｜DSML｜invoke>``, or a list of dicts
    if it contains multiple — same convention as mlx-lm's gemma4
    parser.
    """
    matches = list(_INVOKE_RE.finditer(text))
    if not matches:
        raise ValueError(
            "No <｜DSML｜invoke> block found in DeepSeek V4 tool-call text"
        )

    parsed = [_parse_single_invoke(m.group("name"), m.group("body")) for m in matches]
    if len(parsed) == 1:
        return parsed[0]
    return parsed
