# SPDX-License-Identifier: Apache-2.0
"""Tool parser for Tencent Hy4 (``hy_v4``).

Format, inside ``<tool_calls:opensource> ... </tool_calls:opensource>``::

    <tool_call:opensource>function_name
    <arg_key:opensource>key</arg_key:opensource>
    <arg_value:opensource>value</arg_value:opensource>
    </tool_call:opensource>

Unlike Hy3 there is no ``<tool_sep>`` after the function name, so each call
block is split first and its name ends at the first ``<arg_key>``.
"""

from __future__ import annotations

from typing import Any

import regex as re

from ..hy_v3.hy_v3_tool_parser import _parse_single_call

tool_call_start = "<tool_calls:opensource>"
tool_call_end = "</tool_calls:opensource>"

_SUFFIX = r"(?::[\w-]+)?"
_call_regex = re.compile(rf"<tool_call{_SUFFIX}>(.*?)</tool_call{_SUFFIX}>", re.DOTALL)


def parse_tool_call(text: str, tools: list[Any] | None = None):
    bodies = _call_regex.findall(text) or [text.strip()]
    calls = [_parse_single_call(body.strip(), tools) for body in bodies]
    return calls[0] if len(calls) == 1 else calls


def is_hy_v4_template(template: Any) -> bool:
    """Hy4 renders ``<arg_key:opensource>`` calls without Hy3's ``<tool_sep>``.

    The released template builds its tokens with ``'<arg_key{}>'.format(...)``.
    """
    if not isinstance(template, str) or "<tool_sep" not in template:
        return (
            isinstance(template, str)
            and ":opensource" in template
            and ("<arg_key:opensource>" in template or "<arg_key{}>" in template)
        )
    return False
