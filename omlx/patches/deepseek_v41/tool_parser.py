# SPDX-License-Identifier: MIT
"""V4.1 DSML adapter around the official completion parser."""

import json
import re

from .encoding import parse_message_from_completion_text

# Canonical V4.1 markers (leading space in tag names from encoding.py).
tool_call_start = "<｜DSML｜ calls>"
tool_call_end = "</｜DSML｜ calls>"

# Model drift / V4-compat openers that must still leave the content stream.
_TOOL_CALL_START_MARKERS = (
    tool_call_start,
    "<｜DSML｜calls>",
    "<｜DSML｜tool_calls>",
    "<｜DSML｜ tool_calls>",
)
_TOOL_CALL_END_MARKERS = (
    tool_call_end,
    "</｜DSML｜calls>",
    "</｜DSML｜tool_calls>",
    "</｜DSML｜ tool_calls>",
)

# Match any recognized envelope closer for streaming cutoff.
_TOOL_CALL_END_RE = re.compile(
    r"</｜DSML｜\s*(?:tool_)?calls\s*>",
    re.IGNORECASE,
)


def find_tool_call_start(text: str, start: int = 0):
    """Return (index, marker) for the earliest known DSML tool-calls opener."""
    best = None
    for marker in _TOOL_CALL_START_MARKERS:
        idx = text.find(marker, start)
        if idx < 0:
            continue
        if best is None or idx < best[0]:
            best = (idx, marker)
    return best


def find_tool_call_end(text: str, start: int = 0):
    """Return (index, end_exclusive) for the earliest known DSML closer."""
    match = _TOOL_CALL_END_RE.search(text, start)
    if match is None:
        return None
    return (match.start(), match.end())


def parse_tool_call(text, tools=None):
    parsed = parse_message_from_completion_text(
        "\n\n"
        + tool_call_start
        + "\n"
        + text.strip()
        + "\n"
        + tool_call_end
        + "<｜end▁of▁sentence｜>",
        "chat",
    )
    calls = parsed.get("tool_calls") or []
    if not calls:
        raise ValueError("No complete DeepSeek V4.1 tool invocation")
    for call in calls:
        if not isinstance(json.loads(call["function"]["arguments"]), dict):
            raise ValueError("DeepSeek V4.1 tool arguments must be a JSON object")
    return [
        {
            # oMLX uses the qualified name in the OpenAI function.name field.
            "name": (
                f'{call["namespace"]}::{call["function"]["name"]}'
                if call.get("namespace") is not None
                else call["function"]["name"]
            ),
            "arguments": call["function"]["arguments"],
        }
        for call in calls
    ]
