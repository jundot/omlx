# SPDX-License-Identifier: MIT
"""V4.1 DSML adapter around the official completion parser."""

import json

from .encoding import parse_message_from_completion_text

tool_call_start = "<｜DSML｜ calls>"
tool_call_end = "</｜DSML｜ calls>"


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
