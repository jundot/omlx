# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 function-calling parser compatible with mlx-lm's parser API."""

import json
import re
from typing import Any, Optional

_tool_call_regex = re.compile(r"call:([\w.-]+)(\{.*\})\s*\Z", re.DOTALL)


def parse_tool_call(text: str, _: Optional[Any] = None):
    match = _tool_call_regex.search(text.strip())
    if not match:
        raise ValueError("No function provided.")

    func_name, args_str = match.groups()
    arguments = json.loads(args_str)
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    return dict(name=func_name, arguments=arguments)


tool_call_start = "<|tool_call>"
tool_call_end = "<tool_call|>"
