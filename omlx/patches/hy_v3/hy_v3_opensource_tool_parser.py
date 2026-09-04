# Copyright © 2026 Apple Inc.

"""
Tool parser for Tencent Hy3 (HYV3).

The full Hy3 release renames the Hy3-preview chat tokens with an
":opensource" suffix at the same token ids (``<tool_calls>`` becomes
``<tool_calls:opensource>``). Parsing is shared with ``hy_v3``; only the
start and end sentinels differ.
"""

from .hy_v3 import parse_tool_call

tool_call_start = "<tool_calls:opensource>"
tool_call_end = "</tool_calls:opensource>"
