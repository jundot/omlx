"""Differential check: the prose fast path must be byte-identical.

Runs the real ``ToolCallStreamFilter`` twice over the same chunking — once
with the guard active, once with it disabled — and compares every emitted
fragment plus ``finish()``. Used to validate the fast path in
``ToolCallStreamFilter.feed``.
"""

from __future__ import annotations

from omlx.api.tool_calling import ToolCallStreamFilter

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }
]

CASES = [
    " the quick brown fox ",
    "plain code: x = 1\n",
    "no markers here at all " * 5,
    "<",
    "[",
    "<tool_call>",
    '<tool_call>{"name": "read_file"}</tool_call>',
    'prefix <tool_call>{"a": 1}</tool_call> suffix ',
    "[Calling tool:",
    "[Calling tool: read_file({})]",
    "a < b and c [ d",
    "<|tool_call_start|>{}<|tool_call_end|>",
    "hello\n\n<tool_call>x</tool_call>",
    "trailing <tool_call",  # partial marker at EOF
    "brace ] and bracket [ mix",
]


class _Tok:
    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"


def run(parts: list[str], *, guard: bool) -> list[str]:
    f = ToolCallStreamFilter(_Tok(), tools=TOOLS)
    if not guard:
        # Neutralise the guard: every character becomes a trigger, so the
        # condition can never hold and the slow path always runs.
        f._marker_trigger_chars = frozenset(
            chr(c) for c in range(0x20, 0x7F)
        ) | {"\n", "\t", "<", "["}
    out = [f.feed(p) for p in parts]
    out.append(f.finish())
    return out


def main() -> int:
    mismatches = 0
    total = 0
    for case in CASES:
        for n in (1, 2, 3, 5):
            for offset in (0, 1):
                parts = [case[i::n] for i in range(n)][offset:] or [case]
                total += 1
                a = run(parts, guard=True)
                b = run(parts, guard=False)
                if a != b:
                    mismatches += 1
                    print("MISMATCH", repr(case), "n=", n, "offset=", offset)
                    print("  guard:", a)
                    print("  ref  :", b)
    print(f"{total} chunkings compared, {mismatches} mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
