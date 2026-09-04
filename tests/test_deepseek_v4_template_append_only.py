# SPDX-License-Identifier: Apache-2.0
"""Append-only rendering for the DeepSeek V4 DSML chat template.

With drop_thinking=False (reasoning retained), the rendered transcript must
be append-only across new user turns so prefix caches stay valid: a turn's
rendering never changes once it is historical. Default rendering
(drop_thinking=True) must be byte-identical to the previous behavior.
"""
import pytest

from omlx.patches.deepseek_v4 import chat_template_v4 as tmpl

SYSTEM = {"role": "system", "content": "You are a coding agent."}
U1 = {"role": "user", "content": "Refactor the parser module."}
A1 = {
    "role": "assistant",
    "content": "Done: extracted tokenize().",
    "reasoning_content": "Plan the split, then extract.",
}
U2 = {"role": "user", "content": "Now add tests."}
A_TOOL = {
    "role": "assistant",
    "content": "",
    "reasoning_content": "Need to inspect the file.",
    "tool_calls": [
        {"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
    ],
}
TOOL = {"role": "tool", "content": "file contents"}


def render(messages, **kwargs):
    return tmpl.encode_messages(messages, thinking_mode="thinking", **kwargs)


class TestAppendOnlyRendering:
    def test_new_user_turn_is_append_only_when_reasoning_retained(self):
        turn1 = render([SYSTEM, U1], drop_thinking=False)
        turn2 = render([SYSTEM, U1, A1, U2], drop_thinking=False)
        assert turn2.startswith(turn1)

    def test_tool_loop_then_user_turn_is_append_only(self):
        loop = render([SYSTEM, U1, A_TOOL, TOOL], drop_thinking=False)
        follow = render([SYSTEM, U1, A_TOOL, TOOL, A1, U2], drop_thinking=False)
        assert follow.startswith(loop)

    def test_historical_assistant_reasoning_is_rendered_when_retained(self):
        out = render([SYSTEM, U1, A1, U2], drop_thinking=False)
        assert A1["reasoning_content"] in out

    def test_default_rendering_flips_previous_user_marker(self):
        # Documents the existing default behavior this feature works around:
        # with drop_thinking=True the previous user suffix flips to </think>,
        # so the rendering is NOT append-only.
        turn1 = render([SYSTEM, U1])
        turn2 = render([SYSTEM, U1, A1, U2])
        assert not turn2.startswith(turn1)

    def test_tool_adjacent_reminder_keeps_render_append_only(self):
        # Claude Code appends a role=system reminder after a tool result
        # (tool -> system -> assistant after adapter splitting). Relocation
        # must keep the rendered prefix byte-stable when the next request
        # appends another reminder at the tail.
        reminder = {"role": "system", "content": "Task reminder"}
        turn1 = [SYSTEM, U1, A_TOOL, TOOL, reminder]
        turn2 = [SYSTEM, U1, A_TOOL, TOOL, reminder, A1, U2, reminder]

        r1 = render(tmpl.relocate_mid_system_messages(turn1), drop_thinking=False)
        r2 = render(tmpl.relocate_mid_system_messages(turn2), drop_thinking=False)
        assert r2.startswith(r1)
        assert r1.count("<｜latest_reminder｜>") == 1
        assert r2.count("<｜latest_reminder｜>") == 2

    @pytest.mark.parametrize("mode", ["thinking", "chat"])
    def test_default_rendering_unchanged_across_cases(self, mode):
        # drop_thinking=True is the default; passing it explicitly must be
        # byte-identical to omitting it for every case/mode combination.
        cases = [
            [SYSTEM, U1],
            [SYSTEM, U1, A1, U2],
            [SYSTEM, U1, A_TOOL, TOOL],
            [SYSTEM, U1, A_TOOL, TOOL, A1, U2],
            [U1, A1, U2],
        ]
        for msgs in cases:
            assert tmpl.encode_messages(
                msgs, thinking_mode=mode
            ) == tmpl.encode_messages(msgs, thinking_mode=mode, drop_thinking=True)
