# SPDX-License-Identifier: Apache-2.0
"""Tests for the SpecPrefill static-prefix boundary in both chat engines."""

import re
from types import SimpleNamespace

import pytest

from omlx.engine.batched import BatchedEngine
from omlx.engine.vlm import VLMBatchedEngine

_TOKEN_RE = re.compile(r"\s+|\w+|[^\s\w]")


class FakeTokenizer:
    """Word-level tokenizer with a Qwen-like template.

    Tool instructions and system text share one leading block, and a default
    system block is emitted when the caller supplies no system message.
    """

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def encode(self, text, **kwargs):
        return [
            self._vocab.setdefault(p, len(self._vocab)) for p in _TOKEN_RE.findall(text)
        ]

    def static_prefix(self, messages, tools):
        leading = []
        for m in messages:
            if m["role"] not in ("system", "developer"):
                break
            leading.append(m["content"])
        out = ["<block>"]
        for tool in tools or []:
            out.append(f"use {tool['function']['name']} carefully;")
        out.extend(leading or ["default reasoning instructions go here"])
        out.append("</block>")
        return " ".join(out)

    def apply_chat_template(self, messages, tools=None, **kwargs):
        parts = [self.static_prefix(messages, tools)]
        started = False
        for m in messages:
            started = started or m["role"] not in ("system", "developer")
            if started:
                parts.append(f"<{m['role']}> {m['content']} </{m['role']}>")
        parts.append("<assistant>")
        return " ".join(parts)


SYS = {"role": "system", "content": "Operator policy: never modify production."}
DEV = {"role": "developer", "content": "Follow the repository conventions."}
USER = {"role": "user", "content": "please refactor the widget loader"}


def _tools(n):
    return [
        {"type": "function", "function": {"name": f"tool_{i}", "parameters": {}}}
        for i in range(n)
    ] or None


def _boundaries(tok, messages, tools):
    """Return (batched_end, vlm_end, prompt_ids) for one request."""
    prompt = tok.apply_chat_template(messages, tools=tools)
    ends = []
    for cls, rendered in (
        (BatchedEngine, prompt),
        (VLMBatchedEngine, tok.encode(prompt)),
    ):
        engine = cls(model_name="m")
        engine._model_settings = SimpleNamespace(specprefill_enabled=True)
        engine._tokenizer = tok
        engine._enable_thinking = None
        kwargs = {}
        engine._inject_specprefill_system_end(messages, rendered, tools, None, kwargs)
        ends.append(kwargs.get("specprefill_system_end", 0))
    return ends[0], ends[1], tok.encode(prompt)


@pytest.mark.parametrize("n_tools", [0, 2])
def test_boundary_covers_system_and_tool_block(n_tools):
    # The template's default system block made the old subtraction under-report.
    tok = FakeTokenizer()
    tools = _tools(n_tools)
    batched, vlm, prompt_ids = _boundaries(tok, [SYS, USER], tools)
    static_len = len(tok.encode(tok.static_prefix([SYS, USER], tools)))
    content_start = prompt_ids.index(tok.encode(USER["content"])[0])
    for end in (batched, vlm):
        assert static_len <= end <= content_start


def test_engines_agree_with_system_developer_and_tools():
    tok = FakeTokenizer()
    batched, vlm, _ = _boundaries(tok, [SYS, DEV, USER], _tools(2))
    assert batched == vlm > 0


def test_render_error_leaves_no_boundary():
    tok = FakeTokenizer()

    def broken(*args, **kwargs):
        raise RuntimeError("template refused")

    tok.apply_chat_template = broken
    for cls in (BatchedEngine, VLMBatchedEngine):
        engine = cls(model_name="m")
        engine._model_settings = SimpleNamespace(specprefill_enabled=True)
        engine._tokenizer = tok
        engine._enable_thinking = None
        kwargs = {}
        engine._inject_specprefill_system_end(
            [SYS, USER], [1, 2, 3], None, None, kwargs
        )
        assert "specprefill_system_end" not in kwargs


def test_mid_conversation_system_message_does_not_over_protect():
    tok = FakeTokenizer()
    messages = [
        SYS,
        USER,
        {"role": "system", "content": "Mid-conversation policy update."},
        {"role": "user", "content": "now regenerate it"},
    ]
    batched, vlm, prompt_ids = _boundaries(tok, messages, None)
    content_start = prompt_ids.index(tok.encode(USER["content"])[0])
    assert batched <= content_start
    assert vlm <= content_start
