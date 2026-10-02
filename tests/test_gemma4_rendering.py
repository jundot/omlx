# SPDX-License-Identifier: Apache-2.0
"""Integration tests: Gemma 4 chat-template rendering with real tokenizer.

Skipped when the Gemma 4 26B model is not present at MODEL_PATH.
"""

from __future__ import annotations

import glob
import os

import pytest

from omlx.adapter.gemma4 import extract_gemma4_messages
from omlx.api.openai_models import Message


def _find_gemma4_26b_model() -> str | None:
    pattern = os.path.join(
        os.path.expanduser("~"), ".omlx", "models", "gemma-4-26B-A4B-it*"
    )
    matches = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    return matches[0] if matches else None


MODEL_PATH = _find_gemma4_26b_model()

pytestmark = pytest.mark.skipif(
    MODEL_PATH is None, reason="No gemma-4-26B-A4B-it* model found in ~/.omlx/models/"
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

_TC = {
    "id": "c1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": "{}"},
}


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH)


def _render(messages, tools=None):
    tok = _load_tokenizer()
    return tok.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )


def _marker_counts(rendered: str) -> tuple[int, int]:
    return rendered.count("<|tool_call>"), rendered.count("<tool_call|>")


class TestGemma4TemplateRendering:
    def test_clean_history_renders_balanced(self):
        """Clean multi-turn tool call → balanced <|tool_call> / <tool_call|>."""
        openai_msgs = [
            Message(role="user", content="What's the weather?"),
            Message(role="assistant", content="", tool_calls=[_TC]),
            Message(role="tool", content="sunny", tool_call_id="c1"),
        ]
        processed = extract_gemma4_messages(openai_msgs)
        rendered = _render(processed, tools=_TOOLS)
        opens, closes = _marker_counts(rendered)
        assert opens == closes, f"imbalanced: opens={opens} closes={closes}"
        assert opens >= 1

    def test_stray_close_marker_in_content_causes_imbalance(self):
        """Stray <tool_call|> in assistant content renders an extra close token.

        This test documents the bug: when the client stores the stray marker
        verbatim and feeds it back without sanitisation, the template embeds
        it as a real special token, producing opens != closes.
        """
        raw_msgs = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "", "tool_calls": [_TC]},
            {
                "role": "assistant",
                "content": "",
                "tool_responses": [{"name": "get_weather", "response": "sunny"}],
            },
            {"role": "user", "content": "Thanks"},
            # The model generated only <tool_call|> on its next turn; the client
            # stored it verbatim.
            {"role": "assistant", "content": "<tool_call|>"},
        ]
        rendered = _render(raw_msgs, tools=_TOOLS)
        opens, closes = _marker_counts(rendered)
        assert opens != closes, (
            f"Expected imbalance but got opens={opens} closes={closes}. "
            "Bug may no longer reproduce with this model/template version."
        )

    def test_extract_gemma4_messages_fixes_imbalance(self):
        """extract_gemma4_messages strips the stray marker → balanced rendering."""
        openai_msgs = [
            Message(role="user", content="What's the weather?"),
            Message(role="assistant", content="", tool_calls=[_TC]),
            Message(role="tool", content="sunny", tool_call_id="c1"),
            Message(role="user", content="Thanks"),
            Message(role="assistant", content="<tool_call|>"),
        ]
        processed = extract_gemma4_messages(openai_msgs)
        rendered = _render(processed, tools=_TOOLS)
        opens, closes = _marker_counts(rendered)
        assert opens == closes, (
            f"Still imbalanced after fix: opens={opens} closes={closes}"
        )
