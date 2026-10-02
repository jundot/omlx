# SPDX-License-Identifier: MIT
"""Client reasoning_effort values must not crash the chat template.

Since #2675 the server passes the client's free-form ``reasoning_effort``
through to ``apply_chat_template``, but the DeepSeek V4 encoder only knows
``low`` / ``high`` / ``max`` and asserted on anything else — a 400 that
surfaces in agent clients as an opaque provider failure (observed with
Hermes sending ``xhigh``). Aliases map onto the nearest supported level;
unknown values fall back to the default with a warning.
"""

import pytest

from omlx.patches.deepseek_v4.chat_template_v4 import (
    DEFAULT_REASONING_EFFORT,
    apply_chat_template,
    normalize_reasoning_effort,
)

MESSAGES = [
    {"role": "system", "content": "You are an assistant."},
    {"role": "user", "content": "Hello."},
]


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, DEFAULT_REASONING_EFFORT),
        ("", DEFAULT_REASONING_EFFORT),
        ("low", "low"),
        ("minimal", "low"),
        ("none", "low"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "max"),
        ("max", "max"),
        ("MAX", "max"),
        ("  high  ", "high"),
    ],
)
def test_alias_mapping(value, expected):
    assert normalize_reasoning_effort(value) == expected


def test_unknown_value_falls_back_with_warning(caplog):
    with caplog.at_level("WARNING"):
        assert normalize_reasoning_effort("bogus") == DEFAULT_REASONING_EFFORT
    assert "bogus" in caplog.text


@pytest.mark.parametrize("value", ["xhigh", "medium", "minimal", "bogus"])
def test_template_accepts_client_vocabulary(value):
    prompt = apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True, reasoning_effort=value
    )
    assert isinstance(prompt, str) and prompt


@pytest.mark.parametrize("alias,level", [("xhigh", "max"), ("medium", "high")])
def test_alias_renders_identically_to_mapped_level(alias, level):
    assert apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True, reasoning_effort=alias
    ) == apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True, reasoning_effort=level
    )
