# SPDX-License-Identifier: Apache-2.0
"""Text-extraction helpers in anthropic_utils.py.

``_extract_system_text()`` and the shared ``_text_from_content_blocks()``
helper both walk content blocks and pull ``text`` off an object or a dict.
These tests pin the behavior of ``_extract_system_text()`` before the shared
loop is factored out, then cover the shared helper.
"""

import pytest

from omlx.api.anthropic_models import (
    ContentBlockText,
    ContentBlockToolUse,
    SystemContent,
)
from omlx.api.anthropic_utils import (
    _BILLING_HEADER_PREFIX,
    _extract_system_text,
    _extract_tool_result_content,
    _text_from_content_blocks,
)

# ---------------------------------------------------------------------------
# _extract_system_text
# ---------------------------------------------------------------------------


def test_extract_system_text_from_a_plain_string():
    assert _extract_system_text("You are a helpful assistant.") == (
        "You are a helpful assistant."
    )


def test_extract_system_text_from_system_content_objects():
    system = [
        SystemContent(type="text", text="part one"),
        SystemContent(type="text", text="part two"),
    ]
    assert _extract_system_text(system) == "part one\npart two"


def test_extract_system_text_from_raw_dicts():
    system = [
        {"type": "text", "text": "part one"},
        {"type": "text", "text": "part two"},
    ]
    assert _extract_system_text(system) == "part one\npart two"


def test_extract_system_text_skips_non_text_dicts():
    system = [{"type": "text", "text": "kept"}, {"type": "other", "text": "dropped"}]
    assert _extract_system_text(system) == "kept"


def test_extract_system_text_skips_billing_header_blocks():
    system = [
        SystemContent(type="text", text="x-anthropic-billing-header: abc123"),
        SystemContent(type="text", text="real system prompt"),
    ]
    assert _extract_system_text(system) == "real system prompt"


def test_extract_system_text_strips_client_budget_markers():
    text = "System prompt.\n<total_tokens>500 tokens left</total_tokens>"
    assert _extract_system_text(text) == "System prompt."


def test_extract_system_text_strips_budget_markers_after_joining_blocks():
    system = [
        SystemContent(type="text", text="part one"),
        SystemContent(
            type="text", text="part two\n<total_tokens>10 tokens left</total_tokens>"
        ),
    ]
    assert "total_tokens" not in _extract_system_text(system)


def test_extract_system_text_returns_empty_for_neither_str_nor_list():
    assert _extract_system_text(None) == ""  # type: ignore[arg-type]


# The "<total_tokens>" substring is absent on purpose: it keeps
# _strip_client_budget_markers a no-op, so the case below isolates the
# billing-header filter instead of reimplementing the marker regex.
_BILLING_CASES = [
    (["real system prompt"], "real system prompt"),
    (
        [_BILLING_HEADER_PREFIX + "abc123", "real system prompt"],
        "real system prompt",
    ),
    (
        ["a", _BILLING_HEADER_PREFIX + "abc123", "b"],
        "a\nb",
    ),
    (
        [_BILLING_HEADER_PREFIX + "one", _BILLING_HEADER_PREFIX + "two"],
        "",
    ),
]


@pytest.mark.parametrize(("texts", "expected"), _BILLING_CASES)
def test_extract_system_text_drops_billing_header_blocks(texts, expected):
    """Every non-billing-header block survives, joined in order; every
    billing-header block never appears in the joined result."""
    system = [SystemContent(type="text", text=text) for text in texts]
    assert _extract_system_text(system) == expected


# ---------------------------------------------------------------------------
# _text_from_content_blocks: the shared helper every call site uses.
# ---------------------------------------------------------------------------

_TEXT_BLOCK_CASES = [
    ([], []),
    ([ContentBlockText(type="text", text="only")], ["only"]),
    (
        [
            ContentBlockText(type="text", text="see this "),
            ContentBlockToolUse(type="tool_use", id="t1", name="Bash", input={}),
            ContentBlockText(type="text", text="tool"),
        ],
        ["see this ", "tool"],
    ),
    ([{"type": "text", "text": "from a dict"}], ["from a dict"]),
    ([{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}], []),
    # A non-text block must not contribute, even when it carries a text key.
    ([{"type": "other", "text": "dropped"}], []),
    # Empty text is returned, not filtered: callers decide whether to keep it.
    ([{"type": "text", "text": ""}], [""]),
    # A non-str text is skipped rather than reaching str.join().
    ([{"type": "text", "text": 123}], []),
    # An object the normalizer does not recognize is skipped, not crashed on.
    ([object()], []),
]


@pytest.mark.parametrize(("blocks", "expected"), _TEXT_BLOCK_CASES)
def test_text_from_content_blocks_returns_only_declared_text(blocks, expected):
    """Whatever comes back is built only from blocks whose type is "text"."""
    assert _text_from_content_blocks(blocks) == expected


# ---------------------------------------------------------------------------
# _extract_tool_result_content: the same guard, at the call site.
# ---------------------------------------------------------------------------

_TOOL_RESULT_CASES = [
    ("plain string", "hello", "hello"),
    ("list of strings", ["a", "b"], "a\nb"),
    ("list of text blocks", [{"type": "text", "text": "a"}], "a"),
    ("mixed list", ["a", {"type": "text", "text": "b"}], "a\nb"),
    ("non-str text in a list", [{"type": "text", "text": 123}], ""),
    ("text dict", {"type": "text", "text": "hi"}, "hi"),
    ("non-str text dict", {"type": "text", "text": 123}, "123"),
    ("non-text dict", {"x": 1}, '{"x": 1}'),
]


@pytest.mark.parametrize(("label", "content", "expected"), _TOOL_RESULT_CASES)
def test_extract_tool_result_content_always_returns_text(label, content, expected):
    """The result is a str for every accepted shape, never a TypeError."""
    assert _extract_tool_result_content(content) == expected
