# SPDX-License-Identifier: Apache-2.0
"""Text-extraction helpers in anthropic_utils.py.

_extract_system_text() and the shared _text_from_content_blocks() helper both
walk content blocks pulling .text off an object-or-dict (#8). These tests pin
the existing behavior of _extract_system_text() before factoring the shared
loop out, then cover the shared helper with both example-based and
generative (hypothesis) tests (#5).
"""

from hypothesis import given
from hypothesis import strategies as st

from omlx.api.anthropic_models import ContentBlockText, ContentBlockToolUse, SystemContent
from omlx.api.anthropic_utils import _extract_system_text, _text_from_content_blocks

# ---------------------------------------------------------------------------
# _extract_system_text: pin existing behavior before refactoring (#8)
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


# ---------------------------------------------------------------------------
# _text_from_content_blocks: the shared helper every call site now uses.
# Generative coverage (#5) — property-test-gap-finder's original suggestions.
# ---------------------------------------------------------------------------

_text_block = st.builds(
    lambda text: ContentBlockText(type="text", text=text), text=st.text()
)
_tool_use_block = st.builds(
    lambda: ContentBlockToolUse(type="tool_use", id="t1", name="Bash", input={})
)
_any_block = st.one_of(_text_block, _tool_use_block)


@given(st.lists(_any_block, max_size=10))
def test_text_from_content_blocks_only_ever_returns_declared_text_block_contents(blocks):
    """Whatever comes back is built only from blocks whose type == 'text'."""
    result = _text_from_content_blocks(blocks)
    expected = [b.text for b in blocks if isinstance(b, ContentBlockText)]
    assert result == expected


def test_text_from_content_blocks_skips_non_text_blocks():
    blocks = [
        ContentBlockText(type="text", text="see this "),
        ContentBlockToolUse(type="tool_use", id="t1", name="Bash", input={}),
        ContentBlockText(type="text", text="tool"),
    ]
    assert _text_from_content_blocks(blocks) == ["see this ", "tool"]


def test_text_from_content_blocks_returns_empty_for_no_blocks():
    assert _text_from_content_blocks([]) == []
