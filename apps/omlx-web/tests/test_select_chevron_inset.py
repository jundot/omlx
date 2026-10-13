# SPDX-License-Identifier: Apache-2.0
"""Native ``<select>`` controls must not show the OS arrow at the edge.

On Safari the native dropdown arrow is pinned to the right border of the
control, which reads as cramped against the edge. The global rule in
``theme.css`` drops the native arrow (``appearance: none``) and paints a
chevron inset from the right instead. Selects that already render their
own chevron (``.select-chevron-custom``) are excluded so they never get
a double arrow, and the dark theme swaps the chevron stroke colour.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THEME_CSS = ROOT / "omlx_web/static/css/theme.css"
CHAT_TEMPLATE = ROOT / "omlx_web/templates/chat.html"

LIGHT_RE = re.compile(
    r"^select:not\(\[multiple\]\):not\(\[size\]\):not\(\.select-chevron-custom\)\s*{[^}]*}",
    re.MULTILINE,
)
DARK_RE = re.compile(
    r'\[data-theme="dark"\]\s+select:not\(\[multiple\]\):not\(\[size\]\)'
    r":not\(\.select-chevron-custom\)\s*{[^}]*}"
)


def _css() -> str:
    return THEME_CSS.read_text(encoding="utf-8")


def test_light_rule_drops_native_arrow_and_insets_chevron():
    (rule,) = LIGHT_RE.findall(_css())
    assert "appearance: none" in rule
    assert "background-image: url(" in rule
    assert "background-position: right 0.75rem center" in rule
    assert "padding-right: 2.25rem" in rule


def test_dark_rule_recolours_chevron():
    (rule,) = DARK_RE.findall(_css())
    assert "background-image: url(" in rule
    assert "%23a1a1aa" in rule, "dark chevron should use the lighter stroke"


def test_custom_chevron_selects_are_skipped():
    # Both rules must carry the .select-chevron-custom exclusion so
    # selects with their own lucide chevron never get a double arrow.
    assert len(LIGHT_RE.findall(_css())) == 1
    assert len(DARK_RE.findall(_css())) == 1


def test_chat_profile_select_marked_custom_chevron():
    html = CHAT_TEMPLATE.read_text(encoding="utf-8")
    assert "select-chevron-custom" in html, (
        "chat prompt-profile select renders its own lucide chevron and "
        "must opt out of the global rule"
    )


def test_chat_theme_select_does_not_wipe_chevron():
    # An inline `background:` shorthand would erase the chevron
    # background-image; the theme select must use background-color.
    html = CHAT_TEMPLATE.read_text(encoding="utf-8")
    theme_select = re.search(
        r"<select[^>]*setTheme\([^)]*\)[^>]*>", html, re.DOTALL
    )
    assert theme_select, "theme select not found"
    style = re.search(r'style="([^"]*)"', theme_select.group(0))
    assert style and "background:" not in style.group(1).replace("background-color:", "")
