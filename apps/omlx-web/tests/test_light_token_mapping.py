# SPDX-License-Identifier: Apache-2.0
"""Light mode draws the neutral Tailwind utilities through the theme tokens."""

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "omlx_web/static/css"
DASHBOARD_CSS = (STATIC / "dashboard.css").read_text(encoding="utf-8")
THEME_CSS = (STATIC / "theme.css").read_text(encoding="utf-8")

ROUTED = {
    ".bg-white": "--bg-primary",
    ".bg-neutral-50": "--bg-secondary",
    ".bg-neutral-100": "--bg-tertiary",
    ".text-neutral-900": "--text-primary",
    ".text-neutral-700": "--text-secondary",
    ".text-neutral-500": "--text-tertiary",
    ".text-neutral-400": "--text-muted",
    ".border-neutral-200": "--border-faint",
    ".border-neutral-300": "--border-normal",
}


def _unscoped_rule(selector):
    pattern = rf"(?m)^\s*{re.escape(selector)}[,\s][^{{]*\{{([^}}]*)\}}"
    match = re.search(pattern, DASHBOARD_CSS)
    assert match, f"no theme-independent rule for {selector}"
    return match.group(1)


def test_neutral_utilities_use_the_tokens_in_both_themes():
    for selector, token in ROUTED.items():
        assert f"var({token})" in _unscoped_rule(selector), selector


def test_routing_keeps_variant_classes_winning():
    # hover:/focus: variants beat a plain class only without !important.
    for selector in ROUTED:
        assert "!important" not in _unscoped_rule(selector), selector


def test_light_tokens_match_the_neutral_palette():
    light = THEME_CSS[THEME_CSS.index(":root {") : THEME_CSS.index('[data-theme="dark"] {')]
    for token, step in {
        "--text-primary": "neutral-900",
        "--text-secondary": "neutral-600",
        "--text-tertiary": "neutral-500",
        "--text-muted": "neutral-500",
        "--border-faint": "neutral-200",
        "--border-normal": "neutral-300",
    }.items():
        assert f"{token}: rgb(var(--palette-{step}));" in light, token
