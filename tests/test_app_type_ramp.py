# SPDX-License-Identifier: Apache-2.0
"""The app's type ramp is seven named steps, declared once in `Theme.swift`.

The steps sit next to the rest of the app's own design tokens, with no
cross-end source behind them. A step's value is part of the design: changing
it resizes every screen that names it, so the values are pinned here.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THEME = ROOT / "apps" / "omlx-mac" / "Sources" / "Theme" / "Theme.swift"

RAMP = {
    "aux": 12,
    "body": 14,
    "emphasis": 16,
    "section": 18,
    "page": 24,
    "kpi": 32,
    "display": 48,
}


def _font_size_block() -> str:
    """The `enum FontSize` inside `DesignTokens`, as source text."""
    theme = THEME.read_text(encoding="utf-8")
    match = re.search(
        r"enum DesignTokens \{.*?enum FontSize \{(.*?)\n    \}", theme, re.S
    )
    assert match, "Theme.swift no longer declares DesignTokens.FontSize"
    return match.group(1)


def _declared(name: str) -> float:
    block = _font_size_block()
    match = re.search(rf"static let {name}: CGFloat = (\d+(?:\.\d+)?)", block)
    assert match, f"DesignTokens.FontSize.{name} is not declared"
    return float(match.group(1))


def test_every_step_keeps_its_value():
    """A step changes here on purpose or not at all: every screen that names it
    moves with it."""
    for name, value in RAMP.items():
        declared = _declared(name)
        assert (
            declared == value
        ), f"DesignTokens.FontSize.{name} is {declared:g}pt, expected {value}pt"


def test_no_step_is_absent_and_none_is_added():
    """The scale has seven steps and no others: an eighth cannot be typed in
    without also being added here."""
    block = _font_size_block()
    declared = set(re.findall(r"static let (\w+): CGFloat =", block))
    assert declared == set(RAMP), f"unexpected names in the type ramp: {sorted(declared)}"


def test_the_ramp_is_declared_in_exactly_one_place():
    """Track A of #4400: the two ends stay independent, with no cross-end
    specification file. Nothing outside `apps/omlx-mac/` may supply a size the
    app renders, and there must be no second definition of a step to choose
    between."""
    sources = sorted((ROOT / "apps" / "omlx-mac" / "Sources").rglob("*.swift"))
    for name, value in RAMP.items():
        declaration = f"static let {name}: CGFloat = {value}"
        owners = [
            str(path.relative_to(ROOT))
            for path in sources
            if declaration in path.read_text(encoding="utf-8")
        ]
        assert len(owners) == 1, f"{name} ({value}pt) is declared {len(owners)} times: {owners}"
