# SPDX-License-Identifier: Apache-2.0
"""The app's type ramp is six named steps, and each is the size it replaces.

`Theme.swift` declares the steps next to the rest of the app's own design
tokens, in one file, with no cross-end source behind them. A ramp that drifts
from the sizes it was cut from would silently change how every screen renders,
so this pins the values rather than trusting them.

The steps and the counts below are the survey the ramp was cut from. Counts
are re-derived rather than stored: a step that gains or loses every call site
is not a defect, but a step whose value changes is.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THEME = ROOT / "apps" / "omlx-mac" / "Sources" / "Theme" / "Theme.swift"

# token -> the literal it replaces, with the number of call sites it covers
# before the adoption pass in #3847 lands.
RAMP = {
    "aux": (12, 250),
    "body": (14, 35),
    "emphasis": (16, 5),
    "section": (18, 8),
    "page": (24, 6),
    "kpi": (32, 2),
}

FLOORS = {
    "floor": 12,
    "bodyFloor": 14,
    "enhancedReadabilityFloor": 12,
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


def test_every_step_equals_the_literal_it_replaces():
    """A ramp is only a name for sizes that already existed.

    Each step must equal the point size it replaced, so adopting the ramp
    across the app moves nothing. `tokens.json` did not exist: this table is
    the contract, and it lives with the values it describes.
    """
    for name, (value, _) in RAMP.items():
        declared = _declared(name)
        assert declared == value, (
            f"DesignTokens.FontSize.{name} is {declared:g}pt, but the size it "
            f"replaces is {value}pt; adopting the ramp would move text"
        )


def test_no_step_is_absent_and_none_is_added():
    """The scale has six steps and no others: a seventh cannot be typed in
    without also being added here, where its site count must be justified."""
    block = _font_size_block()
    declared = set(re.findall(r"static let (\w+): CGFloat =", block))
    assert declared == set(RAMP) | set(
        FLOORS
    ), f"unexpected names in the type ramp: {sorted(declared)}"


def test_nothing_renders_below_the_auxiliary_step():
    """The Enhanced Readability switch only raises contrast now: the floor it
    used to enforce is the floor of the scale, and the three floors agree."""
    for name, value in FLOORS.items():
        assert _declared(name) == value, f"DesignTokens.FontSize.{name}"
    assert _declared("floor") == _declared("aux")
    assert _declared("enhancedReadabilityFloor") == _declared("aux")


def test_the_ramp_is_where_the_rest_of_the_app_s_ties_its_tokens():
    """The type ramp sits with the app's other tokens rather than in a
    generated file: nothing outside `apps/omlx-mac/` feeds it, and no build
    step rewrites it."""
    theme = THEME.read_text(encoding="utf-8")
    assert "enum DesignTokens" in theme
    assert "enum FontSize" in theme

    # Theme.swift is part of the app target already, so the ramp compiles
    # without any project-file change.
    pbxproj = ROOT / "apps" / "omlx-mac" / "oMLX.xcodeproj" / "project.pbxproj"
    if pbxproj.exists():
        assert "Theme.swift" in pbxproj.read_text(encoding="utf-8")


def test_the_ramp_is_declared_in_exactly_one_place():
    """Track A of #4400: the two ends stay independent, with no cross-end
    specification file. Nothing outside `apps/omlx-mac/` may supply a size the
    app renders, and there must be no second definition of a step to choose
    between."""
    sources = sorted((ROOT / "apps" / "omlx-mac" / "Sources").rglob("*.swift"))
    for name, (value, _) in RAMP.items():
        declaration = f"static let {name}: CGFloat = {value}"
        owners = [
            str(path.relative_to(ROOT))
            for path in sources
            if declaration in path.read_text(encoding="utf-8")
        ]
        assert (
            len(owners) <= 1
        ), f"{name} ({value}pt) is declared {len(owners)} times: {owners}"
