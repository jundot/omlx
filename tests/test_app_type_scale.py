# SPDX-License-Identifier: Apache-2.0
"""Every text size in the app names one of the seven ramp steps.

The app draws its own text through `DesignTokens.FontSize`, declared in
`Theme.swift` by #4082: the guard below walks every call site, so a size
cannot be typed back into a view and drift from the ramp.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THEME = ROOT / "apps" / "omlx-mac" / "Sources" / "Theme" / "Theme.swift"
# The whole first argument, not just a literal right after the paren: a
# ternary or a computed size must not slip past the guard.
APP_FONT_CALL_ANY = re.compile(r"\.omlx(?:Text|Mono|Display)\(([^,)]*)")

STEPS = {"aux", "body", "emphasis", "section", "page", "kpi", "display"}


def _theme() -> str:
    return THEME.read_text(encoding="utf-8")


def _font_size_block() -> str:
    match = re.search(
        r"enum DesignTokens \{.*?enum FontSize \{(.*?)\n    \}", _theme(), re.S
    )
    assert match, "Theme.swift no longer declares DesignTokens.FontSize"
    return match.group(1)


def _declared() -> dict:
    return {
        name: float(value)
        for name, value in re.findall(
            r"static let (\w+): CGFloat = (\d+(?:\.\d+)?)", _font_size_block()
        )
    }


def _swift_sources() -> list:
    return sorted((ROOT / "apps" / "omlx-mac" / "Sources").rglob("*.swift"))


def test_every_text_size_names_a_step_of_the_ramp():
    """No `.omlx*()` call types a number: each names a `DesignTokens.FontSize`
    value, so a size cannot drift back into a view and away from the ramp."""
    declared = _declared()
    assert set(declared) == STEPS, sorted(declared)

    literals: list = []
    used: set = set()
    for path in _swift_sources():
        text = path.read_text(encoding="utf-8")
        for match in APP_FONT_CALL_ANY.finditer(text):
            first = match.group(1)
            if re.search(r"\d", first) or "DesignTokens.FontSize." not in first:
                literals.append(f"{path.name}: {match.group(0)}")
        used.update(re.findall(r"DesignTokens\.FontSize\.(\w+)", text))

    assert not literals, f"a view types its own text size: {literals[:5]}"
    unknown = sorted(name for name in used if name not in set(declared))
    assert not unknown, f"DesignTokens.FontSize has no such value: {unknown}"
    assert used == STEPS, f"the app does not name every step: {sorted(STEPS - used)}"


