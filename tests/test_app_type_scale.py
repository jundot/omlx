# SPDX-License-Identifier: Apache-2.0
"""Every text size in the app names one of the six ramp steps.

The app draws its own text through `DesignTokens.FontSize`, declared in
`Theme.swift` by #4082: the guard below walks every call site, so a size
cannot be typed back into a view and drift from the ramp. No view may type a
size below the auxiliary floor, and the file declaring the ramp has to be part
of the app target for any of it to compile.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THEME = ROOT / "apps" / "omlx-mac" / "Sources" / "Theme" / "Theme.swift"
APP_FONT_CALL = re.compile(r"(?:omlxText|omlxMono|omlxDisplay)\(\s*(\d+(?:\.\d+)?)")
# The whole first argument, not just a literal right after the paren: a
# ternary or a computed size must not slip past the guard.
APP_FONT_CALL_ANY = re.compile(r"\.omlx(?:Text|Mono|Display)\(([^,)]*)")

STEPS = {"aux", "body", "emphasis", "section", "page", "kpi"}
FLOORS = {"floor", "bodyFloor", "enhancedReadabilityFloor"}


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


def _floor() -> float:
    return _declared()["floor"]


def _swift_sources() -> list:
    return sorted((ROOT / "apps" / "omlx-mac" / "Sources").rglob("*.swift"))


def test_every_text_size_names_a_step_of_the_ramp():
    """No `.omlx*()` call types a number: each names a `DesignTokens.FontSize`
    value, so a size cannot drift back into a view and away from the ramp."""
    declared = _declared()
    assert set(declared) == STEPS | FLOORS, sorted(declared)

    literals: list = []
    used: set = set()
    for path in _swift_sources():
        text = path.read_text(encoding="utf-8")
        for match in APP_FONT_CALL_ANY.finditer(text):
            if re.search(r"\d", match.group(1)):
                literals.append(f"{path.name}: {match.group(0)}")
        used.update(re.findall(r"DesignTokens\.FontSize\.(\w+)", text))

    assert not literals, f"a view types its own text size: {literals[:5]}"
    unknown = sorted(name for name in used if name not in set(declared))
    assert not unknown, f"DesignTokens.FontSize has no such value: {unknown}"
    assert used, "the app stopped drawing with the ramp"


def test_app_text_respects_the_floor():
    """Nothing renders below the auxiliary step, so the Enhanced Readability
    switch only raises contrast: the floor it used to enforce is the floor of
    the scale itself."""
    floor = _floor()
    offenders = {}
    for path in _swift_sources():
        for value in APP_FONT_CALL.findall(path.read_text(encoding="utf-8")):
            if float(value) < floor:
                offenders.setdefault(str(path.relative_to(ROOT)), set()).add(value)
    assert not offenders, f"app text below the {floor:g}pt floor: {offenders}"
    assert _declared()["enhancedReadabilityFloor"] == floor
    assert _declared()["aux"] == floor


def test_the_ramp_is_compiled_by_the_app():
    """The ramp lives in `Theme.swift`, which the project file already lists
    four times: a PBXBuildFile, a PBXFileReference, a group child and a
    Sources-phase entry. #4082 adds no source file, so this stays put."""
    pbxproj = (
        ROOT / "apps" / "omlx-mac" / "oMLX.xcodeproj" / "project.pbxproj"
    ).read_text(encoding="utf-8")
    assert "Theme.swift" in pbxproj
    assert pbxproj.count("Theme.swift") >= 4, "Theme.swift is no longer in the target"
