# SPDX-License-Identifier: Apache-2.0
"""Every text size in the app names one of the generated steps.

The app draws its own text with the six levels ``tokens.json`` declares: the
guard below walks every call site, so a size cannot be typed back into a view
and drift from the token file. The file is also checked to be part of the app
target, and no view may type a size below the auxiliary floor.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENS = json.loads(
    (ROOT / "omlx" / "admin" / "tokens.json").read_text(encoding="utf-8")
)
SWIFT = (
    ROOT / "apps" / "omlx-mac" / "Sources" / "Theme" / "DesignTokens.swift"
).read_text(encoding="utf-8")
SWIFT_SOURCES = sorted((ROOT / "apps" / "omlx-mac" / "Sources").rglob("*.swift"))
PBXPROJ = (
    ROOT / "apps" / "omlx-mac" / "oMLX.xcodeproj" / "project.pbxproj"
).read_text(encoding="utf-8")
FLOOR = TOKENS["typography"]["floor"]
APP_FONT_CALL = re.compile(r"(?:omlxText|omlxMono|omlxDisplay)\(\s*(\d+(?:\.\d+)?)")


def test_the_apps_text_sizes_come_from_the_token_file():
    """The app draws its own text with the six steps: every ``omlxText`` /
    ``omlxMono`` / ``omlxDisplay`` call names a ``DesignTokens.FontSize``
    value, so a size cannot be typed back into a view and drift from
    ``tokens.json``."""
    import json
    import re

    root = Path(__file__).resolve().parents[1]
    tokens = json.loads(
        (root / "omlx" / "admin" / "tokens.json").read_text(encoding="utf-8")
    )
    steps = set(tokens["typography"]["scale"])
    assert len(steps) == 6, sorted(steps)

    generated = (
        root / "apps" / "omlx-mac" / "Sources" / "Theme" / "DesignTokens.swift"
    ).read_text(encoding="utf-8")
    for name in sorted(steps):
        assert re.search(rf"static let {name}: CGFloat = \d", generated), name

    literals: list[str] = []
    used: set[str] = set()
    for path in sorted((root / "apps" / "omlx-mac" / "Sources").rglob("*.swift")):
        text = path.read_text(encoding="utf-8")
        # The whole first argument, not just a literal right after the paren:
        # a ternary or a computed size must not slip past the guard.
        for match in re.finditer(r"\.omlx(?:Text|Mono|Display)\(([^,)]*)", text):
            size_argument = match.group(1).strip()
            if re.search(r"\d", size_argument):
                literals.append(f"{path.name}: {match.group(0)}")
        used.update(re.findall(r"DesignTokens\.FontSize\.(\w+)", text))

    assert not literals, f"a view types its own text size: {literals[:5]}"
    allowed = steps | {"floor", "bodyFloor", "enhancedReadabilityFloor"}
    unknown = sorted(name for name in used if name not in allowed)
    assert not unknown, f"DesignTokens.FontSize has no such value: {unknown}"
    assert used, "the app stopped drawing with the token steps"

# === The floor, and the file being compiled ===


def test_app_text_respects_the_floor():
    offenders = {}
    for path in SWIFT_SOURCES:
        for value in APP_FONT_CALL.findall(path.read_text(encoding="utf-8")):
            if float(value) < FLOOR["aux"]:
                offenders.setdefault(str(path.relative_to(ROOT)), set()).add(value)
    assert not offenders, f"app text below the {FLOOR['aux']}pt floor: {offenders}"


def test_swift_tokens_are_compiled_by_the_app():
    assert "DesignTokens.swift" in PBXPROJ
    assert PBXPROJ.count("DesignTokens.swift") >= 4, (
        "a new Swift file needs a PBXBuildFile, a PBXFileReference, a group "
        "child and a Sources-phase entry"
    )
