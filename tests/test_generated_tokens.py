# SPDX-License-Identifier: Apache-2.0
"""tokens.json is the single source; both generated artifacts must match it."""

import subprocess
import sys
from pathlib import Path


def test_generated_artifacts_match_the_token_file():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "omlx" / "admin" / "build_tokens.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_text_grey_ramp_is_three_steps():
    """The console's text greys are primary/secondary/tertiary (muted joined
    tertiary). Nothing the console ships may still name the deleted token, and
    every surviving grey clears 4.5:1 on its page surface."""
    import json

    root = Path(__file__).resolve().parents[1]
    tokens = json.loads(
        (root / "omlx" / "admin" / "tokens.json").read_text(encoding="utf-8")
    )
    for appearance in ("light", "dark"):
        assert set(tokens["text"][appearance]) == {
            "primary",
            "secondary",
            "tertiary",
            "danger",
            "link",
        }, sorted(tokens["text"][appearance])

    def _luminance(hex_colour):
        value = hex_colour.lstrip("#")
        channels = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    def _ratio(a, b):
        la, lb = _luminance(a), _luminance(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)

    for appearance in ("light", "dark"):
        page = tokens["surface"][appearance]["bgPrimary"]
        for step in ("primary", "secondary", "tertiary"):
            ratio = _ratio(tokens["text"][appearance][step], page)
            assert ratio >= 4.5, f"{appearance} text.{step}: {ratio:.2f}:1 on {page}"

    for path in sorted((root / "omlx" / "admin").rglob("*")):
        if path.is_file() and path.suffix in {".html", ".css", ".js", ".json", ".py"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "text-muted" not in text, path
            assert "text-fg-muted" not in text, path


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


def test_the_layout_width_ids_are_pinned_on_both_sides():
    """One contract in three places: the console's measure set in
    ``tokens.json``, ``dashboard_layout.js``'s ``WIDTH_CLASSES`` and the
    request model's ``Literal``. The console offers these ids and the server
    accepts exactly them."""
    import json
    import re

    root = Path(__file__).resolve().parents[1]
    tokens = json.loads(
        (root / "omlx" / "admin" / "tokens.json").read_text(encoding="utf-8")
    )
    token_ids = list(tokens["layout"]["widths"])

    layout_js = (
        root / "omlx" / "admin" / "static" / "js" / "dashboard_layout.js"
    ).read_text(encoding="utf-8")
    # The console expresses a measure as a Tailwind class in the earlier slices
    # and as a token in the console slice; either way `WIDTH_IDS` is the set
    # that is offered, so the test reads the object that list is built from.
    source = re.search(
        r"WIDTH_IDS\s*=\s*Object\.keys\(\s*(\w+)\s*\)", layout_js
    )
    assert source, "the console stopped deriving its width ids from one list"
    measures = re.search(
        rf"{re.escape(source.group(1))}\s*=\s*\{{(.*?)\}}", layout_js, re.S
    )
    assert measures, f"the console's {source.group(1)} list is gone"
    js_ids = re.findall(r"^\s*(\w+):", measures.group(1), re.M)

    routes = (root / "omlx" / "admin" / "routes.py").read_text(encoding="utf-8")
    literal = re.search(r"width:\s*Literal\[([^\]]*)\]", routes)
    assert literal, "the server stopped validating the measure id"
    route_ids = re.findall(r'"([^"]+)"', literal.group(1))

    assert token_ids == js_ids == route_ids, (token_ids, js_ids, route_ids)
