# SPDX-License-Identifier: Apache-2.0
"""Pin the danger text steps from docs/web-ui-design.md on the settings page.

The authentication card used three different reds for one warning block
(``red-500`` for the loopback note, ``red-600`` for the keyless hint,
``red-700`` for the keyless label). ``text-red-500`` measures 3.76:1 on white,
so 12px text in that colour is under the 4.5:1 text floor the theme editor and
WCAG AA both apply. The card now uses the documented ladder: ``red-700`` for
the label of a dangerous setting, ``red-600`` for its supporting text and for
inline errors, ``red-500`` for accents (hover, borders, rings, icons) only.
The same ladder applies to the other dashboard error strings, which this test
pins so a new page cannot silently fall back to ``red-500`` text.

These are static-template assertions, like test_admin_enhanced_readability.py:
no browser render and no server.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "omlx_web/templates/dashboard/_settings.html"
DESIGN = ROOT.parents[1] / "docs/web-ui-design.md"

HTML = SETTINGS.read_text(encoding="utf-8")


def _classes(pattern: str) -> str:
    match = re.search(pattern, HTML, re.S)
    assert match, f"markup not found: {pattern}"
    return match.group("classes")


def test_settings_page_does_not_use_red_500_as_text():
    """red-500 is an accent step; as text it fails the 4.5:1 floor."""
    offenders = [
        line.strip()
        for line in HTML.splitlines()
        if re.search(r"(?<!hover:)text-red-500", line)
    ]
    assert offenders == []


def test_auth_card_uses_the_documented_danger_steps():
    warning = _classes(
        r'<p class="(?P<classes>[^"]*)"[^>]*>'
        r"\s*\{\{ t\('settings\.auth\.skip_verification_warning'\) \}\}"
    )
    label = _classes(
        r'<label class="(?P<classes>[^"]*)">'
        r"\{\{ t\('settings\.auth\.keyless_inference'\) \}\}"
    )
    hint = _classes(
        r'<p class="(?P<classes>[^"]*)">'
        r"\{\{ t\('settings\.auth\.keyless_inference_hint'\) \}\}"
    )

    assert "text-red-600" in warning.split()
    assert "text-red-700" in label.split()
    assert "text-red-600" in hint.split()


def test_design_rules_document_the_danger_steps():
    design = DESIGN.read_text(encoding="utf-8")
    assert "### Danger text" in design
    for step in ("`text-red-700`", "`text-red-600`", "`text-red-500`"):
        assert step in design


DASHBOARD = ROOT / "omlx_web/templates/dashboard"


def _dashboard(name: str) -> str:
    return (DASHBOARD / name).read_text(encoding="utf-8")


def test_dashboard_error_strings_use_red_600():
    """Inline errors across the dashboard use the 4.5:1 red step."""
    assert 'x-show="logError" class="text-red-600"' in _dashboard("_logs.html")
    assert "text-sm text-red-600" in _dashboard("_bench.html")
    assert "'text-red-600'" in _dashboard("_bench_accuracy.html")

    models = _dashboard("_models.html")
    error_lines = [line for line in models.splitlines() if "task.error" in line]
    assert len(error_lines) >= 4
    assert all("text-red-500" not in line for line in error_lines)
    # The only red-500 left in this template is the hover accent on delete icons.
    assert models.count("text-red-500") == models.count("hover:text-red-500")


def test_dashboard_icons_keep_red_500_as_an_accent():
    """Status icons are accents, not text, so they stay on red-500."""
    cluster = _dashboard("_cluster_v2.html")
    assert cluster.count('class="w-4 h-4 text-red-500"') == 2
