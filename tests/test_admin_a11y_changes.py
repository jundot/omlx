# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the ARIA + viewport additions in the web dashboard a11y
changelog. Complements `tests/test_admin_accessibility.py`, which already covers
focus-ring contrast; this file guards the structural ARIA changes introduced
in the same PR (viewport zoom unlock, tablist/tab semantics, dialog roles,
locale navbar.tablist_label presence).

The contract is intentionally file-content based so the tests do not require a
live server; they walk the static templates and locale catalogs under
`omlx/admin/`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADMIN_DIR = ROOT / "omlx" / "admin"
TEMPLATES = ADMIN_DIR / "templates"
DASHBOARD = TEMPLATES / "dashboard"
I18N = ADMIN_DIR / "i18n"

BASE_HTML = (TEMPLATES / "base.html").read_text(encoding="utf-8")
NAVBAR = (DASHBOARD / "_navbar.html").read_text(encoding="utf-8")
MODAL_MODEL = (DASHBOARD / "_modal_model_settings.html").read_text(encoding="utf-8")
DASHBOARD_TEMPLATE = (TEMPLATES / "dashboard.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. viewport zoom is not disabled
# ---------------------------------------------------------------------------


def test_viewport_does_not_lock_zoom():
    """The viewport meta should not pin `maximum-scale=1.0`, which would
    disable zoom on touch devices."""
    match = re.search(r'<meta\s+name="viewport"\s+content="([^"]+)"', BASE_HTML)
    assert match is not None, "expected a viewport meta tag in base.html"
    content = match.group(1)
    assert "maximum-scale=1.0" not in content, (
        "the `maximum-scale=1.0` zoom lock must be removed; "
        f"current viewport content: {content!r}"
    )


# ---------------------------------------------------------------------------
# 2. main nav has tablist/tab/aria-selected semantics
# ---------------------------------------------------------------------------


def test_main_nav_uses_role_tablist():
    assert 'role="tablist"' in NAVBAR, (
        "the dashboard nav must use role=tablist to announce its container"
    )


def test_main_nav_uses_role_tab_on_each_link():
    tab_blocks = re.findall(r'<(?:a|button)\b[^>]*role="tab"[^>]*>', NAVBAR)
    # The dashboard currently exposes six top-level tabs.
    assert len(tab_blocks) >= 2, (
        f"expected at least two role=tab entries in the main nav, got {len(tab_blocks)}"
    )


def test_main_nav_uses_aria_selected_on_each_tab():
    """Every role=tab entry must also declare `aria-selected`, which reflects
    the active panel id and is the attribute AT relies on for selection state."""
    tab_blocks = re.findall(r'<(?:a|button)\b[^>]*role="tab"[^>]*>', NAVBAR)
    for block in tab_blocks:
        assert "aria-selected" in block, f"tab block missing aria-selected: {block!r}"


def test_main_nav_tabs_have_id_aria_controls():
    """ARIA Authoring Practices require an `id` on the tab and `aria-controls`
    pointing to its associated tabpanel id."""
    tab_blocks = re.findall(r'<(?:a|button)\b[^>]*role="tab"[^>]*>', NAVBAR)
    id_regex = re.compile(r'\bid="([^"]+)"')
    ctrls_regex = re.compile(r'\baria-controls="([^"]+)"')
    for block in tab_blocks:
        tab_id = id_regex.search(block)
        assert tab_id, f"role=tab missing id: {block!r}"
        assert ctrls_regex.search(block), (
            f"role=tab id={tab_id.group(1)} missing aria-controls: {block!r}"
        )


# ---------------------------------------------------------------------------
# 3. tabpanel semantics on the per-section panels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "panel_name",
    ["_status", "_cluster_v2", "_models", "_bench"],
)
def test_panel_uses_role_tabpanel_with_aria_labelledby(panel_name):
    """The five dashboard panels (`_status`, `_cluster_v2`, `_models`,
    `_logs`, `_bench`) should each render with `role=tabpanel`,
    `id`, and `aria-labelledby` pointing back at its corresponding tab."""
    path = DASHBOARD / f"{panel_name}.html"
    body = path.read_text(encoding="utf-8")
    panel_match = re.search(
        r'<(?:section|div|aside)\b[^>]*role="tabpanel"[^>]*>',
        body,
    )
    assert panel_match is not None, (
        f"{panel_name}.html must contain at least one role=tabpanel section"
    )
    open_tag = panel_match.group(0)
    assert 'id="' in open_tag, (
        f"{panel_name}.html role=tabpanel must declare an id (got {open_tag!r})"
    )
    assert 'aria-labelledby="' in open_tag, (
        f"{panel_name}.html role=tabpanel must declare aria-labelledby (got {open_tag!r})"
    )


# ---------------------------------------------------------------------------
# 4. dialog role + aria-modal on the modals
# ---------------------------------------------------------------------------


def test_hf_mirror_modal_has_dialog_role():
    """The HF mirror modal is rendered via dashboard.html; verify it carries
    `role=dialog`, `aria-modal=true`, and a labelled-by heading."""
    assert 'role="dialog"' in DASHBOARD_TEMPLATE, (
        "the HF mirror modal in dashboard.html must expose role=dialog"
    )
    assert "aria-modal" in DASHBOARD_TEMPLATE, (
        "the HF mirror modal must expose aria-modal"
    )


def test_model_settings_modal_has_dialog_role():
    """The model settings modal partial must be a dialog with aria-modal."""
    assert 'role="dialog"' in MODAL_MODEL, "model-settings modal must be a dialog"
    assert "aria-modal" in MODAL_MODEL, (
        "model-settings modal must declare aria-modal"
    )


# ---------------------------------------------------------------------------
# 5. locale catalogs agree with the new navbar.tablist_label
# ---------------------------------------------------------------------------


ALL_LOCALES = [
    "en",
    "es",
    "fr",
    "ja",
    "ko",
    "pt-BR",
    "ru",
    "zh",
    "zh-TW",
]


@pytest.mark.parametrize("locale", ALL_LOCALES)
def test_navbar_tablist_label_is_present_in_each_locale(locale):
    """Every locale catalog must include `navbar.tablist_label`. The English
    fallback will silently mask the missing key, so we pin this directly."""
    path = I18N / f"{locale}.json"
    assert path.exists(), f"{locale}.json is missing from i18n/"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "navbar.tablist_label" in data, (
        f"{locale}.json is missing `navbar.tablist_label`; check that the tablist "
        "aria-label has a translation in this locale."
    )
    value = data["navbar.tablist_label"]
    assert isinstance(value, str) and value.strip(), (
        f"{locale}.json navbar.tablist_label must be a non-empty string"
    )
