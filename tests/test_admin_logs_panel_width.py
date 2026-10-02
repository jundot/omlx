# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the wider server-logs panel (issue #2436).

Every main tab renders inside one shared container (``templates/dashboard.html``)
that caps the page column with ``max-w-7xl``, so log lines wrapped instead of
fitting on one row. These tests pin that the logs tab gets a wider column and
that the escape cannot leak into a sibling tab:

1. Width      - the container binds a wider max-width class on the logs tab.
2. No leak    - every other main tab still resolves to the shared ``max-w-7xl``
                default, evaluated over the real Alpine expression for every tab.
3. Responsive - the escape stays a ``max-width`` cap and never a fixed ``width``,
                so narrow viewports shrink to the viewport instead of overflowing.
"""

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "omlx/admin/templates"
DASHBOARD = (TEMPLATES / "dashboard.html").read_text(encoding="utf-8")
TAILWIND_CSS = (ROOT / "omlx/admin/static/css/tailwind.css").read_text(encoding="utf-8")
SAFELIST = (ROOT / "omlx/admin/tailwind.config.js").read_text(encoding="utf-8")
DASHBOARD_JS = (ROOT / "omlx/admin/static/js/dashboard.js").read_text(encoding="utf-8")

# Kept in sync with DASHBOARD_MAIN_TABS in admin/static/js/dashboard.js.
MAIN_TABS = ["status", "cluster", "settings", "models", "logs", "bench"]
SHARED_WIDTH_CLASS = "max-w-7xl"
LOGS_WIDTH_CLASS = "max-w-[100rem]"

# The single shared page container that every main tab renders inside.
CONTAINER_RE = re.compile(r'<div class="([^"]*)" :class="(mainTab[^"]*)">')


def _container() -> tuple[str, str]:
    match = CONTAINER_RE.search(DASHBOARD)
    assert match is not None, "shared main-tab container not found in dashboard.html"
    return match.group(1), match.group(2)


def _resolve_width_classes() -> dict:
    """Evaluate the container's real Alpine :class expression for every main tab.

    The expression is lifted verbatim out of the template and run in Node, so the
    table these tests assert on describes the shipped markup rather than a
    restatement of it.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to evaluate the dashboard width binding")
    _, expression = _container()
    script = (
        "const tabs = " + json.dumps(MAIN_TABS + ["bogus-tab"]) + ";\n"
        "const expr = " + json.dumps(expression) + ";\n"
        "const fn = new Function('mainTab', 'dashboardWidthClass', 'return ' + expr + ';');\n"
        "const out = {};\n"
        "for (const tab of tabs) out[tab] = fn(tab, 'max-w-7xl');\n"
        # The status tab follows the user's stored layout width; pin that too.
        "out['status@full'] = fn('status', 'max-w-none');\n"
        "console.log(JSON.stringify(out));\n"
    )
    result = subprocess.run(
        [node, "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip())


def test_all_main_tabs_share_the_one_container():
    """Guard: the container really is shared, so the escape must be tab-scoped."""
    for panel in (
        "_status.html",
        "_cluster_v2.html",
        "_settings.html",
        "_models.html",
        "_logs.html",
        "_bench.html",
    ):
        assert f'dashboard/{panel}' in DASHBOARD, panel
    assert DASHBOARD.count(':class="mainTab') == 1, "width binding is not unique"


def test_main_tab_set_is_unchanged():
    """Kept in sync with DASHBOARD_MAIN_TABS so the leak table stays exhaustive."""
    declared = re.search(r"DASHBOARD_MAIN_TABS = new Set\((\[[^\]]*\])\)", DASHBOARD_JS)
    assert declared is not None, "DASHBOARD_MAIN_TABS not found in dashboard.js"
    assert ast.literal_eval(declared.group(1)) == MAIN_TABS


def test_logs_tab_gets_its_own_wider_container():
    """The logs tab must not fall through to the shared max-w-7xl column."""
    static_classes, expression = _container()
    assert "mainTab === 'logs'" in expression, (
        "the shared container must widen for the logs tab, got: " + expression
    )
    assert _resolve_width_classes()["logs"] == LOGS_WIDTH_CLASS


def test_wider_class_is_a_max_width_cap_present_in_compiled_css():
    """max-width caps without forcing width, so narrow viewports still shrink."""
    assert LOGS_WIDTH_CLASS in _container()[1]

    rule = re.search(r"\.max-w-\\\[100rem\\\]\{([^}]*)\}", TAILWIND_CSS)
    assert rule is not None, f"{LOGS_WIDTH_CLASS} missing from compiled tailwind.css"
    assert rule.group(1) == "max-width:100rem", rule.group(1)

    # Safelisted because the class is chosen at runtime by the :class binding.
    assert LOGS_WIDTH_CLASS in SAFELIST


def test_wider_escape_does_not_leak_to_sibling_tabs():
    """Counter-example table: only 'logs' changes, every other tab is untouched."""
    resolved = _resolve_width_classes()
    for tab in MAIN_TABS:
        if tab == "logs":
            assert resolved[tab] == LOGS_WIDTH_CLASS, tab
        elif tab == "status":
            # Unchanged: the status tab still follows the user's stored width.
            assert resolved[tab] == SHARED_WIDTH_CLASS, tab
            assert resolved["status@full"] == "max-w-none", "status width preference lost"
        else:
            assert resolved[tab] == SHARED_WIDTH_CLASS, f"{tab} tab width changed"

    # An unknown tab (stale ?tab= in the URL) also keeps the shared default.
    assert resolved["bogus-tab"] == SHARED_WIDTH_CLASS
    # Nothing may resolve to a fixed width; that is what breaks narrow viewports.
    assert all(cls.startswith("max-w-") for cls in resolved.values()), resolved


def test_container_stays_flow_width_for_narrow_viewports():
    """No fixed width on the shared container: that is what would overflow phones."""
    static_classes, _ = _container()
    # The static classes must set no width/min-width at all; the only width input
    # is the max-width cap bound by :class.
    for token in static_classes.split():
        assert not re.match(r"^(min-)?w-", token), f"fixed width {token!r} on shared container"
    # The column stays centered, and <main> still clips it.
    assert "mx-auto" in static_classes
    assert '<main class="flex-grow relative overflow-hidden">' in DASHBOARD
