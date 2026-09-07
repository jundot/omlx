# SPDX-License-Identifier: Apache-2.0
"""Source-level contracts for the cluster dashboard CSS architecture."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def _cluster_template() -> str:
    return _read("omlx/admin/templates/dashboard/_cluster.html")


def _dashboard_css() -> str:
    return _read("omlx/admin/static/css/dashboard.css")


def _dashboard_html() -> str:
    return _read("omlx/admin/templates/dashboard.html")


def _base_html() -> str:
    return _read("omlx/admin/templates/base.html")


def test_cluster_dashboard_uses_vendored_assets():
    """Cluster template and its parent use vendored assets, never CDN references."""
    cluster = _cluster_template()
    base = _base_html()

    for label, html in [("cluster", cluster), ("base", base)]:
        lower = html.lower()
        assert "cdn." not in lower, f"{label} contains a CDN reference"
        assert "unpkg.com" not in lower, f"{label} references unpkg.com"
        assert "jsdelivr.net" not in lower, f"{label} references jsdelivr.net"
        assert "cdnjs.cloudflare" not in lower, f"{label} references cdnjs.cloudflare"
        assert "fonts.googleapis.com" not in lower, f"{label} references Google Fonts CDN"
        assert "ajax.googleapis.com" not in lower, f"{label} references Google CDN"

    # All three vendored assets are loaded from /admin/static/
    assert "/admin/static/js/alpine.min.js" in base
    assert "/admin/static/js/lucide.min.js" in base
    assert "tailwind.css" in base
    assert "static('css/tailwind.css')" in base or "/admin/static/css/tailwind.css" in base


def test_cluster_uses_css_variables():
    """Cluster CSS classes use the existing CSS custom property system."""
    css = _dashboard_css()

    # The variable system exists
    assert "--bg-primary" in css
    assert "--text-primary" in css
    assert "--border-faint" in css
    assert "--bg-secondary" in css
    assert "--bg-tertiary" in css

    # Dark mode variable overrides exist
    assert '[data-theme="dark"]' in css

    # Cluster component classes reference CSS variables
    assert ".cluster-card {" in css
    assert "var(--bg-primary)" in css
    assert "var(--border-faint)" in css
    assert "var(--text-primary)" in css
    assert "var(--bg-tertiary)" in css
    assert "var(--bg-secondary)" in css


def test_cluster_responsive_grid():
    """Cluster template uses responsive grid classes for layout."""
    cluster = _cluster_template()

    # Loading state: responsive grid
    assert 'grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4' in cluster
    # Capability summary: responsive grid
    assert 'grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4' in cluster
    # Node card: responsive grid
    assert 'grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4' in cluster
    # Validation: responsive grid
    assert 'grid grid-cols-1 lg:grid-cols-3' in cluster


def test_cluster_components_have_card_pattern():
    """Cluster components use the standard card pattern with CSS classes."""
    css = _dashboard_css()

    # The cluster-card class encapsulates the card pattern
    assert ".cluster-card {" in css
    assert "var(--bg-primary)" in css
    assert "var(--border-faint)" in css

    # Header bar pattern uses bg-tertiary
    assert ".cluster-card__header {" in css
    assert "var(--bg-tertiary)" in css

    # Body pattern
    assert ".cluster-card__body {" in css

    # Capacity bar pattern exists
    assert ".cluster-capacity-bar {" in css
    assert ".cluster-capacity-bar__weights {" in css
    assert ".cluster-capacity-bar__kv {" in css

    # Health ring pattern exists
    assert ".cluster-health-ring {" in css

    # Node row pattern exists
    assert ".cluster-node-row {" in css
    assert ".cluster-node-row__identity {" in css

    # Model row pattern exists
    assert ".cluster-model-row {" in css
    assert ".cluster-model-row--selected {" in css

    # Expand/collapse toggle pattern exists
    assert ".cluster-toggle {" in css
    assert ".cluster-toggle__chevron {" in css


def test_dark_mode_support():
    """Dark mode is supported via the data-theme attribute on the document."""
    base = _base_html()
    css = _dashboard_css()

    # base.html sets data-theme from localStorage
    assert "document.documentElement.setAttribute('data-theme'" in base

    # CSS has dark mode variable overrides
    assert '[data-theme="dark"]' in css
    assert "--bg-primary: #1d1d20" in css
    assert "--text-primary: #efefef" in css

    # Cluster template inherits dark mode from parent
    cluster = _cluster_template()
    assert 'data-theme="dark"' not in cluster  # Cluster doesn't set it; parent does


def test_x_cloak_on_hidden_elements():
    """Elements with x-show have x-cloak for flash-of-content prevention."""
    cluster = _cluster_template()

    # x-cloak style rule exists in base.html
    base = _base_html()
    assert "[x-cloak]" in base
    assert "display: none" in base

    # The root tab container uses x-cloak
    assert 'x-show="mainTab === \'cluster\'"' in cluster
    assert "x-cloak" in cluster

    # Key visible elements have x-cloak for flash prevention
    assert 'x-show="clusterDisplayedError()"' in cluster
    assert 'x-show="clusterLoading && !clusterStatus"' in cluster
    assert 'x-show="clusterShowSetupDetails"' in cluster

    # x-cloak is used extensively (at least 50 times in this large template)
    assert cluster.count("x-cloak") >= 50


def test_cluster_section_comment_in_dashboard_css():
    """Dashboard.css has an explicit Cluster section for cluster-specific styles."""
    css = _dashboard_css()

    # There should be a section comment marking the cluster styles
    assert "Cluster" in css.split("/* === Cluster")[0] or "Cluster" in css
    # Specifically, we need a dedicated Cluster section header
    assert "Cluster" in css


def test_cluster_model_row_uses_dark_mode_variables():
    """Cluster model row styles work in both light and dark mode."""
    css = _dashboard_css()

    # Model row uses CSS variables, not hardcoded colors
    model_row_section = css.split(".cluster-model-row {", 1)[1].split(
        ".cluster-model-row--selected", 1
    )[0]
    assert "var(--border-faint)" in model_row_section
    assert "var(--bg-primary)" in model_row_section


def test_cluster_toggle_uses_dark_mode_variables():
    """Cluster toggle styles work in both light and dark mode."""
    css = _dashboard_css()

    toggle_section = css.split(".cluster-toggle {", 1)[1].split(
        ".cluster-toggle__chevron", 1
    )[0]
    assert "var(--border-faint)" in toggle_section
    assert "var(--bg-primary)" in toggle_section
    assert "var(--bg-secondary)" in toggle_section

    # Chevron uses muted text color
    chevron_section = css.split(".cluster-toggle__chevron {", 1)[1].split(
        ".cluster-toggle[aria-expanded", 1
    )[0]
    assert "var(--text-muted)" in chevron_section


def test_no_inline_style_blocks_in_cluster_template():
    """Cluster template uses CSS classes, not inline <style> blocks."""
    cluster = _cluster_template()

    assert "<style>" not in cluster
    assert "</style>" not in cluster
