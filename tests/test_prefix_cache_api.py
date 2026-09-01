# SPDX-License-Identifier: Apache-2.0
"""Source-level tests for prefix-cache topology API (#2767).

Following the pattern in test_cluster_exposure.py: source-read tests that
verify endpoint registration, response schema, template patterns, and
no-CDN policy without requiring a running server.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "omlx"
TEMPLATE = SRC / "admin" / "templates" / "dashboard" / "_cluster.html"
CSS = SRC / "admin" / "static" / "css" / "dashboard.css"
JS = SRC / "admin" / "static" / "js" / "dashboard.js"
API_MODULE = SRC / "cluster" / "prefix_cache_api.py"


def test_prefix_cache_api_module_exists():
    """The backend API module is present in the cluster package."""
    assert API_MODULE.is_file(), f"{API_MODULE} does not exist"


def test_prefix_cache_api_has_router():
    """The module exposes a FastAPI APIRouter with the correct prefix."""
    source = API_MODULE.read_text()
    assert "router = APIRouter(" in source
    assert '/prefix-cache"' in source
    assert "prefix=\"/admin/api/cluster\"" in source


def test_prefix_cache_api_has_get_endpoint():
    """The module defines a GET endpoint for prefix-cache topology."""
    source = API_MODULE.read_text()
    assert '@router.get("/prefix-cache")' in source
    assert "PrefixCacheTopology" in source


def test_prefix_cache_api_response_schema():
    """The response schema contains required fields (nodes, events, totals)."""
    source = API_MODULE.read_text()
    assert "class PrefixCacheTopology" in source
    assert "nodes:" in source
    assert "events:" in source
    assert "totals:" in source


def test_prefix_cache_api_node_schema():
    """NodeCacheState has the required fields for per-node cache state."""
    source = API_MODULE.read_text()
    assert "class NodeCacheState" in source
    assert "total_blocks:" in source
    assert "used_blocks:" in source
    assert "hit_rate:" in source
    assert "prefix_hashes:" in source


def test_prefix_cache_api_event_schema():
    """CacheEvent has kind, ts, and tokens_saved fields."""
    source = API_MODULE.read_text()
    assert "class CacheEvent" in source
    assert "kind:" in source
    assert "ts:" in source
    assert "tokens_saved:" in source


def test_prefix_cache_endpoint_registered_on_app():
    """The cluster prefix-cache router is registered in _register_cluster_routes."""
    server_source = (SRC / "server.py").read_text()
    assert "prefix_cache_api" in server_source or "prefix-cache" in server_source


def test_prefix_cache_uses_cluster_card_pattern():
    """The template uses .cluster-card CSS patterns from #2764."""
    template = TEMPLATE.read_text()
    assert 'data-cluster-prefix-cache' in template
    assert "cluster-card__header" in template
    assert "cluster-card__body" in template


def test_prefix_cache_no_cdn():
    """No CDN references in the new template code."""
    template_source = TEMPLATE.read_text(encoding="utf-8")

    lines = template_source.split("\n")
    in_prefix_cache_section = False
    for line in lines:
        if "data-cluster-prefix-cache" in line:
            in_prefix_cache_section = True
        if in_prefix_cache_section and "cdn" in line.lower():
            if "googleapis" in line or "cloudflare" in line or "jsdelivr" in line:
                raise AssertionError(f"CDN reference found in prefix-cache section: {line.strip()}")


def test_prefix_cache_css_exists():
    """The CSS module contains prefix-cache specific styles."""
    css = CSS.read_text()
    assert "prefix-cache" in css


def test_prefix_cache_css_uses_custom_properties():
    """Prefix-cache CSS uses the shared CSS custom property system."""
    css = CSS.read_text()
    assert "var(--border-faint)" in css
    assert "var(--bg-secondary)" in css
    assert "var(--text-primary)" in css


def test_prefix_cache_css_dark_mode_support():
    """Prefix-cache CSS includes dark mode rules."""
    css = CSS.read_text()
    assert '[data-theme="dark"] .prefix-cache' in css


def test_prefix_cache_js_alpine_data_property():
    """Dashboard JS declares the prefixCacheTopology data property."""
    js = JS.read_text()
    assert "prefixCacheTopology" in js
    assert "prefixCacheTopology: null" in js


def test_prefix_cache_js_fetch_method():
    """Dashboard JS has a loadPrefixCacheTopology method."""
    js = JS.read_text()
    assert "loadPrefixCacheTopology" in js
    assert "/admin/api/cluster/prefix-cache" in js


def test_prefix_cache_js_hit_rate_helpers():
    """Dashboard JS has helper methods for hit rate tone and label."""
    js = JS.read_text()
    assert "prefixCacheHitRateTone" in js
    assert "prefixCacheHitRateLabel" in js


def test_prefix_cache_js_polled_in_refresh():
    """The prefix-cache topology is fetched during cluster refresh."""
    js = JS.read_text()
    assert "loadPrefixCacheTopology" in js


def test_prefix_cache_template_uses_alpine_bind():
    """Template binds to prefixCacheTopology using Alpine.js directives."""
    template = TEMPLATE.read_text()
    assert "x-show=\"clusterShowSetupDetails && prefixCacheTopology\"" in template
    assert "prefixCacheTopology?.totals" in template
    assert "prefixCacheTopology?.nodes" in template
    assert "prefixCacheTopology?.events" in template


def test_prefix_cache_template_uses_utilization_bar():
    """Template renders per-node cache utilization bars."""
    template = TEMPLATE.read_text()
    assert "prefix-cache-util" in template
    assert "prefix-cache-util__used" in template


def test_prefix_cache_template_events_section():
    """Template includes a recent events section with color coding."""
    template = TEMPLATE.read_text()
    assert "prefix-cache-event" in template
    assert "prefix-cache-event--hit" in template
    assert "prefix-cache-event--miss" in template
    assert "prefix-cache-event--evict" in template
