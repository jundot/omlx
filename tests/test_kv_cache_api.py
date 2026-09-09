# SPDX-License-Identifier: Apache-2.0
"""Source-level contracts for KV cache observability."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def _cluster_template() -> str:
    return _read("omlx/admin/templates/dashboard/_cluster.html")


def _dashboard_css() -> str:
    return _read("omlx/admin/static/css/dashboard.css")


def test_kv_cache_endpoint_exists():
    """The KV cache API router is importable and registers a GET /kv-cache."""
    from omlx.cluster.kv_cache_api import router

    paths = [route.path for route in router.routes]
    assert any("kv-cache" in p for p in paths)


def test_kv_cache_response_schema():
    """The handler docstring specifies the required response fields."""
    api_source = _read("omlx/cluster/kv_cache_api.py")
    assert "nodes" in api_source
    assert "timeseries" in api_source
    assert "score" in api_source
    assert "cache_size_mb" in api_source
    assert "hit_rate" in api_source
    assert "eviction_rate" in api_source
    assert "memory_pressure" in api_source


def test_kv_cache_uses_cluster_card():
    """Template section uses the .cluster-card pattern from #2764."""
    cluster = _cluster_template()
    css = _dashboard_css()

    assert "data-cluster-kv-cache" in cluster
    assert 'class="cluster-card' in cluster

    assert ".cluster-card {" in css
    assert "var(--bg-primary)" in css
    assert ".cluster-card__header {" in css
    assert ".cluster-card__body {" in css


def test_kv_cache_no_cdn():
    """KV cache section does not introduce CDN references."""
    cluster = _cluster_template()
    base = ROOT / "omlx/admin/templates/base.html"

    for label, html in [("cluster", cluster), ("base", base.read_text())]:
        lower = html.lower()
        assert "cdn." not in lower, f"{label} contains a CDN reference"
        assert "unpkg.com" not in lower, f"{label} references unpkg.com"
        assert "jsdelivr.net" not in lower, f"{label} references jsdelivr.net"


def test_kv_cache_gauge_css_exists():
    """Gauge bar styles are defined in dashboard.css."""
    css = _dashboard_css()

    assert ".cluster-kv-gauge {" in css
    assert ".cluster-kv-gauge__track {" in css
    assert ".cluster-kv-gauge__fill {" in css
    assert ".cluster-kv-gauge__fill--danger {" in css
    assert ".cluster-kv-gauge__fill--warn {" in css
    assert ".cluster-kv-chart {" in css


def test_kv_cache_section_polls_via_alpine():
    """Template section uses Alpine.js data binding and polling."""
    cluster = _cluster_template()

    assert "x-init=\"loadClusterKvCache()\"" in cluster
    assert "pollClusterKvCache()" in cluster
    assert "clusterKvCacheNodes()" in cluster
    assert "clusterKvCacheScore()" in cluster
    assert "clusterKvChartLinePath()" in cluster
    assert "clusterKvEvictionAlert()" in cluster


def test_kv_cache_section_hidden_when_not_live():
    """KV cache section only shows when cluster is live."""
    cluster = _cluster_template()

    kv_section = cluster.split("<!-- KV Cache observability -->", 1)[1].split(
        "The old full-page shard/runtime console", 1
    )[0]
    assert 'clusterNeuralFabricJob()?.live' in kv_section


def test_kv_cache_api_has_importable_router():
    """kv_cache_api module is importable without side effects."""
    from omlx.cluster.kv_cache_api import router  # noqa: F401
