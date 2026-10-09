# SPDX-License-Identifier: Apache-2.0
"""Status template rendering of cluster badges and rank cache rows."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_status_template_renders_cluster_badge_and_rank_cache_row():
    blocks = ROOT / "omlx_web/templates/dashboard/blocks"
    status = (blocks / "_active_models.html").read_text() + (
        blocks / "_cache_observability.html"
    ).read_text()
    javascript = (ROOT / "omlx_web/static/js/dashboard.js").read_text()
    en = json.loads((ROOT / "omlx_web/i18n/en.json").read_text())

    assert status.count("clusterBadgeLabel(m.cluster)") == 2  # mobile + desktop
    assert "clusterBadgeLabel(cluster)" in javascript
    assert "m.cluster?.live?.stale" in status
    assert "m.cache_tier === 'rank-prompt-snapshot'" in status
    assert "m.rank_prompt_cache" in status
    for key in (
        "cluster.badge.label",
        "cluster.badge.tensor",
        "cluster.badge.pipeline",
        "cluster.badge.stale",
        "cluster.badge.rank_cache",
        "cluster.badge.rank_cache_entries",
    ):
        assert en.get(key), f"en.json missing {key}"
