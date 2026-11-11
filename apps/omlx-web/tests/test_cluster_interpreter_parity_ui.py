# SPDX-License-Identifier: Apache-2.0
"""Cluster wizard rendering of interpreter parity warnings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_does_not_render_a_warned_peer_as_an_unqualified_match():
    wizard = (ROOT / "omlx_web/static/js/cluster_v2.js").read_text(encoding="utf-8")
    template = (ROOT / "omlx_web/templates/dashboard/_cluster_v2.html").read_text(
        encoding="utf-8"
    )

    assert "probe.result?.runtime_warnings" in wizard
    assert "warnings.length" in wizard
    assert "? 'warn'" in wizard
    assert "row.status === 'warn'" in template
    assert "text-amber-700" in template
