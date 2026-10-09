# SPDX-License-Identifier: Apache-2.0
"""Active models dashboard block rendering of DFlash metrics."""

import json
from pathlib import Path


def test_dflash_dashboard_localizes_metrics_and_shows_session_fallbacks():
    root = Path(__file__).resolve().parents[1]
    template = (
        root / "omlx_web/templates/dashboard/blocks/_active_models.html"
    ).read_text(encoding="utf-8")
    dashboard_js = (root / "omlx_web/static/js/dashboard.js").read_text(
        encoding="utf-8"
    )
    i18n_dir = root / "omlx_web/i18n"
    keys = {
        "status.active_models.dflash_fallback_ar",
        "status.active_models.dflash_draft_share",
        "status.active_models.dflash_accepted_draft_per_cycle",
        "status.active_models.dflash_output_per_cycle",
        "status.active_models.dflash_draft_tokens_last_request",
        "status.active_models.dflash_session",
        "status.active_models.dflash_speculative_requests",
        "status.active_models.dflash_fallback_requests",
    }

    assert "m.dflash.speculation.totals.requests > 1" in template
    assert "formatDFlashSessionStats(m.dflash.speculation.totals)" in template
    assert "totals.fallback_requests > 0" in dashboard_js
    for key in keys:
        assert key in template + dashboard_js
    for locale_path in i18n_dir.glob("*.json"):
        locale = json.loads(locale_path.read_text(encoding="utf-8"))
        assert not keys - locale.keys(), f"{locale_path.name} is missing DFlash keys"
