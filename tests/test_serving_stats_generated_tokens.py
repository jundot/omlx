# SPDX-License-Identifier: Apache-2.0
"""Regression guards for the Generated Tokens stat card on the Status tab.

The card displays ``total_completion_tokens`` from the /admin/api/stats
snapshot (already recorded by ServerMetrics alongside prompt/cached tokens)
next to the prefill/cached/efficiency cards, for both session and all-time
scopes. These guards lock the i18n key, the template binding, and the
dashboard.js defaults that keep the card rendering before the first fetch.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = ROOT / "omlx/admin/i18n"
SERVING_STATS_TEMPLATE = (
    ROOT / "omlx/admin/templates/dashboard/blocks/_serving_stats.html"
)
DASHBOARD_JS = ROOT / "omlx/admin/static/js/dashboard.js"

GENERATED_TOKENS_I18N_KEY = "status.stat.generated_tokens"


def _template() -> str:
    return SERVING_STATS_TEMPLATE.read_text(encoding="utf-8")


def test_generated_tokens_i18n_key_present_in_every_locale():
    locales = sorted(I18N_DIR.glob("*.json"))
    assert len(locales) == 9
    for locale_path in locales:
        locale = json.loads(locale_path.read_text(encoding="utf-8"))
        label = locale.get(GENERATED_TOKENS_I18N_KEY)
        assert label, f"{locale_path.name}: missing {GENERATED_TOKENS_I18N_KEY}"
        assert label.strip(), f"{locale_path.name}: empty label"


def test_generated_tokens_card_rendered_after_cache_efficiency():
    html = _template()

    assert f"{{{{ t('{GENERATED_TOKENS_I18N_KEY}') }}}}" in html
    # Bound to the completion-token counter already returned by the snapshot,
    # with the same formatting/font-scaling helpers as the sibling cards.
    assert (
        "x-text=\"formatNumber((statsScope === 'alltime' ? alltimeStats : stats)"
        '.total_completion_tokens)"' in html
    )
    assert (
        ":class=\"getStatFontClass((statsScope === 'alltime' ? alltimeStats : stats)"
        '.total_completion_tokens)"' in html
    )

    # Card order: prefill, cached, efficiency, then generated tokens.
    assert (
        html.index("status.stat.total_tokens")
        < html.index("status.stat.cached_tokens")
        < html.index("status.stat.cache_efficiency")
        < html.index(GENERATED_TOKENS_I18N_KEY)
    )

    # Grid widened from three to four columns.
    assert "sm:grid-cols-2 lg:grid-cols-4" in html
    assert "sm:grid-cols-3" not in html


def test_dashboard_js_defaults_include_completion_tokens():
    """Pre-first-fetch render must not hit formatNumber(undefined)."""
    js = DASHBOARD_JS.read_text(encoding="utf-8")

    for state_name in ("stats", "alltimeStats"):
        section = js.split(f"{state_name}: {{", 1)[1].split("},", 1)[0]
        assert (
            "total_completion_tokens: 0" in section
        ), f"{state_name} default missing total_completion_tokens"
