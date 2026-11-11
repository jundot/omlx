# SPDX-License-Identifier: Apache-2.0
"""Dashboard toggle markup and i18n keys for exposing backend MCP tools."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = ROOT / "omlx_web/i18n"
SETTINGS_TEMPLATE = ROOT / "omlx_web/templates/dashboard/_settings.html"
DASHBOARD_JS = ROOT / "omlx_web/static/js/dashboard.js"

REQUIRED_I18N_KEYS = {
    "settings.mcp.expose_tools",
    "settings.mcp.expose_tools_hint",
}


class TestDashboardToggleMarkup:
    """The Settings > Global Settings > MCP section renders the toggle."""

    def test_toggle_bound_to_expose_tools(self):
        html = SETTINGS_TEMPLATE.read_text(encoding="utf-8")
        assert "globalSettings.mcp.expose_tools" in html
        assert "settings.mcp.expose_tools" in html
        assert "settings.mcp.expose_tools_hint" in html

    def test_dashboard_state_defaults_expose_tools_true(self):
        javascript = DASHBOARD_JS.read_text(encoding="utf-8")
        assert "mcp: { config_path: '', expose_tools: true }" in javascript


class TestI18nKeys:
    """The new labels exist in every locale file."""

    def test_expose_tools_keys_present_in_every_locale(self):
        for locale_path in sorted(I18N_DIR.glob("*.json")):
            locale = json.loads(locale_path.read_text(encoding="utf-8"))
            missing = {key for key in REQUIRED_I18N_KEYS if not locale.get(key)}
            assert not missing, f"{locale_path.name}: missing {sorted(missing)}"

    def test_locale_key_sets_identical(self):
        base = set(json.loads((I18N_DIR / "en.json").read_text(encoding="utf-8")))
        for locale_path in sorted(I18N_DIR.glob("*.json")):
            keys = set(json.loads(locale_path.read_text(encoding="utf-8")))
            assert keys == base, locale_path.name
