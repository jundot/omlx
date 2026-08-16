# SPDX-License-Identifier: Apache-2.0
"""Tests for the "Expose backend MCP tools to clients" dashboard toggle.

Covers the server-side helper (``omlx.server.mcp_tools_exposed``), the admin
dashboard toggle markup, and i18n key presence across all locales. The
end-to-end merge behaviour is covered in
``tests/integration/test_server_endpoints.py`` (TestMCPExposeToolsToggle).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import omlx.server as server
from omlx.settings import GlobalSettings, MCPSettings

ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = ROOT / "omlx/admin/i18n"
SETTINGS_TEMPLATE = ROOT / "omlx/admin/templates/dashboard/_settings.html"
DASHBOARD_JS = ROOT / "omlx/admin/static/js/dashboard.js"

REQUIRED_I18N_KEYS = {
    "settings.mcp.expose_tools",
    "settings.mcp.expose_tools_hint",
}


class TestMcpToolsExposedHelper:
    """Unit tests for ``omlx.server.mcp_tools_exposed``."""

    def test_true_when_global_settings_unavailable(self, monkeypatch):
        """No global settings (e.g. MCP via env var) -> keep exposing."""
        monkeypatch.setattr(server._server_state, "global_settings", None)
        assert server.mcp_tools_exposed() is True

    def test_true_when_expose_tools_enabled(self, monkeypatch):
        monkeypatch.setattr(
            server._server_state,
            "global_settings",
            GlobalSettings(mcp=MCPSettings(expose_tools=True)),
        )
        assert server.mcp_tools_exposed() is True

    def test_false_when_expose_tools_disabled(self, monkeypatch):
        monkeypatch.setattr(
            server._server_state,
            "global_settings",
            GlobalSettings(mcp=MCPSettings(expose_tools=False)),
        )
        assert server.mcp_tools_exposed() is False

    def test_true_for_legacy_settings_without_flag(self, monkeypatch):
        """A settings object without the attribute must default to True."""
        monkeypatch.setattr(
            server._server_state,
            "global_settings",
            SimpleNamespace(mcp=SimpleNamespace()),
        )
        assert server.mcp_tools_exposed() is True


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


class TestAdminApiExposeTools:
    """The /api/global-settings GET/POST round trip carries the toggle."""

    def test_get_global_settings_includes_expose_tools(self, tmp_path, monkeypatch):
        import asyncio

        from omlx.admin import routes as admin_routes

        gs = GlobalSettings(base_path=tmp_path)
        gs.mcp.config_path = "/mcp.json"
        gs.mcp.expose_tools = False
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)

        # Real get_system_memory_info / get_ssd_disk_info are used here; both
        # are pure sysctl/statfs helpers with safe fallbacks on any macOS host.
        result = asyncio.run(admin_routes.get_global_settings(is_admin=True))
        assert result["mcp"]["config_path"] == "/mcp.json"
        assert result["mcp"]["expose_tools"] is False

    def test_post_global_settings_applies_and_persists_expose_tools(
        self, tmp_path, monkeypatch
    ):
        import asyncio

        from omlx.admin import routes as admin_routes

        gs = GlobalSettings(base_path=tmp_path)
        gs.mcp.config_path = "/mcp.json"
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)

        request = admin_routes.GlobalSettingsRequest(mcp_expose_tools=False)
        result = asyncio.run(
            admin_routes.update_global_settings(request=request, is_admin=True)
        )
        assert result["success"] is True
        assert gs.mcp.expose_tools is False
        assert "mcp_expose_tools" in result["runtime_applied"]

        # Persisted to disk, so a server restart keeps the toggle off.
        restored = GlobalSettings.load(base_path=tmp_path)
        assert restored.mcp.expose_tools is False

    def test_post_global_settings_without_toggle_keeps_current(
        self, tmp_path, monkeypatch
    ):
        import asyncio

        from omlx.admin import routes as admin_routes

        gs = GlobalSettings(base_path=tmp_path)
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)

        request = admin_routes.GlobalSettingsRequest()
        result = asyncio.run(
            admin_routes.update_global_settings(request=request, is_admin=True)
        )
        assert result["success"] is True
        assert gs.mcp.expose_tools is True
        assert "mcp_expose_tools" not in result["runtime_applied"]
