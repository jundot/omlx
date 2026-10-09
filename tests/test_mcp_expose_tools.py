# SPDX-License-Identifier: Apache-2.0
"""Tests for the "Expose backend MCP tools to clients" dashboard toggle.

Covers the server-side helper (``omlx.server.mcp_tools_exposed``) and the
admin API round trip. The dashboard markup and i18n checks live in
apps/omlx-web/tests/test_mcp_expose_tools_ui.py, and the end-to-end merge
behaviour is covered in ``tests/integration/test_server_endpoints.py``
(TestMCPExposeToolsToggle).
"""

from types import SimpleNamespace

import omlx.server as server
from omlx.settings import GlobalSettings, MCPSettings


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
