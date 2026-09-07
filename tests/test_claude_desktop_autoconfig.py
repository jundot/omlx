# SPDX-License-Identifier: Apache-2.0
"""Tests for Claude Desktop auto-configuration (T-002).

When the ``Claude Desktop`` switch (``claude_code.desktop_enabled``, T-001)
is enabled, oMLX writes the macOS Claude Desktop JSON configs so the app
talks directly to the oMLX gateway (no external reverse proxy). Disabling
restores the previous configuration.

File logic mirrors ``docs/task/references/ClaudeConfig.swift``; the gateway
``/v1/models`` format follows ``docs/task/references/ModelMap.swift``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from omlx.integrations import claude_desktop
from omlx.integrations.claude_desktop import (
    PROFILE_ID,
    configure_omlx_gateway,
    is_configured,
    restore,
)

OLLAMA_SWITCHER_PROFILE = "00000000-0000-4000-8000-000000000114"


def _force_macos(monkeypatch) -> None:
    """Force the macOS code path regardless of the test runner OS."""
    monkeypatch.setattr(claude_desktop, "_is_macos", lambda: True)


def _force_not_macos(monkeypatch) -> None:
    monkeypatch.setattr(claude_desktop, "_is_macos", lambda: False)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestProfileId:
    def test_stable_and_distinct_from_ollama_switcher(self):
        assert PROFILE_ID != OLLAMA_SWITCHER_PROFILE
        # Valid UUID shape (the T-002 prompt placeholder with "MLX"
        # characters is not a valid UUID, so a fixed v4 value is used).
        parts = PROFILE_ID.split("-")
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]
        int(PROFILE_ID.replace("-", ""), 16)


class TestConfigure:
    def test_writes_all_four_files(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        assert configure_omlx_gateway(8000, "secret", home=tmp_path) is True

        base = tmp_path / "Library" / "Application Support"
        assert _read(base / "Claude/claude_desktop_config.json")["deploymentMode"] == "3p"
        assert _read(base / "Claude-3p/claude_desktop_config.json")["deploymentMode"] == "3p"

        meta = _read(base / "Claude-3p/configLibrary/_meta.json")
        assert {"id": PROFILE_ID, "name": "oMLX"} in meta["entries"]
        assert meta["appliedId"] == PROFILE_ID

        profile = _read(base / "Claude-3p/configLibrary" / f"{PROFILE_ID}.json")
        assert profile["inferenceProvider"] == "gateway"
        assert profile["inferenceGatewayBaseUrl"] == "http://127.0.0.1:8000"
        assert profile["inferenceGatewayApiKey"] == "secret"
        assert profile["inferenceGatewayAuthScheme"] == "bearer"
        assert profile["disableDeploymentModeChooser"] is True

    def test_open_server_defaults_api_key_to_omlx(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        assert configure_omlx_gateway(8000, None, home=tmp_path) is True
        profile = _read(
            tmp_path
            / "Library"
            / "Application Support"
            / "Claude-3p"
            / "configLibrary"
            / f"{PROFILE_ID}.json"
        )
        assert profile["inferenceGatewayApiKey"] == "omlx"
        assert configure_omlx_gateway(9000, "  ", home=tmp_path) is True
        profile = _read(
            tmp_path
            / "Library"
            / "Application Support"
            / "Claude-3p"
            / "configLibrary"
            / f"{PROFILE_ID}.json"
        )
        assert profile["inferenceGatewayBaseUrl"] == "http://127.0.0.1:9000"
        assert profile["inferenceGatewayApiKey"] == "omlx"

    def test_preserves_unrelated_keys(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        base = tmp_path / "Library" / "Application Support"
        target = base / "Claude/claude_desktop_config.json"
        target.parent.mkdir(parents=True)
        target.write_text(
            json.dumps({"deploymentMode": "1p", "theme": "dark"}), encoding="utf-8"
        )
        configure_omlx_gateway(8000, "k", home=tmp_path)
        data = _read(target)
        assert data == {"deploymentMode": "3p", "theme": "dark"}

    def test_idempotent(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        configure_omlx_gateway(8000, "k", home=tmp_path)
        base = tmp_path / "Library" / "Application Support"
        snapshot = {
            str(p): p.read_text(encoding="utf-8")
            for p in sorted(base.rglob("*.json"))
            if not p.name.endswith(".bak")
        }
        configure_omlx_gateway(8000, "k", home=tmp_path)
        rerun = {
            str(p): p.read_text(encoding="utf-8")
            for p in sorted(base.rglob("*.json"))
            if not p.name.endswith(".bak")
        }
        assert snapshot == rerun

    def test_backup_not_overwritten(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        base = tmp_path / "Library" / "Application Support"
        target = base / "Claude/claude_desktop_config.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"deploymentMode": "1p"}), encoding="utf-8")

        configure_omlx_gateway(8000, "k", home=tmp_path)
        backup = target.with_suffix(target.suffix + ".bak")
        assert _read(backup) == {"deploymentMode": "1p"}

        # Second configure must keep the ORIGINAL backup, not the 3p state.
        configure_omlx_gateway(8001, "k2", home=tmp_path)
        assert _read(backup) == {"deploymentMode": "1p"}
        assert _read(target)["deploymentMode"] == "3p"

    def test_non_macos_noop(self, tmp_path, monkeypatch):
        _force_not_macos(monkeypatch)
        assert configure_omlx_gateway(8000, "k", home=tmp_path) is False
        assert not (tmp_path / "Library").exists()


class TestRestore:
    def test_restore_round_trip(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        configure_omlx_gateway(8000, "k", home=tmp_path)
        assert is_configured(home=tmp_path) is True
        assert restore(home=tmp_path) is True

        base = tmp_path / "Library" / "Application Support"
        assert _read(base / "Claude/claude_desktop_config.json")["deploymentMode"] == "1p"
        assert _read(base / "Claude-3p/claude_desktop_config.json")["deploymentMode"] == "1p"
        meta = _read(base / "Claude-3p/configLibrary/_meta.json")
        assert all(e.get("id") != PROFILE_ID for e in meta.get("entries", []))
        assert meta.get("appliedId") != PROFILE_ID
        profile = _read(base / "Claude-3p/configLibrary" / f"{PROFILE_ID}.json")
        for key in (
            "inferenceProvider",
            "inferenceGatewayBaseUrl",
            "inferenceGatewayApiKey",
            "inferenceGatewayAuthScheme",
        ):
            assert key not in profile
        assert profile["disableDeploymentModeChooser"] is False
        assert is_configured(home=tmp_path) is False

    def test_restore_without_configure_is_safe_noop(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        assert restore(home=tmp_path) is True
        assert is_configured(home=tmp_path) is False

    def test_restore_preserves_other_meta_entries(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        base = tmp_path / "Library" / "Application Support"
        meta_path = base / "Claude-3p/configLibrary/_meta.json"
        meta_path.parent.mkdir(parents=True)
        meta_path.write_text(
            json.dumps(
                {
                    "entries": [{"id": "other-id", "name": "Other"}],
                    "appliedId": "other-id",
                }
            ),
            encoding="utf-8",
        )
        configure_omlx_gateway(8000, "k", home=tmp_path)
        restore(home=tmp_path)
        meta = _read(meta_path)
        assert {"id": "other-id", "name": "Other"} in meta["entries"]
        # appliedId pointed at the other profile throughout; untouched.
        assert meta["appliedId"] == "other-id"

    def test_double_configure_restores_original_owner(self, tmp_path, monkeypatch):
        """configure -> configure -> restore hands appliedId to the FIRST owner."""
        _force_macos(monkeypatch)
        base = tmp_path / "Library" / "Application Support"
        meta_path = base / "Claude-3p/configLibrary/_meta.json"
        meta_path.parent.mkdir(parents=True)
        meta_path.write_text(
            json.dumps({"entries": [{"id": "other-id"}], "appliedId": "other-id"}),
            encoding="utf-8",
        )
        configure_omlx_gateway(8000, "k", home=tmp_path)
        configure_omlx_gateway(8000, "k", home=tmp_path)  # re-apply
        restore(home=tmp_path)
        meta = _read(meta_path)
        assert meta["appliedId"] == "other-id"

    def test_non_macos_noop(self, tmp_path, monkeypatch):
        _force_not_macos(monkeypatch)
        assert restore(home=tmp_path) is False
        assert is_configured(home=tmp_path) is False


class TestIsConfigured:
    def test_false_when_nothing_written(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        assert is_configured(home=tmp_path) is False

    def test_true_after_configure(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        configure_omlx_gateway(8000, "k", home=tmp_path)
        assert is_configured(home=tmp_path) is True


class TestGatewayModelFormat:
    """T-002 gateway check: /v1/models tier entries must carry the five
    Anthropic family fields ModelMap.swift::catalog() requires."""

    def test_model_info_exposes_anthropic_fields(self):
        from omlx.api.openai_models import ClaudeTierModelInfo

        entry = ClaudeTierModelInfo(
            id="claude-sonnet-5",
            owned_by="omlx",
            max_model_len=8192,
            display_name="sonnet-phys",
            created_at="2026-06-30T00:00:00Z",
            anthropic_family_tier="sonnet",
            is_family_default=True,
            max_tokens=8192,
        )
        dumped = entry.model_dump(exclude_none=True)
        for field in (
            "display_name",
            "created_at",
            "anthropic_family_tier",
            "is_family_default",
            "max_tokens",
        ):
            assert field in dumped, f"missing gateway field {field}"

    def test_standard_entries_stay_clean(self):
        from omlx.api.openai_models import ModelInfo

        dumped = ModelInfo(id="plain", owned_by="omlx").model_dump(exclude_none=True)
        for field in ("display_name", "created_at", "anthropic_family_tier"):
            assert field not in dumped


class TestAdminRoutes:
    @staticmethod
    def _patch_settings(monkeypatch, tmp_path, *, desktop: bool):
        import omlx.admin.routes as admin_routes
        from omlx.settings import GlobalSettings

        gs = GlobalSettings(base_path=tmp_path)
        gs.claude_code.desktop_enabled = desktop
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
        return admin_routes, gs

    async def test_toggle_on_configures_gateway(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        admin_routes, gs = self._patch_settings(monkeypatch, tmp_path, desktop=False)
        calls = []
        monkeypatch.setattr(
            claude_desktop,
            "configure_omlx_gateway",
            lambda port, api_key: calls.append((port, api_key)) or True,
        )
        request = admin_routes.GlobalSettingsRequest.model_validate(
            {"claude_code_desktop_enabled": True}
        )
        response = await admin_routes.update_global_settings(request, True)
        assert gs.claude_code.desktop_enabled is True
        assert calls and calls[0][0] == gs.server.port
        assert response["claude_desktop"]["applied"] is True
        assert response["claude_desktop"]["action"] == "configure"

    async def test_toggle_off_restores(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        admin_routes, gs = self._patch_settings(monkeypatch, tmp_path, desktop=True)
        calls = []
        monkeypatch.setattr(
            claude_desktop, "restore", lambda home=None: calls.append(True) or True
        )
        request = admin_routes.GlobalSettingsRequest.model_validate(
            {"claude_code_desktop_enabled": False}
        )
        response = await admin_routes.update_global_settings(request, True)
        assert gs.claude_code.desktop_enabled is False
        assert calls == [True]
        assert response["claude_desktop"] == {"applied": True, "action": "restore"}

    async def test_no_transition_no_side_effect(self, tmp_path, monkeypatch):
        admin_routes, _ = self._patch_settings(monkeypatch, tmp_path, desktop=False)
        called = []
        monkeypatch.setattr(
            claude_desktop,
            "configure_omlx_gateway",
            lambda *a, **k: called.append(True) or True,
        )
        monkeypatch.setattr(
            claude_desktop, "restore", lambda *a, **k: called.append(True) or True
        )
        request = admin_routes.GlobalSettingsRequest.model_validate(
            {"claude_code_mode": "local"}
        )
        response = await admin_routes.update_global_settings(request, True)
        assert called == []
        assert response["claude_desktop"] == {"applied": False}

    async def test_restart_flag_triggers_relaunch(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        admin_routes, _ = self._patch_settings(monkeypatch, tmp_path, desktop=False)
        monkeypatch.setattr(
            claude_desktop, "configure_omlx_gateway", lambda *a, **k: True
        )
        restarts = []
        monkeypatch.setattr(
            claude_desktop,
            "restart_claude_desktop",
            lambda: restarts.append(True) or True,
        )
        request = admin_routes.GlobalSettingsRequest.model_validate(
            {"claude_code_desktop_enabled": True, "restart_desktop": True}
        )
        await admin_routes.update_global_settings(request, True)
        assert restarts == [True]

    async def test_dedicated_endpoints(self, tmp_path, monkeypatch):
        _force_macos(monkeypatch)
        admin_routes, _ = self._patch_settings(monkeypatch, tmp_path, desktop=True)
        monkeypatch.setattr(
            claude_desktop, "configure_omlx_gateway", lambda *a, **k: True
        )
        monkeypatch.setattr(claude_desktop, "restore", lambda *a, **k: True)
        monkeypatch.setattr(claude_desktop, "is_configured", lambda *a, **k: True)

        status = await admin_routes.claude_desktop_status(True)
        assert status == {"configured": True, "profile_id": PROFILE_ID}

        response = await admin_routes.claude_desktop_configure(
            admin_routes.ClaudeDesktopConfigureRequest(), True
        )
        assert response["success"] is True
        assert response["profile_id"] == PROFILE_ID

        monkeypatch.setattr(claude_desktop, "is_configured", lambda *a, **k: False)
        response = await admin_routes.claude_desktop_restore(
            admin_routes.ClaudeDesktopRestoreRequest(), True
        )
        assert response["success"] is True
        assert response["configured"] is False


class TestAdminUI:
    def test_toggle_shows_state_and_reapply(self):
        root = Path(__file__).resolve().parents[1]
        html = (root / "omlx/admin/templates/dashboard/_status.html").read_text(
            encoding="utf-8"
        )
        assert "claudeDesktop.configured" in html
        assert "reapplyClaudeDesktopConfig()" in html
        assert "status.claude_code.desktop_side_effect" in html
        js = (root / "omlx/admin/static/js/dashboard.js").read_text(encoding="utf-8")
        assert "fetchClaudeDesktopStatus" in js
        assert "/admin/api/claude-desktop/status" in js
        assert "/admin/api/claude-desktop/configure" in js
        assert "/admin/api/claude-desktop/restore" in js

    def test_i18n_keys_present_in_every_locale(self):
        root = Path(__file__).resolve().parents[1]
        locales = sorted((root / "omlx/admin/i18n").glob("*.json"))
        assert locales, "no locale files found"
        wanted = {
            "status.claude_code.desktop_configured",
            "status.claude_code.desktop_not_configured",
            "status.claude_code.desktop_side_effect",
            "status.claude_code.desktop_reapply",
            "status.claude_code.desktop_working",
        }
        for locale_path in locales:
            data = json.loads(locale_path.read_text(encoding="utf-8"))
            missing = wanted - set(data)
            assert not missing, f"{locale_path.name} missing {sorted(missing)}"
            assert data["status.claude_code.desktop_side_effect"].strip()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only module")
class TestMacOSOnly:
    def test_real_platform_gate(self):
        # On macOS the real gate passes; the no-op paths are covered above
        # by forcing _is_macos off.
        assert claude_desktop._is_macos() is True
