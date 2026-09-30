# SPDX-License-Identifier: Apache-2.0
"""Tests for Claude Desktop tier metadata on native IDs.

When ``ClaudeCodeSettings.desktop_enabled`` is set, oMLX annotates the
native tier models in /v1/models with Anthropic family metadata — no
synthetic slot IDs are exposed (the client matches on the metadata, not
the ID):

- ``claude_code.opus_model`` carries family ``opus``
- ``claude_code.sonnet_model`` carries family ``sonnet``
- ``claude_code.haiku_model`` carries family ``haiku``
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from omlx.engine_pool import build_claude_tier_aliases
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.settings import ClaudeCodeSettings, GlobalSettings

OPUS_SLOT = "claude-opus-5"
SONNET_SLOT = "claude-sonnet-5"
HAIKU_SLOT = "claude-haiku-4-5-20251001"


def _claude_settings(**kwargs) -> ClaudeCodeSettings:
    kwargs.setdefault("desktop_enabled", True)
    kwargs.setdefault("opus_model", "opus-phys")
    kwargs.setdefault("sonnet_model", "sonnet-phys")
    kwargs.setdefault("haiku_model", "haiku-phys")
    return ClaudeCodeSettings(**kwargs)


class TestBuildClaudeTierAliases:
    def test_builds_native_map_when_enabled(self):
        aliases = build_claude_tier_aliases(_claude_settings())
        assert aliases == {
            "opus-phys": "opus",
            "sonnet-phys": "sonnet",
            "haiku-phys": "haiku",
        }

    def test_empty_when_disabled(self):
        settings = _claude_settings(desktop_enabled=False)
        assert build_claude_tier_aliases(settings) == {}

    def test_unconfigured_tier_omitted(self):
        settings = _claude_settings(opus_model=None, haiku_model="")
        aliases = build_claude_tier_aliases(settings)
        assert aliases == {"sonnet-phys": "sonnet"}

    def test_none_settings(self):
        assert build_claude_tier_aliases(None) == {}

    def test_duplicate_model_keeps_priority_and_warns(self, caplog):
        import logging

        settings = _claude_settings(opus_model="shared-phys", sonnet_model="shared-phys")
        with caplog.at_level(logging.WARNING, logger="omlx.engine_pool"):
            aliases = build_claude_tier_aliases(settings)
        assert aliases["shared-phys"] == "opus"
        assert aliases["haiku-phys"] == "haiku"
        assert any("shared-phys" in r.message for r in caplog.records)


class TestResolveNativeIds:
    """Resolution: native IDs resolve naturally, slot IDs never resolve."""

    @staticmethod
    def _pool_with_entries(monkeypatch, entries: list[str]):
        from omlx.engine_pool import EnginePool

        pool = EnginePool.__new__(EnginePool)
        pool._entries = {mid: object() for mid in entries}
        pool._cluster_registry = None
        return pool

    @staticmethod
    def _settings_manager() -> MagicMock:
        manager = MagicMock()
        manager.get_exposed_profile_source_model_id.return_value = None
        manager.get_all_settings.return_value = {}
        return manager

    def test_native_id_resolves_to_itself(self, monkeypatch):
        pool = self._pool_with_entries(
            monkeypatch, ["opus-phys", "sonnet-phys", "haiku-phys"]
        )
        assert (
            pool.resolve_model_id("sonnet-phys", self._settings_manager())
            == "sonnet-phys"
        )
        assert (
            pool.resolve_model_id("opus-phys", self._settings_manager()) == "opus-phys"
        )
        assert (
            pool.resolve_model_id("haiku-phys", self._settings_manager())
            == "haiku-phys"
        )

    def test_slot_id_never_resolves(self, monkeypatch):
        # Legacy synthetic slot IDs are not models: they pass through
        # unresolved even when tiers are configured.
        pool = self._pool_with_entries(
            monkeypatch, ["opus-phys", "sonnet-phys", "haiku-phys"]
        )
        build_claude_tier_aliases(_claude_settings())  # tiers configured
        assert (
            pool.resolve_model_id(SONNET_SLOT, self._settings_manager())
            == SONNET_SLOT
        )
        assert pool.resolve_model_id(OPUS_SLOT, self._settings_manager()) == OPUS_SLOT
        assert pool.resolve_model_id(HAIKU_SLOT, self._settings_manager()) == HAIKU_SLOT

    def test_unknown_id_resolves_to_self(self, monkeypatch):
        pool = self._pool_with_entries(monkeypatch, ["sonnet-phys"])
        assert (
            pool.resolve_model_id("no-such-model", self._settings_manager())
            == "no-such-model"
        )

    def test_custom_alias_still_wins(self, monkeypatch):
        pool = self._pool_with_entries(monkeypatch, ["model-a", "sonnet-phys"])
        manager = self._settings_manager()
        manager.get_all_settings.return_value = {
            "model-a": ModelSettings(model_alias="my-alias"),
        }
        assert pool.resolve_model_id("my-alias", manager) == "model-a"

    def test_exact_entry_still_wins(self, monkeypatch):
        pool = self._pool_with_entries(monkeypatch, [SONNET_SLOT, "sonnet-phys"])
        # A real model directory literally named like a legacy slot resolves
        # as itself — not via any tier mapping.
        assert (
            pool.resolve_model_id(SONNET_SLOT, self._settings_manager())
            == SONNET_SLOT
        )

    def test_provider_prefix_strip_still_works(self, monkeypatch):
        pool = self._pool_with_entries(monkeypatch, ["sonnet-phys"])
        assert (
            pool.resolve_model_id("omlx/sonnet-phys", self._settings_manager())
            == "sonnet-phys"
        )


class _Pool:
    """Engine pool stub exposing a fixed model list via get_status()."""

    def __init__(self, models: list[dict]):
        self._models = models

    def resolve_model_id(
        self, model_id_or_alias: str, settings_manager
    ) -> str:
        return model_id_or_alias

    def get_status(self) -> dict:
        return {
            "final_ceiling": 0,
            "current_model_memory": 0,
            "model_count": len(self._models),
            "loaded_count": 0,
            "models": self._models,
        }


def _model(model_id: str) -> dict:
    return {
        "id": model_id,
        "model_path": f"/models/{model_id}",
        "config_model_type": "llama",
        "source_repo_id": None,
    }


def _state(models: list[dict], tmp_path, *, desktop: bool):
    import omlx.server as server_module

    state = server_module.ServerState()
    state.engine_pool = _Pool(models)
    state.settings_manager = ModelSettingsManager(base_path=tmp_path)
    gs = GlobalSettings()
    gs.claude_code.opus_model = "opus-phys"
    gs.claude_code.sonnet_model = "sonnet-phys"
    gs.claude_code.haiku_model = "haiku-phys"
    gs.claude_code.desktop_enabled = desktop
    state.global_settings = gs
    return state


def _list_models(state) -> list[dict]:
    import omlx.server as server_module

    with (
        patch("omlx.server._server_state", state),
        patch("omlx.server.get_max_context_window", return_value=8192),
    ):
        client = TestClient(server_module.app, raise_server_exceptions=False)
        response = client.get("/v1/models")
    assert response.status_code == 200
    return response.json()["data"]


def _models_by_id(entries: list[dict]) -> dict[str, dict]:
    return {m["id"]: m for m in entries}


class TestListModelsTierAliases:
    def test_absent_when_disabled(self, tmp_path):
        models = [_model("opus-phys"), _model("sonnet-phys"), _model("haiku-phys")]
        data = _list_models(_state(models, tmp_path, desktop=False))
        ids = [m["id"] for m in data]
        assert OPUS_SLOT not in ids
        assert SONNET_SLOT not in ids
        assert HAIKU_SLOT not in ids
        # Standard OpenAI entries carry no Anthropic metadata.
        for entry in data:
            assert "display_name" not in entry
            assert "anthropic_family_tier" not in entry

    def test_native_models_annotated_no_slot_ids_when_enabled(self, tmp_path):
        models = [_model("opus-phys"), _model("sonnet-phys"), _model("haiku-phys")]
        by_id = _models_by_id(_list_models(_state(models, tmp_path, desktop=True)))
        # No synthetic slot IDs are ever added to the listing.
        assert OPUS_SLOT not in by_id
        assert SONNET_SLOT not in by_id
        assert HAIKU_SLOT not in by_id
        # Native tier models carry the correct family metadata.
        assert by_id["opus-phys"]["display_name"]
        assert by_id["opus-phys"]["anthropic_family_tier"] == "opus"
        assert by_id["sonnet-phys"]["display_name"]
        assert by_id["sonnet-phys"]["anthropic_family_tier"] == "sonnet"
        assert by_id["haiku-phys"]["display_name"]
        assert by_id["haiku-phys"]["anthropic_family_tier"] == "haiku"
        for native_id in ("opus-phys", "sonnet-phys", "haiku-phys"):
            entry = by_id[native_id]
            assert entry["owned_by"] == "omlx"
            assert entry["is_family_default"] is True
            assert entry["max_model_len"] == 8192
            assert entry["max_tokens"] is not None
            assert entry["created_at"]

    def test_unassigned_model_has_no_metadata(self, tmp_path):
        models = [_model("sonnet-phys"), _model("plain-model")]
        by_id = _models_by_id(_list_models(_state(models, tmp_path, desktop=True)))
        assert by_id["sonnet-phys"]["anthropic_family_tier"] == "sonnet"
        assert by_id["sonnet-phys"]["is_family_default"] is True
        assert "anthropic_family_tier" not in by_id["plain-model"]
        assert "display_name" not in by_id["plain-model"]
        assert "is_family_default" not in by_id["plain-model"]

    def test_unconfigured_tier_not_annotated(self, tmp_path):
        models = [_model("sonnet-phys")]
        state = _state(models, tmp_path, desktop=True)
        state.global_settings.claude_code.opus_model = None
        state.global_settings.claude_code.haiku_model = None
        by_id = _models_by_id(_list_models(state))
        assert OPUS_SLOT not in by_id
        assert HAIKU_SLOT not in by_id
        assert by_id["sonnet-phys"]["display_name"]
        assert by_id["sonnet-phys"]["anthropic_family_tier"] == "sonnet"

    def test_undiscovered_tier_model_warns_and_skips(self, tmp_path, caplog):
        # Tier points at a model that is not in the listing: warn, add nothing.
        models = [_model("sonnet-phys")]
        with caplog.at_level("WARNING", logger="omlx.server"):
            by_id = _models_by_id(_list_models(_state(models, tmp_path, desktop=True)))
        assert OPUS_SLOT not in by_id
        assert HAIKU_SLOT not in by_id
        assert SONNET_SLOT not in by_id
        assert by_id["sonnet-phys"]["anthropic_family_tier"] == "sonnet"
        assert any("opus-phys" in r.message for r in caplog.records)

    def test_duplicate_tier_model_uses_priority(self, tmp_path, caplog):
        import logging

        models = [_model("shared-phys"), _model("haiku-phys")]
        state = _state(models, tmp_path, desktop=True)
        state.global_settings.claude_code.opus_model = "shared-phys"
        state.global_settings.claude_code.sonnet_model = "shared-phys"
        state.global_settings.claude_code.haiku_model = "haiku-phys"
        with caplog.at_level(logging.WARNING):
            by_id = _models_by_id(_list_models(state))
        # First tier in opus, sonnet, haiku order wins.
        assert by_id["shared-phys"]["anthropic_family_tier"] == "opus"
        assert by_id["shared-phys"]["is_family_default"] is True
        assert by_id["haiku-phys"]["anthropic_family_tier"] == "haiku"
        assert OPUS_SLOT not in by_id
        assert SONNET_SLOT not in by_id
        assert any("shared-phys" in r.message for r in caplog.records)


class TestServerResolveTierAliases:
    def test_native_ids_pass_through(self, tmp_path):
        import omlx.server as server_module

        state = _state([_model("sonnet-phys")], tmp_path, desktop=True)
        with patch("omlx.server._server_state", state):
            assert server_module.resolve_model_id("sonnet-phys") == "sonnet-phys"
            assert server_module.get_claude_tier_aliases() == {
                "opus-phys": "opus",
                "sonnet-phys": "sonnet",
                "haiku-phys": "haiku",
            }

    def test_slot_ids_never_resolve(self, tmp_path):
        import omlx.server as server_module

        state = _state([_model("sonnet-phys")], tmp_path, desktop=True)
        with patch("omlx.server._server_state", state):
            assert server_module.resolve_model_id(SONNET_SLOT) == SONNET_SLOT
            assert server_module.resolve_model_id(OPUS_SLOT) == OPUS_SLOT

    def test_disabled_resolves_to_self(self, tmp_path):
        import omlx.server as server_module

        state = _state([_model("sonnet-phys")], tmp_path, desktop=False)
        with patch("omlx.server._server_state", state):
            assert server_module.resolve_model_id(SONNET_SLOT) == SONNET_SLOT
            assert server_module.get_claude_tier_aliases() == {}


class TestSettingsRoundTrip:
    def test_desktop_enabled_defaults_false(self):
        assert ClaudeCodeSettings().desktop_enabled is False
        assert ClaudeCodeSettings.from_dict({}).desktop_enabled is False

    def test_to_from_dict(self):
        settings = _claude_settings()
        assert ClaudeCodeSettings.from_dict(settings.to_dict()) == settings

    def test_persists_in_settings_json(self, tmp_path):
        gs = GlobalSettings(base_path=tmp_path)
        gs.claude_code.desktop_enabled = True
        gs.claude_code.sonnet_model = "sonnet-phys"
        gs.save()
        raw = json.loads((tmp_path / "settings.json").read_text())
        assert raw["claude_code"]["desktop_enabled"] is True
        reloaded = GlobalSettings.load(base_path=tmp_path)
        assert reloaded.claude_code.desktop_enabled is True
        assert reloaded.claude_code.sonnet_model == "sonnet-phys"

    def test_missing_key_loads_false(self, tmp_path):
        gs = GlobalSettings(base_path=tmp_path)
        gs.save()
        raw = json.loads((tmp_path / "settings.json").read_text())
        del raw["claude_code"]["desktop_enabled"]
        (tmp_path / "settings.json").write_text(json.dumps(raw))
        assert (
            GlobalSettings.load(base_path=tmp_path).claude_code.desktop_enabled is False
        )


class TestAdminRouteAndUI:
    def test_request_field_present_in_model_fields_set(self):
        from omlx.admin.routes import GlobalSettingsRequest

        r = GlobalSettingsRequest.model_validate({"claude_code_desktop_enabled": True})
        assert "claude_code_desktop_enabled" in r.model_fields_set
        assert r.claude_code_desktop_enabled is True
        assert (
            "claude_code_desktop_enabled"
            not in GlobalSettingsRequest().model_fields_set
        )

    async def test_post_handler_persists_desktop_enabled(self, tmp_path):
        import omlx.admin.routes as admin_routes
        import omlx.server  # noqa: F401 — ensure server import first (set_admin_getters)
        from omlx.admin.routes import GlobalSettingsRequest

        gs = GlobalSettings(base_path=tmp_path)
        assert gs.claude_code.desktop_enabled is False
        original = admin_routes._get_global_settings
        admin_routes._get_global_settings = lambda: gs
        try:
            request = GlobalSettingsRequest.model_validate(
                {"claude_code_desktop_enabled": True}
            )
            response = await admin_routes.update_global_settings(request, True)
        finally:
            admin_routes._get_global_settings = original
        assert gs.claude_code.desktop_enabled is True
        assert "claude_code" in response["runtime_applied"]

    def test_toggle_markup_and_payload(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        html = (root / "omlx/admin/templates/dashboard/blocks/_claude_code.html").read_text(
            encoding="utf-8"
        )
        assert "globalSettings.claude_code.desktop_enabled" in html
        assert "status.claude_code.desktop" in html
        assert "status.claude_code.desktop_hint" in html
        js = (root / "omlx/admin/static/js/dashboard.js").read_text(encoding="utf-8")
        assert "claude_code_desktop_enabled" in js

    def test_i18n_keys_present_in_every_locale(self):
        from pathlib import Path

        i18n_dir = Path(__file__).resolve().parents[1] / "omlx/admin/i18n"
        locales = sorted(i18n_dir.glob("*.json"))
        assert locales, "no locale files found"
        for locale_path in locales:
            data = json.loads(locale_path.read_text(encoding="utf-8"))
            assert "status.claude_code.desktop" in data, locale_path.name
            assert "status.claude_code.desktop_hint" in data, locale_path.name


class TestDesktopEnabledGetRoundTrip:
    def test_get_includes_desktop_enabled(self, tmp_path, monkeypatch):
        import asyncio

        import omlx.admin.routes as admin_routes

        gs = GlobalSettings(base_path=tmp_path)
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
        result = asyncio.run(admin_routes.get_global_settings(is_admin=True))
        assert "desktop_enabled" in result["claude_code"]
        assert result["claude_code"]["desktop_enabled"] is False
        gs.claude_code.desktop_enabled = True
        result = asyncio.run(admin_routes.get_global_settings(is_admin=True))
        assert result["claude_code"]["desktop_enabled"] is True

    async def test_patch_true_then_get_true(self, tmp_path, monkeypatch):
        import omlx.admin.routes as admin_routes
        import omlx.server  # noqa: F401 — ensure server import first (set_admin_getters)
        from omlx.admin.routes import GlobalSettingsRequest
        from omlx.integrations import claude_desktop

        gs = GlobalSettings(base_path=tmp_path)
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
        monkeypatch.setattr(
            claude_desktop, "configure_omlx_gateway", lambda *a, **k: True
        )
        monkeypatch.setattr(claude_desktop, "restore", lambda *a, **k: True)
        request = GlobalSettingsRequest.model_validate(
            {"claude_code_desktop_enabled": True}
        )
        await admin_routes.update_global_settings(request, True)
        assert gs.claude_code.desktop_enabled is True
        result = await admin_routes.get_global_settings(is_admin=True)
        assert result["claude_code"]["desktop_enabled"] is True

    async def test_restore_persist_false_triggers_no_configure(self, tmp_path, monkeypatch):
        import omlx.admin.routes as admin_routes
        import omlx.server  # noqa: F401 — ensure server import first (set_admin_getters)
        from omlx.admin.routes import GlobalSettingsRequest
        from omlx.integrations import claude_desktop

        gs = GlobalSettings(base_path=tmp_path)
        gs.claude_code.desktop_enabled = True
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
        calls = {"configure": [], "restore": []}
        monkeypatch.setattr(
            claude_desktop,
            "configure_omlx_gateway",
            lambda *a, **k: calls["configure"].append(True) or True,
        )
        monkeypatch.setattr(
            claude_desktop, "restore", lambda *a, **k: calls["restore"].append(True) or True
        )
        request = GlobalSettingsRequest.model_validate(
            {"claude_code_desktop_enabled": False}
        )
        response = await admin_routes.update_global_settings(request, True)
        assert gs.claude_code.desktop_enabled is False
        assert calls["configure"] == []
        assert calls["restore"] == [True]
        assert response["claude_desktop"] == {"applied": True, "action": "restore"}
        # Second persist while already off must not re-trigger any side effect.
        calls["restore"].clear()
        request = GlobalSettingsRequest.model_validate(
            {"claude_code_desktop_enabled": False}
        )
        response = await admin_routes.update_global_settings(request, True)
        assert calls["configure"] == []
        assert calls["restore"] == []
        assert response["claude_desktop"] == {"applied": False}


class TestRestoreButton:
    def test_restore_markup_and_handler(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        html = (root / "omlx/admin/templates/dashboard/blocks/_claude_code.html").read_text(
            encoding="utf-8"
        )
        assert "restoreClaudeDesktopConfig()" in html
        assert "status.claude_code.desktop_restore" in html
        assert "status.claude_code.desktop_restoring" in html
        js = (root / "omlx/admin/static/js/dashboard.js").read_text(encoding="utf-8")
        assert "async restoreClaudeDesktopConfig()" in js
        assert "/admin/api/claude-desktop/restore" in js
        assert "desktop_enabled = false" in js
        assert "desktop_restore_success" in js
        assert "desktop_restore_error" in js

    def test_restore_i18n_keys_present_in_every_locale(self):
        from pathlib import Path

        i18n_dir = Path(__file__).resolve().parents[1] / "omlx/admin/i18n"
        locales = sorted(i18n_dir.glob("*.json"))
        assert locales, "no locale files found"
        wanted = {
            "status.claude_code.desktop_restore",
            "status.claude_code.desktop_restoring",
            "status.claude_code.desktop_restore_success",
            "status.claude_code.desktop_restore_error",
        }
        for locale_path in locales:
            data = json.loads(locale_path.read_text(encoding="utf-8"))
            missing = wanted - set(data)
            assert not missing, f"{locale_path.name} missing {sorted(missing)}"
            for key in wanted:
                assert data[key].strip(), f"{locale_path.name} {key} empty"
