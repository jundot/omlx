# SPDX-License-Identifier: Apache-2.0
"""Tests for admin profile/template API routes."""

import base64
import json
import re
import zlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import recipe as settings_recipe
from omlx.admin import routes as admin_routes
from omlx.model_settings import ModelSettings, ModelSettingsManager


class _FakeEntry:
    def __init__(
        self,
        model_id: str,
        *,
        engine_type: str = "batched",
        model_type: str = "llm",
        config_model_type: str | None = None,
    ):
        self.engine_type = engine_type
        self.model_type = model_type
        self.config_model_type = config_model_type
        self.engine = None
        self.is_pinned = False
        self.is_loading = False
        self.load_failed = False
        self.load_failure_message = None
        self.load_failure_at = None
        self.model_path = "/fake"


class _FakePool:
    def __init__(self):
        self._entries = {"model-a": _FakeEntry("model-a")}

    def get_entry(self, model_id):
        return self._entries.get(model_id)

    def get_status(self):
        return {
            "models": [
                {
                    "id": "model-a",
                    "loaded": False,
                    "pinned": False,
                    "engine_type": "batched",
                    "model_type": "llm",
                }
            ]
        }

    def get_model_ids(self):
        return list(self._entries)

    def _engine_runtime_signature(self, model_id, runtime_settings=None):
        return ()

    @staticmethod
    def _clear_load_failure(entry):
        entry.load_failed = False
        entry.load_failure_message = None
        entry.load_failure_at = None


class _FakeServerState:
    default_model = None


@pytest.fixture
def client(tmp_path, monkeypatch):
    mgr = ModelSettingsManager(tmp_path)
    pool = _FakePool()
    state = _FakeServerState()

    # Patch the module-level getters
    admin_routes._get_settings_manager = lambda: mgr
    admin_routes._get_engine_pool = lambda: pool
    admin_routes._get_server_state = lambda: state
    admin_routes._get_global_settings = lambda: None

    # Bypass auth
    async def _fake_require_admin():
        return True

    from omlx.admin import auth as admin_auth

    monkeypatch.setattr(admin_auth, "require_admin", _fake_require_admin)

    # Also patch on the router dependency
    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app), mgr


class TestProfileRoutes:
    def test_list_profiles_empty(self, client):
        c, _ = client
        r = c.get("/admin/api/models/model-a/profiles")
        assert r.status_code == 200
        assert r.json() == {"profiles": []}

    def test_create_and_list_profile(self, client):
        c, _ = client
        r = c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0, "is_pinned": True},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["profile"]["name"] == "coding"
        assert body["profile"]["api_name"] == "coding"
        assert "is_pinned" not in body["profile"]["settings"]

        r = c.get("/admin/api/models/model-a/profiles")
        assert len(r.json()["profiles"]) == 1

    def test_create_duplicate_conflicts(self, client):
        c, _ = client
        payload = {"name": "coding", "display_name": "C", "settings": {}}
        r1 = c.post("/admin/api/models/model-a/profiles", json=payload)
        assert r1.status_code == 200
        r2 = c.post("/admin/api/models/model-a/profiles", json=payload)
        assert r2.status_code == 409

    def test_create_invalid_name_400(self, client):
        c, _ = client
        r = c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "Has Space",
                "display_name": "x",
                "settings": {},
            },
        )
        assert r.status_code == 400

    def test_update_profile(self, client):
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        r = c.put(
            "/admin/api/models/model-a/profiles/coding",
            json={
                "display_name": "Coding v2",
                "settings": {"temperature": 0.2},
            },
        )
        assert r.status_code == 200
        assert r.json()["profile"]["display_name"] == "Coding v2"
        assert r.json()["profile"]["api_name"] == "coding"
        assert r.json()["profile"]["settings"]["temperature"] == 0.2

    def test_update_profile_preserves_stored_fields(self, client):
        """A settings patch must not drop stored keys it does not carry.

        Regression: update used to replace the settings record wholesale, so
        a partial update wiped engine-scope fields the client never sent.
        """
        c, mgr = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "p",
                "display_name": "P",
                "settings": {
                    "expert_streaming_io_depth": 8,
                    "cache_reasoning_output": True,
                    "temperature": 0.0,
                },
            },
        )
        r = c.put(
            "/admin/api/models/model-a/profiles/p",
            json={"settings": {"temperature": 0.2}},
        )
        assert r.status_code == 200
        settings = r.json()["profile"]["settings"]
        assert settings["expert_streaming_io_depth"] == 8
        assert settings["cache_reasoning_output"] is True
        assert settings["temperature"] == 0.2
        stored = mgr.get_profile("model-a", "p")["settings"]
        assert stored["expert_streaming_io_depth"] == 8
        assert stored["cache_reasoning_output"] is True

    def test_update_profile_api_name(self, client):
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "p-abc",
                "display_name": "Fast Chat",
                "settings": {"temperature": 0.0},
            },
        )
        r = c.put(
            "/admin/api/models/model-a/profiles/p-abc",
            json={
                "api_name": "fast-chat-api",
            },
        )
        assert r.status_code == 200
        assert r.json()["profile"]["name"] == "p-abc"
        assert r.json()["profile"]["api_name"] == "fast-chat-api"

    def test_rename_invalid_name_400(self, client):
        """Renaming to a non-slug is rejected, like creation."""
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "coding",
                "settings": {},
            },
        )
        r = c.put(
            "/admin/api/models/model-a/profiles/coding", json={"new_name": "Has Space"}
        )
        assert r.status_code == 400

    def test_delete_profile(self, client):
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {},
            },
        )
        r = c.delete("/admin/api/models/model-a/profiles/coding")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_delete_missing_404(self, client):
        c, _ = client
        r = c.delete("/admin/api/models/model-a/profiles/nope")
        assert r.status_code == 404

    def test_apply_profile_sets_active(self, client):
        c, mgr = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        r = c.post("/admin/api/models/model-a/profiles/coding/apply")
        assert r.status_code == 200
        assert r.json()["settings"]["active_profile_name"] == "coding"

    def test_apply_profile_resolves_vlm_mtp_processor_conflict(self, client):
        c, mgr = client
        mgr.set_settings(
            "model-a",
            ModelSettings(
                vlm_mtp_enabled=True,
                vlm_mtp_draft_model="qwen-mtp-drafter",
            ),
        )
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "penalty",
                "display_name": "Penalty",
                "settings": {"presence_penalty": 1.5},
            },
        )

        r = c.post("/admin/api/models/model-a/profiles/penalty/apply")

        assert r.status_code == 200, r.text
        settings = r.json()["settings"]
        assert settings["presence_penalty"] == 1.5
        assert settings["vlm_mtp_enabled"] is False
        assert settings["active_profile_name"] == "penalty"
        assert mgr.get_settings("model-a").vlm_mtp_enabled is False

    def test_apply_profile_validation_error_is_400_without_partial_write(self, client):
        c, mgr = client
        mgr.set_settings(
            "model-a",
            ModelSettings(
                vlm_mtp_enabled=True,
                vlm_mtp_draft_model="qwen-mtp-drafter",
            ),
        )
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "dflash",
                "display_name": "DFlash",
                "settings": {"dflash_enabled": True},
            },
        )

        r = c.post("/admin/api/models/model-a/profiles/dflash/apply")

        assert r.status_code == 400
        assert "vlm_mtp_enabled and dflash_enabled" in r.json()["detail"]
        persisted = mgr.get_settings("model-a")
        assert persisted.vlm_mtp_enabled is True
        assert persisted.dflash_enabled is False
        assert persisted.active_profile_name is None

    def test_apply_profile_sanitizes_diffusion_unsupported_settings(self, client):
        c, mgr = client
        pool = admin_routes._get_engine_pool()
        pool._entries["diffusion"] = _FakeEntry(
            "diffusion",
            engine_type="vlm",
            model_type="vlm",
            config_model_type="diffusion_gemma",
        )
        c.post(
            "/admin/api/models/diffusion/profiles",
            json={
                "name": "fast",
                "display_name": "Fast",
                "settings": {
                    "temperature": 0.0,
                    "top_p": 0.5,
                    "guided_grammar_enabled": True,
                    "guided_grammar": 'root ::= "YES"',
                    "max_tool_result_tokens": 4096,
                    "turboquant_kv_enabled": True,
                    "specprefill_enabled": True,
                    "dflash_enabled": True,
                    "mtp_enabled": True,
                    "vlm_mtp_enabled": True,
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "custom_key": "ok",
                    },
                    "forced_ct_kwargs": ["enable_thinking", "custom_key"],
                },
            },
        )

        r = c.post("/admin/api/models/diffusion/profiles/fast/apply")
        assert r.status_code == 200, r.text
        settings = r.json()["settings"]
        assert settings["temperature"] == 0.0
        assert "top_p" not in settings
        assert settings["guided_grammar_enabled"] is False
        assert "guided_grammar" not in settings
        # Tool calling works on the diffusion lane (prompt-driven +
        # output parsing), so its settings are preserved.
        assert settings["max_tool_result_tokens"] == 4096
        assert settings["turboquant_kv_enabled"] is False
        assert settings["specprefill_enabled"] is False
        assert settings["dflash_enabled"] is False
        assert settings["mtp_enabled"] is False
        assert settings["vlm_mtp_enabled"] is False
        assert settings["chat_template_kwargs"] == {"custom_key": "ok"}
        assert settings["forced_ct_kwargs"] == ["custom_key"]

    def test_apply_missing_404(self, client):
        c, _ = client
        r = c.post("/admin/api/models/model-a/profiles/nope/apply")
        assert r.status_code == 404

    def test_get_profile_fields(self, client):
        c, _ = client
        r = c.get("/admin/api/profile-fields")
        assert r.status_code == 200
        data = r.json()
        assert "universal" in data
        assert "model_specific" in data
        assert "temperature" in data["universal"]
        assert "turboquant_kv_enabled" in data["model_specific"]

    def test_also_save_as_template(self, client):
        c, mgr = client
        r = c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0, "turboquant_kv_enabled": True},
                "also_save_as_template": True,
            },
        )
        assert r.status_code == 200
        tmpl = mgr.get_template("coding")
        assert tmpl is not None
        assert tmpl["settings"] == {"temperature": 0.0}


class TestApplySnapshotSemantics:
    """Applying a profile is authoritative over universal fields: absent
    keys reset to defaults. Profile updates merge into the stored record
    (a settings patch never drops stored keys), so the snapshot that apply
    sees is the union of everything ever saved into the profile.
    Model-specific and excluded fields are not reset."""

    def test_apply_resets_universal_fields_removed_from_profile(self, client):
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "p",
                "display_name": "P",
                "settings": {
                    "max_context_window": 8192,
                    "max_tokens": 512,
                    "temperature": 0.7,
                },
            },
        )
        r = c.post("/admin/api/models/model-a/profiles/p/apply")
        assert r.json()["settings"]["max_tokens"] == 512

        # Update merges into the stored record: keys absent from the patch
        # survive, and apply then snapshots the full merged set.
        r = c.put(
            "/admin/api/models/model-a/profiles/p",
            json={"settings": {"temperature": 0.7}},
        )
        assert r.status_code == 200, r.text
        r = c.post("/admin/api/models/model-a/profiles/p/apply")
        settings = r.json()["settings"]
        assert settings["temperature"] == 0.7
        assert settings["max_context_window"] == 8192
        assert settings["max_tokens"] == 512

    def test_apply_clears_kwargs_absent_from_applied_profile(self, client):
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "p",
                "display_name": "P",
                "settings": {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "forced_ct_kwargs": ["enable_thinking"],
                },
            },
        )
        r = c.post("/admin/api/models/model-a/profiles/p/apply")
        assert r.json()["settings"]["chat_template_kwargs"] == {"enable_thinking": True}

        # Applying a different profile without the kwargs still snapshots:
        # fields the applied profile does not carry reset to defaults.
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "p2",
                "display_name": "P2",
                "settings": {"temperature": 0.5},
            },
        )
        r = c.post("/admin/api/models/model-a/profiles/p2/apply")
        settings = r.json()["settings"]
        assert "chat_template_kwargs" not in settings
        assert "forced_ct_kwargs" not in settings

    def test_profile_settings_sanitized_on_create_and_update(self, client):
        c, mgr = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "p",
                "display_name": "P",
                "settings": {"temperature": 0.5, "max_tokens": "", "top_p": None},
            },
        )
        assert mgr.get_profile("model-a", "p")["settings"] == {"temperature": 0.5}

        c.put(
            "/admin/api/models/model-a/profiles/p",
            json={"settings": {"top_p": 0.9, "max_context_window": ""}},
        )
        # Update merges: the patch's None/"" values are dropped and the
        # previously stored temperature survives.
        assert mgr.get_profile("model-a", "p")["settings"] == {
            "temperature": 0.5,
            "top_p": 0.9,
        }

    def test_apply_preserves_excluded_and_model_specific_settings(self, client):
        c, mgr = client
        mgr.set_settings(
            "model-a",
            ModelSettings(ttl_seconds=300, model_alias="keep-me", dflash_enabled=True),
        )
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "p",
                "display_name": "P",
                "settings": {"temperature": 0.7},
            },
        )
        r = c.post("/admin/api/models/model-a/profiles/p/apply")
        settings = r.json()["settings"]
        assert settings["ttl_seconds"] == 300
        assert settings["model_alias"] == "keep-me"
        assert settings["dflash_enabled"] is True


def test_all_model_settings_fields_classified():
    from dataclasses import fields

    from omlx.model_profiles import (
        EXCLUDED_FROM_PROFILES,
        MODEL_SPECIFIC_PROFILE_FIELDS,
        UNIVERSAL_PROFILE_FIELDS,
    )
    from omlx.model_settings import ModelSettings

    universal = set(UNIVERSAL_PROFILE_FIELDS)
    model_specific = set(MODEL_SPECIFIC_PROFILE_FIELDS)
    excluded = set(EXCLUDED_FROM_PROFILES)
    assert len(UNIVERSAL_PROFILE_FIELDS) == len(universal)
    assert len(MODEL_SPECIFIC_PROFILE_FIELDS) == len(model_specific)
    assert not (universal & model_specific)
    assert not (universal & excluded)
    assert not (model_specific & excluded)
    assert "preserve_thinking" in universal
    assert "preserve_thinking" not in excluded

    classified = universal | model_specific | excluded
    all_fields = {f.name for f in fields(ModelSettings)}
    missing = all_fields - classified
    assert not missing, (
        f"New ModelSettings field(s) {missing} must be classified in "
        f"UNIVERSAL_PROFILE_FIELDS, MODEL_SPECIFIC_PROFILE_FIELDS, or "
        f"EXCLUDED_FROM_PROFILES. If unsure, add to EXCLUDED_FROM_PROFILES."
    )
    stale = classified - all_fields
    assert not stale, (
        f"Stale entries {stale} reference removed ModelSettings fields. "
        f"Remove them from UNIVERSAL_PROFILE_FIELDS, "
        f"MODEL_SPECIFIC_PROFILE_FIELDS, and/or EXCLUDED_FROM_PROFILES."
    )


class TestTemplateRoutes:
    def test_list_empty(self, client):
        c, _ = client
        r = c.get("/admin/api/profile-templates")
        assert r.status_code == 200
        assert r.json() == {"templates": []}

    def test_create_list_get(self, client):
        c, _ = client
        r = c.post(
            "/admin/api/profile-templates",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0, "turboquant_kv_enabled": True},
            },
        )
        assert r.status_code == 200
        # Model-specific field filtered out
        assert r.json()["template"]["settings"] == {"temperature": 0.0}

    def test_duplicate_conflicts(self, client):
        c, _ = client
        c.post(
            "/admin/api/profile-templates",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        r = c.post(
            "/admin/api/profile-templates",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.1},
            },
        )
        assert r.status_code == 409

    def test_update_delete(self, client):
        c, _ = client
        c.post(
            "/admin/api/profile-templates",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        r = c.put(
            "/admin/api/profile-templates/coding", json={"display_name": "Coding v2"}
        )
        assert r.status_code == 200
        assert r.json()["template"]["display_name"] == "Coding v2"
        r = c.delete("/admin/api/profile-templates/coding")
        assert r.status_code == 200
        assert r.json()["deleted"] is True


def test_request_models_import():
    from omlx.admin.routes import (
        CreateProfileRequest,
    )

    # Minimal round-trip
    req = CreateProfileRequest(
        name="coding",
        display_name="Coding",
        description=None,
        settings={"temperature": 0.0},
        also_save_as_template=False,
    )
    assert req.name == "coding"


class TestModelsResponseActiveProfile:
    def test_active_profile_surfaces_in_list_models(self, client):
        c, mgr = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        c.post("/admin/api/models/model-a/profiles/coding/apply")
        r = c.get("/admin/api/models")
        assert r.status_code == 200
        models = r.json()["models"]
        entry = next(m for m in models if m["id"] == "model-a")
        assert entry["settings"]["active_profile_name"] == "coding"

    def test_guided_grammar_surfaces_in_list_models(self, client):
        c, _ = client
        r = c.put(
            "/admin/api/models/model-a/settings",
            json={
                "guided_grammar_enabled": True,
                "guided_grammar": '  root ::= "YES"  ',
            },
        )
        assert r.status_code == 200
        assert r.json()["settings"]["guided_grammar_enabled"] is True
        assert r.json()["settings"]["guided_grammar"] == 'root ::= "YES"'

        r = c.get("/admin/api/models")
        assert r.status_code == 200
        entry = next(m for m in r.json()["models"] if m["id"] == "model-a")
        assert entry["settings"]["guided_grammar_enabled"] is True
        assert entry["settings"]["guided_grammar"] == 'root ::= "YES"'

    def test_diffusion_settings_update_sanitizes_unsupported_fields(self, client):
        c, _ = client
        pool = admin_routes._get_engine_pool()
        pool._entries["diffusion"] = _FakeEntry(
            "diffusion",
            engine_type="vlm",
            model_type="vlm",
            config_model_type="diffusion_gemma",
        )

        r = c.put(
            "/admin/api/models/diffusion/settings",
            json={
                "max_tokens": 32,
                "temperature": 0.0,
                "top_p": 0.8,
                "force_sampling": True,
                "guided_grammar_enabled": True,
                "guided_grammar": 'root ::= "YES"',
                "max_tool_result_tokens": 4096,
                "turboquant_kv_enabled": True,
                "specprefill_enabled": True,
                "dflash_enabled": True,
                "dflash_in_memory_cache": False,
                "dflash_ssd_cache": True,
                "mtp_enabled": True,
                "vlm_mtp_enabled": True,
            },
        )

        assert r.status_code == 200, r.text
        settings = r.json()["settings"]
        assert settings["max_tokens"] == 32
        assert settings["temperature"] == 0.0
        assert "top_p" not in settings
        assert settings["force_sampling"] is False
        assert settings["guided_grammar_enabled"] is False
        assert "guided_grammar" not in settings
        # Tool calling works on the diffusion lane; setting preserved.
        assert settings["max_tool_result_tokens"] == 4096
        assert settings["turboquant_kv_enabled"] is False
        assert settings["specprefill_enabled"] is False
        assert settings["dflash_enabled"] is False
        assert settings["dflash_in_memory_cache"] is True
        assert settings["dflash_ssd_cache"] is False
        assert settings["mtp_enabled"] is False
        assert settings["vlm_mtp_enabled"] is False


class TestActiveProfileDriftClearing:
    def test_active_preserved_when_no_drift(self, client):
        c, mgr = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        c.post("/admin/api/models/model-a/profiles/coding/apply")
        # Re-save with SAME value
        r = c.put("/admin/api/models/model-a/settings", json={"temperature": 0.0})
        assert r.status_code == 200
        assert r.json()["settings"]["active_profile_name"] == "coding"

    def test_active_cleared_on_drift(self, client):
        c, mgr = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "coding",
                "display_name": "Coding",
                "settings": {"temperature": 0.0},
            },
        )
        c.post("/admin/api/models/model-a/profiles/coding/apply")
        # Change temperature
        r = c.put("/admin/api/models/model-a/settings", json={"temperature": 0.5})
        assert r.status_code == 200
        assert r.json()["settings"].get("active_profile_name") is None


class TestDefaultModelPointer:
    """``server_state.default_model`` must track is_default in both directions.

    Unlike is_pinned (which always writes straight through to the engine pool
    entry), is_default also maintains a second, redundant "current default"
    pointer on server_state for fast lookup. Setting is_default=True updates
    it; unsetting is_default=False must clear it too, but only when THIS
    model is the one currently pointed at -- unsetting a model that was never
    the pointer target must leave someone else's default alone.
    """

    def test_setting_default_updates_the_pointer(self, client):
        c, _ = client
        r = c.put("/admin/api/models/model-a/settings", json={"is_default": True})
        assert r.status_code == 200, r.text
        assert admin_routes._get_server_state().default_model == "model-a"
        assert r.json()["settings"]["is_default"] is True

    def test_unsetting_the_current_default_clears_the_pointer(self, client):
        c, _ = client
        c.put("/admin/api/models/model-a/settings", json={"is_default": True})
        assert admin_routes._get_server_state().default_model == "model-a"

        r = c.put("/admin/api/models/model-a/settings", json={"is_default": False})
        assert r.status_code == 200, r.text
        assert admin_routes._get_server_state().default_model is None
        assert r.json()["settings"]["is_default"] is False

    def test_unsetting_a_model_that_is_not_the_pointer_leaves_it_alone(self, client):
        c, _ = client
        pool = admin_routes._get_engine_pool()
        pool._entries["model-b"] = _FakeEntry("model-b")

        c.put("/admin/api/models/model-a/settings", json={"is_default": True})
        assert admin_routes._get_server_state().default_model == "model-a"

        # model-b was never the pointer target; unsetting its (already-false)
        # is_default must not disturb model-a's default status.
        r = c.put("/admin/api/models/model-b/settings", json={"is_default": False})
        assert r.status_code == 200, r.text
        assert admin_routes._get_server_state().default_model == "model-a"


class TestExposeAsModelAPI:
    """Request models and UI round-trip for the expose-as-model flag."""

    def test_profile_requests_accept_expose_as_model_flag(self):
        create = admin_routes.CreateProfileRequest.model_validate(
            {
                "name": "thinking",
                "display_name": "Thinking",
                "api_name": "thinking-api",
                "settings": {},
                "expose_as_model": True,
            }
        )
        update = admin_routes.UpdateProfileRequest.model_validate(
            {"expose_as_model": False}
        )

        assert create.expose_as_model is True
        assert create.api_name == "thinking-api"
        assert update.expose_as_model is False
        assert "expose_as_model" in update.model_fields_set

    def test_create_exposed_profile_surfaces_in_list(self, client):
        """Creating with expose_as_model persists it; list_profiles enriches
        each entry with the derived model_id and has_engine_fields."""
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "thinking",
                "display_name": "thinking",
                "settings": {"temperature": 0.6, "dflash_enabled": True},
                "expose_as_model": True,
            },
        )
        profiles = c.get("/admin/api/models/model-a/profiles").json()["profiles"]
        prof = next(p for p in profiles if p["name"] == "thinking")
        assert prof["expose_as_model"] is True
        assert prof["model_id"] == "model-a:thinking"
        assert prof["has_engine_fields"] is True

    def test_exposed_profile_surfaces_in_model_list(self, client):
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "p-random",
                "display_name": "Think On",
                "api_name": "think-on",
                "settings": {"temperature": 0.6},
                "expose_as_model": True,
            },
        )

        r = c.get("/admin/api/models")

        assert r.status_code == 200
        entry = r.json()["models"][0]
        assert entry["exposed_profiles"][0]["name"] == "p-random"
        assert entry["exposed_profiles"][0]["api_name"] == "think-on"
        assert entry["exposed_profiles"][0]["model_id"] == "model-a:think-on"

    def test_put_toggles_exposure_without_touching_settings(self, client):
        """A flag-only PUT flips exposure and leaves the profile's settings
        intact (None for expose_as_model means 'don't touch' on other fields)."""
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "fast",
                "display_name": "fast",
                "settings": {"temperature": 0.3},
            },
        )
        r = c.put(
            "/admin/api/models/model-a/profiles/fast", json={"expose_as_model": True}
        )
        assert r.status_code == 200
        assert r.json()["profile"]["expose_as_model"] is True
        assert r.json()["profile"]["settings"]["temperature"] == 0.3

        # Unexpose with the same flag-only PUT.
        r = c.put(
            "/admin/api/models/model-a/profiles/fast", json={"expose_as_model": False}
        )
        assert r.json()["profile"]["expose_as_model"] is False

    def test_rename_exposed_profile_preserves_model_id(self, client):
        """Internal profile rename keeps the public API model ID stable."""
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "thinking",
                "display_name": "thinking",
                "settings": {"temperature": 0.6},
                "expose_as_model": True,
            },
        )
        r = c.put(
            "/admin/api/models/model-a/profiles/thinking",
            json={"new_name": "reasoning", "display_name": "reasoning"},
        )
        assert r.status_code == 200

        profiles = c.get("/admin/api/models/model-a/profiles").json()["profiles"]
        names = {p["name"] for p in profiles}
        assert names == {"reasoning"}
        prof = profiles[0]
        assert prof["expose_as_model"] is True
        assert prof["model_id"] == "model-a:thinking"

    def test_api_name_update_changes_exposed_model_id(self, client):
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "thinking",
                "display_name": "Thinking",
                "settings": {"temperature": 0.6},
                "expose_as_model": True,
            },
        )
        r = c.put(
            "/admin/api/models/model-a/profiles/thinking",
            json={"api_name": "reasoning"},
        )
        assert r.status_code == 200

        profiles = c.get("/admin/api/models/model-a/profiles").json()["profiles"]
        assert profiles[0]["name"] == "thinking"
        assert profiles[0]["api_name"] == "reasoning"
        assert profiles[0]["model_id"] == "model-a:reasoning"

    def test_exposed_profile_rejects_model_id_collision(self, client):
        c, _ = client
        pool = admin_routes._get_engine_pool()
        pool._entries["model-a:thinking"] = _FakeEntry("model-a:thinking")

        r = c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "thinking",
                "display_name": "Thinking",
                "api_name": "thinking",
                "settings": {},
                "expose_as_model": True,
            },
        )

        assert r.status_code == 409

    def test_model_alias_rejects_existing_exposed_profile_id(self, client):
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "thinking",
                "display_name": "Thinking",
                "api_name": "thinking",
                "settings": {},
                "expose_as_model": True,
            },
        )

        r = c.put(
            "/admin/api/models/model-a/settings",
            json={"model_alias": "model-a:thinking"},
        )

        assert r.status_code == 400

    def test_model_alias_rejects_profile_id_created_by_new_alias(self, client):
        c, _ = client
        pool = admin_routes._get_engine_pool()
        pool._entries["gpt:thinking"] = _FakeEntry("gpt:thinking")
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "thinking",
                "display_name": "Thinking",
                "api_name": "thinking",
                "settings": {},
                "expose_as_model": True,
            },
        )

        r = c.put("/admin/api/models/model-a/settings", json={"model_alias": "gpt"})

        assert r.status_code == 400

    def test_settings_only_put_preserves_exposure(self, client):
        """An update that omits expose_as_model must not clear it — the Mac
        app and web both rely on this 'absent = don't touch' contract."""
        c, _ = client
        c.post(
            "/admin/api/models/model-a/profiles",
            json={
                "name": "keep",
                "display_name": "keep",
                "settings": {"temperature": 0.3},
                "expose_as_model": True,
            },
        )
        c.put(
            "/admin/api/models/model-a/profiles/keep",
            json={"settings": {"temperature": 0.9}},
        )
        profiles = c.get("/admin/api/models/model-a/profiles").json()["profiles"]
        prof = next(p for p in profiles if p["name"] == "keep")
        assert prof["expose_as_model"] is True
        assert prof["settings"]["temperature"] == 0.9

    def test_dashboard_profile_ui_round_trips_expose_as_model_flag(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        js = (root / "omlx/admin/static/js/dashboard.js").read_text()
        html = (
            root / "omlx/admin/templates/dashboard/_modal_model_settings.html"
        ).read_text()
        settings_html = (
            root / "omlx/admin/templates/dashboard/_settings.html"
        ).read_text()
        dashboard_html = (root / "omlx/admin/templates/dashboard.html").read_text()
        en = (root / "omlx/admin/i18n/en.json").read_text()

        # api_name is the exposed model ID suffix, so it's validated as a
        # slug and rejected on bad input.
        assert "isValidProfileName" in js
        assert "api_name" in js
        assert "modal.model_settings.profiles.invalid_name" in en
        # Profile chips keep the display name visible in both API-on and
        # API-off states. API state is shown as a badge and edited from the
        # edit form, not toggled directly from the chip row.
        assert "profileTooltip(p)" in html
        assert "p.display_name || p.name" in html
        assert "p.expose_as_model ? (p.api_name || p.name)" not in html
        assert "expose_as_model: !p.expose_as_model" not in html
        assert "p.has_engine_fields" in html
        # Editing updates display_name/api_name/description/exposure without
        # renaming the internal profile key.
        assert "updateProfileFromEdit" in js
        assert "api_name: apiName" in js
        assert "description: description" in js
        assert "expose_as_model: exposeAsModel" in js
        edit_method = js.split("updateProfileFromEdit(p) {", 1)[1].split(
            "updateProfileSettingsFromForm(p)", 1
        )[0]
        assert "settings:" not in edit_method
        assert "updateProfileSettingsFromForm(p)" in js
        assert "updateProfileFromEdit(p)" in html
        assert "updateProfileSettingsFromForm(p)" in html
        assert "_editDescription" in html
        assert "_editExposeAsModel" in html
        assert "profileTooltip(profile)" in settings_html
        assert "model.exposed_profiles" in settings_html
        assert "profile.api_name || profile.name" in settings_html
        assert settings_html.index("profile.has_engine_fields") < settings_html.index(
            "profile.api_name || profile.name"
        )
        assert 'x-text="model.settings.active_profile_name"' not in settings_html
        assert "profileTooltip" in js
        assert "whitespace-pre-line" in dashboard_html
        assert "modal.model_settings.profiles.expose_as_model" in html
        assert "modal.model_settings.profiles.exposed_as" in en
        assert "modal.model_settings.profiles.expose_engine_fields_hint" in en


@pytest.mark.parametrize("operation", ["create", "update", "apply"])
@pytest.mark.parametrize("reason", ["hardware", "architecture", "conflict", "floor"])
def test_a8_profile_validation_preserves_settings(
    client, monkeypatch, operation, reason
):
    c, mgr = client
    entry = admin_routes._get_engine_pool().get_entry("model-a")
    entry.config_model_type = "llama" if reason == "architecture" else "qwen3_5"
    monkeypatch.setattr(
        admin_routes, "_oq_a8_kernels_available", lambda: reason != "hardware"
    )
    settings = {"qwen35_oq_a8_enabled": True}
    if reason == "conflict":
        settings["qwen35_ane_prefill_enabled"] = True
    if reason == "floor":
        settings["qwen35_oq_a8_min_tokens"] = 0
    mgr.set_settings("model-a", ModelSettings())
    if operation != "create":
        mgr.save_profile(
            "model-a", "a8", "A8", "", settings=settings if operation == "apply" else {}
        )
    before = mgr.get_settings("model-a").to_dict()
    profiles = mgr.list_profiles("model-a")
    url = "/admin/api/models/model-a/profiles"
    if operation == "create":
        response = c.post(
            url, json={"name": "a8", "display_name": "A8", "settings": settings}
        )
    elif operation == "update":
        response = c.put(url + "/a8", json={"settings": settings})
    else:
        response = c.post(url + "/a8/apply")
    assert response.status_code == 400, response.text
    assert mgr.get_settings("model-a").to_dict() == before
    assert mgr.list_profiles("model-a") == profiles


def test_a8_profile_apply_checks_merged_ane_conflict(client, monkeypatch):
    c, mgr = client
    admin_routes._get_engine_pool().get_entry("model-a").config_model_type = "qwen3_5"
    monkeypatch.setattr(admin_routes, "_oq_a8_kernels_available", lambda: True)
    mgr.set_settings("model-a", ModelSettings(qwen35_ane_prefill_enabled=True))
    mgr.save_profile("model-a", "a8", "A8", "", settings={"qwen35_oq_a8_enabled": True})
    response = c.post("/admin/api/models/model-a/profiles/a8/apply")
    assert response.status_code == 400
    assert mgr.get_settings("model-a").qwen35_ane_prefill_enabled
    assert not mgr.get_settings("model-a").qwen35_oq_a8_enabled


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_6_moe", "qwen3_8"])
def test_a8_profile_roundtrip_supported_architectures(client, monkeypatch, model_type):
    c, mgr = client
    admin_routes._get_engine_pool().get_entry("model-a").config_model_type = model_type
    monkeypatch.setattr(admin_routes, "_oq_a8_kernels_available", lambda: True)
    url = "/admin/api/models/model-a/profiles"
    assert (
        c.post(
            url,
            json={
                "name": "a8",
                "display_name": "A8",
                "settings": {"qwen35_oq_a8_enabled": True},
            },
        ).status_code
        == 200
    )
    assert c.post(url + "/a8/apply").status_code == 200
    assert mgr.get_settings("model-a").qwen35_oq_a8_enabled


@pytest.mark.parametrize("operation", ["settings", "create", "update"])
@pytest.mark.parametrize(
    "values",
    [
        {"turboquant_kv_bits": 5},
        {"dflash_verify_mode": "typo"},
        {"dflash_in_memory_cache_max_entries": -1},
        {"dflash_in_memory_cache_max_entries": 1.5},
        {"specprefill_enabled": "not-a-boolean"},
        {"dflash_in_memory_cache": []},
        {"specprefill_draft_model": "/missing-draft-for-validation"},
        {"vlm_mtp_draft_model": "./missing-draft-for-validation"},
        {"reasoning_parser": "unknown-parser"},
    ],
)
def test_invalid_settings_do_not_change_storage(client, monkeypatch, operation, values):
    c, mgr = client
    monkeypatch.setattr(admin_routes, "_grammar_parser_options", lambda: [])
    mgr.set_settings("model-a", ModelSettings(temperature=0.5))
    url = "/admin/api/models/model-a"
    response = c.post(
        url + "/profiles",
        json={
            "name": "existing",
            "display_name": "Existing",
            "settings": {"temperature": 0.5},
        },
    )
    assert response.status_code == 200, response.text
    before = {p.name: p.read_bytes() for p in mgr.base_path.glob("*.json")}
    payload = {"temperature": 0.9, **values}
    if operation == "settings":
        response = c.put(url + "/settings", json=payload)
    elif operation == "create":
        response = c.post(
            url + "/profiles",
            json={
                "name": "new",
                "display_name": "New",
                "settings": payload,
            },
        )
    else:
        response = c.put(url + "/profiles/existing", json={"settings": payload})
    assert response.status_code == 422, response.text
    assert {p.name: p.read_bytes() for p in mgr.base_path.glob("*.json")} == before
    assert mgr.get_settings("model-a").temperature == 0.5
    assert len(mgr.list_profiles("model-a")) == 1


@pytest.mark.parametrize("operation", ["settings", "create", "update"])
def test_valid_settings_are_normalized_and_preserved(client, tmp_path, operation):
    c, mgr = client
    draft = tmp_path / "assistant"
    draft.mkdir()
    (draft / "config.json").write_text('{"model_type":"muse_glimmer_assistant"}')
    url = "/admin/api/models/model-a"
    payload = {
        "specprefill_enabled": "false",
        "dflash_in_memory_cache": "false",
        "cache_reasoning_output": "false",
        "turboquant_kv_bits": 0,
        "dflash_in_memory_cache_max_entries": 0,
        "dflash_draft_model": str(draft),
    }
    if operation == "settings":
        response = c.put(url + "/settings", json=payload)
    else:
        response = c.post(
            url + "/profiles",
            json={
                "name": "valid",
                "display_name": "Valid",
                "settings": payload if operation == "create" else {},
                "expose_as_model": True,
            },
        )
        assert response.status_code == 200, response.text
        if operation == "update":
            response = c.put(url + "/profiles/valid", json={"settings": payload})
        assert response.status_code == 200, response.text
        saved = response.json()["profile"]["settings"]
        assert "temperature" not in saved
        assert saved["specprefill_enabled"] is False
        model_id, exposed = mgr.get_exposed_profile_runtime_settings_for_request(
            "model-a:valid"
        )
        assert model_id == "model-a"
        assert exposed.specprefill_enabled is False
        assert exposed.dflash_in_memory_cache is False
        assert exposed.cache_reasoning_output is False
        response = c.post(url + "/profiles/valid/apply")
    assert response.status_code == 200, response.text
    result = mgr.get_settings("model-a")
    assert result.specprefill_enabled is False
    assert result.dflash_in_memory_cache is False
    assert result.cache_reasoning_output is False
    assert result.turboquant_kv_bits == 4
    assert result.dflash_in_memory_cache_max_entries == 4
    assert result.dflash_draft_model == str(draft)


def test_profile_empty_values_keep_overlay_semantics(client):
    c, mgr = client
    mgr.set_settings(
        "model-a", ModelSettings(dflash_draft_model="org/draft", temperature=0.9)
    )
    response = c.post(
        "/admin/api/models/model-a/profiles",
        json={
            "name": "empty",
            "display_name": "Empty",
            "settings": {
                "temperature": None,
                "dflash_draft_model": "",
                "ignored": 1,
                "is_pinned": True,
            },
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["profile"]["settings"] == {}
    response = c.post("/admin/api/models/model-a/profiles/empty/apply")
    assert response.status_code == 200, response.text
    result = mgr.get_settings("model-a")
    assert result.dflash_draft_model == "org/draft"
    assert result.temperature is None
    assert result.is_pinned is False


@pytest.mark.parametrize("operation", ["create", "update"])
def test_templates_use_settings_types_and_filter_before_validation(client, operation):
    c, mgr = client
    url = "/admin/api/profile-templates"
    payload = {
        "enable_thinking": "false",
        "cache_reasoning_output": "false",
        "dflash_draft_model": "/ignored",
    }
    response = c.post(
        url, json={"name": "typed", "display_name": "Typed", "settings": payload}
    )
    assert response.status_code == 200, response.text
    assert response.json()["template"]["settings"] == {
        "enable_thinking": False,
        "cache_reasoning_output": False,
    }
    before = mgr.templates_file.read_bytes()
    if operation == "create":
        response = c.post(
            url,
            json={
                "name": "invalid",
                "display_name": "Invalid",
                "settings": {"enable_thinking": []},
            },
        )
    else:
        response = c.put(url + "/typed", json={"settings": {"enable_thinking": []}})
    assert response.status_code == 422, response.text
    assert mgr.templates_file.read_bytes() == before


def test_parser_validation_uses_dropdown_registry(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(
        admin_routes,
        "_grammar_parser_options",
        lambda: [
            {"value": "harmony", "label": "harmony", "models": ["gpt-oss"]},
        ],
    )
    options = c.get("/admin/api/grammar/parsers").json()
    assert options[0]["value"] == "harmony"
    response = c.put(
        "/admin/api/models/model-a/settings",
        json={"reasoning_parser": options[0]["value"]},
    )
    assert response.status_code == 200, response.text
    response = c.post(
        "/admin/api/models/model-a/profiles",
        json={
            "name": "parser",
            "display_name": "Parser",
            "settings": {"reasoning_parser": "harmony"},
        },
    )
    assert response.status_code == 200, response.text


def test_parser_unavailable_preserves_configuration_and_allows_clearing(
    client, monkeypatch, caplog
):
    c, mgr = client
    monkeypatch.setattr(admin_routes, "_grammar_parser_options", lambda: None)
    mgr.set_settings("model-a", ModelSettings(reasoning_parser="harmony"))
    url = "/admin/api/models/model-a/settings"
    response = c.put(url, json={"temperature": 0.5})
    assert response.status_code == 200, response.text
    assert mgr.get_settings("model-a").reasoning_parser == "harmony"
    response = c.put(url, json={"reasoning_parser": "harmony"})
    assert response.status_code == 200, response.text
    assert "Cannot validate reasoning_parser" in caplog.text
    response = c.put(url, json={"reasoning_parser": ""})
    assert response.status_code == 200, response.text
    assert mgr.get_settings("model-a").reasoning_parser is None


@pytest.mark.parametrize("reference", ["org/draft", "draft-name", "./assistant"])
def test_draft_references_follow_local_path_rules(
    client, tmp_path, monkeypatch, reference
):
    c, _ = client
    monkeypatch.chdir(tmp_path)
    draft = tmp_path / "assistant"
    draft.mkdir()
    (draft / "config.json").write_text('{"model_type":"muse_glimmer_assistant"}')
    response = c.put(
        "/admin/api/models/model-a/settings", json={"dflash_draft_model": reference}
    )
    assert response.status_code == 200, response.text
    response = c.put(
        "/admin/api/models/model-a/settings", json={"dflash_draft_model": ""}
    )
    assert response.status_code == 200, response.text
    assert response.json()["settings"].get("dflash_draft_model") is None


@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize(
    "values",
    [
        {"mtp_enabled": True, "dflash_enabled": True},
        {"vlm_mtp_enabled": True, "specprefill_enabled": True},
    ],
)
def test_profile_conflicting_flags_rejected_before_storage(client, operation, values):
    c, mgr = client
    url = "/admin/api/models/model-a/profiles"
    response = c.post(url, json={"name": "existing", "display_name": "Existing"})
    assert response.status_code == 200, response.text
    before = mgr.profiles_file.read_bytes()
    if operation == "create":
        response = c.post(
            url, json={"name": "new", "display_name": "New", "settings": values}
        )
    else:
        response = c.put(url + "/existing", json={"settings": values})
    assert response.status_code == 400, response.text
    assert mgr.profiles_file.read_bytes() == before


def test_template_apply_reads_latest_and_preserves_independent_profile(client):
    c, mgr = client
    mgr.save_profile("model-a", "coding", "Coding", None, {"temperature": 0.1})
    mgr.save_template("coding", "Coding", None, {"temperature": 0.2})
    path = "/admin/api/models/model-a/profile-templates/coding/apply"
    first = c.post(path)
    assert first.status_code == 200
    copy_name = first.json()["settings"]["active_profile_name"]
    assert copy_name != "coding"
    assert (
        c.put(
            "/admin/api/profile-templates/coding",
            json={"settings": {"temperature": 0.9}},
        ).status_code
        == 200
    )
    second = c.post(path)
    assert second.status_code == 200
    assert second.json()["settings"]["temperature"] == 0.9
    assert second.json()["settings"]["active_profile_name"] == copy_name
    assert mgr.get_profile("model-a", "coding")["settings"]["temperature"] == 0.1
    assert mgr.get_settings_for_request("model-a").temperature == 0.9


def test_template_apply_missing_template_and_model(client):
    c, mgr = client
    mgr.save_template("coding", "Coding", None, {})
    assert (
        c.post("/admin/api/models/missing/profile-templates/coding/apply").status_code
        == 404
    )
    assert (
        c.post("/admin/api/models/model-a/profile-templates/missing/apply").status_code
        == 404
    )
    assert mgr.list_profiles("model-a") == []


# ---------------------------------------------------------------------------
# Settings snapshots: recipe codec, reset, recipe and optimal endpoints
# ---------------------------------------------------------------------------


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


class TestSettingsRecipeCodec:
    def test_round_trip_drops_benchmark_context(self):
        settings = {
            "benchmark_context": "Code (Python)",
            "temperature": 0.6,
            "turboquant_kv_enabled": True,
            "turboquant_kv_bits": 4,
            "dflash_draft_model": "Qwen3-4B-DFlash-b16",
        }
        text = settings_recipe.encode_recipe(settings)
        assert re.fullmatch(r"omlx-recipe:1:[A-Za-z0-9_-]+", text)
        expected = {k: v for k, v in settings.items() if k != "benchmark_context"}
        assert settings_recipe.decode_recipe("  " + text + "\n") == expected

    def test_body_is_a_zlib_stream_like_the_site_encoder(self):
        text = settings_recipe.encode_recipe({"top_p": 0.9})
        body = text.split(":", 2)[2]
        raw = zlib.decompress(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        assert json.loads(raw) == {"top_p": 0.9}

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "hello",
            "omlx-recipe:2:abc",
            "omlx-recipe:1:",
            "omlx-recipe:1:not base64!",
            "omlx-recipe:1:" + _b64url(b"not a zlib stream"),
            "omlx-recipe:1:" + _b64url(zlib.compress(b"[1, 2]")),
            "omlx-recipe:1:" + _b64url(zlib.compress(b'{"a": null}'))[:-4] + "AAAA",
        ],
    )
    def test_rejects_malformed_text(self, text):
        with pytest.raises(settings_recipe.RecipeError):
            settings_recipe.decode_recipe(text)

    def test_rejects_oversized_payload(self):
        bomb = zlib.compress(b'{"a": "' + b"x" * 200000 + b'"}', 9)
        with pytest.raises(settings_recipe.RecipeError, match="too large"):
            settings_recipe.decode_recipe("omlx-recipe:1:" + _b64url(bomb))

    def test_resolve_draft_reference(self):
        entries = [
            ("z-lab/Qwen3-4B-DFlash-b16", "/models/z-lab/Qwen3-4B-DFlash-b16"),
            ("other", "/models/other"),
        ]
        resolve = settings_recipe.resolve_draft_reference
        assert resolve("z-lab/Qwen3-4B-DFlash-b16", entries)[1] == (
            "/models/z-lab/Qwen3-4B-DFlash-b16"
        )
        assert resolve("/models/other", entries) == ("other", "/models/other")
        assert resolve("Qwen3-4B-DFlash-b16", entries)[0] == "z-lab/Qwen3-4B-DFlash-b16"
        assert resolve("/elsewhere/Qwen3-4B-DFlash-b16/", entries)[0] == (
            "z-lab/Qwen3-4B-DFlash-b16"
        )
        assert resolve("missing", entries) == (None, None)
        assert resolve("", entries) == (None, None)

    def test_settings_diff_sends_dataclass_defaults_not_null(self):
        current = {"qwen35_ane_prefill_sequence_length": 4096, "temperature": 0.5}
        scope = {"qwen35_ane_prefill_sequence_length", "temperature", "top_p"}
        diff = settings_recipe.settings_diff({}, current, scope)
        assert diff == {"qwen35_ane_prefill_sequence_length": 2048, "temperature": None}

    def test_search_url_targets_the_leaderboard_filters(self):
        url = settings_recipe.search_url("M3", "Ultra", "Qwen3.8-27B-4bit", 4096, 512)
        assert "memory_max=512" in url
        assert url.startswith("https://omlx.ai/benchmarks/performance?")
        assert "model_exact=Qwen3.8-27B-4bit" in url
        assert "context=4096" in url
        assert "%22M3%22" in url and "%22Ultra%22" in url


class TestSettingsSnapshotRoutes:
    def _install_draft(self, tmp_path):
        # The settings PUT validator requires a local draft path to carry a
        # config.json, so the fake draft gets a real directory.
        draft = tmp_path / "Qwen3-4B-DFlash-b16"
        draft.mkdir()
        (draft / "config.json").write_text('{"model_type":"qwen3"}')
        pool = admin_routes._get_engine_pool()
        entry = _FakeEntry("z-lab/Qwen3-4B-DFlash-b16")
        entry.model_path = str(draft)
        pool._entries["z-lab/Qwen3-4B-DFlash-b16"] = entry
        return str(draft)

    def test_reset_returns_every_field_to_defaults(self, client):
        c, mgr = client
        state = admin_routes._get_server_state()
        pool = admin_routes._get_engine_pool()
        mgr.set_settings(
            "model-a",
            ModelSettings(
                temperature=0.1,
                is_pinned=True,
                is_default=True,
                ttl_seconds=300,
                model_alias="alias",
                display_name="Nice name",
                description="notes",
                trust_remote_code=True,
                turboquant_kv_enabled=True,
                qwen35_ane_prefill_sequence_length=4096,
                active_profile_name="p",
            ),
        )
        state.default_model = "model-a"
        pool.get_entry("model-a").is_pinned = True

        r = c.post("/admin/api/models/model-a/settings/reset")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["changed"] is True
        assert body["applied"]["temperature"] is None
        assert body["applied"]["display_name"] is None
        assert body["applied"]["is_pinned"] is False
        assert body["skipped"] == []
        assert body["settings"] == ModelSettings().to_dict()
        assert mgr.get_settings("model-a").to_dict() == ModelSettings().to_dict()
        assert state.default_model is None
        assert pool.get_entry("model-a").is_pinned is False

        again = c.post("/admin/api/models/model-a/settings/reset").json()
        assert again["changed"] is False

    def test_reset_unknown_model_404(self, client):
        c, _ = client
        assert c.post("/admin/api/models/nope/settings/reset").status_code == 404

    def test_recipe_replaces_scoped_fields_and_keeps_the_rest(self, client):
        c, mgr = client
        mgr.set_settings(
            "model-a",
            ModelSettings(
                temperature=0.9,
                top_p=0.5,
                ttl_seconds=300,
                chat_template_kwargs={"a": 1},
            ),
        )
        recipe = settings_recipe.encode_recipe(
            {
                "benchmark_context": "Novel (English)",
                "temperature": 0.2,
                "turboquant_kv_enabled": True,
                "turboquant_kv_bits": 3,
                "display_name": "must be ignored",
            }
        )
        r = c.post("/admin/api/models/model-a/settings/recipe", json={"recipe": recipe})
        assert r.status_code == 200, r.text
        body = r.json()
        settings = body["settings"]
        assert settings["temperature"] == 0.2
        assert "top_p" not in settings
        assert settings["turboquant_kv_enabled"] is True
        assert settings["turboquant_kv_bits"] == 3
        assert settings["ttl_seconds"] == 300
        assert settings["chat_template_kwargs"] == {"a": 1}
        assert "display_name" not in settings
        assert body["skipped"] == []
        assert body["applied"]["temperature"] == 0.2
        assert body["applied"]["top_p"] is None
        assert mgr.get_settings("model-a").temperature == 0.2

    @pytest.mark.parametrize("target_type", ["qwen3_5", "k2_horizon"])
    def test_recipe_handles_inactive_ane_controls_for_target(self, client, target_type):
        from omlx.admin.benchmark import _filter_uploaded_settings
        from omlx.model_settings import validate_ane_prefill

        c, mgr = client
        admin_routes._get_engine_pool().get_entry("model-a").config_model_type = (
            target_type
        )
        source = ModelSettings(
            temperature=0.2,
            qwen35_ane_prefill_enabled=False,
            qwen35_ane_prefill_sequence_length=128,
        )
        validate_ane_prefill(source.to_dict(), "k2_horizon")
        recipe = settings_recipe.encode_recipe(_filter_uploaded_settings(source))
        response = c.post(
            "/admin/api/models/model-a/settings/recipe", json={"recipe": recipe}
        )
        assert response.status_code == 200, response.text
        saved = mgr.get_settings("model-a")
        assert saved.temperature == 0.2
        assert saved.qwen35_ane_prefill_enabled is False
        assert saved.qwen35_ane_prefill_sequence_length == (
            2048 if target_type == "qwen3_5" else 128
        )
        assert [item["feature"] for item in response.json()["skipped"]] == (
            ["ane_prefill"] if target_type == "qwen3_5" else []
        )

    @pytest.mark.parametrize(
        "invalid",
        [
            {"temperature": {"bad": 1}},
            {"qwen35_ane_prefill_fraction": {"bad": 1}},
        ],
    )
    def test_recipe_invalid_value_types_leave_storage_unchanged(self, client, invalid):
        c, mgr = client
        mgr.set_settings("model-a", ModelSettings(temperature=0.7))
        before = {p.name: p.read_bytes() for p in mgr.base_path.glob("*.json")}
        recipe = settings_recipe.encode_recipe({"temperature": 0.2, **invalid})
        response = c.post(
            "/admin/api/models/model-a/settings/recipe", json={"recipe": recipe}
        )
        assert response.status_code == 400, response.text
        assert {p.name: p.read_bytes() for p in mgr.base_path.glob("*.json")} == before
        assert mgr.get_settings("model-a").temperature == 0.7

    def test_recipe_reports_persisted_values_and_resets(self, client):
        c, mgr = client
        mgr.set_settings(
            "model-a", ModelSettings(top_p=0.5, turboquant_kv_enabled=True)
        )
        recipe = settings_recipe.encode_recipe(
            {"temperature": 0.2, "index_cache_freq": 1}
        )
        response = c.post(
            "/admin/api/models/model-a/settings/recipe", json={"recipe": recipe}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["applied"]["top_p"] is None
        assert body["applied"]["turboquant_kv_enabled"] is False
        assert body["applied"]["index_cache_freq"] is None
        saved = mgr.get_settings("model-a")
        assert body["applied"] == {
            key: getattr(saved, key) for key in body["applied"]
        }

    def test_reset_reports_metadata_only_change(self, client):
        c, mgr = client
        mgr.set_settings("model-a", ModelSettings(display_name="Custom name"))
        body = c.post("/admin/api/models/model-a/settings/reset").json()
        assert body["changed"] is True
        assert body["applied"]["display_name"] is None
        assert mgr.get_settings("model-a").display_name is None

    def test_recipe_drops_feature_whose_draft_is_missing(self, client, monkeypatch):
        c, mgr = client
        monkeypatch.setattr(
            admin_routes, "_dflash_compat_for_model", lambda info: (True, "")
        )
        recipe = settings_recipe.encode_recipe(
            {
                "dflash_enabled": True,
                "dflash_draft_model": "Missing-DFlash",
                "dflash_block_size": 16,
                "temperature": 0.3,
            }
        )
        r = c.post("/admin/api/models/model-a/settings/recipe", json={"recipe": recipe})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["settings"]["dflash_enabled"] is False
        assert "dflash_draft_model" not in body["settings"]
        assert "dflash_block_size" not in body["settings"]
        assert body["settings"]["temperature"] == 0.3
        assert body["skipped"] == [
            {
                "feature": "dflash",
                "reason": "draft model 'Missing-DFlash' is not installed",
            }
        ]

    @pytest.mark.parametrize("ssd_available", [False, True])
    def test_recipe_checks_inactive_dflash_ssd_cache(self, client, ssd_available):
        from omlx.admin.benchmark import _filter_uploaded_settings

        c, mgr = client
        admin_routes._get_engine_pool()._scheduler_config = SimpleNamespace(
            paged_ssd_cache_dir="/cache" if ssd_available else None
        )
        source = ModelSettings(
            temperature=0.2, dflash_enabled=False, dflash_ssd_cache=True
        )
        recipe = settings_recipe.encode_recipe(_filter_uploaded_settings(source))
        response = c.post(
            "/admin/api/models/model-a/settings/recipe", json={"recipe": recipe}
        )
        assert response.status_code == 200, response.text
        saved = mgr.get_settings("model-a")
        assert saved.temperature == 0.2
        assert saved.dflash_enabled is False
        assert saved.dflash_ssd_cache is ssd_available
        assert [item["feature"] for item in response.json()["skipped"]] == (
            [] if ssd_available else ["dflash_ssd_cache"]
        )

    @pytest.mark.parametrize(
        "engine_type,target_type,draft_type,mismatch,accepted",
        [
            ("batched", "llama", "gemma4_assistant", None, False),
            ("vlm", "qwen3_5", "gemma4_assistant", None, False),
            ("vlm", "qwen3_5", "qwen3_5_mtp", "hidden_size", False),
            ("vlm", "qwen3_5", "qwen3_5_mtp", "vocab_size", False),
            ("vlm", "qwen3_5", "qwen3_5_mtp", None, True),
            ("vlm", "gemma4", "gemma4_assistant", None, True),
            ("vlm", "gemma4_unified", "gemma4_unified_assistant", None, True),
        ],
    )
    def test_recipe_checks_vlm_mtp_pair(
        self, client, tmp_path, engine_type, target_type, draft_type, mismatch, accepted
    ):
        c, mgr = client
        pool = admin_routes._get_engine_pool()
        target = pool.get_entry("model-a")
        target.engine_type = engine_type
        target.config_model_type = target_type
        draft = _FakeEntry("assistant", config_model_type=draft_type)
        pool._entries["assistant"] = draft
        text_config = {"hidden_size": 1536, "vocab_size": 262144}
        draft_text = dict(text_config)
        if mismatch:
            draft_text[mismatch] //= 2
        if draft_type.startswith("gemma4"):
            draft_text["hidden_size"] = 768
        for name, entry, config in (
            ("target", target, {"model_type": target_type, "text_config": text_config}),
            (
                "draft",
                draft,
                {
                    "model_type": draft_type,
                    "text_config": draft_text,
                    "backbone_hidden_size": 1536,
                },
            ),
        ):
            path = tmp_path / name
            path.mkdir()
            (path / "config.json").write_text(json.dumps(config))
            entry.model_path = str(path)
        recipe = settings_recipe.encode_recipe(
            {
                "temperature": 0.2,
                "vlm_mtp_enabled": True,
                "vlm_mtp_draft_model": "assistant",
            }
        )
        response = c.post(
            "/admin/api/models/model-a/settings/recipe", json={"recipe": recipe}
        )
        assert response.status_code == 200, response.text
        saved = mgr.get_settings("model-a")
        assert saved.temperature == 0.2
        assert saved.vlm_mtp_enabled is accepted
        assert [item["feature"] for item in response.json()["skipped"]] == (
            [] if accepted else ["vlm_mtp"]
        )

    def test_recipe_resolves_installed_draft_by_basename(
        self, client, monkeypatch, tmp_path
    ):
        import omlx.engine.dflash as dflash_engine

        c, mgr = client
        draft_path = self._install_draft(tmp_path)
        monkeypatch.setattr(
            admin_routes, "_dflash_compat_for_model", lambda info: (True, "")
        )
        # The settings PUT re-checks the target with the engine helper.
        monkeypatch.setattr(dflash_engine, "is_dflash_compatible", lambda p: (True, ""))
        recipe = settings_recipe.encode_recipe(
            {"dflash_enabled": True, "dflash_draft_model": "Qwen3-4B-DFlash-b16"}
        )
        r = c.post("/admin/api/models/model-a/settings/recipe", json={"recipe": recipe})
        assert r.status_code == 200, r.text
        assert r.json()["settings"]["dflash_enabled"] is True
        assert r.json()["settings"]["dflash_draft_model"] == draft_path
        assert r.json()["skipped"] == []

    def test_recipe_skips_feature_the_model_cannot_run(self, client):
        c, _ = client
        recipe = settings_recipe.encode_recipe(
            {"mtp_enabled": True, "mtp_num_draft_tokens": 3, "max_tokens": 64}
        )
        r = c.post("/admin/api/models/model-a/settings/recipe", json={"recipe": recipe})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["settings"]["mtp_enabled"] is False
        assert "mtp_num_draft_tokens" not in body["settings"]
        assert body["settings"]["max_tokens"] == 64
        assert [s["feature"] for s in body["skipped"]] == ["mtp"]

    @pytest.mark.parametrize(
        "recipe",
        [
            "garbage",
            settings_recipe.encode_recipe({"benchmark_context": "x"}),
            settings_recipe.encode_recipe({"display_name": "not shareable"}),
        ],
    )
    def test_recipe_rejects_unusable_input(self, client, recipe):
        c, mgr = client
        before = mgr.get_settings("model-a").to_dict()
        r = c.post("/admin/api/models/model-a/settings/recipe", json={"recipe": recipe})
        assert r.status_code == 400, r.text
        assert mgr.get_settings("model-a").to_dict() == before

    def _fake_omlx_ai(self, monkeypatch, handler):
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, params))
            status, payload = handler(url, params)
            return SimpleNamespace(status_code=status, json=lambda: payload, text="")

        monkeypatch.setattr(admin_routes, "requests", SimpleNamespace(get=fake_get))
        monkeypatch.setattr(
            admin_routes,
            "_local_device_info",
            lambda: {
                "chip_name": "M3",
                "chip_variant": "Ultra",
                "memory_gb": 512,
                "gpu_cores": 80,
            },
        )
        return calls

    @staticmethod
    def _row(bid, pp, tg, settings, memory_gb=512):
        return {
            "id": bid,
            "url": f"https://omlx.ai/benchmarks/performance/{bid}",
            "pp_tps": pp,
            "tg_tps": tg,
            "quantization": "4bit",
            "memory_gb": memory_gb,
            "model_settings": settings,
        }

    def test_optimal_candidates_grouped_by_pp_and_tg(self, client, monkeypatch):
        c, _ = client
        rows = {
            "pp": [self._row("aaa", 900, 40, {"temperature": 0.1})],
            "tg": [
                self._row("bbb", 800, 60, {"temperature": 0.2}, memory_gb=256),
                self._row("aaa", 900, 40, {"temperature": 0.1}),
            ],
        }
        calls = self._fake_omlx_ai(
            monkeypatch, lambda url, params: (200, {"results": rows[params["sort"]]})
        )
        r = c.get("/admin/api/models/model-a/settings/optimal")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["found"] is True
        assert body["model_name"] == "model-a"
        assert body["context_length"] == 4096
        assert [x["benchmark_id"] for x in body["by_pp"]] == ["aaa"]
        assert [x["benchmark_id"] for x in body["by_tg"]] == ["bbb", "aaa"]
        assert body["by_tg"][0]["tg_tps"] == 60
        assert body["by_tg"][0]["memory_gb"] == 256
        assert "model_settings" not in body["by_pp"][0]
        assert [call[1]["sort"] for call in calls] == ["pp", "tg"]
        assert calls[0][1] == {
            "chip": "M3",
            "variant": "Ultra",
            "memory_gb": 512,
            "model": "model-a",
            "context": 4096,
            "limit": 3,
            "sort": "pp",
        }

    def test_optimal_candidates_none_found(self, client, monkeypatch):
        c, _ = client
        self._fake_omlx_ai(monkeypatch, lambda url, params: (200, {"results": []}))
        body = c.get("/admin/api/models/model-a/settings/optimal").json()
        assert body["found"] is False
        assert body["by_pp"] == [] and body["by_tg"] == []
        assert "model_exact=model-a" in body["search_url"]
        assert "context=4096" in body["search_url"]
        assert "memory_max=512" in body["search_url"]

    def test_optimal_apply_selected_candidate(self, client, monkeypatch):
        c, mgr = client
        calls = self._fake_omlx_ai(
            monkeypatch,
            lambda url, params: (
                200,
                self._row(
                    "bbb",
                    800,
                    60,
                    {"benchmark_context": "x", "temperature": 0.2, "mtp_enabled": True},
                ),
            ),
        )
        r = c.post(
            "/admin/api/models/model-a/settings/optimal", json={"benchmark_id": "bbb"}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert calls == [(f"{admin_routes.OMLX_AI_API_URL}/bbb", None)]
        assert body["benchmark_id"] == "bbb"
        assert body["benchmark_url"].endswith("/bbb")
        assert body["pp_tps"] == 800
        assert body["settings"]["temperature"] == 0.2
        assert [s["feature"] for s in body["skipped"]] == ["mtp"]
        assert mgr.get_settings("model-a").temperature == 0.2

    def test_optimal_apply_rejects_bad_id_and_upstream_failure(
        self, client, monkeypatch
    ):
        c, _ = client
        r = c.post(
            "/admin/api/models/model-a/settings/optimal", json={"benchmark_id": "../x"}
        )
        assert r.status_code == 422
        self._fake_omlx_ai(monkeypatch, lambda url, params: (404, {"error": "nope"}))
        r = c.post(
            "/admin/api/models/model-a/settings/optimal", json={"benchmark_id": "zzz"}
        )
        assert r.status_code == 502

    def test_snapshot_never_imports_guided_grammar(self, client):
        c, mgr = client
        mgr.set_settings(
            "model-a",
            ModelSettings(guided_grammar_enabled=True, guided_grammar='root ::= "YES"'),
        )
        recipe = settings_recipe.encode_recipe(
            {"temperature": 0.4, "guided_grammar_enabled": False, "guided_grammar": "x"}
        )
        r = c.post("/admin/api/models/model-a/settings/recipe", json={"recipe": recipe})
        assert r.status_code == 200, r.text
        settings = r.json()["settings"]
        assert settings["temperature"] == 0.4
        assert settings["guided_grammar_enabled"] is True
        assert settings["guided_grammar"] == 'root ::= "YES"'
        assert "guided_grammar" not in r.json()["applied"]

    def test_dashboard_wires_snapshot_actions(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        js = (root / "omlx/admin/static/js/dashboard.js").read_text()
        modal = (
            root / "omlx/admin/templates/dashboard/_modal_model_settings.html"
        ).read_text()
        apply_modal = (
            root / "omlx/admin/templates/dashboard/_modal_settings_apply.html"
        ).read_text()
        dashboard = (root / "omlx/admin/templates/dashboard.html").read_text()
        en = json.loads((root / "omlx/admin/i18n/en.json").read_text())
        ko = json.loads((root / "omlx/admin/i18n/ko.json").read_text())

        assert "dashboard/_modal_settings_apply.html" in dashboard
        for mode in ("reset", "optimal", "recipe"):
            assert f"openSettingsApply('{mode}')" in modal
        assert "/settings/${path}`" in js
        assert "_settingsActionRequest('POST', 'reset')" in js
        assert "_settingsActionRequest('GET', 'optimal')" in js
        assert "_settingsActionRequest('POST', 'recipe'" in js
        assert "applyOptimalCandidate(item.benchmark_id)" in apply_modal
        assert "by_pp" in js and "by_tg" in js
        for key in (
            "modal.model_settings.actions.reset_confirm",
            "modal.model_settings.actions.group_pp",
            "modal.model_settings.actions.group_tg",
            "modal.model_settings.actions.none_body",
            "js.error.settings_apply_failed",
        ):
            assert key in en and key in ko

    def test_cache_reasoning_output_round_trips_through_put(self, client):
        c, mgr = client
        r = c.put(
            "/admin/api/models/model-a/settings", json={"cache_reasoning_output": True}
        )
        assert r.status_code == 200, r.text
        assert r.json()["settings"]["cache_reasoning_output"] is True
        r = c.put(
            "/admin/api/models/model-a/settings", json={"cache_reasoning_output": None}
        )
        assert "cache_reasoning_output" not in r.json()["settings"]
