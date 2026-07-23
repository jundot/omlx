# SPDX-License-Identifier: Apache-2.0
"""Tests for admin profile/template API routes."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
    keys reset to defaults so clearing a value in the editor round-trips.
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

        r = c.put(
            "/admin/api/models/model-a/profiles/p",
            json={"settings": {"temperature": 0.7}},
        )
        assert r.status_code == 200, r.text
        r = c.post("/admin/api/models/model-a/profiles/p/apply")
        settings = r.json()["settings"]
        assert settings["temperature"] == 0.7
        assert "max_context_window" not in settings
        assert "max_tokens" not in settings

    def test_apply_clears_kwargs_removed_from_profile(self, client):
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
        assert r.json()["settings"]["chat_template_kwargs"] == {
            "enable_thinking": True
        }

        c.put(
            "/admin/api/models/model-a/profiles/p",
            json={"settings": {"temperature": 0.5}},
        )
        r = c.post("/admin/api/models/model-a/profiles/p/apply")
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
        assert mgr.get_profile("model-a", "p")["settings"] == {"top_p": 0.9}

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
