# SPDX-License-Identifier: Apache-2.0
"""Tests for admin profile/template API routes."""


import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes
from omlx.model_settings import ModelSettingsManager


class _FakeEntry:
    def __init__(self, model_id: str):
        self.engine_type = "batched"
        self.model_type = "llm"
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
        return {"models": [{"id": "model-a", "loaded": False, "pinned": False,
                            "engine_type": "batched", "model_type": "llm"}]}


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
        r = c.post("/admin/api/models/model-a/profiles", json={
            "name": "coding", "display_name": "Coding",
            "settings": {"temperature": 0.0, "is_pinned": True},
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["profile"]["name"] == "coding"
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
        r = c.post("/admin/api/models/model-a/profiles", json={
            "name": "Has Space", "display_name": "x", "settings": {},
        })
        assert r.status_code == 400

    def test_update_profile(self, client):
        c, _ = client
        c.post("/admin/api/models/model-a/profiles", json={
            "name": "coding", "display_name": "Coding",
            "settings": {"temperature": 0.0},
        })
        r = c.put("/admin/api/models/model-a/profiles/coding", json={
            "display_name": "Coding v2",
            "settings": {"temperature": 0.2},
        })
        assert r.status_code == 200
        assert r.json()["profile"]["display_name"] == "Coding v2"
        assert r.json()["profile"]["settings"]["temperature"] == 0.2

    def test_delete_profile(self, client):
        c, _ = client
        c.post("/admin/api/models/model-a/profiles", json={
            "name": "coding", "display_name": "Coding", "settings": {},
        })
        r = c.delete("/admin/api/models/model-a/profiles/coding")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_delete_missing_404(self, client):
        c, _ = client
        r = c.delete("/admin/api/models/model-a/profiles/nope")
        assert r.status_code == 404

    def test_apply_profile_sets_active(self, client):
        c, mgr = client
        c.post("/admin/api/models/model-a/profiles", json={
            "name": "coding", "display_name": "Coding",
            "settings": {"temperature": 0.0},
        })
        r = c.post("/admin/api/models/model-a/profiles/coding/apply")
        assert r.status_code == 200
        assert r.json()["settings"]["active_profile_name"] == "coding"

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
        r = c.post("/admin/api/models/model-a/profiles", json={
            "name": "coding", "display_name": "Coding",
            "settings": {"temperature": 0.0, "turboquant_kv_enabled": True},
            "also_save_as_template": True,
        })
        assert r.status_code == 200
        tmpl = mgr.get_template("coding")
        assert tmpl is not None
        assert tmpl["settings"] == {"temperature": 0.0}


def test_all_model_settings_fields_classified():
    from dataclasses import fields

    from omlx.model_profiles import (
        EXCLUDED_FROM_PROFILES,
        MODEL_SPECIFIC_PROFILE_FIELDS,
        UNIVERSAL_PROFILE_FIELDS,
    )
    from omlx.model_settings import ModelSettings

    classified = set(UNIVERSAL_PROFILE_FIELDS) | set(MODEL_SPECIFIC_PROFILE_FIELDS) | EXCLUDED_FROM_PROFILES
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
        r = c.post("/admin/api/profile-templates", json={
            "name": "coding", "display_name": "Coding",
            "settings": {"temperature": 0.0, "turboquant_kv_enabled": True},
        })
        assert r.status_code == 200
        # Model-specific field filtered out
        assert r.json()["template"]["settings"] == {"temperature": 0.0}

    def test_duplicate_conflicts(self, client):
        c, _ = client
        c.post("/admin/api/profile-templates", json={
            "name": "coding", "display_name": "Coding", "settings": {"temperature": 0.0},
        })
        r = c.post("/admin/api/profile-templates", json={
            "name": "coding", "display_name": "Coding", "settings": {"temperature": 0.1},
        })
        assert r.status_code == 409

    def test_update_delete(self, client):
        c, _ = client
        c.post("/admin/api/profile-templates", json={
            "name": "coding", "display_name": "Coding", "settings": {"temperature": 0.0},
        })
        r = c.put("/admin/api/profile-templates/coding", json={"display_name": "Coding v2"})
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
        c.post("/admin/api/models/model-a/profiles", json={
            "name": "coding", "display_name": "Coding",
            "settings": {"temperature": 0.0},
        })
        c.post("/admin/api/models/model-a/profiles/coding/apply")
        r = c.get("/admin/api/models")
        assert r.status_code == 200
        models = r.json()["models"]
        entry = next(m for m in models if m["id"] == "model-a")
        assert entry["settings"]["active_profile_name"] == "coding"


class TestActiveProfileDriftClearing:
    def test_active_preserved_when_no_drift(self, client):
        c, mgr = client
        c.post("/admin/api/models/model-a/profiles", json={
            "name": "coding", "display_name": "Coding",
            "settings": {"temperature": 0.0},
        })
        c.post("/admin/api/models/model-a/profiles/coding/apply")
        # Re-save with SAME value
        r = c.put("/admin/api/models/model-a/settings", json={"temperature": 0.0})
        assert r.status_code == 200
        assert r.json()["settings"]["active_profile_name"] == "coding"

    def test_active_cleared_on_drift(self, client):
        c, mgr = client
        c.post("/admin/api/models/model-a/profiles", json={
            "name": "coding", "display_name": "Coding",
            "settings": {"temperature": 0.0},
        })
        c.post("/admin/api/models/model-a/profiles/coding/apply")
        # Change temperature
        r = c.put("/admin/api/models/model-a/settings", json={"temperature": 0.5})
        assert r.status_code == 200
        assert r.json()["settings"].get("active_profile_name") is None
