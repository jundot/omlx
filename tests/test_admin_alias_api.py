# SPDX-License-Identifier: Apache-2.0
"""Tests for alias management CLI commands and swap-alias API endpoint."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes
from omlx.model_settings import ModelSettings, ModelSettingsManager


# =============================================================================
# Mock helpers
# =============================================================================


class _FakeEntry:
    def __init__(self, model_id: str):
        self.engine_type = "batched"
        self.model_type = "llm"
        self.engine = None
        self.is_pinned = False
        self.is_loading = False
        self.model_path = "/fake"


class _FakePool:
    def __init__(self, model_ids=None):
        self._entries = {mid: _FakeEntry(mid) for mid in (model_ids or [])}

    def get_entry(self, model_id):
        return self._entries.get(model_id)

    def get_status(self):
        return {
            "models": [
                {
                    "id": mid,
                    "loaded": False,
                    "pinned": False,
                    "engine_type": "batched",
                    "model_type": "llm",
                }
                for mid in self._entries
            ]
        }


class _FakeServerState:
    default_model = None


def _make_client(tmp_path, model_ids=None, settings=None):
    """Build a TestClient with admin routes and a fresh settings manager."""
    from types import SimpleNamespace

    mgr = ModelSettingsManager(tmp_path)
    pool = _FakePool(model_ids or [])
    state = _FakeServerState()

    if settings:
        for mid, s in settings.items():
            mgr.set_settings(mid, s)

    admin_routes._get_settings_manager = lambda: mgr
    admin_routes._get_engine_pool = lambda: pool
    admin_routes._get_server_state = lambda: state
    # Bypass auth entirely so the tests don't need session cookies or API keys
    admin_routes._get_global_settings = lambda: SimpleNamespace(
        auth=SimpleNamespace(api_key="", sub_keys=[], skip_api_key_verification=True)
    )

    async def _fake_require_admin():
        return True

    from omlx.admin import auth as admin_auth

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app), mgr


# =============================================================================
# swap-alias API endpoint (POST /api/models/swap-alias)
# =============================================================================


class TestSwapAliasApi:
    """Tests for POST /api/models/swap-alias."""

    def test_swap_aliases_two_models(self, tmp_path):
        """Swapping aliases between two models transfers them."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
            settings={
                "model-a": ModelSettings(model_alias="alias-a"),
                "model-b": ModelSettings(model_alias="alias-b"),
            },
        )
        r = c.post("/admin/api/models/swap-alias", json={
            "model_a": "model-a",
            "model_b": "model-b",
            "alias_a": "alias-b",
            "alias_b": "alias-a",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["model_a"]["old_alias"] == "alias-a"
        assert body["model_a"]["new_alias"] == "alias-b"
        assert body["model_b"]["old_alias"] == "alias-b"
        assert body["model_b"]["new_alias"] == "alias-a"

        # Verify persistence
        assert mgr.get_settings("model-a").model_alias == "alias-b"
        assert mgr.get_settings("model-b").model_alias == "alias-a"

    def test_swap_clears_both_aliases(self, tmp_path):
        """Swap with None aliases clears both."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
            settings={
                "model-a": ModelSettings(model_alias="alias-a"),
                "model-b": ModelSettings(model_alias="alias-b"),
            },
        )
        r = c.post("/admin/api/models/swap-alias", json={
            "model_a": "model-a",
            "model_b": "model-b",
            "alias_a": None,
            "alias_b": None,
        })
        assert r.status_code == 200
        assert mgr.get_settings("model-a").model_alias is None
        assert mgr.get_settings("model-b").model_alias is None

    def test_swap_transfer_alias_one_side(self, tmp_path):
        """Transfer alias from A to B, clearing A."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
            settings={"model-a": ModelSettings(model_alias="shared")},
        )
        r = c.post("/admin/api/models/swap-alias", json={
            "model_a": "model-a",
            "model_b": "model-b",
            "alias_a": None,
            "alias_b": "shared",
        })
        assert r.status_code == 200
        assert mgr.get_settings("model-a").model_alias is None
        assert mgr.get_settings("model-b").model_alias == "shared"

    def test_swap_conflicts_with_third_model(self, tmp_path):
        """Third model already uses target alias — 400."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b", "model-c"],
            settings={
                "model-a": ModelSettings(model_alias="alias-a"),
                "model-c": ModelSettings(model_alias="alias-b"),
            },
        )
        r = c.post("/admin/api/models/swap-alias", json={
            "model_a": "model-a",
            "model_b": "model-b",
            "alias_a": "alias-b",
            "alias_b": "alias-a",
        })
        assert r.status_code == 400
        assert "already used by model" in r.json()["detail"]

    def test_swap_conflicts_with_directory_name(self, tmp_path):
        """Target alias equals another model's directory name — 400."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
            settings={"model-a": ModelSettings(model_alias="alias-a")},
        )
        r = c.post("/admin/api/models/swap-alias", json={
            "model_a": "model-a",
            "model_b": "model-b",
            "alias_a": "model-b",  # conflicts with model-b's directory name
            "alias_b": "alias-a",
        })
        assert r.status_code == 400
        assert "conflicts with model directory name" in r.json()["detail"]

    def test_swap_missing_model(self, tmp_path):
        """One model doesn't exist — 404."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a"],
            settings={"model-a": ModelSettings(model_alias="a")},
        )
        r = c.post("/admin/api/models/swap-alias", json={
            "model_a": "model-a",
            "model_b": "nope",
            "alias_a": None,
            "alias_b": None,
        })
        assert r.status_code == 404
        assert "not found" in r.json()["detail"]

    def test_swap_empty_string_becomes_none(self, tmp_path):
        """Empty string aliases are normalised to None."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
            settings={
                "model-a": ModelSettings(model_alias="alias-a"),
                "model-b": ModelSettings(model_alias="alias-b"),
            },
        )
        r = c.post("/admin/api/models/swap-alias", json={
            "model_a": "model-a",
            "model_b": "model-b",
            "alias_a": "",
            "alias_b": "   ",
        })
        assert r.status_code == 200
        assert mgr.get_settings("model-a").model_alias is None
        assert mgr.get_settings("model-b").model_alias is None

    def test_swap_same_model(self, tmp_path):
        """Swapping a model with itself should still work."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a"],
            settings={"model-a": ModelSettings(model_alias="alias-a")},
        )
        r = c.post("/admin/api/models/swap-alias", json={
            "model_a": "model-a",
            "model_b": "model-a",
            "alias_a": "new",
            "alias_b": "alias-a",
        })
        # Both references point to the same model, so both are updated
        assert r.status_code == 200

    def test_swap_no_conflict_same_alias(self, tmp_path):
        """Setting the same alias on both models in swap (no conflict)."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
            settings={"model-a": ModelSettings(model_alias="a")},
        )
        r = c.post("/admin/api/models/swap-alias", json={
            "model_a": "model-a",
            "model_b": "model-b",
            "alias_a": "shared",
            "alias_b": "shared",
        })
        assert r.status_code == 200


# =============================================================================
# update_model_settings alias handling
# =============================================================================


class TestUpdateModelSettingsAlias:
    """Tests for model_alias in PUT /api/models/{id}/settings."""

    def test_set_alias_via_update(self, tmp_path):
        """Setting alias via update_model_settings."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a"],
        )
        r = c.put("/admin/api/models/model-a/settings", json={
            "model_alias": "my-alias",
        })
        assert r.status_code == 200
        assert mgr.get_settings("model-a").model_alias == "my-alias"

    def test_set_alias_conflict_with_existing(self, tmp_path):
        """Setting an alias already taken by another model — 400."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
            settings={"model-a": ModelSettings(model_alias="shared")},
        )
        r = c.put("/admin/api/models/model-b/settings", json={
            "model_alias": "shared",
        })
        assert r.status_code == 400
        assert "already used by" in r.json()["detail"]

    def test_set_alias_conflicts_with_directory(self, tmp_path):
        """Alias equals another model's directory name — 400."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
        )
        r = c.put("/admin/api/models/model-a/settings", json={
            "model_alias": "model-b",
        })
        assert r.status_code == 400
        assert "conflicts with model directory name" in r.json()["detail"]

    def test_clear_alias_via_update(self, tmp_path):
        """Clearing alias via update_model_settings."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a"],
            settings={"model-a": ModelSettings(model_alias="my-alias")},
        )
        r = c.put("/admin/api/models/model-a/settings", json={
            "model_alias": None,
        })
        assert r.status_code == 200
        assert mgr.get_settings("model-a").model_alias is None

    def test_set_empty_string_becomes_none(self, tmp_path):
        """Empty string alias is normalised to None."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a"],
        )
        r = c.put("/admin/api/models/model-a/settings", json={
            "model_alias": "  ",
        })
        assert r.status_code == 200
        assert mgr.get_settings("model-a").model_alias is None


# =============================================================================
# CLI alias commands
# =============================================================================


class TestAliasCliSet:
    """Tests for 'omlx alias set' command."""

    def test_set_alias_success(self, tmp_path):
        """Setting an alias on a known model succeeds."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["my-model"],
        )
        r = c.put("/admin/api/models/my-model/settings", json={
            "model_alias": "my-alias",
        })
        assert r.status_code == 200
        assert r.json()["success"] is True
        assert mgr.get_settings("my-model").model_alias == "my-alias"

    def test_set_alias_conflict(self, tmp_path):
        """Setting an alias already held by another model — 400."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
            settings={"model-a": ModelSettings(model_alias="shared")},
        )
        r = c.put("/admin/api/models/model-b/settings", json={
            "model_alias": "shared",
        })
        assert r.status_code == 400
        assert "already used by" in r.json()["detail"]

    def test_set_alias_conflict_with_directory(self, tmp_path):
        """Alias equals another model's directory name — 400."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
        )
        r = c.put("/admin/api/models/model-a/settings", json={
            "model_alias": "model-b",
        })
        assert r.status_code == 400
        assert "conflicts with model directory name" in r.json()["detail"]


class TestAliasCliSwap:
    """Tests for 'omlx alias swap' command."""

    def test_swap_two_models_with_aliases(self, tmp_path):
        """Swapping two models each with an alias transfers them."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
            settings={
                "model-a": ModelSettings(model_alias="alias-a"),
                "model-b": ModelSettings(model_alias="alias-b"),
            },
        )
        r = c.post("/admin/api/models/swap-alias", json={
            "model_a": "model-a",
            "model_b": "model-b",
            "alias_a": "alias-b",
            "alias_b": "alias-a",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert mgr.get_settings("model-a").model_alias == "alias-b"
        assert mgr.get_settings("model-b").model_alias == "alias-a"

    def test_swap_one_side_clear(self, tmp_path):
        """Clear alias on A, give B's alias to A."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
            settings={
                "model-a": ModelSettings(model_alias="alias-a"),
                "model-b": ModelSettings(model_alias="alias-b"),
            },
        )
        r = c.post("/admin/api/models/swap-alias", json={
            "model_a": "model-a",
            "model_b": "model-b",
            "alias_a": None,
            "alias_b": "alias-a",
        })
        assert r.status_code == 200
        assert mgr.get_settings("model-a").model_alias is None
        assert mgr.get_settings("model-b").model_alias == "alias-a"

    def test_swap_missing_model(self, tmp_path):
        """One model doesn't exist — 404."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a"],
        )
        r = c.post("/admin/api/models/swap-alias", json={
            "model_a": "model-a",
            "model_b": "unknown",
            "alias_a": None,
            "alias_b": None,
        })
        assert r.status_code == 404


class TestAliasCliRemove:
    """Tests for 'omlx alias remove' command."""

    def test_remove_alias_success(self, tmp_path):
        """Removing an alias clears it."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["my-model"],
            settings={"my-model": ModelSettings(model_alias="my-alias")},
        )
        r = c.put("/admin/api/models/my-model/settings", json={
            "model_alias": None,
        })
        assert r.status_code == 200
        assert mgr.get_settings("my-model").model_alias is None

    def test_remove_alias_on_model_without_alias(self, tmp_path):
        """Removing alias from a model that already has none — still succeeds."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["my-model"],
        )
        r = c.put("/admin/api/models/my-model/settings", json={
            "model_alias": None,
        })
        assert r.status_code == 200
        assert mgr.get_settings("my-model").model_alias is None


class TestAliasCliList:
    """Tests for 'omlx alias list' command — verifies list_models output."""

    def test_list_includes_aliases(self, tmp_path):
        """list_models response includes model_alias in settings for real models."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a", "model-b"],
            settings={
                "model-a": ModelSettings(model_alias="alias-a"),
                "model-b": ModelSettings(model_alias="alias-b"),
            },
        )
        r = c.get("/admin/api/models")
        assert r.status_code == 200
        models = r.json()["models"]
        # list_models may include virtual models (MarkItDown), so check by ID
        by_id = {m["id"]: m for m in models}
        assert "model-a" in by_id
        assert "model-b" in by_id
        assert by_id["model-a"]["settings"]["model_alias"] == "alias-a"
        assert by_id["model-b"]["settings"]["model_alias"] == "alias-b"

    def test_list_shows_none_for_no_alias(self, tmp_path):
        """Models without alias show None/absent in response."""
        c, mgr = _make_client(
            tmp_path,
            model_ids=["model-a"],
        )
        r = c.get("/admin/api/models")
        assert r.status_code == 200
        models = r.json()["models"]
        by_id = {m["id"]: m for m in models}
        assert "model-a" in by_id
        ms = by_id["model-a"].get("settings", {})
        # None values are excluded from to_dict, so key may be absent
        assert "model_alias" not in ms or ms.get("model_alias") is None


# =============================================================================
# SwapAliasRequest pydantic model
# =============================================================================


class TestSwapAliasRequest:
    """Tests for the SwapAliasRequest pydantic model."""

    def test_default_aliases_are_none(self):
        """alias_a and alias_b default to None."""
        from omlx.admin.routes import SwapAliasRequest
        req = SwapAliasRequest(model_a="a", model_b="b")
        assert req.alias_a is None
        assert req.alias_b is None

    def test_aliases_can_be_set(self):
        """Alias fields accept string values."""
        from omlx.admin.routes import SwapAliasRequest
        req = SwapAliasRequest(
            model_a="a",
            model_b="b",
            alias_a="new-a",
            alias_b="new-b",
        )
        assert req.alias_a == "new-a"
        assert req.alias_b == "new-b"

    def test_aliases_can_be_explicit_none(self):
        """Explicit None is valid."""
        from omlx.admin.routes import SwapAliasRequest
        req = SwapAliasRequest(
            model_a="a",
            model_b="b",
            alias_a=None,
            alias_b=None,
        )
        assert req.alias_a is None
        assert req.alias_b is None


# =============================================================================
# CLI alias_command function
# =============================================================================


class TestAliasCliFunction:
    """Tests for the alias_command CLI function."""

    def _mock_server(self, models_data, monkeypatch):
        """Build a mock HTTP server response for the alias CLI commands."""
        from omlx.settings import GlobalSettings
        from types import SimpleNamespace
        import requests

        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json_data = json_data

            def json(self):
                return self._json_data

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.HTTPError(f"HTTP {self.status_code}")

        responses = {}

        def mock_get(url, **kwargs):
            if "health" in url:
                return MockResponse(200, {"status": "ok"})
            if "/api/models" in url:
                return MockResponse(200, {"models": models_data})
            return MockResponse(404, {})

        def mock_request(method, url, **kwargs):
            if "health" in url:
                return MockResponse(200, {"status": "ok"})
            if "/api/models" in url:
                return MockResponse(200, {"models": models_data})
            if "/admin/api/models" in url:
                body = kwargs.get("json", {})
                if method == "PUT" and "settings" in url:
                    mid = url.split("/")[4]
                    for m in models_data:
                        if m["id"] == mid:
                            if "model_alias" in body:
                                m["settings"] = {
                                    **m.get("settings", {}),
                                    "model_alias": body["model_alias"],
                                }
                            return MockResponse(
                                200,
                                {
                                    "success": True,
                                    "model_id": mid,
                                    "settings": m.get("settings", {}),
                                },
                            )
                if method == "POST" and "swap-alias" in url:
                    return MockResponse(200, {
                        "success": True,
                        "model_a": {
                            "old_alias": models_data[0].get("settings", {}).get("model_alias"),
                            "new_alias": body.get("alias_a"),
                        },
                        "model_b": {
                            "old_alias": models_data[1].get("settings", {}).get("model_alias"),
                            "new_alias": body.get("alias_b"),
                        },
                    })
            return MockResponse(404, {})

        monkeypatch.setattr(requests, "get", mock_get)
        monkeypatch.setattr(requests, "request", mock_request)
        monkeypatch.setattr(
            GlobalSettings, "load",
            lambda: SimpleNamespace(
                server=SimpleNamespace(port=8000),
                auth=SimpleNamespace(api_key=""),
            ),
        )

    def test_alias_set_success(self, monkeypatch, tmp_path):
        """alias set succeeds for a known model."""
        from omlx.cli import alias_command

        models = [
            {"id": "model-a", "settings": {}},
            {"id": "model-b", "settings": {}},
        ]
        self._mock_server(models, monkeypatch)

        args = SimpleNamespace(
            alias_command="set",
            model="model-a",
            alias="new-alias",
        )
        result = alias_command(args)
        assert result == 0

    def test_alias_set_conflict(self, monkeypatch):
        """alias set fails when alias is already used."""
        from omlx.cli import alias_command

        models = [
            {"id": "model-a", "settings": {}},
            {"id": "model-b", "settings": {"model_alias": "existing"}},
        ]
        self._mock_server(models, monkeypatch)

        args = SimpleNamespace(
            alias_command="set",
            model="model-a",
            alias="existing",
        )
        result = alias_command(args)
        assert result == 1

    def test_alias_set_unknown_model(self, monkeypatch):
        """alias set fails for unknown model."""
        from omlx.cli import alias_command

        models = [
            {"id": "model-a", "settings": {}},
        ]
        self._mock_server(models, monkeypatch)

        args = SimpleNamespace(
            alias_command="set",
            model="unknown-model",
            alias="some-alias",
        )
        result = alias_command(args)
        assert result == 1

    def test_alias_remove_success(self, monkeypatch, tmp_path):
        """alias remove succeeds."""
        from omlx.cli import alias_command

        models = [
            {"id": "model-a", "settings": {"model_alias": "old-alias"}},
        ]
        self._mock_server(models, monkeypatch)

        args = SimpleNamespace(
            alias_command="remove",
            model="model-a",
        )
        result = alias_command(args)
        assert result == 0

    def test_alias_swap_success(self, monkeypatch, tmp_path):
        """alias swap succeeds."""
        from omlx.cli import alias_command

        models = [
            {"id": "model-a", "settings": {"model_alias": "alias-a"}},
            {"id": "model-b", "settings": {"model_alias": "alias-b"}},
        ]
        self._mock_server(models, monkeypatch)

        args = SimpleNamespace(
            alias_command="swap",
            model_a="model-a",
            model_b="model-b",
            alias_a=None,
            alias_b=None,
        )
        result = alias_command(args)
        assert result == 0

    def test_alias_list_success(self, monkeypatch, tmp_path):
        """alias list succeeds and shows models with aliases."""
        from omlx.cli import alias_command

        models = [
            {
                "id": "model-a",
                "settings": {"model_alias": "alias-a"},
                "pinned": False,
                "is_default": False,
            },
            {
                "id": "model-b",
                "settings": {"model_alias": None},
                "pinned": True,
                "is_default": False,
            },
            {
                "id": "model-c",
                "settings": {},
                "pinned": False,
                "is_default": True,
            },
        ]
        self._mock_server(models, monkeypatch)

        args = SimpleNamespace(
            alias_command="list",
        )
        result = alias_command(args)
        assert result == 0

    def test_alias_no_command_shows_help(self, monkeypatch):
        """alias with no subcommand shows usage."""
        from omlx.cli import alias_command

        self._mock_server([], monkeypatch)

        args = SimpleNamespace(
            alias_command=None,
        )
        result = alias_command(args)
        assert result == 1

    def test_alias_swap_missing_model(self, monkeypatch):
        """alias swap fails when one model doesn't exist."""
        from omlx.cli import alias_command

        models = [
            {"id": "model-a", "settings": {"model_alias": "alias-a"}},
        ]
        self._mock_server(models, monkeypatch)

        args = SimpleNamespace(
            alias_command="swap",
            model_a="model-a",
            model_b="unknown",
            alias_a=None,
            alias_b=None,
        )
        result = alias_command(args)
        assert result == 1

    def test_server_not_running(self, monkeypatch):
        """alias commands fail when server is not running."""
        import requests

        monkeypatch.setattr(
            requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError())
        )
        from omlx.cli import alias_command

        args = SimpleNamespace(
            alias_command="set",
            model="model-a",
            alias="alias",
        )
        result = alias_command(args)
        assert result == 1
