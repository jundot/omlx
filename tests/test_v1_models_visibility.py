# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/models visibility filtering (per-model hide + global helper hide)."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.server import ServerState, app
from omlx.settings import GlobalSettings


class _Pool:
    """Engine pool stub exposing a fixed model list via get_status()."""

    def __init__(self, models: list[dict]):
        self._models = models

    def resolve_model_id(self, model_id_or_alias: str, settings_manager) -> str:
        return model_id_or_alias

    def get_status(self) -> dict:
        return {
            "final_ceiling": 0,
            "current_model_memory": 0,
            "model_count": len(self._models),
            "loaded_count": 0,
            "models": self._models,
        }


def _model(model_id: str, config_model_type: str = "llama", **extra) -> dict:
    return {
        "id": model_id,
        "model_path": f"/models/{model_id}",
        "config_model_type": config_model_type,
        "source_repo_id": None,
        **extra,
    }


def _state(models: list[dict], tmp_path, *, hide_helpers: bool = False) -> ServerState:
    state = ServerState()
    state.engine_pool = _Pool(models)
    state.settings_manager = ModelSettingsManager(base_path=tmp_path)
    gs = GlobalSettings()
    gs.model.hide_helper_models = hide_helpers
    state.global_settings = gs
    return state


def _list_ids(state) -> list[str]:
    with (
        patch("omlx.server._server_state", state),
        patch("omlx.server.get_max_context_window", return_value=None),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")
    assert response.status_code == 200
    return [m["id"] for m in response.json()["data"]]


def test_lists_all_models_by_default(tmp_path):
    state = _state([_model("chat-a"), _model("chat-b")], tmp_path)
    assert _list_ids(state) == ["chat-a", "chat-b"]


def test_per_model_hidden_excluded(tmp_path):
    state = _state([_model("chat-a"), _model("chat-b")], tmp_path)
    state.settings_manager.set_settings("chat-b", ModelSettings(is_hidden=True))
    ids = _list_ids(state)
    assert "chat-a" in ids
    assert "chat-b" not in ids


def test_hidden_source_profile_excluded(tmp_path):
    state = _state([_model("chat-a")], tmp_path)
    state.settings_manager.set_settings("chat-a", ModelSettings(is_hidden=True))
    state.settings_manager.save_profile(
        "chat-a",
        "fast",
        "Fast",
        None,
        {"temperature": 0.1},
        expose_as_model=True,
    )
    assert _list_ids(state) == []


def test_hidden_ignored_when_helper_toggle_off(tmp_path):
    # A drafter is visible while the global toggle is off.
    models = [_model("chat-a"), _model("drafter", is_helper=True)]
    state = _state(models, tmp_path, hide_helpers=False)
    assert "drafter" in _list_ids(state)


def test_global_hide_helper_excludes_intrinsic_flag(tmp_path):
    # Discovery flags drafters (config marker / architecture) via is_helper.
    models = [_model("chat-a"), _model("drafter", is_helper=True)]
    state = _state(models, tmp_path, hide_helpers=True)
    ids = _list_ids(state)
    assert ids == ["chat-a"]


def test_global_hide_helper_excludes_helper_source_profile(tmp_path):
    models = [_model("chat-a"), _model("drafter", is_helper=True)]
    state = _state(models, tmp_path, hide_helpers=True)
    state.settings_manager.save_profile(
        "drafter",
        "fast",
        "Fast",
        None,
        {"temperature": 0.1},
        expose_as_model=True,
    )
    assert _list_ids(state) == ["chat-a"]


def test_global_hide_helper_excludes_referenced_draft(tmp_path):
    # A plain-LLM draft is a helper only because another model references it.
    models = [_model("chat-a"), _model("draft-llm", config_model_type="llama")]
    state = _state(models, tmp_path, hide_helpers=True)
    state.settings_manager.set_settings(
        "chat-a", ModelSettings(dflash_draft_model="/models/draft-llm")
    )
    ids = _list_ids(state)
    assert "chat-a" in ids
    assert "draft-llm" not in ids


def test_referenced_parent_stays_visible(tmp_path):
    # The referencing chat model itself is never treated as a helper.
    models = [_model("chat-a"), _model("draft-llm")]
    state = _state(models, tmp_path, hide_helpers=True)
    state.settings_manager.set_settings(
        "chat-a", ModelSettings(vlm_mtp_draft_model="/models/draft-llm")
    )
    assert "chat-a" in _list_ids(state)


def test_favorites_listed_first(tmp_path):
    models = [_model("chat-a"), _model("chat-b"), _model("chat-c")]
    state = _state(models, tmp_path)
    state.settings_manager.set_settings("chat-c", ModelSettings(is_favorite=True))
    assert _list_ids(state) == ["chat-c", "chat-a", "chat-b"]


def test_favorites_alphabetical_within_groups(tmp_path):
    models = [_model("chat-a"), _model("chat-b"), _model("chat-c"), _model("chat-d")]
    state = _state(models, tmp_path)
    state.settings_manager.set_settings("chat-b", ModelSettings(is_favorite=True))
    state.settings_manager.set_settings("chat-d", ModelSettings(is_favorite=True))
    assert _list_ids(state) == ["chat-b", "chat-d", "chat-a", "chat-c"]


def test_favorite_alias_listed_first(tmp_path):
    # The favorite set must track display ids, since an alias replaces the id.
    models = [_model("chat-a"), _model("chat-b")]
    state = _state(models, tmp_path)
    state.settings_manager.set_settings(
        "chat-b", ModelSettings(is_favorite=True, model_alias="zz-alias")
    )
    assert _list_ids(state) == ["zz-alias", "chat-a"]


def test_hidden_favorite_still_excluded(tmp_path):
    # Favorite is ordering-only and never overrides the hidden filter.
    models = [_model("chat-a"), _model("chat-b")]
    state = _state(models, tmp_path)
    state.settings_manager.set_settings(
        "chat-b", ModelSettings(is_hidden=True, is_favorite=True)
    )
    assert _list_ids(state) == ["chat-a"]
