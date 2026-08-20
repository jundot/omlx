# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical DS4 identities exposed by the server."""

from contextlib import contextmanager
from unittest.mock import patch
from urllib.parse import quote

from fastapi.testclient import TestClient

from omlx.engine_pool import EnginePool
from omlx.model_settings import ModelSettings
from omlx.server import ServerState, app
from omlx.settings import DS4Settings


class _SettingsManager:
    def __init__(self, settings: dict[str, ModelSettings] | None = None):
        self._settings = settings or {}

    def get_settings(self, model_id: str) -> ModelSettings:
        return self._settings.get(model_id, ModelSettings())

    def get_all_settings(self) -> dict[str, ModelSettings]:
        return self._settings

    def list_exposed_profile_models(self) -> list[dict]:
        return []


def _ds4_pool(
    tmp_path,
    filename: str = "DeepSeek V4 Flash Q2_K.gguf",
    *,
    ds4_settings: DS4Settings | None = None,
) -> EnginePool:
    (tmp_path / filename).write_bytes(b"0" * 1000)
    pool = EnginePool(ds4_settings=ds4_settings)
    pool.discover_models(str(tmp_path))
    return pool


@contextmanager
def _client_for_pool(pool: EnginePool, settings_manager: _SettingsManager | None = None):
    state = ServerState()
    state.engine_pool = pool
    state.settings_manager = settings_manager or _SettingsManager()
    state.api_key = None
    with patch("omlx.server._server_state", state):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


def test_models_list_has_one_entry_for_each_ds4_model(tmp_path):
    pool = _ds4_pool(tmp_path)
    model_id = pool.get_model_ids()[0]

    with _client_for_pool(pool) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == [model_id]


def test_models_list_advertises_only_the_engine_context(tmp_path):
    pool = _ds4_pool(
        tmp_path,
        ds4_settings=DS4Settings(context_default_tokens=100_000),
    )
    model_id = pool.get_model_ids()[0]

    with _client_for_pool(pool) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    models = response.json()["data"]
    assert len(models) == 1
    assert models[0]["id"] == model_id
    assert models[0]["object"] == "model"
    assert models[0]["owned_by"] == "omlx"
    assert models[0]["max_model_len"] == 100_000


def test_models_list_uses_configured_alias_without_generated_variants(tmp_path):
    pool = _ds4_pool(tmp_path, filename="Foo.gguf")
    settings_manager = _SettingsManager({"foo": ModelSettings(model_alias="gpt-4o")})

    with _client_for_pool(pool, settings_manager) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == ["gpt-4o"]


def test_model_detail_requires_the_exact_published_identity(tmp_path):
    pool = _ds4_pool(tmp_path, filename="Foo.gguf")

    with _client_for_pool(pool) as client:
        exact = client.get("/v1/models/foo")
        removed = client.get("/v1/models/foo-" + "reasoner")

    assert exact.status_code == 200
    assert exact.json()["id"] == "foo"
    assert removed.status_code == 404
    assert removed.json()["error"]["code"] == "model_not_found"


def test_models_status_has_no_generated_ds4_identity_list(tmp_path):
    pool = _ds4_pool(tmp_path, filename="Foo.gguf")
    settings_manager = _SettingsManager({"foo": ModelSettings(model_alias="gpt-4o")})

    with _client_for_pool(pool, settings_manager) as client:
        response = client.get("/v1/models/status")

    assert response.status_code == 200
    model = response.json()["models"][0]
    assert model["id"] == "foo"
    assert model["model_alias"] == "gpt-4o"
    assert f"ds4_{'aliases'}" not in model


def test_public_load_resolves_ds4_source_filename(tmp_path):
    """Manual load accepts a discovered source GGUF filename."""
    filename = "DeepSeek V4 Flash Q2_K.gguf"
    pool = _ds4_pool(tmp_path, filename=filename)
    model_id = "deepseek-v4-flash-q2-k"
    pool._entries[model_id].engine = object()

    with _client_for_pool(pool) as client:
        response = client.post(f"/v1/models/{quote(filename)}/load")

    assert response.status_code == 200
    assert response.json()["model_id"] == model_id
