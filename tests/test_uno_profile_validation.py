# SPDX-License-Identifier: Apache-2.0
"""Effective Uno profile validation and recovery of invalid saved aliases."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from omlx.model_settings import (
    InvalidProfileSettingsError,
    ModelSettings,
    ModelSettingsManager,
)


def save_profile(manager, name, settings):
    return manager.save_profile(
        "base",
        name,
        name.title(),
        None,
        settings,
        expose_as_model=True,
        api_name=name,
    )


def uno_settings(**kwargs):
    return ModelSettings(uno_enabled=True, uno_adapter_model="adapter", **kwargs)


def test_profile_create_and_update_reject_inherited_conflict_atomically(tmp_path):
    manager = ModelSettingsManager(tmp_path)
    manager.set_settings("base", uno_settings())
    save_profile(manager, "valid", {"temperature": 0.2})
    before = manager.profiles_file.read_bytes()
    with pytest.raises(InvalidProfileSettingsError, match="base:broken.*min_p"):
        save_profile(manager, "broken", {"min_p": 0.05})
    with pytest.raises(InvalidProfileSettingsError, match="base:valid.*min_p"):
        manager.update_profile("base", "valid", settings={"min_p": 0.05})
    assert manager.profiles_file.read_bytes() == before
    assert manager.get_profile("base", "broken") is None
    assert manager.get_profile("base", "valid")["settings"] == {"temperature": 0.2}


@pytest.mark.parametrize("apply", [False, True])
def test_base_save_and_profile_application_name_all_conflicts_without_writing(
    tmp_path, apply
):
    manager = ModelSettingsManager(tmp_path)
    manager.set_settings("base", ModelSettings(temperature=0.3))
    save_profile(manager, "alpha", {"min_p": 0.05})
    save_profile(manager, "beta", {"repetition_penalty": 1.2})
    save_profile(manager, "uno", uno_settings().to_dict())
    before = manager.settings_file.read_bytes(), manager.profiles_file.read_bytes()
    with pytest.raises(InvalidProfileSettingsError) as error:
        if apply:
            manager.apply_profile("base", "uno")
        else:
            manager.set_settings("base", uno_settings(is_default=True))
    assert "base:alpha" in str(error.value) and "min_p" in str(error.value)
    assert "base:beta" in str(error.value) and "repetition_penalty" in str(error.value)
    assert (
        manager.settings_file.read_bytes(),
        manager.profiles_file.read_bytes(),
    ) == before
    assert manager.get_settings("base") == ModelSettings(temperature=0.3)


def test_profile_engine_overrides_are_respected_in_both_views(tmp_path):
    manager = ModelSettingsManager(tmp_path)
    manager.set_settings("base", ModelSettings(mtp_enabled=True))
    save_profile(
        manager,
        "uno",
        {
            "uno_enabled": True,
            "uno_adapter_model": "adapter",
            "mtp_enabled": False,
        },
    )
    assert manager.get_settings_for_request("base:uno").uno_enabled
    assert not manager.get_settings_for_request("base:uno").mtp_enabled
    manager.set_settings("base", uno_settings())
    save_profile(manager, "ordinary", {"uno_enabled": False, "min_p": 0.05})
    settings = manager.get_settings_for_request("base:ordinary")
    _, runtime = manager.get_exposed_profile_runtime_settings_for_request(
        "base:ordinary"
    )
    assert settings.min_p == runtime.min_p == 0.05
    assert not settings.uno_enabled and not runtime.uno_enabled


@pytest.fixture
def legacy_manager(tmp_path):
    """Load a profile that an older version accepted, without bypassing read paths."""
    manager = ModelSettingsManager(tmp_path)
    manager.set_settings("base", uno_settings())
    save_profile(manager, "broken", {"min_p": 0})
    data = json.loads(manager.profiles_file.read_text())
    data["profiles"]["base"]["broken"]["settings"]["min_p"] = 0.05
    manager.profiles_file.write_text(json.dumps(data))
    return ModelSettingsManager(tmp_path)


def test_invalid_saved_profile_stays_listed_with_reason_and_warning(
    legacy_manager, caplog
):
    manager = legacy_manager
    before = manager.profiles_file.read_bytes()
    [profile] = manager.list_exposed_profile_models()
    assert profile["model_id"] == "base:broken"
    assert profile["invalid"] is True
    assert "min_p" in profile["invalid_reason"]
    assert "base:broken" in caplog.text
    for getter in (
        manager.get_settings_for_request,
        manager.get_exposed_profile_runtime_settings_for_request,
    ):
        with pytest.raises(InvalidProfileSettingsError, match="base:broken.*min_p"):
            getter("base:broken")
    assert manager.profiles_file.read_bytes() == before
    manager.update_profile(
        "base", "broken", settings={"uno_enabled": False, "min_p": 0.05}
    )
    [profile] = manager.list_exposed_profile_models()
    assert profile["invalid"] is False and profile["invalid_reason"] is None


@pytest.fixture
def api_client(legacy_manager, monkeypatch):
    import omlx.server as server

    pool = MagicMock()
    pool.get_status.return_value = {
        "models": [
            {
                "id": "base",
                "loaded": False,
                "pinned": False,
                "engine_type": "batched",
                "model_type": "llm",
                "config_model_type": "k2_horizon",
            }
        ]
    }
    pool.resolve_model_id.side_effect = lambda name, *_: (
        "base" if name == "base:broken" else name
    )
    pool.get_entry.return_value.model_context_length = 4096
    pool.get_engine = AsyncMock(
        side_effect=AssertionError("must reject before loading")
    )
    state = server.ServerState(engine_pool=pool, settings_manager=legacy_manager)
    monkeypatch.setattr(server, "_server_state", state)
    monkeypatch.setattr(server, "_markitdown_is_visible", lambda: False)
    monkeypatch.setitem(
        server.app.dependency_overrides, server.verify_api_key, lambda: True
    )
    return TestClient(server.app, raise_server_exceptions=False), pool


@pytest.mark.parametrize("endpoint", ["/v1/models", "/v1/models/status"])
def test_model_listing_keeps_invalid_profile_and_healthy_base(api_client, endpoint):
    client, pool = api_client
    response = client.get(endpoint)
    assert response.status_code == 200, response.text
    rows = response.json()["data" if endpoint == "/v1/models" else "models"]
    assert {row["id"] for row in rows} == {"base", "base:broken"}
    profile = next(row for row in rows if row["id"] == "base:broken")
    assert profile["invalid"] is True
    assert (
        "base:broken" in profile["invalid_reason"]
        and "min_p" in profile["invalid_reason"]
    )
    pool.get_engine.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "endpoint",
    [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
    ],
)
def test_invalid_alias_returns_400_before_inference(api_client, endpoint, stream):
    client, pool = api_client
    body = {"model": "base:broken", "max_tokens": 16, "stream": stream}
    if endpoint == "/v1/completions":
        body["prompt"] = "Hello"
    elif endpoint == "/v1/responses":
        body["input"] = "Hello"
    else:
        body["messages"] = [{"role": "user", "content": "Hello"}]
    response = client.post(endpoint, json=body)
    assert response.status_code == 400, response.text
    assert "base:broken" in response.text and "min_p" in response.text
    pool.get_engine.assert_not_called()
