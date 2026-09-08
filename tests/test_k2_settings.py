# SPDX-License-Identifier: Apache-2.0
"""K2 request validation and ANE settings."""

import json

import pytest

from omlx.engine_pool import EnginePool
from omlx.model_settings import ModelSettings, ModelSettingsManager


@pytest.fixture
def models(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps(
            dict(
                model_type="k2_horizon",
                hidden_size=1536,
                num_hidden_layers=28,
                intermediate_size=5120,
                vocab_size=64256,
                max_position_embeddings=131072,
            )
        )
    )
    (base / "model.safetensors").write_bytes(b"fixture")
    return base


@pytest.mark.parametrize(
    "kwargs",
    [{"reasoning_effort": value} for value in ("off", "xhigh", "max", 1, None)]
    + [{"enable_thinking": False}],
)
def test_k2_rejects_unsupported_kwargs_before_template_fallback(kwargs):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from omlx.engine.batched import BatchedEngine
    from omlx.exceptions import InvalidRequestError

    for engine in (BatchedEngine("base"),):
        engine._tokenizer = MagicMock()
        engine._model = SimpleNamespace(args=SimpleNamespace(model_type="k2_horizon"))
        render = engine._apply_chat_template
        with pytest.raises(InvalidRequestError, match="K2"):
            render([{"role": "user", "content": "Hello"}], chat_template_kwargs=kwargs)
        engine._tokenizer.apply_chat_template.assert_not_called()


def test_k2_ane_setting_roundtrip_and_reservation(models, tmp_path):
    from omlx.patches.k2_horizon.ane_prefill import prefill_memory_reservation

    base = models
    assert ModelSettings(k2_ane_prefill_enabled=True).k2_ane_prefill_enabled
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    ordinary = ModelSettings()
    ane = ModelSettings(k2_ane_prefill_enabled=True)
    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings(base.name, ane)
    assert (
        ModelSettingsManager(tmp_path / "settings")
        .get_settings(base.name)
        .k2_ane_prefill_enabled
    )
    assert pool._engine_runtime_signature(
        base.name, ordinary
    ) != pool._engine_runtime_signature(base.name, ane)
    entry = pool.get_entry(base.name)
    config = json.loads((base / "config.json").read_text())
    assert pool._entry_runtime_resident_size(
        entry, ane
    ) - pool._entry_runtime_resident_size(
        entry, ordinary
    ) == prefill_memory_reservation(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_admin_rejects_invalid_k2_ane_settings(
    models, tmp_path, monkeypatch, enabled
):
    from fastapi import HTTPException

    from omlx.admin import routes

    base = models
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    manager = ModelSettingsManager(tmp_path / "settings")
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(routes, "_get_server_state", lambda: None)
    for values in (
        {"k2_ane_prefill_fraction": 0},
        {"k2_ane_prefill_sequence_length": 7},
    ):
        request = routes.ModelSettingsRequest(k2_ane_prefill_enabled=enabled, **values)
        with pytest.raises(HTTPException) as error:
            await routes.update_model_settings(base.name, request, is_admin=True)
        assert error.value.status_code == 400
        assert not manager.get_settings(base.name).k2_ane_prefill_enabled


def test_k2_ane_profile_persists_without_becoming_a_global_template(tmp_path):
    from omlx.model_profiles import filter_universal_fields

    fields = dict(
        k2_ane_prefill_enabled=True,
        k2_ane_prefill_fraction=0.5,
        k2_ane_prefill_shared_fraction=1.0,
        k2_ane_prefill_sequence_length=2048,
    )
    manager = ModelSettingsManager(tmp_path)
    manager.save_profile("mova", "ane", "ANE", None, fields)
    restored = ModelSettingsManager(tmp_path).apply_profile("mova", "ane")
    assert all(getattr(restored, key) == value for key, value in fields.items())
    assert filter_universal_fields(fields) == {}
