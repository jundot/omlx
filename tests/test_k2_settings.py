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
    assert ModelSettings(qwen35_ane_prefill_enabled=True).qwen35_ane_prefill_enabled
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    ordinary = ModelSettings()
    ane = ModelSettings(qwen35_ane_prefill_enabled=True)
    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings(base.name, ane)
    assert (
        ModelSettingsManager(tmp_path / "settings")
        .get_settings(base.name)
        .qwen35_ane_prefill_enabled
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
        {"qwen35_ane_prefill_fraction": 0},
        {"qwen35_ane_prefill_sequence_length": 7},
    ):
        request = routes.ModelSettingsRequest(
            qwen35_ane_prefill_enabled=enabled, **values
        )
        with pytest.raises(HTTPException) as error:
            await routes.update_model_settings(base.name, request, is_admin=True)
        assert error.value.status_code == 400
        assert not manager.get_settings(base.name).qwen35_ane_prefill_enabled


def test_k2_ane_profile_persists_without_becoming_a_global_template(tmp_path):
    from omlx.model_profiles import filter_universal_fields

    fields = dict(
        qwen35_ane_prefill_enabled=True,
        qwen35_ane_prefill_fraction=0.5,
        qwen35_ane_prefill_shared_fraction=1.0,
        qwen35_ane_prefill_sequence_length=2048,
    )
    manager = ModelSettingsManager(tmp_path)
    manager.save_profile("mova", "ane", "ANE", None, fields)
    restored = ModelSettingsManager(tmp_path).apply_profile("mova", "ane")
    assert all(getattr(restored, key) == value for key, value in fields.items())
    assert filter_universal_fields(fields) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_type,forced,backend",
    [("k2_horizon", True, "k2"), ("qwen3_5", False, "qwen"),
     ("qwen3-6_moe", False, "qwen"), ("qwen3_8", False, "qwen"),
     ("llama", False, None), ("unknown", False, None)],
)
async def test_admin_model_response_describes_controls(
    tmp_path, monkeypatch, model_type, forced, backend
):
    from unittest.mock import MagicMock

    from omlx.admin import routes

    pool = MagicMock()
    pool.get_status.return_value = {"models": [{"id": "model", "config_model_type": model_type}]}
    manager = ModelSettingsManager(tmp_path)
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(routes, "_get_server_state", lambda: None)
    monkeypatch.setattr(routes, "_get_global_settings", lambda: None)
    model = (await routes.list_models(is_admin=True))["models"][0]
    assert model["thinking_forced"] is forced
    assert model["thinking_modes"] == (
        ["auto", "on_limit"] if forced else ["auto", "on_unlimit", "on_limit", "off"]
    )
    assert model["reasoning_effort_options"] == (
        ["low", "medium", "high"] if forced else ["low", "medium", "high", "xhigh", "max"]
    )
    assert model["reasoning_effort_default"] == ("high" if forced else "low")
    assert model["reasoning_effort_custom"] is not forced
    assert model["ane_prefill_backend"] == backend
    assert model["ane_prefill_mlp_fractions"] == ([1 / 3, 0.5] if forced else [])
    assert model["ane_prefill_shared_fractions"] == ([0, 1 / 3, 1] if forced else [])


def test_helper_model_does_not_offer_k2_ane():
    from omlx.admin.routes import _model_options

    assert _model_options(
        {"config_model_type": "k2_horizon", "is_helper": True}, None
    )["ane_prefill_backend"] is None


@pytest.mark.parametrize(
    "model_type,default", [("k2_horizon", 1 / 3), ("qwen3_5", 0.53)]
)
def test_shared_ane_defaults_and_explicit_settings(model_type, default):
    from omlx.model_settings import ane_prefill_fraction, validate_ane_prefill

    settings = ModelSettings(qwen35_ane_prefill_enabled=True)
    validate_ane_prefill(settings.to_dict(), model_type)
    assert (
        ane_prefill_fraction(settings.qwen35_ane_prefill_fraction, model_type)
        == default
    )
    settings.qwen35_ane_prefill_fraction = 0.53
    restored = ModelSettings.from_dict(settings.to_dict())
    assert (
        ane_prefill_fraction(restored.qwen35_ane_prefill_fraction, model_type) == 0.53
    )
    assert not any(key.startswith("k2_ane_") for key in restored.to_dict())


@pytest.mark.parametrize("enabled", [False, True])
def test_shared_ane_signature_uses_only_effective_backend_controls(
    models, tmp_path, enabled
):
    from dataclasses import replace

    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    settings = ModelSettings(qwen35_ane_prefill_enabled=enabled)
    signature = lambda value: pool._engine_runtime_signature(models.name, value)
    original = signature(settings)
    assert (
        signature(
            replace(
                settings,
                qwen35_ane_prefill_cpu_enabled=True,
                qwen35_ane_prefill_gdn=False,
            )
        )
        == original
    )
    for change in (
        {"qwen35_ane_prefill_fraction": 0.5},
        {"qwen35_ane_prefill_shared_fraction": 0},
        {"qwen35_ane_prefill_sequence_length": 4096},
    ):
        assert (signature(replace(settings, **change)) != original) is enabled
    pool.get_entry(models.name).config_model_type = "qwen3_5"
    qwen = signature(settings)
    assert (qwen != original) is enabled
    assert signature(replace(settings, qwen35_ane_prefill_shared_fraction=0)) == qwen
    assert (
        signature(replace(settings, qwen35_ane_prefill_gdn=False)) != qwen
    ) is enabled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_type,width,fraction", [("k2_horizon", 32, 1.0), ("qwen3_5", 1024, 0.53)]
)
async def test_shared_ane_manual_enable_and_profile_validation(
    models, tmp_path, monkeypatch, model_type, width, fraction
):
    from fastapi import HTTPException
    from omlx.admin import routes

    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    pool.get_entry(models.name).config_model_type = model_type
    manager = ModelSettingsManager(tmp_path / "settings")
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(routes, "_get_server_state", lambda: None)
    request = routes.ModelSettingsRequest(
        qwen35_ane_prefill_enabled=True,
        qwen35_ane_prefill_sequence_length=width,
        qwen35_ane_prefill_fraction=fraction,
    )
    await routes.update_model_settings(models.name, request, is_admin=True)
    saved = manager.get_settings(models.name)
    assert saved.qwen35_ane_prefill_enabled
    assert saved.qwen35_ane_prefill_sequence_length == width
    assert saved.qwen35_ane_prefill_fraction == fraction
    manager.save_profile(
        models.name, "bad", "Bad", None, {"qwen35_ane_prefill_fraction": 2}
    )
    with pytest.raises(HTTPException) as error:
        await routes.apply_model_profile(models.name, "bad", is_admin=True)
    assert error.value.status_code == 400
    assert manager.get_settings(models.name).to_dict() == saved.to_dict()


@pytest.mark.parametrize(
    "flag", ["dflash_enabled", "specprefill_enabled", "mtp_enabled", "vlm_mtp_enabled"]
)
def test_k2_ane_conflicts_checked_with_model_metadata(flag):
    from omlx.model_settings import validate_ane_prefill

    settings = ModelSettings(qwen35_ane_prefill_enabled=True, **{flag: True})
    with pytest.raises(ValueError, match=flag):
        validate_ane_prefill(settings.to_dict(), "k2_horizon")
