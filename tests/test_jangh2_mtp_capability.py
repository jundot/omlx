"""Packed head tensors must not advertise executable Lightning MTP."""

import json
from types import SimpleNamespace

import pytest

from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.utils.model_loading import lightning_mtp_supported


def _checkpoint(path, *, packed):
    path.mkdir()
    config = {
        "model_type": "qwen4_exp",
        "text_config": {"mtp_num_hidden_layers": 1},
    }
    if packed:
        config["jangtq"] = {"version": 2}
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc_hidden.weight": "model.safetensors"}})
    )
    return EngineEntry(
        model_id=path.name,
        model_path=str(path),
        model_type="vlm",
        engine_type="vlm",
        config_model_type="qwen4_exp",
        estimated_size=1,
    )


@pytest.mark.parametrize("packed, supported", [(True, False), (False, True)])
def test_mtp_capability_distinguishes_packed_heads(tmp_path, packed, supported):
    # A model name is not a format declaration.
    entry = _checkpoint(tmp_path / "anything-JANGH2", packed=packed)
    assert lightning_mtp_supported(entry.model_path) is supported
    assert EnginePool._starts_with_lightning_mtp(entry) is supported


def test_initial_settings_do_not_enable_packed_mtp(tmp_path):
    packed = _checkpoint(tmp_path / "packed", packed=True)
    ordinary = _checkpoint(tmp_path / "ordinary", packed=False)
    pool = EnginePool()
    pool._entries = {entry.model_id: entry for entry in (packed, ordinary)}
    manager = ModelSettingsManager(tmp_path / "settings")

    pool.apply_settings_overrides(manager)

    assert manager.get_settings("packed").mtp_enabled is False
    assert manager.get_settings("ordinary").mtp_enabled is True


@pytest.mark.parametrize("packed", [True, False])
def test_effective_settings_gate_stale_packed_mtp_only(tmp_path, monkeypatch, packed):
    entry = _checkpoint(tmp_path / "checkpoint", packed=packed)
    pool = EnginePool()
    # Capability gating must also work if a PLE estimate is unavailable.
    monkeypatch.setattr(
        pool, "_qwen4_ple_offload_status", lambda *a, **kw: (False, False, None)
    )
    settings = SimpleNamespace(mtp_enabled=True, vlm_mtp_enabled=True)

    effective = pool._effective_qwen4_model_settings(entry, settings)

    assert effective.mtp_enabled is not packed
    assert effective.vlm_mtp_enabled is not packed
    assert settings.mtp_enabled is True
    assert settings.vlm_mtp_enabled is True
    assert (effective is settings) is not packed


@pytest.mark.parametrize("packed", [True, False])
def test_stale_mtp_cannot_force_packed_engine_reload(tmp_path, monkeypatch, packed):
    entry = _checkpoint(tmp_path / "checkpoint", packed=packed)
    pool = EnginePool()
    pool._entries[entry.model_id] = entry
    monkeypatch.setattr(
        pool, "_qwen4_ple_offload_status", lambda *a, **kw: (False, False, None)
    )
    enabled = ModelSettings(mtp_enabled=True, mtp_adaptive_max_depth=8)
    disabled = ModelSettings(mtp_enabled=False)

    enabled_signature = pool._engine_runtime_signature(entry.model_id, enabled)
    disabled_signature = pool._engine_runtime_signature(entry.model_id, disabled)

    assert (enabled_signature == disabled_signature) is packed
    assert enabled.mtp_enabled is True
