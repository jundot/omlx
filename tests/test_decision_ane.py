# SPDX-License-Identifier: Apache-2.0
"""Decision serving must use the same opt-in ANE backend as other engines."""

import json
import threading
from types import SimpleNamespace

import pytest

from omlx.engine import decision as engine_module
from omlx.engine.decision import DecisionEngine
from omlx.model_settings import (
    ModelSettings,
    ModelSettingsManager,
)
from omlx.models.decision import DecisionBackbone, ane_memory_reservation
from omlx.patches import qwen35_ane_prefill as ane_patch


@pytest.fixture
def model_path(tmp_path):
    (tmp_path / "joint_head.safetensors").touch()
    (tmp_path / "joint_head_config.json").write_text("{}")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 256,
                    "intermediate_size": 512,
                    "num_hidden_layers": 4,
                    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                    "linear_num_value_heads": 2,
                    "linear_value_head_dim": 64,
                    "linear_num_key_heads": 1,
                    "linear_key_head_dim": 64,
                },
            }
        )
    )
    return tmp_path


@pytest.fixture
def settings():
    return ModelSettings(
        ane_prefill_enabled=True,
        ane_prefill_sequence_length=1024,
        ane_prefill_fraction=0.25,
        ane_prefill_dual_ane=False,
        ane_prefill_gdn=False,
    )


def test_admission_reserves_ane_memory_before_load(model_path, settings):
    from omlx.engine_pool import EngineEntry, EnginePool

    reservation = ane_memory_reservation(str(model_path), settings)
    assert reservation > 4 * 2 * 128 * 256  # At least the selected INT8 rows.
    assert ane_memory_reservation(str(model_path), ModelSettings()) == 0
    pool = EnginePool()
    entry = EngineEntry(
        model_id="clef",
        model_path=str(model_path),
        model_type="decision",
        engine_type="decision",
        estimated_size=1000,
    )
    assert pool._entry_runtime_resident_size(entry, settings) == 1000 + reservation
    assert (
        pool._entry_runtime_resident_size(
            entry, settings, include_ane_reservation=False
        )
        == 1000
    )
    settings.ane_prefill_gdn = True
    assert ane_memory_reservation(str(model_path), settings) > reservation


def test_fused_dual_ane_reserves_fp16_weights_on_both_instances(model_path, settings):
    settings.ane_prefill_dual_ane = True
    settings.ane_prefill_fused_down = True
    # Four layers, three matrices, 128 hidden channels on each ANE, FP16.
    both_weights = 4 * 3 * 128 * 256 * 2 * 2
    assert ane_memory_reservation(str(model_path), settings) > both_weights


@pytest.mark.asyncio
async def test_ane_reservation_can_reject_an_oversized_load(
    model_path, settings, monkeypatch
):
    from omlx import engine_pool
    from omlx.exceptions import ModelTooLargeError

    pool = engine_pool.EnginePool()
    reservation = ane_memory_reservation(str(model_path), settings)
    pool._get_final_ceiling = lambda: reservation
    pool._entries["clef"] = engine_pool.EngineEntry(
        model_id="clef",
        model_path=str(model_path),
        model_type="decision",
        engine_type="decision",
        config_model_type="qwen3_5",
        estimated_size=1000,
    )
    monkeypatch.setattr(engine_pool, "_settled_phys_footprint", lambda: 0)
    monkeypatch.setattr(engine_pool.mx, "get_active_memory", lambda: 0)
    with pytest.raises(ModelTooLargeError):
        await pool.get_engine("clef", runtime_settings=settings)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("namespace", ["neutral", "qwen", "clef"])
async def test_settings_enable_backbone_on_the_mlx_executor(
    model_path, settings, monkeypatch, enabled, namespace
):
    events = []
    backbone = SimpleNamespace(
        model=SimpleNamespace(),
        config={"model_type": "qwen3_5"},
        tokenizer=object(),
        has_vision=False,
    )

    class Model:
        def __init__(self, *args, **kwargs):
            self.backbone = backbone

        def load(self):
            events.append(("load", threading.get_ident()))

        def close(self):
            events.append(("close", threading.get_ident()))

    def enable(model, **kwargs):
        assert model is backbone.model
        assert kwargs["dual_ane"] is False
        assert kwargs["fraction"] == 0.25
        assert kwargs["sequence_length"] == 1024
        events.append(("ane", threading.get_ident()))
        model._omlx_ane_mlp_prefill_count = 4
        return 4

    settings.ane_prefill_enabled = enabled
    if namespace != "neutral":
        prefix = "qwen35_ane_prefill_" if namespace == "qwen" else "clef_flash_ane_"
        settings = ModelSettings.from_dict({
            prefix + "enabled": enabled,
            prefix + "sequence_length": 1024,
            prefix + "fraction": 0.25,
            prefix + "dual_ane": False,
            prefix + "gdn": False,
        })
    monkeypatch.setattr(engine_module, "decision_kind", lambda path: "clef")
    monkeypatch.setitem(engine_module._MODEL_CLASSES, "clef", Model)
    monkeypatch.setattr(ane_patch, "enable_qwen35_ane_prefill", enable)
    engine = DecisionEngine(str(model_path), model_settings=settings)
    await engine.start()
    try:
        assert engine.tokenizer is backbone.tokenizer
        assert engine.backbone_model is backbone.model
        assert engine.get_stats()["ane_prefill"]["configured"] is enabled
        assert [e[0] for e in events] == (["load", "ane"] if enabled else ["load"])
        assert all(e[1] != threading.get_ident() for e in events)
    finally:
        await engine.stop()
    assert events[-1][0] == "close"


@pytest.mark.asyncio
async def test_partial_ane_failure_releases_banks_and_keeps_gpu_model(
    model_path, settings, monkeypatch
):
    released = []
    model = SimpleNamespace(
        load=lambda: None,
        close=lambda: None,
        backbone=SimpleNamespace(config={"model_type": "qwen3_5"}, model=object()),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(engine_module, "decision_kind", lambda path: "clef")
    monkeypatch.setitem(
        engine_module._MODEL_CLASSES, "clef", lambda *args, **kwargs: model
    )
    monkeypatch.setattr(ane_patch, "enable_qwen35_ane_prefill", fail)
    monkeypatch.setattr(ane_patch, "release_qwen35_ane_prefill", released.append)
    engine = DecisionEngine(str(model_path), model_settings=settings)
    await engine.start()
    try:
        assert engine.backbone_model is model.backbone.model
        assert released == [model.backbone.model]
    finally:
        await engine.stop()


def test_backbone_close_releases_programs_before_dropping_model(monkeypatch):
    backbone = DecisionBackbone("unused")
    backbone.model = object()
    released = []
    monkeypatch.setattr(ane_patch, "release_qwen35_ane_prefill", released.append)
    model = backbone.model
    backbone.close()
    assert released == [model]
    assert backbone.model is None


@pytest.mark.parametrize("ceiling, released", [(0, False), (200, True), (300, False)])
def test_driver_weights_count_toward_live_headroom(monkeypatch, ceiling, released):
    engine = DecisionEngine("unused", memory_ceiling=lambda: ceiling)
    model = object()
    engine._model = SimpleNamespace(backbone=SimpleNamespace(model=model))
    calls = []
    monkeypatch.setattr(ane_patch, "ane_prefill_transient_bytes", lambda model: 100)
    monkeypatch.setattr(ane_patch, "release_qwen35_ane_prefill", calls.append)
    monkeypatch.setattr(engine_module, "get_phys_footprint", lambda: 150)
    monkeypatch.setattr(engine_module.mx, "get_active_memory", lambda: 120)
    engine._check_ane_headroom()
    assert calls == ([model] if released else [])


def test_server_status_unwraps_decision_backbone():
    from omlx.server import _ane_prefill_status

    engine = DecisionEngine("unused")
    model = SimpleNamespace(_omlx_ane_mlp_prefill_count=4)
    engine._model = SimpleNamespace(backbone=SimpleNamespace(model=model))
    pool = SimpleNamespace(_entries={"clef": SimpleNamespace(engine=engine)})
    status = _ane_prefill_status(pool)
    assert status["configured_models"] == 1
    assert status["models"][0]["mlp_layers"] == 4


def test_clef_settings_round_trip_and_leave_qwen_record_independent(tmp_path):
    manager = ModelSettingsManager(tmp_path)
    manager.set_settings("qwen", ModelSettings(ane_prefill_fraction=0.53))
    settings = ModelSettings(ane_prefill_enabled=True, ane_prefill_cpu_enabled=False)
    manager.set_settings("clef-flash-oQ4", settings)
    reloaded = ModelSettingsManager(tmp_path)
    assert reloaded.get_settings("clef-flash-oQ4").ane_prefill_enabled is True
    assert reloaded.get_settings("clef-flash-oQ4").ane_prefill_cpu_enabled is False
    assert reloaded.get_settings("qwen").ane_prefill_fraction == 0.53
    assert reloaded.get_settings("qwen").ane_prefill_enabled is False


def test_neutral_names_reserve_memory_and_change_engine_reload_signature(model_path):
    from omlx.engine_pool import EngineEntry, EnginePool

    pool = EnginePool()
    entry = EngineEntry(
        model_id="clef",
        model_path=str(model_path),
        model_type="decision",
        engine_type="decision",
        config_model_type="qwen3_5",
        estimated_size=1000,
    )
    pool._entries["clef"] = entry
    settings = ModelSettings(
        ane_prefill_enabled=True,
        ane_prefill_sequence_length=1024,
        ane_prefill_fraction=0.25,
        ane_prefill_dual_ane=False,
        ane_prefill_gdn=False,
    )
    reservation = ane_memory_reservation(str(model_path), settings)
    assert reservation > 0
    assert pool._entry_runtime_resident_size(entry, settings) == 1000 + reservation
    signature = pool._engine_runtime_signature("clef", settings)
    settings.ane_prefill_enabled = False
    assert ane_memory_reservation(str(model_path), settings) == 0
    assert pool._engine_runtime_signature("clef", settings) != signature


def test_profiles_and_tuning_keep_neutral_settings_effective():
    from omlx.admin.ane_tuning import (
        ANETuningRequest,
        _Candidate,
        _settings_for_candidate,
    )
    from omlx.admin.recipe import FEATURE_GROUPS, drop_group, group_enabled
    from omlx.model_profiles import filter_profile_fields

    settings = ModelSettings(ane_prefill_enabled=True, ane_prefill_dual_ane=False)
    assert filter_profile_fields(settings.to_dict())["ane_prefill_enabled"] is True
    group = next(group for group in FEATURE_GROUPS if group.name == "ane_prefill")
    assert group_enabled(settings.to_dict(), group)
    assert not any(
        name.startswith("ane_prefill_") for name in drop_group(settings.to_dict(), group)
    )
    gpu = _settings_for_candidate(
        settings, ANETuningRequest(model_id="clef"), _Candidate("GPU", False)
    )
    assert gpu.ane_prefill_enabled is False
    assert settings.ane_prefill_enabled is True
