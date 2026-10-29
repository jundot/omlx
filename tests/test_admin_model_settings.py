# SPDX-License-Identifier: Apache-2.0
"""Tests for load-failure invalidation in admin model settings."""

import asyncio
import copy
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic
import pytest
from fastapi import HTTPException

import omlx.server  # noqa: F401 - ensure server module is imported first
from omlx.admin import routes as admin_routes
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_settings import ModelSettings


def _failed_pool() -> tuple[EnginePool, EngineEntry]:
    pool = EnginePool()
    entry = EngineEntry(
        model_id="ling",
        model_path="/tmp/ling",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
        load_failed=True,
        load_failure_message="trust_remote_code=True required",
        load_failure_at=123.0,
    )
    pool._entries[entry.model_id] = entry
    return pool, entry


def _write_qwen4_mtp_checkpoint(tmp_path, *, embedded_mtp: bool) -> None:
    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "num_hidden_layers": 48,
            "mtp_num_hidden_layers": 1,
            "num_nextn_predict_layers": 1,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    weight_key = (
        "mtp.fc_hidden.weight"
        if embedded_mtp
        else "model.layers.48.self_attn.q_proj.weight"
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight_key: "model.safetensors"}})
    )


def _write_minimax_m3_checkpoint(tmp_path, *, dspark: bool) -> None:
    config = {"model_type": "minimax_m3_vl", "text_config": {"num_hidden_layers": 60}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    if dspark:
        draft = tmp_path / "dspark"
        draft.mkdir()
        (draft / "config.json").write_text(
            json.dumps({"architectures": ["DSparkMiniMaxDraftModel"]})
        )
        (draft / "model-dspark-00001.safetensors").write_bytes(b"")


async def _update_settings(
    pool: EnginePool,
    settings: ModelSettings,
    request: admin_routes.ModelSettingsRequest,
) -> dict:
    manager = MagicMock()
    manager.get_settings.return_value = copy.deepcopy(settings)
    manager.set_settings.side_effect = lambda _, updated: settings.__dict__.update(
        updated.__dict__
    )
    state = MagicMock()

    with (
        patch("omlx.admin.routes._get_engine_pool", return_value=pool),
        patch("omlx.admin.routes._get_settings_manager", return_value=manager),
        patch("omlx.admin.routes._get_server_state", return_value=state),
    ):
        result = await admin_routes.update_model_settings(
            "ling", request, is_admin=True
        )

    manager.set_settings.assert_called_once_with("ling", settings)
    return result


@pytest.mark.asyncio
async def test_load_time_setting_change_clears_cached_failure():
    pool, entry = _failed_pool()
    settings = ModelSettings(trust_remote_code=False)

    result = await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(trust_remote_code=True),
    )

    assert settings.trust_remote_code is True
    assert entry.load_failed is False
    assert entry.load_failure_message is None
    assert entry.load_failure_at is None
    assert result["requires_reload"] is False


@pytest.mark.asyncio
async def test_embedding_audio_setting_is_load_time():
    pool, entry = _failed_pool()
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(embedding_audio_enabled=True),
    )

    assert settings.embedding_audio_enabled is True
    assert entry.load_failed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_enabled", [True, False])
async def test_embedding_audio_length_is_load_time_only_with_audio(audio_enabled):
    pool, entry = _failed_pool()
    settings = ModelSettings(embedding_audio_enabled=audio_enabled)

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(embedding_audio_max_seconds=120),
    )

    assert settings.embedding_audio_max_seconds == 120
    assert entry.load_failed is not audio_enabled


@pytest.mark.asyncio
async def test_resent_embedding_audio_settings_keep_loaded_engine():
    pool, entry = _failed_pool()
    entry.engine = _idle_engine()
    entry.load_failed = False
    pool._unload_engine = AsyncMock()

    result = await _update_settings(
        pool,
        ModelSettings(embedding_audio_enabled=True, embedding_audio_max_seconds=120.0),
        admin_routes.ModelSettingsRequest(
            embedding_audio_enabled=True, embedding_audio_max_seconds=120.0
        ),
    )

    assert result["requires_reload"] is False
    pool._unload_engine.assert_not_awaited()


def test_embedding_audio_length_must_be_positive():
    with pytest.raises(ValueError):
        admin_routes.ModelSettingsRequest(embedding_audio_max_seconds=0)


@pytest.mark.asyncio
async def test_unchanged_load_time_setting_keeps_cached_failure():
    pool, entry = _failed_pool()
    settings = ModelSettings(trust_remote_code=False)

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(trust_remote_code=False),
    )

    assert entry.load_failed is True
    assert entry.load_failure_message == "trust_remote_code=True required"
    assert entry.load_failure_at == 123.0


@pytest.mark.asyncio
async def test_sampling_setting_change_keeps_cached_failure():
    pool, entry = _failed_pool()
    settings = ModelSettings(trust_remote_code=False)

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(temperature=0.25),
    )

    assert settings.temperature == 0.25
    assert entry.load_failed is True
    assert entry.load_failure_message == "trust_remote_code=True required"
    assert entry.load_failure_at == 123.0


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_kv_format_change_reloads_only_when_compression_is_enabled(enabled):
    pool, entry = _failed_pool()
    entry.engine = _idle_engine()
    entry.load_failed = False
    pool._unload_engine = AsyncMock()

    result = await _update_settings(
        pool,
        ModelSettings(turboquant_kv_enabled=enabled),
        admin_routes.ModelSettingsRequest(turboquant_kv_scheme="affine4"),
    )

    assert result["settings"]["turboquant_kv_scheme"] == "affine4"
    assert result["requires_reload"] is enabled
    assert result["auto_unloaded"] is enabled
    assert result["reload_deferred"] is False
    assert pool._unload_engine.await_count == int(enabled)


@pytest.mark.asyncio
async def test_legacy_turboquant_bits_only_block_kv_changes():
    pool, _ = _failed_pool()
    legacy = ModelSettings(turboquant_kv_bits=5)

    result = await _update_settings(
        pool, legacy, admin_routes.ModelSettingsRequest(temperature=0.5)
    )
    assert result["settings"]["temperature"] == 0.5

    manager = MagicMock()
    manager.get_settings.return_value = ModelSettings(turboquant_kv_bits=5)
    with (
        patch("omlx.admin.routes._get_engine_pool", return_value=pool),
        patch("omlx.admin.routes._get_settings_manager", return_value=manager),
        patch("omlx.admin.routes._get_server_state", return_value=MagicMock()),
        pytest.raises(HTTPException) as exc,
    ):
        await admin_routes.update_model_settings(
            "ling",
            admin_routes.ModelSettingsRequest(turboquant_kv_scheme="affine8"),
            is_admin=True,
        )
    assert exc.value.status_code == 400
    manager.set_settings.assert_not_called()


@pytest.mark.asyncio
async def test_qwen_ane_prefill_settings_are_persisted():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen3_5"
    settings = ModelSettings()

    result = await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(
            qwen35_ane_prefill_enabled=True,
            qwen35_ane_prefill_sequence_length=2048,
            qwen35_ane_prefill_tail_padding_min_tokens=1357,
            qwen35_ane_prefill_fraction=0.53,
            qwen35_ane_prefill_max_layers=64,
            qwen35_ane_prefill_dual_ane=True,
            qwen35_ane_prefill_gdn=True,
            qwen35_ane_prefill_gdn_fraction=0.50,
            qwen35_ane_prefill_gdn_max_layers=48,
        ),
    )

    assert settings.qwen35_ane_prefill_enabled is True
    assert settings.qwen35_ane_prefill_sequence_length == 2048
    assert settings.qwen35_ane_prefill_tail_padding_min_tokens == 1357
    assert settings.qwen35_ane_prefill_fraction == 0.53
    assert settings.qwen35_ane_prefill_max_layers == 64
    assert settings.qwen35_ane_prefill_dual_ane is True
    assert settings.qwen35_ane_prefill_gdn is True
    assert settings.qwen35_ane_prefill_gdn_fraction == 0.50
    assert settings.qwen35_ane_prefill_gdn_max_layers == 48
    assert result["requires_reload"] is False


def _idle_engine() -> MagicMock:
    engine = MagicMock()
    engine.has_active_requests.return_value = False
    engine.scheduler = None
    engine._engine = None
    return engine


@pytest.mark.asyncio
async def test_gpu_power_limit_saves_without_reload():
    pool, entry = _failed_pool()
    entry.engine = _idle_engine()
    entry.load_failed = False
    pool._unload_engine = AsyncMock()
    settings = ModelSettings()

    def request(watts):
        return admin_routes.ModelSettingsRequest(gpu_power_limit_w=watts)

    result = await _update_settings(pool, settings, request(10))
    assert settings.gpu_power_limit_w == 10
    assert result["requires_reload"] is False
    pool._unload_engine.assert_not_awaited()

    await _update_settings(pool, settings, request(0))
    assert settings.gpu_power_limit_w is None

    # Profiles and templates go through the same request model.
    with pytest.raises(pydantic.ValidationError):
        request(1)
    assert admin_routes._normalize_profile_settings({"gpu_power_limit_w": 0}) == {}


@pytest.mark.asyncio
async def test_qwen_ane_prefill_change_unloads_a_loaded_engine():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen3_5"
    entry.engine = _idle_engine()
    entry.load_failed = False
    pool._unload_engine = AsyncMock()

    result = await _update_settings(
        pool,
        ModelSettings(),
        admin_routes.ModelSettingsRequest(qwen35_ane_prefill_enabled=True),
    )

    assert result["requires_reload"] is True
    assert result["auto_unloaded"] is True
    assert result["reload_deferred"] is False
    pool._unload_engine.assert_awaited_once_with("ling")


@pytest.mark.asyncio
async def test_reload_setting_on_busy_engine_defers_unload():
    """A save during a benchmark run must not abort it (#3961)."""
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen3_5"
    entry.engine = _idle_engine()
    entry.engine.abort_all_requests = AsyncMock()
    entry.load_failed = False
    entry.in_use = 1
    pool._unload_engine = AsyncMock()

    result = await _update_settings(
        pool,
        ModelSettings(),
        admin_routes.ModelSettingsRequest(qwen35_ane_prefill_enabled=True),
    )

    assert result["requires_reload"] is True
    assert result["auto_unloaded"] is False
    assert result["reload_deferred"] is True
    assert entry.pending_unload_reason == "settings changed"
    assert entry.abort_requested is False
    entry.engine.abort_all_requests.assert_not_awaited()
    pool._unload_engine.assert_not_awaited()

    pending = pool._pending_unload_tasks["ling"]
    await pool.release_engine("ling")
    pool._unload_engine.assert_awaited_once_with("ling")
    await asyncio.wait_for(pending, timeout=1)


@pytest.mark.asyncio
async def test_qwen_ane_prefill_accepts_qwen38_config_type():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen3_8"
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(qwen35_ane_prefill_enabled=True),
    )

    assert settings.qwen35_ane_prefill_enabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial, update",
    [
        ({"qwen35_ane_prefill_enabled": True}, {"a8_prefill_enabled": True}),
        ({"a8_prefill_enabled": True}, {"qwen35_ane_prefill_enabled": True}),
        ({}, {"a8_prefill_enabled": True, "qwen35_ane_prefill_enabled": True}),
    ],
)
async def test_oq_a8_is_refused_while_ane_prefill_is_on(tmp_path, initial, update):
    from omlx.model_settings import ModelSettingsManager

    pool, entry = _failed_pool()
    entry.config_model_type = "qwen3_5"
    manager = ModelSettingsManager(tmp_path)
    manager.set_settings("ling", ModelSettings(**initial))
    before = manager.get_settings("ling").to_dict()
    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(admin_routes, "_get_settings_manager", return_value=manager),
        patch.object(admin_routes, "_get_server_state", return_value=MagicMock()),
        patch.object(admin_routes, "_oq_a8_kernels_available", return_value=True),
        pytest.raises(admin_routes.HTTPException) as excinfo,
    ):
        await admin_routes.update_model_settings(
            "ling", admin_routes.ModelSettingsRequest(**update), is_admin=True
        )
    assert excinfo.value.status_code == 400
    assert "cannot both be enabled" in excinfo.value.detail
    assert manager.get_settings("ling").to_dict() == before
    assert ModelSettingsManager(tmp_path).get_settings("ling").to_dict() == before


@pytest.mark.asyncio
async def test_oq_a8_alone_is_persisted():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen3_5"
    settings = ModelSettings()

    with patch.object(admin_routes, "_oq_a8_kernels_available", return_value=True):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(
                a8_prefill_enabled=True, a8_prefill_min_tokens=256
            ),
        )

    assert settings.a8_prefill_enabled is True
    assert settings.a8_prefill_min_tokens == 256


@pytest.mark.asyncio
async def test_oq_a8_is_accepted_for_qwen4_exp_and_keeps_min_tokens():
    """Qwen3.8-Flash-Next shares the existing A8 setting (routed experts)."""
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen4_exp"
    settings = ModelSettings()

    with patch.object(admin_routes, "_oq_a8_kernels_available", return_value=True):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(
                a8_prefill_enabled=True, a8_prefill_min_tokens=512
            ),
        )

    assert settings.a8_prefill_enabled is True
    assert settings.a8_prefill_min_tokens == 512


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config_type", ["qwen4", "qwen4_exp_x", "qwen3", "qwen2", "llama", "gemma4", ""]
)
async def test_oq_a8_is_refused_outside_validated_families(config_type):
    """Only the exact ``qwen4_exp`` family is added, not every ``qwen4*``."""
    pool, entry = _failed_pool()
    entry.config_model_type = config_type
    settings = ModelSettings()

    with patch.object(admin_routes, "_oq_a8_kernels_available", return_value=True):
        with pytest.raises(admin_routes.HTTPException) as excinfo:
            await _update_settings(
                pool,
                settings,
                admin_routes.ModelSettingsRequest(a8_prefill_enabled=True),
            )

    assert excinfo.value.status_code == 400
    assert "Flash-Next" in excinfo.value.detail
    assert settings.a8_prefill_enabled is False


@pytest.mark.asyncio
async def test_oq_a8_for_qwen4_exp_still_conflicts_with_ane_prefill(tmp_path):
    from omlx.model_settings import ModelSettingsManager

    pool, entry = _failed_pool()
    entry.config_model_type = "qwen4_exp"
    manager = ModelSettingsManager(tmp_path)
    manager.set_settings("ling", ModelSettings())
    before = manager.get_settings("ling").to_dict()
    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(admin_routes, "_get_settings_manager", return_value=manager),
        patch.object(admin_routes, "_get_server_state", return_value=MagicMock()),
        patch.object(admin_routes, "_oq_a8_kernels_available", return_value=True),
        pytest.raises(admin_routes.HTTPException) as excinfo,
    ):
        await admin_routes.update_model_settings(
            "ling",
            admin_routes.ModelSettingsRequest(
                a8_prefill_enabled=True, qwen35_ane_prefill_enabled=True
            ),
            is_admin=True,
        )
    assert excinfo.value.status_code == 400
    assert manager.get_settings("ling").to_dict() == before


@pytest.mark.parametrize(
    "config_type, expected",
    [
        ("qwen3_5", True),
        ("qwen3_5_moe", True),
        ("Qwen3-6", True),
        ("qwen3_8", True),
        ("qwen4_exp", True),
        ("Qwen4-Exp", True),
        ("qwen4", False),
        ("qwen4_exp_x", False),
        ("deepseek_v41", True),
        ("deepseek_v4", False),
        ("llama", False),
        (None, False),
    ],
)
def test_oq_a8_model_supported(config_type, expected):
    assert admin_routes._oq_a8_model_supported(config_type) is expected


@pytest.mark.parametrize(
    "config_type,kernels,expected",
    [("qwen3_5", True, True), ("qwen3_5", False, False), ("llama", True, False)],
)
def test_model_options_report_oq_a8_support(config_type, kernels, expected):
    with patch.object(admin_routes, "_oq_a8_kernels_available", return_value=kernels):
        options = admin_routes._model_options({"config_model_type": config_type}, None)
    assert options["a8_prefill_supported"] is expected


@pytest.mark.asyncio
async def test_oq_a8_needs_native_int8_kernels():
    """Nothing on this hardware would run faster, so the setting is refused
    rather than accepted and silently ignored at load."""
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen3_5"
    settings = ModelSettings()

    with patch.object(admin_routes, "_oq_a8_kernels_available", return_value=False):
        with pytest.raises(admin_routes.HTTPException) as excinfo:
            await _update_settings(
                pool,
                settings,
                admin_routes.ModelSettingsRequest(a8_prefill_enabled=True),
            )

    assert excinfo.value.status_code == 400
    assert "M5-series or newer" in excinfo.value.detail
    assert settings.a8_prefill_enabled is False


@pytest.mark.asyncio
async def test_qwen4_ple_ssd_offload_is_persisted_for_qwen4_only():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen4_exp"
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(qwen4_ple_ssd_offload=True),
    )

    assert settings.qwen4_ple_ssd_offload is True


@pytest.mark.asyncio
async def test_qwen4_ple_ssd_offload_is_ignored_for_other_models():
    pool, _ = _failed_pool()
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(qwen4_ple_ssd_offload=True),
    )

    assert settings.qwen4_ple_ssd_offload is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_type", "expected"), [("qwen4_exp", True), ("qwen3_5_moe", False)]
)
async def test_qwen4_wide_decode_is_persisted_for_qwen4_only(model_type, expected):
    pool, entry = _failed_pool()
    entry.config_model_type = model_type
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(qwen4_gdn_decode_wide_proj=True),
    )

    assert settings.qwen4_gdn_decode_wide_proj is expected


@pytest.mark.asyncio
async def test_deepseek_v41_engram_ssd_offload_is_persisted_for_v41_only():
    pool, entry = _failed_pool()
    entry.config_model_type = "deepseek_v41"
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(deepseek_v41_engram_ssd_offload=True),
    )

    assert settings.deepseek_v41_engram_ssd_offload is True


@pytest.mark.asyncio
async def test_deepseek_v41_engram_ssd_offload_is_ignored_for_other_models():
    pool, _ = _failed_pool()
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(deepseek_v41_engram_ssd_offload=True),
    )

    assert settings.deepseek_v41_engram_ssd_offload is False


@pytest.mark.asyncio
async def test_qwen4_mtp_setting_accepts_embedded_head(tmp_path):
    _write_qwen4_mtp_checkpoint(tmp_path, embedded_mtp=True)
    pool, entry = _failed_pool()
    entry.model_path = str(tmp_path)
    entry.config_model_type = "qwen4_exp"
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(mtp_enabled=True),
    )

    assert settings.mtp_enabled is True


@pytest.mark.asyncio
async def test_qwen4_mtp_setting_rejects_nextn_only_layout(tmp_path):
    _write_qwen4_mtp_checkpoint(tmp_path, embedded_mtp=False)
    pool, entry = _failed_pool()
    entry.model_path = str(tmp_path)
    entry.config_model_type = "qwen4_exp"
    settings = ModelSettings()

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(mtp_enabled=True),
        )

    assert exc_info.value.status_code == 400
    assert "native nextn layers are not supported" in exc_info.value.detail
    assert settings.mtp_enabled is False


@pytest.mark.asyncio
async def test_minimax_m3_mtp_setting_accepts_bundled_dspark(tmp_path):
    _write_minimax_m3_checkpoint(tmp_path, dspark=True)
    pool, entry = _failed_pool()
    entry.model_path = str(tmp_path)
    entry.config_model_type = "minimax_m3_vl"
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(mtp_enabled=True),
    )

    assert settings.mtp_enabled is True


@pytest.mark.asyncio
async def test_minimax_m3_mtp_setting_rejects_missing_dspark(tmp_path):
    _write_minimax_m3_checkpoint(tmp_path, dspark=False)
    pool, entry = _failed_pool()
    entry.model_path = str(tmp_path)
    entry.config_model_type = "minimax_m3_vl"
    settings = ModelSettings()

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(mtp_enabled=True),
        )

    assert exc_info.value.status_code == 400
    assert "no bundled dspark/ draft" in exc_info.value.detail
    assert settings.mtp_enabled is False


@pytest.mark.asyncio
async def test_qwen_ane_prefill_rejects_invalid_block_size():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen3_5"

    with pytest.raises(admin_routes.HTTPException, match="multiple of 64"):
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(qwen35_ane_prefill_sequence_length=2000),
        )


@pytest.mark.asyncio
async def test_qwen_ane_prefill_rejects_tail_threshold_at_block_size():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen3_5"

    with pytest.raises(admin_routes.HTTPException, match="less than"):
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(
                qwen35_ane_prefill_tail_padding_min_tokens=2048
            ),
        )


@pytest.mark.asyncio
async def test_qwen_ane_prefill_rejects_fused_down_above_half_fraction():
    """Fused reuses the MLP fraction for down; above 0.50 the loader raises
    and ANE prefill silently disables, so the save must be rejected."""
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen3_5"
    settings = ModelSettings()
    settings.qwen35_ane_prefill_fraction = 0.53

    with pytest.raises(admin_routes.HTTPException, match="0.50 or"):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(qwen35_ane_prefill_fused_down=True),
        )


@pytest.mark.asyncio
async def test_qwen_ane_prefill_allows_fused_down_at_half_fraction():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen3_5"
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(
            qwen35_ane_prefill_fused_down=True,
            qwen35_ane_prefill_fraction=0.5,
        ),
    )

    assert settings.qwen35_ane_prefill_fused_down is True
    assert settings.qwen35_ane_prefill_fraction == 0.5


@pytest.mark.asyncio
async def test_qwen_ane_prefill_rejects_other_model_families():
    pool, entry = _failed_pool()
    entry.config_model_type = "gemma4"

    with pytest.raises(admin_routes.HTTPException, match="ANE prefill is unavailable"):
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(qwen35_ane_prefill_enabled=True),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [3, 4, 5, 6, 8])
async def test_mtp_draft_tokens_is_persisted_not_dropped(depth):
    """#2823: mtp_adaptive_max_depth used to be silently discarded by PUT."""
    pool, _ = _failed_pool()
    settings = ModelSettings(mtp_adaptive_max_depth=None, mtp_fixed_depth=2)

    result = await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(
            mtp_adaptive_max_depth=depth, mtp_fixed_depth=None
        ),
    )

    assert settings.mtp_adaptive_max_depth == depth
    assert result["settings"]["mtp_adaptive_max_depth"] == depth

    assert settings.mtp_fixed_depth is None


@pytest.mark.asyncio
async def test_preserve_thinking_and_turboquant_skip_last_are_persisted():
    """Same silent-drop class as #2823 for the other two engine settings."""
    pool, _ = _failed_pool()
    settings = ModelSettings(
        preserve_thinking=False,
        turboquant_skip_last=True,
    )

    result = await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(
            preserve_thinking=True,
            turboquant_skip_last=False,
        ),
    )

    assert settings.preserve_thinking is True
    assert settings.turboquant_skip_last is False
    assert result["settings"]["preserve_thinking"] is True
    assert result["settings"]["turboquant_skip_last"] is False


@pytest.mark.asyncio
async def test_mtp_fixed_depth_is_persisted_and_cleared():
    pool, _ = _failed_pool()
    settings = ModelSettings()

    result = await _update_settings(
        pool, settings, admin_routes.ModelSettingsRequest(mtp_fixed_depth=4)
    )
    assert settings.mtp_fixed_depth == 4
    assert result["settings"]["mtp_fixed_depth"] == 4

    await _update_settings(
        pool, settings, admin_routes.ModelSettingsRequest(mtp_fixed_depth=None)
    )
    assert settings.mtp_fixed_depth is None


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["mtp_adaptive_max_depth", "mtp_fixed_depth"])
@pytest.mark.parametrize("value", [0, 9])
async def test_mtp_depth_rejects_out_of_range_values(field, value):
    pool, _ = _failed_pool()

    with pytest.raises(admin_routes.HTTPException, match="must be between 1 and 8"):
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(**{field: value}),
        )


@pytest.mark.parametrize("value", [0, -5])
def test_output_token_caps_below_one_are_rejected(value):
    # The global route applies the runtime copy before validate().
    with pytest.raises(pydantic.ValidationError, match="max_tokens"):
        admin_routes.ModelSettingsRequest(max_tokens=value)
    with pytest.raises(pydantic.ValidationError, match="sampling_max_tokens"):
        admin_routes.GlobalSettingsRequest(sampling_max_tokens=value)


def test_unknown_settings_fields_are_rejected_loudly():
    """Unknown keys must 422 instead of silently returning success:true."""
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="bogus_field"):
        # Simulate a client sending a field that has no admin-PUT support.
        admin_routes.ModelSettingsRequest(mtp_adaptive_max_depth=8, bogus_field=1)


@pytest.mark.asyncio
async def test_turboquant_skip_last_null_preserves_default_true():
    """null = clear to the model default; it must not flip the default to
    False via bool(None) (review feedback on the silent-drop fix)."""
    pool, _ = _failed_pool()
    settings = ModelSettings()  # default turboquant_skip_last=True

    result = await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(turboquant_skip_last=None),
    )

    assert settings.turboquant_skip_last is True
    assert result["settings"]["turboquant_skip_last"] is True


def test_runtime_signature_gates_mtp_depth_on_lightning_mtp():
    """mtp_adaptive_max_depth must be part of the engine runtime signature only
    while Lightning MTP (mtp_enabled) is active (review feedback), so a depth
    change reloads a loaded engine, but a stale value never forces one."""
    from omlx.engine_pool import EnginePool

    pool = EnginePool()

    depth_3_on = ModelSettings(mtp_enabled=True, mtp_adaptive_max_depth=3)
    depth_8_on = ModelSettings(mtp_enabled=True, mtp_adaptive_max_depth=8)
    depth_3_off = ModelSettings(mtp_enabled=False, mtp_adaptive_max_depth=3)
    depth_8_off = ModelSettings(mtp_enabled=False, mtp_adaptive_max_depth=8)

    on_keys = {k for k, _ in pool._engine_runtime_signature("m", depth_3_on)}
    assert "mtp_adaptive_max_depth" in on_keys
    off_keys = {k for k, _ in pool._engine_runtime_signature("m", depth_3_off)}
    assert "mtp_adaptive_max_depth" not in off_keys

    # Active MTP: different depths produce different signatures (reload).
    assert pool._engine_runtime_signature(
        "m", depth_3_on
    ) != pool._engine_runtime_signature("m", depth_8_on)
    # Inactive MTP: the value is invisible to the signature (no reload).
    assert pool._engine_runtime_signature(
        "m", depth_3_off
    ) == pool._engine_runtime_signature("m", depth_8_off)


def test_runtime_signature_reloads_qwen35_family_for_a_yarn_window():
    pool = EnginePool()
    pool._entries["q"] = EngineEntry(
        model_id="q",
        model_path="/tmp/q",
        model_type="vlm",
        engine_type="vlm",
        estimated_size=1,
        config_model_type="qwen4_exp",
        model_context_length=262144,
    )

    def signature(window):
        return dict(
            pool._engine_runtime_signature(
                "q", ModelSettings(max_context_window=window)
            )
        )

    assert "yarn_factor" not in signature(None)
    assert "yarn_factor" not in signature(262144)
    assert signature(524288)["yarn_factor"] != signature(1048576)["yarn_factor"]

    pool._entries["q"].config_model_type = "llama"
    assert "yarn_factor" not in signature(1048576)


@pytest.mark.asyncio
async def test_parked_dflash_draft_path_does_not_block_saves(tmp_path):
    pool, _ = _failed_pool()
    deleted = str(tmp_path / "deleted-draft")
    settings = ModelSettings(dflash_enabled=True, dflash_draft_model=deleted)

    await _update_settings(
        pool, settings, admin_routes.ModelSettingsRequest(temperature=0.3)
    )
    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(
            dflash_enabled=False, dflash_draft_model=deleted
        ),
    )

    assert settings.dflash_enabled is False
    assert settings.dflash_draft_model == deleted


@pytest.mark.asyncio
async def test_enabling_dflash_over_a_deleted_draft_path_is_rejected(tmp_path):
    pool, _ = _failed_pool()
    deleted = str(tmp_path / "deleted-draft")
    manager = MagicMock()
    manager.get_settings.side_effect = lambda _: ModelSettings(
        dflash_draft_model=deleted
    )

    with (
        patch("omlx.admin.routes._get_engine_pool", return_value=pool),
        patch("omlx.admin.routes._get_settings_manager", return_value=manager),
        patch("omlx.admin.routes._get_server_state", return_value=MagicMock()),
        patch.object(admin_routes, "dflash_compatibility", return_value=(True, "")),
    ):
        for payload in (
            {"dflash_enabled": True},
            {"dflash_enabled": True, "dflash_draft_model": deleted},
        ):
            request = admin_routes.ModelSettingsRequest(**payload)
            with pytest.raises(HTTPException) as exc:
                await admin_routes.update_model_settings("ling", request, is_admin=True)
            assert exc.value.status_code == 422

    manager.set_settings.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type,accepted", [("qwen3_5", True), ("gemma4", False)])
async def test_paroquant_dflash_allows_only_qwen_targets(
    tmp_path, model_type, accepted
):
    pool, entry = _failed_pool()
    entry.model_path = str(tmp_path)
    entry.config_model_type = model_type
    entry.engine_type = "vlm"
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "paroquant"}})
    )
    settings = ModelSettings()
    request = admin_routes.ModelSettingsRequest(dflash_enabled=True)

    if accepted:
        await _update_settings(pool, settings, request)
        assert settings.dflash_enabled is True
    else:
        with pytest.raises(HTTPException) as exc:
            await _update_settings(pool, settings, request)
        assert exc.value.status_code == 400
        assert exc.value.detail == admin_routes._PAROQUANT_REASON
