# SPDX-License-Identifier: Apache-2.0
"""Tests for load-failure invalidation in admin model settings."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


async def _update_settings(
    pool: EnginePool,
    settings: ModelSettings,
    request: admin_routes.ModelSettingsRequest,
) -> dict:
    manager = MagicMock()
    manager.get_settings.return_value = settings
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
async def test_expert_streaming_cache_policy_is_persisted():
    pool, _ = _failed_pool()
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(
            expert_streaming_cache_policy="lru"
        ),
    )

    assert settings.expert_streaming_cache_policy == "lru"


@pytest.mark.asyncio
async def test_expert_streaming_scratch_experts_is_persisted():
    pool, _ = _failed_pool()
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(expert_streaming_scratch_experts=48),
    )

    assert settings.expert_streaming_scratch_experts == 48


@pytest.mark.asyncio
async def test_expert_streaming_fast_resource_loading_is_persisted():
    pool, _ = _failed_pool()
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(
            expert_streaming_fast_resource_loading=True
        ),
    )

    assert settings.expert_streaming_fast_resource_loading is True


@pytest.mark.asyncio
async def test_invalid_expert_streaming_patch_does_not_mutate_settings():
    pool, _ = _failed_pool()
    settings = ModelSettings(
        expert_streaming_enabled=True,
        expert_streaming_mode="cache_only",
    )

    with pytest.raises(admin_routes.HTTPException, match="Fast Resource Loading"):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(
                expert_streaming_fast_resource_loading=False,
            ),
        )

    assert settings.expert_streaming_fast_resource_loading is True
    assert settings.expert_streaming_direct_io is True
    assert settings.expert_streaming_native_demand is True


@pytest.mark.asyncio
async def test_cache_only_streaming_does_not_require_manifest():
    pool, _ = _failed_pool()
    settings = ModelSettings()

    with patch(
        "omlx.expert_streaming.residency.estimate_for_model_settings"
    ) as estimate:
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(
                expert_streaming_enabled=True,
                expert_streaming_mode="cache_only",
                expert_streaming_manifest=None,
            ),
        )

    assert settings.expert_streaming_enabled is True
    assert settings.expert_streaming_mode == "cache_only"
    assert settings.expert_streaming_manifest is None
    estimate.assert_called_once()


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


@pytest.mark.asyncio
async def test_qwen_ane_prefill_change_unloads_a_loaded_engine():
    pool, entry = _failed_pool()
    entry.config_model_type = "qwen3_5"
    entry.engine = MagicMock()
    entry.load_failed = False
    pool._unload_engine = AsyncMock()

    result = await _update_settings(
        pool,
        ModelSettings(),
        admin_routes.ModelSettingsRequest(qwen35_ane_prefill_enabled=True),
    )

    assert result["requires_reload"] is True
    assert result["auto_unloaded"] is True
    pool._unload_engine.assert_awaited_once_with("ling")


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

    with pytest.raises(admin_routes.HTTPException, match="Qwen3.5/3.6/3.8"):
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(qwen35_ane_prefill_enabled=True),
        )


@pytest.mark.asyncio
async def test_mtp_draft_tokens_is_persisted_not_dropped():
    """#2823: mtp_num_draft_tokens used to be silently discarded by PUT."""
    pool, _ = _failed_pool()
    settings = ModelSettings(mtp_num_draft_tokens=None)

    result = await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(mtp_num_draft_tokens=8),
    )

    assert settings.mtp_num_draft_tokens == 8
    assert result["settings"]["mtp_num_draft_tokens"] == 8


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
@pytest.mark.parametrize("value", [0, 9])
async def test_mtp_draft_tokens_rejects_out_of_range_values(value):
    pool, _ = _failed_pool()

    with pytest.raises(admin_routes.HTTPException, match="must be between 1 and 8"):
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(mtp_num_draft_tokens=value),
        )


def test_unknown_settings_fields_are_rejected_loudly():
    """Unknown keys must 422 instead of silently returning success:true."""
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="bogus_field"):
        # Simulate a client sending a field that has no admin-PUT support.
        admin_routes.ModelSettingsRequest(mtp_num_draft_tokens=8, bogus_field=1)


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
    """mtp_num_draft_tokens must be part of the engine runtime signature only
    while Lightning MTP (mtp_enabled) is active (review feedback), so a depth
    change reloads a loaded engine, but a stale value never forces one."""
    from omlx.engine_pool import EnginePool

    pool = EnginePool()

    depth_3_on = ModelSettings(mtp_enabled=True, mtp_num_draft_tokens=3)
    depth_8_on = ModelSettings(mtp_enabled=True, mtp_num_draft_tokens=8)
    depth_3_off = ModelSettings(mtp_enabled=False, mtp_num_draft_tokens=3)
    depth_8_off = ModelSettings(mtp_enabled=False, mtp_num_draft_tokens=8)

    on_keys = {k for k, _ in pool._engine_runtime_signature("m", depth_3_on)}
    assert "mtp_num_draft_tokens" in on_keys
    off_keys = {k for k, _ in pool._engine_runtime_signature("m", depth_3_off)}
    assert "mtp_num_draft_tokens" not in off_keys

    # Active MTP: different depths produce different signatures (reload).
    assert pool._engine_runtime_signature("m", depth_3_on) != pool._engine_runtime_signature(
        "m", depth_8_on
    )
    # Inactive MTP: the value is invisible to the signature (no reload).
    assert pool._engine_runtime_signature("m", depth_3_off) == pool._engine_runtime_signature(
        "m", depth_8_off
    )
