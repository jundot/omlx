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
            admin_routes.ModelSettingsRequest(
                qwen35_ane_prefill_sequence_length=2000
            ),
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
            admin_routes.ModelSettingsRequest(
                qwen35_ane_prefill_fused_down=True
            ),
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


@pytest.mark.asyncio
async def test_prefill_step_size_is_persisted_and_echoed():
    """#3381: the per-model prefill chunk override must reach ModelSettings."""
    pool, _ = _failed_pool()
    settings = ModelSettings()

    result = await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(prefill_step_size=1024),
    )

    assert settings.prefill_step_size == 1024
    assert result["settings"]["prefill_step_size"] == 1024


@pytest.mark.asyncio
async def test_prefill_step_size_null_clears_to_automatic():
    """null is the meaningful "automatic" value here, not a no-op (#3381)."""
    pool, _ = _failed_pool()
    settings = ModelSettings(prefill_step_size=1024)

    result = await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(prefill_step_size=None),
    )

    assert settings.prefill_step_size is None
    # to_dict() drops None, so the panel reads the cleared override back as
    # Automatic instead of a stale width.
    assert "prefill_step_size" not in result["settings"]


@pytest.mark.asyncio
async def test_prefill_step_size_untouched_when_not_sent():
    """`in sent` semantics: a PUT that never mentions the field keeps it."""
    pool, _ = _failed_pool()
    settings = ModelSettings(prefill_step_size=1024)

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(temperature=0.25),
    )

    assert settings.prefill_step_size == 1024


@pytest.mark.parametrize("value", [999999, 3000])
def test_prefill_step_size_rejects_unsupported_widths(value):
    """An unvalidated width persists and GET echoes it while the engine
    coerces a different one, so the panel would lie about the active config
    (#3381). FastAPI maps this to 422 at the HTTP boundary; _update_settings
    calls the handler directly, so it surfaces at request construction."""
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="256, 512, 1024 or 2048"):
        admin_routes.ModelSettingsRequest(prefill_step_size=value)


@pytest.mark.parametrize("value", [256, 512, 1024, 2048, None])
def test_prefill_step_size_accepts_the_full_select_grid(value):
    """Every width the panel offers, plus Automatic, must survive validation."""
    request = admin_routes.ModelSettingsRequest(prefill_step_size=value)

    assert request.prefill_step_size == value


def test_prefill_step_size_rejection_stays_json_serializable():
    """The rejection has to reach the client as a 422, not a 500 (#3381).

    omlx/server.py:780 hands `{"detail": exc.errors()}` straight to a
    JSONResponse on non-API routes, and Pydantic v2 puts the raw exception
    object into `ctx` for any validator that raises ValueError. json.dumps
    then fails and FastAPI turns the intended 422 into a 500 ("Object of type
    ValueError is not JSON serializable"). This locks the Literal in place
    against anyone "simplifying" it back to a field_validator.
    """
    import pydantic

    with pytest.raises(pydantic.ValidationError) as exc_info:
        admin_routes.ModelSettingsRequest(prefill_step_size=999999)

    errors = exc_info.value.errors()
    # Exactly what the non-API-route branch of the handler does.
    body = json.dumps({"detail": errors})

    assert errors[0]["type"] == "literal_error"
    assert "256, 512, 1024 or 2048" in body


def test_sanitize_diffusion_settings_dict_clears_prefill_step_size():
    """A profile import must not persist a width the diffusion lane ignores.

    The dashboard.js guard only covers the UI path, so this sanitizer is the
    only thing standing between a profile's stored override and settings.json
    (#3381).
    """
    settings = {"prefill_step_size": 512}

    admin_routes._sanitize_diffusion_settings_dict(settings)

    assert settings["prefill_step_size"] is None
    # Neighbouring resets, to prove the clear lands in the live block rather
    # than after an early return.
    assert settings["turboquant_kv_bits"] == 4
    assert settings["turboquant_skip_last"] is True


def test_sanitize_diffusion_model_settings_clears_prefill_step_size():
    """The direct-API hole: PUT {"prefill_step_size": 512} on a diffusion
    model must land back on automatic, since engine/vlm.py pins the width
    (#3381)."""
    settings = ModelSettings(prefill_step_size=512)

    admin_routes._sanitize_diffusion_model_settings(settings)

    assert settings.prefill_step_size is None
    assert settings.turboquant_kv_bits == 4
    assert settings.turboquant_skip_last is True
