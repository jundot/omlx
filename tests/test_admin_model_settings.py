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


_FAKE_REASONING_PARSER_REGISTRY = {"harmony": ["gpt-oss"], "llama": ["Llama-3"]}


@pytest.mark.asyncio
async def test_update_model_settings_rejects_unknown_reasoning_parser():
    pool, entry = _failed_pool()

    with (
        patch(
            "omlx.admin.routes._reasoning_parser_registry",
            return_value=_FAKE_REASONING_PARSER_REGISTRY,
        ),
        pytest.raises(admin_routes.HTTPException) as exc_info,
    ):
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(reasoning_parser="qwen"),
        )

    assert exc_info.value.status_code == 400
    assert "harmony" in exc_info.value.detail
    assert "llama" in exc_info.value.detail


@pytest.mark.asyncio
async def test_update_model_settings_accepts_valid_reasoning_parser():
    pool, entry = _failed_pool()
    settings = ModelSettings()

    with patch(
        "omlx.admin.routes._reasoning_parser_registry",
        return_value=_FAKE_REASONING_PARSER_REGISTRY,
    ):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(reasoning_parser="harmony"),
        )

    assert settings.reasoning_parser == "harmony"


@pytest.mark.asyncio
async def test_update_model_settings_clearing_reasoning_parser_skips_validation():
    pool, entry = _failed_pool()
    settings = ModelSettings(reasoning_parser="harmony")

    with patch(
        "omlx.admin.routes._reasoning_parser_registry",
        return_value=_FAKE_REASONING_PARSER_REGISTRY,
    ):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(reasoning_parser=""),
        )

    assert settings.reasoning_parser is None


@pytest.mark.asyncio
async def test_update_model_settings_skips_validation_when_registry_unavailable():
    """xgrammar unavailable: fail open like the dropdown does (returns [])
    rather than blocking every settings save (#4)."""
    pool, entry = _failed_pool()
    settings = ModelSettings()

    with patch("omlx.admin.routes._reasoning_parser_registry", return_value=None):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(reasoning_parser="qwen"),
        )

    assert settings.reasoning_parser == "qwen"


@pytest.mark.asyncio
async def test_update_model_settings_logs_when_skipping_validation(caplog):
    """The fail-open path (registry unavailable) must leave a trace naming
    the model and the unvalidated value, not just the generic warning
    already logged inside _reasoning_parser_registry() itself."""
    pool, entry = _failed_pool()
    settings = ModelSettings()

    with (
        patch("omlx.admin.routes._reasoning_parser_registry", return_value=None),
        caplog.at_level("WARNING"),
    ):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(reasoning_parser="qwen"),
        )

    assert any(
        "ling" in record.message and "qwen" in record.message
        for record in caplog.records
    )


# --- dflash_verify_mode: reject invalid values instead of reverting to None (#10) ---


@pytest.mark.asyncio
async def test_update_model_settings_rejects_unknown_dflash_verify_mode():
    pool, entry = _failed_pool()

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(dflash_verify_mode="bogus"),
        )

    assert exc_info.value.status_code == 400
    assert "dflash" in exc_info.value.detail
    assert "adaptive" in exc_info.value.detail
    assert "ddtree" in exc_info.value.detail
    assert "off" in exc_info.value.detail


@pytest.mark.asyncio
async def test_update_model_settings_accepts_valid_dflash_verify_mode():
    pool, entry = _failed_pool()
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(dflash_verify_mode="adaptive"),
    )

    assert settings.dflash_verify_mode == "adaptive"


@pytest.mark.asyncio
async def test_update_model_settings_clearing_dflash_verify_mode_skips_validation():
    pool, entry = _failed_pool()
    settings = ModelSettings(dflash_verify_mode="adaptive")

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(dflash_verify_mode=""),
    )

    assert settings.dflash_verify_mode is None


# --- *_draft_model fields: validate existence at write time (#10) ---


@pytest.mark.asyncio
async def test_update_model_settings_rejects_nonexistent_specprefill_draft_model(
    tmp_path,
):
    pool, entry = _failed_pool()
    missing = tmp_path / "does-not-exist"

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(specprefill_draft_model=str(missing)),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_model_settings_accepts_existing_specprefill_draft_model(
    tmp_path,
):
    pool, entry = _failed_pool()
    draft_dir = tmp_path / "draft-model"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text("{}")
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(specprefill_draft_model=str(draft_dir)),
    )

    assert settings.specprefill_draft_model == str(draft_dir)


@pytest.mark.asyncio
async def test_update_model_settings_accepts_repo_id_specprefill_draft_model():
    """A repo id (not an absolute path) is passed through unchecked --
    verifying it exists would require a network round-trip on every
    settings write."""
    pool, entry = _failed_pool()
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(
            specprefill_draft_model="mlx-community/some-draft-4bit"
        ),
    )

    assert settings.specprefill_draft_model == "mlx-community/some-draft-4bit"


@pytest.mark.asyncio
async def test_update_model_settings_rejects_nonexistent_dflash_draft_model(tmp_path):
    pool, entry = _failed_pool()
    missing = tmp_path / "does-not-exist"

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(dflash_draft_model=str(missing)),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_model_settings_rejects_dflash_incompatible_draft_model(
    tmp_path,
):
    pool, entry = _failed_pool()
    draft_dir = tmp_path / "draft-model"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text("{}")

    with (
        patch(
            "omlx.engine.dflash.is_dflash_compatible",
            return_value=(False, "not a supported model_type"),
        ),
        pytest.raises(admin_routes.HTTPException, match="not a supported model_type"),
    ):
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(dflash_draft_model=str(draft_dir)),
        )


@pytest.mark.asyncio
async def test_update_model_settings_accepts_dflash_compatible_draft_model(tmp_path):
    pool, entry = _failed_pool()
    draft_dir = tmp_path / "draft-model"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text("{}")
    settings = ModelSettings()

    with patch(
        "omlx.engine.dflash.is_dflash_compatible",
        return_value=(True, ""),
    ):
        await _update_settings(
            pool,
            settings,
            admin_routes.ModelSettingsRequest(dflash_draft_model=str(draft_dir)),
        )

    assert settings.dflash_draft_model == str(draft_dir)


@pytest.mark.asyncio
async def test_update_model_settings_rejects_nonexistent_vlm_mtp_draft_model(
    tmp_path,
):
    """Validated even when vlm_mtp_enabled is absent from the same payload --
    the gap #10 reported (existing mutex/requires check only fires when
    vlm_mtp_enabled is set in the same request)."""
    pool, entry = _failed_pool()
    missing = tmp_path / "does-not-exist"

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(vlm_mtp_draft_model=str(missing)),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_model_settings_accepts_existing_vlm_mtp_draft_model(tmp_path):
    pool, entry = _failed_pool()
    draft_dir = tmp_path / "draft-model"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text("{}")
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(vlm_mtp_draft_model=str(draft_dir)),
    )

    assert settings.vlm_mtp_draft_model == str(draft_dir)


# --- variant bugs found by pre-pr-review's variant-bug-hunter on #10's diff:
# same silent-coercion pattern as dflash_verify_mode, in the same function. ---


@pytest.mark.asyncio
async def test_update_model_settings_rejects_unknown_turboquant_kv_bits():
    pool, entry = _failed_pool()

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(turboquant_kv_bits=5),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_model_settings_accepts_valid_turboquant_kv_bits():
    pool, entry = _failed_pool()
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(turboquant_kv_bits=2.5),
    )

    assert settings.turboquant_kv_bits == 2.5


@pytest.mark.asyncio
async def test_update_model_settings_clearing_turboquant_kv_bits_resets_to_default():
    pool, entry = _failed_pool()
    settings = ModelSettings(turboquant_kv_bits=8)

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(turboquant_kv_bits=0),
    )

    assert settings.turboquant_kv_bits == 4


@pytest.mark.asyncio
async def test_update_model_settings_rejects_negative_dflash_in_memory_cache_max_entries():
    pool, entry = _failed_pool()

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(dflash_in_memory_cache_max_entries=-3),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_model_settings_accepts_valid_dflash_in_memory_cache_max_entries():
    pool, entry = _failed_pool()
    settings = ModelSettings()

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(dflash_in_memory_cache_max_entries=16),
    )

    assert settings.dflash_in_memory_cache_max_entries == 16


@pytest.mark.asyncio
async def test_update_model_settings_clearing_dflash_in_memory_cache_max_entries_resets_to_default():
    pool, entry = _failed_pool()
    settings = ModelSettings(dflash_in_memory_cache_max_entries=16)

    await _update_settings(
        pool,
        settings,
        admin_routes.ModelSettingsRequest(dflash_in_memory_cache_max_entries=0),
    )

    assert settings.dflash_in_memory_cache_max_entries == 4


# --- P2 fixes from pre-pr-review analyzers on #10's diff ---


@pytest.mark.asyncio
async def test_dflash_incompatible_draft_model_error_names_the_field(tmp_path):
    """security-audit + silent-failure-hunter: the compat-rejection detail
    didn't say which field it was about, unlike the does-not-exist branch
    three lines above it."""
    pool, entry = _failed_pool()
    draft_dir = tmp_path / "draft-model"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text("{}")

    with (
        patch(
            "omlx.engine.dflash.is_dflash_compatible",
            return_value=(False, "not a supported model_type"),
        ),
        pytest.raises(admin_routes.HTTPException) as exc_info,
    ):
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(dflash_draft_model=str(draft_dir)),
        )

    assert "dflash_draft_model" in exc_info.value.detail
    assert "not a supported model_type" in exc_info.value.detail


@pytest.mark.asyncio
async def test_update_model_settings_expands_tilde_in_draft_model_path(
    tmp_path, monkeypatch
):
    """security-audit: a tilde-form local path (~/models/foo) looks like a
    local path but is not Path.is_absolute() -- must be expanded before the
    existence check, or it silently skips validation like a repo id would."""
    pool, entry = _failed_pool()
    monkeypatch.setenv("HOME", str(tmp_path))
    missing = "~/does-not-exist"

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(specprefill_draft_model=missing),
        )

    assert exc_info.value.status_code == 400
