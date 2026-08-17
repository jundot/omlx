# SPDX-License-Identifier: Apache-2.0
"""Tests for load-failure invalidation in admin model settings."""

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
async def test_qwen_ane_prefill_rejects_other_model_families():
    pool, entry = _failed_pool()
    entry.config_model_type = "gemma4"

    with pytest.raises(admin_routes.HTTPException, match="Qwen3.5/3.6/3.8"):
        await _update_settings(
            pool,
            ModelSettings(),
            admin_routes.ModelSettingsRequest(qwen35_ane_prefill_enabled=True),
        )
