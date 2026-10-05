# SPDX-License-Identifier: Apache-2.0
"""Tests for load-failure invalidation in admin model settings."""

import asyncio
import copy
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


def _idle_engine() -> MagicMock:
    engine = MagicMock()
    engine.has_active_requests.return_value = False
    engine.scheduler = None
    engine._engine = None
    return engine


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
        ({"qwen35_ane_prefill_enabled": True}, {"qwen35_oq_a8_enabled": True}),
        ({"qwen35_oq_a8_enabled": True}, {"qwen35_ane_prefill_enabled": True}),
        ({}, {"qwen35_oq_a8_enabled": True, "qwen35_ane_prefill_enabled": True}),
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
                qwen35_oq_a8_enabled=True, qwen35_oq_a8_min_tokens=256
            ),
        )

    assert settings.qwen35_oq_a8_enabled is True
    assert settings.qwen35_oq_a8_min_tokens == 256


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
                admin_routes.ModelSettingsRequest(qwen35_oq_a8_enabled=True),
            )

    assert excinfo.value.status_code == 400
    assert "M5-series or newer" in excinfo.value.detail
    assert settings.qwen35_oq_a8_enabled is False


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


# ---------------------------------------------------------------------------
# Issue #4217: a stale dflash_draft_model must not 422 unrelated settings saves
# ---------------------------------------------------------------------------


def _draft_path_client(tmp_path, monkeypatch):
    """Wire the admin router behind a TestClient with a real settings manager.

    Returns ``(client, settings, manager)``; ``settings`` is the stored state
    the fake manager snapshots and ``manager`` records the persistence calls.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    pool, entry = _failed_pool()
    settings = ModelSettings()
    manager = MagicMock()
    # Snapshot per call so tests can seed stored settings after construction.
    manager.get_settings.side_effect = lambda _mid: copy.deepcopy(settings)
    manager.set_settings.side_effect = lambda _, updated: settings.__dict__.update(
        updated.__dict__
    )
    state = MagicMock()

    async def _fake_require_admin():
        return True

    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: state)
    # The fake entry's /tmp/ling has no config.json, so short-circuit the
    # compatibility probe that otherwise 400s before the draft path is read.
    from omlx.engine import dflash as dflash_engine

    monkeypatch.setattr(dflash_engine, "is_dflash_compatible", lambda _p: (True, ""))

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app), settings, manager


def _draft_dir(tmp_path) -> str:
    """A valid local draft model directory the caller may delete mid-test."""
    draft = tmp_path / "Qwen-DFlash-draft"
    draft.mkdir()
    (draft / "config.json").write_text("{}")
    return str(draft)


def _delete_draft(path: str) -> None:
    """Simulate the user deleting the draft model directory from disk."""
    import shutil

    shutil.rmtree(path)


def test_stale_dflash_draft_path_does_not_block_unrelated_settings_save(
    tmp_path, monkeypatch
):
    """#4217: toggling Guided Grammar must survive a stale draft path."""
    client, settings, _ = _draft_path_client(tmp_path, monkeypatch)
    draft = _draft_dir(tmp_path)

    first = client.put(
        "/admin/api/models/ling/settings",
        json={"dflash_enabled": True, "dflash_draft_model": draft},
    )
    assert first.status_code == 200, first.text

    # The draft model directory is deleted behind oMLX's back.
    _delete_draft(draft)

    # Unrelated setting save -- DFlash is off, stale path is parked in the UI.
    second = client.put(
        "/admin/api/models/ling/settings",
        json={
            "guided_grammar_enabled": True,
            "dflash_enabled": False,
            "dflash_draft_model": draft,
        },
    )
    assert second.status_code == 200, second.text
    assert settings.guided_grammar_enabled is True


def test_disabled_dflash_stale_path_is_not_cleared_from_stored_settings(
    tmp_path, monkeypatch
):
    """#4217: the UI deliberately parks a stale draft path; never mutate it."""
    client, settings, _ = _draft_path_client(tmp_path, monkeypatch)
    draft = _draft_dir(tmp_path)

    assert (
        client.put(
            "/admin/api/models/ling/settings",
            json={"dflash_enabled": True, "dflash_draft_model": draft},
        ).status_code
        == 200
    )

    _delete_draft(draft)

    assert (
        client.put(
            "/admin/api/models/ling/settings",
            json={"dflash_enabled": False, "dflash_draft_model": draft},
        ).status_code
        == 200
    )
    # The value survives so it comes back when Custom is picked again.
    assert settings.dflash_draft_model == draft


def test_null_dflash_enabled_leaves_stale_draft_path_untouched(tmp_path, monkeypatch):
    """#4217: an explicit null means disabled (bool(None) is False downstream)."""
    client, settings, _ = _draft_path_client(tmp_path, monkeypatch)
    draft = _draft_dir(tmp_path)
    _delete_draft(draft)

    response = client.put(
        "/admin/api/models/ling/settings",
        json={"dflash_enabled": None, "dflash_draft_model": draft},
    )
    assert response.status_code == 200, response.text
    assert settings.dflash_draft_model == draft
    assert settings.dflash_enabled is False


def test_dflash_enabled_absent_from_payload_does_not_validate_draft(
    tmp_path, monkeypatch
):
    """#4217: with DFlash stored off, an unsent flag must not hard-fail."""
    client, settings, _ = _draft_path_client(tmp_path, monkeypatch)
    draft = _draft_dir(tmp_path)
    _delete_draft(draft)

    response = client.put(
        "/admin/api/models/ling/settings",
        json={"temperature": 0.3, "dflash_draft_model": draft},
    )
    assert response.status_code == 200, response.text
    assert settings.temperature == 0.3


def test_enabling_dflash_with_a_stale_draft_path_is_still_rejected(
    tmp_path, monkeypatch
):
    """#4217 safety net: actively enabling DFlash must surface the bad path."""
    client, settings, _ = _draft_path_client(tmp_path, monkeypatch)
    draft = _draft_dir(tmp_path)
    _delete_draft(draft)

    response = client.put(
        "/admin/api/models/ling/settings",
        json={"dflash_enabled": True, "dflash_draft_model": draft},
    )
    assert response.status_code == 422
    assert "no config.json" in response.text
    assert settings.dflash_draft_model is None


def test_re_enabled_dflash_after_a_disabled_save_is_rejected_again(
    tmp_path, monkeypatch
):
    """#4217: re-enabling DFlash later must not slip a stale path through."""
    client, settings, _ = _draft_path_client(tmp_path, monkeypatch)
    draft = _draft_dir(tmp_path)

    client.put(
        "/admin/api/models/ling/settings",
        json={"dflash_enabled": True, "dflash_draft_model": draft},
    )
    _delete_draft(draft)
    # Balanced profile keeps the stale value parked.
    assert (
        client.put(
            "/admin/api/models/ling/settings",
            json={"dflash_enabled": False, "dflash_draft_model": draft},
        ).status_code
        == 200
    )
    assert settings.dflash_draft_model == draft

    # Back to Custom with the stale path still selected -> rejected loudly.
    response = client.put(
        "/admin/api/models/ling/settings",
        json={"dflash_enabled": True, "dflash_draft_model": draft},
    )
    assert response.status_code == 422
    assert draft in response.text


def test_remote_hf_draft_repo_id_is_never_treated_as_a_local_path(
    tmp_path, monkeypatch
):
    """#4217: a repo id is never downloaded/resolved, valid while enabled."""
    client, settings, _ = _draft_path_client(tmp_path, monkeypatch)
    repo = "NewHorizonGroup/Qwen3.8-27B-DFlash2-oQ4"

    response = client.put(
        "/admin/api/models/ling/settings",
        json={"dflash_enabled": True, "dflash_draft_model": repo},
    )
    assert response.status_code == 200, response.text
    assert settings.dflash_draft_model == repo


def test_specprefill_draft_path_validation_is_unchanged():
    """#4217 must not relax the sibling SpecPrefill/vlm-mtp validators."""
    for field in ("specprefill_draft_model", "vlm_mtp_draft_model"):
        with pytest.raises(ValueError, match="no config.json"):
            admin_routes.ModelSettingsRequest.model_validate(
                {field: "/nonexistent/spec-draft", "dflash_enabled": False}
            )


def test_dflash_draft_field_unsent_rejects_the_effective_on_state(
    tmp_path, monkeypatch
):
    """#4217: the payload validator is skipped for an unsent draft field, but
    the route still rejects an effective DFlash-on + unusable-path state.

    The stored value is never rewritten, so without the route check the broken
    configuration would be silently re-persisted on every unrelated save.
    """
    client, settings, manager = _draft_path_client(tmp_path, monkeypatch)
    draft = _draft_dir(tmp_path)
    settings.dflash_draft_model = draft
    settings.dflash_enabled = True
    _delete_draft(draft)

    response = client.put("/admin/api/models/ling/settings", json={"temperature": 0.3})
    assert response.status_code == 422, response.text
    assert "no config.json" in response.text
    assert settings.temperature != 0.3
    manager.set_settings.assert_not_called()


def test_single_field_draft_path_patch_with_stored_dflash_on_is_rejected(
    tmp_path, monkeypatch
):
    """macOS single-field patch shape: no ``dflash_enabled`` in the payload.

    ``ModelSettingsScreenVM`` sends only ``dflash_draft_model`` for the draft
    picker and ``ModelsDTO`` uses ``encodeIfPresent``, so the request validator
    takes its OFF branch. The route, which knows the stored flag, must reject
    the save and persist nothing.
    """
    client, settings, manager = _draft_path_client(tmp_path, monkeypatch)
    stored_draft = _draft_dir(tmp_path)
    # A path the user deleted from disk: absolute, no config.json.
    stale = str(tmp_path / "deleted-draft")
    settings.dflash_enabled = True
    settings.dflash_draft_model = stored_draft

    response = client.put(
        "/admin/api/models/ling/settings", json={"dflash_draft_model": stale}
    )
    assert response.status_code == 422, response.text
    assert "no config.json" in response.text
    assert stale in response.text

    # Rejected before persistence: the stored path and flag are untouched.
    manager.set_settings.assert_not_called()
    assert settings.dflash_draft_model == stored_draft
    assert settings.dflash_enabled is True


def test_flag_only_enable_with_a_stale_stored_draft_path_is_rejected(
    tmp_path, monkeypatch
):
    """The macOS app enables DFlash with the flag alone.

    The PR body claimed the path is rejected "when DFlash is next actually
    enabled"; that never happened because the draft-path validator only runs
    when the payload carries the draft field.
    """
    client, settings, manager = _draft_path_client(tmp_path, monkeypatch)
    draft = _draft_dir(tmp_path)
    settings.dflash_enabled = False
    settings.dflash_draft_model = draft
    _delete_draft(draft)

    response = client.put(
        "/admin/api/models/ling/settings", json={"dflash_enabled": True}
    )
    assert response.status_code == 422, response.text
    assert "no config.json" in response.text
    assert draft in response.text

    manager.set_settings.assert_not_called()
    assert settings.dflash_enabled is False


def test_single_field_draft_path_patch_with_stored_dflash_off_is_accepted(
    tmp_path, monkeypatch
):
    """#4217 regression guard: a disabled DFlash parks the stale path."""
    client, settings, manager = _draft_path_client(tmp_path, monkeypatch)
    draft = _draft_dir(tmp_path)
    settings.dflash_enabled = False
    _delete_draft(draft)

    response = client.put(
        "/admin/api/models/ling/settings", json={"dflash_draft_model": draft}
    )
    assert response.status_code == 200, response.text
    manager.set_settings.assert_called_once()
    assert settings.dflash_draft_model == draft


def test_profile_normalization_keeps_a_stale_draft_path_when_dflash_is_off(
    tmp_path, monkeypatch
):
    """Profile create/update behaviour for a dormant broken path (#4217).

    ``_normalize_profile_settings`` runs the request validator, which warns but
    never rewrites the value; the route's effective-state check does not run on
    profile writes, so a profile may legitimately store a dormant bad path.
    """
    monkeypatch.setattr(admin_routes, "_WARNED_DORMANT_DFLASH_DRAFTS", set())
    stale = str(tmp_path / "deleted-draft")

    normalized = admin_routes._normalize_profile_settings(
        {"dflash_enabled": False, "dflash_draft_model": stale}
    )

    assert normalized == {"dflash_enabled": False, "dflash_draft_model": stale}


def test_empty_dflash_draft_path_normalises_to_none_while_disabled():
    """#4217: the off branch keeps the sibling validators' "" -> None rule."""
    request = admin_routes.ModelSettingsRequest(
        dflash_draft_model="", dflash_enabled=False
    )
    assert request.model_dump()["dflash_draft_model"] is None


def test_dormant_draft_warning_fires_once_per_distinct_value(
    tmp_path, monkeypatch, caplog
):
    """#4217: the full-payload console save must not warn on every write."""
    import logging

    monkeypatch.setattr(admin_routes, "_WARNED_DORMANT_DFLASH_DRAFTS", set())
    client, settings, _ = _draft_path_client(tmp_path, monkeypatch)
    draft = _draft_dir(tmp_path)
    _delete_draft(draft)
    payload = {"dflash_enabled": False, "dflash_draft_model": draft}

    with caplog.at_level(logging.WARNING, logger="omlx.admin.routes"):
        for _ in range(3):
            assert (
                client.put("/admin/api/models/ling/settings", json=payload).status_code
                == 200
            )

    warnings = [
        record
        for record in caplog.records
        if "missing or incomplete" in record.getMessage()
    ]
    assert len(warnings) == 1


def test_dflash_draft_warning_only_fires_for_a_broken_path(
    tmp_path, monkeypatch, caplog
):
    """#4217: a healthy draft path parked while DFlash is off stays silent."""
    import logging

    client, settings, _ = _draft_path_client(tmp_path, monkeypatch)
    draft = _draft_dir(tmp_path)

    with caplog.at_level(logging.WARNING, logger="omlx.admin.routes"):
        assert (
            client.put(
                "/admin/api/models/ling/settings",
                json={"dflash_enabled": False, "dflash_draft_model": draft},
            ).status_code
            == 200
        )
    assert caplog.records == []
    assert settings.dflash_draft_model == draft

    _delete_draft(draft)
    with caplog.at_level(logging.WARNING, logger="omlx.admin.routes"):
        assert (
            client.put(
                "/admin/api/models/ling/settings",
                json={"dflash_enabled": False, "dflash_draft_model": draft},
            ).status_code
            == 200
        )
    assert any("missing or incomplete" in r.message for r in caplog.records)
    assert settings.dflash_draft_model == draft


def test_dflash_enabled_is_declared_before_the_draft_model_field():
    """#4217: the sibling-aware validator relies on pydantic field order.

    ``info.data`` only holds fields validated so far, so moving
    ``dflash_enabled`` below ``dflash_draft_model`` would leave ``info.data``
    empty and the validator would take the OFF branch, silently ACCEPTING a
    stale path with no config.json instead of rejecting it. Fail loudly
    instead.
    """
    names = list(admin_routes.ModelSettingsRequest.model_fields)
    assert names.index("dflash_enabled") < names.index("dflash_draft_model")
