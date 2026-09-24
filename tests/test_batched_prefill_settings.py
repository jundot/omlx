"""Settings persistence and runtime metadata for opt-in prefill batching."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import omlx.admin.routes as admin_routes
import omlx.server as server
from omlx.admin.routes import GlobalSettingsRequest
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.settings import GlobalSettings, SchedulerSettings


@pytest.fixture
def global_settings(tmp_path):
    return GlobalSettings(base_path=tmp_path)


@pytest.fixture
def runtime(monkeypatch, global_settings):
    scheduler = SimpleNamespace(
        config=SimpleNamespace(prefill_max_batch_size=1),
        _prefill_groups=object(),
    )
    direct_scheduler = SimpleNamespace(config=SimpleNamespace(prefill_max_batch_size=1))
    pool = EnginePool(scheduler_config=SimpleNamespace(prefill_max_batch_size=1))
    pool._entries = {
        "loaded": SimpleNamespace(
            engine=SimpleNamespace(
                _engine=SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))
            )
        ),
        "direct": SimpleNamespace(engine=SimpleNamespace(scheduler=direct_scheduler)),
        "unloaded": SimpleNamespace(engine=None),
        "missing": None,
        "non_batched": SimpleNamespace(engine=SimpleNamespace()),
    }
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: global_settings)
    monkeypatch.setattr(server, "_server_state", SimpleNamespace(engine_pool=pool))
    return SimpleNamespace(
        pool=pool, scheduler=scheduler, direct_scheduler=direct_scheduler
    )


def test_batching_defaults_off_for_new_and_legacy_settings():
    assert SchedulerSettings().prefill_max_batch_size == 1
    assert SchedulerSettings.from_dict({}).prefill_max_batch_size == 1
    assert SchedulerSettings.from_dict({"max_num_seqs": 4}).prefill_max_batch_size == 1


def test_existing_positional_scheduler_settings_remain_compatible():
    settings = SchedulerSettings(8, 32, True, "speed", False)
    assert settings.chunked_prefill is True
    assert settings.prefill_priority == "speed"
    assert settings.decode_fairness is False
    assert settings.prefill_max_batch_size == 1


@pytest.mark.parametrize("batch_size", [1, 2, 4, 16])
def test_settings_round_trip(batch_size):
    settings = SchedulerSettings(prefill_max_batch_size=batch_size)
    assert settings.to_dict()["prefill_max_batch_size"] == batch_size
    assert SchedulerSettings.from_dict(settings.to_dict()) == settings


@pytest.mark.parametrize("invalid", [0, -1, True, False, "4", 2.0, None])
def test_constructor_and_dict_reject_invalid_batch_size(invalid):
    with pytest.raises(ValueError, match="prefill_max_batch_size"):
        SchedulerSettings(prefill_max_batch_size=invalid)
    with pytest.raises(ValueError, match="prefill_max_batch_size"):
        SchedulerSettings.from_dict({"prefill_max_batch_size": invalid})


@pytest.mark.parametrize("invalid", [0, -1, True, False, "4", 2.0, None])
def test_global_validation_rejects_mutated_batch_size(global_settings, invalid):
    global_settings.scheduler.prefill_max_batch_size = invalid
    assert any(
        "prefill_max_batch_size" in error for error in global_settings.validate()
    )


def test_file_persistence_and_legacy_load(global_settings, tmp_path):
    global_settings.scheduler.prefill_max_batch_size = 4
    global_settings.save()
    assert GlobalSettings.load(base_path=tmp_path).scheduler.prefill_max_batch_size == 4
    settings_path = tmp_path / "settings.json"
    saved = json.loads(settings_path.read_text())
    assert saved["scheduler"].pop("prefill_max_batch_size") == 4
    settings_path.write_text(json.dumps(saved))
    assert GlobalSettings.load(base_path=tmp_path).scheduler.prefill_max_batch_size == 1


@pytest.mark.parametrize("batch_size", [1, 4])
def test_scheduler_config_receives_batch_size(global_settings, batch_size):
    global_settings.scheduler.prefill_max_batch_size = batch_size
    assert global_settings.to_scheduler_config().prefill_max_batch_size == batch_size


@pytest.mark.parametrize("invalid", [True, False, "4", 2.0])
def test_api_rejects_non_integer_batch_size(invalid):
    with pytest.raises(ValidationError, match="prefill_max_batch_size"):
        GlobalSettingsRequest(prefill_max_batch_size=invalid)


@pytest.mark.asyncio
async def test_api_persists_then_updates_new_and_loaded_engines(
    global_settings, runtime, tmp_path
):
    active_groups = runtime.scheduler._prefill_groups

    def save_settings():
        assert runtime.pool._scheduler_config.prefill_max_batch_size == 1
        assert runtime.scheduler.config.prefill_max_batch_size == 1
        GlobalSettings.save(global_settings)

    with patch.object(global_settings, "save", side_effect=save_settings) as save:
        result = await admin_routes.update_global_settings(
            GlobalSettingsRequest(prefill_max_batch_size=4), is_admin=True
        )

    assert result["success"] is True
    assert "prefill_max_batch_size" in result["runtime_applied"]
    assert global_settings.scheduler.prefill_max_batch_size == 4
    assert runtime.pool._scheduler_config.prefill_max_batch_size == 4
    assert runtime.scheduler.config.prefill_max_batch_size == 4
    assert runtime.direct_scheduler.config.prefill_max_batch_size == 4
    assert runtime.scheduler._prefill_groups is active_groups
    assert GlobalSettings.load(base_path=tmp_path).scheduler.prefill_max_batch_size == 4
    save.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [0, -1, True, False, "4", 2.0, None])
async def test_pool_rejects_invalid_batch_size_without_mutation(runtime, invalid):
    with pytest.raises(ValueError, match="prefill_max_batch_size"):
        await runtime.pool.apply_prefill_max_batch_size(invalid)
    assert runtime.pool._scheduler_config.prefill_max_batch_size == 1
    assert runtime.scheduler.config.prefill_max_batch_size == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("old_size, new_size", [(1, 4), (4, 1)])
async def test_api_waits_for_loading_engine_before_publishing_batch_size(
    global_settings, tmp_path, monkeypatch, old_size, new_size
):
    pool = EnginePool(scheduler_config=SimpleNamespace(prefill_max_batch_size=old_size))
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "llama"}')
    entry = EngineEntry(
        model_id="model",
        model_path=str(model_path),
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
    )
    pool._entries["model"] = entry
    global_settings.scheduler.prefill_max_batch_size = old_size
    config_copied = asyncio.Event()
    allow_publication = asyncio.Event()
    settings_saved = asyncio.Event()
    active_groups = object()

    async def load_engine(model_id, **kwargs):
        assert pool._lock.locked()
        scheduler = SimpleNamespace(
            config=copy.copy(pool._scheduler_config),
            _prefill_groups=active_groups,
        )
        engine = SimpleNamespace(
            _engine=SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))
        )
        config_copied.set()
        await allow_publication.wait()
        pool._entries[model_id].engine = engine

    def save_settings():
        GlobalSettings.save(global_settings)
        settings_saved.set()

    monkeypatch.setattr(pool, "_load_engine", load_engine)
    monkeypatch.setattr(global_settings, "save", save_settings)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: global_settings)
    monkeypatch.setattr(server, "_server_state", SimpleNamespace(engine_pool=pool))

    async with asyncio.timeout(5):
        async with asyncio.TaskGroup() as tasks:
            loading = tasks.create_task(pool.get_engine("model"))
            await config_copied.wait()
            updating = tasks.create_task(
                admin_routes.update_global_settings(
                    GlobalSettingsRequest(prefill_max_batch_size=new_size),
                    is_admin=True,
                )
            )
            await settings_saved.wait()
            assert not updating.done()
            assert entry.engine is None
            assert pool._scheduler_config.prefill_max_batch_size == old_size
            allow_publication.set()

    assert updating.result()["success"] is True
    assert "prefill_max_batch_size" in updating.result()["runtime_applied"]
    assert loading.result() is entry.engine
    scheduler = pool._resolve_scheduler_from_engine(entry.engine)
    assert pool._scheduler_config.prefill_max_batch_size == new_size
    assert scheduler.config.prefill_max_batch_size == new_size
    assert scheduler._prefill_groups is active_groups
    assert (
        GlobalSettings.load(base_path=tmp_path).scheduler.prefill_max_batch_size
        == new_size
    )


@pytest.mark.asyncio
async def test_api_can_disable_batching_without_replacing_active_groups(
    global_settings, runtime
):
    global_settings.scheduler.prefill_max_batch_size = 4
    runtime.pool._scheduler_config.prefill_max_batch_size = 4
    runtime.scheduler.config.prefill_max_batch_size = 4
    active_groups = runtime.scheduler._prefill_groups

    result = await admin_routes.update_global_settings(
        GlobalSettingsRequest(prefill_max_batch_size=1), is_admin=True
    )

    assert "prefill_max_batch_size" in result["runtime_applied"]
    assert global_settings.scheduler.prefill_max_batch_size == 1
    assert runtime.pool._scheduler_config.prefill_max_batch_size == 1
    assert runtime.scheduler.config.prefill_max_batch_size == 1
    assert runtime.scheduler._prefill_groups is active_groups


@pytest.mark.asyncio
async def test_api_saves_batch_size_without_engine_pool(global_settings, runtime):
    server._server_state.engine_pool = None
    result = await admin_routes.update_global_settings(
        GlobalSettingsRequest(prefill_max_batch_size=2), is_admin=True
    )
    assert result["success"] is True
    assert global_settings.scheduler.prefill_max_batch_size == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [0, -1])
async def test_api_rejects_invalid_size_before_any_mutation(
    global_settings, runtime, batch_size
):
    previous_port = global_settings.server.port
    with (
        patch.object(global_settings, "save") as save,
        pytest.raises(HTTPException) as raised,
    ):
        await admin_routes.update_global_settings(
            GlobalSettingsRequest(prefill_max_batch_size=batch_size, port=9999),
            is_admin=True,
        )
    assert raised.value.status_code == 400
    assert "prefill_max_batch_size" in raised.value.detail
    assert global_settings.server.port == previous_port
    assert global_settings.scheduler.prefill_max_batch_size == 1
    assert runtime.pool._scheduler_config.prefill_max_batch_size == 1
    assert runtime.scheduler.config.prefill_max_batch_size == 1
    save.assert_not_called()


@pytest.mark.asyncio
async def test_api_does_not_apply_batch_size_when_request_validation_fails(
    global_settings, runtime
):
    with (
        patch.object(global_settings, "save") as save,
        pytest.raises(HTTPException) as raised,
    ):
        await admin_routes.update_global_settings(
            GlobalSettingsRequest(prefill_max_batch_size=4, api_key="abc"),
            is_admin=True,
        )
    assert raised.value.status_code == 400
    assert global_settings.scheduler.prefill_max_batch_size == 1
    assert runtime.pool._scheduler_config.prefill_max_batch_size == 1
    assert runtime.scheduler.config.prefill_max_batch_size == 1
    save.assert_not_called()


@pytest.mark.asyncio
async def test_api_rolls_back_batch_size_when_global_validation_fails(
    global_settings, runtime
):
    with (
        patch.object(global_settings, "validate", return_value=["invalid settings"]),
        patch.object(global_settings, "save") as save,
        pytest.raises(HTTPException) as raised,
    ):
        await admin_routes.update_global_settings(
            GlobalSettingsRequest(prefill_max_batch_size=4), is_admin=True
        )
    assert raised.value.status_code == 400
    assert global_settings.scheduler.prefill_max_batch_size == 1
    assert runtime.pool._scheduler_config.prefill_max_batch_size == 1
    assert runtime.scheduler.config.prefill_max_batch_size == 1
    save.assert_not_called()


@pytest.mark.asyncio
async def test_api_rolls_back_batch_size_when_save_fails(global_settings, runtime):
    with (
        patch.object(global_settings, "save", side_effect=OSError("disk full")),
        pytest.raises(HTTPException) as raised,
    ):
        await admin_routes.update_global_settings(
            GlobalSettingsRequest(prefill_max_batch_size=4), is_admin=True
        )
    assert raised.value.status_code == 500
    assert global_settings.scheduler.prefill_max_batch_size == 1
    assert runtime.pool._scheduler_config.prefill_max_batch_size == 1
    assert runtime.scheduler.config.prefill_max_batch_size == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_request",
    [GlobalSettingsRequest(), GlobalSettingsRequest(prefill_max_batch_size=None)],
)
async def test_api_omitted_or_null_batch_size_preserves_setting(
    global_settings, runtime, api_request
):
    global_settings.scheduler.prefill_max_batch_size = 4
    runtime.pool._scheduler_config.prefill_max_batch_size = 4
    runtime.scheduler.config.prefill_max_batch_size = 4

    result = await admin_routes.update_global_settings(api_request, is_admin=True)

    assert "prefill_max_batch_size" not in result["runtime_applied"]
    assert global_settings.scheduler.prefill_max_batch_size == 4
    assert runtime.pool._scheduler_config.prefill_max_batch_size == 4
    assert runtime.scheduler.config.prefill_max_batch_size == 4


@pytest.mark.asyncio
async def test_get_settings_exposes_prefill_batch_size(global_settings, runtime):
    global_settings.scheduler.prefill_max_batch_size = 4
    with (
        patch.object(admin_routes, "get_system_memory_info", return_value=MagicMock()),
        patch.object(admin_routes, "get_ssd_disk_info", return_value=MagicMock()),
    ):
        response = await admin_routes.get_global_settings(is_admin=True)
    assert response["scheduler"]["prefill_max_batch_size"] == 4
