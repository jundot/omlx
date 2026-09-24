"""Model-level admission limits stay independent across live engines."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

import omlx.admin.routes as routes
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_profiles import EXCLUDED_FROM_PROFILES
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.scheduler import SchedulerConfig


@pytest.mark.parametrize("value", [None, 1, 2, 4])
def test_override_persistence(tmp_path, value):
    manager = ModelSettingsManager(tmp_path)
    manager.set_settings("hy", ModelSettings(prefill_max_batch_size=value))
    assert (
        ModelSettingsManager(tmp_path).get_settings("hy").prefill_max_batch_size
        == value
    )
    assert "prefill_max_batch_size" in EXCLUDED_FROM_PROFILES


@pytest.mark.parametrize("value", [0, -1, True, False, 2.0, "4"])
def test_override_validation(value):
    with pytest.raises(ValueError, match="prefill_max_batch_size"):
        ModelSettings(prefill_max_batch_size=value)
    with pytest.raises(ValidationError):
        routes.ModelSettingsRequest(prefill_max_batch_size=value)


@pytest.fixture
def model_pool(tmp_path):
    pool = EnginePool(scheduler_config=SchedulerConfig(prefill_max_batch_size=1))
    manager = ModelSettingsManager(tmp_path)
    pool._settings_manager = manager
    for name, override in [("hy", 4), ("disabled", 1), ("inherited", None)]:
        manager.set_settings(name, ModelSettings(prefill_max_batch_size=override))
        scheduler = SimpleNamespace(
            config=SchedulerConfig(prefill_max_batch_size=override or 1)
        )
        entry = EngineEntry(
            model_id=name,
            model_path=str(tmp_path / name),
            model_type="llm",
            engine_type="batched",
            estimated_size=1,
        )
        entry.engine = SimpleNamespace(scheduler=scheduler)
        pool._entries[name] = entry
    return pool, manager


@pytest.mark.asyncio
async def test_global_update_retains_overrides_and_clearing_inherits(model_pool):
    pool, manager = model_pool
    await pool.apply_prefill_max_batch_size(2)
    sizes = {
        name: entry.engine.scheduler.config.prefill_max_batch_size
        for name, entry in pool._entries.items()
    }
    assert sizes == {"hy": 4, "disabled": 1, "inherited": 2}
    manager.set_settings("hy", ModelSettings())
    await pool.apply_model_prefill_max_batch_size("hy")
    assert pool._entries["hy"].engine.scheduler.config.prefill_max_batch_size == 2
    assert pool._scheduler_config.prefill_max_batch_size == 2
    assert pool._effective_prefill_batch_size("disabled") == 1
    assert pool._effective_prefill_batch_size("new") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,expected",
    [
        ({}, 4),
        ({"prefill_max_batch_size": 2}, 2),
        ({"prefill_max_batch_size": 1}, 1),
        ({"prefill_max_batch_size": None}, 1),
    ],
)
async def test_model_api_omission_null_and_live_override(
    model_pool, monkeypatch, payload, expected
):
    pool, manager = model_pool
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(routes, "_get_server_state", lambda: None)
    result = await routes.update_model_settings(
        "hy", routes.ModelSettingsRequest(**payload), is_admin=True
    )
    assert not result["requires_reload"]
    assert (
        pool._entries["hy"].engine.scheduler.config.prefill_max_batch_size == expected
    )
    assert pool._entries["disabled"].engine.scheduler.config.prefill_max_batch_size == 1
    assert manager.get_settings("hy").prefill_max_batch_size == payload.get(
        "prefill_max_batch_size", 4
    )


@pytest.mark.asyncio
async def test_failed_model_save_does_not_change_live_limit(model_pool, monkeypatch):
    pool, manager = model_pool
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(routes, "_get_server_state", lambda: None)
    monkeypatch.setattr(
        manager, "set_settings", MagicMock(side_effect=OSError("disk full"))
    )
    with pytest.raises(OSError):
        await routes.update_model_settings(
            "hy", routes.ModelSettingsRequest(prefill_max_batch_size=1), is_admin=True
        )
    assert pool._entries["hy"].engine.scheduler.config.prefill_max_batch_size == 4


@pytest.mark.asyncio
async def test_loading_models_get_independent_configs(
    model_pool, monkeypatch, tmp_path
):
    from unittest.mock import AsyncMock

    import omlx.engine_pool as engine_pool_module

    pool, _manager = model_pool
    pool._get_final_ceiling = lambda: 0
    configs = []

    def make_engine(*args, **kwargs):
        configs.append(kwargs["scheduler_config"])
        engine = MagicMock()
        engine.start = AsyncMock()
        engine.stop = AsyncMock()
        return engine

    monkeypatch.setattr(engine_pool_module, "BatchedEngine", make_engine)
    for model_id in ("hy", "disabled", "inherited"):
        entry = pool._entries[model_id]
        entry.engine = None
        path = tmp_path / model_id
        path.mkdir()
        (path / "config.json").write_text('{"model_type": "llama"}')
        await pool._load_engine(model_id)

    assert [config.prefill_max_batch_size for config in configs] == [4, 1, 1]
    assert len({id(config) for config in configs}) == 3
    assert all(config is not pool._scheduler_config for config in configs)
    assert [config.model_name for config in configs] == ["hy", "disabled", "inherited"]
    assert pool._scheduler_config.prefill_max_batch_size == 1
