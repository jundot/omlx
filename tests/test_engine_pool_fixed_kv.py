# SPDX-License-Identifier: Apache-2.0
"""Engine-pool integration tests for fixed KV launch admission."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import InsufficientMemoryError, ModelUnavailableError
from omlx.fixed_kv_memory import estimate_model_memory
from omlx.scheduler import SchedulerConfig


def _entry(tmp_path, *, weights=1_000):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "test_gqa",
                "num_hidden_layers": 1,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "hidden_size": 64,
                "num_nextn_predict_layers": 1,
                "max_position_embeddings": 4096,
                "torch_dtype": "float16",
            }
        ),
        encoding="utf-8",
    )
    return EngineEntry(
        model_id="model",
        model_path=str(tmp_path),
        model_type="llm",
        engine_type="batched",
        estimated_size=weights,
        model_context_length=4096,
    )


def _pool(entry, *, context=257, slots=8, ceiling=10**9):
    config = SchedulerConfig(
        fixed_kv_cache_enabled=True,
        default_context_window=context,
        max_num_seqs=slots,
        completion_batch_size=slots,
    )
    pool = EnginePool(config)
    pool._entries[entry.model_id] = entry
    pool._get_final_ceiling = lambda: ceiling
    pool._get_admission_ceiling = lambda: ceiling
    return pool


def test_launch_plan_caps_sessions_to_memory_budget(tmp_path, monkeypatch):
    entry = _entry(tmp_path)
    per_session = estimate_model_memory(
        tmp_path,
        257,
        weights_bytes=entry.estimated_size,
        requested_session_slots=1,
        available_memory_bytes=10**9,
    ).per_session_kv_bytes
    budget = entry.estimated_size + 5 * per_session
    pool = _pool(entry, ceiling=budget)
    monkeypatch.setattr(
        "omlx.engine_pool.virtual_memory",
        lambda: SimpleNamespace(available=budget),
    )
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)

    plan = pool._plan_fixed_kv_memory(
        entry,
        SimpleNamespace(max_context_window=257),
        weights_bytes=entry.estimated_size,
        other_fixed_bytes=0,
    )

    assert plan.requested_session_slots == 8
    assert plan.reserved_session_slots == 4
    assert plan.fixed_kv_cache_bytes == 4 * per_session
    assert plan.pool_scratch_bytes == per_session
    assert plan.configured_concurrency_capped is True


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (SimpleNamespace(dflash_enabled=True), "DFlash"),
        (SimpleNamespace(turboquant_kv_enabled=True), "TurboQuant"),
        (
            SimpleNamespace(
                mtp_enabled=False,
                specprefill_enabled=True,
                specprefill_draft_model="draft",
            ),
            "SpecPrefill",
        ),
        (
            SimpleNamespace(
                mtp_enabled=False,
                specprefill_enabled=False,
                vlm_mtp_enabled=True,
                vlm_mtp_draft_model="drafter",
            ),
            "VLM MTP",
        ),
    ],
)
def test_launch_plan_refuses_unplanned_auxiliary_caches(
    tmp_path, monkeypatch, settings, message
):
    entry = _entry(tmp_path)
    pool = _pool(entry)
    monkeypatch.setattr(
        "omlx.engine_pool.virtual_memory",
        lambda: SimpleNamespace(available=10**9),
    )
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    settings.max_context_window = 257

    with pytest.raises(ModelUnavailableError, match=message):
        pool._plan_fixed_kv_memory(
            entry,
            settings,
            weights_bytes=entry.estimated_size,
            other_fixed_bytes=0,
        )


def test_launch_plan_refuses_distributed_without_rank_local_pool(
    tmp_path, monkeypatch
):
    entry = _entry(tmp_path)
    pool = _pool(entry)
    monkeypatch.setattr(
        pool, "_distributed_deployment_for_entry", lambda _entry: object()
    )

    with pytest.raises(ModelUnavailableError, match="distributed models"):
        pool._plan_fixed_kv_memory(
            entry,
            SimpleNamespace(max_context_window=257),
            weights_bytes=entry.estimated_size,
            other_fixed_bytes=0,
        )


def test_launch_plan_includes_native_mtp_cache(tmp_path, monkeypatch):
    entry = _entry(tmp_path)
    pool = _pool(entry, slots=2)
    monkeypatch.setattr(
        "omlx.engine_pool.virtual_memory",
        lambda: SimpleNamespace(available=10**9),
    )
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)

    plan = pool._plan_fixed_kv_memory(
        entry,
        SimpleNamespace(max_context_window=257, mtp_enabled=True),
        weights_bytes=entry.estimated_size,
        other_fixed_bytes=0,
    )

    assert plan.native_mtp_kv_bytes_per_session > 0
    assert plan.per_session_kv_bytes > plan.native_mtp_kv_bytes_per_session


def test_launch_plan_honors_mapping_profile_context(tmp_path, monkeypatch):
    entry = _entry(tmp_path)
    pool = _pool(entry, context=4096, slots=1)
    monkeypatch.setattr(
        "omlx.engine_pool.virtual_memory",
        lambda: SimpleNamespace(available=10**9),
    )
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)

    plan = pool._plan_fixed_kv_memory(
        entry,
        {"max_context_window": 257, "mtp_enabled": True},
        weights_bytes=entry.estimated_size,
        other_fixed_bytes=0,
    )

    assert plan.context_window == 257
    assert plan.native_mtp_kv_bytes_per_session > 0


@pytest.mark.asyncio
async def test_admission_commit_reuse_and_unload_share_one_plan(tmp_path, monkeypatch):
    entry = _entry(tmp_path)
    pool = _pool(entry, slots=2)
    monkeypatch.setattr(
        "omlx.engine_pool.virtual_memory",
        lambda: SimpleNamespace(available=10**9),
    )
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)

    class FakeEngine:
        def __init__(self, **kwargs):
            self.scheduler_config = kwargs["scheduler_config"]
            self.start = AsyncMock()
            self.stop = AsyncMock()

        def get_fixed_kv_memory(self):
            plan = entry.memory
            return {
                "context_window": plan.context_window,
                "reserved_session_slots": plan.reserved_session_slots,
                "fixed_kv_cache_bytes": plan.fixed_kv_cache_bytes,
                "per_session_kv_bytes": plan.per_session_kv_bytes,
                "committed_kv_cache_bytes": plan.fixed_kv_cache_bytes,
                "pool_scratch_bytes": plan.pool_scratch_bytes,
                "committed_pool_bytes": (
                    plan.fixed_kv_cache_bytes + plan.pool_scratch_bytes
                ),
                "materialized_delta_bytes": (
                    plan.fixed_kv_cache_bytes + plan.pool_scratch_bytes
                ),
                "lifecycle": "committed",
            }

        def has_active_requests(self):
            return False

    fake = FakeEngine(scheduler_config=pool._scheduler_config)
    with patch("omlx.engine_pool.BatchedEngine", return_value=fake):
        first = await pool.get_engine("model")
        second = await pool.get_engine("model")

    assert first is second is fake
    fake.start.assert_awaited_once()
    assert entry.memory["lifecycle"] == "committed"
    assert (
        entry.memory["committed_kv_cache_bytes"] == entry.memory["fixed_kv_cache_bytes"]
    )
    assert pool.current_model_memory == entry.memory["estimated_total_bytes"]
    assert fake.scheduler_config.fixed_kv_cache_session_slots == 2
    status_memory = pool.get_status()["models"][0]["memory"]
    assert status_memory["lifecycle"] == "committed"
    assert status_memory["reserved_session_slots"] == 2

    await pool._unload_engine("model")

    fake.stop.assert_awaited_once()
    assert pool.current_model_memory == 0
    assert entry.memory is None


@pytest.mark.asyncio
async def test_per_model_disable_uses_original_growing_cache(tmp_path, monkeypatch):
    entry = _entry(tmp_path)
    pool = _pool(entry, slots=2)
    settings = SimpleNamespace(
        fixed_kv_cache_enabled=False,
        max_context_window=257,
        to_dict=lambda: {
            "fixed_kv_cache_enabled": False,
            "max_context_window": 257,
        },
    )
    pool._settings_manager = SimpleNamespace(get_settings=lambda _model_id: settings)
    monkeypatch.setattr(
        "omlx.engine_pool.virtual_memory",
        lambda: SimpleNamespace(available=10**9),
    )
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)

    class FakeEngine:
        def __init__(self, **kwargs):
            self.scheduler_config = kwargs["scheduler_config"]
            self.start = AsyncMock()
            self.stop = AsyncMock()

        def has_active_requests(self):
            return False

    fake = FakeEngine(scheduler_config=pool._scheduler_config)
    with patch("omlx.engine_pool.BatchedEngine", return_value=fake):
        loaded = await pool.get_engine("model")

    assert loaded is fake
    fake.start.assert_awaited_once()
    assert entry.memory is None
    assert entry.runtime_estimated_size == entry.estimated_size
    assert fake.scheduler_config.fixed_kv_cache_context_window == 0
    assert fake.scheduler_config.fixed_kv_cache_session_slots == 0

    await pool._unload_engine("model")


@pytest.mark.asyncio
async def test_no_full_context_session_fit_refuses_before_engine_start(
    tmp_path, monkeypatch
):
    entry = _entry(tmp_path, weights=10_000)
    pool = _pool(entry, context=256, ceiling=10_000)
    monkeypatch.setattr(
        "omlx.engine_pool.virtual_memory",
        lambda: SimpleNamespace(available=10_000),
    )
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)

    with (
        patch("omlx.engine_pool.BatchedEngine") as engine_factory,
        pytest.raises(InsufficientMemoryError, match="one cache slot"),
    ):
        await pool.get_engine("model")

    engine_factory.assert_not_called()


@pytest.mark.asyncio
async def test_materialization_failure_stops_partial_engine_and_clears_plan(
    tmp_path, monkeypatch
):
    entry = _entry(tmp_path)
    pool = _pool(entry, slots=2)
    monkeypatch.setattr(
        "omlx.engine_pool.virtual_memory",
        lambda: SimpleNamespace(available=10**9),
    )
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    fake = SimpleNamespace(
        start=AsyncMock(side_effect=RuntimeError("cache materialization failed")),
        stop=AsyncMock(),
    )

    with (
        patch("omlx.engine_pool.BatchedEngine", return_value=fake),
        pytest.raises(ModelUnavailableError, match="cache materialization failed"),
    ):
        await pool.get_engine("model")

    fake.stop.assert_awaited_once()
    assert entry.engine is None
    assert entry.memory is None
    assert entry.runtime_estimated_size is None
    assert pool.current_model_memory == 0


def test_context_is_part_of_runtime_reuse_signature(tmp_path):
    entry = _entry(tmp_path)
    pool = _pool(entry)
    first = SimpleNamespace(
        max_context_window=256, to_dict=lambda: {"max_context_window": 256}
    )
    second = SimpleNamespace(
        max_context_window=512, to_dict=lambda: {"max_context_window": 512}
    )

    assert pool._engine_runtime_signature(
        "model", first
    ) != pool._engine_runtime_signature("model", second)


def test_per_model_fixed_kv_toggle_is_part_of_runtime_signature(tmp_path):
    entry = _entry(tmp_path)
    pool = _pool(entry)
    enabled = SimpleNamespace(
        fixed_kv_cache_enabled=True,
        max_context_window=256,
        to_dict=lambda: {
            "fixed_kv_cache_enabled": True,
            "max_context_window": 256,
        },
    )
    disabled = SimpleNamespace(
        fixed_kv_cache_enabled=False,
        max_context_window=256,
        to_dict=lambda: {
            "fixed_kv_cache_enabled": False,
            "max_context_window": 256,
        },
    )

    assert pool._engine_runtime_signature(
        "model", enabled
    ) != pool._engine_runtime_signature("model", disabled)
