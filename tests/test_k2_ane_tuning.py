# SPDX-License-Identifier: Apache-2.0
"""K2 tuner isolation, measured recommendations, and cancellation."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from omlx.admin import ane_tuning
from omlx.model_settings import ModelSettings


@pytest.mark.asyncio
@pytest.mark.parametrize("peak", [102, 108])
async def test_k2_tuner_preserves_settings_and_unloads(monkeypatch, peak, tmp_path):
    from omlx.custom_kernels.qwen35_prefill import fast

    base = ModelSettings(uno_enabled=True, uno_adapter_model="adapter")
    loaded = {"model"}

    async def unload(name):
        loaded.discard(name)

    (tmp_path / "config.json").write_text(
        json.dumps(dict(num_hidden_layers=3, num_experts=4, mlp_only_layers=[0]))
    )
    pool = SimpleNamespace(
        get_entry=lambda _: SimpleNamespace(model_path=tmp_path),
        _settings_manager=SimpleNamespace(get_settings=lambda _: base),
        get_loaded_model_ids=lambda: list(loaded),
        _unload_engine=unload,
    )
    monkeypatch.setattr(ane_tuning, "_pin_speed_priority", lambda _: "old")
    restored = []
    monkeypatch.setattr(
        ane_tuning, "_restore_speed_priority", lambda _, value: restored.append(value)
    )
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(
        fast, "_ext", SimpleNamespace(ane_compile_program_bank=object())
    )
    samples = iter([100, 101, peak, 101, 100])

    async def measure(run, pool, settings, candidate):
        transient = ane_tuning._settings_for_candidate(settings, run.request, candidate)
        assert not transient.uno_enabled
        assert transient.k2_ane_prefill_enabled == candidate.enabled
        assert base.uno_enabled and not base.k2_ane_prefill_enabled
        loaded.add("model")
        row = ane_tuning._empty_result(candidate)
        row["processing_tps"] = next(samples)
        return row

    monkeypatch.setattr(ane_tuning, "_measure_candidate", measure)
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="model", backend="k2")
    )
    await ane_tuning.run_tuning(run, pool)
    assert run.status == "completed"
    assert run.recommendation["backend"] == "k2"
    assert run.recommendation["enabled"] == (peak > 103)
    assert not loaded and restored == ["old"]
    assert base.uno_enabled and not base.k2_ane_prefill_enabled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("compile failed"), asyncio.CancelledError()]
)
async def test_k2_tuner_failure_does_not_apply_partial_results(
    monkeypatch, error, tmp_path
):
    from omlx.custom_kernels.qwen35_prefill import fast

    unload = AsyncMock()
    (tmp_path / "config.json").write_text(
        json.dumps(dict(num_hidden_layers=3, num_experts=4, mlp_only_layers=[0]))
    )
    pool = SimpleNamespace(
        get_entry=lambda _: SimpleNamespace(model_path=tmp_path),
        _settings_manager=SimpleNamespace(get_settings=lambda _: ModelSettings()),
        get_loaded_model_ids=lambda: ["model"],
        _unload_engine=unload,
    )
    monkeypatch.setattr(ane_tuning, "_pin_speed_priority", lambda _: None)
    monkeypatch.setattr(ane_tuning, "_restore_speed_priority", lambda *_: None)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(
        fast, "_ext", SimpleNamespace(ane_compile_program_bank=object())
    )
    monkeypatch.setattr(ane_tuning, "_measure_candidate", AsyncMock(side_effect=error))
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="model", backend="k2")
    )
    await ane_tuning.run_tuning(run, pool)
    assert run.status == (
        "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
    )
    assert run.recommendation is None
    assert unload.await_count == 2


def test_tuner_candidates_follow_dense_and_shared_geometry():
    dense = ane_tuning._k2_candidates(dict(num_hidden_layers=64))
    sparse = ane_tuning._k2_candidates(dict(num_hidden_layers=61, num_experts=192))
    assert len(dense) == 3
    assert len(sparse) == 3
    assert all(row.shared_fraction > 0 for row in sparse[1:])
