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

    base = ModelSettings(thinking_budget_enabled=True, thinking_budget_tokens=128)
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
        fast, "_ext", SimpleNamespace(ane_compile_program=object())
    )
    samples = iter([100, 101, peak, 101, 100, 100])

    async def measure(run, pool, settings, candidate):
        transient = ane_tuning._settings_for_candidate(settings, run.request, candidate)
        assert transient.k2_ane_prefill_enabled == candidate.enabled
        assert base.thinking_budget_enabled and not base.k2_ane_prefill_enabled
        loaded.add("model")
        row = ane_tuning._empty_result(candidate)
        row["processing_tps"] = next(samples)
        row["workloads"] = workloads(row["processing_tps"])
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
    assert base.thinking_budget_enabled and not base.k2_ane_prefill_enabled


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
        fast, "_ext", SimpleNamespace(ane_compile_program=object())
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


def workloads(speed=100):
    return {
        name: {
            "unit": "ms" if "prompt" in name else "tok/s",
            "samples": [10000 / speed if "prompt" in name else speed] * 3,
            "max_ttft_ms": [10000 / speed] * 3,
        }
        for name in ("long_prompt", "short_prompt", "concurrent", "staggered", "cached")
    }


@pytest.mark.parametrize(
    "regression",
    [None, "short_prompt", "concurrent", "staggered", "cached", "latency", "noise"],
)
def test_k2_verdict_requires_serving_improvement(regression):
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="model", backend="k2")
    )
    baseline = dict(
        label="GPU only",
        backend="k2",
        enabled=False,
        processing_tps=100,
        workloads=workloads(),
    )
    candidate = dict(
        label="Dense 33%",
        backend="k2",
        enabled=True,
        processing_tps=120,
        mlp_fraction=1 / 3,
        workloads=workloads(120),
    )
    if regression in candidate["workloads"]:
        candidate["workloads"][regression] = workloads(80)[regression]
    if regression == "latency":
        baseline["workloads"]["concurrent"]["max_ttft_ms"] = [100] * 3
        candidate["workloads"]["concurrent"]["max_ttft_ms"] = [150] * 3
    if regression == "noise":
        candidate["workloads"]["concurrent"]["samples"] = [80, 120, 180]
    run.results = [baseline, candidate]
    result = ane_tuning._select_k2_recommendation(run)
    assert result["enabled"] is (regression is None)
    assert len(run.comparison) == 5
    assert result["comparison_label"] == "Dense 33%"
    assert ane_tuning.run_snapshot(run)["comparison"] == run.comparison


def test_k2_verdict_rejects_missing_concurrency_measurement():
    candidate = {"workloads": workloads(120)}
    del candidate["workloads"]["staggered"]
    with pytest.raises(RuntimeError, match="incomplete"):
        ane_tuning._k2_comparison({"workloads": workloads()}, candidate)


def test_gpu_recheck_can_reject_an_apparent_ane_gain():
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="model", backend="k2")
    )
    run.results = [
        dict(label="GPU", enabled=False, processing_tps=100, workloads=workloads()),
        dict(label="ANE", enabled=True, processing_tps=120, workloads=workloads(120)),
    ]
    assert ane_tuning._select_k2_recommendation(run)["enabled"]
    run.results.append(
        dict(
            label="GPU recheck",
            enabled=False,
            processing_tps=140,
            workloads=workloads(140),
        )
    )
    assert not ane_tuning._select_k2_recommendation(run)["enabled"]


@pytest.mark.parametrize("samples", [[], [1], [float("nan")] * 3, [0] * 3])
def test_k2_verdict_requires_valid_first_token_samples(samples):
    candidate = {"workloads": workloads(120)}
    candidate["workloads"]["staggered"]["max_ttft_ms"] = samples
    with pytest.raises(RuntimeError, match="incomplete|invalid"):
        ane_tuning._k2_comparison({"workloads": workloads()}, candidate)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cache_enabled,cache_hits", [(True, True), (True, False), (False, False)]
)
async def test_k2_evaluation_covers_arrivals_and_proves_cache_reuse(
    monkeypatch, cache_enabled, cache_hits
):
    calls = []

    async def measure(**options):
        calls.append(options)
        return {
            "avg_ttft_ms": 10,
            "aggregate_tps": 100,
            "max_ttft_ms": 15,
            "cached_tokens": [
                256 if cache_hits and not options["skip_cache_store"] else 0
            ]
            * options["batch_size"],
        }

    monkeypatch.setattr(ane_tuning, "_run_batch_test", measure)
    monkeypatch.setattr(
        ane_tuning, "_generate_prompt", lambda _, length, profile: [1] * length
    )
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="model", backend="k2")
    )
    engine = SimpleNamespace(tokenizer=object(), prefix_cache_enabled=cache_enabled)
    candidate = ane_tuning._Candidate("GPU only", False, backend="k2")
    if cache_enabled and not cache_hits:
        with pytest.raises(RuntimeError, match="did not reuse"):
            await ane_tuning._measure_k2_workloads(run, engine, candidate, [10] * 3)
        return
    results = await ane_tuning._measure_k2_workloads(run, engine, candidate, [10] * 3)
    assert {
        "short_prompt",
        "long_prompt",
        "concurrent",
        "staggered",
        "cached",
    } == results.keys()
    assert len(results["concurrent"]["samples"]) == 3
    assert (
        len([call for call in calls if call["staggered"] and call["batch_size"] == 8])
        == 4
    )
    if cache_enabled:
        assert all(all(row) for row in results["cached"]["cached_tokens"])
        assert any(
            call["max_tokens"] == 2 and not call["skip_cache_store"] for call in calls
        )
    else:
        assert results["cached"]["unavailable"] == "cache_disabled"
