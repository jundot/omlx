# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from omlx.admin import ane_tuning
from omlx.model_settings import ModelSettings


@pytest.fixture(autouse=True)
def _clear_runs(monkeypatch):
    ane_tuning._runs.clear()
    monkeypatch.setattr(ane_tuning, "_pin_speed_priority", lambda pool: None)
    monkeypatch.setattr(
        ane_tuning, "_restore_speed_priority", lambda pool, previous: None
    )
    yield
    ane_tuning._runs.clear()


def test_nax_fraction_grid_covers_faster_gpu_balance(monkeypatch):
    import omlx.custom_kernels.nax as nax

    monkeypatch.setattr(nax, "is_nax_available", lambda: True)
    assert ane_tuning._fraction_grid() == [0.15, 0.25, 0.35, 0.45, 0.53]


def test_candidate_settings_are_transient_copy():
    base = ModelSettings()
    request = ane_tuning.ANETuningRequest(model_id="qwen", sequence_length=2048)
    candidate = ane_tuning._Candidate(
        "test", True, 0.25, True, 0.35, True, 0.125, 0.20, 0.10
    )

    tuned = ane_tuning._settings_for_candidate(base, request, candidate)

    assert tuned is not base
    assert tuned.qwen35_ane_prefill_enabled is True
    assert tuned.qwen35_ane_prefill_fraction == 0.25
    assert tuned.qwen35_ane_prefill_gdn_fraction == 0.35
    assert tuned.qwen35_ane_prefill_cpu_enabled is True
    assert tuned.qwen35_ane_prefill_cpu_fraction == 0.125
    assert tuned.qwen35_ane_prefill_cpu_down_fraction == 0.20
    assert tuned.qwen35_ane_prefill_cpu_gdn_fraction == 0.10
    assert base.qwen35_ane_prefill_enabled is False
    assert base.qwen35_ane_prefill_fraction == 0.53


def test_candidate_settings_apply_tuner_boolean_overrides():
    base = ModelSettings(qwen35_ane_prefill_cpu_shared_resource=True)
    request = ane_tuning.ANETuningRequest(
        model_id="qwen",
        allow_cpu=False,
        allow_cpu_gate=False,
        allow_cpu_down=False,
        allow_ane_gdn=False,
        allow_cpu_gdn=False,
        allow_cpu_shared_resource=False,
    )
    candidate = ane_tuning._Candidate(
        "constrained", True, 0.45, True, 0.45, True, 0.14, 0.20, 0.13
    )

    tuned = ane_tuning._settings_for_candidate(base, request, candidate)

    assert tuned.qwen35_ane_prefill_enabled is True
    assert tuned.qwen35_ane_prefill_gdn is False
    assert tuned.qwen35_ane_prefill_cpu_enabled is False
    assert tuned.qwen35_ane_prefill_cpu_fraction == 0.0
    assert tuned.qwen35_ane_prefill_cpu_down_fraction == 0.0
    assert tuned.qwen35_ane_prefill_cpu_gdn_fraction == 0.0
    assert tuned.qwen35_ane_prefill_cpu_shared_resource is False


def test_tuner_overrides_reduce_planned_search_matrix():
    full = ane_tuning.create_run(ane_tuning.ANETuningRequest(model_id="full"))
    constrained = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(
            model_id="constrained",
            allow_cpu=False,
            allow_ane_gdn=False,
        )
    )

    assert constrained.total == 9
    assert constrained.total < full.total


def test_full_model_profile_rebalances_representative_prediction(monkeypatch):
    monkeypatch.setattr(
        ane_tuning, "_fraction_grid", lambda: [0.4, 0.45, 0.5, 0.53, 0.6]
    )
    candidate = ane_tuning._Candidate(
        "predicted", True, 0.5, True, 0.6, True, 0.125, 0.25
    )
    result = {
        "_profile": {
            "mlp": {
                "operations": 192,
                "ane0_eval_ns": 19.03e6 * 192,
                "ane1_eval_ns": 18.97e6 * 192,
                "cpu_completion_ns": 16.33e6 * 192,
                "gpu_completion_ns": 16.20e6 * 192,
            },
            "gdn": {
                "operations": 144,
                "ane0_eval_ns": 11.47e6 * 144,
                "ane1_eval_ns": 11.48e6 * 144,
                "gpu_completion_ns": 8.72e6 * 144,
            },
        }
    }

    refined = ane_tuning._profile_refinement(candidate, result)

    assert refined.mlp_fraction == 0.465
    assert refined.cpu_fraction == 0.135
    assert refined.cpu_down_fraction == 0.25
    assert refined.gdn_fraction == 0.53


def test_full_model_profile_rebalances_three_way_gdn_prediction(monkeypatch):
    monkeypatch.setattr(
        ane_tuning, "_fraction_grid", lambda: [0.4, 0.45, 0.5, 0.53, 0.6]
    )
    candidate = ane_tuning._Candidate(
        "predicted", True, 0.5, True, 0.6, True, 0.0, 0.0, 0.15
    )
    operations = 144
    result = {
        "_profile": {
            "gdn": {
                "operations": operations,
                "ane0_eval_ns": 11.47e6 * operations,
                "ane1_eval_ns": 11.48e6 * operations,
                "cpu_completion_ns": 5.0e6 * operations,
                "gpu_completion_ns": 8.72e6 * operations,
            }
        }
    }

    refined = ane_tuning._profile_refinement(candidate, result)

    assert refined.gdn_fraction == 0.465
    assert refined.cpu_gdn_fraction == 0.25


@pytest.mark.asyncio
async def test_tuner_recommends_best_combined_split(monkeypatch):
    async def measure(run, pool, settings, candidate):
        tps = 100.0 if not candidate.enabled else 125.0
        return {
            "label": candidate.label,
            "enabled": candidate.enabled,
            "mlp_fraction": candidate.mlp_fraction,
            "gdn_enabled": candidate.gdn_enabled,
            "gdn_fraction": candidate.gdn_fraction,
            "cpu_enabled": candidate.cpu_enabled,
            "cpu_fraction": candidate.cpu_fraction,
            "cpu_down_fraction": candidate.cpu_down_fraction,
            "processing_tps": tps,
            "samples": [tps],
        }

    async def calibrate(run, engine, settings):
        return ane_tuning._CalibrationChoice(
            mlp_fraction=0.5,
            cpu_fraction=0.125,
            cpu_down_fraction=0.2,
            gdn_enabled=True,
            gdn_fraction=0.5,
            cpu_enabled=True,
            cpu_threads=8,
            cpu_shared_resource=True,
        )

    monkeypatch.setattr(ane_tuning, "_measure_candidate", measure)
    monkeypatch.setattr(ane_tuning, "_calibrate_components", calibrate)
    async def get_engine(*args, **kwargs):
        return object()

    pool = SimpleNamespace(
        _settings_manager=SimpleNamespace(
            get_settings=lambda model_id: ModelSettings()
        ),
        get_loaded_model_ids=lambda: [],
        get_engine=get_engine,
    )
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="qwen", repeats=1)
    )

    await ane_tuning.run_tuning(run, pool)

    assert run.status == "completed"
    assert run.current == run.total
    assert run.recommendation == {
        "enabled": True,
        "mlp_fraction": 0.5,
        "gdn_enabled": True,
        "gdn_fraction": 0.5,
        "cpu_enabled": True,
        "cpu_fraction": 0.125,
        "cpu_down_fraction": 0.2,
        "cpu_gdn_fraction": None,
        "cpu_threads": 8,
        "cpu_shared_resource": True,
        "processing_tps": 125.0,
        "speedup_percent": 25.0,
        "sequence_length": 2048,
    }


@pytest.mark.asyncio
async def test_tuner_keeps_gpu_for_sub_noise_gain(monkeypatch):
    async def measure(run, pool, settings, candidate):
        tps = 100.5 if candidate.enabled else 100.0
        return {
            "label": candidate.label,
            "enabled": candidate.enabled,
            "mlp_fraction": candidate.mlp_fraction,
            "gdn_enabled": candidate.gdn_enabled,
            "gdn_fraction": candidate.gdn_fraction,
            "cpu_enabled": candidate.cpu_enabled,
            "cpu_fraction": candidate.cpu_fraction,
            "cpu_down_fraction": candidate.cpu_down_fraction,
            "processing_tps": tps,
            "samples": [tps],
        }

    async def calibrate(run, engine, settings):
        return ane_tuning._CalibrationChoice(
            0.5, 0.125, 0.2, True, 0.5, True, 8, True
        )

    monkeypatch.setattr(ane_tuning, "_measure_candidate", measure)
    monkeypatch.setattr(ane_tuning, "_calibrate_components", calibrate)
    async def get_engine(*args, **kwargs):
        return object()

    pool = SimpleNamespace(
        _settings_manager=SimpleNamespace(
            get_settings=lambda model_id: ModelSettings()
        ),
        get_loaded_model_ids=lambda: [],
        get_engine=get_engine,
    )
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="qwen", repeats=1)
    )

    await ane_tuning.run_tuning(run, pool)

    assert run.status == "completed"
    assert run.recommendation["enabled"] is False
    assert run.recommendation["processing_tps"] == 100.0


@pytest.mark.asyncio
async def test_tuner_preserves_partial_matrix_and_failure_reason(monkeypatch):
    async def measure(run, pool, settings, candidate):
        tps = 100.0
        return {
            "label": candidate.label,
            "enabled": candidate.enabled,
            "mlp_fraction": candidate.mlp_fraction,
            "gdn_enabled": candidate.gdn_enabled,
            "gdn_fraction": candidate.gdn_fraction,
            "cpu_enabled": candidate.cpu_enabled,
            "cpu_fraction": candidate.cpu_fraction,
            "cpu_down_fraction": candidate.cpu_down_fraction,
            "processing_tps": tps,
            "samples": [tps],
        }

    async def calibrate(run, engine, settings):
        raise MemoryError("Metal heap exhausted")

    monkeypatch.setattr(ane_tuning, "_measure_candidate", measure)
    monkeypatch.setattr(ane_tuning, "_calibrate_components", calibrate)
    async def get_engine(*args, **kwargs):
        return object()

    pool = SimpleNamespace(
        _settings_manager=SimpleNamespace(
            get_settings=lambda model_id: ModelSettings()
        ),
        get_loaded_model_ids=lambda: [],
        get_engine=get_engine,
    )
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="qwen", repeats=1)
    )

    await ane_tuning.run_tuning(run, pool)
    snapshot = ane_tuning.run_snapshot(run)

    assert run.status == "error"
    assert run.current == 1
    assert len(snapshot["results"]) == 6
    assert [result["state"] for result in snapshot["results"]] == [
        "completed",
        "failed",
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    assert [result["processing_tps"] for result in snapshot["results"]] == [
        100.0,
        None,
        None,
        None,
        None,
        None,
    ]
    assert snapshot["results"][0]["speedup_percent"] == 0.0
    assert snapshot["results"][1]["error"] == "MemoryError: Metal heap exhausted"
    assert snapshot["termination_reason"] == (
        f"Stopped after 1 of {run.total} tests: MemoryError: Metal heap exhausted"
    )
    assert snapshot["recommendation"] is None
