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
    candidate = ane_tuning._Candidate("test", True, 0.25, True, 0.35)

    tuned = ane_tuning._settings_for_candidate(base, request, candidate)

    assert tuned is not base
    assert tuned.qwen35_ane_prefill_enabled is True
    assert tuned.qwen35_ane_prefill_fraction == 0.25
    assert tuned.qwen35_ane_prefill_gdn_fraction == 0.35
    assert base.qwen35_ane_prefill_enabled is False
    assert base.qwen35_ane_prefill_fraction == 0.53


@pytest.mark.asyncio
async def test_tuner_recommends_best_combined_split(monkeypatch):
    monkeypatch.setattr(ane_tuning, "_fraction_grid", lambda: [0.25, 0.5])

    async def measure(run, pool, settings, candidate):
        if not candidate.enabled:
            tps = 100.0
        elif candidate.gdn_enabled:
            tps = 125.0 if candidate.gdn_fraction == 0.5 else 115.0
        else:
            tps = 110.0 if candidate.mlp_fraction == 0.5 else 105.0
        return {
            "label": candidate.label,
            "enabled": candidate.enabled,
            "mlp_fraction": candidate.mlp_fraction,
            "gdn_enabled": candidate.gdn_enabled,
            "gdn_fraction": candidate.gdn_fraction,
            "processing_tps": tps,
            "samples": [tps],
        }

    monkeypatch.setattr(ane_tuning, "_measure_candidate", measure)
    pool = SimpleNamespace(
        _settings_manager=SimpleNamespace(
            get_settings=lambda model_id: ModelSettings()
        ),
        get_loaded_model_ids=lambda: [],
    )
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="qwen", repeats=1)
    )

    await ane_tuning.run_tuning(run, pool)

    assert run.status == "completed"
    assert run.current == 5
    assert run.recommendation == {
        "enabled": True,
        "mlp_fraction": 0.5,
        "gdn_enabled": True,
        "gdn_fraction": 0.5,
        "processing_tps": 125.0,
        "speedup_percent": 25.0,
        "sequence_length": 2048,
    }


@pytest.mark.asyncio
async def test_tuner_keeps_gpu_for_sub_noise_gain(monkeypatch):
    monkeypatch.setattr(ane_tuning, "_fraction_grid", lambda: [0.5])

    async def measure(run, pool, settings, candidate):
        tps = 100.5 if candidate.enabled else 100.0
        return {
            "label": candidate.label,
            "enabled": candidate.enabled,
            "mlp_fraction": candidate.mlp_fraction,
            "gdn_enabled": candidate.gdn_enabled,
            "gdn_fraction": candidate.gdn_fraction,
            "processing_tps": tps,
            "samples": [tps],
        }

    monkeypatch.setattr(ane_tuning, "_measure_candidate", measure)
    pool = SimpleNamespace(
        _settings_manager=SimpleNamespace(
            get_settings=lambda model_id: ModelSettings()
        ),
        get_loaded_model_ids=lambda: [],
    )
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="qwen", repeats=1)
    )

    await ane_tuning.run_tuning(run, pool)

    assert run.status == "completed"
    assert run.recommendation["enabled"] is False
    assert run.recommendation["processing_tps"] == 100.0
