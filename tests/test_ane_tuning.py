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


def test_gdn_override_disables_gdn_without_mutating_base():
    base = ModelSettings(qwen35_ane_prefill_gdn=True)
    request = ane_tuning.ANETuningRequest(model_id="qwen", allow_ane_gdn=False)
    candidate = ane_tuning._Candidate("test", True, 0.45, True, 0.45)

    tuned = ane_tuning._settings_for_candidate(base, request, candidate)

    assert tuned.qwen35_ane_prefill_gdn is False
    assert base.qwen35_ane_prefill_gdn is True
    run = ane_tuning.create_run(request)
    assert run.total == 3 + len(run.fractions)


def test_full_model_profile_rebalances_representative_prediction(monkeypatch):
    monkeypatch.setattr(
        ane_tuning, "_fraction_grid", lambda: [0.4, 0.45, 0.5, 0.53, 0.6]
    )
    candidate = ane_tuning._Candidate("predicted", True, 0.5, True, 0.6)
    result = {
        "_profile": {
            "mlp": {
                "operations": 192,
                "ane0_eval_ns": 19.03e6 * 192,
                "ane1_eval_ns": 18.97e6 * 192,
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
    assert refined.gdn_fraction == 0.53


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
            "processing_tps": tps,
            "samples": [tps],
        }

    async def calibrate(run, engine, settings):
        return ane_tuning._CalibrationChoice(
            mlp_fraction=0.5,
            gdn_enabled=True,
            gdn_fraction=0.5,
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
            "processing_tps": tps,
            "samples": [tps],
        }

    async def calibrate(run, engine, settings):
        return ane_tuning._CalibrationChoice(0.5, True, 0.5)

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


def _trace(mlp_ops: int = 0, gdn_ops: int = 0, profiling: bool = True) -> dict:
    return {
        "profiling_available": profiling,
        "sequence_length": 8192,
        "categories": {
            "mlp": {"operations": mlp_ops},
            "gdn": {"operations": gdn_ops},
        },
    }


def test_ane_execution_observed_distinguishes_compiled_from_ran():
    # Programs compiled but no operation ran: the fixed-shape check failed.
    assert ane_tuning._ane_execution_observed(_trace(mlp_ops=0, gdn_ops=0)) is False
    assert ane_tuning._ane_execution_observed(_trace(mlp_ops=126)) is True
    assert ane_tuning._ane_execution_observed(_trace(gdn_ops=48)) is True


def test_ane_execution_is_unknown_without_the_profiler():
    """Zero counters prove nothing when the profiler never ran.

    qwen35_ane_profile_set_enabled() can return False, and the import is
    wrapped in a bare except, so the counters are zero regardless of what the
    hardware did. Treating that as an idle ANE would reject working candidates.
    """
    assert ane_tuning._ane_execution_observed(None) is None
    assert ane_tuning._ane_execution_observed({}) is None
    assert ane_tuning._ane_execution_observed(_trace(profiling=False)) is None
    assert (
        ane_tuning._ane_execution_observed(_trace(mlp_ops=126, profiling=False)) is None
    )


def test_prefill_step_size_ignores_the_qwen35_floor():
    """The qwen35 floor is zeroed on any ANE-aligned engine, so the config
    value alone is the right hint and the floor must not inflate it."""
    engine = SimpleNamespace(
        _scheduler_config=SimpleNamespace(prefill_step_size=2048),
        _engine=SimpleNamespace(
            engine=SimpleNamespace(
                scheduler=SimpleNamespace(_qwen35_prefill_floor=4096)
            )
        ),
    )
    assert ane_tuning._prefill_step_size(engine) == 2048


def test_prefill_step_size_reads_scheduler_config():
    engine = SimpleNamespace(_scheduler_config=SimpleNamespace(prefill_step_size=4096))
    assert ane_tuning._prefill_step_size(engine) == 4096
    assert ane_tuning._prefill_step_size(SimpleNamespace()) is None
    bad = SimpleNamespace(_scheduler_config=SimpleNamespace(prefill_step_size="nope"))
    assert ane_tuning._prefill_step_size(bad) is None


def _measure_env(monkeypatch, *, trace):
    """Stub out everything _measure_candidate needs except the guard itself."""

    class _Engine:
        tokenizer = object()
        _scheduler_config = SimpleNamespace(prefill_step_size=2048)

        async def stream_generate(self, **kwargs):
            if False:  # pragma: no cover - never yields
                yield None

    class _Pool:
        async def get_engine(self, model_id, **kwargs):
            return _Engine()

    monkeypatch.setattr(ane_tuning, "_ane_is_active", lambda engine: True)
    monkeypatch.setattr(
        ane_tuning, "_generate_prompt", lambda tok, length, profile: [0] * length
    )

    async def _fake_run_single_test(**kwargs):
        return {"processing_tps": 400.0, "ane_trace": trace}

    monkeypatch.setattr(ane_tuning, "_run_single_test", _fake_run_single_test)
    return _Pool()


@pytest.mark.asyncio
async def test_measure_rejects_candidate_whose_ane_never_executed(monkeypatch):
    """A compiled-but-idle ANE must not be reported as a measured result.

    Regression guard: with sequence_length mismatched to the scheduler's
    prefill chunk size, every chunk fails the fixed-shape check, so the
    candidate really measures GPU-only plus the cost of compiling and pinning
    unused programs. Ranking that against real candidates is misleading.
    """
    pool = _measure_env(monkeypatch, trace=_trace(mlp_ops=0))
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="qwen", sequence_length=8192, repeats=1)
    )
    candidate = ane_tuning._Candidate("MLP 35%", True, 0.35, False, None)

    with pytest.raises(RuntimeError) as excinfo:
        await ane_tuning._measure_candidate(run, pool, ModelSettings(), candidate)

    message = str(excinfo.value)
    assert "never executed" in message
    # The message must be actionable: name the chunk size to use.
    assert "sequence_length=2048" in message


@pytest.mark.asyncio
async def test_measure_accepts_candidate_whose_ane_executed(monkeypatch):
    pool = _measure_env(monkeypatch, trace=_trace(mlp_ops=126))
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="qwen", sequence_length=2048, repeats=1)
    )
    candidate = ane_tuning._Candidate("MLP 35%", True, 0.35, False, None)

    result = await ane_tuning._measure_candidate(run, pool, ModelSettings(), candidate)

    assert result["processing_tps"] == 400.0


@pytest.mark.asyncio
async def test_measure_prompt_leaves_a_non_ane_tail_chunk(monkeypatch):
    """sequence_length * 2 keeps a tail chunk in the measured prefill.

    stream_generate prefills tokens[:-1]; the previous * 2 + 1 made that
    exactly two full ANE-shaped blocks with no GPU tail, which overstated
    the ANE gain in every v0.6.2 tuner result.
    """
    pool = _measure_env(monkeypatch, trace=_trace(mlp_ops=126))
    lengths = []
    monkeypatch.setattr(
        ane_tuning,
        "_generate_prompt",
        lambda tok, length, profile: lengths.append(length) or [0] * length,
    )
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="qwen", sequence_length=2048, repeats=1)
    )
    candidate = ane_tuning._Candidate("MLP 35%", True, 0.35, False, None)

    await ane_tuning._measure_candidate(run, pool, ModelSettings(), candidate)

    assert lengths == [2048 + 1, 2048 * 2]


@pytest.mark.asyncio
async def test_gpu_only_candidate_is_never_rejected_for_idle_ane(monkeypatch):
    """The GPU-only baseline has no ANE trace by design."""
    pool = _measure_env(monkeypatch, trace=None)
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="qwen", sequence_length=8192, repeats=1)
    )
    candidate = ane_tuning._Candidate("GPU only", False)

    result = await ane_tuning._measure_candidate(run, pool, ModelSettings(), candidate)

    assert result["enabled"] is False
    assert result["processing_tps"] == 400.0


@pytest.mark.asyncio
async def test_candidate_is_kept_when_the_profiler_is_unavailable(monkeypatch):
    """Without the profiler the guard must not fire: it cannot tell either way."""
    pool = _measure_env(monkeypatch, trace=_trace(mlp_ops=0, profiling=False))
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="qwen", sequence_length=8192, repeats=1)
    )
    candidate = ane_tuning._Candidate("MLP 35%", True, 0.35, False, None)

    result = await ane_tuning._measure_candidate(run, pool, ModelSettings(), candidate)

    assert result["processing_tps"] == 400.0


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
    assert len(snapshot["results"]) == 5
    assert [result["state"] for result in snapshot["results"]] == [
        "completed",
        "failed",
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
    ]
    assert snapshot["results"][0]["speedup_percent"] == 0.0
    assert snapshot["results"][1]["error"] == "MemoryError: Metal heap exhausted"
    assert snapshot["termination_reason"] == (
        f"Stopped after 1 of {run.total} tests: MemoryError: Metal heap exhausted"
    )
    assert snapshot["recommendation"] is None
