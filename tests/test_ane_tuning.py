# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from omlx.admin import ane_tuning
from omlx.custom_kernels.qwen35_prefill import fast
from omlx.model_settings import ModelSettings

_REAL_BANK_COMPILER_AVAILABLE = fast.qwen35_ane_bank_compiler_available


@pytest.fixture(autouse=True)
def _clear_runs(monkeypatch):
    ane_tuning._runs.clear()
    monkeypatch.setattr(ane_tuning, "_pin_speed_priority", lambda pool: None)
    monkeypatch.setattr(
        ane_tuning, "_restore_speed_priority", lambda pool, previous: None
    )
    # run_tuning preflights the compiler probes; the pipeline tests here drive
    # a mocked measurement stack on runners without the extension, so pretend
    # the compiler exists. Unavailable-path tests re-stub the probes.
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "qwen35_ane_bank_compiler_available", lambda: True)
    yield
    ane_tuning._runs.clear()


def test_nax_fraction_grid_covers_faster_gpu_balance(monkeypatch):
    import omlx.custom_kernels.nax as nax

    monkeypatch.setattr(nax, "is_nax_available", lambda: True)
    assert ane_tuning._fraction_grid() == [0.15, 0.25, 0.35, 0.45, 0.53]


def test_cpu_worker_search_space_is_independent_of_saved_settings():
    assert ane_tuning._cpu_thread_grid() == [6, 8, 10, 12, 14, 16]
    assert ane_tuning._COARSE_SAMPLES == 7
    assert ane_tuning._FINALIST_SAMPLES == 9


def test_candidate_settings_are_transient_copy():
    base = ModelSettings(qwen35_ane_prefill_tail_padding_min_tokens=1500)
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
    assert tuned.qwen35_ane_prefill_tail_padding_min_tokens == 0
    assert base.qwen35_ane_prefill_enabled is False
    assert base.qwen35_ane_prefill_fraction == 0.53
    assert base.qwen35_ane_prefill_tail_padding_min_tokens == 1500


def test_candidate_settings_preserve_single_ane_mode():
    base = ModelSettings(qwen35_ane_prefill_dual_ane=False)
    request = ane_tuning.ANETuningRequest(model_id="qwen")
    candidate = ane_tuning._Candidate(
        "single", True, 0.45, True, 0.45, True, 0.14, 0.20, 0.13
    )

    tuned = ane_tuning._settings_for_candidate(base, request, candidate)

    assert tuned.qwen35_ane_prefill_dual_ane is False
    assert tuned.qwen35_ane_prefill_cpu_enabled is True
    assert tuned.qwen35_ane_prefill_cpu_fraction == 0.14
    assert tuned.qwen35_ane_prefill_cpu_down_fraction == 0.20
    assert tuned.qwen35_ane_prefill_cpu_gdn_fraction == 0.13


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


def test_fused_candidate_uses_per_ane_fraction_and_one_cpu_hidden_share():
    base = ModelSettings(qwen35_ane_prefill_cpu_down_fraction=0.2)
    request = ane_tuning.ANETuningRequest(model_id="qwen")
    candidate = ane_tuning._Candidate(
        label="fused",
        enabled=True,
        mlp_fraction=0.19,
        cpu_enabled=True,
        cpu_fraction=0.14,
        cpu_down_fraction=0.2,
        fused_down=True,
        cpu_threads=12,
    )

    tuned = ane_tuning._settings_for_candidate(base, request, candidate)

    assert tuned.qwen35_ane_prefill_fused_down is True
    assert tuned.qwen35_ane_prefill_fraction == 0.19
    assert tuned.qwen35_ane_prefill_cpu_fraction == 0.14
    assert tuned.qwen35_ane_prefill_cpu_down_fraction == 0.0
    assert tuned.qwen35_ane_prefill_cpu_threads == 12


def test_fused_profile_refinement_balances_aggregate_dual_ane_width():
    candidate = ane_tuning._Candidate(
        label="fused",
        enabled=True,
        mlp_fraction=0.19,
        cpu_enabled=True,
        cpu_fraction=0.14,
        fused_down=True,
    )
    operations = 64
    result = {
        "_profile": {
            "mlp": {
                "operations": operations,
                "ane0_eval_ns": 10e6 * operations,
                "ane1_eval_ns": 10e6 * operations,
                "cpu_completion_ns": 10e6 * operations,
                "gpu_completion_ns": 10e6 * operations,
            }
        }
    }

    refined = ane_tuning._profile_refinement(candidate, result)

    assert refined.mlp_fraction == 0.19
    assert refined.cpu_fraction == 0.14
    assert refined.fused_down is True


def test_profile_refinement_rebalances_mlp_without_cpu_share(monkeypatch):
    monkeypatch.setattr(
        ane_tuning, "_fraction_grid", lambda: [0.4, 0.45, 0.5, 0.53, 0.6]
    )
    candidate = ane_tuning._Candidate("predicted", True, 0.5, False, None)
    operations = 192
    result = {
        "_profile": {
            "mlp": {
                "operations": operations,
                "ane0_eval_ns": 19.0e6 * operations,
                "ane1_eval_ns": 19.0e6 * operations,
                "gpu_completion_ns": 10.0e6 * operations,
            }
        }
    }

    refined = ane_tuning._profile_refinement(candidate, result)

    assert refined.mlp_fraction == 0.35
    assert not refined.cpu_fraction


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


def test_profile_refinement_reads_only_native_profile_keys():
    """Every key the refinement consumes must exist in the native schema.

    Regression guard: profile refinement must stay synchronized with the
    metrics exported by the compiled native extension.
    """
    import inspect

    from omlx.custom_kernels.qwen35_prefill import fast

    source = inspect.getsource(ane_tuning._profile_refinement)
    used = set(re.findall(r'\.get\("([a-z0-9_]+_ns|operations)"', source))
    assert used, "expected the refinement to read profile keys"
    missing = used - set(fast._ANE_PROFILE_KEYS)
    assert not missing, f"refinement reads keys absent from the schema: {missing}"


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
        "fused_down": False,
        "cpu_threads": 8,
        "cpu_shared_resource": True,
        "processing_tps": 125.0,
        "speedup_percent": 25.0,
        "sequence_length": 2048,
        "tail_padding_min_tokens": 1639,
    }


@pytest.mark.asyncio
async def test_fused_tuner_verifies_the_actual_calibrated_worker_counts(monkeypatch):
    measured_threads = []

    async def measure(run, pool, settings, candidate):
        measured_threads.append(candidate.cpu_threads)
        tps = {None: 100.0, 8: 120.0, 12: 130.0}[candidate.cpu_threads]
        return {
            **ane_tuning._empty_result(candidate),
            "processing_tps": tps,
            "samples": [tps],
        }

    async def calibrate(run, engine, settings):
        return ane_tuning._CalibrationChoice(
            mlp_fraction=0.19,
            cpu_fraction=0.14,
            cpu_down_fraction=0.0,
            gdn_enabled=True,
            gdn_fraction=0.45,
            cpu_enabled=True,
            cpu_threads=8,
            cpu_shared_resource=True,
            fused_down=True,
            alternate_mlp_fraction=0.22,
            alternate_cpu_fraction=0.11,
            alternate_cpu_threads=12,
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

    assert measured_threads == [None, 8, 12]
    assert run.recommendation["cpu_threads"] == 12
    assert run.recommendation["mlp_fraction"] == 0.22
    assert run.recommendation["cpu_fraction"] == 0.11


@pytest.mark.asyncio
async def test_fused_tuner_can_spend_runner_up_slot_on_gdn(monkeypatch):
    measured = []

    async def measure(run, pool, settings, candidate):
        measured.append(
            (candidate.cpu_threads, candidate.gdn_fraction, candidate.cpu_gdn_fraction)
        )
        if not candidate.enabled:
            tps = 100.0
        elif candidate.gdn_fraction == 0.53:
            tps = 130.0
        else:
            tps = 120.0
        return {
            **ane_tuning._empty_result(candidate),
            "processing_tps": tps,
            "samples": [tps],
        }

    async def calibrate(run, engine, settings):
        return ane_tuning._CalibrationChoice(
            mlp_fraction=0.19,
            cpu_fraction=0.14,
            cpu_down_fraction=0.0,
            gdn_enabled=True,
            gdn_fraction=0.45,
            cpu_enabled=True,
            cpu_threads=8,
            cpu_shared_resource=True,
            cpu_gdn_fraction=0.125,
            fused_down=True,
            alternate_mlp_fraction=0.19,
            alternate_cpu_fraction=0.14,
            alternate_cpu_threads=8,
            alternate_gdn_fraction=0.53,
            alternate_cpu_gdn_fraction=0.0,
            alternate_reason="GDN topology",
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

    assert measured == [
        (None, None, None),
        (8, 0.45, 0.125),
        (8, 0.53, 0.0),
    ]
    assert run.recommendation["gdn_fraction"] == 0.53
    assert run.recommendation["cpu_gdn_fraction"] == 0.0


@pytest.mark.asyncio
async def test_full_model_gdn_correction_precedes_worker_runner_up(monkeypatch):
    measured = []

    async def measure(run, pool, settings, candidate):
        measured.append(
            (candidate.cpu_threads, candidate.gdn_fraction, candidate.cpu_gdn_fraction)
        )
        tps = 100.0 if not candidate.enabled else 120.0
        if candidate.gdn_fraction == 0.45:
            tps = 130.0
        return {
            **ane_tuning._empty_result(candidate),
            "processing_tps": tps,
            "samples": [tps],
        }

    async def calibrate(run, engine, settings):
        return ane_tuning._CalibrationChoice(
            mlp_fraction=0.19,
            cpu_fraction=0.14,
            cpu_down_fraction=0.0,
            gdn_enabled=True,
            gdn_fraction=0.50,
            cpu_enabled=True,
            cpu_threads=8,
            cpu_shared_resource=True,
            cpu_gdn_fraction=0.10,
            fused_down=True,
            alternate_mlp_fraction=0.19,
            alternate_cpu_fraction=0.14,
            alternate_cpu_threads=12,
            alternate_gdn_fraction=0.50,
            alternate_cpu_gdn_fraction=0.10,
            alternate_reason="CPU worker count",
        )

    def profile_refinement(candidate, result, gdn_floor=None):
        return ane_tuning.replace(
            candidate,
            gdn_fraction=0.45,
            cpu_gdn_fraction=0.125,
        )

    monkeypatch.setattr(ane_tuning, "_measure_candidate", measure)
    monkeypatch.setattr(ane_tuning, "_calibrate_components", calibrate)
    monkeypatch.setattr(ane_tuning, "_profile_refinement", profile_refinement)

    async def get_engine(*args, **kwargs):
        return object()

    pool = SimpleNamespace(
        _settings_manager=SimpleNamespace(
            get_settings=lambda model_id: ModelSettings(
                qwen35_ane_prefill_cpu_threads=14,
                qwen35_ane_prefill_gdn_fraction=0.60,
            )
        ),
        get_loaded_model_ids=lambda: [],
        get_engine=get_engine,
    )
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="qwen", repeats=1)
    )

    await ane_tuning.run_tuning(run, pool)

    assert measured == [
        (None, None, None),
        (8, 0.50, 0.10),
        (8, 0.45, 0.125),
    ]
    assert run.recommendation["gdn_fraction"] == 0.45
    assert run.recommendation["cpu_gdn_fraction"] == 0.125


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


def test_required_mlp_execution_cannot_be_masked_by_gdn():
    trace = _trace(mlp_ops=0, gdn_ops=48)

    assert ane_tuning._ane_execution_observed(trace, require_mlp=True) is False
    assert (
        ane_tuning._ane_execution_observed(
            trace, require_mlp=True, require_gdn=True
        )
        is False
    )


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
    assert result["samples"] == [400.0, 400.0, 400.0]


@pytest.mark.asyncio
async def test_measure_prompt_uses_four_exact_ane_tiles(monkeypatch):
    """Account for stream_generate consuming the final input token."""
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

    assert lengths == [2048 + 1, 2048 * 4 + 1]


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
    # A completed GPU-only baseline survives a later failure as the
    # recommendation: keep ANE off.
    assert snapshot["recommendation"] == {
        "enabled": False,
        "mlp_fraction": None,
        "gdn_enabled": False,
        "gdn_fraction": None,
        "fused_down": False,
        "processing_tps": 100.0,
        "speedup_percent": 0.0,
        "sequence_length": run.request.sequence_length,
        "tail_padding_min_tokens": 0,
    }


@pytest.mark.parametrize(
    ("gpu_tps", "tuned_tps", "expected"),
    [
        (349.4, 527.4, 1357),
        (100.0, 125.0, 1639),
        (100.0, 100.0, 0),
        (100.0, 99.0, 0),
        (None, 125.0, 0),
    ],
)
def test_tail_padding_threshold_uses_current_tuner_throughput(
    gpu_tps, tuned_tps, expected
):
    assert ane_tuning._tail_padding_min_tokens(2048, gpu_tps, tuned_tps) == expected


def test_profile_refinement_rebalances_mlp_without_cpu_share(monkeypatch):
    """cpu_fraction 0 must keep the plain two-way ANE/GPU rebalance."""
    monkeypatch.setattr(
        ane_tuning, "_fraction_grid", lambda: [0.4, 0.45, 0.5, 0.53, 0.6]
    )
    candidate = ane_tuning._Candidate("predicted", True, 0.5, False, None)
    operations = 192
    result = {
        "_profile": {
            "mlp": {
                "operations": operations,
                "ane0_eval_ns": 19.0e6 * operations,
                "ane1_eval_ns": 19.0e6 * operations,
                "gpu_completion_ns": 10.0e6 * operations,
            }
        }
    }

    refined = ane_tuning._profile_refinement(candidate, result)

    assert refined.mlp_fraction == 0.35
    assert not refined.cpu_fraction


def test_min_viable_gdn_fraction_matches_bank_rule():
    """The floor is the smallest fraction whose aligned slice covers z."""
    from types import SimpleNamespace

    from omlx.patches import qwen35_ane_prefill as patch

    gdn = SimpleNamespace(
        in_proj_z=SimpleNamespace(weight=SimpleNamespace(shape=(512, 1))),
        in_proj_qkv=SimpleNamespace(weight=SimpleNamespace(shape=(1536, 1))),
    )
    floor = ane_tuning._min_viable_gdn_fraction(patch, gdn, 128)
    assert floor == 0.25
    total = 2048
    assert (int(total * floor) // 128) * 128 >= 512
    assert (int(total * 0.15) // 128) * 128 < 512

    # z larger than the whole projection can never engage
    impossible = SimpleNamespace(
        in_proj_z=SimpleNamespace(weight=SimpleNamespace(shape=(2050, 1))),
        in_proj_qkv=SimpleNamespace(weight=SimpleNamespace(shape=(10, 1))),
    )
    assert ane_tuning._min_viable_gdn_fraction(patch, impossible, 128) is None


def test_profile_refinement_locks_gdn_to_the_z_floor(monkeypatch):
    """The refinement must use the sole recurrent-safe ANE width."""
    monkeypatch.setattr(
        ane_tuning, "_fraction_grid", lambda: [0.15, 0.25, 0.35, 0.45, 0.53]
    )
    candidate = ane_tuning._Candidate("predicted", True, 0.5, True, 0.45)
    operations = 144
    result = {
        "_profile": {
            "gdn": {
                "operations": operations,
                # ANE much slower than GPU -> the balancer wants a tiny fraction
                "ane0_eval_ns": 40.0e6 * operations,
                "ane1_eval_ns": 40.0e6 * operations,
                "gpu_completion_ns": 6.0e6 * operations,
            }
        }
    }

    unclamped = ane_tuning._profile_refinement(candidate, result)
    assert unclamped.gdn_fraction < 0.4  # sanity: the pull downward is real

    clamped = ane_tuning._profile_refinement(candidate, result, gdn_floor=0.42)
    assert clamped.gdn_fraction == 0.42


def test_settings_for_candidate_disables_dflash():
    """Tuner staging must load the plain LM engine, never DFlash (#2914)."""
    from omlx.model_settings import ModelSettings

    base = ModelSettings(dflash_enabled=True)
    request = ane_tuning.ANETuningRequest(model_id="m")
    candidate = ane_tuning._Candidate("GPU only", False)

    settings = ane_tuning._settings_for_candidate(base, request, candidate)

    assert settings.dflash_enabled is False
    assert base.dflash_enabled is True


def test_settings_for_candidate_disables_specprefill():
    """Tuner measurement must prefill the full prompt, never sparse.

    SpecPrefill compresses the measurement prompt below the compiled ANE
    tile width, so the ANE never executes and the idle guard aborts the run.
    """
    from omlx.model_settings import ModelSettings

    base = ModelSettings(
        specprefill_enabled=True, specprefill_draft_model="draft-model"
    )
    request = ane_tuning.ANETuningRequest(model_id="m")
    candidate = ane_tuning._Candidate("GPU only", False)

    settings = ane_tuning._settings_for_candidate(base, request, candidate)

    assert settings.specprefill_enabled is False
    assert base.specprefill_enabled is True


def test_unavailable_bank_compiler_yields_gpu_only_verdict(monkeypatch):
    """#3044: a machine without the private ANE runtime/bank compiler must get
    a completed GPU-only verdict up front — not a failed run after the
    bank-split ladder, and with no model loads or unloads along the way."""
    import asyncio
    from unittest.mock import MagicMock

    from omlx.custom_kernels.qwen35_prefill import fast

    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: False)
    monkeypatch.setattr(fast, "qwen35_ane_bank_compiler_available", lambda: False)

    run = ane_tuning.create_run(ane_tuning.ANETuningRequest(model_id="m"))
    pool = MagicMock()
    asyncio.run(ane_tuning.run_tuning(run, pool))

    assert run.status == "completed"
    assert run.phase == "completed"
    assert run.error_message == ""
    assert run.recommendation is not None
    assert run.recommendation["enabled"] is False
    assert run.recommendation["gdn_enabled"] is False
    assert all(result["state"] == "skipped" for result in run.results)
    assert "ANE prefill is not usable here" in run.message
    # The verdict must not have touched the engine pool at all: the model
    # that was serving before the tuner ran stays exactly as it was.
    assert pool.mock_calls == []


def test_bank_compiler_available_matches_serving_probe(monkeypatch):
    """The tuner guard and qwen35_ane_compile_linear_bank's own gate must
    agree: a False probe is exactly the condition under which the compile
    call raises."""
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: False)
    monkeypatch.setattr(
        fast, "qwen35_ane_bank_compiler_available", _REAL_BANK_COMPILER_AVAILABLE
    )
    assert fast.qwen35_ane_bank_compiler_available() is False
    with pytest.raises(RuntimeError, match="procedure-bank compiler"):
        fast.qwen35_ane_compile_linear_bank([], 2048, 0)
