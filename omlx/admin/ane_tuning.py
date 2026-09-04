# SPDX-License-Identifier: Apache-2.0
"""Hardware-local, profile-guided Qwen ANE/CPU/GPU workload tuning.

Exploratory points run against one representative real MLP and GDN layer.
Their heterogeneous ANE widths are packed into a small temporary procedure
bank, so only the predicted winner is eagerly compiled across the full model.
Persisted model settings are never changed by a tuning run.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel, field_validator

from .benchmark import (
    BenchmarkContextProfile,
    _generate_prompt,
    _pin_speed_priority,
    _restore_speed_priority,
    _run_single_test,
)

logger = logging.getLogger(__name__)

_runs: dict[str, ANETuningRun] = {}
_GPU_SLOT = 0
_GATE_SLOT = 1
_DOWN_SLOT = 2
_GDN_SLOT = 3
_VERIFY_SLOT = 4
_REFINE_SLOT = 5


class ANETuningRequest(BaseModel):
    model_id: str
    sequence_length: int = 2048
    repeats: int = 2
    allow_cpu: bool = True
    allow_cpu_gate: bool = True
    allow_cpu_down: bool = True
    allow_ane_gdn: bool = True
    allow_cpu_gdn: bool = True
    allow_cpu_shared_resource: bool = True

    @field_validator("sequence_length")
    @classmethod
    def validate_sequence_length(cls, value: int) -> int:
        if value < 1024 or value % 64:
            raise ValueError("sequence_length must be a multiple of 64 >= 1024")
        return value

    @field_validator("repeats")
    @classmethod
    def validate_repeats(cls, value: int) -> int:
        if value < 1 or value > 5:
            raise ValueError("repeats must be between 1 and 5")
        return value


@dataclass(frozen=True)
class _Candidate:
    label: str
    enabled: bool
    mlp_fraction: float | None = None
    gdn_enabled: bool = False
    gdn_fraction: float | None = None
    cpu_enabled: bool = False
    cpu_fraction: float | None = None
    cpu_down_fraction: float | None = None
    cpu_gdn_fraction: float | None = None
    fused_down: bool = False
    cpu_threads: int | None = None
    stage: str = "verification"


@dataclass(frozen=True)
class _CalibrationChoice:
    mlp_fraction: float
    cpu_fraction: float
    cpu_down_fraction: float
    gdn_enabled: bool
    gdn_fraction: float | None
    cpu_enabled: bool
    cpu_threads: int
    cpu_shared_resource: bool
    cpu_gdn_fraction: float = 0.0
    fused_down: bool = False
    alternate_mlp_fraction: float | None = None
    alternate_cpu_fraction: float | None = None
    alternate_cpu_threads: int | None = None
    alternate_gdn_fraction: float | None = None
    alternate_cpu_gdn_fraction: float | None = None
    alternate_reason: str | None = None


@dataclass
class ANETuningRun:
    tuning_id: str
    request: ANETuningRequest
    status: str = "running"
    phase: str = "preparing"
    message: str = "Preparing tuner…"
    current: int = 0
    total: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    fractions: list[float] = field(default_factory=list)
    recommendation: dict[str, Any] | None = None
    error_message: str = ""
    termination_reason: str = ""
    # Smallest gdn_fraction whose aligned ANE slice covers the z projection
    # on the calibrated checkpoint; None when GDN is absent (issue #2899).
    gdn_floor: float | None = None
    task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.time)


def _fraction_grid() -> list[float]:
    """ANE widths worth compiling into the representative calibration bank."""
    try:
        from ..custom_kernels.nax import is_nax_available

        if is_nax_available():
            return [0.15, 0.25, 0.35, 0.45, 0.53]
    except Exception:
        pass
    return [0.40, 0.45, 0.50, 0.53, 0.60]


def _cpu_fraction_grid() -> list[float]:
    return [0.0, 0.05, 0.10, 0.125, 0.135, 0.15, 0.20, 0.25]


def _cpu_down_fraction_grid() -> list[float]:
    return [0.0, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]


def _cpu_gdn_fraction_grid() -> list[float]:
    return [0.0, 0.05, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25, 0.35]


def _fused_fraction_grid() -> list[float]:
    """Per-ANE hidden-channel shares for fused SwiGLU/down."""
    return [0.10, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31]


def _fused_cpu_fraction_grid() -> list[float]:
    return [0.0, 0.08, 0.11, 0.14, 0.17, 0.20]


_CALIBRATION_CPU_THREADS = 8
_COARSE_SAMPLES = 7
_FINALIST_SAMPLES = 9


def _cpu_thread_grid() -> list[int]:
    # Deliberately independent of saved model settings: identical hardware and
    # tuner overrides must produce the same search space on every run.
    return [6, 8, 10, 12, 14, 16]


def _gdn_fraction_grid() -> list[float]:
    return [0.35, 0.40, 0.45, 0.50, 0.53, 0.56, 0.60]


def _planned_rows() -> list[_Candidate]:
    return [
        _Candidate("GPU only", False),
        _Candidate("MLP topology calibration", True, stage="calibration"),
        _Candidate("CPU worker calibration", True, stage="calibration"),
        _Candidate("GDN calibration", True, stage="calibration"),
        _Candidate("Predicted optimum", True),
        _Candidate("Full-model uncertainty runner-up", True),
    ]


def create_run(request: ANETuningRequest) -> ANETuningRun:
    fractions = _fraction_grid()
    planned = _planned_rows()
    run = ANETuningRun(
        tuning_id=str(uuid.uuid4()),
        request=request,
        fractions=fractions,
        results=[_empty_result(candidate) for candidate in planned],
    )
    # This is an upper bound until the loaded checkpoint tells us whether CPU
    # sharing and GDN are eligible. The live snapshot is reduced afterwards.
    cpu_gate_points = (
        len(_cpu_fraction_grid())
        if request.allow_cpu and request.allow_cpu_gate
        else 1
    )
    cpu_down_points = (
        len(_cpu_down_fraction_grid())
        if request.allow_cpu and request.allow_cpu_down
        else 1
    )
    gdn_points = 0
    if request.allow_ane_gdn:
        cpu_gdn_points = (
            len(_cpu_gdn_fraction_grid())
            if request.allow_cpu and request.allow_cpu_gdn
            else 1
        )
        gdn_points = len(fractions) * cpu_gdn_points
    run.total = 3 + len(fractions) * cpu_gate_points + cpu_down_points + gdn_points
    _runs[run.tuning_id] = run
    return run


def get_run(tuning_id: str) -> ANETuningRun | None:
    return _runs.get(tuning_id)


def get_active_run() -> ANETuningRun | None:
    return next((run for run in _runs.values() if run.status == "running"), None)


def cleanup_old_runs(max_age_seconds: float = 3600.0) -> None:
    cutoff = time.time() - max_age_seconds
    for tuning_id, run in list(_runs.items()):
        if run.status != "running" and run.created_at < cutoff:
            _runs.pop(tuning_id, None)


def run_snapshot(run: ANETuningRun) -> dict[str, Any]:
    return {
        "tuning_id": run.tuning_id,
        "model_id": run.request.model_id,
        "status": run.status,
        "phase": run.phase,
        "message": run.message,
        "current": run.current,
        "total": run.total,
        "results": [
            {
                key: value
                for key, value in result.items()
                if not key.startswith("_")
            }
            for result in run.results
        ],
        "recommendation": run.recommendation,
        "error": run.error_message or None,
        "termination_reason": run.termination_reason or None,
    }


def _empty_result(candidate: _Candidate) -> dict[str, Any]:
    return {
        "label": candidate.label,
        "detail": None,
        "stage": candidate.stage,
        "enabled": candidate.enabled,
        "mlp_fraction": candidate.mlp_fraction,
        "gdn_enabled": candidate.gdn_enabled,
        "gdn_fraction": candidate.gdn_fraction,
        "cpu_enabled": candidate.cpu_enabled,
        "cpu_fraction": candidate.cpu_fraction,
        "cpu_down_fraction": candidate.cpu_down_fraction,
        "cpu_gdn_fraction": candidate.cpu_gdn_fraction,
        "fused_down": candidate.fused_down,
        "cpu_threads": candidate.cpu_threads,
        "state": "pending",
        "processing_tps": None,
        "latency_ms": None,
        "samples": [],
        "speedup_percent": None,
        "error": None,
    }


def _exception_reason(exc: BaseException) -> str:
    detail = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {detail}" if detail else name


def _refresh_speedups(run: ANETuningRun) -> None:
    baseline = run.results[_GPU_SLOT].get("processing_tps") if run.results else None
    if baseline is None or baseline <= 0:
        return
    for result in run.results:
        processing_tps = result.get("processing_tps")
        if processing_tps is None:
            continue
        result["speedup_percent"] = round(
            (processing_tps / baseline - 1.0) * 100.0, 2
        )


def _tail_padding_min_tokens(
    sequence_length: int,
    gpu_tps: float | None,
    tuned_tps: float | None,
) -> int:
    """Return the first tail length whose GPU time exceeds one hybrid tile.

    For positive throughput values, a GPU tail costs ``rows / gpu_tps`` and a
    padded fixed-shape hybrid call costs ``sequence_length / tuned_tps``. The
    strict integer crossover is therefore floor(S * G / H) + 1. Zero disables
    padding when tuning found no gain or no partial tile can be profitable.
    """
    if (
        sequence_length <= 1
        or gpu_tps is None
        or tuned_tps is None
        or gpu_tps <= 0
        or tuned_tps <= gpu_tps
    ):
        return 0
    threshold = int(sequence_length * gpu_tps / tuned_tps) + 1
    return threshold if 0 < threshold < sequence_length else 0


def _set_phase_running(run: ANETuningRun, slot: int, message: str) -> None:
    run.phase = "calibrating"
    run.message = message
    run.results[slot]["state"] = "running"


def _preview_phase(
    run: ANETuningRun,
    slot: int,
    *,
    detail: str,
    latency_ms: float,
    **values: Any,
) -> None:
    """Publish a provisional calibration leader without completing its row."""
    run.results[slot].update(
        {
            "detail": detail,
            "latency_ms": round(latency_ms, 3),
            "state": "running",
            **values,
        }
    )


def _complete_phase(
    run: ANETuningRun,
    slot: int,
    *,
    detail: str,
    latency_ms: float | None,
    mlp_fraction: float | None = None,
    gdn_enabled: bool = False,
    gdn_fraction: float | None = None,
    cpu_enabled: bool = False,
    cpu_fraction: float | None = None,
    cpu_down_fraction: float | None = None,
    cpu_gdn_fraction: float | None = None,
    fused_down: bool = False,
    cpu_threads: int | None = None,
) -> None:
    result = run.results[slot]
    result.update(
        {
            "detail": detail,
            "state": "completed",
            "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
            "mlp_fraction": mlp_fraction,
            "gdn_enabled": gdn_enabled,
            "gdn_fraction": gdn_fraction,
            "cpu_enabled": cpu_enabled,
            "cpu_fraction": cpu_fraction,
            "cpu_down_fraction": cpu_down_fraction,
            "cpu_gdn_fraction": cpu_gdn_fraction,
            "fused_down": fused_down,
            "cpu_threads": cpu_threads,
        }
    )


async def _measure_result_slot(
    run: ANETuningRun,
    result_index: int,
    engine_pool: Any,
    base_settings: Any,
    candidate: _Candidate,
) -> None:
    slot = _empty_result(candidate)
    slot["state"] = "running"
    run.results[result_index] = slot
    run.phase = "measuring"
    run.message = f"Testing {candidate.label}…"
    try:
        result = await _measure_candidate(run, engine_pool, base_settings, candidate)
    except asyncio.CancelledError:
        slot["state"] = "cancelled"
        slot["error"] = "Cancelled by user"
        raise
    except Exception as exc:
        slot["state"] = "failed"
        slot["error"] = _exception_reason(exc)
        raise
    else:
        result["state"] = "completed"
        result["stage"] = candidate.stage
        result["detail"] = None
        result["latency_ms"] = None
        result["speedup_percent"] = None
        result["error"] = None
        run.results[result_index] = result
        run.current += 1
        _refresh_speedups(run)


def _settings_for_candidate(base: Any, request: ANETuningRequest, candidate: _Candidate):
    settings = replace(base)
    settings.qwen35_ane_prefill_enabled = candidate.enabled
    settings.qwen35_ane_prefill_fused_down = bool(
        candidate.enabled and candidate.fused_down
    )
    settings.qwen35_ane_prefill_sequence_length = request.sequence_length
    # Saved calibration must not alter the search. The winning run computes a
    # fresh crossover from this run's GPU and hybrid throughput measurements.
    settings.qwen35_ane_prefill_tail_padding_min_tokens = 0
    if candidate.mlp_fraction is not None:
        settings.qwen35_ane_prefill_fraction = candidate.mlp_fraction
    settings.qwen35_ane_prefill_gdn = bool(
        candidate.gdn_enabled and request.allow_ane_gdn
    )
    if candidate.gdn_fraction is not None and request.allow_ane_gdn:
        settings.qwen35_ane_prefill_gdn_fraction = candidate.gdn_fraction
    settings.qwen35_ane_prefill_cpu_enabled = bool(
        candidate.cpu_enabled and request.allow_cpu
    )
    if (
        candidate.cpu_fraction is not None
        and request.allow_cpu
        and request.allow_cpu_gate
    ):
        settings.qwen35_ane_prefill_cpu_fraction = candidate.cpu_fraction
    else:
        settings.qwen35_ane_prefill_cpu_fraction = 0.0
    if (
        candidate.cpu_down_fraction is not None
        and request.allow_cpu
        and request.allow_cpu_down
    ):
        settings.qwen35_ane_prefill_cpu_down_fraction = candidate.cpu_down_fraction
    else:
        settings.qwen35_ane_prefill_cpu_down_fraction = 0.0
    if candidate.fused_down:
        settings.qwen35_ane_prefill_cpu_down_fraction = 0.0
    if (
        candidate.cpu_gdn_fraction is not None
        and request.allow_cpu
        and request.allow_ane_gdn
        and request.allow_cpu_gdn
    ):
        settings.qwen35_ane_prefill_cpu_gdn_fraction = candidate.cpu_gdn_fraction
    else:
        settings.qwen35_ane_prefill_cpu_gdn_fraction = 0.0
    settings.qwen35_ane_prefill_cpu_shared_resource = bool(
        request.allow_cpu
        and request.allow_cpu_shared_resource
        and getattr(base, "qwen35_ane_prefill_cpu_shared_resource", True)
    )
    if candidate.cpu_threads is not None:
        settings.qwen35_ane_prefill_cpu_threads = candidate.cpu_threads
    settings.qwen35_ane_prefill_dual_ane = bool(
        getattr(base, "qwen35_ane_prefill_dual_ane", True)
    )
    # The tuner reaches into the engine for the raw model and compares prompt
    # throughput across slots, so it must stage the plain LM engine: a DFlash
    # engine exposes no _model and would skew every measurement (issue #2914).
    if hasattr(settings, "dflash_enabled"):
        settings.dflash_enabled = False
    # SpecPrefill compresses the measurement prompt to a sparse subset before
    # prefill, so delivered chunks run narrower than the compiled ANE tile
    # width: the ANE never executes and the idle guard aborts the run. Tune
    # against the full dense prefill the ANE path is compiled for.
    if hasattr(settings, "specprefill_enabled"):
        settings.specprefill_enabled = False
    return settings


def _prefill_step_size(engine: Any) -> int | None:
    """The configured prefill step size, if determinable.

    The scheduler keeps its normal chunk width and the ANE tiles wider
    chunks internally, so sequence_length must not exceed the delivered
    width. The delivered chunk can be cut below this value by the cache
    block boundary or the memory guard, which is why callers point at
    chunk_tokens in the serve log as the authority.
    """
    config = getattr(engine, "_scheduler_config", None)
    try:
        size = int(getattr(config, "prefill_step_size", 0) or 0)
    except (TypeError, ValueError):
        size = 0
    return size or None


def _ane_execution_observed(
    trace: dict[str, Any] | None,
    *,
    require_mlp: bool = False,
    require_gdn: bool = False,
) -> bool | None:
    """Whether the ANE ran ops, or None when that cannot be determined.

    The operation counters come from the ANE profiler. When the profiler is
    unavailable they are all zero no matter what the hardware did, so that
    case must be reported as unknown rather than as an idle ANE.
    """
    if not trace or not trace.get("profiling_available"):
        return None
    categories = trace.get("categories") or {}
    if require_mlp and int((categories.get("mlp") or {}).get("operations", 0) or 0) <= 0:
        return False
    if require_gdn and int((categories.get("gdn") or {}).get("operations", 0) or 0) <= 0:
        return False
    return any(
        int(values.get("operations", 0) or 0) > 0
        for values in categories.values()
    )


def _ane_is_active(engine: Any) -> bool:
    model = getattr(engine, "_model", None)
    if model is None:
        model = getattr(engine, "_vlm_model", None)
    return bool(
        getattr(model, "_omlx_ane_mlp_prefill_count", 0)
        or getattr(model, "_omlx_ane_gdn_prefill_count", 0)
    )


async def _measure_candidate(
    run: ANETuningRun,
    engine_pool: Any,
    base_settings: Any,
    candidate: _Candidate,
) -> dict[str, Any]:
    settings = _settings_for_candidate(base_settings, run.request, candidate)
    engine = await engine_pool.get_engine(
        run.request.model_id,
        force_lm=True,
        runtime_settings=settings,
    )
    if candidate.enabled and not _ane_is_active(engine):
        raise RuntimeError(
            "The ANE candidate loaded, but no eligible Qwen MLP/GDN layers were compiled"
        )

    tokenizer = engine.tokenizer
    warmup_length = run.request.sequence_length + 1
    # stream_generate prefills tokens[:-1]. Add that consumed token explicitly
    # so verification measures four complete scheduler/ANE tiles instead of a
    # half-ANE, half-GPU tail mixture. The old sequence_length * 2 request
    # produced 4095 prefill tokens at the default 2048 shape.
    measure_length = run.request.sequence_length * 4 + 1
    warmup = _generate_prompt(
        tokenizer, warmup_length, BenchmarkContextProfile.CODE_PYTHON
    )
    prompt = _generate_prompt(
        tokenizer, measure_length, BenchmarkContextProfile.CODE_PYTHON
    )

    run.message = f"Warming {candidate.label}…"
    async for _ in engine.stream_generate(
        prompt=warmup,
        max_tokens=2,
        temperature=0.0,
        top_p=1.0,
        skip_cache_store=True,
    ):
        pass

    profile_enabled = False
    fast = None
    if candidate.enabled:
        try:
            from ..custom_kernels.qwen35_prefill import fast as profile_fast

            fast = profile_fast
            profile_enabled = fast.qwen35_ane_profile_set_enabled(True)
            if profile_enabled:
                fast.qwen35_ane_profile_reset()
        except Exception:
            logger.debug("ANE tuner profiling is unavailable", exc_info=True)

    samples: list[float] = []
    traces: list[dict[str, Any] | None] = []
    profile: dict[str, dict[str, float]] = {}
    try:
        # Accelerated finalists are expensive to reload but relatively cheap
        # to sample once loaded. Three observations provide a real median and
        # avoid the average-of-two instability seen in repeated tuner runs.
        measurement_repeats = (
            max(3, run.request.repeats)
            if candidate.enabled
            else run.request.repeats
        )
        for sample_index in range(measurement_repeats):
            metrics = await _run_single_test(
                engine=engine,
                prompt=prompt,
                max_tokens=2,
                pp_len=measure_length,
                ane_trace_config=(
                    {
                        "sequence_length": run.request.sequence_length,
                        "mlp_layers": int(settings.qwen35_ane_prefill_max_layers),
                        "gdn_layers": (
                            int(settings.qwen35_ane_prefill_gdn_max_layers)
                            if candidate.gdn_enabled
                            else 0
                        ),
                    }
                    if candidate.enabled
                    else None
                ),
            )
            samples.append(float(metrics["processing_tps"]))
            run.message = (
                f"Testing {candidate.label}: sample {sample_index + 1}/"
                f"{measurement_repeats} complete · {samples[-1]:.1f} tok/s"
            )
            if candidate.enabled:
                traces.append(metrics.get("ane_trace"))
        if profile_enabled and fast is not None:
            profile = fast.qwen35_ane_profile_snapshot()
    finally:
        if profile_enabled and fast is not None:
            fast.qwen35_ane_profile_set_enabled(False)

    observations = [
        _ane_execution_observed(
            trace,
            require_mlp=candidate.enabled,
            require_gdn=candidate.gdn_enabled,
        )
        for trace in traces
    ]
    # Reject only a positive observation that compiled programs remained idle.
    # If profiling was unavailable, the result is unknown and stays eligible.
    if candidate.enabled and any(o is False for o in observations) and not any(observations):
        step = _prefill_step_size(engine)
        hint = (
            f"the scheduler is configured for {step}-token prefill chunks, "
            f"so set sequence_length={step} or smaller (wider chunks tile "
            "onto the compiled shape, narrower ones cannot)"
            if step
            else "sequence_length must not exceed the scheduler's prefill "
            "chunk width"
        )
        raise RuntimeError(
            "The ANE compiled but never executed for "
            f"sequence_length={run.request.sequence_length}: no ANE operation "
            "was observed during measurement, so this candidate would report "
            "GPU-only throughput as an ANE result. The most common cause is a "
            f"prefill chunk size mismatch ({hint}; confirm against "
            "chunk_tokens in the serve log, since delivered chunks can shrink "
            "below the configured step). A runtime dispatch failure can also "
            "disable the path; check the serve log for 'Disabling ANE' warnings."
        )

    return {
        "label": candidate.label,
        "enabled": candidate.enabled,
        "mlp_fraction": candidate.mlp_fraction,
        "gdn_enabled": candidate.gdn_enabled,
        "gdn_fraction": candidate.gdn_fraction,
        "cpu_enabled": candidate.cpu_enabled,
        "cpu_fraction": candidate.cpu_fraction,
        "cpu_down_fraction": candidate.cpu_down_fraction,
        "cpu_gdn_fraction": candidate.cpu_gdn_fraction,
        "fused_down": candidate.fused_down,
        "cpu_threads": candidate.cpu_threads,
        "processing_tps": round(statistics.median(samples), 2),
        "samples": [round(value, 2) for value in samples],
        "_profile": profile,
    }


def _nearest(value: float, choices: list[float]) -> float:
    return min(choices, key=lambda choice: abs(choice - value))


def _median_absolute_deviation(samples: list[float]) -> float:
    median = statistics.median(samples)
    return statistics.median(abs(sample - median) for sample in samples)


def _balanced_fractions(
    widths: list[float], times: list[float], choices: list[list[float]]
) -> list[float] | None:
    if any(width <= 0 or duration <= 0 for width, duration in zip(widths, times)):
        return None
    rates = [width / duration for width, duration in zip(widths, times)]
    total_rate = sum(rates)
    if total_rate <= 0:
        return None
    return [
        _nearest(rate / total_rate, allowed)
        for rate, allowed in zip(rates, choices)
    ]


def _min_viable_gdn_fraction(patch: Any, gdn: Any, alignment: int) -> float | None:
    """Smallest gdn_fraction that engages the ANE on ``gdn``.

    Delegates to the patch module, which owns the bank rule this mirrors, so
    the tuner's floor and the one the enable path warns about cannot drift.
    """
    return patch._min_viable_gdn_fraction(gdn, alignment)


def _profile_refinement(
    candidate: _Candidate,
    result: dict[str, Any],
    gdn_floor: float | None = None,
) -> _Candidate:
    """Use full-model branch completion rates for one bounded correction."""
    profile = result.get("_profile") or {}
    mlp = profile.get("mlp") or {}
    gdn = profile.get("gdn") or {}
    ane_fraction = float(candidate.mlp_fraction or 0.0)
    cpu_fraction = float(candidate.cpu_fraction or 0.0)
    gpu_fraction = 1.0 - ane_fraction * (2 if candidate.fused_down else 1) - cpu_fraction
    mlp_ops = float(mlp.get("operations", 0.0))
    if mlp_ops > 0 and ane_fraction > 0 and gpu_fraction > 0:
        ane_time = max(
            float(mlp.get("ane0_eval_ns", 0.0)),
            float(mlp.get("ane1_eval_ns", 0.0)),
        ) / mlp_ops
        gpu_time = float(mlp.get("gpu_completion_ns", 0.0)) / mlp_ops
        ane_width = ane_fraction * (2 if candidate.fused_down else 1)
        widths = [ane_width]
        times = [ane_time]
        choices = [
            (
                [2 * value for value in _fused_fraction_grid()]
                if candidate.fused_down
                else sorted(
                    set([*_fraction_grid(), 0.35, 0.40, 0.465, 0.50, 0.55])
                )
            )
        ]
        if cpu_fraction > 0:
            widths.append(cpu_fraction)
            times.append(float(mlp.get("cpu_completion_ns", 0.0)) / mlp_ops)
            choices.append(
                _fused_cpu_fraction_grid()
                if candidate.fused_down
                else _cpu_fraction_grid()
            )
        widths.append(gpu_fraction)
        times.append(gpu_time)
        choices.append([gpu_fraction])
        balanced = _balanced_fractions(widths, times, choices)
        if balanced is not None:
            ane_fraction = balanced[0] / (2 if candidate.fused_down else 1)
            if cpu_fraction > 0:
                cpu_fraction = balanced[1]

    gdn_fraction = candidate.gdn_fraction
    cpu_gdn_fraction = float(candidate.cpu_gdn_fraction or 0.0)
    gdn_ops = float(gdn.get("operations", 0.0))
    if candidate.gdn_enabled and gdn_fraction is not None and gdn_ops > 0:
        ane_time = max(
            float(gdn.get("ane0_eval_ns", 0.0)),
            float(gdn.get("ane1_eval_ns", 0.0)),
        ) / gdn_ops
        gpu_time = float(gdn.get("gpu_completion_ns", 0.0)) / gdn_ops
        # The production GDN path precision-caps ANE at the token-local z
        # projection. Its structural floor is therefore also its only real
        # ANE width; larger requested fractions compile the same z-only slice.
        effective_gdn_fraction = (
            float(gdn_floor) if gdn_floor is not None else float(gdn_fraction)
        )
        widths = [effective_gdn_fraction]
        times = [ane_time]
        gdn_grid = (
            [float(gdn_floor)]
            if gdn_floor is not None
            else sorted(
                set([*_gdn_fraction_grid(), 0.35, 0.40, 0.465, 0.50, 0.53, 0.55])
            )
        )
        choices = [gdn_grid or [float(gdn_fraction)]]
        if cpu_gdn_fraction > 0:
            widths.append(cpu_gdn_fraction)
            times.append(float(gdn.get("cpu_completion_ns", 0.0)) / gdn_ops)
            choices.append(_cpu_gdn_fraction_grid())
        widths.append(1.0 - effective_gdn_fraction - cpu_gdn_fraction)
        times.append(gpu_time)
        choices.append([widths[-1]])
        balanced = _balanced_fractions(widths, times, choices)
        if balanced is not None:
            gdn_fraction = balanced[0]
            if cpu_gdn_fraction > 0:
                cpu_gdn_fraction = balanced[1]

    return replace(
        candidate,
        label="Profile-refined optimum",
        mlp_fraction=ane_fraction,
        cpu_fraction=cpu_fraction,
        gdn_fraction=gdn_fraction,
        cpu_gdn_fraction=cpu_gdn_fraction,
    )


def _loaded_model(engine: Any) -> Any:
    model = getattr(engine, "_model", None)
    if model is None:
        model = getattr(engine, "_vlm_model", None)
    if model is None:
        raise RuntimeError("Loaded engine does not expose its model")
    return model


def _time_native(factory: Callable[[], Any], repeats: int) -> float:
    import mlx.core as mx

    output = factory()
    if output is None:
        raise RuntimeError("Representative Qwen calibration dispatch was ineligible")
    values = output if isinstance(output, (tuple, list)) else (output,)
    mx.eval(*values)
    mx.synchronize()
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        output = factory()
        if output is None:
            raise RuntimeError("Representative Qwen calibration dispatch failed")
        values = output if isinstance(output, (tuple, list)) else (output,)
        mx.eval(*values)
        mx.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def _restore_attr(target: Any, name: str, previous: Any, missing: object) -> None:
    if previous is missing:
        with suppress(AttributeError):
            delattr(target, name)
    else:
        setattr(target, name, previous)


def _time_mlp_state(
    patch: Any,
    mlp: Any,
    x: Any,
    config: Any,
    state: Any,
    repeats: int,
) -> float:
    missing = object()
    names = (
        "_omlx_ane_prefill_config",
        "_omlx_ane_prefill_state",
        "_omlx_ane_prefill_failed",
    )
    previous = {name: getattr(mlp, name, missing) for name in names}
    mlp._omlx_ane_prefill_config = config
    mlp._omlx_ane_prefill_state = state
    mlp._omlx_ane_prefill_failed = False
    try:
        return _time_native(lambda: patch._backend(mlp, x), repeats)
    finally:
        for name in names:
            _restore_attr(mlp, name, previous[name], missing)


def _time_fused_mlp_state(
    patch: Any,
    mlp: Any,
    x: Any,
    config: Any,
    state: Any,
    repeats: int,
) -> float:
    missing = object()
    names = (
        "_omlx_ane_prefill_config",
        "_omlx_ane_fused_down_state",
        "_omlx_ane_prefill_failed",
    )
    previous = {name: getattr(mlp, name, missing) for name in names}
    mlp._omlx_ane_prefill_config = config
    mlp._omlx_ane_fused_down_state = state
    mlp._omlx_ane_prefill_failed = False
    try:
        return _time_native(lambda: patch._backend(mlp, x), repeats)
    finally:
        for name in names:
            _restore_attr(mlp, name, previous[name], missing)


def _time_fused_mlp_state_once(
    patch: Any,
    mlp: Any,
    x: Any,
    config: Any,
    state: Any,
) -> float:
    """Time one already-warmed native fused dispatch."""
    import mlx.core as mx

    missing = object()
    names = (
        "_omlx_ane_prefill_config",
        "_omlx_ane_fused_down_state",
        "_omlx_ane_prefill_failed",
    )
    previous = {name: getattr(mlp, name, missing) for name in names}
    mlp._omlx_ane_prefill_config = config
    mlp._omlx_ane_fused_down_state = state
    mlp._omlx_ane_prefill_failed = False
    try:
        started = time.perf_counter()
        output = patch._backend(mlp, x)
        if output is None:
            raise RuntimeError("Representative fused Qwen dispatch failed")
        values = output if isinstance(output, (tuple, list)) else (output,)
        mx.eval(*values)
        mx.synchronize()
        return (time.perf_counter() - started) * 1000.0
    finally:
        for name in names:
            _restore_attr(mlp, name, previous[name], missing)


def _time_gdn_state(
    patch: Any,
    gdn: Any,
    x: Any,
    config: Any,
    state: Any,
    repeats: int,
) -> float:
    missing = object()
    names = (
        "_omlx_ane_gdn_config",
        "_omlx_ane_gdn_state",
        "_omlx_ane_gdn_failed",
    )
    previous = {name: getattr(gdn, name, missing) for name in names}
    gdn._omlx_ane_gdn_config = config
    gdn._omlx_ane_gdn_state = state
    gdn._omlx_ane_gdn_failed = False
    try:
        return _time_native(lambda: patch._gdn_backend(gdn, x), repeats)
    finally:
        for name in names:
            _restore_attr(gdn, name, previous[name], missing)


def _time_gdn_state_once(
    patch: Any,
    gdn: Any,
    x: Any,
    config: Any,
    state: Any,
) -> float:
    """Time one already-warmed representative GDN dispatch."""
    import mlx.core as mx

    missing = object()
    names = (
        "_omlx_ane_gdn_config",
        "_omlx_ane_gdn_state",
        "_omlx_ane_gdn_failed",
    )
    previous = {name: getattr(gdn, name, missing) for name in names}
    gdn._omlx_ane_gdn_config = config
    gdn._omlx_ane_gdn_state = state
    gdn._omlx_ane_gdn_failed = False
    try:
        started = time.perf_counter()
        output = patch._gdn_backend(gdn, x)
        if output is None:
            raise RuntimeError("Representative Qwen GDN dispatch failed")
        values = output if isinstance(output, (tuple, list)) else (output,)
        mx.eval(*values)
        mx.synchronize()
        return (time.perf_counter() - started) * 1000.0
    finally:
        for name in names:
            _restore_attr(gdn, name, previous[name], missing)


def _calibrate_fused_components_sync(
    run: ANETuningRun,
    base_settings: Any,
    mlp: Any,
    gdn: Any | None,
    fast: Any,
    patch: Any,
) -> _CalibrationChoice:
    """Tune the fused hidden-channel topology against its real native call."""
    import mlx.core as mx

    cpu_supported = bool(
        run.request.allow_cpu
        and run.request.allow_cpu_gate
        and mlp.gate_proj.scales.dtype == mx.float16
        and fast.has_symbol("qwen35_ane_dual_cpu_fp16_q4_swiglu_down_t")
    )
    gdn_cpu_supported = bool(
        run.request.allow_cpu
        and run.request.allow_cpu_gdn
        and run.request.allow_ane_gdn
        and gdn is not None
        and gdn.in_proj_qkv.scales.dtype == mx.float16
        and fast.has_symbol(patch._cpu_gdn_kernel_symbol(dual=True))
    )
    cpu_shared = bool(
        (cpu_supported or gdn_cpu_supported)
        and run.request.allow_cpu_shared_resource
        and getattr(base_settings, "qwen35_ane_prefill_cpu_shared_resource", True)
        and fast.qwen35_cpu_shared_resource_available()
    )
    calibration_threads = _CALIBRATION_CPU_THREADS
    fractions = _fused_fraction_grid()
    cpu_fractions = _fused_cpu_fraction_grid() if cpu_supported else [0.0]
    gdn_fractions = (
        [float(run.gdn_floor)]
        if gdn is not None and run.gdn_floor is not None
        else []
    )
    gdn_cpu_fractions = _cpu_gdn_fraction_grid() if gdn_cpu_supported else [0.0]

    run.phase = "compiling_calibration"
    run.message = "Compiling fused SwiGLU/down calibration procedures…"
    prepared: list[tuple[float, tuple[Any, ...]]] = []
    for fraction in fractions:
        config = patch._AnePrefillConfig(
            run.request.sequence_length,
            fraction,
            8,
            True,
            ane_down_fraction=fraction,
            fused_down=True,
            cpu_threads=calibration_threads,
            cpu_shared_resource=cpu_shared,
        )
        value = patch._prepare_fused_down_for_bank(mlp, config)
        if value is not None:
            _state, weights = value
            prepared.append((fraction, weights))
    if not prepared:
        raise RuntimeError("No valid fused MLP calibration widths could be prepared")
    mx.eval(*(value for _, weights in prepared for value in weights))
    models0 = fast.qwen35_ane_compile_swiglu_down_bank(
        [weights[0] for _, weights in prepared],
        [weights[1] for _, weights in prepared],
        [weights[2] for _, weights in prepared],
        run.request.sequence_length,
        1,
    )
    models1 = fast.qwen35_ane_compile_swiglu_down_bank(
        [weights[3] for _, weights in prepared],
        [weights[4] for _, weights in prepared],
        [weights[5] for _, weights in prepared],
        run.request.sequence_length,
        2,
    )
    fused_models = {
        fraction: (models0[index], models1[index])
        for index, (fraction, _) in enumerate(prepared)
    }
    prepared.clear()
    mx.clear_cache()

    # GDN remains an independent branch, but its grid must cover the higher
    # ANE share made viable by the shorter fused MLP critical path.
    gdn_models: dict[float, tuple[Any, Any]] = {}
    if gdn is not None:
        gdn_prepared = []
        for fraction in gdn_fractions:
            config = patch._AneGDNConfig(
                run.request.sequence_length, fraction, 8, True
            )
            value = patch._prepare_gdn_for_bank(gdn, config)
            if value is not None and value[2] is not None:
                state, dense0, dense1 = value
                gdn_prepared.append((fraction, state, dense0, dense1))
        if gdn_prepared:
            mx.eval(
                *(entry[2] for entry in gdn_prepared),
                *(entry[3] for entry in gdn_prepared),
            )
            banked = patch._compile_dual_banks(
                [entry[2] for entry in gdn_prepared],
                [entry[3] for entry in gdn_prepared],
                run.request.sequence_length,
            )
            if banked is not None:
                gdn0, gdn1, _ = banked
                gdn_models = {
                    entry[0]: (gdn0[index], gdn1[index])
                    for index, entry in enumerate(gdn_prepared)
                }
        gdn_prepared.clear()
        mx.clear_cache()

    valid_combinations = [
        (fraction, cpu_fraction)
        for fraction in fused_models
        for cpu_fraction in cpu_fractions
        if 2 * fraction + cpu_fraction < 1.0
    ]
    thread_points = (
        3 * len(_cpu_thread_grid()) if cpu_supported else 1
    )
    run.total = (
        3
        + len(valid_combinations)
        + thread_points
        + len(gdn_models) * len(gdn_cpu_fractions)
    )
    gate = mlp.gate_proj
    input_dim = int(gate.weight.shape[1]) * 8
    x = mx.zeros(
        (1, run.request.sequence_length, input_dim), dtype=gate.scales.dtype
    )
    mx.eval(x)
    coarse_repeats = _COARSE_SAMPLES

    _set_phase_running(
        run,
        _GATE_SLOT,
        "Balancing fused MLP hidden channels across two ANEs, CPU and GPU…",
    )
    mlp_results: list[tuple[float, float, float]] = []
    for fraction, cpu_fraction in valid_combinations:
        config = patch._AnePrefillConfig(
            run.request.sequence_length,
            fraction,
            8,
            True,
            cpu_fraction=cpu_fraction,
            cpu_threads=calibration_threads,
            cpu_shared_resource=cpu_shared,
            ane_down_fraction=fraction,
            fused_down=True,
        )
        value = patch._prepare_fused_down_for_bank(mlp, config)
        if value is None:
            continue
        state, _weights = value
        model0, model1 = fused_models[fraction]
        state = replace(state, model=model0, model1=model1)
        latency = _time_fused_mlp_state(
            patch, mlp, x, config, state, coarse_repeats
        )
        mlp_results.append((latency, fraction, cpu_fraction))
        preview_ms, preview_ane, preview_cpu = min(mlp_results)
        _preview_phase(
            run,
            _GATE_SLOT,
            detail=(
                f"Current best · ANE {preview_ane:.1%} each · "
                f"CPU {preview_cpu:.1%} · "
                f"GPU {1.0 - 2 * preview_ane - preview_cpu:.1%}"
            ),
            latency_ms=preview_ms,
            mlp_fraction=preview_ane,
            cpu_enabled=preview_cpu > 0,
            cpu_fraction=preview_cpu,
            cpu_down_fraction=0.0,
            fused_down=True,
        )
        run.current += 1
        run.message = (
            f"Fused MLP: ANE {fraction:.1%} each, CPU {cpu_fraction:.1%}…"
        )
        del state, _weights
        mx.clear_cache()
    if not mlp_results:
        raise RuntimeError("Every representative fused MLP candidate failed")
    mlp_ms, best_mlp, best_cpu = min(mlp_results)
    if best_cpu <= 0:
        run.total -= thread_points - 1
    _complete_phase(
        run,
        _GATE_SLOT,
        detail=(
            f"Fused · ANE {best_mlp:.1%} each · CPU {best_cpu:.1%} · "
            f"GPU {1.0 - 2 * best_mlp - best_cpu:.1%}"
        ),
        latency_ms=mlp_ms,
        mlp_fraction=best_mlp,
        cpu_enabled=best_cpu > 0,
        cpu_fraction=best_cpu,
        cpu_down_fraction=0.0,
        fused_down=True,
    )

    best_threads = calibration_threads
    worker_alternate_mlp: float | None = None
    worker_alternate_cpu: float | None = None
    worker_alternate_threads: int | None = None
    worker_uncertainty = float("inf")
    _set_phase_running(run, _DOWN_SLOT, "Tuning fused CPU branch worker count…")
    if best_cpu > 0:
        thread_results: list[tuple[float, float, float, int, list[float]]] = []
        # Worker count can move the optimum partition, so retest the three
        # fastest coarse splits rather than freezing the split before tuning
        # threads. Interleave worker counts within each split so heat/load drift
        # cannot make the last count tested look artificially best.
        finalists = sorted(mlp_results)[:3]
        for _coarse_ms, fraction, cpu_fraction in finalists:
            model0, model1 = fused_models[fraction]
            contender_states: list[tuple[int, Any, Any]] = []
            for threads in _cpu_thread_grid():
                config = patch._AnePrefillConfig(
                    run.request.sequence_length,
                    fraction,
                    8,
                    True,
                    cpu_fraction=cpu_fraction,
                    cpu_threads=threads,
                    cpu_shared_resource=cpu_shared,
                    ane_down_fraction=fraction,
                    fused_down=True,
                )
                value = patch._prepare_fused_down_for_bank(mlp, config)
                if value is None:
                    continue
                state, _weights = value
                state = replace(state, model=model0, model1=model1)
                contender_states.append((threads, config, state))
                del _weights
            if not contender_states:
                continue
            samples_by_thread = {
                threads: [] for threads, _config, _state in contender_states
            }
            # One untimed dispatch per configuration establishes the same
            # cache/kernel state before the interleaved samples.
            for _threads, config, state in contender_states:
                _time_fused_mlp_state_once(patch, mlp, x, config, state)
            for sample_index in range(_FINALIST_SAMPLES):
                offset = sample_index % len(contender_states)
                ordered = contender_states[offset:] + contender_states[:offset]
                for threads, config, state in ordered:
                    samples_by_thread[threads].append(
                        _time_fused_mlp_state_once(patch, mlp, x, config, state)
                    )
            for threads, _config, _state in contender_states:
                samples_ms = samples_by_thread[threads]
                latency = statistics.median(samples_ms)
                thread_results.append(
                    (latency, fraction, cpu_fraction, threads, samples_ms)
                )
                preview = min(thread_results)
                _preview_phase(
                    run,
                    _DOWN_SLOT,
                    detail=(
                        f"Current best · {preview[3]} workers · "
                        f"ANE {preview[1]:.1%} each · CPU {preview[2]:.1%}"
                    ),
                    latency_ms=preview[0],
                    mlp_fraction=preview[1],
                    cpu_enabled=preview[2] > 0,
                    cpu_fraction=preview[2],
                    cpu_down_fraction=0.0,
                    fused_down=True,
                    cpu_threads=preview[3],
                )
                logger.info(
                    "[ane-tuner-fused-worker] ane_each=%.3f cpu=%.3f "
                    "threads=%d latency_ms=%.3f samples_ms=%s",
                    fraction,
                    cpu_fraction,
                    threads,
                    latency,
                    ",".join(f"{sample:.3f}" for sample in samples_ms),
                )
                run.current += 1
                run.message = (
                    f"Fused CPU branch: ANE {fraction:.1%} each, "
                    f"CPU {cpu_fraction:.1%}, {threads} workers…"
                )
            del contender_states
            if samples_by_thread:
                mx.clear_cache()
        if thread_results:
            ranked_threads = sorted(thread_results)
            fastest_ms = ranked_threads[0][0]
            tied_threads = [
                result
                for result in ranked_threads
                if result[0] <= fastest_ms * 1.005
            ]
            # Resolve a sub-0.5% tie without consulting saved settings: prefer
            # the least dispersed result, then the lower resource count.
            selected = min(
                tied_threads,
                key=lambda result: (
                    _median_absolute_deviation(result[4]),
                    result[3],
                    result[2],
                    result[0],
                ),
            )
            worker_ms, best_mlp, best_cpu, best_threads, _samples = selected
            alternate = next(
                (result for result in ranked_threads if result[3] != best_threads),
                None,
            )
            if alternate is not None:
                (
                    alternate_ms,
                    worker_alternate_mlp,
                    worker_alternate_cpu,
                    worker_alternate_threads,
                    _,
                ) = alternate
                worker_uncertainty = (
                    abs(alternate_ms - worker_ms) / min(alternate_ms, worker_ms)
                )
            logger.info(
                "[ane-tuner-fused-worker-choice] threads=%d latency_ms=%.3f "
                "alternate_threads=%s uncertainty=%.4f",
                best_threads,
                worker_ms,
                worker_alternate_threads,
                worker_uncertainty,
            )
        else:
            worker_ms = None
    else:
        worker_ms = None
        run.current += 1
    _complete_phase(
        run,
        _DOWN_SLOT,
        detail=(
            f"{best_threads} workers; down projection is fused per hidden branch"
            if best_cpu > 0
            else "CPU branch disabled; down projection is fused per ANE/GPU branch"
        ),
        latency_ms=worker_ms,
        mlp_fraction=best_mlp,
        cpu_enabled=best_cpu > 0,
        cpu_fraction=best_cpu,
        cpu_down_fraction=0.0,
        fused_down=True,
    )

    best_gdn: float | None = None
    best_gdn_cpu = 0.0
    gdn_alternate: float | None = None
    gdn_cpu_alternate: float | None = None
    gdn_uncertainty = float("inf")
    if gdn_models:
        _set_phase_running(run, _GDN_SLOT, "Balancing GDN across ANE, CPU and GPU…")
        qkv = patch._gdn_linears(gdn)[0]
        qkv_bits = int(qkv.bits)
        gdn_input_dim = int(qkv.weight.shape[1]) * 32 // qkv_bits
        gdn_x = mx.zeros(
            (1, run.request.sequence_length, gdn_input_dim), dtype=qkv.scales.dtype
        )
        mx.eval(gdn_x)
        gdn_results: list[tuple[float, float, float]] = []
        for fraction, (model0, model1) in gdn_models.items():
            for cpu_fraction in gdn_cpu_fractions:
                if fraction + cpu_fraction >= 1.0:
                    continue
                config = patch._AneGDNConfig(
                    run.request.sequence_length,
                    fraction,
                    8,
                    True,
                    cpu_fraction=cpu_fraction,
                    cpu_threads=best_threads,
                    cpu_shared_resource=cpu_shared,
                )
                state = patch._prepare_gdn_runtime_state(gdn, config, model0, model1)
                if state is None:
                    continue
                latency = _time_gdn_state(
                    patch, gdn, gdn_x, config, state, coarse_repeats
                )
                gdn_results.append((latency, fraction, cpu_fraction))
                preview_ms, preview_ane, preview_cpu = min(gdn_results)
                _preview_phase(
                    run,
                    _GDN_SLOT,
                    detail=(
                        f"Coarse best · ANE {preview_ane:.1%} · "
                        f"CPU {preview_cpu:.1%} · "
                        f"GPU {1.0 - preview_ane - preview_cpu:.1%}"
                    ),
                    latency_ms=preview_ms,
                    gdn_enabled=True,
                    gdn_fraction=preview_ane,
                    cpu_enabled=preview_cpu > 0,
                    cpu_gdn_fraction=preview_cpu,
                    fused_down=True,
                )
                run.current += 1
                run.message = f"GDN: ANE {fraction:.1%}, CPU {cpu_fraction:.1%}…"
        if not gdn_results:
            raise RuntimeError("Every representative GDN candidate failed")

        # Coarse timings identify a shortlist cheaply. Retest that shortlist
        # interleaved so a warm ANE, host load, or traversal order cannot flip
        # the chosen GDN topology between otherwise identical tuner runs.
        gdn_contenders: list[tuple[float, float, Any, Any]] = []
        for _coarse_ms, fraction, cpu_fraction in sorted(gdn_results)[:3]:
            model0, model1 = gdn_models[fraction]
            config = patch._AneGDNConfig(
                run.request.sequence_length,
                fraction,
                8,
                True,
                cpu_fraction=cpu_fraction,
                cpu_threads=best_threads,
                cpu_shared_resource=cpu_shared,
            )
            state = patch._prepare_gdn_runtime_state(
                gdn, config, model0, model1
            )
            if state is not None:
                gdn_contenders.append((fraction, cpu_fraction, config, state))
        if not gdn_contenders:
            raise RuntimeError("No shortlisted GDN candidate could be prepared")

        gdn_samples = {
            (fraction, cpu_fraction): []
            for fraction, cpu_fraction, _config, _state in gdn_contenders
        }
        for _fraction, _cpu_fraction, config, state in gdn_contenders:
            _time_gdn_state_once(patch, gdn, gdn_x, config, state)
        for sample_index in range(_FINALIST_SAMPLES):
            offset = sample_index % len(gdn_contenders)
            ordered = gdn_contenders[offset:] + gdn_contenders[:offset]
            for fraction, cpu_fraction, config, state in ordered:
                gdn_samples[(fraction, cpu_fraction)].append(
                    _time_gdn_state_once(patch, gdn, gdn_x, config, state)
                )

        ranked_gdn: list[tuple[float, float, float, list[float]]] = []
        for fraction, cpu_fraction, _config, _state in gdn_contenders:
            samples_ms = gdn_samples[(fraction, cpu_fraction)]
            latency = statistics.median(samples_ms)
            ranked_gdn.append((latency, fraction, cpu_fraction, samples_ms))
            preview = min(ranked_gdn)
            _preview_phase(
                run,
                _GDN_SLOT,
                detail=(
                    f"Finalist best · ANE {preview[1]:.1%} · "
                    f"CPU {preview[2]:.1%} · "
                    f"GPU {1.0 - preview[1] - preview[2]:.1%}"
                ),
                latency_ms=preview[0],
                gdn_enabled=True,
                gdn_fraction=preview[1],
                cpu_enabled=preview[2] > 0,
                cpu_gdn_fraction=preview[2],
                fused_down=True,
            )
            logger.info(
                "[ane-tuner-gdn-finalist] ane=%.3f cpu=%.3f "
                "latency_ms=%.3f samples_ms=%s",
                fraction,
                cpu_fraction,
                latency,
                ",".join(f"{sample:.3f}" for sample in samples_ms),
            )
        ranked_gdn.sort()
        fastest_gdn_ms = ranked_gdn[0][0]
        tied_gdn = [
            result
            for result in ranked_gdn
            if result[0] <= fastest_gdn_ms * 1.005
        ]
        selected_gdn = min(
            tied_gdn,
            key=lambda result: (
                _median_absolute_deviation(result[3]),
                result[2],
                result[1],
                result[0],
            ),
        )
        gdn_ms, best_gdn, best_gdn_cpu, _samples = selected_gdn
        alternate = next(
            (
                result
                for result in ranked_gdn
                if (result[1], result[2]) != (best_gdn, best_gdn_cpu)
            ),
            None,
        )
        if alternate is not None:
            alternate_ms, gdn_alternate, gdn_cpu_alternate, _ = alternate
            gdn_uncertainty = (
                abs(alternate_ms - gdn_ms) / min(alternate_ms, gdn_ms)
            )
        logger.info(
            "[ane-tuner-gdn-choice] ane=%.3f cpu=%.3f latency_ms=%.3f "
            "alternate_ane=%s alternate_cpu=%s uncertainty=%.4f",
            best_gdn,
            best_gdn_cpu,
            gdn_ms,
            gdn_alternate,
            gdn_cpu_alternate,
            gdn_uncertainty,
        )
        del gdn_contenders
        mx.clear_cache()
        _complete_phase(
            run,
            _GDN_SLOT,
            detail=(
                f"ANE {best_gdn:.1%} · CPU {best_gdn_cpu:.1%} · "
                f"GPU {1.0 - best_gdn - best_gdn_cpu:.1%}"
            ),
            latency_ms=gdn_ms,
            gdn_enabled=True,
            gdn_fraction=best_gdn,
            cpu_enabled=best_gdn_cpu > 0,
            cpu_gdn_fraction=best_gdn_cpu,
            fused_down=True,
        )
    else:
        _complete_phase(
            run,
            _GDN_SLOT,
            detail=(
                "Disabled by tuner override"
                if not run.request.allow_ane_gdn
                else "Not eligible in this checkpoint"
            ),
            latency_ms=None,
            fused_down=True,
        )

    # Spend the one existing runner-up model load on the least certain
    # calibration decision. This improves stability without adding another
    # expensive full-model compile/reload cycle.
    alternate_mlp: float | None = None
    alternate_cpu: float | None = None
    alternate_threads: int | None = None
    alternate_gdn: float | None = None
    alternate_gdn_cpu: float | None = None
    alternate_reason: str | None = None
    if gdn_alternate is not None and gdn_uncertainty <= worker_uncertainty:
        alternate_mlp = best_mlp
        alternate_cpu = best_cpu
        alternate_threads = best_threads
        alternate_gdn = gdn_alternate
        alternate_gdn_cpu = gdn_cpu_alternate
        alternate_reason = "GDN topology"
    elif worker_alternate_threads is not None:
        alternate_mlp = worker_alternate_mlp
        alternate_cpu = worker_alternate_cpu
        alternate_threads = worker_alternate_threads
        alternate_gdn = best_gdn
        alternate_gdn_cpu = best_gdn_cpu
        alternate_reason = "CPU worker count"
    logger.info(
        "[ane-tuner-runner-up-choice] reason=%s worker_uncertainty=%.4f "
        "gdn_uncertainty=%.4f",
        alternate_reason,
        worker_uncertainty,
        gdn_uncertainty,
    )

    return _CalibrationChoice(
        mlp_fraction=best_mlp,
        cpu_fraction=best_cpu,
        cpu_down_fraction=0.0,
        cpu_gdn_fraction=best_gdn_cpu,
        gdn_enabled=best_gdn is not None,
        gdn_fraction=best_gdn,
        cpu_enabled=best_cpu > 0 or best_gdn_cpu > 0,
        cpu_threads=best_threads,
        cpu_shared_resource=cpu_shared,
        fused_down=True,
        alternate_mlp_fraction=alternate_mlp,
        alternate_cpu_fraction=alternate_cpu,
        alternate_cpu_threads=alternate_threads,
        alternate_gdn_fraction=alternate_gdn,
        alternate_cpu_gdn_fraction=alternate_gdn_cpu,
        alternate_reason=alternate_reason,
    )


async def _calibrate_components(
    run: ANETuningRun,
    engine: Any,
    base_settings: Any,
) -> _CalibrationChoice:
    # Bank compilation and the per-width timings hold the GIL-side thread for
    # the whole calibration, so run them on the MLX executor like every other
    # model-touching path; the event loop keeps serving results polling,
    # cancel, and health checks in the meantime.
    from omlx.engine_core import get_mlx_executor

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        get_mlx_executor(),
        _calibrate_components_sync,
        run,
        engine,
        base_settings,
    )


def _calibrate_components_sync(
    run: ANETuningRun,
    engine: Any,
    base_settings: Any,
) -> _CalibrationChoice:
    import mlx.core as mx

    from ..custom_kernels.qwen35_prefill import fast
    from ..patches import qwen35_ane_prefill as patch

    dual_ane = bool(getattr(base_settings, "qwen35_ane_prefill_dual_ane", True))

    model = _loaded_model(engine)
    modules = list(model.modules()) if hasattr(model, "modules") else []
    mlp = next((module for module in modules if patch._eligible_pair(module)), None)
    if mlp is None:
        raise RuntimeError("No eligible Qwen MLP layer is available for calibration")
    gdn = (
        next((module for module in modules if patch._eligible_gdn(module)), None)
        if run.request.allow_ane_gdn
        else None
    )
    if gdn is not None:
        run.gdn_floor = _min_viable_gdn_fraction(
            patch, gdn, 128 if dual_ane else 64
        )

    gate = mlp.gate_proj
    down = mlp.down_proj
    fused_supported = bool(
        dual_ane
        and fast.qwen35_ane_swiglu_down_available()
        and all(
            int(getattr(linear, "bits", 0)) == 4
            and int(getattr(linear, "group_size", 0)) == 128
            for linear in (gate, mlp.up_proj, down)
        )
    )
    if fused_supported:
        return _calibrate_fused_components_sync(
            run,
            base_settings,
            mlp,
            gdn,
            fast,
            patch,
        )

    bits = int(gate.bits)
    cpu_gate_supported = bool(
        run.request.allow_cpu
        and bits in (4, 5, 6, 8)
        and gate.scales.dtype == mx.float16
        and mlp.up_proj.scales.dtype == mx.float16
        and fast.has_symbol(
            patch._cpu_gate_kernel_symbol(bits, dual=dual_ane)
        )
    )
    cpu_down_supported = bool(
        run.request.allow_cpu
        and gate.scales.dtype == mx.float16
        and down.scales.dtype == mx.float16
        and fast.has_symbol("qwen35_cpu_fp16_affine_qmm_t")
    )
    gdn_cpu_supported = bool(
        run.request.allow_cpu
        and run.request.allow_cpu_gdn
        and run.request.allow_ane_gdn
        and gdn is not None
        and gate.scales.dtype == mx.float16
        and gdn.in_proj_qkv.scales.dtype == mx.float16
        and fast.has_symbol(patch._cpu_gdn_kernel_symbol(dual=dual_ane))
    )
    cpu_shared = bool(
        (cpu_gate_supported or cpu_down_supported or gdn_cpu_supported)
        and run.request.allow_cpu_shared_resource
        and getattr(base_settings, "qwen35_ane_prefill_cpu_shared_resource", True)
        and fast.qwen35_cpu_shared_resource_available()
    )
    cpu_threads = _CALIBRATION_CPU_THREADS
    fractions = run.fractions
    cpu_fractions = (
        _cpu_fraction_grid()
        if cpu_gate_supported and run.request.allow_cpu_gate
        else [0.0]
    )
    down_fractions = (
        _cpu_down_fraction_grid()
        if cpu_down_supported and run.request.allow_cpu_down
        else [0.0]
    )
    gdn_cpu_fractions = (
        _cpu_gdn_fraction_grid() if gdn_cpu_supported else [0.0]
    )

    run.phase = "compiling_calibration"
    run.message = "Compiling representative ANE calibration bank…"
    prepared: list[tuple[str, float, Any, Any, Any]] = []
    for fraction in fractions:
        config = patch._AnePrefillConfig(
            run.request.sequence_length,
            fraction,
            8,
            dual_ane,
            cpu_threads=cpu_threads,
            cpu_shared_resource=cpu_shared,
        )
        value = patch._prepare_pair_for_bank(mlp, config)
        if value is not None:
            state, dense0, dense1 = value
            prepared.append(("mlp", fraction, state, dense0, dense1))
    if gdn is not None and run.gdn_floor is not None:
        for fraction in [float(run.gdn_floor)]:
            config = patch._AneGDNConfig(
                run.request.sequence_length, fraction, 8, dual_ane
            )
            value = patch._prepare_gdn_for_bank(gdn, config)
            if value is not None:
                state, dense0, dense1 = value
                prepared.append(("gdn", fraction, state, dense0, dense1))
    if not any(kind == "mlp" for kind, *_ in prepared):
        raise RuntimeError("No valid MLP calibration widths could be prepared")

    dense0 = [entry[3] for entry in prepared]
    if dual_ane:
        dense1 = [entry[4] for entry in prepared]
        if any(weight is None for weight in dense1):
            raise RuntimeError("A dual-ANE calibration width was prepared incompletely")
        mx.eval(*dense0, *dense1)
        banked = patch._compile_dual_banks(
            dense0,
            dense1,
            run.request.sequence_length,
        )
        if banked is None:
            raise RuntimeError("Representative ANE calibration bank could not be loaded")
        models0, models1, _ = banked
    else:
        mx.eval(*dense0)
        single_banked = patch._compile_single_banks(
            dense0, run.request.sequence_length
        )
        if single_banked is None:
            raise RuntimeError("Representative ANE calibration bank could not be loaded")
        models0, _ = single_banked
        models1 = [None] * len(models0)
    ane_models: dict[tuple[str, float], tuple[Any, Any, Any]] = {}
    for index, (kind, fraction, state, _, _) in enumerate(prepared):
        ane_models[(kind, fraction)] = (models0[index], models1[index], state)

    valid_mlp_fractions = [
        fraction for fraction in fractions if ("mlp", fraction) in ane_models
    ]
    valid_gdn_fractions = [
        fraction
        for fraction in ([float(run.gdn_floor)] if run.gdn_floor is not None else [])
        if ("gdn", fraction) in ane_models
    ]
    # The loaded private programs own their compiled weight blobs. Release the
    # much larger temporary FP32 source slices before allocating CPU variants.
    prepared.clear()
    mx.clear_cache()
    run.total = (
        3
        + len(valid_mlp_fractions) * len(cpu_fractions)
        + len(down_fractions)
        + len(valid_gdn_fractions) * len(gdn_cpu_fractions)
    )

    input_dim = int(gate.weight.shape[1]) * 32 // bits
    x = mx.zeros(
        (1, run.request.sequence_length, input_dim), dtype=gate.scales.dtype
    )
    mx.eval(x)
    calibration_repeats = _COARSE_SAMPLES

    _set_phase_running(run, _GATE_SLOT, "Balancing MLP gate/up across ANE, CPU and GPU…")
    gate_results: list[tuple[float, float, float]] = []
    for fraction in valid_mlp_fractions:
        model0, model1, _ = ane_models[("mlp", fraction)]
        for cpu_fraction in cpu_fractions:
            config = patch._AnePrefillConfig(
                run.request.sequence_length,
                fraction,
                8,
                dual_ane,
                cpu_fraction=cpu_fraction,
                cpu_threads=cpu_threads,
                cpu_shared_resource=cpu_shared,
            )
            state = patch._prepare_pair_runtime_state(
                mlp, config, model0, model1
            )
            if state is None:
                continue
            latency = _time_mlp_state(
                patch, mlp, x, config, state, calibration_repeats
            )
            gate_results.append((latency, fraction, cpu_fraction))
            preview = min(gate_results)
            _preview_phase(
                run,
                _GATE_SLOT,
                detail=(
                    f"Current best · ANE {preview[1]:.1%} · "
                    f"CPU {preview[2]:.1%}"
                ),
                latency_ms=preview[0],
                mlp_fraction=preview[1],
                cpu_enabled=preview[2] > 0,
                cpu_fraction=preview[2],
            )
            run.current += 1
            run.message = (
                f"MLP gate/up: ANE {fraction:.1%}, CPU {cpu_fraction:.1%}…"
            )
    if not gate_results:
        raise RuntimeError("Every representative MLP gate/up candidate failed")
    gate_ms, best_mlp, best_cpu = min(gate_results)
    _complete_phase(
        run,
        _GATE_SLOT,
        detail=f"ANE {best_mlp:.1%} · CPU {best_cpu:.1%}",
        latency_ms=gate_ms,
        mlp_fraction=best_mlp,
        cpu_enabled=best_cpu > 0,
        cpu_fraction=best_cpu,
    )

    _set_phase_running(run, _DOWN_SLOT, "Balancing MLP down projection across CPU and GPU…")
    down_results: list[tuple[float, float]] = []
    model0, model1, _ = ane_models[("mlp", best_mlp)]
    for down_fraction in down_fractions:
        config = patch._AnePrefillConfig(
            run.request.sequence_length,
            best_mlp,
            8,
            dual_ane,
            cpu_fraction=best_cpu,
            cpu_down_fraction=down_fraction,
            cpu_threads=cpu_threads,
            cpu_shared_resource=cpu_shared,
        )
        state = patch._prepare_pair_runtime_state(mlp, config, model0, model1)
        if state is None:
            continue
        latency = _time_mlp_state(
            patch, mlp, x, config, state, calibration_repeats
        )
        down_results.append((latency, down_fraction))
        preview = min(down_results)
        _preview_phase(
            run,
            _DOWN_SLOT,
            detail=(
                f"Current best · CPU {preview[1]:.1%} · "
                f"GPU {1.0 - preview[1]:.1%}"
            ),
            latency_ms=preview[0],
            mlp_fraction=best_mlp,
            cpu_enabled=best_cpu > 0 or preview[1] > 0,
            cpu_fraction=best_cpu,
            cpu_down_fraction=preview[1],
        )
        run.current += 1
        run.message = f"MLP down projection: CPU {down_fraction:.1%}…"
    if not down_results:
        raise RuntimeError("Every representative down-projection candidate failed")
    down_ms, best_down = min(down_results)
    _complete_phase(
        run,
        _DOWN_SLOT,
        detail=f"CPU {best_down:.1%} · GPU {1.0 - best_down:.1%}",
        latency_ms=down_ms,
        mlp_fraction=best_mlp,
        cpu_enabled=best_cpu > 0 or best_down > 0,
        cpu_fraction=best_cpu,
        cpu_down_fraction=best_down,
    )

    best_gdn: float | None = None
    best_gdn_cpu = 0.0
    if gdn is not None and valid_gdn_fractions:
        _set_phase_running(run, _GDN_SLOT, "Balancing GDN across ANE and GPU…")
        qkv = patch._gdn_linears(gdn)[0]
        qkv_bits = int(qkv.bits)
        gdn_input_dim = int(qkv.weight.shape[1]) * 32 // qkv_bits
        gdn_x = mx.zeros(
            (1, run.request.sequence_length, gdn_input_dim), dtype=qkv.scales.dtype
        )
        mx.eval(gdn_x)
        gdn_results: list[tuple[float, float, float]] = []
        for fraction in valid_gdn_fractions:
            model0, model1, _ = ane_models[("gdn", fraction)]
            for cpu_fraction in gdn_cpu_fractions:
                config = patch._AneGDNConfig(
                    run.request.sequence_length,
                    fraction,
                    8,
                    dual_ane,
                    cpu_fraction=cpu_fraction,
                    cpu_threads=cpu_threads,
                    cpu_shared_resource=cpu_shared,
                )
                state = patch._prepare_gdn_runtime_state(
                    gdn, config, model0, model1
                )
                if state is None:
                    continue
                latency = _time_gdn_state(
                    patch, gdn, gdn_x, config, state, calibration_repeats
                )
                gdn_results.append((latency, fraction, cpu_fraction))
                preview = min(gdn_results)
                _preview_phase(
                    run,
                    _GDN_SLOT,
                    detail=(
                        f"Current best · ANE {preview[1]:.1%} · "
                        f"CPU {preview[2]:.1%} · "
                        f"GPU {1.0 - preview[1] - preview[2]:.1%}"
                    ),
                    latency_ms=preview[0],
                    gdn_enabled=True,
                    gdn_fraction=preview[1],
                    cpu_enabled=preview[2] > 0,
                    cpu_gdn_fraction=preview[2],
                )
                run.current += 1
                run.message = (
                    f"GDN: ANE {fraction:.1%}, CPU {cpu_fraction:.1%}…"
                )
        if not gdn_results:
            raise RuntimeError("Every representative GDN candidate failed")
        gdn_ms, best_gdn, best_gdn_cpu = min(gdn_results)
        _complete_phase(
            run,
            _GDN_SLOT,
            detail=(
                f"ANE {best_gdn:.1%} · CPU {best_gdn_cpu:.1%} · "
                f"GPU {1.0 - best_gdn - best_gdn_cpu:.1%}"
            ),
            latency_ms=gdn_ms,
            gdn_enabled=True,
            gdn_fraction=best_gdn,
            cpu_enabled=best_gdn_cpu > 0,
            cpu_gdn_fraction=best_gdn_cpu,
        )
    else:
        _complete_phase(
            run,
            _GDN_SLOT,
            detail=(
                "Disabled by tuner override"
                if not run.request.allow_ane_gdn
                else "Not eligible in this checkpoint"
            ),
            latency_ms=None,
            gdn_enabled=False,
        )

    return _CalibrationChoice(
        mlp_fraction=best_mlp,
        cpu_fraction=best_cpu,
        cpu_down_fraction=best_down,
        cpu_gdn_fraction=best_gdn_cpu,
        gdn_enabled=best_gdn is not None,
        gdn_fraction=best_gdn,
        cpu_enabled=best_cpu > 0 or best_down > 0 or best_gdn_cpu > 0,
        cpu_threads=cpu_threads,
        cpu_shared_resource=cpu_shared,
    )


async def run_tuning(run: ANETuningRun, engine_pool: Any) -> None:
    # The serve path skips ANE gracefully when the private runtime is
    # missing, but the tuner used to find out only deep inside the
    # bank-split ladder — failing the run and leaving the model unloaded
    # (#3044, M2 Ultra). Probe up front, before anything is pinned or
    # unloaded, and return a completed GPU-only verdict: on such a machine
    # that IS the tuning answer, not an error.
    try:
        from ..custom_kernels.qwen35_prefill import fast

        runtime_available = fast.qwen35_ane_available()
        bank_available = fast.qwen35_ane_bank_compiler_available()
    except Exception:
        # Never let the guard itself break tuning; fail open and let the
        # ladder report whatever is actually wrong.
        runtime_available = True
        bank_available = True
    if not bank_available:
        reason = (
            "the private ANE runtime is unavailable on this machine"
            if not runtime_available
            else "the private ANE procedure-bank compiler is missing from "
            "this build"
        )
        logger.warning(
            "ANE tuning for %s skipped: %s; recommending GPU-only",
            run.request.model_id,
            reason,
        )
        for result in run.results:
            result["state"] = "skipped"
            result["error"] = f"ANE unavailable: {reason}"
        run.recommendation = {
            "enabled": False,
            "mlp_fraction": None,
            "gdn_enabled": False,
            "gdn_fraction": None,
            "fused_down": False,
            "processing_tps": None,
            "speedup_percent": None,
            "sequence_length": run.request.sequence_length,
            "tail_padding_min_tokens": 0,
        }
        run.status = "completed"
        run.phase = "completed"
        run.current = run.total
        run.message = (
            f"ANE prefill is not usable here ({reason}); GPU-only recommended"
        )
        return

    previous_speed_priority = _pin_speed_priority(engine_pool)
    active_slot = _GPU_SLOT
    try:
        settings_manager = getattr(engine_pool, "_settings_manager", None)
        if settings_manager is None:
            raise RuntimeError("Model settings are unavailable")
        base_settings = settings_manager.get_settings(run.request.model_id)

        run.phase = "unloading"
        run.message = "Unloading models before tuning…"
        for model_id in list(engine_pool.get_loaded_model_ids()):
            await engine_pool._unload_engine(model_id)

        baseline = _Candidate("GPU only", False)
        await _measure_result_slot(
            run, _GPU_SLOT, engine_pool, base_settings, baseline
        )

        gpu_settings = _settings_for_candidate(base_settings, run.request, baseline)
        engine = await engine_pool.get_engine(
            run.request.model_id,
            force_lm=True,
            runtime_settings=gpu_settings,
        )
        active_slot = _GATE_SLOT
        choice = await _calibrate_components(run, engine, base_settings)
        # Release the calibration engine before staging the verify engine:
        # this local reference kept the full model alive across the reload,
        # doubling residency and OOMing 48 GB machines at the verify slot
        # (issue #2908).
        engine = None

        candidate = _Candidate(
            label="Predicted optimum",
            enabled=True,
            mlp_fraction=choice.mlp_fraction,
            gdn_enabled=choice.gdn_enabled,
            gdn_fraction=choice.gdn_fraction,
            cpu_enabled=choice.cpu_enabled,
            cpu_fraction=choice.cpu_fraction,
            cpu_down_fraction=choice.cpu_down_fraction,
            cpu_gdn_fraction=choice.cpu_gdn_fraction,
            fused_down=choice.fused_down,
            cpu_threads=choice.cpu_threads,
        )
        active_slot = _VERIFY_SLOT
        await _measure_result_slot(
            run, _VERIFY_SLOT, engine_pool, base_settings, candidate
        )

        profiled = _profile_refinement(
            candidate,
            run.results[_VERIFY_SLOT],
            gdn_floor=run.gdn_floor,
        )
        gdn_profile_changed = any(
            getattr(profiled, name) != getattr(candidate, name)
            for name in ("gdn_fraction", "cpu_gdn_fraction")
        )
        if choice.fused_down and gdn_profile_changed:
            # The representative GDN call cannot reproduce contention with all
            # surrounding layers. Let the aggregate three-sample native profile
            # propose a stateless correction and spend the existing runner-up
            # load validating it end to end.
            refined = replace(
                candidate,
                label="Full-model profile-refined GDN",
                gdn_fraction=profiled.gdn_fraction,
                cpu_gdn_fraction=profiled.cpu_gdn_fraction,
            )
            refinement_changed = True
            logger.info(
                "[ane-tuner-full-model-correction] gdn=%.3f->%.3f "
                "cpu_gdn=%.3f->%.3f",
                float(candidate.gdn_fraction or 0.0),
                float(refined.gdn_fraction or 0.0),
                float(candidate.cpu_gdn_fraction or 0.0),
                float(refined.cpu_gdn_fraction or 0.0),
            )
        elif choice.fused_down and choice.alternate_cpu_threads is not None:
            refined = replace(
                candidate,
                label=(
                    f"Full-model {choice.alternate_reason} runner-up"
                    if choice.alternate_reason
                    else "Full-model calibration runner-up"
                ),
                mlp_fraction=choice.alternate_mlp_fraction,
                cpu_fraction=choice.alternate_cpu_fraction,
                cpu_threads=choice.alternate_cpu_threads,
                gdn_fraction=(
                    choice.alternate_gdn_fraction
                    if choice.alternate_gdn_fraction is not None
                    else candidate.gdn_fraction
                ),
                cpu_gdn_fraction=(
                    choice.alternate_cpu_gdn_fraction
                    if choice.alternate_cpu_gdn_fraction is not None
                    else candidate.cpu_gdn_fraction
                ),
            )
            refinement_changed = True
        else:
            refined = profiled
            refinement_changed = any(
                getattr(refined, name) != getattr(candidate, name)
                for name in (
                    "mlp_fraction",
                    "cpu_fraction",
                    "gdn_fraction",
                    "cpu_gdn_fraction",
                    "cpu_threads",
                )
            )
        if refinement_changed:
            active_slot = _REFINE_SLOT
            await _measure_result_slot(
                run, _REFINE_SLOT, engine_pool, base_settings, refined
            )
        else:
            _complete_phase(
                run,
                _REFINE_SLOT,
                detail="Initial prediction was already balanced",
                latency_ms=None,
                mlp_fraction=refined.mlp_fraction,
                gdn_enabled=refined.gdn_enabled,
                gdn_fraction=refined.gdn_fraction,
                cpu_enabled=refined.cpu_enabled,
                cpu_fraction=refined.cpu_fraction,
                cpu_down_fraction=refined.cpu_down_fraction,
                cpu_gdn_fraction=refined.cpu_gdn_fraction,
                fused_down=refined.fused_down,
                cpu_threads=refined.cpu_threads,
            )
            run.current += 1

        baseline_result = run.results[_GPU_SLOT]
        completed = [
            run.results[index]
            for index in (_VERIFY_SLOT, _REFINE_SLOT)
            if run.results[index]["processing_tps"] is not None
        ]
        best = max(completed, key=lambda result: result["processing_tps"])
        if (
            best["processing_tps"] is None
            or best["speedup_percent"] is None
            or best["speedup_percent"] < 1.0
        ):
            best = baseline_result
        # The persisted recommendation must match what actually ran: when the
        # winning slot's profile proves GDN executed 0 operations (while MLP
        # profiling worked), gdn_enabled would only mislead (issue #2899).
        gdn_enabled = bool(best["gdn_enabled"])
        gdn_fraction = best["gdn_fraction"]
        profile = best.get("_profile") or {}
        mlp_ops = float((profile.get("mlp") or {}).get("operations", 0) or 0)
        gdn_ops = float((profile.get("gdn") or {}).get("operations", 0) or 0)
        if gdn_enabled and mlp_ops > 0 and gdn_ops <= 0:
            logger.warning(
                "ANE tuner: recommended slot ran 0 GDN operations "
                "(gdn_fraction=%s is below this model's floor %s); "
                "persisting the recommendation with GDN disabled",
                gdn_fraction,
                run.gdn_floor,
            )
            gdn_enabled = False
            gdn_fraction = None
        tail_padding_min_tokens = _tail_padding_min_tokens(
            run.request.sequence_length,
            baseline_result.get("processing_tps"),
            best.get("processing_tps") if best.get("enabled") else None,
        )
        run.recommendation = {
            "enabled": bool(best["enabled"]),
            "mlp_fraction": best["mlp_fraction"],
            "gdn_enabled": gdn_enabled,
            "gdn_fraction": gdn_fraction,
            "cpu_enabled": bool(best.get("cpu_enabled", False)),
            "cpu_fraction": best.get("cpu_fraction"),
            "cpu_down_fraction": best.get("cpu_down_fraction"),
            "cpu_gdn_fraction": best.get("cpu_gdn_fraction"),
            "fused_down": bool(best.get("fused_down", False)),
            "cpu_threads": best.get("cpu_threads") or choice.cpu_threads,
            "cpu_shared_resource": choice.cpu_shared_resource,
            "processing_tps": best["processing_tps"],
            "speedup_percent": best["speedup_percent"],
            "sequence_length": run.request.sequence_length,
            "tail_padding_min_tokens": tail_padding_min_tokens,
        }
        run.status = "completed"
        run.phase = "completed"
        run.current = run.total
        run.message = "Tuning complete"
    except asyncio.CancelledError:
        run.status = "cancelled"
        run.phase = "cancelled"
        run.message = "Tuning cancelled"
        interrupted = next(
            (
                result
                for result in run.results
                if result["state"] == "running"
            ),
            run.results[active_slot],
        )
        if interrupted["state"] in ("pending", "running"):
            interrupted["state"] = "cancelled"
            interrupted["error"] = "Cancelled by user"
        run.termination_reason = (
            f"Cancelled by user after {run.current} of {run.total} tests completed."
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("ANE tuning failed for %s", run.request.model_id)
        run.status = "error"
        run.phase = "error"
        run.error_message = _exception_reason(exc)
        interrupted = next(
            (
                result
                for result in run.results
                if result["state"] == "running"
            ),
            run.results[active_slot],
        )
        if interrupted["state"] in ("pending", "running"):
            interrupted["state"] = "failed"
            interrupted["error"] = run.error_message
        run.termination_reason = (
            f"Stopped after {run.current} of {run.total} tests: "
            f"{run.error_message}"
        )
        run.message = "Tuning stopped early"
        baseline_result = run.results[_GPU_SLOT]
        if (
            run.recommendation is None
            and baseline_result.get("processing_tps") is not None
        ):
            # A completed GPU-only baseline is still a valid answer: keep ANE
            # off. Discarding it because a later measurement failed would
            # throw away the one number the run did establish.
            run.recommendation = {
                "enabled": False,
                "mlp_fraction": None,
                "gdn_enabled": False,
                "gdn_fraction": None,
                "fused_down": False,
                "processing_tps": baseline_result["processing_tps"],
                "speedup_percent": baseline_result.get("speedup_percent"),
                "sequence_length": run.request.sequence_length,
                "tail_padding_min_tokens": 0,
            }
    finally:
        _restore_speed_priority(engine_pool, previous_speed_priority)
        try:
            if run.request.model_id in engine_pool.get_loaded_model_ids():
                await engine_pool._unload_engine(run.request.model_id)
        except Exception:
            logger.warning("Failed to unload model after ANE tuning", exc_info=True)
