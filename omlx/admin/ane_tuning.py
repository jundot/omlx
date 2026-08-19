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
    return [0.0, 0.05, 0.10, 0.125, 0.15, 0.20, 0.25, 0.35]


def _planned_rows() -> list[_Candidate]:
    return [
        _Candidate("GPU only", False),
        _Candidate("MLP gate/up calibration", True, stage="calibration"),
        _Candidate("Down projection calibration", True, stage="calibration"),
        _Candidate("GDN calibration", True, stage="calibration"),
        _Candidate("Predicted optimum", True),
        _Candidate("Profile-refined optimum", True),
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


def _set_phase_running(run: ANETuningRun, slot: int, message: str) -> None:
    run.phase = "calibrating"
    run.message = message
    run.results[slot]["state"] = "running"


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
    settings.qwen35_ane_prefill_sequence_length = request.sequence_length
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
    settings.qwen35_ane_prefill_dual_ane = bool(
        getattr(base, "qwen35_ane_prefill_dual_ane", True)
    )
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


def _ane_execution_observed(trace: dict[str, Any] | None) -> bool | None:
    """Whether the ANE ran ops, or None when that cannot be determined.

    The operation counters come from the ANE profiler. When the profiler is
    unavailable they are all zero no matter what the hardware did, so that
    case must be reported as unknown rather than as an idle ANE.
    """
    if not trace or not trace.get("profiling_available"):
        return None
    return any(
        int(values.get("operations", 0) or 0) > 0
        for values in (trace.get("categories") or {}).values()
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
    # stream_generate prefills tokens[:-1], so sequence_length * 2 leaves a
    # non-ANE-shaped tail chunk in the measurement like real traffic does.
    # The previous * 2 + 1 measured two full ANE blocks with no GPU tail and
    # overstated the ANE gain.
    measure_length = run.request.sequence_length * 2
    warmup = _generate_prompt(
        tokenizer, warmup_length, BenchmarkContextProfile.CODE_PYTHON
    )
    prompt = _generate_prompt(
        tokenizer, measure_length, BenchmarkContextProfile.CODE_PYTHON
    )

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
        for _ in range(run.request.repeats):
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
            if candidate.enabled:
                traces.append(metrics.get("ane_trace"))
        if profile_enabled and fast is not None:
            profile = fast.qwen35_ane_profile_snapshot()
    finally:
        if profile_enabled and fast is not None:
            fast.qwen35_ane_profile_set_enabled(False)

    observations = [_ane_execution_observed(trace) for trace in traces]
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
        "processing_tps": round(statistics.median(samples), 2),
        "samples": [round(value, 2) for value in samples],
        "_profile": profile,
    }


def _nearest(value: float, choices: list[float]) -> float:
    return min(choices, key=lambda choice: abs(choice - value))


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


def _profile_refinement(candidate: _Candidate, result: dict[str, Any]) -> _Candidate:
    """Use full-model branch completion rates for one bounded correction."""
    profile = result.get("_profile") or {}
    mlp = profile.get("mlp") or {}
    gdn = profile.get("gdn") or {}
    ane_fraction = float(candidate.mlp_fraction or 0.0)
    cpu_fraction = float(candidate.cpu_fraction or 0.0)
    gpu_fraction = 1.0 - ane_fraction - cpu_fraction
    mlp_ops = float(mlp.get("operations", 0.0))
    if mlp_ops > 0 and cpu_fraction > 0:
        ane_time = max(
            float(mlp.get("ane0_eval_ns", 0.0)),
            float(mlp.get("ane1_eval_ns", 0.0)),
        ) / mlp_ops
        cpu_time = float(mlp.get("cpu_completion_ns", 0.0)) / mlp_ops
        gpu_time = float(mlp.get("gpu_completion_ns", 0.0)) / mlp_ops
        balanced = _balanced_fractions(
            [ane_fraction, cpu_fraction, gpu_fraction],
            [ane_time, cpu_time, gpu_time],
            [
                sorted(set([*_fraction_grid(), 0.35, 0.40, 0.465, 0.50, 0.55])),
                _cpu_fraction_grid(),
                [gpu_fraction],
            ],
        )
        if balanced is not None:
            ane_fraction, cpu_fraction = balanced[:2]

    gdn_fraction = candidate.gdn_fraction
    cpu_gdn_fraction = float(candidate.cpu_gdn_fraction or 0.0)
    gdn_ops = float(gdn.get("operations", 0.0))
    if candidate.gdn_enabled and gdn_fraction is not None and gdn_ops > 0:
        ane_time = max(
            float(gdn.get("ane0_eval_ns", 0.0)),
            float(gdn.get("ane1_eval_ns", 0.0)),
        ) / gdn_ops
        gpu_time = float(gdn.get("gpu_completion_ns", 0.0)) / gdn_ops
        widths = [float(gdn_fraction)]
        times = [ane_time]
        choices = [
            sorted(set([*_fraction_grid(), 0.35, 0.40, 0.465, 0.50, 0.53, 0.55]))
        ]
        if cpu_gdn_fraction > 0:
            widths.append(cpu_gdn_fraction)
            times.append(float(gdn.get("cpu_completion_ns", 0.0)) / gdn_ops)
            choices.append(_cpu_gdn_fraction_grid())
        widths.append(1.0 - float(gdn_fraction) - cpu_gdn_fraction)
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

    gate = mlp.gate_proj
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
    down = mlp.down_proj
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
    cpu_threads = int(
        getattr(base_settings, "qwen35_ane_prefill_cpu_threads", 8) or 8
    )
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
    if gdn is not None:
        for fraction in fractions:
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
        fraction for fraction in fractions if ("gdn", fraction) in ane_models
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
    calibration_repeats = max(1, min(run.request.repeats, 2))

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

        candidate = _Candidate(
            "Predicted optimum",
            True,
            choice.mlp_fraction,
            choice.gdn_enabled,
            choice.gdn_fraction,
            choice.cpu_enabled,
            choice.cpu_fraction,
            choice.cpu_down_fraction,
            choice.cpu_gdn_fraction,
        )
        active_slot = _VERIFY_SLOT
        await _measure_result_slot(
            run, _VERIFY_SLOT, engine_pool, base_settings, candidate
        )

        refined = _profile_refinement(candidate, run.results[_VERIFY_SLOT])
        refinement_changed = any(
            getattr(refined, name) != getattr(candidate, name)
            for name in (
                "mlp_fraction",
                "cpu_fraction",
                "gdn_fraction",
                "cpu_gdn_fraction",
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
        run.recommendation = {
            "enabled": bool(best["enabled"]),
            "mlp_fraction": best["mlp_fraction"],
            "gdn_enabled": bool(best["gdn_enabled"]),
            "gdn_fraction": best["gdn_fraction"],
            "cpu_enabled": bool(best.get("cpu_enabled", False)),
            "cpu_fraction": best.get("cpu_fraction"),
            "cpu_down_fraction": best.get("cpu_down_fraction"),
            "cpu_gdn_fraction": best.get("cpu_gdn_fraction"),
            "cpu_threads": choice.cpu_threads,
            "cpu_shared_resource": choice.cpu_shared_resource,
            "processing_tps": best["processing_tps"],
            "speedup_percent": best["speedup_percent"],
            "sequence_length": run.request.sequence_length,
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
                "processing_tps": baseline_result["processing_tps"],
                "speedup_percent": baseline_result.get("speedup_percent"),
                "sequence_length": run.request.sequence_length,
            }
    finally:
        _restore_speed_priority(engine_pool, previous_speed_priority)
        try:
            if run.request.model_id in engine_pool.get_loaded_model_ids():
                await engine_pool._unload_engine(run.request.model_id)
        except Exception:
            logger.warning("Failed to unload model after ANE tuning", exc_info=True)
