# SPDX-License-Identifier: Apache-2.0
"""Hardware-local Qwen ANE/GPU split tuning.

The tuner deliberately uses the normal engine and scheduler rather than timing
the projection kernels in isolation.  Each candidate is loaded with transient
model settings, warmed with one exact ANE block, then measured with two blocks.
Persisted model settings are never changed by a tuning run.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
import uuid
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


class ANETuningRequest(BaseModel):
    model_id: str
    sequence_length: int = 2048
    repeats: int = 2

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
    recommendation: dict[str, Any] | None = None
    error_message: str = ""
    task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.time)


def _fraction_grid() -> list[float]:
    """Broad enough for NAX and classic GPUs without an exhaustive reload grid."""
    try:
        from ..custom_kernels.nax import is_nax_available

        if is_nax_available():
            # NAX makes the GPU suffix materially faster, so the balance can
            # move well below the classic M3/M4 ~0.5 optimum.
            return [0.15, 0.25, 0.35, 0.45, 0.53]
    except Exception:
        pass
    return [0.40, 0.45, 0.50, 0.53, 0.60]


def create_run(request: ANETuningRequest) -> ANETuningRun:
    run = ANETuningRun(tuning_id=str(uuid.uuid4()), request=request)
    run.total = 1 + len(_fraction_grid()) * 2
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
        "results": list(run.results),
        "recommendation": run.recommendation,
        "error": run.error_message or None,
    }


def _settings_for_candidate(base: Any, request: ANETuningRequest, candidate: _Candidate):
    settings = replace(base)
    settings.qwen35_ane_prefill_enabled = candidate.enabled
    settings.qwen35_ane_prefill_sequence_length = request.sequence_length
    if candidate.mlp_fraction is not None:
        settings.qwen35_ane_prefill_fraction = candidate.mlp_fraction
    settings.qwen35_ane_prefill_gdn = candidate.gdn_enabled
    if candidate.gdn_fraction is not None:
        settings.qwen35_ane_prefill_gdn_fraction = candidate.gdn_fraction
    # Procedure banks are part of the path being tuned.  Keep the user's
    # layer limits, but always compare the intended dual-instance backend.
    settings.qwen35_ane_prefill_dual_ane = True
    return settings


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
    measure_length = run.request.sequence_length * 2 + 1
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

    samples: list[float] = []
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

    return {
        "label": candidate.label,
        "enabled": candidate.enabled,
        "mlp_fraction": candidate.mlp_fraction,
        "gdn_enabled": candidate.gdn_enabled,
        "gdn_fraction": candidate.gdn_fraction,
        "processing_tps": round(statistics.median(samples), 2),
        "samples": [round(value, 2) for value in samples],
    }


async def run_tuning(run: ANETuningRun, engine_pool: Any) -> None:
    previous_speed_priority = _pin_speed_priority(engine_pool)
    try:
        settings_manager = getattr(engine_pool, "_settings_manager", None)
        if settings_manager is None:
            raise RuntimeError("Model settings are unavailable")
        base_settings = settings_manager.get_settings(run.request.model_id)

        run.phase = "unloading"
        run.message = "Unloading models before tuning…"
        for model_id in list(engine_pool.get_loaded_model_ids()):
            await engine_pool._unload_engine(model_id)

        fractions = _fraction_grid()
        # Coordinate search keeps the number of expensive eager-compilation
        # reloads bounded: tune MLP alone, then tune GDN around the winning MLP.
        candidates = [_Candidate("GPU only", False)]
        candidates.extend(
            _Candidate(f"MLP {fraction:.0%}", True, fraction)
            for fraction in fractions
        )
        run.total = 1 + len(fractions) * 2

        for candidate in candidates:
            run.phase = "measuring"
            run.message = f"Testing {candidate.label}…"
            result = await _measure_candidate(
                run, engine_pool, base_settings, candidate
            )
            run.results.append(result)
            run.current += 1

        best_mlp = max(
            (result for result in run.results if result["enabled"]),
            key=lambda result: result["processing_tps"],
        )
        for fraction in fractions:
            candidate = _Candidate(
                f"MLP {best_mlp['mlp_fraction']:.0%} + GDN {fraction:.0%}",
                True,
                float(best_mlp["mlp_fraction"]),
                True,
                fraction,
            )
            run.phase = "measuring"
            run.message = f"Testing {candidate.label}…"
            result = await _measure_candidate(
                run, engine_pool, base_settings, candidate
            )
            run.results.append(result)
            run.current += 1

        baseline = run.results[0]["processing_tps"]
        for result in run.results:
            result["speedup_percent"] = round(
                (result["processing_tps"] / baseline - 1.0) * 100.0, 2
            )

        best = max(run.results, key=lambda result: result["processing_tps"])
        # A sub-1% lead is smaller than the normal run-to-run noise and is not
        # enough to justify private-API load time and memory overhead.
        if best["enabled"] and best["speedup_percent"] < 1.0:
            best = run.results[0]
        run.recommendation = {
            "enabled": bool(best["enabled"]),
            "mlp_fraction": best["mlp_fraction"],
            "gdn_enabled": bool(best["gdn_enabled"]),
            "gdn_fraction": best["gdn_fraction"],
            "processing_tps": best["processing_tps"],
            "speedup_percent": best["speedup_percent"],
            "sequence_length": run.request.sequence_length,
        }
        run.status = "completed"
        run.phase = "completed"
        run.message = "Tuning complete"
    except asyncio.CancelledError:
        run.status = "cancelled"
        run.phase = "cancelled"
        run.message = "Tuning cancelled"
    except Exception as exc:  # noqa: BLE001
        logger.exception("ANE tuning failed for %s", run.request.model_id)
        run.status = "error"
        run.phase = "error"
        run.error_message = str(exc)
        run.message = "Tuning failed"
    finally:
        _restore_speed_priority(engine_pool, previous_speed_priority)
        try:
            if run.request.model_id in engine_pool.get_loaded_model_ids():
                await engine_pool._unload_engine(run.request.model_id)
        except Exception:
            logger.warning("Failed to unload model after ANE tuning", exc_info=True)
