# SPDX-License-Identifier: Apache-2.0
"""Context window benchmark for oMLX admin panel.

Measures the largest prompt the current machine can actually prefill for
a model, then writes the result into the model's ``max_context_window``
setting so clients get an honest upfront limit instead of mid-prefill
aborts.

Method: a short calibration prefill seeds the scheduler's transient
tracker, then the deterministic admission boundary is found by bisecting
``scheduler.preflight_or_raise`` in-process — microseconds per probe, no
GPU work. One real prefill at the (2k-floored) boundary verifies it end
to end; on a mid-prefill abort a single conservative retry runs, sized
from where the first prefill actually died (90% of its processed
tokens). A clean first-try completion instead climbs upward in 1.2x
extension probes until the cap or the first failure, since a boundary
that binds near the ceiling can be conservative. Probe requests carry
``skip_cache_store`` and the bench clears the model's paged cache
between probes, so nothing leaks into the prefix/SSD cache tiers.
"""

import asyncio
import contextlib
import gc
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, field_validator

from ..exceptions import PrefillMemoryAbortedError, PrefillMemoryExceededError
from ..prefill_progress import get_prefill_tracker
from .benchmark import _generate_prompt

logger = logging.getLogger(__name__)

# Module-level storage for context benchmark runs
_context_runs: dict[str, "ContextBenchmarkRun"] = {}

# Valid measurement targets (16k .. 512k)
VALID_TARGET_TOKENS = [16384, 32768, 65536, 131072, 262144, 524288]

# Applied values are floored to this granularity (user-facing "2k units").
_APPLY_GRANULARITY = 2048

# Smallest boundary worth reporting; below this the machine cannot hold a
# usable context for the model at all.
_MIN_USEFUL_TOKENS = 2048

# Calibration prefill size — seeds the transient tracker EWMA and the
# GDN/Mamba fixed-state measurement before the analytic search.
_CALIBRATION_TOKENS = 4096

# Two real-prefill attempts total: the boundary try, then one
# conservative retry informed by where the first prefill actually
# aborted. More rounds add minutes of near-ceiling crawling for
# marginal precision. Instant admission rejections (nothing prefilled)
# do not consume an attempt; the spin cap bounds those instead.
_MAX_VERIFY_ATTEMPTS = 2
_MAX_VERIFY_SPINS = 6

# Retry candidate = this fraction of the token count the aborted prefill
# actually completed — physical evidence of what fits, with headroom.
_ABORT_EVIDENCE_SAFETY = 0.9

# When the verify completed straight at the analytic boundary (no abort
# evidence, memory-bound), extension probes climb in steps of this
# factor until the cap or the first failure — a conservative boundary
# gets the chance to prove more. Only COMPLETED extensions are ever
# applied; a failed one keeps the last completed value.
_EXTENSION_FACTOR = 1.2

_CTX_TERMINAL_TYPES = frozenset({"done", "error"})

# Overall progress bands per phase (0-100).
_PROGRESS_BANDS = {
    "prepare": (0.0, 8.0),
    "calibrate": (8.0, 16.0),
    "estimate": (16.0, 20.0),
    "verify": (20.0, 95.0),
    "apply": (95.0, 100.0),
}


class ContextBenchmarkRequest(BaseModel):
    """Request model for starting a context benchmark."""

    model_id: str
    target_tokens: int = 131072

    @field_validator("target_tokens")
    @classmethod
    def validate_target_tokens(cls, v: int) -> int:
        if v not in VALID_TARGET_TOKENS:
            raise ValueError(
                f"Invalid target {v}. Must be one of {VALID_TARGET_TOKENS}"
            )
        return v


@dataclass
class ContextBenchmarkRun:
    """Tracks the state of a running context benchmark.

    Same SSE delivery model as ``BenchmarkRun``: events are appended to
    `events` under `cond`, subscribers replay from offset 0 then wait for
    new entries, and `terminal` closes the stream. `phase` / `progress` /
    `message` mirror the latest progress event so REST pollers (the
    native app) never need to parse the event log.
    """

    bench_id: str
    request: ContextBenchmarkRequest
    status: str = "running"  # running, completed, cancelled, error
    events: list[dict] = field(default_factory=list)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    terminal: bool = False
    task: asyncio.Task | None = None
    phase: str = "prepare"
    progress: float = 0.0
    message: str = ""
    result: dict | None = None
    error_message: str = ""


def get_run(bench_id: str) -> "ContextBenchmarkRun | None":
    """Get a context benchmark run by ID."""
    return _context_runs.get(bench_id)


def get_active_run() -> "ContextBenchmarkRun | None":
    """Return the currently-running context benchmark, if any."""
    for run in _context_runs.values():
        if run.status == "running":
            return run
    return None


def create_run(request: ContextBenchmarkRequest) -> ContextBenchmarkRun:
    """Create and register a new context benchmark run."""
    bench_id = f"ctx-{uuid.uuid4().hex[:12]}"
    run = ContextBenchmarkRun(bench_id=bench_id, request=request)
    _context_runs[bench_id] = run
    return run


def cleanup_old_runs(max_runs: int = 10) -> None:
    """Remove old completed runs to prevent memory leaks."""
    completed = [
        (bid, r)
        for bid, r in _context_runs.items()
        if r.status in ("completed", "cancelled", "error")
    ]
    if len(completed) > max_runs:
        for bid, _ in completed[:-max_runs]:
            del _context_runs[bid]


async def _send_event(run: ContextBenchmarkRun, event: dict) -> None:
    """Append an event, mirror progress state, wake subscribers."""
    async with run.cond:
        run.events.append(event)
        if event.get("type") == "progress":
            run.phase = event.get("phase", run.phase)
            run.progress = float(event.get("progress", run.progress))
            run.message = event.get("message", run.message)
        if event.get("type") in _CTX_TERMINAL_TYPES:
            run.terminal = True
        run.cond.notify_all()


async def _progress(
    run: ContextBenchmarkRun, phase: str, fraction: float, message: str
) -> None:
    """Emit a progress event mapped into the phase's overall band."""
    lo, hi = _PROGRESS_BANDS.get(phase, (0.0, 100.0))
    overall = lo + (hi - lo) * min(1.0, max(0.0, fraction))
    await _send_event(
        run,
        {
            "type": "progress",
            "phase": phase,
            "progress": round(overall, 1),
            "message": message,
        },
    )


def floor_to_apply_granularity(tokens: int) -> int:
    """Floor a token count to the applied 2k granularity."""
    return (max(0, tokens) // _APPLY_GRANULARITY) * _APPLY_GRANULARITY


def ceil_to_apply_granularity(tokens: int) -> int:
    """Ceil a token count to the applied 2k granularity."""
    return -(-max(0, tokens) // _APPLY_GRANULARITY) * _APPLY_GRANULARITY


def bisect_admission(fits: Callable[[int], bool], lo: int, hi: int) -> int:
    """Largest n in [lo, hi] with fits(n) True; 0 if even lo fails.

    Assumes fits is monotone non-increasing over n (the admission
    formula is: estimate grows with n, limit is fixed).
    """
    if hi < lo:
        return 0
    if not fits(lo):
        return 0
    if fits(hi):
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _resolve_scheduler(engine: Any) -> Any | None:
    """Pool engine → async engine → core → scheduler (None when absent)."""
    async_core = getattr(engine, "_engine", None)
    core = getattr(async_core, "engine", None) if async_core is not None else None
    return getattr(core, "scheduler", None) if core is not None else None


def _resolve_core(engine: Any) -> Any | None:
    async_core = getattr(engine, "_engine", None)
    return getattr(async_core, "engine", None) if async_core is not None else None


def _guard_ready(scheduler: Any) -> bool:
    """preflight_or_raise no-ops unless all three of these hold."""
    return bool(
        getattr(scheduler, "_prefill_memory_guard", False)
        and getattr(scheduler, "_memory_hard_limit_bytes", 0) > 0
        and getattr(scheduler, "memory_monitor", None) is not None
    )


def _make_fits(scheduler: Any) -> Callable[[int], bool]:
    def fits(n: int) -> bool:
        try:
            scheduler.preflight_or_raise(num_prompt_tokens=n, text_only=True)
        except PrefillMemoryExceededError:
            return False
        return True

    return fits


def _clear_probe_residue(scheduler: Any) -> None:
    """Release probe KV blocks and pooled Metal buffers.

    Runs ON the engine's MLX executor thread (thread-local stream +
    serialization with scheduler steps). The bench loads its model fresh
    with everything else unloaded, so clearing the paged cache only drops
    probe blocks — there is no user prefix cache to lose.
    """
    from ..scheduler import _sync_and_clear_cache

    cache = getattr(scheduler, "block_aware_cache", None)
    if cache is not None:
        try:
            cache.clear()
        except Exception as exc:
            logger.debug("Context bench: paged cache clear failed: %s", exc)
    gc.collect()
    _sync_and_clear_cache(getattr(scheduler, "_stream", None))
    # Refresh the cached mlx-active sample so event-loop preflight probes
    # see post-cleanup usage instead of the prefill-peak snapshot.
    try:
        scheduler._current_usage_bytes()
    except Exception as exc:
        logger.debug("Context bench: usage refresh failed: %s", exc)


async def _cleanup_between_probes(engine: Any, scheduler: Any) -> None:
    """Dispatch probe-residue cleanup to the engine's executor and settle."""
    core = _resolve_core(engine)
    executor = getattr(core, "_mlx_executor", None)
    loop = asyncio.get_running_loop()
    for _ in range(2):
        if executor is not None:
            await loop.run_in_executor(executor, _clear_probe_residue, scheduler)
        await asyncio.sleep(0.5)


async def _run_probe_prefill(engine: Any, prompt: str) -> tuple[Any, float]:
    """Prefill the prompt with max_tokens=1; return (final output, seconds).

    Raises PrefillMemoryExceededError / PrefillMemoryAbortedError when the
    memory guard refuses or aborts the probe. With max_tokens=1 the wall
    clock is dominated by the prefill, so callers can derive a prefill
    tok/s from it when the engine does not report prompt_tps.
    """
    started = time.perf_counter()
    last_output = None
    async for output in engine.stream_generate(
        prompt=prompt,
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        skip_cache_store=True,
    ):
        last_output = output
    return last_output, time.perf_counter() - started


def next_verify_candidate(
    candidate: int, observed_processed: int, new_boundary: int
) -> int:
    """Conservative retry candidate after a failed verify prefill.

    Physical evidence caps from above: the aborted prefill completed
    ``observed_processed`` tokens before dying, so 90% of that is the
    most a retry should attempt. The re-measured boundary caps too —
    the failure-path cleanup resets the transient tracker first, so the
    re-bisect reflects honest KV/static pricing at the current baseline
    rather than the dead prefill's contaminated last-chunk delta.
    Without evidence (nothing prefilled), halve.
    """
    if observed_processed > 0:
        nxt = min(
            int(observed_processed * _ABORT_EVIDENCE_SAFETY),
            candidate - _APPLY_GRANULARITY,
        )
    else:
        nxt = min(candidate // 2, candidate - _APPLY_GRANULARITY)
    if new_boundary > 0:
        nxt = min(nxt, new_boundary)
    return floor_to_apply_granularity(nxt)


def _reset_transient_tracker(scheduler: Any) -> None:
    """Drop the scheduler's chunk-transient measurements after a failed probe.

    An enforcer-killed prefill leaves a giant last-chunk phys delta in the
    tracker; under speed priority that poisons every subsequent admission
    (a 2048-token charge of several GB), and since rejected probes never
    run a chunk, the poison would never decay within the bench. Resetting
    returns admission to the static + exact-KV estimate until real chunks
    re-seed the EWMA.
    """
    tracker = getattr(scheduler, "_prefill_transient_tracker", None)
    if tracker is not None:
        try:
            tracker.reset()
        except Exception as exc:
            logger.debug("Context bench: tracker reset failed: %s", exc)


async def _relay_prefill_progress(
    run: ContextBenchmarkRun,
    model_id: str,
    attempt: int,
    candidate: int,
    progress_holder: dict,
) -> None:
    """Poll the global prefill tracker and mirror % into verify progress.

    Also records the last seen processed-token count into
    ``progress_holder`` so a failed attempt can derive its retry
    candidate from where the prefill actually died.
    """
    tracker = get_prefill_tracker()
    last_emitted = -1.0
    while True:
        await asyncio.sleep(1.5)
        entries = tracker.get_model_progress(model_id)
        if not entries:
            continue
        entry = entries[0]
        progress_holder["processed"] = max(
            int(progress_holder.get("processed", 0)), int(entry["processed"])
        )
        total = max(1, entry["total"])
        fraction = entry["processed"] / total
        if fraction - last_emitted < 0.01:
            continue
        last_emitted = fraction
        message = (
            f"Verify prefill {candidate:,} tokens (attempt {attempt}): "
            f"{entry['processed']:,}/{entry['total']:,}"
        )
        details = []
        speed = entry.get("speed")
        if speed:
            details.append(f"{speed:,.0f} tok/s")
        eta = entry.get("eta")
        if eta is not None:
            details.append(f"~{int(eta)}s left")
        if details:
            message += f" ({', '.join(details)})"
        await _progress(run, "verify", fraction, message)


async def run_context_benchmark(run: ContextBenchmarkRun, engine_pool: Any) -> None:
    """Execute a context benchmark run.

    Phases: prepare (unload all → load target) → calibrate → estimate
    (bisect admission) → verify (real prefill, step down on abort) →
    apply (write max_context_window) → cleanup (unload).
    """
    request = run.request
    overall_start = time.perf_counter()

    try:
        # Phase 1: prepare — unload everything, load the target model.
        loaded_ids = engine_pool.get_loaded_model_ids()
        if loaded_ids:
            await _progress(
                run, "prepare", 0.1, f"Unloading {len(loaded_ids)} model(s)..."
            )
            for model_id in loaded_ids:
                try:
                    await engine_pool._unload_engine(model_id)
                    logger.info("Context bench: unloaded %s", model_id)
                except Exception as exc:
                    logger.warning(
                        "Context bench: failed to unload %s: %s", model_id, exc
                    )

        await _progress(run, "prepare", 0.4, f"Loading {request.model_id}...")
        engine = await engine_pool.get_engine(request.model_id)
        logger.info("Context bench: loaded %s", request.model_id)

        scheduler = _resolve_scheduler(engine)
        if scheduler is None:
            raise RuntimeError(
                "This model's engine does not expose a scheduler, so the "
                "context benchmark cannot probe its admission boundary. "
                "Disable speculative engine features (e.g. DFlash) for this "
                "model and retry."
            )
        if not _guard_ready(scheduler):
            raise RuntimeError(
                "The prefill memory guard is disabled, so there is no "
                "admission boundary to measure. Enable Memory Guard and "
                "retry."
            )

        entry = engine_pool.get_entry(request.model_id)
        native = getattr(entry, "model_context_length", None) or 0
        cap = request.target_tokens
        cap_source = "target"
        if 0 < native < cap:
            cap = native
            cap_source = "native"

        # Phase 2: calibrate — JIT warmup, then one real mid-size prefill
        # to seed the transient tracker EWMA and the GDN fixed-state probe.
        tokenizer = engine.tokenizer
        await _progress(run, "calibrate", 0.1, "Warming up (JIT compile)...")
        async for _ in engine.stream_generate(
            prompt=_generate_prompt(tokenizer, 32),
            max_tokens=8,
            temperature=0.0,
            skip_cache_store=True,
        ):
            pass

        calibration_tokens = min(_CALIBRATION_TOKENS, max(1024, cap // 2))
        await _progress(
            run,
            "calibrate",
            0.4,
            f"Calibration prefill ({calibration_tokens:,} tokens)...",
        )
        try:
            await _run_probe_prefill(
                engine, _generate_prompt(tokenizer, calibration_tokens)
            )
        except (PrefillMemoryExceededError, PrefillMemoryAbortedError) as exc:
            raise RuntimeError(
                f"Not enough memory to prefill even {calibration_tokens:,} "
                f"tokens on this machine: {exc}"
            ) from exc
        await _cleanup_between_probes(engine, scheduler)

        # Phase 3: estimate — bisect the deterministic admission boundary.
        await _progress(run, "estimate", 0.3, "Estimating admission boundary...")
        fits = _make_fits(scheduler)
        boundary = bisect_admission(fits, 1024, cap)
        if floor_to_apply_granularity(boundary) < _MIN_USEFUL_TOKENS:
            raise RuntimeError(
                "Admission boundary is below 2k tokens — not enough free "
                "memory to serve a usable context for this model."
            )
        logger.info(
            "Context bench: admission boundary %d tokens (cap %d, %s)",
            boundary,
            cap,
            cap_source,
        )

        # Phase 4: verify — real prefill at the floored candidate. On a
        # mid-prefill abort, one conservative retry sized from where the
        # first prefill actually died (90% of its processed tokens). An
        # INSTANT admission rejection (0 tokens processed — current drifted
        # between the bisection and the probe) costs no GPU time and does
        # not consume a real-prefill attempt; it just re-bisects with the
        # fresh usage and retries, bounded by the spin cap.
        candidate = floor_to_apply_granularity(min(boundary, cap))
        verified = 0
        verified_prompt_tokens = 0
        verify_prefill_tps = 0.0
        attempts = 0
        spins = 0
        had_abort = False
        while spins < _MAX_VERIFY_SPINS:
            spins += 1
            attempt_no = attempts + 1
            await _progress(
                run,
                "verify",
                0.0,
                f"Verify prefill {candidate:,} tokens (attempt {attempt_no})...",
            )
            progress_holder: dict = {}
            relay = asyncio.create_task(
                _relay_prefill_progress(
                    run, request.model_id, attempt_no, candidate, progress_holder
                )
            )
            try:
                output, probe_seconds = await _run_probe_prefill(
                    engine, _generate_prompt(tokenizer, candidate)
                )
            except (PrefillMemoryExceededError, PrefillMemoryAbortedError) as exc:
                observed = int(progress_holder.get("processed", 0))
                instant_reject = (
                    not isinstance(exc, PrefillMemoryAbortedError) and observed == 0
                )
                if not instant_reject:
                    attempts += 1
                    had_abort = True
                logger.info(
                    "Context bench: verify failed at %d tokens "
                    "(attempt %d, %d processed%s): %s",
                    candidate,
                    attempt_no,
                    observed,
                    ", instant reject" if instant_reject else "",
                    exc,
                )
                if attempts >= _MAX_VERIFY_ATTEMPTS:
                    raise RuntimeError(
                        f"Verification failed {_MAX_VERIFY_ATTEMPTS} times — "
                        f"the machine could not complete a prefill at the "
                        f"measured sizes. Raise the Memory Guard ceiling or "
                        f"pick a smaller target and rerun."
                    ) from exc
                await _cleanup_between_probes(engine, scheduler)
                # Drop the dead prefill's contaminated transient sample so
                # the re-bisect prices the next candidate honestly.
                _reset_transient_tracker(scheduler)
                new_boundary = bisect_admission(fits, 1024, candidate)
                if instant_reject:
                    # Free retry: one grain below the freshly measured
                    # boundary.
                    candidate = floor_to_apply_granularity(
                        min(new_boundary, candidate - _APPLY_GRANULARITY)
                    )
                else:
                    candidate = next_verify_candidate(candidate, observed, new_boundary)
                if candidate < _MIN_USEFUL_TOKENS:
                    raise RuntimeError(
                        "Verification kept failing above the 2k floor — "
                        "not enough stable memory for a usable context."
                    ) from exc
                continue
            finally:
                relay.cancel()
            attempts += 1
            verified = candidate
            verified_prompt_tokens = int(
                getattr(output, "prompt_tokens", 0) or candidate
            )
            # Prefill speed of the successful verify: prefer the engine's
            # own figure, fall back to the probe wall clock (max_tokens=1,
            # so it is essentially all prefill).
            verify_prefill_tps = float(getattr(output, "prompt_tps", 0.0) or 0.0)
            if verify_prefill_tps <= 0 and probe_seconds > 0:
                verify_prefill_tps = verified_prompt_tokens / probe_seconds
            break
        if verified == 0:
            raise RuntimeError(
                "Verification did not converge within the retry budget. "
                "Raise the Memory Guard ceiling or pick a smaller target "
                "and rerun."
            )

        # Extension probes: the verified value came straight from the
        # analytic boundary (clean first-try completion, memory-bound).
        # Near the ceiling the boundary can be conservative, so keep
        # probing at 1.2x steps until the cap is reached or a probe
        # fails. Every probe re-passes the live admission gate at the
        # settled baseline, and only COMPLETED prefills raise the value;
        # the first rejection or abort ends the climb and keeps the last
        # completed size.
        extended = False
        extension_aborted = False
        while not had_abort and verified < floor_to_apply_granularity(cap):
            ext_target = min(
                ceil_to_apply_granularity(int(verified * _EXTENSION_FACTOR)), cap
            )
            if ext_target <= verified:
                break
            await _cleanup_between_probes(engine, scheduler)
            await _progress(
                run,
                "verify",
                0.98,
                f"Extension probe {ext_target:,} tokens " f"({_EXTENSION_FACTOR}x)...",
            )
            progress_holder = {}
            relay = asyncio.create_task(
                _relay_prefill_progress(
                    run,
                    request.model_id,
                    attempts + 1,
                    ext_target,
                    progress_holder,
                )
            )
            try:
                output, probe_seconds = await _run_probe_prefill(
                    engine, _generate_prompt(tokenizer, ext_target)
                )
            except (
                PrefillMemoryExceededError,
                PrefillMemoryAbortedError,
            ) as exc:
                observed = int(progress_holder.get("processed", 0))
                if observed > 0 or isinstance(exc, PrefillMemoryAbortedError):
                    attempts += 1
                    extension_aborted = True
                logger.info(
                    "Context bench: extension probe failed at %d tokens "
                    "(%d processed): %s",
                    ext_target,
                    observed,
                    exc,
                )
                await _cleanup_between_probes(engine, scheduler)
                _reset_transient_tracker(scheduler)
                break
            else:
                attempts += 1
                extended = True
                verified = ext_target
                verified_prompt_tokens = int(
                    getattr(output, "prompt_tokens", 0) or ext_target
                )
                verify_prefill_tps = float(getattr(output, "prompt_tps", 0.0) or 0.0)
                if verify_prefill_tps <= 0 and probe_seconds > 0:
                    verify_prefill_tps = verified_prompt_tokens / probe_seconds
                logger.info(
                    "Context bench: extension probe completed at %d tokens",
                    ext_target,
                )
            finally:
                relay.cancel()

        # Phase 5: apply — re-measure with the post-verify tracker state
        # (the near-ceiling prefill can raise the floor-chunk transient
        # charge) and write the tighter of the two, floored to 2k.
        await _progress(run, "apply", 0.2, "Re-checking admission boundary...")
        await _cleanup_between_probes(engine, scheduler)
        post_boundary = bisect_admission(fits, 1024, verified)
        final = verified
        if post_boundary > 0:
            # The re-bisect can be contaminated by the verify prefill's own
            # residue (buffer pool, last-chunk transient sample) — never
            # tighten below 90% of what physically completed. After a
            # FAILED extension probe the contamination is fresh and the
            # verified size completed moments ago, so it stands as-is.
            if extension_aborted:
                evidence_floor = floor_to_apply_granularity(verified)
            else:
                evidence_floor = floor_to_apply_granularity(
                    int(verified * _ABORT_EVIDENCE_SAFETY)
                )
            final = min(
                final,
                max(floor_to_apply_granularity(post_boundary), evidence_floor),
            )
        final = floor_to_apply_granularity(final)
        if final < _MIN_USEFUL_TOKENS:
            final = _MIN_USEFUL_TOKENS

        capped_by = "memory"
        if final >= floor_to_apply_granularity(cap):
            capped_by = cap_source

        applied = False
        sm = getattr(engine_pool, "_settings_manager", None)
        if sm is not None:
            try:
                settings = sm.get_settings(request.model_id)
                settings.max_context_window = final
                sm.set_settings(request.model_id, settings)
                applied = True
                logger.info(
                    "Context bench: applied max_context_window=%d for %s",
                    final,
                    request.model_id,
                )
            except Exception as exc:
                logger.warning(
                    "Context bench: failed to apply setting for %s: %s",
                    request.model_id,
                    exc,
                )
        await _progress(run, "apply", 0.8, "Applying context window setting...")

        result = {
            "model_id": request.model_id,
            "target_tokens": request.target_tokens,
            "native_context_length": native or None,
            "measured_tokens": boundary,
            "verified_tokens": verified,
            "verified_prompt_tokens": verified_prompt_tokens,
            "applied_tokens": final,
            "applied": applied,
            "capped_by": capped_by,
            "attempts": attempts,
            # True when the 1.2x extension probe completed and raised the
            # verified value beyond the analytic boundary.
            "extended": extended,
            # Prefill tok/s of the successful verify run — what a prompt at
            # the applied size actually prefills at in the current mode.
            "prefill_tps": round(verify_prefill_tps, 1),
            "duration_s": round(time.perf_counter() - overall_start, 1),
            # Mode the measurement ran under — the applied value only holds
            # while serving keeps the same prefill priority.
            "prefill_priority": (
                "speed"
                if getattr(scheduler, "_prefill_speed_priority", False)
                else "context"
            ),
        }
        run.result = result
        await _send_event(run, {"type": "result", "data": result})

        # Phase 6: cleanup — unload the bench model (throughput parity).
        try:
            await engine_pool._unload_engine(request.model_id)
            logger.info("Context bench: unloaded %s after run", request.model_id)
        except Exception as exc:
            logger.warning(
                "Context bench: failed to unload %s: %s", request.model_id, exc
            )

        run.status = "completed"
        await _send_event(
            run,
            {
                "type": "done",
                "summary": {
                    "model_id": request.model_id,
                    "applied_tokens": final,
                    "total_time": result["duration_s"],
                },
            },
        )

    except asyncio.CancelledError:
        run.status = "cancelled"
        run.error_message = "Context benchmark cancelled by user"
        await _send_event(run, {"type": "error", "message": run.error_message})
        with contextlib.suppress(Exception):
            await engine_pool._unload_engine(request.model_id)

    except Exception as exc:
        logger.error("Context bench error: %s", exc, exc_info=True)
        run.status = "error"
        run.error_message = str(exc)
        await _send_event(run, {"type": "error", "message": str(exc)})
        with contextlib.suppress(Exception):
            await engine_pool._unload_engine(request.model_id)
