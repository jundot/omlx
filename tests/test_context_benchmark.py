# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin context benchmark module."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.admin.context_benchmark import (
    VALID_TARGET_TOKENS,
    ContextBenchmarkRequest,
    ContextBenchmarkRun,
    bisect_admission,
    cleanup_old_runs,
    create_run,
    floor_to_apply_granularity,
    get_active_run,
    get_run,
    next_verify_candidate,
    run_context_benchmark,
)
from omlx.exceptions import PrefillMemoryAbortedError, PrefillMemoryExceededError

# =============================================================================
# Request validation
# =============================================================================


class TestContextBenchmarkRequest:
    def test_valid_request(self):
        req = ContextBenchmarkRequest(model_id="m", target_tokens=65536)
        assert req.target_tokens == 65536

    def test_default_target_is_128k(self):
        req = ContextBenchmarkRequest(model_id="m")
        assert req.target_tokens == 131072

    def test_invalid_target_rejected(self):
        with pytest.raises(ValueError, match="Invalid target 100000"):
            ContextBenchmarkRequest(model_id="m", target_tokens=100000)

    def test_all_documented_targets_accepted(self):
        for t in VALID_TARGET_TOKENS:
            assert ContextBenchmarkRequest(model_id="m", target_tokens=t)


# =============================================================================
# Pure helpers
# =============================================================================


class TestFloorToApplyGranularity:
    def test_floors_to_2k(self):
        assert floor_to_apply_granularity(50000) == 49152
        assert floor_to_apply_granularity(49152) == 49152
        assert floor_to_apply_granularity(2047) == 0
        assert floor_to_apply_granularity(0) == 0
        assert floor_to_apply_granularity(-5) == 0


class TestBisectAdmission:
    def test_all_fit_returns_hi(self):
        assert bisect_admission(lambda n: True, 1024, 131072) == 131072

    def test_none_fit_returns_zero(self):
        assert bisect_admission(lambda n: False, 1024, 131072) == 0

    def test_exact_boundary(self):
        assert bisect_admission(lambda n: n <= 50000, 1024, 131072) == 50000

    def test_boundary_at_lo(self):
        assert bisect_admission(lambda n: n <= 1024, 1024, 131072) == 1024

    def test_hi_below_lo_returns_zero(self):
        assert bisect_admission(lambda n: True, 1024, 512) == 0

    def test_probe_count_is_logarithmic(self):
        calls = []

        def fits(n):
            calls.append(n)
            return n <= 77777

        assert bisect_admission(fits, 1024, 524288) == 77777
        assert len(calls) <= 24


class TestNextVerifyCandidate:
    def test_uses_abort_point_evidence(self):
        # Died at 45,000 processed of a 131,072 attempt: 90% of the
        # abort point, floored to 2k.
        assert next_verify_candidate(131072, 45000, 0) == 38912

    def test_re_measured_boundary_caps_evidence(self):
        # The failure path resets the transient tracker before re-bisecting,
        # so the boundary is honest and caps the evidence candidate:
        # min(0.9 * 64000 = 57600, 63488, 40960) -> 40960.
        assert next_verify_candidate(65536, 64000, 40960) == 40960

    def test_no_evidence_honors_re_measured_boundary(self):
        assert next_verify_candidate(65536, 0, 20480) == 20480

    def test_no_evidence_halves(self):
        assert next_verify_candidate(49152, 0, 0) == 24576

    def test_always_strictly_below_failed_candidate(self):
        # Evidence near the candidate still steps down at least one grain:
        # min(90% of 8192 = 7372, 8192 - 2048) -> floor2k -> 6144.
        assert next_verify_candidate(8192, 8192, 0) == 6144


# =============================================================================
# Run registry
# =============================================================================


class TestRunRegistry:
    def test_create_get_active(self):
        run = create_run(ContextBenchmarkRequest(model_id="m"))
        assert run.bench_id.startswith("ctx-")
        assert get_run(run.bench_id) is run
        assert get_active_run() is run
        run.status = "completed"
        assert get_active_run() is None

    def test_cleanup_old_runs(self):
        from omlx.admin.context_benchmark import _context_runs

        _context_runs.clear()
        for _ in range(15):
            run = create_run(ContextBenchmarkRequest(model_id="m"))
            run.status = "completed"
        cleanup_old_runs(max_runs=10)
        assert len(_context_runs) == 10
        _context_runs.clear()


# =============================================================================
# Runner fakes
# =============================================================================


class _FakeTokenizer:
    def encode(self, text):
        return list(range(len(text) // 4))

    def decode(self, tokens):
        return "x" * (len(tokens) * 4)


class _FakeScheduler:
    """preflight_or_raise passes while n <= boundary."""

    def __init__(self, boundary=50000, guard=True):
        self.boundary = boundary
        self._prefill_memory_guard = guard
        self._memory_hard_limit_bytes = 10 * 1024**3 if guard else 0
        self.memory_monitor = object() if guard else None
        self.block_aware_cache = None
        self._stream = None
        self._prefill_transient_tracker = MagicMock()

    def preflight_or_raise(
        self,
        *,
        num_prompt_tokens,
        cached_tokens=0,
        request_id=None,
        text_only=False,
    ):
        if num_prompt_tokens > self.boundary:
            raise PrefillMemoryExceededError(
                message="too big",
                request_id=request_id or "probe",
                estimated_bytes=num_prompt_tokens,
                limit_bytes=self.boundary,
            )


class _FakeEngine:
    """stream_generate consumes one entry of probe_plan per probe call
    (max_tokens == 1). None = success, an exception instance = raised."""

    def __init__(self, scheduler, probe_plan=None, prompt_tokens=12345):
        self.tokenizer = _FakeTokenizer()
        self._engine = SimpleNamespace(
            engine=SimpleNamespace(scheduler=scheduler, _mlx_executor=None)
        )
        self.probe_plan = list(probe_plan or [])
        self.prompt_tokens = prompt_tokens
        self.probe_calls = []

    async def stream_generate(self, **kwargs):
        if kwargs.get("max_tokens") == 1:
            self.probe_calls.append(kwargs)
            if self.probe_plan:
                planned = self.probe_plan.pop(0)
                if planned is not None:
                    raise planned
        yield SimpleNamespace(
            completion_tokens=1,
            prompt_tokens=self.prompt_tokens,
            prompt_tps=1234.5,
            cached_tokens=0,
            new_text="x",
            finished=True,
            finish_reason="length",
        )


class _FakeSettingsManager:
    def __init__(self):
        self.settings = SimpleNamespace(max_context_window=None)
        self.applied = []

    def get_settings(self, model_id):
        return self.settings

    def set_settings(self, model_id, settings):
        self.applied.append((model_id, settings.max_context_window))


class _FakePool:
    def __init__(self, engine, native=0, loaded=None):
        self._engine = engine
        self._settings_manager = _FakeSettingsManager()
        self.native = native
        self.loaded = list(loaded or [])
        self.unloaded = []

    def get_loaded_model_ids(self):
        return list(self.loaded)

    async def get_engine(self, model_id, force_lm=False):
        return self._engine

    async def _unload_engine(self, model_id):
        self.unloaded.append(model_id)

    def get_entry(self, model_id):
        return SimpleNamespace(model_context_length=self.native, model_type="llm")


def _make_run(target_tokens=131072):
    return ContextBenchmarkRun(
        bench_id="ctx-test",
        request=ContextBenchmarkRequest(
            model_id="test-model", target_tokens=target_tokens
        ),
    )


async def _run_bench(run, pool):
    with patch("omlx.admin.context_benchmark._cleanup_between_probes", AsyncMock()):
        await run_context_benchmark(run, pool)


# =============================================================================
# Runner tests
# =============================================================================


class TestRunContextBenchmark:
    @pytest.mark.asyncio
    async def test_happy_path_applies_floored_boundary(self):
        scheduler = _FakeScheduler(boundary=50000)
        abort = PrefillMemoryAbortedError(
            message="climb over",
            request_id="probe",
            estimated_bytes=1,
            limit_bytes=1,
        )
        # Calibration ok, verify at the boundary ok, extension 1
        # (ceil2k(58982) = 59392) ok, extension 2 (71680) aborts.
        engine = _FakeEngine(scheduler, probe_plan=[None, None, None, abort])
        pool = _FakePool(engine, loaded=["other-model"])
        run = _make_run()

        await _run_bench(run, pool)

        assert run.status == "completed"
        assert run.result is not None
        assert run.result["measured_tokens"] == 50000
        # The last COMPLETED size wins; the aborted climb keeps it as-is.
        assert run.result["verified_tokens"] == 59392
        assert run.result["extended"] is True
        assert run.result["applied_tokens"] == 59392
        assert run.result["verified_prompt_tokens"] == 12345
        assert run.result["prefill_tps"] == 1234.5
        assert run.result["capped_by"] == "memory"
        assert run.result["attempts"] == 3
        assert run.result["applied"] is True
        assert pool._settings_manager.applied == [("test-model", 59392)]
        # Both the pre-bench sweep and the post-bench cleanup unload.
        assert pool.unloaded == ["other-model", "test-model"]
        assert run.events[-1]["type"] == "done"
        assert run.terminal is True

    @pytest.mark.asyncio
    async def test_probes_carry_skip_cache_store(self):
        scheduler = _FakeScheduler(boundary=50000)
        engine = _FakeEngine(scheduler)
        pool = _FakePool(engine)
        run = _make_run()

        await _run_bench(run, pool)

        assert engine.probe_calls, "expected calibration + verify probes"
        assert all(c.get("skip_cache_store") for c in engine.probe_calls)

    @pytest.mark.asyncio
    async def test_capped_by_target(self):
        scheduler = _FakeScheduler(boundary=10**9)
        engine = _FakeEngine(scheduler)
        pool = _FakePool(engine)
        run = _make_run(target_tokens=16384)

        await _run_bench(run, pool)

        assert run.status == "completed"
        assert run.result["applied_tokens"] == 16384
        assert run.result["capped_by"] == "target"
        assert run.result["extended"] is False
        # Target-capped runs never probe beyond the cap: calib + verify.
        assert len(engine.probe_calls) == 2
        # No failure -> the tracker's measurements are kept.
        scheduler._prefill_transient_tracker.reset.assert_not_called()

    @pytest.mark.asyncio
    async def test_capped_by_native_context_length(self):
        scheduler = _FakeScheduler(boundary=10**9)
        engine = _FakeEngine(scheduler)
        pool = _FakePool(engine, native=20000)
        run = _make_run(target_tokens=131072)

        await _run_bench(run, pool)

        assert run.status == "completed"
        assert run.result["applied_tokens"] == 18432  # floor2k(20000)
        assert run.result["capped_by"] == "native"
        assert run.result["native_context_length"] == 20000

    @pytest.mark.asyncio
    async def test_verify_abort_steps_down_and_retries(self):
        scheduler = _FakeScheduler(boundary=50000)
        abort = PrefillMemoryAbortedError(
            message="mid-prefill abort",
            request_id="probe",
            estimated_bytes=1,
            limit_bytes=1,
        )
        # Plan: calibration ok, first verify aborts, second verify ok.
        engine = _FakeEngine(scheduler, probe_plan=[None, abort, None])
        pool = _FakePool(engine)
        run = _make_run()

        await _run_bench(run, pool)

        assert run.status == "completed"
        assert run.result["attempts"] == 2
        # 49152 aborted with no observed progress -> halve -> floor2k(24576)
        assert run.result["applied_tokens"] == 24576
        assert run.result["verified_tokens"] == 24576
        # An abort happened, so no extension probe: calib + 2 verifies.
        assert run.result["extended"] is False
        assert len(engine.probe_calls) == 3
        # The failure retry drops the dead prefill's transient poison.
        scheduler._prefill_transient_tracker.reset.assert_called()

    @pytest.mark.asyncio
    async def test_apply_rebisect_collapse_floored_by_verified_evidence(self):
        """A post-verify re-bisect contaminated by the probe's own residue
        must not drag the applied value below 90% of what completed."""
        scheduler = _FakeScheduler(boundary=50000)

        abort = PrefillMemoryAbortedError(
            message="climb over",
            request_id="probe",
            estimated_bytes=1,
            limit_bytes=1,
        )

        class _CollapsingEngine(_FakeEngine):
            async def stream_generate(self, **kwargs):
                async for out in super().stream_generate(**kwargs):
                    yield out
                # Probe 1 is calibration, probe 2 is the verify prefill —
                # collapse the boundary only after the verify completes.
                if len(self.probe_calls) >= 2:
                    scheduler.boundary = 1024

        engine = _CollapsingEngine(scheduler, probe_plan=[None, None, abort])
        pool = _FakePool(engine)
        run = _make_run()

        await _run_bench(run, pool)

        assert run.status == "completed"
        # The collapsed re-bisect (1024) must not drag the applied value
        # below the size that physically completed moments ago.
        assert run.result["verified_tokens"] == 49152
        assert run.result["applied_tokens"] == 49152

    @pytest.mark.asyncio
    async def test_instant_reject_does_not_consume_an_attempt(self):
        """A preflight rejection with nothing prefilled re-bisects and
        retries for free; only real prefills count against the cap."""
        scheduler = _FakeScheduler(boundary=50000)
        reject = PrefillMemoryExceededError(
            message="current drifted",
            request_id="probe",
            estimated_bytes=1,
            limit_bytes=1,
        )
        abort = PrefillMemoryAbortedError(
            message="climb over",
            request_id="probe",
            estimated_bytes=1,
            limit_bytes=1,
        )
        # Calibration ok, first verify instant-rejects, the free retry
        # succeeds, the extension climb aborts immediately.
        engine = _FakeEngine(scheduler, probe_plan=[None, reject, None, abort])
        pool = _FakePool(engine)
        run = _make_run()

        await _run_bench(run, pool)

        assert run.status == "completed"
        # Free retry ran one grain below the re-measured boundary; the
        # instant reject did not count as an attempt.
        assert run.result["attempts"] == 2
        assert run.result["verified_tokens"] == 47104
        assert run.result["applied_tokens"] == 47104

    @pytest.mark.asyncio
    async def test_extension_abort_keeps_verified_value(self):
        """A failed extension probe keeps the already-completed verify
        value; the fresh contamination must not shrink it either."""
        scheduler = _FakeScheduler(boundary=50000)
        abort = PrefillMemoryAbortedError(
            message="extension died",
            request_id="probe",
            estimated_bytes=1,
            limit_bytes=1,
        )
        # Calibration ok, verify at the boundary ok, extension aborts.
        engine = _FakeEngine(scheduler, probe_plan=[None, None, abort])
        pool = _FakePool(engine)
        run = _make_run()

        await _run_bench(run, pool)

        assert run.status == "completed"
        assert run.result["extended"] is False
        assert run.result["verified_tokens"] == 49152
        assert run.result["applied_tokens"] == 49152
        assert run.result["attempts"] == 2

    @pytest.mark.asyncio
    async def test_extension_climbs_until_cap(self):
        """Successful extensions keep multiplying by 1.2 until the cap
        (min(target, native)) is reached."""
        scheduler = _FakeScheduler(boundary=50000)
        engine = _FakeEngine(scheduler)
        pool = _FakePool(engine)
        run = _make_run(target_tokens=65536)

        await _run_bench(run, pool)

        assert run.status == "completed"
        assert run.result["extended"] is True
        # 49152 -> ceil2k(58982) = 59392 -> min(ceil2k(71270), 65536) =
        # 65536 = the cap; the climb stops there.
        assert run.result["verified_tokens"] == 65536
        assert run.result["attempts"] == 3
        # calib + verify + 2 extensions, no probe beyond the cap.
        assert len(engine.probe_calls) == 4

    @pytest.mark.asyncio
    async def test_two_verify_failures_error_out(self):
        scheduler = _FakeScheduler(boundary=50000)

        def abort():
            return PrefillMemoryAbortedError(
                message="mid-prefill abort",
                request_id="probe",
                estimated_bytes=1,
                limit_bytes=1,
            )

        # Calibration ok, both verify attempts abort.
        engine = _FakeEngine(scheduler, probe_plan=[None, abort(), abort()])
        pool = _FakePool(engine)
        run = _make_run()

        await _run_bench(run, pool)

        assert run.status == "error"
        assert "Raise the Memory Guard ceiling" in run.error_message
        assert pool._settings_manager.applied == []
        assert "test-model" in pool.unloaded

    @pytest.mark.asyncio
    async def test_guard_disabled_errors_out(self):
        scheduler = _FakeScheduler(guard=False)
        engine = _FakeEngine(scheduler)
        pool = _FakePool(engine)
        run = _make_run()

        await _run_bench(run, pool)

        assert run.status == "error"
        assert "memory guard" in run.error_message.lower()
        assert pool._settings_manager.applied == []
        assert "test-model" in pool.unloaded  # cleanup still runs

    @pytest.mark.asyncio
    async def test_missing_scheduler_errors_out(self):
        scheduler = _FakeScheduler()
        engine = _FakeEngine(scheduler)
        engine._engine = None
        pool = _FakePool(engine)
        run = _make_run()

        await _run_bench(run, pool)

        assert run.status == "error"
        assert "scheduler" in run.error_message.lower()

    @pytest.mark.asyncio
    async def test_calibration_failure_errors_out(self):
        scheduler = _FakeScheduler(boundary=50000)
        reject = PrefillMemoryExceededError(
            message="no room",
            request_id="probe",
            estimated_bytes=1,
            limit_bytes=1,
        )
        engine = _FakeEngine(scheduler, probe_plan=[reject])
        pool = _FakePool(engine)
        run = _make_run()

        await _run_bench(run, pool)

        assert run.status == "error"
        assert "Not enough memory" in run.error_message

    @pytest.mark.asyncio
    async def test_boundary_below_2k_errors_out(self):
        scheduler = _FakeScheduler(boundary=1500)
        engine = _FakeEngine(scheduler)
        pool = _FakePool(engine)
        run = _make_run()

        await _run_bench(run, pool)

        assert run.status == "error"
        assert "below 2k" in run.error_message

    @pytest.mark.asyncio
    async def test_cancellation_marks_cancelled_and_unloads(self):
        scheduler = _FakeScheduler(boundary=50000)
        started = asyncio.Event()

        class _HangingEngine(_FakeEngine):
            async def stream_generate(self, **kwargs):
                if kwargs.get("max_tokens") == 1:
                    started.set()
                    await asyncio.Event().wait()
                yield SimpleNamespace(
                    completion_tokens=1,
                    prompt_tokens=32,
                    cached_tokens=0,
                    new_text="x",
                    finished=True,
                    finish_reason="length",
                )

        engine = _HangingEngine(scheduler)
        pool = _FakePool(engine)
        run = _make_run()

        with patch("omlx.admin.context_benchmark._cleanup_between_probes", AsyncMock()):
            task = asyncio.create_task(run_context_benchmark(run, pool))
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            await task

        assert run.status == "cancelled"
        assert "test-model" in pool.unloaded
        assert run.terminal is True

    @pytest.mark.asyncio
    async def test_progress_state_mirrored_for_rest_polling(self):
        scheduler = _FakeScheduler(boundary=50000)
        engine = _FakeEngine(scheduler)
        pool = _FakePool(engine)
        run = _make_run()

        await _run_bench(run, pool)

        assert run.phase == "apply"
        assert run.progress > 90
        assert run.message
