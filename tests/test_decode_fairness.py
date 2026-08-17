# SPDX-License-Identifier: Apache-2.0
"""
Tests for decode fairness (SchedulerConfig.decode_fairness).

While decodes run (own engine or another engine on the shared GPU),
prefill is force-chunked, chunks are capped, and each chunk accrues a
decode time debt that must be repaid before the next chunk runs.
"""

from unittest.mock import MagicMock, patch

import pytest

from omlx.decode_activity import get_decode_activity
from omlx.scheduler import (
    _CONTENDED_PREFILL_CHUNK,
    Scheduler,
    SchedulerConfig,
)


@pytest.fixture(autouse=True)
def _quiet_decode_activity():
    from omlx.prefill_progress import get_prefill_tracker

    get_decode_activity().clear()
    get_prefill_tracker().clear()
    yield
    get_decode_activity().clear()
    get_prefill_tracker().clear()


def _make_scheduler(**config_kwargs) -> Scheduler:
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    config = SchedulerConfig(
        max_num_seqs=8,
        paged_cache_block_size=0,
        **config_kwargs,
    )
    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)
    mock_bg = MagicMock()
    mock_bg.insert.return_value = [42]
    mock_bg.next_generated.return_value = iter([])
    scheduler.batch_generator = mock_bg
    scheduler._current_sampler_params = ()
    return scheduler


class TestPrefillGate:
    def test_open_when_fairness_disabled(self):
        s = _make_scheduler(decode_fairness=False)
        s.running = {"r1": MagicMock()}
        s._decode_time_owed_s = 1.0
        assert s._prefill_gate_open()

    def test_open_and_debt_reset_when_no_decode_running(self):
        s = _make_scheduler()
        s._decode_time_owed_s = 1.0
        assert s._prefill_gate_open()
        assert s._decode_time_owed_s == 0.0

    def test_closed_while_debt_outstanding(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        s._decode_time_owed_s = 0.5
        assert not s._prefill_gate_open()
        s._repay_decode_debt(0.2)
        assert not s._prefill_gate_open()
        s._repay_decode_debt(0.4)
        assert s._prefill_gate_open()

    def test_accrue_only_while_contended(self):
        s = _make_scheduler()
        s._accrue_decode_debt(0.5)
        assert s._decode_time_owed_s == 0.0
        assert s._prefill_hold_until == 0.0
        s.running = {"r1": MagicMock()}
        s._accrue_decode_debt(0.5)
        assert s._decode_time_owed_s > 0.0

    def test_accrue_sets_hold_deadline_for_other_engines(self):
        import time

        s = _make_scheduler()
        get_decode_activity().publish("other-engine", 1)
        s._accrue_decode_debt(0.5)
        assert s._decode_time_owed_s == 0.0
        assert s._prefill_hold_until > time.perf_counter()
        assert not s._prefill_gate_open()

    def test_hold_deadline_expires(self):
        import time

        s = _make_scheduler()
        s._prefill_hold_until = time.perf_counter() - 0.01
        assert s._prefill_gate_open()

    def test_shared_hold_blocks_other_prefillers(self):
        import time

        # Engine A accrues a hold; engine B (a different scheduler with no
        # local hold) must pause too, or B's chunks cover A's hold window.
        a = _make_scheduler()
        b = _make_scheduler()
        get_decode_activity().publish("victim-engine", 1)
        a._accrue_decode_debt(0.5)
        assert time.perf_counter() < a._prefill_hold_until
        assert not b._prefill_gate_open()
        assert b._prefill_hold_until == 0.0  # local stays untouched

    def test_shared_hold_keeps_max(self):
        import time

        reg = get_decode_activity()
        now = time.perf_counter()
        reg.extend_hold(now + 2.0)
        reg.extend_hold(now + 1.0)  # shorter deadline must not shrink it
        assert reg.hold_until() == pytest.approx(now + 2.0)
        reg.clear()
        assert reg.hold_until() == 0.0

    def test_accrue_noop_when_fairness_disabled(self):
        s = _make_scheduler(decode_fairness=False)
        s.running = {"r1": MagicMock()}
        s._accrue_decode_debt(0.5)
        assert s._decode_time_owed_s == 0.0


class TestContendedChunkCap:
    def test_no_cap_without_contention(self):
        s = _make_scheduler()
        assert s._contended_prefill_cap() == 0
        assert s._prefill_step_size_for_progress(0, 100000) == 2048

    def test_cap_with_own_running_decode(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        assert s._contended_prefill_cap() == _CONTENDED_PREFILL_CHUNK
        assert (
            s._prefill_step_size_for_progress(0, 100000)
            == _CONTENDED_PREFILL_CHUNK
        )

    def test_cap_with_other_engine_decoding(self):
        s = _make_scheduler()
        get_decode_activity().publish("other-engine", 1)
        assert s._contended_prefill_cap() == _CONTENDED_PREFILL_CHUNK

    def test_no_cap_when_fairness_disabled(self):
        s = _make_scheduler(decode_fairness=False)
        s.running = {"r1": MagicMock()}
        assert s._contended_prefill_cap() == 0

    def test_cap_never_grows_small_steps(self):
        s = _make_scheduler(prefill_step_size=256)
        s.running = {"r1": MagicMock()}
        assert s._prefill_step_size_for_progress(0, 100000) == 256


class TestQwen35PrefillFloor:
    """Qwen3.5/3.6 chunk floor (measured +3.2% prefill at 4k on the 27B)."""

    def test_floor_applies(self):
        s = _make_scheduler()
        s._qwen35_prefill_floor = 4096
        assert s._prefill_step_size_for_progress(0, 100000) == 4096

    def test_contended_cap_still_wins(self):
        s = _make_scheduler()
        s._qwen35_prefill_floor = 4096
        s.running = {"r1": MagicMock()}
        assert (
            s._prefill_step_size_for_progress(0, 100000)
            == _CONTENDED_PREFILL_CHUNK
        )

    def test_non_qwen_model_unaffected(self):
        s = _make_scheduler()
        assert s._qwen35_prefill_floor == 0
        assert s._prefill_step_size_for_progress(0, 100000) == 2048


class TestAdaptiveChunkCap:
    """Contended chunks are sized by stall time x measured prefill tps."""

    def test_fallback_before_first_measurement(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        assert s._contended_prefill_cap() == _CONTENDED_PREFILL_CHUNK

    def test_cap_derives_from_measured_prefill_tps(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        # 500ms stall target -> 500 tokens, floored to the 64-token grid.
        s._prefill_tps_best = 1000.0
        assert s._contended_prefill_cap() == 448

    def test_cap_stays_on_64_grid(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        # Keep scheduler chunk sizing stable even though model-specific native
        # kernels now handle partial tiles internally.
        for tps in (594.0, 733.0, 999.0, 1601.0, 5000.0):
            s._prefill_tps_best = tps
            assert s._contended_prefill_cap() % 64 == 0

    def test_cap_floors_for_slow_prefill(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        s._prefill_tps_best = 100.0
        assert s._contended_prefill_cap() == 256

    def test_cap_ceils_at_step_size_for_fast_prefill(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        s._prefill_tps_best = 100000.0
        assert s._contended_prefill_cap() == 2048

    def test_decode_rate_sampling_solo_vs_contended(self):
        from omlx.prefill_progress import get_prefill_tracker

        s = _make_scheduler()
        s._sample_decode_rate(10, 0.1)  # no prefill anywhere -> solo
        assert s._solo_decode_tps_ema == pytest.approx(100.0)
        assert s._contended_decode_tps_ema is None
        get_prefill_tracker().update("r", 10, 100, "m")
        s._sample_decode_rate(10, 0.2)  # prefill live -> contended
        assert s._contended_decode_tps_ema == pytest.approx(50.0)
        assert s._solo_decode_tps_ema == pytest.approx(100.0)

    def test_decode_rate_buckets_microsecond_steps(self):
        s = _make_scheduler()
        # MTP queue pops: absurd instantaneous rates must not leak into
        # the EMA until >=100ms of decode wall time accumulates.
        for _ in range(3):
            s._sample_decode_rate(1, 0.00005)
        assert s._solo_decode_tps_ema is None
        s._sample_decode_rate(4, 0.1)  # bucket now 7 tok / 0.10015s
        assert s._solo_decode_tps_ema == pytest.approx(7 / 0.10015, rel=0.01)

    def test_prefill_tps_best_only_ratchets_up(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        s._prefill_tps_best = 1000.0
        assert s._contended_prefill_cap() == 448
        # A contended (slower) measurement must not shrink the cap.
        s._prefill_tps_best = max(s._prefill_tps_best, 400.0)
        assert s._contended_prefill_cap() == 448


class TestConditionalChunkClear:
    def test_clears_when_fairness_disabled(self):
        s = _make_scheduler(decode_fairness=False)
        assert s._should_clear_after_chunk()

    def test_clears_without_contention(self):
        s = _make_scheduler()
        assert s._should_clear_after_chunk()

    def test_clears_when_guard_off(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        s._memory_limit_bytes = 0
        assert s._should_clear_after_chunk()

    def test_skips_below_soft_watermark_under_contention(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        s._memory_limit_bytes = 100
        s._current_usage_bytes = lambda: 50
        assert not s._should_clear_after_chunk()

    def test_clears_at_soft_watermark(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        s._memory_limit_bytes = 100
        s._current_usage_bytes = lambda: 100
        assert s._should_clear_after_chunk()


class TestStepGating:
    def test_step_skips_chunk_advance_while_debt_outstanding(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        s.prefilling.append(MagicMock())
        s._decode_time_owed_s = 10.0
        with patch.object(s, "_advance_chunked_prefills") as advance:
            with patch.object(s, "_schedule_waiting", return_value=([], [])):
                s.step()
        advance.assert_not_called()

    def test_step_advances_chunks_when_debt_repaid(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        s.prefilling.append(MagicMock())
        s._decode_time_owed_s = 0.0
        with patch.object(s, "_advance_chunked_prefills") as advance:
            with patch.object(s, "_schedule_waiting", return_value=([], [])):
                s.step()
        advance.assert_called_once()

    def test_step_repays_debt_from_decode_wall_time(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        s._decode_time_owed_s = 10.0
        s.batch_generator.next_generated.return_value = iter([])
        with patch.object(s, "_schedule_waiting", return_value=([], [])):
            s.step()
        assert s._decode_time_owed_s < 10.0

    def test_chunk_only_step_reports_has_work(self):
        s = _make_scheduler()
        s.prefilling.append(MagicMock())
        with patch.object(s, "_advance_chunked_prefills"):
            with patch.object(s, "_schedule_waiting", return_value=([], [])):
                out = s.step()
        assert out.has_work

    def test_step_publishes_decode_activity(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        with patch.object(s, "_schedule_waiting", return_value=([], [])):
            s.step()
        assert get_decode_activity().others_decoding("someone-else")


class TestAdmissionDeferral:
    def test_waiting_deferred_while_debt_outstanding(self):
        s = _make_scheduler()
        s.running = {"r1": MagicMock()}
        s._decode_time_owed_s = 10.0
        s.waiting.append(MagicMock())
        scheduled, rejected = s._schedule_waiting()
        assert scheduled == []
        assert rejected == []
        assert len(s.waiting) == 1

    def test_waiting_deferred_while_holding_for_other_engine(self):
        import time

        s = _make_scheduler()
        s._prefill_hold_until = time.perf_counter() + 5.0
        s.waiting.append(MagicMock())
        scheduled, rejected = s._schedule_waiting()
        assert scheduled == []
        assert len(s.waiting) == 1


class TestHoldStepBehavior:
    def test_holding_step_reports_no_work(self):
        import time

        s = _make_scheduler()
        s.prefilling.append(MagicMock())
        s._prefill_hold_until = time.perf_counter() + 5.0
        with patch.object(s, "_advance_chunked_prefills") as advance:
            with patch.object(s, "_schedule_waiting", return_value=([], [])):
                out = s.step()
        advance.assert_not_called()
        assert not out.has_work
