# SPDX-License-Identifier: Apache-2.0
"""Tests for _StoreCacheGate.

The gate is a non-blocking counter that bounds how many KV caches are alive
in the post-completion store-cache pipeline. Backpressure is applied at
admission (_schedule_waiting declines new prefills while in_flight >= cap)
rather than by blocking the generation step, so cache persistence never
stalls token generation (#1496). The cap still bounds concurrent extracted-KV
copies, the OOM guard for the burst-finish RAM growth reported in #1383.
"""

import threading
import time

import pytest

from omlx.scheduler import Scheduler, _StoreCacheGate


class TestCounter:
    def test_note_submitted_increments(self):
        gate = _StoreCacheGate(cap=3)
        gate.note_submitted()
        gate.note_submitted()
        assert gate.in_flight == 2

    def test_note_done_decrements(self):
        gate = _StoreCacheGate(cap=3)
        gate.note_submitted()
        gate.note_submitted()
        gate.note_done()
        assert gate.in_flight == 1

    def test_note_done_does_not_underflow(self):
        gate = _StoreCacheGate(cap=2)
        gate.note_done()
        gate.note_done()
        assert gate.in_flight == 0

    def test_note_submitted_never_blocks_past_cap(self):
        """Submitting beyond cap must not block the caller (#1496).

        The generation step calls note_submitted() for a completed request
        even when the pipeline is already at cap; it records the slot and
        returns immediately rather than waiting on an SSD write.
        """
        gate = _StoreCacheGate(cap=1)
        gate.note_submitted()

        returned = threading.Event()

        def submit_again():
            gate.note_submitted()
            returned.set()

        t = threading.Thread(target=submit_again)
        t.start()
        assert returned.wait(1.0), "note_submitted blocked past cap"
        t.join()
        assert gate.in_flight == 2


class TestHasCapacity:
    def test_true_below_cap(self):
        gate = _StoreCacheGate(cap=2)
        assert gate.has_capacity is True
        gate.note_submitted()
        assert gate.has_capacity is True

    def test_false_at_cap(self):
        gate = _StoreCacheGate(cap=2)
        gate.note_submitted()
        gate.note_submitted()
        assert gate.has_capacity is False

    def test_recovers_after_note_done(self):
        gate = _StoreCacheGate(cap=1)
        gate.note_submitted()
        assert gate.has_capacity is False
        gate.note_done()
        assert gate.has_capacity is True

    def test_shrinking_cap_below_in_flight_blocks_admission(self):
        """Cap shrink under load removes capacity until writes drain."""
        gate = _StoreCacheGate(cap=3)
        gate.note_submitted()
        gate.note_submitted()
        assert gate.has_capacity is True
        gate.set_cap(2)  # shrink under memory pressure
        assert gate.has_capacity is False
        gate.note_done()
        assert gate.has_capacity is True


class TestSetCap:
    def test_clamps_to_minimum_one(self):
        gate = _StoreCacheGate(cap=4)
        gate.set_cap(0)
        assert gate.cap == 1
        gate.set_cap(-5)
        assert gate.cap == 1

    def test_set_cap_updates_value(self):
        gate = _StoreCacheGate(cap=4)
        gate.set_cap(2)
        assert gate.cap == 2
        gate.set_cap(7)
        assert gate.cap == 7


class TestAdjustStoreCacheCap:
    """Tests for Scheduler.adjust_store_cache_cap pressure mapping."""

    def _fake_scheduler(self, cap, max_num_seqs=8):
        from unittest.mock import MagicMock

        sched = MagicMock(spec=["config", "_store_cache_gate"])
        sched.config = MagicMock(spec=["max_num_seqs"])
        sched.config.max_num_seqs = max_num_seqs
        sched._store_cache_gate = _StoreCacheGate(cap=cap)
        return sched

    def test_ok_grows_by_one(self):
        sched = self._fake_scheduler(cap=4, max_num_seqs=8)
        Scheduler.adjust_store_cache_cap(sched, "ok")
        assert sched._store_cache_gate.cap == 5

    def test_ok_clamps_to_max_num_seqs(self):
        sched = self._fake_scheduler(cap=8, max_num_seqs=8)
        Scheduler.adjust_store_cache_cap(sched, "ok")
        assert sched._store_cache_gate.cap == 8

    def test_soft_shrinks_by_one(self):
        sched = self._fake_scheduler(cap=5)
        Scheduler.adjust_store_cache_cap(sched, "soft")
        assert sched._store_cache_gate.cap == 4

    def test_hard_shrinks_by_one(self):
        sched = self._fake_scheduler(cap=3)
        Scheduler.adjust_store_cache_cap(sched, "hard")
        assert sched._store_cache_gate.cap == 2

    def test_shrink_floor_at_one(self):
        sched = self._fake_scheduler(cap=1)
        Scheduler.adjust_store_cache_cap(sched, "hard")
        assert sched._store_cache_gate.cap == 1

    def test_no_op_when_gate_missing(self):
        from unittest.mock import MagicMock

        sched = MagicMock(spec=["config", "_store_cache_gate"])
        sched.config = MagicMock(spec=["max_num_seqs"])
        sched.config.max_num_seqs = 8
        sched._store_cache_gate = None
        # Should not raise.
        Scheduler.adjust_store_cache_cap(sched, "ok")


class TestThreadSafety:
    @pytest.mark.timeout(5)
    def test_counter_consistent_under_contention(self):
        """Concurrent submit/done pairs must net to zero in_flight."""
        gate = _StoreCacheGate(cap=4)

        def worker():
            for _ in range(200):
                gate.note_submitted()
                time.sleep(0.0005)
                gate.note_done()

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert gate.in_flight == 0
