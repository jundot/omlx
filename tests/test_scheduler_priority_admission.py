# SPDX-License-Identifier: Apache-2.0
"""Tests for Scheduler._admit_to_waiting (opt-in priority-ordered admission).

FCFS (the default) must stay byte-identical to the plain-append behavior it
replaced. PRIORITY policy (OMLX_PRIORITY_SCHEDULING=1) must insert
lower-priority-value requests ahead of higher-priority-value ones while
preserving arrival order among requests of equal priority — i.e. every
existing caller, which leaves priority at its default of 0, observes
unchanged FCFS ordering even under the PRIORITY policy until something
actually sets a non-zero priority.
"""

from collections import deque
from unittest.mock import MagicMock

import pytest

from omlx.scheduler import Scheduler, SchedulingPolicy


@pytest.fixture
def scheduler():
    """Bare Scheduler instance (see test_scheduler_admission.py for why)."""
    s = Scheduler.__new__(Scheduler)
    s.config = MagicMock()
    s.waiting = deque()
    return s


def _req(rid: str, priority: int = 0):
    r = MagicMock()
    r.request_id = rid
    r.priority = priority
    return r


def _ids(scheduler):
    return [r.request_id for r in scheduler.waiting]


class TestFcfsUnchanged:
    """The default policy must be indistinguishable from plain deque.append."""

    def test_appends_to_back_regardless_of_priority(self, scheduler):
        scheduler.config.policy = SchedulingPolicy.FCFS
        scheduler._admit_to_waiting(_req("a", priority=0))
        scheduler._admit_to_waiting(_req("b", priority=-5))  # would outrank under PRIORITY
        scheduler._admit_to_waiting(_req("c", priority=0))
        assert _ids(scheduler) == ["a", "b", "c"]


class TestPriorityOrdering:
    def test_empty_queue(self, scheduler):
        scheduler.config.policy = SchedulingPolicy.PRIORITY
        scheduler._admit_to_waiting(_req("a"))
        assert _ids(scheduler) == ["a"]

    def test_equal_priority_preserves_arrival_order(self, scheduler):
        # Every caller today leaves priority at its default (0) — this is
        # the case that MUST match today's FCFS behavior exactly.
        scheduler.config.policy = SchedulingPolicy.PRIORITY
        for rid in ("a", "b", "c", "d"):
            scheduler._admit_to_waiting(_req(rid, priority=0))
        assert _ids(scheduler) == ["a", "b", "c", "d"]

    def test_higher_priority_request_jumps_ahead_of_lower_priority_waiters(
        self, scheduler
    ):
        # Lower numeric value = higher priority (interactive = 0).
        scheduler.config.policy = SchedulingPolicy.PRIORITY
        scheduler._admit_to_waiting(_req("bg-1", priority=10))
        scheduler._admit_to_waiting(_req("bg-2", priority=10))
        scheduler._admit_to_waiting(_req("interactive", priority=0))
        assert _ids(scheduler) == ["interactive", "bg-1", "bg-2"]

    def test_new_same_priority_request_goes_behind_existing_same_priority_ones(
        self, scheduler
    ):
        # A second interactive request must NOT jump ahead of the first —
        # only the priority TIER jumps the background queue, not each new
        # arrival within a tier.
        scheduler.config.policy = SchedulingPolicy.PRIORITY
        scheduler._admit_to_waiting(_req("int-1", priority=0))
        scheduler._admit_to_waiting(_req("bg-1", priority=10))
        scheduler._admit_to_waiting(_req("int-2", priority=0))
        assert _ids(scheduler) == ["int-1", "int-2", "bg-1"]

    def test_three_tier_interleaving(self, scheduler):
        scheduler.config.policy = SchedulingPolicy.PRIORITY
        order = [
            ("bg-1", 10),
            ("mid-1", 5),
            ("bg-2", 10),
            ("int-1", 0),
            ("mid-2", 5),
            ("int-2", 0),
        ]
        for rid, prio in order:
            scheduler._admit_to_waiting(_req(rid, priority=prio))
        # Stable sort by priority: 0s in arrival order, then 5s, then 10s.
        assert _ids(scheduler) == ["int-1", "int-2", "mid-1", "mid-2", "bg-1", "bg-2"]

    def test_does_not_disturb_a_leading_appendleft_rescheduled_request(self, scheduler):
        # Simulates a retry/reschedule (cache eviction, corruption, overflow)
        # that jumped the front via appendleft BEFORE this new admission. A
        # SAME-tier new arrival must NOT displace it — resumed work with
        # sunk cost still beats a freshly-arriving request of equal priority.
        scheduler.config.policy = SchedulingPolicy.PRIORITY
        scheduler.waiting.appendleft(_req("resumed", priority=10))
        scheduler._admit_to_waiting(_req("new-bg", priority=10))
        assert _ids(scheduler) == ["resumed", "new-bg"]

    def test_strictly_higher_priority_arrival_still_outranks_a_resumed_request(
        self, scheduler
    ):
        # A resumed (appendleft'd) request only holds its place against
        # SAME-or-lower actual priority. A genuinely more urgent new arrival
        # (e.g. a live interactive turn vs. a resumed BACKGROUND request)
        # correctly cuts ahead — the whole point of this feature is that
        # interactive never waits behind background work, resumed or not.
        scheduler.config.policy = SchedulingPolicy.PRIORITY
        scheduler.waiting.appendleft(_req("resumed-bg", priority=10))
        scheduler._admit_to_waiting(_req("interactive", priority=0))
        assert _ids(scheduler) == ["interactive", "resumed-bg"]
