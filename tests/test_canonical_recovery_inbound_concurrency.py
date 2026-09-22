# SPDX-License-Identifier: Apache-2.0
"""The inbound marker is written from one thread and read from another.

``note_inbound_request`` runs on the asyncio event loop: ``EngineCore.add_request``
announces the arrival *before* handing the request to the executor, because the
executor is the same single worker that runs ``step()`` and the request is
invisible to the scheduler until that hand-off completes.

``_canonical_recovery_inbound_count`` runs on that executor, and it does not
just read — it expires stale entries, so it iterates the mapping and then
deletes from it.

That is not the ``_pending_abort_ids`` idiom. A ``set.add`` / ``set.pop`` pair
is a single bytecode each and the GIL makes it atomic; iterating a dict is not,
and a dict that grows mid-iteration raises ``RuntimeError: dictionary changed
size during iteration``. The exception surfaces inside the recovery predicate,
on the engine loop.

These tests drive both sides at once, with the window deliberately widened:
the interpreter's thread-switch interval is dropped to a microsecond and the
mapping is pre-loaded to a size no real arrival rate would reach. At realistic
sizes the comprehension finishes well inside one switch interval, so the race
is rare rather than absent — the amplification is what makes it a test instead
of a coin flip. Without it these same tests pass against the unlocked dict.

The assertion is that no exception escaped, which cannot produce a false
failure.
"""

import sys
import threading
from unittest.mock import MagicMock

import pytest

from omlx.scheduler import Scheduler, SchedulerConfig

# Large enough that iterating it spans several thread switches at the interval
# below. Both numbers exist only to make the window observable.
SEEDS = 50_000
SWITCH_INTERVAL_S = 1e-6


@pytest.fixture
def fine_grained_switching():
    previous = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_INTERVAL_S)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _make_scheduler() -> Scheduler:
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8,
            prefill_step_size=64,
            chunked_prefill=True,
            paged_cache_block_size=256,
            canonical_state_recovery_enabled=True,
        ),
    )
    scheduler.block_aware_cache = MagicMock()
    return scheduler


def _run_race(scheduler, *, writers, reader, seconds: float = 2.0):
    """Run *writers* and *reader* together and collect anything they raise."""
    errors: list[BaseException] = []
    stop = threading.Event()

    def guard(fn):
        def run():
            try:
                fn(stop)
            except BaseException as exc:  # noqa: BLE001 - the whole point
                errors.append(exc)
                stop.set()

        return run

    threads = [threading.Thread(target=guard(fn)) for fn in (*writers, reader)]
    for thread in threads:
        thread.start()
    stop.wait(seconds)
    stop.set()
    for thread in threads:
        thread.join(timeout=10.0)
        assert not thread.is_alive(), "a race thread did not stop"
    return errors


class TestInboundBookkeepingIsThreadSafe:
    def test_arrival_during_expiry_does_not_raise(self, fine_grained_switching):
        """The reader iterates while the writer inserts.

        Every entry is already stale — the TTL is zero — so the reader takes
        the expiry branch on every call, which is the branch that iterates.
        """
        scheduler = _make_scheduler()
        scheduler._canonical_recovery_inbound_ttl_s = 0.0
        # Pre-load enough entries that the comprehension is not over in one
        # bytecode dispatch; a two-entry dict races far too rarely to test.
        for i in range(SEEDS):
            scheduler.note_inbound_request(f"seed-{i}")

        def writer(stop):
            i = 0
            while not stop.is_set():
                scheduler.note_inbound_request(f"w-{i}")
                i += 1

        def reader(stop):
            while not stop.is_set():
                scheduler._canonical_recovery_inbound_count()

        errors = _run_race(scheduler, writers=[writer], reader=reader)
        assert not errors, f"inbound bookkeeping raised under concurrency: {errors!r}"

    def test_admission_during_expiry_does_not_raise(self, fine_grained_switching):
        """The other writer: admission deletes, expiry deletes, both at once."""
        scheduler = _make_scheduler()
        scheduler._canonical_recovery_inbound_ttl_s = 0.0
        ids = [f"seed-{i}" for i in range(SEEDS)]
        for rid in ids:
            scheduler.note_inbound_request(rid)

        def writer(stop):
            i = 0
            while not stop.is_set():
                rid = f"w-{i}"
                scheduler.note_inbound_request(rid)
                scheduler.note_admitted_request(rid)
                scheduler.note_admitted_request(ids[i % len(ids)])
                i += 1

        def reader(stop):
            while not stop.is_set():
                scheduler._canonical_recovery_inbound_count()

        errors = _run_race(scheduler, writers=[writer], reader=reader)
        assert not errors, f"inbound bookkeeping raised under concurrency: {errors!r}"

    def test_two_readers_expiring_at_once_do_not_raise(self, fine_grained_switching):
        """Both engine loops in one process can reach the predicate together."""
        scheduler = _make_scheduler()
        scheduler._canonical_recovery_inbound_ttl_s = 0.0
        for i in range(SEEDS):
            scheduler.note_inbound_request(f"seed-{i}")

        def writer(stop):
            i = 0
            while not stop.is_set():
                scheduler.note_inbound_request(f"w-{i}")
                i += 1

        def reader(stop):
            while not stop.is_set():
                scheduler._canonical_recovery_inbound_count()

        errors = _run_race(scheduler, writers=[writer, reader], reader=reader)
        assert not errors, f"inbound bookkeeping raised under concurrency: {errors!r}"


class TestTheSemanticsAreUnchanged:
    """The lock must not change what the counter counts."""

    def test_an_arrival_counts_until_it_departs(self):
        scheduler = _make_scheduler()
        assert scheduler._canonical_recovery_inbound_count() == 0
        scheduler.note_inbound_request("incoming")
        assert scheduler._canonical_recovery_inbound_count() == 1
        scheduler.note_admitted_request("incoming")
        assert scheduler._canonical_recovery_inbound_count() == 1
        scheduler.note_request_departed("incoming")
        assert scheduler._canonical_recovery_inbound_count() == 0

    def test_an_arrival_that_never_lands_expires(self):
        scheduler = _make_scheduler()
        scheduler.note_inbound_request("lost")
        scheduler._canonical_recovery_inbound_ttl_s = 0.0
        assert scheduler._canonical_recovery_inbound_count() == 0

    def test_admitting_an_unknown_id_is_a_no_op(self):
        """Admission re-stamps an entry; it never creates one. A request that
        departed before this ran must not be resurrected by it."""
        scheduler = _make_scheduler()
        scheduler.note_admitted_request("never-arrived")
        assert scheduler._canonical_recovery_inbound_count() == 0
