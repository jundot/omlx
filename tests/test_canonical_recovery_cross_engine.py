# SPDX-License-Identifier: Apache-2.0
"""Recovery slack is a property of the process, not of one scheduler.

`EnginePool` gives every loaded model its own `Scheduler` and its own engine
loop, and they share one GPU. Every other clause in the runnable predicate
reads one scheduler's own lists, which answer "am I idle" rather than "is the
machine idle". A recovery chunk admitted on the strength of the first answer
lands on a GPU another engine is using, and a chunk cannot be interrupted once
it starts — so the foreground request it collides with is one this scheduler
never saw and cannot yield to.

The foreground prefill path already treats this as a process-wide question and
already has the registry for it. These tests pin recovery asking the same
question: the decode half through `others_decoding`, and the prefill half
through the process-global prefill tracker, which is what a *foreign prefill*
shows up in. A recovery chunk is itself a prefill, so its own entry is
excluded — a job that stood down for itself would never run.
"""

from unittest.mock import MagicMock

import pytest

from omlx.decode_activity import get_decode_activity
from omlx.foreground_arrivals import get_foreground_arrivals
from omlx.prefill_progress import PrefillProgressTracker, get_prefill_tracker
from omlx.scheduler import Scheduler, SchedulerConfig


def _make_scheduler(**config_over) -> Scheduler:
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2

    config_kwargs = dict(
        max_num_seqs=8,
        prefill_step_size=64,
        chunked_prefill=True,
        paged_cache_block_size=256,
        canonical_state_recovery_enabled=True,
        canonical_state_recovery_global_budget_pct=10.0,
    )
    config_kwargs.update(config_over)
    scheduler = Scheduler(
        model=model, tokenizer=tokenizer, config=SchedulerConfig(**config_kwargs)
    )
    scheduler.block_aware_cache = MagicMock()
    scheduler._unreconstructible_cache_model = False
    mock_bg = MagicMock()
    mock_bg.insert.return_value = [42]
    mock_bg.next_generated.return_value = iter([])
    scheduler.batch_generator = mock_bg
    return scheduler


def _sparse_request(prompt_tokens: int, rid: str = "r1", scheduler=None):
    request = MagicMock()
    request.request_id = rid
    request.prompt_token_ids = list(range(prompt_tokens))
    request.specprefill_indices = [1, 2, 3]
    request._serving_prefix_cache_id = (
        id(scheduler.block_aware_cache) if scheduler is not None else None
    )
    return request


@pytest.fixture(autouse=True)
def _clean_registries():
    """Both registries are process-global singletons, so a test that left an
    entry behind would decide the next one."""
    get_decode_activity().clear()
    get_prefill_tracker()._progress.clear()
    get_foreground_arrivals().clear()
    yield
    get_decode_activity().clear()
    get_prefill_tracker()._progress.clear()
    get_foreground_arrivals().clear()


def _idle(scheduler: Scheduler) -> None:
    scheduler._canonical_recovery_note_step(did_foreground_work=False)
    scheduler._canonical_recovery_note_step(did_foreground_work=False)


class TestAnotherEnginesDecodeWithdrawsTheChunk:
    def test_a_foreign_decode_makes_the_job_not_runnable(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        _idle(scheduler)
        assert scheduler._canonical_recovery_runnable()

        get_decode_activity().publish("another-engine:beef", 1)
        assert not scheduler._canonical_recovery_runnable()

        get_decode_activity().publish("another-engine:beef", 0)
        assert scheduler._canonical_recovery_runnable()

    def test_this_engines_own_entry_is_not_a_foreign_one(self):
        """Its own decode is already covered by `running`, and counting it
        here would mean a scheduler stood down for itself."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        _idle(scheduler)
        get_decode_activity().publish(scheduler._decode_activity_key, 1)
        assert scheduler._canonical_recovery_runnable()


class TestAnotherEnginesPrefillWithdrawsTheChunk:
    """The half the decode registry does not cover.

    An engine that is prefilling publishes a running-decode count of zero,
    which *removes* its registry entry. Prefill is exactly the work a recovery
    chunk contends with, so without the tracker the loudest case is the one
    that goes unseen.
    """

    def test_a_foreign_prefill_makes_the_job_not_runnable(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        _idle(scheduler)
        assert scheduler._canonical_recovery_runnable()

        get_prefill_tracker().update("other-request", 100, 8000, "another-model")
        assert not scheduler._canonical_recovery_runnable()

        get_prefill_tracker().remove("other-request")
        assert scheduler._canonical_recovery_runnable()

    def test_the_jobs_own_prefill_entry_is_excluded(self):
        """A job between chunks still holds its entry, so counting it would
        stop the job it belongs to from ever taking a second chunk."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        _idle(scheduler)
        rid = scheduler._canonical_recovery_request_id(scheduler._canonical_recovery_job)
        get_prefill_tracker().update(rid, 256, 768, "this-model")
        assert scheduler._canonical_recovery_runnable()

    def test_a_foreign_prefill_parks_the_loop_and_keeps_the_job(self):
        """Standing down for another engine is waiting, not finishing.

        Waiting behind a peer is the same kind of wait as a spent window and
        gets the same treatment: the loop parks rather than stepping twenty
        times a second for a job that cannot run, and the job survives so the
        chunk can run once the peer is done. The stall deadline must not count
        it either — there is a reason, and the reason ends.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        assert scheduler.has_requests()

        get_prefill_tracker().update("other-request", 100, 8000, "another-model")
        assert not scheduler.has_requests()

        # Drive the loop's own shape: it re-reads the predicate every step
        # interval, so a parked job costs polls rather than steps.
        steps = 0
        for _ in range(50):
            if scheduler.has_requests():
                scheduler.step()
                steps += 1
        assert steps == 0
        assert scheduler._canonical_recovery_job is not None
        assert scheduler._canonical_recovery_blocked_idle_steps == 0

        get_prefill_tracker().remove("other-request")
        assert scheduler.has_requests()


class TestAnotherEnginesArrivalWithdrawsTheChunk:
    """The half neither progress registry covers.

    Both of the registries above are progress signals: an entry exists once
    work is running. Between the transport accepting a request and that
    request's first forward, every progress signal in the process reads idle —
    and that is precisely the window in which recovery decides to start
    something it cannot interrupt.

    The two sub-windows are separate, and a fix for one does not close the
    other. Before admission nothing anywhere knows the request exists. After
    admission the *admitting* engine's own ``waiting``/``running`` lists know,
    and no peer does, because the prefill tracker is written after the first
    chunk's forward rather than before it.
    """

    def test_an_arrival_at_a_peer_that_is_not_yet_admitted_withdraws_the_chunk(self):
        """Window one: accepted by the transport, not yet handed to an engine."""
        engine_a = _make_scheduler()
        engine_b = _make_scheduler()
        engine_a.note_canonical_recovery_candidate(
            _sparse_request(1000, scheduler=engine_a)
        )
        _idle(engine_a)
        assert engine_a._canonical_recovery_runnable()

        # The request reaches engine B. B has not admitted it, so B's own
        # lists are empty too — nothing in this process is running yet.
        engine_b.note_inbound_request("fg-on-b")
        assert not engine_b._canonical_recovery_local_requests()
        assert not engine_a._canonical_recovery_runnable()

    def test_an_arrival_admitted_at_a_peer_still_withdraws_the_chunk(self):
        """Window two: admitted at B, first prefill chunk not yet forwarded.

        B knows through its own lists. A has nothing to read: the tracker
        entry does not exist until the chunk has already run, which is one
        chunk too late for a slice A cannot interrupt.
        """
        engine_a = _make_scheduler()
        engine_b = _make_scheduler()
        engine_a.note_canonical_recovery_candidate(
            _sparse_request(1000, scheduler=engine_a)
        )
        _idle(engine_a)

        engine_b.note_inbound_request("fg-on-b")
        engine_b.note_admitted_request("fg-on-b")
        assert get_prefill_tracker().any_active() is False
        assert not engine_a._canonical_recovery_runnable()

    def test_the_chunk_returns_once_the_peers_request_departs(self):
        engine_a = _make_scheduler()
        engine_b = _make_scheduler()
        engine_a.note_canonical_recovery_candidate(
            _sparse_request(1000, scheduler=engine_a)
        )
        _idle(engine_a)

        engine_b.note_inbound_request("fg-on-b")
        engine_b.note_admitted_request("fg-on-b")
        assert not engine_a._canonical_recovery_runnable()

        engine_b.note_request_departed("fg-on-b")
        assert engine_a._canonical_recovery_runnable()

    def test_a_recovery_job_is_not_an_arrival(self):
        """Recovery must not stand down for recovery through this signal.

        The synthetic request is built inside the scheduler and never passes
        through the transport, so it never reaches the registry at all —
        by construction rather than by an exclusion rule that could drift.
        """
        engine_a = _make_scheduler()
        engine_b = _make_scheduler()
        for engine in (engine_a, engine_b):
            engine.note_canonical_recovery_candidate(
                _sparse_request(1000, scheduler=engine)
            )
            _idle(engine)

        engine_b._canonical_recovery_begin_state = MagicMock(return_value=None)
        assert get_foreground_arrivals().count() == 0
        assert engine_a._canonical_recovery_runnable()

    def test_recovery_still_excludes_recovery_through_the_claim(self):
        """Mutual exclusion between jobs is the budget's claim, and the
        arrival signal must not be doing that work by accident."""
        engine_a = _make_scheduler()
        engine_b = _make_scheduler()
        # One budget object for the process, as `EnginePool` arranges.
        engine_b._canonical_recovery_budget = engine_a._canonical_recovery_budget
        for engine in (engine_a, engine_b):
            engine.note_canonical_recovery_candidate(
                _sparse_request(1000, scheduler=engine)
            )
            _idle(engine)
        assert engine_a._canonical_recovery_runnable()

        assert engine_b._canonical_recovery_budget.try_claim(
            engine_b._canonical_recovery_owner_key
        )
        try:
            assert engine_a._canonical_recovery_claim_blocked()
            assert not engine_a._canonical_recovery_runnable()
        finally:
            engine_b._canonical_recovery_budget.release_claim(
                engine_b._canonical_recovery_owner_key
            )
        assert engine_a._canonical_recovery_runnable()


class TestAnyActive:
    """The tracker method the scheduler asks with."""

    def test_an_empty_tracker_reports_nothing_active(self):
        assert PrefillProgressTracker().any_active() is False

    def test_a_live_entry_is_active(self):
        tracker = PrefillProgressTracker()
        tracker.update("r1", 10, 100, "m")
        assert tracker.any_active() is True

    def test_an_excluded_entry_is_not(self):
        tracker = PrefillProgressTracker()
        tracker.update("r1", 10, 100, "m")
        assert tracker.any_active(("r1",)) is False
        tracker.update("r2", 10, 100, "m")
        assert tracker.any_active(("r1",)) is True

    def test_a_completed_prefill_is_not_active(self):
        """`update` removes the entry at completion, so this is the tracker's
        own contract rather than a second rule."""
        tracker = PrefillProgressTracker()
        tracker.update("r1", 10, 100, "m")
        tracker.update("r1", 100, 100, "m")
        assert tracker.any_active() is False
