# SPDX-License-Identifier: Apache-2.0
"""Published canonical state is durable; in-progress state is disposable.

A recovery slice's ``prefill_state`` holds a *materialised* KV cache --
``make_prompt_cache`` filled by dense forwards, or ``reconstruct_cache``,
which allocates rather than referencing the paged pool. Measured on a small
hybrid model it is 12 KiB per token plus 18.6 MiB of fixed recurrent state;
by arithmetic over the 64-layer production geometry (16 full-attention layers,
4 KV heads, head_dim 256, bf16) it is 64 KiB per token, so a parked 32k-token
job holds 2 GiB. ``mx.get_active_memory()`` falls by exactly that amount the
moment the reference goes.

Between slices the state is worth keeping: the next slice continues from it.
It stops being worth keeping the moment recovery stands down for something
that is not a gap between slices, because recovery is the lowest-priority
work in the process and must not be its largest idle allocation. So the state
goes back when foreground work exists anywhere in the process, when the budget
window is spent, when the prefill memory throttle fires, and when a chunk is
aborted -- and it is kept across the two-idle-step spacing between slices and
across a peer holding the recovery claim, both of which last one slice.

What survives retirement is everything the job is: its committed boundary, its
published blocks, its lineage and its ability to resume. What is lost is at
most one block of dense re-read, because publication floors to a block and
runs after every chunk.

The throttle case has an ordering requirement of its own. ``_PrefillEvictionNeeded``
is raised because predicted usage crossed a cap measured against current usage,
and the job's own state is the largest part of that current usage recovery
owns. Reclaiming around it and then keeping it is the one ordering that leaves
the memory-pressure-causing allocation alive exactly when the runtime said
there was not enough memory.

No claim is made here that foreground work goes faster. The measurement behind
this file establishes the retained size and that ``mx.get_active_memory()``
returns it; it does not establish a foreground effect, and none is asserted.
"""

import gc
import weakref
from unittest.mock import MagicMock, patch

import pytest

from omlx.decode_activity import get_decode_activity
from omlx.foreground_arrivals import get_foreground_arrivals
from omlx.prefill_progress import get_prefill_tracker
from omlx.request import Request, SamplingParams
from omlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    _PrefillAbortedError,
    _PrefillEvictionNeeded,
)

BLOCK = 256


def _make_scheduler(**config_over) -> Scheduler:
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    config_kwargs = dict(
        max_num_seqs=8,
        prefill_step_size=64,
        chunked_prefill=True,
        paged_cache_block_size=BLOCK,
        canonical_state_recovery_enabled=True,
        canonical_state_recovery_global_budget_pct=10.0,
    )
    config_kwargs.update(config_over)
    scheduler = Scheduler(
        model=model, tokenizer=tokenizer, config=SchedulerConfig(**config_kwargs)
    )
    scheduler.block_aware_cache = MagicMock()
    scheduler._unreconstructible_cache_model = False
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
    for reset in (
        lambda: get_decode_activity().clear(),
        lambda: get_prefill_tracker()._progress.clear(),
        lambda: get_foreground_arrivals().clear(),
    ):
        reset()
    yield
    for reset in (
        lambda: get_decode_activity().clear(),
        lambda: get_prefill_tracker()._progress.clear(),
        lambda: get_foreground_arrivals().clear(),
    ):
        reset()


def _idle(scheduler: Scheduler) -> None:
    scheduler._canonical_recovery_note_step(did_foreground_work=False)
    scheduler._canonical_recovery_note_step(did_foreground_work=False)


def _live_job(scheduler: Scheduler, *, committed: int = 2 * BLOCK, processed: int = 0):
    """A job mid-slice: state built, a boundary already published."""
    scheduler.note_canonical_recovery_candidate(
        _sparse_request(8 * BLOCK, scheduler=scheduler)
    )
    job = scheduler._canonical_recovery_job
    assert job is not None
    job.note_published(committed)
    job.processed_tokens = processed or committed + 10
    job.prefill_state = MagicMock(
        base_size=committed,
        tokens_processed=job.processed_tokens - committed,
        cache=[MagicMock()],
        canonical_recovery_target_tokens=job.target_tokens,
    )
    return job


def _after_step(scheduler: Scheduler) -> None:
    output = MagicMock()
    output.has_work = False
    scheduler._canonical_recovery_after_step(output)


# --------------------------------------------------------------------------
# 1-2. Foreground anywhere in the process retires the state
# --------------------------------------------------------------------------


class TestForegroundRetiresTheState:
    def test_a_local_foreground_request_retires_it(self):
        scheduler = _make_scheduler()
        job = _live_job(scheduler)
        _idle(scheduler)
        scheduler.waiting.append(MagicMock())

        _after_step(scheduler)

        assert job.prefill_state is None
        assert scheduler._canonical_recovery_job is job
        assert scheduler._canonical_recovery_counters.states_retired == 1

    def test_a_request_that_has_arrived_and_not_been_admitted_retires_it(self):
        """The window the arrival registry exists for: nothing is in any list
        yet, so only the direct arrival signal can see this request."""
        scheduler = _make_scheduler()
        job = _live_job(scheduler)
        _idle(scheduler)
        scheduler.note_inbound_request("incoming")

        _after_step(scheduler)

        assert job.prefill_state is None

    def test_a_foreign_decode_retires_it(self):
        scheduler = _make_scheduler()
        job = _live_job(scheduler)
        _idle(scheduler)
        get_decode_activity().publish("another-engine:beef", 1)

        _after_step(scheduler)

        assert job.prefill_state is None

    def test_a_foreign_prefill_retires_it(self):
        scheduler = _make_scheduler()
        job = _live_job(scheduler)
        _idle(scheduler)
        get_prefill_tracker().update("other-request", 100, 8000, "another-model")

        _after_step(scheduler)

        assert job.prefill_state is None


class TestAGapBetweenSlicesDoesNot:
    """The controls. Retiring across these would rebuild before every slice."""

    def test_the_two_idle_step_spacing_keeps_the_state(self):
        scheduler = _make_scheduler()
        job = _live_job(scheduler)
        scheduler._consecutive_idle_steps = 0  # as a slice that just ran leaves it
        assert not scheduler._canonical_recovery_runnable()

        _after_step(scheduler)

        assert job.prefill_state is not None
        assert scheduler._canonical_recovery_counters.states_retired == 0

    def test_a_peer_holding_the_recovery_claim_keeps_the_state(self):
        """A claim is released in the `finally` of a slice, so it is held for
        a slice. Standing down for one is a gap, not a pause."""
        scheduler = _make_scheduler()
        job = _live_job(scheduler)
        _idle(scheduler)
        scheduler._canonical_recovery_budget.try_claim("another-engine")
        assert not scheduler._canonical_recovery_runnable()

        _after_step(scheduler)

        assert job.prefill_state is not None


class TestASpentBudgetWindowRetiresTheState:
    """A window is 30 seconds and the allowance a percentage of it, so a spent
    window is tens of seconds of holding a dense cache for nothing."""

    def test_an_exhausted_allowance_retires_it(self):
        scheduler = _make_scheduler()
        job = _live_job(scheduler)
        _idle(scheduler)
        budget = scheduler._canonical_recovery_budget
        budget.note_service(budget.allowance_s + 1.0)
        assert not budget.allows()

        _after_step(scheduler)

        assert job.prefill_state is None
        assert job.committed_tokens == 2 * BLOCK


# --------------------------------------------------------------------------
# 3-4. The two in-slice pauses
# --------------------------------------------------------------------------


def _step_raising(scheduler: Scheduler, exc: BaseException):
    """Run one slice whose chunk raises, recording the reclaim's view.

    The reclaim is called as ``Scheduler._clear_cache(self)``, so it is
    patched on the class; the recorded value is what ``job.prefill_state``
    was at the moment it ran, which is the ordering under test.
    """
    seen: list = []
    job = scheduler._canonical_recovery_job

    def record(_self):
        seen.append(job.prefill_state)

    tracker = MagicMock()
    tracker.any_active.return_value = False
    tracker.recently_active.return_value = False
    with patch.object(scheduler, "_step_prefill_chunk", side_effect=exc), patch.object(
        Scheduler, "_clear_cache", record
    ), patch("omlx.scheduler.get_prefill_tracker", return_value=tracker):
        scheduler._canonical_recovery_step()
    assert seen, "the reclaim did not run"
    return seen[0]


class TestTheThrottleRetiresBeforeItReclaims:
    def test_the_state_is_gone_by_the_time_the_reclaim_runs(self):
        scheduler = _make_scheduler()
        job = _live_job(scheduler)

        at_reclaim = _step_raising(scheduler, _PrefillEvictionNeeded(MagicMock()))

        assert at_reclaim is None, (
            "the memory throttle reclaimed around the recovery state instead "
            "of giving it back first"
        )
        assert job.prefill_state is None
        assert scheduler._canonical_recovery_counters.states_retired == 1

    def test_the_job_and_its_committed_prefix_survive(self):
        scheduler = _make_scheduler()
        job = _live_job(scheduler)

        _step_raising(scheduler, _PrefillEvictionNeeded(MagicMock()))

        assert scheduler._canonical_recovery_job is job
        assert not job.cancelled
        assert job.committed_tokens == 2 * BLOCK


class TestAnAbortedChunkRetiresTheState:
    def test_the_state_is_given_back(self):
        scheduler = _make_scheduler()
        job = _live_job(scheduler)

        at_reclaim = _step_raising(scheduler, _PrefillAbortedError([], 0))

        assert at_reclaim is None
        assert job.prefill_state is None
        assert scheduler._canonical_recovery_job is job


# --------------------------------------------------------------------------
# 5-6. What survives, and resuming from it
# --------------------------------------------------------------------------


class TestWhatSurvivesRetirement:
    def test_the_committed_boundary_and_published_blocks_survive(self):
        scheduler = _make_scheduler()
        job = _live_job(scheduler, committed=3 * BLOCK)
        _idle(scheduler)
        scheduler.waiting.append(MagicMock())

        _after_step(scheduler)

        assert job.committed_tokens == 3 * BLOCK
        assert job.published_boundaries == [3 * BLOCK]
        assert job.session_key == "r1"
        assert not job.cancelled
        assert not job.done

    def test_a_later_idle_window_rebuilds_from_the_published_prefix(self):
        """Resumability: the next slice restores what was published through
        the ordinary prefix-cache path and continues from there."""
        scheduler = _make_scheduler()
        job = _live_job(scheduler, committed=3 * BLOCK)
        _idle(scheduler)
        scheduler.waiting.append(MagicMock())
        _after_step(scheduler)
        assert job.prefill_state is None

        scheduler.waiting.clear()
        rebuilt = MagicMock(
            base_size=3 * BLOCK,
            tokens_processed=0,
            cache=[MagicMock()],
            canonical_recovery_target_tokens=job.target_tokens,
        )
        tracker = MagicMock()
        tracker.any_active.return_value = False
        tracker.recently_active.return_value = False
        with patch.object(
            scheduler, "_canonical_recovery_begin_state", return_value=rebuilt
        ) as begin, patch.object(
            scheduler, "_step_prefill_chunk", return_value=False
        ), patch("omlx.scheduler.get_prefill_tracker", return_value=tracker):
            assert scheduler._canonical_recovery_step() is True

        begin.assert_called_once_with(job)
        assert job.prefill_state is rebuilt
        assert job.committed_tokens == 3 * BLOCK

    def test_nothing_still_refers_to_the_retired_state(self):
        """The point is the arrays going away, so reachability is what is
        asserted rather than the attribute that used to name them. A state
        the scheduler still holds somewhere else would satisfy
        ``prefill_state is None`` and free nothing."""
        scheduler = _make_scheduler()
        job = _live_job(scheduler)
        state = job.prefill_state
        witness = weakref.ref(state)
        _idle(scheduler)
        scheduler.waiting.append(MagicMock())

        _after_step(scheduler)

        del state
        gc.collect()
        assert witness() is None, (
            "the recovery state survived the retirement that unlinked it: "
            "something else in the scheduler is still holding it"
        )


class TestForegroundStateIsUntouched:
    def test_no_foreground_request_or_prefill_state_is_disturbed(self):
        scheduler = _make_scheduler()
        job = _live_job(scheduler)
        foreground = Request(
            request_id="fg-1",
            prompt=None,
            prompt_token_ids=list(range(64)),
            sampling_params=SamplingParams(max_tokens=1),
        )
        scheduler.requests["fg-1"] = foreground
        fg_state = MagicMock()
        scheduler._prefill_states["fg-1"] = fg_state
        scheduler.waiting.append(foreground)
        _idle(scheduler)

        _after_step(scheduler)

        assert job.prefill_state is None
        assert scheduler.requests["fg-1"] is foreground
        assert scheduler._prefill_states["fg-1"] is fg_state
        assert list(scheduler.waiting) == [foreground]


# --------------------------------------------------------------------------
# The scheduling seam: somebody has to be awake to do the freeing
# --------------------------------------------------------------------------


class TestTheLoopStaysAwakeLongEnoughToFreeIt:
    """`_has_canonical_recovery_work` gates the engine loop, and every reason
    the state is retired for is also a reason that predicate answers False.
    Without the live-state clause, a peer's foreground work parks this loop
    with the dense cache still resident and no step in which to drop it."""

    def test_a_live_state_keeps_the_loop_stepping_through_a_peers_work(self):
        scheduler = _make_scheduler()
        job = _live_job(scheduler)
        get_prefill_tracker().update("other-request", 100, 8000, "another-model")

        assert scheduler._has_canonical_recovery_work() is True
        assert scheduler.has_requests() is True

        _idle(scheduler)
        _after_step(scheduler)

        assert job.prefill_state is None
        # And having freed it, the loop is allowed to park again.
        assert scheduler._has_canonical_recovery_work() is False

    def test_a_live_state_keeps_the_loop_stepping_through_a_spent_window(self):
        scheduler = _make_scheduler()
        _live_job(scheduler)
        budget = scheduler._canonical_recovery_budget
        budget.note_service(budget.allowance_s + 1.0)

        assert scheduler._has_canonical_recovery_work() is True

    def test_the_predicate_is_a_plain_attribute_read(self):
        """It is called from `has_requests` on the asyncio thread, so it must
        not touch MLX or the cache. Asserting the *absence* of a call is the
        only way to keep that true as the predicate grows."""
        scheduler = _make_scheduler()
        _live_job(scheduler)
        with patch.object(scheduler, "_canonical_recovery_stand_down") as stand_down:
            scheduler._has_canonical_recovery_work()
        stand_down.assert_not_called()

    def test_a_disabled_recovery_does_not_spin_on_a_stale_state(self):
        """`after_step` is gated on the same flag, so reporting work for a
        state nothing will come back for would spin an idle loop forever."""
        scheduler = _make_scheduler()
        _live_job(scheduler)
        scheduler._canonical_recovery_enabled_flag = False

        assert scheduler._has_canonical_recovery_work() is False
