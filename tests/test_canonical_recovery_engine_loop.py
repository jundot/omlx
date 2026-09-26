# SPDX-License-Identifier: Apache-2.0
"""What a live recovery job costs the engine loop while it is not running.

A recovery job is only allowed to run on an idle engine, and it keeps the engine
loop stepping so that the idle moment it needs can actually arrive. Those two
facts together are a resource question, not a scheduling one: for as long as a
job is live and waiting, the loop runs a full scheduler step twenty times a
second, and a scheduler step advances the counter that gates ``gc.collect()``
and the periodic *process-global* ``mx.clear_cache()``. That pool is shared
with every other model the process serves, so a background job on one session
is in a position to disturb foreground work it has nothing to do with.

These tests pin the three things that keep that bounded:

- the engine loop re-reads ``has_requests()`` once per ``step_interval``
  whether or not it stepped, so a job that is out of allowance can be parked
  and picked back up without anything having to wake it;
- a job out of allowance is parked rather than polled;
- a job that is allowed to run and cannot is given up on, because the runtime
  can leave a condition installed that nothing the job does will ever clear.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omlx.engine_core import EngineConfig, EngineCore
from omlx.scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from omlx.canonical_recovery import MAX_BLOCKED_IDLE_STEPS


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


def _spend_the_window(scheduler: Scheduler) -> None:
    """Charge a whole window's wall time, which no allowance can cover."""
    scheduler._canonical_recovery_budget.note_service(scheduler._canonical_recovery_budget.window_s)


class TestTheLoopRepollsWithoutAWake:
    """The premise the parking decision rests on, measured rather than assumed.

    An earlier version of ``_has_canonical_recovery_work`` held the predicate true through
    a spent window on the belief that reporting no work would park the loop
    until an unrelated request woke it — "which on an idle server is never".
    The loop does not behave that way, and these two tests are why the belief
    could be dropped rather than worked around.
    """

    @pytest.mark.asyncio
    async def test_an_idle_loop_re_reads_has_requests_every_interval(
        self, mock_model, mock_tokenizer
    ):
        with patch("omlx.engine_core.get_registry") as registry:
            registry.return_value.acquire.return_value = True
            engine = EngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=EngineConfig(step_interval=0.01),
            )
            try:
                engine.scheduler.has_requests = MagicMock(return_value=False)
                await engine.start()
                await asyncio.sleep(0.2)
                # No wake, no request, no work: the predicate is still read.
                assert engine.scheduler.has_requests.call_count > 3
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_work_appearing_on_its_own_is_picked_up_without_a_wake(
        self, mock_model, mock_tokenizer
    ):
        """This is the case a parked job is in when its window replenishes."""
        with patch("omlx.engine_core.get_registry") as registry:
            registry.return_value.acquire.return_value = True
            engine = EngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=EngineConfig(step_interval=0.01),
            )
            try:
                state = SimpleNamespace(polls=0)

                def has_requests():
                    state.polls += 1
                    return state.polls > 5

                engine.scheduler.has_requests = MagicMock(side_effect=has_requests)
                engine.scheduler.step = MagicMock(
                    return_value=SchedulerOutput(has_work=False)
                )
                await engine.start()
                for _ in range(100):
                    if engine.scheduler.step.call_count:
                        break
                    await asyncio.sleep(0.01)
                assert engine.scheduler.step.call_count >= 1
            finally:
                await engine.stop()
                engine.close()


class TestASpentWindowCostsNoSteps:
    """A job waiting for its allowance must not step the engine for it.

    The cost being avoided is not the step itself. It is ``_step_counter``,
    which gates ``gc.collect()`` and the periodic process-global Metal cache
    clear: at the 50 ms step interval a permanently-waiting job advances that
    counter about 20 times a second, which reaches the 512-step clear interval
    roughly every 26 seconds for as long as the job is live.
    """

    def _drive(self, scheduler: Scheduler, steps: int) -> int:
        """Run the engine loop's own predicate-then-step shape *steps* times."""
        taken = 0
        for _ in range(steps):
            if scheduler.has_requests():
                scheduler.step()
                taken += 1
        return taken

    def test_a_spent_window_takes_no_scheduler_steps(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        _spend_the_window(scheduler)
        counter_before = scheduler._step_counter
        assert self._drive(scheduler, 50) == 0
        assert scheduler._step_counter == counter_before

    def test_an_allowed_window_does_step(self):
        """The contrast, so the test above is not passing for the wrong reason."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        counter_before = scheduler._step_counter
        with patch.object(scheduler, "_canonical_recovery_step", return_value=False):
            assert self._drive(scheduler, 5) == 5
        assert scheduler._step_counter == counter_before + 5

    def test_the_job_survives_being_parked(self):
        """Parking is not cancelling: the job and its committed prefix stay."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        job.note_published(512)
        _spend_the_window(scheduler)
        assert not scheduler.has_requests()
        assert scheduler._canonical_recovery_job is job
        assert not job.cancelled
        assert job.committed_tokens == 512

    def test_a_zero_budget_is_still_a_different_case(self):
        """Zero percent never replenishes, so its job is not waiting at all."""
        scheduler = _make_scheduler(canonical_state_recovery_global_budget_pct=0.0)
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        assert scheduler._canonical_recovery_job is None
        assert not scheduler.has_requests()


class TestAStalledJobIsGivenUpOn:
    """A job that may run and cannot must not pin the loop forever.

    `_specprefill_rope_installed` refuses a dense forward while a SpecPrefill
    RoPE wrapper is on the shared model, and `_unwrap_rope` documents a wrapper
    left installed between requests as an expected state (#766). Nothing the
    recovery job does takes it off. Without a deadline the job is live forever:
    the loop keeps stepping for it, and the engine never becomes quiescent, so
    the model can never be unloaded.

    The yield limit does not cover this. `consecutive_yields` is raised from
    inside a chunk and no chunk is ever reached here.
    """

    def _run_idle_steps(self, scheduler: Scheduler, steps: int) -> None:
        for _ in range(steps):
            scheduler.step()

    def test_a_job_blocked_by_a_leftover_rope_wrapper_is_dropped(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        with patch.object(scheduler, "_specprefill_rope_installed", return_value=True):
            self._run_idle_steps(scheduler, MAX_BLOCKED_IDLE_STEPS - 1)
            assert scheduler._canonical_recovery_job is not None
            assert scheduler._canonical_recovery_blocked_idle_steps == MAX_BLOCKED_IDLE_STEPS - 1
            self._run_idle_steps(scheduler, 1)
        assert scheduler._canonical_recovery_job is None
        assert not scheduler.has_requests()

    def test_a_busy_engine_is_not_a_stall(self):
        """A job waiting behind foreground work has a reason, and it will end.

        Dropping it here would punish exactly the sessions the feature is for,
        and it buys nothing: the loop is stepping for the foreground anyway.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        # A request that has arrived and not yet been admitted: foreground
        # pressure the scheduler's own lists cannot see, and the hardest case
        # for the deadline to get right.
        scheduler.note_inbound_request("inbound-1")
        self._run_idle_steps(scheduler, MAX_BLOCKED_IDLE_STEPS * 2)
        assert scheduler._canonical_recovery_job is not None
        assert scheduler._canonical_recovery_blocked_idle_steps == 0

    def test_a_spent_budget_is_not_a_stall(self):
        """Waiting for an allowance is the ordinary case, and parking covers it."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        _spend_the_window(scheduler)
        with patch.object(scheduler, "_specprefill_rope_installed", return_value=True):
            self._run_idle_steps(scheduler, MAX_BLOCKED_IDLE_STEPS * 2)
        assert scheduler._canonical_recovery_job is not None
        assert scheduler._canonical_recovery_blocked_idle_steps == 0

    def test_a_chunk_that_runs_clears_the_deadline(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        with patch.object(scheduler, "_specprefill_rope_installed", return_value=True):
            self._run_idle_steps(scheduler, 10)
        assert scheduler._canonical_recovery_blocked_idle_steps == 10
        with patch.object(scheduler, "_canonical_recovery_step", return_value=True):
            self._run_idle_steps(scheduler, 2)
        assert scheduler._canonical_recovery_blocked_idle_steps == 0


class TestForegroundPriorityIsEngineGlobal:
    """Recovery admission follows engine slack, not one session's idleness.

    Every clause below is a *global* scheduler list. A session waiting on a
    tool is not slack while another request is being served, and the predicate
    has no way to express "this session is idle" even if it wanted to — which
    is the property that keeps recovery out of agent-identity semantics.
    """

    def _idle(self, scheduler: Scheduler) -> None:
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        scheduler._canonical_recovery_note_step(did_foreground_work=False)

    def test_an_idle_engine_admits_a_chunk(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        self._idle(scheduler)
        assert scheduler._canonical_recovery_runnable()

    @pytest.mark.parametrize("queue", ["waiting", "running", "prefilling"])
    def test_any_foreground_queue_withdraws_the_chunk(self, queue):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        self._idle(scheduler)
        assert scheduler._canonical_recovery_runnable()
        held = getattr(scheduler, queue)
        if isinstance(held, dict):
            held["foreground"] = MagicMock()
        else:
            held.append(MagicMock())
        assert not scheduler._canonical_recovery_runnable()

    def test_a_request_that_has_arrived_but_not_been_admitted_withdraws_it(self):
        """The window the scheduler's own lists cannot see.

        Admission runs on the same single-worker executor as step(), so between
        the HTTP layer accepting a request and add_request() running, every
        list says idle while a request is already waiting.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        self._idle(scheduler)
        assert scheduler._canonical_recovery_runnable()
        scheduler.note_inbound_request("inbound-1")
        assert not scheduler._canonical_recovery_runnable()
        # Admission is not the withdrawal: a peer engine cannot see this
        # request's prefill until its first chunk has already run.
        scheduler.note_admitted_request("inbound-1")
        assert not scheduler._canonical_recovery_runnable()
        scheduler.note_request_departed("inbound-1")
        assert scheduler._canonical_recovery_runnable()

    def test_foreground_work_resets_the_idle_run(self):
        """A chunk holds the interpreter for its whole duration, so the rule
        is two idle steps, not one: the second is the window an arriving
        request has to announce itself in."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        self._idle(scheduler)
        assert scheduler._canonical_recovery_runnable()
        scheduler._canonical_recovery_note_step(did_foreground_work=True)
        assert not scheduler._canonical_recovery_runnable()
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        assert not scheduler._canonical_recovery_runnable()
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        assert scheduler._canonical_recovery_runnable()

    def test_a_second_lineage_does_not_buy_a_second_budget(self):
        """The bound is engine-global: one job slot, one budget object.

        Parallel sessions cannot multiply the allowance because there is
        nothing per-session to multiply.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(
            _sparse_request(1000, rid="a", scheduler=scheduler)
        )
        first = scheduler._canonical_recovery_job
        budget = scheduler._canonical_recovery_budget
        other = _sparse_request(1000, rid="b", scheduler=scheduler)
        other.prompt_token_ids = list(range(5000, 6000))
        scheduler.note_canonical_recovery_candidate(other)
        assert scheduler._canonical_recovery_job is not first
        assert scheduler._canonical_recovery_budget is budget
        assert first.cancelled
