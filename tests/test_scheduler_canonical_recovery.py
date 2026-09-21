# SPDX-License-Identifier: Apache-2.0
"""Scheduler-level tests for progressive canonical state recovery.

The model is mocked; what is under test is when the scheduler decides a
recovery chunk may run, what it publishes, and what it releases. Each test in
``TestSafetyReviewConditions`` corresponds to a defect found in an earlier
background-densification prototype, so a regression there fails loudly rather
than being rediscovered by a later review.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omlx.patches.specprefill import (
    _OffsetAdjustedRoPE,
    _PositionMappedRoPE,
    _ROPE_WRAPPERS,
    _unwrap_rope,
    is_specprefill_rope,
)
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
    # A prefix cache must exist for canonical state recovery to be enabled at
    # all; the publish tests replace it with their own double.
    scheduler.block_aware_cache = MagicMock()
    scheduler._unreconstructible_cache_model = False
    mock_bg = MagicMock()
    mock_bg.insert.return_value = [42]
    mock_bg.next_generated.return_value = iter([])
    scheduler.batch_generator = mock_bg
    return scheduler


def _sparse_request(prompt_tokens: int, rid: str = "r1", scheduler=None):
    """A finished sparse request, stamped with the cache that served it.

    The stamp is part of the contract now: a recovery job is only queued by the
    scheduler whose prefix-cache instance actually served the request, because
    one served model can present more than one instance and state published
    into the wrong one is valid, durable and unreachable.
    """
    request = MagicMock()
    request.request_id = rid
    request.prompt_token_ids = list(range(prompt_tokens))
    request.specprefill_indices = [1, 2, 3]
    request._serving_prefix_cache_id = (
        id(scheduler.block_aware_cache) if scheduler is not None else None
    )
    return request


def _raise_on_call(*_args, **_kwargs):
    raise RuntimeError("begin_state failed")


class TestCandidateAdmission:
    def test_a_sparse_request_queues_a_canonical_recovery_job(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        assert scheduler._canonical_recovery_job is not None
        # 1000 tokens, 256-token blocks: only whole blocks are publishable,
        # and the target carries one token past the last boundary so the
        # prefill's held-back final token does not cost the job that block.
        assert scheduler._canonical_recovery_job.target_tokens == 769

    def test_a_dense_request_queues_nothing(self):
        """A dense request already stored its own checkpoint. There is no debt."""
        scheduler = _make_scheduler()
        request = _sparse_request(1000, scheduler=scheduler)
        request.specprefill_indices = None
        scheduler.note_canonical_recovery_candidate(request)
        assert scheduler._canonical_recovery_job is None

    def test_a_prompt_shorter_than_one_block_queues_nothing(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(100, scheduler=scheduler))
        assert scheduler._canonical_recovery_job is None

    def test_an_unreconstructible_model_queues_nothing(self):
        scheduler = _make_scheduler()
        scheduler._unreconstructible_cache_model = True
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        assert scheduler._canonical_recovery_job is None

    def test_disabled_queues_nothing(self):
        scheduler = _make_scheduler(canonical_state_recovery_enabled=False)
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        assert scheduler._canonical_recovery_job is None

    def test_an_append_extends_rather_than_replacing(self):
        """Single-flight: two jobs on one session recompute the same prefix."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        first = scheduler._canonical_recovery_job
        first.committed_tokens = 512
        scheduler.note_canonical_recovery_candidate(_sparse_request(2000, rid="r2", scheduler=scheduler))
        assert scheduler._canonical_recovery_job is first
        assert scheduler._canonical_recovery_job.target_tokens == 1793
        assert scheduler._canonical_recovery_job.committed_tokens == 512

    def test_a_rewritten_prefix_replaces_the_job(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        first = scheduler._canonical_recovery_job
        rewritten = _sparse_request(2000, rid="r2", scheduler=scheduler)
        rewritten.prompt_token_ids[10] = -1
        scheduler.note_canonical_recovery_candidate(rewritten)
        assert scheduler._canonical_recovery_job is not first
        assert scheduler._canonical_recovery_job.committed_tokens == 0


class TestSafetyReviewConditions:
    """One test per defect the earlier prototype's review found."""

    def test_an_active_specprefill_makes_the_canonical_recovery_unrunnable(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._consecutive_idle_steps = 5
        assert scheduler._canonical_recovery_runnable()
        scheduler._specprefill_active_request_id = "r1"
        assert not scheduler._canonical_recovery_runnable()

    def test_an_inbound_request_makes_the_canonical_recovery_unrunnable(self):
        """Arrival visibility: a request exists before its admission runs."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._consecutive_idle_steps = 5
        assert scheduler._canonical_recovery_runnable()
        scheduler.note_inbound_request("incoming")
        assert not scheduler._canonical_recovery_runnable()
        scheduler.note_admitted_request("incoming")
        assert scheduler._canonical_recovery_runnable()

    def test_a_stale_inbound_marker_expires(self):
        """Liveness: an inbound request that is never admitted must not block
        the recovery job for the life of the process."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._consecutive_idle_steps = 5
        scheduler.note_inbound_request("lost")
        assert not scheduler._canonical_recovery_runnable()
        scheduler._canonical_recovery_inbound["lost"] -= scheduler._canonical_recovery_inbound_ttl_s + 1
        assert scheduler._canonical_recovery_runnable()
        assert scheduler._canonical_recovery_inbound_count() == 0

    def test_one_idle_step_is_not_enough_to_start_a_chunk(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._consecutive_idle_steps = 1
        assert not scheduler._canonical_recovery_runnable()
        scheduler._consecutive_idle_steps = 2
        assert scheduler._canonical_recovery_runnable()

    def test_the_unsupported_cache_gate_is_rechecked_before_publishing(self):
        """The gate must be consulted at publish time, not only at queue time.

        A job queued while the model's cache was reconstructible must not
        publish if that answer has changed.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        state = MagicMock(base_size=0, tokens_processed=256)
        scheduler._unreconstructible_cache_model = True
        scheduler._canonical_recovery_publish(job, 256, state)
        scheduler.block_aware_cache.store_cache.assert_not_called()
        assert scheduler._canonical_recovery_job is None

    def test_publication_goes_through_the_locked_store_worker(self):
        """Publication uses the ordinary store worker, with the job's tokens.

        This asserts the route and its arguments only. What the route is worth
        is asserted separately, against the worker itself, in
        TestStoreWorkerLifecycle — a test that mocks the worker cannot say
        anything about what the worker does.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        state = MagicMock(base_size=0, tokens_processed=256, cache=[MagicMock()])
        with patch.object(
            scheduler, "_extract_cache_states", return_value=([{"state": None}], None)
        ), patch.object(
            scheduler, "_collect_arrays_from_extracted_cache", return_value=[]
        ), patch.object(
            scheduler, "_detect_boundary_snapshot_need", return_value=False
        ), patch.object(
            scheduler, "_get_boundary_store_override", return_value=None
        ), patch.object(
            scheduler, "_canonical_recovery_readback_tokens", return_value=256
        ), patch.object(
            scheduler,
            "_async_store_cache_worker",
            return_value=SimpleNamespace(block_ids=[7]),
        ) as worker:
            scheduler._canonical_recovery_publish(job, 256, state)
        worker.assert_called_once()
        assert worker.call_args.args[0] == "canonical-recovery:r1"
        assert worker.call_args.args[1] == list(range(256))
        assert job.committed_tokens == 256

    def test_a_store_that_persisted_nothing_is_not_counted_as_committed(self):
        """The failure that made this check exist.

        A hybrid model's non-sliceable layers cannot be stored without the
        boundary snapshots, and the store declines by stopping at zero tokens
        rather than by raising. The log said the prefix had been published, the
        job recorded the commit, and the next turn restored nothing.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        state = MagicMock(base_size=0, tokens_processed=256, cache=[MagicMock()])
        with patch.object(
            scheduler, "_extract_cache_states", return_value=([{"state": None}], None)
        ), patch.object(
            scheduler, "_collect_arrays_from_extracted_cache", return_value=[]
        ), patch.object(
            scheduler, "_detect_boundary_snapshot_need", return_value=False
        ), patch.object(
            scheduler, "_get_boundary_store_override", return_value=None
        ), patch.object(
            scheduler,
            "_async_store_cache_worker",
            return_value=SimpleNamespace(block_ids=[]),
        ):
            scheduler._canonical_recovery_publish(job, 256, state)
        assert job.committed_tokens == 0
        assert scheduler._canonical_recovery_counters.publishes == 0

    def test_a_hybrid_model_does_not_publish_without_a_boundary_snapshot(self):
        """Publishing the live cache would write placeholders for every block
        but the last, which a later restore rejects."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        state = MagicMock(base_size=0, tokens_processed=256, cache=[MagicMock()])
        with patch.object(
            scheduler, "_get_boundary_store_override", return_value=None
        ), patch.object(
            scheduler, "_detect_boundary_snapshot_need", return_value=True
        ), patch.object(
            scheduler, "_async_store_cache_worker"
        ) as worker:
            scheduler._canonical_recovery_publish(job, 256, state)
        worker.assert_not_called()
        assert job.committed_tokens == 0

    def test_a_truncating_boundary_override_does_not_publish(self):
        """An assertion, not a fix: this should not be reachable.

        A publish only runs on the chunk that landed exactly on the boundary,
        and that chunk emits a boundary snapshot at the same token count before
        it returns, so the override should always end on the boundary. Every
        way the capture can be skipped fails closed instead — no snapshots at
        all leaves `_detect_boundary_snapshot_need` to refuse, and a failed SSD
        load returns no override rather than a short one.

        The invariant is held by that argument and by nothing in the code, so
        it is asserted here. If it ever stops holding:

        `_get_boundary_store_override` truncates to the latest block-aligned
        snapshot at or below the boundary, and it deliberately leaves that
        snapshot out of the provider it returns. The block the range now ends
        on therefore has no snapshot of its own, so `store_cache` falls to
        `live_state_at_true_end` and stores the live recurrent state as that
        block's — but the live state is at the boundary, not at the end of the
        truncated range. A later restore would get a block whose state has
        already ingested tokens past its own end.

        The read-back cannot catch this: it verifies the range is restorable,
        not that what restores is the right state. So the publish is declined.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(
            _sparse_request(1000, scheduler=scheduler)
        )
        job = scheduler._canonical_recovery_job
        state = MagicMock(base_size=0, tokens_processed=8192, cache=[MagicMock()])
        # a snapshot at 4096 and none at 8192: the override truncates
        truncated = (list(range(4096)), [{"state": None}], None, {})
        with patch.object(
            scheduler, "_get_boundary_store_override", return_value=truncated
        ), patch.object(
            scheduler, "_async_store_cache_worker"
        ) as worker:
            scheduler._canonical_recovery_publish(job, 8192, state)
        worker.assert_not_called()
        assert job.committed_tokens == 0

    def test_a_job_whose_committed_prefix_was_evicted_stops(self):
        """Losing ground already verified ends the job, it does not retry.

        A recovery request is in neither `running` nor `prefilling`, and
        `get_active_hot_cache_block_hashes` builds the hot cache's protection
        set from exactly those two, so the job's published blocks are the first
        chosen for eviction. Under `hot_cache_only` an evicted block is dropped
        rather than demoted to SSD, so a hole can open low in the chain.

        Refusing the publish keeps the counter honest about this boundary, but
        `committed_tokens` only rises, so every later boundary sits at or below
        the stale watermark and is refused by the same probe. Left alone the
        job spends its whole allowance committing nothing while still
        reporting a prefix the cache cannot honour.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(
            _sparse_request(1000, scheduler=scheduler)
        )
        job = scheduler._canonical_recovery_job
        job.note_published(8192)
        state = MagicMock(base_size=0, tokens_processed=12288, cache=[MagicMock()])
        with patch.object(
            scheduler, "_get_boundary_store_override", return_value=None
        ), patch.object(
            scheduler, "_detect_boundary_snapshot_need", return_value=False
        ), patch.object(
            scheduler, "_extract_cache_states", return_value=([{"state": None}], None)
        ), patch.object(
            scheduler,
            "_async_store_cache_worker",
            return_value=SimpleNamespace(block_ids=list(range(64))),
        ), patch.object(
            # the chain lost a block below what this job already committed
            scheduler, "_canonical_recovery_readback_tokens", return_value=0
        ):
            scheduler._canonical_recovery_publish(job, 12288, state)

        assert job.committed_tokens == 8192
        assert scheduler._canonical_recovery_job is None

    def test_a_dropped_job_releases_its_cache_footprint(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        with patch.object(
            scheduler, "_release_paged_cache_for_request"
        ) as release, patch.object(
            scheduler, "_drop_boundary_snapshots_for_request"
        ) as drop:
            scheduler._canonical_recovery_drop_job("test")
        release.assert_called_once_with("canonical-recovery:r1")
        drop.assert_called_once_with("canonical-recovery:r1")
        assert scheduler._canonical_recovery_job is None


class TestPublication:
    def test_a_new_boundary_publishes_mid_target(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        job.processed_tokens = 512
        assert job.publishable_boundary() == 512

    def test_publication_is_refused_when_live_state_is_past_the_boundary(self):
        """Storing a state that has ingested tokens past a block as that
        block's state double-ingests them on restore."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        state = MagicMock(base_size=0, tokens_processed=300)
        with patch.object(scheduler, "_async_store_cache_worker") as worker:
            scheduler._canonical_recovery_publish(job, 256, state)
        worker.assert_not_called()
        assert job.committed_tokens == 0


class TestIdleAccounting:
    def test_a_step_with_foreground_work_resets_the_idle_run(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._consecutive_idle_steps = 4
        scheduler._canonical_recovery_note_step(did_foreground_work=True)
        assert scheduler._consecutive_idle_steps == 0
        assert scheduler._canonical_recovery_counters.yielded_steps == 1


class TestLoopLiveness:
    """The engine loop only steps while has_requests() is true.

    A recovery job that is allowed to run only when the engine is idle is,
    without this, never run at all: the loop stops stepping at exactly the
    moment the recovery job becomes eligible.
    """

    def test_a_live_job_keeps_the_loop_stepping(self):
        scheduler = _make_scheduler()
        assert not scheduler.has_requests()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        assert scheduler.has_requests()

    def test_a_finished_job_does_not_hold_the_loop_awake(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        job.processed_tokens = job.target_tokens
        assert not scheduler.has_requests()

    def test_a_cancelled_job_does_not_hold_the_loop_awake(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._canonical_recovery_job.cancelled = True
        assert not scheduler.has_requests()

    def test_a_budget_that_can_never_grant_service_does_not_hold_the_loop_awake(self):
        """Otherwise an idle server polls forever on a job it may not serve.

        Zero percent is not a spent allowance, it is no allowance in any
        window, so the job is never going to run and the loop has no reason to
        keep stepping for it.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._canonical_recovery_budget.pct = 0.0
        assert not scheduler.has_requests()

    def test_a_spent_window_parks_the_loop_and_a_roll_wakes_it(self):
        """An out-of-allowance job is late, and waiting is not work.

        The engine loop re-reads has_requests() once per step_interval whether
        or not it stepped last time, so parking it here costs at most 50 ms of
        latency on the roll. Holding it awake instead costs a full scheduler
        step 20 times a second for a job that may not run, and each of those
        steps advances the counter that gates the process-global Metal cache
        clear.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        budget = scheduler._canonical_recovery_budget
        budget.note_service(budget.window_s)       # far past this window's allowance
        assert not budget.allows()
        assert not scheduler.has_requests()
        assert not scheduler._canonical_recovery_runnable()

        # One window of lockout, because the overrun is carried forward and
        # the carry is capped at a single allowance. The roll has to be taken
        # a window at a time: a carry is computed once per roll however many
        # windows it skips, so jumping two at once discharges nothing.
        budget.window_start_s -= budget.window_s
        assert not budget.allows()
        assert not scheduler.has_requests()

        budget.window_start_s -= budget.window_s
        assert budget.allows()
        assert scheduler.has_requests()


class TestFinishedJobStopsRunning:
    """A job that reached its target must stop being runnable.

    The prefill state is retired on completion, so a finished job that still
    counts as runnable rebuilds it on the next idle step, re-reads its whole
    target from the last committed boundary, publishes nothing new and
    finishes again — every idle window, charged to the budget. This was
    observed: one 8K session spent its entire run re-reading the same 4,096
    tokens.
    """

    def test_a_job_that_reached_its_target_is_not_runnable(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        assert scheduler._canonical_recovery_runnable()
        scheduler._canonical_recovery_job.note_reached_target()
        assert not scheduler._canonical_recovery_runnable()
        assert not scheduler.has_requests()

    def test_an_append_makes_it_runnable_again(self):
        """Finishing retires the work, it does not retire the job: the next
        turn extends the same job rather than starting over from nothing."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._canonical_recovery_job.note_reached_target()
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        assert not scheduler._canonical_recovery_runnable()
        scheduler.note_canonical_recovery_candidate(_sparse_request(2000, scheduler=scheduler))
        assert scheduler._canonical_recovery_runnable()


class TestJobSurvivesATurnWithNothingNew:
    """A turn that adds no new whole block must not destroy the job.

    `extend` refuses a target that did not move, and reading that refusal as
    "not an append" cost the job its committed prefix, the canonical prefix
    it reports, and the append path every later turn would have taken. With
    a 4,096-token block and turns of a few hundred tokens that is most turns.
    """

    def test_a_turn_below_the_next_boundary_keeps_the_job(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        job.note_published(768)
        scheduler.note_canonical_recovery_candidate(_sparse_request(1010, scheduler=scheduler))
        assert scheduler._canonical_recovery_job is job
        assert scheduler._canonical_recovery_job.committed_tokens == 768

    def test_a_turn_past_the_next_boundary_still_extends(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        scheduler.note_canonical_recovery_candidate(_sparse_request(1400, scheduler=scheduler))
        assert scheduler._canonical_recovery_job is job
        assert scheduler._canonical_recovery_job.target_tokens == 1281

    def test_a_rewritten_history_still_replaces_the_job(self):
        """The no-op path must not swallow a prompt that is not an append."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        rewritten = _sparse_request(1000, scheduler=scheduler)
        rewritten.prompt_token_ids = [9] + list(range(1, 1000))
        scheduler.note_canonical_recovery_candidate(rewritten)
        assert scheduler._canonical_recovery_job is not job


class TestParking:
    """A job with nothing to do yet is parked, not destroyed."""

    def test_a_parked_job_is_not_runnable_and_keeps_its_prefix(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        job.note_published(768)
        scheduler._canonical_recovery_park_job(job, "nothing_to_do")
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        assert scheduler._canonical_recovery_job is job
        assert job.committed_tokens == 768
        assert not scheduler._canonical_recovery_runnable()

    def test_an_append_wakes_a_parked_job(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        scheduler._canonical_recovery_park_job(job, "nothing_to_do")
        scheduler.note_canonical_recovery_candidate(_sparse_request(1400, scheduler=scheduler))
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        assert scheduler._canonical_recovery_job is job
        assert scheduler._canonical_recovery_runnable()


class TestTheWholeStepIsCharged:
    """Restoring and publishing hold the engine thread exactly as the forward
    does, so a budget that charged only the forward reported a lower bound on
    the recovery's real wall cost rather than a measurement of it."""

    def test_a_step_that_never_reaches_the_model_still_costs_budget(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._canonical_recovery_begin_state = _raise_on_call
        assert not scheduler._canonical_recovery_step()
        assert scheduler._canonical_recovery_counters.service_s > 0
        assert scheduler._canonical_recovery_budget.service_s > 0


class TestTelemetrySurvivesNothing:
    """An engine switch must not splice two runs into one share.

    `reset()` cancels the recovery job, and the counters and the budget's own
    wall clock used to survive it. The two service shares then covered two
    runs with nothing on the wire saying so.
    """

    def test_reset_clears_the_recovery_telemetry(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._canonical_recovery_counters.service_s = 12.0
        scheduler._canonical_recovery_budget.note_service(12.0)
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        scheduler.reset()
        assert scheduler._canonical_recovery_counters.service_s == 0.0
        assert scheduler._canonical_recovery_budget.service_s == 0.0

    def test_a_shared_budget_is_not_reset_by_one_engine(self):
        """The counters are this scheduler's; the budget is not."""
        scheduler = _make_scheduler()
        scheduler._canonical_recovery_budget.shared = True
        scheduler._canonical_recovery_budget.note_service(12.0)
        scheduler._canonical_recovery_counters.service_s = 12.0
        scheduler.reset()
        assert scheduler._canonical_recovery_counters.service_s == 0.0
        assert scheduler._canonical_recovery_budget.service_s == 12.0


class TestIdleAccounting:
    """The recovery job must not count itself as the work that blocks it."""

    def test_a_live_job_alone_does_not_reset_the_idle_run(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        assert scheduler.has_requests()          # keeps the loop stepping
        assert not scheduler._canonical_recovery_foreground_busy()
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        assert scheduler._consecutive_idle_steps == 2
        assert scheduler._canonical_recovery_runnable()

    def test_an_active_specprefill_counts_as_foreground_busy(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._specprefill_active_request_id = "r1"
        assert scheduler._canonical_recovery_foreground_busy()
        scheduler._canonical_recovery_note_step(did_foreground_work=False)
        assert scheduler._consecutive_idle_steps == 0


class TestCancellation:
    """Background work must never be the reason a model cannot be unloaded.

    The unload path drains on the same predicate the recovery job uses to keep
    the engine loop stepping. Without an explicit cancel, an engine with a live
    recovery job never becomes quiescent: the unload is queued "until active
    scheduler work drains", it never drains, and every later request to that
    model is refused with 409.
    """

    def test_cancel_canonical_recovery_work_makes_the_scheduler_quiescent(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        assert scheduler.has_requests()
        assert scheduler.cancel_canonical_recovery_work("unload") is True
        assert not scheduler.has_requests()
        assert scheduler._canonical_recovery_job is None

    def test_cancel_is_idempotent_and_reports_no_work(self):
        scheduler = _make_scheduler()
        assert scheduler.cancel_canonical_recovery_work("unload") is False

    def test_a_cancelled_job_releases_its_own_footprint_and_nothing_else(self):
        """The drop path releases the job's in-flight footprint.

        It must not try to unpublish: published blocks are ordinary canonical
        state and the next request is entitled to restore from them. There is
        no unpublish call to assert the absence of, so this asserts the two
        things the drop path does do, and that the published boundary survives
        on the job object that recorded it.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        job.note_published(512)
        with patch.object(
            scheduler, "_release_paged_cache_for_request"
        ) as release, patch.object(
            scheduler, "_drop_boundary_snapshots_for_request"
        ) as drop:
            scheduler.cancel_canonical_recovery_work("unload")
        release.assert_called_once_with("canonical-recovery:r1")
        drop.assert_called_once_with("canonical-recovery:r1")
        assert job.published_boundaries == [512]


class TestStoreWorkerLifecycle:
    """What routing through the store worker is actually worth.

    The worker's tail releases the stored blocks for eviction and drops the
    request entry. That is right once, at request completion. At every boundary
    of a job that is still running it is wrong twice over: dropping the entry
    makes the next publish re-serialize the whole prefix instead of appending
    to it, and releasing the blocks lets the prefix the job is still building on
    be evicted underneath it.
    """

    def _scheduler_with_cache(self):
        scheduler = _make_scheduler()
        scheduler.paged_cache_manager = MagicMock()
        scheduler.block_aware_cache.store_cache.return_value = SimpleNamespace(
            block_ids=[1, 2]
        )
        return scheduler

    def test_a_completed_request_releases_and_clears(self):
        scheduler = self._scheduler_with_cache()
        scheduler._async_store_cache_worker(
            "r1", [1, 2, 3], [{"state": None}], None, None, None, None, None, True
        )
        scheduler.paged_cache_manager.release_for_eviction.assert_called_once_with(
            [1, 2]
        )
        scheduler.block_aware_cache.clear_request_entry.assert_called_once_with("r1")

    def test_a_retained_entry_is_neither_released_nor_cleared(self):
        scheduler = self._scheduler_with_cache()
        scheduler._async_store_cache_worker(
            "canonical-recovery:r1", [1, 2, 3], [{"state": None}], None, None, None, None, None,
            True, retain_request_entry=True,
        )
        scheduler.paged_cache_manager.release_for_eviction.assert_not_called()
        scheduler.block_aware_cache.clear_request_entry.assert_not_called()

    def test_the_worker_reports_what_it_stored(self):
        """A store that declined stops at zero tokens rather than raising, so
        the caller has to be able to see the difference."""
        scheduler = self._scheduler_with_cache()
        table = scheduler._async_store_cache_worker(
            "r1", [1, 2, 3], [{"state": None}], None, None, None, None, None, True
        )
        assert table.block_ids == [1, 2]

    def test_a_live_job_publishes_with_the_entry_retained(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(2000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        state = MagicMock(base_size=0, tokens_processed=256, cache=[MagicMock()])
        with patch.object(
            scheduler, "_extract_cache_states", return_value=([{"state": None}], None)
        ), patch.object(
            scheduler, "_collect_arrays_from_extracted_cache", return_value=[]
        ), patch.object(
            scheduler, "_detect_boundary_snapshot_need", return_value=False
        ), patch.object(
            scheduler, "_get_boundary_store_override", return_value=None
        ), patch.object(
            scheduler, "_canonical_recovery_readback_tokens", return_value=256
        ), patch.object(
            scheduler,
            "_async_store_cache_worker",
            return_value=SimpleNamespace(block_ids=[7]),
        ) as worker:
            scheduler._canonical_recovery_publish(job, 256, state)
        assert worker.call_args.kwargs["retain_request_entry"] is True


class TestRopeGuard:
    """The guard has to read the model, not the bookkeeping variable.

    `_handle_prefill_oom` clears `_specprefill_active_request_id` without
    calling `cleanup_rope`, and the patch module documents the leftover wrapper
    as an expected state. A dense forward taken while it is installed reads
    another request's position offset, and the recovery job would publish
    positionally wrong KV as ordinary canonical state.
    """

    def _layer_with_rope(self, rope):
        attn = SimpleNamespace(rope=rope)
        return SimpleNamespace(self_attn=attn)

    def test_a_leftover_wrapper_blocks_the_canonical_recovery_with_the_id_already_clear(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._consecutive_idle_steps = 5
        assert scheduler._specprefill_active_request_id is None

        wrapper = _OffsetAdjustedRoPE(SimpleNamespace(), adjustment=8)
        scheduler.model.layers = [self._layer_with_rope(wrapper)]
        assert scheduler._specprefill_rope_installed()
        assert not scheduler._canonical_recovery_runnable()

    def test_the_guard_does_not_depend_on_the_wrapper_class_name(self):
        """A rename must not turn the guard off.

        The guard used to compare ``type(rope).__name__`` against two literals,
        so renaming a wrapper in the patch module would have reported that no
        wrapper was installed and allowed the dense forward the guard exists to
        prevent — a failure that is silent and open, and that no test with a
        correctly named stub could have caught.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._consecutive_idle_steps = 5

        wrapper = _PositionMappedRoPE(SimpleNamespace(dims=64, base=10000.0, scale=1.0), [0, 1, 2])
        renamed = type("SomethingElseEntirely", (type(wrapper),), {})
        wrapper.__class__ = renamed
        assert type(wrapper).__name__ not in (
            "_OffsetAdjustedRoPE",
            "_PositionMappedRoPE",
        )

        scheduler.model.layers = [self._layer_with_rope(wrapper)]
        assert scheduler._specprefill_rope_installed()
        assert not scheduler._canonical_recovery_runnable()

    def test_every_registered_wrapper_both_unwinds_and_is_recognised(self):
        """The two questions read one registry and cannot disagree.

        A wrapper that unwinds but is not recognised lets a dense forward run
        under a sparse prefill's positions; one that is recognised but does not
        unwind nests on the next request. Both follow from a wrapper being
        added to the module and left out of ``_ROPE_WRAPPERS``.
        """
        original = SimpleNamespace(dims=64, base=10000.0, scale=1.0)
        built = {
            _OffsetAdjustedRoPE: _OffsetAdjustedRoPE(original, adjustment=8),
            _PositionMappedRoPE: _PositionMappedRoPE(original, [0, 1, 2]),
        }
        assert set(built) == set(_ROPE_WRAPPERS)
        for wrapper in built.values():
            assert is_specprefill_rope(wrapper)
            assert _unwrap_rope(wrapper) is original

    def test_an_ordinary_rope_does_not_block_recovery(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._consecutive_idle_steps = 5

        class RoPE:
            pass

        scheduler.model.layers = [self._layer_with_rope(RoPE())]
        assert not is_specprefill_rope(scheduler.model.layers[0].self_attn.rope)
        assert not scheduler._specprefill_rope_installed()
        assert scheduler._canonical_recovery_runnable()

    def test_a_model_that_cannot_be_inspected_is_treated_as_patched(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        scheduler._consecutive_idle_steps = 5
        type(scheduler.model).layers = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("no layers"))
        )
        try:
            assert scheduler._specprefill_rope_installed()
            assert not scheduler._canonical_recovery_runnable()
        finally:
            del type(scheduler.model).layers


class TestYieldBudget:
    """A yield that nothing ever satisfies is not a pause.

    The memory throttle is not something the recovery job can satisfy by
    waiting, so an unbounded retry keeps the job live, keeps `has_requests()`
    true, and spins an idle engine forever while holding the job's prefill
    state.
    """

    def test_repeated_yields_eventually_drop_the_job(self):
        from omlx.canonical_recovery import MAX_CONSECUTIVE_YIELDS

        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        for _ in range(MAX_CONSECUTIVE_YIELDS - 1):
            scheduler._canonical_recovery_note_yield(job, "the throttle")
        assert scheduler._canonical_recovery_job is job
        scheduler._canonical_recovery_note_yield(job, "the throttle")
        assert scheduler._canonical_recovery_job is None

    def test_a_chunk_that_ran_clears_the_yield_run(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        scheduler._canonical_recovery_note_yield(job, "the throttle")
        scheduler._canonical_recovery_note_yield(job, "the throttle")
        assert job.consecutive_yields == 2
        state = MagicMock(
            base_size=0, tokens_processed=100,
            canonical_recovery_target_tokens=job.target_tokens,
        )
        job.prefill_state = state
        with patch.object(scheduler, "_step_prefill_chunk", return_value=False):
            scheduler._canonical_recovery_step()
        assert job.consecutive_yields == 0


class TestPerModelSettings:
    """SchedulerConfig is one object shared by every engine in the pool.

    A scheduler that read the feature flag on every step would have it switched
    on and off under it whenever another model was loaded.
    """

    def test_a_later_config_change_does_not_reach_a_live_scheduler(self):
        scheduler = _make_scheduler(canonical_state_recovery_enabled=True)
        assert scheduler._canonical_recovery_enabled()
        scheduler.config.canonical_state_recovery_enabled = False
        assert scheduler._canonical_recovery_enabled()


class TestDisabledIsInert:
    def test_a_disabled_scheduler_does_no_canonical_recovery_work_in_a_step(self):
        scheduler = _make_scheduler(canonical_state_recovery_enabled=False)
        with patch.object(scheduler, "_canonical_recovery_step") as step, patch.object(
            scheduler, "_canonical_recovery_note_step"
        ) as note:
            scheduler.step()
        step.assert_not_called()
        note.assert_not_called()


class TestServingCacheBinding:
    """A recovery job belongs to the prefix-cache instance that served its request.

    One served model can present more than one `BlockAwarePrefixCache`. State
    published into the instance that did not serve the request is valid,
    durable and unreachable — a restore on the serving path never looks there.
    A traced run showed exactly that: the publication succeeded, the sidecar
    committed, a restore against the publishing cache recovered the boundary in
    full, and the foreground cache saw nothing.
    """

    def test_a_request_this_cache_did_not_serve_is_declined(self):
        scheduler = _make_scheduler()
        request = _sparse_request(1000)          # no serving stamp at all
        scheduler.note_canonical_recovery_candidate(request)
        assert scheduler._canonical_recovery_job is None

    def test_a_request_served_by_another_instance_is_declined(self):
        scheduler = _make_scheduler()
        request = _sparse_request(1000, scheduler=scheduler)
        request._serving_prefix_cache_id = id(scheduler.block_aware_cache) + 1
        scheduler.note_canonical_recovery_candidate(request)
        assert scheduler._canonical_recovery_job is None

    def test_a_queued_job_records_the_instance_it_is_bound_to(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        assert scheduler._canonical_recovery_job.serving_cache_id == id(scheduler.block_aware_cache)

    def test_publication_fails_closed_when_the_instance_changes(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        state = MagicMock(base_size=0, tokens_processed=256, cache=[MagicMock()])
        scheduler.block_aware_cache = MagicMock()      # a different instance
        with patch.object(scheduler, "_async_store_cache_worker") as worker:
            scheduler._canonical_recovery_publish(job, 256, state)
        worker.assert_not_called()
        assert job.committed_tokens == 0
        assert scheduler._canonical_recovery_job is None           # dropped, not retried


class TestRestorableInvariant:
    """canonical_committed_tokens <= independently_restorable_tokens.

    A store that reports success is not evidence of canonical publication. The
    counter advances only after the ordinary matching path, on the serving
    cache, can see the boundary.
    """

    def _publish(self, scheduler, job, readback):
        state = MagicMock(base_size=0, tokens_processed=256, cache=[MagicMock()])
        with patch.object(
            scheduler, "_extract_cache_states", return_value=([{"state": None}], None)
        ), patch.object(
            scheduler, "_collect_arrays_from_extracted_cache", return_value=[]
        ), patch.object(
            scheduler, "_detect_boundary_snapshot_need", return_value=False
        ), patch.object(
            scheduler, "_get_boundary_store_override", return_value=None
        ), patch.object(
            scheduler, "_canonical_recovery_readback_tokens", return_value=readback
        ), patch.object(
            scheduler,
            "_async_store_cache_worker",
            return_value=SimpleNamespace(block_ids=[7]),
        ):
            scheduler._canonical_recovery_publish(job, 256, state)

    def test_an_unrestorable_publication_does_not_advance_the_counter(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        self._publish(scheduler, job, readback=0)
        assert job.committed_tokens == 0
        assert scheduler._canonical_recovery_counters.publishes == 0

    def test_a_restorable_publication_advances_it(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        self._publish(scheduler, job, readback=256)
        assert job.committed_tokens == 256
        assert scheduler._canonical_recovery_counters.publishes == 1

    def test_a_partial_readback_does_not_advance_it(self):
        """The invariant is <=, so a boundary only partly visible is not one."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        self._publish(scheduler, job, readback=128)
        assert job.committed_tokens == 0

    def test_the_readback_probe_releases_its_own_entry(self):
        """It runs the real lookup, so it must not leave a block table behind."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(_sparse_request(1000, scheduler=scheduler))
        job = scheduler._canonical_recovery_job
        scheduler.block_aware_cache.fetch_cache.return_value = (
            SimpleNamespace(num_tokens=256),
            [],
        )
        assert scheduler._canonical_recovery_readback_tokens(job, list(range(256))) == 256
        probe_id = "canonical-recovery-readback:r1"
        scheduler.block_aware_cache.release_cache.assert_any_call(probe_id)
        scheduler.block_aware_cache.clear_request_entry.assert_any_call(probe_id)
