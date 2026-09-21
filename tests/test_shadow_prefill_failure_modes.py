# SPDX-License-Identifier: Apache-2.0
"""Failure injection for progressive shadow prefill.

``test_scheduler_shadow_prefill.py`` asks what the recovery job does when its
collaborators behave. This file asks what it does when they do not.

The invariant is **fail closed**. A recovery failure may cost reuse — that is
the whole of what it is allowed to cost. It must never cause a foreground
request to fail, invalid canonical state to become visible, a false committed
token count, a leaked cache or block reference, an unbounded queue, or an
engine that cannot be unloaded (a live job holds the engine loop awake, and the
unload path drains on the same predicate).

Every test asserts five things about one injected fault:

a. no exception escapes into the caller — the scheduler step path returns;
b. ``_shadow_job`` is left somewhere defensible, and the test says which of
   dropped / parked / intact it observed;
c. ``job.committed_tokens`` advanced only behind a real publish *and* read-back;
d. the cleanup calls fired for ``shadow:{session_key}`` — the paged-cache
   release, the boundary-snapshot drop, the prefill tracker's ``remove`` and
   the ``requests`` pop;
e. ``has_requests()`` afterwards reflects reality, because a dropped job that
   still reports work spins an idle engine and blocks model unload.

These tests document what the code does, not what it should do. Where a fault
leaves state that this file judges wrong, the docstring says so under "Gap" and
the assertion still pins the behaviour that is actually there, so a fix changes
a test rather than being discovered by a later review.
"""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omlx.request import Request, SamplingParams
from omlx.scheduler import (
    PrefillEvictionRequest,
    Scheduler,
    SchedulerConfig,
    _PrefillAbortedError,
    _PrefillEvictionNeeded,
)
from omlx.shadow_prefill import MAX_CONSECUTIVE_YIELDS


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
        shadow_prefill_enabled=True,
        shadow_prefill_global_budget_pct=10.0,
    )
    config_kwargs.update(config_over)
    scheduler = Scheduler(
        model=model, tokenizer=tokenizer, config=SchedulerConfig(**config_kwargs)
    )
    # A prefix cache must exist for the shadow to be enabled at all; the
    # publish tests replace it with their own double.
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


# --------------------------------------------------------------------------
# Fault-injection scaffolding
# --------------------------------------------------------------------------

BLOCK = 256
RID = "shadow:r1"
PROBE_ID = "shadow-readback:r1"


class _RecordingRequests(dict):
    """``scheduler.requests`` that remembers the ids popped out of it.

    A spy rather than a mock: the pop still happens, so "the entry was popped"
    and "the entry is gone" are the same assertion rather than two.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.popped: list[str] = []

    def pop(self, key, *default):
        self.popped.append(key)
        return super().pop(key, *default)


@contextmanager
def _cleanup_spy(scheduler, rid: str = RID):
    """Watch the four calls that give a shadow request's footprint back.

    The request entry and the prepared-prefix marker are seeded first, so the
    removals are observable as state changes and not only as mock calls.
    """
    scheduler.requests = _RecordingRequests(scheduler.requests)
    scheduler.requests[rid] = MagicMock(name="live-shadow-request")
    scheduler._prefix_cache_prepared.add(rid)
    tracker = MagicMock()
    # The spy exists to watch `remove`. Left as a bare MagicMock its
    # `any_active` returns a truthy Mock, which the scheduler reads as "some
    # other engine is prefilling" — so every predicate that consults the
    # process-global tracker silently answers the opposite of what the test
    # set up.
    tracker.any_active.return_value = False
    tracker.recently_active.return_value = False
    with ExitStack() as stack:
        release = stack.enter_context(
            patch.object(scheduler, "_release_paged_cache_for_request")
        )
        drop = stack.enter_context(
            patch.object(scheduler, "_drop_boundary_snapshots_for_request")
        )
        stack.enter_context(
            patch("omlx.scheduler.get_prefill_tracker", return_value=tracker)
        )
        yield SimpleNamespace(
            release=release,
            drop=drop,
            tracker=tracker,
            requests=scheduler.requests,
            prepared=scheduler._prefix_cache_prepared,
            rid=rid,
        )


def _assert_cleanup_fired(spy):
    spy.release.assert_any_call(spy.rid)
    spy.drop.assert_any_call(spy.rid)
    spy.tracker.remove.assert_any_call(spy.rid)
    assert spy.rid in spy.requests.popped
    assert spy.rid not in spy.requests
    assert spy.rid not in spy.prepared


def _assert_cleanup_not_fired(spy):
    spy.release.assert_not_called()
    spy.drop.assert_not_called()
    spy.tracker.remove.assert_not_called()
    assert spy.rid not in spy.requests.popped
    assert spy.rid in spy.requests


def _queued(scheduler, prompt_tokens: int = 1000):
    scheduler.note_shadow_candidate(
        _sparse_request(prompt_tokens, scheduler=scheduler)
    )
    job = scheduler._shadow_job
    assert job is not None
    return job


def _live_state(job, *, processed: int, base: int = 0):
    """A prefill state that is mid-chunk, at ``base + processed`` tokens."""
    return MagicMock(
        base_size=base,
        tokens_processed=processed,
        cache=[MagicMock()],
        shadow_target_tokens=job.target_tokens,
    )


def _eviction_needed() -> _PrefillEvictionNeeded:
    return _PrefillEvictionNeeded(
        PrefillEvictionRequest(
            request_id=RID,
            model_id="m",
            current_bytes=1,
            target_cap_bytes=1,
            predicted_transient_bytes=1,
            requested_tokens=64,
            reason="the memory throttle",
        )
    )


_EXTRACTED = ([{"state": None}], None)
_STORED_ONE_BLOCK = SimpleNamespace(block_ids=[7])


def _as_patch_kwargs(value):
    if isinstance(value, BaseException):
        return {"side_effect": value}
    return {"return_value": value}


@contextmanager
def _publish_path(
    scheduler,
    *,
    extract=_EXTRACTED,
    worker=_STORED_ONE_BLOCK,
    readback=BLOCK,
    snapshot_needed=False,
):
    """Stub everything ``_shadow_publish`` talks to below the fault under test.

    ``extract`` and ``worker`` take either a return value or an exception
    instance to raise; ``readback=None`` leaves the real read-back probe in
    place so the cache double answers it.
    """
    with ExitStack() as stack:
        enter = stack.enter_context
        enter(
            patch.object(
                scheduler, "_extract_cache_states", **_as_patch_kwargs(extract)
            )
        )
        enter(
            patch.object(
                scheduler, "_collect_arrays_from_extracted_cache", return_value=[]
            )
        )
        enter(
            patch.object(
                scheduler, "_detect_boundary_snapshot_need",
                return_value=snapshot_needed,
            )
        )
        enter(patch.object(scheduler, "_get_boundary_store_override", return_value=None))
        worker_mock = enter(
            patch.object(
                scheduler, "_async_store_cache_worker", **_as_patch_kwargs(worker)
            )
        )
        readback_mock = None
        if readback is not None:
            readback_mock = enter(
                patch.object(
                    scheduler, "_shadow_readback_tokens", return_value=readback
                )
            )
        yield SimpleNamespace(worker=worker_mock, readback=readback_mock)


# --------------------------------------------------------------------------
# 1-2: the state the chunk runs on cannot be built
# --------------------------------------------------------------------------


class TestStateBuildFails:
    """`_shadow_begin_state` raises: the job is dropped, not retried."""

    def test_a_failed_state_build_drops_the_job_and_frees_the_loop(self):
        """Fault 1. Dropped. Nothing published, everything released.

        The drop is the right answer rather than a retry because the failure
        is in building the state, so a retry would fail the same way every
        idle window and charge the budget for it.
        """
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        job.note_published(512)

        with _cleanup_spy(scheduler) as spy, patch.object(
            scheduler,
            "_shadow_begin_state",
            side_effect=RuntimeError("state build failed"),
        ):
            # (a) the step path returns rather than raising.
            assert scheduler._shadow_step() is False
            # (d) the footprint went back.
            _assert_cleanup_fired(spy)

        # (b) dropped.
        assert scheduler._shadow_job is None
        assert job.cancelled is True
        assert job.prefill_state is None
        # (c) the committed prefix is the one a real publish left; the failure
        #     did not advance it, and the drop does not unpublish it either.
        assert job.committed_tokens == 512
        assert scheduler._shadow_counters.publishes == 0
        # (e) a dropped job must not hold the engine loop awake.
        assert scheduler.has_requests() is False

    def test_a_failed_state_build_still_charges_its_seconds(self):
        """The failed step held the engine thread, so it is service.

        Reporting it as free would understate the recovery's wall cost by
        exactly the failures, which is the wrong direction for a budget.
        """
        scheduler = _make_scheduler()
        _queued(scheduler)
        with patch.object(
            scheduler, "_shadow_begin_state", side_effect=RuntimeError("boom")
        ):
            scheduler._shadow_step()
        assert scheduler._shadow_counters.service_s > 0
        assert scheduler._shadow_budget.service_s > 0


class TestNothingToReRead:
    """`_shadow_begin_state` returns None: the job is parked, not destroyed."""

    def test_nothing_to_re_read_parks_the_job_and_keeps_its_prefix(self):
        """Fault 2. Parked. The job survives; its committed prefix survives.

        Destroying it would forfeit the committed prefix it reports and the
        append path a later turn would take, so the next turn would pay a full
        prefix reconstruct to learn what this job already knew.
        """
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        job.note_published(768)

        with _cleanup_spy(scheduler) as spy, patch.object(
            scheduler, "_shadow_begin_state", return_value=None
        ):
            # (a)
            assert scheduler._shadow_step() is False
            # (d) park releases the same four things a drop does.
            _assert_cleanup_fired(spy)

        # (b) parked: same object, still installed.
        assert scheduler._shadow_job is job
        assert job.cancelled is False
        assert job.prefill_state is None
        assert job.reached_target is True
        # (c)
        assert job.committed_tokens == 768
        assert scheduler._shadow_counters.publishes == 0
        # (e) a parked job is `done`, so it is neither runnable nor work.
        scheduler._shadow_note_step(did_foreground_work=False)
        scheduler._shadow_note_step(did_foreground_work=False)
        assert scheduler._shadow_runnable() is False
        assert scheduler.has_requests() is False

    def test_a_parked_job_wakes_on_an_append_rather_than_on_a_retry(self):
        """Parking is not a drop: the job is still there to be extended.

        This is what makes the park worth more than the drop, so it is
        asserted rather than left implied by "the object survived".
        """
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        with patch.object(scheduler, "_shadow_begin_state", return_value=None):
            scheduler._shadow_step()
        assert scheduler._shadow_job is job
        scheduler.note_shadow_candidate(
            _sparse_request(1400, scheduler=scheduler)
        )
        scheduler._shadow_note_step(did_foreground_work=False)
        scheduler._shadow_note_step(did_foreground_work=False)
        assert scheduler._shadow_job is job
        assert scheduler._shadow_runnable() is True
        assert scheduler.has_requests() is True


# --------------------------------------------------------------------------
# 3-5: the chunk itself fails
# --------------------------------------------------------------------------


class TestChunkYields:
    """A chunk that did not run is a pause, and the pause is bounded.

    Dropping on the first throttle reading cost the shadow a 12,288-token
    prefix to a transient memory sample. Retrying forever is the opposite
    failure: nothing the shadow does satisfies the throttle, so the job stays
    live, `has_requests()` stays true, and an idle engine spins holding the
    job's whole prefill state resident.
    """

    def _drive_to_the_limit(self, scheduler, job, state, error_factory):
        for turn in range(1, MAX_CONSECUTIVE_YIELDS + 1):
            with patch.object(
                scheduler, "_step_prefill_chunk", side_effect=error_factory()
            ):
                # (a) every one of them returns rather than raising.
                assert scheduler._shadow_step() is False
            if turn < MAX_CONSECUTIVE_YIELDS:
                # (b) intact, mid-pause.
                assert scheduler._shadow_job is job
                assert job.consecutive_yields == turn
                assert job.prefill_state is state
                # (e) a job that is only pausing is still work.
                assert scheduler.has_requests() is True

    def test_an_eviction_yield_keeps_the_job_then_the_limit_drops_it(self):
        """Fault 3. Intact while yielding, dropped at the limit."""
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        job.note_published(512)
        state = _live_state(job, processed=100)
        job.prefill_state = state

        with _cleanup_spy(scheduler) as spy:
            self._drive_to_the_limit(scheduler, job, state, _eviction_needed)
            # (d) the give-up drops through the ordinary drop path.
            _assert_cleanup_fired(spy)

        # (b) dropped, after exactly MAX_CONSECUTIVE_YIELDS.
        assert scheduler._shadow_job is None
        assert job.consecutive_yields == MAX_CONSECUTIVE_YIELDS
        assert scheduler._shadow_counters.yielded_steps == MAX_CONSECUTIVE_YIELDS
        # (c) a yield publishes nothing and forfeits nothing.
        assert job.committed_tokens == 512
        assert job.published_boundaries == [512]
        assert scheduler._shadow_counters.publishes == 0
        # (e)
        assert scheduler.has_requests() is False

    def test_an_aborted_chunk_yields_on_the_same_terms(self):
        """Fault 4. `_PrefillAbortedError` is the same pause as the throttle."""
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        job.note_published(512)
        state = _live_state(job, processed=100)
        job.prefill_state = state

        def _aborted():
            return _PrefillAbortedError([42], 100)

        with _cleanup_spy(scheduler) as spy:
            self._drive_to_the_limit(scheduler, job, state, _aborted)
            _assert_cleanup_fired(spy)

        assert scheduler._shadow_job is None
        assert job.consecutive_yields == MAX_CONSECUTIVE_YIELDS
        assert job.committed_tokens == 512
        assert scheduler._shadow_counters.publishes == 0
        assert scheduler.has_requests() is False

    def test_a_single_yield_does_not_drop_or_publish(self):
        """The one-shot case, stated on its own so the limit test is not the
        only evidence that a yield is survivable."""
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        job.note_published(512)
        state = _live_state(job, processed=100)
        job.prefill_state = state

        with _cleanup_spy(scheduler) as spy:
            with patch.object(
                scheduler, "_step_prefill_chunk", side_effect=_eviction_needed()
            ):
                assert scheduler._shadow_step() is False
            _assert_cleanup_not_fired(spy)

        assert scheduler._shadow_job is job
        assert job.prefill_state is state
        assert job.committed_tokens == 512
        assert scheduler.has_requests() is True


class TestChunkFails:
    def test_a_generic_chunk_failure_drops_the_job(self):
        """Fault 5. Dropped. A chunk that raised anything else is not a pause.

        The state it was running on is not trustworthy after an arbitrary
        exception — the cache may be half-ingested — so nothing is published
        from it and the job goes.
        """
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        job.note_published(512)
        state = _live_state(job, processed=100)
        job.prefill_state = state

        with _cleanup_spy(scheduler) as spy, patch.object(
            scheduler, "_step_prefill_chunk", side_effect=RuntimeError("chunk blew up")
        ):
            # (a)
            assert scheduler._shadow_step() is False
            # (d)
            _assert_cleanup_fired(spy)

        # (b) dropped.
        assert scheduler._shadow_job is None
        assert job.cancelled is True
        assert job.prefill_state is None
        # (c) the already-published prefix stands; nothing new was counted.
        assert job.committed_tokens == 512
        assert job.published_boundaries == [512]
        assert scheduler._shadow_counters.publishes == 0
        assert scheduler._shadow_counters.chunks == 0
        # (e)
        assert scheduler.has_requests() is False


# --------------------------------------------------------------------------
# 6-9: publication fails, at each of the four places it can
# --------------------------------------------------------------------------


class TestPublishFailsWithoutCommitting:
    """A publish that fails costs reuse. It must not cost accounting.

    Every test here asserts the same shape: the publish returns, the counter
    does not move, the boundary list stays empty, and the job is left intact
    and runnable rather than dropped — because the next boundary may well
    publish, and the fault is in one attempt rather than in the job.
    """

    def _assert_nothing_was_published(self, scheduler, job):
        assert job.committed_tokens == 0
        assert job.published_boundaries == []
        assert scheduler._shadow_counters.publishes == 0

    def test_a_failed_cache_extraction_publishes_nothing(self):
        """Fault 6. Intact. `_extract_cache_states` raised inside the publish.

        The extraction is where the live cache becomes a storable payload, so
        a failure here means there is nothing to store — and the job must not
        record a commit for a boundary that never left the live cache.
        """
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        state = _live_state(job, processed=BLOCK)

        with _cleanup_spy(scheduler) as spy, _publish_path(
            scheduler, extract=RuntimeError("extract failed")
        ) as path:
            # (a)
            assert scheduler._shadow_publish(job, BLOCK, state) is None
            # The store was never reached.
            path.worker.assert_not_called()
            # (d) nothing was released, because nothing was dropped.
            _assert_cleanup_not_fired(spy)

        # (b) intact.
        assert scheduler._shadow_job is job
        assert job.cancelled is False
        # (c)
        self._assert_nothing_was_published(scheduler, job)
        # (e) a live job is still work.
        assert scheduler.has_requests() is True

    def test_a_failed_store_worker_publishes_nothing(self):
        """Fault 7. Intact. The store worker raised.

        Gap: the worker may have taken block references before raising, and
        the publish path releases nothing on this branch — it returns. The
        references are given back only when the job is later dropped, parked
        or finished, each of which calls `_release_paged_cache_for_request`
        for the same request id. This test asserts that deferral rather than
        an immediate release, because deferral is what the code does
        (omlx/scheduler.py:13761-13763).
        """
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        state = _live_state(job, processed=BLOCK)

        with _cleanup_spy(scheduler) as spy, _publish_path(
            scheduler, worker=RuntimeError("store worker failed")
        ) as path:
            # (a)
            assert scheduler._shadow_publish(job, BLOCK, state) is None
            path.worker.assert_called_once()
            # No read-back was attempted: there is nothing to read back.
            path.readback.assert_not_called()
            # (d) deferred, not immediate.
            _assert_cleanup_not_fired(spy)

        # (b) intact.
        assert scheduler._shadow_job is job
        # (c)
        self._assert_nothing_was_published(scheduler, job)
        # (e)
        assert scheduler.has_requests() is True

        # The deferral is only sound if the eventual drop does release it.
        with _cleanup_spy(scheduler) as spy:
            scheduler._shadow_drop_job("later")
            _assert_cleanup_fired(spy)

    def test_a_store_that_persisted_less_than_claimed_is_not_counted(self):
        """Fault 8. Intact. The store reported success and wrote less.

        A hybrid model's non-sliceable layers cannot be stored without the
        boundary snapshots, and the store declines by stopping short rather
        than by raising. Counting the claim rather than the result is how the
        log came to say a prefix had been published while the next turn
        restored nothing.
        """
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        job.note_published(512)                 # two blocks already canonical
        state = _live_state(job, processed=768)  # live state sits on block three

        with _cleanup_spy(scheduler) as spy, _publish_path(
            scheduler,
            worker=SimpleNamespace(block_ids=[1]),   # one block: 256 tokens
            readback=768,                            # the read-back would have passed
        ) as path:
            # (a)
            assert scheduler._shadow_publish(job, 768, state) is None
            path.worker.assert_called_once()
            # The short store is caught before the read-back, so the probe is
            # never even run — the claim is checked against the store's own
            # result first.
            path.readback.assert_not_called()
            _assert_cleanup_not_fired(spy)

        # (b) intact.
        assert scheduler._shadow_job is job
        # (c) the counter stands where the last *real* publish left it.
        assert job.committed_tokens == 512
        assert job.published_boundaries == [512]
        assert scheduler._shadow_counters.publishes == 0
        # (e)
        assert scheduler.has_requests() is True


class TestRestorableInvariantUnderFault:
    """canonical_committed_tokens <= independently_restorable_tokens.

    This is the correctness invariant the whole experiment turns on. A store
    that reports success is not evidence of canonical publication: the
    committed count is a claim that a later turn can restore that prefix, so
    it may only advance once the ordinary matching path, on the serving cache,
    can actually see the boundary.

    The failure this guards is silent. A false commit does not raise, does not
    log an error and does not lose a request — it reports a canonical prefix
    that is not there, which is the one failure mode a measurement column must
    not have.
    """

    def test_a_short_read_back_does_not_advance_the_committed_count(self):
        """Fault 9. Intact. Store succeeded; the serving path sees less.

        768 tokens stored, 512 restorable: the boundary is only partly
        visible, and a partly visible boundary is not one.
        """
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        state = _live_state(job, processed=768)

        with _cleanup_spy(scheduler) as spy, _publish_path(
            scheduler,
            worker=SimpleNamespace(block_ids=[1, 2, 3]),  # store claims 768
            readback=512,                                 # serving path sees 512
        ) as path:
            # (a)
            assert scheduler._shadow_publish(job, 768, state) is None
            path.worker.assert_called_once()
            path.readback.assert_called_once()
            _assert_cleanup_not_fired(spy)

        # (b) intact: the store is not evidence of a bad job, only of a
        #     boundary that is not yet reachable.
        assert scheduler._shadow_job is job
        # (c) the invariant holds: committed (0) <= restorable (512).
        assert job.committed_tokens == 0
        assert job.published_boundaries == []
        assert scheduler._shadow_counters.publishes == 0
        # (e)
        assert scheduler.has_requests() is True

    def test_a_read_back_that_covers_the_boundary_does_advance_it(self):
        """The control. Without it, the test above is satisfied by a publish
        path that never advances the counter at all."""
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        state = _live_state(job, processed=768)

        with _publish_path(
            scheduler,
            worker=SimpleNamespace(block_ids=[1, 2, 3]),
            readback=768,
        ):
            scheduler._shadow_publish(job, 768, state)

        assert job.committed_tokens == 768
        assert scheduler._shadow_counters.publishes == 1

    def test_a_read_back_probe_that_raises_reports_zero_and_still_cleans_up(self):
        """Fault 10. Intact. `fetch_cache` raised inside the probe.

        The probe runs the real lookup under a throwaway request id, so it
        registers a block table of its own. A probe that raised and left that
        entry behind would leak a block reference on every failed publish.
        """
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        scheduler.block_aware_cache.fetch_cache.side_effect = RuntimeError(
            "fetch failed"
        )

        # (a) the probe answers rather than raising, and (c) its answer is the
        #     conservative one.
        assert scheduler._shadow_readback_tokens(job, list(range(BLOCK))) == 0
        # (d) the probe's own footprint went back anyway.
        scheduler.block_aware_cache.release_cache.assert_any_call(PROBE_ID)
        scheduler.block_aware_cache.clear_request_entry.assert_any_call(PROBE_ID)
        # (b)/(e) the probe does not touch the job.
        assert scheduler._shadow_job is job
        assert scheduler.has_requests() is True

    def test_a_raising_read_back_probe_blocks_the_commit(self):
        """The probe's zero has to reach the counter, not only be returned.

        Asserted through the whole publish rather than against the probe
        alone, because "returns 0" is worth nothing if the caller then commits
        anyway.
        """
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        state = _live_state(job, processed=768)
        scheduler.block_aware_cache.fetch_cache.side_effect = RuntimeError(
            "fetch failed"
        )

        with _cleanup_spy(scheduler) as spy, _publish_path(
            scheduler,
            worker=SimpleNamespace(block_ids=[1, 2, 3]),
            readback=None,          # the real probe, against the raising cache
        ):
            assert scheduler._shadow_publish(job, 768, state) is None
            _assert_cleanup_not_fired(spy)

        assert scheduler._shadow_job is job
        assert job.committed_tokens == 0
        assert scheduler._shadow_counters.publishes == 0
        assert scheduler.has_requests() is True


# --------------------------------------------------------------------------
# 11-12: the ground moves under a live job
# --------------------------------------------------------------------------


class TestServingCacheChangesUnderTheJob:
    """State published into an instance that did not serve the request is
    valid, durable and unreachable, so the publish fails closed."""

    def test_a_changed_serving_cache_refuses_and_drops(self):
        """Fault 11. Dropped. The job is bound to one prefix-cache instance.

        A restore on the serving path never looks in another instance, so
        publishing here would write state that is correct, counted and
        invisible — which is worse than not publishing, because the counter
        would then report canonical coverage that no request can reach.
        """
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        state = _live_state(job, processed=BLOCK)
        scheduler.block_aware_cache = MagicMock()      # a different instance

        with _cleanup_spy(scheduler) as spy, _publish_path(scheduler) as path:
            # (a)
            assert scheduler._shadow_publish(job, BLOCK, state) is None
            path.worker.assert_not_called()
            # (d)
            _assert_cleanup_fired(spy)

        # (b) dropped, not retried: the binding cannot come back.
        assert scheduler._shadow_job is None
        assert job.cancelled is True
        # (c)
        assert job.committed_tokens == 0
        assert scheduler._shadow_counters.publishes == 0
        # (e)
        assert scheduler.has_requests() is False

    def test_the_drop_releases_against_the_bound_cache_as_well(self):
        """The blocks live in the instance that served the job, not the new one.

        `_release_paged_cache_for_request` releases against
        `self.block_aware_cache`, the instance that is current now — which on
        this path is by construction not the one holding this job's block
        references. Without the second release the drop left them pinned in a
        cache nothing would ever ask again. The job keeps a weak reference so
        that a replaced cache which is genuinely being torn down is not kept
        alive by the recovery job holding it.
        """
        scheduler = _make_scheduler()
        bound_cache = scheduler.block_aware_cache
        job = _queued(scheduler)
        assert job.serving_cache_id == id(bound_cache)
        state = _live_state(job, processed=BLOCK)

        new_cache = MagicMock()
        scheduler.block_aware_cache = new_cache
        with _publish_path(scheduler):
            scheduler._shadow_publish(job, BLOCK, state)

        bound_cache.release_cache.assert_any_call(RID)
        assert scheduler._shadow_job is None


class TestModelBecomesUnreconstructible:
    def test_an_unreconstructible_model_refuses_and_drops(self):
        """Fault 12. Dropped. The gate is re-read at publish time.

        A job queued while the model's cache was reconstructible must not
        publish once that answer has changed: the stored blocks would restore
        as placeholders, and a later restore walks them back to nothing or
        rejects the chain outright.
        """
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        job.note_published(512)
        state = _live_state(job, processed=768)

        with _cleanup_spy(scheduler) as spy, patch.object(
            scheduler, "_model_has_unreconstructible_cache", return_value=True
        ), _publish_path(scheduler) as path:
            # (a)
            assert scheduler._shadow_publish(job, 768, state) is None
            path.worker.assert_not_called()
            # (d)
            _assert_cleanup_fired(spy)

        # (b) dropped.
        assert scheduler._shadow_job is None
        assert job.cancelled is True
        # (c) what was already published stays published and stays counted on
        #     the job object; the refused boundary is not added.
        assert job.committed_tokens == 512
        assert job.published_boundaries == [512]
        assert scheduler._shadow_counters.publishes == 0
        # (e)
        assert scheduler.has_requests() is False


# --------------------------------------------------------------------------
# 13: cancellation
# --------------------------------------------------------------------------


class TestCancellationHalfway:
    """Background work must never be the reason a model cannot be unloaded.

    The unload path drains on `has_requests()`, which reports a live shadow
    job as work. Without a cancel that really clears it, the unload is queued
    "until active scheduler work drains", it never drains, and every later
    request to that model is refused with 409.
    """

    def test_cancelling_a_job_holding_prefill_state_cleans_up_fully(self):
        """Fault 13. Dropped, mid-chunk, with live state attached."""
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        job.note_published(512)
        job.prefill_state = _live_state(job, processed=600)
        assert scheduler.has_requests() is True

        with _cleanup_spy(scheduler) as spy:
            # (a)
            assert scheduler.cancel_shadow_work("unload") is True
            # (d)
            _assert_cleanup_fired(spy)

        # (b) dropped, and the state reference is let go.
        assert scheduler._shadow_job is None
        assert job.cancelled is True
        assert job.prefill_state is None
        # (c) a cancel does not unpublish: published blocks are ordinary
        #     canonical state and the next request may restore from them.
        assert job.committed_tokens == 512
        assert job.published_boundaries == [512]
        # (e) the engine can now be unloaded.
        assert scheduler.has_requests() is False

    def test_cancelling_again_reports_no_work_and_does_not_raise(self):
        """Idempotent: the unload path may call it more than once, and a
        second call must not raise or re-release anything."""
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        job.prefill_state = _live_state(job, processed=600)
        assert scheduler.cancel_shadow_work("unload") is True

        with _cleanup_spy(scheduler) as spy:
            assert scheduler.cancel_shadow_work("unload") is False
            _assert_cleanup_not_fired(spy)
        assert scheduler.has_requests() is False

    def test_a_cancel_during_a_chunk_stops_the_next_step_cleanly(self):
        """The step after a cancel finds no job and says so, rather than
        running against the state the cancel just retired."""
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        job.prefill_state = _live_state(job, processed=600)
        scheduler.cancel_shadow_work("unload")

        with patch.object(scheduler, "_step_prefill_chunk") as chunk:
            assert scheduler._shadow_step() is False
        chunk.assert_not_called()
        assert scheduler.has_requests() is False


# --------------------------------------------------------------------------
# The foreground
# --------------------------------------------------------------------------


class TestTheForegroundIsUnaffected:
    """The point of all of the above: a recovery failure costs reuse only.

    `step()` is driven for real here rather than `_shadow_step()`, because the
    claim is about the step the engine loop calls, not about the shadow method
    it calls in turn.
    """

    def test_a_step_whose_shadow_drops_returns_normally(self):
        """The shadow becomes runnable on the second idle step, fails to build
        its state, and drops. The step must not notice."""
        scheduler = _make_scheduler()
        _queued(scheduler)

        with patch.object(
            scheduler,
            "_shadow_begin_state",
            side_effect=RuntimeError("state build failed"),
        ):
            first = scheduler.step()
            second = scheduler.step()      # the shadow runs and drops in this one

        # (a) the step path returned, twice.
        assert first.has_work is False
        assert second.has_work is False
        # No request was failed, finished or rejected by the shadow's failure.
        for output in (first, second):
            assert output.outputs == []
            assert output.finished_request_ids == set()
            assert output.scheduled_request_ids == []
            assert output.prefill_eviction_request is None
        # (b)/(e) the job is gone and the loop is free to park.
        assert scheduler._shadow_job is None
        assert scheduler.has_requests() is False

    def test_a_step_whose_shadow_chunk_raises_returns_normally(self):
        """The same claim for the other drop path, where the fault happens
        inside the model call rather than before it."""
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        job.prefill_state = _live_state(job, processed=100)

        with patch.object(
            scheduler, "_step_prefill_chunk", side_effect=RuntimeError("chunk blew up")
        ):
            scheduler.step()
            output = scheduler.step()

        assert output.outputs == []
        assert output.finished_request_ids == set()
        assert output.prefill_eviction_request is None
        assert scheduler._shadow_job is None
        assert scheduler.has_requests() is False


class TestAShadowFailureCannotFailTheBatch:
    """The shadow block runs after `step()`'s own try/except.

    Anything escaping it leaves `step()` altogether, and the engine loop
    answers an escaped exception by calling `fail_all_requests()` — every live
    request in the batch errored, by background work whose worst permitted
    outcome is losing its own progress. Several calls in that window are
    unguarded; `_specprefill_rope_installed` is the clearest, because its
    `try` covers the import and `_find_attention_layers` but not the loop that
    reads each layer's `.rope`.
    """

    def test_a_raising_idle_judgement_does_not_escape_the_step(self):
        scheduler = _make_scheduler()
        _queued(scheduler)
        with patch.object(
            scheduler, "_shadow_note_step", side_effect=RuntimeError("boom")
        ):
            output = scheduler.step()
        assert output is not None
        assert scheduler._shadow_job is None
        assert not scheduler.has_requests()

    def test_a_raising_runnable_predicate_does_not_escape_the_step(self):
        """The RoPE inspection lives behind this one."""
        scheduler = _make_scheduler()
        _queued(scheduler)
        with patch.object(
            scheduler, "_shadow_runnable", side_effect=RuntimeError("no rope")
        ):
            output = scheduler.step()
        assert output is not None
        assert scheduler._shadow_job is None

    def test_a_raising_chunk_path_does_not_escape_the_step(self):
        scheduler = _make_scheduler()
        _queued(scheduler)
        scheduler._shadow_note_step(did_foreground_work=False)
        scheduler._shadow_note_step(did_foreground_work=False)
        with patch.object(
            scheduler, "_shadow_step", side_effect=RuntimeError("boom")
        ):
            output = scheduler.step()
        assert output is not None
        assert scheduler._shadow_job is None

    def test_the_foreground_outputs_are_untouched_by_the_failure(self):
        scheduler = _make_scheduler()
        _queued(scheduler)
        with patch.object(
            scheduler, "_shadow_note_step", side_effect=RuntimeError("boom")
        ):
            output = scheduler.step()
        assert output.outputs == []
        assert output.finished_request_ids == set()
        assert output.prefill_eviction_request is None


class TestRetiringStateMidChunkGivesTheFootprintBack:
    """An extension that lands during a chunk retires the state, and the state
    is holding a request entry and its paged blocks.

    The publish immediately above kept them on purpose —
    `retain_request_entry=not job.done`, and a job about to be rebuilt is not
    done. The next window rebuilds under the same request id, which overwrites
    the block table and orphans the previous references: never decremented,
    never evictable, filling the paged cache until the memory throttle starts
    refusing the recovery chunks outright. `_shadow_finish`'s own comment names
    that consequence for the path it guards; this is the path that did not.
    """

    def _extended_mid_chunk(self, scheduler):
        job = _queued(scheduler)
        state = _live_state(job, processed=BLOCK)
        state.shadow_target_tokens = job.target_tokens
        job.prefill_state = state
        # The turn that lands while the chunk is in flight.
        job.target_tokens += BLOCK
        job.tokens = job.tokens + list(range(9_000_000, 9_000_000 + BLOCK))
        return job, state

    def test_the_request_entry_and_blocks_go_back(self):
        scheduler = _make_scheduler()
        job, _state = self._extended_mid_chunk(scheduler)
        with _cleanup_spy(scheduler) as spy, patch.object(
            scheduler, "_step_prefill_chunk", return_value=False
        ), patch.object(scheduler, "_shadow_publish"):
            assert scheduler._shadow_step_inner() is True
            _assert_cleanup_fired(spy)
        assert job.prefill_state is None

    def test_the_job_itself_survives_and_stays_runnable(self):
        """Only the work is retired. The job keeps its committed prefix and
        its lineage so the next window extends rather than starts over."""
        scheduler = _make_scheduler()
        job, _state = self._extended_mid_chunk(scheduler)
        job.note_published(BLOCK)
        with patch.object(
            scheduler, "_step_prefill_chunk", return_value=False
        ), patch.object(scheduler, "_shadow_publish"):
            scheduler._shadow_step_inner()
        assert scheduler._shadow_job is job
        assert not job.cancelled
        assert not job.done
        assert job.committed_tokens == BLOCK


# --------------------------------------------------------------------------
# The recovery request is not a user request
# --------------------------------------------------------------------------


class TestTheRecoveryRequestIsInvisibleToTheRequestSweeps:
    """Three existing sweeps walk ``self.requests`` and reach the job's own
    synthetic request, which sits there in no queue at all.

    That is precisely the shape two of them exist to find: a request popped
    off ``waiting`` and being prefilled right now is reachable through no
    queue either, and missing it hung clients (#2372). The recovery request
    has the same shape and none of the meaning — no collector, no client, no
    output — so failing it names an id nobody sent, and re-prefilling it
    schedules background work as foreground with nothing to emit to.

    Recovery is allowed to lose its work. It is not allowed to become a
    request, and it is not allowed to be the reason an engine cannot quiesce
    after an unrecoverable error.
    """

    @staticmethod
    def _with_both(scheduler):
        """One foreground request and one recovery request, both queue-less.

        The foreground one is the control: every assertion about the recovery
        request has a matching one saying the sweep still does its job.
        """
        foreground = Request(
            request_id="user-1",
            prompt=None,
            prompt_token_ids=[1, 2, 3, 4],
            sampling_params=SamplingParams(max_tokens=8),
        )
        recovery = Request(
            request_id=RID,
            prompt=None,
            prompt_token_ids=list(range(4 * BLOCK)),
            sampling_params=SamplingParams(max_tokens=1),
        )
        recovery.is_shadow = True
        scheduler.requests[foreground.request_id] = foreground
        scheduler.requests[recovery.request_id] = recovery
        return foreground, recovery

    def test_a_fatal_error_does_not_fail_it_as_a_user_request(self):
        scheduler = _make_scheduler()
        foreground, _recovery = self._with_both(scheduler)
        failed = scheduler.fail_all_requests()
        assert foreground.request_id in failed
        assert RID not in failed

    def test_a_fatal_error_ends_the_job_so_the_engine_can_quiesce(self):
        """`has_requests()` reports a live recovery job, and the unload path
        drains on that predicate. A job that survives the failure that killed
        every request keeps the engine awake and unloadable forever."""
        scheduler = _make_scheduler()
        scheduler.note_shadow_candidate(_sparse_request(4 * BLOCK, scheduler=scheduler))
        assert scheduler._shadow_job is not None
        assert scheduler.has_requests()
        scheduler.fail_all_requests()
        assert scheduler._shadow_job is None
        assert not scheduler.has_requests()

    def test_cache_corruption_recovery_does_not_resurrect_it(self):
        scheduler = _make_scheduler()
        foreground, _recovery = self._with_both(scheduler)
        collected = scheduler._collect_corruption_retry_requests()
        assert foreground in collected
        assert all(not request.is_shadow for request in collected)

    def test_generation_overflow_rescheduling_does_not_resurrect_it(self):
        scheduler = _make_scheduler()
        foreground, recovery = self._with_both(scheduler)
        scheduler._reschedule_generation_overflow_requests()
        assert recovery not in scheduler.waiting
        assert recovery.request_id in scheduler.requests
        assert foreground in scheduler.waiting
