# SPDX-License-Identifier: Apache-2.0
"""``committed_tokens`` is a claim about the cache, not a record of work done.

The invariant

    canonical_committed_tokens <= independently_restorable_tokens

is verified at publication, by storing and then reading the boundary back
through the ordinary serving path. That is the right check at the right moment,
and it is not enough on its own, because the two sides of it move
independently afterwards:

* ``committed_tokens`` only ever rises;
* the blocks backing an already-published prefix can be evicted, and under
  ``hot_cache_only`` an evicted block is dropped rather than demoted to SSD, so
  a hole can open low in a chain that was verified when it was written.

``_canonical_recovery_publish`` re-probes whenever a *higher* boundary comes
along and drops the job if the old ground has gone. The gap is a session that
never reaches a higher boundary: publish 12,288, lose blocks, then take turns
that stay inside the same block. The job sits parked with a claim the cache
cannot honour, and — because ``publishable_boundary`` refuses anything at or
below the watermark — if it ever resumes it declines to republish exactly the
range that went missing.

So the claim is re-checked where the answer is already in hand. A job resuming
has just asked the serving cache what it can restore for this prompt; that
number is the authority, and the watermark is walked back to meet it. The cache
is not asked a second time and no reference is taken that would have to be
given back.

These tests use a real ``CanonicalRecoveryJob`` and a stub prefix-cache
preparation, because what is under test is the arithmetic of the claim, not the
cache's own eviction policy.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omlx.canonical_recovery import CanonicalRecoveryJob
from omlx.scheduler import Scheduler, SchedulerConfig

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


def _job(committed: int, target: int = 8 * BLOCK) -> CanonicalRecoveryJob:
    job = CanonicalRecoveryJob(
        session_key="s",
        tokens=list(range(target)),
        target_tokens=target,
        block_size=BLOCK,
    )
    job.note_published(committed)
    return job


def _restoring(scheduler, cached_tokens: int):
    """Stub the prefix-cache preparation to report *cached_tokens* restorable."""

    def prepare(request):
        request.cached_tokens = cached_tokens
        request.remaining_tokens = list(range(cached_tokens, len(request.prompt_token_ids)))

    return patch.object(
        scheduler, "_prepare_prefix_cache_for_request", side_effect=prepare
    )


class TestTheWatermarkFollowsTheCache:
    def test_ground_that_is_gone_walks_the_watermark_back(self):
        scheduler = _make_scheduler()
        job = _job(committed=4 * BLOCK)
        request = SimpleNamespace(cached_tokens=2 * BLOCK)

        scheduler._canonical_recovery_revalidate_ground(job, request)

        assert job.committed_tokens == 2 * BLOCK
        assert job.published_boundaries == []

    def test_ground_that_is_intact_is_left_alone(self):
        scheduler = _make_scheduler()
        job = _job(committed=4 * BLOCK)
        request = SimpleNamespace(cached_tokens=4 * BLOCK)

        scheduler._canonical_recovery_revalidate_ground(job, request)

        assert job.committed_tokens == 4 * BLOCK
        assert job.published_boundaries == [4 * BLOCK]

    def test_more_ground_than_claimed_does_not_raise_the_watermark(self):
        """This path only ever lowers. Raising is a publication's job, and a
        publication is the only thing that verified a boundary was written."""
        scheduler = _make_scheduler()
        job = _job(committed=4 * BLOCK)
        request = SimpleNamespace(cached_tokens=6 * BLOCK)

        scheduler._canonical_recovery_revalidate_ground(job, request)

        assert job.committed_tokens == 4 * BLOCK

    def test_a_partial_block_floors_to_the_block_below(self):
        """Only whole blocks are restorable canonical state."""
        scheduler = _make_scheduler()
        job = _job(committed=4 * BLOCK)
        request = SimpleNamespace(cached_tokens=2 * BLOCK + 17)

        scheduler._canonical_recovery_revalidate_ground(job, request)

        assert job.committed_tokens == 2 * BLOCK

    def test_no_cache_means_unknown_not_gone(self):
        """A zero from a model whose cache cannot be reconstructed is the
        absence of an answer. Walking the watermark to zero on it would
        forfeit a verified prefix on the strength of a question nobody asked."""
        scheduler = _make_scheduler()
        scheduler.block_aware_cache = None
        job = _job(committed=4 * BLOCK)

        scheduler._canonical_recovery_revalidate_ground(
            job, SimpleNamespace(cached_tokens=0)
        )
        assert job.committed_tokens == 4 * BLOCK

        scheduler.block_aware_cache = MagicMock()
        with patch.object(
            scheduler, "_model_has_unreconstructible_cache", return_value=True
        ):
            scheduler._canonical_recovery_revalidate_ground(
                job, SimpleNamespace(cached_tokens=0)
            )
        assert job.committed_tokens == 4 * BLOCK


class TestTheScenarioEndToEnd:
    """Publish, lose the ground, take a turn inside the block, then resume."""

    def _parked_job_with_lost_ground(self, scheduler):
        job = _job(committed=4 * BLOCK)
        job.note_reached_target()
        scheduler._canonical_recovery_job = job
        return job

    def test_a_turn_inside_the_same_block_keeps_the_job_and_publishes_nothing(self):
        """Step 3: nothing crosses a boundary, so no publish-time probe runs.

        The old ground is still claimed at this point, and that is inert: a
        parked job publishes nothing, decides nothing, and is read by nothing
        outside itself. What must not happen is the job being destroyed, which
        would forfeit the lineage a later turn extends.
        """
        scheduler = _make_scheduler()
        job = self._parked_job_with_lost_ground(scheduler)

        request = MagicMock()
        request.request_id = "s"
        request.prompt_token_ids = list(range(8 * BLOCK + 10))
        request.specprefill_indices = [1, 2, 3]
        request._serving_prefix_cache_id = id(scheduler.block_aware_cache)
        scheduler.note_canonical_recovery_candidate(request)

        assert scheduler._canonical_recovery_job is job
        assert job.committed_tokens == 4 * BLOCK

    def test_resuming_the_job_corrects_the_claim_before_it_reads_anything(self):
        """Step 4: the claim does not survive contact with the cache again."""
        scheduler = _make_scheduler()
        job = self._parked_job_with_lost_ground(scheduler)
        job.reached_target = False  # a later turn extended it

        with _restoring(scheduler, cached_tokens=2 * BLOCK), patch.object(
            scheduler, "_begin_prefill", return_value=MagicMock()
        ):
            scheduler._canonical_recovery_begin_state(job)

        assert job.committed_tokens == 2 * BLOCK

    def test_the_corrected_job_will_republish_the_range_it_lost(self):
        """The point of walking back rather than only refusing to advance.

        ``publishable_boundary`` refuses anything at or below the watermark, so
        a stale-high claim makes the job decline to republish exactly the range
        that went missing.
        """
        job = _job(committed=4 * BLOCK)
        job.processed_tokens = 3 * BLOCK
        assert job.publishable_boundary() == 0  # refused while the claim stands

        job.note_ground_lost(2 * BLOCK)
        assert job.publishable_boundary() == 3 * BLOCK
