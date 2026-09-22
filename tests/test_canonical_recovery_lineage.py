# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for canonical-recovery *lineage* handling.

These tests do not say what the recovery job ought to do. They say what it
does, for the three timelines a real session actually produces:

**B2 — rapid turns.** Four turns arrive before any chunk runs. What the code
does with them is the difference between one job that re-reads the newest
prompt and four jobs that each re-read the same prefix.

**B3 — a shared-prefix fork.** Two branches off one common prefix. A token
list is the cache key, so the question is not whether a branch is allowed to
reuse the other's blocks but whether either job can ever hold a token list
that is a mixture of the two.

**B4 — compaction.** A turn whose prompt shares only its opening with the
previous one. The old lineage is gone and its in-flight work is worthless; the
committed count that described it must not be carried onto its replacement.

Where the implementation does not satisfy the invariant the scenario was
written for, the test asserts the behaviour that is there and the docstring
says so under "Gap:". Nothing in ``omlx/`` was changed to make one pass.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omlx.scheduler import Scheduler, SchedulerConfig
from omlx.canonical_recovery import CanonicalRecoveryJob

BLOCK = 4096


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


def _sparse_request_on(tokens: list[int], rid: str, scheduler):
    """The same request, over a token list the test wrote out itself.

    Every lineage question here is a question about token lists, so these are
    built explicitly rather than by a length: a test that asked for "26,000
    tokens" would be asserting against whatever `_sparse_request` happened to
    generate, not against a prefix-extension it can point at.
    """
    request = _sparse_request(0, rid=rid, scheduler=scheduler)
    request.prompt_token_ids = list(tokens)
    return request


def _target_for(prompt_len: int) -> int:
    """What `note_canonical_recovery_candidate` will set `target_tokens` to.

    The last whole block, exactly. Only whole blocks are publishable, and the
    recovery state prefills every token it is given, so there is nothing to
    compensate for.
    """
    return (prompt_len // BLOCK) * BLOCK


@contextmanager
def _lineage_spies(scheduler):
    """Count job construction, growth and replacement without changing any.

    `extend` and `_canonical_recovery_drop_job` keep doing exactly what they did; the
    wrappers only tally. `omlx.scheduler.CanonicalRecoveryJob` is rebound rather than
    mocked out, so every job under test is a real `CanonicalRecoveryJob`.
    """
    counts = SimpleNamespace(
        jobs_created=0,
        target_extensions=0,
        extend_refusals=0,
        no_op_candidates=0,
        superseded_jobs=0,
        cancelled_jobs=0,
        drop_reasons=[],
    )
    real_extend = CanonicalRecoveryJob.extend
    real_drop = scheduler._canonical_recovery_drop_job

    def counting_extend(job, tokens):
        grew = real_extend(job, tokens)
        if grew:
            counts.target_extensions += 1
        else:
            counts.extend_refusals += 1
        return grew

    def counting_drop(reason):
        job = scheduler._canonical_recovery_job
        counts.drop_reasons.append(reason)
        if reason == "replaced":
            counts.superseded_jobs += 1
        real_drop(reason)
        if job is not None and job.cancelled:
            counts.cancelled_jobs += 1

    def counting_ctor(*args, **kwargs):
        counts.jobs_created += 1
        return CanonicalRecoveryJob(*args, **kwargs)

    with patch.object(CanonicalRecoveryJob, "extend", counting_extend), patch.object(
        scheduler, "_canonical_recovery_drop_job", counting_drop
    ), patch("omlx.scheduler.CanonicalRecoveryJob", counting_ctor):
        yield counts


def _publish(scheduler, job, boundary, state_tokens, block_ids, readback=None):
    """Run `_canonical_recovery_publish` with everything below the decision stubbed out.

    Returns the recorded call to the store worker, which is where the token
    list a boundary is keyed by is actually decided.
    """
    state = MagicMock(
        base_size=0, tokens_processed=state_tokens, cache=[MagicMock()]
    )
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
        "_canonical_recovery_readback_tokens",
        side_effect=lambda _job, tokens: (
            readback if readback is not None else len(tokens)
        ),
    ), patch.object(
        scheduler,
        "_async_store_cache_worker",
        return_value=SimpleNamespace(block_ids=list(block_ids)),
    ) as worker:
        scheduler._canonical_recovery_publish(job, boundary, state)
    return worker


class TestB2RapidTurnsCoalesce:
    """Four turns, no chunk in between.

    Measured: jobs_created=1, target_extensions=3, superseded_jobs=0,
    cancelled_jobs=0, publishes=0.
    """

    PROMPTS = [20000, 26000, 31000, 38000]

    def test_four_growing_turns_are_one_job_created_once_and_extended_three_times(self):
        """jobs_created=1, target_extensions=3, superseded_jobs=0, cancelled_jobs=0."""
        scheduler = _make_scheduler()
        identities = []
        with _lineage_spies(scheduler) as counts:
            for turn, prompt_len in enumerate(self.PROMPTS):
                scheduler.note_canonical_recovery_candidate(
                    _sparse_request_on(
                        list(range(prompt_len)), rid=f"r{turn}", scheduler=scheduler
                    )
                )
                assert scheduler._canonical_recovery_job is not None
                identities.append(id(scheduler._canonical_recovery_job))

        # One job object, never replaced, for the whole burst.
        assert len(set(identities)) == 1
        assert counts.jobs_created == 1
        assert counts.target_extensions == 3
        assert counts.extend_refusals == 0
        assert counts.superseded_jobs == 0
        assert counts.cancelled_jobs == 0
        assert counts.drop_reasons == []
        assert scheduler._canonical_recovery_counters.publishes == 0

    def test_the_one_job_tracks_the_newest_boundary(self):
        scheduler = _make_scheduler()
        first = None
        for turn, prompt_len in enumerate(self.PROMPTS):
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(
                    list(range(prompt_len)), rid=f"r{turn}", scheduler=scheduler
                )
            )
            job = scheduler._canonical_recovery_job
            if first is None:
                first = job
            assert job is first
            assert job.target_tokens == _target_for(prompt_len)
            assert job.tokens == list(range(_target_for(prompt_len)))

        # 38,000 tokens over 4,096-token blocks: nine whole blocks, and the
        # 1,616-token remainder is not publishable state.
        assert first.target_tokens == 9 * BLOCK == 36864

    def test_the_committed_prefix_survives_every_extension(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(list(range(20000)), rid="r0", scheduler=scheduler)
        )
        job = scheduler._canonical_recovery_job
        job.note_published(4 * BLOCK)
        for turn, prompt_len in enumerate(self.PROMPTS[1:], start=1):
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(
                    list(range(prompt_len)), rid=f"r{turn}", scheduler=scheduler
                )
            )
            assert scheduler._canonical_recovery_job is job
            assert job.committed_tokens == 4 * BLOCK
        assert job.published_boundaries == [4 * BLOCK]

    def test_an_append_inside_the_same_block_neither_replaces_nor_extends(self):
        """38,000 -> 38,100 adds no whole block: jobs_created=0, extensions=0.

        `extend` is never even reached — the equal-target guard in
        `note_canonical_recovery_candidate` returns first — so `extend_refusals` stays 0
        too. That guard is the whole point: reading `extend`'s refusal as "not
        an append" is what used to destroy the job on a short turn.
        """
        scheduler = _make_scheduler()
        for turn, prompt_len in enumerate(self.PROMPTS):
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(
                    list(range(prompt_len)), rid=f"r{turn}", scheduler=scheduler
                )
            )
        job = scheduler._canonical_recovery_job
        job.note_published(9 * BLOCK)

        with _lineage_spies(scheduler) as counts:
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(list(range(38100)), rid="r4", scheduler=scheduler)
            )

        assert scheduler._canonical_recovery_job is job
        assert counts.jobs_created == 0
        assert counts.target_extensions == 0
        assert counts.extend_refusals == 0
        assert counts.superseded_jobs == 0
        assert counts.cancelled_jobs == 0
        assert job.target_tokens == 9 * BLOCK
        assert job.committed_tokens == 9 * BLOCK

    def test_a_longer_turn_after_the_no_op_still_extends_the_same_job(self):
        """The no-op must leave the append path usable, not just the object."""
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(list(range(38000)), rid="r0", scheduler=scheduler)
        )
        job = scheduler._canonical_recovery_job
        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(list(range(38100)), rid="r1", scheduler=scheduler)
        )
        with _lineage_spies(scheduler) as counts:
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(list(range(43000)), rid="r2", scheduler=scheduler)
            )
        assert scheduler._canonical_recovery_job is job
        assert counts.target_extensions == 1
        assert counts.jobs_created == 0
        assert job.target_tokens == _target_for(43000) == 10 * BLOCK


def _common_prefix(blocks: int = 8) -> list[int]:
    return list(range(blocks * BLOCK))


def _branch(common: list[int], marker: int, length: int = BLOCK) -> list[int]:
    """A branch off *common* whose own tokens share no value with the other's."""
    return common + [marker + i for i in range(length)]


class TestB3SharedPrefixFork:
    """Two branches off a 32,768-token common prefix.

    Measured over feed(A), feed(B), feed(A): jobs_created=3,
    target_extensions=0, superseded_jobs=2, cancelled_jobs=2, publishes=0.
    """

    def _branches(self):
        common = _common_prefix(8)
        assert len(common) == 32768
        branch_a = _branch(common, 1_000_000)
        branch_b = _branch(common, 2_000_000)
        assert branch_a[:32768] == branch_b[:32768]
        assert set(branch_a[32768:]).isdisjoint(branch_b[32768:])
        return common, branch_a, branch_b

    def test_a_fork_replaces_the_job_it_does_not_extend_it(self):
        """jobs_created=3, target_extensions=0, superseded_jobs=2, cancelled_jobs=2.

        `extend` requires a strict prefix-extension and B is the same length as
        A, so it is refused twice over — on length and on content — and the
        caller replaces. The shared 32,768 tokens buy the second branch
        nothing at the job level; whatever they buy it is bought later, in the
        prefix cache, by the token list being the key.
        """
        _common, branch_a, branch_b = self._branches()
        scheduler = _make_scheduler()
        with _lineage_spies(scheduler) as counts:
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(branch_a, rid="a1", scheduler=scheduler)
            )
            job_a = scheduler._canonical_recovery_job
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(branch_b, rid="b1", scheduler=scheduler)
            )
            job_b = scheduler._canonical_recovery_job
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(branch_a, rid="a2", scheduler=scheduler)
            )
            job_a2 = scheduler._canonical_recovery_job

        assert job_b is not job_a
        assert job_a2 is not job_b
        assert job_a2 is not job_a
        assert counts.jobs_created == 3
        assert counts.target_extensions == 0
        assert counts.extend_refusals == 2
        assert counts.superseded_jobs == 2
        assert counts.cancelled_jobs == 2
        assert counts.drop_reasons == ["replaced", "replaced"]
        assert job_a.cancelled is True
        assert job_b.cancelled is True
        assert job_a2.cancelled is False
        assert scheduler._canonical_recovery_counters.publishes == 0

    def test_the_drop_that_replaces_a_job_is_the_replaced_reason(self):
        _common, branch_a, branch_b = self._branches()
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(branch_a, rid="a1", scheduler=scheduler)
        )
        with patch.object(
            scheduler, "_canonical_recovery_drop_job", wraps=scheduler._canonical_recovery_drop_job
        ) as drop:
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(branch_b, rid="b1", scheduler=scheduler)
            )
        drop.assert_called_once_with("replaced")

    def test_a_longer_fork_is_refused_on_content_rather_than_on_length(self):
        """The two refusals inside `extend` are different, and both are load-bearing.

        Equal-length branches are refused on `len(tokens) <= target_tokens`
        alone and never reach the prefix comparison. A branch that is also
        longer does reach it, and is refused there — which is the check that
        keeps a job from adopting a token list whose middle it never computed.
        """
        common = _common_prefix(8)
        branch_a = _branch(common, 1_000_000, length=BLOCK)
        longer_fork = _branch(common, 2_000_000, length=2 * BLOCK)
        assert len(longer_fork) > len(branch_a)

        scheduler = _make_scheduler()
        with _lineage_spies(scheduler) as counts:
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(branch_a, rid="a1", scheduler=scheduler)
            )
            job_a = scheduler._canonical_recovery_job
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(longer_fork, rid="b1", scheduler=scheduler)
            )

        assert counts.extend_refusals == 1          # `extend` was reached
        assert counts.target_extensions == 0        # and said no
        assert counts.superseded_jobs == 1
        assert scheduler._canonical_recovery_job is not job_a
        assert scheduler._canonical_recovery_job.tokens[: len(common)] == common
        assert set(scheduler._canonical_recovery_job.tokens).isdisjoint(branch_a[len(common):])

    def test_a_job_never_holds_a_mixture_of_two_branches(self):
        """The token list a job would publish is its own lineage, entire."""
        _common, branch_a, branch_b = self._branches()
        scheduler = _make_scheduler()

        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(branch_a, rid="a1", scheduler=scheduler)
        )
        job_a = scheduler._canonical_recovery_job
        assert job_a.tokens == branch_a[: _target_for(len(branch_a))]

        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(branch_b, rid="b1", scheduler=scheduler)
        )
        job_b = scheduler._canonical_recovery_job
        assert job_b.tokens == branch_b[: _target_for(len(branch_b))]
        # Not a splice of the two: no token of A's branch survives into B's job.
        assert set(job_b.tokens).isdisjoint(branch_a[32768:])
        assert job_a.tokens == branch_a[: _target_for(len(branch_a))]

        for boundary in range(BLOCK, job_b.target_tokens + 1, BLOCK):
            assert job_b.tokens[:boundary] == branch_b[:boundary]

    def test_a_replaced_job_keeps_what_it_published(self):
        """Gap-free by design: the drop path has no unpublish to call.

        Published blocks are ordinary canonical state keyed by their own token
        list, and the next request is entitled to restore from them. There is
        no unpublish call to assert the absence of, so this asserts that the
        drop touches the prefix cache not at all, and that the boundary the
        job recorded survives on the dropped object.
        """
        _common, branch_a, branch_b = self._branches()
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(branch_a, rid="a1", scheduler=scheduler)
        )
        job_a = scheduler._canonical_recovery_job
        job_a.note_published(8 * BLOCK)

        before = list(scheduler.block_aware_cache.mock_calls)
        with patch.object(
            scheduler, "_release_paged_cache_for_request"
        ) as release, patch.object(
            scheduler, "_drop_boundary_snapshots_for_request"
        ) as drop_snapshots:
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(branch_b, rid="b1", scheduler=scheduler)
            )
        after = list(scheduler.block_aware_cache.mock_calls)

        assert after == before          # nothing was evicted, removed or rewound
        release.assert_called_once_with("canonical-recovery:a1")
        drop_snapshots.assert_called_once_with("canonical-recovery:a1")
        assert job_a.committed_tokens == 8 * BLOCK
        assert job_a.published_boundaries == [8 * BLOCK]

    def test_two_branches_are_stored_under_different_token_keys(self):
        """Where the cross-branch question is actually decided.

        `_canonical_recovery_publish` keys the store by `list(job.tokens[:boundary])`, so
        two branches published past their fork point are two different keys.
        The common prefix is two blocks here so that the published range
        reaches past the fork; with the fork beyond the boundary the two
        stores would be identical and the test would prove nothing.
        """
        common = _common_prefix(2)
        branch_a = _branch(common, 1_000_000)
        branch_b = _branch(common, 2_000_000)
        boundary = 3 * BLOCK
        assert len(branch_a) == len(branch_b) == boundary

        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(branch_a, rid="a1", scheduler=scheduler)
        )
        job_a = scheduler._canonical_recovery_job
        worker_a = _publish(
            scheduler, job_a, boundary, boundary, block_ids=[1, 2, 3]
        )

        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(branch_b, rid="b1", scheduler=scheduler)
        )
        job_b = scheduler._canonical_recovery_job
        assert job_b is not job_a
        worker_b = _publish(
            scheduler, job_b, boundary, boundary, block_ids=[4, 5, 6]
        )

        stored_a = worker_a.call_args.args[1]
        stored_b = worker_b.call_args.args[1]
        assert stored_a == branch_a
        assert stored_b == branch_b
        assert stored_a != stored_b
        assert set(stored_a).isdisjoint(branch_b[2 * BLOCK:])
        assert set(stored_b).isdisjoint(branch_a[2 * BLOCK:])
        # Different jobs, so different request ids for the block tables too.
        assert worker_a.call_args.args[0] == "canonical-recovery:a1"
        assert worker_b.call_args.args[0] == "canonical-recovery:b1"
        assert job_a.committed_tokens == boundary
        assert job_b.committed_tokens == boundary


class TestB4CompactionDivergence:
    """A compacted turn: `A B C D E` becomes `A Summary E`.

    Measured over the divergence: jobs_created=1 (the replacement),
    target_extensions=0, superseded_jobs=1, cancelled_jobs=1, publishes=0.
    """

    def _lineage_a(self):
        return list(range(5 * BLOCK))

    def _compacted(self):
        """Shares its first block with A and nothing after it."""
        return list(range(BLOCK)) + [9_000_000 + i for i in range(3 * BLOCK)]

    def _started_job(self, scheduler, processed=2 * BLOCK, published=2 * BLOCK):
        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(self._lineage_a(), rid="r1", scheduler=scheduler)
        )
        job = scheduler._canonical_recovery_job
        job.processed_tokens = processed
        job.note_published(published)
        return job

    def test_a_compacted_stream_replaces_the_job_and_releases_the_old_footprint(self):
        """superseded_jobs=1, cancelled_jobs=1, jobs_created=1.

        The release is keyed by the *old* job's request id, which is the
        session key of the turn that created it — so the new turn carrying a
        different request id is what makes this assertion mean anything.
        """
        scheduler = _make_scheduler()
        old = self._started_job(scheduler)
        assert old.session_key == "r1"

        with patch.object(
            scheduler, "_release_paged_cache_for_request"
        ) as release, patch.object(
            scheduler, "_drop_boundary_snapshots_for_request"
        ) as drop_snapshots, _lineage_spies(scheduler) as counts:
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(self._compacted(), rid="r2", scheduler=scheduler)
            )

        new = scheduler._canonical_recovery_job
        assert new is not old
        assert old.cancelled is True
        assert old.prefill_state is None
        release.assert_called_once_with("canonical-recovery:r1")
        drop_snapshots.assert_called_once_with("canonical-recovery:r1")
        assert counts.drop_reasons == ["replaced"]
        assert counts.superseded_jobs == 1
        assert counts.cancelled_jobs == 1
        assert counts.jobs_created == 1
        assert counts.target_extensions == 0
        assert new.session_key == "r2"
        assert new.tokens == self._compacted()[: _target_for(4 * BLOCK)]

    def test_the_new_job_does_not_inherit_the_old_lineages_committed_count(self):
        """No false accounting: a fresh lineage starts at zero committed.

        The old job had 8,192 tokens of canonical prefix on a token list the
        session no longer has. Carrying that number onto the replacement would
        report a canonical prefix that does not exist and would suppress the
        very publishes that would create one.
        """
        scheduler = _make_scheduler()
        old = self._started_job(scheduler, published=2 * BLOCK)
        assert old.committed_tokens == 2 * BLOCK

        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(self._compacted(), rid="r2", scheduler=scheduler)
        )
        new = scheduler._canonical_recovery_job
        assert new.committed_tokens == 0
        assert new.published_boundaries == []
        assert new.processed_tokens == 0
        assert new.reached_target is False

    def test_nothing_is_published_for_the_old_lineage_across_the_divergence(self):
        scheduler = _make_scheduler()
        old = self._started_job(scheduler)
        with patch.object(scheduler, "_canonical_recovery_publish") as publish:
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(self._compacted(), rid="r2", scheduler=scheduler)
            )
            # Whatever the old job had reached, the drop offers none of it.
            assert old.publishable_boundary() == 0
        publish.assert_not_called()
        assert scheduler._canonical_recovery_counters.publishes == 0

    def test_an_extension_mid_chunk_publishes_the_old_boundary_of_the_same_lineage(self):
        """The branch `_canonical_recovery_step_inner` takes when the target moved under it.

        The state was built for the old target and would report `done` at it,
        so it is retired. The boundary the chunk landed on is still valid —
        `extend` verified the append, so those tokens are unchanged — and it
        is published before the state goes.
        """
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(list(range(20000)), rid="r1", scheduler=scheduler)
        )
        job = scheduler._canonical_recovery_job
        old_target = job.target_tokens
        state = MagicMock(
            base_size=0,
            tokens_processed=2 * BLOCK,
            cache=[MagicMock()],
            canonical_recovery_target_tokens=old_target,
        )
        job.prefill_state = state

        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(list(range(24600)), rid="r2", scheduler=scheduler)
        )
        assert scheduler._canonical_recovery_job is job
        assert job.target_tokens > old_target

        with patch.object(
            scheduler, "_step_prefill_chunk", return_value=False
        ), patch.object(scheduler, "_canonical_recovery_publish") as publish:
            assert scheduler._canonical_recovery_step_inner() is True

        publish.assert_called_once()
        published_job, boundary, published_state = publish.call_args.args
        assert published_job is job
        assert boundary == 2 * BLOCK
        assert published_state is state
        # The same lineage, so the tokens the publish would key on are the
        # ones the retired state actually computed.
        assert job.tokens[:boundary] == list(range(2 * BLOCK))
        assert job.prefill_state is None

    def test_a_replacement_mid_chunk_publishes_nothing(self):
        """Gap: the step keeps a stale `job` local across the chunk.

        `_canonical_recovery_step_inner` reads `self._canonical_recovery_job` once, at the top. A
        candidate that arrives while `_step_prefill_chunk` is running swaps
        the job underneath it, and the rest of the step goes on operating on
        the object that is no longer live: it writes `processed_tokens` onto
        the dropped job and reports `True`, meaning "a chunk ran", for a chunk
        whose result belongs to a discarded lineage.

        Nothing is published only because `_canonical_recovery_drop_job` sets `cancelled`
        and `publishable_boundary` returns 0 for a cancelled job. That flag,
        not the step, is what stands between a replaced lineage and a publish.
        """
        scheduler = _make_scheduler()
        old = self._started_job(scheduler, processed=0, published=0)
        state = MagicMock(
            base_size=0,
            tokens_processed=2 * BLOCK,
            cache=[MagicMock()],
            canonical_recovery_target_tokens=old.target_tokens,
        )
        old.prefill_state = state

        def swap_then_return(_state):
            scheduler.note_canonical_recovery_candidate(
                _sparse_request_on(self._compacted(), rid="r2", scheduler=scheduler)
            )
            return False

        with patch.object(
            scheduler, "_step_prefill_chunk", side_effect=swap_then_return
        ), patch.object(scheduler, "_canonical_recovery_publish") as publish:
            assert scheduler._canonical_recovery_step_inner() is True

        publish.assert_not_called()
        new = scheduler._canonical_recovery_job
        assert new is not old
        assert old.cancelled is True
        assert old.publishable_boundary() == 0
        # The stale local was still written to, and the live job saw no work.
        assert old.processed_tokens == 2 * BLOCK
        assert new.processed_tokens == 0
        assert new.prefill_state is None

    def test_a_swap_between_steps_is_seen_because_the_step_re_reads_the_job(self):
        """The same swap one step earlier is handled correctly.

        `job = self._canonical_recovery_job` at the top of `_canonical_recovery_step_inner` is read
        per step, so a replacement that completes before the step starts is
        picked up: the new job builds its own state and the old job's
        in-flight state is never handed to the model again.
        """
        scheduler = _make_scheduler()
        old = self._started_job(scheduler, processed=0, published=0)
        old_state = MagicMock(
            base_size=0,
            tokens_processed=2 * BLOCK,
            cache=[MagicMock()],
            canonical_recovery_target_tokens=old.target_tokens,
        )
        old.prefill_state = old_state

        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(self._compacted(), rid="r2", scheduler=scheduler)
        )
        new = scheduler._canonical_recovery_job
        assert new is not old
        assert old.prefill_state is None      # the drop retired it

        new_state = MagicMock(
            base_size=0,
            tokens_processed=BLOCK,
            cache=[MagicMock()],
            canonical_recovery_target_tokens=new.target_tokens,
        )
        begun_for = []
        stepped = []

        with patch.object(
            scheduler,
            "_canonical_recovery_begin_state",
            side_effect=lambda job: (begun_for.append(job), new_state)[1],
        ), patch.object(
            scheduler,
            "_step_prefill_chunk",
            side_effect=lambda state: (stepped.append(state), False)[1],
        ), patch.object(scheduler, "_canonical_recovery_publish") as publish:
            assert scheduler._canonical_recovery_step_inner() is True

        assert begun_for == [new]
        assert stepped == [new_state]
        assert old_state not in stepped
        assert new.processed_tokens == BLOCK
        # One whole block of the new lineage is publishable, and it is the new
        # lineage's tokens that would be keyed on.
        publish.assert_called_once()
        published_job, boundary, _state = publish.call_args.args
        assert published_job is new
        assert boundary == BLOCK
        assert new.tokens[:boundary] == self._compacted()[:BLOCK]


class TestB7FanOutIsBoundedByHavingOneSlot:
    """Many lineages at once, and what the single job slot does with them.

    An agent session fans out: a parent, four subagents, and short-lived
    lineages that end after a turn or two. The safety question is whether
    recovery can multiply — N lineages buying N budgets, or a queue that grows
    with the fan-out. The answer is structural rather than policy: there is one
    job slot and one budget object on the scheduler, and no queue at all.

    The cost of that structure is the economic result, and it belongs beside
    the safety one rather than instead of it: a lineage that arrives evicts the
    one before it, so under a fan-out that interleaves, no job survives long
    enough to reach a publishable boundary and the recovered-token yield is
    zero. That is an admission-policy signal, not a correctness failure.
    """

    def _branch(self, index: int, blocks: int = 4) -> list[int]:
        """A lineage sharing a 2-block parent prefix and then diverging."""
        parent = list(range(2 * BLOCK))
        own = [1_000_000 * (index + 1) + i for i in range(blocks * BLOCK - 2 * BLOCK)]
        return parent + own

    def test_four_interleaved_lineages_hold_one_job_and_one_budget(self):
        scheduler = _make_scheduler()
        budget = scheduler._canonical_recovery_budget
        branches = [self._branch(i) for i in range(4)]
        with _lineage_spies(scheduler) as counts:
            for _round in range(3):
                for index, tokens in enumerate(branches):
                    scheduler.note_canonical_recovery_candidate(
                        _sparse_request_on(tokens, rid=f"b{index}", scheduler=scheduler)
                    )
                    # One slot, always: there is nowhere for a second job to go.
                    assert scheduler._canonical_recovery_job is not None
                    assert scheduler._canonical_recovery_budget is budget

        # Twelve candidates, twelve jobs, eleven of them superseded. Nothing
        # queued, because no queue exists to hold it.
        assert counts.jobs_created == 12
        assert counts.superseded_jobs == 11
        assert counts.target_extensions == 0
        assert scheduler._canonical_recovery_counters.publishes == 0

    def test_an_evicted_lineage_takes_its_in_flight_work_with_it(self):
        """The yield under an interleaved fan-out, stated as a number."""
        scheduler = _make_scheduler()
        branches = [self._branch(i) for i in range(4)]
        recovered = 0
        for _round in range(3):
            for index, tokens in enumerate(branches):
                scheduler.note_canonical_recovery_candidate(
                    _sparse_request_on(tokens, rid=f"b{index}", scheduler=scheduler)
                )
                job = scheduler._canonical_recovery_job
                # Each job gets as far as one block of dense work before the
                # next lineage arrives, which is more than a real 4,096-token
                # chunk would manage inside one of these gaps.
                job.processed_tokens = BLOCK
                recovered += BLOCK
        assert recovered == 12 * BLOCK
        # None of it was ever published, so none of it can ever be reused.
        assert scheduler._canonical_recovery_counters.publishes == 0
        assert scheduler._canonical_recovery_job.committed_tokens == 0

    def test_a_lineage_that_ended_still_owns_the_slot(self):
        """Nothing tells recovery that a session is over.

        A short-lived lineage's job stays live until some other candidate
        replaces it or it finishes its target. It is bounded — one job, one
        token list, one set of blocks — and it is held on a prompt nobody will
        ever send again, which is the other half of why fan-out yield is low.
        """
        scheduler = _make_scheduler()
        ephemeral = self._branch(9)
        scheduler.note_canonical_recovery_candidate(
            _sparse_request_on(ephemeral, rid="gone", scheduler=scheduler)
        )
        job = scheduler._canonical_recovery_job
        assert job is not None
        for _ in range(50):
            scheduler._canonical_recovery_note_step(did_foreground_work=False)
        assert scheduler._canonical_recovery_job is job
        assert not job.cancelled
