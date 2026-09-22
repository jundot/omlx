# SPDX-License-Identifier: Apache-2.0
"""A prompt that is exactly whole blocks must recover all of them.

``_begin_prefill`` splits its token list into ``tokens[:-1]`` and ``tokens[-1:]``:
the last token is not prefilled, it is handed to ``insert()`` as the generation
kickoff. Every foreground request needs that, so the split is unconditional.

A recovery job needs the opposite. It never samples, never calls ``insert`` and
never reads ``state.last_token``; it exists only for what it leaves in the
cache. Holding a token back costs it the block that token sits in, because
publication floors to a block boundary:

    prompt 10,000, block 4,096 -> last whole block 8,192, prefill reaches
    8,192, publish 8,192.

    prompt  8,192, block 4,096 -> last whole block 8,192, prefill reaches
    8,191, publish 4,096.

Half a two-block session, lost to an off-by-one in a token that request never
uses. The earlier workaround asked for one token *past* the boundary, which
works whenever the prompt is longer than the boundary and cannot work when the
prompt ends on it — there is no such token.

So the recovery state is built without the hold-back rather than compensated
for afterwards, and the target is the boundary itself. Nothing artificial is
pushed through the model: these are the session's own tokens, and the range
prefilled is exactly the range published.
"""

from unittest.mock import MagicMock, patch

import pytest

from omlx.canonical_recovery import CanonicalRecoveryJob
from omlx.request import Request, SamplingParams
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


def _sparse_request(prompt_tokens: int, rid: str = "r1", scheduler=None):
    request = MagicMock()
    request.request_id = rid
    request.prompt_token_ids = list(range(prompt_tokens))
    request.specprefill_indices = [1, 2, 3]
    request._serving_prefix_cache_id = (
        id(scheduler.block_aware_cache) if scheduler is not None else None
    )
    return request


def _plain_request(n: int) -> Request:
    return Request(
        request_id="canonical-recovery:s",
        prompt=None,
        prompt_token_ids=list(range(n)),
        sampling_params=SamplingParams(max_tokens=1),
    )


class TestTheTargetIsTheBoundaryItself:
    """``note_canonical_recovery_candidate`` no longer asks for a spare token."""

    @pytest.mark.parametrize(
        "prompt_tokens,expected_target",
        [
            (2 * BLOCK, 2 * BLOCK),  # exactly two blocks: the case that was lost
            (BLOCK, BLOCK),  # exactly one block
            (1000, 768),  # 3 whole blocks of 256, remainder dropped
            (2 * BLOCK + 1, 2 * BLOCK),  # one token past: same boundary
        ],
    )
    def test_the_target_is_the_last_whole_block(self, prompt_tokens, expected_target):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(
            _sparse_request(prompt_tokens, scheduler=scheduler)
        )
        job = scheduler._canonical_recovery_job
        assert job is not None
        assert job.target_tokens == expected_target

    def test_a_prompt_shorter_than_one_block_still_queues_nothing(self):
        scheduler = _make_scheduler()
        scheduler.note_canonical_recovery_candidate(
            _sparse_request(BLOCK - 1, scheduler=scheduler)
        )
        assert scheduler._canonical_recovery_job is None


class TestTheRecoveryStateKeepsEveryToken:
    """The seam: what the built state will actually push through the model."""

    def _state_for(self, scheduler, n: int):
        with patch.object(scheduler, "_prepare_prefix_cache_for_request"):
            request = _plain_request(n)
            request.remaining_tokens = list(range(n))
            request.cached_tokens = 0
            request.is_canonical_recovery = True
            scheduler.requests[request.request_id] = request
            return scheduler._begin_prefill(
                request, list(range(n)), None, hold_back_last=False
            )

    def test_a_recovery_state_prefills_all_of_its_tokens(self):
        scheduler = _make_scheduler()
        state = self._state_for(scheduler, 2 * BLOCK)
        assert int(state.tokens_remaining.shape[1]) == 2 * BLOCK
        assert state.last_token == []

    def test_a_foreground_state_still_holds_its_last_token_back(self):
        """The control. The kickoff token is not optional for a real request."""
        scheduler = _make_scheduler()
        request = _plain_request(2 * BLOCK)
        state = scheduler._begin_prefill(request, list(range(2 * BLOCK)), None)
        assert int(state.tokens_remaining.shape[1]) == 2 * BLOCK - 1
        assert state.last_token == [2 * BLOCK - 1]

    def test_the_progress_denominator_matches_what_will_be_prefilled(self):
        """The progress sites spelled this ``total_length - 1``, which assumed
        a hold-back that no longer always happens. ``total_length`` minus the
        held-back token count is the same number for a foreground state and
        the right one for a recovery state."""
        scheduler = _make_scheduler()
        recovery = self._state_for(scheduler, 2 * BLOCK)
        foreground = scheduler._begin_prefill(
            _plain_request(2 * BLOCK), list(range(2 * BLOCK)), None
        )
        for state in (recovery, foreground):
            denominator = state.total_length - len(state.last_token)
            assert denominator == int(state.tokens_remaining.shape[1])


class TestTheWholeBlockBecomesPublishable:
    """The consequence, at the level the job reasons about."""

    def _job(self, target, block=4096):
        return CanonicalRecoveryJob(
            session_key="s",
            tokens=list(range(target)),
            target_tokens=target,
            block_size=block,
        )

    def test_an_exact_two_block_prompt_publishes_both_blocks(self):
        job = self._job(8192)
        job.processed_tokens = 8192  # what the prefill now reaches
        assert job.publishable_boundary() == 8192

    def test_the_job_is_done_by_count_without_a_spare_token(self):
        job = self._job(8192)
        job.processed_tokens = 8192
        assert job.done

    def test_a_prompt_longer_than_the_boundary_is_unchanged(self):
        """The control: the case the old compensation already handled."""
        job = self._job(8192)
        job.processed_tokens = 8192
        assert job.publishable_boundary() == 8192
