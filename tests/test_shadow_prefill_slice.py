# SPDX-License-Identifier: Apache-2.0
"""The recovery execution slice is not the canonical publication grain.

A recovery slice cannot be interrupted once it is handed to the model, so the
worst wait a foreground request can suffer is one slice. At the block grain
that unit is a whole cache-block forward, which is long enough that an
interactive request cannot be asked to absorb it.

Lowering the budget does not help, and the reason is worth stating because it
is what these tests exist to act on: a budget is a ceiling on how *often* a
slice runs. At 5% three probes of twenty-four collided and at 100% six did,
while the worst collision stayed in the same 12-15 s band either way. The
budget owns collision frequency; the slice owns collision severity.

What makes the slice free to shrink is that publication does not depend on it.
Canonical state for a non-sliceable layer exists only at a cache block
boundary, so publication is fixed at the block. Execution is not:
``_step_prefill_chunk`` advances one slice of a *persistent* prefill state,
``clamp_prefill_chunk_to_boundary`` already refuses to overshoot a boundary,
and ``safe_publish_boundary`` floors publication to block multiples. These
tests pin that separation — that a smaller slice reaches the same boundaries,
publishes the same prefixes in the same order, and recomputes nothing.
"""

from unittest.mock import MagicMock

import pytest

from omlx.model_settings import ModelSettings
from omlx.prefill_boundaries import clamp_prefill_chunk_to_boundary
from omlx.scheduler import Scheduler, SchedulerConfig
from omlx.shadow_prefill import (
    ShadowJob,
    apply_shadow_prefill_settings,
    safe_publish_boundary,
    shadow_slice_cap,
)

BLOCK = 4096


def _make_scheduler(**config_over) -> Scheduler:
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2

    config_kwargs = dict(
        max_num_seqs=8,
        prefill_step_size=2048,
        chunked_prefill=True,
        paged_cache_block_size=BLOCK,
        shadow_prefill_enabled=True,
        shadow_prefill_global_budget_pct=10.0,
    )
    config_kwargs.update(config_over)
    scheduler = Scheduler(
        model=model, tokenizer=tokenizer, config=SchedulerConfig(**config_kwargs)
    )
    scheduler.block_aware_cache = MagicMock()
    scheduler._unreconstructible_cache_model = False
    return scheduler


def _request(is_shadow: bool):
    request = MagicMock()
    request.is_shadow = is_shadow
    return request


class TestTheCapAppliesToRecoveryOnly:
    def test_a_foreground_request_keeps_the_ordinary_step_size(self):
        scheduler = _make_scheduler(shadow_prefill_slice_tokens=256)
        assert shadow_slice_cap(scheduler.config, _request(False), 2048) == 2048

    def test_a_recovery_request_is_capped(self):
        scheduler = _make_scheduler(shadow_prefill_slice_tokens=256)
        assert shadow_slice_cap(scheduler.config, _request(True), 2048) == 256

    def test_zero_leaves_recovery_on_the_ordinary_step_size(self):
        """The default: recovery runs at the ordinary prefill step size.

        This is the state the block-grain blocking interval was characterised
        in, and it is what the cap exists to narrow.
        """
        scheduler = _make_scheduler(shadow_prefill_slice_tokens=0)
        assert shadow_slice_cap(scheduler.config, _request(True), 2048) == 2048

    def test_the_cap_only_ever_lowers(self):
        scheduler = _make_scheduler(shadow_prefill_slice_tokens=8192)
        assert shadow_slice_cap(scheduler.config, _request(True), 512) == 512

    def test_a_request_with_no_shadow_marker_is_foreground(self):
        scheduler = _make_scheduler(shadow_prefill_slice_tokens=256)
        plain = MagicMock(spec=[])
        assert shadow_slice_cap(scheduler.config, plain, 2048) == 2048


def _walk(slice_tokens: int, target: int, block: int = BLOCK):
    """Drive a job to *target* in slices, the way the chunk loop would.

    Uses the runtime's own clamp rather than a re-implementation of it, so the
    test fails if the clamp stops being what makes this work.
    """
    job = ShadowJob(
        session_key="s",
        tokens=list(range(target)),
        target_tokens=target,
        block_size=block,
    )
    processed = 0
    slices = 0
    while processed < target:
        step = min(slice_tokens, target - processed)
        step = clamp_prefill_chunk_to_boundary(
            step, cache_tokens=processed, block_size=block
        )
        processed += step
        slices += 1
        job.processed_tokens = processed
        boundary = job.publishable_boundary()
        if boundary:
            job.note_published(boundary)
    return job, slices


class TestPublicationDoesNotMoveWithTheSlice:
    @pytest.mark.parametrize("slice_tokens", [256, 512, 1024, 2048, 4096])
    def test_every_slice_size_publishes_the_same_boundaries(self, slice_tokens):
        job, _slices = _walk(slice_tokens, 4 * BLOCK)
        assert job.published_boundaries == [BLOCK, 2 * BLOCK, 3 * BLOCK, 4 * BLOCK]

    @pytest.mark.parametrize("slice_tokens", [256, 512, 1024, 2048, 4096])
    def test_no_slice_size_publishes_a_partial_block(self, slice_tokens):
        job, _slices = _walk(slice_tokens, 3 * BLOCK + 1000)
        assert all(b % BLOCK == 0 for b in job.published_boundaries)
        assert job.committed_tokens == 3 * BLOCK

    def test_a_smaller_slice_costs_more_slices_and_nothing_else(self):
        big, big_slices = _walk(4096, 4 * BLOCK)
        small, small_slices = _walk(256, 4 * BLOCK)
        assert small_slices > big_slices
        assert small.published_boundaries == big.published_boundaries
        assert small.processed_tokens == big.processed_tokens

    def test_progress_only_ever_moves_forward(self):
        """A slice resumes the same state; it does not restart the range.

        If a smaller slice made the job recompute, `processed_tokens` would go
        backwards somewhere in the walk.
        """
        job = ShadowJob(
            session_key="s",
            tokens=list(range(4 * BLOCK)),
            target_tokens=4 * BLOCK,
            block_size=BLOCK,
        )
        seen = []
        processed = 0
        while processed < 4 * BLOCK:
            step = clamp_prefill_chunk_to_boundary(
                min(256, 4 * BLOCK - processed),
                cache_tokens=processed,
                block_size=BLOCK,
            )
            processed += step
            job.processed_tokens = processed
            seen.append(processed)
        assert seen == sorted(seen)
        assert len(set(seen)) == len(seen)
        assert seen[-1] == 4 * BLOCK

    def test_the_clamp_is_what_keeps_a_slice_off_a_boundary(self):
        """The property the whole separation rests on, asserted directly."""
        for slice_tokens in (256, 512, 1024, 2048, 4096, 8192):
            processed = 0
            while processed < 3 * BLOCK:
                step = clamp_prefill_chunk_to_boundary(
                    min(slice_tokens, 3 * BLOCK - processed),
                    cache_tokens=processed,
                    block_size=BLOCK,
                )
                assert processed // BLOCK == (processed + step - 1) // BLOCK, (
                    f"slice {slice_tokens} crossed a boundary at {processed}"
                )
                processed += step

    def test_the_publication_floor_is_a_pure_function_of_the_block(self):
        """`safe_publish_boundary` has no slice term, and that is the point."""
        for tokens in (1, 255, 4095, 4096, 4097, 9000):
            assert safe_publish_boundary(
                tokens_committed=tokens, block_size=BLOCK
            ) == (tokens // BLOCK) * BLOCK


class TestTheSliceReachesTheScheduler:
    def test_a_model_setting_carries_onto_the_scheduler_config(self):
        config = SchedulerConfig()
        apply_shadow_prefill_settings(
            config,
            ModelSettings(
                shadow_prefill_enabled=True,
                shadow_prefill_slice_tokens=512,
            ),
        )
        assert config.shadow_prefill_slice_tokens == 512

    def test_an_absent_setting_is_zero_rather_than_a_guess(self):
        config = SchedulerConfig()
        apply_shadow_prefill_settings(config, ModelSettings())
        assert config.shadow_prefill_slice_tokens == 0

    def test_a_negative_slice_is_refused(self):
        with pytest.raises(ValueError, match="shadow_prefill_slice_tokens"):
            ModelSettings(shadow_prefill_slice_tokens=-1)
