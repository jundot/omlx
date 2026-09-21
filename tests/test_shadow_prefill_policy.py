# SPDX-License-Identifier: Apache-2.0
"""Tests for the shadow-prefill decision layer.

These are the conditions an earlier background-densification prototype got
wrong, written as assertions rather than as review notes. Nothing here loads a
model: the point of keeping the policy free of MLX is that it can be tested at
this speed.
"""

import pytest

from omlx.shadow_prefill import (
    DEFAULT_BUDGET_WINDOW_S,
    PublishMode,
    ShadowBudget,
    ShadowCounters,
    ShadowJob,
    canonical_debt,
    catch_up_ratio,
    safe_publish_boundary,
    shadow_is_runnable,
)


class TestSafeBoundary:
    def test_partial_block_is_never_publishable(self):
        assert safe_publish_boundary(tokens_committed=1023, block_size=1024) == 0

    def test_exact_boundary_publishes_itself(self):
        assert safe_publish_boundary(tokens_committed=2048, block_size=1024) == 2048

    def test_publication_floors_to_the_block_below(self):
        assert safe_publish_boundary(tokens_committed=2500, block_size=1024) == 2048

    @pytest.mark.parametrize("block_size", [0, -1])
    def test_a_nonpositive_block_size_publishes_nothing(self, block_size):
        assert safe_publish_boundary(tokens_committed=9999, block_size=block_size) == 0


class TestDebt:
    def test_debt_is_the_uncanonicalized_remainder(self):
        assert canonical_debt(prompt_tokens=48000, longest_committed_prefix=28672) == 19328

    def test_debt_never_goes_negative(self):
        assert canonical_debt(prompt_tokens=1000, longest_committed_prefix=4096) == 0

    def test_catch_up_ratio_is_none_when_context_did_not_grow(self):
        assert catch_up_ratio(
            delta_committed_tokens=1024, delta_required_context_tokens=0
        ) is None

    def test_catch_up_ratio_above_one_means_gaining(self):
        assert catch_up_ratio(
            delta_committed_tokens=16384, delta_required_context_tokens=8192
        ) == 2.0


class TestBudget:
    """The lifetime side of the budget: what it reports, not what it permits.

    Permission is per window and is pinned in ``TestReplenishingBudget``.
    What lives here is the measured lifetime share, which is the number the
    experiment reads back as the recovery compute share actually received.
    """

    def test_a_zero_budget_never_allows_service(self):
        budget = ShadowBudget(pct=0.0, wall_start_s=0.0, window_start_s=0.0)
        assert not budget.allows(now=100.0)

    def test_the_lifetime_share_is_service_over_wall_time(self):
        budget = ShadowBudget(pct=10.0, wall_start_s=0.0, window_start_s=0.0)
        budget.note_service(5.0)
        assert budget.share(now=100.0) == pytest.approx(0.05)

    def test_the_share_is_zero_before_any_time_has_passed(self):
        """A ratio against a zero denominator is not a measurement."""
        budget = ShadowBudget(pct=10.0, wall_start_s=0.0, window_start_s=0.0)
        assert budget.share(now=0.0) == 0.0

    def test_negative_service_cannot_buy_back_budget(self):
        budget = ShadowBudget(pct=10.0, wall_start_s=0.0, window_start_s=0.0)
        budget.note_service(10.0)
        budget.note_service(-5.0)
        assert budget.service_s == 10.0


class TestReplenishingBudget:
    """The windowed budget: an allowance that comes back, and a debt that clears.

    The lifetime cumulative-share budget this replaced had one failure mode.
    One chunk that overran the allowance early held the measured share above
    the cap until the denominator grew back, and on a long session that is the
    rest of the run: the job overshot once and was finished rather than late.
    Each test here pins a property that failure violated, and each one drives
    the clock with an explicit ``now`` so nothing reads ``perf_counter``.
    """

    def _budget(self, pct=10.0, window_s=10.0):
        """A budget whose two clocks both start at zero.

        ``wall_start_s`` and ``window_start_s`` have independent default
        factories, so a budget that sets only one of them takes the other from
        ``perf_counter`` and can never be driven by an explicit ``now``. The
        defaults give a 1 s allowance in a 10 s window.
        """
        return ShadowBudget(
            pct=pct, window_s=window_s, wall_start_s=0.0, window_start_s=0.0
        )

    def test_a_spent_allowance_comes_back_in_the_next_window(self):
        """A fully consumed window refuses service, and the roll restores it."""
        budget = self._budget()
        assert budget.allowance_s == pytest.approx(1.0)
        assert budget.allows(now=0.0)
        budget.note_service(1.0)
        assert not budget.allows(now=0.0)
        assert not budget.allows(now=9.9)
        assert budget.allows(now=10.0)

    def test_unused_windows_do_not_bank_credit(self):
        """Five idle windows buy one allowance, not five.

        Banked credit is what turns a bounded share into one long slice taken
        in front of the foreground, so the roll discards whatever a window did
        not use rather than carrying it forward.
        """
        budget = self._budget()
        taken = 0.0
        while budget.allows(now=50.0):      # five whole windows of idleness
            budget.note_service(0.25)
            taken += 0.25
        assert taken == pytest.approx(1.0)
        assert budget.windows == 6

    def test_an_overshoot_locks_the_job_out_for_exactly_one_window(self):
        """A chunk six times the allowance costs one window of service, not six.

        The carried debt is capped at a single allowance, so the window after
        the overrun opens already spent and the window after *that* opens
        clean. The bound is asserted as a specific window, not as "eventually".
        """
        budget = self._budget()
        assert budget.allows(now=0.0)
        budget.note_service(6.0)            # one indivisible chunk, 6x allowance
        assert not budget.allows(now=5.0)   # the rest of the window it overran
        assert not budget.allows(now=10.0)  # the next window, paying the debt
        assert budget.allows(now=20.0)      # and no later than the one after
        assert budget.locked_out_windows == 1
        assert budget.overshoot_s == pytest.approx(5.0)

    def test_a_zero_budget_allows_nothing_in_any_window(self):
        """``pct=0`` is off, not "off until enough wall time has accumulated"."""
        budget = self._budget(pct=0.0)
        assert budget.allowance_s == 0.0
        for now in (0.0, 5.0, 10.0, 100.0, 3600.0, 86400.0):
            assert not budget.allows(now=now)
        assert budget.share(now=86400.0) == 0.0

    def test_repeated_overshoot_never_starves_the_job_permanently(self):
        """Chunks that each overrun 6x still earn service across 200 windows.

        The debt cap is the whole of why this holds: an uncapped carry would
        have the job paying off its first window for the rest of the run,
        which is the behaviour the lifetime budget actually had.
        """
        budget = self._budget()
        windows = 200
        grants = 0
        for tick in range(windows * 10):    # probe once a second
            if budget.allows(now=float(tick)):
                budget.note_service(6.0)
                grants += 1
        # A grant costs its own window and the next, so the floor is one grant
        # per two windows. Assert well inside that rather than at the edge.
        assert grants >= 50

    def test_lifetime_share_stays_within_the_stated_bound(self):
        """Measured share is bounded by the nominal pct plus one chunk a window.

        A chunk cannot be interrupted once it is handed to the model, so the
        last chunk a window admits may start with the allowance all but spent
        and still run to completion. A window therefore grants at most
        ``allowance + chunk`` seconds, and over ``W`` windows of ``window_s``
        each the lifetime share is bounded by ``(allowance + chunk) /
        window_s``, which is ``pct/100 + chunk/window_s``. With a 1 s
        allowance in a 10 s window and a 0.4 s chunk that is 0.10 + 0.04 =
        0.14. The bound is derived from the documented semantics, not read off
        what this implementation happens to produce.
        """
        budget = self._budget()
        chunk_s = 0.4
        windows = 100
        for tick in range(windows * 100):   # probe ten times a second
            if budget.allows(now=tick / 10.0):
                budget.note_service(chunk_s)
        end = float(windows) * budget.window_s
        bound = budget.pct / 100.0 + chunk_s / budget.window_s
        assert budget.share(now=end) <= bound
        # And it is a budget that actually fired, not one that granted nothing.
        assert budget.share(now=end) > 0.5 * (budget.pct / 100.0)

    def test_a_gap_does_not_discharge_the_carried_debt(self):
        """Skipping windows does not clear the overshoot.

        The gap between two `allows` calls cannot tell "the job had nothing
        to do" from "the foreground was busy for a minute", and the second is
        the case the budget exists for. Discharging on the gap meant any busy
        period longer than two windows wiped the debt, so the cap stopped
        holding on exactly the sessions it was written for. The debt is still
        capped at one allowance, so the delay it can impose is still one
        window however long the gap was.
        """
        budget = self._budget()
        assert budget.allows(now=0.0)
        budget.note_service(6.0)            # 5 s over a 1 s allowance
        assert not budget.allows(now=30.0)  # three windows later, still owed
        assert budget.window_service_s == pytest.approx(1.0)
        assert budget.windows == 4
        assert budget.allows(now=40.0)      # and never more than one window

    def test_a_part_spent_window_is_not_a_lockout(self):
        """A window that opens in debt but still grants service is not a
        lockout. Counting it as one reported 500 lockouts across 667 windows
        on a sweep where no service was ever withheld."""
        budget = self._budget()                 # 1 s allowance, 10 s window
        assert budget.allows(now=0.0)
        budget.note_service(1.5)                # half an allowance of debt
        assert budget.allows(now=10.0)          # 0.5 s still available
        assert budget.locked_out_windows == 0

    def test_a_fully_spent_window_is_a_lockout(self):
        budget = self._budget()
        assert budget.allows(now=0.0)
        budget.note_service(2.5)                # a full allowance of debt
        assert not budget.allows(now=10.0)
        assert budget.locked_out_windows == 1

    def test_a_non_positive_window_is_corrected_not_interpreted(self):
        """Nothing writes this field today, so a bad value would arrive by
        mistake. Left alone it published a zero allowance beside a non-zero
        service share, which is telemetry that contradicts itself."""
        budget = ShadowBudget(pct=10.0, window_s=-5.0, wall_start_s=0.0,
                              window_start_s=0.0)
        assert budget.window_s == DEFAULT_BUDGET_WINDOW_S
        assert budget.allowance_s > 0

    def test_as_dict_reports_the_service_that_was_actually_noted(self):
        """The reported share, windows, lockouts and overshoot match the run.

        The sequence below is: 3 s served in window 1 against a 1 s allowance
        (2 s of overshoot), window 2 locked out paying 1 s of that debt, then
        two 0.5 s chunks in window 3. Total service is 4 s over 30 s of wall
        time. ``as_dict`` does not roll the window, so ``budget_windows`` is
        the count as of the last ``allows``, which is window 3.
        """
        budget = self._budget()
        assert budget.allows(now=0.0)
        budget.note_service(3.0)
        assert not budget.allows(now=10.0)
        assert budget.allows(now=20.0)
        budget.note_service(0.5)
        assert budget.allows(now=25.0)
        budget.note_service(0.5)

        payload = budget.as_dict(now=30.0)
        assert budget.service_s == pytest.approx(3.0 + 0.5 + 0.5)
        assert payload["budget_allowance_s"] == pytest.approx(1.0)
        assert payload["service_share"] == pytest.approx(4.0 / 30.0, abs=1e-6)
        assert payload["budget_windows"] == 3
        assert payload["budget_locked_out_windows"] == 1
        assert payload["budget_overshoot_s"] == pytest.approx(2.0)
        assert payload["window_service_s"] == pytest.approx(1.0)


class TestRunnable:
    def _kwargs(self, **over):
        base = dict(
            enabled=True,
            budget=ShadowBudget(pct=10.0, wall_start_s=0.0),
            has_job=True,
            waiting_requests=0,
            running_requests=0,
            prefilling_requests=0,
            specprefill_active=False,
            inbound_requests=0,
            consecutive_idle_steps=2,
            now=1.0,
        )
        base.update(over)
        return base

    def test_idle_scheduler_with_budget_is_runnable(self):
        assert shadow_is_runnable(**self._kwargs())

    def test_an_active_specprefill_blocks_the_shadow(self):
        """The offset RoPE wrapper is installed on the shared model.

        A dense forward taken while it is installed would read the foreground
        request's position offset, so this is a correctness condition, not a
        fairness one.
        """
        assert not shadow_is_runnable(**self._kwargs(specprefill_active=True))

    def test_an_inbound_request_blocks_the_shadow(self):
        """A request is invisible to the scheduler until admission runs."""
        assert not shadow_is_runnable(**self._kwargs(inbound_requests=1))

    @pytest.mark.parametrize(
        "field", ["waiting_requests", "running_requests", "prefilling_requests"]
    )
    def test_any_foreground_work_blocks_the_shadow(self, field):
        assert not shadow_is_runnable(**self._kwargs(**{field: 1}))

    def test_one_idle_step_is_not_enough(self):
        """Back-to-back slices leave no window for a request to announce itself."""
        assert not shadow_is_runnable(**self._kwargs(consecutive_idle_steps=1))

    def test_an_exhausted_budget_blocks_the_shadow(self):
        budget = ShadowBudget(pct=10.0, wall_start_s=0.0)
        budget.note_service(50.0)
        assert not shadow_is_runnable(**self._kwargs(budget=budget, now=100.0))

    def test_no_job_is_not_runnable(self):
        assert not shadow_is_runnable(**self._kwargs(has_job=False))

    def test_disabled_is_not_runnable(self):
        assert not shadow_is_runnable(**self._kwargs(enabled=False))


class TestPublication:
    def _job(self, mode, **over):
        kwargs = dict(
            session_key="s", tokens=list(range(4096)), target_tokens=4096,
            block_size=1024, publish_mode=mode,
        )
        kwargs.update(over)
        return ShadowJob(**kwargs)

    def test_terminal_publishes_nothing_before_the_target_finishes(self):
        job = self._job(PublishMode.TERMINAL)
        job.processed_tokens = 3072
        assert job.publishable_boundary() == 0

    def test_terminal_publishes_at_the_target(self):
        job = self._job(PublishMode.TERMINAL)
        job.processed_tokens = 4096
        assert job.publishable_boundary() == 4096

    def test_progressive_publishes_each_new_boundary(self):
        job = self._job(PublishMode.PROGRESSIVE)
        job.processed_tokens = 1024
        assert job.publishable_boundary() == 1024
        job.note_published(1024)
        job.processed_tokens = 2048
        assert job.publishable_boundary() == 2048

    def test_progressive_does_not_republish_a_boundary(self):
        job = self._job(PublishMode.PROGRESSIVE)
        job.processed_tokens = 2048
        job.note_published(2048)
        job.processed_tokens = 2500
        assert job.publishable_boundary() == 0

    def test_an_interrupted_progressive_job_keeps_what_it_published(self):
        """This is the whole difference between PASS and Shadow-End."""
        progressive = self._job(PublishMode.PROGRESSIVE)
        terminal = self._job(PublishMode.TERMINAL)
        for job in (progressive, terminal):
            job.processed_tokens = 3072
            boundary = job.publishable_boundary()
            if boundary:
                job.note_published(boundary)
            job.cancelled = True
        assert progressive.committed_tokens == 3072
        assert terminal.committed_tokens == 0

    def test_a_cancelled_job_publishes_nothing_further(self):
        job = self._job(PublishMode.PROGRESSIVE)
        job.processed_tokens = 4096
        job.cancelled = True
        assert job.publishable_boundary() == 0


class TestHeldBackToken:
    """The prefill keeps the last token back, and the target has to allow for it.

    ``_step_prefill_chunk`` stops one token short of the range it was given,
    because that token is the generation kickoff. A job whose target is
    exactly a block boundary therefore tops out at ``boundary - 1`` and
    publishes the block below it. On a two-block session that is the
    difference between recovering all of it and recovering half: the job
    reached its target on every idle window, committed the same 4,096 of
    8,192 each time, and re-read the same tokens for the rest of the session.
    """

    def _job(self, target, block=4096):
        return ShadowJob(
            session_key="s", tokens=list(range(target)), target_tokens=target,
            block_size=block,
        )

    def test_a_target_on_the_boundary_loses_the_block_below_it(self):
        job = self._job(8192)
        job.processed_tokens = 8191          # what the prefill actually reaches
        assert job.publishable_boundary() == 4096

    def test_a_target_one_past_the_boundary_publishes_it(self):
        job = self._job(8193)
        job.processed_tokens = 8192
        assert job.publishable_boundary() == 8192


class TestSingleFlightGrowth:
    def test_an_append_extends_the_live_job(self):
        job = ShadowJob(
            session_key="s", tokens=list(range(1000)), target_tokens=1000, block_size=256
        )
        assert job.extend(list(range(1500)))
        assert job.target_tokens == 1500

    def test_a_shorter_prompt_is_not_a_growth(self):
        job = ShadowJob(
            session_key="s", tokens=list(range(1000)), target_tokens=1000, block_size=256
        )
        assert not job.extend(list(range(500)))
        assert job.target_tokens == 1000

    def test_a_rewritten_prefix_is_refused(self):
        """Publishing state for a prefix the session no longer has is the failure
        the placeholder rejection exists to prevent. Refuse it here too."""
        job = ShadowJob(
            session_key="s", tokens=list(range(1000)), target_tokens=1000, block_size=256
        )
        rewritten = list(range(1500))
        rewritten[10] = -1
        assert not job.extend(rewritten)
        assert job.target_tokens == 1000


class TestCounters:
    def test_counters_round_trip_to_a_dict(self):
        counters = ShadowCounters(runnable_steps=3, scheduled_steps=2, service_s=1.23456789)
        payload = counters.as_dict()
        assert payload["runnable_steps"] == 3
        assert payload["scheduled_steps"] == 2
        assert payload["service_s"] == pytest.approx(1.234568)

    def test_service_and_runnable_are_reported_separately(self):
        """"The shadow got no service" and "the shadow got service and it was
        not enough" are different results; the counters must tell them apart."""
        counters = ShadowCounters(runnable_steps=100, scheduled_steps=0)
        payload = counters.as_dict()
        assert payload["runnable_steps"] and not payload["scheduled_steps"]


class TestTargetCompletion:
    """The prefill path holds the last token back for the generation kickoff.

    A job whose completion is decided by ``processed_tokens >= target_tokens``
    therefore never completes: it tops out one token short, stays "live", and
    keeps the engine loop awake on an idle server forever.
    """

    def _job(self):
        return ShadowJob(
            session_key="s", tokens=list(range(8192)), target_tokens=8192,
            block_size=4096, publish_mode=PublishMode.TERMINAL,
        )

    def test_one_token_short_is_not_done_by_count_alone(self):
        job = self._job()
        job.processed_tokens = 8191
        assert not job.done

    def test_the_chunk_loop_can_declare_the_target_reached(self):
        job = self._job()
        job.processed_tokens = 8191
        job.note_reached_target()
        assert job.done
        # 8191 processed tokens floor to the block below, not to the target.
        assert job.publishable_boundary() == 4096

    def test_extending_a_finished_job_reopens_it(self):
        job = self._job()
        job.note_reached_target()
        assert job.done
        assert job.extend(list(range(12288)))
        assert not job.done
