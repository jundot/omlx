# SPDX-License-Identifier: Apache-2.0
"""One recovery budget for every engine sharing the accelerator.

A recovery budget rations an accelerator, and an engine pool has exactly one
between all of its engines. Built per scheduler — which is what a float on a
shallow-copied config gives you — M loaded models granted M times the
configured share, each measuring its own against its own wall clock, and
nothing anywhere added them up.

The fix follows the shape the runtime already uses for its one other
cross-engine ceiling: the pool creates an *object* before any engine loads and
hangs it on the shared scheduler config, where the same two shallow copies that
snapshot every scalar pass the reference through untouched.

Two separate things live on that object and they are not substitutes:

- the **budget** bounds the aggregate *share* of wall time;
- the **claim** bounds concurrent *execution*, and closes the window between
  a job clearing its tracker entry to rebuild state and writing a new one at
  the end of its first chunk — an interval in which a peer sees a process with
  no recovery running and starts a slice of its own.

These are the six scenarios the design review named, plus the two the ownership
model implies: a scheduler with no pool still gets a budget, and the shared one
refuses to be reset by any single owner.
"""

from unittest.mock import MagicMock, patch

import pytest

from omlx.decode_activity import get_decode_activity
from omlx.engine_pool import EnginePool
from omlx.prefill_progress import get_prefill_tracker
from omlx.scheduler import Scheduler, SchedulerConfig
from omlx.shadow_prefill import ShadowBudget

WINDOW_S = 30.0
PCT = 10.0
ALLOWANCE_S = WINDOW_S * PCT / 100.0  # 3.0 s across the whole process


def _pool_config(pct: float = PCT) -> SchedulerConfig:
    """A config carrying a shared budget, built by the pool's own method.

    `EnginePool.__new__` rather than a real pool: what is under test is the
    ownership wiring, and constructing a pool would drag in the model registry
    and the memory enforcer without making the assertion stronger.
    """
    config = SchedulerConfig(
        max_num_seqs=8,
        prefill_step_size=64,
        chunked_prefill=True,
        paged_cache_block_size=256,
        shadow_prefill_enabled=True,
        shadow_prefill_budget_window_s=WINDOW_S,
        shadow_prefill_global_budget_pct=pct,
    )
    pool = EnginePool.__new__(EnginePool)
    pool._scheduler_config = config
    EnginePool.configure_shadow_budget(pool)
    return config


def _engine(config: SchedulerConfig | None = None, label: str = "m") -> Scheduler:
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=config if config is not None else SchedulerConfig(
            max_num_seqs=8,
            prefill_step_size=64,
            chunked_prefill=True,
            paged_cache_block_size=256,
            shadow_prefill_enabled=True,
            shadow_prefill_budget_pct=PCT,
            shadow_prefill_budget_window_s=WINDOW_S,
        ),
    )
    scheduler.block_aware_cache = MagicMock()
    scheduler._unreconstructible_cache_model = False
    mock_bg = MagicMock()
    mock_bg.insert.return_value = [42]
    mock_bg.next_generated.return_value = iter([])
    scheduler.batch_generator = mock_bg
    return scheduler


def _sparse_request(prompt_tokens: int, rid: str, scheduler: Scheduler):
    request = MagicMock()
    request.request_id = rid
    request.prompt_token_ids = list(range(prompt_tokens))
    request.specprefill_indices = [1, 2, 3]
    request._serving_prefix_cache_id = id(scheduler.block_aware_cache)
    return request


def _idle(scheduler: Scheduler) -> None:
    scheduler._shadow_note_step(did_foreground_work=False)
    scheduler._shadow_note_step(did_foreground_work=False)


@pytest.fixture(autouse=True)
def _clean_registries():
    """Both registries are process-global, so a leftover entry would decide
    the next test rather than the code under test."""
    get_decode_activity().clear()
    get_prefill_tracker()._progress.clear()
    yield
    get_decode_activity().clear()
    get_prefill_tracker()._progress.clear()


@pytest.fixture
def two_engines():
    config = _pool_config()
    a = _engine(config, "a")
    b = _engine(config, "b")
    assert a._shadow_budget is b._shadow_budget
    return a, b


class TestOneBudgetForTheProcess:
    """1 — two engines cannot receive twice the cap."""

    def test_the_two_schedulers_hold_the_same_budget_object(self, two_engines):
        a, b = two_engines
        assert a._shadow_budget.shared is True
        assert a._shadow_budget.owner_count() == 2
        assert a._shadow_owner_key != b._shadow_owner_key

    def test_service_on_one_engine_is_charged_against_the_other(self, two_engines):
        a, b = two_engines
        assert a._shadow_budget.allows()
        assert b._shadow_budget.allows()

        a._shadow_budget.note_service(ALLOWANCE_S * 0.75)
        assert b._shadow_budget.allows()      # a quarter of one allowance left

        b._shadow_budget.note_service(ALLOWANCE_S * 0.5)
        # Together they have spent 1.25 allowances. Neither may run again in
        # this window; a per-engine budget would have granted each of them a
        # whole allowance of their own.
        assert not a._shadow_budget.allows()
        assert not b._shadow_budget.allows()
        assert a._shadow_budget.window_service_s == pytest.approx(
            ALLOWANCE_S * 1.25
        )

    def test_the_aggregate_never_exceeds_one_allowance_per_window(self, two_engines):
        """Drive both engines across several windows and total what was
        granted. The charge lands after the grant, so the bound is one
        allowance plus at most the slice that overran it."""
        a, b = two_engines
        budget = a._shadow_budget
        slice_s = 0.4
        granted = 0.0
        for window in range(4):
            budget.window_start_s -= WINDOW_S      # roll one window forward
            for engine in (a, b):
                while engine._shadow_budget.allows():
                    engine._shadow_budget.note_service(slice_s)
                    granted += slice_s
        # The bound comes out tight — exactly one allowance per window plus
        # the single slice that overran the last of them — so it is compared
        # with a tolerance rather than exactly, against float accumulation
        # over twelve additions.
        assert granted <= 4 * ALLOWANCE_S + slice_s + 1e-9


class TestForegroundOnOneEngineBlocksRecoveryOnTheOther:
    """2 — foreground on B blocks new recovery admission on A."""

    def test_a_foreign_decode_withdraws_the_chunk(self, two_engines):
        a, b = two_engines
        a.note_shadow_candidate(_sparse_request(1000, "r1", a))
        _idle(a)
        assert a._shadow_runnable()

        get_decode_activity().publish(b._decode_activity_key, 1)
        assert not a._shadow_runnable()
        assert not a.has_requests()

        get_decode_activity().publish(b._decode_activity_key, 0)
        assert a._shadow_runnable()

    def test_a_foreign_foreground_prefill_withdraws_the_chunk(self, two_engines):
        """The half the decode registry cannot see: an engine that is
        prefilling publishes a decode count of zero, which *removes* its
        registry entry."""
        a, _b = two_engines
        a.note_shadow_candidate(_sparse_request(1000, "r1", a))
        _idle(a)
        assert a._shadow_runnable()

        get_prefill_tracker().update("b-foreground", 100, 8000, "model-b")
        assert not a._shadow_runnable()
        assert not a.has_requests()

        get_prefill_tracker().remove("b-foreground")
        assert a._shadow_runnable()

    def test_a_foreign_recovery_job_is_not_foreground(self, two_engines):
        """And the reason it must not be.

        A recovery job holds its tracker entry from its first chunk until it
        parks, finishes or is dropped — across every wait in between. Read as
        foreground, one engine stood down for another engine's *waiting*,
        indefinitely. Mutual exclusion between recovery jobs is the claim.
        """
        a, b = two_engines
        a.note_shadow_candidate(_sparse_request(1000, "r1", a))
        b.note_shadow_candidate(_sparse_request(1000, "r2", b))
        _idle(a)
        get_prefill_tracker().update(
            b._shadow_request_id(b._shadow_job), 256, 8000, "model-b"
        )
        assert not a._shadow_foreign_engine_busy()


class TestTheFirstSliceIsNotInvisible:
    """3 — no first-slice visibility hole."""

    def test_the_claim_is_held_before_the_state_build(self, two_engines):
        a, b = two_engines
        a.note_shadow_candidate(_sparse_request(1000, "r1", a))
        b.note_shadow_candidate(_sparse_request(1000, "r2", b))
        _idle(a)
        _idle(b)
        assert a._shadow_runnable() and b._shadow_runnable()

        seen = {}

        def _inner():
            # The instant the state build could begin, the peer must already
            # see a process with recovery running. Before the claim this was
            # the whole duration of the first slice.
            seen["b_blocked"] = b._shadow_claim_blocked()
            seen["b_runnable"] = b._shadow_runnable()
            seen["b_has_work"] = b.has_requests()
            return False

        with patch.object(a, "_shadow_step_inner", side_effect=_inner):
            a._shadow_step()

        assert seen["b_blocked"] is True
        assert seen["b_runnable"] is False
        assert seen["b_has_work"] is False

    def test_the_claim_is_released_when_the_slice_ends(self, two_engines):
        a, b = two_engines
        a.note_shadow_candidate(_sparse_request(1000, "r1", a))
        b.note_shadow_candidate(_sparse_request(1000, "r2", b))
        _idle(b)
        with patch.object(a, "_shadow_step_inner", return_value=False):
            a._shadow_step()
        assert not b._shadow_claim_blocked()
        assert b._shadow_runnable()

    def test_a_slice_that_raises_still_releases_the_claim(self, two_engines):
        """Otherwise one failure stops recovery for the whole process."""
        a, b = two_engines
        a.note_shadow_candidate(_sparse_request(1000, "r1", a))
        b.note_shadow_candidate(_sparse_request(1000, "r2", b))
        _idle(b)
        with patch.object(
            a, "_shadow_step_inner", side_effect=RuntimeError("boom")
        ), pytest.raises(RuntimeError):
            a._shadow_step()
        assert not b._shadow_claim_blocked()

    def test_a_stale_claim_expires(self, two_engines):
        """A holder that dies mid-slice must not hold it for the process's
        life. The backstop the decode registry uses, for the same reason."""
        a, b = two_engines
        budget = a._shadow_budget
        assert budget.try_claim(a._shadow_owner_key)
        assert not budget.try_claim(b._shadow_owner_key)
        budget.claim_at_s -= budget.claim_ttl_s + 1
        assert budget.try_claim(b._shadow_owner_key)


class TestAnExhaustedBudgetParksEveryEngine:
    """4 — no starvation or spin while the shared window is spent."""

    def test_neither_engine_steps_while_the_window_is_spent(self, two_engines):
        a, b = two_engines
        a.note_shadow_candidate(_sparse_request(1000, "r1", a))
        b.note_shadow_candidate(_sparse_request(1000, "r2", b))
        assert a.has_requests() and b.has_requests()

        a._shadow_budget.note_service(WINDOW_S)     # far past one allowance
        assert not a.has_requests()
        assert not b.has_requests()

        steps = 0
        for _ in range(60):
            for engine in (a, b):
                if engine.has_requests():
                    engine.step()
                    steps += 1
        assert steps == 0
        assert a._shadow_job is not None and b._shadow_job is not None

    def test_both_wake_when_the_window_replenishes(self, two_engines):
        a, b = two_engines
        a.note_shadow_candidate(_sparse_request(1000, "r1", a))
        b.note_shadow_candidate(_sparse_request(1000, "r2", b))
        budget = a._shadow_budget
        budget.note_service(WINDOW_S)
        assert not a.has_requests()

        budget.window_start_s -= WINDOW_S    # one window of carried lockout
        assert not a.has_requests()
        budget.window_start_s -= WINDOW_S
        assert a.has_requests() and b.has_requests()

    def test_waiting_on_a_peer_is_not_a_stall(self, two_engines):
        """The deadline that drops a job which may run and cannot must not
        fire on a job that is waiting for a reason that ends."""
        a, b = two_engines
        a.note_shadow_candidate(_sparse_request(1000, "r1", a))
        get_prefill_tracker().update("b-foreground", 100, 8000, "model-b")
        for _ in range(200):
            a._shadow_after_step(MagicMock(has_work=False))
        assert a._shadow_job is not None
        assert a._shadow_blocked_idle_steps == 0


class TestOneOwnersResetSparesTheRest:
    """5 — resetting B does not alter A's or the global spent allowance."""

    def test_resetting_one_scheduler_leaves_the_shared_accounting_alone(
        self, two_engines
    ):
        a, b = two_engines
        budget = a._shadow_budget
        a._shadow_budget.note_service(ALLOWANCE_S * 1.5)   # A overran
        spent = budget.window_service_s
        overshoot = budget.overshoot_s
        window_start = budget.window_start_s
        lifetime = budget.service_s
        assert overshoot > 0

        b.reset()

        assert budget.window_service_s == spent
        assert budget.overshoot_s == overshoot
        assert budget.window_start_s == window_start
        assert budget.service_s == lifetime
        assert not budget.allows()

    def test_a_shared_budget_refuses_a_direct_reset(self, two_engines):
        a, _b = two_engines
        with pytest.raises(RuntimeError, match="shared recovery budget"):
            a._shadow_budget.reset()

    def test_a_private_budget_still_resets(self):
        """The bare-Scheduler path keeps the semantics it always had."""
        solo = _engine()
        assert solo._shadow_budget.shared is False
        solo._shadow_budget.note_service(1.0)
        solo.reset()
        assert solo._shadow_budget.service_s == 0.0


class TestReloadBuysNothing:
    """6 — unload and reload get no fresh allowance and no forgiven debt."""

    def test_a_returning_owner_finds_the_window_where_it_left_it(self, two_engines):
        a, b = two_engines
        budget = a._shadow_budget
        budget.note_service(ALLOWANCE_S * 1.5)
        spent = budget.window_service_s
        overshoot = budget.overshoot_s
        window_start = budget.window_start_s
        assert not budget.allows()

        # unload
        budget.deregister(a._shadow_owner_key)
        assert budget.owner_count() == 1
        assert budget.window_service_s == spent
        assert budget.overshoot_s == overshoot

        # reload
        budget.register(a._shadow_owner_key)
        assert budget.owner_count() == 2
        assert budget.window_service_s == spent
        assert budget.overshoot_s == overshoot
        assert budget.window_start_s == window_start
        assert not budget.allows()

    def test_reconfiguring_the_cap_keeps_the_window_and_the_debt(self):
        """A settings change must not hand every engine a clean slate either."""
        config = _pool_config(pct=PCT)
        budget = config.shadow_budget
        budget.note_service(ALLOWANCE_S * 1.5)
        spent, overshoot = budget.window_service_s, budget.overshoot_s

        config.shadow_prefill_global_budget_pct = 20.0
        pool = EnginePool.__new__(EnginePool)
        pool._scheduler_config = config
        EnginePool.configure_shadow_budget(pool)

        assert config.shadow_budget is budget
        assert budget.pct == 20.0
        assert budget.window_service_s == spent
        assert budget.overshoot_s == overshoot

    def test_a_departing_owner_releases_its_claim(self, two_engines):
        a, b = two_engines
        budget = a._shadow_budget
        assert budget.try_claim(a._shadow_owner_key)
        budget.deregister(a._shadow_owner_key)
        assert budget.try_claim(b._shadow_owner_key)


class TestABareSchedulerStillRecovers:
    """No pool means no peers to share an accelerator with, so a private
    budget is the same invariant rather than a degraded one."""

    def test_a_scheduler_without_a_pool_gets_its_own_budget(self):
        solo = _engine()
        assert solo._shadow_budget.shared is False
        assert solo._shadow_budget.pct == PCT
        assert solo._shadow_budget.owner_count() == 1

    def test_two_bare_schedulers_do_not_share(self):
        first, second = _engine(), _engine()
        assert first._shadow_budget is not second._shadow_budget

    def test_recovery_still_runs_on_a_bare_scheduler(self):
        solo = _engine()
        solo.note_shadow_candidate(_sparse_request(1000, "r1", solo))
        _idle(solo)
        assert solo._shadow_runnable()
        assert solo.has_requests()
