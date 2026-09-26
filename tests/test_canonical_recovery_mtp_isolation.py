# SPDX-License-Identifier: Apache-2.0
"""Canonical recovery reconstructs target state, and nothing else.

Lightning-MTP keeps a small prompt-history sidecar per request, owned through
``prompt_priming``: ``prepare_prefix_context`` builds a plan and files it under
the request id, each prefill chunk re-activates it, the forward folds hidden
states into it, and ``release_request`` gives it back when the request ends.

A recovery job is a synthetic request that never decodes, never reads that
history and never finishes through the ordinary finish path. So on a model with
both SpecPrefill and MTP enabled it would create priming state that nothing
consumes, run head-history folding nobody asked for, and file the result under
an id no release path visits.

The contract these tests pin:

    canonical-state recovery must not create, capture, publish or retain
    request-owned MTP prompt-priming state, and foreground priming must be
    unchanged.

Three separate mechanisms have to hold, because each covers a hole the others
do not: the sidecar is never *prepared* for a recovery request, capture is
*suppressed* for the duration of a recovery slice, and every exit path
*releases* the synthetic id anyway — defence in depth for a model that acquired
ownership by some route this file did not think of.
"""

from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest

from omlx.patches.mlx_lm_mtp import prompt_priming
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig

BLOCK = 256
RID = "canonical-recovery:r1"


def _mtp_model() -> MagicMock:
    """A model double that ``prompt_priming`` accepts as an eligible host.

    ``_host_eligible`` wants all three of these, and ``_owned`` refuses a host
    with DSpark decode on. A bare ``MagicMock`` answers every one of those
    ``getattr`` calls with a truthy child mock, so leaving any of them implicit
    would make the fixture lie in whichever direction the test did not check.
    """
    model = MagicMock()
    model.layers = []
    model._omlx_mtp_decode_enabled = True
    model._omlx_mtp_chain = True
    model._omlx_dspark_decode_enabled = False
    model.mtp = MagicMock(name="mtp-head")
    # ``_host_candidates`` walks these before settling on the model itself.
    model.language_model = None
    model._language_model = None
    return model


def _make_scheduler(**config_over) -> Scheduler:
    config_kwargs = dict(
        max_num_seqs=8,
        prefill_step_size=64,
        chunked_prefill=True,
        paged_cache_block_size=BLOCK,
        canonical_state_recovery_enabled=True,
        canonical_state_recovery_global_budget_pct=10.0,
    )
    config_kwargs.update(config_over)
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=_mtp_model(),
        tokenizer=tokenizer,
        config=SchedulerConfig(**config_kwargs),
    )
    scheduler.block_aware_cache = MagicMock()
    scheduler._unreconstructible_cache_model = False
    return scheduler


@pytest.fixture(autouse=True)
def _clean_priming_slot():
    """No owned priming state may leak between tests through the module."""
    yield
    prompt_priming._SUPPRESS.value = False


def _sparse_request(prompt_tokens: int, rid: str = "r1", scheduler=None):
    request = MagicMock()
    request.request_id = rid
    request.prompt_token_ids = list(range(prompt_tokens))
    request.specprefill_indices = [1, 2, 3]
    request._serving_prefix_cache_id = (
        id(scheduler.block_aware_cache) if scheduler is not None else None
    )
    return request


def _queued(scheduler, prompt_tokens: int = 1000):
    scheduler.note_canonical_recovery_candidate(
        _sparse_request(prompt_tokens, scheduler=scheduler)
    )
    job = scheduler._canonical_recovery_job
    assert job is not None
    return job


def _own(scheduler, request_id: str) -> dict:
    """File a priming record under *request_id* and return the owning dict."""
    _, state = prompt_priming._owned(scheduler.model, create=True)
    assert state is not None, "the model double is not an eligible priming host"
    state.requests[request_id] = (None, MagicMock(name="prime-plan"))
    return state.requests


# --------------------------------------------------------------------------
# 1. The sidecar is never prepared for a recovery request
# --------------------------------------------------------------------------


@contextmanager
def _bare_prefix_path(scheduler):
    """Reach the MTP block in ``_prepare_prefix_cache_for_request``.

    With no paged cache the method takes its shortest route — every token is
    uncached — which is the route that still ends at the priming hook.
    """
    scheduler.block_aware_cache = None
    scheduler.paged_cache_manager = None
    with patch.object(
        prompt_priming, "prepare_prefix_context"
    ) as prepare:
        yield prepare


def _plain_request(rid: str, *, recovery: bool) -> Request:
    request = Request(
        request_id=rid,
        prompt=None,
        prompt_token_ids=list(range(64)),
        sampling_params=SamplingParams(max_tokens=1),
    )
    request.is_canonical_recovery = recovery
    return request


class TestTheSidecarIsNeverPreparedForRecovery:
    def test_a_recovery_request_does_not_prepare_mtp_prefix_context(self):
        scheduler = _make_scheduler()
        with _bare_prefix_path(scheduler) as prepare:
            scheduler._prepare_prefix_cache_for_request(
                _plain_request(RID, recovery=True)
            )
        prepare.assert_not_called()

    def test_a_foreground_request_still_prepares_mtp_prefix_context(self):
        """The control. Without it the fix above could be a global disable."""
        scheduler = _make_scheduler()
        with _bare_prefix_path(scheduler) as prepare:
            scheduler._prepare_prefix_cache_for_request(
                _plain_request("fg-1", recovery=False)
            )
        prepare.assert_called_once()
        assert prepare.call_args.kwargs["request_id"] == "fg-1"


# --------------------------------------------------------------------------
# 2. Capture is suppressed for the whole recovery slice
# --------------------------------------------------------------------------


@contextmanager
def _live_job(scheduler, *, chunk):
    """A job with state already built, so the step goes straight to *chunk*."""
    job = _queued(scheduler)
    job.prefill_state = MagicMock(
        base_size=0,
        tokens_processed=0,
        cache=[MagicMock()],
        canonical_recovery_target_tokens=job.target_tokens,
    )
    tracker = MagicMock()
    tracker.any_active.return_value = False
    tracker.recently_active.return_value = False
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(scheduler, "_step_prefill_chunk", side_effect=chunk)
        )
        stack.enter_context(
            patch("omlx.scheduler.get_prefill_tracker", return_value=tracker)
        )
        yield job


class TestRecoveryForwardsDoNotCapture:
    def test_the_chunk_forward_runs_with_capture_suppressed(self):
        seen = []

        def chunk(_state):
            seen.append(prompt_priming._suppressed())
            return False

        scheduler = _make_scheduler()
        with _live_job(scheduler, chunk=chunk):
            scheduler._canonical_recovery_step()

        assert seen == [True], (
            "a recovery chunk forward reached the model with MTP prompt "
            "capture still armed"
        )

    def test_suppression_is_lifted_when_the_step_returns(self):
        scheduler = _make_scheduler()
        with _live_job(scheduler, chunk=lambda _state: False):
            scheduler._canonical_recovery_step()
        assert prompt_priming._suppressed() is False

    def test_suppression_is_lifted_when_the_chunk_fails(self):
        """A failing chunk is caught inside the step; suppression must not
        outlive it and silence the next foreground prefill."""
        scheduler = _make_scheduler()
        with _live_job(scheduler, chunk=MagicMock(side_effect=RuntimeError("boom"))):
            scheduler._canonical_recovery_step()
        assert prompt_priming._suppressed() is False

    def test_foreground_prefill_is_not_suppressed(self):
        """The control: suppression is scoped to the recovery slice."""
        scheduler = _make_scheduler()
        with _live_job(scheduler, chunk=lambda _state: False):
            scheduler._canonical_recovery_step()
            assert prompt_priming._suppressed() is False


# --------------------------------------------------------------------------
# 3. Every exit path releases the synthetic request id
# --------------------------------------------------------------------------


def _exit_paths():
    """Each way a recovery job can stop, as (name, callable(scheduler, job))."""

    def finish(scheduler, job):
        scheduler._canonical_recovery_finish(job)

    def park(scheduler, job):
        scheduler._canonical_recovery_park_job(job, "nothing_to_do")

    def retire(scheduler, job):
        scheduler._canonical_recovery_retire_state(job)

    def drop_failure(scheduler, job):
        scheduler._canonical_recovery_drop_job("chunk_failed")

    def drop_replacement(scheduler, job):
        scheduler._canonical_recovery_drop_job("replaced")

    def cancel(scheduler, job):
        scheduler.cancel_canonical_recovery_work("unload")

    return [
        ("finish", finish),
        ("park", park),
        ("retire", retire),
        ("drop_failure", drop_failure),
        ("drop_replacement", drop_replacement),
        ("cancel", cancel),
    ]


class TestTheSyntheticRequestIsAlwaysReleased:
    @pytest.mark.parametrize("name,exit_path", _exit_paths(), ids=lambda v: v)
    def test_no_priming_ownership_survives(self, name, exit_path):
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        rid = scheduler._canonical_recovery_request_id(job)
        owned = _own(scheduler, rid)
        assert rid in owned

        tracker = MagicMock()
        tracker.any_active.return_value = False
        tracker.recently_active.return_value = False
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(scheduler, "_release_paged_cache_for_request")
            )
            stack.enter_context(
                patch.object(scheduler, "_drop_boundary_snapshots_for_request")
            )
            stack.enter_context(
                patch("omlx.scheduler.get_prefill_tracker", return_value=tracker)
            )
            exit_path(scheduler, job)

        assert rid not in owned, (
            f"the {name} path left MTP priming state owned by the synthetic "
            "recovery request"
        )

    def test_a_foreground_request_keeps_its_priming_across_a_recovery_exit(self):
        """The control: releasing the synthetic id releases only that id."""
        scheduler = _make_scheduler()
        job = _queued(scheduler)
        rid = scheduler._canonical_recovery_request_id(job)
        owned = _own(scheduler, rid)
        _own(scheduler, "fg-1")

        tracker = MagicMock()
        tracker.any_active.return_value = False
        tracker.recently_active.return_value = False
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(scheduler, "_release_paged_cache_for_request")
            )
            stack.enter_context(
                patch.object(scheduler, "_drop_boundary_snapshots_for_request")
            )
            stack.enter_context(
                patch("omlx.scheduler.get_prefill_tracker", return_value=tracker)
            )
            scheduler._canonical_recovery_drop_job("chunk_failed")

        assert rid not in owned
        assert "fg-1" in owned

