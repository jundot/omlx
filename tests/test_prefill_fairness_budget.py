"""Regressions for sharing decode time between consecutive prefill forwards."""

from itertools import count
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.decode_activity import get_decode_activity
from omlx.prefill_progress import get_prefill_tracker
from omlx.request import Request, SamplingParams
from omlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    _PrefillAbortedError,
    _PrefillState,
)


@pytest.fixture(autouse=True)
def isolated_activity():
    get_decode_activity().clear()
    get_prefill_tracker().clear()
    with patch("omlx.scheduler._sync_and_clear_cache"):
        yield
    get_decode_activity().clear()
    get_prefill_tracker().clear()


@pytest.fixture
def clock(monkeypatch):
    current = SimpleNamespace(now=100.0)
    monkeypatch.setattr("omlx.scheduler.time.perf_counter", lambda: current.now)
    return current


def make_scheduler(**settings) -> Scheduler:
    model = MagicMock()
    del model._omlx_prefill
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    config = SchedulerConfig(
        **{
            "max_num_seqs": 8,
            "prefill_step_size": 4,
            "chunked_prefill": True,
            "paged_cache_block_size": 0,
            **settings,
        }
    )
    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)
    batch_generator = MagicMock()
    next_uid = count(42)
    batch_generator.insert.side_effect = lambda *args, **kwargs: [next(next_uid)]
    batch_generator.next_generated.return_value = iter([])
    scheduler.batch_generator = batch_generator
    scheduler._current_sampler_params = ()
    return scheduler


def make_request(request_id: str, token_count: int = 20) -> Request:
    tokens = list(range(token_count))
    request = Request(
        request_id=request_id,
        prompt=tokens,
        sampling_params=SamplingParams(max_tokens=32),
    )
    request.prompt_token_ids = tokens
    request.num_prompt_tokens = len(tokens)
    request.remaining_tokens = tokens
    return request


def add_prefill(scheduler: Scheduler, request_id: str) -> Request:
    request = make_request(request_id)
    scheduler.requests[request_id] = request
    scheduler.prefilling.append(request)
    scheduler._prefill_states[request_id] = _PrefillState(
        request=request,
        cache=[],
        tokens_remaining=mx.zeros((1, request.num_prompt_tokens - 1), dtype=mx.int32),
        last_token=request.prompt_token_ids[-1:],
        tokens_processed=0,
        base_size=0,
        emitted_boundaries={},
        boundary_enabled=False,
        block_size=0,
        total_length=request.num_prompt_tokens,
        sampler=MagicMock(),
        sm=MagicMock(),
        per_row_lps=[],
    )
    return request


def prefill_ids(scheduler: Scheduler) -> list[str]:
    return [request.request_id for request in scheduler.prefilling]


def test_each_forward_rechecks_debt_and_rotates_without_starvation():
    scheduler = make_scheduler()
    scheduler.running["decoder"] = make_request("decoder")
    request_ids = ["prefill-a", "prefill-b", "prefill-c"]
    for request_id in request_ids:
        add_prefill(scheduler, request_id)
    advanced = []

    def advance(state):
        advanced.append(state.request.request_id)
        scheduler._accrue_decode_debt(0.5)
        return False

    with patch.object(scheduler, "_step_prefill_chunk", side_effect=advance):
        for turn, request_id in enumerate(request_ids * 2):
            scheduler._advance_chunked_prefills([], [])
            assert advanced == (request_ids * 2)[: turn + 1]
            assert prefill_ids(scheduler)[-1] == request_id
            assert scheduler._decode_time_owed_s > 0
            scheduler._advance_chunked_prefills([], [])
            assert len(advanced) == turn + 1
            scheduler._repay_decode_debt(scheduler._decode_time_owed_s)

    assert prefill_ids(scheduler) == request_ids


def test_shared_hold_from_another_prefiller_interrupts_current_loop(clock):
    scheduler = make_scheduler()
    other_prefiller = make_scheduler()
    get_decode_activity().publish("decoding-engine", 1)
    for request_id in ("prefill-a", "prefill-b", "prefill-c"):
        add_prefill(scheduler, request_id)
    advanced = []

    def advance(state):
        advanced.append(state.request.request_id)
        other_prefiller._accrue_decode_debt(0.5)
        return False

    with patch.object(scheduler, "_step_prefill_chunk", side_effect=advance):
        scheduler._advance_chunked_prefills([], [])
        assert advanced == ["prefill-a"]
        assert scheduler._prefill_hold_until == 0
        assert prefill_ids(scheduler) == ["prefill-b", "prefill-c", "prefill-a"]
        scheduler._advance_chunked_prefills([], [])
        assert advanced == ["prefill-a"]
        clock.now = get_decode_activity().hold_until()
        scheduler._advance_chunked_prefills([], [])

    assert advanced == ["prefill-a", "prefill-b"]
    assert prefill_ids(scheduler) == ["prefill-c", "prefill-a", "prefill-b"]


@pytest.mark.parametrize(
    ("decode_fairness", "has_decoder"), [(True, False), (False, True)]
)
def test_uncontended_or_disabled_fairness_keeps_advancing_all_requests(
    decode_fairness, has_decoder
):
    scheduler = make_scheduler(decode_fairness=decode_fairness)
    if has_decoder:
        scheduler.running["decoder"] = make_request("decoder")
    request_ids = ["prefill-a", "prefill-b", "prefill-c"]
    for request_id in request_ids:
        add_prefill(scheduler, request_id)
    advanced = []

    def advance(state):
        advanced.append(state.request.request_id)
        scheduler._accrue_decode_debt(0.5)
        return False

    with patch.object(scheduler, "_step_prefill_chunk", side_effect=advance):
        scheduler._advance_chunked_prefills([], [])

    assert advanced == request_ids
    assert prefill_ids(scheduler) == request_ids
    assert scheduler._decode_time_owed_s == 0


def test_completed_request_creates_contention_for_remaining_prefills():
    scheduler = make_scheduler()
    completed = add_prefill(scheduler, "completed")
    add_prefill(scheduler, "prefill-a")
    add_prefill(scheduler, "prefill-b")
    scheduled = []
    rejected = []
    advanced = []

    def advance(state):
        advanced.append(state.request.request_id)
        scheduler._accrue_decode_debt(0.5)
        return state.request is completed

    with patch.object(scheduler, "_step_prefill_chunk", side_effect=advance):
        scheduler._advance_chunked_prefills(scheduled, rejected)

    assert advanced == ["completed", "prefill-a"]
    assert scheduled == [completed]
    assert rejected == []
    assert scheduler.running == {"completed": completed}
    assert "completed" not in scheduler._prefill_states
    assert prefill_ids(scheduler) == ["prefill-b", "prefill-a"]


def test_missing_and_aborted_rows_do_not_discard_waiting_prefills():
    scheduler = make_scheduler()
    scheduler.running["decoder"] = make_request("decoder")
    scheduler.prefilling.append(make_request("missing"))
    add_prefill(scheduler, "aborted")
    add_prefill(scheduler, "prefill-a")
    add_prefill(scheduler, "prefill-b")
    advanced = []

    def advance(state):
        advanced.append(state.request.request_id)
        scheduler._accrue_decode_debt(0.5)
        if state.request.request_id == "aborted":
            raise _PrefillAbortedError([], 4)
        return False

    with patch.object(scheduler, "_step_prefill_chunk", side_effect=advance):
        scheduler._advance_chunked_prefills([], [])
        assert advanced == ["aborted"]
        assert prefill_ids(scheduler) == ["prefill-a", "prefill-b"]
        assert "aborted" not in scheduler._prefill_states
        scheduler._repay_decode_debt(scheduler._decode_time_owed_s)
        scheduler._advance_chunked_prefills([], [])

    assert advanced == ["aborted", "prefill-a"]
    assert prefill_ids(scheduler) == ["prefill-b", "prefill-a"]


def test_inflight_chunk_debt_defers_new_admission_until_decode_runs():
    scheduler = make_scheduler()
    scheduler.running["decoder"] = make_request("decoder")
    add_prefill(scheduler, "inflight-a")
    add_prefill(scheduler, "inflight-b")
    waiting = make_request("waiting")
    scheduler.add_request(waiting)
    events = []

    def advance(state):
        events.append(state.request.request_id)
        scheduler._accrue_decode_debt(0.5)
        return False

    def decode():
        events.append("decode")
        return iter([])

    scheduler.batch_generator.next_generated.side_effect = decode
    with (
        patch.object(scheduler, "_step_prefill_chunk", side_effect=advance),
        patch.object(scheduler, "_begin_prefill") as begin,
        patch.object(scheduler, "_do_external_prefill") as external,
    ):
        output = scheduler.step()

    assert events == ["inflight-a", "decode"]
    assert list(scheduler.waiting) == [waiting]
    assert output.has_work
    begin.assert_not_called()
    external.assert_not_called()


@pytest.mark.parametrize("ane_prefill", [False, True])
def test_short_external_prefills_share_admission_debt(clock, ane_prefill):
    scheduler = make_scheduler()
    scheduler.running["decoder"] = make_request("decoder")
    requests = [make_request(request_id, 3) for request_id in ("short-a", "short-b")]
    for request in requests:
        scheduler.add_request(request)
    forwarded = []

    def forward(tokens, **kwargs):
        forwarded.append(tokens.shape[1])
        clock.now += 0.5

    scheduler.model.side_effect = forward
    if ane_prefill:
        # The K2 ANE hook's signature: no extra keyword arguments.
        def ane_forward(tokens, *, cache):
            forward(tokens)

        scheduler.model._omlx_prefill = ane_forward
    with patch("omlx.scheduler.make_prompt_cache", return_value=[]):
        scheduled, rejected = scheduler._schedule_waiting()
        assert scheduled == [requests[0]]
        assert rejected == []
        assert list(scheduler.waiting) == [requests[1]]
        assert forwarded == [2]
        scheduler._repay_decode_debt(scheduler._decode_time_owed_s)
        scheduled, rejected = scheduler._schedule_waiting()

    assert scheduled == [requests[1]]
    assert rejected == []
    assert not scheduler.waiting
    assert forwarded == [2, 2]


@pytest.mark.parametrize("other_engine", [False, True])
def test_waiting_and_inflight_prefills_both_progress_under_decode_load(
    clock, other_engine
):
    scheduler = make_scheduler()
    if other_engine:
        get_decode_activity().publish("other-decoder", 1)
    else:
        scheduler.running["decoder"] = make_request("decoder")
    add_prefill(scheduler, "inflight")
    for index in range(3):
        scheduler.add_request(make_request(f"waiting-{index}", 3))
    events = []

    def advance(state):
        events.append("inflight")
        clock.now += 0.5
        scheduler._accrue_decode_debt(0.5)
        return False

    def forward(tokens, **kwargs):
        events.append("waiting")
        clock.now += 0.5

    def decode():
        clock.now += 0.125
        return iter([])

    scheduler.model.side_effect = forward
    scheduler.batch_generator.next_generated.side_effect = decode
    with (
        patch.object(scheduler, "_step_prefill_chunk", side_effect=advance),
        patch("omlx.scheduler.make_prompt_cache", return_value=[]),
    ):
        for _ in range(24):
            scheduler.step()
            clock.now += 0.125
            if not scheduler.waiting:
                break

    assert not scheduler.waiting
    assert events == ["inflight", "waiting"] * 3
    assert "inflight" in scheduler._prefill_states
    assert all(f"waiting-{index}" in scheduler.running for index in range(3))


@pytest.mark.parametrize("blocker", ["capacity", "memory", "store", "freshness"])
def test_admission_turn_keeps_guards_and_allows_inflight_progress(clock, blocker):
    scheduler = make_scheduler()
    scheduler.running["decoder"] = make_request("decoder")
    add_prefill(scheduler, "inflight")
    waiting = make_request("waiting", 3)
    scheduler.add_request(waiting)
    advanced = []

    def advance(state):
        advanced.append(state.request.request_id)
        scheduler._accrue_decode_debt(0.5)
        return False

    with patch.object(scheduler, "_step_prefill_chunk", side_effect=advance):
        scheduler.step()
        scheduler._repay_decode_debt(scheduler._decode_time_owed_s)
        if blocker == "capacity":
            scheduler.config.max_num_seqs = 2
        elif blocker == "memory":
            scheduler._admission_paused = True
        elif blocker == "store":
            scheduler._store_cache_gate = SimpleNamespace(
                has_capacity=False, in_flight=2, cap=2
            )
        else:
            scheduler._should_defer_for_cache_freshness = lambda request: True
        scheduler.step()

    assert advanced == ["inflight", "inflight"]
    assert list(scheduler.waiting) == [waiting]


def test_cross_engine_admission_chunk_reports_work_without_advancing_twice(clock):
    scheduler = make_scheduler()
    get_decode_activity().publish("other-decoder", 1)
    add_prefill(scheduler, "inflight")
    scheduler.add_request(make_request("waiting"))
    advanced = []

    def advance(state):
        advanced.append(state.request.request_id)
        scheduler._accrue_decode_debt(0.5)
        return False

    with (
        patch.object(scheduler, "_step_prefill_chunk", side_effect=advance),
        patch("omlx.scheduler.make_prompt_cache", return_value=[]),
    ):
        scheduler.step()
        clock.now = get_decode_activity().hold_until()
        output = scheduler.step()

    assert advanced == ["inflight", "waiting"]
    assert output.has_work
    assert prefill_ids(scheduler) == ["inflight", "waiting"]
    assert not scheduler.running
    assert not scheduler.waiting


def test_cancel_after_admission_turn_preserves_inflight_prefill(clock):
    scheduler = make_scheduler()
    scheduler.running["decoder"] = make_request("decoder")
    add_prefill(scheduler, "inflight")
    scheduler.add_request(make_request("cancelled"))
    advanced = []

    def advance(state):
        advanced.append(state.request.request_id)
        scheduler._accrue_decode_debt(0.5)
        return False

    with (
        patch.object(scheduler, "_step_prefill_chunk", side_effect=advance),
        patch("omlx.scheduler.make_prompt_cache", return_value=[]),
    ):
        scheduler.step()
        scheduler._repay_decode_debt(scheduler._decode_time_owed_s)
        scheduler.step()
        assert prefill_ids(scheduler) == ["inflight", "cancelled"]
        scheduler.abort_request("cancelled")
        scheduler._repay_decode_debt(scheduler._decode_time_owed_s)
        scheduler.step()

    assert advanced == ["inflight", "cancelled", "inflight"]
    assert "cancelled" not in scheduler.requests
    assert "cancelled" not in scheduler._prefill_states
    assert prefill_ids(scheduler) == ["inflight"]
    assert "decoder" in scheduler.running


def test_uncontended_short_request_is_admitted_before_inflight_chunk(clock):
    """With nothing decoding, a short request that arrives mid-prefill is
    admitted before the in-flight chunk, which then shrinks to share the GPU
    with its decode instead of running at full width first (#5148)."""
    scheduler = make_scheduler(prefill_step_size=4096)
    add_prefill(scheduler, "inflight")
    waiting = make_request("waiting", 3)
    scheduler.add_request(waiting)
    events = []
    chunk_sizes = []

    def advance(state):
        events.append("inflight")
        chunk_sizes.append(
            scheduler._prefill_step_size_for_progress(state.tokens_processed, 8192)
        )
        clock.now += 0.5
        scheduler._accrue_decode_debt(0.5)
        return False

    def forward(tokens, **kwargs):
        events.append("waiting")
        clock.now += 0.01

    scheduler.model.side_effect = forward
    with (
        patch.object(scheduler, "_step_prefill_chunk", side_effect=advance),
        patch("omlx.scheduler.make_prompt_cache", return_value=[]),
    ):
        scheduler.step()
        assert events == ["waiting", "inflight"]
        assert chunk_sizes[0] < 4096
        assert "waiting" in scheduler.running
        assert not scheduler.waiting
        # The in-flight prefill keeps advancing once the debt is repaid.
        scheduler._repay_decode_debt(scheduler._decode_time_owed_s)
        scheduler.step()

    assert events == ["waiting", "inflight", "inflight"]
    assert prefill_ids(scheduler) == ["inflight"]


def test_uncontended_long_request_runs_one_chunk_in_its_admission_step(clock):
    """A long request admitted ahead of the in-flight chunk already ran its
    first chunk, so that step does not advance it a second time."""
    scheduler = make_scheduler()
    add_prefill(scheduler, "inflight")
    scheduler.add_request(make_request("long", 20))
    events = []

    def advance(state):
        events.append(state.request.request_id)
        state.tokens_processed += 4
        clock.now += 0.5
        return False

    with (
        patch.object(scheduler, "_step_prefill_chunk", side_effect=advance),
        patch("omlx.scheduler.make_prompt_cache", return_value=[]),
    ):
        scheduler.step()
        assert events == ["long", "inflight"]
        assert prefill_ids(scheduler) == ["inflight", "long"]
        scheduler.step()

    assert events == ["long", "inflight", "inflight", "long"]
    assert prefill_ids(scheduler) == ["inflight", "long"]


def test_disabled_fairness_keeps_chunk_before_admission(clock):
    """Without decode fairness the chunk does not shrink, so admitting first
    would gain nothing; the order stays chunk first."""
    scheduler = make_scheduler(decode_fairness=False)
    add_prefill(scheduler, "inflight")
    scheduler.add_request(make_request("waiting", 3))
    events = []

    def advance(state):
        events.append("inflight")
        return False

    scheduler.model.side_effect = lambda tokens, **kwargs: events.append("waiting")
    with (
        patch.object(scheduler, "_step_prefill_chunk", side_effect=advance),
        patch("omlx.scheduler.make_prompt_cache", return_value=[]),
    ):
        scheduler.step()

    assert events == ["inflight", "waiting"]
    assert "waiting" in scheduler.running


def run_long_admission_step(scheduler, during_inflight=None, inflight_error=None):
    """One uncontended step that admits "long" ahead of "inflight"."""

    def advance(state):
        if state.request.request_id == "inflight":
            if during_inflight is not None:
                during_inflight()
            if inflight_error is not None:
                raise inflight_error
        state.tokens_processed += 4
        return False

    with (
        patch.object(scheduler, "_step_prefill_chunk", side_effect=advance),
        patch("omlx.scheduler.make_prompt_cache", return_value=[]),
    ):
        return scheduler.step()


def test_held_request_stays_queued_during_inflight_chunk(clock):
    """Other threads read the queues mid-step (memory enforcer, stats), so the
    admitted request stays in prefilling while the in-flight chunk runs."""
    scheduler = make_scheduler()
    add_prefill(scheduler, "inflight")
    scheduler.add_request(make_request("long", 20))
    seen = {}

    def during_inflight():
        seen["prefilling"] = prefill_ids(scheduler)
        seen["num_prefilling"] = scheduler.get_stats()["num_prefilling"]

    run_long_admission_step(scheduler, during_inflight)

    assert seen == {"prefilling": ["inflight", "long"], "num_prefilling": 2}


def test_inflight_chunk_error_keeps_held_request(clock):
    scheduler = make_scheduler()
    add_prefill(scheduler, "inflight")
    scheduler.add_request(make_request("long", 20))

    with pytest.raises(KeyError):
        run_long_admission_step(scheduler, inflight_error=KeyError("boom"))

    assert "long" in prefill_ids(scheduler)
    assert "long" in scheduler._prefill_states


def test_held_request_is_not_packed_into_inflight_forward(clock):
    """A held request cannot ride along as a packed companion either."""
    scheduler = make_scheduler()
    add_prefill(scheduler, "inflight")
    long_request = make_request("long", 20)
    # Unpacked admission, so its first chunk runs while it is admitted.
    long_request.packed_prefill_excluded = True
    scheduler.add_request(long_request)
    companions = []

    def plan_rows(state, later):
        companions.append([request.request_id for request in later])
        return []

    with (
        patch.object(Scheduler, "_packed_prefill_ready", return_value=True),
        patch.object(Scheduler, "_packing_fits_contended_cap", return_value=True),
        patch.object(scheduler, "_plan_prefill_rows", side_effect=plan_rows),
    ):
        run_long_admission_step(scheduler)

    assert companions == [[]]
    assert "long" in prefill_ids(scheduler)


def test_step_reports_whether_a_prefill_chunk_ran(clock):
    """EngineCore ends a burst only after a step that ran a chunk; a step that
    held the prefill for decode reports none."""
    scheduler = make_scheduler()
    add_prefill(scheduler, "inflight")

    def advance(state):
        scheduler._prefill_forwards += 1
        return False

    with patch.object(scheduler, "_step_prefill_chunk", side_effect=advance):
        assert scheduler.step().ran_prefill_chunk
        scheduler.running["decoder"] = make_request("decoder")
        scheduler._accrue_decode_debt(0.5)
        assert not scheduler.step().ran_prefill_chunk
