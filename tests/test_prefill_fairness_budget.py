"""Regressions for sharing decode time between consecutive prefill forwards."""

from contextlib import ExitStack
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
    _PrefillChunkPlan,
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


def fake_chunk_forwards(forwards, clock, seconds_per_token=0.001):
    """Replace chunk planning and forwards with token bookkeeping."""

    def plan(self, state, *, limit=None, guarded=True):
        remaining = state.tokens_remaining.shape[1]
        step = self._prefill_step_size_for_progress(state.tokens_processed, remaining)
        tokens = min(step, remaining) if limit is None else min(step, remaining, limit)
        return _PrefillChunkPlan(
            state=state,
            tokens=tokens,
            requested_step=step,
            gathered_core=False,
            qwen4_accounting=False,
            throttled=False,
            started=clock.now,
        )

    def run(self, plans):
        self._prefill_forwards = getattr(self, "_prefill_forwards", 0) + 1
        forwards.append([(plan.state.request.request_id, plan.tokens) for plan in plans])
        done = []
        for plan in plans:
            state = plan.state
            left = state.tokens_remaining.shape[1] - plan.tokens
            state.tokens_remaining = mx.zeros((1, left), dtype=mx.int32)
            state.tokens_processed += plan.tokens
            done.append(left == 0)
        seconds = sum(plan.tokens for plan in plans) * seconds_per_token
        clock.now += seconds
        self._accrue_decode_debt(seconds)
        return done

    return (
        patch.object(Scheduler, "_plan_prefill_chunk", plan),
        patch.object(Scheduler, "_run_prefill_chunks", run),
        patch("omlx.scheduler.make_prompt_cache", return_value=[]),
    )


def add_long_prefill(scheduler, request_id, remaining, processed=0):
    request = add_prefill(scheduler, request_id)
    state = scheduler._prefill_states[request_id]
    state.tokens_remaining = mx.zeros((1, remaining), dtype=mx.int32)
    state.tokens_processed = processed
    return request


@pytest.mark.parametrize("processed", [0, 8192])
def test_uncontended_long_request_keeps_its_turn(clock, processed):
    """A long arrival does not put its full-width first chunk ahead of the
    in-flight request's chunk; that order and its costs stay as on main."""
    scheduler = make_scheduler(prefill_step_size=4096)
    add_long_prefill(scheduler, "inflight", 4096, processed)
    scheduler.add_request(make_request("long", 20000))
    forwards = []
    with ExitStack() as stack:
        for patcher in fake_chunk_forwards(forwards, clock):
            stack.enter_context(patcher)
        scheduler.step()

    assert forwards[0] == [("inflight", 4096)]
    assert "inflight" in scheduler.running


def test_uncontended_mixed_waiting_keeps_chunk_first(clock):
    scheduler = make_scheduler(prefill_step_size=4096)
    add_long_prefill(scheduler, "inflight", 20000)
    scheduler.add_request(make_request("short", 30))
    scheduler.add_request(make_request("long", 20000))
    forwards = []
    with ExitStack() as stack:
        for patcher in fake_chunk_forwards(forwards, clock):
            stack.enter_context(patcher)
        scheduler.step()

    assert forwards[0] == [("inflight", 4096)]


def test_uncontended_short_packable_request_prefills_before_inflight_chunk(clock):
    """A chat-sized prompt that takes the packed path prefills alone ahead of
    the head, starts decoding, and the head chunk shrinks (#5148)."""
    scheduler = make_scheduler(prefill_step_size=4096)
    scheduler._packed_min_row_tokens = 64
    add_long_prefill(scheduler, "inflight", 20000, 8192)
    scheduler.add_request(make_request("chat", 300))
    forwards = []
    with ExitStack() as stack:
        for patcher in fake_chunk_forwards(forwards, clock):
            stack.enter_context(patcher)
        stack.enter_context(
            patch.object(Scheduler, "_packed_prefill_ready", return_value=True)
        )
        stack.enter_context(
            patch.object(Scheduler, "_packed_prefill_fits", return_value=True)
        )
        scheduler.step()

    assert forwards[0] == [("chat", 299)]
    assert "chat" in scheduler.running
    assert all(tokens < 4096 for _, tokens in forwards[1])
    assert "inflight" in scheduler._prefill_states
