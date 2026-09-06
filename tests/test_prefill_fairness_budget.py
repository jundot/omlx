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


def test_short_external_prefills_share_admission_debt(clock):
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
