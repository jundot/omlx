# SPDX-License-Identifier: Apache-2.0
"""Tests for interactive preemption of background chunked prefills.

Strategy: mock model/tokenizer, manipulate scheduler queues directly,
verify parking/resuming logic without GPU.
"""

from unittest.mock import MagicMock

from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, _PrefillState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scheduler(
    prefill_preemption: bool = True,
    max_num_seqs: int = 8,
) -> Scheduler:
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        prefill_step_size=4,
        chunked_prefill=True,
        paged_cache_block_size=0,
        prefill_preemption=prefill_preemption,
    )
    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)
    mock_bg = MagicMock()
    mock_bg.insert.return_value = [42]
    mock_bg.next_generated.return_value = iter([])
    scheduler.batch_generator = mock_bg
    scheduler._current_sampler_params = ()
    return scheduler


def _make_request(
    request_id: str = "req-1",
    n_tokens: int = 10,
    priority: int = 0,
) -> Request:
    req = Request(
        request_id=request_id,
        prompt=list(range(n_tokens)),
        sampling_params=SamplingParams(max_tokens=32),
        priority=priority,
    )
    req.prompt_token_ids = list(range(n_tokens))
    req.num_prompt_tokens = n_tokens
    req.remaining_tokens = list(range(n_tokens))
    return req


def _make_prefill_state(
    request: Request,
    n_remaining: int = 20,
) -> _PrefillState:
    import mlx.core as mx

    tokens_remaining = mx.zeros((1, n_remaining), dtype=mx.int32)
    return _PrefillState(
        request=request,
        cache=[],
        tokens_remaining=tokens_remaining,
        last_token=[99],
        tokens_processed=0,
        base_size=0,
        emitted_boundaries={},
        boundary_enabled=False,
        block_size=0,
        total_length=n_remaining + 1,
        sampler=MagicMock(),
        sm=MagicMock(),
        per_row_lps=[],
    )


def _enqueue_prefill(scheduler: Scheduler, request: Request) -> _PrefillState:
    """Put a request into the prefilling queue with a _PrefillState."""
    state = _make_prefill_state(request)
    scheduler.prefilling.append(request)
    scheduler._prefill_states[request.request_id] = state
    scheduler.requests[request.request_id] = request
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHasWaitingInteractive:
    def test_empty_waiting(self):
        s = _make_scheduler()
        assert not s._has_waiting_interactive()

    def test_only_background_waiting(self):
        s = _make_scheduler()
        s.waiting.append(_make_request("bg1", priority=10))
        assert not s._has_waiting_interactive()

    def test_interactive_waiting(self):
        s = _make_scheduler()
        s.waiting.append(_make_request("chat1", priority=0))
        assert s._has_waiting_interactive()

    def test_negative_priority_also_interactive(self):
        s = _make_scheduler()
        req = _make_request("fast", priority=-1)
        s.waiting.append(req)
        assert s._has_waiting_interactive()


class TestHasActiveInteractive:
    def test_no_running(self):
        s = _make_scheduler()
        assert not s._has_active_interactive()

    def test_only_background_running(self):
        s = _make_scheduler()
        req = _make_request("bg1", priority=10)
        s.running[req.request_id] = req
        assert not s._has_active_interactive()

    def test_interactive_running(self):
        s = _make_scheduler()
        req = _make_request("chat1", priority=0)
        s.running[req.request_id] = req
        assert s._has_active_interactive()


class TestParkBackgroundPrefills:
    def test_no_parking_when_disabled(self):
        s = _make_scheduler(prefill_preemption=False)
        bg = _make_request("bg1", priority=10)
        _enqueue_prefill(s, bg)
        s.waiting.append(_make_request("chat1", priority=0))

        parked = s._park_background_prefills_for_interactive()
        assert parked == 0
        assert len(s.prefilling) == 1
        assert len(s._suspended_prefills) == 0

    def test_no_parking_when_no_interactive_waiting(self):
        s = _make_scheduler()
        bg = _make_request("bg1", priority=10)
        _enqueue_prefill(s, bg)

        parked = s._park_background_prefills_for_interactive()
        assert parked == 0
        assert len(s.prefilling) == 1

    def test_parks_background_prefills(self):
        s = _make_scheduler()
        bg1 = _make_request("bg1", priority=10)
        bg2 = _make_request("bg2", priority=5)
        _enqueue_prefill(s, bg1)
        _enqueue_prefill(s, bg2)
        s.waiting.append(_make_request("chat1", priority=0))

        parked = s._park_background_prefills_for_interactive()
        assert parked == 2
        assert len(s.prefilling) == 0
        assert len(s._suspended_prefills) == 2
        assert s._prefill_preemptions == 2

    def test_keeps_interactive_prefills(self):
        s = _make_scheduler()
        interactive = _make_request("chat1", priority=0)
        bg = _make_request("bg1", priority=10)
        _enqueue_prefill(s, interactive)
        _enqueue_prefill(s, bg)
        s.waiting.append(_make_request("chat2", priority=0))

        parked = s._park_background_prefills_for_interactive()
        assert parked == 1
        assert len(s.prefilling) == 1
        assert s.prefilling[0].request_id == "chat1"

    def test_retains_prefill_state_identity(self):
        s = _make_scheduler()
        bg = _make_request("bg1", priority=10)
        state = _enqueue_prefill(s, bg)
        s.waiting.append(_make_request("chat1", priority=0))

        s._park_background_prefills_for_interactive()
        _, resumed_state = s._suspended_prefills[0]
        assert resumed_state is state
        assert resumed_state.cache == state.cache


class TestResumeSuspendedPrefills:
    def test_no_resume_when_empty(self):
        s = _make_scheduler()
        assert s._resume_suspended_prefills() == 0

    def test_resumes_all(self):
        s = _make_scheduler()
        bg1 = _make_request("bg1", priority=10)
        bg2 = _make_request("bg2", priority=5)
        _enqueue_prefill(s, bg1)
        _enqueue_prefill(s, bg2)
        s.waiting.append(_make_request("chat1", priority=0))

        s._park_background_prefills_for_interactive()
        assert len(s.prefilling) == 0

        resumed = s._resume_suspended_prefills()
        assert resumed == 2
        assert len(s.prefilling) == 2
        assert len(s._suspended_prefills) == 0
        assert s._prefill_resumes == 2

    def test_preserves_order(self):
        s = _make_scheduler()
        bg1 = _make_request("bg1", priority=10)
        bg2 = _make_request("bg2", priority=5)
        _enqueue_prefill(s, bg1)
        _enqueue_prefill(s, bg2)
        s.waiting.append(_make_request("chat1", priority=0))

        s._park_background_prefills_for_interactive()
        s._resume_suspended_prefills()

        assert s.prefilling[0].request_id == "bg1"
        assert s.prefilling[1].request_id == "bg2"

    def test_restores_prefill_states(self):
        s = _make_scheduler()
        bg = _make_request("bg1", priority=10)
        original_state = _enqueue_prefill(s, bg)
        s.waiting.append(_make_request("chat1", priority=0))

        s._park_background_prefills_for_interactive()
        s._resume_suspended_prefills()

        assert s._prefill_states["bg1"] is original_state


class TestAbortClearsSuspended:
    def test_abort_removes_from_suspended(self):
        s = _make_scheduler()
        bg = _make_request("bg1", priority=10)
        _enqueue_prefill(s, bg)
        s.waiting.append(_make_request("chat1", priority=0))
        s._park_background_prefills_for_interactive()
        assert len(s._suspended_prefills) == 1

        s._do_abort_request("bg1")
        assert len(s._suspended_prefills) == 0
        assert "bg1" not in s._prefill_states


class TestResetClearsSuspended:
    def test_reset_clears_all_suspended_state(self):
        s = _make_scheduler()
        bg = _make_request("bg1", priority=10)
        _enqueue_prefill(s, bg)
        s.waiting.append(_make_request("chat1", priority=0))
        s._park_background_prefills_for_interactive()
        s._prefill_preemptions = 5
        s._prefill_resumes = 3

        s.reset()

        assert len(s._suspended_prefills) == 0
        assert s._prefill_rr_cursor == 0
        assert s._prefill_preemptions == 0
        assert s._prefill_resumes == 0
