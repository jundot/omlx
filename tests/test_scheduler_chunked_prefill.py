# SPDX-License-Identifier: Apache-2.0
"""
Tests for interleaved chunked prefill + decode (SchedulerConfig.chunked_prefill).

Strategy: keep tests fast by mocking MLX model calls and cache operations.
_begin_prefill() and _step_prefill_chunk() are tested by patching
make_prompt_cache and mx.eval; the scheduler-level flow is tested by
patching _step_prefill_chunk directly.
"""

from collections import deque
from unittest.mock import MagicMock, patch

from omlx.request import Request, RequestStatus, SamplingParams
from omlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    _PrefillAbortedError,
    _PrefillState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scheduler(chunked_prefill: bool = True, step_size: int = 4) -> Scheduler:
    """Return a Scheduler with a mock model/tokenizer and chunked_prefill config."""
    model = MagicMock()
    model.layers = []  # No attention layers — keeps _build_state_machine simple

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2

    config = SchedulerConfig(
        max_num_seqs=8,
        prefill_step_size=step_size,
        chunked_prefill=chunked_prefill,
        paged_cache_block_size=0,  # Disable boundary snapshots
    )

    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)

    # Replace the real batch_generator factory so insert() returns a uid.
    mock_bg = MagicMock()
    mock_bg.insert.return_value = [42]
    mock_bg.next_generated.return_value = iter([])
    scheduler.batch_generator = mock_bg
    scheduler._current_sampler_params = ()

    return scheduler


def _make_request(request_id: str = "req-1", n_tokens: int = 10) -> Request:
    """Return a pre-tokenized request with *n_tokens* prompt tokens."""
    req = Request(
        request_id=request_id,
        prompt=list(range(n_tokens)),
        sampling_params=SamplingParams(max_tokens=32),
    )
    req.prompt_token_ids = list(range(n_tokens))
    req.num_prompt_tokens = n_tokens
    req.remaining_tokens = list(range(n_tokens))
    return req


def _make_prefill_state(scheduler: Scheduler, request: Request, n_remaining: int = 20) -> _PrefillState:
    """Build a minimal _PrefillState for direct testing."""
    import mlx.core as mx

    tokens_remaining = mx.zeros((1, n_remaining), dtype=mx.int32)
    state = _PrefillState(
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
    return state


# ---------------------------------------------------------------------------
# SchedulerConfig
# ---------------------------------------------------------------------------

class TestSchedulerConfigChunkedPrefill:
    def test_default_is_false(self):
        config = SchedulerConfig()
        assert config.chunked_prefill is False

    def test_can_be_enabled(self):
        config = SchedulerConfig(chunked_prefill=True)
        assert config.chunked_prefill is True


# ---------------------------------------------------------------------------
# _PrefillState
# ---------------------------------------------------------------------------

class TestPrefillState:
    def test_fields_accessible(self):
        import mlx.core as mx

        state = _PrefillState(
            request=MagicMock(),
            cache=[],
            tokens_remaining=mx.zeros((1, 5), dtype=mx.int32),
            last_token=[7],
            tokens_processed=0,
            base_size=0,
            emitted_boundaries={},
            boundary_enabled=False,
            block_size=256,
            total_length=6,
        )
        assert state.tokens_processed == 0
        assert state.sampler is None
        assert state.per_row_lps is None

    def test_insert_params_settable(self):
        import mlx.core as mx

        state = _PrefillState(
            request=MagicMock(),
            cache=[],
            tokens_remaining=mx.zeros((1, 3), dtype=mx.int32),
            last_token=[1],
            tokens_processed=0,
            base_size=0,
            emitted_boundaries={},
            boundary_enabled=False,
            block_size=256,
            total_length=4,
        )
        state.sampler = "s"
        state.sm = "sm"
        state.per_row_lps = []
        assert state.sampler == "s"


# ---------------------------------------------------------------------------
# Scheduler queues initialised
# ---------------------------------------------------------------------------

class TestSchedulerQueues:
    def test_prefilling_queue_exists(self):
        sched = _make_scheduler()
        assert hasattr(sched, "prefilling")
        assert isinstance(sched.prefilling, deque)
        assert len(sched.prefilling) == 0

    def test_prefill_states_dict_exists(self):
        sched = _make_scheduler()
        assert hasattr(sched, "_prefill_states")
        assert isinstance(sched._prefill_states, dict)


# ---------------------------------------------------------------------------
# has_requests includes prefilling
# ---------------------------------------------------------------------------

class TestHasRequests:
    def test_false_when_all_empty(self):
        sched = _make_scheduler()
        assert not sched.has_requests()

    def test_true_when_prefilling(self):
        sched = _make_scheduler()
        req = _make_request()
        sched.prefilling.append(req)
        assert sched.has_requests()

    def test_still_true_with_waiting_only(self):
        sched = _make_scheduler()
        req = _make_request()
        sched.waiting.append(req)
        assert sched.has_requests()


# ---------------------------------------------------------------------------
# get_stats includes num_prefilling
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_num_prefilling_in_stats(self):
        sched = _make_scheduler()
        stats = sched.get_stats()
        assert "num_prefilling" in stats
        assert stats["num_prefilling"] == 0

    def test_num_prefilling_counts_correctly(self):
        sched = _make_scheduler()
        sched.prefilling.append(_make_request("r1"))
        sched.prefilling.append(_make_request("r2"))
        assert sched.get_stats()["num_prefilling"] == 2


# ---------------------------------------------------------------------------
# reset() clears prefilling
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_prefilling(self):
        sched = _make_scheduler()
        req = _make_request()
        sched.prefilling.append(req)
        sched._prefill_states[req.request_id] = MagicMock()
        sched.requests[req.request_id] = req

        sched.reset()

        assert len(sched.prefilling) == 0
        assert len(sched._prefill_states) == 0


# ---------------------------------------------------------------------------
# fail_all_requests() includes prefilling
# ---------------------------------------------------------------------------

class TestFailAllRequests:
    def test_fail_all_includes_prefilling(self):
        sched = _make_scheduler()
        req = _make_request("pf-req")
        sched.prefilling.append(req)
        sched._prefill_states[req.request_id] = MagicMock()
        sched.requests[req.request_id] = req

        failed = sched.fail_all_requests()

        assert "pf-req" in failed
        assert len(sched.prefilling) == 0
        assert len(sched._prefill_states) == 0


# ---------------------------------------------------------------------------
# _do_abort_request() cleans up prefilling
# ---------------------------------------------------------------------------

class TestAbortPrefilling:
    def test_abort_removes_from_prefilling(self):
        sched = _make_scheduler()
        req = _make_request("abort-me")
        req.status = RequestStatus.WAITING
        sched.prefilling.append(req)
        sched._prefill_states[req.request_id] = MagicMock()
        sched.requests[req.request_id] = req

        sched._do_abort_request(req.request_id)

        assert req.request_id not in sched._prefill_states
        assert all(r.request_id != req.request_id for r in sched.prefilling)


# ---------------------------------------------------------------------------
# _advance_chunked_prefills(): core logic
# ---------------------------------------------------------------------------

class TestAdvanceChunkedPrefills:
    def test_no_op_when_queue_empty(self):
        sched = _make_scheduler()
        scheduled = []
        rejected = []
        # Should not raise
        sched._advance_chunked_prefills(scheduled, rejected)
        assert scheduled == []
        assert rejected == []

    def test_advances_chunk_when_not_done(self):
        """Requests that still have tokens stay in prefilling queue."""
        sched = _make_scheduler()
        req = _make_request("r1")
        sched.requests[req.request_id] = req
        state = _make_prefill_state(sched, req, n_remaining=20)
        sched.prefilling.append(req)
        sched._prefill_states[req.request_id] = state

        with patch.object(sched, "_step_prefill_chunk", return_value=False) as mock_step:
            scheduled = []
            rejected = []
            sched._advance_chunked_prefills(scheduled, rejected)

        mock_step.assert_called_once_with(state)
        # Not done → stays in prefilling, not moved to running
        assert req in sched.prefilling
        assert scheduled == []
        assert rejected == []
        assert req.request_id not in sched.running

    def test_inserts_when_done(self):
        """Completed prefill is inserted into BatchGenerator and moved to running."""
        sched = _make_scheduler()
        req = _make_request("r1")
        sched.requests[req.request_id] = req
        state = _make_prefill_state(sched, req, n_remaining=1)
        state.sampler = MagicMock()
        state.sm = MagicMock()
        state.per_row_lps = []
        sched.prefilling.append(req)
        sched._prefill_states[req.request_id] = state

        with patch.object(sched, "_step_prefill_chunk", return_value=True):
            with patch.object(sched, "_emit_final_boundary_if_needed"):
                scheduled = []
                rejected = []
                sched._advance_chunked_prefills(scheduled, rejected)

        # Moved to running, removed from prefilling
        assert req not in sched.prefilling
        assert req.request_id not in sched._prefill_states
        assert req.request_id in sched.running
        assert req in scheduled
        assert rejected == []
        assert req.status == RequestStatus.RUNNING

    def test_skips_aborted_request(self):
        """Request whose state was cleared by abort is silently skipped."""
        sched = _make_scheduler()
        req = _make_request("gone")
        # State NOT added to _prefill_states (simulates post-abort cleanup)
        sched.prefilling.append(req)

        scheduled = []
        rejected = []
        sched._advance_chunked_prefills(scheduled, rejected)  # Must not raise

        assert scheduled == []
        assert rejected == []
        assert len(sched.prefilling) == 0

    def test_abort_during_chunk_discards_state(self):
        """_PrefillAbortedError from _step_prefill_chunk is swallowed cleanly."""
        sched = _make_scheduler()
        req = _make_request("r1")
        sched.requests[req.request_id] = req
        state = _make_prefill_state(sched, req)
        sched.prefilling.append(req)
        sched._prefill_states[req.request_id] = state

        with patch.object(
            sched, "_step_prefill_chunk",
            side_effect=_PrefillAbortedError([], 4)
        ):
            scheduled = []
            rejected = []
            sched._advance_chunked_prefills(scheduled, rejected)  # Must not raise

        assert req.request_id not in sched._prefill_states
        assert req not in sched.prefilling
        assert scheduled == []
        assert rejected == []

    def test_runtime_error_surfaces_as_request_error(self):
        """RuntimeError mid-chunk yields a finish_reason=\"error\" RequestOutput."""
        sched = _make_scheduler()
        req = _make_request("oom")
        sched.requests[req.request_id] = req
        state = _make_prefill_state(sched, req)
        sched.prefilling.append(req)
        sched._prefill_states[req.request_id] = state

        with patch.object(
            sched, "_step_prefill_chunk",
            side_effect=RuntimeError("Memory limit exceeded")
        ):
            scheduled = []
            rejected = []
            sched._advance_chunked_prefills(scheduled, rejected)

        assert req.request_id not in sched._prefill_states
        assert req not in sched.prefilling
        assert req.request_id not in sched.requests
        assert scheduled == []
        assert len(rejected) == 1
        out = rejected[0]
        assert out.request_id == "oom"
        assert out.finished is True
        assert out.finish_reason == "error"
        assert "Memory limit" in out.error

    def test_multiple_requests_all_advanced(self):
        """All requests in prefilling get one chunk advanced per call."""
        sched = _make_scheduler()
        reqs = [_make_request(f"r{i}") for i in range(3)]
        for req in reqs:
            sched.requests[req.request_id] = req
            state = _make_prefill_state(sched, req, n_remaining=20)
            state.sampler = MagicMock()
            state.sm = MagicMock()
            state.per_row_lps = []
            sched.prefilling.append(req)
            sched._prefill_states[req.request_id] = state

        call_count = 0
        def fake_step(state):
            nonlocal call_count
            call_count += 1
            return False  # All still in-progress

        with patch.object(sched, "_step_prefill_chunk", side_effect=fake_step):
            sched._advance_chunked_prefills([], [])

        assert call_count == 3  # One chunk per request


# ---------------------------------------------------------------------------
# _schedule_waiting(): chunked fork is taken for long prompts
# ---------------------------------------------------------------------------

class TestScheduleWaitingChunkedFork:
    def _setup(self, n_tokens: int, chunked: bool = True, step_size: int = 4):
        sched = _make_scheduler(chunked_prefill=chunked, step_size=step_size)
        req = _make_request("r1", n_tokens=n_tokens)
        sched.add_request(req)
        return sched, req

    def test_short_prompt_stays_on_normal_path(self):
        """Prompts that fit in one chunk use the normal prefill path."""
        # step_size=4, prompt=3 tokens → not long enough to trigger chunked fork
        sched, req = self._setup(n_tokens=3, step_size=4)

        with patch.object(sched, "_do_external_prefill", return_value=([], [0])) as mock_ep:
            with patch.object(sched, "_begin_prefill") as mock_bp:
                sched._schedule_waiting()

        mock_ep.assert_called_once()
        mock_bp.assert_not_called()

    def test_long_prompt_enters_prefilling_queue(self):
        """Prompts longer than step_size+1 enter the chunked prefill queue."""
        # step_size=4, 10 tokens → triggers chunked path
        sched, req = self._setup(n_tokens=10, step_size=4)

        with patch.object(sched, "_begin_prefill", return_value=_make_prefill_state(sched, req)) as mock_bp:
            with patch.object(sched, "_step_prefill_chunk", return_value=False):
                sched._schedule_waiting()

        mock_bp.assert_called_once()
        assert req.request_id in sched._prefill_states
        assert req in sched.prefilling
        assert req.request_id not in sched.running

    def test_long_prompt_completes_in_first_chunk_goes_to_running(self):
        """If the first chunk happens to finish the prefill, request goes to running."""
        sched, req = self._setup(n_tokens=10, step_size=4)
        fake_state = _make_prefill_state(sched, req, n_remaining=1)

        with patch.object(sched, "_begin_prefill", return_value=fake_state):
            with patch.object(sched, "_step_prefill_chunk", return_value=True):
                with patch.object(sched, "_emit_final_boundary_if_needed"):
                    with patch("omlx.scheduler._sync_and_clear_cache"):
                        sched._schedule_waiting()

        assert req.request_id not in sched._prefill_states
        assert req not in sched.prefilling
        assert req.request_id in sched.running

    def test_chunked_disabled_uses_normal_path(self):
        """chunked_prefill=False always uses the full-prefill path."""
        sched, req = self._setup(n_tokens=100, chunked=False, step_size=4)

        with patch.object(sched, "_do_external_prefill", return_value=([], [0])) as mock_ep:
            with patch.object(sched, "_begin_prefill") as mock_bp:
                sched._schedule_waiting()

        mock_ep.assert_called_once()
        mock_bp.assert_not_called()
