# SPDX-License-Identifier: Apache-2.0
"""Tests for decode floor: suspend excess background decode rows when interactive active.

Strategy: mock model/tokenizer/BatchGenerator, manipulate scheduler queues directly,
verify suspend/resume logic without GPU.
"""

from unittest.mock import MagicMock, patch

from omlx.request import Request, SamplingParams
from omlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    _SuspendedDecodeState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scheduler(
    decode_floor: float = 0.5,
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
        prefill_preemption=True,
        interactive_decode_floor=decode_floor,
    )
    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)
    mock_bg = MagicMock()
    mock_bg.insert.return_value = [42]
    mock_bg.remove.return_value = {}
    mock_bg.next_generated.return_value = iter([])
    scheduler.batch_generator = mock_bg
    scheduler._current_sampler_params = ()
    return scheduler


def _make_request(
    request_id: str = "req-1",
    n_tokens: int = 10,
    priority: int = 0,
    max_tokens: int = 32,
) -> Request:
    req = Request(
        request_id=request_id,
        prompt=list(range(n_tokens)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        priority=priority,
    )
    req.prompt_token_ids = list(range(n_tokens))
    req.num_prompt_tokens = n_tokens
    req.remaining_tokens = list(range(n_tokens))
    return req


def _enqueue_running(
    scheduler: Scheduler,
    request: Request,
    uid: int = 10,
) -> None:
    """Put a request directly into running state (simulating post-prefill)."""
    scheduler.running[request.request_id] = request
    scheduler.requests[request.request_id] = request
    scheduler.uid_to_request_id[uid] = request.request_id
    scheduler.request_id_to_uid[request.request_id] = uid
    request.batch_uid = uid


# ---------------------------------------------------------------------------
# Tests: floor calculation
# ---------------------------------------------------------------------------

class TestTargetBackgroundDecodeRows:
    def test_floor_zero_keeps_all(self):
        s = _make_scheduler(decode_floor=0.0)
        r1 = _make_request("bg1", priority=10)
        r2 = _make_request("bg2", priority=10)
        _enqueue_running(s, r1, uid=1)
        _enqueue_running(s, r2, uid=2)
        # floor=0 means no limit
        assert s._target_background_decode_rows() == 2

    def test_floor_one_removes_all_with_interactive(self):
        s = _make_scheduler(decode_floor=1.0)
        interactive = _make_request("chat1", priority=0)
        bg1 = _make_request("bg1", priority=10)
        _enqueue_running(s, interactive, uid=1)
        _enqueue_running(s, bg1, uid=2)
        # floor=1.0, 1 interactive → floor((1*0)/1) = 0
        assert s._target_background_decode_rows() == 0

    def test_floor_one_keeps_all_when_no_interactive(self):
        s = _make_scheduler(decode_floor=1.0)
        r1 = _make_request("bg1", priority=10)
        _enqueue_running(s, r1, uid=1)
        # floor=1.0, 0 interactive → return current count (1)
        assert s._target_background_decode_rows() == 1

    def test_floor_half_one_per_interactive(self):
        s = _make_scheduler(decode_floor=0.5)
        interactive = _make_request("chat1", priority=0)
        bg1 = _make_request("bg1", priority=10)
        bg2 = _make_request("bg2", priority=10)
        _enqueue_running(s, interactive, uid=1)
        _enqueue_running(s, bg1, uid=2)
        _enqueue_running(s, bg2, uid=3)
        # 1 interactive, floor=0.5 → floor((1*(1-0.5))/0.5) = floor(1) = 1
        assert s._target_background_decode_rows() == 1

    def test_floor_quarter_two_per_interactive(self):
        s = _make_scheduler(decode_floor=0.25)
        interactive = _make_request("chat1", priority=0)
        bg1 = _make_request("bg1", priority=10)
        bg2 = _make_request("bg2", priority=10)
        bg3 = _make_request("bg3", priority=10)
        _enqueue_running(s, interactive, uid=1)
        _enqueue_running(s, bg1, uid=2)
        _enqueue_running(s, bg2, uid=3)
        _enqueue_running(s, bg3, uid=4)
        # 1 interactive, floor=0.25 → floor((1*0.75)/0.25) = floor(3) = 3
        assert s._target_background_decode_rows() == 3

    def test_no_interactive_returns_current(self):
        s = _make_scheduler(decode_floor=0.5)
        bg1 = _make_request("bg1", priority=10)
        bg2 = _make_request("bg2", priority=10)
        _enqueue_running(s, bg1, uid=1)
        _enqueue_running(s, bg2, uid=2)
        # No interactive → return current count
        assert s._target_background_decode_rows() == 2


# ---------------------------------------------------------------------------
# Tests: suspend/resume
# ---------------------------------------------------------------------------

class TestSuspendDecodeRequest:
    def test_suspend_extracts_cache_and_removes_from_batch(self):
        s = _make_scheduler(decode_floor=0.5)
        req = _make_request("bg1", priority=10)
        _enqueue_running(s, req, uid=5)

        fake_cache = [MagicMock(), MagicMock()]
        fake_tokens = [1, 2, 3]
        s.batch_generator.remove.return_value = {5: (fake_cache, fake_tokens)}

        result = s._suspend_decode_request("bg1")
        assert result is True
        assert "bg1" in s._suspended_decodes
        assert s._decode_suspensions == 1
        s.batch_generator.remove.assert_called_once_with([5], return_prompt_caches=True)
        # uid mappings cleaned up
        assert 5 not in s.uid_to_request_id
        assert "bg1" not in s.request_id_to_uid
        # batch_uid cleared
        assert req.batch_uid is None
        # request stays in running
        assert "bg1" in s.running

    def test_suspend_preserves_state(self):
        s = _make_scheduler(decode_floor=0.5)
        req = _make_request("bg1", priority=10, max_tokens=100)
        req.output_token_ids = [10, 11, 12]
        _enqueue_running(s, req, uid=5)

        fake_cache = [MagicMock()]
        fake_tokens = [1, 2, 3, 10, 11, 12]
        s.batch_generator.remove.return_value = {5: (fake_cache, fake_tokens)}

        s._suspend_decode_request("bg1")
        state = s._suspended_decodes["bg1"]
        assert state.request is req
        assert state.prompt_cache == fake_cache
        assert state.remaining_max_tokens == 100 - 3  # max_tokens - len(output)

    def test_suspend_skips_interactive(self):
        s = _make_scheduler(decode_floor=0.5)
        req = _make_request("chat1", priority=0)
        _enqueue_running(s, req, uid=5)

        result = s._suspend_decode_request("chat1")
        assert result is False
        assert "chat1" not in s._suspended_decodes

    def test_suspend_skips_already_suspended(self):
        s = _make_scheduler(decode_floor=0.5)
        req = _make_request("bg1", priority=10)
        _enqueue_running(s, req, uid=5)

        fake_cache = [MagicMock()]
        s.batch_generator.remove.return_value = {5: (fake_cache, [1])}
        s._suspend_decode_request("bg1")
        s.batch_generator.remove.reset_mock()

        # Second call should return False (not in request_id_to_uid)
        result = s._suspend_decode_request("bg1")
        assert result is False
        s.batch_generator.remove.assert_not_called()

    def test_suspend_skips_vlm_mtp(self):
        s = _make_scheduler(decode_floor=0.5)
        req = _make_request("mtp1", priority=10)
        _enqueue_running(s, req, uid=-1)  # negative uid = vlm_mtp

        result = s._suspend_decode_request("mtp1")
        assert result is False


class TestResumeSuspendedDecodes:
    def test_resume_reinserts_into_batch(self):
        s = _make_scheduler(decode_floor=0.5)
        req = _make_request("bg1", priority=10)
        _enqueue_running(s, req, uid=5)

        fake_cache = [MagicMock()]
        fake_tokens = [1, 2, 3]
        s.batch_generator.remove.return_value = {5: (fake_cache, fake_tokens)}
        s._suspend_decode_request("bg1")
        s.batch_generator.insert.return_value = [99]  # new uid on re-insert

        resumed = s._resume_suspended_decodes()
        assert resumed == 1
        assert "bg1" not in s._suspended_decodes
        assert s._decode_resumes == 1
        s.batch_generator.insert.assert_called_once()
        # New uid mapping established
        assert s.request_id_to_uid["bg1"] == 99
        assert s.uid_to_request_id[99] == "bg1"
        assert req.batch_uid == 99

    def test_resume_skips_aborted_requests(self):
        s = _make_scheduler(decode_floor=0.5)
        req = _make_request("bg1", priority=10)
        _enqueue_running(s, req, uid=5)

        fake_cache = [MagicMock()]
        s.batch_generator.remove.return_value = {5: (fake_cache, [1])}
        s._suspend_decode_request("bg1")

        # Simulate abort: remove from running
        del s.running["bg1"]

        resumed = s._resume_suspended_decodes()
        assert resumed == 0

    def test_resume_empty_dict(self):
        s = _make_scheduler(decode_floor=0.5)
        resumed = s._resume_suspended_decodes()
        assert resumed == 0


class TestCleanupSuspendedDecode:
    def test_cleanup_removes_from_dict(self):
        s = _make_scheduler(decode_floor=0.5)
        req = _make_request("bg1", priority=10)
        _enqueue_running(s, req, uid=5)

        fake_cache = [MagicMock()]
        s.batch_generator.remove.return_value = {5: (fake_cache, [1])}
        s._suspend_decode_request("bg1")
        assert "bg1" in s._suspended_decodes

        s._cleanup_suspended_decode("bg1")
        assert "bg1" not in s._suspended_decodes

    def test_cleanup_noop_for_unknown(self):
        s = _make_scheduler(decode_floor=0.5)
        s._cleanup_suspended_decode("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# Tests: apply_decode_floor
# ---------------------------------------------------------------------------

class TestApplyDecodeFloor:
    def test_floor_zero_noop(self):
        s = _make_scheduler(decode_floor=0.0)
        bg1 = _make_request("bg1", priority=10)
        _enqueue_running(s, bg1, uid=1)
        s._apply_decode_floor()
        assert s._decode_suspensions == 0

    def test_no_interactive_resumes_all(self):
        s = _make_scheduler(decode_floor=0.5)
        bg1 = _make_request("bg1", priority=10)
        _enqueue_running(s, bg1, uid=1)
        fake_cache = [MagicMock()]
        s.batch_generator.remove.return_value = {1: (fake_cache, [1])}
        s._suspend_decode_request("bg1")
        s.batch_generator.insert.return_value = [1]

        s._apply_decode_floor()
        assert s._decode_resumes == 1
        assert "bg1" not in s._suspended_decodes

    def test_suspend_excess_when_interactive_active(self):
        s = _make_scheduler(decode_floor=0.5)
        interactive = _make_request("chat1", priority=0)
        bg1 = _make_request("bg1", priority=10)
        bg2 = _make_request("bg2", priority=10)
        _enqueue_running(s, interactive, uid=1)
        _enqueue_running(s, bg1, uid=2)
        _enqueue_running(s, bg2, uid=3)

        fake_cache = [MagicMock()]
        s.batch_generator.remove.side_effect = [
            {2: (fake_cache, [1])},  # suspend bg1
        ]

        s._apply_decode_floor()
        # 1 interactive, floor=0.5 → target=1 bg, 2 active → 1 excess
        assert s._decode_suspensions == 1
        assert "bg1" in s._suspended_decodes or "bg2" in s._suspended_decodes

    def test_within_budget_no_suspend(self):
        s = _make_scheduler(decode_floor=0.5)
        interactive = _make_request("chat1", priority=0)
        bg1 = _make_request("bg1", priority=10)
        _enqueue_running(s, interactive, uid=1)
        _enqueue_running(s, bg1, uid=2)

        s._apply_decode_floor()
        # 1 interactive, floor=0.5 → target=1 bg, 1 active → within budget
        assert s._decode_suspensions == 0

    def test_batch_generator_required(self):
        s = _make_scheduler(decode_floor=0.5)
        s.batch_generator = None
        interactive = _make_request("chat1", priority=0)
        bg1 = _make_request("bg1", priority=10)
        _enqueue_running(s, interactive, uid=1)
        _enqueue_running(s, bg1, uid=2)

        s._apply_decode_floor()
        assert s._decode_suspensions == 0


# ---------------------------------------------------------------------------
# Tests: reset clears decode floor state
# ---------------------------------------------------------------------------

class TestResetClearsDecodeFloor:
    def test_reset_clears_suspended_decodes(self):
        s = _make_scheduler(decode_floor=0.5)
        req = _make_request("bg1", priority=10)
        _enqueue_running(s, req, uid=5)

        fake_cache = [MagicMock()]
        s.batch_generator.remove.return_value = {5: (fake_cache, [1])}
        s._suspend_decode_request("bg1")
        assert len(s._suspended_decodes) == 1

        s.reset()
        assert len(s._suspended_decodes) == 0
        assert s._decode_suspensions == 0
        assert s._decode_resumes == 0


# ---------------------------------------------------------------------------
# Tests: abort cleans up suspended decodes
# ---------------------------------------------------------------------------

class TestAbortCleansSuspendedDecodes:
    def test_abort_removes_from_suspended(self):
        s = _make_scheduler(decode_floor=0.5)
        req = _make_request("bg1", priority=10)
        _enqueue_running(s, req, uid=5)

        fake_cache = [MagicMock()]
        s.batch_generator.remove.return_value = {5: (fake_cache, [1])}
        s._suspend_decode_request("bg1")
        assert "bg1" in s._suspended_decodes

        # Abort the request
        s._do_abort_request("bg1")
        assert "bg1" not in s._suspended_decodes
        assert "bg1" not in s.running
