# SPDX-License-Identifier: Apache-2.0
"""
Tests for interleaved chunked prefill + decode (SchedulerConfig.chunked_prefill).

Strategy: keep tests fast by mocking MLX model calls and cache operations.
_begin_prefill() and _step_prefill_chunk() are tested by patching
make_prompt_cache and mx.eval; the scheduler-level flow is tested by
patching _step_prefill_chunk directly.
"""

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omlx.exceptions import PrefillMemoryExceededError
from omlx.request import Request, RequestStatus, SamplingParams
from omlx.scheduler import (
    PrefillEvictionRequest,
    Scheduler,
    SchedulerConfig,
    _PrefillAbortedError,
    _PrefillEvictionNeeded,
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


def _make_prefill_state(
    scheduler: Scheduler, request: Request, n_remaining: int = 20
) -> _PrefillState:
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


class _RecordingModel:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.layers = []
        self.chunk_lengths: list[int] = []

    def __call__(self, tokens, cache=None):
        self.chunk_lengths.append(int(tokens.shape[1]))


def _make_recording_scheduler(
    model_type: str,
    *,
    uses_minimax_m3_positions: bool = False,
    nested_vlm_model_type: str | None = None,
    model_name: str = "",
) -> tuple[Scheduler, _RecordingModel]:
    model = _RecordingModel(model_type)
    if uses_minimax_m3_positions:
        model._uses_minimax_m3_positions = True
    if nested_vlm_model_type is not None:
        model._vlm_model = SimpleNamespace(
            config=SimpleNamespace(model_type=nested_vlm_model_type)
        )
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            prefill_step_size=2048,
            chunked_prefill=True,
            paged_cache_block_size=0,
            model_name=model_name,
        ),
    )
    return scheduler, model


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
# GLM adaptive chunked prefill
# ---------------------------------------------------------------------------


class TestGLMAdaptiveChunkedPrefill:
    def test_glm_uses_adaptive_prefill_chunk_size(self, monkeypatch):
        monkeypatch.delenv("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP", raising=False)
        monkeypatch.delenv("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP_SIZE", raising=False)
        monkeypatch.delenv("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_AFTER", raising=False)
        monkeypatch.delenv(
            "MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_MIN_REMAINING", raising=False
        )

        sched, model = _make_recording_scheduler("glm_moe_dsa")
        req = _make_request("glm", n_tokens=8194)
        state = _make_prefill_state(sched, req, n_remaining=8193)

        with patch("omlx.scheduler._sync_and_clear_cache"):
            done = sched._step_prefill_chunk(state)

        assert not done
        assert model.chunk_lengths == [8192]
        assert state.tokens_processed == 8192

    def test_non_glm_keeps_configured_prefill_chunk_size(self, monkeypatch):
        monkeypatch.delenv("MLX_LM_GLM_DSA_ADAPTIVE_PREFILL_STEP", raising=False)

        sched, model = _make_recording_scheduler("deepseek_v32")
        req = _make_request("deepseek", n_tokens=8193)
        state = _make_prefill_state(sched, req, n_remaining=8192)

        with patch("omlx.scheduler._sync_and_clear_cache"):
            done = sched._step_prefill_chunk(state)

        assert not done
        assert model.chunk_lengths == [2048]
        assert state.tokens_processed == 2048


# ---------------------------------------------------------------------------
# MiniMax M3 adaptive chunked prefill
# ---------------------------------------------------------------------------


class TestMiniMaxM3AdaptiveChunkedPrefill:
    def test_minimax_m3_uses_4096_for_long_prefill(self, monkeypatch):
        monkeypatch.delenv("MLX_MINIMAX_M3_ADAPTIVE_PREFILL_STEP", raising=False)
        monkeypatch.delenv("MLX_MINIMAX_M3_ADAPTIVE_PREFILL_STEP_SIZE", raising=False)
        monkeypatch.delenv("MLX_MINIMAX_M3_ADAPTIVE_PREFILL_AFTER", raising=False)
        monkeypatch.delenv(
            "MLX_MINIMAX_M3_ADAPTIVE_PREFILL_MIN_REMAINING", raising=False
        )

        sched, model = _make_recording_scheduler("minimax_m3")
        req = _make_request("minimax", n_tokens=4098)
        state = _make_prefill_state(sched, req, n_remaining=4097)

        with patch("omlx.scheduler._sync_and_clear_cache"):
            done = sched._step_prefill_chunk(state)

        assert not done
        assert model.chunk_lengths == [4096]
        assert state.tokens_processed == 4096

    def test_minimax_m3_keeps_2048_for_short_prefill(self, monkeypatch):
        monkeypatch.delenv("MLX_MINIMAX_M3_ADAPTIVE_PREFILL_STEP", raising=False)

        sched, model = _make_recording_scheduler("minimax_m3_vl")
        req = _make_request("minimax-short", n_tokens=4096)
        state = _make_prefill_state(sched, req, n_remaining=4095)

        with patch("omlx.scheduler._sync_and_clear_cache"):
            done = sched._step_prefill_chunk(state)

        assert not done
        assert model.chunk_lengths == [2048]
        assert state.tokens_processed == 2048

    def test_minimax_m3_env_can_disable_adaptive_prefill(self, monkeypatch):
        monkeypatch.setenv("MLX_MINIMAX_M3_ADAPTIVE_PREFILL_STEP", "0")

        sched, model = _make_recording_scheduler("minimax_m3")
        req = _make_request("minimax-disabled", n_tokens=4098)
        state = _make_prefill_state(sched, req, n_remaining=4097)

        with patch("omlx.scheduler._sync_and_clear_cache"):
            done = sched._step_prefill_chunk(state)

        assert not done
        assert model.chunk_lengths == [2048]
        assert state.tokens_processed == 2048

    def test_minimax_m3_vlm_adapter_flag_enables_adaptive_prefill(self, monkeypatch):
        monkeypatch.delenv("MLX_MINIMAX_M3_ADAPTIVE_PREFILL_STEP", raising=False)

        sched, model = _make_recording_scheduler(
            "vlm",
            uses_minimax_m3_positions=True,
        )
        req = _make_request("minimax-adapter", n_tokens=4098)
        state = _make_prefill_state(sched, req, n_remaining=4097)

        with patch("omlx.scheduler._sync_and_clear_cache"):
            done = sched._step_prefill_chunk(state)

        assert not done
        assert model.chunk_lengths == [4096]
        assert state.tokens_processed == 4096

    def test_minimax_m3_nested_vlm_model_enables_adaptive_prefill(self, monkeypatch):
        monkeypatch.delenv("MLX_MINIMAX_M3_ADAPTIVE_PREFILL_STEP", raising=False)

        sched, model = _make_recording_scheduler(
            "vlm",
            nested_vlm_model_type="minimax_m3_vl",
        )
        req = _make_request("minimax-nested-vlm", n_tokens=4098)
        state = _make_prefill_state(sched, req, n_remaining=4097)

        with patch("omlx.scheduler._sync_and_clear_cache"):
            done = sched._step_prefill_chunk(state)

        assert not done
        assert model.chunk_lengths == [4096]
        assert state.tokens_processed == 4096

    def test_minimax_m3_model_path_enables_adaptive_prefill(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("MLX_MINIMAX_M3_ADAPTIVE_PREFILL_STEP", raising=False)
        (tmp_path / "config.json").write_text(
            '{"model_type": "minimax_m3_vl"}',
            encoding="utf-8",
        )

        sched, model = _make_recording_scheduler(
            "vlm",
            model_name=str(tmp_path),
        )
        req = _make_request("minimax-model-path", n_tokens=4098)
        state = _make_prefill_state(sched, req, n_remaining=4097)

        with patch("omlx.scheduler._sync_and_clear_cache"):
            done = sched._step_prefill_chunk(state)

        assert not done
        assert model.chunk_lengths == [4096]
        assert state.tokens_processed == 4096


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

        with patch.object(
            sched, "_step_prefill_chunk", return_value=False
        ) as mock_step:
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
            sched, "_step_prefill_chunk", side_effect=_PrefillAbortedError([], 4)
        ):
            scheduled = []
            rejected = []
            sched._advance_chunked_prefills(scheduled, rejected)  # Must not raise

        assert req.request_id not in sched._prefill_states
        assert req not in sched.prefilling
        assert scheduled == []
        assert rejected == []

    def test_runtime_error_surfaces_as_request_error(self):
        """A non-memory RuntimeError mid-chunk yields a finish_reason="error"
        RequestOutput immediately (only memory-pressure errors are requeued)."""
        sched = _make_scheduler()
        req = _make_request("oom")
        sched.requests[req.request_id] = req
        state = _make_prefill_state(sched, req)
        sched.prefilling.append(req)
        sched._prefill_states[req.request_id] = state

        with patch.object(
            sched, "_step_prefill_chunk", side_effect=RuntimeError("kernel panic")
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
        assert "kernel panic" in out.error

    def test_memory_error_requeues_instead_of_surfacing(self):
        """A memory-pressure RuntimeError mid-chunk requeues the request for a
        fresh attempt instead of immediately surfacing an error to the client."""
        sched = _make_scheduler()
        req = _make_request("oom-mem")
        sched.requests[req.request_id] = req
        state = _make_prefill_state(sched, req)
        sched.prefilling.append(req)
        sched._prefill_states[req.request_id] = state

        with patch.object(
            sched,
            "_step_prefill_chunk",
            side_effect=RuntimeError("Memory limit exceeded during chunked prefill"),
        ):
            scheduled = []
            rejected = []
            sched._advance_chunked_prefills(scheduled, rejected)

        # No client-facing error; the request is reset and back on the queue.
        assert rejected == []
        assert req.request_id not in sched._prefill_states
        assert req not in sched.prefilling
        assert sched.requests.get(req.request_id) is req
        assert req in sched.waiting
        assert req.prefill_oom_retries == 1

    def test_capacity_error_surfaces_as_typed_request_error(self):
        """A deterministic capacity rejection is not retried as transient OOM."""
        sched = _make_scheduler()
        req = _make_request("capacity")
        sched.requests[req.request_id] = req
        state = _make_prefill_state(sched, req)
        sched.prefilling.append(req)
        sched._prefill_states[req.request_id] = state

        err = PrefillMemoryExceededError(
            message="Prefill context too large for available memory",
            request_id=req.request_id,
            estimated_bytes=123,
            limit_bytes=100,
        )
        with patch.object(sched, "_step_prefill_chunk", side_effect=err):
            scheduled = []
            rejected = []
            sched._advance_chunked_prefills(scheduled, rejected)

        assert scheduled == []
        assert len(rejected) == 1
        out = rejected[0]
        assert out.error == str(err)
        assert out.error_code == "prefill_memory_exceeded"
        assert out.error_metadata == {
            "request_id": req.request_id,
            "estimated_bytes": 123,
            "limit_bytes": 100,
        }
        assert req.prefill_oom_retries == 0

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

        with patch.object(
            sched, "_do_external_prefill", return_value=([], [0])
        ) as mock_ep:
            with patch.object(sched, "_begin_prefill") as mock_bp:
                sched._schedule_waiting()

        mock_ep.assert_called_once()
        mock_bp.assert_not_called()

    def test_long_prompt_enters_prefilling_queue(self):
        """Prompts longer than step_size+1 enter the chunked prefill queue."""
        # step_size=4, 10 tokens → triggers chunked path
        sched, req = self._setup(n_tokens=10, step_size=4)

        with patch.object(
            sched, "_begin_prefill", return_value=_make_prefill_state(sched, req)
        ) as mock_bp:
            with patch.object(sched, "_step_prefill_chunk", return_value=False):
                sched._schedule_waiting()

        mock_bp.assert_called_once()
        assert req.request_id in sched._prefill_states
        assert req in sched.prefilling
        assert req.request_id not in sched.running

    def test_prefilling_request_counts_against_concurrency_cap(self):
        """A chunked prefill already in flight consumes a scheduler slot."""
        sched = _make_scheduler(chunked_prefill=True, step_size=4)
        sched.config.max_num_seqs = 1

        inflight = _make_request("inflight", n_tokens=10)
        sched.requests[inflight.request_id] = inflight
        sched.prefilling.append(inflight)
        sched._prefill_states[inflight.request_id] = _make_prefill_state(
            sched,
            inflight,
        )

        queued = _make_request("queued", n_tokens=10)
        sched.add_request(queued)

        with patch.object(sched, "_begin_prefill") as mock_begin:
            scheduled, rejected = sched._schedule_waiting()

        mock_begin.assert_not_called()
        assert scheduled == []
        assert rejected == []
        assert queued in sched.waiting
        assert inflight in sched.prefilling

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

        with patch.object(
            sched, "_do_external_prefill", return_value=([], [0])
        ) as mock_ep:
            with patch.object(sched, "_begin_prefill") as mock_bp:
                sched._schedule_waiting()

        mock_ep.assert_called_once()
        mock_bp.assert_not_called()

    def test_non_chunked_path_runtime_error_cleans_up_and_rejects(self):
        """RuntimeError from _do_external_prefill in the non-chunked path
        must pop self.requests, drop the temp uid mappings, remove the
        PrefillProgressTracker entry, and emit a finish_reason=\"error\"
        RequestOutput so the client sees the failure (#1405)."""
        from omlx.prefill_progress import get_prefill_tracker

        sched, req = self._setup(n_tokens=3, step_size=4)
        rid = req.request_id
        tracker = get_prefill_tracker()
        tracker.clear()
        tracker.update(rid, processed=1, total=3, model_id="test")
        assert tracker.get_model_progress("test"), "tracker entry not set up"

        try:
            with patch.object(
                sched,
                "_do_external_prefill",
                side_effect=RuntimeError("Memory limit exceeded during prefill"),
            ):
                scheduled, rejected = sched._schedule_waiting()

            assert rid not in sched.requests
            assert rid not in sched.request_id_to_uid
            assert not any(v == rid for v in sched.uid_to_request_id.values())
            assert tracker.get_model_progress("test") == []
            assert scheduled == []
            assert len(rejected) == 1
            out = rejected[0]
            assert out.request_id == rid
            assert out.finished is True
            assert out.finish_reason == "error"
            assert "Memory limit" in out.error
        finally:
            tracker.clear()

    def _setup_throttle(self, max_bytes_gb=10, hard_cap_gb=12):
        """Build a scheduler with watermark fields set for throttle tests."""
        sched = _make_scheduler()
        sched._memory_limit_bytes = max_bytes_gb * 1024**3
        sched._memory_hard_limit_bytes = hard_cap_gb * 1024**3
        sched._prefill_safe_zone_ratio = 0.80
        sched._prefill_min_chunk_tokens = 32
        return sched

    def _mock_current(self, sched, current_gb):
        """Context manager-ish — patch both memory probes to current_gb."""
        target = int(current_gb * 1024**3)
        return patch("omlx.scheduler.mx.get_active_memory", return_value=target), patch(
            "omlx.scheduler.get_phys_footprint", return_value=target
        )

    def test_adaptive_throttle_below_soft_watermark_passthrough(self):
        """current < soft watermark → no throttle, full chunk."""
        sched = self._setup_throttle(max_bytes_gb=10, hard_cap_gb=12)
        # soft_watermark = 10 * 0.80 = 8 GB; current 5 GB is below
        a, b = self._mock_current(sched, 5)
        with a, b:
            result = sched._adaptive_chunk_size(
                2048, request_id="r1", loop_label="external"
            )
        assert result == 2048

    def test_adaptive_throttle_tier_1024(self):
        """First quarter of the soft-to-hard band → 1024."""
        sched = self._setup_throttle(max_bytes_gb=10, hard_cap_gb=12)
        # soft_wm = 8 GB, band = 12 - 8 = 4 GB. 10% into band = 8.4 GB.
        a, b = self._mock_current(sched, 8.4)
        with a, b:
            result = sched._adaptive_chunk_size(
                2048, request_id="r1", loop_label="external"
            )
        assert result == 1024

    def test_adaptive_throttle_tier_512(self):
        """50%+ of band → 512."""
        sched = self._setup_throttle(max_bytes_gb=10, hard_cap_gb=12)
        # 60% of band: 8 + 4*0.60 = 10.4 GB
        a, b = self._mock_current(sched, 10.4)
        with a, b:
            result = sched._adaptive_chunk_size(
                2048, request_id="r1", loop_label="external"
            )
        assert result == 512

    def test_adaptive_throttle_requested_smaller_than_tier(self):
        """Requested chunk already smaller than the tier target → pass through."""
        sched = self._setup_throttle(max_bytes_gb=10, hard_cap_gb=12)
        # 60% of band → tier 512. But requested=256 < 512.
        a, b = self._mock_current(sched, 10.4)
        with a, b:
            result = sched._adaptive_chunk_size(
                256, request_id="r1", loop_label="external"
            )
        assert result == 256

    def test_adaptive_throttle_no_cap_passthrough(self):
        """When hard limit or soft base is unset (=0), no throttle."""
        sched = self._setup_throttle()
        sched._memory_hard_limit_bytes = 0
        result = sched._adaptive_chunk_size(
            2048, request_id="r1", loop_label="external"
        )
        assert result == 2048

        sched._memory_hard_limit_bytes = 10 * 1024**3
        sched._memory_limit_bytes = 0
        result = sched._adaptive_chunk_size(
            2048, request_id="r1", loop_label="external"
        )
        assert result == 2048

    def test_chunked_first_chunk_runtime_error_cleans_up_and_rejects(self):
        """RuntimeError on the chunked first chunk must pop self.requests,
        remove the PrefillProgressTracker entry, and emit an error
        RequestOutput. _step_prefill_chunk updates the tracker before the
        hard-limit check, so without this catch the entry would leak
        (#1405)."""
        from omlx.prefill_progress import get_prefill_tracker

        sched, req = self._setup(n_tokens=10, step_size=4)
        rid = req.request_id
        tracker = get_prefill_tracker()
        tracker.clear()
        tracker.update(rid, processed=2, total=10, model_id="test")
        assert tracker.get_model_progress("test"), "tracker entry not set up"

        try:
            with patch.object(
                sched,
                "_begin_prefill",
                return_value=_make_prefill_state(sched, req),
            ):
                with patch.object(
                    sched,
                    "_step_prefill_chunk",
                    side_effect=RuntimeError(
                        "Memory limit exceeded during chunked prefill"
                    ),
                ):
                    scheduled, rejected = sched._schedule_waiting()

            assert rid not in sched.requests
            assert rid not in sched._prefill_states
            assert req not in sched.prefilling
            assert tracker.get_model_progress("test") == []
            assert scheduled == []
            assert len(rejected) == 1
            out = rejected[0]
            assert out.request_id == rid
            assert out.finished is True
            assert out.finish_reason == "error"
            assert "Memory limit" in out.error
        finally:
            tracker.clear()


# ---------------------------------------------------------------------------
# Prefill-rejection paged-cache cleanup
# ---------------------------------------------------------------------------


class TestPrefillRejectionReleasesPagedCache:
    """Rejection paths must release block_aware_cache refs / paged_cache
    block_table entries that ``add_request`` populated via ``fetch_cache``.

    Without this, every rejected request leaks an entry in
    ``BlockAwarePrefixCache._request_tables`` plus the ref counts on its
    prefix-matched blocks — pinning the paged cache and compounding the
    very memory pressure that triggered the rejection. The existing
    ``self.requests.pop(...)`` and ``get_prefill_tracker().remove(...)``
    cleanups handle scheduler-side state but never reach into the
    paged-cache layer.
    """

    def test_helper_calls_block_aware_cache_release(self):
        """The helper delegates to block_aware_cache.release_cache when one
        is attached — the normal production wiring."""
        sched = _make_scheduler()
        sched.block_aware_cache = MagicMock()
        sched.paged_cache_manager = MagicMock()

        sched._release_paged_cache_for_request("rid-1")

        sched.block_aware_cache.release_cache.assert_called_once_with("rid-1")
        # release_cache delegates to delete_block_table internally; the
        # helper must NOT also call it directly (double-delete).
        sched.paged_cache_manager.delete_block_table.assert_not_called()

    def test_helper_falls_back_to_paged_cache_manager(self):
        """Without a BlockAwarePrefixCache, fall back to deleting the block
        table directly on the paged cache manager."""
        sched = _make_scheduler()
        sched.block_aware_cache = None
        sched.paged_cache_manager = MagicMock()

        sched._release_paged_cache_for_request("rid-2")

        sched.paged_cache_manager.delete_block_table.assert_called_once_with("rid-2")

    def test_helper_is_noop_without_any_paged_cache(self):
        """No paged-cache layer attached → silent no-op."""
        sched = _make_scheduler()
        sched.block_aware_cache = None
        sched.paged_cache_manager = None

        # Should not raise.
        sched._release_paged_cache_for_request("rid-3")

    def test_advance_chunked_prefills_releases_on_runtime_error(self):
        """_advance_chunked_prefills' RuntimeError handler must call
        release_cache so the paged-cache block refs from the request's
        prefix-cache lookup don't leak."""
        sched = _make_scheduler()
        sched.block_aware_cache = MagicMock()
        req = _make_request("oom-chunked")
        sched.requests[req.request_id] = req
        state = _make_prefill_state(sched, req)
        sched.prefilling.append(req)
        sched._prefill_states[req.request_id] = state

        with patch.object(
            sched,
            "_step_prefill_chunk",
            side_effect=RuntimeError("Memory limit exceeded"),
        ):
            sched._advance_chunked_prefills([], [])

        sched.block_aware_cache.release_cache.assert_called_once_with("oom-chunked")

    def test_schedule_waiting_non_chunked_releases_on_runtime_error(self):
        """The non-chunked _do_external_prefill rejection path must release
        the paged-cache footprint before popping self.requests."""
        sched = _make_scheduler(step_size=4)
        sched.block_aware_cache = MagicMock()
        # No prefix-cache hit: fetch_cache returns (None, prompt_tokens) so
        # add_request falls through to the waiting queue without trying to
        # preload/reconstruct.
        sched.block_aware_cache.fetch_cache.return_value = (None, [0, 1, 2])
        req = _make_request("oom-direct", n_tokens=3)
        sched.add_request(req)
        sched.block_aware_cache.reset_mock()

        with patch.object(
            sched,
            "_do_external_prefill",
            side_effect=RuntimeError("kernel panic"),
        ):
            sched._schedule_waiting()

        sched.block_aware_cache.release_cache.assert_called_once_with("oom-direct")

    def test_schedule_waiting_chunked_first_chunk_releases_on_runtime_error(self):
        """The chunked first-chunk rejection path must release the
        paged-cache footprint before popping self.requests."""
        sched = _make_scheduler(step_size=4)
        sched.block_aware_cache = MagicMock()
        sched.block_aware_cache.fetch_cache.return_value = (None, list(range(10)))
        req = _make_request("oom-first-chunk", n_tokens=10)
        sched.add_request(req)
        sched.block_aware_cache.reset_mock()

        with patch.object(
            sched,
            "_begin_prefill",
            return_value=_make_prefill_state(sched, req),
        ):
            with patch.object(
                sched,
                "_step_prefill_chunk",
                side_effect=RuntimeError("kernel panic"),
            ):
                sched._schedule_waiting()

        sched.block_aware_cache.release_cache.assert_called_once_with("oom-first-chunk")

    def test_schedule_waiting_preflight_rejection_releases(self):
        """_preflight_memory_check rejection (the non-RuntimeError path
        inside _schedule_waiting) must also release the paged-cache
        footprint. Same leak shape as the RuntimeError rejections — the
        request reached this point via add_request → fetch_cache so
        _request_tables is populated and prefix block refs are held."""
        sched = _make_scheduler(step_size=4)
        sched.block_aware_cache = MagicMock()
        sched.block_aware_cache.fetch_cache.return_value = (None, list(range(5)))
        req = _make_request("oom-preflight", n_tokens=5)
        sched.add_request(req)
        sched.block_aware_cache.reset_mock()

        from omlx.scheduler import _PreflightRejection

        with patch.object(
            sched,
            "_preflight_memory_check",
            return_value=_PreflightRejection(
                message="Memory limit exceeded by preflight estimate",
                estimated_bytes=1,
                limit_bytes=1,
            ),
        ):
            scheduled, rejected = sched._schedule_waiting()

        assert scheduled == []
        assert len(rejected) == 1
        assert rejected[0].request_id == "oom-preflight"
        assert rejected[0].finish_reason == "error"
        sched.block_aware_cache.release_cache.assert_called_once_with("oom-preflight")


# ---------------------------------------------------------------------------
# First-chunk eviction pause must preserve a reconstructed prefix (#2180)
# ---------------------------------------------------------------------------


class TestFirstChunkEvictionPreservesPrefix:
    def test_first_chunk_eviction_pause_keeps_reconstructed_prefix(self):
        """_PrefillEvictionNeeded raised before the first chunk's forward
        pass must not discard a reconstructed SSD prefix. The eviction pause
        keeps prompt_cache / block_table / cached_tokens / remaining_tokens
        attached, so when no idle model can be evicted the retry prefills
        only the uncached suffix instead of recomputing the whole prompt
        cold (#2180)."""
        sched = _make_scheduler(step_size=4)
        sched.block_aware_cache = MagicMock()
        sched.block_aware_cache.fetch_cache.return_value = (None, list(range(100)))
        req = _make_request("evict-first-chunk", n_tokens=100)
        sched.add_request(req)
        sched.block_aware_cache.reset_mock()

        # Simulate the state _prepare_prefix_cache_for_request leaves after a
        # successful paged/SSD cache hit + reconstruction: 90 cached tokens,
        # a 10-token uncached suffix, and a live block table.
        prompt_cache = [MagicMock()]
        block_table = MagicMock()
        sched._prefix_cache_prepared.add(req.request_id)
        req.prompt_cache = prompt_cache
        req.cached_tokens = 90
        req.remaining_tokens = req.prompt_token_ids[90:]
        req.block_table = block_table
        req.shared_prefix_blocks = 3

        eviction = PrefillEvictionRequest(
            request_id=req.request_id,
            model_id="test",
            current_bytes=1,
            target_cap_bytes=1,
            predicted_transient_bytes=1,
            requested_tokens=4,
            reason="adaptive_prefill_throttle",
        )
        with patch.object(
            sched,
            "_begin_prefill",
            return_value=_make_prefill_state(sched, req),
        ):
            with patch.object(
                sched,
                "_step_prefill_chunk",
                side_effect=_PrefillEvictionNeeded(eviction),
            ):
                scheduled, rejected = sched._schedule_waiting()

        assert scheduled == []
        assert rejected == []
        # Paused back into the waiting queue with the eviction request pending.
        assert req in sched.waiting
        assert sched._pending_prefill_eviction_request is eviction
        # The reconstructed prefix must survive the pause untouched.
        assert req.prompt_cache is prompt_cache
        assert req.cached_tokens == 90
        assert req.remaining_tokens == req.prompt_token_ids[90:]
        assert req.block_table is block_table
        assert req.shared_prefix_blocks == 3
        sched.block_aware_cache.release_cache.assert_not_called()
