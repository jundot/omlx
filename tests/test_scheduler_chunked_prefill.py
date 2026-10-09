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

import mlx.core as mx
import pytest
from mlx_lm.models import qwen3_5_moe
from mlx_lm.models.cache import make_prompt_cache
from mlx_vlm.models.qwen3_5 import config as qwen3_5_config
from mlx_vlm.models.qwen3_5 import language as qwen3_5_language

import omlx.scheduler as scheduler_module
from omlx.exceptions import PrefillMemoryExceededError
from omlx.models.vlm import VLMModelAdapter
from omlx.patches import mlx_vlm_qwen4_exp_compat
from omlx.patches.hy_v3 import apply_hy_v3_patch
from omlx.patches.mlx_vlm_glm5_next_compat import (
    apply_mlx_vlm_glm5_next_compat_patch,
)
from omlx.prefill.packed import (
    PackedBatch,
    PackedRow,
    PackedRows,
    install_packed_prefill,
    packed_min_row_tokens,
    run_packed_prefill,
)
from omlx.request import Request, RequestStatus, SamplingParams
from omlx.scheduler import (
    PrefillEvictionRequest,
    Scheduler,
    SchedulerConfig,
    _bind_text_prefill_rope_delta,
    _default_generation_stream,
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


@pytest.mark.parametrize("chunked", [False, True])
def test_prefill_interrupts_mtp_cost_timing(chunked):
    from omlx.patches.mlx_lm_mtp.batch_policy import BatchPolicy

    scheduler, model = _make_recording_scheduler("qwen3_5_moe")
    policy = BatchPolicy([0, 1], 3)
    scheduler.batch_generator = SimpleNamespace(
        _generation_batch=SimpleNamespace(_omlx_mtp_batch_policy=policy)
    )
    policy.cycle_time_ms("mtp", 1.0, 1.02)
    request = _make_request("timing", n_tokens=9)
    with patch("omlx.scheduler._sync_and_clear_cache"):
        if chunked:
            state = _make_prefill_state(scheduler, request, n_remaining=8)
            scheduler._step_prefill_chunk(state)
        else:
            cache = [SimpleNamespace(state=mx.array([0]))]
            scheduler._do_external_prefill(request, list(range(9)), cache)
    assert model.chunk_lengths == [8]
    assert policy.cycle_time_ms("mtp", 2.0, 2.02) is None
    assert abs(policy.cycle_time_ms("mtp", 2.025, 2.04) - 20) < 1e-6


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
# Chunk-local mRoPE ownership
# ---------------------------------------------------------------------------


class TestChunkedPrefillMRoPE:
    def test_text_prefill_rebinds_delta_after_interleaved_cleanup(self):
        class MRoPERecordingModel(_RecordingModel):
            _uses_mrope = True

            def __init__(self):
                super().__init__("vlm")
                self.batch_deltas = None
                self.delta_history = []

            def set_text_prefill_rope_delta(self, delta):
                self.batch_deltas = mx.array([delta])
                self.delta_history.append([delta])

            def set_batch_rope_deltas(self, deltas):
                raise AssertionError("text prefill must use the bounded binder")

            def __call__(self, tokens, cache=None):
                assert self.batch_deltas is not None
                super().__call__(tokens, cache=cache)

        model = MRoPERecordingModel()
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2
        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                prefill_step_size=4,
                chunked_prefill=True,
                paged_cache_block_size=0,
            ),
        )
        request = _make_request("mrope-interleaved", n_tokens=9)
        request.rope_deltas = 7.0
        state = _make_prefill_state(scheduler, request, n_remaining=8)

        with patch("omlx.scheduler._sync_and_clear_cache"):
            assert not scheduler._step_prefill_chunk(state)
            # Reproduce a concurrent request's completion cleanup.
            model.batch_deltas = None
            assert scheduler._step_prefill_chunk(state)

        assert model.chunk_lengths == [4, 4]
        assert model.delta_history == [[7.0], [7.0]]

    def test_text_prefill_chunk_records_text_positions_proof_on_request(self):
        """Each text chunk proves the request text-only; insert() later marks its batch uid."""

        class MRoPEMarkingModel(_RecordingModel):
            _uses_mrope = True

            def __init__(self):
                super().__init__("vlm")
                self.batch_deltas = None
                self.marked = []

            def set_text_prefill_rope_delta(self, delta):
                self.batch_deltas = mx.array([delta])

            def mark_text_positions(self, uid):
                self.marked.append(uid)

            def __call__(self, tokens, cache=None):
                super().__call__(tokens, cache=cache)

        model = MRoPEMarkingModel()
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2
        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                prefill_step_size=4,
                chunked_prefill=True,
                paged_cache_block_size=0,
            ),
        )
        request = _make_request("mrope-marked", n_tokens=9)
        request.rope_deltas = 0.0
        scheduler.request_id_to_uid[request.request_id] = 42
        state = _make_prefill_state(scheduler, request, n_remaining=8)

        with patch("omlx.scheduler._sync_and_clear_cache"):
            assert not scheduler._step_prefill_chunk(state)
            assert scheduler._step_prefill_chunk(state)

        # The prefill-time uid is a temporary one (id(request)); the chunk only
        # records the proof on the request, and insert() marks the batch uid.
        assert request.text_positions_proven is True
        assert model.marked == []

    def test_mock_request_without_rope_delta_uses_text_default(self):
        """Legacy/minimal request doubles retain the canonical text delta."""

        class MRoPERecordingModel(_RecordingModel):
            _uses_mrope = True

            def __init__(self):
                super().__init__("vlm")
                self.delta_history = []

            def set_text_prefill_rope_delta(self, delta):
                self.delta_history.append([delta])

        model = MRoPERecordingModel()
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2
        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                prefill_step_size=4,
                chunked_prefill=True,
                paged_cache_block_size=0,
            ),
        )
        request = SimpleNamespace(request_id="mock-without-rope-delta")
        state = _PrefillState(
            request=request,
            cache=[],
            tokens_remaining=mx.zeros((1, 4), dtype=mx.int32),
            last_token=[99],
            tokens_processed=0,
            base_size=0,
            emitted_boundaries={},
            boundary_enabled=False,
            block_size=0,
            total_length=5,
        )

        with patch("omlx.scheduler._sync_and_clear_cache"):
            assert scheduler._step_prefill_chunk(state)

        assert model.chunk_lengths == [4]
        assert model.delta_history == [[0.0]]


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


# ---------------------------------------------------------------------------
# _schedule_waiting(): specprefill guard defers everything while one is active
# ---------------------------------------------------------------------------


class TestScheduleWaitingSpecPrefillGuard:
    def test_second_specprefill_deferred_while_one_active(self):
        """A second specprefill request must wait for the active one (#766).

        Admitting it would replace the live _OffsetAdjustedRoPE on the shared
        model and corrupt the remaining decode of the active request.
        """
        sched = _make_scheduler(chunked_prefill=False)
        sched._specprefill_active_request_id = "active-req"

        req = _make_request("spec-2", n_tokens=10)
        req.specprefill_indices = mx.array([0, 2, 4])
        sched.add_request(req)

        with patch.object(sched, "_do_external_prefill") as mock_ep:
            scheduled, rejected = sched._schedule_waiting()

        mock_ep.assert_not_called()
        assert scheduled == []
        assert rejected == []
        assert req in sched.waiting

    def test_normal_request_deferred_while_specprefill_active(self):
        """Non-specprefill requests keep deferring while one is active."""
        sched = _make_scheduler(chunked_prefill=False)
        sched._specprefill_active_request_id = "active-req"

        req = _make_request("normal", n_tokens=10)
        sched.add_request(req)

        with patch.object(sched, "_do_external_prefill") as mock_ep:
            scheduled, rejected = sched._schedule_waiting()

        mock_ep.assert_not_called()
        assert scheduled == []
        assert rejected == []
        assert req in sched.waiting


# ---------------------------------------------------------------------------
# Prefill error paths must drain the ENGINE stream before clearing the cache
# ---------------------------------------------------------------------------


class TestPrefillCleanupUsesEngineStream:
    """Every prefill error/rejection path must pass the per-engine stream to
    _sync_and_clear_cache, like its sibling success/abort branches do.

    mx.clear_cache() can release Metal buffers that in-flight command buffers
    still reference (#300), so the clear must be preceded by a drain of the
    stream that carried the work. The drain only covers the stream it is given:
    an mlx ThreadLocalStream resolves to a *different* concrete mx.Stream per
    calling thread, so a no-argument call drains mlx-lm's generation_stream and
    the calling thread's default stream -- never the engine stream the prefill
    forward and the BatchGenerator's async_eval actually ran on.
    """

    @staticmethod
    def _engine_scheduler(**kwargs) -> Scheduler:
        """Scheduler with a per-engine stream, the way EngineCore builds it."""
        sched = _make_scheduler(**kwargs)
        sched._stream = mx.new_thread_local_stream(mx.default_device())
        assert sched._stream is not _default_generation_stream
        return sched

    @staticmethod
    def _capacity_error(request_id: str) -> PrefillMemoryExceededError:
        return PrefillMemoryExceededError(
            message="Prefill context too large for available memory",
            request_id=request_id,
            estimated_bytes=123,
            limit_bytes=100,
        )

    @staticmethod
    def _recorder() -> tuple[list, object]:
        """Patch the module-level helper so calls record the stream argument."""
        streams: list = []
        return streams, patch(
            "omlx.scheduler._sync_and_clear_cache",
            side_effect=lambda stream=None: streams.append(stream),
        )

    def _assert_engine_stream(self, streams: list, sched: Scheduler) -> None:
        assert streams, "prefill cleanup did not clear the Metal buffer cache"
        assert all(s is sched._stream for s in streams), (
            "prefill cleanup cleared the cache without draining the engine "
            f"stream: {streams!r} != {sched._stream!r}"
        )

    def _queued_chunked_request(self, sched: Scheduler) -> Request:
        req = _make_request("r1")
        sched.requests[req.request_id] = req
        sched.prefilling.append(req)
        sched._prefill_states[req.request_id] = _make_prefill_state(sched, req)
        return req

    # _advance_chunked_prefills(): in-flight chunk

    def test_advance_chunked_capacity_rejection_drains_engine_stream(self):
        sched = self._engine_scheduler()
        req = self._queued_chunked_request(sched)
        streams, recording = self._recorder()

        with (
            recording,
            patch.object(
                sched,
                "_step_prefill_chunk",
                side_effect=self._capacity_error(req.request_id),
            ),
        ):
            rejected: list = []
            sched._advance_chunked_prefills([], rejected)

        assert len(rejected) == 1
        self._assert_engine_stream(streams, sched)

    def test_advance_chunked_runtime_error_drains_engine_stream(self):
        sched = self._engine_scheduler()
        self._queued_chunked_request(sched)
        streams, recording = self._recorder()

        with (
            recording,
            patch.object(
                sched, "_step_prefill_chunk", side_effect=RuntimeError("kernel panic")
            ),
        ):
            rejected: list = []
            sched._advance_chunked_prefills([], rejected)

        assert len(rejected) == 1
        self._assert_engine_stream(streams, sched)

    # _schedule_waiting(): first chunk of a chunked prefill

    def test_first_chunk_capacity_rejection_drains_engine_stream(self):
        sched = self._engine_scheduler()
        req = _make_request("r1", n_tokens=10)  # > step_size + 1 → chunked fork
        sched.add_request(req)
        streams, recording = self._recorder()

        with (
            recording,
            patch.object(
                sched, "_begin_prefill", return_value=_make_prefill_state(sched, req)
            ),
            patch.object(
                sched,
                "_step_prefill_chunk",
                side_effect=self._capacity_error(req.request_id),
            ),
        ):
            _, rejected = sched._schedule_waiting()

        assert len(rejected) == 1
        self._assert_engine_stream(streams, sched)

    def test_first_chunk_runtime_error_drains_engine_stream(self):
        sched = self._engine_scheduler()
        req = _make_request("r1", n_tokens=10)
        sched.add_request(req)
        streams, recording = self._recorder()

        with (
            recording,
            patch.object(
                sched, "_begin_prefill", return_value=_make_prefill_state(sched, req)
            ),
            patch.object(
                sched, "_step_prefill_chunk", side_effect=RuntimeError("kernel panic")
            ),
        ):
            _, rejected = sched._schedule_waiting()

        assert len(rejected) == 1
        self._assert_engine_stream(streams, sched)

    # _schedule_waiting(): non-chunked full prefill

    def test_non_chunked_capacity_rejection_drains_engine_stream(self):
        sched = self._engine_scheduler()
        req = _make_request("r1", n_tokens=3)  # short → normal prefill path
        sched.add_request(req)
        streams, recording = self._recorder()

        with (
            recording,
            patch.object(
                sched,
                "_do_external_prefill",
                side_effect=self._capacity_error(req.request_id),
            ),
        ):
            _, rejected = sched._schedule_waiting()

        assert len(rejected) == 1
        self._assert_engine_stream(streams, sched)

    def test_non_chunked_runtime_error_drains_engine_stream(self):
        sched = self._engine_scheduler()
        req = _make_request("r1", n_tokens=3)
        sched.add_request(req)
        streams, recording = self._recorder()

        with (
            recording,
            patch.object(
                sched, "_do_external_prefill", side_effect=RuntimeError("kernel panic")
            ),
        ):
            _, rejected = sched._schedule_waiting()

        assert len(rejected) == 1
        self._assert_engine_stream(streams, sched)


def test_step_prefill_chunk_announces_the_next_chunk_to_the_model():
    """Each chunk step tells a model with prefetch_ple which tokens follow, so it can gather ahead."""

    class LookaheadModel(_RecordingModel):
        def __init__(self):
            super().__init__("vlm")
            self.seen = []

        def prefetch_ple(self, next_ids, current_ids):
            self.seen.append((next_ids.tolist()[0], current_ids.tolist()[0]))

    model = LookaheadModel()
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(prefill_step_size=4, chunked_prefill=True, paged_cache_block_size=0),
    )
    request = _make_request("lookahead", n_tokens=11)
    state = _make_prefill_state(scheduler, request, n_remaining=10)
    state.tokens_remaining = mx.arange(10, 20, dtype=mx.int32)[None]
    with patch("omlx.scheduler._sync_and_clear_cache"):
        while not scheduler._step_prefill_chunk(state):
            pass
    assert model.chunk_lengths == [4, 4, 2]
    assert model.seen == [([10, 11, 12, 13], []), ([14, 15, 16, 17], [10, 11, 12, 13]), ([18, 19], [14, 15, 16, 17])]


def test_external_prefill_announces_the_next_chunk_to_the_model():
    """The non-chunked prefill loop announces the next chunk too; the last chunk announces nothing."""
    import types

    class LookaheadModel(_RecordingModel):
        def __init__(self):
            super().__init__("vlm")
            self.seen = []

        def prefetch_ple(self, next_ids, current_ids):
            self.seen.append((next_ids.tolist()[0], current_ids.tolist()[0]))

    model = LookaheadModel()
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(prefill_step_size=4, paged_cache_block_size=0),
    )
    tokens = list(range(10, 21))  # 10 prefill tokens, the last token goes to the batch generator
    request = _make_request("lookahead-external", n_tokens=11)
    cache = [types.SimpleNamespace(state=mx.array([0]))]
    with patch("omlx.scheduler._sync_and_clear_cache"):
        scheduler._do_external_prefill(request, tokens, cache)
    assert model.chunk_lengths == [4, 4, 2]
    assert model.seen == [([10, 11, 12, 13], []), ([14, 15, 16, 17], [10, 11, 12, 13]), ([18, 19], [14, 15, 16, 17])]


# ---------------------------------------------------------------------------
# Packed prefill
# ---------------------------------------------------------------------------


def _tiny_qwen3_5():
    text = qwen3_5_config.TextConfig(
        model_type="qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=512,
        head_dim=8,
        rope_parameters={
            "type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
        },
    )
    return qwen3_5_language.LanguageModel(text), SimpleNamespace(
        model_type="qwen3_5", text_config=text
    )


def _tiny_qwen4_exp():
    mlx_vlm_qwen4_exp_compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models import qwen4_exp

    text = qwen4_exp.TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        hc_count=2,
        hc_lowrank=8,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )
    vision = qwen4_exp.VisionConfig(
        model_type="qwen4_exp",
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=32,
        num_heads=4,
        patch_size=14,
        in_channels=3,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    config = qwen4_exp.ModelConfig(
        text_config=text,
        vision_config=vision,
        model_type="qwen4_exp",
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
        vocab_size=64,
    )
    return qwen4_exp.Model(config).language_model, config


def _tiny_glm5_next():
    apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models import glm5_next
    from mlx_vlm.models.glm5_next import language

    text = glm5_next.TextConfig(
        model_type="glm5_next_text",
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=32,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        v_head_dim=16,
        qk_nope_head_dim=16,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        # Small enough that the 7- and 21-token prefixes take the sparse path.
        index_topk=8,
        index_head_dim=16,
        index_n_heads=2,
        layer_types=[
            "linear_attention",
            "deepseek_sparse_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_kpool=2,
        hc_mult=2,
        hc_sinkhorn_iters=5,
    )
    return language.LanguageModel(text), SimpleNamespace(
        model_type="glm5_next", text_config=text
    )


def _tiny_mlx_lm_qwen3_5_moe():

    mx.random.seed(7)
    model = qwen3_5_moe.Model(
        qwen3_5_moe.ModelArgs.from_dict(
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "shared_expert_intermediate_size": 16,
                "moe_intermediate_size": 16,
                "vocab_size": 64,
                "max_position_embeddings": 512,
                "rope_parameters": {
                    "type": "default",
                    "mrope_section": [2, 1, 1],
                    "rope_theta": 10000,
                    "partial_rotary_factor": 1.0,
                },
            }
        )
    )
    mx.eval(model.parameters())
    return model


def _tiny_hy_v3():
    apply_hy_v3_patch()
    from mlx_lm.models import hy_v3

    mx.random.seed(7)
    model = hy_v3.Model(
        hy_v3.ModelArgs(
            model_type="hy_v3",
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            expert_hidden_dim=16,
            first_k_dense_replace=1,
            rms_norm_eps=1e-6,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )
    )
    mx.eval(model.parameters())
    return model


def _tiny_vlm_adapter(builder):

    mx.random.seed(7)
    language_model, config = builder()
    mx.eval(language_model.parameters())
    return VLMModelAdapter(
        SimpleNamespace(language_model=language_model, config=config)
    )


def _single_chunk(model, cache, tokens):

    _bind_text_prefill_rope_delta(model, 0.0)
    kwargs = (
        {"skip_lm_head": True} if getattr(model, "supports_skip_lm_head", False) else {}
    )
    model(mx.array(tokens, dtype=mx.int32)[None], cache=cache, **kwargs)
    mx.eval([c.state for c in cache])


def _cache_with_prefix(model, tokens):

    cache = make_prompt_cache(model)
    if tokens:
        _single_chunk(model, cache, tokens)
    return cache


def _cache_leaves(cache):
    def flatten(value):
        if isinstance(value, mx.array):
            return [value]
        if isinstance(value, (list, tuple)):
            return [leaf for item in value for leaf in flatten(item)]
        return []

    return flatten([c.state for c in cache])


def _assert_same_cache(actual, expected):
    actual, expected = _cache_leaves(actual), _cache_leaves(expected)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a.shape == b.shape
        assert mx.array_equal(a, b).item()


@pytest.mark.parametrize(
    "build",
    [
        lambda: _tiny_vlm_adapter(_tiny_qwen3_5),
        lambda: _tiny_vlm_adapter(_tiny_qwen4_exp),
        lambda: _tiny_vlm_adapter(_tiny_glm5_next),
        _tiny_mlx_lm_qwen3_5_moe,
        _tiny_hy_v3,
    ],
    ids=[
        "qwen3_5",
        "qwen4_exp",
        "glm5_next",
        "mlx_lm_qwen3_5_moe",
        "hy_v3",
    ],
)
def test_packed_prefill_matches_single_request_chunks(build):
    """Rows at different offsets and lengths match their own single forwards."""

    model = build()
    assert install_packed_prefill(model)
    sequences = [list(range(3, 30)), list(range(5, 40)), list(range(1, 50))]
    rows, expected = [], []
    for index, (sequence, prefix, length) in enumerate(
        zip(sequences, (0, 7, 21), (5, 11, 3))
    ):
        chunk = sequence[prefix : prefix + length]
        reference = _cache_with_prefix(model, sequence[:prefix])
        _single_chunk(model, reference, chunk)
        expected.append(reference)
        rows.append(
            PackedRow(
                request_id=f"row-{index}",
                tokens=mx.array(chunk, dtype=mx.int32)[None],
                cache=_cache_with_prefix(model, sequence[:prefix]),
            )
        )
    run_packed_prefill(model, rows)
    for row, reference in zip(rows, expected):
        _assert_same_cache(row.cache, reference)


def _glm5_next_router():
    from mlx_vlm.models.glm5_next import language

    return _tiny_vlm_adapter(_tiny_glm5_next), language.Glm5NextMoEGate


def _hy_v3_row_op(name):
    def build():
        model = _tiny_hy_v3()
        from mlx_lm.models import hy_v3

        return model, getattr(hy_v3, name)

    return build


@pytest.mark.parametrize(
    "build",
    [_glm5_next_router, _hy_v3_row_op("MoEGate"), _hy_v3_row_op("MLP")],
    ids=["glm5_next_router", "hy_v3_router", "hy_v3_shared_mlp"],
)
def test_packed_row_ops_run_once_per_row(monkeypatch, build):
    """Ops that pick kernels by row count see each row alone."""

    model, op = build()
    call = op.__call__
    seen = []

    def record(self, x):
        seen.append(x.shape[1])
        return call(self, x)

    monkeypatch.setattr(op, "__call__", record)
    assert install_packed_prefill(model)
    rows = [
        PackedRow(f"row-{n}", mx.array([list(range(3, 3 + n))], dtype=mx.int32), cache)
        for n, cache in (
            (5, _cache_with_prefix(model, [])),
            (9, _cache_with_prefix(model, [])),
        )
    ]
    run_packed_prefill(model, rows)
    assert seen and set(seen) == {5, 9}


def test_packed_min_row_tokens_keeps_exact_moe_rows_on_the_sorted_expert_gather():

    def adapter(model_type, **args):
        language_model = SimpleNamespace(args=SimpleNamespace(**args))
        return SimpleNamespace(
            model_type=model_type, _language_model=language_model, _vlm_model=None
        )

    # mlx's segmented gather needs 4 rows per expert: 4 * 288 / 8 routes.
    glm = adapter("glm5_next", n_routed_experts=288, num_experts_per_tok=8)
    assert packed_min_row_tokens(glm) == 144
    # Qwen MoE rows differ from single chunks at any length; they keep the gain.
    qwen = adapter("qwen4_exp", num_experts=512, num_experts_per_tok=10)
    assert packed_min_row_tokens(qwen) == 64
    assert packed_min_row_tokens(adapter("qwen3_5")) == 64
    # mlx-lm models keep their config on ``args``.
    hy = SimpleNamespace(
        args=SimpleNamespace(model_type="hy_v3", num_experts=192, num_experts_per_tok=8)
    )
    assert packed_min_row_tokens(hy) == 96


def test_packed_rows_reject_unregistered_cache_access():

    batch = PackedBatch([PackedRow("a", mx.zeros((1, 2), dtype=mx.int32), [None])])
    rows = PackedRows(batch, [None])
    with pytest.raises(AttributeError, match="update_and_fetch"):
        _ = rows.update_and_fetch
    assert rows.offset == 0 and rows.left_padding is None


def _make_packed_scheduler(step_size: int = 16):
    model = _tiny_vlm_adapter(_tiny_qwen3_5)
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=8,
            prefill_step_size=step_size,
            chunked_prefill=True,
            paged_cache_block_size=0,
        ),
    )
    scheduler._qwen35_prefill_floor = 0
    # Tiny rows stand in for full-size chunks.
    scheduler._packed_min_row_tokens = 1
    mock_bg = MagicMock()
    mock_bg.insert.return_value = [42]
    scheduler.batch_generator = mock_bg
    scheduler._current_sampler_params = ()
    return scheduler


def _stage_prefill(scheduler, request_id: str, n_tokens: int, request=None):
    request = request or _make_request(request_id, n_tokens=n_tokens)
    request.prompt_token_ids = list(range(3, 3 + n_tokens))
    request.remaining_tokens = list(request.prompt_token_ids)
    scheduler.requests[request_id] = request
    state = scheduler._begin_prefill(request, request.prompt_token_ids, None)
    state.sampler = MagicMock()
    state.sm = MagicMock()
    state.per_row_lps = []
    scheduler.prefilling.append(request)
    scheduler._prefill_states[request_id] = state
    return request, state


def _record_packed_forwards(monkeypatch):

    forwards = []
    run = scheduler_module.run_packed_prefill

    def record(model, rows):
        forwards.append([(row.request_id, int(row.tokens.shape[1])) for row in rows])
        return run(model, rows)

    monkeypatch.setattr(scheduler_module, "run_packed_prefill", record)
    return forwards


def _record_inserted_caches(monkeypatch, scheduler):
    inserted = {}
    insert = scheduler._insert_prefilled_request

    def record(request, state, scheduled):
        inserted[request.request_id] = state.cache
        return insert(request, state, scheduled)

    monkeypatch.setattr(scheduler, "_insert_prefilled_request", record)
    return inserted


def _advance(scheduler):
    scheduled, rejected = [], []
    with patch("omlx.scheduler._sync_and_clear_cache"):
        scheduler._advance_chunked_prefills(scheduled, rejected)
    return [r.request_id for r in scheduled], rejected


def test_packed_prefill_keeps_the_head_chunk_and_fills_the_forward(monkeypatch):
    scheduler = _make_packed_scheduler()
    forwards = _record_packed_forwards(monkeypatch)
    inserted = _record_inserted_caches(monkeypatch, scheduler)
    for request_id, n_tokens in (("a", 5), ("b", 7), ("c", 7)):
        _stage_prefill(scheduler, request_id, n_tokens)
    scheduled, rejected = _advance(scheduler)
    # Rows keep the chunks they would run alone inside the head's 16 tokens.
    assert forwards == [[("a", 4), ("b", 6), ("c", 6)]]
    assert scheduled == ["a", "b", "c"] and rejected == []
    model = scheduler.model
    for request_id, n_tokens in (("a", 5), ("b", 7), ("c", 7)):
        expected = _cache_with_prefix(model, list(range(3, 2 + n_tokens)))
        _assert_same_cache(inserted[request_id], expected)
    assert scheduler.get_stats()["packed_prefill"]["rows"] == 3


def test_packed_forwards_leave_the_single_chunk_rate_alone(monkeypatch):
    """The contended cap prices single-row chunks, which run slower per token."""

    monkeypatch.setattr(scheduler_module, "_CONTENDED_CHUNK_FLOOR", 1)
    scheduler = _make_packed_scheduler()
    forwards = _record_packed_forwards(monkeypatch)
    _stage_prefill(scheduler, "a", 5)
    _stage_prefill(scheduler, "b", 7)
    _advance(scheduler)
    assert forwards == [[("a", 4), ("b", 6)]]
    assert scheduler._prefill_tps_best is None
    _stage_prefill(scheduler, "c", 7)
    _advance(scheduler)
    assert scheduler._prefill_tps_best is not None


def test_packed_prefill_never_cuts_a_companion_chunk(monkeypatch):
    scheduler = _make_packed_scheduler()
    forwards = _record_packed_forwards(monkeypatch)
    _stage_prefill(scheduler, "a", 9)
    _stage_prefill(scheduler, "b", 31)
    _stage_prefill(scheduler, "c", 6)
    _advance(scheduler)
    # b's 16-token chunk does not fit the 8 tokens left; c's whole chunk does.
    assert forwards == [[("a", 8), ("c", 5)]]


def test_contended_packed_prefill_finishes_the_shortest_rows_first(monkeypatch):
    scheduler = _make_packed_scheduler(step_size=512)
    forwards = _record_packed_forwards(monkeypatch)
    monkeypatch.setattr(Scheduler, "_contended_prefill_cap", lambda self: 192)
    for request_id, n_tokens in (("a", 201), ("b", 41), ("c", 81), ("d", 301)):
        _stage_prefill(scheduler, request_id, n_tokens)
    scheduled, _ = _advance(scheduler)
    # Decode waits: b and c finish inside one 192-token chunk, the head keeps
    # a grid step, and the longest row waits.
    assert forwards == [[("a", 64), ("b", 40), ("c", 80)]]
    assert scheduled == ["b", "c"]


@pytest.mark.parametrize(
    "head_tokens, head_chunk",
    [
        # The head would not finish: it keeps its grid step only.
        (301, 64),
        # The head would finish, but its rest is longer than b.
        (151, 64),
        # The head's rest is shorter than b, so both finish together.
        (101, 100),
    ],
)
def test_contended_rows_that_finish_do_not_wait_for_a_longer_head_chunk(
    monkeypatch, head_tokens, head_chunk
):
    scheduler = _make_packed_scheduler(step_size=512)
    forwards = _record_packed_forwards(monkeypatch)
    monkeypatch.setattr(Scheduler, "_contended_prefill_cap", lambda self: 192)
    _stage_prefill(scheduler, "a", head_tokens)
    _stage_prefill(scheduler, "b", 41)
    _advance(scheduler)
    assert forwards == [[("a", head_chunk), ("b", 40)]]


def test_packed_prefill_keeps_short_chunks_out_of_the_pack(monkeypatch):
    scheduler = _make_packed_scheduler()
    scheduler._packed_min_row_tokens = 6
    forwards = _record_packed_forwards(monkeypatch)
    for request_id, n_tokens in (("a", 9), ("b", 5), ("c", 8)):
        _stage_prefill(scheduler, request_id, n_tokens)
    # b's 4-token chunk would take small-row kernels in a pack: it runs alone,
    # first, so it does not wait behind a. a and c then share one forward.
    assert _advance(scheduler) == (["b", "a", "c"], [])
    assert forwards == [[("a", 8), ("c", 7)]]


def test_contended_packed_rows_get_at_least_the_packable_minimum(monkeypatch):
    scheduler = _make_packed_scheduler(step_size=512)
    scheduler._packed_min_row_tokens = 70
    forwards = _record_packed_forwards(monkeypatch)
    monkeypatch.setattr(Scheduler, "_contended_prefill_cap", lambda self: 256)
    for request_id, n_tokens in (("a", 201), ("b", 41), ("c", 81), ("d", 301)):
        _stage_prefill(scheduler, request_id, n_tokens)
    # b's 40 tokens are too short to pack, so b runs alone first.
    assert _advance(scheduler) == (["b"], [])
    _advance(scheduler)
    # The head keeps two grid steps (>= 70), and only c fits the 128 left.
    assert forwards == [[("a", 128), ("c", 80)]]


def test_late_request_joins_the_next_packed_forward(monkeypatch):
    scheduler = _make_packed_scheduler()
    # No decode step runs here, so fairness would hold prefills after "a".
    scheduler._decode_fairness = False
    forwards = _record_packed_forwards(monkeypatch)
    _stage_prefill(scheduler, "c", 40)
    _advance(scheduler)
    _advance(scheduler)
    # d arrives while c is mid-prefill and rides in c's last chunk.
    _, d_state = _stage_prefill(scheduler, "d", 4)
    _advance(scheduler)
    # c ran two full 16-token chunks alone before d arrived.
    assert forwards == [[("c", 7), ("d", 3)]]
    assert "d" in scheduler.running
    _assert_same_cache(
        d_state.cache, _cache_with_prefix(scheduler.model, list(range(3, 6)))
    )


def test_failed_packed_forward_requeues_rows_and_disables_packing(monkeypatch):

    scheduler = _make_packed_scheduler()

    def fail(model, rows):
        raise ValueError("unsupported cache access")

    monkeypatch.setattr(scheduler_module, "run_packed_prefill", fail)
    for request_id, n_tokens in (("a", 5), ("b", 7)):
        _stage_prefill(scheduler, request_id, n_tokens)
    assert _advance(scheduler) == ([], [])
    assert not scheduler.prefilling and not scheduler._prefill_states
    assert [r.request_id for r in scheduler.waiting] == ["a", "b"]
    assert all(r.prefill_oom_retries == 0 for r in scheduler.waiting)
    assert scheduler._packed_prefill_disabled is not None
    assert not Scheduler._packed_prefill_ready(scheduler)


def test_rows_of_a_pack_that_ran_out_of_memory_retry_alone(monkeypatch):

    scheduler = _make_packed_scheduler()
    run = scheduler_module.run_packed_prefill
    packs = []

    def fail_first_pack(model, rows):
        packs.append([row.request_id for row in rows])
        if len(packs) == 1:
            raise MemoryError("pack does not fit")
        return run(model, rows)

    monkeypatch.setattr(scheduler_module, "run_packed_prefill", fail_first_pack)
    sizes = {"a": 5, "b": 7}
    requests = [_stage_prefill(scheduler, rid, n)[0] for rid, n in sizes.items()]
    assert _advance(scheduler) == ([], [])
    assert list(scheduler.waiting) == requests
    # A memory failure keeps packing for other requests.
    assert Scheduler._packed_prefill_ready(scheduler)
    scheduler.waiting.clear()
    states = [
        _stage_prefill(scheduler, r.request_id, sizes[r.request_id], request=r)[1]
        for r in requests
    ]
    assert _advance(scheduler) == (["a", "b"], [])
    assert packs == [["a", "b"]]
    assert [state.tokens_processed for state in states] == [4, 6]


def test_rows_of_a_pack_out_of_memory_retries_fail_cleanly(monkeypatch):
    scheduler = _make_packed_scheduler()

    def fail(model, rows):
        raise MemoryError("pack does not fit")

    monkeypatch.setattr(scheduler_module, "run_packed_prefill", fail)
    for request_id, n_tokens in (("a", 5), ("b", 7)):
        request, _ = _stage_prefill(scheduler, request_id, n_tokens)
        request.prefill_oom_retries = scheduler._MAX_PREFILL_OOM_RETRIES
    scheduled, rejected = _advance(scheduler)
    assert scheduled == [] and not scheduler.waiting
    assert sorted(output.request_id for output in rejected) == ["a", "b"]
    assert {output.finish_reason for output in rejected} == {"error"}


@pytest.mark.parametrize(
    "routes, priced_gathered", [((True, True), True), ((True, False), False)]
)
def test_packed_forward_with_a_dense_row_is_priced_dense(
    monkeypatch, routes, priced_gathered
):
    scheduler = _make_packed_scheduler()
    plans = []
    for request_id, gathered in zip(("a", "b"), routes):
        _, state = _stage_prefill(scheduler, request_id, 9)
        plan = scheduler._plan_prefill_chunk(state, guarded=False)
        plan.gathered_core = gathered
        plans.append(plan)
    priced = []

    def bound(self, n_tokens, kv_len, *, gathered_core=False):
        priced.append(gathered_core)
        return 0

    monkeypatch.setattr(Scheduler, "_adaptive_chunk_size", lambda self, n, **_: n)
    monkeypatch.setattr(
        Scheduler, "_prefill_abort_description", lambda self: (None, 1 << 40, None)
    )
    monkeypatch.setattr(Scheduler, "_current_usage_bytes", lambda self: 0)
    monkeypatch.setattr(Scheduler, "_admission_transient_bound", bound)
    assert scheduler._packed_prefill_fits(plans)
    assert priced == [priced_gathered]


def test_a_companion_that_fails_to_plan_leaves_the_head_alone(monkeypatch):
    scheduler = _make_packed_scheduler()
    forwards = _record_packed_forwards(monkeypatch)
    reserve = Scheduler._reserve_prefill_capacity

    def fail_for_b(self, cache, tokens, request_id):
        if request_id == "b":
            raise RuntimeError("cannot reserve b")
        return reserve(self, cache, tokens, request_id)

    monkeypatch.setattr(Scheduler, "_reserve_prefill_capacity", fail_for_b)
    _stage_prefill(scheduler, "a", 9)
    _stage_prefill(scheduler, "b", 7)
    scheduled, rejected = _advance(scheduler)
    # b's error surfaces on its own turn; a still prefills.
    assert scheduled == ["a"]
    assert [output.request_id for output in rejected] == ["b"]
    assert forwards == []


def test_contended_head_keeps_its_chunk_when_its_companions_drop_out(monkeypatch):
    scheduler = _make_packed_scheduler(step_size=512)
    forwards = _record_packed_forwards(monkeypatch)
    monkeypatch.setattr(Scheduler, "_contended_prefill_cap", lambda self: 192)
    monkeypatch.setattr(Scheduler, "_packed_prefill_fits", lambda self, plans: False)
    _, a_state = _stage_prefill(scheduler, "a", 301)
    _stage_prefill(scheduler, "b", 41)
    _advance(scheduler)
    assert forwards == []
    assert a_state.tokens_processed == 192


def test_ane_prefill_on_the_wrapped_vlm_model_disables_packing():
    scheduler = _make_packed_scheduler()
    assert Scheduler._packed_prefill_ready(scheduler)
    scheduler.model._vlm_model._omlx_ane_mlp_prefill_count = 1
    assert not Scheduler._packed_prefill_ready(scheduler)


def test_insert_rollback_releases_drafter_rows(monkeypatch):
    scheduler = _make_packed_scheduler()
    drafter = MagicMock()
    monkeypatch.setattr(scheduler_module, "_block_drafter_for", lambda model: drafter)
    monkeypatch.setattr(
        scheduler_module,
        "_mark_text_positions",
        MagicMock(side_effect=RuntimeError("positions")),
    )
    request, state = _stage_prefill(scheduler, "a", 5)
    state.tokens_remaining = state.tokens_remaining[:, :0]
    with pytest.raises(RuntimeError, match="positions"):
        scheduler._insert_prefilled_request(request, state, [])
    drafter.release.assert_called_once_with([42])


def test_packed_prefill_drops_rows_that_do_not_fit(monkeypatch):
    scheduler = _make_packed_scheduler()
    forwards = _record_packed_forwards(monkeypatch)
    monkeypatch.setattr(Scheduler, "_packed_prefill_fits", lambda self, plans: False)
    _stage_prefill(scheduler, "a", 5)
    _stage_prefill(scheduler, "b", 7)
    # Neither runs packed. "a" ran uncontended, so b follows in the same step,
    # as unpacked prefills do.
    assert _advance(scheduler) == (["a", "b"], [])
    assert forwards == []


@pytest.mark.parametrize(
    "a_tokens, scheduled",
    [
        # a's 16-token chunk fills the step.
        (41, []),
        # a finishes and then decodes, but its forward ran uncontended.
        (9, ["a"]),
    ],
)
def test_lone_heads_without_contention_let_later_prefills_advance(
    monkeypatch, a_tokens, scheduled
):
    scheduler = _make_packed_scheduler()
    forwards = _record_packed_forwards(monkeypatch)
    _stage_prefill(scheduler, "a", a_tokens)
    _, b_state = _stage_prefill(scheduler, "b", 31)
    assert _advance(scheduler) == (scheduled, [])
    # b's 16-token chunk does not fit next to a's, yet b still advances.
    assert b_state.tokens_processed == 16
    assert forwards == []


def test_chunks_too_short_to_pack_do_not_wait_behind_a_long_head(monkeypatch):
    scheduler = _make_packed_scheduler()
    scheduler._packed_min_row_tokens = 6
    forwards = _record_packed_forwards(monkeypatch)
    order = []
    run = Scheduler._run_prefill_chunks

    def record(self, plans):
        order.append([plan.state.request.request_id for plan in plans])
        return run(self, plans)

    monkeypatch.setattr(Scheduler, "_run_prefill_chunks", record)
    _stage_prefill(scheduler, "a", 41)
    _stage_prefill(scheduler, "b", 5)
    assert _advance(scheduler) == (["b"], [])
    assert order == [["b"], ["a"]] and forwards == []


def test_packing_steps_aside_when_a_contended_chunk_holds_one_row(monkeypatch):
    scheduler = _make_packed_scheduler(step_size=512)
    scheduler._packed_min_row_tokens = 100
    forwards = _record_packed_forwards(monkeypatch)
    monkeypatch.setattr(Scheduler, "_contended_prefill_cap", lambda self: 192)
    _, a_state = _stage_prefill(scheduler, "a", 301)
    _stage_prefill(scheduler, "b", 121)
    # A 128-token head floor and a 100-token row exceed the 192-token cap, so
    # both prefills take one unpacked chunk each, as without packing.
    assert _advance(scheduler) == (["b"], [])
    assert forwards == []
    assert a_state.tokens_processed == 192


def test_packed_rows_emit_their_own_boundary_snapshots(monkeypatch):
    scheduler = _make_packed_scheduler()
    forwards = _record_packed_forwards(monkeypatch)
    emitted = []
    monkeypatch.setattr(
        scheduler,
        "_emit_prefill_boundary_snapshot",
        lambda request, cache, total: emitted.append(
            (request.request_id, total, cache)
        ),
    )
    states = {}
    for request_id, n_tokens in (("a", 6), ("b", 7), ("c", 31)):
        _, state = _stage_prefill(scheduler, request_id, n_tokens)
        state.boundary_enabled = True
        state.block_size = 4
        states[request_id] = state
    _advance(scheduler)
    assert forwards == [[("a", 4), ("b", 4), ("c", 4)]]
    assert [(rid, total) for rid, total, _ in emitted] == [("a", 4), ("b", 4), ("c", 4)]
    assert all(cache is states[rid].cache for rid, _, cache in emitted)


@pytest.mark.parametrize(
    "chunked, contended, min_row, cap_fits, excluded, packed",
    [
        (True, False, 1, True, False, True),
        # Decode fairness chunks prompts under contention even when chunked
        # prefill is off, so those prompts pack too.
        (False, True, 1, True, False, True),
        (False, False, 1, True, False, False),
        (True, False, 64, True, False, False),
        # The resumable path holds the last token back: 2 tokens < 3.
        (True, False, 3, True, False, False),
        (True, False, 1, False, False, False),
        # A retry after a failed pack runs alone.
        (True, False, 1, True, True, False),
    ],
)
def test_schedule_waiting_admits_text_prompts_for_packed_prefill(
    monkeypatch, chunked, contended, min_row, cap_fits, excluded, packed
):
    sched = _make_scheduler(chunked_prefill=chunked, step_size=4)
    sched._packed_min_row_tokens = min_row
    monkeypatch.setattr(Scheduler, "_packed_prefill_ready", lambda self: True)
    monkeypatch.setattr(Scheduler, "_decode_contention", lambda self: contended)
    monkeypatch.setattr(Scheduler, "_packing_fits_contended_cap", lambda self: cap_fits)
    req = _make_request("short", n_tokens=3)
    req.packed_prefill_excluded = excluded
    sched.add_request(req)
    state = _make_prefill_state(sched, req, 2)
    aborted = _PrefillAbortedError([], 0)
    with (
        patch.object(sched, "_begin_prefill", return_value=state),
        patch.object(sched, "_step_prefill_chunk") as step,
        patch.object(sched, "_do_external_prefill", side_effect=aborted) as external,
    ):
        sched._schedule_waiting()
    step.assert_not_called()
    # Otherwise the prompt takes the regular path.
    assert (req in sched.prefilling) is packed
    assert (req.request_id in sched._prefill_states) is packed
    assert external.called is not packed


def test_packed_rows_activate_their_own_priming_slot_for_tail_snapshots(monkeypatch):

    scheduler = _make_packed_scheduler()
    _record_packed_forwards(monkeypatch)
    events = []
    monkeypatch.setattr(
        scheduler_module._mtp_priming,
        "activate_request",
        lambda model, request_id: events.append(("activate", request_id)),
    )
    monkeypatch.setattr(
        scheduler,
        "_emit_prefill_tail_snapshot",
        lambda request, cache, total: events.append(("tail", request.request_id)),
    )
    for request_id, n_tokens in (("a", 9), ("b", 9)):
        _, state = _stage_prefill(scheduler, request_id, n_tokens)
        state.boundary_enabled = True
        state.block_size = 64
        state.tail_at = 5
    _advance(scheduler)
    for request_id in ("a", "b"):
        # The prefill end (8 tokens) is off the 64-token grid: an end tail.
        scheduler._prefill_states[request_id].end_tail = True
    _advance(scheduler)
    tails = [i for i, event in enumerate(events) if event[0] == "tail"]
    assert [events[i][1] for i in tails] == ["a", "b", "a", "b"]
    assert all(events[i - 1] == ("activate", events[i][1]) for i in tails)
