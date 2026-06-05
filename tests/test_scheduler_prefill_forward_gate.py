# SPDX-License-Identifier: Apache-2.0
"""Tests for the forward-FRONT prefill memory gate (P0c).

The gate (_prefill_forward_gate) predicts a prefill chunk's peak memory
BEFORE running self.model(...) and raises RuntimeError when it would breach
the hard cap, so the request is aborted cleanly instead of the transient
landing on the Metal ceiling and kernel-panicking the machine. The legacy
chunk-END check only fires after the allocation has already happened, which
on Apple Silicon is too late.

Strategy: pure mocks, no model load. The discriminating assertion is that
when the predicted peak exceeds the cap the model forward is NOT called --
on pre-change code (no forward-front gate) the forward WOULD run.
"""

from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.request import Request, RequestStatus, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, _PrefillState

GB = 1024**3


# ---------------------------------------------------------------------------
# Direct unit tests of _prefill_forward_gate
# ---------------------------------------------------------------------------


def _gate_scheduler(
    *,
    hard_limit: int,
    recent_peak: int,
    estimate: int,
    margin: int,
    guard: bool = True,
    monitor: bool = True,
):
    """Build a bare Scheduler wired only for _prefill_forward_gate."""
    s = Scheduler.__new__(Scheduler)
    s._prefill_memory_guard = guard
    s._memory_hard_limit_bytes = hard_limit
    s._memory_recent_peak_bytes = recent_peak
    s._prefill_transient_margin_bytes = margin
    s.config = MagicMock(prefill_step_size=2048)
    if monitor:
        s.memory_monitor = MagicMock()
        s.memory_monitor.estimate_prefill_peak_bytes = MagicMock(
            return_value=estimate
        )
    else:
        s.memory_monitor = None
    return s


def _call_gate(s, chunk_tokens, *, instant):
    """Invoke the gate with patched instant memory probes."""
    with patch("omlx.scheduler.mx") as mock_mx, patch(
        "omlx.scheduler.get_phys_footprint", return_value=instant
    ):
        mock_mx.get_active_memory.return_value = instant
        s._prefill_forward_gate(
            chunk_tokens, request_id="rid-1", loop_label="external"
        )


class TestPrefillForwardGateUnit:
    """Direct tests of the gate predicate."""

    def test_raises_when_predicted_peak_exceeds_cap(self):
        """current(high-water) + estimate + margin > cap -> RuntimeError.

        Numbers chosen so the instant reading alone (low) + estimate would
        fit, but the high-water recent_peak + estimate + margin overflow.
        """
        hard = 107 * GB
        estimate = 2 * GB
        margin = 10 * GB
        instant = 50 * GB
        recent_peak = 96 * GB
        # Instant + estimate (no margin) fits; this is the trough the legacy
        # check could read.
        assert instant + estimate <= hard
        # High-water + estimate + margin overflows -> must refuse.
        assert recent_peak + estimate + margin > hard

        s = _gate_scheduler(
            hard_limit=hard,
            recent_peak=recent_peak,
            estimate=estimate,
            margin=margin,
        )
        with pytest.raises(RuntimeError, match="refused before forward"):
            _call_gate(s, 256, instant=instant)

    def test_passes_when_predicted_peak_fits(self):
        """current + estimate + margin <= cap -> no raise."""
        hard = 107 * GB
        s = _gate_scheduler(
            hard_limit=hard,
            recent_peak=80 * GB,
            estimate=2 * GB,
            margin=10 * GB,
        )
        # 80 + 2 + 10 = 92 < 107.
        _call_gate(s, 256, instant=80 * GB)  # must not raise

    def test_margin_is_what_tips_it_over(self):
        """Without the margin it would pass; the margin alone forces refusal.

        Pins that the margin term is actually applied (not dropped).
        """
        hard = 100 * GB
        estimate = 1 * GB
        instant = 90 * GB
        recent_peak = 90 * GB
        # current + estimate (no margin) = 91 < 100 -> would pass.
        assert recent_peak + estimate < hard
        # current + estimate + margin = 101 > 100 -> must refuse.
        margin = 10 * GB
        assert recent_peak + estimate + margin > hard

        s = _gate_scheduler(
            hard_limit=hard,
            recent_peak=recent_peak,
            estimate=estimate,
            margin=margin,
        )
        with pytest.raises(RuntimeError):
            _call_gate(s, 256, instant=instant)

        # Same setup, margin=0 -> passes (control).
        s0 = _gate_scheduler(
            hard_limit=hard,
            recent_peak=recent_peak,
            estimate=estimate,
            margin=0,
        )
        _call_gate(s0, 256, instant=instant)  # must not raise

    def test_uses_recent_peak_high_water_not_just_instant(self):
        """A mid-prefill trough in the instant reading must not mask the
        real footprint: recent_peak high + low instant still refuses."""
        hard = 107 * GB
        s = _gate_scheduler(
            hard_limit=hard,
            recent_peak=100 * GB,  # real footprint
            estimate=2 * GB,
            margin=10 * GB,
        )
        # Instant reads a trough at 50GB; without recent_peak it would pass.
        assert 50 * GB + 2 * GB + 10 * GB < hard
        with pytest.raises(RuntimeError):
            _call_gate(s, 256, instant=50 * GB)

    def test_noop_when_guard_off(self):
        s = _gate_scheduler(
            hard_limit=107 * GB,
            recent_peak=200 * GB,
            estimate=200 * GB,
            margin=10 * GB,
            guard=False,
        )
        _call_gate(s, 256, instant=200 * GB)  # guard off -> never raises

    def test_noop_when_hard_limit_unset(self):
        s = _gate_scheduler(
            hard_limit=0,
            recent_peak=200 * GB,
            estimate=200 * GB,
            margin=10 * GB,
        )
        _call_gate(s, 256, instant=200 * GB)  # no limit -> never raises

    def test_noop_when_monitor_missing(self):
        s = _gate_scheduler(
            hard_limit=107 * GB,
            recent_peak=200 * GB,
            estimate=200 * GB,
            margin=10 * GB,
            monitor=False,
        )
        _call_gate(s, 256, instant=200 * GB)  # no monitor -> never raises

    def test_noop_when_estimate_zero(self):
        """estimate==0 means the model can't be estimated -> leave it to the
        legacy chunk-end check, do not raise here."""
        s = _gate_scheduler(
            hard_limit=107 * GB,
            recent_peak=200 * GB,
            estimate=0,
            margin=10 * GB,
        )
        _call_gate(s, 256, instant=200 * GB)  # estimate 0 -> never raises

    def test_noop_when_chunk_zero(self):
        s = _gate_scheduler(
            hard_limit=107 * GB,
            recent_peak=200 * GB,
            estimate=2 * GB,
            margin=10 * GB,
        )
        _call_gate(s, 0, instant=200 * GB)  # nothing to process -> never raises


# ---------------------------------------------------------------------------
# Integration: gate fires BEFORE the model forward in the real chunked loop
# ---------------------------------------------------------------------------


def _integration_scheduler(*, hard_gb: float, estimate_bytes: int, margin_gb: float):
    """Scheduler with a mock model, hard cap on but soft off (so the adaptive
    throttle passes through and only the forward-front gate can fire)."""
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    config = SchedulerConfig(
        max_num_seqs=8,
        prefill_step_size=256,
        chunked_prefill=True,
        paged_cache_block_size=0,
    )
    s = Scheduler(model=model, tokenizer=tokenizer, config=config)
    s.batch_generator = MagicMock()
    # Soft limit 0 -> _adaptive_chunk_size is a pure passthrough.
    s._memory_limit_bytes = 0
    s._memory_hard_limit_bytes = int(hard_gb * GB)
    s._prefill_memory_guard = True
    s._prefill_transient_margin_bytes = int(margin_gb * GB)
    s.memory_monitor = MagicMock()
    s.memory_monitor.estimate_prefill_peak_bytes = MagicMock(
        return_value=estimate_bytes
    )
    return s, model


def _prefill_state(n_tokens: int) -> _PrefillState:
    req = Request(
        request_id="rid-int",
        prompt=list(range(n_tokens + 1)),
        sampling_params=SamplingParams(max_tokens=8),
    )
    req.prompt_token_ids = list(range(n_tokens + 1))
    req.num_prompt_tokens = n_tokens + 1
    req.status = RequestStatus.WAITING
    return _PrefillState(
        request=req,
        cache=[],
        tokens_remaining=mx.array(list(range(n_tokens)))[None],
        last_token=[n_tokens],
        tokens_processed=0,
        base_size=0,
        emitted_boundaries={},
        boundary_enabled=False,
        block_size=0,
        total_length=n_tokens + 1,
    )


class TestForwardGateBlocksForward:
    """The gate must abort the chunk BEFORE self.model(...) runs."""

    def test_over_cap_does_not_call_model_forward(self):
        """Predicted peak over cap -> RuntimeError raised and model NOT called.

        This is the discriminating assertion that pins the fix: pre-change
        code (no forward-front gate) reaches self.model(chunk, ...) and the
        transient lands on the cap (kernel panic on real hardware). With the
        gate, the forward never runs.
        """
        # recent_peak high (set via instant probes) + estimate + margin > cap.
        s, model = _integration_scheduler(
            hard_gb=107.0, estimate_bytes=2 * GB, margin_gb=10 * 1.0
        )
        state = _prefill_state(n_tokens=200)

        high = int(100 * GB)
        with patch(
            "omlx.scheduler.mx.get_active_memory", return_value=high
        ), patch("omlx.scheduler.get_phys_footprint", return_value=high), patch(
            "omlx.scheduler.mx.eval"
        ) as mock_eval:
            with pytest.raises(RuntimeError, match="refused before forward"):
                s._step_prefill_chunk(state)

        # The whole point: the model forward must not have executed.
        model.assert_not_called()
        mock_eval.assert_not_called()

    def test_under_cap_runs_model_forward(self):
        """Predicted peak under cap -> forward runs as normal (control)."""
        s, model = _integration_scheduler(
            hard_gb=107.0, estimate_bytes=1 * GB, margin_gb=2.0
        )
        state = _prefill_state(n_tokens=200)

        low = int(50 * GB)  # 50 + 1 + 2 = 53 < 107
        with patch(
            "omlx.scheduler.mx.get_active_memory", return_value=low
        ), patch("omlx.scheduler.get_phys_footprint", return_value=low), patch(
            "omlx.scheduler.mx.eval"
        ), patch("omlx.scheduler._sync_and_clear_cache"), patch(
            "omlx.scheduler.get_prefill_tracker"
        ):
            done = s._step_prefill_chunk(state)

        # Forward ran exactly once; prefill consumed the only chunk.
        assert model.call_count == 1
        assert done is True


class TestForwardGateExternalLoopWiring:
    """Sanity that the external loop wiring calls the gate before the forward.

    Patch _prefill_forward_gate to raise; the model forward must not run.
    Uses a tiny text-only request through _do_external_prefill.
    """

    def test_external_loop_calls_gate_before_forward(self):
        model = MagicMock()
        model.layers = []
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2
        config = SchedulerConfig(
            max_num_seqs=8,
            prefill_step_size=256,
            chunked_prefill=False,
            paged_cache_block_size=0,
        )
        s = Scheduler(model=model, tokenizer=tokenizer, config=config)

        req = Request(
            request_id="rid-ext",
            prompt=[1, 2, 3, 4, 5],
            sampling_params=SamplingParams(max_tokens=8),
        )
        req.prompt_token_ids = [1, 2, 3, 4, 5]
        req.num_prompt_tokens = 5

        with patch.object(
            s,
            "_prefill_forward_gate",
            side_effect=RuntimeError("Prefill refused before forward"),
        ) as mock_gate, patch(
            "omlx.scheduler.make_prompt_cache", return_value=[]
        ):
            with pytest.raises(RuntimeError, match="refused before forward"):
                s._do_external_prefill(req, [1, 2, 3, 4, 5], None)

        mock_gate.assert_called_once()
        # Gate raised -> forward must not have run.
        model.assert_not_called()
