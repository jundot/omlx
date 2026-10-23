# SPDX-License-Identifier: Apache-2.0
"""A request that BatchGenerator.insert refuses fails alone.

mlx-lm's insert raises ValueError for a row it cannot accept (an empty prompt,
max_tokens <= 0). It runs after the request's prefill, inside step(), so before
this fix the exception reached engine_core's loop, whose recovery calls
scheduler.fail_all_requests() and fails every request on the engine. The
scheduler refuses max_tokens < 1 at admission (_insert_max_tokens would
otherwise serve it one token after the full prefill) and turns an insert
ValueError into an error output for that request alone. An empty prompt is the
refusal that still reaches insert, so the isolation tests use it.
"""

from collections import deque
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.engine_core import EngineCore
from omlx.exceptions import InvalidRequestError
from omlx.request import Request, RequestStatus, SamplingParams
from omlx.scheduler import Scheduler, _PrefillState

_REFUSAL = "Sequence 0 has an empty prompt."


def _insert_like_mlx_lm(prompts, **kwargs):
    """Stand-in for BatchGenerator.insert with mlx-lm's validate-first checks."""
    if not prompts[0]:
        raise ValueError(_REFUSAL)
    if kwargs["max_tokens"][0] <= 0:
        raise ValueError("Sequence 0's max_tokens must be > 0.")
    return [42]


def _make_scheduler(mock_model, mock_tokenizer):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.insert.side_effect = _insert_like_mlx_lm
    scheduler.batch_generator.next_generated.return_value = []
    scheduler._ensure_batch_generator = MagicMock()
    scheduler._build_sampler_and_processors = MagicMock(return_value=(MagicMock(), []))
    scheduler._build_state_machine = MagicMock(return_value=MagicMock())
    scheduler._preflight_memory_check = MagicMock(return_value=None)
    scheduler._validate_cache = MagicMock(return_value=True)
    return scheduler


def _queue_cache_hit(scheduler, request_id, prompt=(11, 12, 13, 14)):
    """Queue a request whose prompt is cached up to its last token, so
    _schedule_waiting inserts it without running a prefill. An empty prompt
    reaches insert as an empty row."""
    prompt = list(prompt)
    request = Request(
        request_id=request_id,
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = prompt
    request.num_prompt_tokens = len(prompt)
    request.remaining_tokens = prompt[-1:]
    request.cached_tokens = max(len(prompt) - 1, 0)
    request.prompt_cache = [MagicMock()]
    scheduler.waiting.append(request)
    scheduler.requests[request_id] = request
    return request


def _prefill_state(request, last_token=(14,)):
    """A prefill state one token from its end; insert receives last_token."""
    return _PrefillState(
        request=request,
        cache=[MagicMock()],
        tokens_remaining=mx.array([[]]),
        last_token=list(last_token),
        tokens_processed=3,
        base_size=0,
        emitted_boundaries={},
        boundary_enabled=False,
        block_size=0,
        total_length=4,
        sampler=MagicMock(),
        sm=MagicMock(),
        per_row_lps=[],
    )


def _queue_chunked_prefill(scheduler, request_id, last_token=(14,)):
    """Track a request mid chunked prefill, one token from its last chunk."""
    request = Request(
        request_id=request_id,
        prompt=[11, 12, 13, 14],
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = [11, 12, 13, 14]
    request.num_prompt_tokens = 4
    state = _prefill_state(request, last_token)
    scheduler.requests[request_id] = request
    scheduler.prefilling.append(request)
    scheduler._prefill_states[request_id] = state
    return state


def _add_running(scheduler, request_id, uid):
    request = Request(
        request_id=request_id,
        prompt=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        status=RequestStatus.RUNNING,
        batch_uid=uid,
    )
    scheduler.running[request_id] = request
    scheduler.requests[request_id] = request
    scheduler.request_id_to_uid[request_id] = uid
    scheduler.uid_to_request_id[uid] = request_id
    return request


def _assert_refused(output, request_id):
    assert output.request_id == request_id
    assert output.finished is True
    assert output.finish_reason == "error"
    assert output.error == _REFUSAL


def _tiny_llama():
    from mlx_lm.models.llama import Model, ModelArgs

    return Model(
        ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=64,
        )
    )


def test_mlx_lm_insert_refuses_bad_rows_before_touching_batch_state():
    """The isolation relies on insert validating every row before it
    changes batch state; pin that for the installed mlx-lm."""
    from mlx_lm.generate import BatchGenerator

    generator = BatchGenerator(_tiny_llama(), max_tokens=4)
    try:
        for bad in (0, -5):
            with pytest.raises(ValueError, match="max_tokens must be > 0"):
                generator.insert([[1, 2, 3]], max_tokens=[bad])
        with pytest.raises(ValueError, match="empty prompt"):
            generator.insert([[]], max_tokens=[4])
        assert generator._uid_count == 0
        assert not generator._unprocessed_sequences
        assert generator.insert([[1, 2, 3]], max_tokens=[4]) == [0]
    finally:
        generator.close()


@pytest.mark.parametrize("max_tokens", [0, -5])
def test_add_request_refuses_non_positive_max_tokens(
    max_tokens, mock_model, mock_tokenizer
):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    request = Request(
        request_id="bad",
        prompt=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )

    with pytest.raises(InvalidRequestError, match="max_tokens must be at least 1"):
        scheduler.add_request(request)

    assert scheduler.requests == {}
    assert list(scheduler.waiting) == []


def test_add_request_accepts_one_max_token(mock_model, mock_tokenizer):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    request = Request(
        request_id="one",
        prompt=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
    )

    scheduler.add_request(request)

    assert list(scheduler.waiting) == [request]


def test_refused_insert_fails_only_that_request(mock_model, mock_tokenizer):
    scheduler = _make_scheduler(mock_model, mock_tokenizer)
    live = _add_running(scheduler, "live", uid=7)
    _queue_cache_hit(scheduler, "bad", prompt=[])
    good = _queue_cache_hit(scheduler, "good")
    # Admit both in one pass; decode fairness would otherwise defer "good"
    # to the next step behind the running "live" decode.
    scheduler._prefill_gate_open = MagicMock(return_value=True)

    scheduled, rejected = scheduler._schedule_waiting()

    assert scheduled == [good]
    assert len(rejected) == 1
    _assert_refused(rejected[0], "bad")
    assert "bad" not in scheduler.requests
    assert "bad" not in scheduler.running
    assert "bad" not in scheduler.request_id_to_uid
    assert scheduler.running == {"live": live, "good": good}
    assert scheduler.request_id_to_uid == {"live": 7, "good": 42}


def test_refused_chunked_insert_fails_only_that_request(mock_model, mock_tokenizer):
    scheduler = _make_scheduler(mock_model, mock_tokenizer)
    live = _add_running(scheduler, "live", uid=7)
    states = {
        "bad": _queue_chunked_prefill(scheduler, "bad", last_token=()),
        "good": _queue_chunked_prefill(scheduler, "good"),
    }
    scheduler._prefill_gate_open = MagicMock(return_value=True)
    scheduler._step_prefill_chunk = MagicMock(return_value=True)
    scheduler._emit_final_boundary_if_needed = MagicMock()

    scheduled, rejected = [], []
    scheduler._advance_chunked_prefills(scheduled, rejected)

    good = states["good"].request
    assert scheduled == [good]
    assert len(rejected) == 1
    _assert_refused(rejected[0], "bad")
    assert "bad" not in scheduler.requests
    assert "bad" not in scheduler._prefill_states
    assert list(scheduler.prefilling) == []
    assert scheduler.running == {"live": live, "good": good}


def test_refused_chunked_insert_leaves_other_prefills_in_flight(
    mock_model, mock_tokenizer
):
    scheduler = _make_scheduler(mock_model, mock_tokenizer)
    _queue_chunked_prefill(scheduler, "bad", last_token=())
    slow = _queue_chunked_prefill(scheduler, "slow")
    scheduler._prefill_gate_open = MagicMock(return_value=True)
    # "bad" finishes its prefill this step; "slow" still has chunks to go.
    scheduler._step_prefill_chunk = MagicMock(
        side_effect=lambda state: state.request.request_id == "bad"
    )
    scheduler._emit_final_boundary_if_needed = MagicMock()

    scheduled, rejected = [], []
    scheduler._advance_chunked_prefills(scheduled, rejected)

    assert scheduled == []
    assert [output.request_id for output in rejected] == ["bad"]
    assert list(scheduler.prefilling) == [slow.request]
    assert scheduler._prefill_states == {"slow": slow}
    assert set(scheduler.requests) == {"slow"}


def test_refused_first_chunk_insert_fails_only_that_request(mock_model, mock_tokenizer):
    """A chunked prefill that finishes in its first chunk is inserted from
    _schedule_waiting; a refusal there fails only that request."""
    scheduler = _make_scheduler(mock_model, mock_tokenizer)
    live = _add_running(scheduler, "live", uid=7)
    bad = _queue_cache_hit(scheduler, "bad")
    bad.remaining_tokens = list(bad.prompt_token_ids)
    bad.cached_tokens = 0
    bad.prompt_cache = None
    # Four uncached tokens exceed a two-token chunk, so the prefill is chunked.
    scheduler.config.chunked_prefill = True
    scheduler.config.prefill_step_size = 2
    scheduler._prefill_gate_open = MagicMock(return_value=True)
    # The first chunk finishes the prefill and leaves an empty row to insert.
    scheduler._begin_prefill = MagicMock(return_value=_prefill_state(bad, ()))
    scheduler._step_prefill_chunk = MagicMock(return_value=True)
    scheduler._emit_final_boundary_if_needed = MagicMock()

    scheduled, rejected = scheduler._schedule_waiting()

    scheduler._begin_prefill.assert_called_once()
    assert scheduled == []
    assert len(rejected) == 1
    _assert_refused(rejected[0], "bad")
    assert "bad" not in scheduler.requests
    assert scheduler.running == {"live": live}


def test_refused_packed_companion_fails_only_that_request(mock_model, mock_tokenizer):
    """A companion that finishes inside the head's packed forward is inserted
    on its own; a refusal there fails only the companion."""
    scheduler = _make_scheduler(mock_model, mock_tokenizer)
    live = _add_running(scheduler, "live", uid=7)
    head = _queue_chunked_prefill(scheduler, "head")
    bad = _queue_chunked_prefill(scheduler, "bad", last_token=())
    scheduler._prefill_gate_open = MagicMock(return_value=True)
    scheduler._plan_prefill_rows = MagicMock(
        return_value=[MagicMock(state=head), MagicMock(state=bad)]
    )
    scheduler._run_prefill_chunks = MagicMock(return_value=[True, True])
    scheduler._emit_final_boundary_if_needed = MagicMock()

    with (
        patch.object(Scheduler, "_packed_prefill_ready", return_value=True),
        patch.object(Scheduler, "_packing_fits_contended_cap", return_value=True),
        patch.object(Scheduler, "_contended_prefill_cap", return_value=0),
    ):
        scheduled, rejected = [], []
        scheduler._advance_chunked_prefills(scheduled, rejected)

    assert scheduled == [head.request]
    assert len(rejected) == 1
    _assert_refused(rejected[0], "bad")
    assert "bad" not in scheduler.requests
    assert list(scheduler.prefilling) == []
    assert scheduler.running == {"live": live, "head": head.request}


def test_refused_insert_releases_its_paged_cache_blocks(mock_model, mock_tokenizer):
    scheduler = _make_scheduler(mock_model, mock_tokenizer)
    paged = MagicMock()
    paged.get_block_table.return_value = MagicMock(block_ids=[5, 6, 7])
    paged.release_for_eviction.return_value = 3
    scheduler.paged_cache_manager = paged
    _queue_cache_hit(scheduler, "bad", prompt=[])
    scheduler._prefill_gate_open = MagicMock(return_value=True)

    scheduled, rejected = scheduler._schedule_waiting()

    assert scheduled == []
    assert [output.request_id for output in rejected] == ["bad"]
    paged.get_block_table.assert_any_call("bad")
    paged.release_for_eviction.assert_called_once_with([5, 6, 7])


def test_scheduler_goes_idle_after_a_lone_refusal(mock_model, mock_tokenizer):
    scheduler = _make_scheduler(mock_model, mock_tokenizer)
    _queue_cache_hit(scheduler, "bad", prompt=[])

    with patch("omlx.scheduler._sync_and_clear_cache"):
        output = scheduler.step()
        refusals = [o for o in output.outputs if o.request_id == "bad"]
        assert len(refusals) == 1
        assert not (scheduler.waiting or scheduler.prefilling or scheduler.running)
        assert scheduler.requests == {}
        # The abort path may defer one Metal cache clear; nothing else is left.
        for _ in range(8):
            if not scheduler.has_requests():
                break
            scheduler.step()
        assert scheduler.has_requests() is False
        # engine_core may still abort the refused id afterwards; that is a no-op.
        scheduler.abort_request("bad")
        scheduler.step()
    assert scheduler.has_requests() is False


def test_step_reports_refused_insert_without_raising(mock_model, mock_tokenizer):
    """step() returns the refusal as that request's output; nothing reaches
    the engine loop's fail-all recovery."""
    scheduler = _make_scheduler(mock_model, mock_tokenizer)
    live = _add_running(scheduler, "live", uid=7)
    _queue_cache_hit(scheduler, "bad", prompt=[])

    with patch("omlx.scheduler._sync_and_clear_cache"):
        output = scheduler.step()

    refusals = [o for o in output.outputs if o.request_id == "bad"]
    assert len(refusals) == 1
    _assert_refused(refusals[0], "bad")
    assert all(
        o.finish_reason != "error" for o in output.outputs if o is not refusals[0]
    )
    assert scheduler.running == {"live": live}
    assert scheduler.waiting == deque()
    assert "bad" not in scheduler.requests
    assert scheduler.has_requests() is True


def test_empty_prompt_fails_alone_beside_a_decoding_request(mock_tokenizer):
    """With mlx-lm's real BatchGenerator, a row insert refuses fails alone
    while another request decodes to its cap."""
    mx.random.seed(0)
    scheduler = Scheduler(model=_tiny_llama(), tokenizer=mock_tokenizer)
    finished, tokens = {}, {}

    def run(steps):
        for _ in range(steps):
            if not scheduler.has_requests():
                return
            for output in scheduler.step().outputs:
                count = len(output.new_token_ids or [])
                tokens[output.request_id] = tokens.get(output.request_id, 0) + count
                if output.finished:
                    finished[output.request_id] = output.finish_reason

    scheduler.add_request(
        Request(
            request_id="live",
            prompt=[5, 6, 7, 8, 9, 10, 11, 12],
            sampling_params=SamplingParams(max_tokens=24, temperature=0.0),
        )
    )
    run(3)
    assert "live" in scheduler.running
    scheduler.add_request(
        Request(
            request_id="empty",
            prompt=[],
            sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
        )
    )

    run(80)

    assert finished == {"empty": "error", "live": "length"}
    assert tokens["live"] == 24
    assert scheduler.has_requests() is False


@pytest.mark.asyncio
async def test_engine_add_request_refuses_non_positive_max_tokens_alone(
    mock_model, mock_tokenizer
):
    """The refusal reaches only its caller, as the typed 400 error, and leaves
    no tracking behind; a request already admitted is untouched."""
    with patch("omlx.engine_core.get_registry") as mock_registry:
        mock_registry.return_value.acquire.return_value = True
        engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
        try:
            engine.scheduler.fail_all_requests = MagicMock(return_value=[])
            kept = await engine.add_request(
                prompt="Hello", sampling_params=SamplingParams(max_tokens=8)
            )

            with pytest.raises(InvalidRequestError) as refused:
                await engine.add_request(
                    prompt="Hello", sampling_params=SamplingParams(max_tokens=0)
                )

            assert refused.value.field == "max_tokens"
            assert set(engine._output_collectors) == {kept}
            assert set(engine._stream_states) == {kept}
            assert set(engine.scheduler.requests) == {kept}
            engine.scheduler.fail_all_requests.assert_not_called()
        finally:
            engine.close()
