"""Characterize existing prefill boundaries without changing execution policy."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from omlx.prefill_progress import get_prefill_tracker
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, _PrefillState


@pytest.fixture
def boundary_scheduler():
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            prefill_step_size=6,
            chunked_prefill=True,
            paged_cache_block_size=0,
        ),
    )
    scheduler.config.paged_cache_block_size = 4
    scheduler.block_aware_cache = MagicMock()
    scheduler._memory_limit_bytes = 0
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.insert.return_value = [42]
    events = []
    forwarded = []
    snapshots = []
    limit = SimpleNamespace(tokens=None)

    def adaptive(token_count, **kwargs):
        events.append(("adaptive", token_count))
        return token_count

    def guard(token_count, **kwargs):
        events.append(("guard", token_count))
        if limit.tokens is not None:
            return min(token_count, limit.tokens)
        return token_count

    def forward(tokens, *, cache, **kwargs):
        row = tokens.tolist()[0]
        forwarded.extend(row)
        events.append(("forward", len(row)))
        keys = mx.zeros((1, 1, len(row), 1))
        cache[0].update_and_fetch(keys, keys)

    def snapshot(request, cache, total_tokens):
        assert cache[0].offset == total_tokens
        snapshots.append(total_tokens)
        events.append(("snapshot", total_tokens))

    model.side_effect = forward
    get_prefill_tracker().clear()
    with ExitStack() as patches:
        patches.enter_context(patch("omlx.scheduler._sync_and_clear_cache"))
        patches.enter_context(
            patch("omlx.scheduler.get_phys_footprint", return_value=0)
        )
        patches.enter_context(
            patch("omlx.scheduler._prompt_cache_needs_snapshots", return_value=True)
        )
        patches.enter_context(
            patch(
                "omlx.scheduler._cache_base_sizes",
                side_effect=lambda cache: cache[0].offset,
            )
        )
        patches.enter_context(
            patch(
                "omlx.scheduler.mx.eval",
                side_effect=lambda *args: events.append(("eval",)),
            )
        )
        patches.enter_context(
            patch.object(scheduler, "_adaptive_chunk_size", side_effect=adaptive)
        )
        patches.enter_context(
            patch.object(scheduler, "_guard_prefill_chunk", side_effect=guard)
        )
        patches.enter_context(
            patch.object(scheduler, "_supports_skip_lm_head", return_value=False)
        )
        patches.enter_context(patch.object(scheduler, "_record_chunk_transient"))
        patches.enter_context(
            patch.object(scheduler, "_maybe_record_fixed_state_bytes")
        )
        patches.enter_context(
            patch.object(scheduler, "_should_clear_after_chunk", return_value=False)
        )
        patches.enter_context(
            patch.object(
                scheduler, "_emit_prefill_boundary_snapshot", side_effect=snapshot
            )
        )
        yield SimpleNamespace(
            scheduler=scheduler,
            events=events,
            forwarded=forwarded,
            snapshots=snapshots,
            limit=limit,
        )
    get_prefill_tracker().clear()


@pytest.mark.parametrize("path", ["external", "resumable"])
@pytest.mark.parametrize(
    "base_tokens,prefill_tokens,boundaries,guard_limit,expected_chunks,expected_snapshots",
    [
        (0, 10, True, None, [4, 4, 2], [4, 8]),
        (3, 9, True, None, [1, 4, 4], [4, 8, 12]),
        (4, 7, True, None, [4, 3], [8]),
        (0, 8, True, None, [4, 4], [4, 8]),
        (3, 1, True, None, [1], [4]),
        (0, 1, True, None, [1], []),
        (0, 0, True, None, [], []),
        (3, 10, False, None, [6, 4], []),
        (0, 10, True, 2, [2, 2, 2, 2, 2], [4, 8]),
    ],
)
def test_existing_chunk_and_snapshot_sequence(
    boundary_scheduler,
    path,
    base_tokens,
    prefill_tokens,
    boundaries,
    guard_limit,
    expected_chunks,
    expected_snapshots,
):
    context = boundary_scheduler
    scheduler = context.scheduler
    context.limit.tokens = guard_limit
    if not boundaries:
        scheduler.block_aware_cache = None
    tokens = list(range(base_tokens + prefill_tokens + 1))
    request = Request(
        request_id="boundaries",
        prompt=tokens,
        sampling_params=SamplingParams(max_tokens=8),
    )
    request.prompt_token_ids = tokens
    request.num_prompt_tokens = len(tokens)
    request.cached_tokens = base_tokens
    request.remaining_tokens = tokens[base_tokens:]
    cache = [KVCache()]
    if base_tokens:
        keys = mx.zeros((1, 1, base_tokens, 1))
        cache[0].update_and_fetch(keys, keys)

    if path == "external":
        returned_cache, last_token = scheduler._do_external_prefill(
            request, request.remaining_tokens, cache
        )
        assert returned_cache is cache
        assert last_token == tokens[-1:]
    else:
        state = scheduler._begin_prefill(request, request.remaining_tokens, cache)
        state.sampler = MagicMock()
        state.sm = MagicMock()
        state.per_row_lps = []
        while not scheduler._step_prefill_chunk(state):
            pass
        assert state.tokens_processed == prefill_tokens
        assert state.tokens_remaining.shape[1] == 0
        scheduler._emit_final_boundary_if_needed(state)
        scheduled = []
        scheduler._insert_prefilled_request(request, state, scheduled)
        assert scheduled == [request]
        assert scheduler.running[request.request_id] is request
        assert scheduler.batch_generator.insert.call_args.args == ([tokens[-1:]],)
        assert scheduler.batch_generator.insert.call_args.kwargs["caches"][0] is cache

    assert context.forwarded == tokens[base_tokens:-1]
    assert cache[0].offset == base_tokens + prefill_tokens
    assert context.snapshots == expected_snapshots
    assert [
        event[1] for event in context.events if event[0] == "forward"
    ] == expected_chunks
    for event_index, event in enumerate(context.events):
        if event[0] == "forward":
            assert context.events[event_index - 2][0] == "adaptive"
            assert context.events[event_index - 1][0] == "guard"
            assert context.events[event_index + 1] == ("eval",)
        elif event[0] == "snapshot":
            assert context.events[event_index - 1] == ("eval",)


@pytest.mark.parametrize(
    "total_tokens,last_emitted,boundaries,expected",
    [
        (0, -1, True, []),
        (7, -1, True, []),
        (8, -1, True, [8]),
        (8, 4, True, [8]),
        (8, 8, True, []),
        (8, 12, True, []),
        (8, -1, False, []),
    ],
)
def test_existing_final_snapshot_eligibility(
    boundary_scheduler, total_tokens, last_emitted, boundaries, expected
):
    context = boundary_scheduler
    request = SimpleNamespace(request_id="final-boundary")
    cache = [KVCache()]
    cache[0].offset = total_tokens
    emitted = {request.request_id: last_emitted}
    state = _PrefillState(
        request=request,
        cache=cache,
        tokens_remaining=None,
        last_token=[1],
        tokens_processed=total_tokens,
        base_size=0,
        emitted_boundaries=emitted,
        boundary_enabled=boundaries,
        block_size=4,
        total_length=total_tokens + 1,
    )

    context.scheduler._emit_final_boundary_if_needed(state)

    assert context.snapshots == expected
    assert emitted == {request.request_id: last_emitted}
