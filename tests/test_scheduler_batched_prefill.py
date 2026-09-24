"""Scheduler integration with real tiny-model prefill and per-request caches."""

from __future__ import annotations

import importlib
import logging
from itertools import count
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, make_prompt_cache

from omlx.decode_activity import get_decode_activity
from omlx.prefill.memory import PrefillTransition
from omlx.prefill.planning import PrefillDefer, PrefillReason, PrefillRun, plan_prefill_batch
from omlx.prefill.timing import PrefillTiming
from omlx.prefill_progress import get_prefill_tracker
from omlx.request import Request, RequestStatus, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


@pytest.fixture(autouse=True)
def isolated_activity():
    get_decode_activity().clear()
    get_prefill_tracker().clear()
    yield
    get_decode_activity().clear()
    get_prefill_tracker().clear()


@pytest.fixture
def scheduler_factory(monkeypatch):
    schedulers = []

    def create(model_type="llama", **settings):
        module = importlib.import_module(f"mlx_lm.models.{model_type}")
        stream = mx.new_thread_local_stream(mx.default_device())
        with mx.stream(stream):
            model = module.Model(
                module.ModelArgs.from_dict(
                    {
                        "model_type": model_type,
                        "hidden_size": 32,
                        "num_hidden_layers": 2,
                        "intermediate_size": 64,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "rms_norm_eps": 1e-5,
                        "vocab_size": 64,
                        "head_dim": 8,
                        "max_position_embeddings": 2048,
                        "rope_theta": 10000,
                        "tie_word_embeddings": True,
                    }
                )
            )
            model.eval()
            mx.eval(model.parameters())

        harness = SimpleNamespace(
            model=model,
            calls=[],
            failure=None,
            forward_hook=None,
            original_forward=type(model).__call__,
        )

        def record_forward(instance, inputs, cache=None, **kwargs):
            if instance is model:
                harness.calls.append(
                    SimpleNamespace(
                        shape=tuple(inputs.shape),
                        tokens=inputs.tolist(),
                        cache_lengths=tuple(layer.size() for layer in cache),
                        stream=mx.default_stream(mx.default_device()),
                    )
                )
            output = harness.original_forward(instance, inputs, cache=cache, **kwargs)
            if instance is model:
                if harness.forward_hook is not None:
                    harness.forward_hook()
                if harness.failure is not None:
                    mx.eval([layer.state for layer in cache])
                    raise RuntimeError(harness.failure)
            return output

        monkeypatch.setattr(type(model), "__call__", record_forward)
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 63
        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                **{
                    "max_num_seqs": 8,
                    "max_num_batched_tokens": 64,
                    "prefill_step_size": 4,
                    "prefill_max_batch_size": 2,
                    "chunked_prefill": True,
                    "decode_fairness": False,
                    "paged_cache_block_size": 0,
                    **settings,
                }
            ),
        )
        scheduler._stream = stream
        scheduler._memory_limit_bytes = 10**12
        scheduler._memory_hard_limit_bytes = 10**12
        scheduler._memory_abort_limit_bytes = 10**12
        scheduler._memory_limits_propagated = True
        batch_generator = MagicMock()
        next_uid = count(42)
        batch_generator.insert.side_effect = lambda *args, **kwargs: [next(next_uid)]
        batch_generator.next_generated.side_effect = lambda: iter([])
        scheduler.batch_generator = batch_generator
        scheduler._current_sampler_params = ()
        harness.scheduler = scheduler
        harness.batch_generator = batch_generator
        schedulers.append(scheduler)
        return harness

    yield create
    for scheduler in schedulers:
        scheduler.reset()
        scheduler.shutdown()


def make_request(request_id, token_count=9, *, start=1, priority=0):
    tokens = list(range(start, start + token_count))
    return Request(
        request_id=request_id,
        prompt=tokens,
        prompt_token_ids=tokens,
        num_prompt_tokens=len(tokens),
        sampling_params=SamplingParams(max_tokens=16, temperature=0),
        priority=priority,
    )


def add_requests(scheduler, *requests):
    for request in requests:
        scheduler.add_request(request)


def forward_shapes(harness):
    return [call.shape for call in harness.calls]


def step_with_token_count(harness):
    first_call = len(harness.calls)
    output = harness.scheduler.step()
    token_count = sum(
        call.shape[0] * call.shape[1] for call in harness.calls[first_call:]
    )
    return output, token_count


def finish_prefills(scheduler):
    scheduled = []
    rejected = []
    for _ in range(32):
        if not scheduler.prefilling:
            return scheduled, rejected
        scheduler._advance_chunked_prefills(scheduled, rejected)
    pytest.fail("Prefill failed to finish within the finite prompt length")


def assert_insert_matches_single(harness, request, insertion):
    assert insertion.args[0] == [request.prompt_token_ids[-1:]]
    assert insertion.kwargs["all_tokens"] == [request.prompt_token_ids[:-1]]
    extracted = insertion.kwargs["caches"][0]
    assert all(type(layer) is KVCache for layer in extracted)
    assert all(layer.offset == request.num_prompt_tokens - 1 for layer in extracted)
    with mx.stream(harness.scheduler._stream):
        baseline = make_prompt_cache(harness.model)
        harness.original_forward(
            harness.model,
            mx.array([request.prompt_token_ids[:-1]], dtype=mx.int32),
            cache=baseline,
        )
        mx.eval([layer.state for layer in baseline])
        for actual, expected in zip(extracted, baseline):
            assert mx.allclose(
                actual.keys_and_values()[0], expected.keys_and_values()[0], atol=2e-5
            ).item()
            assert mx.allclose(
                actual.keys_and_values()[1], expected.keys_and_values()[1], atol=2e-5
            ).item()
        kickoff = mx.array([request.prompt_token_ids[-1:]], dtype=mx.int32)
        expected_logits = harness.original_forward(
            harness.model, kickoff, cache=baseline
        )
        actual_logits = harness.original_forward(
            harness.model, kickoff, cache=extracted
        )
        mx.eval(expected_logits, actual_logits)
        assert mx.allclose(actual_logits, expected_logits, atol=2e-5).item()


@pytest.mark.parametrize("model_type", ["llama", "qwen2", "qwen3"])
def test_custom_prefill_entrypoint_remains_scalar(
    scheduler_factory, monkeypatch, model_type
):
    harness = scheduler_factory(model_type)
    hook_shapes = []

    def custom_prefill(inputs, cache=None, **kwargs):
        hook_shapes.append(tuple(inputs.shape))
        return harness.model(inputs, cache=cache, **kwargs)

    monkeypatch.setattr(harness.model, "_omlx_prefill", custom_prefill, raising=False)
    requests = [make_request("first", 5), make_request("second", 5, start=20)]
    add_requests(harness.scheduler, *requests)

    scheduled, rejected = harness.scheduler._schedule_waiting()

    assert scheduled == requests
    assert rejected == []
    assert hook_shapes == [(1, 4), (1, 4)]
    assert forward_shapes(harness) == hook_shapes
    assert harness.scheduler._batched_prefill_stats["groups"] == 0
    assert (
        harness.scheduler._batched_prefill_stats["fallbacks"]["custom_execution"] == 2
    )


@pytest.mark.parametrize("model_type", ["llama", "qwen2", "qwen3"])
def test_batched_handoff_marks_text_positions(
    scheduler_factory, monkeypatch, model_type
):
    harness = scheduler_factory(model_type)
    marked = []

    def mark_text_positions(model, uid):
        assert model is harness.model
        marked.append(uid)

    monkeypatch.setattr(
        type(harness.model), "mark_text_positions", mark_text_positions, raising=False
    )
    requests = [make_request("first", 5), make_request("second", 5, start=20)]
    add_requests(harness.scheduler, *requests)

    scheduled, rejected = harness.scheduler._schedule_waiting()

    assert scheduled == requests
    assert rejected == []
    assert forward_shapes(harness) == [(2, 4)]
    assert all(request.text_positions_proven for request in requests)
    assert marked == [request.batch_uid for request in requests]


@pytest.mark.parametrize("model_type", ["llama", "qwen2", "qwen3"])
def test_two_cold_requests_use_real_batch_and_preserve_decode_kickoff(
    scheduler_factory, model_type
):
    harness = scheduler_factory(model_type)
    requests = [make_request("first", 5), make_request("second", 5, start=20)]
    add_requests(harness.scheduler, *requests)

    scheduled, rejected = harness.scheduler._schedule_waiting()

    assert scheduled == requests
    assert rejected == []
    assert forward_shapes(harness) == [(2, 4)]
    assert harness.calls[0].tokens == [
        request.prompt_token_ids[:-1] for request in requests
    ]
    with mx.stream(harness.scheduler._stream):
        assert harness.calls[0].stream == mx.default_stream(mx.default_device())
    assert set(harness.scheduler.running) == {"first", "second"}
    assert not harness.scheduler.prefilling
    assert not harness.scheduler._prefill_states
    assert harness.batch_generator.insert.call_count == 2
    for request, insertion in zip(
        requests, harness.batch_generator.insert.call_args_list
    ):
        assert_insert_matches_single(harness, request, insertion)


def test_unequal_tails_complete_rows_and_keep_remaining_cache(scheduler_factory):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    shorter = make_request("shorter", 7)
    longer = make_request("longer", 11, start=20)
    add_requests(scheduler, shorter, longer)

    assert scheduler._schedule_waiting() == ([], [])
    group = scheduler._prefill_runtime.group_for("shorter")
    assert group is scheduler._prefill_runtime.group_for("longer")
    assert forward_shapes(harness) == [(2, 4)]
    scheduled = []
    rejected = []
    scheduler._advance_chunked_prefills(scheduled, rejected)

    assert scheduled == [shorter]
    assert rejected == []
    assert not group.valid
    assert scheduler._prefill_runtime.group_for("longer") is None
    assert all(
        type(layer) is KVCache and layer.offset == 6
        for layer in scheduler._prefill_states["longer"].cache
    )
    assert scheduler._prefill_states["longer"].tokens_processed == 6
    assert scheduler._prefill_states["longer"].tokens_remaining.shape == (1, 4)
    scheduler._advance_chunked_prefills(scheduled, rejected)

    assert scheduled == [shorter, longer]
    assert rejected == []
    assert forward_shapes(harness) == [(2, 4), (2, 2), (1, 4)]
    assert not group.valid
    assert group.cache_nbytes == 0
    assert not scheduler._prefill_states
    for request, insertion in zip(
        (shorter, longer), harness.batch_generator.insert.call_args_list
    ):
        assert_insert_matches_single(harness, request, insertion)


def test_lone_request_runs_immediately_without_waiting_for_a_peer(scheduler_factory):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    first = make_request("first")
    scheduler.add_request(first)

    scheduler._schedule_waiting()

    assert forward_shapes(harness) == [(1, 4)]
    assert scheduler._prefill_states["first"].tokens_processed == 4
    assert scheduler._prefill_runtime.group_for("first") is None
    scheduler.add_request(make_request("later", start=20))
    scheduler._schedule_waiting()
    assert forward_shapes(harness) == [(1, 4), (1, 4)]
    assert scheduler._prefill_states["first"].tokens_processed == 4
    assert scheduler._prefill_runtime.group_for("later") is None


def test_new_admission_does_not_advance_existing_group_twice(scheduler_factory):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [
        make_request("first", 13),
        make_request("second", 13, start=15),
        make_request("third", 13, start=30),
        make_request("fourth", 13, start=45),
    ]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    first_group = scheduler._prefill_runtime.group_for("first")
    scheduler._schedule_waiting()
    second_group = scheduler._prefill_runtime.group_for("third")

    assert second_group is not first_group
    assert forward_shapes(harness) == [(2, 4), (2, 4)]
    assert all(
        state.tokens_processed == 4 for state in scheduler._prefill_states.values()
    )
    scheduler._advance_chunked_prefills([], [])

    assert forward_shapes(harness) == [(2, 4)] * 4
    assert all(
        state.tokens_processed == 8 for state in scheduler._prefill_states.values()
    )


@pytest.mark.parametrize("barrier_kind", ["priority", "position_state"])
def test_staging_does_not_cross_queue_barriers(scheduler_factory, barrier_kind):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    first = make_request("first", 3)
    barrier = make_request("barrier", 3, start=20)
    last = make_request("last", 3, start=40)
    if barrier_kind == "priority":
        barrier.priority = 1
    else:
        barrier.rope_deltas = 1
    add_requests(scheduler, first, barrier, last)

    for _ in range(3):
        scheduler._schedule_waiting()
        finish_prefills(scheduler)

    assert forward_shapes(harness) == [(1, 2), (1, 2), (1, 2)]
    assert [call.tokens[0][0] for call in harness.calls] == [1, 20, 40]
    assert not scheduler.waiting


def test_each_row_retains_its_sampler_processors_and_stop_machine(
    scheduler_factory, monkeypatch
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first"), make_request("second", start=20)]
    samplers = {request.request_id: MagicMock() for request in requests}
    processors = {
        request.request_id: [MagicMock(), MagicMock()] for request in requests
    }

    def build_params(params, request):
        return samplers[request.request_id], processors[request.request_id]

    monkeypatch.setattr(scheduler, "_build_sampler_and_processors", build_params)
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    stop_machines = {
        request_id: state.sm for request_id, state in scheduler._prefill_states.items()
    }
    scheduled, rejected = finish_prefills(scheduler)

    assert scheduled == requests
    assert rejected == []
    for request, insertion in zip(
        requests, harness.batch_generator.insert.call_args_list
    ):
        request_id = request.request_id
        assert insertion.kwargs["samplers"] == [samplers[request_id]]
        assert insertion.kwargs["logits_processors"] == [processors[request_id]]
        assert insertion.kwargs["stop_sequences"] == [stop_machines[request_id]]


def test_cancelling_middle_row_preserves_remaining_group_and_caches(
    scheduler_factory,
):
    harness = scheduler_factory(prefill_max_batch_size=3)
    scheduler = harness.scheduler
    first = make_request("first", 11)
    cancelled = make_request("cancelled", 11, start=15)
    last = make_request("last", 11, start=30)
    add_requests(scheduler, first, cancelled, last)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")

    scheduler.abort_request("cancelled")
    scheduler._process_pending_aborts()

    assert cancelled.status is RequestStatus.FINISHED_ABORTED
    assert "cancelled" not in scheduler.requests
    assert "cancelled" not in scheduler._prefill_states
    assert group.request_ids == ("first", "last")
    scheduled, rejected = finish_prefills(scheduler)
    assert scheduled == [first, last]
    assert rejected == []
    assert forward_shapes(harness) == [(3, 4), (2, 4), (2, 2)]
    assert not group.valid
    for request, insertion in zip(
        (first, last), harness.batch_generator.insert.call_args_list
    ):
        assert_insert_matches_single(harness, request, insertion)


@pytest.mark.parametrize("survivor_index", [0, 1, 2])
def test_single_survivor_demotes_from_original_batch_without_compaction(
    scheduler_factory, monkeypatch, survivor_index
):
    harness = scheduler_factory(prefill_max_batch_size=3)
    scheduler = harness.scheduler
    requests = [
        make_request(
            f"row-{row_index}",
            11 if row_index == survivor_index else 9,
            start=1 + row_index * 15,
        )
        for row_index in range(3)
    ]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for(requests[0].request_id)
    original_guard = scheduler._guard_prefill_group_transition
    transitions = []

    def guard(current_group, transition):
        transitions.append(
            (
                transition,
                current_group.batch_size,
                scheduler._prefill_runtime.owned_ids(current_group),
            )
        )
        return original_guard(current_group, transition)

    def forbidden_compaction(*args, **kwargs):
        pytest.fail("A sole survivor must be extracted from the original batch")

    monkeypatch.setattr(group, "remove", forbidden_compaction)
    monkeypatch.setattr(scheduler, "_guard_prefill_group_transition", guard)
    scheduled, rejected = [], []
    scheduler._advance_chunked_prefills(scheduled, rejected)

    survivor = requests[survivor_index]
    assert not rejected
    assert scheduled == [request for request in requests if request is not survivor]
    assert transitions[-1] == (PrefillTransition.DEMOTE, 3, (survivor.request_id,))
    assert all(
        transition[0] is not PrefillTransition.COMPACT for transition in transitions
    )
    assert not group.valid
    assert not scheduler._prefill_runtime.has_groups
    assert scheduler._prefill_states[survivor.request_id].cache[0].offset == 8
    assert forward_shapes(harness) == [(3, 4), (3, 4)]
    completed, rejected = finish_prefills(scheduler)
    assert completed == [survivor]
    assert not rejected
    assert forward_shapes(harness) == [(3, 4), (3, 4), (1, 2)]
    for request, insertion in zip(
        scheduled + completed, harness.batch_generator.insert.call_args_list
    ):
        assert_insert_matches_single(harness, request, insertion)


@pytest.mark.parametrize("cancelled_id", ["first", "second"])
@pytest.mark.parametrize("during_forward", [False, True])
def test_cancellation_demotes_single_unfinished_row_without_compaction(
    scheduler_factory, monkeypatch, cancelled_id, during_forward
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first", 11), make_request("second", 11, start=20)]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")

    def forbidden_compaction(*args, **kwargs):
        pytest.fail("Cancellation must demote the sole unfinished row directly")

    monkeypatch.setattr(group, "remove", forbidden_compaction)
    scheduled, rejected = [], []
    if during_forward:
        harness.forward_hook = lambda: scheduler.abort_request(cancelled_id)
        scheduler._advance_chunked_prefills(scheduled, rejected)
        harness.forward_hook = None
    else:
        scheduler.abort_request(cancelled_id)
        scheduler._process_pending_aborts()
    survivor = next(
        request for request in requests if request.request_id != cancelled_id
    )
    assert not scheduled
    assert not rejected
    assert not group.valid
    assert not scheduler._prefill_runtime.has_groups
    assert scheduler._prefill_states[survivor.request_id].cache[0].offset == (
        8 if during_forward else 4
    )
    scheduled, rejected = finish_prefills(scheduler)
    assert scheduled == [survivor]
    assert not rejected
    assert forward_shapes(harness) == (
        [(2, 4), (2, 4), (1, 2)] if during_forward else [(2, 4), (1, 4), (1, 2)]
    )
    assert_insert_matches_single(
        harness, survivor, harness.batch_generator.insert.call_args
    )


@pytest.mark.parametrize("cleanup", ["abort_all", "reset", "fail_all"])
def test_group_storage_is_released_by_scheduler_cleanup(scheduler_factory, cleanup):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    add_requests(scheduler, make_request("first"), make_request("second", start=20))
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    assert group.cache_nbytes > 0

    if cleanup == "abort_all":
        scheduler.abort_request("first")
        scheduler.abort_request("second")
        scheduler._process_pending_aborts()
    elif cleanup == "reset":
        scheduler.reset()
    else:
        assert set(scheduler.fail_all_requests()) == {"first", "second"}

    assert not group.valid
    assert group.request_ids == ()
    assert group.cache_nbytes == 0
    assert not scheduler._prefill_states
    assert not scheduler.prefilling
    assert not scheduler.running
    assert not scheduler.waiting
    assert not scheduler.requests


def test_failed_forward_discards_all_partially_updated_group_rows(scheduler_factory):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    add_requests(scheduler, make_request("first"), make_request("second", start=20))
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    harness.failure = "batched model kernel failed"
    scheduled = []
    rejected = []

    scheduler._advance_chunked_prefills(scheduled, rejected)

    assert scheduled == []
    assert {output.request_id for output in rejected} == {"first", "second"}
    assert all(
        output.finished and output.finish_reason == "error" for output in rejected
    )
    assert not group.valid
    assert group.cache_nbytes == 0
    assert not scheduler.requests
    assert not scheduler._prefill_states
    assert not scheduler.prefilling
    assert not scheduler.waiting
    harness.batch_generator.insert.assert_not_called()


@pytest.mark.parametrize("phase", ["forward", "extract", "demotion"])
@pytest.mark.parametrize("error_type", [MemoryError, AttributeError])
def test_batch_failures_release_all_rows_and_retry_only_allocation_errors(
    scheduler_factory, monkeypatch, phase, error_type
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first"), make_request("second", start=20)]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")

    def fail(*args, **kwargs):
        raise error_type("injected batch failure")

    if phase == "forward":
        harness.forward_hook = fail
    else:
        monkeypatch.setattr(group, "extract", fail)
        if phase == "demotion":
            scheduler._prefill_speed_priority = True
    scheduled = []
    rejected = []

    scheduler._advance_chunked_prefills(scheduled, rejected)

    assert not scheduled
    assert not group.valid
    assert group.cache_nbytes == 0
    assert not scheduler._prefill_states
    assert not scheduler.prefilling
    assert not scheduler.running
    if error_type is MemoryError:
        assert not rejected
        assert list(scheduler.waiting) == requests
        assert all(request.prefill_oom_retries == 1 for request in requests)
        assert all(request.prompt_cache is None for request in requests)
        assert all(request._batched_prefill_disabled for request in requests)
        harness.forward_hook = None
        scheduler._schedule_waiting()
        completed, failures = finish_prefills(scheduler)
        assert completed == requests
        assert not failures
    else:
        assert not scheduler.waiting
        assert not scheduler.requests
        assert {output.request_id for output in rejected} == {"first", "second"}
        assert all(
            output.finished and output.finish_reason == "error" for output in rejected
        )


@pytest.mark.parametrize("phase", ["remove", "extract"])
@pytest.mark.parametrize("error_type", [MemoryError, RuntimeError])
@pytest.mark.parametrize("during_forward", [False, True])
def test_cancellation_cache_failure_finishes_abort_and_recovers_survivor(
    scheduler_factory, monkeypatch, phase, error_type, during_forward
):
    harness = scheduler_factory(prefill_max_batch_size=3)
    scheduler = harness.scheduler
    cancelled = make_request("cancelled", 13)
    survivors = [make_request("survivor", 13, start=20)]
    if phase == "remove":
        survivors.append(make_request("other-survivor", 13, start=40))
    add_requests(scheduler, cancelled, *survivors)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("cancelled")

    def fail(*args, **kwargs):
        raise error_type("injected cancellation cache failure")

    monkeypatch.setattr(group, phase, fail)
    scheduled = []
    rejected = []
    if during_forward:
        harness.forward_hook = lambda: scheduler.abort_request("cancelled")
        scheduler._advance_chunked_prefills(scheduled, rejected)
    else:
        scheduler.abort_request("cancelled")
        scheduler._process_pending_aborts()

    assert cancelled.status is RequestStatus.FINISHED_ABORTED
    assert "cancelled" not in scheduler.requests
    assert "cancelled" not in scheduler._prefill_states
    assert not group.valid
    assert group.cache_nbytes == 0
    scheduler._advance_chunked_prefills(scheduled, rejected)

    assert not scheduled
    assert not scheduler._prefill_states
    assert not scheduler.prefilling
    if error_type is MemoryError:
        assert list(scheduler.waiting) == survivors
        assert all(survivor.prefill_oom_retries == 1 for survivor in survivors)
        assert all(survivor._batched_prefill_disabled for survivor in survivors)
        assert not rejected
    else:
        assert not scheduler.waiting
        assert not scheduler.requests
        assert {output.request_id for output in rejected} == {
            survivor.request_id for survivor in survivors
        }
        assert all(
            output.finished and output.finish_reason == "error" for output in rejected
        )


@pytest.mark.parametrize("error_type", [MemoryError, RuntimeError])
def test_abort_all_during_forward_does_not_retry_failed_cache_cleanup(
    scheduler_factory, monkeypatch, error_type
):
    harness = scheduler_factory(prefill_max_batch_size=3)
    scheduler = harness.scheduler
    requests = [
        make_request("first"),
        make_request("second", start=20),
        make_request("third", start=40),
    ]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")

    def fail(*args, **kwargs):
        raise error_type("injected cancellation cache failure")

    def cancel_all():
        for request in requests:
            scheduler.abort_request(request.request_id)

    monkeypatch.setattr(group, "remove", fail)
    harness.forward_hook = cancel_all
    scheduled = []
    rejected = []
    scheduler._advance_chunked_prefills(scheduled, rejected)

    assert all(request.status is RequestStatus.FINISHED_ABORTED for request in requests)
    assert not scheduled
    assert not rejected
    assert not scheduler.requests
    assert not scheduler.waiting
    assert not scheduler.prefilling
    assert not scheduler._prefill_states
    assert not scheduler.running
    assert not group.valid
    assert group.cache_nbytes == 0


def test_group_setup_allocation_failure_falls_back_to_scalar(
    scheduler_factory, monkeypatch
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first"), make_request("second", start=20)]
    add_requests(scheduler, *requests)

    def fail(*args, **kwargs):
        raise MemoryError("batch cache construction failed")

    monkeypatch.setattr("omlx.prefill.execution.create_cold_batch_cache", fail)
    scheduled, rejected = scheduler._schedule_waiting()

    assert not scheduled
    assert not rejected
    assert forward_shapes(harness) == [(1, 4), (1, 4)]
    assert scheduler._batched_prefill_stats["groups"] == 0
    assert scheduler._batched_prefill_stats["fallbacks"]["cache_setup"] == 1
    scheduled, rejected = finish_prefills(scheduler)
    assert scheduled == requests
    assert not rejected


def test_later_insert_failure_preserves_already_handed_off_group_row(
    scheduler_factory,
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    first = make_request("first")
    second = make_request("second", start=20)
    add_requests(scheduler, first, second)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    harness.batch_generator.insert.side_effect = [
        [42],
        RuntimeError("decode insertion failed"),
    ]
    scheduled = []
    rejected = []

    scheduler._advance_chunked_prefills(scheduled, rejected)

    assert scheduled == [first]
    assert len(rejected) == 1
    assert rejected[0].request_id == "second"
    assert rejected[0].finished
    assert rejected[0].finish_reason == "error"
    assert scheduler.running == {"first": first}
    assert scheduler.requests == {"first": first}
    assert scheduler.request_id_to_uid == {"first": 42}
    assert not scheduler.prefilling
    assert not scheduler._prefill_states
    assert not scheduler.waiting
    assert not group.valid
    assert group.cache_nbytes == 0
    assert_insert_matches_single(
        harness, first, harness.batch_generator.insert.call_args_list[0]
    )


def test_insert_failure_after_cancellation_demotion_does_not_orphan_survivor(
    scheduler_factory,
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    first = make_request("first")
    second = make_request("second", start=20)
    add_requests(scheduler, first, second)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    harness.forward_hook = lambda: scheduler.abort_request("first")
    harness.batch_generator.insert.side_effect = RuntimeError("decode insertion failed")
    scheduled = []
    rejected = []

    scheduler._advance_chunked_prefills(scheduled, rejected)

    assert scheduled == []
    assert len(rejected) == 1
    assert rejected[0].request_id == "second"
    assert rejected[0].finished
    assert rejected[0].finish_reason == "error"
    assert first.status is RequestStatus.FINISHED_ABORTED
    assert not scheduler.requests
    assert not scheduler.running
    assert not scheduler.prefilling
    assert not scheduler._prefill_states
    assert not scheduler.waiting
    assert not group.valid
    assert group.cache_nbytes == 0


def test_memory_failure_retries_are_bounded_and_restart_from_cold_cache(
    scheduler_factory,
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first"), make_request("second", start=20)]
    add_requests(scheduler, *requests)
    harness.failure = "Memory limit exceeded during batched prefill"

    assert scheduler._schedule_waiting() == ([], [])
    assert {request.request_id for request in scheduler.waiting} == {"first", "second"}
    assert not scheduler._prefill_states
    assert not scheduler.prefilling
    for request in requests:
        assert request.prefill_oom_retries == 1
        assert request.prompt_cache is None
        assert request.cached_tokens == 0
        assert request.remaining_tokens == request.prompt_token_ids

    scheduled, rejected = scheduler._schedule_waiting()

    assert scheduled == []
    assert {output.request_id for output in rejected} == {"first", "second"}
    assert not scheduler.requests
    assert not scheduler.waiting
    assert not scheduler._prefill_states
    assert all(
        request.prefill_oom_retries == scheduler._MAX_PREFILL_OOM_RETRIES
        for request in requests
    )
    assert forward_shapes(harness) == [(2, 4)] + [(1, 4)] * (
        2 * scheduler._MAX_PREFILL_OOM_RETRIES
    )
    assert all(
        all(length == 0 for length in call.cache_lengths) for call in harness.calls
    )
    harness.batch_generator.insert.assert_not_called()


def test_failed_batch_can_recover_with_independent_cold_prefills(scheduler_factory):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first"), make_request("second", start=20)]
    add_requests(scheduler, *requests)
    harness.failure = "Memory limit exceeded during batched prefill"
    assert scheduler._schedule_waiting() == ([], [])
    harness.failure = None

    assert scheduler._schedule_waiting() == ([], [])
    scheduled, rejected = finish_prefills(scheduler)

    assert scheduled == requests
    assert rejected == []
    assert forward_shapes(harness) == [(2, 4), (1, 4), (1, 4), (1, 4), (1, 4)]
    assert all(request.prefill_oom_retries == 1 for request in requests)
    for request, insertion in zip(
        requests, harness.batch_generator.insert.call_args_list
    ):
        assert_insert_matches_single(harness, request, insertion)


@pytest.mark.parametrize("cancelled_id", ["first", "second"])
def test_cancellation_during_final_group_forward_keeps_other_row_correct(
    scheduler_factory, monkeypatch, cancelled_id
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first"), make_request("second", start=20)]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    harness.forward_hook = lambda: scheduler.abort_request(cancelled_id)

    def forbidden_transform(*args, **kwargs):
        pytest.fail(
            "A completed survivor must be handed off without filtering or demotion"
        )

    monkeypatch.setattr("omlx.prefill.execution.filter_rows", forbidden_transform)
    monkeypatch.setattr(scheduler, "_dissolve_prefill_group", forbidden_transform)
    scheduled, rejected = finish_prefills(scheduler)

    survivor = next(
        request for request in requests if request.request_id != cancelled_id
    )
    assert scheduled == [survivor]
    assert rejected == []
    assert cancelled_id not in scheduler.requests
    assert not scheduler._prefill_states
    assert not group.valid
    assert group.cache_nbytes == 0
    assert_insert_matches_single(
        harness, survivor, harness.batch_generator.insert.call_args
    )


def test_batch_members_each_consume_an_admission_slot(scheduler_factory):
    harness = scheduler_factory(max_num_seqs=2, prefill_max_batch_size=4)
    scheduler = harness.scheduler
    first = make_request("first")
    second = make_request("second", start=20)
    waiting = make_request("waiting", start=40)
    add_requests(scheduler, first, second, waiting)

    scheduler._schedule_waiting()

    assert forward_shapes(harness) == [(2, 4)]
    assert scheduler._num_admitted_requests() == 2
    assert list(scheduler.waiting) == [waiting]
    scheduler._schedule_waiting()
    assert forward_shapes(harness) == [(2, 4)]
    finish_prefills(scheduler)
    assert scheduler._num_admitted_requests() == 2
    scheduler._do_abort_request("first")
    scheduler._schedule_waiting()
    assert forward_shapes(harness)[-1] == (1, 4)
    assert not scheduler.waiting
    assert scheduler._num_admitted_requests() == 2


def test_token_budget_charges_every_row_in_a_forward(scheduler_factory):
    harness = scheduler_factory(max_num_batched_tokens=4)
    scheduler = harness.scheduler
    add_requests(scheduler, make_request("first"), make_request("second", start=20))

    scheduler._schedule_waiting()

    assert forward_shapes(harness) == [(2, 2)]
    assert all(
        state.tokens_processed == 2 for state in scheduler._prefill_states.values()
    )


def test_existing_group_finishes_safely_after_batching_is_disabled(scheduler_factory):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    add_requests(scheduler, make_request("first"), make_request("second", start=20))
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    scheduler.config.prefill_max_batch_size = 1

    scheduled, rejected = finish_prefills(scheduler)

    assert {request.request_id for request in scheduled} == {"first", "second"}
    assert rejected == []
    assert forward_shapes(harness) == [(2, 4), (2, 4)]
    assert not group.valid
    add_requests(
        scheduler, make_request("third", 5), make_request("fourth", 5, start=40)
    )
    scheduler._schedule_waiting()
    assert forward_shapes(harness)[-2:] == [(1, 4), (1, 4)]


def test_missing_memory_ceiling_falls_back_to_singleton_prefill(scheduler_factory):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    scheduler._memory_limit_bytes = 0
    scheduler._memory_hard_limit_bytes = 0
    scheduler._memory_abort_limit_bytes = 0
    requests = [make_request("first", 5), make_request("second", 5, start=20)]
    add_requests(scheduler, *requests)

    scheduled, rejected = scheduler._schedule_waiting()

    assert scheduled == requests
    assert rejected == []
    assert forward_shapes(harness) == [(1, 4), (1, 4)]


def test_batched_chunk_yields_to_decode_before_more_admission(
    scheduler_factory, monkeypatch
):
    harness = scheduler_factory(decode_fairness=True)
    scheduler = harness.scheduler
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("omlx.scheduler.time.perf_counter", lambda: clock.now)

    def advance_clock():
        clock.now += 0.5

    harness.forward_hook = advance_clock
    decoder = make_request("decoder")
    scheduler.running[decoder.request_id] = decoder
    scheduler.requests[decoder.request_id] = decoder
    requests = [
        make_request("first"),
        make_request("second", start=15),
        make_request("waiting", start=30),
    ]
    add_requests(scheduler, *requests)

    scheduler._schedule_waiting()
    scheduler._advance_chunked_prefills([], [])
    scheduler._schedule_waiting()

    assert forward_shapes(harness) == [(2, 4)]
    assert list(scheduler.waiting) == [requests[-1]]
    assert scheduler._decode_time_owed_s > 0
    scheduler._repay_decode_debt(scheduler._decode_time_owed_s)
    scheduled = []
    scheduler._advance_chunked_prefills(scheduled, [])
    scheduler._schedule_waiting()
    assert scheduled == requests[:2]
    assert forward_shapes(harness) == [(2, 4), (2, 4)]
    assert list(scheduler.waiting) == [requests[-1]]


def test_step_shares_one_budget_across_existing_groups_and_new_admission(
    scheduler_factory,
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    existing = [
        make_request("first", 13),
        make_request("second", 13, start=10),
        make_request("third", 13, start=20),
        make_request("fourth", 13, start=30),
    ]
    add_requests(scheduler, *existing)
    scheduler._schedule_waiting()
    scheduler._schedule_waiting()
    arriving = [make_request("fifth", 5, start=40), make_request("sixth", 5, start=50)]
    add_requests(scheduler, *arriving)
    scheduler.config.max_num_batched_tokens = 10

    for _ in range(3):
        output, token_count = step_with_token_count(harness)
        assert token_count == 10
        assert not output.outputs

    assert all(request.request_id in scheduler.running for request in existing[:2])
    assert all(
        scheduler._prefill_states[request.request_id].tokens_processed == 1
        for request in arriving
    )
    for _ in range(16):
        if not scheduler.prefilling and not scheduler.waiting:
            break
        output, token_count = step_with_token_count(harness)
        assert 0 < token_count <= 10
        assert not output.outputs

    assert not scheduler.prefilling
    assert not scheduler.waiting
    assert set(scheduler.running) == {
        request.request_id for request in existing + arriving
    }
    assert all(request.prefill_oom_retries == 0 for request in existing + arriving)


def test_remaining_budget_smaller_than_group_width_yields_without_retry(
    scheduler_factory,
):
    harness = scheduler_factory(prefill_max_batch_size=3)
    scheduler = harness.scheduler
    singleton = make_request("singleton")
    scheduler.add_request(singleton)
    scheduler._schedule_waiting()
    grouped = [
        make_request("first", start=15),
        make_request("second", start=30),
        make_request("third", start=45),
    ]
    add_requests(scheduler, *grouped)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    scheduler.config.max_num_batched_tokens = 5
    first_call = len(harness.calls)

    output, token_count = step_with_token_count(harness)

    assert token_count == 4
    assert [call.shape for call in harness.calls[first_call:]] == [(1, 4)]
    assert not output.outputs
    assert group.valid
    assert group.tokens_processed == 4
    assert all(request.prefill_oom_retries == 0 for request in grouped)
    assert not scheduler.waiting
    output, token_count = step_with_token_count(harness)
    assert token_count == 3
    assert not output.outputs
    assert group.tokens_processed == 5
    for _ in range(8):
        if not scheduler.prefilling:
            break
        output, token_count = step_with_token_count(harness)
        assert 0 < token_count <= 5
        assert not output.outputs

    assert not scheduler.prefilling
    assert set(scheduler.running) == {"singleton", "first", "second", "third"}
    assert all(request.prefill_oom_retries == 0 for request in grouped)


def test_live_budget_reduction_dissolves_group_without_losing_progress(
    scheduler_factory,
):
    harness = scheduler_factory(prefill_max_batch_size=3)
    scheduler = harness.scheduler
    requests = [
        make_request("first"),
        make_request("second", start=15),
        make_request("third", start=30),
    ]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    scheduler.config.max_num_batched_tokens = 2
    first_call = len(harness.calls)

    output, token_count = step_with_token_count(harness)

    assert token_count == 2
    assert not output.outputs
    assert not group.valid
    assert harness.calls[first_call].cache_lengths == (4, 4)
    for _ in range(16):
        if not scheduler.prefilling:
            break
        output, token_count = step_with_token_count(harness)
        assert 0 < token_count <= 2
        assert not output.outputs

    assert not scheduler.prefilling
    assert all(call.shape[0] == 1 for call in harness.calls[first_call:])
    assert all(request.prefill_oom_retries == 0 for request in requests)
    inserted = {
        insertion.kwargs["all_tokens"][0][0]: insertion
        for insertion in harness.batch_generator.insert.call_args_list
    }
    for request in requests:
        assert_insert_matches_single(
            harness, request, inserted[request.prompt_token_ids[0]]
        )


@pytest.mark.parametrize("fallback", ["position_state", "missing_memory_ceiling"])
def test_scalar_fallback_respects_the_shared_step_budget(scheduler_factory, fallback):
    harness = scheduler_factory(max_num_batched_tokens=3, chunked_prefill=False)
    scheduler = harness.scheduler
    requests = [make_request("first", 5), make_request("second", 5, start=20)]
    if fallback == "position_state":
        for request in requests:
            request.rope_deltas = 1
    else:
        scheduler._memory_limit_bytes = 0
        scheduler._memory_hard_limit_bytes = 0
        scheduler._memory_abort_limit_bytes = 0
    add_requests(scheduler, *requests)

    for _ in range(8):
        if not scheduler.waiting and not scheduler.prefilling:
            break
        output, token_count = step_with_token_count(harness)
        assert 0 < token_count <= 3
        assert not output.outputs

    assert not scheduler.waiting
    assert not scheduler.prefilling
    assert set(scheduler.running) == {"first", "second"}
    assert all(call.shape[0] == 1 for call in harness.calls)
    inserted = {
        insertion.kwargs["all_tokens"][0][0]: insertion
        for insertion in harness.batch_generator.insert.call_args_list
    }
    for request in requests:
        assert_insert_matches_single(
            harness, request, inserted[request.prompt_token_ids[0]]
        )


@pytest.mark.parametrize("batched", [True, False])
def test_failed_forward_still_consumes_budget_before_retry_admission(
    scheduler_factory, batched
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first"), make_request("second", start=20)]
    if not batched:
        for request in requests:
            request.rope_deltas = 1
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    scheduler.add_request(make_request("waiting", start=40))
    scheduler.config.max_num_batched_tokens = 8 if batched else 4
    harness.failure = "Memory limit exceeded during budgeted prefill"
    first_call = len(harness.calls)

    output, token_count = step_with_token_count(harness)

    assert not output.outputs
    assert token_count == scheduler.config.max_num_batched_tokens
    assert len(harness.calls) == first_call + 1
    assert requests[0].prefill_oom_retries == 1
    assert requests[1].prefill_oom_retries == int(batched)
    assert "waiting" not in scheduler._prefill_states
    harness.failure = None
    for _ in range(16):
        if not scheduler.prefilling and not scheduler.waiting:
            break
        output, token_count = step_with_token_count(harness)
        assert 0 < token_count <= scheduler.config.max_num_batched_tokens
        assert not output.outputs

    assert not scheduler.prefilling
    assert not scheduler.waiting
    assert set(scheduler.running) == {"first", "second", "waiting"}


def test_default_singleton_mode_does_not_activate_experimental_token_budget(
    scheduler_factory,
):
    harness = scheduler_factory(
        prefill_max_batch_size=1,
        max_num_batched_tokens=1,
        chunked_prefill=False,
    )
    requests = [make_request("first", 5), make_request("second", 5, start=20)]
    add_requests(harness.scheduler, *requests)

    output, token_count = step_with_token_count(harness)

    assert not output.outputs
    assert output.scheduled_request_ids == ["first", "second"]
    assert token_count == 8
    assert forward_shapes(harness) == [(1, 4), (1, 4)]


def test_enabling_batching_during_forward_applies_on_the_next_step(
    scheduler_factory,
):
    harness = scheduler_factory(prefill_max_batch_size=1, max_num_batched_tokens=4)
    scheduler = harness.scheduler
    requests = [
        make_request("first", 5),
        make_request("second", 5, start=10),
        make_request("third", 5, start=20),
    ]
    add_requests(scheduler, *requests)
    harness.forward_hook = lambda: setattr(
        scheduler.config, "prefill_max_batch_size", 2
    )

    output, token_count = step_with_token_count(harness)

    assert not output.outputs
    assert token_count == 12
    assert forward_shapes(harness) == [(1, 4), (1, 4), (1, 4)]
    assert not scheduler.prefilling
    add_requests(
        scheduler,
        make_request("fourth", 5, start=30),
        make_request("fifth", 5, start=40),
    )
    output, token_count = step_with_token_count(harness)
    assert not output.outputs
    assert token_count == 4
    assert forward_shapes(harness)[-1] == (2, 2)


def test_disabling_batching_during_forward_applies_on_the_next_step(
    scheduler_factory,
):
    harness = scheduler_factory(max_num_batched_tokens=10)
    scheduler = harness.scheduler
    first = make_request("first", 3)
    first.rope_deltas = 1
    add_requests(
        scheduler,
        first,
        make_request("second", 5, start=10),
        make_request("third", 5, start=20),
    )
    harness.forward_hook = lambda: setattr(
        scheduler.config, "prefill_max_batch_size", 1
    )

    output, token_count = step_with_token_count(harness)

    assert not output.outputs
    assert token_count == 10
    assert forward_shapes(harness) == [(1, 2), (2, 4)]
    assert not scheduler.prefilling
    add_requests(
        scheduler,
        make_request("fourth", 5, start=30),
        make_request("fifth", 5, start=40),
    )
    output, token_count = step_with_token_count(harness)
    assert not output.outputs
    assert token_count == 8
    assert forward_shapes(harness)[-2:] == [(1, 4), (1, 4)]


def test_duration_estimates_use_one_prompt_bucket_for_search_and_observation(
    scheduler_factory, monkeypatch
):
    harness = scheduler_factory(prefill_step_size=8, decode_fairness=True)
    scheduler = harness.scheduler
    decoder = make_request("decoder")
    scheduler.running["decoder"] = decoder
    scheduler.requests["decoder"] = decoder
    requests = [make_request("first", 17), make_request("second", 25, start=25)]
    prompt_bucket = max(request.num_prompt_tokens for request in requests).bit_length()
    competing_buckets = {(2, 1): 1000.0, (2, 2): 0.1, (2, 3): 1000.0, (2, 4): 0.01}
    for key, rate in {**competing_buckets, (2, prompt_bucket): 8.0}.items():
        timing = PrefillTiming()
        timing.observe(8, 8 / rate, now=100.0)
        scheduler._batched_prefill_timings[key] = timing
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("omlx.scheduler.time.perf_counter", lambda: clock.now)
    observations = []

    def observe_planning(candidates, **kwargs):
        estimates = [
            kwargs["estimate"](2, chunk_tokens, 0) for chunk_tokens in range(1, 9)
        ]
        durations = [estimate.duration_seconds for estimate in estimates]
        assert durations == sorted(durations)
        assert durations == pytest.approx(
            [chunk_tokens / 4 for chunk_tokens in range(1, 9)]
        )
        plan = plan_prefill_batch(candidates, **kwargs)
        observations.append(plan)
        return plan

    def advance_clock():
        clock.now += 1.0

    monkeypatch.setattr("omlx.scheduler.plan_prefill_batch", observe_planning)
    harness.forward_hook = advance_clock
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()

    assert observations
    assert all(
        isinstance(decision, PrefillRun) and decision.plan.chunk_tokens == 2
        for decision in observations
    )
    assert forward_shapes(harness) == [(2, 2)]
    assert scheduler._batched_prefill_timings[(2, prompt_bucket)].estimate(
        4
    ) == pytest.approx(1.0)
    assert all(
        scheduler._batched_prefill_timings[key].estimate(8) == pytest.approx(8 / rate)
        for key, rate in competing_buckets.items()
    )


def test_handoff_time_accrues_decode_debt_without_poisoning_forward_estimates(
    scheduler_factory, monkeypatch
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("omlx.scheduler.time.perf_counter", lambda: clock.now)
    original_insert = scheduler._insert_prefilled_request
    debt_observations = []

    def forward_delay():
        clock.now += 0.25

    def insert_with_delay(*args, **kwargs):
        clock.now += 0.5
        return original_insert(*args, **kwargs)

    harness.forward_hook = forward_delay
    monkeypatch.setattr(scheduler, "_insert_prefilled_request", insert_with_delay)
    monkeypatch.setattr(scheduler, "_accrue_decode_debt", debt_observations.append)
    add_requests(scheduler, make_request("first", 5), make_request("second", 5))

    scheduler._schedule_waiting()

    timing = scheduler._batched_prefill_timings[(2, (5).bit_length())]
    assert timing.estimate(8) == pytest.approx(0.25)
    assert debt_observations == pytest.approx([1.25])
    assert harness.batch_generator.insert.call_count == 2


def test_time_only_rejection_demotes_live_group_without_replaying_prefill(
    scheduler_factory, monkeypatch
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first"), make_request("second", start=15)]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    first_call = len(harness.calls)
    scheduler._decode_fairness = True
    monkeypatch.setattr(scheduler, "_decode_contention", lambda: True)
    monkeypatch.setattr(scheduler, "_prefill_gate_open", lambda: True)
    timing = PrefillTiming()
    timing.observe(2, 1.0)
    scheduler._batched_prefill_timings[(2, (9).bit_length())] = timing
    rejected = []

    scheduler._advance_chunked_prefills([], rejected)

    assert not rejected
    assert not group.valid
    assert len(harness.calls) == first_call
    assert scheduler._batched_prefill_stats["failures"] == 0
    assert scheduler._batched_prefill_stats["fallbacks"]["duration_limit"] == 1
    for state in scheduler._prefill_states.values():
        assert scheduler._prefill_runtime.group_for(state.request.request_id) is None
        assert state.tokens_processed == 4
        assert all(
            type(layer) is KVCache and layer.offset == 4 for layer in state.cache
        )
    _, rejected = finish_prefills(scheduler)
    assert not rejected
    assert all(call.shape == (1, 4) for call in harness.calls[first_call:])
    assert all(request.prefill_oom_retries == 0 for request in requests)
    for request, insertion in zip(
        requests, harness.batch_generator.insert.call_args_list
    ):
        assert_insert_matches_single(harness, request, insertion)


def test_memory_rejection_preserves_kv_when_transition_fits(
    scheduler_factory, monkeypatch
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first"), make_request("second", start=15)]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    first_call = len(harness.calls)
    original_plan = scheduler._plan_batched_prefill
    monkeypatch.setattr(
        scheduler, "_plan_batched_prefill",
        lambda *args: PrefillDefer(PrefillReason.MEMORY_LIMIT),
    )
    rejected = []
    scheduler._advance_chunked_prefills([], rejected)
    assert not rejected
    assert not scheduler._prefill_runtime.has_groups
    assert len(harness.calls) == first_call
    assert scheduler._batched_prefill_stats["memory_demotions"] == 1
    assert scheduler._batched_prefill_stats["discarded_tokens"] == 0
    assert scheduler._batched_prefill_stats["requeued_tokens"] == 0
    for state in scheduler._prefill_states.values():
        assert state.tokens_processed == 4
        assert all(type(layer) is KVCache and layer.offset == 4 for layer in state.cache)
    monkeypatch.setattr(scheduler, "_plan_batched_prefill", original_plan)
    _, rejected = finish_prefills(scheduler)
    assert not rejected
    assert all(call.shape == (1, 4) for call in harness.calls[first_call:])
    assert all(request.prefill_oom_retries == 0 for request in requests)
    for request, insertion in zip(requests, harness.batch_generator.insert.call_args_list):
        assert_insert_matches_single(harness, request, insertion)


def test_memory_rejection_cannot_demote_without_transition_headroom(
    scheduler_factory, monkeypatch
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    add_requests(scheduler, make_request("first"), make_request("second", start=15))
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    first_call = len(harness.calls)
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda: 10**13)
    monkeypatch.setattr(
        scheduler, "_requeue_or_fail_prefill", lambda *args, **kwargs: False
    )
    rejected = []

    scheduler._advance_chunked_prefills([], rejected)

    assert not group.valid
    assert len(harness.calls) == first_call
    assert not scheduler._prefill_states
    assert scheduler._batched_prefill_stats["failures"] == 0
    assert scheduler._batched_prefill_stats["fallbacks"]["duration_limit"] == 0
    assert len(rejected) == 2
    assert all(output.error_code == "prefill_memory_exceeded" for output in rejected)
    assert scheduler._batched_prefill_stats["fallbacks"]["memory_limit"] == 1


def test_speed_priority_bypasses_grouping_for_new_and_already_staged_rows(
    scheduler_factory, monkeypatch
):
    def with_speed_priority(scheduler):
        original_advance = scheduler._advance_chunked_prefills

        def enable_speed_and_advance(*args, **kwargs):
            scheduler._prefill_speed_priority = True
            return original_advance(*args, **kwargs)

        return enable_speed_and_advance

    for staged_before_change in (False, True):
        harness = scheduler_factory(prefill_speed_priority=not staged_before_change)
        scheduler = harness.scheduler
        if staged_before_change:
            monkeypatch.setattr(
                scheduler, "_advance_chunked_prefills", with_speed_priority(scheduler)
            )
        add_requests(
            scheduler,
            make_request("first", 5),
            make_request("second", 5, start=20),
        )

        output, token_count = step_with_token_count(harness)

        assert not output.outputs
        assert token_count == 8
        assert forward_shapes(harness) == [(1, 4), (1, 4)]
        assert scheduler._batched_prefill_stats["groups"] == 0
        assert not scheduler.prefilling
        if not staged_before_change:
            assert scheduler._batched_prefill_stats["fallbacks"]["speed_priority"] == 2


def test_enabling_speed_priority_dissolves_existing_group_and_preserves_cache(
    scheduler_factory,
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first"), make_request("second", start=20)]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    scheduler._prefill_speed_priority = True

    output, token_count = step_with_token_count(harness)

    assert not output.outputs
    assert token_count == 8
    assert forward_shapes(harness) == [(2, 4), (1, 4), (1, 4)]
    assert not group.valid
    assert group.cache_nbytes == 0
    assert not scheduler._prefill_states
    assert all(request.prefill_oom_retries == 0 for request in requests)
    for request, insertion in zip(
        requests, harness.batch_generator.insert.call_args_list
    ):
        assert_insert_matches_single(harness, request, insertion)


@pytest.mark.parametrize("error_type", [MemoryError, RuntimeError])
@pytest.mark.parametrize("survivor_count", [1, 2])
def test_survivor_transition_failure_recovers_only_uncommitted_rows(
    scheduler_factory, monkeypatch, error_type, survivor_count
):
    harness = scheduler_factory(prefill_max_batch_size=4)
    scheduler = harness.scheduler
    first = make_request("first", 9)
    second = make_request("second", 9, start=15)
    survivors = [make_request("last", 13, start=30)]
    if survivor_count == 2:
        survivors.append(make_request("other-last", 13, start=45))
    add_requests(scheduler, first, second, *survivors)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")

    def fail_after_commits(request_ids):
        assert request_ids == ["first", "second"]
        assert scheduler._prefill_runtime.owned_ids(group) == ("last", "other-last")
        raise error_type("compaction failed")

    original_extract = group.extract

    def fail_demotion(request_id):
        if request_id == "last":
            assert scheduler._prefill_runtime.owned_ids(group) == ("last",)
            assert group.batch_size == 3
            raise error_type("demotion failed")
        return original_extract(request_id)

    if survivor_count == 1:
        monkeypatch.setattr(group, "extract", fail_demotion)
    else:
        monkeypatch.setattr(group, "remove", fail_after_commits)
    scheduled, rejected = [], []
    scheduler._advance_chunked_prefills(scheduled, rejected)

    assert scheduled == [first, second]
    assert scheduler.running == {"first": first, "second": second}
    assert not scheduler._prefill_runtime.has_groups
    assert not scheduler._prefill_states
    assert not group.valid
    assert harness.batch_generator.insert.call_count == 2
    if error_type is MemoryError:
        assert list(scheduler.waiting) == survivors
        assert all(survivor.prefill_oom_retries == 1 for survivor in survivors)
        assert not rejected
    else:
        assert not scheduler.waiting
        assert {output.request_id for output in rejected} == {
            survivor.request_id for survivor in survivors
        }
    assert first.prefill_oom_retries == second.prefill_oom_retries == 0
    for request, insertion in zip(
        [first, second], harness.batch_generator.insert.call_args_list
    ):
        assert_insert_matches_single(harness, request, insertion)


@pytest.mark.parametrize("transition", ["handoff", "demotion", "cancellation"])
def test_transition_rechecks_memory_before_allocating_scalar_caches(
    scheduler_factory, monkeypatch, transition
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    first = make_request("first")
    second = make_request("second", start=20)
    add_requests(scheduler, first, second)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    extractions = []

    def forbidden_extract(request_id):
        extractions.append(request_id)
        raise AssertionError("memory guard must run before extraction")

    def lose_headroom():
        # Exceed transition admission headroom, but stay below the physical
        # cap so this exercises the transition guard, not the post-forward guard.
        monkeypatch.setattr(
            scheduler,
            "_current_usage_bytes",
            lambda: scheduler._prefill_abort_cap() + 1,
        )

    monkeypatch.setattr(group, "extract", forbidden_extract)
    monkeypatch.setattr(
        scheduler, "_requeue_or_fail_prefill", lambda *args, **kwargs: False
    )
    scheduled, rejected = [], []
    if transition == "handoff":
        harness.forward_hook = lose_headroom
    else:
        lose_headroom()
        if transition == "demotion":
            scheduler._prefill_speed_priority = True
        else:
            scheduler.abort_request("first")
            scheduler._process_pending_aborts()
            assert first.status is RequestStatus.FINISHED_ABORTED

    scheduler._advance_chunked_prefills(scheduled, rejected)

    assert not scheduled
    assert not extractions
    assert not group.valid
    assert not scheduler._prefill_runtime.has_groups
    assert not scheduler._prefill_states
    assert not scheduler.running
    assert {output.request_id for output in rejected} == (
        {"second"} if transition == "cancellation" else {"first", "second"}
    )
    assert all(output.error_code == "prefill_memory_exceeded" for output in rejected)
    assert scheduler._batched_prefill_stats["failures"] == 0


def test_memory_backpressure_can_reclaim_and_continue_same_group(
    scheduler_factory, monkeypatch
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first", 13), make_request("second", 13, start=20)]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    usage = SimpleNamespace(value=10**13)
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda: usage.value)

    def reclaim():
        usage.value = 1
        return usage.value

    monkeypatch.setattr(scheduler, "_reclaim_prefill_headroom", reclaim)
    scheduled, rejected = [], []
    scheduler._advance_chunked_prefills(scheduled, rejected)

    assert not scheduled
    assert not rejected
    assert group is scheduler._prefill_runtime.group_for("first")
    assert group.tokens_processed == 8
    assert forward_shapes(harness) == [(2, 4), (2, 4)]
    assert all(request.prefill_oom_retries == 0 for request in requests)
    assert scheduler._batched_prefill_stats["failures"] == 0


def test_empty_decode_acceptance_cannot_commit_a_prefill_row(scheduler_factory):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    add_requests(scheduler, make_request("first"), make_request("second", start=20))
    scheduler._schedule_waiting()
    harness.batch_generator.insert.side_effect = lambda *args, **kwargs: []
    scheduled, rejected = [], []
    scheduler._advance_chunked_prefills(scheduled, rejected)
    assert not scheduled
    assert not scheduler.running
    assert not scheduler._prefill_runtime.has_groups
    assert {output.request_id for output in rejected} == {"first", "second"}


@pytest.mark.parametrize("transition", ["handoff", "demotion", "cancellation"])
def test_transition_reclaim_preserves_completed_work(
    scheduler_factory, monkeypatch, transition
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first"), make_request("second", start=20)]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    group = scheduler._prefill_runtime.group_for("first")
    usage = SimpleNamespace(value=1, reclaims=0)
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda: usage.value)

    def reclaim():
        usage.reclaims += 1
        usage.value = 1
        return usage.value

    def lose_headroom():
        usage.value = scheduler._prefill_abort_cap() + 1

    monkeypatch.setattr(scheduler, "_reclaim_prefill_headroom", reclaim)
    scheduled, rejected = [], []
    if transition == "handoff":
        harness.forward_hook = lose_headroom
        scheduler._advance_chunked_prefills(scheduled, rejected)
    else:
        lose_headroom()
        if transition == "demotion":
            scheduler._dissolve_prefill_group(group)
        else:
            scheduler.abort_request("first")
            scheduler._process_pending_aborts()
        assert scheduler._prefill_states["second"].cache[0].offset == 4
        assert forward_shapes(harness) == [(2, 4)]
        scheduled, rejected = finish_prefills(scheduler)

    assert not rejected
    assert usage.reclaims == 1
    assert scheduler._batched_prefill_stats["transition_reclaims"] == 1
    assert scheduler._batched_prefill_stats["transition_rejections"] == 0
    assert scheduler._batched_prefill_stats["discarded_tokens"] == 0
    assert all(request.prefill_oom_retries == 0 for request in requests)
    assert not scheduler._prefill_runtime.has_groups
    expected = requests[1:] if transition == "cancellation" else requests
    assert {request.request_id for request in scheduled} == {
        request.request_id for request in expected
    }
    for request, insertion in zip(
        expected, harness.batch_generator.insert.call_args_list
    ):
        assert_insert_matches_single(harness, request, insertion)


def test_partial_handoff_compacts_without_reserving_committed_rows_again(
    scheduler_factory, monkeypatch
):
    harness = scheduler_factory(prefill_max_batch_size=4)
    scheduler = harness.scheduler
    requests = [
        make_request("short-first", 5),
        make_request("short-second", 5, start=10),
        make_request("long-first", 9, start=20),
        make_request("long-second", 9, start=30),
    ]
    limit = SimpleNamespace(value=10**12)
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda: 1)
    monkeypatch.setattr(scheduler, "_prefill_abort_cap", lambda: limit.value)
    monkeypatch.setattr(scheduler, "_admission_limit_bytes", lambda: limit.value)
    original_insert = harness.batch_generator.insert.side_effect
    insertions = []

    def insert_and_reduce_headroom(*args, **kwargs):
        inserted = original_insert(*args, **kwargs)
        insertions.extend(inserted)
        if len(insertions) == 2:
            group = scheduler._prefill_runtime.group_for("long-first")
            context = scheduler._prefill_memory_context(group)
            cost = context.estimate_transition(
                PrefillTransition.COMPACT,
                physical_rows=4,
                owned_rows=2,
                cache_tokens=4,
            )
            limit.value = context.current_usage_bytes + cost.additional_peak_bytes
            assert (
                context.estimate(4, 1, 9).additional_peak_bytes
                > cost.additional_peak_bytes
            )
        return inserted

    harness.batch_generator.insert.side_effect = insert_and_reduce_headroom
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()

    group = scheduler._prefill_runtime.group_for("long-first")
    assert group is not None
    assert group.request_ids == ("long-first", "long-second")
    assert group.tokens_processed == 4
    assert set(scheduler.running) == {"short-first", "short-second"}
    assert scheduler._batched_prefill_stats["transition_reclaims"] == 0
    assert all(request.prefill_oom_retries == 0 for request in requests)
    limit.value = 10**12
    scheduled, rejected = finish_prefills(scheduler)
    assert not rejected
    assert len(scheduled) == 2
    assert forward_shapes(harness) == [(4, 4), (2, 4)]
    for request, insertion in zip(
        requests, harness.batch_generator.insert.call_args_list
    ):
        assert_insert_matches_single(harness, request, insertion)


def test_completed_forward_is_observed_even_when_handoff_fails(
    scheduler_factory, monkeypatch, caplog
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("omlx.scheduler.time.perf_counter", lambda: clock.now)

    def forward_delay():
        clock.now += 0.25

    def fail_insert(*args, **kwargs):
        clock.now += 0.5
        raise RuntimeError("injected handoff failure")

    harness.forward_hook = forward_delay
    harness.batch_generator.insert.side_effect = fail_insert
    add_requests(scheduler, make_request("first", 5), make_request("second", 5))
    with caplog.at_level(logging.DEBUG, logger="omlx.scheduler"):
        scheduler._schedule_waiting()

    stats = scheduler._batched_prefill_stats
    assert stats["chunks"] == 1
    assert stats["tokens"] == stats["discarded_tokens"] == 8
    assert stats["requeued_tokens"] == 0
    assert stats["forward_seconds"] == pytest.approx(0.25)
    assert stats["step_seconds"] == pytest.approx(0.75)
    assert scheduler._batched_prefill_timings[(2, 3)].estimate(8) == pytest.approx(0.25)
    assert (
        "forward_ms=250.000 boundary_ms=500.000 outcome=failed phase=handoff"
        in caplog.text
    )


def test_failed_forward_counts_only_previously_materialized_requeued_work(
    scheduler_factory, caplog
):
    harness = scheduler_factory()
    scheduler = harness.scheduler
    requests = [make_request("first"), make_request("second", start=20)]
    add_requests(scheduler, *requests)
    scheduler._schedule_waiting()
    harness.failure = "Memory limit exceeded during forward"
    with caplog.at_level(logging.DEBUG, logger="omlx.scheduler"):
        scheduler._advance_chunked_prefills([], [])

    stats = scheduler._batched_prefill_stats
    assert stats["chunks"] == 1
    assert stats["tokens"] == stats["discarded_tokens"] == stats["requeued_tokens"] == 8
    assert all(request.prefill_oom_retries == 1 for request in requests)
    assert "outcome=failed phase=forward" in caplog.text


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("prompt_tokens", [5, 9])
def test_post_forward_over_cap_stops_before_cache_transitions(
    scheduler_factory, monkeypatch, batch_size, prompt_tokens
):
    """Check actual occupancy even when a chunk has no completed rows."""
    harness = scheduler_factory(prefill_max_batch_size=batch_size)
    scheduler = harness.scheduler
    usage = SimpleNamespace(value=1)
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda: usage.value)
    reclaim = MagicMock(side_effect=lambda: usage.value)
    monkeypatch.setattr(scheduler, "_reclaim_prefill_headroom", reclaim)
    monkeypatch.setattr(
        scheduler, "_requeue_or_fail_prefill", lambda *args, **kwargs: False
    )
    transition = MagicMock(wraps=scheduler._guard_prefill_group_transition)
    monkeypatch.setattr(scheduler, "_guard_prefill_group_transition", transition)
    harness.forward_hook = lambda: setattr(usage, "value", 2 * 10**12)
    requests = [make_request(f"row-{i}", prompt_tokens) for i in range(batch_size)]
    add_requests(scheduler, *requests)

    scheduled, rejected = scheduler._schedule_waiting()

    assert forward_shapes(harness) == [(batch_size, 4)]
    reclaim.assert_called_once()
    transition.assert_not_called()
    harness.batch_generator.insert.assert_not_called()
    assert not scheduled
    assert {output.request_id for output in rejected} == {
        r.request_id for r in requests
    }
    assert all("Memory limit exceeded" in output.error for output in rejected)
    assert not scheduler._prefill_runtime.has_groups
    assert not scheduler._prefill_states
    assert not scheduler.running
    if batch_size > 1:
        stats = scheduler._batched_prefill_stats
        assert stats["tokens"] == stats["discarded_tokens"] == batch_size * 4


@pytest.mark.parametrize("batch_size", [1, 2])
def test_post_forward_reclaim_precedes_handoff_and_retains_work(
    scheduler_factory, monkeypatch, batch_size
):
    harness = scheduler_factory(prefill_max_batch_size=batch_size)
    scheduler = harness.scheduler
    usage = SimpleNamespace(value=1)
    events = []
    monkeypatch.setattr(scheduler, "_current_usage_bytes", lambda: usage.value)

    def reclaim():
        events.append("reclaim")
        usage.value = 1
        return usage.value

    original_transition = scheduler._guard_prefill_group_transition

    def transition(*args):
        events.append("transition")
        return original_transition(*args)

    monkeypatch.setattr(scheduler, "_reclaim_prefill_headroom", reclaim)
    monkeypatch.setattr(scheduler, "_guard_prefill_group_transition", transition)
    harness.forward_hook = lambda: setattr(usage, "value", 2 * 10**12)
    requests = [make_request(f"row-{i}", 5) for i in range(batch_size)]
    add_requests(scheduler, *requests)

    scheduled, rejected = scheduler._schedule_waiting()

    assert not rejected
    assert events[0] == "reclaim"
    assert events.count("reclaim") == 1
    assert forward_shapes(harness) == [(batch_size, 4)]
    assert scheduled == requests
    assert harness.batch_generator.insert.call_count == batch_size
    assert all(request.prefill_oom_retries == 0 for request in requests)
    assert not scheduler._prefill_runtime.has_groups
