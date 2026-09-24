"""Tiny-model contracts through real scheduler prefill and batched decode."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import make_prompt_cache
from prefill_helpers import make_request
from prefill_helpers import model_factory as model_factory

from omlx import scheduler as scheduler_module
from omlx.decode_activity import get_decode_activity
from omlx.prefill_progress import get_prefill_tracker
from omlx.request import Request, RequestStatus
from omlx.scheduler import Scheduler, SchedulerConfig


class TinyTokenizer:
    eos_token_id = None
    eos_token_ids = set()
    name_or_path = ""
    vocab_size = 64

    def encode(self, text, add_special_tokens=True):
        return [40, 41] if text == "<>" else [0]

    def decode(self, token_ids, **kwargs):
        pieces = {40: "<", 41: ">"}
        return "".join(pieces.get(token_id, f"{token_id} ") for token_id in token_ids)

    def get_vocab(self):
        return {str(token_id): token_id for token_id in range(self.vocab_size)}


@dataclass
class TokenHistoryAudit:
    request_id: str
    prompt: tuple[int, ...]
    history: list[tuple[int, ...]] = field(default_factory=list)

    def __call__(self, tokens, logits):
        context = tuple(tokens.tolist())
        assert context[: len(self.prompt)] == self.prompt
        assert len(context) == len(self.prompt) + len(self.history)
        self.history.append(context)
        return logits


@pytest.fixture(autouse=True)
def isolated_decode_activity():
    get_decode_activity().clear()
    get_prefill_tracker().clear()
    yield
    get_decode_activity().clear()
    get_prefill_tracker().clear()


@pytest.fixture
def scheduler_factory(monkeypatch):
    schedulers = []

    def create(harness, batch_size, *, chunk_tokens=4, forced_tokens=None):
        scheduler = Scheduler(
            model=harness.model,
            tokenizer=TinyTokenizer(),
            config=SchedulerConfig(
                max_num_seqs=4,
                completion_batch_size=4,
                max_num_batched_tokens=1024,
                prefill_step_size=chunk_tokens,
                prefill_max_batch_size=batch_size,
                chunked_prefill=True,
                decode_fairness=False,
                paged_cache_block_size=0,
            ),
            stream=harness.stream,
        )
        scheduler._memory_limit_bytes = 10**12
        scheduler._memory_hard_limit_bytes = 10**12
        scheduler._memory_abort_limit_bytes = 10**12
        scheduler._memory_limits_propagated = True
        run = SimpleNamespace(
            scheduler=scheduler,
            model=harness,
            audits={},
            insertions={},
            outputs=defaultdict(list),
            finished={},
            checked_join=False,
            checked_memberships=set(),
            cache_checked_ids=set(),
        )
        original_build = scheduler._build_sampler_and_processors
        original_create = scheduler._create_batch_generator

        def build_processors(params, request=None):
            sampler, processors = original_build(params, request)
            if request is None:
                return (sampler, processors)
            audit = TokenHistoryAudit(
                request.request_id, tuple(request.prompt_token_ids)
            )
            assert request.request_id not in run.audits
            run.audits[request.request_id] = audit
            processors = [audit, *processors]
            if forced_tokens and request.request_id in forced_tokens:
                sequence = forced_tokens[request.request_id]

                def force_sequence(tokens, logits):
                    generated = len(tokens) - len(audit.prompt)
                    selected = sequence[min(generated, len(sequence) - 1)]
                    forced = mx.full_like(logits, -10000.0)
                    forced[..., selected] = 0
                    return forced

                processors.append(force_sequence)
            return (sampler, processors)

        def create_generator(params):
            generator = original_create(params)
            assert type(generator) is BatchGenerator
            original_insert = generator.insert

            def record_insert(prompts, **kwargs):
                processors = kwargs["logits_processors"][0]
                audit = next(
                    processor
                    for processor in processors
                    if isinstance(processor, TokenHistoryAudit)
                )
                assert kwargs["all_tokens"] == [list(audit.prompt[:-1])]
                assert prompts == [list(audit.prompt[-1:])]
                assert all(
                    layer.offset == len(audit.prompt) - 1
                    for layer in kwargs["caches"][0]
                )
                run.insertions[audit.request_id] = SimpleNamespace(
                    state_machine=kwargs["stop_sequences"][0],
                    processors=tuple(processors),
                    all_tokens=tuple(kwargs["all_tokens"][0]),
                )
                return original_insert(prompts, **kwargs)

            monkeypatch.setattr(generator, "insert", record_insert)
            return generator

        monkeypatch.setattr(
            scheduler, "_build_sampler_and_processors", build_processors
        )
        monkeypatch.setattr(scheduler, "_create_batch_generator", create_generator)
        schedulers.append(scheduler)
        return run

    yield create
    for scheduler in schedulers:
        scheduler.reset()
        scheduler.shutdown()


def step(run):
    result = run.scheduler.step()
    for output in result.outputs:
        assert output.error is None
        run.outputs[output.request_id].extend(output.new_token_ids)
        if output.finished:
            assert output.request_id not in run.finished
            run.finished[output.request_id] = output.finish_reason
    return result


def assert_decode_rows_match_single(run):
    scheduler = run.scheduler
    generator = scheduler.batch_generator
    if generator is None or len(generator._generation_batch.uids) < 2:
        return False
    batch = generator._generation_batch
    membership = tuple(batch.uids)
    if membership in run.checked_memberships:
        return False
    run.checked_memberships.add(membership)
    harness = run.model
    with mx.stream(harness.stream):
        harness.record = False
        try:
            for row_index, uid in enumerate(batch.uids):
                request_id = scheduler.uid_to_request_id[uid]
                run.cache_checked_ids.add(request_id)
                request = scheduler.requests[request_id]
                token_ids = batch.tokens[row_index]
                assert token_ids == request.prompt_token_ids + request.output_token_ids
                extracted = batch.extract_cache(row_index)
                expected = make_prompt_cache(harness.model)
                harness.original_forward(
                    harness.model, mx.array([token_ids], dtype=mx.int32), cache=expected
                )
                mx.eval([layer.state for layer in extracted + expected])
                for actual_layer, expected_layer in zip(extracted, expected):
                    assert (
                        actual_layer.offset == expected_layer.offset == len(token_ids)
                    )
                    for actual_array, expected_array in zip(
                        actual_layer.keys_and_values(), expected_layer.keys_and_values()
                    ):
                        assert mx.allclose(
                            actual_array,
                            expected_array,
                            atol=harness.tolerance,
                            rtol=harness.tolerance,
                        ).item(), f"request={request_id} max_abs={mx.max(mx.abs(actual_array.astype(mx.float32) - expected_array.astype(mx.float32))).item()}"
        finally:
            harness.record = True
    return True


def finish(run, requests, *, check_join=True):
    request_ids = {request.request_id for request in requests}
    for _step_index in range(96):
        if request_ids == run.finished.keys():
            break
        step(run)
        if check_join:
            run.checked_join |= assert_decode_rows_match_single(run)
    else:
        pytest.fail(f"Decode did not finish in {_step_index + 1} scheduler steps")
    scheduler = run.scheduler
    assert not scheduler.waiting
    assert not scheduler.prefilling
    assert not scheduler._prefill_states
    assert not scheduler.running
    assert not scheduler.requests
    assert not scheduler.request_id_to_uid
    assert not scheduler.uid_to_request_id
    assert not scheduler._request_detokenizers
    assert not scheduler._output_parser_sessions
    assert not scheduler._pending_async_removes
    assert not get_prefill_tracker().get_model_progress("")
    if scheduler.batch_generator is not None:
        assert not scheduler.batch_generator._generation_batch.uids
        assert not scheduler.batch_generator._prompt_batch.uids
        assert not scheduler.batch_generator._unprocessed_sequences
    assert request_ids == run.finished.keys()


def run_overlapping_decode(
    harness, scheduler_factory, batch_size, *, long_prompts=False
):
    run = scheduler_factory(
        harness, batch_size, chunk_tokens=255 if long_prompts else 4
    )
    resident = make_request("resident", 5, max_tokens=16, repetition_penalty=1.1)
    run.scheduler.add_request(resident)
    for _step_index in range(12):
        step(run)
        if len(resident.output_token_ids) >= 2:
            break
    else:
        pytest.fail(f"Resident did not start decode in {_step_index + 1} steps")
    lengths = (258, 514, 306) if long_prompts else (7, 14, 11)
    arrivals = [
        make_request(
            "short",
            lengths[0],
            start=9,
            repetition_penalty=1.25,
            repetition_context_size=32,
        ),
        make_request(
            "long", lengths[1], start=19, presence_penalty=0.35, frequency_penalty=0.2
        ),
        make_request("middle", lengths[2], start=29, repetition_penalty=1.7),
    ]
    for request in arrivals:
        run.scheduler.add_request(request)
    requests = [resident, *arrivals]
    finish(run, requests)
    assert run.checked_join
    assert run.cache_checked_ids == {request.request_id for request in requests}
    for request in requests:
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert run.finished[request.request_id] == "length"
        assert (
            len(run.outputs[request.request_id]) == request.sampling_params.max_tokens
        )
        assert run.outputs[request.request_id] == request.output_token_ids
        audit = run.audits[request.request_id]
        for generated, context in enumerate(audit.history):
            assert context == tuple(request.prompt_token_ids) + tuple(
                request.output_token_ids[:generated]
            )
    return run


@pytest.mark.parametrize("model_type", ["llama", "qwen2", "qwen3", "hy_v3"])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_real_decode_matches_single_prefill_with_live_batch_join(
    model_factory, scheduler_factory, model_type, batch_size
):
    harness = model_factory(model_type)
    baseline = run_overlapping_decode(harness, scheduler_factory, 1)
    harness.forwards.clear()
    batched = run_overlapping_decode(harness, scheduler_factory, batch_size)
    assert batched.outputs == baseline.outputs
    assert batched.scheduler._batched_prefill_stats["max_batch_size"] == batch_size
    assert any(
        forward.shape[0] == batch_size and forward.shape[1] > 1
        for forward in harness.forwards
    )


def test_fp16_quantized_weights_keep_decode_parity(model_factory, scheduler_factory):
    harness = model_factory("qwen3", quantized=True)
    assert isinstance(
        harness.model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear
    )
    baseline = run_overlapping_decode(harness, scheduler_factory, 1)
    batched = run_overlapping_decode(harness, scheduler_factory, 3)
    assert batched.outputs == baseline.outputs
    assert batched.scheduler._batched_prefill_stats["max_batch_size"] == 3


def test_allocator_boundary_preserves_decode_cache_positions(
    model_factory, scheduler_factory
):
    harness = model_factory("llama")
    run = run_overlapping_decode(harness, scheduler_factory, 3, long_prompts=True)
    assert run.checked_join
    assert any(
        forward.shape[0] > 1
        and forward.cache_size < 256
        and (forward.cache_size + forward.shape[1] > 256)
        for forward in harness.forwards
    )


@pytest.mark.parametrize("cancelled_indices", [(1,), (0, 1)])
def test_inflight_cancellation_preserves_real_decode_cache_and_tokens(
    model_factory, scheduler_factory, cancelled_indices
):
    harness = model_factory("llama")
    run = scheduler_factory(harness, 3)
    requests = [
        make_request("first", 14, start=3),
        make_request("middle", 22, start=17),
        make_request("last", 18, start=29),
    ]
    for request in requests:
        run.scheduler.add_request(request)
    step(run)
    group = run.scheduler._prefill_runtime.group_for("first")
    assert group is not None
    assert group.request_ids == ("first", "middle", "last")
    assert group.tokens_processed == 4
    for request_index in cancelled_indices:
        assert run.scheduler.abort_request(requests[request_index].request_id)
    step(run)
    survivors = [
        request
        for request_index, request in enumerate(requests)
        if request_index not in cancelled_indices
    ]
    if len(survivors) == 1:
        assert not group.valid
        assert group.cache == ()
        assert run.scheduler._prefill_runtime.group_for("last") is None
    else:
        assert group.request_ids == ("first", "last")
    finish(run, survivors)
    for request_index in cancelled_indices:
        request = requests[request_index]
        assert request.status == RequestStatus.FINISHED_ABORTED
        assert request.request_id not in run.insertions
        assert not run.audits[request.request_id].history
        assert request.request_id not in run.outputs
    assert not group.valid
    assert group.cache == ()
    assert run.cache_checked_ids == (
        {request.request_id for request in survivors} if len(survivors) > 1 else set()
    )
    baseline = scheduler_factory(harness, 1)
    baseline_requests = [
        Request(
            request_id=request.request_id,
            prompt=request.prompt_token_ids,
            prompt_token_ids=request.prompt_token_ids,
            num_prompt_tokens=request.num_prompt_tokens,
            sampling_params=request.sampling_params,
            skip_cache_store=True,
        )
        for request in survivors
    ]
    for request in baseline_requests:
        baseline.scheduler.add_request(request)
    finish(baseline, baseline_requests)
    assert run.outputs == baseline.outputs


@pytest.mark.parametrize("failed_registration", [0, 1])
@pytest.mark.parametrize("partial_registration", [False, True])
def test_handoff_registration_failure_rolls_back_only_unaccepted_decode_rows(
    model_factory,
    scheduler_factory,
    monkeypatch,
    failed_registration,
    partial_registration,
):
    harness = model_factory("llama")
    run = scheduler_factory(harness, 2)
    scheduler = run.scheduler
    requests = [make_request("first", 9, start=3), make_request("second", 9, start=17)]
    for request in requests:
        scheduler.add_request(request)
    step(run)
    group = scheduler._prefill_runtime.group_for("first")
    assert group is not None
    original_register = scheduler_module._register_uid_rows
    registrations = []
    failed_uids = set()

    def fail_registration(model, uids, samplers, processors):
        registrations.append(tuple(uids))
        if len(registrations) - 1 != failed_registration:
            return original_register(model, uids, samplers, processors)
        failed_uids.update(uids)
        if partial_registration:
            original_register(model, uids, samplers, processors)
        raise MemoryError("injected post-insert UID registration failure")

    monkeypatch.setattr(scheduler_module, "_register_uid_rows", fail_registration)
    scheduled = []
    rejected = []
    scheduler._advance_chunked_prefills(scheduled, rejected)
    accepted = requests[:failed_registration]
    retried = requests[failed_registration:]
    assert scheduled == accepted
    assert not rejected
    assert list(scheduler.waiting) == retried
    assert set(scheduler.running) == {request.request_id for request in accepted}
    assert not scheduler._prefill_states
    assert not scheduler.prefilling
    assert not group.valid
    assert group.cache == ()
    generator = scheduler.batch_generator
    assert type(generator) is BatchGenerator
    accepted_uids = set(scheduler.request_id_to_uid.values())
    generator_uids = {
        *generator._generation_batch.uids,
        *generator._prompt_batch.uids,
        *(sequence[0] for sequence in generator._unprocessed_sequences),
    }
    assert generator_uids == accepted_uids
    assert failed_uids.isdisjoint(generator_uids)
    registered_uids = {
        uid
        for model_id, uid in scheduler_module._uid_row_registry
        if model_id == id(harness.model)
    }
    assert registered_uids == accepted_uids
    assert set(scheduler.uid_to_request_id) == accepted_uids
    for request in accepted:
        assert request.status is RequestStatus.RUNNING
        assert request.prefill_oom_retries == 0
    for request in retried:
        assert request.status is RequestStatus.WAITING
        assert request.prefill_oom_retries == 1
        assert request._batched_prefill_disabled
        assert request.prompt_cache is None
        assert request.batch_uid is None
        assert not run.audits.pop(request.request_id).history
    finish(run, requests)
    assert run.checked_join
    assert run.cache_checked_ids == {request.request_id for request in requests}
    assert not any(
        (
            model_id == id(harness.model)
            for model_id, _uid in scheduler_module._uid_row_registry
        )
    )
    for request in requests:
        assert request.status is RequestStatus.FINISHED_LENGTH_CAPPED
        assert (
            len(run.outputs[request.request_id]) == request.sampling_params.max_tokens
        )
        assert run.outputs[request.request_id] == request.output_token_ids


def test_request_processors_and_stop_machines_stay_isolated(
    model_factory, scheduler_factory
):
    harness = model_factory("llama")
    sequences = {
        "sequence_stop": [7, 40, 41, 8],
        "token_stop": [40, 41, 50, 9],
        "continues": [50, 40, 41, 9, 10, 11],
    }
    run = scheduler_factory(harness, 3, forced_tokens=sequences)
    requests = [
        make_request("sequence_stop", 5, stop=["<>"], repetition_penalty=1.2),
        make_request("token_stop", 8, start=11, stop_token_ids=[50]),
        make_request("continues", 11, start=23, max_tokens=5, presence_penalty=0.4),
    ]
    for request in requests:
        run.scheduler.add_request(request)
    finish(run, requests, check_join=False)
    assert run.outputs == {
        "sequence_stop": [7],
        "token_stop": [40, 41],
        "continues": [50, 40, 41, 9, 10],
    }
    assert run.finished == {
        "sequence_stop": "stop",
        "token_stop": "stop",
        "continues": "length",
    }
    assert len({id(insert.state_machine) for insert in run.insertions.values()}) == 3
    assert len({id(audit) for audit in run.audits.values()}) == 3
    assert run.scheduler._batched_prefill_stats["max_batch_size"] == 3
    for request in requests:
        audit = run.audits[request.request_id]
        for generated, context in enumerate(audit.history):
            assert context == tuple(request.prompt_token_ids) + tuple(
                sequences[request.request_id][:generated]
            )


def test_handoff_priming_binding_failure_releases_inserted_uid(
    model_factory, scheduler_factory, monkeypatch
):
    harness = model_factory("llama")
    run = scheduler_factory(harness, 2)
    requests = [make_request("first", 9), make_request("second", 9, start=17)]
    for request in requests:
        run.scheduler.add_request(request)
    step(run)
    bound_uids = set()
    released_uids = set()

    def fail_binding(model, request_id, uid):
        bound_uids.add(uid)
        raise MemoryError("injected partial priming binding failure")

    def release_binding(model, uids):
        released_uids.update(uids)
        bound_uids.difference_update(uids)

    monkeypatch.setattr(scheduler_module._mtp_priming, "bind_uid", fail_binding)
    monkeypatch.setattr(scheduler_module._mtp_priming, "release_uids", release_binding)
    scheduled, rejected = [], []
    run.scheduler._advance_chunked_prefills(scheduled, rejected)

    assert released_uids and not bound_uids
    assert not scheduled and not rejected
    assert list(run.scheduler.waiting) == requests
    assert not run.scheduler.running
    assert not run.scheduler.request_id_to_uid
    assert not run.scheduler.uid_to_request_id
    generator = run.scheduler.batch_generator
    assert not generator._generation_batch.uids
    assert not generator._prompt_batch.uids
    assert not generator._unprocessed_sequences
    assert not any(
        model_id == id(harness.model)
        for model_id, _uid in scheduler_module._uid_row_registry
    )
