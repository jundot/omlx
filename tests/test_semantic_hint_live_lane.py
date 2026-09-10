# SPDX-License-Identifier: Apache-2.0
"""CPU-small differential tests for the bounded regular OoO-Spec lane."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs as LlamaModelArgs

import omlx.scheduler as scheduler_module
from omlx.engine.batched import BatchedEngine
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig
from omlx.speculative.semantic_hints import (
    SemanticHintCandidate,
    SemanticHintConfig,
    SemanticHintMailbox,
    SemanticHintRequestContext,
    SemanticToolCall,
    TargetTemplateMismatchError,
    render_live_target_hint,
)
from omlx.speculative.semantic_verification import (
    SemanticVerificationError,
    assert_dense_kv_offset,
    clone_dense_kv_cache,
    verify_greedy_suffix,
)


@pytest.fixture(autouse=True)
def _force_mlx_cpu_device(monkeypatch):
    previous = mx.default_device()
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _StopOnFourFactory:
    kind = "semantic-test"
    stop_token_ids = set()
    thinking_end_text = None
    create_session_with_tools = None

    def create_session(self, tokenizer):
        return _StopOnFourSession()


class _StopOnFourSession:
    def process_token(self, token_id):
        from omlx.adapter.output_parser import OutputParserTokenResult

        if token_id == 4:
            return OutputParserTokenResult(is_stop=True, record_token=False)
        return OutputParserTokenResult(
            stream_text=chr(64 + token_id),
            visible_text=chr(64 + token_id),
            record_token=True,
        )

    def finalize(self):
        from omlx.adapter.output_parser import OutputParserFinalizeResult

        return OutputParserFinalizeResult()


class _TinyTarget:
    """One-layer deterministic target whose KV values record input tokens."""

    def __init__(self, transitions: dict[int, int] | None = None) -> None:
        self.transitions = transitions or {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}
        self.config = SimpleNamespace(
            model_type="tiny",
            num_hidden_layers=1,
            hidden_size=1,
            num_attention_heads=1,
            num_key_value_heads=1,
        )
        self.forward_inputs: list[list[int]] = []

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def parameters(self):
        return []

    def __call__(self, inputs, cache):
        rows = [[int(token) for token in row] for row in inputs.tolist()]
        self.forward_inputs.extend(rows)
        stored = inputs.astype(mx.float32)[:, None, :, None]
        cache[0].update_and_fetch(stored, stored)
        logits = mx.full((1, inputs.shape[1], 32), -6.0)
        for position, token in enumerate(rows[0]):
            logits[0, position, self.transitions.get(token, 0)] = 2.0
        return logits


class _TinyDetokenizer:
    def __init__(self, tokenizer: _TinyTokenizer) -> None:
        self._tokenizer = tokenizer
        self.reset()

    def reset(self) -> None:
        self.text = ""
        self.last_segment = ""

    def add_token(self, token: int) -> None:
        self.last_segment = self._tokenizer.decode([token])
        self.text += self.last_segment

    def finalize(self) -> None:
        self.last_segment = ""


class _TinyTokenizer:
    eos_token_id = 31
    pad_token_id = 0
    name_or_path = ""

    def __init__(self, hint_tokens: list[int]) -> None:
        self.hint_tokens = list(hint_tokens)
        self.stop_encodings: dict[str, list[int]] = {}
        self.pieces = {token: chr(64 + token) for token in range(1, 27)}

    @property
    def detokenizer(self) -> _TinyDetokenizer:
        return _TinyDetokenizer(self)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        tools=None,
        **kwargs,
    ):
        assert tokenize is False
        assert tools
        if len(messages) == 1 and add_generation_prompt:
            return "BASE"
        if (
            len(messages) == 2
            and messages[-1].get("role") == "assistant"
            and not add_generation_prompt
        ):
            return "BASE_HINT"
        raise AssertionError("unexpected template mode")

    def encode(self, text, add_special_tokens=True):
        if text == "BASE":
            return [1, 2]
        if text == "BASE_HINT":
            return [1, 2, *self.hint_tokens]
        if text == "\n":
            return [30]
        if text in self.stop_encodings:
            return list(self.stop_encodings[text])
        return [29]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self.pieces.get(int(token), "") for token in token_ids)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
MESSAGES = [{"role": "user", "content": "weather?"}]
HINT = SemanticToolCall(0, "weather", {"city": "Boston"})

_CANDIDATE_RESULTS: dict[int, Any] = {}


def _candidate(
    result: Any,
    *,
    max_prompt_tokens: int = 32,
    max_suffix_tokens: int = 16,
) -> SemanticHintCandidate:
    candidate = SemanticHintCandidate(
        messages_json=json.dumps(MESSAGES),
        final_tools_json=json.dumps(TOOLS),
        template_tools_json=json.dumps(TOOLS),
        template_options_json="{}",
        config=SemanticHintConfig(
            enabled=True,
            endpoint="http://127.0.0.1:9876/hint",
            timeout_s=0.01,
            max_prompt_tokens=max_prompt_tokens,
            max_suffix_tokens=max_suffix_tokens,
        ),
    )
    _CANDIDATE_RESULTS[id(candidate)] = result
    return candidate


def _context_for_candidate(
    candidate: SemanticHintCandidate,
) -> SemanticHintRequestContext:
    future: concurrent.futures.Future[Any] = concurrent.futures.Future()
    result = _CANDIDATE_RESULTS[id(candidate)]
    if isinstance(result, concurrent.futures.Future):
        future = result
    else:
        future.set_result(result)
    return SemanticHintRequestContext(
        mailbox=SemanticHintMailbox(future),
        messages_json=candidate.messages_json,
        template_tools_json=candidate.template_tools_json,
        template_options_json=candidate.template_options_json,
        config=candidate.config,
    )


def _resolved_context(
    result: Any,
    *,
    max_prompt_tokens: int = 32,
    max_suffix_tokens: int = 16,
) -> SemanticHintRequestContext:
    async def resolve():
        return result

    mailbox = SemanticHintMailbox(asyncio.create_task(resolve()))
    return SemanticHintRequestContext(
        mailbox=mailbox,
        messages_json=json.dumps(MESSAGES),
        template_tools_json=json.dumps(TOOLS),
        template_options_json="{}",
        config=SemanticHintConfig(
            enabled=True,
            endpoint="http://127.0.0.1:9876/hint",
            timeout_s=0.01,
            max_prompt_tokens=max_prompt_tokens,
            max_suffix_tokens=max_suffix_tokens,
        ),
    )


def _pending_candidate() -> tuple[SemanticHintCandidate, concurrent.futures.Future]:
    future: concurrent.futures.Future[Any] = concurrent.futures.Future()
    return _candidate(future), future


def _prefilled_cache(model: _TinyTarget) -> list[KVCache]:
    cache = model.make_cache()
    model(mx.array([[1]]), cache=cache)
    mx.eval(cache[0].keys, cache[0].values)
    return cache


@pytest.mark.parametrize(
    ("suffix", "accepted", "queue"),
    [
        ([3, 4], 2, (3, 4, 5)),
        ([3, 9], 1, (3, 4)),
        ([9, 8], 0, (3,)),
    ],
)
def test_target_verification_full_partial_zero_and_correction(suffix, accepted, queue):
    model = _TinyTarget()
    baseline = _prefilled_cache(model)

    verified = verify_greedy_suffix(
        model=model,
        baseline_cache=baseline,
        prompt_token_ids=[1, 2],
        suffix_token_ids=suffix,
    )

    assert verified.accepted_tokens == accepted
    assert verified.token_ids == queue
    assert len(verified.logprobs) == len(queue)
    assert [int(mx.argmax(lp).item()) for lp in verified.logprobs] == list(queue)
    assert_dense_kv_offset(baseline, 1)
    assert_dense_kv_offset(verified.cache, 2 + len(queue))
    assert all(len(row) == 1 for row in model.forward_inputs)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("mode", ["full", "partial", "zero"])
def test_real_dense_llama_matches_public_serial_decode_exactly(dtype, mode):
    """Compare tokens, full logprob vectors, and every valid KV array."""
    mx.random.seed(17)
    model = LlamaModel(
        LlamaModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            rms_norm_eps=1e-5,
            vocab_size=32,
        )
    )
    model.set_dtype(dtype)
    mx.eval(model.parameters())

    prompt = [1, 2]
    prefilled = make_prompt_cache(model)
    model(mx.array([[prompt[0]]]), cache=prefilled)
    mx.eval(*[array for layer in prefilled for array in (layer.keys, layer.values)])

    serial_probe = clone_dense_kv_cache(prefilled, 1)
    expected: list[int] = []
    current = prompt[-1]
    for _ in range(3):
        logits = model(mx.array([[current]]), cache=serial_probe)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        current = int(mx.argmax(logprobs[0, 0]).item())
        expected.append(current)

    different_first = (expected[0] + 1) % 32
    different_second = (expected[1] + 1) % 32
    suffix = {
        "full": expected[:2],
        "partial": [expected[0], different_second],
        "zero": [different_first, different_second],
    }[mode]
    verified = verify_greedy_suffix(
        model=model,
        baseline_cache=prefilled,
        prompt_token_ids=prompt,
        suffix_token_ids=suffix,
    )

    generator = BatchGenerator(
        model,
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        completion_batch_size=1,
        prefill_batch_size=1,
        stream=mx.new_stream(mx.cpu),
    )
    try:
        baseline_cache = clone_dense_kv_cache(prefilled, 1)
        uid = generator.insert(
            [[prompt[-1]]],
            max_tokens=[16],
            caches=[baseline_cache],
            all_tokens=[[prompt[0]]],
        )[0]
        responses = []
        for _ in verified.token_ids:
            responses.extend(generator.next_generated())

        assert [response.token for response in responses] == list(verified.token_ids)
        assert len(responses) == len(verified.logprobs)
        for actual, response in zip(verified.logprobs, responses):
            assert mx.array_equal(actual, response.logprobs).item()

        serial_cache, all_tokens = generator.extract_cache([uid])[uid]
        assert all_tokens == [*prompt, *verified.token_ids]
        assert [layer.offset for layer in serial_cache] == [
            layer.offset for layer in verified.cache
        ]
        for actual_layer, serial_layer in zip(verified.cache, serial_cache):
            offset = actual_layer.offset
            assert mx.array_equal(
                actual_layer.keys[..., :offset, :],
                serial_layer.keys[..., :offset, :],
            ).item()
            assert mx.array_equal(
                actual_layer.values[..., :offset, :],
                serial_layer.values[..., :offset, :],
            ).item()
    finally:
        generator.close()


@pytest.mark.asyncio
async def test_live_render_must_bind_exact_request_prompt_tokens():
    candidate, _ = _pending_candidate()
    context = _context_for_candidate(candidate)
    with pytest.raises(TargetTemplateMismatchError, match="live request tokens"):
        render_live_target_hint(
            context=context,
            hint=HINT,
            tokenizer=_TinyTokenizer([3, 4]),
            live_prompt_token_ids=[1, 9],
        )
    context.cancel()
    assert await context.mailbox.wait() is None


def _make_scheduler(
    monkeypatch,
    *,
    hint_tokens: list[int],
    transitions: dict[int, int] | None = None,
) -> Scheduler:
    monkeypatch.setenv("OMLX_OOO_SPEC_ENABLED", "1")
    monkeypatch.setenv("OMLX_OOO_SPEC_ENDPOINT", "http://127.0.0.1:9876/hint")
    monkeypatch.setattr(
        scheduler_module,
        "start_semantic_hint_request_context",
        _context_for_candidate,
    )
    scheduler = Scheduler(
        model=_TinyTarget(transitions),
        tokenizer=_TinyTokenizer(hint_tokens),
        config=SchedulerConfig(max_num_seqs=1, completion_batch_size=1),
        stream=mx.new_stream(mx.cpu),
    )
    assert scheduler.supports_semantic_hint_verification
    return scheduler


def _add_request(
    scheduler: Scheduler,
    *,
    candidate: SemanticHintCandidate | None,
    request_id: str | None = None,
    max_tokens: int = 8,
    stop: list[str] | None = None,
    stop_token_ids: list[int] | None = None,
) -> Request:
    request = Request(
        request_id=request_id or f"request-{id(scheduler)}",
        prompt=[1, 2],
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
            stop=stop or [],
            stop_token_ids=stop_token_ids or [],
            logprobs=True,
        ),
        tools=TOOLS,
        semantic_hint_candidate=candidate,
    )
    scheduler.add_request(request)
    return request


def _run_steps(scheduler: Scheduler, count: int):
    tokens: list[int] = []
    logprobs: list[list[float]] = []
    outputs = []
    finish_offsets: list[tuple[int, ...]] = []
    original = scheduler._process_batch_responses

    def capture(responses):
        for response in responses:
            tokens.append(int(response.token))
            if response.logprobs is not None:
                logprobs.append(response.logprobs.tolist())
        result = original(responses)
        for response in responses:
            if response.finish_reason is not None and response.prompt_cache is not None:
                finish_offsets.append(
                    tuple(layer.offset for layer in response.prompt_cache)
                )
        return result

    scheduler._process_batch_responses = capture
    try:
        for _ in range(count):
            step = scheduler.step()
            outputs.extend(step.outputs)
            if step.finished_request_ids:
                break
    finally:
        scheduler._process_batch_responses = original
    return tokens, logprobs, outputs, finish_offsets


def _live_cache_receipt(scheduler: Scheduler, request: Request):
    uid = scheduler.request_id_to_uid[request.request_id]
    cache, all_tokens = scheduler.batch_generator.extract_cache([uid])[uid]
    layer = cache[0]
    return layer.offset, all_tokens, layer.values[..., : layer.offset, :].tolist()


def _output_receipts(outputs):
    return [
        (
            output.new_token_ids,
            output.new_text,
            output.output_token_ids,
            output.output_text,
            output.finished,
            output.finish_reason,
            output.completion_tokens,
            output.tool_calls,
        )
        for output in outputs
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("hint_tokens", ([3, 4], [3, 9], [9, 8]))
async def test_live_lane_matches_serial_tokens_logprobs_cache_and_output(
    monkeypatch, hint_tokens
):
    hinted = _make_scheduler(monkeypatch, hint_tokens=list(hint_tokens))
    baseline = _make_scheduler(monkeypatch, hint_tokens=list(hint_tokens))
    await asyncio.sleep(0)
    hinted_request = _add_request(hinted, candidate=_candidate(HINT))
    baseline_request = _add_request(baseline, candidate=None)
    await asyncio.sleep(0)

    hinted_receipt = _run_steps(hinted, 5)
    baseline_receipt = _run_steps(baseline, 5)

    assert hinted_receipt[:2] == baseline_receipt[:2]
    assert _output_receipts(hinted_receipt[2]) == _output_receipts(baseline_receipt[2])
    assert [len(step.new_token_ids) for step in hinted_receipt[2]] == [1] * 5
    assert hinted_request.output_token_ids == baseline_request.output_token_ids
    assert _live_cache_receipt(hinted, hinted_request) == _live_cache_receipt(
        baseline, baseline_request
    )
    assert not hinted._semantic_hint_active


@pytest.mark.asyncio
async def test_pending_late_timeout_and_malformed_are_unchanged_baseline(monkeypatch):
    pending_scheduler = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    baseline = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    pending_candidate, pending_future = _pending_candidate()
    pending_request = _add_request(pending_scheduler, candidate=pending_candidate)
    baseline_request = _add_request(baseline, candidate=None)

    pending_receipt = _run_steps(pending_scheduler, 3)
    baseline_receipt = _run_steps(baseline, 3)
    assert pending_receipt[:2] == baseline_receipt[:2]
    assert _output_receipts(pending_receipt[2]) == _output_receipts(baseline_receipt[2])
    assert pending_request.output_token_ids == baseline_request.output_token_ids
    assert not pending_scheduler._semantic_hint_active

    assert pending_future.cancelled()
    assert pending_request.semantic_hint_context is None

    malformed = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    malformed_request = _add_request(malformed, candidate=_candidate(object()))
    await asyncio.sleep(0)
    malformed_receipt = _run_steps(malformed, 3)
    assert malformed_receipt[:2] == baseline_receipt[:2]
    assert _output_receipts(malformed_receipt[2]) == _output_receipts(
        baseline_receipt[2]
    )
    assert malformed_request.output_token_ids == baseline_request.output_token_ids


@pytest.mark.asyncio
async def test_target_verification_error_preserves_prefill_baseline(monkeypatch):
    hinted = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    baseline = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    hinted_request = _add_request(hinted, candidate=_candidate(HINT))
    baseline_request = _add_request(baseline, candidate=None)
    await asyncio.sleep(0)

    def fail_verification(**kwargs):
        raise SemanticVerificationError("mutant target verification failure")

    monkeypatch.setattr(scheduler_module, "verify_greedy_suffix", fail_verification)
    hinted_receipt = _run_steps(hinted, 4)
    baseline_receipt = _run_steps(baseline, 4)

    assert hinted_receipt[:2] == baseline_receipt[:2]
    assert _output_receipts(hinted_receipt[2]) == _output_receipts(baseline_receipt[2])
    assert hinted_request.output_token_ids == baseline_request.output_token_ids
    assert _live_cache_receipt(hinted, hinted_request) == _live_cache_receipt(
        baseline, baseline_request
    )


@pytest.mark.asyncio
async def test_activation_bookkeeping_error_rolls_back_to_waiting_baseline(monkeypatch):
    scheduler = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    request = Request(
        request_id="activation-mutant",
        prompt=[1, 2],
        prompt_token_ids=[1, 2],
        num_prompt_tokens=2,
        sampling_params=SamplingParams(max_tokens=8, temperature=0),
        tools=TOOLS,
        semantic_hint_context=_resolved_context(HINT),
    )
    await asyncio.sleep(0)
    cache = _prefilled_cache(scheduler.model)
    sampler, processors = scheduler._build_sampler_and_processors(
        request.sampling_params, request
    )

    class _RejectAppend(list):
        def append(self, value):
            raise RuntimeError("mutant schedule append failure")

    activated = scheduler._try_activate_semantic_hint(
        request,
        cache,
        [2],
        sampler,
        scheduler._build_state_machine(request),
        processors,
        _RejectAppend(),
    )

    assert activated is False
    assert request.status.name == "WAITING"
    assert request.batch_uid is None
    assert request.request_id not in scheduler.running
    assert request.request_id not in scheduler.request_id_to_uid
    assert not scheduler._semantic_hint_active
    assert scheduler.total_prompt_tokens == 0
    assert_dense_kv_offset(cache, 1)


@pytest.mark.asyncio
async def test_abort_cancels_mailbox_and_drops_active_queue(monkeypatch):
    waiting = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    pending_candidate, pending_future = _pending_candidate()
    starts: list[SemanticHintCandidate] = []

    def record_start(candidate):
        starts.append(candidate)
        return _context_for_candidate(candidate)

    monkeypatch.setattr(
        scheduler_module, "start_semantic_hint_request_context", record_start
    )
    waiting_request = _add_request(waiting, candidate=pending_candidate)
    waiting.abort_request(waiting_request.request_id)
    waiting.step()
    assert starts == []
    assert pending_future.cancelled() is False
    assert waiting_request.semantic_hint_candidate is None
    assert waiting_request.request_id not in waiting.requests

    active = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    active_request = _add_request(active, candidate=_candidate(HINT))
    await asyncio.sleep(0)
    active.step()
    assert active._semantic_hint_active
    active.abort_request(active_request.request_id)
    active.step()
    assert not active._semantic_hint_active
    assert active_request.request_id not in active.requests


def test_abort_defers_semantic_sidecar_cancellation_while_store_owns_request(
    monkeypatch,
):
    """The async store-cache barrier must run before semantic cancellation."""
    scheduler = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    request = _add_request(
        scheduler,
        candidate=None,
        request_id="semantic-store-pending",
    )
    cancelled: list[bool] = []
    request.semantic_hint_context = SimpleNamespace(
        cancel=lambda: cancelled.append(True)
    )
    store_future: concurrent.futures.Future[None] = concurrent.futures.Future()
    scheduler._inflight_store_futures[request.request_id] = store_future

    assert scheduler._do_abort_request(request.request_id) is False
    assert request.request_id in scheduler.requests
    assert request.semantic_hint_context is not None
    assert cancelled == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_tokens", "stop_token_ids", "expected_tokens", "expected_offset", "reason"),
    [
        (8, [3], [3], 3, "stop"),
        (8, [4], [3, 4], 4, "stop"),
        (8, [5], [3, 4, 5], 5, "stop"),
        (2, [], [3, 4], 4, "length"),
    ],
)
async def test_stop_and_length_mid_queue_trim_unexposed_tail(
    monkeypatch,
    max_tokens,
    stop_token_ids,
    expected_tokens,
    expected_offset,
    reason,
):
    scheduler = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    baseline = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    request = _add_request(
        scheduler,
        candidate=_candidate(HINT),
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )
    baseline_request = _add_request(
        baseline,
        candidate=None,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )
    await asyncio.sleep(0)

    receipt = _run_steps(scheduler, 4)
    baseline_receipt = _run_steps(baseline, 4)
    tokens, _, outputs, finish_offsets = receipt

    assert receipt[:2] == baseline_receipt[:2]
    assert _output_receipts(outputs) == _output_receipts(baseline_receipt[2])
    assert finish_offsets == baseline_receipt[3]
    assert request.output_token_ids == baseline_request.output_token_ids
    assert tokens == expected_tokens
    assert outputs[-1].finish_reason == reason
    assert finish_offsets == [(expected_offset,)]
    if reason == "stop":
        assert request.output_token_ids == expected_tokens[:-1]
        assert outputs[-1].new_token_ids == []
    else:
        assert request.output_token_ids == [3, 4]


@pytest.mark.asyncio
async def test_parser_stop_mid_queue_uses_regular_finish_and_exact_trim(monkeypatch):
    scheduler = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    baseline = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    scheduler._output_parser_factory = _StopOnFourFactory()
    baseline._output_parser_factory = _StopOnFourFactory()
    real_trim = scheduler_module.trim_dense_kv_cache_exact
    trim_calls = []

    def record_trim(cache, n_tokens, *, expected_before):
        trim_calls.append((n_tokens, expected_before))
        return real_trim(cache, n_tokens, expected_before=expected_before)

    monkeypatch.setattr(scheduler_module, "trim_dense_kv_cache_exact", record_trim)
    request = _add_request(scheduler, candidate=_candidate(HINT))
    baseline_request = _add_request(baseline, candidate=None)
    await asyncio.sleep(0)

    receipt = _run_steps(scheduler, 4)
    baseline_receipt = _run_steps(baseline, 4)
    tokens, _, outputs, finish_offsets = receipt

    assert receipt[:2] == baseline_receipt[:2]
    assert _output_receipts(outputs) == _output_receipts(baseline_receipt[2])
    assert finish_offsets == [(4,)]
    assert baseline_receipt[3] == []
    assert request.output_token_ids == baseline_request.output_token_ids
    assert (1, 5) in trim_calls
    assert tokens == [3, 4]
    assert outputs[-1].finished is True
    assert outputs[-1].finish_reason == "stop"
    assert request.output_token_ids == [3]
    assert request.output_text == "C"
    assert finish_offsets == [(4,)]


@pytest.mark.asyncio
async def test_text_fallback_stop_stores_trimmed_semantic_cache(monkeypatch):
    scheduler = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    baseline = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    scheduler.tokenizer.stop_encodings["D"] = [20]
    baseline.tokenizer.stop_encodings["D"] = [20]
    request = _add_request(
        scheduler,
        candidate=_candidate(HINT),
        stop=["D"],
    )
    baseline_request = _add_request(baseline, candidate=None, stop=["D"])

    first = scheduler.step()
    assert request.output_token_ids == [3]

    extracted: dict[str, Any] = {}

    def capture_cache(raw_cache):
        extracted["cache"] = raw_cache
        extracted["offsets"] = tuple(layer.offset for layer in raw_cache)
        return ([{"semantic": True}], {"kind": "dense"})

    scheduler.block_aware_cache = object()
    monkeypatch.setattr(scheduler, "_extract_cache_states", capture_cache)
    monkeypatch.setattr(scheduler, "_cleanup_finished", lambda finished: None)
    second = scheduler.step()
    baseline_receipt = _run_steps(baseline, 2)

    outputs = [*first.outputs, *second.outputs]
    assert _output_receipts(outputs) == _output_receipts(baseline_receipt[2])
    assert request.output_token_ids == baseline_request.output_token_ids == [3, 4]
    assert outputs[-1].finish_reason == "stop"
    assert outputs[-1].output_text == "C"
    assert extracted["offsets"] == (4,)
    assert extracted["cache"] is not None
    assert request._extracted_cache == [{"semantic": True}]


@pytest.mark.asyncio
async def test_finish_trim_failure_cold_replays_only_exposed_tokens(monkeypatch):
    scheduler = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    request = _add_request(
        scheduler,
        candidate=_candidate(HINT),
        stop_token_ids=[3],
    )
    await asyncio.sleep(0)

    real_trim = scheduler_module.trim_dense_kv_cache_exact
    failed = False

    def fail_finish_trim(cache, n_tokens, *, expected_before):
        nonlocal failed
        if not failed and n_tokens == 2 and expected_before == 5:
            failed = True
            raise SemanticVerificationError("mutant finish trim failure")
        return real_trim(cache, n_tokens, expected_before=expected_before)

    monkeypatch.setattr(scheduler_module, "trim_dense_kv_cache_exact", fail_finish_trim)
    tokens, _, outputs, finish_offsets = _run_steps(scheduler, 2)

    assert failed
    assert tokens == [3]
    assert request.output_token_ids == []
    assert outputs[-1].finish_reason == "stop"
    assert finish_offsets == [(3,)]


@pytest.mark.asyncio
async def test_stop_string_crosses_public_handback_with_primed_matcher(monkeypatch):
    scheduler = _make_scheduler(
        monkeypatch,
        hint_tokens=[9],
        transitions={1: 2, 2: 4, 4: 5, 5: 6},
    )
    baseline = _make_scheduler(
        monkeypatch,
        hint_tokens=[9],
        transitions={1: 2, 2: 4, 4: 5, 5: 6},
    )
    scheduler.tokenizer.stop_encodings["DE"] = [4, 5]
    baseline.tokenizer.stop_encodings["DE"] = [4, 5]
    request = _add_request(
        scheduler,
        candidate=_candidate(HINT),
        stop=["DE"],
    )
    baseline_request = _add_request(baseline, candidate=None, stop=["DE"])
    await asyncio.sleep(0)

    receipt = _run_steps(scheduler, 3)
    baseline_receipt = _run_steps(baseline, 3)
    tokens, _, outputs, finish_offsets = receipt

    assert receipt[:2] == baseline_receipt[:2]
    assert _output_receipts(outputs) == _output_receipts(baseline_receipt[2])
    assert finish_offsets == baseline_receipt[3]
    assert request.output_token_ids == baseline_request.output_token_ids
    assert tokens == [4, 5]
    assert outputs[-1].finish_reason == "stop"
    assert outputs[-1].output_text == ""
    assert request.output_token_ids == [4]
    assert finish_offsets == [(4,)]


@pytest.mark.asyncio
async def test_handback_trim_failure_cold_replays_exact_serial_state(
    monkeypatch,
):
    hinted = _make_scheduler(monkeypatch, hint_tokens=[9])
    baseline = _make_scheduler(monkeypatch, hint_tokens=[9])
    hinted_request = _add_request(hinted, candidate=_candidate(HINT))
    baseline_request = _add_request(baseline, candidate=None)
    await asyncio.sleep(0)

    real_trim = scheduler_module.trim_dense_kv_cache_exact
    failed = False

    def fail_once(cache, n_tokens, *, expected_before):
        nonlocal failed
        if not failed and n_tokens == 1 and expected_before == 3:
            failed = True
            raise SemanticVerificationError("mutant trim failure")
        return real_trim(cache, n_tokens, expected_before=expected_before)

    monkeypatch.setattr(scheduler_module, "trim_dense_kv_cache_exact", fail_once)
    hinted_receipt = _run_steps(hinted, 4)
    baseline_receipt = _run_steps(baseline, 4)

    assert failed
    assert hinted_receipt[:2] == baseline_receipt[:2]
    assert _output_receipts(hinted_receipt[2]) == _output_receipts(baseline_receipt[2])
    assert hinted_request.output_token_ids == baseline_request.output_token_ids
    assert _live_cache_receipt(hinted, hinted_request) == _live_cache_receipt(
        baseline, baseline_request
    )


def test_capability_gate_rejects_parallel_batch_quantized_and_processors(
    monkeypatch,
):
    scheduler = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    scheduler.config.max_num_seqs = 2
    assert not scheduler.supports_semantic_hint_verification
    scheduler.config.max_num_seqs = 1
    scheduler._turboquant_kv_bits = 4
    assert not scheduler.supports_semantic_hint_verification
    scheduler._turboquant_kv_bits = None
    scheduler.model.config.quantization = {"bits": 4}
    assert not scheduler.supports_semantic_hint_verification
    scheduler.model.config.quantization = None
    scheduler.model.config.sliding_window = 32
    assert not scheduler.supports_semantic_hint_verification
    scheduler.model.config.sliding_window = None
    assert scheduler.supports_semantic_hint_verification

    request = Request(
        request_id="ineligible",
        prompt=[1, 2],
        sampling_params=SamplingParams(temperature=0, repetition_penalty=1.1),
        semantic_hint_context=SimpleNamespace(),
    )
    assert not scheduler._semantic_hint_request_eligible(
        request,
        _prefilled_cache(scheduler.model),
        [2],
        [lambda tokens, logits: logits],
    )


@pytest.mark.parametrize(
    "case",
    [
        "temperature",
        "repetition",
        "presence",
        "frequency",
        "xtc",
        "grammar",
        "processor",
        "specprefill",
        "prompt_bound",
        "scheduler_capacity",
        "quantized_model",
        "sliding_model",
        "unsupported_cache",
    ],
)
def test_comprehensive_ineligible_gate_makes_zero_sidecar_calls(monkeypatch, case):
    scheduler = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    candidate = _candidate(
        HINT,
        max_prompt_tokens=1 if case == "prompt_bound" else 32,
    )
    params = SamplingParams(max_tokens=8, temperature=0)
    processors: list[Any] = []
    cache: list[Any] | None = None
    request = Request(
        request_id=f"ineligible-{case}",
        prompt=[1, 2],
        prompt_token_ids=[1, 2],
        num_prompt_tokens=2,
        sampling_params=params,
        tools=TOOLS,
        semantic_hint_candidate=candidate,
    )
    if case == "temperature":
        params.temperature = 0.1
    elif case == "repetition":
        params.repetition_penalty = 1.1
    elif case == "presence":
        params.presence_penalty = 0.1
    elif case == "frequency":
        params.frequency_penalty = 0.1
    elif case == "xtc":
        params.xtc_probability = 0.1
    elif case == "grammar":
        params.compiled_grammar = object()
    elif case == "processor":
        processors.append(object())
    elif case == "specprefill":
        request._specprefill_enabled = True
    elif case == "scheduler_capacity":
        scheduler.config.max_num_seqs = 2
    elif case == "quantized_model":
        scheduler.model.config.quantization = {"bits": 4}
    elif case == "sliding_model":
        scheduler.model.config.sliding_window = 32
    elif case == "unsupported_cache":
        cache = [object()]

    starts: list[SemanticHintCandidate] = []
    monkeypatch.setattr(
        scheduler_module,
        "start_semantic_hint_request_context",
        lambda candidate: starts.append(candidate),
    )
    scheduler._maybe_start_semantic_hint(
        request,
        cache,
        [1, 2],
        processors,
    )

    assert starts == []
    assert request.semantic_hint_candidate is None
    assert request.semantic_hint_context is None


def test_sidecar_starts_only_after_tokenization_and_memory_admission(monkeypatch):
    scheduler = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    candidate = _candidate(HINT)
    starts: list[SemanticHintCandidate] = []
    forward_counts: list[int] = []

    def record_start(candidate):
        starts.append(candidate)
        forward_counts.append(len(scheduler.model.forward_inputs))
        return _context_for_candidate(candidate)

    monkeypatch.setattr(
        scheduler_module, "start_semantic_hint_request_context", record_start
    )
    request = Request(
        request_id="eligible-after-admission",
        prompt="BASE",
        sampling_params=SamplingParams(max_tokens=8, temperature=0),
        tools=TOOLS,
        semantic_hint_candidate=candidate,
    )
    scheduler.add_request(request)
    assert request.prompt_token_ids == [1, 2]
    assert starts == []

    scheduler.step()

    assert starts == [candidate]
    assert forward_counts == [0]
    assert request.semantic_hint_candidate is None


def test_memory_rejection_makes_zero_sidecar_calls(monkeypatch):
    scheduler = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    starts: list[SemanticHintCandidate] = []
    monkeypatch.setattr(
        scheduler_module,
        "start_semantic_hint_request_context",
        lambda candidate: starts.append(candidate),
    )
    monkeypatch.setattr(
        scheduler,
        "_preflight_memory_check",
        lambda request: SimpleNamespace(
            message="rejected before sidecar",
            estimated_bytes=2,
            limit_bytes=1,
        ),
    )
    request = _add_request(
        scheduler,
        candidate=_candidate(HINT),
        request_id="memory-rejected",
    )

    result = scheduler.step()

    assert starts == []
    assert result.outputs[-1].finish_reason == "error"
    assert request.semantic_hint_candidate is None


def test_waiting_behind_live_request_makes_zero_sidecar_calls(monkeypatch):
    scheduler = _make_scheduler(monkeypatch, hint_tokens=[3, 4])
    _add_request(
        scheduler,
        candidate=None,
        request_id="occupies-capacity",
        max_tokens=8,
    )
    scheduler.step()
    starts: list[SemanticHintCandidate] = []
    monkeypatch.setattr(
        scheduler_module,
        "start_semantic_hint_request_context",
        lambda candidate: starts.append(candidate),
    )
    waiting = _add_request(
        scheduler,
        candidate=_candidate(HINT),
        request_id="waiting-for-capacity",
    )

    scheduler.step()

    assert starts == []
    assert waiting in scheduler.waiting
    assert waiting.semantic_hint_candidate is not None


def test_batched_engine_gate_excludes_subclasses_partial_and_parallel(monkeypatch):
    engine = object.__new__(BatchedEngine)
    engine._engine = SimpleNamespace(
        engine=SimpleNamespace(
            scheduler=SimpleNamespace(supports_semantic_hint_verification=True)
        )
    )
    engine._enable_thinking = False
    assert engine.supports_semantic_hint_verification

    class _Subclass(BatchedEngine):
        pass

    subclass = object.__new__(_Subclass)
    subclass._engine = engine._engine
    assert not subclass.supports_semantic_hint_verification

    captured = []

    def prepare(*args):
        captured.append(args)
        return "candidate"

    monkeypatch.setattr("omlx.engine.batched.prepare_semantic_hint_candidate", prepare)
    common = {
        "messages": MESSAGES,
        "tools": TOOLS,
        "template_tools": TOOLS,
        "chat_template_kwargs": {"mode": "exact"},
        "tool_choice": "auto",
    }
    assert (
        engine._prepare_semantic_hint_candidate(
            **common, is_partial=True, parallel_tool_calls=False
        )
        is None
    )
    assert (
        engine._prepare_semantic_hint_candidate(
            **common, is_partial=False, parallel_tool_calls=None
        )
        is None
    )
    assert (
        engine._prepare_semantic_hint_candidate(
            **common, is_partial=False, parallel_tool_calls=True
        )
        is None
    )
    assert (
        engine._prepare_semantic_hint_candidate(
            **common, is_partial=False, parallel_tool_calls=False
        )
        == "candidate"
    )
    assert captured == [
        (
            MESSAGES,
            TOOLS,
            TOOLS,
            {"enable_thinking": False, "mode": "exact"},
        )
    ]
