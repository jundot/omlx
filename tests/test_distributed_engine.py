# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.performance import execution_profile
from omlx.cluster.planner import PipelineAssignment
from omlx.cluster.strategy_benchmarks import configure_strategy_benchmark_store
from omlx.engine import distributed
from omlx.engine.distributed import (
    DistributedBatchedEngine,
    DistributedInferenceError,
)


def _deployment() -> ClusterDeployment:
    return ClusterDeployment(
        deployment_id="engine-test",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("peer", "peer.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("local", 0, 2, 4, 2, 0, 0, 4),
            PipelineAssignment("peer", 1, 0, 2, 2, 0, 0, 4),
        ),
        plan_hash="d" * 64,
    )


class _Tokenizer:
    @staticmethod
    def encode(text):
        return list(text.encode())


def _ready_engine(handler) -> DistributedBatchedEngine:
    engine = DistributedBatchedEngine(_deployment())
    engine._loaded = True
    engine._tokenizer = _Tokenizer()
    engine._client = httpx.AsyncClient(
        base_url="http://127.0.0.1:1",
        transport=httpx.MockTransport(handler),
    )
    return engine


def test_backend_chat_messages_serialize_native_tool_history_once():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather",
            "content": '{"temperature_c":18}',
        },
    ]

    prepared = DistributedBatchedEngine._backend_chat_messages(messages)

    assert prepared[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"city": "Paris"}'
    )
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert prepared[1] == messages[1]


@pytest.mark.asyncio
async def test_private_rank_zero_client_has_finite_inactivity_timeouts():
    engine = DistributedBatchedEngine(_deployment(), request_read_timeout=12.5)
    client = engine._new_client("http://127.0.0.1:1")
    try:
        assert client.timeout.connect == 10.0
        assert client.timeout.read == 12.5
        assert client.timeout.write == 30.0
        assert client.timeout.pool == 10.0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_request_read_timeout_defaults_from_env_var(monkeypatch):
    monkeypatch.setenv("OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT", "600")
    engine = DistributedBatchedEngine(_deployment())
    client = engine._new_client("http://127.0.0.1:1")
    try:
        assert client.timeout.read == 600.0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_request_read_timeout_env_var_takes_backseat_to_explicit_arg(monkeypatch):
    monkeypatch.setenv("OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT", "600")
    engine = DistributedBatchedEngine(_deployment(), request_read_timeout=12.5)
    client = engine._new_client("http://127.0.0.1:1")
    try:
        assert client.timeout.read == 12.5
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_request_read_timeout_env_var_rejects_non_numeric(monkeypatch):
    monkeypatch.setenv("OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT", "not-a-number")
    with pytest.raises(ValueError, match="must be a number"):
        DistributedBatchedEngine(_deployment())


@pytest.mark.asyncio
async def test_request_read_timeout_rejects_non_finite_and_non_positive(monkeypatch):
    for bad in ("nan", "inf", "0", "-5"):
        monkeypatch.setenv("OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT", bad)
        with pytest.raises(ValueError, match="finite positive"):
            DistributedBatchedEngine(_deployment())

    monkeypatch.delenv("OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT")
    with pytest.raises(ValueError, match="finite positive"):
        DistributedBatchedEngine(_deployment(), request_read_timeout=float("nan"))
    with pytest.raises(ValueError, match="finite positive"):
        DistributedBatchedEngine(_deployment(), request_read_timeout=0.0)


def _stalled_engine():
    def handler(request):
        raise httpx.ReadTimeout("collective stalled", request=request)

    engine = _ready_engine(handler)
    status_calls = []

    def status():
        status_calls.append(True)
        return SimpleNamespace(
            returncode=None,
            failure_reason=None,
            phase="ready",
        )

    engine._supervisor.status = status
    return engine, status_calls


@pytest.mark.asyncio
async def test_distributed_generate_bounds_rank_zero_read_stalls():
    engine, status_calls = _stalled_engine()
    try:
        with pytest.raises(
            DistributedInferenceError,
            match="request timed out.*no rank-zero data.*cluster was ready",
        ):
            await engine.generate("hello")
    finally:
        await engine._client.aclose()

    assert len(status_calls) == 2, "availability must be rechecked after timeout"


@pytest.mark.asyncio
async def test_distributed_stream_bounds_rank_zero_read_stalls():
    engine, status_calls = _stalled_engine()
    try:
        with pytest.raises(
            DistributedInferenceError,
            match="stream timed out.*no rank-zero data.*cluster was ready",
        ):
            [output async for output in engine.stream_generate("hello")]
    finally:
        await engine._client.aclose()

    assert len(status_calls) == 2, "availability must be rechecked after timeout"


def test_chat_payload_folds_thinking_budget_into_chat_template_kwargs():
    engine = DistributedBatchedEngine(_deployment())
    payload = engine._chat_payload(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs={
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "thinking_budget": 2048,
        },
    )
    assert payload["chat_template_kwargs"] == {
        "reasoning_effort": "low",
        "thinking_budget": 2048,
    }


def test_chat_payload_without_thinking_budget_leaves_template_kwargs_untouched():
    engine = DistributedBatchedEngine(_deployment())
    payload = engine._chat_payload(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs={"chat_template_kwargs": {"reasoning_effort": "low"}},
    )
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}


def test_completion_payload_folds_thinking_budget_into_chat_template_kwargs():
    engine = DistributedBatchedEngine(_deployment())
    payload = engine._completion_payload(
        prompt="hi",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs={"thinking_budget": 512},
    )
    assert payload["chat_template_kwargs"] == {"thinking_budget": 512}


def test_payloads_forward_repetition_context_size_when_requested():
    engine = DistributedBatchedEngine(_deployment())
    kwargs = {"repetition_context_size": 128}
    chat = engine._chat_payload(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.1,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs=dict(kwargs),
    )
    completion = engine._completion_payload(
        prompt="hi",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.1,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs=dict(kwargs),
    )
    assert chat["repetition_context_size"] == 128
    assert completion["repetition_context_size"] == 128


def test_payloads_omit_repetition_context_size_by_default():
    # The key must stay off the wire unless the client asked for it: ranks
    # running mlx-lm default the window to 20 tokens when it is absent.
    engine = DistributedBatchedEngine(_deployment())
    chat = engine._chat_payload(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.1,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs={},
    )
    completion = engine._completion_payload(
        prompt="hi",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.1,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs={},
    )
    assert "repetition_context_size" not in chat
    assert "repetition_context_size" not in completion


def test_model_thinking_budget_is_supported_by_distributed_engine():
    engine = DistributedBatchedEngine(
        _deployment(),
        model_settings=SimpleNamespace(thinking_budget_enabled=True),
    )

    engine._validate_model_settings()


@pytest.mark.asyncio
async def test_distributed_generate_translates_backend_completion():
    def handler(request):
        body = json.loads(request.content)
        assert body["prompt"] == "Hello"
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [{"text": " world", "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 1},
                },
            },
        )

    engine = _ready_engine(handler)
    try:
        output = await engine.generate("Hello", max_tokens=8)
    finally:
        await engine._client.aclose()

    assert output.text == " world"
    assert output.prompt_tokens == 1
    assert output.completion_tokens == 2
    assert output.cached_tokens == 1
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_distributed_chat_preserves_rank_zero_tool_calls_and_reasoning():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]

    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["messages"] == [{"role": "user", "content": "Weather?"}]
        assert body["tools"] == tools
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'll check.",
                            "reasoning": "A weather lookup is required.",
                            "tool_calls": [
                                {
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "prompt_tokens_details": {"cached_tokens": 3},
                },
            },
        )

    engine = _ready_engine(handler)
    try:
        output = await engine.chat(
            [{"role": "user", "content": "Weather?"}],
            tools=tools,
        )
    finally:
        await engine._client.aclose()

    assert output.text == ("<think>A weather lookup is required.</think>I'll check.")
    assert output.finish_reason == "tool_calls"
    assert output.tool_calls == [
        {
            "id": "call_weather",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
        }
    ]
    assert output.cached_tokens == 3


@pytest.mark.asyncio
async def test_distributed_stream_chat_preserves_structured_tool_calls():
    events = [
        {
            "choices": [
                {
                    "delta": {"role": "assistant", "reasoning": "Need lookup."},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Paris"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"

    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )

    engine = _ready_engine(handler)
    try:
        outputs = [
            output
            async for output in engine.stream_chat(
                [{"role": "user", "content": "Weather?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
        ]
    finally:
        await engine._client.aclose()

    assert outputs[0].new_text == "<think>Need lookup."
    assert outputs[-1].new_text == ""
    assert outputs[-1].text == "<think>Need lookup.</think>"
    assert outputs[-1].finish_reason == "tool_calls"
    assert outputs[-1].tool_calls == [
        {
            "id": "call_weather",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        }
    ]
    assert outputs[-1].prompt_tokens == 12
    assert outputs[-1].completion_tokens == 5
    assert outputs[-1].cached_tokens == 2


@pytest.mark.asyncio
async def test_distributed_stream_waits_for_usage_before_final_output():
    events = [
        {
            "choices": [
                {"text": "A", "finish_reason": None},
            ]
        },
        {
            "choices": [
                {"text": "B", "finish_reason": "length"},
            ]
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )

    engine = _ready_engine(handler)
    try:
        outputs = [output async for output in engine.stream_generate("test")]
    finally:
        await engine._client.aclose()

    assert [output.new_text for output in outputs] == ["A", "B"]
    assert outputs[0].finished is False
    assert outputs[0].completion_tokens == 1
    assert outputs[0].generated_at is not None
    assert outputs[0].generated_until == outputs[0].generated_at
    assert outputs[-1].finished is True
    assert outputs[-1].text == "AB"
    assert outputs[-1].finish_reason == "length"
    assert outputs[-1].prompt_tokens == 4
    assert outputs[-1].completion_tokens == 2
    assert outputs[-1].cached_tokens == 3
    assert outputs[-1].generated_at == outputs[0].generated_at


@pytest.mark.asyncio
async def test_stream_records_real_prefill_and_decode_for_automatic_choice(
    monkeypatch,
    tmp_path,
):
    from omlx.engine import distributed

    events = [
        {"choices": [{"text": "A", "finish_reason": None}]},
        {"choices": [{"text": "B", "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 32,
                "completion_tokens": 2,
                "total_tokens": 34,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"
    engine = _ready_engine(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )
    )
    store = configure_strategy_benchmark_store(tmp_path)
    ticks = iter((10.0, 12.0, 16.0))
    monkeypatch.setattr(
        distributed,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    try:
        [output async for output in engine.stream_generate("x" * 32)]
    finally:
        await engine._client.aclose()

    measurements = store.measurements(
        model="org/model",
        node_ids=("local", "peer"),
        backend="ring",
        target_context_tokens=1024,
    )
    assert measurements[1].prompt_tokens_per_second == 16.0
    assert measurements[1].decode_tokens_per_second == 0.25
    assert measurements[1].time_to_first_token_seconds == 2.0


@pytest.mark.asyncio
async def test_strategy_benchmark_buckets_total_context_but_rates_uncached_prefill(
    tmp_path, monkeypatch
):
    events = [
        {"choices": [{"text": "A", "finish_reason": None}]},
        {"choices": [{"text": "B", "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 8192,
                "completion_tokens": 2,
                "total_tokens": 8194,
                "prompt_tokens_details": {"cached_tokens": 7168},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"
    engine = _ready_engine(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )
    )
    store = configure_strategy_benchmark_store(tmp_path)
    ticks = iter((10.0, 12.0, 16.0))
    monkeypatch.setattr(
        distributed,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    try:
        [output async for output in engine.stream_generate("x" * 8192)]
    finally:
        await engine._client.aclose()

    measurements = store.measurements(
        model="org/model",
        node_ids=("local", "peer"),
        backend="ring",
        target_context_tokens=8192,
    )
    assert measurements[1].context_tokens == 8192
    assert measurements[1].prompt_tokens_per_second == 512.0


@pytest.mark.asyncio
async def test_distributed_stream_rejects_malformed_usage():
    event = {
        "choices": [],
        "usage": {"prompt_tokens_details": "not-an-object"},
    }

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\n",
        )

    engine = _ready_engine(handler)
    try:
        with pytest.raises(
            DistributedInferenceError,
            match="invalid token details",
        ):
            [output async for output in engine.stream_generate("test")]
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_distributed_engine_surfaces_bounded_backend_error():
    def handler(request):
        return httpx.Response(503, json={"error": "rank 1 failed"})

    engine = _ready_engine(handler)
    try:
        with pytest.raises(DistributedInferenceError, match="HTTP 503.*rank 1"):
            await engine.generate("hello")
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_distributed_transport_error_surfaces_peer_failure_reason():
    def handler(request):
        raise httpx.RemoteProtocolError(
            "server disconnected",
            request=request,
        )

    engine = _ready_engine(handler)
    engine._supervisor.status = lambda: SimpleNamespace(
        returncode=1,
        failure_reason=(
            "Studio stopped publishing its runtime heartbeat. "
            "Check oMLX is running on that Mac."
        ),
        phase="failed",
        stderr_tail=(),
    )
    try:
        with pytest.raises(
            DistributedInferenceError,
            match="Studio stopped publishing its runtime heartbeat",
        ):
            await engine.generate("hello")
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_distributed_transport_error_reports_bounded_launcher_exit():
    def handler(request):
        raise httpx.RemoteProtocolError(
            "server disconnected",
            request=request,
        )

    engine = _ready_engine(handler)
    engine._supervisor.status = lambda: SimpleNamespace(
        returncode=1,
        failure_reason=None,
        phase="failed",
        stderr_tail=("rank 1 out of memory",),
    )
    try:
        with pytest.raises(
            DistributedInferenceError,
            match="exited with code 1.*rank 1 out of memory",
        ):
            await engine.generate("hello")
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_distributed_engine_rejects_unimplemented_grammar():
    def handler(request):
        raise AssertionError("backend should not be called")

    engine = _ready_engine(handler)
    try:
        with pytest.raises(ValueError, match="guided grammar"):
            await engine.generate("hello", compiled_grammar=object())
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_experimental_token_only_output_rejects_seeded_single_request():
    deployment = replace(
        _deployment(),
        execution=replace(
            execution_profile("balanced"),
            sampling_rank_only=True,
        ),
    )
    engine = DistributedBatchedEngine(deployment)
    engine._loaded = True
    engine._tokenizer = _Tokenizer()
    engine._client = httpx.AsyncClient(
        base_url="http://127.0.0.1:1",
        transport=httpx.MockTransport(
            lambda request: pytest.fail("backend should not be called")
        ),
    )
    try:
        with pytest.raises(ValueError, match="sampling-rank-only"):
            await engine.generate("hello", seed=7)
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_distributed_preflight_rejects_features_before_stream_starts():
    engine = _ready_engine(lambda request: httpx.Response(500))
    try:
        # thinking_budget is now supported: it is forwarded to the rank inside
        # chat_template_kwargs instead of being rejected.
        with pytest.raises(ValueError, match="SpecPrefill"):
            await engine.preflight_chat(
                [{"role": "user", "content": "hello"}],
                specprefill=True,
            )
    finally:
        await engine._client.aclose()


# ---------------------------------------------------------------------------
# reasoning_effort fallback: the distributed engine cannot render the chat
# template itself (only rank-zero can), so an unsupported value must be
# retried against rank-zero's HTTP endpoint rather than caught locally the
# way the batched/vlm/dflash engines do.
# ---------------------------------------------------------------------------


def test_reasoning_effort_retry_payloads_maps_alias_first():
    from omlx.engine.distributed import _reasoning_effort_retry_payloads

    payload = {"chat_template_kwargs": {"reasoning_effort": "high"}}
    variants = _reasoning_effort_retry_payloads(
        payload, "Unexpected reasoning effort high. Supported types are xhigh."
    )
    assert len(variants) == 2
    assert variants[0]["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    # Second tier drops the field entirely (template's own default).
    assert "reasoning_effort" not in variants[1].get("chat_template_kwargs", {})


def test_reasoning_effort_retry_payloads_drops_when_no_alias_helps():
    from omlx.engine.distributed import _reasoning_effort_retry_payloads

    # "xhigh" has no further fallback in _ALIAS_FALLBACKS beyond "max", but if
    # the alias candidate equals the normalized value there is nothing to
    # retry with as an alias -- only the drop tier applies. Use a value with a
    # real alias to prove the two-tier ordering, and a bogus value to prove
    # single-tier (drop only) when there's no useful candidate.
    payload = {"chat_template_kwargs": {"reasoning_effort": "not-a-real-level"}}
    variants = _reasoning_effort_retry_payloads(
        payload, "Unexpected reasoning effort not-a-real-level."
    )
    assert len(variants) == 1
    assert "reasoning_effort" not in variants[0].get("chat_template_kwargs", {})


def test_reasoning_effort_retry_payloads_ignores_unrelated_failures():
    from omlx.engine.distributed import _reasoning_effort_retry_payloads

    payload = {"chat_template_kwargs": {"reasoning_effort": "high"}}
    assert _reasoning_effort_retry_payloads(payload, "model not found") == []


def test_reasoning_effort_retry_payloads_ignores_when_not_requested():
    from omlx.engine.distributed import _reasoning_effort_retry_payloads

    payload = {"chat_template_kwargs": {}}
    assert (
        _reasoning_effort_retry_payloads(
            payload, "Unexpected reasoning effort high."
        )
        == []
    )


@pytest.mark.asyncio
async def test_distributed_chat_retries_unsupported_reasoning_effort():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        effort = body.get("chat_template_kwargs", {}).get("reasoning_effort")
        calls.append(effort)
        if effort == "high":
            return httpx.Response(
                404,
                json={
                    "error": "Unexpected reasoning effort high. Supported "
                    "types are xhigh (default), medium, and low."
                },
            )
        assert effort == "xhigh"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    engine = _ready_engine(handler)
    try:
        output = await engine.chat(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "high"},
        )
    finally:
        await engine._client.aclose()

    assert calls == ["high", "xhigh"]
    assert output.text == "ok"


@pytest.mark.asyncio
async def test_distributed_chat_tries_the_normalized_value_first():
    # Local engines normalize before the first render, so "High" succeeds
    # locally; the cluster path must land on the same value, not jump
    # straight to the alias tier.
    calls = []

    def handler(request):
        body = json.loads(request.content)
        effort = body.get("chat_template_kwargs", {}).get("reasoning_effort")
        calls.append(effort)
        if effort == "high":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        return httpx.Response(
            404,
            json={"error": "Unexpected reasoning effort High."},
        )

    engine = _ready_engine(handler)
    try:
        output = await engine.chat(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "High"},
        )
    finally:
        await engine._client.aclose()

    assert calls == ["High", "high"]
    assert output.text == "ok"


@pytest.mark.asyncio
async def test_distributed_generate_retries_unsupported_reasoning_effort():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        effort = body.get("chat_template_kwargs", {}).get("reasoning_effort")
        calls.append(effort)
        if effort == "minimal":
            return httpx.Response(
                404,
                json={"error": "Unexpected reasoning effort minimal."},
            )
        assert effort == "low"
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "ok", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    engine = _ready_engine(handler)
    try:
        output = await engine.generate(
            "hi", chat_template_kwargs={"reasoning_effort": "minimal"}
        )
    finally:
        await engine._client.aclose()

    assert calls == ["minimal", "low"]
    assert output.text == "ok"


@pytest.mark.asyncio
async def test_distributed_stream_chat_retries_unsupported_reasoning_effort():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        effort = body.get("chat_template_kwargs", {}).get("reasoning_effort")
        calls.append(effort)
        if effort == "high":
            return httpx.Response(
                404,
                json={"error": "Unexpected reasoning effort high."},
            )
        assert effort == "xhigh"
        lines = [
            'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], '
            '"usage": {"prompt_tokens": 1, "completion_tokens": 1}}',
            "data: [DONE]",
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="\n".join(lines) + "\n",
        )

    engine = _ready_engine(handler)
    try:
        outputs = [
            output
            async for output in engine.stream_chat(
                [{"role": "user", "content": "hi"}],
                chat_template_kwargs={"reasoning_effort": "high"},
            )
        ]
    finally:
        await engine._client.aclose()

    assert calls == ["high", "xhigh"]
    assert "".join(o.new_text for o in outputs) == "ok"


@pytest.mark.asyncio
async def test_distributed_stream_generate_bounds_retries_and_gives_up():
    # Every attempt is rejected. "High" walks the full ladder — original,
    # normalized ("high"), alias ("xhigh"), dropped — exactly 4 requests,
    # then raise; never an unbounded loop.
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(
            404,
            json={"error": "Unexpected reasoning effort High."},
        )

    engine = _ready_engine(handler)
    try:
        with pytest.raises(DistributedInferenceError, match="HTTP 404"):
            async for _ in engine.stream_generate(
                "hi", chat_template_kwargs={"reasoning_effort": "High"}
            ):
                pass
    finally:
        await engine._client.aclose()

    assert len(calls) == 4


@pytest.mark.asyncio
async def test_distributed_chat_does_not_retry_unrelated_404():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404, json={"error": "model not found"})

    engine = _ready_engine(handler)
    try:
        with pytest.raises(DistributedInferenceError, match="model not found"):
            await engine.chat([{"role": "user", "content": "hi"}])
    finally:
        await engine._client.aclose()

    assert len(calls) == 1


def _healthy_supervisor_status():
    return SimpleNamespace(returncode=None, failure_reason=None)


@pytest.mark.asyncio
async def test_preflight_rejects_an_unhealthy_rank_before_streaming(monkeypatch):
    # The 200 commits before a streaming body runs, so preflight is the last
    # point a half-dead cluster can still become a clean HTTP error (#2708).
    engine = _ready_engine(lambda request: httpx.Response(200))
    monkeypatch.setattr(engine._supervisor, "status", _healthy_supervisor_status)
    monkeypatch.setattr(
        distributed,
        "check_peers",
        lambda hosts, **kwargs: (
            SimpleNamespace(healthy=True),
            SimpleNamespace(healthy=False),
        ),
    )
    monkeypatch.setattr(
        distributed,
        "describe_failure",
        lambda health: "rank 1 (peer) stopped heartbeating",
    )
    try:
        with pytest.raises(DistributedInferenceError, match="not serving"):
            await engine.preflight_chat([{"role": "user", "content": "hi"}])
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_preflight_caches_the_peer_health_read(monkeypatch):
    engine = _ready_engine(lambda request: httpx.Response(200))
    monkeypatch.setattr(engine._supervisor, "status", _healthy_supervisor_status)
    calls = []

    def fake_check_peers(hosts, **kwargs):
        calls.append(hosts)
        return (SimpleNamespace(healthy=True),)

    monkeypatch.setattr(distributed, "check_peers", fake_check_peers)
    try:
        await engine.preflight_chat([{"role": "user", "content": "hi"}])
        await engine.preflight_completion("hi")
        assert len(calls) == 1  # second preflight served from the TTL cache
        assert calls[0] == {0: ("local", "127.0.0.1"), 1: ("peer", "peer.local")}
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_preflight_rejects_a_reported_failure_without_probing(monkeypatch):
    engine = _ready_engine(lambda request: httpx.Response(200))
    monkeypatch.setattr(
        engine._supervisor,
        "status",
        lambda: SimpleNamespace(
            returncode=None, failure_reason="rank 1 connection closed"
        ),
    )
    probed = []
    monkeypatch.setattr(
        distributed, "check_peers", lambda *a, **k: probed.append(1) or ()
    )
    try:
        with pytest.raises(DistributedInferenceError, match="rank 1 connection"):
            await engine.preflight_chat([{"role": "user", "content": "hi"}])
        assert probed == []
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_preflight_fails_open_when_the_probe_itself_breaks(monkeypatch):
    # A broken probe must not take down a serving cluster; the supervisor
    # checks still catch hard failures.
    engine = _ready_engine(lambda request: httpx.Response(200))
    monkeypatch.setattr(engine._supervisor, "status", _healthy_supervisor_status)

    def broken_check_peers(hosts, **kwargs):
        raise OSError("ssh binary missing")

    monkeypatch.setattr(distributed, "check_peers", broken_check_peers)
    try:
        await engine.preflight_chat([{"role": "user", "content": "hi"}])
    finally:
        await engine._client.aclose()
