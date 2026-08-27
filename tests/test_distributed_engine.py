# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import time
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


def _phase_deployment(*, prefill_rank: int) -> ClusterDeployment:
    base = _deployment()
    assignments = tuple(
        PipelineAssignment(
            host.node_id,
            rank,
            0,
            4,
            4,
            0,
            0,
            8,
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            kv_cache_bytes=1,
            kv_bytes_per_token=1,
            max_context_tokens=8,
        )
        for rank, host in enumerate(base.hosts)
    )
    return replace(
        base,
        assignments=assignments,
        serving_mode="disaggregated",
        prefill_rank=prefill_rank,
        decode_rank=1 - prefill_rank,
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
    engine._supervisor.port = 8001
    return engine


@pytest.mark.asyncio
async def test_distributed_ssd_clear_reaches_every_rank(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={"status": "ok", "rank": 0, "ssd_deleted": 3, "hot_cleared": 0},
        )

    remote_calls = []

    def remote(ssh_target, command, timeout, runner):
        remote_calls.append((ssh_target, command, timeout))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"status": "ok", "rank": 1, "ssd_deleted": 5, "hot_cleared": 0}
            ),
            stderr="",
        )

    monkeypatch.setattr(distributed, "_run_cluster_ssh", remote)
    engine = _ready_engine(handler)
    try:
        result = await engine.clear_prompt_caches(ssd=True)
    finally:
        await engine._client.aclose()

    assert result["ssd_deleted"] == 8
    assert len(result["ranks"]) == 2
    assert requests[0].url.path == "/omlx/internal/cache/ssd/clear"
    assert requests[0].headers["X-oMLX-Plan-Hash"] == "d" * 64
    assert remote_calls[0][0] == "peer.local"
    assert "engine-test-cache-clear.json" in remote_calls[0][1]
    assert '"ssd":true' in remote_calls[0][1]


@pytest.mark.asyncio
async def test_distributed_cache_clear_refuses_active_requests():
    engine = _ready_engine(lambda _request: httpx.Response(200, json={}))
    engine._active_requests = 1
    try:
        with pytest.raises(DistributedInferenceError, match="requests are active"):
            await engine.clear_prompt_caches(ssd=True)
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_reversed_phase_clear_reaches_local_prefill_owner(tmp_path, monkeypatch):
    deployment = _phase_deployment(prefill_rank=0)

    def handler(_request):
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "rank": 1,
                "ssd_deleted": 0,
                "hot_cleared": 0,
            },
        )

    engine = DistributedBatchedEngine(deployment)
    engine._loaded = True
    engine._client = httpx.AsyncClient(
        base_url="http://10.0.0.2:8001",
        transport=httpx.MockTransport(handler),
    )
    engine._supervisor.port = 8001
    engine._supervisor.state_dir = str(tmp_path)
    monkeypatch.setattr(distributed.time, "time_ns", lambda: 7)
    (tmp_path / f"{deployment.deployment_id}-cache-clear-rank-0.json").write_text(
        json.dumps(
            {
                "epoch": 7,
                "status": "ok",
                "rank": 0,
                "ssd_deleted": 3,
                "hot_cleared": 0,
            }
        )
    )
    monkeypatch.setattr(
        distributed,
        "_run_cluster_ssh",
        lambda *_args, **_kwargs: pytest.fail("local prefill must not use SSH"),
    )
    try:
        result = await engine.clear_prompt_caches(ssd=True)
    finally:
        await engine._client.aclose()

    assert result["ssd_deleted"] == 3
    assert {item["rank"] for item in result["ranks"]} == {0, 1}
    request = json.loads(
        (tmp_path / f"{deployment.deployment_id}-cache-clear.json").read_text()
    )
    assert request["ssd"] is True


def test_text_backbone_only_contract_is_explicit_and_default_off():
    assert DistributedBatchedEngine(_deployment()).text_backbone_only is False
    assert (
        DistributedBatchedEngine(
            _deployment(), text_backbone_only=True
        ).text_backbone_only
        is True
    )


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


def test_nonstream_rank_zero_json_has_no_false_long_prefill_deadline():
    timeout = DistributedBatchedEngine._nonstream_timeout()
    assert timeout.connect == 10.0
    assert timeout.read is None
    assert timeout.write == 30.0
    assert timeout.pool == 10.0


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
    # Read timeouts now drop a rank-side cancel file; keep it out of the
    # real runtime state dir.
    engine._supervisor.state_dir = tempfile.mkdtemp(prefix="omlx-test-runtime-")
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


def test_greedy_seed_is_forwarded_with_sampling_rank_only():
    engine = DistributedBatchedEngine(_deployment())
    payload = engine._chat_payload(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs={"seed": 42},
    )
    assert payload["seed"] == 42


def test_stochastic_seed_remains_rejected_with_sampling_rank_only():
    engine = DistributedBatchedEngine(_deployment())
    with pytest.raises(ValueError, match="stochastic seeded generation"):
        engine._completion_payload(
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
            kwargs={"seed": 42},
        )


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


def test_distributed_mtp_is_signed_into_a_pure_tensor_deployment():
    deployment = _deployment()
    tensor_assignments = tuple(
        replace(
            assignment,
            start_layer=0,
            end_layer=4,
            tensor_parallel_rank=assignment.rank,
            tensor_parallel_size=2,
        )
        for assignment in deployment.assignments
    )
    engine = DistributedBatchedEngine(
        replace(
            deployment,
            assignments=tensor_assignments,
            tensor_parallel_size=2,
            mtp_enabled=True,
            mtp_num_draft_tokens=5,
        ),
        model_settings=SimpleNamespace(
            mtp_enabled=False,
            mtp_num_draft_tokens=2,
        ),
    )

    engine._validate_model_settings()
    assert engine.deployment.mtp_enabled is True
    assert engine.deployment.mtp_num_draft_tokens == 5


def test_stale_local_mtp_toggle_cannot_override_signed_disabled_deployment():
    engine = DistributedBatchedEngine(
        _deployment(),
        model_settings=SimpleNamespace(
            mtp_enabled=True,
            mtp_num_draft_tokens=5,
        ),
    )

    assert engine.deployment.mtp_enabled is False
    assert engine.deployment.mtp_num_draft_tokens is None


def test_distributed_mtp_refuses_pipeline_parallelism():
    engine = DistributedBatchedEngine(
        replace(_deployment(), mtp_enabled=True, mtp_num_draft_tokens=3),
        model_settings=SimpleNamespace(
            mtp_enabled=False,
            mtp_num_draft_tokens=None,
        ),
    )

    with pytest.raises(ValueError, match="requires pure tensor parallelism"):
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
async def test_distributed_chat_recovers_raw_ds4_dsml_without_leaking_markup():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    dsml = (
        '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="bash">\n'
        '<｜DSML｜parameter name="command" string="true">true'
        "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    )

    def handler(_request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": dsml,
                            "reasoning_content": "Need the shell tool.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            },
        )

    engine = _ready_engine(handler)
    engine._model_type = "deepseek_v4"
    try:
        output = await engine.chat(
            [{"role": "user", "content": "Run true"}], tools=tools
        )
    finally:
        await engine._client.aclose()

    assert "DSML" not in output.text
    assert output.text == "<think>Need the shell tool.</think>"
    assert output.finish_reason == "tool_calls"
    assert output.tool_calls is not None
    assert output.tool_calls[0]["name"] == "bash"
    assert json.loads(output.tool_calls[0]["arguments"]) == {"command": "true"}


@pytest.mark.asyncio
async def test_distributed_chat_repairs_one_tail_truncated_dsml_call():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    truncated = (
        'Ready.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="bash">\n'
        '<｜DSML｜parameter name="command" string="true">curl -fsS example.test'
    )

    def handler(_request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": truncated},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            },
        )

    engine = _ready_engine(handler)
    engine._model_type = "deepseek_v4"
    try:
        output = await engine.chat(
            [{"role": "user", "content": "Fetch it"}], tools=tools
        )
    finally:
        await engine._client.aclose()

    assert output.text == "Ready."
    assert output.finish_reason == "tool_calls"
    assert output.tool_calls is not None
    assert json.loads(output.tool_calls[0]["arguments"]) == {
        "command": "curl -fsS example.test"
    }


def test_dsml_repair_refuses_multiple_unmatched_invokes():
    malformed = (
        '<｜DSML｜tool_calls><｜DSML｜invoke name="a">'
        '<｜DSML｜invoke name="b">'
    )
    assert DistributedBatchedEngine._repair_truncated_dsml(malformed) is None


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
async def test_distributed_stream_chat_buffers_split_dsml_and_emits_structured_call():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    chunks = [
        "Before the call.\n\n<｜DSML｜tool",
        '_calls>\n<｜DSML｜invoke name="bash">\n',
        '<｜DSML｜parameter name="command" string="true">true',
        "</｜DSML｜parameter>\n</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>",
    ]
    events = [
        {
            "choices": [
                {"delta": {"content": chunk}, "finish_reason": None}
            ]
        }
        for chunk in chunks
    ]
    events.append(
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }
    )
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"

    engine = _ready_engine(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )
    )
    engine._model_type = "deepseek_v4"
    try:
        outputs = [
            output
            async for output in engine.stream_chat(
                [{"role": "user", "content": "Run true"}], tools=tools
            )
        ]
    finally:
        await engine._client.aclose()

    assert all("DSML" not in output.new_text for output in outputs)
    assert outputs[-1].text == "Before the call."
    assert outputs[-1].finish_reason == "tool_calls"
    assert outputs[-1].tool_calls is not None
    assert outputs[-1].tool_calls[0]["name"] == "bash"
    assert json.loads(outputs[-1].tool_calls[0]["arguments"]) == {
        "command": "true"
    }


@pytest.mark.asyncio
async def test_distributed_stream_chat_repairs_truncated_dsml_without_raw_recovery():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            },
        }
    ]
    chunks = [
        "Visible.\n\n<｜DSML｜tool_calls>",
        '\n<｜DSML｜invoke name="bash">',
        '\n<｜DSML｜parameter name="command" string="true">true',
    ]
    events = [
        {"choices": [{"delta": {"content": chunk}, "finish_reason": None}]}
        for chunk in chunks
    ]
    events.append(
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
    )
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"

    engine = _ready_engine(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )
    )
    engine._model_type = "deepseek_v4"
    try:
        outputs = [
            output
            async for output in engine.stream_chat(
                [{"role": "user", "content": "Run true"}], tools=tools
            )
        ]
    finally:
        await engine._client.aclose()

    assert all("DSML" not in output.new_text for output in outputs)
    assert outputs[-1].text == "Visible."
    assert outputs[-1].finish_reason == "tool_calls"
    assert outputs[-1].tool_calls is not None
    assert json.loads(outputs[-1].tool_calls[0]["arguments"]) == {
        "command": "true"
    }


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
    # Monotonic call order in stream_generate: request_started_at(10.0),
    # _enter_request state timestamp (11.0, unused for rates), first-chunk
    # now/first_token_at(12.0), _mark_backend_finished (15.0, unused),
    # finished_at(16.0). Rates: prefill 32/(12-10)=16, decode (2-1)/(16-12)=0.25.
    ticks = iter((10.0, 11.0, 12.0, 15.0, 16.0))
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
    # Same monotonic call order as the automatic-choice test above; only the
    # started/first-token/finished values feed the rates.
    ticks = iter((10.0, 11.0, 12.0, 15.0, 16.0))
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
        assert "Studio stopped publishing" in engine.runtime_failed_reason
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
async def test_distributed_transport_error_marks_ready_runtime_terminal(monkeypatch):
    def handler(request):
        raise httpx.RemoteProtocolError(
            "server disconnected",
            request=request,
        )

    engine = _ready_engine(handler)
    engine._supervisor.status = lambda: SimpleNamespace(
        returncode=None,
        failure_reason=None,
        phase="ready",
        stderr_tail=("request accepted",),
    )

    async def no_wait(_delay):
        return None

    monkeypatch.setattr(distributed.asyncio, "sleep", no_wait)
    try:
        with pytest.raises(
            DistributedInferenceError,
            match="connection closed while the cluster was ready",
        ):
            await engine.generate("hello")
        assert engine.runtime_failed_reason is not None
        assert "connection closed" in engine.runtime_failed_reason
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


def _rank_zero_marker(active_requests: int) -> dict:
    return {
        "schema_version": 1,
        "deployment_id": "engine-test",
        "pid": 424242,
        "rank": 0,
        "world_size": 2,
        "model": "org/model",
        "backend": "ring",
        "plan_hash": "d" * 64,
        "phase": "ready",
        "updated_at": "2026-08-17T00:00:00+00:00",
        "metrics": {"active_requests": active_requests},
    }


@pytest.mark.asyncio
async def test_abort_request_flags_and_closes_only_that_request(tmp_path):
    engine = _ready_engine(lambda request: httpx.Response(200, json={}))
    engine._supervisor.state_dir = str(tmp_path)
    first = await engine._enter_request()
    second = await engine._enter_request()
    closed = []

    class FakeResponse:
        async def aclose(self):
            closed.append(True)

    engine._request_states[first].response = FakeResponse()

    assert await engine.abort_request(first, reason="test") is True
    assert engine._request_states[first].aborted is True
    assert engine._request_states[second].aborted is False
    assert closed == [True]
    marker = json.loads(
        (tmp_path / "engine-test-cancel.json").read_text(encoding="utf-8")
    )
    assert marker["scope"] == "requests"
    assert marker["request_ids"] == [first]
    assert await engine.abort_request("engine-test-999") is False

    await engine._leave_request(first)
    await engine._leave_request(second)
    await engine._client.aclose()


@pytest.mark.asyncio
async def test_private_rank_request_carries_the_targeted_transport_id():
    seen = []

    def handler(request):
        seen.append(request.headers.get("x-omlx-request-id"))
        event = {
            "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
        )

    engine = _ready_engine(handler)
    try:
        outputs = [
            output
            async for output in engine.stream_chat(
                [{"role": "user", "content": "hi"}],
                _request_id="transport-public-1",
            )
        ]
    finally:
        await engine._client.aclose()

    assert outputs[-1].finished is True
    assert seen == ["transport-public-1"]


@pytest.mark.asyncio
async def test_concurrent_targeted_cancels_merge_until_rank_ack(tmp_path):
    engine = _ready_engine(lambda request: httpx.Response(200, json={}))
    engine._supervisor.state_dir = str(tmp_path)
    first = await engine._enter_request("transport-first")
    second = await engine._enter_request("transport-second")
    try:
        assert await engine.abort_request(first, reason="first socket closed") is True
        assert await engine.abort_request(second, reason="second socket closed") is True
        payload = json.loads(
            (tmp_path / "engine-test-cancel.json").read_text(encoding="utf-8")
        )
        assert payload["scope"] == "requests"
        assert set(payload["request_ids"]) == {first, second}
    finally:
        await engine._leave_request(first)
        await engine._leave_request(second)
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_stale_pre_restart_cancel_all_never_widens_new_disconnect(tmp_path):
    engine = _ready_engine(lambda request: httpx.Response(200, json={}))
    engine._supervisor.state_dir = str(tmp_path)
    cancel_path = tmp_path / "engine-test-cancel.json"
    cancel_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": "engine-test",
                "plan_hash": "d" * 64,
                "epoch": 9_999_999_999_999,
                "scope": "all",
                "reason": "old process",
            }
        ),
        encoding="utf-8",
    )
    request_id = await engine._enter_request("transport-new-client")
    try:
        assert await engine.abort_request(request_id, reason="socket closed") is True
        payload = json.loads(cancel_path.read_text(encoding="utf-8"))
        assert payload["scope"] == "requests"
        assert payload["request_ids"] == [request_id]
    finally:
        await engine._leave_request(request_id)
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_malformed_targeted_cancel_never_degrades_to_cancel_all(tmp_path):
    engine = _ready_engine(lambda request: httpx.Response(200, json={}))
    engine._supervisor.state_dir = str(tmp_path)

    assert (
        engine._write_rank_cancel_request(
            reason="bad caller",
            request_id="invalid\nheader",
        )
        is None
    )
    assert not (tmp_path / "engine-test-cancel.json").exists()
    await engine._client.aclose()


@pytest.mark.asyncio
async def test_aborted_stream_raises_at_the_next_yield_boundary(tmp_path):
    events = [
        {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]},
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
    engine._supervisor.state_dir = str(tmp_path)
    try:
        stream = engine.stream_chat([{"role": "user", "content": "hi"}])
        first = await stream.__anext__()
        assert first.new_text == "hello"
        request_id = next(iter(engine._request_states))
        assert await engine.abort_request(request_id, reason="operator") is True
        with pytest.raises(distributed.DistributedRequestAborted):
            await stream.__anext__()
        # The abort unwound the generator's finally: the counter drained.
        assert engine._active_requests == 0
        assert engine.has_active_requests() is False
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_abort_all_writes_cancel_file_and_swaps_client(tmp_path):
    engine = _ready_engine(lambda request: httpx.Response(200, json={}))
    engine._supervisor.state_dir = str(tmp_path)
    # The client swap needs a supervisor endpoint; the test double never
    # launches, so set the port a real launch would have allocated.
    engine._supervisor.port = 18010
    old_client = engine._client

    count = await engine.abort_all_requests(reason="unload requested")

    assert count == 0
    assert engine._client is not old_client
    cancel_path = tmp_path / "engine-test-cancel.json"
    payload = json.loads(cancel_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["deployment_id"] == "engine-test"
    assert payload["scope"] == "all"
    assert payload["reason"] == "unload requested"
    assert payload["epoch"] > 0
    await engine._client.aclose()


@pytest.mark.asyncio
async def test_backend_drain_waits_for_rank_side_evidence(tmp_path):
    engine = _ready_engine(lambda request: httpx.Response(200, json={}))
    engine._supervisor.state_dir = str(tmp_path)
    (tmp_path / "engine-test-rank-0.json").write_text(
        json.dumps(_rank_zero_marker(0)), encoding="utf-8"
    )
    try:
        assert await engine._wait_for_backend_drain(timeout=1.0) is True

        (tmp_path / "engine-test-rank-0.json").write_text(
            json.dumps(_rank_zero_marker(2)), encoding="utf-8"
        )
        started = time.monotonic()
        assert await engine._wait_for_backend_drain(timeout=0.3) is False
        assert time.monotonic() - started >= 0.3
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_abort_all_reports_rank_side_survivors(tmp_path):
    engine = _ready_engine(lambda request: httpx.Response(200, json={}))
    engine._supervisor.state_dir = str(tmp_path)
    engine._supervisor.port = 18010  # endpoint needed for the post-abort client swap
    engine._abort_drain_timeout = 0.2
    (tmp_path / "engine-test-rank-0.json").write_text(
        json.dumps(_rank_zero_marker(1)), encoding="utf-8"
    )
    try:
        # No local requests, but the rank still reports one: the drain wait
        # must engage and expire unconfirmed rather than trust client close.
        count = await engine.abort_all_requests(reason="memory pressure")
        assert count == 0
    finally:
        await engine._client.aclose()


def test_orphan_reaper_drops_finished_but_abandoned_requests():
    engine = DistributedBatchedEngine(_deployment())
    abandoned = distributed._DistributedRequestState("a", 100.0)
    abandoned.finished_at = 100.0
    live = distributed._DistributedRequestState("b", 100.0)
    fresh = distributed._DistributedRequestState("c", 100.0)
    fresh.finished_at = 199.0
    engine._request_states.update({"a": abandoned, "b": live, "c": fresh})
    engine._active_requests = 3

    reaped = engine.reap_orphaned_generators(now=200.0, grace=5.0)

    assert reaped == 1
    assert set(engine._request_states) == {"b", "c"}
    assert engine._active_requests == 2


def test_orphan_reaper_runs_from_has_active_requests():
    engine = DistributedBatchedEngine(_deployment())
    stale = distributed._DistributedRequestState("a", 0.0)
    stale.finished_at = 0.0
    engine._request_states["a"] = stale
    engine._active_requests = 1

    assert engine.has_active_requests() is False
    assert engine._request_states == {}


@pytest.mark.asyncio
async def test_read_timeout_drops_rank_side_cancel_file(tmp_path):
    engine, _ = _stalled_engine()
    engine._supervisor.state_dir = str(tmp_path)
    try:
        with pytest.raises(DistributedInferenceError):
            await engine.generate("hello")
    finally:
        await engine._client.aclose()

    cancel_path = tmp_path / "engine-test-cancel.json"
    payload = json.loads(cancel_path.read_text(encoding="utf-8"))
    assert payload["scope"] == "requests"
    assert len(payload["request_ids"]) == 1
    assert payload["request_ids"][0].startswith("engine-test-")
    assert "read timeout" in payload["reason"]
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


@pytest.mark.asyncio
async def test_rank_prefill_rejection_keeps_typed_memory_error_surface():
    from omlx.exceptions import PrefillMemoryExceededError

    detail = (
        "Cluster prefill rejected by rank 1: Prefill would require ~131.36 GB "
        "peak but effective ceiling is 126.00 GB"
    )
    engine = _ready_engine(
        lambda request: httpx.Response(404, json={"error": detail})
    )
    try:
        with pytest.raises(PrefillMemoryExceededError, match="rank 1"):
            await engine.generate("hello")
    finally:
        await engine._client.aclose()


def _healthy_supervisor_status():
    return SimpleNamespace(returncode=None, failure_reason=None)


def test_runtime_failure_reconciles_supervisor_terminal_state(monkeypatch):
    """Pool status/release must see a rank death even after a 200 response."""

    engine = _ready_engine(lambda request: httpx.Response(200))
    monkeypatch.setattr(
        engine._supervisor,
        "status",
        lambda: SimpleNamespace(
            returncode=0,
            failure_reason=(
                "rank 0 exited with code 75 after JACCL all_reduce made no progress"
            ),
            phase="failed",
        ),
    )

    assert engine.runtime_failed_reason is not None
    assert "rank 0 exited with code 75" in engine.runtime_failed_reason


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
        assert engine.runtime_failed_reason == "rank 1 (peer) stopped heartbeating"
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
        assert engine.runtime_failed_reason == "rank 1 connection closed"
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
