"""HTTP-only benchmark tests: no running server or model is required."""

import asyncio
import json

import httpx
import pytest

from benchmarks.batched_prefill_bench import (
    BenchmarkConfig,
    RequestResult,
    RequestSpec,
    consume_payload,
    endpoint_url,
    parse_args,
    percentile,
    request_specs,
    run_benchmark,
    run_request,
    summarize,
    synthetic_prompt,
)


class ControlledStream(httpx.AsyncByteStream):
    def __init__(self, chunks, block=None):
        self.chunks = chunks
        self.block = block
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.block is not None:
            await self.block.wait()

    async def aclose(self):
        self.closed = True


def sse(event):
    return ("data: " + json.dumps(event) + "\n\n").encode()


def output_event(text="two tokens"):
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}]}


def complete_stream():
    return ControlledStream(
        [
            sse(output_event()),
            sse(
                {"choices": [], "usage": {"prompt_tokens": 91, "completion_tokens": 2}}
            ),
            b"data: [DONE]\n\n",
        ]
    )


def result_for(spec):
    return RequestResult(spec.request_id, spec.role, len(spec.prompt), spec.max_tokens)


def test_distinct_leading_prompts_are_reproducible():
    first = synthetic_prompt(5, 0, 10)
    second = synthetic_prompt(5, 1, 10)
    assert first == synthetic_prompt(5, 0, 10)
    assert first[:16] != second[:16]
    assert first != synthetic_prompt(6, 0, 10)


def test_staggered_plan_contains_decode_long_and_short_requests():
    specs = request_specs(BenchmarkConfig("model", "test", scenario="staggered"))
    assert [spec.role for spec in specs] == [
        "decode_anchor",
        "prefill_long",
        "prefill_short",
        "prefill_long",
    ]
    assert specs[0].max_tokens == 512
    assert len(specs[1].prompt) > len(specs[2].prompt)


def test_endpoint_redacts_credentials_query_and_fragment():
    base = "https://user:secret@example.test:8443/v1/?api_key=hidden#private"
    assert (
        endpoint_url(base, redact=True)
        == "https://example.test:8443/v1/chat/completions"
    )
    assert "api_key=hidden" in endpoint_url(base)


@pytest.mark.parametrize(
    "options",
    [
        {"concurrency": 0},
        {"prompt_words": 65537},
        {"duration": float("nan")},
        {"request_timeout": 0},
        {"scenario": "staggered", "concurrency": 1},
    ],
)
def test_config_rejects_unbounded_or_invalid_workloads(options):
    with pytest.raises(ValueError):
        BenchmarkConfig("model", "test", **options)


def test_cli_defaults_and_validation():
    assert parse_args(["--model", "model", "--label", "before"]).concurrency == 4
    with pytest.raises(SystemExit) as error:
        parse_args(["--model", "model", "--label", "before", "--max-tokens", "0"])
    assert error.value.code == 2


def test_percentiles_and_reasoning_chunks_are_not_tokens():
    result = RequestResult(0, "prefill_long", 30, 10)
    result.dispatched_seconds = 1.0
    assert not consume_payload(
        json.dumps({"choices": [{"delta": {"role": "assistant"}}]}), result, 1.1
    )
    assert consume_payload(
        json.dumps(
            {"choices": [{"delta": {"reasoning_content": "many reasoning tokens"}}]}
        ),
        result,
        2.0,
    )
    consume_payload(json.dumps(output_event()), result, 2.5)
    consume_payload(json.dumps(output_event()), result, 4.0)
    report = result.report()
    assert report["ttft_seconds"] == 1.0
    assert report["output_chunks"] == 3
    assert report["chunk_gap_p50_seconds"] == 1.0
    assert report["chunk_gap_p95_seconds"] == pytest.approx(1.45)
    assert percentile([], 0.95) is None


async def test_request_parses_multiline_sse_usage_and_synthetic_timing():
    stream = ControlledStream(
        [
            b': keepalive\n\ndata: {"choices":\n',
            b'data: [{"delta": {"role": "assistant"}}]}\n\n',
            sse(output_event("several output tokens")),
            sse(output_event(" next")),
            sse(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 67,
                        "completion_tokens": 4,
                        "total_tokens": 71,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, stream=stream)

    spec = RequestSpec(0, "prefill_long", "private prompt body", 8)
    result = result_for(spec)
    observed = iter([10.0, 10.1, 11.0, 11.5, 11.6, 12.0])
    first_output = asyncio.Event()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await run_request(
            client,
            BenchmarkConfig("model", "test"),
            spec,
            result,
            10.0,
            first_output,
            lambda: next(observed),
        )
    report = result.report()
    assert report["status"] == "completed"
    assert report["ttft_seconds"] == 1.0
    assert report["end_to_end_seconds"] == 2.0
    assert report["chunk_gap_p95_seconds"] == 0.5
    assert report["usage"]["completion_tokens"] == 4
    assert report["usage"]["cached_prompt_tokens"] == 0
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert first_output.is_set() and stream.closed
    assert spec.prompt not in json.dumps(report)


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        ([sse(output_event())], "incomplete_stream"),
        ([b"data: invalid secret\n\n"], "invalid_sse_json"),
        ([sse({"error": {"message": "sensitive details"}})], "server_error_event"),
    ],
)
async def test_stream_failures_are_explicit_and_safe(chunks, expected):
    spec = RequestSpec(0, "prefill_long", "secret prompt", 8)
    result = result_for(spec)
    stream = ControlledStream(chunks)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream)
        )
    ) as client:
        await run_request(
            client, BenchmarkConfig("model", "test"), spec, result, 0, asyncio.Event()
        )
    assert result.status == "failed" and result.error == expected
    assert stream.closed
    assert "secret" not in json.dumps(result.report())


async def test_http_error_does_not_echo_server_body_or_credentials():
    config = BenchmarkConfig(
        "model", "test", base_url="http://user:password@example.test/v1?key=secret"
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text="private server body")
        )
    ) as client:
        report = await run_benchmark(config, client)
    assert all(result["error"] == "http_status_error" for result in report["requests"])
    for secret in ("password", "secret", "private server body"):
        assert secret not in json.dumps(report)


async def test_simultaneous_report_uses_authoritative_usage():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=complete_stream())
        )
    ) as client:
        report = await run_benchmark(
            BenchmarkConfig("model", "test", concurrency=2), client
        )
    assert report["summary"]["requests_completed"] == 2
    assert report["summary"]["successful_prompt_tokens"] == 182
    assert report["summary"]["successful_completion_tokens"] == 4
    assert report["summary"]["successful_completion_tokens_per_second"] > 0
    assert "multiple tokens" in report["metadata"]["latency_unit"]


async def test_staggered_waits_for_anchor_output_and_reports_overlap():
    anchor_release = asyncio.Event()
    dispatched = []
    anchor_stream = ControlledStream([sse(output_event())], block=anchor_release)

    async def handler(request):
        dispatched.append(json.loads(request.content))
        if len(dispatched) == 1:
            return httpx.Response(200, stream=anchor_stream)
        if len(dispatched) == 3:
            anchor_release.set()
        return httpx.Response(200, stream=complete_stream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_benchmark(
            BenchmarkConfig(
                "model", "test", scenario="staggered", concurrency=3, arrival_interval=0
            ),
            client,
        )
    assert dispatched[0]["max_tokens"] == 512
    assert all(result["decode_active_at_dispatch"] for result in report["requests"][1:])
    assert report["requests"][0]["error"] == "incomplete_stream"
    assert report["summary"]["requests_completed"] == 2


async def test_failed_anchor_skips_prefills():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))
    ) as client:
        report = await run_benchmark(
            BenchmarkConfig("model", "test", scenario="staggered"), client
        )
    assert report["requests"][0]["status"] == "failed"
    assert all(result["status"] == "skipped" for result in report["requests"][1:])


async def test_missing_usage_is_not_invented_from_chunk_count():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=sse(output_event()) + b"data: [DONE]\n\n"
            )
        )
    ) as client:
        report = await run_benchmark(
            BenchmarkConfig("model", "test", concurrency=1), client
        )
    assert report["requests"][0]["output_chunks"] == 1
    assert report["summary"]["successful_completion_tokens_per_second"] is None


async def test_request_timeout_and_global_deadline_close_streams():
    streams = []

    def handler(request):
        stream = ControlledStream([], block=asyncio.Event())
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_benchmark(
            BenchmarkConfig("model", "test", concurrency=1, request_timeout=0.001),
            client,
        )
        assert report["requests"][0]["status"] == "timed_out"
        report = await run_benchmark(
            BenchmarkConfig("model", "test", scenario="staggered", duration=0.001),
            client,
        )
        assert report["metadata"]["deadline_exceeded"]
        assert all(result["status"] == "cancelled" for result in report["requests"])
    assert all(stream.closed for stream in streams)


async def test_cancellation_remains_observable_and_propagates():
    connected = asyncio.Event()
    stream = ControlledStream([], block=asyncio.Event())

    def handler(request):
        connected.set()
        return httpx.Response(200, stream=stream)

    spec = RequestSpec(0, "prefill_long", "prompt", 8)
    result = result_for(spec)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(
            run_request(
                client,
                BenchmarkConfig("model", "test"),
                spec,
                result,
                0,
                asyncio.Event(),
            )
        )
        await connected.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert result.status == "cancelled"
    assert result.ended_seconds is not None
    assert stream.closed


@pytest.mark.parametrize(
    "options",
    [
        {"request_count": 0},
        {"request_count": 4097},
        {"request_count": True},
        {"prompts": ()},
        {"prompts": (" ",)},
        {"prompts": (3,)},
        {"prompts": ("one",), "request_count": 2},
    ],
)
def test_sustained_config_rejects_invalid_workloads(options):
    with pytest.raises(ValueError):
        BenchmarkConfig("model", "test", scenario="sustained", **options)


def test_corpus_cli_and_synthetic_job_size(tmp_path):
    corpus = tmp_path / "prompts.json"
    corpus.write_text(json.dumps(["First unique input", "第二条输入"]))
    config = parse_args(
        [
            "--model",
            "model",
            "--label",
            "test",
            "--scenario",
            "sustained",
            "--prompts-file",
            str(corpus),
            "--include-output",
        ]
    )
    assert config.include_output
    assert [spec.prompt for spec in request_specs(config)] == [
        "First unique input",
        "第二条输入",
    ]
    synthetic = BenchmarkConfig("model", "test", scenario="sustained", request_count=13)
    assert len(request_specs(synthetic)) == 13
    assert all(spec.role == "generation" for spec in request_specs(synthetic))
    with pytest.raises(ValueError, match="sustained"):
        BenchmarkConfig("model", "test", request_count=10)
    with pytest.raises(ValueError, match="sustained"):
        BenchmarkConfig("model", "test", prompts=("one",))


@pytest.mark.parametrize("contents", ["not json", "{}", "[]", "[null]", '[" "]'])
def test_corpus_cli_rejects_invalid_input(tmp_path, contents):
    corpus = tmp_path / "prompts.json"
    corpus.write_text(contents)
    with pytest.raises(SystemExit) as error:
        parse_args(
            [
                "--model",
                "model",
                "--label",
                "test",
                "--scenario",
                "sustained",
                "--prompts-file",
                str(corpus),
            ]
        )
    assert error.value.code == 2


def test_output_retention_is_explicit_and_separates_reasoning():
    retained = RequestResult(0, "generation", 5, 128, retain_output=True)
    private = RequestResult(0, "generation", 5, 128)
    for result in (retained, private):
        consume_payload(
            json.dumps({"choices": [{"delta": {"reasoning_content": "think"}}]}),
            result,
            1,
        )
        consume_payload(json.dumps(output_event("short ")), result, 2)
        consume_payload(json.dumps(output_event("answer")), result, 3)
    assert retained.report()["output_text"] == "short answer"
    assert retained.report()["reasoning_text"] == "think"
    assert "output_text" not in private.report()
    assert not private.content_parts and not private.reasoning_parts


async def test_sustained_job_is_bounded_and_refills_until_corpus_is_exhausted():
    admitted = []
    streams = []
    both_started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    peak_active = 0

    class JobStream(ControlledStream):
        async def __aiter__(self):
            await release.wait()
            for chunk in self.chunks:
                yield chunk

        async def aclose(self):
            nonlocal active
            active -= 1
            await super().aclose()

    def handler(request):
        nonlocal active, peak_active
        admitted.append(json.loads(request.content)["messages"][0]["content"])
        active += 1
        peak_active = max(peak_active, active)
        if len(admitted) == 2:
            both_started.set()
        stream = JobStream(
            [
                sse(output_event("translation")),
                sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                sse(
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 40, "completion_tokens": 3},
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        )
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    config = BenchmarkConfig(
        "model",
        "test",
        scenario="sustained",
        concurrency=2,
        prompts=("first", "second", "third", "fourth", "fifth"),
        include_output=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(run_benchmark(config, client))
        await asyncio.wait_for(both_started.wait(), timeout=2)
        assert admitted == ["first", "second"] and active == 2
        release.set()
        report = await task
    assert admitted == list(config.prompts)
    assert peak_active == 2 and active == 0
    assert all(stream.closed for stream in streams)
    assert report["summary"]["requests_completed"] == 5
    assert report["summary"]["finish_reasons"] == {"stop": 5}
    assert report["metadata"]["request_count"] == 5
    assert len(report["metadata"]["corpus_sha256"]) == 64
    assert report["metadata"]["prompt_words"] is None
    for result in report["requests"]:
        assert result["output_text"] == "translation"
        assert result["client_queue_wait_seconds"] == result["dispatched_seconds"]
        assert result["job_completion_seconds"] == result["ended_seconds"]
        assert result["job_completion_seconds"] >= result["end_to_end_seconds"]


async def test_sustained_deadline_closes_active_streams_and_marks_queued_work():
    streams = []

    def handler(request):
        stream = ControlledStream([], block=asyncio.Event())
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_benchmark(
            BenchmarkConfig(
                "model",
                "test",
                scenario="sustained",
                concurrency=2,
                request_count=5,
                duration=0.02,
            ),
            client,
        )
    assert len(streams) == 2 and all(stream.closed for stream in streams)
    assert report["metadata"]["deadline_exceeded"]
    assert all(result["status"] == "cancelled" for result in report["requests"])
    assert all(
        result["dispatched_seconds"] is None for result in report["requests"][2:]
    )
    assert report["summary"]["requests_not_completed"] == 5


def test_summary_separates_request_throughput_latency_and_truncation():
    results = [
        RequestResult(
            0,
            "generation",
            40,
            128,
            status="completed",
            dispatched_seconds=1,
            ended_seconds=3,
            finish_reason="stop",
        ),
        RequestResult(
            1,
            "generation",
            40,
            128,
            status="completed",
            dispatched_seconds=3,
            ended_seconds=7,
            finish_reason="length",
        ),
        RequestResult(2, "generation", 40, 128, status="failed"),
    ]
    summary = summarize(results, 8)
    assert summary["successful_requests_per_second"] == 0.25
    assert summary["successful_end_to_end_p50_seconds"] == 3
    assert summary["successful_end_to_end_p95_seconds"] == pytest.approx(3.9)
    assert summary["finish_reasons"] == {"stop": 1, "length": 1}
