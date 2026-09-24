#!/usr/bin/env python3
"""Observe prefill throughput and decode fairness through a running HTTP server.

Run identical workloads against separately prepared before/after servers::

    python benchmarks/batched_prefill_bench.py --model MODEL --label baseline \
        --base-url http://127.0.0.1:8000/v1 > baseline.json
    python benchmarks/batched_prefill_bench.py --model MODEL --label candidate \
        --base-url http://127.0.0.1:8001/v1 > candidate.json

Use the same scenario, seed, sizes, model, and server settings for comparisons.
No server is started and no model is downloaded. OPENAI_API_KEY supplies optional
authentication. Prompt sizes are approximate word counts, not token counts;
server usage is authoritative. Distinct leading text reduces cross-request
prefix reuse, but repeated runs with the same seed can hit an existing cache.
Prepare cache state externally and record it with --cache-state.

TTFT measures the first nonempty content/reasoning SSE delta. Gaps measure
output-bearing SSE chunks, which may contain multiple tokens; they are not
individual token latencies. Throughput uses successful requests' reported
usage over the entire workload wall time, not isolated GPU prefill time.
The staggered scenario waits for the first decoding response, then injects
alternating long/short prompts. Early EOS can prevent actual decode overlap;
each request records whether the first request was still active at dispatch.

The sustained scenario keeps at most --concurrency requests in flight until a
fixed job completes. Supply --request-count for synthetic prompts or
--prompts-file with a JSON array of distinct, nonempty prompt strings. All job
items are ready at the start; per-request latency starts at HTTP dispatch, while
job completion latency includes waiting in the client queue. Use natural EOS
for short generation; max_tokens is a safety cap, not a target output length.
--include-output explicitly retains generated content and reasoning in the
report for offline correctness checks. Outputs are omitted by default.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx


@dataclass(frozen=True)
class BenchmarkConfig:
    model: str
    label: str
    base_url: str = "http://127.0.0.1:8000/v1"
    scenario: str = "simultaneous"
    concurrency: int = 4
    prompt_words: int = 2048
    short_prompt_words: int = 64
    max_tokens: int = 64
    decode_max_tokens: int = 512
    arrival_interval: float = 0.1
    request_timeout: float = 120.0
    duration: float = 180.0
    seed: int = 0
    cache_state: str = "unspecified"
    server_revision: str = "unspecified"
    server_hardware: str = "unspecified"
    request_count: int | None = None
    prompts: tuple[str, ...] | None = None
    include_output: bool = False

    def __post_init__(self):
        for name, upper in (
            ("concurrency", 32),
            ("prompt_words", 65536),
            ("short_prompt_words", 65536),
            ("max_tokens", 16384),
            ("decode_max_tokens", 16384),
        ):
            if not 1 <= getattr(self, name) <= upper:
                raise ValueError(f"{name} must be between 1 and {upper}")
        for name in ("arrival_interval", "request_timeout", "duration"):
            value = getattr(self, name)
            minimum = 0 if name == "arrival_interval" else 0.001
            if not math.isfinite(value) or not minimum <= value <= 3600:
                raise ValueError(f"{name} must be between {minimum} and 3600")
        if self.scenario not in ("simultaneous", "staggered", "sustained"):
            raise ValueError("scenario must be simultaneous, staggered or sustained")
        if self.scenario == "staggered" and self.concurrency < 2:
            raise ValueError("staggered requires concurrency >= 2")
        if self.request_count is not None:
            if (
                type(self.request_count) is not int
                or not 1 <= self.request_count <= 4096
            ):
                raise ValueError("request_count must be between 1 and 4096")
            if self.scenario != "sustained":
                raise ValueError("request_count requires the sustained scenario")
        if self.prompts is not None:
            if (
                not isinstance(self.prompts, tuple)
                or not 1 <= len(self.prompts) <= 4096
                or any(
                    not isinstance(prompt, str) or not prompt.strip()
                    for prompt in self.prompts
                )
            ):
                raise ValueError("prompts must contain 1 to 4096 nonempty strings")
            if self.scenario != "sustained":
                raise ValueError("prompts requires the sustained scenario")
            if self.request_count is not None and self.request_count != len(
                self.prompts
            ):
                raise ValueError("request_count must match the number of prompts")
        endpoint_url(self.base_url)


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    role: str
    prompt: str
    max_tokens: int


@dataclass
class RequestResult:
    request_id: int
    role: str
    input_characters: int
    max_tokens: int
    status: str = "not_started"
    error: str | None = None
    http_status: int | None = None
    dispatched_seconds: float | None = None
    ended_seconds: float | None = None
    decode_active_at_dispatch: bool | None = None
    finish_reason: str | None = None
    output_times: list[float] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    retain_output: bool = False
    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)

    def report(self):
        gaps = [
            current - previous
            for previous, current in zip(self.output_times, self.output_times[1:])
        ]
        report = {
            "request_id": self.request_id,
            "role": self.role,
            "input_characters": self.input_characters,
            "max_tokens": self.max_tokens,
            "status": self.status,
            "error": self.error,
            "http_status": self.http_status,
            "dispatched_seconds": self.dispatched_seconds,
            "ended_seconds": self.ended_seconds,
            "decode_active_at_dispatch": self.decode_active_at_dispatch,
            "finish_reason": self.finish_reason,
            "ttft_seconds": (
                self.output_times[0] - self.dispatched_seconds
                if self.output_times and self.dispatched_seconds is not None
                else None
            ),
            "end_to_end_seconds": (
                self.ended_seconds - self.dispatched_seconds
                if self.ended_seconds is not None
                and self.dispatched_seconds is not None
                else None
            ),
            "output_chunks": len(self.output_times),
            "output_chunk_times_seconds": self.output_times,
            "chunk_gap_p50_seconds": percentile(gaps, 0.5),
            "chunk_gap_p95_seconds": percentile(gaps, 0.95),
            "usage": self.usage,
        }
        if self.retain_output:
            report["output_text"] = "".join(self.content_parts)
            report["reasoning_text"] = "".join(self.reasoning_parts)
        return report


class StreamError(Exception):
    """A safe protocol error code without response or request contents."""


def endpoint_url(base_url: str, *, redact: bool = False) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("base_url must be an HTTP(S) API base URL")
    authority = parsed.netloc.rsplit("@", 1)[-1] if redact else parsed.netloc
    return urlunsplit(
        (
            parsed.scheme,
            authority,
            parsed.path.rstrip("/") + "/chat/completions",
            "" if redact else parsed.query,
            "",
        )
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def synthetic_prompt(seed: int, request_id: int, word_count: int) -> str:
    identity = hashlib.sha256(f"{seed}:{request_id}".encode()).hexdigest()
    words = [
        "river",
        "forest",
        "observatory",
        "copper",
        "signal",
        "meadow",
        "harbor",
        "distance",
    ]
    body = " ".join(words[index % len(words)] for index in range(word_count))
    return (
        f"{identity}\n{body}\n"
        "Write an extensive numbered explanation connecting these observations. "
        "Continue with detailed examples for as long as possible."
    )


def request_specs(config: BenchmarkConfig) -> list[RequestSpec]:
    specs = []
    count = (
        len(config.prompts)
        if config.prompts is not None
        else (config.request_count or config.concurrency)
    )
    for request_id in range(count):
        role, words, maximum = "prefill_long", config.prompt_words, config.max_tokens
        if config.scenario == "sustained":
            role = "generation"
        if config.scenario == "staggered":
            if request_id == 0:
                role, words, maximum = (
                    "decode_anchor",
                    config.short_prompt_words,
                    config.decode_max_tokens,
                )
            elif request_id % 2 == 0:
                role, words = "prefill_short", config.short_prompt_words
        specs.append(
            RequestSpec(
                request_id,
                role,
                (
                    config.prompts[request_id]
                    if config.prompts is not None
                    else synthetic_prompt(config.seed, request_id, words)
                ),
                maximum,
            )
        )
    return specs


async def sse_payloads(response: httpx.Response) -> AsyncIterator[str]:
    data = []
    size = 0
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data, size = [], 0
        elif line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            size += len(value)
            if size > 1024 * 1024:
                raise StreamError("sse_event_too_large")
            data.append(value)
    if data:
        yield "\n".join(data)


def consume_payload(payload: str, result: RequestResult, observed: float) -> bool:
    try:
        event = json.loads(payload)
    except json.JSONDecodeError as error:
        raise StreamError("invalid_sse_json") from error
    if not isinstance(event, dict):
        raise StreamError("invalid_sse_event")
    if "error" in event:
        raise StreamError("server_error_event")
    usage = event.get("usage")
    if isinstance(usage, dict):
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(name)
            if type(value) is int and value >= 0:
                result.usage[name] = value
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            value = details.get("cached_tokens")
            if type(value) is int and value >= 0:
                result.usage["cached_prompt_tokens"] = value
    has_output = False
    choices = event.get("choices", [])
    if not isinstance(choices, list):
        raise StreamError("invalid_sse_choices")
    for choice in choices:
        if not isinstance(choice, dict):
            raise StreamError("invalid_sse_choice")
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise StreamError("invalid_sse_delta")
        has_output |= any(
            isinstance(delta.get(name), str) and bool(delta[name])
            for name in ("content", "reasoning_content", "reasoning")
        )
        if result.retain_output:
            content = delta.get("content")
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(content, str):
                result.content_parts.append(content)
            if isinstance(reasoning, str):
                result.reasoning_parts.append(reasoning)
        reason = choice.get("finish_reason")
        if reason in (
            "stop",
            "length",
            "content_filter",
            "tool_calls",
            "function_call",
        ):
            result.finish_reason = reason
    if has_output:
        result.output_times.append(observed)
    return has_output


async def run_request(
    client: httpx.AsyncClient,
    config: BenchmarkConfig,
    spec: RequestSpec,
    result: RequestResult,
    started: float,
    first_output: asyncio.Event,
    clock: Callable[[], float] = time.perf_counter,
    decode_anchor: RequestResult | None = None,
) -> None:
    result.dispatched_seconds = clock() - started
    if decode_anchor is not None:
        result.decode_active_at_dispatch = decode_anchor.status == "running"
    result.status = "running"
    try:
        async with asyncio.timeout(config.request_timeout):
            async with client.stream(
                "POST",
                endpoint_url(config.base_url),
                json={
                    "model": config.model,
                    "messages": [{"role": "user", "content": spec.prompt}],
                    "max_tokens": spec.max_tokens,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                timeout=config.request_timeout,
            ) as response:
                result.http_status = response.status_code
                response.raise_for_status()
                completed = False
                async for payload in sse_payloads(response):
                    if payload.strip() == "[DONE]":
                        completed = True
                        break
                    if consume_payload(payload, result, clock() - started):
                        first_output.set()
                if not completed:
                    raise StreamError("incomplete_stream")
                result.status = "completed"
    except asyncio.CancelledError:
        result.status, result.error = "cancelled", "cancelled"
        raise
    except (TimeoutError, httpx.TimeoutException):
        result.status, result.error = "timed_out", "request_timeout"
    except httpx.HTTPStatusError:
        result.status, result.error = "failed", "http_status_error"
    except httpx.HTTPError as error:
        result.status, result.error = "failed", type(error).__name__
    except StreamError as error:
        result.status, result.error = "failed", str(error)
    finally:
        result.ended_seconds = clock() - started


def summarize(results: list[RequestResult], elapsed: float) -> dict:
    completed = [result for result in results if result.status == "completed"]
    summary = {
        "wall_seconds": elapsed,
        "requests_completed": len(completed),
        "requests_not_completed": len(results) - len(completed),
        "successful_requests_per_second": (
            len(completed) / elapsed if elapsed > 0 else None
        ),
    }
    ttfts = [
        result.output_times[0] - result.dispatched_seconds
        for result in completed
        if result.output_times and result.dispatched_seconds is not None
    ]
    summary["successful_ttft_p50_seconds"] = percentile(ttfts, 0.5)
    summary["successful_ttft_p95_seconds"] = percentile(ttfts, 0.95)
    latencies = [
        result.ended_seconds - result.dispatched_seconds
        for result in completed
        if result.ended_seconds is not None and result.dispatched_seconds is not None
    ]
    summary["successful_end_to_end_p50_seconds"] = percentile(latencies, 0.5)
    summary["successful_end_to_end_p95_seconds"] = percentile(latencies, 0.95)
    summary["finish_reasons"] = {
        reason
        or "unspecified": sum(result.finish_reason == reason for result in completed)
        for reason in (
            "stop",
            "length",
            "content_filter",
            "tool_calls",
            "function_call",
            None,
        )
        if any(result.finish_reason == reason for result in completed)
    }
    for name in ("prompt_tokens", "completion_tokens"):
        complete_usage = bool(completed) and all(
            name in result.usage for result in completed
        )
        total = (
            sum(result.usage[name] for result in completed) if complete_usage else None
        )
        summary[f"successful_{name}"] = total
        summary[f"successful_{name}_per_second"] = (
            total / elapsed if total is not None and elapsed > 0 else None
        )
    return summary


async def run_benchmark(config: BenchmarkConfig, client: httpx.AsyncClient) -> dict:
    specs = request_specs(config)
    results = [
        RequestResult(
            spec.request_id,
            spec.role,
            len(spec.prompt),
            spec.max_tokens,
            retain_output=config.include_output,
        )
        for spec in specs
    ]
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    tasks = []
    output_waiter = None
    first_output = asyncio.Event()
    deadline_exceeded = False

    pending = iter(range(len(specs)))

    async def worker():
        for request_id in pending:
            await run_request(
                client,
                config,
                specs[request_id],
                results[request_id],
                started,
                first_output,
            )

    def dispatch(request_id: int):
        result = results[request_id]
        decode_anchor = (
            results[0] if config.scenario == "staggered" and request_id else None
        )
        tasks.append(
            asyncio.create_task(
                run_request(
                    client,
                    config,
                    specs[request_id],
                    result,
                    started,
                    first_output,
                    decode_anchor=decode_anchor,
                )
            )
        )

    try:
        async with asyncio.timeout(config.duration):
            if config.scenario == "sustained":
                tasks.extend(
                    asyncio.create_task(worker())
                    for _ in range(min(config.concurrency, len(specs)))
                )
            else:
                dispatch(0)
            if config.scenario == "staggered":
                output_waiter = asyncio.create_task(first_output.wait())
                await asyncio.wait(
                    [tasks[0], output_waiter], return_when=asyncio.FIRST_COMPLETED
                )
                if not first_output.is_set():
                    for result in results[1:]:
                        result.status, result.error = (
                            "skipped",
                            "anchor_produced_no_output",
                        )
                    await tasks[0]
                    return benchmark_report(config, results, started_at, started, False)
            if config.scenario != "sustained":
                for request_id in range(1, len(specs)):
                    if config.scenario == "staggered":
                        await asyncio.sleep(config.arrival_interval)
                    dispatch(request_id)
            await asyncio.gather(*tasks)
    except TimeoutError:
        deadline_exceeded = True
    finally:
        cleanup = tasks + ([output_waiter] if output_waiter is not None else [])
        for task in cleanup:
            if not task.done():
                task.cancel()
        await asyncio.gather(*cleanup, return_exceptions=True)
        for result in results:
            if result.status == "not_started":
                result.status, result.error = "cancelled", "cancelled_before_dispatch"
    return benchmark_report(config, results, started_at, started, deadline_exceeded)


def benchmark_report(config, results, started_at, started, deadline_exceeded):
    summary = summarize(results, time.perf_counter() - started)
    requests = [result.report() for result in results]
    if config.scenario == "sustained":
        for result in requests:
            result["client_queue_wait_seconds"] = result["dispatched_seconds"]
            result["job_completion_seconds"] = result["ended_seconds"]
        for name in ("client_queue_wait_seconds", "job_completion_seconds"):
            values = [
                result[name]
                for result in requests
                if result["status"] == "completed" and result[name] is not None
            ]
            for suffix, fraction in (("p50", 0.5), ("p95", 0.95)):
                summary[
                    f"successful_{name.removesuffix('_seconds')}_{suffix}_seconds"
                ] = percentile(values, fraction)
    return {
        "schema_version": 2,
        "metadata": {
            "label": config.label,
            "model": config.model,
            "endpoint": endpoint_url(config.base_url, redact=True),
            "started_at": started_at,
            "client_python": platform.python_version(),
            "server_revision": config.server_revision,
            "server_hardware": config.server_hardware,
            "cache_state": config.cache_state,
            "scenario": config.scenario,
            "concurrency": config.concurrency,
            "request_count": len(results),
            "prompt_source": "corpus" if config.prompts is not None else "synthetic",
            "corpus_sha256": (
                hashlib.sha256(
                    json.dumps(config.prompts, ensure_ascii=False).encode()
                ).hexdigest()
                if config.prompts is not None
                else None
            ),
            "include_output": config.include_output,
            "prompt_words": config.prompt_words if config.prompts is None else None,
            "short_prompt_words": (
                config.short_prompt_words if config.prompts is None else None
            ),
            "max_tokens": config.max_tokens,
            "decode_max_tokens": config.decode_max_tokens,
            "arrival_interval_seconds": config.arrival_interval,
            "request_timeout_seconds": config.request_timeout,
            "duration_limit_seconds": config.duration,
            "seed": config.seed,
            "latency_unit": "output-bearing SSE chunk; may contain multiple tokens",
            "token_counts": "server usage; prompt word counts are approximate",
            "cache_note": "prepare cache state externally; repeated prompts can reuse cached runs",
            "deadline_exceeded": deadline_exceeded,
        },
        "summary": summary,
        "requests": requests,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default=BenchmarkConfig.base_url)
    parser.add_argument(
        "--scenario",
        choices=("simultaneous", "staggered", "sustained"),
        default="simultaneous",
    )
    for name in (
        "concurrency",
        "prompt_words",
        "short_prompt_words",
        "max_tokens",
        "decode_max_tokens",
        "seed",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=getattr(BenchmarkConfig, name),
        )
    for name in ("arrival_interval", "request_timeout", "duration"):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            default=getattr(BenchmarkConfig, name),
        )
    parser.add_argument(
        "--cache-state", choices=("unspecified", "cold", "warm"), default="unspecified"
    )
    parser.add_argument("--server-revision", default="unspecified")
    parser.add_argument("--server-hardware", default="unspecified")
    parser.add_argument("--request-count", type=int)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--include-output", action="store_true")
    args = parser.parse_args(argv)
    prompts_file = vars(args).pop("prompts_file")
    try:
        if prompts_file is not None:
            with prompts_file.open("rb") as stream:
                payload = stream.read(16 * 1024 * 1024 + 1)
            if len(payload) > 16 * 1024 * 1024:
                raise ValueError("prompts_file exceeds 16 MiB")
            prompts = json.loads(payload)
            if not isinstance(prompts, list):
                raise ValueError("prompts_file must be a JSON array of strings")
            args.prompts = tuple(prompts)
        return BenchmarkConfig(**vars(args))
    except OSError:
        parser.error("cannot read prompts_file")
    except (json.JSONDecodeError, UnicodeDecodeError):
        parser.error("prompts_file must contain valid JSON")
    except ValueError as error:
        parser.error(str(error))


async def main_async(config):
    api_key = os.environ.get("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(
        headers=headers,
        limits=httpx.Limits(max_connections=config.concurrency),
    ) as client:
        return await run_benchmark(config, client)


def main(argv=None):
    config = parse_args(argv)
    try:
        report = asyncio.run(main_async(config))
    except KeyboardInterrupt:
        print("Benchmark cancelled; active HTTP requests were closed.", file=sys.stderr)
        return 130
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if report["summary"]["requests_not_completed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
