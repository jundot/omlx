#!/usr/bin/env python3
"""Benchmark: interactive preemption under daemon load.

Measures TTFT for priority-0 (interactive) requests while priority-10
(background) requests are running concurrently.

Usage:
    python scripts/bench_interactive_preemption.py \
        --base-url http://127.0.0.1:18200 \
        --model qwen3.6-35b-a3b-4bit \
        --background 8 \
        --background-prompt-tokens 16384 \
        --interactive-runs 20 \
        --turns 4

Requires a running oMLX server with the model loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    request_id: str
    is_interactive: bool
    time_to_first_token: float | None = None
    prompt_eval_duration: float | None = None
    generation_duration: float | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None
    cached_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


@dataclass
class BenchResult:
    run_label: str
    background_count: int
    interactive_count: int
    interactive_ttfts: list[float] = field(default_factory=list)
    background_ttfts: list[float] = field(default_factory=list)
    interactive_p50: float | None = None
    interactive_p95: float | None = None
    interactive_p99: float | None = None
    background_p50: float | None = None
    all_metrics: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])


def _prompt_tokens(n: int) -> list[int]:
    """Generate n tokens (repeating a small vocabulary)."""
    vocab = list(range(1000, 1100))
    return [vocab[i % len(vocab)] for i in range(n)]


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def check_health(base_url: str) -> dict:
    r = requests.get(f"{base_url}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def get_model_id(base_url: str, model: str) -> str:
    """Resolve model name via /v1/models."""
    r = requests.get(f"{base_url}/v1/models", timeout=10)
    r.raise_for_status()
    models = r.json().get("data", [])
    for m in models:
        mid = m.get("id", "")
        if model.lower() in mid.lower():
            return mid
    return model


def fire_request(
    base_url: str,
    model: str,
    prompt_tokens: list[int],
    priority: int,
    max_tokens: int = 64,
    stream: bool = True,
) -> RequestMetrics:
    """Send a single request and parse metrics from the final SSE chunk."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": max_tokens,
        "stream": stream,
        "stream_options": {"include_usage": True},
    }
    # Non-streaming fallback
    if not stream:
        payload["stream"] = False

    t_start = time.monotonic()

    try:
        if stream:
            return _fire_stream(base_url, payload, priority, t_start)
        else:
            return _fire_non_stream(base_url, payload, priority, t_start)
    except Exception as e:
        return RequestMetrics(
            request_id="error",
            is_interactive=(priority == 0),
            error=str(e),
        )


def _fire_stream(
    base_url: str, payload: dict, priority: int, t_start: float
) -> RequestMetrics:
    """Fire a streaming request, parse TTFT and usage from SSE."""
    headers = {"X-Request-Priority": str(priority)}
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers=headers,
        stream=True,
        timeout=120,
    )
    r.raise_for_status()

    ttft = None
    usage = None
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # TTFT: first chunk with content
        if ttft is None:
            choices = chunk.get("choices", [])
            if choices and choices[0].get("delta", {}).get("content"):
                ttft = time.monotonic() - t_start

        # Usage in final chunk
        if "usage" in chunk:
            usage = chunk["usage"]

    elapsed = time.monotonic() - t_start

    if usage:
        return RequestMetrics(
            request_id="",
            is_interactive=(priority == 0),
            time_to_first_token=usage.get("time_to_first_token") or ttft,
            prompt_eval_duration=usage.get("prompt_eval_duration"),
            generation_duration=usage.get("generation_duration"),
            prompt_tokens_per_second=usage.get("prompt_tokens_per_second"),
            generation_tokens_per_second=usage.get("generation_tokens_per_second"),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
    else:
        return RequestMetrics(
            request_id="",
            is_interactive=(priority == 0),
            time_to_first_token=ttft or elapsed,
        )


def _fire_non_stream(
    base_url: str, payload: dict, priority: int, t_start: float
) -> RequestMetrics:
    """Fire a non-streaming request, parse TTFT from timing."""
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    resp = r.json()
    elapsed = time.monotonic() - t_start
    usage = resp.get("usage", {})
    return RequestMetrics(
        request_id="",
        is_interactive=(priority == 0),
        time_to_first_token=usage.get("time_to_first_token") or elapsed,
        prompt_eval_duration=usage.get("prompt_eval_duration"),
        generation_duration=usage.get("generation_duration"),
        prompt_tokens_per_second=usage.get("prompt_tokens_per_second"),
        generation_tokens_per_second=usage.get("generation_tokens_per_second"),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_bench(
    base_url: str,
    model: str,
    bg_count: int,
    bg_prompt_tokens: int,
    interactive_runs: int,
    turns: int,
    max_tokens: int,
    label: str,
) -> BenchResult:
    """Run one benchmark pass: bg_count background + interactive_runs interactive."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"  background: {bg_count} x {bg_prompt_tokens} prompt tokens")
    print(f"  interactive: {interactive_runs} runs x {turns} turns")
    print(f"{'='*60}")

    model_id = get_model_id(base_url, model)
    print(f"  resolved model: {model_id}")

    # Build prompts
    bg_prompt = _prompt_tokens(bg_prompt_tokens)

    result = BenchResult(run_label=label, background_count=bg_count, interactive_count=interactive_runs)

    # Phase 1: fire background requests
    print(f"\n  Firing {bg_count} background requests (priority=10)...")
    bg_futures = {}
    with ThreadPoolExecutor(max_workers=bg_count) as pool:
        for i in range(bg_count):
            fut = pool.submit(
                fire_request, base_url, model_id, bg_prompt, priority=10, max_tokens=max_tokens
            )
            bg_futures[fut] = i
        # Don't wait yet — let them start, then fire interactive

    # Brief pause so background requests start prefilling
    time.sleep(0.5)

    # Phase 2: fire interactive requests
    print(f"  Firing {interactive_runs} interactive requests (priority=0)...")
    interactive_metrics: list[RequestMetrics] = []
    turn_prompt = _prompt_tokens(256)

    for run_i in range(interactive_runs):
        for turn in range(turns):
            m = fire_request(base_url, model_id, turn_prompt, priority=0, max_tokens=64)
            m.request_id = f"interactive-{run_i}-turn{turn}"
            interactive_metrics.append(m)
            if m.time_to_first_token is not None:
                print(f"    run {run_i} turn {turn}: TTFT={m.time_to_first_token:.3f}s")
            elif m.error:
                print(f"    run {run_i} turn {turn}: ERROR={m.error}")

    # Collect background results
    print(f"\n  Waiting for {bg_count} background requests to finish...")
    bg_metrics: list[RequestMetrics] = []
    for fut in as_completed(bg_futures):
        m = fut.result()
        m.request_id = f"background-{bg_futures[fut]}"
        bg_metrics.append(m)

    # Aggregate
    result.interactive_ttfts = [
        m.time_to_first_token for m in interactive_metrics
        if m.time_to_first_token is not None and m.error is None
    ]
    result.background_ttfts = [
        m.time_to_first_token for m in bg_metrics
        if m.time_to_first_token is not None and m.error is None
    ]
    result.interactive_p50 = _percentile(result.interactive_ttfts, 0.50)
    result.interactive_p95 = _percentile(result.interactive_ttfts, 0.95)
    result.interactive_p99 = _percentile(result.interactive_ttfts, 0.99)
    result.background_p50 = _percentile(result.background_ttfts, 0.50)
    result.all_metrics = [asdict(m) for m in interactive_metrics + bg_metrics]

    print(f"\n  Results ({label}):")
    print(f"    interactive TTFT p50={result.interactive_p50:.3f}s  p95={result.interactive_p95:.3f}s  p99={result.interactive_p99:.3f}s  n={len(result.interactive_ttfts)}")
    print(f"    background  TTFT p50={result.background_p50:.3f}s  n={len(result.background_ttfts)}")

    errors = [m for m in interactive_metrics if m.error]
    if errors:
        print(f"    errors: {len(errors)}")
        for e in errors[:5]:
            print(f"      {e.request_id}: {e.error}")

    return result


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def run_assertions(
    result: BenchResult,
    assert_ttft_p95: float | None,
    assert_throughput_retention: float | None,
    baseline: BenchResult | None,
) -> list[str]:
    """Return list of failure messages. Empty = all pass."""
    failures: list[str] = []

    if assert_ttft_p95 is not None:
        if result.interactive_p95 is None:
            failures.append(f"p95 TTFT: no data (all requests failed?)")
        elif result.interactive_p95 > assert_ttft_p95:
            failures.append(
                f"p95 TTFT {result.interactive_p95:.3f}s > {assert_ttft_p95}s target"
            )
        else:
            print(f"  PASS: p95 TTFT {result.interactive_p95:.3f}s <= {assert_ttft_p95}s")

    if baseline and assert_throughput_retention is not None:
        if baseline.background_p50 and result.background_p50:
            retention = result.background_p50 / baseline.background_p50
            if retention < assert_throughput_retention:
                failures.append(
                    f"background throughput retention {retention:.2f} < {assert_throughput_retention}"
                )
            else:
                print(f"  PASS: background throughput retention {retention:.2f} >= {assert_throughput_retention}")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark interactive preemption")
    parser.add_argument("--base-url", default="http://127.0.0.1:18200")
    parser.add_argument("--model", default="qwen3.6-35b-a3b-4bit")
    parser.add_argument("--background", type=int, default=8)
    parser.add_argument("--background-prompt-tokens", type=int, default=16384)
    parser.add_argument("--interactive-runs", type=int, default=20)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline JSON")
    parser.add_argument("--assert-interactive-ttft-p95", type=float, default=None)
    parser.add_argument("--assert-quiet-throughput-retention", type=float, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    # Health check
    try:
        health = check_health(args.base_url)
        print(f"Server healthy: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.base_url}: {e}", file=sys.stderr)
        return 1

    # Load baseline if provided
    baseline = None
    if args.baseline:
        bp = Path(args.baseline)
        if bp.exists():
            data = json.loads(bp.read_text())
            baseline = BenchResult(**{k: v for k, v in data.items() if k in BenchResult.__dataclass_fields__})
            print(f"Loaded baseline from {args.baseline}")

    # Run
    result = run_bench(
        base_url=args.base_url,
        model=args.model,
        bg_count=args.background,
        bg_prompt_tokens=args.background_prompt_tokens,
        interactive_runs=args.interactive_runs,
        turns=args.turns,
        max_tokens=args.max_tokens,
        label="candidate",
    )

    # Assert
    failures = run_assertions(
        result,
        assert_ttft_p95=args.assert_interactive_ttft_p95,
        assert_throughput_retention=args.assert_quiet_throughput_retention,
        baseline=baseline,
    )

    # Save
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(result), indent=2))
        print(f"\nSaved to {args.json_out}")

    if failures:
        print(f"\nFAILED: {len(failures)} assertion(s):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"\nAll assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
