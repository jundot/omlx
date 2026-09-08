#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, single-request-at-a-time Qwen3.8-27B cache benchmark.

This intentionally uses only the Python standard library.  It is designed to
be run against an explicitly supplied, isolated oMLX server (usually port
8845).  It refuses the production port unless ``--allow-8843`` is supplied.
The harness does not change prompts between repetitions: prefix identity is a
measurement property, especially for the multi-turn and cache-boundary cases.

The generated JSONL is an append-only cache-boundary record. It contains
synthetic prompts only; no caller supplied logs or private prompts are
accepted by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MODEL = "Qwen--Qwen3.8-27B-oQ4e-mtp"
DEFAULT_OUTPUT = "results/qwen-cache-bench.jsonl"
FACT = "PLANTED_FACT_3070: the cache boundary marker is amber-otter-4821."
SHORT_PROMPTS = {
    "count": "Count from 1 to 200, one number per line. Output only the numbers.",
    "code": (
        "Write a Python function parse_log_line(line) using a compiled regex to "
        "parse an nginx access log into ip, ts, method, path, status, bytes, "
        "referer, and user_agent. Include type conversions and five doctest "
        "examples. Output only code."
    ),
}
WORD_BANK = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "maple",
    "north",
    "orbit",
    "paper",
    "quartz",
    "river",
    "sable",
    "tango",
    "umber",
    "violet",
    "willow",
    "xenon",
    "yellow",
    "zulu",
)
# Measured from the local Qwen tokenizer.json's ByteLevel-BPE vocabulary for
# this deterministic ASCII fixture (raw user content, before chat-template
# wrappers). Keep the estimator dependency-free; the tokenizer audit remains a
# CPU-only preflight check rather than a runtime requirement.
QWEN_APPROX_CHARS_PER_TOKEN = 3.68


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def approx_tokens(text: str) -> int:
    """Return a tokenizer-free estimate calibrated to the local Qwen BPE."""

    return max(1, round(len(text.encode("utf-8")) / QWEN_APPROX_CHARS_PER_TOKEN))


def build_long_prompt(target_tokens: int, seed: int, regime: str = "retrieval") -> str:
    """Build a repeatable synthetic prompt near ``target_tokens``.

    The planted fact is deliberately placed at one-third of the context for a
    separate retrieval case. Count/code cases use the same filler construction
    and exercise long-context decode without asking the model to retrieve the
    answer.
    """

    rng = random.Random(seed)
    header = (
        "Synthetic benchmark context. This is generated filler, not a user log.\n"
        "Read the complete context before answering.\n"
    )
    if regime == "retrieval":
        trailer = (
            "\nQuestion: What is the exact value after PLANTED_FACT_3070? Output "
            "only that value. /no_think"
        )
    elif regime == "count":
        trailer = "\nCount from 1 to 200, one number per line. Output only the numbers. /no_think"
    elif regime == "code":
        trailer = (
            "\nWrite a Python class implementing an LRU cache with get, put, and "
            "eviction callbacks, with docstrings. Output only code. /no_think"
        )
    else:
        raise ValueError(f"unknown long-context regime: {regime}")
    target_chars = max(
        len(header) + len(trailer) + 32,
        round(target_tokens * QWEN_APPROX_CHARS_PER_TOKEN),
    )
    # Keep a fixed prefix and place the fact at a stable interior location.  A
    # fresh RNG is used for every construction, so every arm sees byte-identical
    # text when given the same seed and target.
    filler_target = max(0, target_chars - len(header) - len(trailer))
    pieces: list[str] = []
    length = 0
    i = 0
    while length < filler_target:
        line = (
            f"record-{i:06d} "
            + " ".join(rng.choice(WORD_BANK) for _ in range(16))
            + "\n"
        )
        pieces.append(line)
        length += len(line)
        i += 1
    filler = "".join(pieces)[:filler_target]
    if regime == "retrieval":
        # Keep the marker in the context body, rather than in the question,
        # so retrieval and cache-boundary behaviour are both exercised.
        split = len(filler) // 3
        return header + filler[:split] + "\n" + FACT + "\n" + filler[split:] + trailer
    return header + filler + trailer


def fake_tool_result(turn: int, words: int = 730) -> str:
    """Produce a deterministic, private-data-free ~750-token tool result."""

    rng = random.Random(0x3070 + turn * 7919)
    lines = [f"Synthetic tool result turn={turn}; source=generated fixture."]
    while approx_tokens("\n".join(lines)) < words:
        n = len(lines)
        lines.append(
            f"item-{n:04d}: "
            + " ".join(rng.choice(WORD_BANK) for _ in range(14))
        )
    return "\n".join(lines)


def validate_base_url(base_url: str, allow_8843: bool) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--base-url must be an http(s) URL with a hostname")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port == 8843 and not allow_8843:
        raise ValueError(
            "refusing production port 8843; use an isolated port (normally 8845) "
            "or pass --allow-8843 deliberately"
        )
    return base_url.rstrip("/")


def read_log_end(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


MTP_RE = re.compile(
    r"MTP\[(?P<id>[^\]]+)\].*?tokens=(?P<tokens>\d+)\s+"
    r"cycles=(?P<cycles>\d+)\s+tok/cycle=(?P<tok_cycle>[\d.]+)\s+"
    r"accept=(?P<accepted>\d+)/(?P<attempted>\d+)\s+\((?P<accept_pct>[\d.]+)%\)"
    r".*?timing\[backbone=(?P<backbone_ms>[\d.]+)ms\s+"
    r"mtp=(?P<mtp_ms>[\d.]+)ms\s+sample=(?P<sample_ms>[\d.]+)ms\s+"
    r"cache=(?P<cache_ms>[\d.]+)ms\]"
)
DEPTH_RE = re.compile(r"depth\[(?P<depth>[^\]]+)\]")
DEPTH_ENTRY_RE = re.compile(r"(?P<name>d\d+)=(?P<num>\d+)/(?:\d+)")
BOUNDARY_RE = re.compile(
    r"boundary_snapshot_(?P<event>\w+)\s+tokens=(?P<tokens>\d+)\s+"
    r"block_size=(?P<block_size>\d+)\s+"
    r"available_boundaries=(?P<available>\d+)"
)
BOUNDARY_STORE_RE = re.compile(
    r"Using boundary cache snapshot for [^:]+:\s+storing\s+"
    r"(?P<stored>\d+)/(?P<total>\d+) tokens\s+"
    r"\(skipping trailing partial block,\s+(?P<intermediate>\d+)\s+"
    r"intermediate snapshots\)"
)
CACHE_PHASE_RE = re.compile(r"Cache phase timings:\s*(?P<phases>.*)$")
CACHE_PHASE_ENTRY_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_]+)=(?P<ms>[\d.]+)ms/(?P<count>\d+)"
)


def parse_log_slice(path: Path | None, start: int | None, end: int | None = None) -> dict[str, Any]:
    if path is None or start is None:
        return {"start": start, "end": end, "mtp": [], "cache_boundary": [], "cache_phase_timings": []}
    try:
        file_end = path.stat().st_size
        start = max(0, int(start))
        if start >= file_end:
            return {"start": start, "end": file_end, "mtp": [], "cache_boundary": [], "cache_phase_timings": []}
        actual_end = file_end if end is None else max(start, min(int(end), file_end))
        with path.open("rb") as handle:
            handle.seek(start)
            text = handle.read(actual_end - start).decode("utf-8", "replace")
    except FileNotFoundError:
        return {"start": start, "end": end, "mtp": [], "cache_boundary": [], "cache_phase_timings": []}

    mtp: list[dict[str, Any]] = []
    boundary: list[dict[str, Any]] = []
    phase_timings: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = MTP_RE.search(line)
        if match:
            row = {key: value for key, value in match.groupdict().items()}
            for key in ("tokens", "cycles", "accepted", "attempted"):
                row[key] = int(row[key])
            for key in ("tok_cycle", "accept_pct", "backbone_ms", "mtp_ms", "sample_ms", "cache_ms"):
                row[key] = float(row[key])
            row["decode_ms"] = sum(row[key] for key in ("backbone_ms", "mtp_ms", "sample_ms", "cache_ms"))
            row["decode_tps"] = row["tokens"] / (row["decode_ms"] / 1000.0) if row["decode_ms"] else None
            depth_match = DEPTH_RE.search(line)
            row["depth_buckets"] = (
                {entry.group("name"): int(entry.group("num")) for entry in DEPTH_ENTRY_RE.finditer(depth_match.group("depth"))}
                if depth_match else {}
            )
            mtp.append(row)
        match = BOUNDARY_RE.search(line)
        if match:
            row = match.groupdict()
            row.update(
                event="boundary_snapshot_" + row.pop("event"),
                tokens=int(row["tokens"]),
                block_size=int(row["block_size"]),
                available_boundaries=int(row["available"]),
            )
            row.pop("available", None)
            boundary.append(row)
        store_match = BOUNDARY_STORE_RE.search(line)
        if store_match:
            boundary.append(
                {
                    "event": "boundary_cache_snapshot",
                    "stored_tokens": int(store_match.group("stored")),
                    "total_tokens": int(store_match.group("total")),
                    "intermediate_snapshots": int(store_match.group("intermediate")),
                }
            )
        phase_match = CACHE_PHASE_RE.search(line)
        if phase_match:
            phases: dict[str, dict[str, Any]] = {}
            for phase in CACHE_PHASE_ENTRY_RE.finditer(phase_match.group("phases")):
                phases[phase.group("name")] = {
                    "ms": float(phase.group("ms")),
                    "count": int(phase.group("count")),
                }
            if phases:
                phase_timings.append(phases)
        lower = line.lower()
        if "cache" in lower and any(word in lower for word in ("phase", "restore", "prefix", "boundary")) and not (phase_match or store_match or BOUNDARY_RE.search(line)):
            # Do not store raw log lines: an unrelated request must never leak
            # into the campaign artifact.  The digest is enough to correlate a
            # phase line while keeping the content private.
            boundary.append({"event": "cache_phase_line", "line_sha256": sha256_text(line)})
    return {"start": start, "end": actual_end, "mtp": mtp, "cache_boundary": boundary, "cache_phase_timings": phase_timings}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def usage_metrics(usage: dict[str, Any]) -> dict[str, Any]:
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    cache = next(
        (
            usage.get(key)
            for key in ("cache_tokens", "cached_tokens", "cache_read_input_tokens")
            if usage.get(key) is not None
        ),
        prompt_details.get("cached_tokens"),
    )
    reasoning = next(
        (
            usage.get(key)
            for key in ("reasoning_tokens", "reasoning_output_tokens")
            if usage.get(key) is not None
        ),
        completion_details.get("reasoning_tokens"),
    )
    return {
        "prompt_tokens": _int_or_none(usage.get("prompt_tokens")),
        "cache_tokens": _int_or_none(cache),
        "completion_tokens": _int_or_none(usage.get("completion_tokens")),
        "reasoning_tokens": _int_or_none(reasoning),
        "usage_raw": usage,
    }


def request_stream(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    seed: int,
    timeout: float,
    log_path: Path | None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    log_start = read_log_end(log_path)
    sent = time.monotonic()
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    first: float | None = None
    last: float | None = None
    error: str | None = None
    finish_reason: str | None = None
    done_seen = False
    malformed_sse_events = 0
    stream_error: Any = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done_seen = True
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    malformed_sse_events += 1
                    continue
                if event.get("error") is not None:
                    stream_error = event.get("error")
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage.update(event_usage)
                choices = event.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    reason = choices[0].get("finish_reason")
                    if reason is not None:
                        finish_reason = str(reason)
                delta = choices[0].get("delta") if choices else {}
                if not isinstance(delta, dict):
                    continue
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if text:
                    content_parts.append(str(text))
                if reasoning:
                    reasoning_parts.append(str(reasoning))
                if text or reasoning:
                    now = time.monotonic()
                    first = now if first is None else first
                    last = now
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    received = time.monotonic()
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    output = content + reasoning
    gen_window = (last - first) if first is not None and last is not None else None
    metrics = usage_metrics(usage)
    completion_tokens = metrics["completion_tokens"]
    invalid_reasons: list[str] = []
    if stream_error is not None:
        invalid_reasons.append(
            "event_error=" + json.dumps(stream_error, ensure_ascii=False, sort_keys=True)
        )
    if malformed_sse_events:
        invalid_reasons.append(f"malformed_sse_events={malformed_sse_events}")
    if not done_seen:
        invalid_reasons.append("missing_done")
    if not usage:
        invalid_reasons.append("missing_usage")
    if metrics["prompt_tokens"] is None:
        invalid_reasons.append("missing_prompt_tokens")
    if completion_tokens is None:
        invalid_reasons.append("missing_completion_tokens")
    elif completion_tokens <= 0:
        invalid_reasons.append("no_completion_tokens")
    if first is None:
        invalid_reasons.append("no_output_delta")
    if not output and completion_tokens and completion_tokens > 0:
        invalid_reasons.append("empty_output")
    if error is None and invalid_reasons:
        error = "invalid_stream: " + "; ".join(invalid_reasons)
    metrics.update(
        {
            "ttft_s": (first - sent) if first is not None else None,
            "end_to_end_s": received - sent,
            "generation_window_s": gen_window,
            "output_tokens_per_s": (
                max(metrics["completion_tokens"] - 1, 0) / gen_window
                if metrics["completion_tokens"] is not None and gen_window and gen_window > 0
                else None
            ),
            "output_sha256": sha256_text(output),
            "output": output,
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": finish_reason,
            "stream_done": done_seen,
            "malformed_sse_events": malformed_sse_events,
            "stream_error": stream_error,
            "stream_valid": not invalid_reasons and error is None,
            "invalid_reasons": invalid_reasons,
            "error": error,
            "log": parse_log_slice(log_path, log_start),
        }
    )
    return metrics


def case_messages(case: str, seed: int, context_override: int | None) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    short_kind = case[:-3] if case.endswith("128") else case
    if short_kind in SHORT_PROMPTS:
        prompt = SHORT_PROMPTS[short_kind] + " /no_think"
        return [{"role": "user", "content": prompt}], 128 if case.endswith("128") else 400, {
            "kind": case,
            "regime": short_kind,
        }
    if case.startswith("context16k") or case.startswith("context57k"):
        context_name = "context16k" if case.startswith("context16k") else "context57k"
        suffix = case[len(context_name):].lstrip("-_") or "retrieval"
        if suffix.startswith("continuation-"):
            regime = suffix[len("continuation-"):]
            if regime not in {"count", "code"}:
                raise ValueError(f"unknown continuation regime: {regime}")
            target = context_override or (16_000 if context_name == "context16k" else 57_000)
            prime = build_long_prompt(target, seed, "retrieval")
            return [
                {"role": "user", "content": prime},
                {"role": "assistant", "content": "amber-otter-4821"},
                {"role": "user", "content": SHORT_PROMPTS[regime] + " /no_think"},
            ], 400, {
                "kind": case,
                "regime": regime,
                "target_context_tokens": target,
                "approx_prompt_tokens": approx_tokens(prime),
                "planted_fact": FACT,
                "continuation": True,
            }
        regime = suffix if suffix in {"count", "code", "retrieval"} else "retrieval"
        target = context_override or (16_000 if context_name == "context16k" else 57_000)
        prompt = build_long_prompt(target, seed, regime)
        max_tokens = 32 if regime == "retrieval" else 400
        return [{"role": "user", "content": prompt}], max_tokens, {
            "kind": case,
            "regime": regime,
            "target_context_tokens": target,
            "approx_prompt_tokens": approx_tokens(prompt),
            "planted_fact": FACT if regime == "retrieval" else None,
        }
    if case == "tool128":
        prompt = (
            "Return one compact JSON tool call with keys tool and args. Use tool "
            "name inspect_fixture and argument path synthetic/module_3.py. "
            "Output only JSON. /no_think"
        )
        return [{"role": "user", "content": prompt}], 128, {"kind": case}
    raise ValueError(f"case_messages does not build {case}")


def prompt_metadata(messages: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    prompt_text = "\n".join(str(message.get("content", "")) for message in messages)
    return {
        **meta,
        "prompt_sha256": sha256_text(serialized),
        "prompt_prefix_sha256": sha256_text(serialized[:2048]),
        "prompt_chars": len(prompt_text),
        "approx_prompt_tokens": approx_tokens(prompt_text),
        "approx_chars_per_token": QWEN_APPROX_CHARS_PER_TOKEN,
        "approx_token_estimator": "local Qwen tokenizer.json ByteLevel-BPE audit; chat wrappers excluded",
    }


def run_multiturn(args: argparse.Namespace, output_handle: Any) -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a concise synthetic coding agent. Thinking is disabled.",
        },
        {
            "role": "user",
            "content": "Inspect the generated repository fixture and report the next safe action. /no_think",
        },
    ]
    for turn in range(1, args.turns + 1):
        meta = prompt_metadata(messages, {"kind": "multiturn", "turn": turn, "growth_target_tokens": 750})
        start = time.time()
        metrics = request_stream(
            args.base_url, args.model, messages, 8, args.seed + turn, args.timeout, args.log_path
        )
        row = {
            "schema": 1,
            "campaign": "qwen27b-m5-campaign",
            "arm": args.arm,
            "case": "multiturn",
            "warmup": False,
            "run": turn,
            "seed": args.seed + turn,
            "request_started_unix": start,
            "model": args.model,
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking_enabled": False,
            **meta,
            **metrics,
        }
        output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_handle.flush()
        print(f"multiturn turn={turn} prompt~{meta['approx_prompt_tokens']} tok", flush=True)
        assistant = metrics.get("output") or ""
        messages.append({"role": "assistant", "content": assistant})
        # Append an independent synthetic tool result; do not rewrite any
        # previous message or nonce the prefix.
        messages.append({"role": "user", "content": fake_tool_result(turn) + "\nContinue. /no_think"})


def write_request_row(
    args: argparse.Namespace,
    output_handle: Any,
    case: str,
    run: int,
    warmup: bool,
    messages: list[dict[str, Any]],
    max_tokens: int,
    meta: dict[str, Any],
) -> None:
    started = time.time()
    prompt_info = prompt_metadata(messages, meta)
    metrics = request_stream(
        args.base_url, args.model, messages, max_tokens, args.seed, args.timeout, args.log_path
    )
    row = {
        "schema": 1,
        "campaign": "qwen27b-m5-campaign",
        "arm": args.arm,
        "case": case,
        "run": run,
        "warmup": warmup,
        "seed": args.seed,
        "request_started_unix": started,
        "model": args.model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking_enabled": False,
        **prompt_info,
        **metrics,
    }
    output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_handle.flush()
    status = "FAILED" if metrics.get("error") else "ok"
    print(
        f"{case} {'warmup' if warmup else f'run={run}'} {status} "
        f"ttft={metrics.get('ttft_s')} e2e={metrics.get('end_to_end_s')} "
        f"completion={metrics.get('completion_tokens')}",
        flush=True,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Isolated oMLX URL, normally http://127.0.0.1:8845")
    parser.add_argument("--allow-8843", action="store_true", help="Deliberately permit production port 8843")
    parser.add_argument("--model", default=MODEL, help=argparse.SUPPRESS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Append-only JSONL destination")
    parser.add_argument("--serve-log", help="Optional server log for per-request offsets and MTP extraction")
    parser.add_argument("--arm", default="baseline", help="Experiment arm label recorded in every row")
    parser.add_argument("--seed", type=int, default=3070, help="Explicit deterministic request/prompt seed")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=(
            "count", "code", "count128", "code128", "context16k", "context16k-count", "context16k-code",
            "context16k-retrieval", "context16k-continuation-count", "context16k-continuation-code",
            "context57k", "context57k-count", "context57k-code", "context57k-retrieval",
            "context57k-continuation-count", "context57k-continuation-code", "multiturn", "tool128",
        ),
        default=[
            "count", "code", "count128", "code128", "context16k-count", "context16k-code", "context16k-retrieval",
            "context57k-count", "context57k-code", "context57k-retrieval", "multiturn", "tool128",
        ],
    )
    parser.add_argument("--runs", type=int, default=3, help="Recorded repetitions for one-shot cases")
    parser.add_argument("--warmup", type=int, default=1, help="Discarded warmup repetitions for one-shot cases")
    parser.add_argument("--turns", type=int, default=10, help="Number of append-only multi-turn requests")
    parser.add_argument("--context-target", type=int, help="Override approximate context size for both context cases")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Per-request timeout in seconds")
    parser.add_argument("--validate-only", action="store_true", help="Validate arguments and print plan without HTTP/GPU work")
    args = parser.parse_args(argv)
    try:
        args.base_url = validate_base_url(args.base_url, args.allow_8843)
    except ValueError as exc:
        parser.error(str(exc))
    if args.runs < 1 or args.warmup < 0 or args.turns < 1:
        parser.error("--runs and --turns must be positive; --warmup cannot be negative")
    args.log_path = Path(args.serve_log).expanduser() if args.serve_log else None
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "model": args.model,
                    "arm": args.arm,
                    "seed": args.seed,
                    "cases": args.cases,
                    "output": args.output,
                    "serve_log": str(args.log_path) if args.log_path else None,
                    "gpu_work": False,
                },
                indent=2,
            )
        )
        return 0

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_handle:
        for case in args.cases:
            if case == "multiturn":
                run_multiturn(args, output_handle)
                continue
            messages, max_tokens, meta = case_messages(case, args.seed, args.context_target)
            for run in range(1, args.warmup + args.runs + 1):
                write_request_row(args, output_handle, case, run, run <= args.warmup, messages, max_tokens, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
