#!/usr/bin/env python3
"""Concurrent TP2 decode and mixed-prefill acceptance probe.

The client keeps one record per stream while polling the rank-zero marker, so
the result checks both actual throughput and dashboard request separation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import statistics
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE = "http://127.0.0.1:8000"
MODEL = "DeepSeek-V4-Flash-0731-MXFP4-MLX"
RUNTIME_GLOB = "~/.omlx/cluster/runtime/*rank-0.json"


@dataclass
class StreamResult:
    label: str
    started: float = 0.0
    first_token: float | None = None
    finished: float = 0.0
    event_times: list[float] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        intervals = [
            right - left for left, right in zip(self.event_times, self.event_times[1:])
        ]
        return {
            "label": self.label,
            "prompt_tokens": self.usage.get("prompt_tokens"),
            "cached_tokens": (self.usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0
            ),
            "completion_tokens": self.usage.get("completion_tokens"),
            "prefill_tps": self.usage.get("prompt_tokens_per_second"),
            "decode_tps": self.usage.get("generation_tokens_per_second"),
            "wall_seconds": self.finished - self.started,
            "ttft_seconds": (
                self.first_token - self.started
                if self.first_token is not None
                else None
            ),
            "itl_p50_ms": (statistics.median(intervals) * 1000 if intervals else None),
            "itl_p95_ms": _percentile(intervals, 0.95),
            "itl_max_ms": max(intervals) * 1000 if intervals else None,
            "output_sha256": hashlib.sha256(self.output.encode()).hexdigest(),
            "error": self.error,
        }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index] * 1000


def _latest_marker() -> dict[str, Any]:
    paths = sorted(
        glob.glob(os.path.expanduser(RUNTIME_GLOB)),
        key=os.path.getmtime,
    )
    if not paths:
        return {}
    try:
        return json.loads(Path(paths[-1]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _stream(label: str, prompt: str, max_tokens: int, timeout: int) -> StreamResult:
    result = StreamResult(label=label, started=time.perf_counter())
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    pieces: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    result.usage = event["usage"]
                for choice in event.get("choices") or ():
                    text = choice.get("text")
                    if not text:
                        continue
                    now = time.perf_counter()
                    if result.first_token is None:
                        result.first_token = now
                    result.event_times.append(now)
                    pieces.append(text)
    except Exception as exc:  # benchmark reports the failure in-band
        result.error = f"{type(exc).__name__}: {exc}"
    result.output = "".join(pieces)
    result.finished = time.perf_counter()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--prefill-repetitions", type=int, default=0)
    parser.add_argument("--prefill-max-tokens", type=int, default=16)
    parser.add_argument("--word", default="tensor")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    # Keep decoder and long-prefill prefixes disjoint so the mixed cold gate
    # cannot inherit the decoder's just-completed 64-token cache entry.
    decoder_prompt = ("decode " * 64) + args.word
    barrier = threading.Barrier(args.batch)
    snapshots: list[dict[str, Any]] = []
    results: list[StreamResult] = []

    def decoder(index: int) -> StreamResult:
        barrier.wait(timeout=30)
        return _stream(
            f"decode-{index}",
            decoder_prompt,
            args.decode_tokens,
            args.timeout,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.batch + (1 if args.prefill_repetitions else 0)
    ) as pool:
        decoder_futures = [pool.submit(decoder, index) for index in range(args.batch)]
        prefill_future = None
        deadline = time.monotonic() + 45
        while args.prefill_repetitions and time.monotonic() < deadline:
            marker = _latest_marker()
            metrics = marker.get("metrics") or {}
            active = metrics.get("active_request_metrics") or []
            snapshots.append(metrics)
            decoding = sum(
                int(item.get("completion_tokens", 0) or 0) > 0 for item in active
            )
            if decoding >= args.batch:
                long_prompt = (args.word + " ") * args.prefill_repetitions
                prefill_future = pool.submit(
                    _stream,
                    "prefill",
                    long_prompt,
                    args.prefill_max_tokens,
                    args.timeout,
                )
                break
            time.sleep(0.1)

        futures = list(decoder_futures)
        if prefill_future is not None:
            futures.append(prefill_future)
        while any(not future.done() for future in futures):
            metrics = _latest_marker().get("metrics") or {}
            snapshots.append(metrics)
            time.sleep(0.1)
        results = [future.result() for future in futures]

    decoders = [result for result in results if result.label.startswith("decode-")]
    first_decode = min(
        (result.first_token for result in decoders if result.first_token is not None),
        default=None,
    )
    last_decode = max((result.finished for result in decoders), default=0.0)
    total_decode_tokens = sum(
        int(result.usage.get("completion_tokens", 0) or 0) for result in decoders
    )
    decode_window = last_decode - first_decode if first_decode is not None else None
    active_rows = [
        item
        for metrics in snapshots
        for item in (metrics.get("active_request_metrics") or [])
    ]
    marker = _latest_marker()
    metrics = marker.get("metrics") or {}
    output_hashes = {
        hashlib.sha256(result.output.encode()).hexdigest() for result in decoders
    }
    report = {
        "batch": args.batch,
        "decode_tokens_requested": args.decode_tokens,
        "prefill_repetitions": args.prefill_repetitions,
        "streams": [result.summary() for result in results],
        "decode_output_hashes_equal": len(output_hashes) == 1,
        "client_aggregate_decode_tps": (
            total_decode_tokens / decode_window
            if decode_window is not None and decode_window > 0
            else None
        ),
        "max_dashboard_active_requests": max(
            (int(item.get("active_requests", 0) or 0) for item in snapshots),
            default=0,
        ),
        "dashboard_request_ids_seen": sorted(
            {
                int(item["request_id"])
                for item in active_rows
                if item.get("request_id") is not None
            }
        ),
        "dashboard_aggregate_decode_tps": metrics.get("aggregate_decode_tps"),
        "dashboard_average_request_decode_tps": metrics.get(
            "average_request_decode_tps"
        ),
        "dashboard_last_batch": (metrics.get("pipeline") or {}).get("last_batch"),
        "requests_failed_total": metrics.get("requests_failed"),
        "requests_cancelled_total": metrics.get("requests_cancelled"),
        "deployment_id": marker.get("deployment_id"),
        "plan_hash": marker.get("plan_hash"),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
