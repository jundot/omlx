#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the DeepSeek V4 Flash M3 Ultra cold-prefill benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO_ROOT / "omlx/admin/bench_corpora/code_python.txt"
DEFAULT_MODEL = "DeepSeek-V4-Flash-0731"
DEFAULT_MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
PROMPT_PREFIX = (
    "Read the following fixed code corpus. Reply with exactly the word "
    "READY and no other text.\n\n"
)
PROMPT_BODY_CHARS = 71_000
EXPECTED_PROMPT_CHARS = 71_092
EXPECTED_PROMPT_SHA256 = (
    "a1465f4b5ee68dbd173c138dd65718bd06957439b66e99069e91f50364bf81f1"
)
EXPECTED_PROMPT_TOKENS = 17_219
EXPECTED_OUTPUT = "READY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--revision")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Prior run record to validate and compare against",
    )
    parser.add_argument("--verify-prompt", action="store_true")
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not args.verify_prompt and (not args.revision or args.output is None):
        parser.error("--revision and --output are required for a benchmark run")
    return args


def make_prompt(corpus_path: Path) -> tuple[str, dict[str, Any]]:
    corpus = corpus_path.read_text(encoding="utf-8")
    if not corpus:
        raise RuntimeError(f"benchmark corpus is empty: {corpus_path}")
    repeats = PROMPT_BODY_CHARS // len(corpus) + 1
    prompt = PROMPT_PREFIX + (corpus * repeats)[:PROMPT_BODY_CHARS]
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if len(prompt) != EXPECTED_PROMPT_CHARS or digest != EXPECTED_PROMPT_SHA256:
        raise RuntimeError(
            "benchmark prompt does not match the recorded workload: "
            f"chars={len(prompt)}, sha256={digest}"
        )
    resolved = corpus_path.resolve()
    try:
        corpus_reference = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        corpus_reference = str(resolved)
    return prompt, {
        "corpus": corpus_reference,
        "body_chars": PROMPT_BODY_CHARS,
        "prompt_chars": len(prompt),
        "prompt_sha256": digest,
        "expected_api_prompt_tokens": EXPECTED_PROMPT_TOKENS,
        "expected_output": EXPECTED_OUTPUT,
    }


def request_once(
    endpoint: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    usage: dict[str, Any] = {}
    content_parts: list[str] = []
    finish_reason = None
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices:
                    choice = choices[0]
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        content_parts.append(str(content))
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"server returned HTTP {error.code}: {detail}") from error

    content = "".join(content_parts)
    required = (
        "prompt_tokens",
        "completion_tokens",
        "prompt_eval_duration",
        "prompt_tokens_per_second",
    )
    missing = [field for field in required if usage.get(field) is None]
    if missing:
        raise RuntimeError(f"response is missing timing fields: {missing}")
    if int(usage["prompt_tokens"]) != EXPECTED_PROMPT_TOKENS:
        raise RuntimeError(
            f"expected {EXPECTED_PROMPT_TOKENS} API prompt tokens, "
            f"got {usage['prompt_tokens']}"
        )
    cached_tokens = int(
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    )
    if cached_tokens:
        raise RuntimeError(f"request was not cold: cached_tokens={cached_tokens}")
    if content != EXPECTED_OUTPUT:
        raise RuntimeError(f"expected {EXPECTED_OUTPUT!r}, got {content!r}")
    return {
        "wall_time_seconds": time.perf_counter() - started,
        "usage": usage,
        "finish_reason": finish_reason,
        "content": content,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "correct": True,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    prefill = [float(run["usage"]["prompt_eval_duration"]) for run in runs]
    prompt_tps = [float(run["usage"]["prompt_tokens_per_second"]) for run in runs]
    return {
        "prefill_seconds_median": statistics.median(prefill),
        "prefill_seconds_mean": statistics.mean(prefill),
        "prefill_seconds_stdev": statistics.stdev(prefill) if len(prefill) > 1 else 0,
        "prompt_tps_median": statistics.median(prompt_tps),
        "prompt_tps_mean": statistics.mean(prompt_tps),
        "all_cold": True,
        "all_correct": True,
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    for field in ("model", "host", "prompt", "request"):
        if baseline[field] != candidate[field]:
            raise RuntimeError(f"baseline and candidate differ in {field}")
    if baseline["source_revision"] == candidate["source_revision"]:
        raise RuntimeError("baseline and candidate revisions must differ")
    baseline_prefill = float(baseline["summary"]["prefill_seconds_median"])
    candidate_prefill = float(candidate["summary"]["prefill_seconds_median"])
    baseline_tps = float(baseline["summary"]["prompt_tps_median"])
    candidate_tps = float(candidate["summary"]["prompt_tps_median"])
    return {
        "baseline_revision": baseline["source_revision"],
        "candidate_revision": candidate["source_revision"],
        "prefill_seconds_saved": baseline_prefill - candidate_prefill,
        "prefill_time_reduction_percent": (
            (baseline_prefill - candidate_prefill) / baseline_prefill * 100
        ),
        "prompt_throughput_gain_percent": (candidate_tps / baseline_tps - 1) * 100,
        "speedup": candidate_tps / baseline_tps,
    }


def main() -> int:
    args = parse_args()
    prompt, prompt_metadata = make_prompt(args.corpus)
    if args.verify_prompt:
        print(json.dumps(prompt_metadata, indent=2))
        return 0

    runs = []
    for repetition in range(1, args.repetitions + 1):
        run = request_once(args.endpoint, args.model, prompt, args.timeout_seconds)
        run["repetition"] = repetition
        runs.append(run)
        print(
            json.dumps(
                {
                    "repetition": repetition,
                    "prefill_seconds": run["usage"]["prompt_eval_duration"],
                    "prompt_tps": run["usage"]["prompt_tokens_per_second"],
                    "wall_seconds": run["wall_time_seconds"],
                }
            ),
            flush=True,
        )

    record = {
        "schema_version": 2,
        "recorded_at": datetime.now(UTC).isoformat(),
        "source_revision": args.revision,
        "endpoint": args.endpoint,
        "model": {"id": args.model, "revision": args.model_revision},
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "prompt": prompt_metadata,
        "request": {
            "repetitions": args.repetitions,
            "temperature": 0,
            "max_tokens": 8,
            "stream": True,
            "cache_requirement": "cached_tokens=0",
        },
        "runs": runs,
        "summary": summarize(runs),
    }
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if baseline.get("schema_version") != 2:
            raise RuntimeError(f"unsupported baseline record: {args.baseline}")
        record["comparison_to_baseline"] = compare(baseline, record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["summary"], indent=2))
    if "comparison_to_baseline" in record:
        print(json.dumps(record["comparison_to_baseline"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
