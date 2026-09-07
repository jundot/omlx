#!/usr/bin/env python3
"""Compare local server KV formats with cold-cache and endurance trials.

Start serve_affine4_benchmark.py first. The API key is read from OMLX_API_KEY.
Only the selected server's model settings and caches are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import httpx


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plan(models):
    for model in models:
        for mode in ("native16", "tq4", "affine4"):
            yield model, mode, "forward", [4096, 8192, 32768, 65536], 1024
            yield model, mode, "endurance", [8192], 4096
        for mode in ("affine4", "tq4", "native16"):
            yield model, mode, "reverse", [8192, 32768], 1024


def request(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


def validate(result, lengths, generation):
    errors = []
    rows = result.get("results", [])
    if result.get("status") != "completed":
        errors.append(result.get("error") or result.get("status"))
    if len(rows) != len(lengths):
        errors.append(f"Expected {len(lengths)} rows, received {len(rows)}")
    for row, length in zip(rows, lengths):
        if row.get("prompt_tokens") != length + 1:
            errors.append(f"Unexpected prompt length at {length}")
        if row.get("completion_tokens") != generation:
            errors.append(
                f"Early termination at {length}: {row.get('completion_tokens')}"
            )
        if row.get("cached_tokens") != 0:
            errors.append(f"Unexpected prefix reuse at {length}")
        for metric in ("gen_tps", "processing_tps", "peak_memory_bytes"):
            if not math.isfinite(row.get(metric, float("nan"))):
                errors.append(f"Nonfinite {metric} at {length}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--models", nargs="+", default=["Qwen3.6-35B-A3B", "Qwen3.8-27B"]
    )
    args = parser.parse_args()
    key = os.environ["OMLX_API_KEY"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        if not args.resume:
            parser.error("Output exists; use --resume to continue it")
        data = json.loads(args.output.read_text())
    else:
        sources = [
            "omlx/affine4.py",
            "omlx/turboquant_kv.py",
            "omlx/scheduler.py",
            "omlx/admin/benchmark.py",
            "benchmarks/serve_affine4_benchmark.py",
            "benchmarks/bench_affine4_server.py",
        ]
        data = {
            "started_at": datetime.now(UTC).isoformat(),
            "scope": "Local server; identical oQ4e weights; native BF16 KV vs TQ4 vs Affine4",
            "method": {
                "context_profile": "code_python",
                "prompt_alignment": "N+1 prompt tokens, N prefill rows",
                "warmup": "2048 prefill rows and up to eight generated tokens",
                "sampling": "greedy; benchmark-only termination-token suppression",
                "speculation": "MTP, VLM MTP, DFlash and SpecPrefill disabled",
                "quantized_skip_last": True,
                "cold_cache": True,
                "public_upload": False,
                "peak_memory": "Peak active MLX allocations, including resident model; physical footprint is sampled separately",
                "repeat": "8K and 32K repeated in reversed format order",
            },
            "source_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in sources
            },
            "runs": [],
        }
        save(args.output, data)

    with httpx.Client(base_url=args.base_url, timeout=60, follow_redirects=True) as client:
        response = client.get(
            "/admin/auto-login",
            params={"key": key, "redirect": "/admin/dashboard"},
        )
        response.raise_for_status()
        previous_model = None
        for model, mode, phase, lengths, generation in plan(args.models):
            label = f"{model}/{mode}/{phase}"
            stored = next((r for r in data["runs"] if r["label"] == label), None)
            if stored and stored.get("result", {}).get("status") in (
                "completed",
                "error",
                "cancelled",
            ):
                continue
            if not stored:
                if previous_model and previous_model != model:
                    unloaded = client.post(
                        f"/admin/api/models/{quote(previous_model, safe='')}/unload"
                    )
                    if unloaded.status_code != 400:
                        unloaded.raise_for_status()
                settings = {
                    "turboquant_kv_enabled": mode != "native16",
                    "turboquant_kv_scheme": "affine4"
                    if mode == "affine4"
                    else "turboquant",
                    "turboquant_kv_bits": 4,
                    "turboquant_skip_last": True,
                    "mtp_enabled": False,
                    "vlm_mtp_enabled": False,
                    "dflash_enabled": False,
                    "specprefill_enabled": False,
                    "qwen35_ane_prefill_enabled": False,
                    "max_context_window": 131072,
                    "max_tokens": 8192,
                }
                request(
                    client,
                    "PUT",
                    f"/admin/api/models/{quote(model, safe='')}/settings",
                    json=settings,
                )
                for cache in ("ssd-cache", "hot-cache"):
                    request(client, "POST", f"/admin/api/{cache}/clear", json={})
                body = {
                    "model_id": model,
                    "prompt_lengths": lengths,
                    "generation_length": generation,
                    "batch_sizes": [],
                    "warmup_mode": "ane_2048",
                    "align_prompt_to_ane": True,
                    "force_lm_engine": False,
                    "context_profile": "code_python",
                }
                started = request(client, "POST", "/admin/api/bench/start", json=body)
                stored = {
                    "label": label,
                    "model": model,
                    "mode": mode,
                    "phase": phase,
                    "request": body,
                    "settings": settings,
                    "bench_id": started["bench_id"],
                }
                data["runs"].append(stored)
                save(args.output, data)
                print(f"START {label} {stored['bench_id']}", flush=True)
            previous_model = model
            observed = 0
            last_notice = time.monotonic()
            while True:
                result = request(
                    client, "GET", f"/admin/api/bench/{stored['bench_id']}/results"
                )
                stored["result"] = result
                save(args.output, data)
                for row in result.get("results", [])[observed:]:
                    print(
                        json.dumps(
                            {
                                "label": label,
                                **{
                                    key: row.get(key)
                                    for key in (
                                        "pp",
                                        "completion_tokens",
                                        "processing_tps",
                                        "gen_tps",
                                        "peak_memory_bytes",
                                        "cached_tokens",
                                    )
                                },
                            }
                        ),
                        flush=True,
                    )
                observed = len(result.get("results", []))
                if result.get("status") != "running":
                    stored["validation_errors"] = validate(result, lengths, generation)
                    save(args.output, data)
                    print(
                        f"DONE {label}: {stored['validation_errors'] or 'valid'}",
                        flush=True,
                    )
                    break
                if time.monotonic() - last_notice >= 30:
                    print(
                        f"RUNNING {label}: {observed}/{len(lengths)} rows", flush=True
                    )
                    last_notice = time.monotonic()
                time.sleep(5)
    data["finished_at"] = datetime.now(UTC).isoformat()
    save(args.output, data)
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
