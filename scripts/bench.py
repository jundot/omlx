#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone benchmark script using omlx's native benchmark machinery.

Runs the same single-request and continuous-batching tests as the UI benchmark,
but directly in Python — no server or HTTP needed.  Pass multiple model paths
to run each in sequence and print a side-by-side comparison table.

Usage
-----
    # Single model
    ~/.venv/bin/python scripts/bench.py ~/models/Bonsai-27B

    # Compare two variants
    ~/.venv/bin/python scripts/bench.py ~/models/bonsai-27b ~/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-27B-mlx-2bit/snapshots/70f75f3ad081ab840a42f3304c02c27e7f89bfb7

    # With batch tests
    ~/.venv/bin/python scripts/bench.py model-a model-b --pp 1024 4096 --batch 2 4

Metrics (single-request)
--------------------------
  pp       prompt tokens
  ttft     time-to-first-token (ms)
  tpot     time-per-output-token (ms)
  gen_tps  decode tokens/sec
  pp_tps   prefill tokens/sec
  mem      peak GPU memory

Metrics (batch)
----------------
  bs       batch size
  pp_tps   aggregate prefill tokens/sec
  tg_tps   aggregate decode tokens/sec
  ttft     average time-to-first-token (ms)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="omlx native benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("models", nargs="+", help="Path(s) to model directory")
    p.add_argument(
        "--pp",
        metavar="N",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192],
        help="Prompt token lengths to test (default: 1024 4096 8192)",
    )
    p.add_argument(
        "--gen",
        metavar="N",
        type=int,
        default=128,
        help="Tokens to generate per request (default: 128)",
    )
    p.add_argument(
        "--batch",
        metavar="N",
        type=int,
        nargs="+",
        default=[],
        help="Batch sizes for continuous-batching tests (default: none)",
    )
    p.add_argument(
        "--warmup",
        metavar="N",
        type=int,
        default=1,
        help="Warmup runs before timing (default: 1)",
    )
    return p.parse_args()


# ── formatting helpers ────────────────────────────────────────────────────────

def _fmt_mem(peak_bytes: int) -> str:
    if peak_bytes <= 0:
        return "—"
    return f"{peak_bytes / 1e9:.1f}G"


def _short_name(path: str) -> str:
    """Return a short display label for a model path."""
    p = Path(path)
    name = p.name
    # HF snapshot paths: …/models--org--name/snapshots/<hash> → org/name
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "snapshots" and i >= 1:
            repo = parts[i - 1]  # models--org--name
            label = repo.removeprefix("models--").replace("--", "/")
            return label
    return name


# ── per-model benchmark runner ────────────────────────────────────────────────

async def _bench_model(
    model_path: str,
    pp_lengths: list[int],
    gen_tokens: int,
    batch_sizes: list[int],
    warmup: int,
) -> tuple[list[dict], list[dict]]:
    """Load one model, run all tests, unload.  Returns (single_results, batch_results)."""
    from omlx.admin.benchmark import (
        _generate_prompt,
        _run_batch_test,
        _run_single_test,
    )
    from omlx.engine.vlm import VLMBatchedEngine

    print(f"\nLoading {model_path} …")
    t0 = time.perf_counter()
    engine = VLMBatchedEngine(model_path)
    await engine.start()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    tokenizer = engine.tokenizer
    prompts: dict[int, str] = {pp: _generate_prompt(tokenizer, pp) for pp in sorted(set(pp_lengths))}

    if warmup > 0 and pp_lengths:
        warmup_pp = min(pp_lengths)
        print(f"Warming up ({warmup}× pp={warmup_pp}) …")
        for _ in range(warmup):
            await _run_single_test(engine, prompts[warmup_pp], gen_tokens, warmup_pp)

    single_results: list[dict] = []
    for pp in sorted(pp_lengths):
        print(f"  pp={pp} gen={gen_tokens} …", end="", flush=True)
        r = await _run_single_test(engine, prompts[pp], gen_tokens, pp)
        single_results.append(r)
        print(f"  ttft={r['ttft_ms']:.0f}ms  {r['gen_tps']:.1f} t/s")

    batch_results: list[dict] = []
    batch_pp = sorted(pp_lengths)[0] if pp_lengths else 1024
    for bs in sorted(batch_sizes):
        batch_prompts = [_generate_prompt(tokenizer, batch_pp) for _ in range(bs)]
        print(f"  batch={bs} pp={batch_pp} gen={gen_tokens} …", end="", flush=True)
        r = await _run_batch_test(engine, batch_prompts, batch_pp, gen_tokens, bs)
        batch_results.append(r)
        print(f"  pp={r['pp_tps']:.0f}/s  tg={r['tg_tps']:.0f}/s")

    await engine.stop()
    return single_results, batch_results


# ── table printers ────────────────────────────────────────────────────────────

def _print_single_comparison(
    labels: list[str],
    all_results: list[list[dict]],
    pp_lengths: list[int],
) -> None:
    """Print a side-by-side comparison table for single-request results."""
    # Column widths: fixed per metric, repeated per model
    col = 9  # width of one model's metric block
    n = len(labels)

    # Header: model names spanning their columns
    metrics = ["ttft", "gen_tps", "pp_tps", "mem"]
    block_w = col * len(metrics) + len(metrics) - 1  # e.g. 4*9+3 = 39

    print()
    print("  Single-request")
    # Model name header row
    name_row = f"  {'pp':>6}  "
    for label in labels:
        # Truncate/pad label to block_w
        display = label[:block_w] if len(label) > block_w else label
        name_row += f"{display:^{block_w}}  "
    print(name_row.rstrip())

    # Sub-header: metric names per model
    sub_row = f"  {'':>6}  "
    for _ in labels:
        sub_row += f"{'ttft':>{col}} {'gen_tps':>{col}} {'pp_tps':>{col}} {'mem':>{col}}  "
    print(sub_row.rstrip())

    sep = "─" * (8 + (block_w + 2) * n)
    print("  " + sep)

    # Data rows
    for pp in sorted(pp_lengths):
        row = f"  {pp:>6}  "
        for model_results in all_results:
            r = next((x for x in model_results if x["prompt_tokens"] == pp), None)
            if r is None:
                row += f"{'—':>{col}} {'—':>{col}} {'—':>{col}} {'—':>{col}}  "
            else:
                row += (
                    f"{r['ttft_ms']:>{col-2}.0f}ms "
                    f"{r['gen_tps']:>{col-2}.1f}/s "
                    f"{r['processing_tps']:>{col-2}.0f}/s "
                    f"{_fmt_mem(r['peak_memory_bytes']):>{col}}  "
                )
        print(row.rstrip())

    print("  " + sep)


def _print_batch_comparison(
    labels: list[str],
    all_results: list[list[dict]],
    batch_sizes: list[int],
) -> None:
    """Print a side-by-side comparison table for batch results."""
    col = 9
    n = len(labels)
    metrics = ["pp_tps", "tg_tps", "ttft"]
    block_w = col * len(metrics) + len(metrics) - 1

    print()
    print("  Continuous-batching")
    name_row = f"  {'bs':>4}  "
    for label in labels:
        display = label[:block_w] if len(label) > block_w else label
        name_row += f"{display:^{block_w}}  "
    print(name_row.rstrip())

    sub_row = f"  {'':>4}  "
    for _ in labels:
        sub_row += f"{'pp_tps':>{col}} {'tg_tps':>{col}} {'ttft':>{col}}  "
    print(sub_row.rstrip())

    sep = "─" * (6 + (block_w + 2) * n)
    print("  " + sep)

    for bs in sorted(batch_sizes):
        row = f"  {bs:>4}  "
        for model_results in all_results:
            r = next((x for x in model_results if x["batch_size"] == bs), None)
            if r is None:
                row += f"{'—':>{col}} {'—':>{col}} {'—':>{col}}  "
            else:
                row += (
                    f"{r['pp_tps']:>{col-2}.0f}/s "
                    f"{r['tg_tps']:>{col-2}.0f}/s "
                    f"{r['avg_ttft_ms']:>{col-2}.0f}ms  "
                )
        print(row.rstrip())

    print("  " + sep)


# ── main ──────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    model_paths = [str(Path(m).expanduser().resolve()) for m in args.models]
    labels = [_short_name(p) for p in model_paths]
    pp_lengths = sorted(set(args.pp))

    all_single: list[list[dict]] = []
    all_batch: list[list[dict]] = []

    for path in model_paths:
        single, batch = await _bench_model(
            path, pp_lengths, args.gen, sorted(args.batch), args.warmup
        )
        all_single.append(single)
        all_batch.append(batch)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  gen_tokens={args.gen}")

    if len(model_paths) == 1:
        # Single model: original compact table
        _print_single_comparison(labels, all_single, pp_lengths)
        if all_batch[0]:
            _print_batch_comparison(labels, all_batch, sorted(args.batch))
    else:
        # Multiple models: side-by-side
        _print_single_comparison(labels, all_single, pp_lengths)
        if any(all_batch):
            _print_batch_comparison(labels, all_batch, sorted(args.batch))

    print()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
