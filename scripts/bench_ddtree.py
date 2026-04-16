#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark AR vs DFlash vs DDTree on a target model + DFlash drafter.

Uses dflash-mlx and ddtree-mlx directly (no omlx engine) so the measurements
reflect the underlying algorithms rather than server overhead.

Usage:
    uv run python scripts/bench_ddtree.py \\
        --model mlx-community/Qwen3.5-4B-4bit \\
        --draft z-lab/Qwen3.5-4B-DFlash \\
        --max-tokens 256 \\
        --budgets 2 4 6 \\
        --prompts 3
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import mlx.core as mx


CODE_PROMPTS = [
    "Write a Python function `mergesort` that sorts a list of integers in-place. Include a brief test harness.",
    "Implement a minimal HTTP GET in Go using net/http. Show the full file including imports and error handling.",
    "Write a Rust function that computes the fibonacci numbers up to N using iteration, with a doctest.",
]

PROSE_PROMPTS = [
    "Write a short paragraph describing a foggy morning at a coastal lighthouse. No dialogue.",
    "Explain to a beginner, in two short paragraphs, how a CPU pipeline stall happens.",
    "Compose a three-sentence summary of the plot of Moby Dick.",
]


def _tokenize(tokenizer, text: str) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def _run_ar(
    target_model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_new: int,
    stop_ids: list[int],
) -> dict:
    from dflash_mlx.generate import decode_token

    start = time.perf_counter()
    tokens: list[int] = []
    past = None
    cur = mx.array(prompt_tokens)[None, :]
    for _ in range(max_new):
        logits, past = decode_token(target_model, cur, past)
        tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        tokens.append(tok)
        if tok in stop_ids:
            break
        cur = mx.array([[tok]])
    mx.eval(cur)
    elapsed = time.perf_counter() - start
    return {
        "tokens": tokens,
        "elapsed_s": elapsed,
        "tok_s": len(tokens) / elapsed if elapsed else 0.0,
    }


def _run_dflash(
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_new: int,
    stop_ids: list[int],
) -> dict:
    from dflash_mlx.runtime import generate_dflash_once

    start = time.perf_counter()
    summary = generate_dflash_once(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt="",
        max_new_tokens=max_new,
        stop_token_ids=stop_ids,
        prompt_tokens_override=prompt_tokens,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - start
    tokens = summary.get("generated_token_ids", [])
    return {
        "tokens": tokens,
        "elapsed_s": elapsed,
        "tok_s": len(tokens) / elapsed if elapsed else 0.0,
        "acceptance": float(summary.get("acceptance_ratio", 0.0)),
    }


def _run_ddtree(
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_new: int,
    stop_ids: list[int],
    budget: int,
) -> dict:
    from ddtree_mlx.runtime import generate_ddtree_once

    start = time.perf_counter()
    summary = generate_ddtree_once(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new,
        tree_budget=budget,
        stop_token_ids=stop_ids,
    )
    elapsed = time.perf_counter() - start
    tokens = summary.get("generated_token_ids", [])
    return {
        "tokens": tokens,
        "elapsed_s": elapsed,
        "tok_s": float(summary.get("tokens_per_second", len(tokens) / elapsed if elapsed else 0.0)),
        "avg_acceptance": float(summary.get("avg_acceptance", 0.0)),
        "fast_path_ratio": float(summary.get("fast_path_ratio", 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-4bit")
    parser.add_argument("--draft", default=None, help="DFlash drafter ref (None=registry lookup)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--prompts", type=int, default=2, help="Prompts per category")
    parser.add_argument("--categories", nargs="+", default=["code", "prose"], choices=["code", "prose"])
    args = parser.parse_args()

    from dflash_mlx.generate import get_stop_token_ids, load_runtime_components

    print(f"Loading {args.model} ...")
    target_model, tokenizer, draft_model, draft_ref = load_runtime_components(
        model_ref=args.model, draft_ref=args.draft,
    )
    stop_ids = get_stop_token_ids(tokenizer)
    print(f"Loaded target={args.model} draft={draft_ref}\n")

    prompt_sets: dict[str, list[str]] = {}
    if "code" in args.categories:
        prompt_sets["code"] = CODE_PROMPTS[: args.prompts]
    if "prose" in args.categories:
        prompt_sets["prose"] = PROSE_PROMPTS[: args.prompts]

    # Warm up with one small generation so MLX compiles its kernels.
    print("Warming up...")
    warm_tokens = _tokenize(tokenizer, "Hi")
    _run_ar(target_model, tokenizer, warm_tokens, 8, stop_ids)
    _run_dflash(target_model, draft_model, tokenizer, warm_tokens, 8, stop_ids)
    _run_ddtree(target_model, draft_model, tokenizer, warm_tokens, 8, stop_ids, budget=4)
    print("Warmup done.\n")

    results: dict[str, dict[str, list[float]]] = {}
    for cat, prompts in prompt_sets.items():
        cat_results: dict[str, list[float]] = {
            "ar_tps": [], "dflash_tps": [], "dflash_acc": [],
        }
        for b in args.budgets:
            cat_results[f"ddtree_b{b}_tps"] = []
            cat_results[f"ddtree_b{b}_acc"] = []
            cat_results[f"ddtree_b{b}_fast"] = []

        print(f"=== category: {cat} ({len(prompts)} prompts, max_new={args.max_tokens}) ===")
        for i, p in enumerate(prompts):
            ptoks = _tokenize(tokenizer, p)
            print(f"  [{i + 1}/{len(prompts)}] prompt={len(ptoks)} tok")

            ar = _run_ar(target_model, tokenizer, ptoks, args.max_tokens, stop_ids)
            cat_results["ar_tps"].append(ar["tok_s"])
            print(f"    AR        : {ar['tok_s']:6.1f} tok/s  ({len(ar['tokens'])} tok / {ar['elapsed_s']:.2f}s)")

            df = _run_dflash(target_model, draft_model, tokenizer, ptoks, args.max_tokens, stop_ids)
            cat_results["dflash_tps"].append(df["tok_s"])
            cat_results["dflash_acc"].append(df["acceptance"])
            print(f"    DFlash    : {df['tok_s']:6.1f} tok/s  (accept={df['acceptance']:.0%})")

            for b in args.budgets:
                dt = _run_ddtree(
                    target_model, draft_model, tokenizer, ptoks,
                    args.max_tokens, stop_ids, budget=b,
                )
                cat_results[f"ddtree_b{b}_tps"].append(dt["tok_s"])
                cat_results[f"ddtree_b{b}_acc"].append(dt["avg_acceptance"])
                cat_results[f"ddtree_b{b}_fast"].append(dt["fast_path_ratio"])
                print(
                    f"    DDTree b={b}: {dt['tok_s']:6.1f} tok/s  "
                    f"(avg_accept={dt['avg_acceptance']:.2f}, fast={dt['fast_path_ratio']:.0%})"
                )
        results[cat] = cat_results

    print("\n=== summary (mean across prompts) ===")
    for cat, m in results.items():
        ar_mean = statistics.mean(m["ar_tps"]) if m["ar_tps"] else 0.0
        df_mean = statistics.mean(m["dflash_tps"]) if m["dflash_tps"] else 0.0
        df_acc = statistics.mean(m["dflash_acc"]) if m["dflash_acc"] else 0.0
        print(f"\n  [{cat}]")
        print(f"    AR        : {ar_mean:6.1f} tok/s")
        print(
            f"    DFlash    : {df_mean:6.1f} tok/s "
            f"({df_mean / ar_mean:.2f}x AR, accept={df_acc:.0%})"
        )
        for b in args.budgets:
            tps_key = f"ddtree_b{b}_tps"
            acc_key = f"ddtree_b{b}_acc"
            fast_key = f"ddtree_b{b}_fast"
            tps = statistics.mean(m[tps_key]) if m[tps_key] else 0.0
            acc = statistics.mean(m[acc_key]) if m[acc_key] else 0.0
            fast = statistics.mean(m[fast_key]) if m[fast_key] else 0.0
            vs_ar = tps / ar_mean if ar_mean else 0.0
            vs_df = tps / df_mean if df_mean else 0.0
            print(
                f"    DDTree b={b}: {tps:6.1f} tok/s "
                f"({vs_ar:.2f}x AR, {vs_df:.2f}x DFlash, "
                f"accept={acc:.2f}, fast={fast:.0%})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
