#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark tiled decode attention vs monolithic on a real MLX model.

Measures memory and decode tok/s at increasing context lengths.
Validates the reviewer's claim that tiled + online softmax keeps
throughput flat from short to long context.
"""
from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx


def _cos_sim(a, b):
    af = a.astype(mx.float32).flatten()
    bf = b.astype(mx.float32).flatten()
    num = (af * bf).sum()
    den = mx.sqrt((af * af).sum()) * mx.sqrt((bf * bf).sum()) + 1e-9
    return float((num / den).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--contexts", default="1024,4096,8192,16384",
                    help="comma-separated context lengths")
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--tile-size", type=int, default=4096)
    ap.add_argument("--pq-bits", type=int, default=3)
    args = ap.parse_args()

    try:
        from mlx_lm import load
        from mlx_lm.models import cache as mlx_cache_mod
    except ImportError:
        print("mlx_lm not available", file=sys.stderr)
        sys.exit(1)
    from omlx.patches.planarquant_cache import (
        enable_planarquant_cache,
        disable_planarquant_cache,
    )
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
    from omlx.cache.planarquant.kv_cache import PlanarQuantKVCache

    apply_turboquant_attention_patch()
    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    base_prompt = (
        "The history of computing spans centuries, from the abacus to quantum "
        "computers. Each era brought revolutionary changes. "
    )
    all_tokens = tokenizer.encode(base_prompt)

    contexts = [int(x) for x in args.contexts.split(",")]
    print()
    print("=" * 90)
    print(f"{'T':>8}  {'mode':>16}  {'prefill t':>10}  {'decode tok/s':>14}  "
          f"{'cache MB':>10}  {'peak MB':>10}  {'cos_sim':>10}")
    print("-" * 90)

    for t_target in contexts:
        reps = t_target // len(all_tokens) + 1
        toks = (all_tokens * reps)[:t_target]
        tokens = mx.array(toks)[None, :]

        def _run(mode: str, baseline_last_logits=None, baseline_decode_out=None):
            if mode == "fp16":
                disable_planarquant_cache()
            else:
                enable_planarquant_cache(bits=args.pq_bits, quantize_v=True)
            cache = mlx_cache_mod.make_prompt_cache(model)

            mx.eval(tokens)
            t0 = time.perf_counter()
            logits = model(tokens, cache=cache)
            mx.eval(logits)
            prefill_s = time.perf_counter() - t0

            # Cache size
            cache_bytes = 0
            for c in cache:
                if hasattr(c, "nbytes"):
                    try:
                        nb = c.nbytes
                        if isinstance(nb, int):
                            cache_bytes += nb
                    except Exception:
                        pass
            cache_mb = cache_bytes / 1e6

            # Enable tiled path for the memory-pressure variant
            if mode == "pq3_tiled":
                for c in cache:
                    if isinstance(c, PlanarQuantKVCache):
                        c.enable_memory_pressure_mode(tile_size=args.tile_size)

            # Warm up
            last = mx.argmax(logits[0, -1, :])[None, None]
            for _ in range(2):
                logits = model(last, cache=cache)
                mx.eval(logits)
                last = mx.argmax(logits[0, -1, :])[None, None]

            # Time decode
            t0 = time.perf_counter()
            for _ in range(args.decode_steps):
                logits = model(last, cache=cache)
                mx.eval(logits)
                last = mx.argmax(logits[0, -1, :])[None, None]
            decode_s = time.perf_counter() - t0
            tps = args.decode_steps / decode_s

            # cos_sim vs baseline logits
            last_logits = logits[0, -1, :]
            sim = 1.0
            if baseline_last_logits is not None:
                sim = _cos_sim(last_logits, baseline_last_logits)

            # Peak memory (rough — MLX active)
            peak_mb = float(mx.get_active_memory()) / 1e6

            print(f"{t_target:>8}  {mode:>16}  {prefill_s*1000:>9.1f}ms  "
                  f"{tps:>14.2f}  {cache_mb:>10.2f}  {peak_mb:>10.1f}  {sim:>10.6f}")
            return last_logits

        # FP16 baseline
        fp16_logits = _run("fp16")
        # PQ3 monolithic
        _run("pq3_monolithic", baseline_last_logits=fp16_logits)
        # PQ3 tiled
        _run("pq3_tiled", baseline_last_logits=fp16_logits)
        print()
        disable_planarquant_cache()

    print("=" * 90)


if __name__ == "__main__":
    main()
