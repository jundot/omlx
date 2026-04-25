#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PlanarQuant3 KV cache benchmark.

Measures quality (logit cosine similarity vs FP16), latency (forward-pass
time for prefill + single decode step), and memory (cache.nbytes ratio)
on a real MLX model.

Usage:
    uv run python scripts/bench_planarquant.py \
        --model mlx-community/Qwen3.5-4B-MLX-4bit \
        --prompt "The capital of France is" \
        --decode-steps 16
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx


def _ensure_imports():
    try:
        from mlx_lm import load  # noqa: F401
    except ImportError:
        print("mlx_lm not available — run `uv sync` first.", file=sys.stderr)
        sys.exit(1)


def bench_config(
    label: str,
    model,
    tokenizer,
    prompt: str,
    decode_steps: int,
    enable_pq: bool,
    pq_bits: int,
) -> dict:
    from mlx_lm.models import cache as mlx_cache_mod

    from omlx.patches.planarquant_cache import (
        disable_planarquant_cache,
        enable_planarquant_cache,
    )
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()

    if enable_pq:
        enable_planarquant_cache(pq_bits)
    else:
        disable_planarquant_cache()

    tokens = mx.array(tokenizer.encode(prompt))[None, :]  # (1, L)
    prompt_len = tokens.shape[1]

    # Warm up MLX kernel compilation before timing.
    warm_cache = mlx_cache_mod.make_prompt_cache(model)
    warm_logits = model(tokens, cache=warm_cache)
    mx.eval(warm_logits)
    # Also warm a single decode step
    _ = model(mx.argmax(warm_logits[0, -1, :])[None, None], cache=warm_cache)
    mx.eval(_)

    # Prefill timing
    cache = mlx_cache_mod.make_prompt_cache(model)
    mx.eval(tokens)
    t0 = time.perf_counter()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    prefill_s = time.perf_counter() - t0

    # Capture prefill logits for quality comparison
    last_logits = logits[0, -1, :]

    # Decode timing — generate `decode_steps` tokens
    decode_start = time.perf_counter()
    next_token = mx.argmax(last_logits)[None, None]
    decoded = []
    for _ in range(decode_steps):
        logits = model(next_token, cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[0, -1, :])[None, None]
        decoded.append(int(next_token.item()))
    decode_s = time.perf_counter() - decode_start

    # Memory snapshot
    total_bytes = 0
    n_pq = 0
    for c in cache:
        if hasattr(c, "nbytes"):
            try:
                nb = c.nbytes
                if isinstance(nb, int):
                    total_bytes += nb
            except Exception:
                pass
        if type(c).__name__ == "PlanarQuantKVCache":
            n_pq += 1

    disable_planarquant_cache()

    decoded_text = tokenizer.decode(decoded)

    return {
        "label": label,
        "prompt_len": prompt_len,
        "prefill_s": prefill_s,
        "prefill_tps": prompt_len / prefill_s,
        "decode_s": decode_s,
        "decode_tps": decode_steps / decode_s,
        "decoded_text": decoded_text[:80],
        "cache_bytes": total_bytes,
        "cache_mb": total_bytes / 1e6,
        "n_layers": len(cache),
        "n_pq_layers": n_pq,
        "last_logits": last_logits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="mlx-community/Qwen3.5-4B-MLX-4bit", help="HF model id"
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is a city known for its art, cuisine, and architecture. It is called",
        help="Benchmark prompt",
    )
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--pq-bits", type=int, default=3)
    args = parser.parse_args()

    _ensure_imports()

    from mlx_lm import load

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    print(f"Prompt: {args.prompt}")
    print(f"Decode steps: {args.decode_steps}")
    print()

    fp16 = bench_config(
        "FP16", model, tokenizer, args.prompt, args.decode_steps, False, args.pq_bits
    )
    pq = bench_config(
        f"PlanarQuant{args.pq_bits}",
        model,
        tokenizer,
        args.prompt,
        args.decode_steps,
        True,
        args.pq_bits,
    )

    # Cosine sim between the two last-logit vectors
    fp16_l = fp16["last_logits"].astype(mx.float32)
    pq_l = pq["last_logits"].astype(mx.float32)
    dot = float(mx.sum(fp16_l * pq_l).item())
    nfp = float(mx.sqrt(mx.sum(fp16_l * fp16_l)).item())
    npq = float(mx.sqrt(mx.sum(pq_l * pq_l)).item())
    cos_sim = dot / (nfp * npq + 1e-10)

    # Format table
    print("=" * 88)
    print(f"{'metric':<22} {'FP16':>20} {'PlanarQuant' + str(args.pq_bits):>20} {'delta':>20}")
    print("-" * 88)
    print(
        f"{'prefill tok/s':<22} {fp16['prefill_tps']:>20.2f} "
        f"{pq['prefill_tps']:>20.2f} "
        f"{(pq['prefill_tps'] / fp16['prefill_tps'] - 1) * 100:>19.1f}%"
    )
    print(
        f"{'decode tok/s':<22} {fp16['decode_tps']:>20.2f} "
        f"{pq['decode_tps']:>20.2f} "
        f"{(pq['decode_tps'] / fp16['decode_tps'] - 1) * 100:>19.1f}%"
    )
    print(
        f"{'cache MB':<22} {fp16['cache_mb']:>20.3f} "
        f"{pq['cache_mb']:>20.3f} "
        f"{(pq['cache_mb'] / fp16['cache_mb'] - 1) * 100:>19.1f}%"
    )
    print("-" * 88)
    print(f"layers wrapped (PQ): {pq['n_pq_layers']}/{pq['n_layers']}")
    print(f"logit cos sim:       {cos_sim:.6f}")
    print()
    print(f"FP16 decoded: {fp16['decoded_text']!r}")
    print(f"PQ   decoded: {pq['decoded_text']!r}")
    print("=" * 88)


if __name__ == "__main__":
    main()
