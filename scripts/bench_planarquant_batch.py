#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""BatchPlanarQuantKVCache benchmark — batch ops + batched decode throughput."""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx


def bench_batch_ops(H: int = 16, D: int = 128, T: int = 256, B: int = 4,
                     bits: float = 3.0, quantize_v: bool = True, n_iter: int = 20):
    """Benchmark individual batch operations."""
    from omlx.cache.planarquant.kv_cache import (
        BatchPlanarQuantKVCache,
        PlanarQuantKVCache,
    )

    results = {}

    def _make_single(t: int) -> PlanarQuantKVCache:
        c = PlanarQuantKVCache(bits=bits, quantize_v=quantize_v)
        x = mx.random.normal((1, H, t, D)) * 0.1
        c.update_and_fetch(x, x)
        c.finalize_prefill()
        mx.eval(c._k_packed, c._k_norms)
        return c

    # --- merge ---
    caches = [_make_single(T) for _ in range(B)]
    times = []
    for _ in range(n_iter):
        mx.synchronize()
        t0 = time.perf_counter()
        batch = BatchPlanarQuantKVCache.merge(caches)
        mx.eval(batch._k_packed)
        times.append(time.perf_counter() - t0)
    results["merge"] = (sum(times) / len(times)) * 1000  # ms

    # --- filter ---
    times = []
    indices = list(range(1, B))
    for _ in range(n_iter):
        mx.synchronize()
        t0 = time.perf_counter()
        batch.filter(indices)
        mx.eval(batch._k_packed)
        times.append(time.perf_counter() - t0)
    results["filter"] = (sum(times) / len(times)) * 1000

    # --- prepare ---
    batch2 = BatchPlanarQuantKVCache(left_padding=[0] * B, bits=bits, quantize_v=quantize_v)
    times = []
    lp = list(range(B))
    for _ in range(n_iter):
        batch2r = BatchPlanarQuantKVCache(left_padding=[0] * B, bits=bits, quantize_v=quantize_v)
        mx.synchronize()
        t0 = time.perf_counter()
        batch2r.prepare(left_padding=mx.array(lp))
        times.append(time.perf_counter() - t0)
    results["prepare"] = (sum(times) / len(times)) * 1000

    # --- extend ---
    c1 = _make_single(T)
    c2 = _make_single(T)
    b1 = BatchPlanarQuantKVCache.merge([c1])
    b2 = BatchPlanarQuantKVCache.merge([c2])
    times = []
    for _ in range(n_iter):
        # Reset b1/b2 for each iteration
        c1r = _make_single(T)
        c2r = _make_single(T)
        b1r = BatchPlanarQuantKVCache.merge([c1r])
        b2r = BatchPlanarQuantKVCache.merge([c2r])
        mx.synchronize()
        t0 = time.perf_counter()
        b1r.extend(b2r)
        mx.eval(b1r._k_packed)
        times.append(time.perf_counter() - t0)
    results["extend"] = (sum(times) / len(times)) * 1000

    # --- extract ---
    batch3 = BatchPlanarQuantKVCache.merge([_make_single(T + i * 10) for i in range(B)])
    times = []
    for _ in range(n_iter):
        mx.synchronize()
        t0 = time.perf_counter()
        extracted = batch3.extract(0)
        mx.eval(extracted._k_packed)
        times.append(time.perf_counter() - t0)
    results["extract"] = (sum(times) / len(times)) * 1000

    # --- finalize ---
    times = []
    for _ in range(n_iter):
        batch4r = BatchPlanarQuantKVCache(left_padding=[0] * 2, bits=bits, quantize_v=quantize_v)
        xr = mx.random.normal((2, H, T, D)) * 0.1
        batch4r.update_and_fetch(xr, xr)
        batch4r.finalize_prefill()
        batch4r._right_padding = mx.array([3, 0])
        mx.synchronize()
        t0 = time.perf_counter()
        batch4r.finalize()
        mx.eval(batch4r._k_packed)
        times.append(time.perf_counter() - t0)
    results["finalize"] = (sum(times) / len(times)) * 1000

    # --- evict_dequant_caches ---
    batch5 = BatchPlanarQuantKVCache.merge([_make_single(T) for _ in range(B)])
    batch5._ensure_k_dequant_cache()
    mx.eval(batch5._k_dequant_cache)
    times = []
    for _ in range(n_iter):
        batch5r = BatchPlanarQuantKVCache.merge([_make_single(T) for _ in range(B)])
        batch5r._ensure_k_dequant_cache()
        mx.eval(batch5r._k_dequant_cache)
        mx.synchronize()
        t0 = time.perf_counter()
        freed = batch5r.evict_dequant_caches()
        times.append(time.perf_counter() - t0)
    results["evict_dequant"] = (sum(times) / len(times)) * 1000

    return results


def bench_batch_decode(model, tokenizer, prompt_lens: list[int], decode_steps: int = 32,
                        pq_bits: int = 3, batch_sizes: list[int] | None = None):
    """Benchmark batched decode throughput for PlanarQuant vs FP16."""
    from mlx_lm.models import cache as mlx_cache_mod

    from omlx.patches.planarquant_cache import (
        disable_planarquant_cache,
        enable_planarquant_cache,
    )
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()
    if batch_sizes is None:
        batch_sizes = [1, 2, 4]

    results = []

    for pq_enabled in [False, True]:
        if pq_enabled:
            enable_planarquant_cache(pq_bits)
        else:
            disable_planarquant_cache()

        for B in batch_sizes:
            # Create B prompts of different lengths
            prompts = []
            for i in range(B):
                base = "The capital of France is a city known for its art. "
                target_len = prompt_lens[i % len(prompt_lens)]
                # Repeat to approximate target length
                text = base * max(1, target_len // len(base.split()) + 1)
                prompts.append(text)

            # Encode all prompts
            all_tokens = []
            for p in prompts:
                toks = tokenizer.encode(p)[:max(prompt_lens)]
                all_tokens.append(toks)

            # Pad to same length (left-pad with 0)
            max_len = max(len(t) for t in all_tokens)
            padded = []
            for t in all_tokens:
                pad_len = max_len - len(t)
                padded.append([0] * pad_len + t)

            # Batch tensor: (B, max_len)
            batch_tokens = mx.array(padded)

            # Create cache and run
            cache = mlx_cache_mod.make_prompt_cache(model)

            # Prefill
            mx.eval(batch_tokens)
            t0 = time.perf_counter()
            try:
                logits = model(batch_tokens, cache=cache)
                mx.eval(logits)
            except Exception as e:
                # Some models don't support batched prefill directly
                # Fall back to sequential prefill
                for i in range(B):
                    single_tok = mx.array([padded[i]])
                    try:
                        logits = model(single_tok, cache=cache)
                        mx.eval(logits)
                    except Exception:
                        break
                max_len = 1  # Can't do batched prefill
            prefill_s = time.perf_counter() - t0

            # Decode steps
            # For simplicity, decode the same token for all batch elements
            next_tok = mx.array([[1]] * B)
            decode_times = []
            for step in range(decode_steps):
                mx.synchronize()
                t0 = time.perf_counter()
                try:
                    logits = model(next_tok, cache=cache)
                    mx.eval(logits)
                    next_tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
                    mx.eval(next_tok)
                except Exception:
                    # If batch fails, try single
                    break
                decode_times.append(time.perf_counter() - t0)

            avg_decode_s = sum(decode_times) / len(decode_times) if decode_times else float("inf")
            total_decode_tokens = len(decode_times) * B
            decode_tps = total_decode_tokens / sum(decode_times) if decode_times else 0

            # Memory
            total_bytes = 0
            for c in cache:
                if hasattr(c, "nbytes"):
                    try:
                        nb = c.nbytes
                        if isinstance(nb, int):
                            total_bytes += nb
                    except Exception:
                        pass

            label = f"{'PQ' if pq_enabled else 'FP16'}"
            results.append({
                "label": label,
                "pq_enabled": pq_enabled,
                "B": B,
                "prompt_len": max_len,
                "prefill_s": prefill_s,
                "decode_tps": decode_tps,
                "avg_step_ms": avg_decode_s * 1000,
                "cache_mb": total_bytes / 1e6,
                "decode_steps": len(decode_times),
            })

    disable_planarquant_cache()
    return results


def bench_single_decode(model, tokenizer, prompt: str, decode_steps: int = 64,
                         pq_bits: int = 3, prompt_tokens_override: int | None = None):
    """Benchmark single-request decode (B=1) PlanarQuant vs FP16."""
    from mlx_lm.models import cache as mlx_cache_mod

    from omlx.patches.planarquant_cache import (
        disable_planarquant_cache,
        enable_planarquant_cache,
    )
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()

    results = []
    for pq_enabled in [False, True]:
        if pq_enabled:
            enable_planarquant_cache(pq_bits)
        else:
            disable_planarquant_cache()

        tokens = mx.array(tokenizer.encode(prompt))[None, :]
        if prompt_tokens_override:
            tokens = tokens[:, :prompt_tokens_override]

        # Warm up
        warm_cache = mlx_cache_mod.make_prompt_cache(model)
        _ = model(tokens, cache=warm_cache)
        mx.eval(_)

        # Prefill
        cache = mlx_cache_mod.make_prompt_cache(model)
        mx.eval(tokens)
        t0 = time.perf_counter()
        logits = model(tokens, cache=cache)
        mx.eval(logits)
        prefill_s = time.perf_counter() - t0
        prompt_len = tokens.shape[1]

        # Decode
        next_tok = mx.argmax(logits[0, -1, :])[None, None]
        decode_times = []
        for _ in range(decode_steps):
            mx.synchronize()
            t0 = time.perf_counter()
            logits = model(next_tok, cache=cache)
            mx.eval(logits)
            next_tok = mx.argmax(logits[0, -1, :])[None, None]
            mx.eval(next_tok)
            decode_times.append(time.perf_counter() - t0)

        avg_step_ms = (sum(decode_times) / len(decode_times)) * 1000
        decode_tps = 1.0 / (sum(decode_times) / len(decode_times))

        # Memory
        total_bytes = 0
        for c in cache:
            if hasattr(c, "nbytes"):
                try:
                    nb = c.nbytes
                    if isinstance(nb, int):
                        total_bytes += nb
                except Exception:
                    pass

        results.append({
            "label": f"{'PQ' if pq_enabled else 'FP16'}",
            "pq_enabled": pq_enabled,
            "prompt_len": prompt_len,
            "prefill_tps": prompt_len / prefill_s,
            "decode_tps": decode_tps,
            "avg_step_ms": avg_step_ms,
            "cache_mb": total_bytes / 1e6,
            "last_logits": logits[0, -1, :],
        })

    disable_planarquant_cache()

    # Cosine sim
    fp16_l = results[0]["last_logits"].astype(mx.float32)
    pq_l = results[1]["last_logits"].astype(mx.float32)
    dot = float(mx.sum(fp16_l * pq_l).item())
    n0 = float(mx.sqrt(mx.sum(fp16_l * fp16_l)).item())
    n1 = float(mx.sqrt(mx.sum(pq_l * pq_l)).item())
    cos_sim = dot / (n0 * n1 + 1e-10)

    return results, cos_sim


def bench_memory_per_token(H: int = 16, D: int = 128, bits: float = 3.0):
    """Benchmark memory per token across storage modes."""
    from omlx.cache.planarquant.kv_cache import PlanarQuantKVCache

    # FP16 K+V
    fp16_bytes = H * D * 2 * 2  # K+V, 2 bytes each

    # PlanarQuant K only (quantize_v=False)
    packed_last = D // 4 + D // 8
    pq_k_bytes = packed_last + 2  # packed + 1 norm (2 bytes)
    # Plus dequant cache
    pq_k_dequant = H * D * 2  # fp16 dequant cache

    # PlanarQuant K+V
    pq_kv_bytes = (packed_last + 2) * 2

    # PlanarQuant K+V with dequant caches
    pq_kv_dequant_bytes = pq_kv_bytes + pq_k_dequant * 2  # K+V dequant

    return {
        "fp16_kv": fp16_bytes,
        "pq_k_only": pq_k_bytes,
        "pq_kv": pq_kv_bytes,
        "pq_kv_dequant": pq_kv_dequant_bytes,
        "compression_k_only": fp16_bytes / pq_k_bytes,
        "compression_kv": fp16_bytes / pq_kv_bytes,
        "compression_kv_dequant": fp16_bytes / pq_kv_dequant_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--pq-bits", type=int, default=3)
    parser.add_argument("--skip-model", action="store_true", help="Skip model-dependent benchmarks")
    parser.add_argument("--batch-ops-only", action="store_true", help="Only benchmark batch ops")
    args = parser.parse_args()

    mx.random.seed(42)

    # ---- Batch operation benchmarks (no model needed) ----
    print("=" * 90)
    print("BATCH OPERATION LATENCY (per layer)")
    print("=" * 90)

    for T in [64, 256, 1024]:
        for B in [2, 4, 8]:
            r = bench_batch_ops(H=16, D=128, T=T, B=B, bits=args.pq_bits, n_iter=10)
            print(f"\n  T={T}, B={B}, H=16, D=128, bits={args.pq_bits}")
            print(f"  {'operation':<22} {'latency (ms)':>12}")
            print(f"  {'-'*34}")
            for op, lat in sorted(r.items()):
                print(f"  {op:<22} {lat:>12.3f}")

    # ---- Memory per token ----
    print("\n" + "=" * 90)
    print("MEMORY PER TOKEN-ROW PER HEAD (D=128)")
    print("=" * 90)
    mem = bench_memory_per_token(D=128, bits=args.pq_bits)
    print(f"  {'mode':<30} {'bytes':>10} {'vs FP16':>10}")
    print(f"  {'-'*52}")
    fp16 = mem["fp16_kv"]
    for key, label in [
        ("fp16_kv", "FP16 K+V"),
        ("pq_k_only", "PQ K-only (packed)"),
        ("pq_kv", "PQ K+V (packed)"),
        ("pq_kv_dequant", "PQ K+V + dequant caches"),
    ]:
        ratio = fp16 / mem[key] if mem[key] > 0 else 0
        print(f"  {label:<30} {mem[key]:>10} {ratio:>9.2f}x")

    if args.skip_model or args.batch_ops_only:
        print("\n(Skipped model-dependent benchmarks)")
        return

    # ---- Model-dependent benchmarks ----
    try:
        from mlx_lm import load
    except ImportError:
        print("mlx_lm not available — skipping model benchmarks")
        return

    print(f"\nLoading {args.model}...")
    model, tokenizer = load(args.model)

    prompt = "The capital of France is a city known for its art, cuisine, and architecture. " * 8

    # ---- Single-request decode (B=1) ----
    for prompt_tokens in [81, 241, 641]:
        print(f"\n{'=' * 90}")
        print(f"SINGLE-REQUEST DECODE (B=1, prompt={prompt_tokens} tokens, {args.decode_steps} decode steps)")
        print(f"{'=' * 90}")

        results, cos_sim = bench_single_decode(
            model, tokenizer, prompt, args.decode_steps,
            pq_bits=args.pq_bits, prompt_tokens_override=prompt_tokens,
        )

        fp16_r = results[0]
        pq_r = results[1]

        print(f"  {'metric':<22} {'FP16':>14} {'PlanarQuant':>14} {'ratio':>10}")
        print(f"  {'-'*60}")
        print(f"  {'decode tok/s':<22} {fp16_r['decode_tps']:>14.1f} {pq_r['decode_tps']:>14.1f} "
              f"{pq_r['decode_tps']/fp16_r['decode_tps']:>9.3f}x")
        print(f"  {'avg step (ms)':<22} {fp16_r['avg_step_ms']:>14.2f} {pq_r['avg_step_ms']:>14.2f} "
              f"{pq_r['avg_step_ms']/fp16_r['avg_step_ms']:>9.3f}x")
        print(f"  {'prefill tok/s':<22} {fp16_r['prefill_tps']:>14.1f} {pq_r['prefill_tps']:>14.1f} "
              f"{pq_r['prefill_tps']/fp16_r['prefill_tps']:>9.3f}x")
        print(f"  {'cache MB':<22} {fp16_r['cache_mb']:>14.2f} {pq_r['cache_mb']:>14.2f} "
              f"{pq_r['cache_mb']/fp16_r['cache_mb']:>9.2f}x")
        print(f"  {'logit cos sim':<22} {'':>14} {cos_sim:>14.6f}")
        print(f"  {'speed parity':<22} {'1.000x':>14} {pq_r['decode_tps']/fp16_r['decode_tps']:>14.3f}x")

    # ---- Batched decode (B>1) ----
    print(f"\n{'=' * 90}")
    print(f"BATCHED DECODE THROUGHPUT (prompt~80 tokens, {args.decode_steps} decode steps)")
    print(f"{'=' * 90}")

    for B in [1, 2, 4]:
        batch_results = bench_batch_decode(
            model, tokenizer,
            prompt_lens=[80, 60, 100, 40][:B],
            decode_steps=args.decode_steps,
            pq_bits=args.pq_bits,
            batch_sizes=[B],
        )

        fp16_r = [r for r in batch_results if not r["pq_enabled"]][0]
        pq_r = [r for r in batch_results if r["pq_enabled"]][0]

        speedup = pq_r["decode_tps"] / fp16_r["decode_tps"] if fp16_r["decode_tps"] > 0 else 0
        mem_ratio = pq_r["cache_mb"] / fp16_r["cache_mb"] if fp16_r["cache_mb"] > 0 else 0

        print(f"\n  B={B}")
        print(f"  {'metric':<22} {'FP16':>14} {'PlanarQuant':>14} {'ratio':>10}")
        print(f"  {'-'*60}")
        print(f"  {'total decode tps':<22} {fp16_r['decode_tps']:>14.1f} {pq_r['decode_tps']:>14.1f} "
              f"{speedup:>9.3f}x")
        print(f"  {'per-request tps':<22} {fp16_r['decode_tps']/B:>14.1f} {pq_r['decode_tps']/B:>14.1f} "
              f"{speedup:>9.3f}x")
        print(f"  {'avg step (ms)':<22} {fp16_r['avg_step_ms']:>14.2f} {pq_r['avg_step_ms']:>14.2f} "
              f"{pq_r['avg_step_ms']/fp16_r['avg_step_ms']:>9.3f}x")
        print(f"  {'cache MB':<22} {fp16_r['cache_mb']:>14.2f} {pq_r['cache_mb']:>14.2f} "
              f"{mem_ratio:>9.2f}x")

    print("\n" + "=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
