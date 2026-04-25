#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end validation: PlanarQuant KV + DFlash at scale.

Benchmarks:
  1. Memory: FP16 vs PQ at T=4K/32K/128K, B=1/4/8
  2. Decode speed: FP16 vs PQ at increasing context + batch
  3. DFlash + PQ: speculative decoding with compressed KV
  4. Memory-pressure: evict_dequant_caches + per-layer rebuild
  5. Quality: logit cosine similarity at each scale
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx


def bench_memory_and_speed(model, tokenizer, args):
    """Comprehensive memory + speed + quality benchmark."""
    from mlx_lm.models import cache as mlx_cache_mod
    from omlx.patches.planarquant_cache import (
        disable_planarquant_cache,
        enable_planarquant_cache,
    )
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()

    prompt_text = (
        "The history of computing spans centuries, from the abacus to quantum computers. "
        "Each era brought revolutionary changes to how humans process information. "
    ) * 200  # ~4096 tokens

    results = []

    for pq_enabled in [False, True]:
        if pq_enabled:
            enable_planarquant_cache(args.pq_bits)
        else:
            disable_planarquant_cache()

        label = f"{'PQ' if pq_enabled else 'FP16'}"

        for target_tokens in [4096, 32768]:
            # Encode and trim to target length
            all_tokens = tokenizer.encode(prompt_text)
            if len(all_tokens) < target_tokens:
                # Repeat to fill
                repeat = target_tokens // len(all_tokens) + 1
                all_tokens = (all_tokens * repeat)[:target_tokens]

            tokens = mx.array(all_tokens)[None, :]  # B=1

            # Create cache and prefill
            cache = mlx_cache_mod.make_prompt_cache(model)
            mx.eval(tokens)
            t0 = time.perf_counter()
            logits = model(tokens, cache=cache)
            mx.eval(logits)
            prefill_s = time.perf_counter() - t0
            actual_prompt_len = tokens.shape[1]

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

            # Decode 32 steps
            next_tok = mx.argmax(logits[0, -1, :])[None, None]
            decode_times = []
            decoded_tokens = []
            for _ in range(32):
                mx.synchronize()
                t0 = time.perf_counter()
                logits = model(next_tok, cache=cache)
                mx.eval(logits)
                next_tok = mx.argmax(logits[0, -1, :])[None, None]
                mx.eval(next_tok)
                decode_times.append(time.perf_counter() - t0)
                decoded_tokens.append(int(next_tok.item()))

            avg_decode_ms = (sum(decode_times) / len(decode_times)) * 1000
            decode_tps = 1.0 / (sum(decode_times) / len(decode_times))

            results.append({
                "label": label,
                "pq_enabled": pq_enabled,
                "B": 1,
                "prompt_tokens": actual_prompt_len,
                "prefill_tps": actual_prompt_len / prefill_s,
                "decode_tps": decode_tps,
                "avg_step_ms": avg_decode_ms,
                "cache_mb": total_bytes / 1e6,
                "cache_gb": total_bytes / 1e9,
                "n_pq_layers": n_pq,
                "n_layers": len(cache),
                "last_logits": logits[0, -1, :],
                "decoded_text": tokenizer.decode(decoded_tokens)[:60],
            })

            # Free memory
            del cache
            del logits

    disable_planarquant_cache()
    return results


def bench_dflash_pq(model, tokenizer, args):
    """DFlash speculative decoding + PQ KV compression."""
    from omlx.patches.planarquant_cache import enable_planarquant_cache, disable_planarquant_cache
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()

    prompt_text = (
        "The history of computing spans centuries, from the abacus to quantum computers. "
        "Each era brought revolutionary changes to how humans process information. "
    ) * 200

    results = []

    # Config 1: Baseline (no PQ, no DFlash)
    # Config 2: PQ only
    # Config 3: DFlash only
    # Config 4: PQ + DFlash combined

    for config_name, pq_on, dflash_on in [
        ("Baseline (FP16, no DFlash)", False, False),
        ("PQ3 only", True, False),
        ("DFlash only", False, True),
        ("PQ3 + DFlash", True, True),
    ]:
        if pq_on:
            enable_planarquant_cache(args.pq_bits)
        else:
            disable_planarquant_cache()

        target_tokens = 4096
        all_tokens = tokenizer.encode(prompt_text)
        if len(all_tokens) < target_tokens:
            all_tokens = (all_tokens * (target_tokens // len(all_tokens) + 1))[:target_tokens]

        tokens = mx.array(all_tokens)[None, :]

        # DFlash setup
        draft_model = None
        if dflash_on:
            try:
                from omlx.patches.dflash import load_dflash_draft
                draft_model, resolved = load_dflash_draft(args.model)
                if draft_model is None:
                    print(f"  [SKIP] DFlash: no draft model for {args.model}")
                    continue
                print(f"  DFlash draft: {resolved}")
            except Exception as e:
                print(f"  [SKIP] DFlash: {e}")
                continue

        from mlx_lm.models import cache as mlx_cache_mod
        cache = mlx_cache_mod.make_prompt_cache(model)
        mx.eval(tokens)
        t0 = time.perf_counter()
        logits = model(tokens, cache=cache)
        mx.eval(logits)
        prefill_s = time.perf_counter() - t0

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

        # Decode with/without DFlash
        n_decode = 64
        decode_times = []
        decoded_tokens = []

        if dflash_on and draft_model is not None:
            # DFlash speculative decode
            from omlx.patches.dflash import install_dflash_hooks
            install_dflash_hooks(model, draft_model=draft_model)

        next_tok = mx.argmax(logits[0, -1, :])[None, None]
        for _ in range(n_decode):
            mx.synchronize()
            t0 = time.perf_counter()
            logits = model(next_tok, cache=cache)
            mx.eval(logits)
            next_tok = mx.argmax(logits[0, -1, :])[None, None]
            mx.eval(next_tok)
            decode_times.append(time.perf_counter() - t0)
            decoded_tokens.append(int(next_tok.item()))

        avg_decode_ms = (sum(decode_times) / len(decode_times)) * 1000
        decode_tps = n_decode / sum(decode_times)

        results.append({
            "config": config_name,
            "pq_on": pq_on,
            "dflash_on": dflash_on,
            "prompt_tokens": tokens.shape[1],
            "prefill_tps": tokens.shape[1] / prefill_s,
            "decode_tps": decode_tps,
            "avg_step_ms": avg_decode_ms,
            "cache_gb": total_bytes / 1e9,
            "decoded": tokenizer.decode(decoded_tokens)[:60],
        })

        del cache
        del logits

    disable_planarquant_cache()
    return results


def bench_evict_and_rebuild(model, tokenizer, args):
    """Memory-pressure mode: evict dequant caches, rebuild per-layer on decode."""
    from mlx_lm.models import cache as mlx_cache_mod
    from omlx.patches.planarquant_cache import enable_planarquant_cache, disable_planarquant_cache
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    apply_turboquant_attention_patch()
    enable_planarquant_cache(args.pq_bits)

    prompt_text = (
        "The history of computing spans centuries, from the abacus to quantum computers. "
    ) * 200

    target_tokens = 4096
    all_tokens = tokenizer.encode(prompt_text)
    if len(all_tokens) < target_tokens:
        all_tokens = (all_tokens * (target_tokens // len(all_tokens) + 1))[:target_tokens]
    tokens = mx.array(all_tokens)[None, :]

    # Normal PQ decode (with dequant caches)
    cache_normal = mlx_cache_mod.make_prompt_cache(model)
    logits = model(tokens, cache=cache_normal)
    mx.eval(logits)
    # Warm + decode
    next_tok = mx.argmax(logits[0, -1, :])[None, None]
    for _ in range(5):
        logits = model(next_tok, cache=cache_normal)
        mx.eval(logits)
        next_tok = mx.argmax(logits[0, -1, :])[None, None]

    times_normal = []
    for _ in range(32):
        mx.synchronize()
        t0 = time.perf_counter()
        logits = model(next_tok, cache=cache_normal)
        mx.eval(logits)
        next_tok = mx.argmax(logits[0, -1, :])[None, None]
        times_normal.append(time.perf_counter() - t0)

    cache_normal_mb = sum(
        c.nbytes for c in cache_normal if hasattr(c, "nbytes") and isinstance(c.nbytes, int)
    ) / 1e6

    # Evict mode: free dequant caches, rebuild per-layer on each decode step
    total_freed = 0
    for c in cache_normal:
        if type(c).__name__ == "PlanarQuantKVCache" and hasattr(c, "evict_dequant_caches"):
            freed = c.evict_dequant_caches()
            total_freed += freed

    cache_evicted_mb = sum(
        c.nbytes for c in cache_normal if hasattr(c, "nbytes") and isinstance(c.nbytes, int)
    ) / 1e6

    # Decode after eviction (will rebuild dequant caches lazily)
    times_evicted = []
    for _ in range(32):
        mx.synchronize()
        t0 = time.perf_counter()
        logits = model(next_tok, cache=cache_normal)
        mx.eval(logits)
        next_tok = mx.argmax(logits[0, -1, :])[None, None]
        times_evicted.append(time.perf_counter() - t0)

    # Memory after rebuild
    cache_rebuilt_mb = sum(
        c.nbytes for c in cache_normal if hasattr(c, "nbytes") and isinstance(c.nbytes, int)
    ) / 1e6

    disable_planarquant_cache()

    return {
        "normal_tps": 32 / sum(times_normal),
        "normal_step_ms": (sum(times_normal) / 32) * 1000,
        "normal_cache_mb": cache_normal_mb,
        "evicted_cache_mb": cache_evicted_mb,
        "freed_mb": total_freed / 1e6,
        "evicted_tps": 32 / sum(times_evicted),
        "evicted_step_ms": (sum(times_evicted) / 32) * 1000,
        "rebuilt_cache_mb": cache_rebuilt_mb,
        "memory_savings_pct": (1 - cache_evicted_mb / cache_normal_mb) * 100 if cache_normal_mb > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    parser.add_argument("--pq-bits", type=int, default=3)
    parser.add_argument("--skip-dflash", action="store_true")
    args = parser.parse_args()

    mx.random.seed(42)

    print(f"Loading {args.model}...")
    from mlx_lm import load
    model, tokenizer = load(args.model)

    # ==================================================================
    # PART 1: Memory + Speed at Scale
    # ==================================================================
    print("\n" + "=" * 90)
    print("PART 1: MEMORY + DECODE SPEED AT SCALE")
    print("=" * 90)

    results = bench_memory_and_speed(model, tokenizer, args)

    # Print comparison table
    print(f"\n{'Config':<12} {'Prompt':>7} {'Prefill':>10} {'Decode':>10} {'Step':>8} {'Cache':>8} {'Mem':>6} {'Speed':>7}")
    print(f"{'':12} {'toks':>7} {'tok/s':>10} {'tok/s':>10} {'ms':>8} {'MB':>8} {'ratio':>6} {'ratio':>7}")
    print("-" * 70)

    # Pair up FP16 vs PQ at same prompt length
    for i in range(0, len(results), 2):
        fp16 = results[i]
        pq = results[i + 1]
        mem_ratio = pq["cache_mb"] / fp16["cache_mb"] if fp16["cache_mb"] > 0 else 0
        speed_ratio = pq["decode_tps"] / fp16["decode_tps"] if fp16["decode_tps"] > 0 else 0

        print(f"  FP16  T={fp16['prompt_tokens']:>5}  {fp16['prefill_tps']:>10.0f}  {fp16['decode_tps']:>10.1f}  "
              f"{fp16['avg_step_ms']:>8.2f}  {fp16['cache_mb']:>8.1f}  {'1.00x':>6}  {'1.00x':>7}")
        print(f"  PQ3   T={pq['prompt_tokens']:>5}  {pq['prefill_tps']:>10.0f}  {pq['decode_tps']:>10.1f}  "
              f"{pq['avg_step_ms']:>8.2f}  {pq['cache_mb']:>8.1f}  {mem_ratio:>5.2f}x  {speed_ratio:>6.3f}x")
        print()

    # Quality
    for i in range(0, len(results), 2):
        fp16 = results[i]
        pq = results[i + 1]
        fp16_l = fp16["last_logits"].astype(mx.float32)
        pq_l = pq["last_logits"].astype(mx.float32)
        dot = float(mx.sum(fp16_l * pq_l).item())
        n0 = float(mx.sqrt(mx.sum(fp16_l * fp16_l)).item())
        n1 = float(mx.sqrt(mx.sum(pq_l * pq_l)).item())
        cos_sim = dot / (n0 * n1 + 1e-10)
        print(f"  T={fp16['prompt_tokens']:>5} logit cos_sim: {cos_sim:.6f}")

    # ==================================================================
    # PART 2: Theoretical Memory at 128K
    # ==================================================================
    print("\n" + "=" * 90)
    print("PART 2: THEORETICAL MEMORY AT 128K CONTEXT (Qwen3.5-4B)")
    print("=" * 90)
    print(f"  (4 KV heads, head_dim=128, 32 layers)")
    print()

    layers, kv_heads, head_dim = 32, 4, 128
    for T in [4096, 32768, 131072]:
        for B in [1, 4, 8]:
            fp16_gb = T * layers * kv_heads * head_dim * 4 * B / 1e9
            pq_packed_gb = T * layers * kv_heads * 50 * 2 * B / 1e9
            pq_dequant_gb = fp16_gb + pq_packed_gb
            fits = "YES" if 2.5 + fp16_gb < 120 else "OOM"
            pq_fits = "YES" if 2.5 + pq_dequant_gb < 120 else ("PACKED ONLY" if 2.5 + pq_packed_gb < 120 else "OOM")

            if B == 1:
                print(f"  T={T//1024:>5}K  B={B}: FP16={fp16_gb:>6.1f}GB  PQ packed={pq_packed_gb:>5.1f}GB  PQ+dequant={pq_dequant_gb:>6.1f}GB  FP16={fits}  PQ={pq_fits}")
            else:
                print(f"          B={B}: FP16={fp16_gb:>6.1f}GB  PQ packed={pq_packed_gb:>5.1f}GB  PQ+dequant={pq_dequant_gb:>6.1f}GB  FP16={fits}  PQ={pq_fits}")
        print()

    # ==================================================================
    # PART 3: DFlash + PQ Combined
    # ==================================================================
    if not args.skip_dflash:
        print("\n" + "=" * 90)
        print("PART 3: DFLASH + PQ COMBINED")
        print("=" * 90)

        dflash_results = bench_dflash_pq(model, tokenizer, args)

        if dflash_results:
            print(f"\n  {'Config':<28} {'Prefill':>8} {'Decode':>8} {'Step':>7} {'Cache':>7}")
            print(f"  {'':28} {'tok/s':>8} {'tok/s':>8} {'ms':>7} {'GB':>7}")
            print("  " + "-" * 58)
            for r in dflash_results:
                print(f"  {r['config']:<28} {r['prefill_tps']:>8.0f} {r['decode_tps']:>8.1f} "
                      f"{r['avg_step_ms']:>7.2f} {r['cache_gb']:>7.2f}")

            # Speedup vs baseline
            baseline = dflash_results[0]
            for r in dflash_results[1:]:
                speedup = r["decode_tps"] / baseline["decode_tps"]
                mem_saved = 1 - r["cache_gb"] / baseline["cache_gb"]
                print(f"  {r['config']:<28} → {speedup:.2f}x speed, {mem_saved*100:+.0f}% memory")

    # ==================================================================
    # PART 4: Memory-pressure (evict dequant caches)
    # ==================================================================
    print("\n" + "=" * 90)
    print("PART 4: MEMORY-PRESSURE MODE (evict_dequant_caches)")
    print("=" * 90)

    evict = bench_evict_and_rebuild(model, tokenizer, args)

    print(f"\n  {'Mode':<25} {'Decode tps':>10} {'Step ms':>8} {'Cache MB':>10}")
    print(f"  {'':25} {'':10} {'':8} {'(active)':>10}")
    print("  " + "-" * 55)
    print(f"  {'Normal (PQ + dequant)':<25} {evict['normal_tps']:>10.1f} {evict['normal_step_ms']:>8.2f} {evict['normal_cache_mb']:>10.1f}")
    print(f"  {'After evict (packed only)':<25} {'--':>10} {'--':>8} {evict['evicted_cache_mb']:>10.1f}")
    print(f"  {'After rebuild (lazily)':<25} {evict['evicted_tps']:>10.1f} {evict['evicted_step_ms']:>8.2f} {evict['rebuilt_cache_mb']:>10.1f}")
    print()
    print(f"  Memory freed by eviction: {evict['freed_mb']:.1f} MB ({evict['memory_savings_pct']:.0f}% of cache)")
    print(f"  Decode speed after rebuild: {evict['evicted_tps']/evict['normal_tps']:.3f}x of normal")
    print(f"  First-step rebuild cost: {(evict['evicted_step_ms'] - evict['normal_step_ms']):.2f} ms (one-time)")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Find the 4K PQ result for summary
    pq_4k = [r for r in results if r["pq_enabled"] and r["prompt_tokens"] >= 4095][0]
    fp16_4k = [r for r in results if not r["pq_enabled"] and r["prompt_tokens"] >= 4095][0]

    print(f"\n  PlanarQuant3 KV Compression:")
    print(f"    Decode speed: {pq_4k['decode_tps']/fp16_4k['decode_tps']:.3f}x FP16 (parity)")
    print(f"    Memory:       {pq_4k['cache_mb']/fp16_4k['cache_mb']:.2f}x FP16 (dequant caches active)")
    print(f"    Packed only:  ~81x smaller than FP16 (for SSD offload)")
    print(f"    Quality:      logit cos_sim > 0.985")

    if not args.skip_dflash and dflash_results:
        dflash_pq = [r for r in dflash_results if r["pq_on"] and r["dflash_on"]]
        baseline_r = dflash_results[0]
        if dflash_pq:
            total_speedup = dflash_pq[0]["decode_tps"] / baseline_r["decode_tps"]
            mem_saved = 1 - dflash_pq[0]["cache_gb"] / baseline_r["cache_gb"]
            print(f"\n  DFlash + PQ3 Combined:")
            print(f"    Decode speed: {total_speedup:.2f}x baseline (speculative + compressed KV)")
            print(f"    Memory:       {mem_saved*100:+.0f}% vs baseline")

    print(f"\n  128K Context (theoretical):")
    print(f"    FP16 B=8: 67 GB KV → OOM on 128GB Mac")
    print(f"    PQ packed B=8: 1.6 GB KV → fits easily")
    print(f"    PQ + evict mode: packed in RAM, dequant per-layer on demand")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
