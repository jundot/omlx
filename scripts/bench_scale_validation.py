#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scale validation: PQ3 vs FP16 at increasing context + DFlash combined."""
from __future__ import annotations
import time, mlx.core as mx

def main():
    model_id = "mlx-community/Qwen3.5-27B-4bit"
    PQ_BITS = 3
    DECODE_STEPS = 32
    CONTEXTS = [80, 2000, 8000, 32000]

    from mlx_lm import load
    from mlx_lm.models import cache as mlx_cache_mod
    from omlx.patches.planarquant_cache import enable_planarquant_cache, disable_planarquant_cache
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
    apply_turboquant_attention_patch()

    print(f"Loading {model_id}...")
    model, tokenizer = load(model_id)

    base = "The history of computing spans centuries, from the abacus to quantum computers. "

    def _bench(pq_on, target_t, dflash_on=False):
        if pq_on:
            enable_planarquant_cache(PQ_BITS)
        else:
            disable_planarquant_cache()

        enc = tokenizer.encode(base)
        reps = max(1, target_t // len(enc) + 1)
        toks = (enc * reps)[:target_t]
        tokens = mx.array(toks)[None, :]
        T = tokens.shape[1]

        # DFlash setup
        draft = None
        if dflash_on:
            try:
                from omlx.patches.dflash import load_dflash_draft, install_dflash_hooks
                draft, ref = load_dflash_draft(model_id)
                if draft:
                    install_dflash_hooks(model, draft_model=draft, target_model=model)
            except Exception:
                pass

        cache = mlx_cache_mod.make_prompt_cache(model)
        mx.eval(tokens)
        t0 = time.perf_counter()
        logits = model(tokens, cache=cache)
        mx.eval(logits)
        prefill_s = time.perf_counter() - t0

        mb = sum(c.nbytes for c in cache if hasattr(c, "nbytes") and isinstance(c.nbytes, int)) / 1e6
        last_logits = logits[0, -1, :]

        # warm
        nt = mx.argmax(last_logits)[None, None]
        for _ in range(4):
            logits = model(nt, cache=cache)
            mx.eval(logits)
            nt = mx.argmax(logits[0, -1, :])[None, None]

        # timed decode
        dt = []
        for _ in range(DECODE_STEPS):
            mx.synchronize()
            t0 = time.perf_counter()
            logits = model(nt, cache=cache)
            mx.eval(logits)
            nt = mx.argmax(logits[0, -1, :])[None, None]
            dt.append(time.perf_counter() - t0)

        tps = DECODE_STEPS / sum(dt)
        step_ms = (sum(dt) / DECODE_STEPS) * 1000
        return T, prefill_s, tps, step_ms, mb, last_logits

    # ================================================================
    # PART 1: PQ3 vs FP16 at scale
    # ================================================================
    print("\n" + "=" * 80)
    print("PART 1: PQ3 vs FP16 — DECODE SPEED + MEMORY + QUALITY")
    print("=" * 80)
    print(f"{'':8} {'T':>6} {'Pre':>7} {'Dec':>7} {'Step':>6} {'MB':>8} {'Spd':>6} {'Mem':>6} {'cos':>7}")
    print(f"{'':8} {'toks':>6} {'tok/s':>7} {'tok/s':>7} {'ms':>6} {'':>8} {'rat':>6} {'rat':>6} {'sim':>7}")
    print("-" * 70)

    for target in CONTEXTS:
        T, pf, f_tps, f_ms, f_mb, f_logits = _bench(False, target)
        T, pf, p_tps, p_ms, p_mb, p_logits = _bench(True, target)

        sr = p_tps / f_tps if f_tps else 0
        mr = p_mb / f_mb if f_mb else 0

        fp16_l = f_logits.astype(mx.float32)
        pq_l = p_logits.astype(mx.float32)
        d = float(mx.sum(fp16_l * pq_l).item())
        n0 = float(mx.sqrt(mx.sum(fp16_l * fp16_l)).item())
        n1 = float(mx.sqrt(mx.sum(pq_l * pq_l)).item())
        cs = d / (n0 * n1 + 1e-10)

        print(f"  FP16   {T:>6} {T/pf:>7.0f} {f_tps:>7.1f} {f_ms:>6.2f} {f_mb:>8.1f} {'1.00':>5}x {'1.00':>5}x")
        print(f"  PQ3    {T:>6} {T/pf:>7.0f} {p_tps:>7.1f} {p_ms:>6.2f} {p_mb:>8.1f} {sr:>5.3f}x {mr:>5.3f}x {cs:>7.6f}")
        print()

    # ================================================================
    # PART 2: 128K theoretical memory
    # ================================================================
    print("=" * 80)
    print("PART 2: MEMORY AT 128K CONTEXT (Qwen3.5-27B, 4 KV heads, D=128, 64 layers)")
    print("=" * 80)
    L, H, D = 64, 4, 256
    for T in [4096, 32768, 131072]:
        for B in [1, 4, 8]:
            fp16 = T * L * H * D * 4 * B / 1e9
            packed = T * L * H * 96 * 2 * B / 1e9  # 96 bytes per 256-elem block per head
            fits = "YES" if 15 + fp16 < 120 else "OOM"
            pf = "YES" if 15 + packed < 120 else "OOM"
            print(f"  T={T//1024:>5}K B={B}: FP16={fp16:>6.1f}GB ({fits})  PQ packed={packed:>5.1f}GB ({pf})  savings={fp16/packed:.0f}x")
        print()

    # ================================================================
    # PART 3: DFlash + PQ
    # ================================================================
    print("=" * 80)
    print("PART 3: DFLASH + PQ3 COMBINED (T=~4K)")
    print("=" * 80)

    configs = [
        ("FP16 baseline", False, False),
        ("PQ3 only", True, False),
        ("DFlash only (FP16)", False, True),
        ("DFlash + PQ3", True, True),
    ]

    print(f"  {'Config':<25} {'Decode':>8} {'Step':>7} {'Cache':>7} {'vs base':>8}")
    print(f"  {'':25} {'tok/s':>8} {'ms':>7} {'MB':>7} {'speedup':>8}")
    print("  " + "-" * 55)

    baseline_tps = None
    for name, pq, df in configs:
        try:
            T, pf, tps, ms, mb, _ = _bench(pq, 4000, dflash_on=df)
            if baseline_tps is None:
                baseline_tps = tps
            vs = f"{tps/baseline_tps:.2f}x"
            print(f"  {name:<25} {tps:>8.1f} {ms:>7.2f} {mb:>7.1f} {vs:>8}")
        except Exception as e:
            print(f"  {name:<25}  FAILED: {e}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    disable_planarquant_cache()


if __name__ == "__main__":
    main()
