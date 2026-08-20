#!/usr/bin/env python
"""Quick laguna decode bench borrowing dflash_mlx's measurement core.

Measures end-to-end decode tok/s on the Poolside Laguna-XS-2.1-NVFP4-mlx
model with the ported laguna_nvfp4 custom kernels OFF vs ON
(OMLX_LAGUNA_NVFP4_KERNELS), using the same paired-run / perf_counter_ns /
memory-snapshot pattern as dflash_mlx.benchmark.

Usage:
  python tools/qwen38_mtp/bench_laguna_kernels.py [--tokens 128] [--runs 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/Users/gg/Documents/GitHub/omlx")

MODEL = (
    os.environ.get(
        "LAGUNA_MODEL",
        "/Users/gg/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4-mlx"
        "/snapshots/841778bda563a36104dd521e37d99218e46f4f25",
    )
)
PROMPT = os.environ.get(
    "LAGUNA_PROMPT",
    "The quick brown fox jumps over the lazy dog. Machine learning systems "
    "process large amounts of text every day, and efficient inference is "
    "critical for real-world deployment at scale across many different "
    "hardware platforms and workloads.",
)


def _thermal_pressure() -> str:
    try:
        import subprocess

        return subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=3
        ).stdout.strip().splitlines()[0][:80]
    except Exception:
        return "n/a"


def _memory_gb() -> float | None:
    try:
        import subprocess

        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=3
        ).stdout
        free = int(out.split("free:", 1)[1].split()[0].strip("."))
        page = 16384
        return free * page / 1e9
    except Exception:
        return None


def _run_once(model, tokenizer, prompt_ids, max_tokens, no_eos=True) -> dict:
    """One timed decode run (borrows dflash's perf_counter_ns measurement)."""
    orig_eos = (getattr(tokenizer, "eos_token_ids", None),
                getattr(tokenizer, "eos_token_id", None))
    if no_eos:
        try:
            tokenizer.eos_token_ids = set()
        except Exception:
            tokenizer.eos_token_ids = []
        try:
            tokenizer.eos_token_id = None
        except Exception:
            pass
    from mlx_lm.generate import batch_generate

    ids = []
    start_ns = time.perf_counter_ns()
    try:
        resp = batch_generate(
            model, tokenizer, prompts=[prompt_ids],
            max_tokens=[max_tokens], return_token_ids=True, verbose=False,
        )
        ids = resp.token_ids[0]
    finally:
        if no_eos:
            tokenizer.eos_token_ids = orig_eos[0]
            tokenizer.eos_token_id = orig_eos[1]
    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    return {
        "elapsed_us": elapsed_us,
        "generation_tokens": len(ids),
        "tok_s": len(ids) / (elapsed_us / 1e6) if elapsed_us > 0 else 0.0,
        "ms_tok": elapsed_us / 1e3 / len(ids) if ids else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    import json as j

    import mlx.core as mx
    from mlx_lm import load

    from omlx.patches.laguna import apply_laguna_patch
    from omlx.utils.model_loading import normalize_laguna_compressed_quant

    apply_laguna_patch()
    cfg = j.load(open(os.path.join(MODEL, "config.json")))
    normalize_laguna_compressed_quant(cfg)

    results = {"model": MODEL, "tokens": args.tokens, "runs": args.runs,
               "thermal": _thermal_pressure(), "free_gb": _memory_gb(),
               "kernels_off": [], "kernels_on": []}

    # interleaved A/B: load both once, alternate off/on so thermal drift
    # and load-order effects are cancelled
    loaded = {}
    for kernels_on in (False, True):
        os.environ["OMLX_LAGUNA_NVFP4_KERNELS"] = "1" if kernels_on else "0"
        model_loaded, tok = load(MODEL, model_config=cfg)
        loaded[kernels_on] = model_loaded
    prompt_ids = tok.encode(PROMPT)
    for k, (torig) in loaded.items():
        _run_once(torig, tok, prompt_ids, 8)  # warmup both models

    for i in range(args.runs):
        for kernels_on in (True, False):  # alternate on-first, then off
            os.environ["OMLX_LAGUNA_NVFP4_KERNELS"] = "1" if kernels_on else "0"
            model = loaded[kernels_on]
            # capture the model's kernel flag at load time; env is read in
            # the wiring paths too, so re-set before each call
            r = _run_once(model, tok, prompt_ids, args.tokens)
            label = "on" if kernels_on else "off"
            results[f"kernels_{label}"].append(r)
            print(f"[kernels={label}] {r['generation_tokens']} tok "
                  f"{r['ms_tok']:.2f} ms/tok ({r['tok_s']:.1f} tok/s)",
                  flush=True)

    def med(vals):
        vs = sorted(v["tok_s"] for v in vals)
        return vs[len(vs) // 2]

    off, on = med(results["kernels_off"]), med(results["kernels_on"])
    print(f"\nmedian tok/s: kernels-off {off:.1f} | kernels-on {on:.1f} "
          f"| delta {(on / off - 1) * 100:+.1f}%")

    out = os.path.expanduser("~/laguna_nvfp4_bench.json")
    json.dump(results, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
