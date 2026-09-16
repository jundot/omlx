# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 expert offload on one machine: load, prefill, decode.

Loads a V4.1 checkpoint through the oMLX loader with Engram on SSD and MoE
expert offload at each requested resident fraction, then runs a chunked
prefill and a greedy decode directly on the model (no scheduler), reporting
load time, Metal and process memory, time to first token, decode speed, and
the expert cache's hit rate and fetch throughput per phase. Prints the
resident set per fraction from the shard headers and the largest fraction
that fits the Metal working-set limit first, so a run that cannot fit is
visible before any weights load::

    python benchmarks/deepseek_v41_offload_bench.py --model /path/to/oQ3e \\
        --fractions 0.125 0.25 --prompt-tokens 512 --decode-tokens 32
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx

from omlx.utils.proc_memory import get_lifetime_max_phys_footprint, get_phys_footprint


def _offload_stats(model) -> dict:
    hits = misses = fetched = 0
    for layer in model.language_model.layers:
        slots = getattr(layer.ffn.experts, "slots", None)
        if slots is None:
            continue
        hits += slots.hits
        misses += slots.misses
        fetched += slots.fetched_bytes
    return {"hits": hits, "misses": misses, "fetched_bytes": fetched}


def _delta(after: dict, before: dict, seconds: float) -> dict:
    misses = after["misses"] - before["misses"]
    hits = after["hits"] - before["hits"]
    fetched = after["fetched_bytes"] - before["fetched_bytes"]
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / max(1, hits + misses),
        "fetched_gib": fetched / 2**30,
        "fetch_gbps": fetched / max(seconds, 1e-9) / 1e9,
    }


def sizing(path: Path, budget: int, engram_ssd: bool) -> dict:
    from omlx.patches.deepseek_v41.moe_offload import (
        _plan,
        admission_bytes,
        fit_resident_fraction,
    )

    plan = _plan(path, 1.0)
    rows = []
    for fraction in (0.125, 0.25, 1 / 3, 0.375, 0.5, 1.0):
        capacity = min(plan.count, max(plan.floor, round(plan.count * fraction)))
        rows.append(
            {
                "fraction": fraction,
                "experts_per_layer": capacity,
                "admission_gib": admission_bytes(
                    path, fraction, engram_ssd_offload=engram_ssd
                )
                / 2**30,
            }
        )
    fit = fit_resident_fraction(path, budget, engram_ssd_offload=engram_ssd)
    print(
        f"experts {plan.full_bytes / 2**30:.1f} GiB, draft {plan.draft_bytes / 2**30:.1f} GiB, "
        f"budget {budget / 2**30:.1f} GiB"
    )
    for row in rows:
        print(
            f"  {row['fraction'] * 100:5.1f}%  {row['experts_per_layer']:4d} experts/layer  "
            f"admission {row['admission_gib']:6.1f} GiB"
        )
    print(
        "largest fraction within budget: "
        + (
            "none"
            if fit is None
            else f"{fit:.4f} ({round(fit * plan.count)} experts/layer)"
        )
    )
    return {"rows": rows, "fit": fit, "budget": budget}


def run(path: Path, fraction: float, args) -> dict:
    from omlx.patches.deepseek_v41.loading import load

    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    model, processor = load(
        path,
        engram_ssd_offload=args.engram == "ssd",
        moe_expert_offload_resident_fraction=fraction,
    )
    load_s = time.perf_counter() - t0
    result = {
        "fraction": fraction,
        "capacity": model._moe_offload_plan.capacity,
        "load_s": load_s,
        "after_load": {
            "active_gib": mx.get_active_memory() / 2**30,
            "peak_gib": mx.get_peak_memory() / 2**30,
            "footprint_gib": get_phys_footprint() / 2**30,
        },
    }
    print(
        f"[{fraction:.4f}] loaded in {load_s:.1f} s: {result['capacity']} experts/layer, "
        f"active {result['after_load']['active_gib']:.1f} GiB, "
        f"peak {result['after_load']['peak_gib']:.1f} GiB, footprint {result['after_load']['footprint_gib']:.1f} GiB"
    )
    try:
        tokenizer = processor.tokenizer
        text = (args.prompt + " ") * (args.prompt_tokens // 8 + 1)
        ids = tokenizer.encode(text)[: args.prompt_tokens]
        cache = model.language_model.make_cache()
        before = _offload_stats(model)
        t0 = time.perf_counter()
        logits = None
        for start in range(0, len(ids), args.prefill_chunk):
            chunk = mx.array([ids[start : start + args.prefill_chunk]])
            logits = model(chunk, cache=cache)
            mx.eval(logits)
        ttft = time.perf_counter() - t0
        result["prefill"] = {
            "tokens": len(ids),
            "seconds": ttft,
            "tok_s": len(ids) / ttft,
            **_delta(_offload_stats(model), before, ttft),
        }
        p = result["prefill"]
        print(
            f"[{fraction:.4f}] prefill {p['tokens']} tokens: {ttft:.1f} s "
            f"({p['tok_s']:.1f} tok/s), hit rate {p['hit_rate']:.2f}, "
            f"fetched {p['fetched_gib']:.1f} GiB at {p['fetch_gbps']:.2f} GB/s"
        )
        token = int(mx.argmax(logits[0, -1]).item())
        generated = [token]
        before = _offload_stats(model)
        t0 = time.perf_counter()
        for _ in range(args.decode_tokens - 1):
            logits = model(mx.array([[token]]), cache=cache)
            token = int(mx.argmax(logits[0, -1]).item())
            generated.append(token)
        decode_s = time.perf_counter() - t0
        result["decode"] = {
            "tokens": len(generated) - 1,
            "seconds": decode_s,
            "tok_s": (len(generated) - 1) / decode_s,
            **_delta(_offload_stats(model), before, decode_s),
        }
        result["generated"] = tokenizer.decode(generated)
        d = result["decode"]
        print(
            f"[{fraction:.4f}] decode {d['tokens']} tokens: {d['tok_s']:.2f} tok/s, "
            f"hit rate {d['hit_rate']:.2f}, fetched {d['fetched_gib']:.1f} GiB at "
            f"{d['fetch_gbps']:.2f} GB/s"
        )
        print(f"[{fraction:.4f}] text: {result['generated'][:200]!r}")
        result["after_run"] = {
            "active_gib": mx.get_active_memory() / 2**30,
            "peak_gib": mx.get_peak_memory() / 2**30,
            "footprint_gib": get_phys_footprint() / 2**30,
            "max_footprint_gib": get_lifetime_max_phys_footprint() / 2**30,
        }
        a = result["after_run"]
        print(
            f"[{fraction:.4f}] after run: active {a['active_gib']:.1f} GiB, "
            f"peak {a['peak_gib']:.1f} GiB, footprint {a['footprint_gib']:.1f} GiB, "
            f"max footprint {a['max_footprint_gib']:.1f} GiB"
        )
    finally:
        model.close()
        del model, processor
        gc.collect()
        mx.clear_cache()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.125])
    ap.add_argument("--engram", choices=["ssd", "ram"], default="ssd")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument(
        "--prompt",
        default="The expert offload path streams routed experts from the checkpoint on demand.",
    )
    ap.add_argument("--budget-gib", type=float, default=None)
    ap.add_argument("--sizing-only", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if args.prompt_tokens <= 0 or args.prefill_chunk <= 0 or args.decode_tokens < 1:
        ap.error(
            "--prompt-tokens and --prefill-chunk must be positive, --decode-tokens >= 1"
        )
    if any(not 0 < fraction <= 1 for fraction in args.fractions):
        ap.error("--fractions must be in (0, 1]")
    budget = (
        int(args.budget_gib * 2**30)
        if args.budget_gib is not None
        else int(mx.device_info()["max_recommended_working_set_size"])
    )
    report = {
        "model": str(args.model),
        "sizing": sizing(args.model, budget, args.engram == "ssd"),
    }
    if not args.sizing_only:
        report["runs"] = [
            run(args.model, fraction, args) for fraction in args.fractions
        ]
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
