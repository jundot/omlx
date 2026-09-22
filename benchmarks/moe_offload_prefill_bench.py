# SPDX-License-Identifier: Apache-2.0
"""MoE expert offload (common adapter) on one machine: prefill and decode.

Loads a checkpoint lazily through the oMLX loader, wraps its experts with
``apply_moe_expert_offload`` at each requested resident fraction, then runs
one prompt twice directly on the model (no scheduler, no chat template):
cold (first request after load, pays the initial fill) and warm (same prompt
again). Reports time to first token, expert fetches through the first yielded token,
decode speed, and process memory. Meant for before/after comparison of the
over-capacity prefill path; the git revision is recorded in the output::

    python benchmarks/moe_offload_prefill_bench.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --fractions 0.125 0.25 0.5 --prompt-tokens 585 --decode-tokens 32
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from omlx.utils.proc_memory import get_phys_footprint

TEXT = (
    "The expert tables of a mixture-of-experts model are mostly idle on any "
    "given token, so keeping a fraction resident and streaming the rest from "
    "the checkpoint trades latency for memory without changing which expert "
    "runs. "
)


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _prompt(tok, n_tokens: int) -> list[int]:
    ids: list[int] = []
    while len(ids) < n_tokens:
        ids = tok.encode(TEXT * (1 + len(ids) // 40 + n_tokens // 40))
    return ids[:n_tokens]


def _first_capacity(model) -> int | None:
    """Slot capacity of the first offloaded layer (module names vary by family)."""
    stack, seen = [model], set()
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        cache = getattr(obj, "cache", None)
        if getattr(cache, "moe_offload_cache", False):
            return cache.capacity
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
    return None


def _one_request(model, tok, prompt: list[int], decode_tokens: int) -> dict:
    from mlx_lm import stream_generate

    from omlx.patches.moe_expert_offload import moe_offload_stats

    before = moe_offload_stats(model)
    t0 = time.perf_counter()
    ttft = None
    gen_tps = None
    n = 0
    for resp in stream_generate(model, tok, prompt=prompt, max_tokens=decode_tokens):
        if ttft is None:
            ttft = time.perf_counter() - t0
            # The generator can run a decode step before its first yield.
            # These counters describe that yield boundary, not pure prefill.
            at_first_token = moe_offload_stats(model)
        n += 1
        gen_tps = resp.generation_tps
    total = time.perf_counter() - t0
    after = moe_offload_stats(model)
    return {
        "ttft_s": ttft,
        "first_token_fetches": at_first_token["misses"] - before["misses"],
        "first_token_hits": at_first_token["hits"] - before["hits"],
        "after_first_token_fetches": after["misses"] - at_first_token["misses"],
        "decode_tokens": n,
        "decode_tps": gen_tps,
        "total_s": total,
    }


def run_fraction(
    model_repo: str, fraction: float, prompt_tokens: int, decode_tokens: int
) -> dict:
    from omlx.patches.moe_expert_offload import apply_moe_expert_offload
    from omlx.utils.model_loading import lm_load_compat, materialize_lazy_state

    t0 = time.perf_counter()
    model, tok = lm_load_compat(model_repo, lazy=True)
    wrapped = apply_moe_expert_offload(model, model_repo, fraction)
    materialize_lazy_state(model)
    mx.synchronize()
    load_s = time.perf_counter() - t0
    if wrapped == 0:
        raise SystemExit(
            f"no layers wrapped at fraction {fraction}; nothing to measure"
        )
    prompt = _prompt(tok, prompt_tokens)
    capacity = _first_capacity(model)
    result = {
        "fraction": fraction,
        "layers_wrapped": wrapped,
        "capacity": capacity,
        "load_s": round(load_s, 2),
        "prompt_tokens": len(prompt),
        "cold": _one_request(model, tok, prompt, decode_tokens),
        "warm": _one_request(model, tok, prompt, decode_tokens),
        "footprint_gib": round(get_phys_footprint() / 2**30, 2),
        "metal_peak_gib": round(mx.get_peak_memory() / 2**30, 2),
    }
    del model, tok
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.125, 0.25, 0.5])
    ap.add_argument("--prompt-tokens", type=int, default=585)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    out = {"model": args.model, "git": _git_rev(), "mlx": mx.__version__, "runs": []}
    print(f"model {args.model}  rev {out['git']}  mlx {out['mlx']}")
    print(
        "| residency | capacity | TTFT cold | TTFT warm | first-token fetches cold / warm "
        "| decode tok/s | process GiB |"
    )
    print("|---|---|---|---|---|---|---|")
    for fraction in args.fractions:
        r = run_fraction(args.model, fraction, args.prompt_tokens, args.decode_tokens)
        out["runs"].append(r)
        c, w = r["cold"], r["warm"]
        print(
            f"| {100 * fraction:g}% | {r['capacity']} | {c['ttft_s']:.2f} s | "
            f"{w['ttft_s']:.2f} s | {c['first_token_fetches']} / {w['first_token_fetches']} | "
            f"{w['decode_tps']:.1f} | {r['footprint_gib']} |",
            flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
