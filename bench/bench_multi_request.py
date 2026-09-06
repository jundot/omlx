# SPDX-License-Identifier: Apache-2.0
"""Multi-request (concurrent) MoE streaming bench.

The single-request harness answers per-request latency; this one answers
THROUGHPUT: N concurrent stream_chat calls against the same engine (the
BatchedEngine batches decode), sharing one app-level expert cache. This is
where an app-level LRU budget should beat page-cache-only: recency ordering
across requests. Run on the target machine:

    .venv/bin/python bench/bench_multi_request.py --model <dir> \
        --budget 2 --concurrency 4 --prompt-len 512 --decode 96

"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time


async def _one_request(engine, messages, decode: int, tag: int) -> dict:
    t0 = time.perf_counter()
    first = None
    n = 0
    async for out in engine.stream_chat(messages, max_tokens=decode, temperature=0.0):
        if first is None and (out.completion_tokens > 0 or out.new_text):
            first = time.perf_counter()
        n = int(out.completion_tokens)
    end = time.perf_counter()
    ttft = (first or end) - t0
    dec_s = end - (first or end)
    return {"tag": tag, "ttft_s": ttft, "decode_s": dec_s, "tokens": n,
            "tok_s": n / dec_s if dec_s > 0 else 0.0}


_FILLER = (
    "The scientist wrote a detailed report about the river ecosystem, "
    "describing seasonal changes, sediment transport, and the fish "
    "population dynamics observed across several years of field work. "
)


def _make_prompt(plen: str, seed: int) -> list[dict]:
    import random

    rng = random.Random(seed)
    targets = {"short": 1, "512": 8, "2k": 32, "8k": 128}
    body = " ".join(rng.choice([_FILLER, _FILLER.replace("The scientist", "A researcher"),
                                 _FILLER.replace("river", "forest")])
                     for _ in range(targets[plen] * 4))
    body = " ".join(body.split()[: max(1, targets[plen] * 7)])
    return [{"role": "user", "content": f"{body}\nSummarize the passage in one sentence."}]


async def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--cache-policy", choices=["lru", "s3fifo"], default=None,
                    help="page-cache-only budgets ignore it; default leaves the env untouched")
    ap.add_argument("--prompt-len", choices=["short", "512", "2k", "8k"], default="512")
    ap.add_argument("--decode", type=int, default=96)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--runs", type=int, default=1, help="repeat the wave; cache stays warm across runs")
    ap.add_argument("--min-free-gb", type=float, default=22.0)
    ap.add_argument("--mem-ceiling-gib", type=float, default=28.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cache_policy:
        os.environ["OMLX_EXPERT_STREAMING_CACHE"] = args.cache_policy

    try:
        import psutil

        free_gb = psutil.virtual_memory().available / 1024**3
        if free_gb < args.min_free_gb:
            raise SystemExit(f"bench aborted: only {free_gb:.1f} GB available")
    except ImportError:
        pass

    from omlx.engine_pool import EnginePool
    from omlx.model_settings import ModelSettings
    from omlx.patches.expert_streaming import expert_streaming_summary

    settings = ModelSettings(
        expert_streaming_enabled=True,
        expert_streaming_budget_gib=args.budget,
    )
    pool = EnginePool()
    from pathlib import Path

    pool.discover_models(str(Path(args.model).parent))
    model_id = Path(args.model).name
    t0 = time.perf_counter()
    engine = await pool.get_engine(model_id, runtime_settings=settings)
    load_s = time.perf_counter() - t0
    print(f"engine loaded in {load_s:.1f}s")

    results = []
    for run in range(args.runs):
        t_wave = time.perf_counter()
        tasks = [_one_request(engine, _make_prompt(args.prompt_len, 1000 + run * 97 + i),
                              args.decode, i)
                 for i in range(args.concurrency)]
        outs = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t_wave
        total_tok = sum(o["tokens"] for o in outs)
        ttfts = [o["ttft_s"] for o in outs]
        print(f"run {run}: wall={wall:.1f}s total_tok={total_tok} "
              f"agg={total_tok / wall:.2f} tok/s ttft_med={sorted(ttfts)[len(ttfts)//2]:.1f}s")
        results.append({"run": run, "wall_s": wall, "total_tokens": total_tok,
                        "agg_tok_s": total_tok / wall,
                        "ttft_med_s": sorted(ttfts)[len(ttfts)//2],
                        "requests": outs})

    backing = getattr(engine, "_expert_streaming_backing", None)
    cache = getattr(backing, "_streaming_cache", None) if backing is not None else None
    summary = expert_streaming_summary(cache, backing) if backing is not None else {}
    print("cache summary:", json.dumps(summary))

    try:
        await pool.release_engine(model_id)
    except Exception:
        pass

    payload = {
        "model": args.model,
        "budget_gib": args.budget,
        "cache_policy": args.cache_policy or os.environ.get("OMLX_EXPERT_STREAMING_CACHE", "lru"),
        "prompt_len": args.prompt_len,
        "decode": args.decode,
        "concurrency": args.concurrency,
        "runs": args.runs,
        "load_s": load_s,
        "waves": results,
        "cache_summary": summary,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
