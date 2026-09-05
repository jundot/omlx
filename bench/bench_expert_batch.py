"""Fase A3 bench: expert streaming under concurrent (batched) decode.

Fires N concurrent chat requests through the EnginePool so the scheduler's
continuous batching unions their experts per step — the amortization
hypothesis (uniq experts grows sublinearly with batch size, so bytes/token
falls).

Usage:
    .venv/bin/python bench/bench_expert_batch.py --model glm --budget 8 --concurrency 4 --decode 16 --out bench/results/glm_b4_8g.json

Controls:
    OMLX_EXPERT_STREAMING_PROFILE=1  (per-stage profiling per layer)
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_expert_streaming import FakeEnforcer, DEFAULT_ENTRIES, find_streaming_cache  # noqa: E402

PROMPT_POOL = [
    "Explain photosynthesis in one sentence.",
    "What is the capital of France?",
    "Write a haiku about rain.",
    "Why is the sky blue?",
    "Name three primary colors.",
    "How do airplanes fly?",
    "What is 17 times 23?",
    "Describe the water cycle briefly.",
]


async def run(model_key: str, budget: float, concurrency: int, decode: int, out: str | None):
    from omlx.engine_pool import EnginePool
    from omlx.model_settings import ModelSettings
    from omlx.scheduler import SchedulerConfig
    from omlx.utils.proc_memory import get_phys_footprint
    import mlx.core as mx
    from resource_sampler import ResourceSampler

    entry_name = DEFAULT_ENTRIES[model_key]
    print(f"=== {model_key} batch {concurrency} budget {budget}G decode {decode} ===", flush=True)

    pool = EnginePool(scheduler_config=SchedulerConfig(hot_cache_max_size=0))
    pool.discover_models("/Volumes/SSD 4TB/AI Models")
    entry = pool.get_entry(entry_name)
    if not entry:
        print("entry not found")
        return

    settings = ModelSettings(
        expert_streaming_enabled=True,
        expert_streaming_budget_gib=budget,
        qwen4_ple_ssd_offload=True,
    )
    runtime = pool._entry_runtime_resident_size(entry, settings)
    print(f"runtime est {runtime / 1024**3:.2f}G", flush=True)

    phys0 = get_phys_footprint() / 1024**3
    t0 = time.perf_counter()
    engine = await pool.get_engine(entry_name, runtime_settings=settings)
    t_load = time.perf_counter() - t0
    print(f"engine loaded {t_load:.1f}s phys {get_phys_footprint() / 1024**3:.2f}G", flush=True)

    vlm_model = getattr(engine, "_vlm_model", None)
    cache = find_streaming_cache(vlm_model)

    sampler = ResourceSampler(
        interval=1.0,
        mlx_callbacks={
            "mlx_active_gib": mx.get_active_memory,
            "mlx_cache_gib": mx.get_cache_memory,
        },
    )
    sampler.start()

    prompts = [PROMPT_POOL[i % len(PROMPT_POOL)] for i in range(concurrency)]

    # Warm the tokenizer/template pipeline once (1 token).
    await engine.chat([{"role": "user", "content": prompts[0]}], max_tokens=1, temperature=0.0)

    sampler.mark("batch")
    t1 = time.perf_counter()
    tasks = [
        engine.chat([{"role": "user", "content": p}], max_tokens=decode, temperature=0.0)
        for p in prompts
    ]
    outputs = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t1
    sampler.mark("single")
    batch_tokens = sum(o.completion_tokens or 0 for o in outputs)
    batch_tok_s = batch_tokens / wall if wall > 0 else 0.0
    print(
        f"batch {concurrency}: {batch_tokens} tok in {wall:.1f}s -> aggregate {batch_tok_s:.3f} tok/s"
        f" ({batch_tok_s / concurrency:.3f} per request)",
        flush=True,
    )

    # In-process single-request reference (warm cache — conservative comparison).
    sampler.mark("decode")
    t2 = time.perf_counter()
    out_single = await engine.chat(
        [{"role": "user", "content": prompts[0]}], max_tokens=decode, temperature=0.0
    )
    single_wall = time.perf_counter() - t2
    sampler.mark("teardown")
    sampler.stop()
    single_tokens = out_single.completion_tokens or 0
    single_tok_s = single_tokens / single_wall if single_wall > 0 else 0.0
    print(f"single (warm): {single_tokens} tok in {single_wall:.1f}s -> {single_tok_s:.3f} tok/s", flush=True)

    res_summary = sampler.summary()
    print(f"resources {res_summary['phases']}")
    json.dump(
        sampler.samples(),
        open(f"bench/results/{model_key}_batch{concurrency}_{budget}g_samples.json", "w"),
    )

    stats = profile = None
    if cache is not None:
        stats = {
            "hits": cache.stats.hits,
            "misses": cache.stats.misses,
            "evictions": cache.stats.evictions,
            "hit_rate": cache.stats.hit_rate(),
            "size": cache.size,
            "capacity": cache.capacity,
        }
        print(f"cache {stats}")
        if cache.profile.enabled:
            profile = cache.profile.report()
            print(f"profile totals {profile['totals']}")

    results = {
        "model": model_key,
        "budget_gib": budget,
        "concurrency": concurrency,
        "decode": decode,
        "runtime_est_gib": runtime / 1024**3,
        "load_s": round(t_load, 1),
        "phys_after_load_gib": round(phys0, 2),
        "batch_tokens": batch_tokens,
        "batch_wall_s": round(wall, 2),
        "batch_aggregate_tok_s": round(batch_tok_s, 4),
        "batch_per_request_tok_s": round(batch_tok_s / concurrency, 4),
        "single_warm_tok_s": round(single_tok_s, 4),
        "speedup_vs_warm_single": round(batch_tok_s / single_tok_s, 3) if single_tok_s else None,
        "cache_stats": stats,
        "profile": profile,
        "resources": res_summary,
        "per_request_tokens": [o.completion_tokens or 0 for o in outputs],
    }

    await pool.release_engine(entry_name)
    await pool._unload_engine(entry_name)

    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {out}")
    print("=== DONE ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen", "glm"])
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    asyncio.run(run(args.model, args.budget, args.concurrency, args.decode, args.out))


if __name__ == "__main__":
    main()
