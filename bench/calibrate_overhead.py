# SPDX-License-Identifier: Apache-2.0
"""FU4: calibrate MODEL_OVERHEAD_FACTOR against live loads.

Loads each given model twice (resident, then page-cache-only streaming)
and reports observed resident/streaming bytes vs the estimate, i.e. the
empirical overhead factor per model. Run on the target machine:

    .venv/bin/python bench/calibrate_overhead.py --model <dir> [--model <dir>...]

Output is a table + a suggested constant (max observed resident ratio
with 2 decimals of headroom). It does NOT rewrite residency.py — a
human promotes the value after judging the sample.
"""

from __future__ import annotations

import argparse
import asyncio
import gc

import mlx.core as mx


async def _measure(model_path: str, streaming: bool) -> dict:
    from omlx.engine_pool import EnginePool
    from omlx.model_settings import ModelSettings
    from omlx.patches.expert_streaming.residency import expert_streaming_estimate

    est = expert_streaming_estimate(model_path)
    pool = EnginePool()
    from pathlib import Path

    pool.discover_models(str(Path(model_path).parent))
    settings = ModelSettings(
        expert_streaming_enabled=streaming,
        expert_streaming_budget_gib=(0.0 if streaming else None),
    )
    mx.clear_cache()
    gc.collect()
    before = mx.get_active_memory()
    await pool.get_engine(model_path, runtime_settings=settings)
    await asyncio.sleep(1.0)
    active = mx.get_active_memory()
    observed = max(0, active - before)
    predicted = est.streaming_bytes if streaming else est.resident_bytes
    try:
        await pool.release_engine(model_path)
    except Exception:
        pass
    mx.clear_cache()
    gc.collect()
    return {
        "checkpoint": est.checkpoint_bytes,
        "predicted": predicted,
        "observed": observed,
        "ratio": observed / predicted if predicted else 0.0,
    }


async def _main(models: list[str]) -> int:
    print(f"{'model':40s} {'mode':10s} {'predicted':>12s} {'observed':>12s} {'ratio':>7s}")
    ratios: list[float] = []
    for m in models:
        for streaming in (False, True):
            try:
                r = await _measure(m, streaming)
            except Exception as e:
                print(f"{m[:40]:40s} SKIP: {e}")
                continue
            mode = "stream" if streaming else "resident"
            print(
                f"{m[:40]:40s} {mode:10s} "
                f"{r['predicted'] / 1024**3:11.2f}G {r['observed'] / 1024**3:11.2f}G "
                f"{r['ratio']:6.3f}"
            )
            if not streaming:
                ratios.append(r["ratio"])
    if ratios:
        sug = max(ratios)
        print(f"\nmax resident ratio: {sug:.3f} -> suggested factor: {sug + 0.02:.2f}")
        print("(promote to _MODEL_OVERHEAD_FACTOR in residency.py after review)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_main(args.model)))


if __name__ == "__main__":
    main()
