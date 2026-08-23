#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Opt-in full-model DeepSeek-V4-Flash ANE/CPU/GPU prefill benchmark.

The reference checkpoint has been observed above 100 GiB after load and does
not safely fit a 96 GiB system. The script refuses to load any weights unless
``--allow-large-model`` is supplied. Use ``dsv4_ane_shape_bench.py`` for the
default sub-GiB microbenchmarks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


def _median_prefill_ms(mx: Any, model: Any, tokens: Any, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = model.model(tokens)
        mx.eval(output)
        mx.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-fraction", type=float, default=0.125)
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--disable-cpu-shared-resource", action="store_true")
    parser.add_argument(
        "--allow-large-model",
        action="store_true",
        help="Explicitly allow loading a checkpoint that may exceed 96 GiB",
    )
    args = parser.parse_args()
    if not args.allow_large_model:
        parser.error(
            "full-model loading is disabled by default; pass --allow-large-model "
            "only on a host with enough unified memory"
        )
    if args.tokens < 1024 or args.tokens % 64:
        parser.error("--tokens must be a multiple of 64 and at least 1024")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    import mlx.core as mx

    from omlx.patches.deepseek_v4.ane_prefill import (
        enable_deepseek_v4_ane_prefill,
    )
    from omlx.utils.model_loading import load_text_model

    print(
        "WARNING: full DeepSeek-V4-Flash validation has exceeded 100 GiB in "
        "practice; memory pressure or process termination is possible.",
        flush=True,
    )
    print(f"Loading {args.model}", flush=True)
    model, _ = load_text_model(str(args.model))
    mx.random.seed(0)
    tokens = mx.random.randint(0, 1000, (1, args.tokens), dtype=mx.int32)
    mx.eval(tokens)

    gpu_ms = _median_prefill_ms(mx, model, tokens, args.repeats)
    started = time.perf_counter()
    procedures = enable_deepseek_v4_ane_prefill(
        model,
        sequence_length=args.tokens,
        cpu_fraction=args.cpu_fraction,
        cpu_threads=args.cpu_threads,
        cpu_shared_resource=not args.disable_cpu_shared_resource,
    )
    compile_seconds = time.perf_counter() - started
    if not procedures:
        raise SystemExit("DeepSeek ANE prefill did not attach to this model")
    hybrid_ms = _median_prefill_ms(mx, model, tokens, args.repeats)

    print(
        json.dumps(
            {
                "model": str(args.model),
                "tokens": args.tokens,
                "repeats": args.repeats,
                "cpu_fraction": args.cpu_fraction,
                "cpu_threads": args.cpu_threads,
                "cpu_shared_resource": not args.disable_cpu_shared_resource,
                "procedures": procedures,
                "cpu_procedures": int(getattr(model, "_omlx_ane_cpu_prefill_count", 0)),
                "compile_seconds": compile_seconds,
                "gpu_prefill_ms": gpu_ms,
                "hybrid_prefill_ms": hybrid_ms,
                "speedup": gpu_ms / hybrid_ms,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
