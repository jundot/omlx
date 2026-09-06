#!/usr/bin/env python3
"""Compare the current CPU staging path with Metal FRL plus bank blit."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from omlx.expert_streaming.safetensors import ExpertReader, SafetensorExpertIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.experts <= 0:
        parser.error("--experts must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    index = SafetensorExpertIndex(args.model.expanduser().resolve())
    locations = index.layer(index.expert_layer_ids()[0])
    count = locations[next(iter(locations))].shape[0]
    if args.experts > count:
        parser.error(f"--experts cannot exceed the checkpoint's {count} experts")
    experts = (
        [0]
        if args.experts == 1
        else [
            round(i * (count - 1) / (args.experts - 1))
            for i in range(args.experts)
        ]
    )
    cpu = ExpertReader(index)
    frl = ExpertReader(index, fast_resource_loading=True)
    targets = {
        key: mx.zeros(
            (args.experts, *location.shape[1:]),
            dtype={
                "U32": mx.uint32,
                "U8": mx.uint8,
                "I8": mx.int8,
                "F32": mx.float32,
                "F16": mx.float16,
                "BF16": mx.bfloat16,
            }[location.dtype],
        )
        for key, location in locations.items()
    }
    mx.eval(*targets.values())
    cpu_times: list[float] = []
    frl_times: list[float] = []
    exact = True
    try:
        for _ in range(args.repeats):
            started = time.perf_counter()
            components = cpu.read_many(locations, experts, use_file_cache=True)
            for key, rows in components.items():
                targets[key][:] = rows
            mx.eval(*targets.values())
            cpu_times.append(time.perf_counter() - started)

            expected = {key: mx.array(value) for key, value in targets.items()}
            mx.eval(*expected.values())
            started = time.perf_counter()
            load = frl.begin_fast_many(locations, experts)
            frl.finish_fast_many(
                load,
                list(range(args.experts)),
                {key: (target, 0) for key, target in targets.items()},
            )
            mx.synchronize()
            frl_times.append(time.perf_counter() - started)
            exact &= all(
                bool(mx.array_equal(targets[key], expected[key]).item())
                for key in targets
            )
    finally:
        cpu.close()
        frl.close()
    cpu_median = statistics.median(cpu_times)
    frl_median = statistics.median(frl_times)
    print(
        json.dumps(
            {
                "experts": args.experts,
                "bytes_per_wave": frl.fast_bytes_read // args.repeats,
                "cpu_median_ms": round(cpu_median * 1000, 3),
                "frl_median_ms": round(frl_median * 1000, 3),
                "speedup": round(cpu_median / frl_median, 3),
                "byte_exact": exact,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
