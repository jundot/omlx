#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sweep bit-exact DS4 target/draft vocabulary-head schedules.

The production DSpark path uses the same dense Metal body for three small-M
vocabulary projections:

* target verification: M=6, BF16 output;
* the next draft block: M=5, FP32 output;
* each Markov correction: M=1, FP32 output with K=256.

This probe changes only outer output ownership: output rows per simdgroup and
simdgroups per threadgroup.  Each output keeps production's four consecutive
K values per lane, 128-value K stride, FP32 accumulation, and explicit
shuffle-down reduction.  Every candidate must therefore match the current
kernel with ``mx.array_equal`` before its timing can be considered.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from omlx.patches.deepseek_v4 import verify_qmv


class _WeightOnlyLinear:
    """Small module surface consumed by the production head helpers."""

    def __init__(self, weight: mx.array) -> None:
        self.weight = weight

    def __contains__(self, _name: str) -> bool:
        return False


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _evaluate(fn: Callable[[], mx.array]) -> None:
    value = fn()
    mx.eval(value)
    mx.synchronize()


def _abba(
    baseline: Callable[[], mx.array],
    candidate: Callable[[], mx.array],
    *,
    warmup: int,
    cycles: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        _evaluate(candidate)
        _evaluate(baseline)

    timings = {"baseline": [], "candidate": []}
    for _ in range(cycles):
        for name, fn in (
            ("baseline", baseline),
            ("candidate", candidate),
            ("candidate", candidate),
            ("baseline", baseline),
        ):
            started = time.perf_counter_ns()
            _evaluate(fn)
            timings[name].append((time.perf_counter_ns() - started) / 1e6)

    baseline_stats = _summary(timings["baseline"])
    candidate_stats = _summary(timings["candidate"])
    return {
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "speedup": (
            baseline_stats["median_ms"] / candidate_stats["median_ms"]
        ),
    }


def _scheduled_source(
    *,
    rows_per_simdgroup: int,
    simdgroups_per_threadgroup: int,
    fp32_output: bool,
) -> str:
    if rows_per_simdgroup not in (1, 2, 4):
        raise ValueError("rows per simdgroup must be 1, 2, or 4")
    if simdgroups_per_threadgroup not in (1, 2, 4, 8):
        raise ValueError("simdgroups per threadgroup must be 1, 2, 4, or 8")

    source = (
        verify_qmv._DENSE_FP32_SOURCE
        if fp32_output
        else verify_qmv._DENSE_SOURCE
    )
    expected_rows = "constexpr int THREAD_ROWS = 4;"
    expected_outputs = "constexpr int OUTPUTS_PER_THREADGROUP = 32;"
    if expected_rows not in source or expected_outputs not in source:
        raise RuntimeError("production dense-head kernel source changed")
    return source.replace(
        expected_rows,
        f"constexpr int THREAD_ROWS = {rows_per_simdgroup};",
    ).replace(
        expected_outputs,
        "constexpr int OUTPUTS_PER_THREADGROUP = "
        f"{rows_per_simdgroup * simdgroups_per_threadgroup};",
    )


@lru_cache(maxsize=None)
def _scheduled_kernel(
    rows_per_simdgroup: int,
    simdgroups_per_threadgroup: int,
    fp32_output: bool,
):
    suffix = "f32" if fp32_output else "bf16"
    return mx.fast.metal_kernel(
        name=(
            "omlx_deepseek_vocab_head_tile_"
            f"r{rows_per_simdgroup}_sg{simdgroups_per_threadgroup}_{suffix}"
        ),
        input_names=["input", "weight"],
        output_names=["output"],
        source=_scheduled_source(
            rows_per_simdgroup=rows_per_simdgroup,
            simdgroups_per_threadgroup=simdgroups_per_threadgroup,
            fp32_output=fp32_output,
        ),
        ensure_row_contiguous=True,
    )


def _candidate(
    weight: mx.array,
    inputs: mx.array,
    *,
    rows_per_simdgroup: int,
    simdgroups_per_threadgroup: int,
    fp32_output: bool,
) -> mx.array:
    input_dims = int(inputs.shape[-1])
    rows = int(inputs.size) // input_dims
    output_dims = int(weight.shape[0])
    outputs_per_threadgroup = (
        rows_per_simdgroup * simdgroups_per_threadgroup
    )
    if output_dims % outputs_per_threadgroup:
        raise ValueError("output shard does not divide the candidate tile")

    flat = inputs.reshape(rows, input_dims)
    kernel = _scheduled_kernel(
        rows_per_simdgroup,
        simdgroups_per_threadgroup,
        fp32_output,
    )
    threadgroup_width = 32 * simdgroups_per_threadgroup
    threadgroups = output_dims // outputs_per_threadgroup
    (output,) = kernel(
        inputs=[flat, weight],
        template=[
            ("T", inputs.dtype),
            ("M", rows),
            ("K", input_dims),
            ("N", output_dims),
        ],
        grid=(threadgroups * threadgroup_width, 1, 1),
        threadgroup=(threadgroup_width, 1, 1),
        output_shapes=[(rows, output_dims)],
        output_dtypes=[mx.float32 if fp32_output else inputs.dtype],
    )
    return output.reshape((*inputs.shape[:-1], output_dims))


def _load_tensor(model: Path, key: str) -> mx.array:
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    shard = index[key]
    return mx.load(str(model / shard))[key]


def _load_vocab_shard(
    model: Path,
    key: str,
    *,
    rank: int,
    world_size: int,
) -> mx.array:
    full = _load_tensor(model, key)
    rows = int(full.shape[0])
    if rows % world_size:
        raise ValueError(f"{key} rows do not divide {world_size} ranks")
    local_rows = rows // world_size
    start = rank * local_rows
    weight = mx.contiguous(full[start : start + local_rows])
    mx.eval(weight)
    mx.synchronize()
    return weight


def _shape_report(
    *,
    name: str,
    weight: mx.array,
    rows: int,
    input_dims: int,
    fp32_output: bool,
    warmup: int,
    cycles: int,
    min_speedup: float,
) -> dict[str, Any]:
    mx.random.seed(20260823 + rows + input_dims)
    inputs = mx.random.normal((rows, input_dims)).astype(mx.bfloat16)
    mx.eval(inputs)
    module = _WeightOnlyLinear(weight)

    if fp32_output:
        baseline = lambda: verify_qmv.dspark_head_gemv(module, inputs)
    else:
        baseline = lambda: verify_qmv.exact_verify_gemv(module, inputs)

    expected = baseline()
    mx.eval(expected)
    schedules: dict[str, Any] = {}
    for rows_per_simdgroup in (1, 2, 4):
        for simdgroups_per_threadgroup in (1, 2, 4, 8):
            schedule = f"r{rows_per_simdgroup}_sg{simdgroups_per_threadgroup}"

            def candidate(
                r=rows_per_simdgroup,
                sg=simdgroups_per_threadgroup,
            ):
                return _candidate(
                    weight,
                    inputs,
                    rows_per_simdgroup=r,
                    simdgroups_per_threadgroup=sg,
                    fp32_output=fp32_output,
                )

            actual = candidate()
            mx.eval(actual)
            exact = bool(mx.array_equal(expected, actual).item())
            timing = _abba(
                baseline,
                candidate,
                warmup=warmup,
                cycles=cycles,
            )
            schedules[schedule] = {
                "array_equal": exact,
                **timing,
                "qualified": exact and timing["speedup"] >= min_speedup,
            }

    # Every row compares against the same production function, but its paired
    # median can drift as the sweep warms/cools the device. Rank exact
    # candidates by their own absolute median; using the ratio can crown a
    # slower candidate merely because its adjacent baseline sample spiked.
    exact_schedules = [
        schedule
        for schedule, result in schedules.items()
        if result["array_equal"]
    ]
    best = min(
        exact_schedules,
        key=lambda schedule: float(
            schedules[schedule]["candidate"]["median_ms"]
        ),
    )
    return {
        "name": name,
        "rows": rows,
        "input_dims": input_dims,
        "output_dims": int(weight.shape[0]),
        "output_dtype": "float32" if fp32_output else str(inputs.dtype),
        "production_schedule": "r4_sg8",
        "minimum_speedup": min_speedup,
        "best_schedule": best,
        "best_qualified": bool(schedules[best]["qualified"]),
        "schedules": schedules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--min-speedup", type=float, default=1.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank is outside world size")

    model = args.model.expanduser().resolve()
    vocab_weight = _load_vocab_shard(
        model,
        "head.weight",
        rank=args.rank,
        world_size=args.world_size,
    )
    markov_weight = _load_vocab_shard(
        model,
        "mtp.2.markov_head.markov_w2.weight",
        rank=args.rank,
        world_size=args.world_size,
    )

    shapes = [
        _shape_report(
            name="target_verify",
            weight=vocab_weight,
            rows=6,
            input_dims=4096,
            fp32_output=False,
            warmup=args.warmup,
            cycles=args.cycles,
            min_speedup=args.min_speedup,
        ),
        _shape_report(
            name="draft_head",
            weight=vocab_weight,
            rows=5,
            input_dims=4096,
            fp32_output=True,
            warmup=args.warmup,
            cycles=args.cycles,
            min_speedup=args.min_speedup,
        ),
        _shape_report(
            name="markov_head",
            weight=markov_weight,
            rows=1,
            input_dims=256,
            fp32_output=True,
            warmup=args.warmup,
            cycles=args.cycles,
            min_speedup=args.min_speedup,
        ),
    ]
    report = {
        "schema_version": 1,
        "scope": "isolated_real_weight_ds4_mtp_vocab_heads",
        "model": str(model),
        "rank": args.rank,
        "world_size": args.world_size,
        "device": dict(mx.device_info()),
        "contract": (
            "outer output ownership only; production K mapping and reduction "
            "order; mx.array_equal required"
        ),
        "shapes": shapes,
        "all_schedules_exact": all(
            schedule["array_equal"]
            for shape in shapes
            for schedule in shape["schedules"].values()
        ),
        "all_best_qualified": all(shape["best_qualified"] for shape in shapes),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
