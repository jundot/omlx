#!/usr/bin/env python3
"""Physical A/B for DS4's certified hierarchy upper-bound postprocess."""

from __future__ import annotations

import argparse
import statistics
import time
from types import SimpleNamespace

import mlx.core as mx

from omlx.patches.deepseek_v4 import hierarchical_indexer as hi


def _timed(fn):
    started = time.perf_counter()
    mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - started) * 1e3


def _measure_interleaved(reference, candidate, warmup: int, repeats: int):
    for index in range(warmup):
        order = (reference, candidate) if index % 2 == 0 else (candidate, reference)
        for fn in order:
            mx.eval(fn())
            mx.synchronize()
    samples = {"reference": [], "candidate": []}
    for index in range(repeats):
        order = (
            (("reference", reference), ("candidate", candidate))
            if index % 2 == 0
            else (("candidate", candidate), ("reference", reference))
        )
        for name, fn in order:
            samples[name].append(_timed(fn))
    return samples["reference"], samples["candidate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--pooled", default="7500,25000,62500")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.rows % hi._GROUP_ROWS:
        raise SystemExit("rows must be divisible by the hierarchy group size")

    hi._NATIVE_UPPER_ENABLED = True
    hi._NATIVE_UPPER_FAILED = False
    mx.random.seed(20260827)
    row_residual = mx.random.uniform(0, 0.2, (args.rows,)).astype(mx.float32)
    row_error = mx.random.uniform(0, 0.02, (args.rows,)).astype(mx.float32)
    row_norm = mx.random.uniform(0, 3, (args.rows,)).astype(mx.float32)
    mx.eval(row_residual, row_error, row_norm)

    print("pool\treference_ms\tnative_ms\tgain\toutward\tmax_delta")
    for pool in (int(value) for value in args.pooled.split(",")):
        approximate = mx.random.uniform(-2, 2, (args.rows, pool)).astype(
            mx.bfloat16
        )
        state = SimpleNamespace(
            key_orthogonal_residual=mx.random.uniform(0, 0.2, (pool,)).astype(
                mx.float32
            ),
            key_coordinate_norm=mx.random.uniform(0, 3, (pool,)).astype(mx.float32),
            key_coordinate_error=mx.random.uniform(0, 0.02, (pool,)).astype(
                mx.float32
            ),
        )
        mx.eval(
            approximate,
            state.key_orthogonal_residual,
            state.key_coordinate_norm,
            state.key_coordinate_error,
        )

        def reference():
            approximate_f = approximate.astype(mx.float32)
            error_bound = (
                row_residual[:, None] * state.key_orthogonal_residual[None]
                + row_error[:, None] * state.key_coordinate_norm[None]
                + (row_norm + row_error)[:, None]
                * state.key_coordinate_error[None]
            )
            upper = (
                approximate_f
                + error_bound
                + hi._NUMERIC_ABS_GUARD
                + hi._NUMERIC_REL_GUARD * mx.abs(approximate_f)
            )
            return mx.max(
                upper.reshape(
                    args.rows // hi._GROUP_ROWS,
                    hi._GROUP_ROWS,
                    pool,
                ),
                axis=1,
            )

        def candidate():
            result = hi._native_group_upper(
                approximate,
                row_residual,
                row_error,
                row_norm,
                state,
            )
            assert result is not None
            return result

        ref = reference()
        got = candidate()
        mx.eval(ref, got)
        delta = got - ref
        outward = bool(mx.all(delta >= 0).item())
        max_delta = float(mx.max(delta).item())
        reference_samples, candidate_samples = _measure_interleaved(
            reference,
            candidate,
            args.warmup,
            args.repeats,
        )
        reference_ms = statistics.median(reference_samples)
        candidate_ms = statistics.median(candidate_samples)
        print(
            f"{pool}\t{reference_ms:.3f}\t{candidate_ms:.3f}\t"
            f"{reference_ms / candidate_ms:.3f}\t{outward}\t{max_delta:.8f}"
        )
        if not outward or max_delta >= 5e-4:
            raise SystemExit("native upper bound failed the outward exactness gate")


if __name__ == "__main__":
    main()
