# SPDX-License-Identifier: Apache-2.0
"""Bounded synthetic MLX benchmark executed once on every cluster rank."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from typing import Any


def _evaluate(mx: Any, value: Any) -> None:
    mx.eval(value)


def _weight_rate(
    mx: Any,
    *,
    rows: int,
    width: int,
    warmup: int,
    iterations: int,
) -> float:
    weights = mx.random.uniform(shape=(width, width)).astype(mx.float16)
    inputs = mx.random.uniform(shape=(rows, width)).astype(mx.float16)
    for _ in range(warmup):
        _evaluate(mx, inputs @ weights)
    started = time.perf_counter()
    for _ in range(iterations):
        _evaluate(mx, inputs @ weights)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return float(weights.nbytes * iterations / elapsed)


def _collective_measurements(
    mx: Any,
    *,
    payload_bytes: int,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        _evaluate(mx, mx.distributed.all_sum(mx.array(1.0)))
    started = time.perf_counter()
    for _ in range(iterations):
        _evaluate(mx, mx.distributed.all_sum(mx.array(1.0)))
    latency = (time.perf_counter() - started) / iterations

    element_count = max(1, payload_bytes // 4)
    payload = mx.ones((element_count,), dtype=mx.float32)
    for _ in range(warmup):
        _evaluate(mx, mx.distributed.all_sum(payload))
    started = time.perf_counter()
    for _ in range(iterations):
        _evaluate(mx, mx.distributed.all_sum(payload))
    elapsed = max(time.perf_counter() - started, 1e-9)
    bandwidth = payload.nbytes * iterations / elapsed
    return float(latency), float(bandwidth)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    group = mx.distributed.init()
    rank = group.rank()
    barrier = mx.distributed.all_sum(mx.array(1), stream=mx.cpu)
    _evaluate(mx, barrier)
    decode_rate = _weight_rate(
        mx,
        rows=1,
        width=args.matrix_width,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    prefill_rate = _weight_rate(
        mx,
        rows=args.prefill_rows,
        width=args.matrix_width,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    latency, bandwidth = _collective_measurements(
        mx,
        payload_bytes=args.payload_bytes,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    _evaluate(mx, mx.distributed.all_sum(mx.array(1), stream=mx.cpu))
    return {
        "type": "performance_result",
        "rank": rank,
        "size": group.size(),
        "decode_weight_bytes_per_second": decode_rate,
        "prefill_weight_bytes_per_second": prefill_rate,
        "collective_latency_seconds": latency,
        "collective_bandwidth_bytes_per_second": bandwidth,
        "payload_bytes": args.payload_bytes,
        "matrix_width": args.matrix_width,
        "prefill_rows": args.prefill_rows,
        "samples": args.iterations,
        "measured_at": datetime.now(UTC).isoformat(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal oMLX performance probe")
    parser.add_argument("--payload-bytes", type=int, default=1024**2)
    parser.add_argument("--matrix-width", type=int, default=2048)
    parser.add_argument("--prefill-rows", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 4096 <= args.payload_bytes <= 64 * 1024**2:
        raise SystemExit("--payload-bytes must be between 4 KiB and 64 MiB")
    if not 128 <= args.matrix_width <= 8192:
        raise SystemExit("--matrix-width must be between 128 and 8192")
    if not 1 <= args.prefill_rows <= 1024:
        raise SystemExit("--prefill-rows must be between 1 and 1024")
    if not 0 <= args.warmup <= 20 or not 1 <= args.iterations <= 100:
        raise SystemExit("probe iteration counts are out of range")
    try:
        print(json.dumps(run_probe(args), sort_keys=True), flush=True)
    except Exception as exc:
        print(
            f"oMLX performance probe failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
