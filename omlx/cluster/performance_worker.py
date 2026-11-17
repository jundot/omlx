# SPDX-License-Identifier: Apache-2.0
"""Bounded synthetic MLX benchmark executed once on every cluster rank."""

from __future__ import annotations

import argparse
import json
import os
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
    """Model-weight bytes per second for a (rows x width) @ (width x width) GEMM.

    Used for PREFILL with a large ``rows`` so the GEMM is compute-bound: the
    weight-read rate then tracks fp16 compute throughput and orders the
    hardware correctly (M5 Max > M2 Ultra > M1 Max). A small ``rows`` (the old
    64) is not compute-bound and the kernel's per-architecture efficiency
    inverts the ordering, so it is unfit. Decode uses ``_bandwidth_rate``
    instead, because decode is memory-bound (B10/E4=B).
    """
    weights = mx.random.uniform(shape=(width, width)).astype(mx.float16)
    inputs = mx.random.uniform(shape=(rows, width)).astype(mx.float16)
    for _ in range(warmup):
        _evaluate(mx, inputs @ weights)
    started = time.perf_counter()
    for _ in range(iterations):
        _evaluate(mx, inputs @ weights)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return float(weights.nbytes * iterations / elapsed)


def _bandwidth_rate(
    mx: Any,
    *,
    elements: int,
    warmup: int,
    iterations: int,
) -> float:
    """Streaming memory bandwidth (bytes/s) via a large reduction.

    Decode is memory-bound: a rank streams its weights once per generated
    token, so the decode calibration signal must track memory bandwidth. A
    large streaming reduction saturates the memory controllers and orders the
    hardware reliably. A small-M GEMV does NOT: MLX's rows=1 kernel efficiency
    varies by architecture (it measured an M5 Max below an M1 Max, inverting
    the true 600 > 400 GB/s ordering), so it is unfit for the decode rate.
    This is a relative calibration signal, not a tokens/s promise (B10/E4=B).
    """
    buf = mx.random.uniform(shape=(max(1, elements // 8192), 8192)).astype(
        mx.float16
    )
    _evaluate(mx, buf)
    for _ in range(warmup):
        _evaluate(mx, mx.sum(buf))
    started = time.perf_counter()
    for _ in range(iterations):
        _evaluate(mx, mx.sum(buf))
    elapsed = max(time.perf_counter() - started, 1e-9)
    return float(buf.nbytes * iterations / elapsed)


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


def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    from .jaccl_side_channel import init_cluster_group

    group = init_cluster_group(
        mx,
        backend="jaccl" if os.environ.get("MLX_JACCL_COORDINATOR") else None,
        strict=bool(os.environ.get("MLX_JACCL_COORDINATOR")),
    )
    rank = group.rank()
    barrier = mx.distributed.all_sum(mx.array(1), stream=mx.cpu)
    _evaluate(mx, barrier)
    decode_rate = _bandwidth_rate(
        mx,
        elements=args.bandwidth_mib * 1024 * 1024 // 2,
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
        "bandwidth_mib": args.bandwidth_mib,
        "matrix_width": args.matrix_width,
        "prefill_rows": args.prefill_rows,
        "samples": args.iterations,
        "measured_at": datetime.now(UTC).isoformat(),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("MLX_JACCL_COORDINATOR"):
        return _run_probe(args)

    from .jaccl_lease import acquire_jaccl_communicator_lease

    with acquire_jaccl_communicator_lease(deployment_id="performance-probe"):
        return _run_probe(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal oMLX performance probe")
    parser.add_argument("--payload-bytes", type=int, default=1024**2)
    # Decode is memory-bound: its rate is a streaming-reduction bandwidth
    # measurement (--bandwidth-mib), which orders the hardware reliably. A
    # small-M GEMV does not (MLX's rows=1 kernel inverted M5 Max vs M1 Max).
    parser.add_argument("--bandwidth-mib", type=int, default=128)
    # Prefill is compute-bound: a large-rows GEMM (--prefill-rows x
    # --matrix-width) makes the weight-read rate track fp16 compute, ordering
    # M5 Max > M2 Ultra > M1 Max. rows=64 was too small to be compute-bound.
    parser.add_argument("--matrix-width", type=int, default=8192)
    parser.add_argument("--prefill-rows", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 4096 <= args.payload_bytes <= 64 * 1024**2:
        raise SystemExit("--payload-bytes must be between 4 KiB and 64 MiB")
    if not 16 <= args.bandwidth_mib <= 1024:
        raise SystemExit("--bandwidth-mib must be between 16 and 1024")
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
