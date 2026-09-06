"""Fase A5: synthetic cost of the profiling instrumentation (no model).

Measures, in isolation (no SSD, no MLX workload), the per-call CPU cost
of the pieces PROFILE=1 adds to the read path:
  - ReadTelemetry.record_call with a full stage dict (the per-call lock)
  - the bounded percentile reservoirs behind stages_us
  - RunPoolTelemetry wrap + submit_notice with the owner tag (Fase A4)
  - the per-component stage dict build

Every bench result carries instrumentation_overhead (null until the
PROFILE=0 vs PROFILE=1 gate pair runs, Fase B). This probe bounds the
per-call cost so that pair can be interpreted without model noise.

Usage:
    .venv/bin/python -m bench.overhead_probe [--calls 20000]
                                               [--out bench/results/overhead_probe.json]

Runs in microseconds with light CPU — safe even on a busy machine.
"""

import argparse
import json
import time
from pathlib import Path


def _measure(calls: int) -> dict:
    from omlx.patches.expert_streaming.shard_bank import (
        ReadTelemetry,
        RunPoolTelemetry,
        _READ_METRICS,
    )

    # One realistic record_call payload: a fragmented decode component
    # (4 runs, 4 in-flight, every stage bucket populated).
    timings = {m: [120 + i * 7, 500 + i * 13] for i, m in enumerate(_READ_METRICS)}

    tel = ReadTelemetry(enabled=True, sample_capacity=2048)
    for _ in range(2000):  # warmup: allocator + reservoir fill
        tel.record_call(
            runs=4, bytes_=11_289_600, run_sizes=[64, 4, 2, 1],
            requested_inflight=4, timings=timings,
        )
    t0 = time.perf_counter()
    for _ in range(calls):
        tel.record_call(
            runs=4, bytes_=11_289_600, run_sizes=[64, 4, 2, 1],
            requested_inflight=4, timings=timings,
        )
    record_us = (time.perf_counter() - t0) * 1e6 / calls

    t0 = time.perf_counter()
    tel.summary()
    summary_us = (time.perf_counter() - t0) * 1e6

    def _build_stages():
        return {m: [] for m in _READ_METRICS}

    t0 = time.perf_counter()
    for _ in range(calls):
        _build_stages()
    build_us = (time.perf_counter() - t0) * 1e6 / calls

    # RunPoolTelemetry wrap + submit_notice with an owner (A4 path).
    ptel = RunPoolTelemetry()
    owner = "probe"

    def _noop():
        return 7

    ts = time.perf_counter_ns()
    for _ in range(2000):
        fn = ptel.wrap(ts, _noop, owner=owner)
        ptel.submit_notice(owner=owner)
        fn()
    t0 = time.perf_counter()
    for _ in range(calls):
        fn = ptel.wrap(ts, _noop, owner=owner)
        ptel.submit_notice(owner=owner)
        fn()
    pool_us = (time.perf_counter() - t0) * 1e6 / calls

    return {
        "calls": calls,
        "stage_metrics": len(_READ_METRICS),
        "record_call_us_per_call": round(record_us, 3),
        "summary_us_per_snapshot": round(summary_us, 3),
        "stages_dict_build_us_per_call": round(build_us, 3),
        "pool_wrap_pair_us_per_run": round(pool_us, 3),
        "estimated_read_path_overhead_us_per_component": round(
            record_us + build_us + pool_us * 4, 3
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calls", type=int, default=20000)
    ap.add_argument("--out", default="bench/results/overhead_probe.json")
    args = ap.parse_args()
    report = _measure(max(100, int(args.calls)))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
