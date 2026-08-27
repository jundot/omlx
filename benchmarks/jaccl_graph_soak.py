#!/usr/bin/env python3
"""Soak JACCL with lazy dependent collective graphs shaped like TP inference."""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx


def _chain(group, *, elements: int, collectives: int, rank: int) -> float:
    value = mx.full((elements,), rank + 1, dtype=mx.bfloat16)
    for _ in range(collectives):
        value = mx.distributed.all_sum(value, group=group) * 0.5
    started = time.perf_counter()
    mx.eval(value)
    elapsed = time.perf_counter() - started
    observed = float(value[0].item())
    if observed != 1.5:
        raise RuntimeError(f"collective chain returned {observed}, expected 1.5")
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--collectives", type=int, default=86)
    parser.add_argument(
        "--elements",
        default="28672,4096",
        help="Comma-separated BF16 element counts (7-row prefill, decode)",
    )
    args = parser.parse_args()

    group = mx.distributed.init(backend="jaccl", strict=True)
    rank = int(group.rank())
    if int(group.size()) != 2:
        raise RuntimeError(f"expected two ranks, got {group.size()}")
    shapes = tuple(int(item) for item in args.elements.split(","))
    elapsed = {str(elements): 0.0 for elements in shapes}
    started = time.perf_counter()
    for cycle in range(args.cycles):
        for elements in shapes:
            elapsed[str(elements)] += _chain(
                group,
                elements=elements,
                collectives=args.collectives,
                rank=rank,
            )
        if cycle % 10 == 9:
            barrier = mx.distributed.all_sum(mx.array(rank + 1), group=group)
            mx.eval(barrier)
            if int(barrier.item()) != 3:
                raise RuntimeError("barrier result is invalid")

    print(
        json.dumps(
            {
                "type": "jaccl_graph_soak",
                "rank": rank,
                "cycles": args.cycles,
                "collectives_per_shape": args.collectives,
                "total_collectives": args.cycles * len(shapes) * args.collectives,
                "shape_seconds": elapsed,
                "wall_seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
