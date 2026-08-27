#!/usr/bin/env python3
"""Physical TP2 gate for DS4 sparse-indexer row transport.

The production 1,024-token prompt chunk assigns 512 query rows to each rank.
Every one of DS4 Flash's 21 ratio-4 layers exchanges a ``(512, 1, 512)``
uint32 top-k block.  At 100K context that is 21 * ceil(100000 / 1024) = 2058
exchanges.  This probe compares the current all-gather with the rollback-gated
ordered point-to-point implementation for at least that many iterations.

Run through the same two-host JACCL launcher used by oMLX, for example::

    mlx.launch --hostfile hosts.json --backend jaccl -- \
      python benchmarks/bench_ds4_indexer_gather_transport.py

Promotion requires both ranks to finish, exact rows in every round, no JACCL
progress timeout, and median point-to-point wall time no more than 2% above
all-gather.  Use ``--rank-rows 384,640`` for the experimental weighted 3:5
row split; production remains 512,512.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from typing import Any

import mlx.core as mx


def _p2p_barrier(group: Any, rank: int) -> None:
    """Ordered point-to-point barrier without another collective family."""

    marker = mx.array([0xD54], dtype=mx.uint32)
    if rank == 0:
        mx.eval(mx.distributed.send(marker, 1, group=group))
        received = mx.distributed.recv_like(marker, 1, group=group)
        mx.eval(received)
    else:
        received = mx.distributed.recv_like(marker, 0, group=group)
        mx.eval(received)
        mx.eval(mx.distributed.send(marker, 0, group=group))
    if int(received[0].item()) != 0xD54:
        raise RuntimeError("TP2 barrier payload was corrupted")


def _ordered_p2p_rows(
    local_rows: mx.array,
    *,
    peer_rows: int,
    group: Any,
    rank: int,
) -> mx.array:
    """Mirror ``_gather_indexer_rows``'s TP2 transport exactly."""

    peer = 1 - rank
    template = mx.zeros(
        (peer_rows, *local_rows.shape[1:]),
        dtype=local_rows.dtype,
    )
    # The benchmark payload is pre-materialized. Production performs this
    # same boundary so both GPUs finish their independent score graphs before
    # either transport direction begins.
    mx.eval(local_rows)
    if rank == 0:
        mx.eval(mx.distributed.send(local_rows, peer, group=group))
        remote = mx.distributed.recv_like(template, peer, group=group)
        mx.eval(remote)
        parts = (local_rows, remote)
    else:
        remote = mx.distributed.recv_like(template, peer, group=group)
        mx.eval(remote)
        mx.eval(mx.distributed.send(local_rows, peer, group=group))
        parts = (remote, local_rows)
    return mx.concatenate(parts, axis=0)


def _ordered_graph_p2p_rows(
    local_rows: mx.array,
    *,
    peer_rows: int,
    group: Any,
    rank: int,
) -> mx.array:
    """Rank-asymmetric graph order with one transport evaluation.

    Both GPUs first complete their independent local work. Rank 0 then makes
    its concatenation depend on send before recv, while rank 1 depends on recv
    before send. The final concatenate is the single evaluation root, keeping
    the exact deadlock-free direction order without two extra Python/Metal
    synchronization boundaries.
    """

    peer = 1 - rank
    template = mx.zeros(
        (peer_rows, *local_rows.shape[1:]),
        dtype=local_rows.dtype,
    )
    mx.eval(local_rows)
    sent = mx.distributed.send(local_rows, peer, group=group)
    remote = mx.distributed.recv_like(template, peer, group=group)
    parts = (sent, remote) if rank == 0 else (remote, sent)
    return mx.concatenate(parts, axis=0)


def _all_gather_rows(
    local_rows: mx.array,
    *,
    rank_rows: tuple[int, int],
    group: Any,
    rank: int,
) -> mx.array:
    """Mirror the current padded all-gather, including uneven row slicing."""

    max_rows = max(rank_rows)
    padded = local_rows
    if local_rows.shape[0] < max_rows:
        padded = mx.concatenate(
            [
                local_rows,
                mx.zeros(
                    (max_rows - local_rows.shape[0], *local_rows.shape[1:]),
                    dtype=local_rows.dtype,
                ),
            ],
            axis=0,
        )
    gathered = mx.distributed.all_gather(padded, group=group)
    if rank_rows[0] == rank_rows[1]:
        return gathered
    parts = tuple(
        gathered[index * max_rows : index * max_rows + rows]
        for index, rows in enumerate(rank_rows)
    )
    return mx.concatenate(parts, axis=0)


def _parse_rank_rows(raw: str) -> tuple[int, int]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rank rows must be integers") from exc
    if len(values) != 2 or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("rank rows must contain two positive values")
    return values


def _exact_payload(
    rank_rows: tuple[int, int],
    *,
    topk: int,
    rank: int,
) -> tuple[mx.array, mx.array]:
    row_start = sum(rank_rows[:rank])
    element_start = row_start * topk
    element_stop = element_start + rank_rows[rank] * topk
    local = mx.arange(element_start, element_stop, dtype=mx.uint32).reshape(
        rank_rows[rank],
        1,
        topk,
    )
    expected = mx.arange(
        0,
        sum(rank_rows) * topk,
        dtype=mx.uint32,
    ).reshape(sum(rank_rows), 1, topk)
    mx.eval(local, expected)
    return local, expected


def _run_round(
    mode: str,
    *,
    local: mx.array,
    expected: mx.array,
    rank_rows: tuple[int, int],
    group: Any,
    rank: int,
    iterations: int,
) -> float:
    _p2p_barrier(group, rank)
    started = time.perf_counter()
    output = None
    for _ in range(iterations):
        if mode == "all_gather":
            output = _all_gather_rows(
                local,
                rank_rows=rank_rows,
                group=group,
                rank=rank,
            )
        elif mode == "p2p":
            output = _ordered_p2p_rows(
                local,
                peer_rows=rank_rows[1 - rank],
                group=group,
                rank=rank,
            )
        else:
            output = _ordered_graph_p2p_rows(
                local,
                peer_rows=rank_rows[1 - rank],
                group=group,
                rank=rank,
            )
        mx.eval(output)
    elapsed = time.perf_counter() - started
    assert output is not None
    exact = mx.array_equal(output, expected)
    mx.eval(exact)
    if not bool(exact.item()):
        raise RuntimeError(f"{mode} reconstructed rows out of rank order")
    _p2p_barrier(group, rank)
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-rows", type=_parse_rank_rows, default=(512, 512))
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument(
        "--iterations",
        type=int,
        default=21 * math.ceil(100_000 / 1024),
        help="exchanges per measured round (default: one 100K DS4 lifetime)",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.topk < 1 or args.iterations < 1 or args.warmup < 0 or args.rounds < 1:
        parser.error("topk, iterations, and rounds must be positive; warmup >= 0")

    group = mx.distributed.init(backend="jaccl", strict=True)
    rank = int(group.rank())
    if int(group.size()) != 2 or rank not in (0, 1):
        raise RuntimeError(f"this gate requires pure TP2, got size={group.size()}")

    local, expected = _exact_payload(args.rank_rows, topk=args.topk, rank=rank)
    for mode in ("all_gather", "p2p", "p2p_graph"):
        if args.warmup:
            _run_round(
                mode,
                local=local,
                expected=expected,
                rank_rows=args.rank_rows,
                group=group,
                rank=rank,
                iterations=args.warmup,
            )

    timings = {"all_gather": [], "p2p": [], "p2p_graph": []}
    # Alternate the first mode to reduce thermal/order bias over three rounds.
    for round_index in range(args.rounds):
        order = (
            ("all_gather", "p2p", "p2p_graph")
            if round_index % 2 == 0
            else ("p2p_graph", "p2p", "all_gather")
        )
        for mode in order:
            timings[mode].append(
                _run_round(
                    mode,
                    local=local,
                    expected=expected,
                    rank_rows=args.rank_rows,
                    group=group,
                    rank=rank,
                    iterations=args.iterations,
                )
            )

    medians = {
        mode: statistics.median(values) for mode, values in timings.items()
    }
    ratios = {
        mode: medians[mode] / medians["all_gather"]
        for mode in ("p2p", "p2p_graph")
    }
    print(
        json.dumps(
            {
                "type": "ds4_indexer_gather_transport",
                "rank": rank,
                "world_size": 2,
                "rank_rows": args.rank_rows,
                "topk": args.topk,
                "dtype": "uint32",
                "payload_bytes_by_rank": [
                    rows * args.topk * 4 for rows in args.rank_rows
                ],
                "iterations_per_round": args.iterations,
                "rounds": args.rounds,
                "seconds": timings,
                "median_seconds": medians,
                "over_all_gather": ratios,
                "within_two_percent": {
                    mode: ratio <= 1.02 for mode, ratio in ratios.items()
                },
                "exact": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if ratios["p2p_graph"] <= 1.02 else 2


if __name__ == "__main__":
    raise SystemExit(main())
