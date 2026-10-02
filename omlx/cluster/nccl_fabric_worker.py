# SPDX-License-Identifier: Apache-2.0
"""Bounded two-rank NCCL probe used by the dashboard's fabric verifier."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections.abc import Sequence
from typing import Any


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _rank_device_lists(value: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("fabric devices must be a JSON array") from exc
    if not isinstance(payload, list) or len(payload) != 2:
        raise argparse.ArgumentTypeError("exactly two fabric ranks are required")
    ranks: list[tuple[str, ...]] = []
    for raw_rank in payload:
        if not isinstance(raw_rank, list) or not raw_rank:
            raise argparse.ArgumentTypeError(
                "each fabric rank requires at least one device"
            )
        devices = tuple(str(item).strip() for item in raw_rank)
        if any(
            not device
            or len(device) > 64
            or not all(char.isalnum() or char in "_.:-" for char in device)
            for device in devices
        ):
            raise argparse.ArgumentTypeError(
                "fabric devices contain an invalid name"
            )
        ranks.append(devices)
    return ranks[0], ranks[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a two-node NCCL fabric")
    parser.add_argument("--interfaces", required=True, type=_rank_device_lists)
    parser.add_argument("--rdma-devices", required=True, type=_rank_device_lists)
    parser.add_argument("--payload-mib", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def _configure_nccl_fabric_environment(
    args: argparse.Namespace,
    rank: int,
) -> None:
    """Pin NCCL bootstrap and payload traffic to this rank's verified rails."""

    os.environ["NCCL_SOCKET_IFNAME"] = "=" + ",".join(args.interfaces[rank])
    # Socket selection controls NCCL bootstrap only. Pin the data HCA as well,
    # otherwise a machine with multiple NICs can pass while NCCL silently uses
    # a different device. The leading '=' requests exact NCCL HCA matches.
    os.environ["NCCL_IB_HCA"] = "=" + ",".join(args.rdma_devices[rank])
    os.environ["NCCL_IB_DISABLE"] = "0"


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("MLX_RANK", "-1"))
    if rank not in {0, 1}:
        raise RuntimeError(f"NCCL fabric probe received invalid rank {rank}")
    _configure_nccl_fabric_environment(args, rank)

    # Import only after the rank-specific NCCL interface has been selected.
    import mlx.core as mx

    if not mx.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on this fabric member")
    if not mx.distributed.is_available("nccl"):
        raise RuntimeError("MLX was installed without NCCL support")
    group = mx.distributed.init(backend="nccl", strict=True)
    if group.size() != 2:
        raise RuntimeError(f"NCCL fabric probe expected 2 ranks, got {group.size()}")

    elements = args.payload_mib * 1024 * 1024 // 4
    payload = mx.ones((elements,), dtype=mx.float32)
    warmup = mx.distributed.all_sum(payload, group=group)
    mx.eval(warmup)
    durations: list[float] = []
    for _ in range(args.repeats):
        barrier = mx.distributed.all_sum(mx.ones((1,), dtype=mx.float32), group=group)
        mx.eval(barrier)
        started = time.perf_counter()
        reduced = mx.distributed.all_sum(payload, group=group)
        mx.eval(reduced)
        elapsed = time.perf_counter() - started
        if not math.isclose(float(reduced[0].item()), 2.0, abs_tol=1e-5):
            raise RuntimeError("NCCL all-sum returned an invalid result")
        durations.append(elapsed)

    local = mx.array(durations, dtype=mx.float32)
    gathered = mx.distributed.all_gather(local, group=group)
    mx.eval(gathered)
    all_durations = [float(item) for item in gathered.tolist()]
    slowest = max(all_durations)
    payload_bytes = elements * 4
    return {
        "type": "nccl_fabric_result",
        "rank": rank,
        "world_size": group.size(),
        "hostname": platform.node(),
        "interfaces": list(args.interfaces[rank]),
        "rdma_devices": list(args.rdma_devices[rank]),
        "payload_mib": args.payload_mib,
        "repeats": args.repeats,
        "slowest_seconds": slowest,
        "payload_bytes_per_second": payload_bytes / slowest if slowest > 0 else 0.0,
        "device_info": _json_safe(mx.device_info()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.payload_mib <= 1024:
        raise SystemExit("--payload-mib must be between 1 and 1024")
    if not 1 <= args.repeats <= 20:
        raise SystemExit("--repeats must be between 1 and 20")
    try:
        print(json.dumps(run_probe(args), sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "type": "nccl_fabric_error",
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
