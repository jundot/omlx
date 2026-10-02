#!/usr/bin/env python3
"""Probe whether one MLX Ring can span Metal and CUDA ranks.

Run locally first::

    python3 benchmarks/heterogeneous_pool_probe.py

Then run the same checkout and MLX version on every host::

    mlx.launch --backend ring --hostfile hosts.json -- \
      python3 benchmarks/heterogeneous_pool_probe.py \
      --distributed \
      --expect-ranks 6 \
      --require-accelerators metal,cuda \
      --cuda-supernode-ranks 4,5 \
      --collective-mib 1,64

The distributed form checks the actual prerequisites for a heterogeneous oMLX
model pool: every rank can execute representative BF16 attention and 4-bit
quantized matrix work, every rank joins the same TCP Ring, Metal and CUDA are
both present, results remain numerically close, and small/large collectives
complete. It does not prove that a particular model supports every backend;
the real unequal pipeline smoke remains the next gate.

``--cuda-supernode-ranks`` additionally verifies that a proposed ConnectX
pair is adjacent in the outer Ring and has NCCL support on both members. It
does not time ``Group.split()``: that would remain a Ring subgroup (and Ring
split is not supported by all MLX releases), so it could not prove NCCL or the
direct link. The dashboard performs the real isolated NCCL fabric test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

_ACCELERATOR_CODES = {"cpu": 0, "metal": 1, "cuda": 2}
_ACCELERATORS_BY_CODE = {value: key for key, value in _ACCELERATOR_CODES.items()}
_OS_CODES = {"unknown": 0, "darwin": 1, "linux": 2, "windows": 3}
_OS_BY_CODE = {value: key for key, value in _OS_CODES.items()}


@dataclass(frozen=True)
class LocalCapability:
    hostname: str
    os_name: str
    os_version: str
    architecture: str
    accelerator: str
    mlx_version: str
    python_version: str
    device: str
    device_info: dict[str, Any]
    ring_available: bool
    jaccl_available: bool
    nccl_available: bool
    runtime_fingerprint: str


def _safe_distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _available(callable_obj: Any) -> bool:
    try:
        return bool(callable_obj())
    except Exception:
        return False


def detect_local_capability(mx: Any) -> LocalCapability:
    """Return platform-neutral facts without relying on oMLX's Mac probes."""

    metal = _available(getattr(getattr(mx, "metal", None), "is_available", None))
    cuda = _available(getattr(getattr(mx, "cuda", None), "is_available", None))
    accelerator = "cuda" if cuda else "metal" if metal else "cpu"
    os_name = platform.system().strip().lower() or "unknown"
    mlx_version = _safe_distribution_version("mlx")
    try:
        device_info = _json_safe(mx.device_info())
    except Exception as exc:
        device_info = {"error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(device_info, dict):
        device_info = {"value": device_info}
    try:
        device = str(mx.default_device())
    except Exception:
        device = "unknown"
    fingerprint_source = json.dumps(
        {
            "mlx": mlx_version,
            "os": os_name,
            "architecture": platform.machine(),
            "accelerator": accelerator,
            "device": device_info,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    distributed = mx.distributed

    def backend_available(name: str) -> bool:
        try:
            return bool(distributed.is_available(name))
        except Exception:
            return False

    return LocalCapability(
        hostname=platform.node(),
        os_name=os_name,
        os_version=platform.version(),
        architecture=platform.machine(),
        accelerator=accelerator,
        mlx_version=mlx_version,
        python_version=platform.python_version(),
        device=device,
        device_info=device_info,
        ring_available=backend_available("ring"),
        jaccl_available=backend_available("jaccl"),
        nccl_available=backend_available("nccl"),
        runtime_fingerprint=hashlib.sha256(fingerprint_source).hexdigest(),
    )


def _parse_positive_ints(value: str, *, label: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for item in value.split(","):
        try:
            number = int(item.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{label} must be a comma-separated integer list"
            ) from exc
        if number <= 0:
            raise argparse.ArgumentTypeError(f"{label} values must be positive")
        parsed.append(number)
    if not parsed:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    return tuple(parsed)


def _parse_rank_set(value: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for item in value.split(","):
        try:
            rank = int(item.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "supernode ranks must be comma-separated integers"
            ) from exc
        if rank < 0:
            raise argparse.ArgumentTypeError("supernode ranks must be non-negative")
        if rank in parsed:
            raise argparse.ArgumentTypeError("supernode ranks must be unique")
        parsed.append(rank)
    if len(parsed) < 2:
        raise argparse.ArgumentTypeError("a CUDA supernode needs at least two ranks")
    return tuple(parsed)


def _version_triplet(version: str) -> tuple[int, int, int]:
    numbers: list[int] = []
    for component in version.split("."):
        digits = "".join(char for char in component if char.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
        if len(numbers) == 3:
            break
    return tuple((numbers + [0, 0, 0])[:3])


def _representative_compute(mx: Any) -> dict[str, float]:
    """Exercise kernels used by quantized transformer inference."""

    import mlx.nn as nn

    weight = ((mx.arange(64 * 64, dtype=mx.float32) % 31) - 15).reshape(64, 64) / 32
    linear = nn.Linear(64, 64, bias=False)
    linear.weight = weight
    quantized = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=4)
    inputs = ((mx.arange(8 * 64, dtype=mx.float32) % 17) - 8).reshape(8, 64) / 16
    qmm_output = quantized(inputs)

    query = mx.arange(1 * 4 * 16 * 32, dtype=mx.float32).reshape(1, 4, 16, 32)
    query = (query / 2048).astype(mx.bfloat16)
    key = query[..., ::-1]
    value = (query.astype(mx.float32) * 0.5).astype(mx.bfloat16)
    attention = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=32**-0.5,
    )
    mx.eval(qmm_output, attention)
    return {
        "qmm_checksum": float(mx.sum(qmm_output.astype(mx.float32)).item()),
        "sdpa_checksum": float(mx.sum(attention.astype(mx.float32)).item()),
    }


def _sync(group: Any, mx: Any) -> None:
    marker = mx.distributed.all_sum(mx.ones((1,), dtype=mx.float32), group=group)
    mx.eval(marker)


def _all_gather_rows(group: Any, mx: Any, values: Sequence[float]) -> list[list[float]]:
    local = mx.array([float(value) for value in values], dtype=mx.float32)
    gathered = mx.distributed.all_gather(local, group=group)
    mx.eval(gathered)
    flat = [float(item) for item in gathered.tolist()]
    width = len(values)
    return [flat[index : index + width] for index in range(0, len(flat), width)]


def _collective_benchmark(
    group: Any,
    mx: Any,
    *,
    sizes_mib: Sequence[int],
    repeats: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    world_size = group.size()
    for size_mib in sizes_mib:
        element_count = size_mib * 1024 * 1024 // 4
        payload = mx.ones((element_count,), dtype=mx.float32)
        warmup = mx.distributed.all_sum(payload, group=group)
        mx.eval(warmup)
        durations: list[float] = []
        for _ in range(repeats):
            _sync(group, mx)
            started_at = time.perf_counter()
            total = mx.distributed.all_sum(payload, group=group)
            mx.eval(total)
            durations.append(time.perf_counter() - started_at)
            first = float(total[0].item())
            if not math.isclose(first, float(world_size), rel_tol=0, abs_tol=1e-5):
                raise RuntimeError(
                    f"{size_mib} MiB all-sum returned {first}, expected {world_size}"
                )
        rank_rows = _all_gather_rows(group, mx, durations)
        all_durations = [duration for row in rank_rows for duration in row]
        slowest = max(all_durations)
        payload_bytes = element_count * 4
        results.append(
            {
                "payload_mib": size_mib,
                "repeats": repeats,
                "slowest_seconds": slowest,
                "payload_gib_per_second": (
                    payload_bytes / slowest / 1024**3 if slowest > 0 else 0.0
                ),
                "rank_durations_seconds": rank_rows,
            }
        )
    return results


def _distributed_topology(
    group: Any,
    mx: Any,
    capability: LocalCapability,
) -> dict[str, Any]:
    version = _version_triplet(capability.mlx_version)
    os_code = _OS_CODES.get(capability.os_name, _OS_CODES["unknown"])
    local = [
        _ACCELERATOR_CODES[capability.accelerator],
        os_code,
        version[0],
        version[1],
        version[2],
        int(capability.nccl_available),
    ]
    rows = _all_gather_rows(group, mx, local)
    ranks = []
    for rank, row in enumerate(rows):
        accelerator_code, rank_os, major, minor, patch, nccl_available = (
            int(value) for value in row
        )
        ranks.append(
            {
                "rank": rank,
                "accelerator": _ACCELERATORS_BY_CODE.get(
                    accelerator_code, f"unknown-{accelerator_code}"
                ),
                "os": _OS_BY_CODE.get(rank_os, f"unknown-{rank_os}"),
                "mlx_version": f"{major}.{minor}.{patch}",
                "nccl_available": bool(nccl_available),
            }
        )
    return {
        "world_size": group.size(),
        "accelerators": sorted({rank["accelerator"] for rank in ranks}),
        "mlx_versions": sorted({rank["mlx_version"] for rank in ranks}),
        "ranks": ranks,
    }


def _checksum_report(
    group: Any,
    mx: Any,
    checksums: dict[str, float],
    *,
    tolerance: float,
) -> dict[str, Any]:
    names = sorted(checksums)
    rows = _all_gather_rows(group, mx, [checksums[name] for name in names])
    values_by_name = {
        name: [row[index] for row in rows] for index, name in enumerate(names)
    }
    spreads = {
        name: max(values) - min(values) for name, values in values_by_name.items()
    }
    return {
        "tolerance": tolerance,
        "values": values_by_name,
        "spreads": spreads,
        "ok": all(spread <= tolerance for spread in spreads.values()),
    }


def _parse_required_accelerators(value: str) -> set[str]:
    required = {item.strip().lower() for item in value.split(",") if item.strip()}
    unknown = required - set(_ACCELERATOR_CODES)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown accelerator(s): {', '.join(sorted(unknown))}"
        )
    return required


def _supernode_failures(
    specs: Sequence[Sequence[int]],
    topology: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    ranks = topology["ranks"]
    world_size = int(topology["world_size"])
    claimed: set[int] = set()
    for index, members_value in enumerate(specs, start=1):
        members = tuple(sorted(int(rank) for rank in members_value))
        outside = [rank for rank in members if rank >= world_size]
        if outside:
            failures.append(
                f"CUDA supernode {index} references missing rank(s): "
                + ", ".join(str(rank) for rank in outside)
            )
            continue
        overlap = claimed.intersection(members)
        if overlap:
            failures.append(
                f"CUDA supernode {index} reuses rank(s): "
                + ", ".join(str(rank) for rank in sorted(overlap))
            )
        claimed.update(members)
        non_cuda = [rank for rank in members if ranks[rank]["accelerator"] != "cuda"]
        if non_cuda:
            failures.append(
                f"CUDA supernode {index} contains non-CUDA rank(s): "
                + ", ".join(str(rank) for rank in non_cuda)
            )
        without_nccl = [rank for rank in members if not ranks[rank]["nccl_available"]]
        if without_nccl:
            failures.append(
                f"CUDA supernode {index} lacks NCCL on rank(s): "
                + ", ".join(str(rank) for rank in without_nccl)
            )
        ring_edges = {
            tuple(sorted((rank, (rank + 1) % world_size))) for rank in range(world_size)
        }
        if any(
            tuple(sorted((left, right))) not in ring_edges
            for left, right in zip(members, members[1:])
        ):
            failures.append(
                f"CUDA supernode {index} ranks must be adjacent in the outer Ring"
            )
    return failures


def _cuda_supernode_records(
    specs: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """Describe structurally admissible pairs without claiming a transport test."""

    results: list[dict[str, Any]] = []
    for index, members_value in enumerate(specs, start=1):
        members = tuple(sorted(int(rank) for rank in members_value))
        results.append(
            {
                "id": f"cuda-supernode-{index}",
                "members": list(members),
                "transport_tested": None,
                "nccl_ready": True,
                "verified": False,
                "next": "verify this pair from the oMLX dashboard",
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an MLX Ring spanning Metal and CUDA ranks."
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Join the MLX group configured by mlx.launch.",
    )
    parser.add_argument("--backend", default="ring", choices=("ring",))
    parser.add_argument("--expect-ranks", type=int, default=None)
    parser.add_argument(
        "--require-accelerators",
        type=_parse_required_accelerators,
        default=set(),
        metavar="KINDS",
        help="Comma-separated required set, for example metal,cuda.",
    )
    parser.add_argument(
        "--collective-mib",
        type=lambda value: _parse_positive_ints(value, label="collective MiB"),
        default=(1, 64),
        metavar="SIZES",
    )
    parser.add_argument(
        "--cuda-supernode-ranks",
        action="append",
        type=_parse_rank_set,
        default=[],
        metavar="RANKS",
        help=(
            "Comma-separated adjacent CUDA ranks in one ConnectX supernode; "
            "repeat for multiple groups."
        ),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--checksum-atol", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expect_ranks is not None and args.expect_ranks < 1:
        raise SystemExit("--expect-ranks must be positive")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if not math.isfinite(args.checksum_atol) or args.checksum_atol < 0:
        raise SystemExit("--checksum-atol must be finite and non-negative")

    try:
        import mlx.core as mx

        capability = detect_local_capability(mx)
        if not capability.ring_available:
            raise RuntimeError("the MLX Ring backend is unavailable")

        if not args.distributed:
            checksums = _representative_compute(mx)
            print(
                json.dumps(
                    {
                        "type": "heterogeneous_pool_probe_local",
                        "ok": True,
                        "capability": asdict(capability),
                        "compute": checksums,
                        "next": "run under mlx.launch with --distributed",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0

        group = mx.distributed.init(backend=args.backend, strict=True)
        topology = _distributed_topology(group, mx, capability)
        failures: list[str] = []
        if args.expect_ranks is not None and group.size() != args.expect_ranks:
            failures.append(
                f"world size is {group.size()}, expected {args.expect_ranks}"
            )
        available_accelerators = set(topology["accelerators"])
        missing = args.require_accelerators - available_accelerators
        if missing:
            failures.append(f"missing accelerator(s): {', '.join(sorted(missing))}")
        if len(topology["mlx_versions"]) != 1:
            failures.append(
                "nominal MLX versions differ: " + ", ".join(topology["mlx_versions"])
            )
        supernode_validation_failures = _supernode_failures(
            args.cuda_supernode_ranks,
            topology,
        )
        failures.extend(supernode_validation_failures)

        checksums: dict[str, float] = {}
        compute_error: str | None = None
        try:
            checksums = _representative_compute(mx)
        except Exception as exc:
            compute_error = f"{type(exc).__name__}: {exc}"
        compute_status = _all_gather_rows(
            group,
            mx,
            [0.0 if compute_error else 1.0],
        )
        failed_compute_ranks = [
            rank for rank, row in enumerate(compute_status) if row != [1.0]
        ]
        if failed_compute_ranks:
            failures.append(
                "representative compute failed on rank(s): "
                + ", ".join(str(rank) for rank in failed_compute_ranks)
            )
            checksum_report: dict[str, Any] = {
                "ok": False,
                "skipped": True,
                "reason": "one or more ranks failed representative compute",
            }
            collectives: list[dict[str, Any]] = []
        else:
            checksum_report = _checksum_report(
                group,
                mx,
                checksums,
                tolerance=args.checksum_atol,
            )
            if not checksum_report["ok"]:
                failures.append(
                    "cross-rank compute checksum spread exceeds "
                    f"{args.checksum_atol}: {checksum_report['spreads']}"
                )
            collectives = _collective_benchmark(
                group,
                mx,
                sizes_mib=args.collective_mib,
                repeats=args.repeats,
            )
        supernodes: list[dict[str, Any]] = []
        if (
            args.cuda_supernode_ranks
            and not failed_compute_ranks
            and not supernode_validation_failures
        ):
            supernodes = _cuda_supernode_records(args.cuda_supernode_ranks)

        record = {
            "type": "heterogeneous_pool_probe_rank",
            "ok": not failures,
            "rank": group.rank(),
            "world_size": group.size(),
            "capability": asdict(capability),
            "compute": checksums,
            "compute_error": compute_error,
            "failures": failures,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        if group.rank() == 0:
            print(
                json.dumps(
                    {
                        "type": "heterogeneous_pool_probe_summary",
                        "ok": not failures,
                        "topology": topology,
                        "compute_parity": checksum_report,
                        "collectives": collectives,
                        "cuda_supernodes": supernodes,
                        "failures": failures,
                        "next": (
                            "run oMLX's unequal pipeline smoke on this hostfile"
                            if not failures
                            else "resolve failures before loading model weights"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return 0 if not failures else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "type": "heterogeneous_pool_probe_error",
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
