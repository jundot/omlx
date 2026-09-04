# SPDX-License-Identifier: Apache-2.0
"""Safe localhost proof of the MLX distributed collective path."""

from __future__ import annotations

import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any


class CollectiveSmokeError(RuntimeError):
    """Raised when the local MLX collective diagnostic does not complete."""


LauncherRunner = Callable[..., subprocess.CompletedProcess[str]]


def _find_loopback_port_span(count: int = 2) -> int:
    """Find a currently free consecutive TCP port span on loopback."""

    if count < 1:
        raise ValueError("count must be positive")
    for _ in range(64):
        sockets: list[socket.socket] = []
        try:
            first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(first)
            first.bind(("127.0.0.1", 0))
            starting_port = first.getsockname()[1]
            if starting_port + count - 1 > 65535:
                continue
            for offset in range(1, count):
                candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(candidate)
                candidate.bind(("127.0.0.1", starting_port + offset))
            return starting_port
        except OSError:
            continue
        finally:
            for candidate in sockets:
                candidate.close()
    raise CollectiveSmokeError("could not reserve a loopback port span")


def _run_launcher(
    argv: Sequence[str],
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Run MLX's launcher in its own process group with a hard deadline."""

    process = subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy() | {"PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        detail = stderr.strip() or stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise CollectiveSmokeError(
            f"MLX collective did not finish within {timeout:.2f}s{suffix}"
        ) from exc
    return subprocess.CompletedProcess(
        args=list(argv),
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _parse_collective_records(stdout: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "collective_result":
            records.append(payload)
    return records


def run_local_collective_smoke(
    *,
    timeout: float = 20.0,
    runner: LauncherRunner = _run_launcher,
    starting_port: int | None = None,
) -> dict[str, Any]:
    """Run two local MLX ranks and verify a ring all-sum of ``1 + 2``."""

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if starting_port is None:
        starting_port = _find_loopback_port_span(2)
    if not 1 <= starting_port <= 65534:
        raise ValueError("starting_port must leave room for two ranks")

    launcher = (
        "from mlx._distributed_utils.launch import main; raise SystemExit(main() or 0)"
    )
    argv = [
        sys.executable,
        "-c",
        launcher,
        "--backend",
        "ring",
        "--hosts",
        "127.0.0.1",
        "--repeat-hosts",
        "2",
        "--starting-port",
        str(starting_port),
        "--",
        sys.executable,
        "-m",
        "omlx.cluster.collective_worker",
    ]

    started_at = time.monotonic()
    try:
        completed = runner(argv, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CollectiveSmokeError(f"could not launch MLX collective: {exc}") from exc

    records = _parse_collective_records(completed.stdout)
    detail = completed.stderr.strip()
    if completed.returncode != 0:
        suffix = f": {detail}" if detail else ""
        raise CollectiveSmokeError(
            f"MLX launcher exited with code {completed.returncode}{suffix}"
        )

    records_by_rank = {
        record["rank"]: record
        for record in records
        if isinstance(record.get("rank"), int)
    }
    expected_ranks = {0, 1}
    if len(records) != 2 or set(records_by_rank) != expected_ranks:
        suffix = f": {detail}" if detail else ""
        raise CollectiveSmokeError(
            "MLX collective did not return one result from each rank" + suffix
        )
    for rank, record in records_by_rank.items():
        if record.get("size") != 2 or record.get("sum") != 3:
            raise CollectiveSmokeError(
                f"rank {rank} returned an invalid collective result: {record}"
            )

    try:
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version = "unknown"
    return {
        "ok": True,
        "backend": "ring",
        "loopback_only": True,
        "rank_count": 2,
        "starting_port": starting_port,
        "mlx_version": mlx_version,
        "expected_sum": 3,
        "elapsed_seconds": time.monotonic() - started_at,
        "ranks": [records_by_rank[rank] for rank in sorted(expected_ranks)],
    }


def run_local_pipeline_smoke(
    *,
    timeout: float = 30.0,
    runner: LauncherRunner = _run_launcher,
    starting_port: int | None = None,
) -> dict[str, Any]:
    """Execute a tiny sharded Nemotron-H forward pass across two local ranks."""

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if starting_port is None:
        starting_port = _find_loopback_port_span(2)
    if not 1 <= starting_port <= 65534:
        raise ValueError("starting_port must leave room for two ranks")

    launcher = (
        "from mlx._distributed_utils.launch import main; raise SystemExit(main() or 0)"
    )
    argv = [
        sys.executable,
        "-c",
        launcher,
        "--backend",
        "ring",
        "--hosts",
        "127.0.0.1",
        "--repeat-hosts",
        "2",
        "--starting-port",
        str(starting_port),
        "--",
        sys.executable,
        "-m",
        "omlx.cluster.pipeline_smoke_worker",
    ]
    started_at = time.monotonic()
    try:
        completed = runner(argv, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CollectiveSmokeError(f"could not launch pipeline smoke: {exc}") from exc

    records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "pipeline_result":
            records.append(payload)
    detail = completed.stderr.strip()
    if completed.returncode != 0:
        suffix = f": {detail}" if detail else ""
        raise CollectiveSmokeError(
            f"MLX pipeline smoke exited with code {completed.returncode}{suffix}"
        )
    by_rank = {
        record["rank"]: record
        for record in records
        if isinstance(record.get("rank"), int)
    }
    if len(records) != 2 or set(by_rank) != {0, 1}:
        suffix = f": {detail}" if detail else ""
        raise CollectiveSmokeError(
            "pipeline smoke did not return one result from each rank" + suffix
        )
    for rank, record in by_rank.items():
        if (
            record.get("size") != 2
            or record.get("model_type") != "nemotron_h"
            or record.get("local_layer_count") != 2
            or record.get("local_cache_count") != 1
            or record.get("output_shape") != [1, 3, 32]
        ):
            raise CollectiveSmokeError(
                f"rank {rank} returned an invalid pipeline result: {record}"
            )
    checksums = [float(by_rank[rank]["checksum"]) for rank in (0, 1)]
    if abs(checksums[0] - checksums[1]) > 1e-4:
        raise CollectiveSmokeError(f"pipeline rank checksums differ: {checksums}")
    return {
        "ok": True,
        "backend": "ring",
        "loopback_only": True,
        "model_type": "nemotron_h",
        "rank_count": 2,
        "starting_port": starting_port,
        "elapsed_seconds": time.monotonic() - started_at,
        "ranks": [by_rank[rank] for rank in (0, 1)],
    }


def _run_local_minimax_decode_smoke(
    *,
    timeout: float = 45.0,
    runner: LauncherRunner = _run_launcher,
    starting_port: int | None = None,
) -> dict[str, Any]:
    """Run three tiny MiniMax decode steps through two real MLX ranks."""

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if starting_port is None:
        starting_port = _find_loopback_port_span(2)
    if not 1 <= starting_port <= 65534:
        raise ValueError("starting_port must leave room for two ranks")

    launcher = (
        "from mlx._distributed_utils.launch import main; raise SystemExit(main() or 0)"
    )
    argv = [
        sys.executable,
        "-c",
        launcher,
        "--backend",
        "ring",
        "--hosts",
        "127.0.0.1",
        "--repeat-hosts",
        "2",
        "--starting-port",
        str(starting_port),
        "--",
        sys.executable,
        "-m",
        "omlx.cluster.minimax_decode_smoke_worker",
    ]
    started_at = time.monotonic()
    try:
        completed = runner(argv, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CollectiveSmokeError(
            f"could not launch MiniMax decode smoke: {exc}"
        ) from exc

    records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "minimax_decode_result":
            records.append(payload)
    detail = completed.stderr.strip()
    if completed.returncode != 0:
        suffix = f": {detail}" if detail else ""
        raise CollectiveSmokeError(
            f"MLX MiniMax decode smoke exited with code "
            f"{completed.returncode}{suffix}"
        )
    by_rank = {
        record["rank"]: record
        for record in records
        if isinstance(record.get("rank"), int)
    }
    if len(records) != 2 or set(by_rank) != {0, 1}:
        suffix = f": {detail}" if detail else ""
        raise CollectiveSmokeError(
            "MiniMax decode smoke did not return one result from each rank" + suffix
        )
    for rank, record in by_rank.items():
        expected_skip = rank != 0
        if (
            record.get("size") != 2
            or record.get("model_type") != "minimax_m3_vl"
            or record.get("steps") != 3
            or record.get("skip_logits") is not expected_skip
            or record.get("local_layer_count") != 2
            or record.get("local_cache_count") != 2
            or record.get("logprobs_width") != 128
        ):
            raise CollectiveSmokeError(
                f"rank {rank} returned an invalid MiniMax decode result: {record}"
            )
    tokens = [int(by_rank[rank]["next_token"]) for rank in (0, 1)]
    if tokens[0] != tokens[1]:
        raise CollectiveSmokeError(f"MiniMax pipeline rank tokens differ: {tokens}")
    return {
        "ok": True,
        "backend": "ring",
        "loopback_only": True,
        "model_type": "minimax_m3_vl",
        "rank_count": 2,
        "steps": 3,
        "starting_port": starting_port,
        "elapsed_seconds": time.monotonic() - started_at,
        "ranks": [by_rank[rank] for rank in (0, 1)],
    }
