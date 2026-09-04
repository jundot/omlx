# SPDX-License-Identifier: Apache-2.0
"""Minimal isolated cluster worker protocol.

This process intentionally does not import MLX or initialize a distributed group.
It proves the supervisor/process/protocol boundary that later JACCL workers use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .models import WORKER_PROTOCOL_VERSION


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _ready_event(rank: int, plan_hash: str) -> dict[str, Any]:
    return {
        "type": "ready",
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "pid": os.getpid(),
        "rank": rank,
        "plan_hash": plan_hash,
        "jaccl_environment": {
            "coordinator_configured": bool(os.environ.get("MLX_JACCL_COORDINATOR")),
            "ibv_devices_configured": bool(os.environ.get("MLX_IBV_DEVICES")),
            "ring": os.environ.get("MLX_JACCL_RING") == "1",
        },
    }


def run_stdio_worker(*, rank: int, plan_hash: str) -> int:
    """Serve the small newline-delimited JSON control protocol on stdio."""

    _emit(_ready_event(rank, plan_hash))
    for raw_line in sys.stdin:
        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            _emit({"type": "error", "error": "invalid_json"})
            continue
        if not isinstance(message, dict):
            _emit({"type": "error", "error": "message_must_be_an_object"})
            continue

        command = message.get("command")
        if command == "ping":
            _emit(
                {
                    "type": "pong",
                    "protocol_version": WORKER_PROTOCOL_VERSION,
                    "rank": rank,
                    "nonce": message.get("nonce"),
                }
            )
        elif command == "shutdown":
            _emit(
                {
                    "type": "stopped",
                    "protocol_version": WORKER_PROTOCOL_VERSION,
                    "rank": rank,
                }
            )
            return 0
        else:
            _emit(
                {
                    "type": "error",
                    "error": "unknown_command",
                    "command": command,
                }
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Internal oMLX cluster worker")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run the newline-delimited JSON control protocol on stdin/stdout",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--plan-hash", default="smoke")
    args = parser.parse_args(argv)

    if args.rank < 0:
        parser.error("--rank must be non-negative")
    if not args.stdio:
        parser.error("no worker transport selected")
    return run_stdio_worker(rank=args.rank, plan_hash=args.plan_hash)


if __name__ == "__main__":
    raise SystemExit(main())
