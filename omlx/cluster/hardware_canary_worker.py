# SPDX-License-Identifier: Apache-2.0
"""Private fault-injection worker for the local Apple-silicon canary."""

from __future__ import annotations

import argparse
import json
import signal
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation-bytes", type=int, required=True)
    args = parser.parse_args()

    try:
        import mlx.core as mx

        if args.allocation_bytes <= 0:
            raise ValueError("allocation must be positive")
        # The supervisor has to escalate a rank which is stuck below Python's
        # normal teardown path, as a Metal/JACCL collective can be. Keep the
        # allocation modest; this worker tests ownership and cleanup, not OOM.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        values = mx.ones((args.allocation_bytes // 4,), dtype=mx.float32)
        checksum = mx.sum(values)
        mx.eval(values, checksum)
        print(
            json.dumps(
                {
                    "type": "metal_holder_ready",
                    "active_memory_bytes": int(mx.get_active_memory()),
                    "allocation_bytes": int(args.allocation_bytes),
                    "checksum": float(checksum.item()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        while True:
            time.sleep(60)
    except Exception as exc:  # pragma: no cover - reported to the parent
        print(
            f"oMLX Metal holder failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
