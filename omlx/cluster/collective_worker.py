# SPDX-License-Identifier: Apache-2.0
"""Internal MLX collective worker used by the local cluster smoke test."""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        import mlx.core as mx

        group = mx.distributed.init()
        rank = group.rank()
        size = group.size()
        value = mx.array(rank + 1)
        total = mx.distributed.all_sum(value, stream=mx.cpu)
        mx.eval(total)
        result = {
            "type": "collective_result",
            "backend": "ring",
            "rank": rank,
            "size": size,
            "input": rank + 1,
            "sum": total.item(),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(f"MLX collective worker failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
