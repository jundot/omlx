# SPDX-License-Identifier: Apache-2.0
"""One rank of tests/test_mtp_depth_lockstep.py (ring backend over loopback).

Forces a different MTP max draft depth on each rank (what the per-chip ceiling
does on a cluster mixing an M5 with an older Mac), runs the install step the
cluster worker runs, and prints the depth the rank would build its model with.
"""

import json
import os
import sys

import mlx.core as mx


def main() -> int:
    group = mx.distributed.init(backend="ring", strict=True)
    rank = group.rank()
    from omlx.patches import mlx_lm_mtp
    from omlx.patches.mlx_lm_mtp import distributed_sync

    forced = [int(v) for v in os.environ["DEPTHS"].split(",")][rank]
    mlx_lm_mtp.set_mtp_depth(forced, fixed=False)
    distributed_sync.install()
    print(
        json.dumps(
            {
                "rank": rank,
                "depth": mlx_lm_mtp.get_mtp_depth(),
                "fixed": mlx_lm_mtp.is_mtp_depth_fixed(),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
