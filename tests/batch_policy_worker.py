# SPDX-License-Identifier: Apache-2.0
"""One rank of tests/test_batch_policy_lockstep.py (ring backend over loopback).

Drives the real multi-row ``BatchPolicy`` the way the Lightning MTP batch path
does, with a per-rank clock: the MTP cycles of rank ``SKEW_RANK`` (default 1)
take ``SKEW`` times as long (default 1.3); the ordinary decode step takes the
same time everywhere. Each rank
prints the (mode, depth) sequence it decided; with ``SYNC=1`` the distributed
sync is installed first. Ranks that decide differently would send verify windows
of different length through the pipelined forward.
"""

import json
import os
import sys

import mlx.core as mx


def main() -> int:
    group = mx.distributed.init(backend="ring", strict=True)
    rank = group.rank()
    slow_rank = int(os.environ.get("SKEW_RANK", "1"))
    skew = float(os.environ.get("SKEW", "1.3")) if rank == slow_rank else 1.0
    if os.environ.get("SYNC") == "1":
        from omlx.patches.mlx_lm_mtp import distributed_sync

        assert distributed_sync.install_batch_policy()
    from omlx.patches.mlx_lm_mtp.batch_policy import BatchPolicy

    policy = BatchPolicy((1, 2, 3), 4)
    decisions = []
    now = 0.0
    for _ in range(160):
        started = now
        if policy.needs_standard():
            now += 0.030
            policy.observe_standard(policy.cycle_time_ms("standard", started, now))
            decisions.append(["std", 0])
            continue
        depth = policy.cur
        now += (0.060 + 0.0075 * depth) * skew
        elapsed = policy.cycle_time_ms("mtp", started, now)
        policy.observe_mtp(depth, [min(depth, 2)] * 3, elapsed, stable=True)
        decisions.append(["mtp", depth])
        if policy.should_park():
            policy.park()
            decisions.append(["park", 0])
    print(json.dumps({"rank": rank, "decisions": decisions}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
