# SPDX-License-Identifier: Apache-2.0
"""Keep Lightning MTP in lockstep across the ranks of a distributed group.

Every rank of an mlx-lm pipeline/tensor deployment runs the same generation
loop (SPMD) and issues the same collectives in the same order. The MTP draft
depth is the one decision that is NOT a pure function of the data: the
adaptive ``_DepthController`` picks it from wall-clock cycle times (and decides
when to hand the sequence back to the standard step), and every rank measures
its own clock. Ranks that disagree on the verify-window length then send and
receive activations of different shapes and the group deadlocks silently.

``install()`` makes rank 0 authoritative: after each controller update every
rank adopts rank 0's depth and exit streak through one tiny all-sum. With a
fixed depth (``mtp_fixed_depth``) no controller exists and nothing is needed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MARKER = "_omlx_distributed_sync"


def install() -> bool:
    """Wrap ``_DepthController.observe`` so all ranks adopt rank 0's decision."""

    import mlx.core as mx

    from . import batch_generator

    controller = getattr(batch_generator, "_DepthController", None)
    if controller is None:
        return False
    original = controller.observe
    if getattr(original, _MARKER, False):
        return True

    def observe(self, *args, **kwargs):
        original(self, *args, **kwargs)
        group = mx.distributed.init()
        if group.size() <= 1:
            return
        local = mx.array([int(self.cur), int(self.exit_streak)], dtype=mx.int32)
        if group.rank() != 0:
            local = mx.zeros_like(local)
        agreed = mx.distributed.all_sum(local, group=group)
        mx.eval(agreed)
        self.cur, self.exit_streak = (int(v) for v in agreed.tolist())

    setattr(observe, _MARKER, True)
    controller.observe = observe
    logger.info("MTP depth controller synchronised across ranks (rank 0 decides)")
    return True
