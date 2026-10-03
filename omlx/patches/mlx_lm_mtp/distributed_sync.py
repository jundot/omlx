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
rank adopts rank 0's depth and exit streak through one tiny all-sum. With 2+
rows in flight the decisions come from ``BatchPolicy`` instead, whose only
non-deterministic input is the measured cycle time; ``install_batch_policy``
feeds every rank rank 0's value. With a fixed depth (``mtp_fixed_depth``) no
controller exists and nothing is needed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MARKER = "_omlx_distributed_sync"


def _agree_on_cycle_time(mx, group, value):
    """Rank 0's cycle time (or None) as seen by every rank: one 8-byte all-sum."""
    local = mx.array(
        [0.0, 0.0] if value is None else [1.0, float(value)], dtype=mx.float32
    )
    if group.rank() != 0:
        local = mx.zeros_like(local)
    agreed = mx.distributed.all_sum(local, group=group)
    mx.eval(agreed)
    present, milliseconds = agreed.tolist()
    return float(milliseconds) if present else None


def install_batch_policy() -> bool:
    """Make the multi-row ``BatchPolicy`` a pure function of rank 0's inputs.

    With 2+ rows in flight the draft depth, the standard-vs-MTP mode and parking
    all come from ``BatchPolicy``. Its only non-deterministic input is the
    measured cycle time, and each rank reads its own clock, so one rank can park
    MTP (or pick another depth) while its peer does not. The ranks then verify
    windows of different length through the pipelined forward and the group
    hangs or dies in the collective. Feeding every rank rank 0's cycle time
    keeps the whole policy state identical.
    """

    import mlx.core as mx

    from .batch_policy import BatchPolicy

    original = BatchPolicy.cycle_time_ms
    if getattr(original, _MARKER, False):
        return True

    def cycle_time_ms(self, mode, started, finished):
        value = original(self, mode, started, finished)
        group = mx.distributed.init()
        if group.size() <= 1:
            return value
        return _agree_on_cycle_time(mx, group, value)

    setattr(cycle_time_ms, _MARKER, True)
    BatchPolicy.cycle_time_ms = cycle_time_ms
    logger.info("MTP batch policy synchronised across ranks (rank 0 clock)")
    return True


def sync_max_depth() -> tuple[int, bool]:
    """Rank 0's max draft depth (and fixed flag) on every rank, before the model is built.

    ``maybe_apply_pre_load_patches`` derives the depth ceiling on each rank from
    its own model and chip (``qwen3_5``: 4 where the M5 packed verify kernels
    exist, 3 elsewhere). The verify window length is what the pipeline's
    all_gather shapes depend on, so a cluster mixing an M5 with an older Mac
    disagreed on the very first verify. The patched ``TextModel.__init__`` copies
    the global onto the instance when the model is built, so this has to run
    after the patches and before the load; ``install`` does.
    """

    import mlx.core as mx

    from . import get_mtp_depth, is_mtp_depth_fixed, set_mtp_depth

    group = mx.distributed.init()
    depth, fixed = int(get_mtp_depth()), bool(is_mtp_depth_fixed())
    if group.size() <= 1:
        return depth, fixed
    local = mx.array([depth, int(fixed)], dtype=mx.int32)
    if group.rank() != 0:
        local = mx.zeros_like(local)
    agreed = mx.distributed.all_sum(local, group=group)
    mx.eval(agreed)
    agreed_depth, agreed_fixed = int(agreed[0].item()), bool(agreed[1].item())
    if (agreed_depth, agreed_fixed) != (depth, fixed):
        logger.warning(
            "MTP max depth differs across ranks (local %d fixed=%s, rank 0 %d "
            "fixed=%s): adopting rank 0's",
            depth,
            fixed,
            agreed_depth,
            agreed_fixed,
        )
        set_mtp_depth(agreed_depth, fixed=agreed_fixed)
    return agreed_depth, agreed_fixed


def install() -> bool:
    """Wrap ``_DepthController.observe`` so all ranks adopt rank 0's decision."""

    import mlx.core as mx

    from . import batch_generator

    sync_max_depth()
    install_batch_policy()

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
