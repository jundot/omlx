# SPDX-License-Identifier: Apache-2.0
"""LRU expert slots inside a MoE module's own projection tensors.

Shared by the adapters that keep their module's forward and swap what it
computes on (DeepSeek V4 / glm5_next ``SwitchGLU``, Inkling 2 experts): each
projection in ``projs`` has its stacked tensors replaced by ``capacity`` slots,
expert ids are translated to slot ids through :attr:`SlotCache.map`, and a miss
reads the expert's slabs from the checkpoint into a slot. ModelOpt NVFP4
per-expert ``global_scale`` tables stay whole; a per-slot copy is kept beside
the slots.
"""

from __future__ import annotations

import time
from concurrent.futures import wait

import mlx.core as mx

from .moe_expert_offload import (
    _OVERLAP_GRACE_S,
    CheckpointExpertStore,
    _GLUStoreView,
    _gpu_keepalive,
    _io_batch,
    _io_pool,
)

# Bound large expert reads in addition to the shared pool's expert-count limit.
INFLIGHT_BYTES = 512 * 1024 * 1024


def _fields(lin) -> tuple[str, ...]:
    return ("weight", "scales") + (("biases",) if lin.get("biases") is not None else ())


class SlotCache:
    """LRU slots living inside the module's own projection tensors."""

    moe_offload_cache = True

    def __init__(self, glu, capacity: int, view: _GLUStoreView, projs):
        self.glu = glu
        self.view = view
        self.n_experts = glu[projs[0]]["weight"].shape[0]
        self.capacity = min(capacity, self.n_experts)
        self.resident: dict[str, list] = {}
        self.global_scales: dict[str, mx.array] = {}
        for proj in projs:
            lin = glu[proj]
            arrays = []
            for field in _fields(lin):
                src = lin[field]
                slots = mx.zeros((self.capacity, *src.shape[1:]), dtype=src.dtype)
                setattr(lin, field, slots)  # drops the lazy full-size array
                arrays.append(slots)
            if "global_scale" in lin:
                # ModelOpt NVFP4 tensor scales are tiny: keep all, slot a copy.
                self.global_scales[proj] = lin["global_scale"]
                lin.global_scale = mx.ones((self.capacity,), dtype=mx.float32)
                arrays += [self.global_scales[proj], lin["global_scale"]]
            self.resident[proj] = arrays
        self.slot_of: dict[int, int] = {}  # expert id -> slot, LRU ordered
        self.free = list(range(self.capacity))
        self.map = mx.full((self.n_experts,), -1, dtype=mx.int32)
        self.hits = self.misses = 0
        self.fetched_bytes = 0
        self.warm = False
        # (projection, field, checkpoint source projection) for one expert, in
        # slot-write order; the byte total sizes the inflight window.
        self._writes = [
            (proj, field, proj) for proj in projs for field in _fields(glu[proj])
        ]
        self.expert_bytes = sum(
            view.plan(src, field, 0).nbytes for _, field, src in self._writes
        )

    def ensure(self, idx: mx.array) -> None:
        if self.warm:
            return
        self.ensure_ids(idx.reshape(-1).tolist())

    def ensure_ids(self, ids, on_slow_read=None) -> bool:
        """Load missing experts through the shared reader and protect current hits.

        ``on_slow_read`` is the decode overlap hook of
        ``ExpertCache._ensure_ids``: if the first miss is still being read
        ``_OVERLAP_GRACE_S`` after its reads were issued, it runs (and may
        dispatch GPU work), what it returns is evaluated right before the first
        slot write, and the remaining reads are waited on with the GPU kept
        busy. Returns whether that happened.
        """
        needed = list(dict.fromkeys(int(e) for e in ids))
        misses = []
        for e in needed:
            if e in self.slot_of:
                self.slot_of[e] = self.slot_of.pop(e)  # re-insert: LRU order
                self.hits += 1
            else:
                misses.append(e)
        if not misses:
            return False
        protected = set(needed)
        pending: dict[int, list] = {}
        pool = _io_pool()
        window = 0
        if pool is not None:
            window = max(
                1, min(_io_batch(), INFLIGHT_BYTES // max(1, self.expert_bytes))
            )
        submitted = 0

        def plans(e):
            return [
                (write, self.view.plan(write[2], write[1], e)) for write in self._writes
            ]

        def submit(limit):
            nonlocal submitted
            if pool is None:
                return
            while submitted < min(limit, len(misses)):
                e = misses[submitted]
                submitted += 1
                pending[e] = [
                    (write, plan, pool.submit(CheckpointExpertStore.read, plan))
                    for write, plan in plans(e)
                ]

        keepalive = barrier = None
        slow = False
        try:
            submit(window)
            deadline = time.perf_counter() + _OVERLAP_GRACE_S
            for done, e in enumerate(misses):
                # Refill before this expert's writes so at most ``window``
                # experts' bytes exist at once, counting the one written here.
                submit(done + window)
                futures = [f for *_, f in pending.get(e, ())]
                if on_slow_read is not None:  # the first miss: nothing written yet
                    timeout = max(0.0, deadline - time.perf_counter())
                    if futures and wait(futures, timeout=timeout).not_done:
                        slow = True
                        barrier = on_slow_read()
                        keepalive = _gpu_keepalive()
                    on_slow_read = None
                if keepalive is not None and futures:
                    keepalive.wait(futures)
                if e in pending:
                    raws = [(write, plan, f.result()) for write, plan, f in pending[e]]
                else:
                    raws = [
                        (write, plan, CheckpointExpertStore.read(plan))
                        for write, plan in plans(e)
                    ]
                if self.free:
                    slot = self.free.pop()
                else:
                    victim = next(v for v in self.slot_of if v not in protected)
                    slot = self.slot_of.pop(victim)
                    self.map[victim] = -1
                if barrier is not None:
                    mx.eval(barrier)  # right before the first slot write
                    barrier = None
                for (proj, field, _), plan, raw in raws:
                    self.glu[proj][field][slot] = CheckpointExpertStore.to_mx(plan, raw)
                for proj, scale in self.global_scales.items():
                    self.glu[proj]["global_scale"][slot] = scale[e]
                self.slot_of[e] = slot
                self.map[e] = slot
                self.misses += 1
                self.fetched_bytes += sum(plan.nbytes for _, plan, _ in raws)
                pending.pop(e, None)
                del raws
        finally:
            futures = [f for group in pending.values() for _, _, f in group]
            for future in futures:
                future.cancel()
            if futures:
                wait(futures)
        self.warm = len(self.slot_of) == self.n_experts
        return slow
