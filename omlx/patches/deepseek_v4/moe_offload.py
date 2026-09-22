# SPDX-License-Identifier: Apache-2.0
"""Expert offload for the DeepSeek V4 / GLM-5.x-flash MoE block.

The ``deepseek_v4`` and ``glm5_next`` checkpoints run their routed experts
through this package's own :class:`~omlx.patches.deepseek_v4.switch_layers
.SwitchGLU`, which the common adapter cannot wrap: it is not the stock
mlx-lm module, and its forward carries native Metal fast paths (the
``deepseek_*_gather_qmm`` block kernels, the gate/up pair kernels and the
native ``glm_moe_weighted_sum``). Re-implementing that forward would fork
it, so this adapter keeps the module and swaps what it computes on.

Each projection's parameters are replaced, before lazy weights materialize,
by slot tensors of ``capacity`` experts; expert ids are translated to slot
ids and the module's own ``__call__`` runs unchanged on them. Every kernel
decision inside it is a function of the routes, the projections' quantization
metadata and the tensors' shapes — ``num_experts`` is a property of the
weight shape, so it follows the slots — and every use of an index is a
gather, so a route computes the same numbers against its slot as it would
against its expert. A miss reads the expert's gate, up and down slabs from
the checkpoint's own safetensors with positional reads on a bounded pool, as
the GLM DSA and DeepSeek V4.1 adapters do.

Over-capacity prefill, where one call routes to more distinct experts than
the cache holds, is chunked on expert boundaries exactly as the other
adapters do: each expert is installed at most once per call, chunks run
under the module's own forward with one route per row, and the weighted sum,
when the module's own forward would have applied it natively, is applied to
the reassembled routes the way the caller does when the kernel is
unavailable.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import wait
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ...custom_kernels.glm_moe_dsa import fast as glm_fast
from ...scheduler import _sync_and_clear_cache
from ..moe_expert_offload import (
    _DTYPES,
    CheckpointExpertStore,
    _GLUStoreView,
    _io_batch,
    _io_pool,
    _minimum_experts,
    _resolve_model_dir,
)
from .switch_layers import _sort_threshold

logger = logging.getLogger(__name__)

_PROJS = ("gate_proj", "up_proj", "down_proj")

# Bound large expert reads in addition to the shared pool's expert-count limit.
INFLIGHT_BYTES = 512 * 1024 * 1024


def is_deepseek_v4_switch_glu(obj) -> bool:
    """This package's SwitchGLU (never an already-wrapped one)."""
    return (
        type(obj).__name__ == "SwitchGLU"
        and type(obj).__module__ == "omlx.patches.deepseek_v4.switch_layers"
    )


def _fields(lin) -> tuple[str, ...]:
    return ("weight", "scales") + (("biases",) if lin.get("biases") is not None else ())


def _is_quantized(lin) -> bool:
    return type(lin).__name__ == "QuantizedSwitchLinear" and all(
        hasattr(lin, a) for a in ("group_size", "bits", "mode")
    )


def resolve_view(glu, store: CheckpointExpertStore, path: str):
    """Validate the checkpoint against the module; ``(view, None)`` or ``(None, reason)``.

    The checkpoint must hold the stacked ``gate_proj``/``up_proj``/``down_proj``
    tensors under the module's tree path, in shapes the module's projections
    match and a storage dtype the store can read.
    """
    for proj in _PROJS:
        lin = glu.get(proj)
        if lin is None or not _is_quantized(lin):
            return None, f"{proj} is not a QuantizedSwitchLinear"
        if "bias" in lin:
            return None, f"{proj} has per-expert bias (unsupported)"
    view = _GLUStoreView(store, path)
    for proj in _PROJS:
        lin = glu[proj]
        for field in _fields(lin):
            name = view._name(proj, field, 0)
            if not store.has(name):
                return None, f"checkpoint has no tensor {name!r}"
            shape, dtype = store.spec(name)
            if shape != tuple(lin[field].shape):
                return (
                    None,
                    f"{name!r} shape {shape} != expected {tuple(lin[field].shape)}",
                )
            if dtype not in _DTYPES:
                return None, f"{name!r} has unsupported dtype {dtype!r}"
    return view, None


class _SlotCache:
    """LRU slots living inside the module's own projection tensors."""

    moe_offload_cache = True

    def __init__(self, glu, capacity: int, view: _GLUStoreView):
        self.glu = glu
        self.view = view
        self.n_experts = glu[_PROJS[0]]["weight"].shape[0]
        self.capacity = min(capacity, self.n_experts)
        self.resident: dict[str, list] = {}
        for proj in _PROJS:
            lin = glu[proj]
            arrays = []
            for field in _fields(lin):
                src = lin[field]
                slots = mx.zeros((self.capacity, *src.shape[1:]), dtype=src.dtype)
                setattr(lin, field, slots)  # drops the lazy full-size array
                arrays.append(slots)
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
            (proj, field, proj) for proj in _PROJS for field in _fields(glu[proj])
        ]
        self.expert_bytes = sum(
            view.plan(src, field, 0).nbytes for _, field, src in self._writes
        )

    def ensure(self, idx: mx.array) -> None:
        if self.warm:
            return
        self.ensure_ids(idx.reshape(-1).tolist())

    def ensure_ids(self, ids) -> None:
        """Load missing experts through the shared reader and protect current hits."""
        needed = list(dict.fromkeys(int(e) for e in ids))
        misses = []
        for e in needed:
            if e in self.slot_of:
                self.slot_of[e] = self.slot_of.pop(e)  # re-insert: LRU order
                self.hits += 1
            else:
                misses.append(e)
        if not misses:
            return
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

        try:
            submit(window)
            for done, e in enumerate(misses):
                # Refill before this expert's writes so at most ``window``
                # experts' bytes exist at once, counting the one written here.
                submit(done + window)
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
                for (proj, field, _), plan, raw in raws:
                    self.glu[proj][field][slot] = CheckpointExpertStore.to_mx(plan, raw)
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


class OffloadedSwitchGLU(nn.Module):
    """DeepSeek V4 SwitchGLU whose experts live in a :class:`_SlotCache`."""

    def __init__(self, glu, capacity: int, view: _GLUStoreView):
        super().__init__()
        # A plain attribute, like the other adapters: the module with the slot
        # tensors stays out of the tree so parameter walks see the wrapper.
        self.cache = _SlotCache(glu, capacity, view)

    def _forward_expert_major(self, flat_x: mx.array, ids: list[int], k: int):
        """Routes sorted by expert, cut into chunks of ``capacity`` experts."""
        c = self.cache
        d_model = flat_x.shape[-1]
        ids_np = np.asarray(ids, dtype=np.int64)
        order = np.argsort(ids_np, kind="stable")
        sorted_ids = ids_np[order]
        run_starts = np.flatnonzero(np.diff(sorted_ids)) + 1
        run_starts = np.concatenate(([0], run_starts))
        cuts = run_starts[:: c.capacity].tolist() + [len(ids)]
        outs = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            chunk_ids = sorted_ids[start:end]
            c.ensure_ids(np.unique(chunk_ids).tolist())
            slots = mx.take(c.map, mx.array(chunk_ids, dtype=mx.int32))
            t_idx = mx.array(order[start:end] // k, dtype=mx.int32)
            xe = mx.take(flat_x, t_idx, axis=0)
            o = c.glu(xe, slots.reshape(-1, 1))[:, 0, :]
            mx.eval(o)
            outs.append(o)
        out = mx.concatenate(outs, axis=0)
        inverse = mx.array(np.argsort(order, kind="stable"), dtype=mx.int32)
        return mx.take(out, inverse, axis=0).reshape(-1, k, d_model)

    def __call__(self, x: mx.array, indices: mx.array, scores=None, weighted_sum=False):
        c = self.cache
        flat_i = indices.reshape(-1, indices.shape[-1])
        n_tok, k = flat_i.shape
        if k > c.capacity:
            raise ValueError("Expert cache capacity is smaller than routing top-k")
        fits = n_tok * k <= c.capacity or n_tok == 1
        ids = None
        if not fits:
            ids = flat_i.reshape(-1).tolist()
            fits = len(set(ids)) <= c.capacity
        if fits:
            c.ensure(indices)
            slots = mx.take(c.map, indices)
            return c.glu(x, slots, scores=scores, weighted_sum=weighted_sum)
        y = self._forward_expert_major(x.reshape(-1, x.shape[-1]), ids, k)
        y = y.reshape(indices.shape + (x.shape[-1],))
        # Sum exactly when the module's own forward would have: it returns the
        # routes unsummed unless the call is sorted, carries float32 scores of
        # top-k 6 or 8 on half-precision activations, and the native kernel is
        # present — and the caller applies the scores itself otherwise.
        if (
            weighted_sum
            and scores is not None
            and indices.size
            >= _sort_threshold(c.glu.gate_proj, c.glu.up_proj, c.glu.down_proj)
            and scores.shape[-1] in (6, 8)
            and scores.dtype == mx.float32
            and y.dtype in (mx.float16, mx.bfloat16)
            and glm_fast.has_symbol("glm_moe_weighted_sum")
        ):
            y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        return y


def _iter_deepseek_v4_switch_glus(model):
    seen = set()

    def walk(parent, key, obj, path):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if is_deepseek_v4_switch_glu(obj):
            yield (parent, key, obj, path)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from walk(obj, k, v, f"{path}.{k}" if path else k)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                yield from walk(obj, i, v, f"{path}.{i}")

    yield from walk(None, None, model, "")


def apply_deepseek_v4_moe_expert_offload(
    model, model_path: str | Path, resident_fraction: float = 0.25
) -> int:
    """Wrap every covered DeepSeek V4 SwitchGLU; returns the number wrapped.

    Same contract as ``apply_moe_expert_offload``: runs before lazy weights
    materialize, honors the kill switch, and skips (with a logged reason)
    any module the checkpoint does not cover.
    """
    if os.environ.get("OMLX_MOE_EXPERT_OFFLOAD", "1") == "0":
        return 0
    targets = list(_iter_deepseek_v4_switch_glus(model))
    if not targets:
        return 0
    model_dir = _resolve_model_dir(model_path)
    if model_dir is None:
        return 0
    minimum = _minimum_experts(model_dir)
    store = CheckpointExpertStore(model_dir)
    if not store:
        logger.warning(
            "deepseek_v4 moe expert offload: no safetensors under %s", model_dir
        )
        return 0
    wrapped = 0
    total_bytes = resident_bytes = 0
    for parent, key, glu, path in targets:
        view, reason = resolve_view(glu, store, path)
        if view is None:
            logger.info(
                "deepseek_v4 moe expert offload: skipping %s (%s)", path, reason
            )
            continue
        n_experts = glu[_PROJS[0]]["weight"].shape[0]
        capacity = min(n_experts, max(minimum, round(n_experts * resident_fraction)))
        layer_bytes = sum(
            int(np.prod(glu[proj][field].shape)) * glu[proj][field].dtype.size
            for proj in _PROJS
            for field in _fields(glu[proj])
        )
        total_bytes += layer_bytes
        resident_bytes += layer_bytes * capacity // n_experts
        new = OffloadedSwitchGLU(glu, capacity, view)
        if isinstance(parent, nn.Module):
            setattr(parent, key, new)
        else:
            parent[key] = new
        wrapped += 1
        _sync_and_clear_cache()

    # glm5_next decoder layers compile their FFN block at decode shapes
    # (mlx_vlm glm5_next language.py ``compile_ffn``). The offloaded block
    # manages slots host-side — LRU map, pread fetches — and cannot be
    # traced into a compiled graph: ``tolist()`` inside a trace dies with
    # "eval during function transformations". Keep those layers eager; the
    # native gather kernels still run, only the graph fusion is lost, and
    # offload trades speed for memory anyway.
    for module in model.modules():
        if not getattr(module, "compile_ffn", False):
            continue
        if any(isinstance(c, OffloadedSwitchGLU) for c in module.modules()):
            module.compile_ffn = False
            module._ffn_c = None

    if wrapped:
        logger.info(
            "deepseek_v4 moe expert offload: wrapped %d layers at %.1f%% residency "
            "(expert tables: %.2f GB total, %.2f GB resident)",
            wrapped,
            100 * resident_fraction,
            total_bytes / 1e9,
            resident_bytes / 1e9,
        )
    return wrapped


__all__ = [
    "OffloadedSwitchGLU",
    "apply_deepseek_v4_moe_expert_offload",
    "is_deepseek_v4_switch_glu",
    "resolve_view",
]
