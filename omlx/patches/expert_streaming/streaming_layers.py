# SPDX-License-Identifier: Apache-2.0
"""Streaming SwitchLinear/GLU modules over the expert cache + bank IO.

``StreamingSwitchLinear`` (bf16) and ``StreamingQuantizedSwitchLinear``
(INT4/INT8, HOBBIT dual-tier) resolve their routed expert demand through
the per-layer ``_LayerLoadContext`` and the ``ExpertLRUCache``, with the
F_RDADVISE readahead advisor and staged decode prefetch driving the next
layer's predicted misses. ``StreamingSwitchGLU`` owns the per-call
``_RemapPlan`` (one host sync per MoE layer), the gather-sort, the
warm/pin hooks and the persistent slot-arena path. The env knobs at the
top bound bank size, coalescing, speculation and arena residency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .._switch_sort import (
    gather_sort as _gather_sort,
)
from .._switch_sort import (
    scatter_unsort as _scatter_unsort,
)
from ._env import env_bool, env_int
from .bank_io import (
    _accepts_kwarg,
    _decode_call_shape,
    _io_pool,
    _layer_ctx_mode,
    _LayerLoadContext,
    promote_np_array,
)
from .expert_cache import ExpertLRUCache
from .shard_bank import segment_runs
from .slot_cache import SlotArena
from .speculation import _STAGED_MAX_IDS, SpeculationState
from .staging import stage_headroom

logger = logging.getLogger(__name__)


_COALESCE_ENV = env_bool("OMLX_EXPERT_STREAMING_COALESCE", True)
# Suppress speculative staging when residency has no headroom
# (governor at floor or desperate-free band). 0 restores eager staging.
_STAGED_HEADROOM_ENV = env_bool("OMLX_EXPERT_STREAMING_STAGED_HEADROOM", True)
_BANK_MAX_BYTES = env_int(
    "OMLX_EXPERT_STREAMING_BANK_MAX_BYTES", 256 * 1024**2, lo=1
)
# Chunk oversized demand banks instead of declining them. Prefill demand
# (~500/512 experts per projection, ~450 MB) exceeds _BANK_MAX_BYTES, so
# the bank is allocated once and filled by cap-bounded read_expert_into
# slices — the segments/promotion contract is unchanged.
# Bound each contiguous run inside a bank read. Uncapped, a dense
# (prefill) demand set is one run = one preadv on one worker — the device
# sits at single-stream speed (~1.5 GB/s) instead of the ~2.7 GB/s
# it reaches with several reads in flight. The cap is in bytes; each
# component converts it to an expert count. 0 disables (runs unbounded).
# Measured constant (was OMLX_EXPERT_STREAMING_BANK_RUN_MAX_BYTES).
_BANK_RUN_MAX_BYTES = 64 * 1024**2
_RUN_MAX = env_int("OMLX_EXPERT_STREAMING_RUN_MAX", 16, lo=1)
_LAYER_BARRIER_ENV = env_bool("OMLX_EXPERT_STREAMING_LAYER_BARRIER", True)


# Cross-layer speculation via F_RDADVISE (default on;
# OMLX_EXPERT_STREAMING_RA=0 disables). Each MoE layer advises the next
# layer's previous-token experts so the NVMe fetch overlaps compute.
# Hints only: nothing is copied into userspace and the LRU is untouched.
#
# The advisor targets the NEXT layer's banks (ids come from
# spec_state.prev_uniq_by_layer[next_layer]); the F_RDADVISE key must be
# the next layer's stacking key, resolved through the converted-linears
# registry (spec_state.linears_by_layer), never this linear's own key.
# The whole speculation state (history, registry, stats, pending futures)
# is PER CONVERSION — it hangs off the cache/backing and dies with them,
# so two engines never share state. The advisory is capped at
# _MAX_ADVISE_ROWS and deduped per layer call (_RemapPlan.advised_runs)
# so the projections of one layer issue each next-layer run at most once.
_RA_ENV = env_bool("OMLX_EXPERT_STREAMING_RA", True)
# Persistent slot-arena residency (opt-out). Misses are bound into fixed
# rows of a per-(layer, proj) bank at admission and gather_qmm indexes it
# by slot id, so a cache hit costs zero per-call assembly — the bundle
# path pays one mx.stack per projection per call. The ExpertLRUCache
# keeps owning admission/eviction policy via the per-layer caps it
# reports; the arena owns payload rows and the id->slot map.
# OMLX_EXPERT_STREAMING_ARENA=0 opts out; every unsupported condition
# (non-quantized proj, dict backing, hot/cold split, demand > arena
# bound) falls back to the bundle path.
_ARENA_ENV = env_bool("OMLX_EXPERT_STREAMING_ARENA", True)
# Physical bound for one layer's arena across ALL its projections
# (weight+scales+biases rows). The cache's per-layer cap is a residency
# bound, not a byte budget — a governor-grown cap can otherwise commit
# tens of GiB of banks.
# A byte ceiling keeps the arena decode-shaped: demand sets larger than
# the row bound keep the bundle path (prefill is dense streaming — the
# bank-read path is already optimal there).
_ARENA_MAX_BYTES = (
    env_int("OMLX_EXPERT_STREAMING_ARENA_MAX_MIB", 128, lo=1) << 20
)
# Read jobs per projection inside arena produce. 1 = one coalesced bank
# read per projection (the union path's shape). >1 splits a projection's
# missing set into that many chunks — more pool parents, but each chunk
# plans a narrower run (breaks intra-projection coalescing). The run pool
# (_RUN_IO_QD=16 process-wide) stays the device queue-depth bound either
# way. Measured constant (was OMLX_EXPERT_STREAMING_ARENA_PROJ_JOBS).
_ARENA_PRODUCE_JOBS = 3


# Rows/positions above which a routed call is prefill-shaped rather than
# decode speculation: the advisory cap and the prefill-bypass cache guard
# share this boundary.
_PREFILL_SHAPE_MIN_ROWS = 64
# Hard cap on the advisory row set (rows above the boundary are
# prefill-shaped, not decode speculation).
_MAX_ADVISE_ROWS = _PREFILL_SHAPE_MIN_ROWS


def _spec_state_of(*holders: Any) -> SpeculationState | None:
    """First non-None ``spec_state`` across *holders* (backing, then cache).

    The converter stamps the per-conversion SpeculationState on the
    backing when it is an object and always on the cache; either may be
    the only carrier.
    """
    for holder in holders:
        state = getattr(holder, "spec_state", None)
        if state is not None:
            return state
    return None


@dataclass
class _RemapPlan:
    """Routing plan shared by every streaming linear of one MoE layer call.

    The first linear invoked in a layer builds the plan (mx.eval + host copy
    + np.unique + compact remap); the other projections (up/gate/down) reuse
    it — one sync per MoE layer instead of three.
    """

    indices_shape: Tuple[int, ...] = ()
    flat_np: Any = None
    uniq_list: list = field(default_factory=list)
    remapped: Any = None  # mx.array of compact ids, original indices shape
    positions: int = 0
    # Real token count (indices.shape[-2]) — the authoritative
    # decode-vs-prefill signal. Set by the GLU before the projections run;
    # _build_plan_into fills it from a 2-D indices array when unset. A
    # 1-token top_k=8 decode is 8 routed rows, and a 4-token verify can
    # have few rows — positions alone mislabels both.
    seq_len: Any = None
    uniq_np: Any = None  # host-side unique ids; promoted lazily for bias
    ctx: Any = None  # per-layer load context (quantized GLU)
    # (target_linear_id, run_first, run_count) already advised in this
    # layer call — the projections share one plan, so dedupe here.
    advised_runs: set = field(default_factory=set)
    # Arena mode (OMLX_EXPERT_STREAMING_ARENA): when the GLU binds the
    # demand set into persistent slot rows, arena_rhs is rhs_indices in
    # slot space and each linear reads the arena bank — no per-call
    # assembly.
    arena_rhs: Any = None


def _build_plan_into(plan: _RemapPlan, indices) -> None:
    """Populate a shared routing plan in place (called once per MoE layer)."""
    mx.eval(indices)
    # np.asarray, not np.array(copy=False): NumPy >=2.0 makes copy=False
    # mean "never copy" and an MLX array must copy device->host, so that
    # call raises. asarray(copy=None) is the direct host path.
    flat_np = np.asarray(indices).reshape(-1)
    uniq_np = np.unique(flat_np)
    uniq_list = uniq_np.tolist()
    # compact remap via searchsorted (uniq is sorted ascending):
    # vectorized C lookup
    remapped_np = np.searchsorted(uniq_np, flat_np).astype(np.int32)
    plan.indices_shape = tuple(indices.shape)
    # Only fill seq_len when the caller did not set it — the GLU stamps the
    # true token count before dispatch (a gather-sorted flat `idx` is 1-D
    # and would lose it here).
    if plan.seq_len is None:
        plan.seq_len = (
            int(indices.shape[-2])
            if getattr(indices, "ndim", 0) >= 2
            else None
        )
    plan.flat_np = flat_np
    plan.uniq_list = uniq_list
    plan.remapped = mx.array(remapped_np.reshape(indices.shape))
    plan.uniq_np = uniq_np
    plan.positions = int(flat_np.size)


# ---------------------------------------------------------------------------
# Streaming SwitchLinear variants
# ---------------------------------------------------------------------------

class _StreamingLinearBase(nn.Module):
    """Shared core of the streaming SwitchLinear variants (bf16 + quantized).

    Both carry the same layer/expert identity, dims, backing/cache handles,
    residual-bias tail and per-model IO override slots. Underscore attrs
    stay out of the nn.Module parameter tree.
    """

    def __init__(
        self,
        layer_idx: int,
        proj_name: str,
        num_experts: int,
        input_dims: int,
        output_dims: int,
        backing: Any,
        cache: ExpertLRUCache,
        has_bias: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.proj_name = proj_name
        self.num_experts = num_experts
        self._input_dims = input_dims
        self._output_dims = output_dims
        self.backing = backing
        self.cache = cache
        # Bias per expert (small, keep resident)
        self._bias: mx.array | None = None
        self._has_bias = has_bias
        # Per-model IO overrides (expert_streaming_io_depth/coalesce
        # settings). None → module env defaults (_EXPERT_IO_POOL /
        # _COALESCE_ENV).
        self._io_pool_override: Any = None
        self._coalesce_override: bool | None = None

    @property
    def input_dims(self) -> int:
        return self._input_dims

    @property
    def output_dims(self) -> int:
        return self._output_dims

    def set_bias(self, bias: mx.array | None) -> None:
        self._bias = bias

    def _finish(self, plan, out):
        """Shared tail: residual bias."""
        if self._bias is not None and self._has_bias:
            b_mini = mx.take(self._bias, mx.array(plan.uniq_np), axis=0)  # (U,O)
            out = out + mx.expand_dims(b_mini[plan.remapped], -2)
        return out


class StreamingSwitchLinear(_StreamingLinearBase):
    """BF16 SwitchLinear with streaming cache."""

    def __init__(
        self,
        layer_idx: int,
        proj_name: str,
        stacked_key: str,
        num_experts: int,
        input_dims: int,
        output_dims: int,
        backing: Any,
        cache: ExpertLRUCache,
        bias: bool = False,
    ):
        super().__init__(
            layer_idx,
            proj_name,
            num_experts,
            input_dims,
            output_dims,
            backing,
            cache,
            bias,
        )
        self.stacked_key = stacked_key

    def _read_expert_weight(self, key, expert_id: int) -> mx.array:
        """Backing fetch + cache put for a demand miss — the caller already
        ran the ``get`` that produced the miss, so hit/miss telemetry
        derives from that same probe (no separate membership test that a
        concurrent eviction could falsify between probe and load)."""
        # Load slice from backing
        if hasattr(self.backing, "load_expert"):
            w = self.backing.load_expert(self.stacked_key, expert_id)
        else:
            # dict-backed for tests: backing is dict[(layer, proj)] -> mx.array[E,O,I]
            bank = self.backing[(self.layer_idx, self.proj_name)]  # type: ignore[index]
            w = bank[expert_id]
            # ensure mx.array
            if not isinstance(w, mx.array):
                w = mx.array(w)
        self.cache.put(key, w)
        return w

    def __call__(self, x, indices, sorted_indices=False, plan: _RemapPlan | None = None):
        if plan is None:
            plan = _RemapPlan()
        if plan.flat_np is None:
            _build_plan_into(plan, indices)
        # Load each unique expert weight. Derive hit/miss from the get
        # result actually used for the load — a probe-then-load read can
        # go stale when a concurrent eviction lands between the two
        # (TOCTOU). get_many takes the cache lock once for the whole set.
        keys = [
            (self.layer_idx, int(eid), self.stacked_key)
            for eid in plan.uniq_list
        ]
        values = self.cache.get_many(keys)
        mini_weights = []
        for eid, key, w in zip(plan.uniq_list, keys, values):
            if w is None:
                w = self._read_expert_weight(key, int(eid))
            mini_weights.append(w)
        # Stack into mini-bank (U, O, I)
        if len(mini_weights) == 1:
            mini_bank = mx.expand_dims(mini_weights[0], 0)
        else:
            mini_bank = mx.stack(mini_weights, axis=0)
        # Call gather_mm with mini-bank
        out = mx.gather_mm(x, mini_bank.swapaxes(-1, -2), rhs_indices=plan.remapped, sorted_indices=sorted_indices)
        return self._finish(plan, out)


class _ArenaBank:
    """Plain (non-Module) holder for a linear's persistent slot rows.

    Keeping the mx.array banks off the nn.Module attribute path stops them
    registering as parameters — the GLU's SlotArena owns them through the
    arrays/bind closures instead.
    """

    __slots__ = ("weight", "scales", "biases")

    def __init__(self) -> None:
        self.weight = None
        self.scales = None
        self.biases = None


class _ResolvedDemand(NamedTuple):
    """One projection's resolved expert rows for a layer call.

    Output of ``StreamingQuantizedSwitchLinear._resolve_demand``: the raw
    NumPy bundles keyed by expert id, the cache hit/miss split, the
    single-promoted bank when the whole demand set was one contiguous
    read.
    """

    bundles: Dict[int, tuple]
    banked: Any


class StreamingQuantizedSwitchLinear(_StreamingLinearBase):
    """INT4/INT8 quantized SwitchLinear with streaming cache."""

    def __init__(
        self,
        layer_idx: int,
        proj_name: str,
        stacked_weight_key: str,
        stacked_scales_key: str,
        stacked_biases_key: str | None,
        num_experts: int,
        input_dims: int,
        output_dims: int,
        backing: Any,
        cache: ExpertLRUCache,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        has_bias: bool = False,
    ):
        super().__init__(
            layer_idx,
            proj_name,
            num_experts,
            input_dims,
            output_dims,
            backing,
            cache,
            has_bias,
        )
        self.stacked_weight_key = stacked_weight_key
        self.stacked_scales_key = stacked_scales_key
        self.stacked_biases_key = stacked_biases_key
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        # HOBBIT hot/cold split: hot experts keep the ORIGINAL packing
        # (source bits/gs below); the rest compute at the cold tier
        # (self._cold_bits/_cold_gs from expert_cold/ metadata). Empty set
        # or None bits = uniform tier — the single-bank path.
        self._hot_experts: set | None = None
        self._cold_bits: int | None = None
        self._cold_gs: int | None = None
        self._split_active = False
        # Memoized _per_slot_bytes: the summed per-slot size is fixed for a
        # given backing, but a failed reader probe (0) is not cached.
        self._per_slot_memo: int | None = None
        # The arena bank lives in a plain object so its mx.array rows stay
        # out of the module parameter tree.
        self._arena_bank = _ArenaBank()

    def set_hobbit_split(self, hot_experts, cold_bits: int, cold_gs: int) -> None:
        """Enable the dual-tier path for this linear (convert-time only)."""
        self._hot_experts = {int(e) for e in (hot_experts or [])}
        self._cold_bits = int(cold_bits)
        self._cold_gs = int(cold_gs)
        self._split_active = bool(
            self._hot_experts and self._cold_bits != self.bits
        )

    def _is_split_active(self) -> bool:
        # Conversion-time invariant — computed once at set_hobbit_split,
        # re-checked per expert on the bundle_key hot path.
        return self._split_active

    def _tier_of(self, expert_id: int) -> int:
        """0 = hot (source packing), 1 = cold (tier packing)."""
        return 0 if int(expert_id) in (self._hot_experts or ()) else 1

    def _slice_dtypes_lazy(self):
        if not hasattr(self, "_slice_dtypes"):
            td = getattr(self.backing, "tensor_dtype", None)
            self._slice_dtypes = (
                td(self.stacked_scales_key) if td else None,
                td(self.stacked_biases_key) if td and self.stacked_biases_key else None,
            )
        return self._slice_dtypes

    def _promote_np(self, v, dtype_str: str | None = None):
        """Promote a cached/staged np.ndarray to mx.array on this thread."""
        return promote_np_array(v, dtype_str)
    def _slice_bytes(self, key: str) -> int:
        """Per-expert byte size of *key* (truthful: read from the backing reader).

        Tier-blind (cold-first); only used for sizing estimates — actual
        reads resolve per expert.
        """
        try:
            reader = self.backing._reader_for_key(key)
            return int(reader._rp_for(key).expert_bytes)
        except Exception:
            return 0

    @property
    def _stacked_keys(self) -> list:
        """[weight, scales] + biases when the projection declares them."""
        keys = [self.stacked_weight_key, self.stacked_scales_key]
        if self.stacked_biases_key:
            keys.append(self.stacked_biases_key)
        return keys

    def _per_slot_bytes(self) -> int:
        """Summed per-SLOT bytes across this projection's stacked tensors.

        One slot = one expert's slice of THIS projection's keys
        (weight+scales+biases) — the same unit the cache's per_slot_bytes
        budget counts. Renamed from ``_per_expert_bytes``: a whole expert
        spans every projection and is n_proj times this.

        Memoized: the union/prefetch paths re-resolve it per call (~150x
        per decode token), and the per-key reader resolution is constant
        for a fixed backing. A 0 from a failed probe is NOT cached so a
        transient backing error still self-heals.
        """
        psb = self._per_slot_memo
        if psb is None:
            psb = sum(self._slice_bytes(k) for k in self._stacked_keys)
            if psb:
                self._per_slot_memo = psb
        return psb

    def _tier_groups(self, ids: list[int]) -> dict[int, list[int]]:
        """Group ids by tier with ONE pass (split-aware segmentation).

        Shared by _tier_bank_bytes_for and the read path so reader
        resolution happens once per (key, tier-run) instead of once per
        expert. Without the split everything is tier 0.
        """
        groups: Dict[int, list[int]] = {}
        if not ids:
            return groups
        try:
            if not self._is_split_active():
                groups[0] = list(ids)
                return groups
            for eid in ids:
                groups.setdefault(self._tier_of(int(eid)), []).append(int(eid))
            return groups
        except Exception:
            return {0: list(ids)}

    def _tier_bank_bytes_for(self, ids: list[int]) -> int:
        """True raw-bank bytes for an id set under the HOBBIT split.

        Sums per tier group with the TIER's own reader width (hot = source
        packing, cold = tier packing). Without the split it reduces to
        _bank_bytes_for. Never raises: a resolution failure falls back to
        the cold-first estimate so the caps stay at least as strict for
        unknown layouts. One reader resolution per (key, tier) via
        _tier_groups.
        """
        if not ids:
            return 0
        try:
            if not self._is_split_active():
                return len(ids) * self._per_slot_bytes()
            keys = self._stacked_keys
            total = 0
            for _t, g in self._tier_groups(ids).items():
                per_t = 0
                for key in keys:
                    reader = self.backing._reader_for_key(key, g[0])
                    per_t += int(reader._rp_for(key).expert_bytes)
                total += len(g) * per_t
            return total
        except Exception:
            return len(ids) * self._per_slot_bytes()


    def _read_expert_banks(self, expert_ids: list[int]):
        """Read a contiguous demand bank per (key, tier).

        Returns (segments, rows): segments is a list of (tier_ids, banks)
        with banks[i] a raw (n_tier, per_bytes) uint8 buffer per stacked
        key (weight, scales, bias); rows are the per-expert typed views in
        expert_ids order that the LRU caches. None when the backing cannot
        serve the demand set as banks (dict backing, unsupported layout,
        oversized demand set, or a tier-mixed component the backing rejects).
        """
        if not hasattr(self.backing, "read_expert_into") or not expert_ids:
            return None
        keys = self._stacked_keys
        try:
            # One pass, split-aware (_tier_groups): {0: all ids} when the
            # HOBBIT split is off.
            groups = sorted(self._tier_groups(expert_ids).items())
            segments: list[tuple[list[int], list]] = []
            rows: list[tuple] = []
            total = 0
            for _t, ids_t in groups:
                per_bytes = []
                # Resolve each key's reader ONCE per tier-run (all ids
                # in ids_t share the tier by construction) and reuse it for
                # the slice views below.
                _readers = []
                for key in keys:
                    reader = self.backing._reader_for_key(key, ids_t[0])
                    _readers.append(reader)
                    per_bytes.append(reader._rp_for(key).expert_bytes)
                if any(size <= 0 for size in per_bytes):
                    return None
                total += len(ids_t) * sum(per_bytes)
                # Raw uint8 demand banks cannot be pooled: the LRU retains
                # rows as views into them, so recycling a bank would corrupt
                # cached experts (aliasing). One np.empty per (key, tier)
                # per layer call is small next to the preadv payload.
                banks = [
                    np.empty((len(ids_t), size), dtype=np.uint8) for size in per_bytes
                ]
                if len(ids_t) * sum(per_bytes) > _BANK_MAX_BYTES:
                    # Read the bank in row windows bounded by the
                    # cap. Rows are views into `banks` either way, so peak
                    # residency is identical to the single-read shape; only
                    # the preadv side is chunked, which also bounds the temp
                    # run buffers inside read_expert_into.
                    rows_per_chunk = max(1, _BANK_MAX_BYTES // sum(per_bytes))
                    for lo in range(0, len(ids_t), rows_per_chunk):
                        hi = min(len(ids_t), lo + rows_per_chunk)
                        components = [(key, ids_t[lo:hi]) for key in keys]
                        views = [b[lo:hi] for b in banks]
                        if not self.backing.read_expert_into(
                            components,
                            views,
                            merge_gap=0,
                            max_run_bytes=_BANK_RUN_MAX_BYTES,
                        ):
                            return None
                else:
                    components = [(key, ids_t) for key in keys]
                    if not self.backing.read_expert_into(
                        components,
                        banks,
                        merge_gap=0,
                        max_run_bytes=_BANK_RUN_MAX_BYTES,
                    ):
                        return None
                segments.append((ids_t, banks))
                # Reinterpret rows with the already-resolved per-tier
                # readers (same packing for the whole run by contract) —
                # _slice_view would re-resolve per expert.
                _rps = [r._rp_for(k) for r, k in zip(_readers, keys)]
                for i in range(len(ids_t)):
                    w = np.frombuffer(banks[0][i], dtype=_rps[0].np_dtype).reshape(_rps[0].per_shape)
                    s = np.frombuffer(banks[1][i], dtype=_rps[1].np_dtype).reshape(_rps[1].per_shape)
                    b = (
                        np.frombuffer(banks[2][i], dtype=_rps[2].np_dtype).reshape(_rps[2].per_shape)
                        if self.stacked_biases_key
                        else None
                    )
                    rows.append((w, s, b))
            if len(groups) > 1:
                # rows arrive tier-grouped; restore expert_ids order so the
                # caller can zip missing -> rows directly.
                flat = [e for _t, ids_t in groups for e in ids_t]
                by_id = {int(e): r for e, r in zip(flat, rows)}
                rows = [by_id[int(e)] for e in expert_ids]
            return segments, rows
        except Exception:
            # Bank-read failures fall back to per-expert loads — count
            # them (with the layer) instead of failing silently, so a
            # rotting backing shows up in the per-request summary.
            try:
                self.cache._count_ctx_fallback(f"bank_read_l{self.layer_idx}")
            except Exception:
                pass
            return None


    def _load_expert_bank_np(self, expert_ids: list[int]) -> list[tuple] | None:
        """Read a demand set into one raw NumPy bank per (key, tier).

        The backing performs coalesced contiguous reads into caller-owned
        banks; rows are then exposed as views for the existing LRU
        representation. Returning None preserves the per-expert fallback
        for dict backings and unsupported layouts.
        """
        got = self._read_expert_banks(expert_ids)
        return None if got is None else got[1]

    def _load_expert_bank_np_full(self, expert_ids: list[int]):
        """Like _load_expert_bank_np, but keeps the raw contiguous banks.

        Needed by the layer context: the NumPy read may happen on an
        IO pool worker, yet promoting those buffers to MLX must happen later
        on the inference thread (MLX ops may not be bound off-stream). Same
        failure contract as _load_expert_bank_np — None whenever the backing
        cannot serve the demand set as banks.
        """
        return self._read_expert_banks(expert_ids)

    def _promote_banks(self, segments: list) -> list | None:
        """Promote raw contiguous per-tier banks into one mx.array per key.

        Shared by _load_expert_bank_mx (read + promote together) and the
        layer context (read on a pool thread, promote here on the
        inference thread).

        Bit-identical to promoting U per-expert arrays and stacking them:
        each bank is reinterpreted with exactly the dtype and per-expert
        shape that _slice_view applies to a single row, so gather_qmm
        receives the same bytes, dtype and layout. Only the allocation count
        differs — one mx.array per key instead of U of them plus the stack,
        which halves the Metal transient at the promotion point (the U
        copies and the bank would briefly coexist).

        Returns a list aligned with segments: one (w_bank, s_bank, b_bank)
        triple per (tier_ids, banks) segment.
        """
        try:
            dt = self._slice_dtypes_lazy()
            promoted = []
            for ids_t, banks in segments:
                n = len(ids_t)
                one: list = []
                for i, key in enumerate(self._stacked_keys):
                    reader = self.backing._reader_for_key(key, ids_t[0])
                    rp = reader._rp_for(key)
                    typed = np.frombuffer(banks[i], dtype=rp.np_dtype).reshape(
                        n, *rp.per_shape
                    )
                    arr = promote_np_array(typed, (None, dt[0], dt[1])[i])
                    # promote_np_array handles the mx.array fast path, but
                    # here the input is always numpy; fall back to a plain
                    # mx.array copy when the registry passed through
                    # (non-BF16 stored dtypes).
                    if not isinstance(arr, mx.array):
                        arr = mx.array(typed)
                    one.append(arr)
                one += [None] * (3 - len(one))
                promoted.append(tuple(one))
            return promoted
        except Exception:
            # Same noisy-fallback contract as _read_expert_banks.
            try:
                self.cache._count_ctx_fallback("bank_promote_fail")
            except Exception:
                pass
            return None

    def _load_expert_bank_mx(self, expert_ids: list[int]):
        """Promote an all-miss demand bank in one shot (per tier).

        Returns (segments_promoted, rows) or None when the demand set cannot
        be served as banks. segments_promoted is a list of
        (tier_ids, (w_bank, s_bank, b_bank)); rows are the per-expert raw
        views the caller still has to seed into the LRU, so the hit-rate
        path is unaffected.

        Bit-identical to promoting U per-expert arrays and stacking them:
        each bank is reinterpreted with exactly the dtype and per-expert
        shape that _slice_view uses per row, so gather_qmm receives the same
        bytes in the same layout. Only the allocation count differs.
        """
        got = self._read_expert_banks(expert_ids)
        if got is None:
            return None
        segments, rows = got
        promoted = self._promote_banks(segments)
        if promoted is None:
            return None
        return [(ids_t, triple) for (ids_t, _banks), triple in zip(segments, promoted)], rows

    def _group_runs(
        self, sorted_ids: list[int], max_run: int | None = None
    ) -> list[tuple[int, int]]:
        """Split ascending expert ids into bounded contiguous runs.

        Under the HOBBIT split a run must NOT cross a tier boundary: the
        coalesced pread reads from ONE backing reader (resolved by the first
        id — source shard vs expert_cold/), so experts past the boundary
        would come back in the wrong packing. Runs therefore end at the
        first id whose tier differs from the run's first id. The bound comes
        from _RUN_MAX (env OMLX_EXPERT_STREAMING_RUN_MAX).

        Gap bridging is deliberately NOT wired: a run *could* be stretched to
        bridge a small hole of missing ids within the same tier (the caller
        drops rows outside its scatter set, so the extra bytes are harmless
        semantically), but reads lose more to the idle bytes than they gain
        from sequentiality. The capability stays in
        shard_bank.segment_runs(merge_gap=...); the runtime never opts in.
        """
        tier_of = self._tier_of if self._is_split_active() else None
        max_run = _RUN_MAX if max_run is None else max(1, int(max_run))
        merge_gap = 0
        return segment_runs(
            sorted_ids,
            same=(lambda a, b: tier_of(a) == tier_of(b)) if tier_of is not None else None,
            merge_gap=merge_gap,
            max_run=max_run,
        )


    def bundle_key(self, expert_id: int):
        # Tier-suffixed under the HOBBIT split so a hot (source-packing)
        # bundle and a cold (tier-packing) bundle of the same expert can
        # coexist in the LRU without aliasing.
        tier = self._tier_of(expert_id) if self._is_split_active() else 0
        base = self.stacked_weight_key if tier == 0 else self.stacked_weight_key + "#c"
        return (self.layer_idx, expert_id, base)

    def _admit(self, key: tuple[int, int, str], row: Any) -> None:
        """Demand admission: enqueue for the detached worker copy+put."""
        adm = getattr(self.cache, "admission", None)
        if adm is not None:
            adm.submit(key, row)

    def _read_slice_triple(self, expert_id: int) -> tuple:
        """One ``(w, s, b)`` bundle via the backing's slice API.

        Propagates weight/scales read errors to the caller (each caller's
        exception contract differs); a missing/ailing biases slice
        degrades to None.
        """
        w = self.backing.load_expert_slice(self.stacked_weight_key, expert_id)
        s = self.backing.load_expert_slice(self.stacked_scales_key, expert_id)
        b = None
        if self.stacked_biases_key:
            try:
                b = self.backing.load_expert_slice(self.stacked_biases_key, expert_id)
            except Exception:
                b = None
        return (w, s, b)

    def _load_expert_np(self, expert_id: int) -> tuple | None:
        """Numpy-only load for the prefetch worker.

        Never touches the LRU and never allocates MLX arrays (worker threads
        must not bind MLX ops to a non-existent default stream). Returns None
        when the backing has no slice-level API or the read fails.
        """
        if not hasattr(self.backing, "load_expert_slice"):
            return None
        # Tier contract: the backing's hot set (same ids as this linear's
        # _hot_experts) routes hot ids to the source shards; everyone else
        # reads expert_cold/. The LRU key (bundle_key) keeps the two apart.
        try:
            return self._read_slice_triple(expert_id)
        except Exception as exc:
            # Repeated per-expert read failures warn once per key
            # (a rotting shard otherwise degrades silently to fallbacks).
            try:
                # Locked RMW on the cache (this runs on IO workers, so a
                # bare dict get/set would lose concurrent increments).
                key = self.bundle_key(int(expert_id))
                n = int(self.cache.note_read_failure(key))
                if n == 3:
                    logger.warning(
                        "expert_streaming: repeated read failure layer=%d expert=%d (%d fails): %s",
                        self.layer_idx, int(expert_id), n, str(exc)[:160],
                    )
            except Exception:
                pass
            return None

    def _load_expert_run_np(self, first_id: int, count: int) -> list[tuple] | None:
        """Numpy-only load of *count* consecutive experts in one pread per key.

        Returns None when the run read is unsupported/fails (caller falls
        back to per-expert loads). Runs exploit row-major contiguity: one
        sequential transfer instead of *count* scattered ones.
        """
        if not hasattr(self.backing, "load_expert_run"):
            return None
        try:
            ws = self.backing.load_expert_run(self.stacked_weight_key, first_id, count)
            ss = self.backing.load_expert_run(self.stacked_scales_key, first_id, count)
            bs: list | None = None
            if self.stacked_biases_key:
                try:
                    bs = self.backing.load_expert_run(self.stacked_biases_key, first_id, count)
                except Exception:
                    bs = None
            return [
                (w, s, bs[i] if bs is not None and i < len(bs) else None)
                for i, (w, s) in enumerate(zip(ws, ss))
            ]
        except Exception:
            return None

    def _bundle_cached_or_staged(self, expert_id: int):
        """Resolve a bundle without touching the disk (inference thread only).

        Returns the cached bundle (mx or raw np tuple) or None when the expert
        must be fetched from the backing store.
        """
        cached = self.cache.get(self.bundle_key(expert_id))
        # Only the bundle format (w, s, b) is ever written under the weight
        # key — every current writer (warmer seed, ctx writeback, bank
        # promote, the bundle loader itself) stores 3-tuples, and a raw
        # mx.array only appears under the non-quantized streamer's own key
        # namespace, which a quantized reader never resolves.
        if isinstance(cached, tuple) and len(cached) == 3:
            return cached  # type: ignore[return-value]
        # Staged prefetch rows (same 3-tuple contract, NumPy only).
        spec = self._spec_state()
        if spec is not None:
            try:
                row = spec.stage_resolve(self.bundle_key(expert_id))
            except Exception:
                row = None
            if isinstance(row, tuple) and len(row) == 3:
                try:
                    self.cache.put(self.bundle_key(expert_id), row)
                except Exception:
                    pass
                return row  # type: ignore[return-value]
        return None

    def _spec_state(self) -> SpeculationState | None:
        """Per-conversion speculation state.

        The converter hangs one instance on the cache and (when the
        backing is an object) on the backing; it dies with them, so two
        engines can never share ring bytes or routing history.
        """
        return _spec_state_of(self.backing, self.cache)

    def _advise_next_layer_prev_token(self, plan: _RemapPlan) -> None:
        """Speculate the NEXT layer's previous-token experts.

        spec_state.prev_uniq_by_layer holds layer N+1's expert ids; the
        advisory must therefore hit layer N+1's banks. The converted-
        linears registry resolves the next layer's real stacking keys (and
        its HOBBIT tier routing — backing.advise_expert_run segments runs
        per resolved reader, so hot/cold boundaries are respected
        automatically).

        Guards: the advisory is capped at _MAX_ADVISE_ROWS experts
        (prefill-shaped sets are skipped: they are dense demand, not decode
        speculation, and advising them would flood the device queue with
        speculative traffic), and each (target, run) fires at most once per
        layer call through plan.advised_runs.
        """
        state = self._spec_state()
        if state is None or state.is_closed():
            return
        # Per-conversion readahead contract wins when conversion stamped
        # it (expert_streaming_readahead); None keeps the env default.
        _ra = getattr(state, "readahead_enabled", None)
        if _ra is None:
            _ra = _RA_ENV
        if not _ra:
            return
        next_layer = self.layer_idx + 1
        targets = state.linears_by_layer.get(next_layer)
        if not targets:
            return
        advised_runs = plan.advised_runs
        # Prev-token routing is the advisory prediction (advisory ids,
        # never output).
        prev = state.prev_uniq_by_layer.get(next_layer)
        if not prev or len(prev) > _MAX_ADVISE_ROWS:
            return
        # When staging already covers this layer (recall gate open), the
        # staged READ warms the same pages for real — the hint is redundant.
        try:
            if state.stage_gate(next_layer):
                return
        except Exception:
            logger.debug(
                "expert_streaming: stage_gate probe failed for layer %d",
                next_layer,
                exc_info=True,
            )
        try:
            sorted_prev = sorted(int(e) for e in prev)
            if not sorted_prev:
                return
            # TTL dedup — the same prev-set was just hinted last token;
            # F_RDADVISE is sticky, re-issuing it is pure overhead.
            sorted_prev = state.advise_fresh(next_layer, sorted_prev)
            if not sorted_prev:
                return
            # k+1 overfetch — union the transition-table candidates
            # for the next layer's prev set into the advisory. Same caps
            # and dedup as the base set; hints only, never output.
            # Gated by measured precision of the last extras.
            try:
                _extra = (
                    state.predict_next(next_layer, sorted_prev)
                    if state.trans_overfetch_ok()
                    else []
                )
            except Exception:
                _extra = []
            if _extra:
                _have = set(sorted_prev)
                _added = []
                for _c in _extra:
                    if _c not in _have and len(sorted_prev) < _MAX_ADVISE_ROWS:
                        sorted_prev.append(int(_c))
                        _have.add(int(_c))
                        _added.append(int(_c))
                sorted_prev.sort()
                state.note_trans_extras(next_layer, _added)
                try:
                    state.bump("trans_overfetch", len(_added))
                except Exception:
                    pass
            # One shared segmentation with the demand path —
            # consecutive ids within one resolved reader become one run for
            # a single F_RDADVISE (tier boundaries break the run). Without
            # the hot/cold split every id resolves to the same reader, so
            # the runs are plain consecutive spans.
            if self._is_split_active():
                rid_of = {
                    e: id(self.backing._reader_for_key(self.stacked_weight_key, e))
                    for e in sorted_prev
                }
                runs = segment_runs(
                    sorted_prev, same=lambda a, b: rid_of[a] == rid_of[b]
                )
            else:
                runs = segment_runs(sorted_prev)
            n_adv = 0
            for target in targets:
                # Every bank the demand path reads needs the hint, not
                # just the weight — a hit on weight but a miss on scales
                # still stalls the projection. advise_expert_run
                # re-segments each key's run at its own tier boundaries.
                tkeys = [target.stacked_weight_key]
                skey = getattr(target, "stacked_scales_key", None)
                if skey:
                    tkeys.append(skey)
                bkey = getattr(target, "stacked_biases_key", None)
                if bkey:
                    tkeys.append(bkey)
                for tkey in tkeys:
                    for first, count in runs:
                        dedupe_key = (id(target), tkey, first, count)
                        if dedupe_key in advised_runs:
                            continue
                        advised_runs.add(dedupe_key)
                        try:
                            ok, _adv_bytes, _adv_segs = (
                                target.backing.advise_expert_run(
                                    tkey, first, count
                                )
                            )
                            # "advised" counts expert rows hinted, once —
                            # scales/biases rides don't re-count the row.
                            if ok and tkey == target.stacked_weight_key:
                                n_adv += count
                        except Exception:
                            logger.debug(
                                "expert_streaming: advisory run failed for "
                                "layer %d",
                                next_layer,
                                exc_info=True,
                            )
            if n_adv:
                state.bump("advised", n_adv)
        except Exception:
            logger.debug(
                "expert_streaming: advisory skipped for layer %d",
                next_layer,
                exc_info=True,
            )

    def _stage_next_layer(self, plan: _RemapPlan | None = None) -> None:
        """Read the next layer's predicted demand into staging.

        Decode-shaped calls only: predicts layer L+1's experts as
        prev_uniq_by_layer[L+1] (last token's routing) + transition-table
        candidates, capped at _STAGED_MAX_IDS, and submits ONE coalesced
        bank read for this projection's counterpart linear. Rows land in
        spec_state.staged as NumPy — no MLX off-stream — and are consumed
        by _split/_bundle_cached_or_staged when layer L+1 resolves.

        Gated by stage_gate (per-layer EWMA recall of the prev-token
        prediction): layers whose routing diverges stop being prefetched
        after the evidence accumulates, so a bad predictor costs a bounded
        burst, not a permanent I/O tax.
        """
        state = self._spec_state()
        if state is None or not state.stage_enabled():
            return
        # Decode-shaped calls only, and the call shape is the real token
        # count: a multi-token verify/prefill with few routed rows must
        # not stage (positions is routed rows, seq_len is tokens).
        if plan is None or not _decode_call_shape(
            int(getattr(plan, "positions", 0)),
            getattr(plan, "seq_len", None),
        ):
            return
        # Headroom gate: speculative reads only pay when residency has
        # slack — floored capacity or free memory in the governor's
        # desperate band means mispredicts compete with demand fetches for
        # slots that cannot hold them.
        if _STAGED_HEADROOM_ENV and not self._stage_headroom():
            return
        next_layer = self.layer_idx + 1
        if not state.stage_gate(next_layer) or not state.stage_room():
            return
        try:
            targets = state.linears_by_layer.get(next_layer)
            if not targets:
                return
            proj = getattr(self, "proj_name", None)
            target = next(
                (l for l in targets if getattr(l, "proj_name", None) == proj),
                None,
            )
            if target is None:
                return
            # Prev-token routing is the advisory prediction.
            prev = state.prev_uniq_by_layer.get(next_layer)
            if not prev:
                return
            ids = list(dict.fromkeys(int(e) for e in prev))
            try:
                extra = state.predict_next(next_layer, ids)
            except Exception:
                extra = []
            for e in extra:
                if len(ids) >= _STAGED_MAX_IDS:
                    break
                if e not in ids:
                    ids.append(int(e))
            ids = ids[:_STAGED_MAX_IDS]
            # If the target layer can't hold even one token's working
            # set, every staged row is a guaranteed drop — the global
            # headroom gate misses this because governor pressure shrinks
            # per-layer caps before the global floor binds.
            try:
                _cap = self.cache._cap_for(next_layer)
            except Exception:
                _cap = 0
            if _cap and _cap < min(len(prev), _STAGED_MAX_IDS):
                return
            # Drop what the cache already serves; staging duplicates read.
            # Membership probe only — get() would mint hits/misses and
            # promote recency for rows no demand ever touched. peek_many
            # takes the cache lock once for the whole set.
            _keys = [target.bundle_key(e) for e in ids]
            want = [
                e for e, v in zip(ids, self.cache.peek_many(_keys))
                if v is None
            ]
            if not want:
                return
            want.sort()
            reader = target._load_expert_bank_np_full
            pool = _io_pool(target)
            # Re-staging replaces last token's staged set for this layer:
            # drop stale rows/futures keyed under next_layer BEFORE
            # registering.
            # stage_register skips ids already staged, so leaving stale
            # entries would both keep a wrong prediction alive and split
            # the new read around it. Keys are exact bundle keys
            # (layer, expert, kind); shared futures cancel only when no
            # sibling staged key still references them. The key snapshot
            # is registry-locked — iterating the dicts directly would
            # race a concurrent resolve's fan-out writes.
            for stale in state.staged_keys():
                if stale[0] == next_layer:
                    state.stage_drop(stale)
            fut = pool.submit(reader, want)
            state.stage_register(fut, want, target)
        except Exception:
            logger.debug(
                "expert_streaming: staging skipped for layer %d",
                getattr(self, "layer_idx", -1) + 1,
                exc_info=True,
            )

    def _stage_headroom(self) -> bool:
        """True when speculative staged reads have residency slack.

        Suppress staging when the shared cache capacity is at the
        governor floor or the last observed free memory was inside the
        desperate-clear band — under memory pressure staged rows get
        dropped before they are ever hit. Static caches (no governor)
        always have slack. Shared predicate: staging.stage_headroom.
        """
        return stage_headroom(getattr(self.cache, "governor", None))

    def _load_expert_bundle(self, expert_id: int) -> tuple[mx.array, mx.array, mx.array | None]:
        key = self.bundle_key(expert_id)
        # Cache / staging resolution (shared with the parallel demand-set path)
        resolved = self._bundle_cached_or_staged(expert_id)
        if resolved is not None:
            return resolved  # type: ignore[return-value]
        # 3) synchronous load from backing
        if hasattr(self.backing, "load_expert_slice"):
            # Async-friendly: store plain np.ndarray slices in the cache and
            # promote them to mx.array on the inference thread at use time
            # (avoids cross-thread stream errors from MLX op allocation —
            # the prefetch worker must never allocate MLX arrays).
            w, s, b = self._read_slice_triple(expert_id)
        elif hasattr(self.backing, "load_expert"):
            w = self.backing.load_expert(self.stacked_weight_key, expert_id)
            s = self.backing.load_expert(self.stacked_scales_key, expert_id)
            b = None
            if self.stacked_biases_key:
                try:
                    b = self.backing.load_expert(self.stacked_biases_key, expert_id)
                except Exception:
                    b = None
        else:
            # dict backing for tests
            def _mx(v):
                return v if isinstance(v, mx.array) else mx.array(v)

            w_bank = self.backing[(self.layer_idx, self.proj_name, "weight")]
            s_bank = self.backing[(self.layer_idx, self.proj_name, "scales")]
            b_bank = self.backing.get((self.layer_idx, self.proj_name, "biases"))
            w = _mx(w_bank[expert_id])
            s = _mx(s_bank[expert_id])
            b = _mx(b_bank[expert_id]) if b_bank is not None else None
        bundle = (w, s, b)
        self.cache.put(key, bundle)  # type: ignore[arg-type]
        return bundle

    def __call__(self, x, indices, sorted_indices=False, plan: _RemapPlan | None = None):
        if plan is None:
            plan = _RemapPlan()
        # Speculation for layer+1, deduped per layer call
        # through plan.advised_runs (the GLU shares one plan).
        _spec_state = self._spec_state()
        # The advisor itself applies the readahead gate: spec_state's
        # per-conversion readahead_enabled wins over the _RA_ENV default
        # when conversion stamped it, so the env cannot veto a per-model
        # enable (nor enable a per-model disable).
        if _spec_state is not None and _spec_state.prev_uniq_by_layer:
            try:
                self._advise_next_layer_prev_token(plan)
            except Exception:
                logger.debug(
                    "expert_streaming: advise failed for layer %d",
                    getattr(self, "layer_idx", -1),
                    exc_info=True,
                )
        # Staged decode prefetch — real reads into staging buffers
        # for the next layer's predicted demand (recall-gated; see
        # _stage_next_layer). Independent of the F_RDADVISE advisor.
        if _spec_state is not None:
            try:
                self._stage_next_layer(plan)
            except Exception:
                logger.debug(
                    "expert_streaming: stage_next failed for layer %d",
                    getattr(self, "layer_idx", -1),
                    exc_info=True,
                )
        if plan.flat_np is None:
            _build_plan_into(plan, indices)
        # While the hotness seeder is active a prefill demand set is not
        # cached — seeding the LRU with prefill-only experts would evict the
        # decode working set.
        cache_result = not (
            getattr(self.cache, "prefill_bypass", False)
            and plan.positions > _PREFILL_SHAPE_MIN_ROWS
        )
        if getattr(plan, "arena_rhs", None) is not None:
            return self._arena_call(x, plan, sorted_indices)
        res = self._resolve_demand(plan, cache_result)
        out = self._assemble_banks(x, plan, res, sorted_indices)
        return self._finish(plan, out)

    def _arena_call(self, x, plan, sorted_indices):
        """Arena path: residents were bound into fixed slot rows at
        admission, so gather_qmm reads the persistent bank by slot index —
        a hit costs zero assembly (the bundle path re-stacks every call)."""
        bank = self._arena_bank
        out = mx.gather_qmm(
            x,
            bank.weight,
            bank.scales,
            bank.biases,
            rhs_indices=plan.arena_rhs,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        return self._finish(plan, out)

    def _resolve_demand(self, plan, cache_result) -> _ResolvedDemand:
        """Cover ``plan.uniq_list`` with raw ``(w, s, b)`` bundles.

        Order: the layer context (which prefetches projections in the
        background) -> the expert LRU -> one contiguous bank read for a
        full miss -> the coalesced/per-expert fallback.
        """
        bundles: Dict[int, tuple] = {}
        missing: list[int] = []
        served = False
        if plan.ctx is not None:
            # Resolve *this* projection through the layer context; the
            # context prefetches the next one in the background so banks
            # are not all resident at once.
            plan.ctx.ensure(self, plan.uniq_list)
            # Count every fallback to the per-expert resolution so runs
            # can prove the fast path engaged. The ctx records WHICH
            # reason when the read came back unusable.
            if plan.ctx.failed:
                self.cache._count_ctx_fallback(
                    getattr(plan.ctx, "fallback_reason", None) or "read_failure"
                )
            elif getattr(plan.ctx, "declined", False):
                self.cache._count_ctx_fallback("bank_too_large")
            else:
                context_bundles = plan.ctx.bundles.get(id(self))
                if context_bundles is not None and len(context_bundles) == len(plan.uniq_list):
                    bundles.update(context_bundles)
                    served = True
                else:
                    self.cache._count_ctx_fallback("tier_mismatch")
        if not served:
            for eid in plan.uniq_list:
                eid = int(eid)
                b = self._bundle_cached_or_staged(eid)
                if b is not None:
                    bundles[eid] = b
                else:
                    missing.append(eid)
        banked = None
        if missing:
            # ascending expert id = ascending file offset within the stacked
            # bank (row-major) — sorted reads keep the NVMe's locality
            missing.sort()
            if (
                len(missing) == len(plan.uniq_list)
                and hasattr(self.backing, "read_expert_into")
            ):
                # Every demanded expert is a miss, so the demand set
                # is one contiguous bank per key (two segments under the
                # HOBBIT split) — promote each once instead of building U
                # per-expert mx arrays and stacking them.
                banked = self._load_expert_bank_mx(missing)
            if banked is not None:
                rows = banked[1]
                for eid, raw in zip(missing, rows):
                    bundles[eid] = raw
                    if cache_result:
                        self._admit(self.bundle_key(eid), raw)
            elif hasattr(self.backing, "load_expert_slice"):
                io_pool = _io_pool(self)
                # Bank-first path: read all missing experts
                # into one raw bank per (key, tier) on this thread, then
                # expose rows as views. Avoids one task/result allocation per
                # expert on dense demand sets and one reader resolution per
                # expert on every set.
                raws = self._load_expert_bank_np(missing)
                if raws is None:
                    coalesce_on = (
                        _COALESCE_ENV
                        if self._coalesce_override is None
                        else bool(self._coalesce_override)
                    )
                    raws = [None] * len(missing)
                    # Fallback: coalesce consecutive ids into
                    # single-pread runs (dense in long-prompt prefill; rare in
                    # decode, where runs are size 1 and the path degenerates
                    # to the per-expert fetch). map keeps a sliding window of
                    # 16 in flight (singleton pool), so the device queue stays
                    # full; batch drain/sawtooth is avoided without moving
                    # promotion off the inference thread.
                    runs = self._group_runs(missing)
                    if coalesce_on and len(runs) < len(missing):
                        results_by_run = list(
                            io_pool.map(
                                lambda r: (r, self._load_expert_run_np(r[0], r[1])),
                                runs,
                            )
                        )
                        idx_of = {eid: i for i, eid in enumerate(missing)}
                        leftover: list[int] = []
                        for (first, count), out in results_by_run:
                            if out is not None:
                                for j in range(count):
                                    eid = first + j
                                    if eid in idx_of:
                                        raws[idx_of[eid]] = out[j]
                                    # else: bridged gap row — read but
                                    # never promoted/used, so dropped here
                            else:
                                leftover.extend(
                                    e for e in range(first, first + count) if e in idx_of
                                )
                        if leftover:
                            for eid, raw in zip(
                                leftover, io_pool.map(self._load_expert_np, leftover)
                            ):
                                raws[idx_of[eid]] = raw
                    else:
                        raws = list(io_pool.map(self._load_expert_np, missing))
                for eid, raw in zip(missing, raws):
                    if raw is None:
                        bundles[eid] = self._load_expert_bundle(eid)
                        continue
                    # Raw np bundles in the LRU by design: Metal only holds
                    # the per-call stack. Caching promoted mx copies here
                    # double-holds the same weights in wired memory when the
                    # budget is positive, so wired residency would overshoot
                    # the budget outright.
                    bundles[eid] = raw
                    if cache_result:
                        # Tier-suffixed key (bundle_key): under the HOBBIT
                        # split a hot (source-packing) and cold
                        # (tier-packing) bundle of the same expert must
                        # never alias in the LRU — a mixed-width bundle
                        # served to the wrong tier crashes mx.stack.
                        self._admit(self.bundle_key(eid), raw)
            else:
                # dict-backed test doubles: sequential fallback
                for eid in missing:
                    bundles[eid] = self._load_expert_bundle(eid)
        return _ResolvedDemand(bundles=bundles, banked=banked)

    def _assemble_banks(self, x, plan, res: _ResolvedDemand, sorted_indices):
        """Promote the resolved rows into the bank(s) gather_qmm consumes."""
        dt = self._slice_dtypes_lazy()
        ctx_banks = None
        if res.banked is None and plan.ctx is not None:
            # The layer context read this projection's demand set
            # as one contiguous NumPy bank per key (possibly on an IO pool
            # worker). Promote it *here*, on the inference thread, so MLX
            # ops stay on-stream and the U per-expert mx arrays plus the
            # stack copy are both skipped. Guarded by bank_ids: only a bank
            # that describes exactly this demand set may be promoted, so a
            # stale or partial bank cannot silently mis-pair experts.
            segs = plan.ctx.bank_raw.get(id(self))
            if (
                segs is not None
                and plan.ctx.bank_ids.get(id(self)) == plan.uniq_list
            ):
                promoted = self._promote_banks(segs)
                if promoted is not None:
                    ctx_banks = [
                        (ids_t, triple)
                        for (ids_t, _banks), triple in zip(segs, promoted)
                    ]

        split = self._is_split_active()
        # Single-promoted per-tier banks. A segment whose ids
        # match EXACTLY the tier's demanded ids can feed gather_qmm directly
        # (no per-expert promote, no mx.stack) — bit-identical by
        # construction. Anything else falls back per tier.
        tier_single: dict[int, tuple] = {}
        segs_in = res.banked[0] if res.banked is not None else ctx_banks
        if segs_in:
            for ids_t, triple in segs_in:
                tier = self._tier_of(ids_t[0]) if split else 0
                tier_single[tier] = triple

        if split:
            return self._qmm_dual_tier(x, plan, res.bundles, tier_single, dt, sorted_indices)
        return self._qmm_uniform(x, plan, res.bundles, tier_single, dt, sorted_indices)

    def _stack_bundles(self, rows: list, dt, label: str) -> tuple:
        """Promote + stack raw ``(w, s, b)`` bundles into a bank triple.

        A single bundle takes expand_dims (mx.stack on a singleton is
        identical). Partial-bias coverage must fail loudly — silently
        stacking fewer biases than weights would mis-pair rows or drop a
        declared bias.
        """
        mini_w, mini_s, mini_b = [], [], []
        for w, s, b in rows:
            mini_w.append(self._promote_np(w))
            mini_s.append(self._promote_np(s, dt[0]))
            if b is not None:
                mini_b.append(self._promote_np(b, dt[1]))
        if mini_b and len(mini_b) != len(mini_w):
            raise RuntimeError(
                f"expert_streaming: inconsistent bias coverage in {label} "
                f"({len(mini_b)} biases for {len(mini_w)} weights)"
            )
        if len(mini_w) == 1:
            return (
                mx.expand_dims(mini_w[0], 0),
                mx.expand_dims(mini_s[0], 0),
                mx.expand_dims(mini_b[0], 0) if mini_b else None,
            )
        return (
            mx.stack(mini_w, axis=0),
            mx.stack(mini_s, axis=0),
            mx.stack(mini_b, axis=0) if mini_b else None,
        )

    def _qmm_uniform(self, x, plan, bundles, tier_single, dt, sorted_indices):
        """Uniform tier: one mini-bank and one gather_qmm."""
        if 0 in tier_single:
            w_bank, s_bank, b_bank = tier_single[0]
        else:
            w_bank, s_bank, b_bank = self._stack_bundles(
                [bundles[int(eid)] for eid in plan.uniq_list],
                dt,
                self.proj_name,
            )
        return mx.gather_qmm(
            x,
            w_bank,
            s_bank,
            b_bank,
            rhs_indices=plan.remapped,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )

    def _qmm_dual_tier(self, x, plan, bundles, tier_single, dt, sorted_indices):
        """HOBBIT dual-tier assembly: one mini-bank per tier and a
        masked add — positions are mutually exclusive (each position
        consumes exactly one expert), so the two gather_qmm outputs
        partition the positions and zeros fill the rest."""
        # Per-tier bundle lists under the HOBBIT split: hot (source packing)
        # and cold (tier packing) widths differ (e.g. 8 vs 6 u32 cols per
        # row at gs 64), so a single stacked mini-bank is impossible — build
        # one per tier and combine the two gather_qmm outputs.
        tier_rows = ([], [])  # hot, cold — raw (w, s, b) bundles
        uniq = [int(e) for e in plan.uniq_list]
        hot_idx = [i for i, e in enumerate(uniq) if self._tier_of(e) == 0]
        cold_idx = [i for i, e in enumerate(uniq) if self._tier_of(e) == 1]
        for t, idxs in ((0, hot_idx), (1, cold_idx)):
            if t in tier_single:
                continue
            for i in idxs:
                tier_rows[t].append(bundles[uniq[i]])

        flat_np = np.asarray(plan.flat_np).reshape(-1)
        out = None
        # Hot tier first (the masked add is elementwise-commutative so the
        # order is bit-exact).
        for t, idxs in ((0, hot_idx), (1, cold_idx)):
            if not idxs:
                continue
            if t in tier_single:
                w_b, s_b, b_b = tier_single[t]
            else:
                w_b, s_b, b_b = self._stack_bundles(
                    tier_rows[t], dt, f"{self.proj_name} tier {t}"
                )
            bits_ = self.bits if t == 0 else self._cold_bits
            gs_ = self.group_size if t == 0 else self._cold_gs
            # expert-id -> rank within THIS tier's bank (flat ids here,
            # not compact uniq ranks); -1 where the other tier owns it.
            tier_map = np.full((self.num_experts,), -1, dtype=np.int32)
            for rank, i in enumerate(idxs):
                tier_map[uniq[i]] = rank
            tier_remapped_np = tier_map[flat_np].reshape(plan.indices_shape)
            # gather_qmm takes UNSIGNED row indices — -1 wraps to a huge
            # OOB index (garbage/nan) that the keep mask cannot undo
            # (nan * 0 = nan). Clamp the gather indices to 0 (any valid
            # rank: the row is zeroed by the keep mask below); the -1
            # survives only in keep_np, which is what selects the tier.
            gather_np = np.maximum(tier_remapped_np, 0)
            tier_remapped = mx.array(gather_np)
            tier_out = mx.gather_qmm(
                x,
                w_b,
                s_b,
                b_b,
                rhs_indices=tier_remapped,
                transpose=True,
                group_size=gs_,
                bits=bits_,
                mode=self.mode,
                sorted_indices=sorted_indices,
            )
            # Mask: keep only the positions this tier owns (-1 elsewhere).
            # gather_qmm inserts the indices' shape at dims 2.. so the
            # keep mask is the (index-shaped) validity, expanded over the
            # trailing (x_exp singleton, output) dims: [.., topk, 1, 1].
            keep_np = (tier_remapped_np >= 0).astype(np.float32)
            keep_shape = tuple(plan.indices_shape) + (1,) * (tier_out.ndim - len(plan.indices_shape))
            keep = mx.array(keep_np).reshape(keep_shape)
            tier_out = tier_out * keep
            out = tier_out if out is None else out + tier_out
        # out is None only when uniq_list was empty — unreachable: a call
        # that routed >= 1 position (top_k >= 1) always has a uniq set, and
        # the old "all-hot fallback" here would have stacked an empty bank
        # anyway (mx.stack([])). Fail loud instead of assembling garbage.
        assert out is not None, (
            f"_qmm_dual_tier: empty uniq_list for layer {self.layer_idx} "
            f"{self.proj_name}"
        )
        return out


class StreamingSwitchGLU(nn.Module):
    """Streaming SwitchGLU that delegates to streaming linears."""

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        layer_idx: int,
        backing: Any,
        cache: ExpertLRUCache,
        fused_gate_up: bool = False,
        inverse_scatter: bool = False,
        quantized: bool = False,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        activation: Any | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.inverse_scatter = inverse_scatter
        self.quantized = quantized
        # Original SwitchGLU activation (e.g. DeepSeek V4's LimitedSwiGLU with
        # swiglu_limit / fp32). None falls back to the stock mlx-lm swiglu.
        # Underscore attr keeps it out of the nn.Module parameter tree.
        self._activation = activation

        # Populated by the converter after construction; dims/quant
        # metadata live on the linears, not here.
        self._num_experts = num_experts
        self._backing = backing
        self._cache = cache

    @property
    def activation(self) -> Any:
        """Stock SwitchGLU surface for verify paths.

        The MTP target-verify path (mlx_vlm qwen3_5_moe
        ``_target_verify_switch_glu``) reaches into
        ``switch_mlp.activation`` directly. The captured callable lives
        in ``_activation`` to stay out of the parameter tree, so
        re-expose it here; fall back to the stock SwiGLU convention.
        """
        act = getattr(self, "_activation", None)
        if act is not None:
            return act
        from mlx_lm.models.switch_layers import SwiGLU

        return SwiGLU()

    def _apply_activation(self, x_up: Any, x_gate: Any) -> Any:
        act = getattr(self, "_activation", None)
        if act is not None:
            # Same call order as the original SwitchGLU: activation(up, gate)
            return act(x_up, x_gate)
        from mlx_lm.models.activations import swiglu

        return swiglu(x_gate, x_up)

    @staticmethod
    def _hook_conventions(hook):
        """Resolve this hook's call conventions once per hook instance.

        ``seq_len`` postdates some hook implementations; probing
        ``inspect.signature`` once and caching the verdict ON the hook
        (``_omlx_call_conv``) means per-call dispatch never retries — an
        ``except TypeError`` retry shim would re-invoke the hook with a
        different signature, so a TypeError raised INSIDE a new-signature
        hook body would execute its side effects twice.
        """
        conv = getattr(hook, "_omlx_call_conv", None)
        if conv is None:
            conv = (
                _accepts_kwarg(
                    getattr(hook, "on_layer_start", None), "seq_len"
                ),
                _accepts_kwarg(
                    getattr(hook, "on_layer_plan", None), "seq_len"
                ),
            )
            try:
                hook._omlx_call_conv = conv
            except Exception:
                pass  # unattr-able hook — re-probe per call; still no retry
        return conv

    def _notify_layer_start(self, hook, positions: int, seq_len) -> None:
        start_kw, _plan_kw = self._hook_conventions(hook)
        if start_kw:
            hook.on_layer_start(self.layer_idx, positions, seq_len=seq_len)
        else:
            hook.on_layer_start(self.layer_idx, positions)

    def _notify_layer_plan(self, hook, plan, seq_len) -> None:
        # Hotness signal: per-TOKEN usage over the routing plan
        # (bincount of the flat ids), computed only when a consumer
        # wants it — the readahead warmer keeps the uniq-list contract
        # and pays nothing. flat_np is already on the host (built by
        # _build_plan_into), so the bincount is a cheap vectorized pass.
        counts = (
            np.bincount(
                np.asarray(plan.flat_np).reshape(-1),
                minlength=self._num_experts,
            )
            if getattr(hook, "wants_usage_counts", False)
            else None
        )
        _start_kw, plan_kw = self._hook_conventions(hook)
        if plan_kw:
            hook.on_layer_plan(
                self.layer_idx,
                plan.uniq_list,
                plan.positions,
                counts,
                seq_len=seq_len,
            )
        else:
            hook.on_layer_plan(
                self.layer_idx, plan.uniq_list, plan.positions, counts
            )

    def _engage_arena_and_ctx(self, plan, idx, has_fused, seq_len) -> None:
        # Arena mode: engage before the layer-ctx decision — when the
        # demand set fits the slot bank, gather_qmm reads slot rows and the
        # ctx/bundle machinery is skipped for this call. The _arena_failed
        # latch keeps a permanently failing engage from paying a per-token
        # exception: one warning, then the bundle path forever (the same
        # flag _arena_get sets when the arena cannot be built).
        if _ARENA_ENV and not getattr(self, "_arena_failed", False):
            try:
                self._arena_engage(plan, idx)
            except Exception:
                plan.arena_rhs = None
                self._arena_failed = True
                logger.warning(
                    "expert_streaming: arena engage failed on layer %d — "
                    "arena disabled for the rest of this engine's life",
                    getattr(self, "layer_idx", -1),
                    exc_info=True,
                )
        if self.quantized and _LAYER_BARRIER_ENV:
            projections = (
                [self.gate_up_proj, self.down_proj]
                if has_fused
                else [self.up_proj, self.gate_proj, self.down_proj]
            )
            if getattr(plan, "arena_rhs", None) is not None:
                pass  # arena owns residency this call — no layer ctx
            elif all(hasattr(proj, '_load_expert_bank_np') for proj in projections):
                # Hybrid: decode-shaped calls (<64 routed rows) read all
                # projections at once (union); prefill keeps rolling so
                # all banks are never resident simultaneously. (idx.size
                # == indices.size: the gather-sort is a permutation.)
                positions = int(idx.size)
                plan.ctx = _LayerLoadContext(
                    projections, self._cache,
                    mode=_layer_ctx_mode(positions),
                    positions=positions,
                    seq_len=seq_len,
                )
            else:
                # No bank reader on every projection (dict-backed test
                # doubles / bf16 mixes) — no context to resolve through.
                self._cache._count_ctx_fallback("dict_backing")

    def _run_projections(self, x_exp, idx, do_sort, has_fused, plan):
        if has_fused:
            x_gate_up = self.gate_up_proj(x_exp, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
            x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
            x_act = self._apply_activation(x_up, x_gate)
            return self.down_proj(x_act, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
        x_up = self.up_proj(x_exp, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
        x_gate = self.gate_proj(x_exp, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
        x_act = self._apply_activation(x_up, x_gate)
        return self.down_proj(x_act, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]

    def _record_prev(self, plan) -> None:
        # Remember this layer's routing for next token's speculation —
        # once per layer call. The projections share plan.uniq_list, so a
        # per-projection record would triple-count it.
        _spec_state = _spec_state_of(self._backing, self._cache)
        if _spec_state is not None:
            try:
                _spec_state.record_prev(
                    self.layer_idx, plan.uniq_list, positions=plan.positions
                )
            except Exception:
                logger.debug(
                    "expert_streaming: record_prev skipped for layer %d",
                    getattr(self, "layer_idx", -1),
                    exc_info=True,
                )

    def _call_prologue(self, x, indices):
        """-> (has_fused, hook, seq_len).

        Call shape for hooks/phase: seq_len (indices.shape[-2]) is the
        token count — the authoritative decode-vs-prefill signal (a
        1-token top_k=8 decode is 8 routed rows, not "prefill").
        """
        # Mirror SwitchGLU.__call__ but route through streaming linears
        # Determine fused vs split by presence of gate_up_proj
        has_fused = hasattr(self, "gate_up_proj")
        # Opt-in warm/pin hook (warmer.py): fires previous-token reads for
        # the next layer before this layer's demand loads; decode-only.
        hook = getattr(self, "_warm_pins", None)
        _seq_len = (
            int(indices.shape[-2])
            if getattr(indices, "ndim", 0) >= 2
            else None
        )
        return has_fused, hook, _seq_len

    def _sort_call_inputs(self, x, indices):
        """Expand + gather-sort the call inputs -> (x_exp, idx, do_sort,
        inv_order)."""
        x_exp = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x_exp, idx, inv_order = _gather_sort(x_exp, indices, inverse_scatter=self.inverse_scatter)
        return x_exp, idx, do_sort, inv_order

    def _unsort_output(self, x_out, inv_order, do_sort, indices):
        """Scatter-unsort + squeeze — the reverse of _sort_call_inputs."""
        if do_sort:
            x_out = _scatter_unsort(x_out, inv_order, indices.shape)
        return x_out.squeeze(-2)

    def __call__(self, x, indices, scores=None, weighted_sum: bool = False):
        has_fused, hook, _seq_len = self._call_prologue(x, indices)
        if hook is not None:
            self._notify_layer_start(hook, int(indices.size), _seq_len)
        x_exp, idx, do_sort, inv_order = self._sort_call_inputs(x, indices)

        # One shared routing plan for the whole layer: the first linear
        # invoked builds it (single mx.eval + unique + remap), the rest reuse.
        # seq_len rides on the plan so every per-projection decode-shape
        # decision (staging gate, advisory) sees the token count rather
        # than routed rows.
        plan = _RemapPlan()
        plan.seq_len = _seq_len
        self._engage_arena_and_ctx(plan, idx, has_fused, _seq_len)
        # The projection block is the whole lifetime of the layer context:
        # close() in a finally so a raising projection still flushes the
        # per-layer stall counters (a partial layer is still a stalled layer).
        _ctx = getattr(plan, "ctx", None)
        try:
            x_out = self._run_projections(x_exp, idx, do_sort, has_fused, plan)
        finally:
            if _ctx is not None:
                _ctx.close()

        self._record_prev(plan)
        if hook is not None:
            self._notify_layer_plan(hook, plan, _seq_len)

        # Weighted-sum fast path: the family's fused kernel, injected by the
        # converter from model_hooks (glm_moe_dsa.fast.glm_moe_weighted_sum,
        # which itself falls back to mx.fast). Plain scatter-unsort when no
        # kernel is registered — the caller's ndim contract then applies
        # the scores itself.
        if weighted_sum and scores is not None and do_sort:
            _ws_kernel = getattr(self, "_weighted_sum_kernel", None)
            if _ws_kernel is not None:
                try:
                    return _ws_kernel(x_out, inv_order, scores)
                except Exception:
                    # A real kernel failure must not be invisible — the
                    # unflatten path below is the fallback, not a fix.
                    logger.debug(
                        "expert_streaming: weighted-sum kernel failed for "
                        "layer %d — falling back",
                        getattr(self, "layer_idx", -1),
                        exc_info=True,
                    )

        out = self._unsort_output(x_out, inv_order, do_sort, indices)
        return out

    # ------------------------------------------------------------------
    # Arena mode (OMLX_EXPERT_STREAMING_ARENA)
    # ------------------------------------------------------------------

    def _arena_engage(self, plan: _RemapPlan, idx) -> bool:
        """Bind this call's demand set into persistent slot rows.

        On success plan.arena_rhs holds rhs_indices in slot space and every
        quantized linear of the layer reads its own arena bank — the bundle
        resolve + per-call mx.stack assembly is skipped entirely. Returns
        False (leaving plan.arena_rhs None) whenever the shape cannot be
        served: non-bank backing, hot/cold split active, or a demand set
        larger than the layer cap — those calls keep the bundle path.
        """
        projections = getattr(self, "_arena_projs", None)
        if projections is None:
            cand = (
                [self.gate_up_proj, self.down_proj]
                if hasattr(self, "gate_up_proj")
                else [self.up_proj, self.gate_proj, self.down_proj]
            )
            if not all(
                isinstance(p, StreamingQuantizedSwitchLinear)
                and not p._is_split_active()
                and hasattr(p.backing, "load_expert_slice")
                for p in cand
            ):
                self._arena_projs = False  # permanent shape: latch the decline
                return False
            self._arena_projs = projections = cand
        elif projections is False:
            return False
        if plan.flat_np is None:
            _build_plan_into(plan, idx)
        uniq = plan.uniq_list
        # Phase from the call's real sequence length: plan.seq_len is the
        # authoritative stamp (set pre-sort, from indices.shape[-2]); the
        # 1-D indices_shape of a sorted plan would drop it and a ≤64-row
        # multi-token call would misclassify as decode.
        seq_len = getattr(plan, "seq_len", None)
        if seq_len is None and len(plan.indices_shape) >= 2:
            seq_len = int(plan.indices_shape[-2])
        is_decode = _decode_call_shape(plan.positions, seq_len)
        # The ctx constructor issues this note in bundle mode; arena skips
        # the ctx, so issue it here — otherwise the active cap pair (and
        # the tighter prefill bound) lags the call's real phase.
        try:
            _note = getattr(self._cache, "note_phase", None)
            if _note is not None:
                _note(is_decode)
        except Exception:
            logger.debug(
                "expert_streaming: note_phase skipped for layer %d",
                getattr(self, "layer_idx", -1),
                exc_info=True,
            )
        # The cache keeps owning capacity: its per-layer cap bounds the
        # arena, and a demand set that cannot fit keeps the bundle path —
        # checked before the arena banks are ever allocated.
        try:
            cap = int(self._cache._cap_for(self.layer_idx))
        except Exception:
            cap = 0
        if not uniq or cap < 1 or len(uniq) > cap:
            return False
        arena = self._arena_get(projections, cap)
        if arena is None:
            return False
        # Keep book.cap synced to the active cap at engage time (a governor
        # action between engages is also pushed via cache._sync_arena_caps;
        # set_cap clamps to the build-time byte bound).
        try:
            arena.set_cap(cap)
        except Exception:
            logger.debug(
                "expert_streaming: arena set_cap skipped for layer %d",
                getattr(self, "layer_idx", -1),
                exc_info=True,
            )
        # The arena's own bound is physical (byte-ceilinged at build);
        # demand sets above it keep the bundle path.
        if len(uniq) > arena.book.cap:
            return False
        m0 = arena.book.misses
        with arena.lock:
            # frozen for non-decode calls: verify/prefill
            # experts commit at the eviction-candidate end and hits skip
            # LRU promotion, so a multi-token call churns its own cold-end
            # rows instead of sweeping the decode-hot set.
            arena.ensure_set(
                uniq,
                not is_decode,
                self._arena_produce(projections, int(plan.positions)),
            )
            plan.arena_rhs = arena.rows_for(uniq)[plan.remapped]
        arena_misses = arena.book.misses - m0
        # The layer-ctx close() feeds these counters in bundle mode; arena
        # skips the ctx, so report the same visit signal directly — the
        # governor's dynamic-budget loop stays informed either way. The
        # per-expert hit/miss pair mirrors _LayerLoadContext._count_demand:
        # covered = demanded − missed, so decode_hit_rate() and
        # streaming_gate_state() see arena-mode residency instead of a
        # permanent 0.0.
        st = getattr(self._cache, "stats", None)
        if st is not None:
            missed = arena_misses > 0
            n_demanded = len(uniq)
            n_missed = int(arena_misses)
            n_covered = max(0, n_demanded - n_missed)
            if is_decode:
                st.note_visit(self.layer_idx, missed)
                st.decode_hits += n_covered
                st.decode_misses += n_missed
            else:
                st.prefill_layers += 1
                st.prefill_layers_missed += 1 if missed else 0
                st.prefill_hits += n_covered
                st.prefill_misses += n_missed
        gov = getattr(self._cache, "governor", None)
        if gov is not None:
            try:
                gov.tick()
            except Exception:
                logger.debug(
                    "expert_streaming: governor tick skipped", exc_info=True
                )
        return True

    def _arena_get(self, projections, cap):
        """Lazily build the per-layer SlotArena over the linears' banks.

        Row shape/dtype come from a promoted expert-0 slice so arena rows
        are byte-identical to what the bundle path would feed qmm.
        """
        arena = getattr(self, "_arena", None)
        if arena is not None:
            return arena
        if getattr(self, "_arena_failed", False):
            return None
        try:
            # Sample expert 0 per field: row shape/dtype and the bytes a
            # slot costs across ALL projections — the arena bound is a
            # byte ceiling, not the cache's residency cap.
            samples: dict = {}
            row_bytes = 0
            for lin in projections:
                dt = lin._slice_dtypes_lazy()
                fields: dict = {
                    "weight": lin._promote_np(
                        lin.backing.load_expert_slice(
                            lin.stacked_weight_key, 0
                        )
                    )
                }
                s0 = lin.backing.load_expert_slice(lin.stacked_scales_key, 0)
                if s0 is not None:
                    fields["scales"] = lin._promote_np(s0, dt[0])
                if lin.stacked_biases_key:
                    try:
                        b0 = lin.backing.load_expert_slice(
                            lin.stacked_biases_key, 0
                        )
                    except Exception:
                        b0 = None
                    if b0 is not None:
                        fields["biases"] = lin._promote_np(b0, dt[1])
                    elif (
                        getattr(lin.backing, "tensor_dtype", None) is not None
                        and lin.backing.tensor_dtype(lin.stacked_biases_key)
                        is not None
                    ):
                        # Bias consistency: the biases tensor exists but
                        # the sample read came back empty — running
                        # gather_qmm without it would silently serve
                        # un-biased numerics. Refuse the arena; the bundle
                        # path surfaces its own consistency error.
                        raise ValueError(
                            "arena sample: %s has biases tensor but read "
                            "returned none" % lin.proj_name
                        )
                samples[lin.proj_name] = fields
                row_bytes += sum(
                    int(getattr(a, "nbytes", 0) or 0)
                    for a in fields.values()
                )
            rooms0 = min(cap, self._num_experts or cap)
            if row_bytes > 0:
                # The byte ceiling is a hard bound: floor 1 row (a usable
                # arena); a larger floor would overshoot the env ceiling
                # whenever _ARENA_MAX_BYTES fits under that many rows.
                rooms0 = min(rooms0, max(1, _ARENA_MAX_BYTES // row_bytes))
            if rooms0 < 1:
                self._arena_failed = True
                return None
            for lin in projections:
                bank = lin._arena_bank
                for field, arr in samples[lin.proj_name].items():
                    setattr(
                        bank,
                        field,
                        mx.zeros((rooms0, *arr.shape), dtype=arr.dtype),
                    )
            lin_by_proj = {lin.proj_name: lin for lin in projections}

            def _arrays(proj):
                bank = lin_by_proj[proj]._arena_bank
                d = {"weight": bank.weight}
                if bank.scales is not None:
                    d["scales"] = bank.scales
                if bank.biases is not None:
                    d["biases"] = bank.biases
                return d

            def _bind(proj, values):
                bank = lin_by_proj[proj]._arena_bank
                bank.weight = values["weight"]
                bank.scales = values.get("scales")
                bank.biases = values.get("biases")

            def _eval_params():
                mx.eval(
                    *[
                        a
                        for lin in projections
                        for a in (
                            lin._arena_bank.weight,
                            lin._arena_bank.scales,
                            lin._arena_bank.biases,
                        )
                        if a is not None
                    ]
                )

            self._arena = SlotArena(
                rooms0,
                [lin.proj_name for lin in projections],
                _arrays,
                _bind,
                _eval_params,
            )
            # Governor resize/set_layer_caps propagate the cache's active
            # per-layer cap into the arena (weakly held; the GLU owns it).
            try:
                reg = getattr(self._cache, "_register_arena", None)
                if reg is not None:
                    reg(self._arena, self.layer_idx)
            except Exception:
                logger.debug(
                    "expert_streaming: arena registration skipped for "
                    "layer %d",
                    getattr(self, "layer_idx", -1),
                    exc_info=True,
                )
            return self._arena
        except Exception:
            self._arena_failed = True
            logger.debug(
                "expert_streaming: arena engage permanently disabled for "
                "layer %d",
                getattr(self, "layer_idx", -1),
                exc_info=True,
            )
            return None

    def _arena_produce(self, projections, positions: int = 0):
        """Payload producer for arena.ensure_set: staged-bundle join first
        (same contract as _bundle_cached_or_staged), then one coalesced
        bank read per linear for the still-missing ids.

        ``positions`` is the CALL's routed-row count — the pool regime
        input. The arena miss count must not drive it: a prefill that
        evicts few residents still wants the prefill pool."""
        linears = list(projections)

        def _fields(lin, bundle, dt):
            w, s, b = bundle
            out = {"weight": lin._promote_np(w)}
            if s is not None:
                out["scales"] = lin._promote_np(s, dt[0])
            if b is not None:
                out["biases"] = lin._promote_np(b, dt[1])
            return out

        def produce(fetch_list):
            ids = [int(e) for e, _s, _v in fetch_list]
            spec = linears[0]._spec_state()
            # Phase 1 — staged-bundle join per linear (cheap dict ops on
            # the inference thread; same contract as _bundle_cached_or_staged).
            per_lin = []  # (lin, dtypes, rows, missing[(row_idx, eid)])
            for lin in linears:
                dt = lin._slice_dtypes_lazy()
                rows = [None] * len(ids)
                missing = []
                for i, eid in enumerate(ids):
                    row = None
                    if spec is not None:
                        try:
                            r = spec.stage_resolve(lin.bundle_key(eid))
                            if isinstance(r, tuple) and len(r) == 3:
                                row = r
                        except Exception:
                            row = None
                    if row is not None:
                        rows[i] = _fields(lin, row, dt)
                    else:
                        missing.append((i, eid))
                per_lin.append((lin, dt, rows, missing))
            # Phase 2 — bank reads for the still-missing ids submitted to
            # the io pool TOGETHER (cross-projection parallelism — serial
            # per-projection reads cost ~3x the miss latency).
            # _ARENA_PRODUCE_JOBS>1 additionally splits each projection's
            # set into chunks.
            jobs = []  # (per_lin_idx, missing_chunk)
            for li, (_lin, _dt, _rows, missing) in enumerate(per_lin):
                if not missing:
                    continue
                if _ARENA_PRODUCE_JOBS > 1 and len(missing) > 1:
                    step = -(-len(missing) // _ARENA_PRODUCE_JOBS)
                    jobs.extend(
                        (li, missing[i : i + step])
                        for i in range(0, len(missing), step)
                    )
                else:
                    jobs.append((li, missing))
            if len(jobs) > 1:
                pool = _io_pool(linears[0])
                futs = [
                    pool.submit(
                        per_lin[li][0]._load_expert_bank_np,
                        [e for _, e in chunk],
                    )
                    for li, chunk in jobs
                ]
                results = [f.result() for f in futs]
            else:
                results = [
                    per_lin[jobs[0][0]][0]._load_expert_bank_np(
                        [e for _, e in jobs[0][1]]
                    )
                ] if jobs else []
            for (li, chunk), banks in zip(jobs, results):
                lin, dt, rows, _m = per_lin[li]
                if banks is None or len(banks) != len(chunk):
                    raise RuntimeError("arena produce: bank read failed")
                for j, b in enumerate(banks):
                    rows[chunk[j][0]] = _fields(lin, b, dt)
            return [
                {lin.proj_name: rows[i] for lin, _dt, rows, _m in per_lin}
                for i in range(len(ids))
            ]

        return produce
