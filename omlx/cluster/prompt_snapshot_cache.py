# SPDX-License-Identifier: Apache-2.0
"""SSD-backed prompt-cache snapshots for the distributed rank server.

A distributed rank runs the pinned ``mlx_lm.server`` with a rank-local
in-memory prompt cache. That cache is bounded and dies with the process, and it
cannot help a model whose per-layer state is not sliceable: a ``RotatingKVCache``
overwrites its window and a gated-delta-net keeps a single recurrent state, so
the state at an interior prefix boundary is gone the moment prefill moves past
it. This store persists chains of boundary files to local SSD, using MLX-LM's
own ``save_prompt_cache`` / ``load_prompt_cache`` so every cache type
serialises through its declared ``state`` / ``meta_state``.

A boundary file holds what that boundary added, mirroring the local paged SSD
cache's policy: plain ``KVCache`` members store only their newest step-sized
slab (positionally immutable, so a chain holds one copy of the KV rather than
one cumulative copy per boundary). Rotating windows and recurrent slots keep
their exact state at each boundary. PoolingCache keeps those boundary-local
carry slots plus only the append-only pooled rows added by that segment; chain
restore validates and joins their absolute ranges.

Each rank keeps its own directory holding its own layer-slice snapshots. The
keys are a hash of ``(model, prefix tokens)`` and are therefore identical across
ranks that processed the same broadcast request, while the bytes under a key are
this rank's shard alone; prompts sharing a prefix share the early chain files.
Eviction is a deterministic bounded LRU keyed on the sequence of operations
rather than a wall clock. Unequal shards can produce different file sizes, so
the caller supplies a rank-agreed capacity charge for each boundary; ranks that
see the same requests then make the same byte-budget decision. Coordinating the
*hit* across ranks (so a disk write that failed on one rank cannot desync the
pipeline) is also the caller's job and lives in the telemetry integration,
which owns the reliable control plane; this module stays pure and unit-testable.

When persistence is enabled, a compact atomic manifest records the digest,
boundary and byte size of every chain segment. The digest already commits to
the model identity and exact prefix tokens, so the manifest does not duplicate
the token arrays (which would grow quadratically at long context). Directories
are scoped by plan hash and rank by the worker, preventing a changed tensor
split from ever reading another shard layout. A missing or invalid manifest
fails closed by clearing otherwise-unindexable files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import struct
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omlx.cache.pooling_delta import POOLING_CACHE_DELTA_FORMAT_VERSION

from .performance import DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES

logger = logging.getLogger(__name__)

# Chain files are cheap (one step-sized slab plus constant-size window and
# recurrent states), so the count bound mainly limits how many distinct
# reusable boundaries exist across all prompts; the byte bound is the backstop.
_MAX_ENTRIES_DEFAULT = 512
_MAX_PENDING_WRITES_DEFAULT = 2
_PENDING_MAX_BYTES_DEFAULT = 512 * 1024 * 1024

# This marker is deliberately independent of the Python class name.  The
# class name lets mlx-lm find the deserialiser, while the marker/version below
# is the durable wire contract that the chain assembler validates before it
# trusts any absolute range metadata.
_POOLING_DELTA_WIRE_MARKER = "omlx.cluster.pooling-cache-delta"


def _slot_layout(state: tuple[Any, ...]) -> str:
    """Encode every PoolingCache state slot without storing empty leaves."""

    layout = []
    for value in state:
        if value is None:
            layout.append({"kind": "none"})
            continue
        shape = [int(dim) for dim in value.shape]
        layout.append(
            {
                "kind": "empty" if int(value.size) == 0 else "array",
                "shape": shape,
                "dtype": str(value.dtype),
            }
        )
    return json.dumps(layout, separators=(",", ":"), sort_keys=True)


def _pool_schema(value: Any) -> str:
    """Stable pooled-row schema, excluding its append-only sequence axis."""

    if value is None:
        return "none"
    shape = [int(dim) for dim in value.shape]
    if len(shape) < 2:
        raise ValueError("PoolingCache pooled state has no sequence axis")
    return json.dumps(
        {
            "dtype": str(value.dtype),
            "shape_without_sequence": [shape[0], *shape[2:]],
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_pooling_state_layout(state: tuple[Any, ...]) -> None:
    if len(state) not in (3, 5):
        raise ValueError("PoolingCache state must have three or five slots")
    if (state[0] is None) != (state[1] is None):
        raise ValueError("PoolingCache remainder KV/gate slots disagree")
    if len(state) == 5 and (state[3] is None) != (state[4] is None):
        raise ValueError("PoolingCache previous-window KV/gate slots disagree")


class PoolingCacheDeltaSnapshot:
    """Versioned wire record for one absolute PoolingCache pooled-row delta.

    Remainder and overlap-carry slots are the exact state at ``source_end``;
    only the append-only pooled slot is sliced to ``[pool_start:pool_end]``.
    ``from_state`` intentionally returns another instance of this carrier,
    rather than a live PoolingCache: a single segment is incomplete and must
    never escape without the chain assembler validating and joining it.
    """

    def __init__(
        self,
        state: tuple[Any, ...],
        *,
        ratio: int,
        source_start: int,
        source_end: int,
        pool_start: int,
        pool_end: int,
        pool_schema: str,
    ) -> None:
        _validate_pooling_state_layout(state)
        self._wire_state = state
        self.ratio = int(ratio)
        self.source_start = int(source_start)
        self.source_end = int(source_end)
        self.pool_start = int(pool_start)
        self.pool_end = int(pool_end)
        self.pool_schema = str(pool_schema)
        self.state_arity = len(state)
        self.layout = _slot_layout(state)

    @classmethod
    def from_cache(
        cls,
        inner: Any,
        *,
        source_start: int,
        source_end: int,
    ) -> PoolingCacheDeltaSnapshot | None:
        """Return a compact record only when absolute geometry is provable."""

        try:
            import mlx.core as mx

            ratio = int(inner.ratio)
            state = tuple(inner.state)
            _validate_pooling_state_layout(state)
            if ratio <= 0 or source_start < 0 or source_end <= source_start:
                return None
            pool_start = source_start // ratio
            pool_end = source_end // ratio
            pooled = state[2]
            if pooled is None:
                if pool_end != 0:
                    return None
                delta = None
            else:
                if len(pooled.shape) < 2 or int(pooled.shape[1]) != pool_end:
                    return None
                delta = pooled[:, pool_start:pool_end]
                # A bare view pins the entire cumulative allocation until an
                # async writer has copied it.  Materialise the small range now.
                if hasattr(mx, "contiguous"):
                    delta = mx.contiguous(delta)
            compacted = list(state)
            compacted[2] = delta
            return cls(
                tuple(compacted),
                ratio=ratio,
                source_start=source_start,
                source_end=source_end,
                pool_start=pool_start,
                pool_end=pool_end,
                pool_schema=_pool_schema(pooled),
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            # An unfamiliar PoolingCache variant stays a cumulative snapshot;
            # correctness is more important than forcing compaction.
            return None

    @property
    def state(self) -> tuple[Any, ...]:
        import mlx.core as mx

        kept = tuple(
            value
            for value in self._wire_state
            if value is not None and int(value.size) > 0
        )
        return kept or (mx.zeros((1,), dtype=mx.uint8),)

    @property
    def meta_state(self) -> tuple[str, ...]:
        return (
            _POOLING_DELTA_WIRE_MARKER,
            POOLING_CACHE_DELTA_FORMAT_VERSION,
            str(self.ratio),
            str(self.source_start),
            str(self.source_end),
            str(self.pool_start),
            str(self.pool_end),
            str(self.state_arity),
            self.pool_schema,
            self.layout,
        )

    @classmethod
    def from_state(cls, state: Any, meta_state: Any) -> PoolingCacheDeltaSnapshot:
        import mlx.core as mx

        if not isinstance(meta_state, (list, tuple)) or len(meta_state) != 10:
            raise ValueError("invalid PoolingCache delta metadata layout")
        (
            marker,
            version,
            ratio_text,
            source_start_text,
            source_end_text,
            pool_start_text,
            pool_end_text,
            arity_text,
            pool_schema,
            layout_text,
        ) = meta_state
        if marker != _POOLING_DELTA_WIRE_MARKER:
            raise ValueError("invalid PoolingCache delta wire marker")
        if version != POOLING_CACHE_DELTA_FORMAT_VERSION:
            raise ValueError("unsupported PoolingCache delta wire version")
        try:
            ratio = int(ratio_text)
            source_start = int(source_start_text)
            source_end = int(source_end_text)
            pool_start = int(pool_start_text)
            pool_end = int(pool_end_text)
            state_arity = int(arity_text)
            layout = json.loads(layout_text)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("corrupt PoolingCache delta metadata") from error
        if (
            ratio <= 0
            or source_start < 0
            or source_end <= source_start
            or pool_start < 0
            or pool_end < pool_start
            or state_arity not in (3, 5)
            or not isinstance(layout, list)
            or len(layout) != state_arity
        ):
            raise ValueError("unsafe PoolingCache delta ranges or layout")

        arrays = list(state) if isinstance(state, (list, tuple)) else [state]
        substantive = sum(
            1
            for spec in layout
            if isinstance(spec, dict) and spec.get("kind") == "array"
        )
        if substantive and len(arrays) != substantive:
            raise ValueError("PoolingCache delta tensor count disagrees with layout")
        if not substantive and len(arrays) != 1:
            raise ValueError("PoolingCache delta placeholder is missing")

        rebuilt = []
        array_position = 0
        for spec in layout:
            if not isinstance(spec, dict) or spec.get("kind") not in {
                "none",
                "empty",
                "array",
            }:
                raise ValueError("invalid PoolingCache delta slot descriptor")
            kind = spec["kind"]
            if kind == "none":
                rebuilt.append(None)
                continue
            try:
                shape = tuple(int(dim) for dim in spec["shape"])
                dtype_text = str(spec["dtype"])
                dtype = getattr(mx, dtype_text.rsplit(".", 1)[-1])
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid PoolingCache delta tensor schema") from error
            if any(dim < 0 for dim in shape):
                raise ValueError("negative PoolingCache delta tensor dimension")
            if kind == "empty":
                if all(dim > 0 for dim in shape):
                    raise ValueError("empty PoolingCache delta slot has nonempty shape")
                rebuilt.append(mx.zeros(shape, dtype=dtype))
                continue
            value = arrays[array_position]
            array_position += 1
            if tuple(int(dim) for dim in value.shape) != shape or value.dtype != dtype:
                raise ValueError("PoolingCache delta tensor disagrees with schema")
            rebuilt.append(value)
        if array_position != substantive:
            raise ValueError("unused PoolingCache delta tensors")

        rebuilt_state = tuple(rebuilt)
        _validate_pooling_state_layout(rebuilt_state)
        obj = cls(
            rebuilt_state,
            ratio=ratio,
            source_start=source_start,
            source_end=source_end,
            pool_start=pool_start,
            pool_end=pool_end,
            pool_schema=str(pool_schema),
        )
        # Preserve the original layout string exactly after it has been
        # validated; a re-encode must match or the metadata was noncanonical.
        if obj.layout != layout_text or obj.state_arity != state_arity:
            raise ValueError("noncanonical PoolingCache delta layout")
        return obj


@dataclass
class _FrozenPromptSnapshot:
    """Detached, fully evaluated payload safe to hand to the writer thread."""

    tensors: dict[str, Any]
    metadata: dict[str, str]
    nbytes: int
    capacity_charge_bytes: int


class PoolingCacheSnapshot:
    """Legacy cumulative stand-in for the DeepSeek sparse-attention pool cache.

    ``save_prompt_cache`` serialises through ``state`` / ``meta_state`` and
    reconstructs via ``globals()[name].from_state`` inside
    ``mlx_lm.models.cache``. A ``PoolingCache`` breaks both directions: its
    state tuple carries ``None`` slots safetensors cannot hold, and its
    meta_state is a bare int where the metadata tree must be all strings. The
    in-process state contract is load-bearing for the paged-cache handlers, so
    rather than changing the class this stand-in presents the same cache in
    wire form (arrays-only state, with the slot layout encoded as a flag
    string), and its ``from_state`` hands back a real ``PoolingCache``. New
    aligned writes use ``PoolingCacheDeltaSnapshot``; this carrier remains the
    safe fallback for unfamiliar geometry and the migration base for existing
    persistent files.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def state(self) -> tuple[Any, ...]:
        import mlx.core as mx

        kept = tuple(a for a in self._inner.state if a is not None)
        # An all-empty state must still occupy its slot in the flattened file
        # or every later cache's arrays would shift onto the wrong class. The
        # placeholder is never consumed: the flags drive consumption.
        return kept or (mx.zeros((1,)),)

    @property
    def meta_state(self) -> tuple[str, str]:
        flags = "".join("0" if a is None else "1" for a in self._inner.state)
        return (str(int(self._inner.ratio)), flags)

    @classmethod
    def from_state(cls, state: Any, meta_state: tuple[str, str]) -> Any:
        from omlx.patches.deepseek_v4.cache_extras import PoolingCache

        ratio, flags = meta_state
        cache = PoolingCache(int(ratio))
        arrays = iter(state if isinstance(state, (list, tuple)) else [state])
        # The existing state setter replays any remainder rows through
        # accumulate_windows, so the restored cache continues exactly like the
        # live one it was copied from.
        cache.state = tuple(next(arrays) if f == "1" else None for f in flags)
        return cache


class EmptyLeafSnapshot:
    """Serialisation stand-in for a cache state with unserialisable leaves.

    safetensors can hold neither a zero-size array nor a ``None`` leaf, yet
    both are legitimate cache states: a sparse-attention branch below its
    engagement length keeps an untouched rotating member whose state slices are
    zero-size, and a recurrent slot may simply not be written yet. This
    stand-in stores only the substantive arrays and records the position,
    shape and dtype of every dropped leaf, so the restored cache is exactly
    what a deepcopy of the live one would have held.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def state(self) -> tuple[Any, ...]:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        kept = tuple(
            leaf
            for _key, leaf in tree_flatten(self._inner.state)
            if leaf is not None and leaf.size > 0
        )
        # Same slot-holding placeholder as PoolingCacheSnapshot: the layout
        # below never marks it for consumption.
        return kept or (mx.zeros((1,)),)

    @property
    def meta_state(self) -> tuple[Any, ...]:
        from mlx.utils import tree_flatten

        layout = []
        for key, leaf in tree_flatten(self._inner.state):
            if leaf is None:
                layout.append(f"{key}=none")
            elif leaf.size == 0:
                shape = "x".join(str(d) for d in leaf.shape)
                layout.append(f"{key}=empty:{shape}:{leaf.dtype}")
            else:
                layout.append(f"{key}=array")
        return (type(self._inner).__name__, tuple(layout), self._inner.meta_state)

    @classmethod
    def from_state(cls, state: Any, meta_state: Any) -> Any:
        import mlx.core as mx
        import mlx_lm.models.cache as cache_module
        from mlx.utils import tree_unflatten

        inner_name, layout, inner_meta = meta_state
        arrays = iter(state if isinstance(state, (list, tuple)) else [state])
        pairs = []
        for item in layout:
            key, _, kind = item.partition("=")
            if kind == "array":
                pairs.append((key, next(arrays)))
            elif kind == "none":
                pairs.append((key, None))
            else:
                _, shape_text, dtype_text = kind.split(":")
                shape = tuple(int(d) for d in shape_text.split("x") if d)
                dtype = getattr(mx, dtype_text.rsplit(".", 1)[-1])
                pairs.append((key, mx.zeros(shape, dtype=dtype)))
        inner_cls = getattr(cache_module, inner_name)
        return inner_cls.from_state(tree_unflatten(pairs), inner_meta)


class KVCacheSegment:
    """Wire stand-in holding only the tokens a boundary added to a KVCache.

    Plain attention KV is positionally immutable: the K/V rows for tokens
    (start, b] never change after prefill writes them. Storing just that slab
    per boundary makes a chain of boundary files hold one copy of the KV
    instead of one cumulative copy per boundary, exactly like the local paged
    SSD cache's block policy. ``from_state`` hands back a real ``KVCache``
    carrying the slab, tagged with its start so the chain assembler knows to
    concatenate it rather than treat it as a whole cache.
    """

    def __init__(self, inner: Any, start: int) -> None:
        self._inner = inner
        self._start = start

    def _slabs(self) -> tuple[Any, Any]:
        keys, values = self._inner.state
        return (keys[..., self._start :, :], values[..., self._start :, :])

    @property
    def state(self) -> tuple[Any, ...]:
        import mlx.core as mx

        # An MLA-style cache keeps a zero-width half (all data in the latent
        # keys, values shaped (..., 0)); safetensors cannot hold it, so only
        # substantive slabs are stored and the layout below rebuilds the rest.
        kept = tuple(slab for slab in self._slabs() if slab.size > 0)
        return kept or (mx.zeros((1,)),)

    @property
    def meta_state(self) -> tuple[str, tuple[str, str]]:
        layout = []
        for slab in self._slabs():
            if slab.size == 0:
                shape = "x".join(str(d) for d in slab.shape)
                layout.append(f"empty:{shape}:{slab.dtype}")
            else:
                layout.append("array")
        return (str(int(self._start)), tuple(layout))

    @classmethod
    def from_state(cls, state: Any, meta_state: Any) -> Any:
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        start, layout = meta_state
        arrays = iter(state if isinstance(state, (list, tuple)) else [state])
        pair = []
        for kind in layout:
            if kind == "array":
                pair.append(next(arrays))
            else:
                _, shape_text, dtype_text = kind.split(":")
                shape = tuple(int(d) for d in shape_text.split("x") if d)
                pair.append(
                    mx.zeros(shape, dtype=getattr(mx, dtype_text.rsplit(".", 1)[-1]))
                )
        keys, values = pair
        cache = KVCache()
        cache.keys = keys
        cache.values = values
        # A zero-width half still carries the sequence axis, so the shape is a
        # valid length source either way.
        cache.offset = keys.shape[2]
        cache._omlx_segment_start = int(start)  # type: ignore[attr-defined]
        return cache


def _has_unserialisable_leaves(entry: Any) -> bool:
    from mlx.utils import tree_flatten

    return any(
        leaf is None or leaf.size == 0 for _key, leaf in tree_flatten(entry.state)
    )


def _register_snapshot_classes() -> None:
    """Make the stand-ins resolvable where ``load_prompt_cache`` looks them up.

    Both the top-level loader and ``CacheList.from_state`` resolve class names
    against ``mlx_lm.models.cache`` globals, so the names written into a
    snapshot file must exist there when any rank loads it.
    """

    import mlx_lm.models.cache as cache_module

    for snapshot_class in (
        PoolingCacheSnapshot,
        PoolingCacheDeltaSnapshot,
        EmptyLeafSnapshot,
        KVCacheSegment,
    ):
        name = snapshot_class.__name__
        if getattr(cache_module, name, None) is not snapshot_class:
            setattr(cache_module, name, snapshot_class)


def _wrap_for_save(
    cache: list[Any], *, boundary: int = 0, segment_start: int = 0
) -> list[Any]:
    """Swap serialisation-hostile entries for their wire stand-ins.

    Returns a parallel list; the live cache is never touched. When ``boundary``
    is set, plain ``KVCache`` members that are in step with the token stream
    (offset equals the boundary) are stored as segments holding only the
    tokens past ``segment_start``; everything else keeps its full state, so a
    member with any unusual offset stays whole rather than risking a slab
    that does not compose.
    """

    from mlx_lm.models.cache import CacheList, KVCache

    try:
        from omlx.patches.deepseek_v4.cache_extras import PoolingCache
    except ImportError:
        pooling_class: Any = None
    else:
        pooling_class = PoolingCache

    def wrap(entry: Any) -> Any:
        if pooling_class is not None and isinstance(entry, pooling_class):
            if boundary > 0:
                delta = PoolingCacheDeltaSnapshot.from_cache(
                    entry,
                    source_start=segment_start,
                    source_end=boundary,
                )
                if delta is not None:
                    return delta
            return PoolingCacheSnapshot(entry)
        if isinstance(entry, CacheList):
            members = [wrap(m) for m in entry.caches]
            if any(m is not o for m, o in zip(members, entry.caches)):
                return CacheList(*members)
            return entry
        if (
            boundary > 0
            and type(entry).__name__ == "KVCache"
            and isinstance(entry, KVCache)
            and getattr(entry, "offset", 0) == boundary
        ):
            return KVCacheSegment(entry, segment_start)
        if _has_unserialisable_leaves(entry):
            return EmptyLeafSnapshot(entry)
        return entry

    return [wrap(entry) for entry in cache]


def _prepare_snapshot_payload(
    cache: list[Any], *, boundary: int, segment_start: int
) -> tuple[dict[str, Any], dict[str, str], int]:
    """Flatten one boundary while its live cache still names that boundary.

    This deliberately does not copy the arrays yet.  The caller first uses the
    returned byte count to reserve bounded write-behind capacity, then copies
    and evaluates every leaf before returning control to generation.  Metadata
    is also sampled here: rotating offsets and recurrent layouts are just as
    boundary-sensitive as the tensor bytes.
    """

    import mlx.core as mx
    from mlx.utils import tree_flatten

    wrapped = _wrap_for_save(
        cache,
        boundary=boundary,
        segment_start=segment_start,
    )
    tensors = dict(tree_flatten([entry.state for entry in wrapped]))
    if not tensors or any(not isinstance(value, mx.array) for value in tensors.values()):
        raise TypeError("prompt snapshot state must contain only MLX arrays")
    metadata = dict(
        tree_flatten(
            [
                [entry.meta_state for entry in wrapped],
                {},
                [type(entry).__name__ for entry in wrapped],
            ]
        )
    )
    if any(not isinstance(value, str) for value in metadata.values()):
        raise TypeError("prompt snapshot metadata must contain only strings")
    nbytes = sum(int(value.nbytes) for value in tensors.values())
    return tensors, metadata, nbytes


def _snapshot_capacity_charge(
    tensors: dict[str, Any], metadata: dict[str, str], nbytes: int
) -> int:
    """Conservative on-disk charge known before the async write starts.

    Safetensors stores the raw tensor bytes plus one JSON header. The header
    depends on rank-local tensor shapes and metadata, so charge a deliberately
    loose upper bound and let the distributed caller agree on the largest
    rank's value. That common charge makes byte-budget eviction choose the
    same LRU victims even when pipeline stages hold different layer counts.
    The writer still checks real file bytes as a fail-safe.
    """

    text_bytes = sum(
        len(str(key).encode("utf-8")) + len(value.encode("utf-8"))
        for key, value in metadata.items()
    )
    schema_bytes = 0
    for name, value in tensors.items():
        schema_bytes += len(str(name).encode("utf-8"))
        schema_bytes += len(str(tuple(value.shape)).encode("ascii"))
        schema_bytes += len(str(value.dtype).encode("ascii"))
    # JSON escaping can expand a code point to six ASCII bytes. 64 KiB also
    # covers safetensors' framing/alignment for ordinary small snapshots.
    header_bound = max(64 * 1024, 8 * (text_bytes + schema_bytes + 1024))
    return max(1, int(nbytes) + header_bound)


def _freeze_snapshot_payload(
    tensors: dict[str, Any],
    metadata: dict[str, str],
    nbytes: int,
    capacity_charge_bytes: int,
) -> _FrozenPromptSnapshot:
    """Copy and fully materialize a payload on its owning inference thread.

    ``KVCache`` and ``RotatingKVCache`` use slice assignment, so merely keeping
    another Python reference would let later prefill/decode overwrite the
    alleged boundary before the writer reaches it.  ``mx.array`` creates a
    detached buffer and the blocking eval is load-bearing: lazy arrays created
    on a thread-local generation stream cannot safely be evaluated by the
    background writer.
    """

    import mlx.core as mx

    from omlx.utils.metal_sync import _mx_buffer_access_lock

    with _mx_buffer_access_lock:
        frozen = {name: mx.array(value) for name, value in tensors.items()}
        mx.eval(*frozen.values())
    return _FrozenPromptSnapshot(
        frozen,
        dict(metadata),
        nbytes,
        max(int(capacity_charge_bytes), int(nbytes)),
    )


@dataclass
class _Entry:
    tokens: tuple[int, ...]
    filename: str
    nbytes: int
    boundary: int
    capacity_charge_bytes: int


def candidate_boundaries(prompt_len: int, step: int) -> tuple[int, ...]:
    """Prefix lengths a snapshot may exist at, longest first.

    Boundaries are the ``step`` multiples at or below ``prompt_len``: the exact
    positions prefill pauses at. Keeping them aligned means a restored prefix
    always leaves the next request's prefill on the same grid, so later
    snapshots keep landing on boundaries other requests can reuse. All ranks
    derive the same list from the same broadcast prompt length, so a one-hot
    vote over these indices lines up across ranks.
    """

    if prompt_len <= 0 or step <= 0:
        return ()
    return tuple(k * step for k in range(prompt_len // step, 0, -1))


def agreed_boundary(
    candidates: tuple[int, ...],
    summed_votes: list[int],
    world_size: int,
) -> int:
    """Longest boundary present on every rank, from summed one-hot votes.

    ``candidates`` is longest-first and ``summed_votes[i]`` is how many ranks
    reported ``candidates[i]``. A boundary is taken only when all of them did, so
    a snapshot missing on any single rank is never restored; the alternative is
    one rank reusing a prefix the others recompute, which desyncs the pipeline.
    """

    for boundary, count in zip(candidates, summed_votes):
        if int(count) == world_size:
            return int(boundary)
    return 0


class SSDPromptSnapshotStore:
    """A rank-local, deterministic, chain-of-segments store of cache snapshots.

    A boundary file holds what that boundary added: plain ``KVCache`` members
    contribute only their new (b - step, b] slab, while non-sliceable members
    (rotating windows, recurrent slots, pooling state) contribute their full
    state at b, which is the only representation they have. Restoring boundary
    B therefore needs every chain file at step, 2*step, ..., B; a chain with a
    hole simply does not offer the boundaries past the hole. Files are keyed by
    the token prefix they end at, so two prompts sharing a prefix share the
    early files, exactly like the local paged SSD cache shares blocks.

    Eviction is a deterministic bounded LRU (entry count and bytes) keyed on
    the sequence of operations rather than a wall clock. Distributed callers
    provide one shared capacity charge per boundary, so unequal shard file
    sizes still choose the same victims. Evicting an
    early file orphans the deeper files of its chain; they stop being offered,
    are never touched again, age to the LRU front and fall out on their own.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        step: int = 2048,
        max_entries: int = _MAX_ENTRIES_DEFAULT,
        max_bytes: int = DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES,
        persistent: bool = False,
        write_behind: bool = False,
        max_pending_writes: int = _MAX_PENDING_WRITES_DEFAULT,
        pending_max_bytes: int = _PENDING_MAX_BYTES_DEFAULT,
        capacity_agreement: Callable[[int | None], int | None] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.step = max(1, int(step))
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(1, int(max_bytes))
        self.persistent = bool(persistent)
        self.write_behind = bool(write_behind)
        self.max_pending_writes = max(1, int(max_pending_writes))
        self.pending_max_bytes = max(1, int(pending_max_bytes))
        self._capacity_agreement = capacity_agreement
        self._lock = threading.RLock()
        # Access-ordered: most-recently-used at the end. The order is advanced
        # only by put/load, both driven by the identical request stream every
        # rank sees, so eviction is the same decision on every rank.
        self._index: OrderedDict[str, _Entry] = OrderedDict()
        self._nbytes = 0
        self._capacity_bytes = 0
        self._evictions = 0
        self._capacity_drops = 0
        # A cache type save_prompt_cache rejects will be rejected on every
        # boundary, so the store disables itself after the first such failure
        # rather than paying a doomed write per request. Known-hostile types
        # get a wire stand-in instead (see ``_wrap_for_save``); this flag is
        # the backstop for a type nobody has taught the store about yet.
        self._serialisable = True
        self._closed = False
        # Write-behind owns detached, fully evaluated payloads only.  Entries
        # remain pending until their atomic rename and manifest update finish;
        # present_boundaries/load intentionally expose committed files only.
        self._pending: dict[str, int] = {}
        self._capturing: set[str] = set()
        self._pending_cond = threading.Condition(self._lock)
        self._pending_bytes = 0
        self._pending_peak_bytes = 0
        self._write_failures = 0
        self._write_queue: queue.Queue[Any] | None = None
        self._writer_thread: threading.Thread | None = None
        self._stop_enqueued = False
        _register_snapshot_classes()
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.persistent:
            self._load_manifest_or_reset()
        else:
            self._clear_directory()
        if self.write_behind:
            self._write_queue = queue.Queue(maxsize=self.max_pending_writes)
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="cluster-prompt-snapshot-writer",
                daemon=True,
            )
            self._writer_thread.start()

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._nbytes

    @property
    def capacity_bytes(self) -> int:
        """Rank-symmetric bytes charged against ``max_bytes``."""

        with self._lock:
            return self._capacity_bytes

    @property
    def evictions(self) -> int:
        with self._lock:
            return self._evictions

    @property
    def capacity_drops(self) -> int:
        with self._lock:
            return self._capacity_drops

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def pending_bytes(self) -> int:
        with self._lock:
            return self._pending_bytes

    @property
    def pending_peak_bytes(self) -> int:
        with self._lock:
            return self._pending_peak_bytes

    @property
    def write_failures(self) -> int:
        with self._lock:
            return self._write_failures

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.safetensors"

    @property
    def _manifest_path(self) -> Path:
        return self.directory / "index.json"

    def _clear_directory(self) -> None:
        self._index.clear()
        self._nbytes = 0
        self._capacity_bytes = 0
        for stale in self.directory.iterdir():
            if stale.is_file():
                with suppress(OSError):
                    stale.unlink()

    @staticmethod
    def _valid_key(value: object) -> bool:
        return bool(
            isinstance(value, str)
            and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value)
        )

    def _load_manifest_or_reset(self) -> None:
        """Restore the durable LRU, or fail closed on any malformed state."""

        manifest = self._manifest_path
        if not manifest.is_file():
            self._clear_directory()
            self._persist_index_locked()
            return
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            version = int(payload.get("version", 0))
            if version not in (1, 2) or int(payload.get("step")) != self.step:
                raise ValueError("snapshot manifest contract changed")
            rows = payload.get("entries")
            if not isinstance(rows, list):
                raise ValueError("snapshot manifest entries are invalid")
            indexed_files = {manifest.name}
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("snapshot manifest entry is invalid")
                key = row.get("key")
                filename = row.get("filename")
                boundary = int(row.get("boundary"))
                nbytes = int(row.get("nbytes"))
                capacity_charge_bytes = int(
                    row.get("capacity_charge_bytes", nbytes)
                )
                if (
                    not self._valid_key(key)
                    or filename != f"{key}.safetensors"
                    or boundary <= 0
                    or boundary % self.step != 0
                    or nbytes <= 0
                    or capacity_charge_bytes < nbytes
                ):
                    raise ValueError("snapshot manifest entry is unsafe")
                path = self.directory / filename
                if key in self._index:
                    raise ValueError("snapshot manifest contains a duplicate key")
                if not path.is_file() or path.stat().st_size != nbytes:
                    raise ValueError("snapshot manifest file is missing or changed")
                self._index[key] = _Entry(
                    (), filename, nbytes, boundary, capacity_charge_bytes
                )
                self._nbytes += nbytes
                self._capacity_bytes += capacity_charge_bytes
                indexed_files.add(filename)
            for stale in self.directory.iterdir():
                if stale.is_file() and stale.name not in indexed_files:
                    with suppress(OSError):
                        stale.unlink()
            self._evict_locked()
            self._persist_index_locked()
        except Exception as error:
            logger.warning("resetting invalid prompt snapshot manifest: %s", error)
            self._clear_directory()
            self._persist_index_locked()

    def _persist_index_locked(self) -> None:
        if not self.persistent:
            return
        payload = {
            "version": 2,
            "step": self.step,
            "entries": [
                {
                    "key": key,
                    "filename": entry.filename,
                    "nbytes": entry.nbytes,
                    "boundary": entry.boundary,
                    "capacity_charge_bytes": entry.capacity_charge_bytes,
                }
                for key, entry in self._index.items()
            ],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".index.", suffix=".json", dir=self.directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._manifest_path)
        except OSError as error:
            with suppress(OSError):
                os.close(descriptor)
            with suppress(OSError):
                os.unlink(temporary)
            logger.warning("could not persist prompt snapshot manifest: %s", error)

    def _chain_keys(self, model: Any, tokens: tuple[int, ...]) -> list[str]:
        """Digest per chain boundary, shortest first, sharing one hash walk."""

        hasher = hashlib.sha256()
        hasher.update(repr(model).encode("utf-8"))
        hasher.update(b"\x00")
        keys = []
        for start in range(0, len(tokens) - self.step + 1, self.step):
            chunk = tokens[start : start + self.step]
            hasher.update(struct.pack(f"<{len(chunk)}q", *chunk))
            keys.append(hasher.copy().hexdigest())
        return keys

    def put(self, model: Any, tokens: list[int], cache: list[Any] | None) -> bool:
        """Persist the boundary file for ``tokens``. Best effort.

        ``tokens`` must end on the step grid. Plain KVCache members are stored
        as their newest slab, everything else as full state. A file that
        already exists is this boundary written by an earlier request sharing
        the prefix; it is kept and touched rather than rewritten, which is
        what lets branching prompts share their common chain.  By default the
        write is synchronous.  With ``write_behind=True`` a detached boundary
        payload is queued and True means accepted, not yet durable; reads and
        rank votes continue to expose committed files only.
        """

        token_tuple = tuple(int(t) for t in tokens)
        boundary = len(token_tuple)
        if boundary == 0 or boundary % self.step != 0:
            return False
        key = self._chain_keys(model, token_tuple)[-1]
        with self._lock:
            if (
                (self._closed or not self._serialisable)
                and not (self.write_behind and self._capacity_agreement is not None)
            ):
                return False
            entry = self._index.get(key)
            if entry is not None and (
                entry.tokens == token_tuple
                or (not entry.tokens and entry.boundary == boundary)
            ):
                # In write-behind mode a peer rank may still have this key
                # pending.  Treating an already-committed duplicate as an LRU
                # touch on only the faster rank would make count eviction
                # depend on SSD timing.  Loads remain rank-agreed touches.
                if not self.write_behind:
                    self._index.move_to_end(key)
                    self._persist_index_locked()
                    return True
                if self._capacity_agreement is None:
                    return True
            if (
                self.write_behind
                and key in self._pending
                and self._capacity_agreement is None
            ):
                return True

        if self.write_behind:
            return self._put_write_behind(
                key=key,
                token_tuple=token_tuple,
                boundary=boundary,
                cache=cache,
            )

        from mlx_lm.models.cache import save_prompt_cache

        wrapped = _wrap_for_save(
            cache, boundary=boundary, segment_start=boundary - self.step
        )
        target = self._path(key)
        temporary = None
        try:
            # A ``.safetensors`` suffix matters: ``mx.save_safetensors`` appends
            # it otherwise, and the atomic rename below would miss the real file.
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{key}.", suffix=".safetensors", dir=self.directory
            )
            os.close(descriptor)
            save_prompt_cache(temporary, wrapped)
            size = os.path.getsize(temporary)
            if size > self.max_bytes:
                with suppress(OSError):
                    os.unlink(temporary)
                with self._lock:
                    self._capacity_drops += 1
                return False
            os.replace(temporary, target)
        except OSError:
            # A transient disk problem: drop this snapshot, keep the store live.
            if temporary is not None:
                with suppress(OSError):
                    os.unlink(temporary)
            return False
        except Exception as error:
            # The cache type itself cannot be serialised. Stop trying for this
            # model so a boundary is not paid for on every request.
            if temporary is not None:
                with suppress(OSError):
                    os.unlink(temporary)
            self._serialisable = False
            logger.warning(
                "prompt snapshot store disabled: %s: %s",
                type(error).__name__,
                error,
            )
            return False
        with self._lock:
            previous = self._index.pop(key, None)
            if previous is not None:
                self._nbytes -= previous.nbytes
                self._capacity_bytes -= previous.capacity_charge_bytes
            self._index[key] = _Entry(
                token_tuple, target.name, size, boundary, size
            )
            self._nbytes += size
            self._capacity_bytes += size
            self._index.move_to_end(key)
            self._evict_locked()
            self._persist_index_locked()
        return True

    def _put_write_behind(
        self,
        *,
        key: str,
        token_tuple: tuple[int, ...],
        boundary: int,
        cache: list[Any],
    ) -> bool:
        """Freeze one exact boundary and enqueue it without SSD backpressure.

        Capacity is reserved before copying.  If the detached payload would
        exceed either the byte or count budget, this checkpoint is dropped.
        In particular, a single oversized long-context snapshot never gets a
        special exemption: duplicating multi-gigabyte state on a constrained
        rank would defeat the memory safety the bound exists to provide.
        """

        preparation_error: Exception | None = None
        tensors: dict[str, Any] = {}
        metadata: dict[str, str] = {}
        nbytes = 0
        local_charge: int | None = None
        with self._lock:
            unavailable = self._closed or not self._serialisable
        try:
            if unavailable:
                raise RuntimeError("prompt snapshot store is unavailable")
            tensors, metadata, nbytes = _prepare_snapshot_payload(
                cache,  # type: ignore[arg-type]
                boundary=boundary,
                segment_start=boundary - self.step,
            )
            local_charge = _snapshot_capacity_charge(tensors, metadata, nbytes)
        except Exception as error:
            preparation_error = error

        capacity_charge = self._agree_capacity_charge(local_charge)
        if capacity_charge is None or preparation_error is not None:
            if preparation_error is not None and not unavailable:
                logger.warning(
                    "prompt snapshot store disabled: %s: %s",
                    type(preparation_error).__name__,
                    preparation_error,
                )
            with self._lock:
                self._serialisable = False
            return False

        with self._lock:
            if self._closed or not self._serialisable:
                return False
            if key in self._pending:
                return True
            entry = self._index.get(key)
            if entry is not None and (
                entry.tokens == token_tuple
                or (not entry.tokens and entry.boundary == boundary)
            ):
                # A legacy manifest may carry only physical bytes. Promote it
                # to the newly agreed common charge before future eviction.
                if entry.capacity_charge_bytes != capacity_charge:
                    self._capacity_bytes -= entry.capacity_charge_bytes
                    entry.capacity_charge_bytes = capacity_charge
                    self._capacity_bytes += capacity_charge
                    self._evict_locked()
                    self._persist_index_locked()
                return True
            if capacity_charge > self.max_bytes:
                self._capacity_drops += 1
                return False
            if (
                nbytes > self.pending_max_bytes
                or len(self._pending) >= self.max_pending_writes
                or self._pending_bytes + nbytes > self.pending_max_bytes
            ):
                return False
            self._pending[key] = nbytes
            self._pending_bytes += nbytes
            self._pending_peak_bytes = max(
                self._pending_peak_bytes,
                self._pending_bytes,
            )
            self._capturing.add(key)

        try:
            frozen = _freeze_snapshot_payload(
                tensors,
                metadata,
                nbytes,
                capacity_charge,
            )
        except Exception as error:
            self._release_pending(key, failed=True)
            with self._lock:
                self._serialisable = False
            logger.warning(
                "prompt snapshot store disabled: %s: %s",
                type(error).__name__,
                error,
            )
            return False

        item = (key, token_tuple, boundary, frozen)
        write_queue = self._write_queue
        if write_queue is None:
            self._release_pending(key, failed=True)
            return False
        try:
            write_queue.put_nowait(item)
        except queue.Full:
            # Reservation count and queue capacity normally make this
            # unreachable.  Fail closed instead of falling back to a blocking
            # inline SSD write on the latency-sensitive prefill path.
            self._release_pending(key, failed=True)
            return False
        with self._pending_cond:
            self._capturing.discard(key)
            self._pending_cond.notify_all()
        return True

    def _agree_capacity_charge(self, local_charge: int | None) -> int | None:
        """Return the rank-shared charge, failing closed on disagreement."""

        if self._capacity_agreement is None:
            return local_charge
        try:
            agreed = self._capacity_agreement(local_charge)
        except Exception as error:
            logger.warning("prompt snapshot capacity agreement failed: %s", error)
            return None
        if agreed is None or local_charge is None:
            return None
        try:
            value = int(agreed)
        except (TypeError, ValueError):
            return None
        return value if value >= local_charge > 0 else None

    def _release_pending(self, key: str, *, failed: bool = False) -> None:
        with self._pending_cond:
            nbytes = self._pending.pop(key, None)
            self._capturing.discard(key)
            if nbytes is not None:
                self._pending_bytes = max(0, self._pending_bytes - nbytes)
            if failed:
                self._write_failures += 1
            self._pending_cond.notify_all()

    def _write_frozen_snapshot(
        self,
        key: str,
        token_tuple: tuple[int, ...],
        boundary: int,
        frozen: _FrozenPromptSnapshot,
    ) -> None:
        """Atomically write and index one already detached payload."""

        import mlx.core as mx

        from omlx.utils.metal_sync import _mx_buffer_access_lock

        target = self._path(key)
        # Reserve the common charge before creating the temporary file. Since
        # the charge is an upper bound on the final safetensors bytes, existing
        # snapshots plus the in-progress file stay within the disk ceiling.
        with self._lock:
            self._evict_for_incoming_locked(frozen.capacity_charge_bytes)
            self._persist_index_locked()
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{key}.", suffix=".safetensors", dir=self.directory
        )
        os.close(descriptor)
        try:
            # save_safetensors reads MLX buffers through the Python buffer
            # protocol.  Serialize it against process-wide clear_cache calls,
            # just like the scheduler's asynchronous tiered-cache writer.
            with _mx_buffer_access_lock:
                mx.save_safetensors(
                    temporary,
                    frozen.tensors,
                    frozen.metadata,
                )
            size = os.path.getsize(temporary)
            if size > frozen.capacity_charge_bytes:
                raise ValueError(
                    "prompt snapshot exceeded its pre-agreed capacity charge"
                )
            if size > self.max_bytes:
                with self._lock:
                    self._capacity_drops += 1
                with suppress(OSError):
                    os.unlink(temporary)
                return
            os.replace(temporary, target)
        except Exception:
            with suppress(OSError):
                os.unlink(temporary)
            raise

        with self._lock:
            previous = self._index.pop(key, None)
            if previous is not None:
                self._nbytes -= previous.nbytes
                self._capacity_bytes -= previous.capacity_charge_bytes
            self._index[key] = _Entry(
                token_tuple,
                target.name,
                size,
                boundary,
                frozen.capacity_charge_bytes,
            )
            self._nbytes += size
            self._capacity_bytes += frozen.capacity_charge_bytes
            self._index.move_to_end(key)
            self._evict_locked()
            self._persist_index_locked()

    def _writer_loop(self) -> None:
        write_queue = self._write_queue
        if write_queue is None:
            return
        while True:
            item = write_queue.get()
            if item is None:
                write_queue.task_done()
                return
            key, token_tuple, boundary, frozen = item
            failed = False
            try:
                with self._lock:
                    serialisable = self._serialisable
                if not serialisable:
                    failed = True
                    continue
                self._write_frozen_snapshot(
                    key,
                    token_tuple,
                    boundary,
                    frozen,
                )
            except OSError:
                # Transient disk failures omit only this rank's boundary.  The
                # unanimous restore vote prevents peers from using theirs.
                failed = True
            except Exception as error:
                failed = True
                with self._lock:
                    self._serialisable = False
                logger.warning(
                    "prompt snapshot store disabled: %s: %s",
                    type(error).__name__,
                    error,
                )
            finally:
                self._release_pending(key, failed=failed)
                write_queue.task_done()

    def flush(self, timeout: float = 30.0) -> bool:
        """Boundedly wait until every accepted write has committed or failed.

        A False result means the daemon writer may still own a detached payload;
        callers must not remove its directory.  It can be retried after the
        filesystem recovers, and process exit remains the final fallback for a
        permanently wedged filesystem.
        """

        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._pending_cond:
            while self._pending or self._capturing:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._pending_cond.wait(timeout=remaining)
        return True

    def close(self, timeout: float = 30.0) -> bool:
        """Stop submissions and boundedly drain/join the daemon writer.

        Returns False rather than hanging teardown when storage is wedged.  On
        False the snapshot directory must be left intact because the writer may
        still complete its atomic rename there.
        """

        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._pending_cond:
            self._closed = True
        write_queue = self._write_queue
        writer = self._writer_thread
        if write_queue is None or writer is None:
            return True
        if not self.flush(timeout=max(0.0, deadline - time.monotonic())):
            return False
        with self._lock:
            if not self._stop_enqueued:
                # No pending item remains, so the bounded queue has room and
                # the sentinel cannot overtake a still-capturing submission.
                write_queue.put_nowait(None)
                self._stop_enqueued = True
        writer.join(timeout=max(0.0, deadline - time.monotonic()))
        return not writer.is_alive()

    def clear(self, timeout: float = 30.0) -> int:
        """Drain accepted writes, then atomically forget every snapshot.

        A live distributed cache clear must update both the files and this
        process's in-memory manifest. Deleting files from the coordinator GUI
        alone leaves peer stores advertising stale boundaries and allows a
        later deployment to restore them. Pending write-behind work is drained
        first so it cannot repopulate the directory immediately after clear.
        """

        if not self.flush(timeout=timeout):
            raise TimeoutError("prompt snapshot writes did not drain before clear")
        with self._lock:
            count = len(self._index)
            self._clear_directory()
            self._serialisable = True
            self._persist_index_locked()
            return count

    def present_boundaries(self, model: Any, tokens: list[int]) -> tuple[int, ...]:
        """Boundaries whose whole chain is on disk, longest first.

        A missing or evicted interior file ends the chain there, so a failed
        write simply removes the deeper boundaries from this rank's vote.
        """

        token_tuple = tuple(int(t) for t in tokens)
        found: list[int] = []
        with self._lock:
            for position, key in enumerate(self._chain_keys(model, token_tuple)):
                boundary = (position + 1) * self.step
                entry = self._index.get(key)
                if (
                    entry is None
                    or (
                        entry.tokens != token_tuple[:boundary]
                        and not (
                            not entry.tokens and entry.boundary == boundary
                        )
                    )
                    or not self._path(key).is_file()
                ):
                    break
                found.append(boundary)
        return tuple(reversed(found))

    def load(self, model: Any, tokens: list[int], boundary: int) -> list[Any] | None:
        """Assemble the cache for ``tokens[:boundary]`` from its chain."""

        from mlx_lm.models.cache import load_prompt_cache

        token_tuple = tuple(int(t) for t in tokens)
        if boundary <= 0 or boundary % self.step != 0 or boundary > len(token_tuple):
            return None
        chain = self._chain_keys(model, token_tuple[:boundary])
        with self._lock:
            for position, key in enumerate(chain):
                entry = self._index.get(key)
                prefix = token_tuple[: (position + 1) * self.step]
                expected_boundary = (position + 1) * self.step
                if entry is None or (
                    entry.tokens != prefix
                    and not (
                        not entry.tokens and entry.boundary == expected_boundary
                    )
                ):
                    return None
                if not self._path(key).is_file():
                    self._index.pop(key, None)
                    self._nbytes -= entry.nbytes
                    self._capacity_bytes -= entry.capacity_charge_bytes
                    self._persist_index_locked()
                    return None
        try:
            files = [load_prompt_cache(str(self._path(key))) for key in chain]
            assembled = _assemble_chain(files, boundary)
        except Exception as error:
            logger.info("Rejecting corrupt prompt snapshot chain: %s", error)
            with self._lock:
                self._discard_chain_locked(chain)
            return None
        if assembled is None:
            # A syntactically valid safetensors file can still carry a gap,
            # overlap, mixed schema or corrupt absolute PoolingCache range.
            # Remove the rejected chain from the durable index so duplicate
            # detection cannot keep returning a poisoned boundary forever.
            with self._lock:
                self._discard_chain_locked(chain)
            return None
        with self._lock:
            for key in chain:
                if key in self._index:
                    self._index.move_to_end(key)
            self._persist_index_locked()
        return assembled

    def _discard_chain_locked(self, keys: list[str]) -> None:
        """Forget an invalid chain atomically, including shared manifest rows."""

        changed = False
        for key in keys:
            entry = self._index.pop(key, None)
            if entry is None:
                continue
            self._nbytes -= entry.nbytes
            self._capacity_bytes -= entry.capacity_charge_bytes
            with suppress(OSError):
                self._path(key).unlink()
            changed = True
        if changed:
            self._persist_index_locked()

    def _evict_locked(self) -> None:
        while len(self._index) > self.max_entries or (
            self._capacity_bytes > self.max_bytes or self._nbytes > self.max_bytes
        ):
            self._evict_oldest_locked()

    def _evict_for_incoming_locked(self, capacity_charge_bytes: int) -> None:
        while (
            self._index
            and self._capacity_bytes + capacity_charge_bytes > self.max_bytes
        ):
            self._evict_oldest_locked()

    def _evict_oldest_locked(self) -> None:
        key, entry = self._index.popitem(last=False)
        self._nbytes -= entry.nbytes
        self._capacity_bytes -= entry.capacity_charge_bytes
        self._evictions += 1
        with suppress(OSError):
            self._path(key).unlink()


def _assemble_pooling_delta_chain(
    members: list[Any], expected_source_ends: list[int]
) -> Any | None:
    """Validate and rebuild one PoolingCache absolute-range delta chain.

    A run of legacy cumulative PoolingCache snapshots may precede the first
    versioned delta after an in-place upgrade.  It is accepted as a base only
    when its ratio, absolute pooled length and tensor schema agree with the new
    records.  A cumulative snapshot after a delta, or any other schema mix,
    fails closed rather than guessing which state is authoritative.
    """

    import mlx.core as mx

    from omlx.patches.deepseek_v4.cache_extras import PoolingCache

    delta_positions = [
        position
        for position, member in enumerate(members)
        if isinstance(member, PoolingCacheDeltaSnapshot)
    ]
    if not delta_positions:
        return members[-1]
    first_delta = delta_positions[0]
    if any(
        not isinstance(member, PoolingCache) for member in members[:first_delta]
    ) or any(
        not isinstance(member, PoolingCacheDeltaSnapshot)
        for member in members[first_delta:]
    ):
        return None

    ratio: int | None = None
    concrete_pool_schema: str | None = None
    base_pooled = None
    pool_cursor = 0
    previous_source_end = 0

    # Cumulative V1/V2 snapshots are still readable and can seed a chain
    # written by the new binary after a persistent rank restart.
    for position, member in enumerate(members[:first_delta]):
        try:
            member_ratio = int(member.ratio)
            state = tuple(member.state)
            _validate_pooling_state_layout(state)
            expected_end = int(expected_source_ends[position])
            expected_pool_end = expected_end // member_ratio
            pooled = state[2]
            pooled_length = 0 if pooled is None else int(pooled.shape[1])
            schema = _pool_schema(pooled)
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        if member_ratio <= 0 or pooled_length != expected_pool_end:
            return None
        if ratio is None:
            ratio = member_ratio
        elif member_ratio != ratio:
            return None
        if schema != "none":
            if concrete_pool_schema is None:
                concrete_pool_schema = schema
            elif schema != concrete_pool_schema:
                return None
        base_pooled = pooled
        pool_cursor = pooled_length
        previous_source_end = expected_end

    pooled_parts = [] if base_pooled is None else [base_pooled]
    final_state: tuple[Any, ...] | None = None
    state_arity: int | None = None
    for position in range(first_delta, len(members)):
        member = members[position]
        expected_source_end = int(expected_source_ends[position])
        expected_source_start = (
            0 if position == 0 else int(expected_source_ends[position - 1])
        )
        try:
            state = tuple(member._wire_state)
            _validate_pooling_state_layout(state)
            observed_schema = _pool_schema(state[2])
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        if (
            member.source_start != expected_source_start
            or member.source_end != expected_source_end
            or member.source_start != previous_source_end
            or member.pool_start != member.source_start // member.ratio
            or member.pool_end != member.source_end // member.ratio
            or member.pool_start != pool_cursor
        ):
            return None
        if ratio is None:
            ratio = member.ratio
        elif member.ratio != ratio:
            return None
        if state_arity is None:
            state_arity = member.state_arity
        elif member.state_arity != state_arity:
            return None
        if member.state_arity != len(state) or member.pool_schema != observed_schema:
            return None
        if observed_schema != "none":
            if concrete_pool_schema is None:
                concrete_pool_schema = observed_schema
            elif observed_schema != concrete_pool_schema:
                return None
        elif member.pool_end != 0:
            # Once any pooled row exists, even a zero-row delta retains its
            # typed empty slice. None would be an incompatible/corrupt layout.
            return None

        delta_length = member.pool_end - member.pool_start
        pooled_delta = state[2]
        if pooled_delta is None:
            if delta_length != 0 or member.pool_schema != "none":
                return None
        else:
            if int(pooled_delta.shape[1]) != delta_length:
                return None
            if delta_length:
                pooled_parts.append(pooled_delta)
        pool_cursor = member.pool_end
        previous_source_end = member.source_end
        final_state = state

    if ratio is None or final_state is None:
        return None
    if pool_cursor != int(expected_source_ends[-1]) // ratio:
        return None
    if pooled_parts:
        try:
            pooled = (
                pooled_parts[0]
                if len(pooled_parts) == 1
                else mx.concatenate(pooled_parts, axis=1)
            )
        except Exception:
            return None
        if int(pooled.shape[1]) != pool_cursor:
            return None
    else:
        # Before the first completed pool window, preserve whether the live
        # cache represented its pooled slot as None or as a typed empty array.
        pooled = final_state[2]

    rebuilt_state = list(final_state)
    rebuilt_state[2] = pooled
    try:
        cache = PoolingCache(ratio)
        cache.state = tuple(rebuilt_state)
    except Exception:
        return None
    return cache


def _assemble_chain(files: list[list[Any]], boundary: int) -> list[Any] | None:
    """Stitch one restorable cache list out of a chain of boundary files.

    Members tagged as segments concatenate across the chain; every other
    member is whatever the deepest file holds, because a non-sliceable state
    at boundary B already is the state for the whole prefix.
    """

    import mlx.core as mx
    from mlx_lm.models.cache import CacheList, KVCache

    if not files or boundary <= 0 or boundary % len(files) != 0:
        return None
    step = boundary // len(files)
    expected_source_ends = [(position + 1) * step for position in range(len(files))]

    def stitch(members: list[Any]) -> Any | None:
        deepest = members[-1]
        if any(isinstance(entry, CacheList) for entry in members):
            if not all(isinstance(entry, CacheList) for entry in members):
                return None
            if any(
                len(entry.caches) != len(deepest.caches) for entry in members
            ):
                return None
            rebuilt = []
            for position in range(len(deepest.caches)):
                member = stitch([entry.caches[position] for entry in members])
                if member is None:
                    return None
                rebuilt.append(member)
            return CacheList(*rebuilt)
        if hasattr(deepest, "_omlx_segment_start"):
            if not all(hasattr(member, "_omlx_segment_start") for member in members):
                return None
            slabs = [member.state for member in members]
            keys = mx.concatenate([keys for keys, _ in slabs], axis=2)
            values = mx.concatenate([values for _, values in slabs], axis=2)
            if keys.shape[2] != boundary:
                return None
            cache = KVCache()
            cache.keys = keys
            cache.values = values
            cache.offset = boundary
            return cache
        if any(isinstance(member, PoolingCacheDeltaSnapshot) for member in members):
            return _assemble_pooling_delta_chain(members, expected_source_ends)
        return deepest

    length = len(files[-1])
    if any(len(entry) != length for entry in files):
        return None
    assembled = []
    for position in range(length):
        member = stitch([entry[position] for entry in files])
        if member is None:
            return None
        assembled.append(member)
    return assembled
