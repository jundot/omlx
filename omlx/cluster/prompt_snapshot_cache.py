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
one cumulative copy per boundary), while non-sliceable members (rotating
windows, recurrent slots, pooling state) store their full state at that
boundary, which is the only representation they have and is what makes every
boundary an independent restore point for them.

Each rank keeps its own directory holding its own layer-slice snapshots. The
keys are a hash of ``(model, prefix tokens)`` and are therefore identical across
ranks that processed the same broadcast request, while the bytes under a key are
this rank's shard alone; prompts sharing a prefix share the early chain files.
Eviction is a deterministic bounded LRU keyed on the sequence of operations
rather than a wall clock, so ranks that see the same requests keep identical key
sets without any coordination. Coordinating the *hit* across ranks (so a disk
write that failed on one rank cannot desync the pipeline) is the caller's job
and lives in the telemetry integration, which has the collective; this module
stays pure and unit-testable.

Snapshots are process-lifetime. The digest filenames cannot be re-indexed
without their token tuples, and a file that is not in the index is invisible to
hits yet still holds disk, so a new store starts by clearing its directory and
the telemetry teardown removes it. A rank that restarts simply begins empty,
which the boundary vote already handles.
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
import tempfile
import threading
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Chain files are cheap (one step-sized slab plus constant-size window and
# recurrent states), so the count bound mainly limits how many distinct
# reusable boundaries exist across all prompts; the byte bound is the backstop.
_MAX_ENTRIES_DEFAULT = 512


class PoolingCacheSnapshot:
    """Serialisation stand-in for the DeepSeek sparse-attention pool cache.

    ``save_prompt_cache`` serialises through ``state`` / ``meta_state`` and
    reconstructs via ``globals()[name].from_state`` inside
    ``mlx_lm.models.cache``. A ``PoolingCache`` breaks both directions: its
    state tuple carries ``None`` slots safetensors cannot hold, and its
    meta_state is a bare int where the metadata tree must be all strings. The
    in-process state contract is load-bearing for the paged-cache handlers, so
    rather than changing the class this stand-in presents the same cache in
    wire form (arrays-only state, with the slot layout encoded as a flag
    string), and its ``from_state`` hands back a real ``PoolingCache``, so a
    restored snapshot is indistinguishable from the cache it was taken from.
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

    for snapshot_class in (PoolingCacheSnapshot, EmptyLeafSnapshot, KVCacheSegment):
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


@dataclass
class _Entry:
    tokens: tuple[int, ...]
    filename: str
    nbytes: int


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

    Eviction is a deterministic bounded LRU (entry count, optionally bytes)
    keyed on the sequence of operations rather than a wall clock. Evicting an
    early file orphans the deeper files of its chain; they stop being offered,
    are never touched again, age to the LRU front and fall out on their own.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        step: int = 2048,
        max_entries: int = _MAX_ENTRIES_DEFAULT,
        max_bytes: int | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.step = max(1, int(step))
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        # Access-ordered: most-recently-used at the end. The order is advanced
        # only by put/load, both driven by the identical request stream every
        # rank sees, so eviction is the same decision on every rank.
        self._index: OrderedDict[str, _Entry] = OrderedDict()
        self._nbytes = 0
        # A cache type save_prompt_cache rejects will be rejected on every
        # boundary, so the store disables itself after the first such failure
        # rather than paying a doomed write per request. Known-hostile types
        # get a wire stand-in instead (see ``_wrap_for_save``); this flag is
        # the backstop for a type nobody has taught the store about yet.
        self._serialisable = True
        _register_snapshot_classes()
        self.directory.mkdir(parents=True, exist_ok=True)
        # Snapshots are process-lifetime (see the module docstring): whatever a
        # dead process left behind is unreachable, so reclaim it up front and
        # keep the directory exactly as large as the live index says it is.
        for stale in self.directory.iterdir():
            if stale.is_file():
                with suppress(OSError):
                    stale.unlink()

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._nbytes

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.safetensors"

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

    def put(self, model: Any, tokens: list[int], cache: list[Any]) -> bool:
        """Persist the boundary file for ``tokens``. Best effort.

        ``tokens`` must end on the step grid. Plain KVCache members are stored
        as their newest slab, everything else as full state. A file that
        already exists is this boundary written by an earlier request sharing
        the prefix; it is kept and touched rather than rewritten, which is
        what lets branching prompts share their common chain.
        """

        from mlx_lm.models.cache import save_prompt_cache

        if not self._serialisable:
            return False
        token_tuple = tuple(int(t) for t in tokens)
        boundary = len(token_tuple)
        if boundary == 0 or boundary % self.step != 0:
            return False
        key = self._chain_keys(model, token_tuple)[-1]
        with self._lock:
            entry = self._index.get(key)
            if entry is not None and entry.tokens == token_tuple:
                self._index.move_to_end(key)
                return True
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
            self._index[key] = _Entry(token_tuple, target.name, size)
            self._nbytes += size
            self._index.move_to_end(key)
            self._evict_locked()
        return True

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
                    or entry.tokens != token_tuple[:boundary]
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
                if entry is None or entry.tokens != prefix:
                    return None
                if not self._path(key).is_file():
                    self._index.pop(key, None)
                    self._nbytes -= entry.nbytes
                    return None
        try:
            files = [load_prompt_cache(str(self._path(key))) for key in chain]
            assembled = _assemble_chain(files, boundary)
        except Exception:
            return None
        if assembled is None:
            return None
        with self._lock:
            for key in chain:
                if key in self._index:
                    self._index.move_to_end(key)
        return assembled

    def _evict_locked(self) -> None:
        while len(self._index) > self.max_entries or (
            self.max_bytes is not None and self._nbytes > self.max_bytes
        ):
            key, entry = self._index.popitem(last=False)
            self._nbytes -= entry.nbytes
            with suppress(OSError):
                self._path(key).unlink()


def _assemble_chain(files: list[list[Any]], boundary: int) -> list[Any] | None:
    """Stitch one restorable cache list out of a chain of boundary files.

    Members tagged as segments concatenate across the chain; every other
    member is whatever the deepest file holds, because a non-sliceable state
    at boundary B already is the state for the whole prefix.
    """

    import mlx.core as mx
    from mlx_lm.models.cache import CacheList, KVCache

    def stitch(members: list[Any]) -> Any | None:
        deepest = members[-1]
        if isinstance(deepest, CacheList):
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
