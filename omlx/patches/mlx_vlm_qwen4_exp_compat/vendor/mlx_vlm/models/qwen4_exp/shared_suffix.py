# SPDX-License-Identifier: Apache-2.0
"""Opt-in QSA storage: immutable prefix and request-owned suffixes.

Port of the storage contract in mlx2 at de836806961e561f2fd7fd848590d7b2e3a44f44,
src/mlx2/runtime/qsa_shared_suffix.py (Apache-2.0), adapted to oMLX's position
ledger and pooled-indexer API. Serving activation is model-owned and default
off; only persistence and unsupported attention contracts materialize full KV.
"""

import copy
from dataclasses import dataclass

import mlx.core as mx
from mlx.utils import tree_flatten

from .language import BatchQSAKVCache, QSAKVCache


def ordinary_text_positions(positions, expected):
    """Prove the native 2-D or three-axis text representation is ordinary."""
    if positions is None:
        return True
    if positions.ndim == 3 and positions.shape[0] == 3:
        expected = mx.broadcast_to(expected[None], (3, *expected.shape))
    return (
        positions.shape == expected.shape and mx.array_equal(positions, expected).item()
    )


@dataclass(frozen=True)
class SharedQSAPrefix:
    keys: mx.array
    values: mx.array
    raw_keys: mx.array
    positions: mx.array
    pooled_keys: mx.array
    length: int
    ratio: int
    owner: object
    retained_tensor_bytes: int
    attention_owner: object = None
    weight_revision: tuple = ()
    lineage: object = None
    cache_scope: object = None

    @classmethod
    def freeze(cls, cache, indexer, *, length=None, attention=None):
        from .language import QSAKVCache

        if type(cache) is not QSAKVCache:
            raise TypeError("a shared prefix requires a plain QSA cache")
        length = int(cache.offset) if length is None else int(length)
        ratio = int(indexer.compress_ratio)
        if length > cache.offset:
            raise ValueError("shared prefix exceeds source lineage")
        if length <= 0 or length % ratio:
            raise ValueError("shared prefix must contain aligned complete blocks")
        if cache.keys.shape[0] != 1 or not ordinary_text_positions(
            cache.index_position_ids,
            mx.arange(cache.offset, dtype=mx.int32)[None],
        ):
            raise ValueError("shared prefix requires ordinary singleton text positions")
        if cache.index_keys.shape[1] != cache.offset:
            raise ValueError("shared prefix indexer and KV are misaligned")
        pooled = cache.pooled_indexer_keys(
            ratio, indexer.k_layernorm, indexer._apply_rope, cache_tag=indexer
        )
        return cls(
            cache.keys[..., :length, :],
            cache.values[..., :length, :],
            cache.index_keys[:, :length],
            cache.index_position_ids[..., :length],
            pooled[:, : length // ratio],
            length,
            ratio,
            indexer,
            cache.nbytes,
            attention,
            weight_revision(attention) if attention is not None else (),
            object(),
            getattr(cache, "_shared_cache_scope", None),
        )

    @property
    def nbytes(self):
        # Prefix views retain the source's larger allocation slabs. Count
        # those full tensor capacities rather than only the logical views.
        return self.retained_tensor_bytes


class SharedSuffixQSAKVCache:
    """Request-owned writable suffix; no implicit full-prefix materialization."""

    _omlx_qsa_shared_suffix = True
    _omlx_mtp_verify_attention_cache = True
    _omlx_mtp_batched_head_cache = True
    preserve_auxiliary_kv_state = True

    def __init__(self, base: SharedQSAPrefix):
        from .language import QSAKVCache

        self.base = base
        self.suffix = QSAKVCache()
        # The stock raw ledger reserves for long prefills. A private decode
        # suffix grows in small slabs, without re-reserving the prefix width.
        self.suffix.index_step = 256
        self.suffix.step = 256
        self.gathered_tokens = 0

    @property
    def offset(self):
        return self.base.length + self.suffix.offset

    @property
    def private_nbytes(self):
        return self.suffix.nbytes

    @property
    def state(self):
        # Evaluation state only. Persistence handlers explicitly materialize.
        return (
            self.base.keys,
            self.base.values,
            self.base.raw_keys,
            self.base.positions,
            self.base.pooled_keys,
            self.suffix.state,
        )

    @property
    def nbytes(self):
        return self.base.nbytes + self.private_nbytes

    def size(self):
        return self.offset

    def empty(self):
        return False

    def is_trimmable(self):
        return True

    def make_mask(self, length, **kwargs):
        return "causal" if length > 1 else None

    def to_batch(self, left_padding):
        if list(left_padding) != [0]:
            raise ValueError("warm shared QSA cache cannot acquire left padding")
        return SharedBatchQSAKVCache([self.clone()])

    def bind_cache_scope(self, scope):
        if self.base.cache_scope != scope:
            raise ValueError("shared QSA cache scope cannot be rebound")

    def clone(self):
        row = type(self)(self.base)
        row.suffix = _clone_plain(self.suffix)
        return row

    def __deepcopy__(self, memo):
        return self.clone()

    def extract(self, index):
        if index != 0:
            raise IndexError(index)
        return self.clone()

    def filter(self, indices):
        if list(indices) != [0]:
            raise ValueError("singleton cache cancellation must remove its batch row")

    @classmethod
    def merge(cls, caches):
        return SharedBatchQSAKVCache.merge(caches)

    def update_and_fetch(self, keys, values):
        raise RuntimeError("shared QSA requires its split-storage attention arm")

    def append(self, raw_keys, positions, keys, values):
        if keys.shape[0] != 1 or positions.ndim not in (2, 3):
            raise ValueError("shared QSA suffix requires singleton text positions")
        self.suffix.update_indexer(raw_keys, positions)
        self.suffix.update_and_fetch(keys, values)

    def pooled_keys(self, indexer):
        if indexer is not self.base.owner or indexer.compress_ratio != self.base.ratio:
            raise ValueError("shared QSA pooled prefix belongs to another indexer")
        if self.suffix.offset < self.base.ratio:
            return self.base.pooled_keys
        pooled = self.suffix.pooled_indexer_keys(
            self.base.ratio, indexer.k_layernorm, indexer._apply_rope, cache_tag=indexer
        )
        # Summaries are small. Full raw ledgers and KV remain split.
        return mx.concatenate([self.base.pooled_keys, pooled], axis=1)

    def gather(self, logical_tokens):
        flat = logical_tokens.reshape(1, 1, -1, 1)
        from_base = flat < self.base.length
        base_indices = mx.minimum(flat, self.base.length - 1)
        suffix_indices = mx.clip(flat - self.base.length, 0, self.suffix.offset - 1)
        parts = []
        for base, private in (
            (self.base.keys, self.suffix.keys),
            (self.base.values, self.suffix.values),
        ):
            parts.append(
                mx.where(
                    from_base,
                    mx.take_along_axis(base, base_indices, axis=2),
                    mx.take_along_axis(private, suffix_indices, axis=2),
                )
            )
        self.gathered_tokens = int(logical_tokens.shape[-1])
        return tuple(parts)

    def trim(self, n):
        if type(n) is not int or n < 0 or n > self.suffix.offset:
            raise ValueError("rollback cannot trim an immutable shared prefix")
        return self.suffix.trim(n)

    def materialize(self):
        from .language import QSAKVCache

        cache = QSAKVCache()
        keys = (
            mx.concatenate(
                [self.base.keys, self.suffix.keys[..., : self.suffix.offset, :]], axis=2
            )
            if self.suffix.offset
            else self.base.keys
        )
        values = (
            mx.concatenate(
                [self.base.values, self.suffix.values[..., : self.suffix.offset, :]],
                axis=2,
            )
            if self.suffix.offset
            else self.base.values
        )
        raw = (
            mx.concatenate([self.base.raw_keys, self.suffix.index_keys], axis=1)
            if self.suffix.offset
            else self.base.raw_keys
        )
        base_positions = self.base.positions
        suffix_positions = self.suffix.index_position_ids
        if self.suffix.offset and base_positions.ndim != suffix_positions.ndim:
            if base_positions.ndim == 2:
                base_positions = mx.broadcast_to(
                    base_positions[None], (3, *base_positions.shape)
                )
            else:
                suffix_positions = mx.broadcast_to(
                    suffix_positions[None], (3, *suffix_positions.shape)
                )
        positions = (
            mx.concatenate([base_positions, suffix_positions], axis=-1)
            if self.suffix.offset
            else self.base.positions
        )
        cache.state = keys, values, raw, positions
        if self.base.attention_owner is not None:
            cache._shared_loaded_owner = (
                self.base.attention_owner,
                self.base.weight_revision,
            )
        cache._shared_cache_scope = self.base.cache_scope
        return cache


def shared_suffix_attention(attention, x, cache, *, target_verify=False):
    """Singleton uses the same projection and selector as the serving batch."""
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1:
        raise ValueError("shared QSA requires one text request")
    base = cache.base
    if base.owner is not attention.indexer or (
        base.attention_owner is not None
        and (
            base.attention_owner is not attention
            or base.weight_revision != weight_revision(attention)
        )
    ):
        raise ValueError("shared QSA prefix belongs to another layer/revision")
    batch = SharedBatchQSAKVCache([cache])
    return batch._batched_attention(attention, x, target_verify=target_verify)


def weight_revision(attention):
    """Loaded module/parameter identity, never imported from persisted state."""
    return tuple(
        (name, id(value)) for name, value in tree_flatten(attention.parameters())
    )


def bind_loaded_owner(cache, attention):
    revision = (attention, weight_revision(attention))
    old = getattr(cache, "_shared_loaded_owner", None)
    if old is not None and (old[0] is not attention or old[1] != revision[1]):
        raise ValueError("QSA cache belongs to another loaded layer/revision")
    cache._shared_loaded_owner = revision


def _clone_plain(cache):
    from .language import QSAKVCache

    out = QSAKVCache()
    # New array wrappers retain the full capacity allocations without aliasing
    # the mutable Python object's graph binding. Slice assignment to a donor
    # must not rebind the clone's bank (or vice versa).
    out.__dict__ = {
        name: mx.array(value) if isinstance(value, mx.array) else value
        for name, value in cache.__dict__.items()
    }
    return out


class SharedBatchQSAKVCache(BatchQSAKVCache):
    """Model-owned row cache ABI; row attention keeps long prefixes unjoined.

    ``state`` is an evaluation graph, not a serialized layout. Registered
    handlers call ``materialize`` explicitly for persistence. Projection,
    selection and selected-KV attention retain the ordinary batch geometry;
    only storage and bounded gathering operate per row.
    """

    _omlx_qsa_shared_suffix = True
    def __init__(self, rows):
        self.rows = list(rows)
        self._pending_right_padding = None
        self._left_padding = [0] * len(rows)
        self._fallback = None
        self._fallback_metadata = []
        self._loaded_owner = None
        self.metrics = dict(
            shared_attention_calls=0,
            shared_rows=0,
            prefix_comparisons=0,
            prefix_comparison_tokens=0,
            prefix_reuses=0,
            materializations=0,
            rotary_comparisons=0,
        )

    def __deepcopy__(self, memo):
        out = type(self)([self._clone(row) for row in self.rows])
        out._left_padding = list(self._left_padding)
        out._pending_right_padding = copy.deepcopy(self._pending_right_padding)
        out._fallback = copy.deepcopy(self._fallback, memo)
        out._fallback_metadata = [
            dict(metadata) for metadata in self._fallback_metadata
        ]
        out._loaded_owner = self._loaded_owner
        out.metrics = dict(self.metrics)
        return out

    @staticmethod
    def _clone(row):
        return (
            row.clone()
            if isinstance(row, SharedSuffixQSAKVCache)
            else _clone_plain(row)
        )

    @classmethod
    def merge(cls, caches):
        if any(
            getattr(cache, "_pending_right_padding", None) is not None
            for cache in caches
        ):
            raise ValueError(
                "cannot join shared QSA caches during an unfinished transaction"
            )
        rows = []
        for cache in caches:
            if isinstance(cache, cls):
                if cache._fallback is not None:
                    rows.extend(cache.extract(i) for i in range(len(cache.offset)))
                else:
                    rows.extend(cls._clone(row) for row in cache.rows)
            elif isinstance(cache, SharedSuffixQSAKVCache):
                rows.append(cache.clone())
            elif isinstance(cache, BatchQSAKVCache):
                rows.extend(cache.extract(i) for i in range(len(cache.offset)))
            elif type(cache) is QSAKVCache:
                rows.append(_clone_plain(cache))
            else:
                raise TypeError(f"Cannot merge shared QSA cache with {type(cache)}")
        return cls(rows)

    @property
    def offset(self):
        if self._fallback is not None:
            return self._fallback.offset
        return mx.array(
            [row.offset - pad for row, pad in zip(self.rows, self._left_padding)],
            dtype=mx.int32,
        )

    @property
    def _idx(self):
        return (
            self._fallback._idx
            if self._fallback is not None
            else max((row.offset for row in self.rows), default=0)
        )

    @property
    def left_padding(self):
        if self._fallback is not None:
            return self._fallback.left_padding
        # Every row owns an unpadded cache. Logical batch padding is metadata.
        return mx.array(
            [self._idx - int(offset) for offset in self.offset.tolist()], dtype=mx.int32
        )

    @property
    def keys(self):
        # Rollback dispatch only asks whether live KV exists. Full tensor
        # materialization is deliberately not available through this property.
        if self._fallback is not None:
            return self._fallback.keys
        for row in self.rows:
            if isinstance(row, SharedSuffixQSAKVCache):
                return row.base.keys
            if row.keys is not None:
                return row.keys
        return None

    @property
    def values(self):
        if self._fallback is not None:
            return self._fallback.values
        for row in self.rows:
            if isinstance(row, SharedSuffixQSAKVCache):
                return row.base.values
            if row.values is not None:
                return row.values
        return None

    @property
    def state(self):
        if self._fallback is not None:
            return self._fallback.state
        bases = {
            id(row.base): row.base
            for row in self.rows
            if isinstance(row, SharedSuffixQSAKVCache)
        }
        return tuple(
            (base.keys, base.values, base.raw_keys, base.positions, base.pooled_keys)
            for base in bases.values()
        ) + tuple(
            row.suffix.state if isinstance(row, SharedSuffixQSAKVCache) else row.state
            for row in self.rows
        )

    @property
    def nbytes(self):
        if self._fallback is not None:
            return self._fallback.nbytes
        bases = {
            id(row.base): row.base
            for row in self.rows
            if isinstance(row, SharedSuffixQSAKVCache)
        }
        return sum(base.nbytes for base in bases.values()) + sum(
            (
                row.private_nbytes
                if isinstance(row, SharedSuffixQSAKVCache)
                else row.nbytes
            )
            for row in self.rows
        )

    def shared_prefix_metrics(self):
        return {
            **self.metrics,
            "current_rows": len(self.rows),
            "current_shared_rows": sum(
                isinstance(row, SharedSuffixQSAKVCache) for row in self.rows
            ),
            "retained_tensor_bytes": self.nbytes,
            "fallback": self._fallback is not None,
        }

    def materialize(self):
        self.metrics["materializations"] += 1
        if self._fallback is not None:
            return self._fallback
        rows = [
            row.materialize() if isinstance(row, SharedSuffixQSAKVCache) else row
            for row in self.rows
        ]
        # Calling the ordinary batch class explicitly bypasses opt-in merge.
        return BatchQSAKVCache.merge(rows)

    def fallback(self):
        if self._fallback is None:
            self._fallback_metadata = [self._row_metadata(row) for row in self.rows]
            self._fallback = self.materialize()
            if any(self._left_padding):
                self._fallback.prepare(left_padding=self._left_padding)
            if self._pending_right_padding is not None:
                self._fallback.prepare(right_padding=self._pending_right_padding)
            self.rows.clear()
        return self._fallback

    @staticmethod
    def _row_metadata(row):
        if isinstance(row, SharedSuffixQSAKVCache):
            metadata = {
                "_shared_cache_scope": row.base.cache_scope,
                "_shared_serving": True,
            }
            if row.base.attention_owner is not None:
                metadata["_shared_loaded_owner"] = (
                    row.base.attention_owner,
                    row.base.weight_revision,
                )
            return metadata
        return {
            name: getattr(row, name)
            for name in (
                "_shared_cache_scope",
                "_shared_loaded_owner",
                "_shared_serving",
            )
            if hasattr(row, name)
        }

    def filter(self, indices):
        if self._fallback is not None:
            indices = [int(i) for i in indices]
            if any(i < 0 or i >= len(self._fallback_metadata) for i in indices) or len(
                set(indices)
            ) != len(indices):
                raise ValueError("invalid shared QSA row selection")
            self._fallback_metadata = [self._fallback_metadata[int(i)] for i in indices]
            if self._pending_right_padding is not None:
                self._pending_right_padding = [
                    self._pending_right_padding[int(i)] for i in indices
                ]
            if not indices:
                self._fallback = None
                self._fallback_metadata = []
                self._pending_right_padding = None
                self._left_padding = []
                return
            self._fallback.filter(indices)
            if self._pending_right_padding is not None:
                self._fallback.prepare(right_padding=self._pending_right_padding)
            return
        indices = [int(i) for i in indices]
        if any(i < 0 or i >= len(self.rows) for i in indices) or len(
            set(indices)
        ) != len(indices):
            raise ValueError("invalid shared QSA row selection")
        self.rows = [self.rows[i] for i in indices]
        self._left_padding = [self._left_padding[i] for i in indices]
        if self._pending_right_padding is not None:
            self._pending_right_padding = [
                self._pending_right_padding[i] for i in indices
            ]

    def extract(self, index):
        if self._fallback is None:
            return self._clone(self.rows[index])
        row = self._fallback.extract(index)
        row.__dict__.update(self._fallback_metadata[index])
        return row

    def extend(self, other):
        if (
            self._pending_right_padding is not None
            or getattr(other, "_pending_right_padding", None) is not None
        ):
            raise ValueError(
                "cannot join shared QSA caches during an unfinished transaction"
            )
        if self._fallback is not None:
            donor = (
                other if isinstance(other, type(self)) else type(self).merge([other])
            )
            rows = [donor.extract(i) for i in range(len(donor.offset))]
            if self._loaded_owner is not None:
                for row in rows:
                    owner = self._row_metadata(row).get("_shared_loaded_owner")
                    if owner is not None and (
                        owner[0] is not self._loaded_owner[0]
                        or owner[1] != self._loaded_owner[1]
                    ):
                        raise ValueError(
                            "QSA cache belongs to another loaded layer/revision"
                        )
            self._fallback.extend(BatchQSAKVCache.merge(rows))
            self._fallback_metadata.extend(self._row_metadata(row) for row in rows)
            return
        if not isinstance(other, type(self)):
            other = type(self).merge([other])
        if other._fallback is not None:
            self.fallback()
            self.extend(other)
            return
        if (
            self._pending_right_padding is not None
            or other._pending_right_padding is not None
        ):
            raise ValueError(
                "cannot join shared QSA caches during an unfinished transaction"
            )
        self.rows.extend(self._clone(row) for row in other.rows)
        self._left_padding.extend(other._left_padding)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None and (
            len(left_padding) != len(self.offset)
            or any(int(pad) < 0 for pad in left_padding)
            or any(self.offset.tolist())
        ):
            raise ValueError("left padding can only initialize empty QSA rows")
        if right_padding is not None:
            pads = [int(pad) for pad in right_padding]
            if len(pads) != len(self.offset) or any(pad < 0 for pad in pads):
                raise ValueError("invalid shared QSA rollback padding")
        if self._fallback is not None:
            if right_padding is not None:
                self._pending_right_padding = pads
            return self._fallback.prepare(
                left_padding=left_padding, lengths=lengths, right_padding=right_padding
            )
        if left_padding is not None:
            if len(left_padding) != len(self.rows) or any(
                row.offset for row in self.rows
            ):
                raise ValueError("left padding can only initialize empty QSA rows")
            self._left_padding = [int(pad) for pad in left_padding]
        if right_padding is not None:
            pads = [int(pad) for pad in right_padding]
            if len(pads) != len(self.rows) or any(pad < 0 for pad in pads):
                raise ValueError("invalid shared QSA rollback padding")
            self._pending_right_padding = pads

    def finalize(self):
        if self._fallback is not None:
            if self._pending_right_padding is not None and any(
                pad > offset
                for pad, offset in zip(
                    self._pending_right_padding, self.offset.tolist()
                )
            ):
                raise ValueError("rollback exceeds materialized QSA history")
            result = self._fallback.finalize()
            self._pending_right_padding = None
            return result
        pads = self._pending_right_padding
        if pads is None:
            return
        # Validate every row before mutating any; never trim an immutable base.
        for row, pad in zip(self.rows, pads):
            available = (
                row.suffix.offset
                if isinstance(row, SharedSuffixQSAKVCache)
                else row.offset
            )
            if pad > available:
                raise ValueError("rollback cannot trim an immutable shared prefix")
        for row, pad in zip(self.rows, pads):
            row.trim(pad)
        self._pending_right_padding = None

    def make_mask(self, length, **kwargs):
        return (
            self._fallback.make_mask(length, **kwargs)
            if self._fallback is not None
            else ("causal" if length > 1 else None)
        )

    def size(self):
        return self._idx

    def empty(self):
        return self._idx == 0

    def is_trimmable(self):
        return True

    def trim(self, count):
        if type(count) is not int or count < 0:
            raise ValueError("invalid shared QSA trim")
        if self._fallback is not None:
            if any(count > offset for offset in self.offset.tolist()):
                raise ValueError("rollback exceeds materialized QSA history")
            return self._fallback.trim(count)
        for row in self.rows:
            available = (
                row.suffix.offset
                if isinstance(row, SharedSuffixQSAKVCache)
                else row.offset
            )
            if count > available:
                raise ValueError("rollback cannot trim an immutable shared prefix")
        for row in self.rows:
            row.trim(count)
        return count

    def bind_cache_scope(self, scope):
        if self._fallback is not None:
            if len(self.offset) != 1:
                raise ValueError("request cache scope requires one row")
            row = self.extract(0)
            row.bind_cache_scope(scope)
            self._fallback_metadata[0] = self._row_metadata(row)
            return
        if len(self.rows) != 1:
            raise ValueError("request cache scope requires one row")
        self.rows[0].bind_cache_scope(scope)

    def _activate_rows(self, attention, position_ids=None):
        candidates = [
            row.base for row in self.rows if isinstance(row, SharedSuffixQSAKVCache)
        ]
        # A pending right-padded prefill can include provisional tokens from
        # earlier chunks. Freeze only after its transaction is finalized.
        if self._pending_right_padding is not None:
            return
        for i in sorted(range(len(self.rows)), key=lambda i: self.rows[i].offset):
            row = self.rows[i]
            if isinstance(row, SharedSuffixQSAKVCache):
                continue
            bind_loaded_owner(row, attention)
            length = (
                row.offset
                // attention.indexer.compress_ratio
                * attention.indexer.compress_ratio
            )
            if (
                length <= 0
                or row.index_position_ids is None
                or not ordinary_text_positions(
                    row.index_position_ids,
                    mx.arange(row.offset, dtype=mx.int32)[None],
                )
            ):
                continue
            if (
                position_ids is not None
                and position_ids.ndim == 3
                and row.index_position_ids.ndim == 2
            ):
                backing = row._index_position_ids
                row.index_position_ids = mx.broadcast_to(
                    backing[None], (3, *backing.shape)
                )
            base = SharedQSAPrefix.freeze(
                row, attention.indexer, length=length, attention=attention
            )
            scope = getattr(row, "_shared_cache_scope", None)
            for candidate in candidates:
                if (
                    candidate.attention_owner is not attention
                    or candidate.weight_revision != base.weight_revision
                    or candidate.length > length
                    or getattr(candidate, "cache_scope", None) != scope
                ):
                    continue
                self.metrics["prefix_comparisons"] += 1
                compare_length = candidate.length
                self.metrics["prefix_comparison_tokens"] += compare_length
                if all(
                    mx.array_equal(a, b).item()
                    for a, b in zip(
                        (
                            candidate.keys,
                            candidate.values,
                            candidate.raw_keys,
                            candidate.positions,
                        ),
                        (
                            base.keys[..., :compare_length, :],
                            base.values[..., :compare_length, :],
                            base.raw_keys[:, :compare_length],
                            base.positions[..., :compare_length],
                        ),
                    )
                ):
                    base = candidate
                    length = candidate.length
                    self.metrics["prefix_reuses"] += 1
                    break
            else:
                candidates.append(base)
            private = SharedSuffixQSAKVCache(base)
            if row.offset > length:
                private.suffix.state = (
                    row.keys[..., length : row.offset, :],
                    row.values[..., length : row.offset, :],
                    row.index_keys[:, length : row.offset],
                    row.index_position_ids[..., length : row.offset],
                )
            self.rows[i] = private
            self.metrics["shared_rows"] += 1

    def attend(
        self,
        attention,
        x,
        *,
        mask=None,
        position_ids=None,
        position_embeddings=None,
        target_verify=False,
    ):
        revision = weight_revision(attention)
        loaded_owner = (attention, revision)
        if self._loaded_owner is not None and (
            self._loaded_owner[0] is not attention or self._loaded_owner[1] != revision
        ):
            raise ValueError("QSA cache belongs to another loaded layer/revision")
        if self._fallback is not None:
            # A cache may materialize before its first attend (for example
            # during a join). Validate every retained owner before forwarding.
            for metadata in self._fallback_metadata:
                owner = metadata.get("_shared_loaded_owner")
                if owner is not None and (
                    owner[0] is not attention or owner[1] != revision
                ):
                    raise ValueError(
                        "QSA cache belongs to another loaded layer/revision"
                    )
            for metadata in self._fallback_metadata:
                metadata["_shared_loaded_owner"] = loaded_owner
            self._loaded_owner = loaded_owner
            return attention(
                x,
                mask=mask,
                cache=self._fallback,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                target_verify=target_verify,
            )
        if x.shape[0] != len(self.rows):
            raise ValueError("shared QSA cache row count differs from input")
        offsets = self.offset.tolist()
        ordinary_positions = (
            mx.array(offsets, dtype=mx.int32)[:, None] + mx.arange(x.shape[1])[None]
        )
        ordinary_positions = mx.maximum(ordinary_positions, 0)
        supported_mask = (
            mask is None
            or isinstance(mask, str)
            and mask in ("causal", "left_padded_decode")
        )
        ordinary = ordinary_text_positions(position_ids, ordinary_positions)
        # Validate every row before appending to any, including mixed owners.
        for row in self.rows:
            if isinstance(row, SharedSuffixQSAKVCache):
                base = row.base
                if base.owner is not attention.indexer or (
                    base.attention_owner is not None
                    and (
                        base.attention_owner is not attention
                        or base.weight_revision != revision
                    )
                ):
                    raise ValueError(
                        "shared QSA prefix belongs to another layer/revision"
                    )
            else:
                bind_loaded_owner(row, attention)
        self._loaded_owner = loaded_owner
        # Native text models precompute MRoPE cos/sin. Verify their authority
        # before reusing them; arbitrary caller-supplied rotary values fall back.
        ordinary_rotary = True
        if position_embeddings is not None:
            if ordinary and position_ids is not None:
                expected = attention.rotary_emb(x, position_ids)
                self.metrics["rotary_comparisons"] += 1
                ordinary_rotary = all(
                    a.shape == b.shape and mx.array_equal(a, b).item()
                    for a, b in zip(position_embeddings, expected)
                )
            else:
                ordinary_rotary = False
        if not ordinary or not supported_mask or not ordinary_rotary:
            return attention(
                x,
                mask=mask,
                cache=self.fallback(),
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                target_verify=target_verify,
            )
        # Restored caches deliberately have no serialized owner. The active
        # model binds them before activation; prefix-cache engine identity is
        # the authority for selecting the restored state.
        self._activate_rows(attention, position_ids)
        if any(self._left_padding):
            # Initial left padding uses the established dense cache ABI. Warm
            # row-relative histories themselves carry no physical padding.
            ordinary_cache = self.fallback()
            return attention(
                x,
                mask=mask,
                cache=ordinary_cache,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                target_verify=target_verify,
            )
        return self._batched_attention(
            attention,
            x,
            target_verify=target_verify,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

    def _batched_attention(
        self,
        attention,
        x,
        *,
        target_verify,
        position_ids=None,
        position_embeddings=None,
    ):
        """Preserve batched projection/selection/SDPA arithmetic; split storage."""
        from .language import _target_verify_linear, _target_verify_linears
        from .qsa_fast import batched_causal_block_selection

        batch, length, _ = x.shape
        indexer = attention.indexer
        positions = (
            mx.array([row.offset for row in self.rows], dtype=mx.int32)[:, None]
            + mx.arange(length)[None]
        )
        if position_ids is not None:
            positions = position_ids
        projected = (
            _target_verify_linear(indexer.index_qk_proj, x)
            if target_verify
            else indexer.index_qk_proj(x)
        ).reshape(batch, length, indexer.n_heads + indexer.kv_heads, indexer.head_dim)
        index_queries = indexer._apply_rope(
            indexer.q_layernorm(projected[:, :, : indexer.n_heads]).transpose(
                0, 2, 1, 3
            ),
            positions,
        )
        qkv = (
            _target_verify_linears(
                (attention.q_proj, attention.k_proj, attention.v_proj), x
            )
            if target_verify
            else tuple(
                module(x)
                for module in (attention.q_proj, attention.k_proj, attention.v_proj)
            )
        )
        queries, keys, values, gate, _ = attention._prepare_projected_qkv(
            *qkv, None, positions, position_embeddings, None
        )
        pooled_rows = []
        for i, row in enumerate(self.rows):
            row_positions = (
                positions[:, i : i + 1] if positions.ndim == 3 else positions[i : i + 1]
            )
            raw = projected[i : i + 1, :, indexer.n_heads :].squeeze(2)
            if isinstance(row, SharedSuffixQSAKVCache):
                row.append(raw, row_positions, keys[i : i + 1], values[i : i + 1])
                pooled = row.pooled_keys(indexer)
                self.metrics["shared_attention_calls"] += 1
            else:
                row.update_indexer(raw, row_positions)
                row.update_and_fetch(keys[i : i + 1], values[i : i + 1])
                pooled = row.pooled_indexer_keys(
                    indexer.compress_ratio,
                    indexer.k_layernorm,
                    indexer._apply_rope,
                    cache_tag=indexer,
                )
            pooled_rows.append(pooled)
        blocks = max(pooled.shape[1] for pooled in pooled_rows)
        pooled = mx.concatenate(
            [
                mx.pad(bank, [(0, 0), (0, blocks - bank.shape[1]), (0, 0)])
                for bank in pooled_rows
            ],
            axis=0,
        )
        visible = (positions[0] if positions.ndim == 3 else positions) + 1
        if blocks > indexer.block_topk:
            tokens, valid = batched_causal_block_selection(
                index_queries,
                pooled,
                visible,
                compress_ratio=indexer.compress_ratio,
                block_topk=indexer.block_topk,
                indexer_head_dim=indexer.head_dim,
            )
        else:
            width = max(row.offset for row in self.rows)
            tokens = mx.broadcast_to(
                mx.arange(width)[None, None], (batch, length, width)
            )
            valid = tokens < visible[..., None]
        selected_rows = []
        for i, row in enumerate(self.rows):
            if isinstance(row, SharedSuffixQSAKVCache):
                selected_rows.append(row.gather(tokens[i : i + 1]))
            else:
                gather = mx.clip(tokens[i : i + 1], 0, row.offset - 1).reshape(
                    1, 1, -1, 1
                )
                selected_rows.append(
                    (
                        mx.take_along_axis(row.keys, gather, axis=2),
                        mx.take_along_axis(row.values, gather, axis=2),
                    )
                )
        selected_keys = mx.concatenate([pair[0] for pair in selected_rows], axis=0)
        selected_values = mx.concatenate([pair[1] for pair in selected_rows], axis=0)
        selected = tokens.shape[-1]
        heads, dim = selected_keys.shape[1], selected_keys.shape[-1]
        if length == 1:
            output = mx.fast.scaled_dot_product_attention(
                queries,
                selected_keys,
                selected_values,
                scale=attention.scale,
                mask=valid[:, None],
            )
            output = output.transpose(0, 2, 1, 3)
        else:

            def per_query(rows):
                return (
                    rows.reshape(batch, heads, length, selected, dim)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(batch * length, heads, selected, dim)
                )

            output = mx.fast.scaled_dot_product_attention(
                queries.transpose(0, 2, 1, 3).reshape(batch * length, -1, 1, dim),
                per_query(selected_keys),
                per_query(selected_values),
                scale=attention.scale,
                mask=valid.reshape(batch * length, 1, 1, selected),
            ).reshape(batch, length, -1, dim)
        output = output.reshape(batch, length, -1) * mx.sigmoid(gate)
        return (
            _target_verify_linear(attention.o_proj, output)
            if target_verify
            else attention.o_proj(output)
        )
