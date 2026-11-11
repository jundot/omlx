# SPDX-License-Identifier: MIT
"""Request-local CSA2 state, including compressor tails and Engram lookback."""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from ...cache.type_handlers import ArraysCacheHandler, CacheType


class DeepseekV41Cache(ArraysCache):
    """Per-layer V4.1 state.

    A batch (B > 1) keeps its requests as a list of single-request caches ("rows") instead
    of padded, concatenated arrays. The model's forward runs every request on its own row
    (Language._forward), so with rows a decode step neither slices the batch apart nor
    re-concatenates it -- which cost B - 1 successive concatenations of every slot in every
    layer per step, a host sync per layer and row to trim the padding, and dropped each
    row's in-place growth buffers. Padding / lengths are mirrored on the host so masks need
    no GPU round trip. The padded dense arrays are still produced on demand (``cache`` /
    ``state``) for code that reads a batch as arrays.
    """

    # offset, window KV, compressed KV, index K, partial KV, partial gates, history.
    # After a CED chunk that skipped its decoder tail, the midpoint layer also holds
    # the decoder replay input in slots 7-10 (see pending()).
    def __init__(self, compress_ratio=0):
        self._rows = None
        self._dense = None
        self._lp = None    # host mirror of left_padding (row-backed batches)
        self._len = None   # host mirror of lengths (row-backed batches)
        super().__init__(7)
        self.compress_ratio = compress_ratio

    # -- storage: dense arrays or rows ------------------------------------------------
    @property
    def cache(self):
        if self._rows is not None and self._dense is None:
            self._dense = self._materialize()
        return self._dense

    @cache.setter
    def cache(self, value):
        self._rows = None
        self._dense = value

    def __setitem__(self, idx, value):
        if self._rows is not None:
            if any(row.has_pending() for row in self._rows) and len(self._rows) > 1:
                # The dense layout has no per-row replay slots.
                raise ValueError("A pending V4.1 decoder replay cannot be batched")
            dense = self.cache
            self._rows = None
            self._dense = dense
        self._dense[idx] = value

    def _materialize(self):
        """Padded dense arrays of a row-backed batch (same layout extend() builds).
        A single row keeps its replay slots."""
        if len(self._rows) == 1:
            return list(self._rows[0].cache)
        states = []
        for slot in range(7):
            values = [row.cache[slot] for row in self._rows]
            template = next((v for v in values if v is not None), None)
            if template is None:
                states.append(None)
                continue
            fill = -1 if slot == 6 else 0
            width = max(v.shape[1] for v in values if v is not None) if template.ndim > 1 else None
            parts = []
            for v in values:
                if v is None:
                    shape = (1, *template.shape[1:]) if width is None else (1, width, *template.shape[2:])
                    v = mx.full(shape, fill, template.dtype)
                elif width is not None and v.shape[1] != width:
                    v = mx.pad(v, [(0, 0), (0, width - v.shape[1])] + [(0, 0)] * (v.ndim - 2), constant_values=fill)
                parts.append(v)
            states.append(mx.concatenate(parts, 0))
        return states

    def _row_copy(self):
        """A single-request cache sharing this one's arrays and growth registry."""
        row = type(self)(self.compress_ratio)
        if getattr(self, "_ds41_buffers", None) is None:
            self._ds41_buffers = {}
        row.cache = list(self.cache)
        row._ds41_buffers = self._ds41_buffers
        return row

    def _as_rows(self):
        if self._rows is not None:
            return list(self._rows)
        if self.empty() or self.batch_size == 1:
            return [self._row_copy()]
        return [self._dense_extract(i) for i in range(self.batch_size)]

    def _set_rows(self, rows, left_padding=None):
        self._rows, self._dense = list(rows), None
        self._lp = None if left_padding is None else [int(v) for v in left_padding]
        self._len = None
        self._sync_padding()

    def _sync_padding(self):
        self.left_padding = None if self._lp is None else mx.array(self._lp, mx.int32)
        self.lengths = None if self._len is None else mx.array(self._len, mx.int32)

    def adopt_rows(self, rows):
        """Take back the rows a forward pass ran on (the rows extract() handed out)."""
        if self._rows is None and len(rows) == 1:
            self.cache = list(rows[0].cache)
            if getattr(rows[0], "_ds41_buffers", None) is not None:
                self._ds41_buffers = rows[0]._ds41_buffers
            return
        self._rows, self._dense = list(rows), None

    @property
    def batch_size(self):
        if self._rows is not None:
            return len(self._rows)
        return super().batch_size

    def empty(self):
        if self._rows is not None:
            return all(row.empty() for row in self._rows)
        return self._dense[0] is None

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls()
        obj.cache = list(state)
        obj.meta_state = meta_state
        return obj

    def has_pending(self):
        if self._rows is not None:
            return any(row.has_pending() for row in self._rows)
        return self._dense is not None and len(self._dense) > 7

    def pending(self):
        """The decoder replay input: midpoint output hidden, pre weights, top-k
        ids and candidate blocks of the last window, or None."""
        if not self.has_pending():
            return None
        if self._rows is not None and len(self._rows) > 1:
            raise ValueError("A batch holds its decoder replay inputs per row")
        return tuple(self.cache[7:])

    def set_pending(self, values):
        self.cache = list(self.cache[:7]) + ([] if values is None else list(values))

    @property
    def offset(self):
        return self.cache[0] if self.cache[0] is not None else 0

    def size(self):
        if self._rows is not None:
            return max(row.size() for row in self._rows)
        offsets = self.cache[0]
        if offsets is None:
            return 0
        # A single row already contains its maximum; avoid a GPU reduction.
        return int(offsets.item() if offsets.size == 1 else mx.max(offsets).item())

    @property
    def meta_state(self):
        if self.compress_ratio is None:
            return ("deepseek_v41", "2")
        # Version 4 marks the replay slots, which older readers must reject.
        version = "4" if len(self.cache) > 7 else "3"
        return ("deepseek_v41", version, str(self.compress_ratio))

    @meta_state.setter
    def meta_state(self, value):
        if tuple(value) == ("deepseek_v41", "2"):
            self.compress_ratio = None
        elif len(value) == 3 and tuple(value[:2]) in (
            ("deepseek_v41", "3"),
            ("deepseek_v41", "4"),
        ):
            self.compress_ratio = int(value[2])
            if self.compress_ratio < 0:
                raise ValueError("Invalid DeepSeek V4.1 compression ratio")
        else:
            raise ValueError("Unknown DeepSeek V4.1 cache format")

    def prepare(self, lengths=None, **kwargs):
        if self._rows is not None:
            self._lp, self._len = None, None if lengths is None else [int(v) for v in lengths]
            self._sync_padding()
            return
        self.left_padding = None
        self.lengths = mx.array(lengths) if lengths is not None else None

    def finalize(self):
        if self._rows is not None:
            self._lp = self._len = None
        self.left_padding = None
        self.lengths = None

    def advance(self, N):
        if self._rows is not None:
            self._lp = None if self._lp is None else [v - N for v in self._lp]
            self._len = None if self._len is None else [v - N for v in self._len]
            self._sync_padding()
            return
        super().advance(N)

    def make_mask(self, N):
        if self._rows is None:
            return super().make_mask(N)
        import numpy as np

        pos, mask = np.arange(N), None
        if self._lp is not None:
            mask = pos >= np.asarray(self._lp)[:, None]
        if self._len is not None:
            within = pos < np.asarray(self._len)[:, None]
            mask = within if mask is None else mask & within
        return mask

    def filter(self, batch_indices):
        if self._rows is None:
            return super().filter(batch_indices)
        keep = batch_indices.tolist() if hasattr(batch_indices, "tolist") else list(batch_indices)
        keep = [int(i) for i in keep]
        self._rows = [self._rows[i] for i in keep]
        self._dense = None
        self._lp = None if self._lp is None else [self._lp[i] for i in keep]
        self._len = None if self._len is None else [self._len[i] for i in keep]
        self._sync_padding()

    @property
    def state(self):
        return self.cache, self.left_padding, self.lengths

    @state.setter
    def state(self, v):
        self.cache, self.left_padding, self.lengths = v
        self._lp = self._len = None

    def extract(self, idx):
        if self._rows is not None:
            return self._rows[idx]._row_copy()
        return self._dense_extract(idx)

    def _dense_extract(self, idx):
        result = type(self)(self.compress_ratio)
        if self.batch_size == 1 and idx in (0, -1):
            # Forward passes run on extracted rows. Share the arrays and the
            # growth registry so appends keep extending the same storage.
            if getattr(self, "_ds41_buffers", None) is None:
                self._ds41_buffers = {}
            result.cache = list(self.cache)
            result._ds41_buffers = self._ds41_buffers
        else:
            result.cache = [
                x[idx : idx + 1] if x is not None else None for x in self.cache
            ]
        ratio = self.compress_ratio
        if result.cache[0] is None or ratio is None:
            return result
        # extend() pads every slot to the widest row; a padded row fails the
        # cumulative-state checks in the prefix cache, so trim to this row's
        # own offset. The window keeps min(offset, width) rows.
        offset = int(result.cache[0].item())
        limits = {
            1: offset,
            2: offset // ratio if ratio else 0,
            3: offset // ratio if ratio else 0,
            4: offset % ratio if ratio > 1 else 0,
            5: offset % ratio if ratio > 1 else 0,
        }
        for slot, limit in limits.items():
            value = result.cache[slot]
            if value is not None and value.ndim > 1 and value.shape[1] > limit:
                result.cache[slot] = value[:, :limit]
        return result

    @classmethod
    def merge(cls, caches):
        result = cls(caches[0].compress_ratio if caches else 0)
        rows = [row for other in caches for row in other._as_rows()]
        result._set_rows(rows, left_padding=[0] * len(rows))
        return result

    def extend(self, other):
        if self.compress_ratio != other.compress_ratio:
            raise ValueError("Cannot batch different V4.1 compression ratios")
        a, b = self._as_rows(), other._as_rows()
        lp_a = self._lp if self._rows is not None else (
            None if self.left_padding is None else [int(v) for v in self.left_padding.tolist()])
        lp_b = other._lp if other._rows is not None else (
            None if other.left_padding is None else [int(v) for v in other.left_padding.tolist()])
        self._set_rows(a + b, left_padding=(lp_a or [0] * len(a)) + (lp_b or [0] * len(b)))


class DeepseekV41CacheHandler(ArraysCacheHandler):
    @property
    def cache_type(self):
        return CacheType.DEEPSEEK_V41

    def reconstruct_cache(self, state, meta_state=None, token_count=0):
        states = state.get("states", [])
        if len(states) not in (7, 11):
            raise ValueError("V4.1 delta requires its complete block chain")
        return DeepseekV41Cache.from_state(states, meta_state)
