# SPDX-License-Identifier: MIT
"""Request-local CSA2 state, including compressor tails and Engram lookback."""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from ...cache.type_handlers import ArraysCacheHandler, CacheType


class DeepseekV41Cache(ArraysCache):
    # offset, window KV, compressed KV, index K, partial KV, partial gates, history
    def __init__(self, compress_ratio=0):
        super().__init__(7)
        self.compress_ratio = compress_ratio

    @property
    def offset(self):
        return self.cache[0] if self.cache[0] is not None else 0

    def size(self):
        offsets = self.cache[0]
        if offsets is None:
            return 0
        # A single row already contains its maximum; avoid a GPU reduction.
        return int(offsets.item() if offsets.size == 1 else mx.max(offsets).item())

    @property
    def meta_state(self):
        return (
            ("deepseek_v41", "2")
            if self.compress_ratio is None
            else ("deepseek_v41", "3", str(self.compress_ratio))
        )

    @meta_state.setter
    def meta_state(self, value):
        if tuple(value) == ("deepseek_v41", "2"):
            self.compress_ratio = None
        elif len(value) == 3 and tuple(value[:2]) == ("deepseek_v41", "3"):
            self.compress_ratio = int(value[2])
            if self.compress_ratio < 0:
                raise ValueError("Invalid DeepSeek V4.1 compression ratio")
        else:
            raise ValueError("Unknown DeepSeek V4.1 cache format")

    def prepare(self, lengths=None, **kwargs):
        self.left_padding = None
        self.lengths = mx.array(lengths) if lengths is not None else None

    def extract(self, idx):
        result = type(self)(self.compress_ratio)
        result.cache = [x[idx : idx + 1] if x is not None else None for x in self.cache]
        return result

    @classmethod
    def merge(cls, caches):
        result = cls(caches[0].compress_ratio if caches else 0)
        for other in caches:
            if result.cache[0] is None and result.left_padding is None:
                result.cache = list(other.cache)
                result.left_padding = mx.zeros((other.batch_size,), mx.int32)
            else:
                result.extend(other)
        return result

    def extend(self, other):
        if self.compress_ratio != other.compress_ratio:
            raise ValueError("Cannot batch different V4.1 compression ratios")
        a_batch, b_batch = self.batch_size, other.batch_size
        states = []
        for i, (a, b) in enumerate(zip(self.cache, other.cache)):
            if a is None and b is None:
                states.append(None)
                continue
            template = a if a is not None else b
            fill = -1 if i == 6 else 0
            if a is None:
                a = mx.full((a_batch, *template.shape[1:]), fill, template.dtype)
            if b is None:
                b = mx.full((b_batch, *template.shape[1:]), fill, template.dtype)
            if a.ndim > 1 and a.shape[1] != b.shape[1]:
                width = max(a.shape[1], b.shape[1])
                a = mx.pad(
                    a,
                    [(0, 0), (0, width - a.shape[1])] + [(0, 0)] * (a.ndim - 2),
                    constant_values=fill,
                )
                b = mx.pad(
                    b,
                    [(0, 0), (0, width - b.shape[1])] + [(0, 0)] * (b.ndim - 2),
                    constant_values=fill,
                )
            states.append(mx.concatenate([a, b], 0))
        self.cache = states
        self.left_padding = mx.concatenate(
            [
                (
                    self.left_padding
                    if self.left_padding is not None
                    else mx.zeros((a_batch,), mx.int32)
                ),
                (
                    other.left_padding
                    if other.left_padding is not None
                    else mx.zeros((b_batch,), mx.int32)
                ),
            ]
        )
        self.lengths = None


class DeepseekV41CacheHandler(ArraysCacheHandler):
    @property
    def cache_type(self):
        return CacheType.DEEPSEEK_V41

    def reconstruct_cache(self, state, meta_state=None, token_count=0):
        states = state.get("states", [])
        if len(states) != 7:
            raise ValueError("V4.1 delta requires its complete block chain")
        return DeepseekV41Cache.from_state(states, meta_state)
