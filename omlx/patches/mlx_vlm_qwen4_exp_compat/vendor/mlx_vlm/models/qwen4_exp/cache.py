from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten


def should_quantize_kv_layer(layer_idx: int, num_layers: int) -> bool:
    """Whether layer ``layer_idx`` should use a quantized KV cache.

    Live batch generation (``_make_cache``), stream quantize, and APC warm
    restore must share this policy so continuous-batching ``extend`` always
    joins same-typed peers.

    For deep stacks (``num_layers > 2``) the last full-attention layer stays
    unquantized — it is sensitive to quantization (see gemma-4-class models).
    Shallow stacks (``num_layers <= 2``) quantize every layer when kv-bits is on.
    """
    if num_layers <= 2:
        return True
    return layer_idx < num_layers - 1


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    right_padding: Optional[mx.array] = None,
    left_padding: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims((offset + N) - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


def make_prompt_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
) -> List[Any]:
    """
    Construct the model's cache for use in generation.

    This function will defer the cache construction to the model if it has a
    ``make_cache`` method, otherwise it will make a default KV cache.

    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``RotatingKVCache`` is used with a maximum
            size of ``max_kv_size``
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()

    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [
            RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]


def create_attention_mask(
    N: int, offset: int, return_array: bool, window_size: Optional[int]
):
    if window_size is not None:
        return create_causal_mask(N, offset, window_size=window_size)
    elif N == 1:
        return None
    elif return_array:
        return create_causal_mask(N, offset, window_size=window_size)
    else:
        return "causal"


class _BaseCache:
    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self):
        return False

    def size(self):
        """
        Return the size (i.e. sequence length) of the cache.

        Not every cache is required to implement this, in which case the size
        will always be 0 (though the cache may not be empty).
        """
        return 0

    @property
    def nbytes(self):
        """Return the size of this cache in bytes"""
        raise NotImplementedError("Cache sub-class must implement nbytes")

    def empty(self):
        """
        Return if the cache is empty or not.
        """
        raise NotImplementedError("Cache sub-class must implement this.")

    @classmethod
    def from_state(cls, state, meta_state):
        # Create an instance of cls without calling __init__
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj

    def prefix_cache_snapshot(self):
        """Return an opaque, restorable snapshot of this cache's state.

        The returned object must round-trip through ``prefix_cache_restore``
        into a fresh cache from ``model.make_cache()``. References are returned
        as-is; the caller (adapter) is responsible for any detaching copy.
        """
        return {"state": self.state, "meta_state": self.meta_state}

    def prefix_cache_restore(self, snapshot):
        """Restore a snapshot from :meth:`prefix_cache_snapshot` into ``self``."""
        self.state = snapshot["state"]
        self.meta_state = snapshot["meta_state"]

    def prefix_cache_merge(self, rows, prefix_lens):
        """Merge single-row snapshots into a batched cache, or ``None``.

        Default: not batch-mergeable. Pageable/windowed caches override to
        return a batched cache built from ``rows``.
        """
        return None


class ConcatenateKVCache(_BaseCache):
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=-2)
            self.values = mx.concatenate([self.values, values], axis=-2)
        self.offset = self.keys.shape[-2]
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, value):
        self.keys, self.values = value
        self.offset = 0 if self.keys is None else self.keys.shape[-2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        if self.keys is not None:
            self.keys = self.keys[..., : self.offset, :]
            self.values = self.values[..., : self.offset, :]
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


def _dequantize_uniform(keys_tuple, values_tuple, length, group_size, bits):
    """Dequantize uniform-quantized K/V tuples to raw float arrays.

    Shared by QuantizedKVCache and BatchQuantizedKVCache for APC storage.
    Returns None, None if the cache is empty.
    """
    if keys_tuple is None or values_tuple is None or length == 0:
        return None, None
    keys = mx.dequantize(
        mx.contiguous(keys_tuple[0][..., :length, :]),
        mx.contiguous(keys_tuple[1][..., :length, :]),
        mx.contiguous(keys_tuple[2][..., :length, :]),
        group_size=group_size,
        bits=bits,
    )
    values = mx.dequantize(
        mx.contiguous(values_tuple[0][..., :length, :]),
        mx.contiguous(values_tuple[1][..., :length, :]),
        mx.contiguous(values_tuple[2][..., :length, :]),
        group_size=group_size,
        bits=bits,
    )
    return keys, values


class QuantizedKVCache(_BaseCache):
    step = 256

    def __init__(self, group_size: int = 64, bits: int = 8):
        self.keys = None
        self.values = None
        self.offset = 0
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )

                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys, self.values = init_quant(k_head_dim), init_quant(v_head_dim)

        self.offset += num_steps

        keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = keys[i]
            self.values[i][..., prev : self.offset, :] = values[i]

        return tree_map(lambda x: x[..., : self.offset, :], (self.keys, self.values))

    @property
    def state(self):
        if self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        else:
            return tree_map(
                lambda x: x[..., : self.offset, :], (self.keys, self.values)
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.group_size, self.bits = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def dequantize_for_apc(self):
        """Return raw float (keys, values) sliced to current offset for APC storage.

        Returns (None, None) if the cache is empty.
        """
        if self.keys is None or self.offset == 0:
            return None, None
        return _dequantize_uniform(
            self.keys, self.values, self.offset, self.group_size, self.bits
        )

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


class KVCache(_BaseCache):
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def size(self):
        return self.offset

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def extract(self, idx):
        cache = KVCache()
        if self.keys is None:
            if idx not in (0, -1):
                raise IndexError("KVCache row index out of range")
            return cache

        batch_size = int(self.keys.shape[0])
        if idx < 0:
            idx += batch_size
        if idx < 0 or idx >= batch_size:
            raise IndexError(
                f"KVCache row index {idx} out of range for batch size {batch_size}"
            )

        cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, : self.offset, :])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, : self.offset, :])
        cache.offset = self.offset
        return cache

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        quant_cache = QuantizedKVCache(group_size=group_size, bits=bits)
        quant_cache.offset = self.offset
        if self.keys is not None:
            quant_cache.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
            quant_cache.values = mx.quantize(
                self.values, group_size=group_size, bits=bits
            )
        return quant_cache

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @classmethod
    def merge(_, caches):
        return BatchKVCache.merge(caches)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class RotatingKVCache(_BaseCache):
    step = 256

    def __init__(self, max_size, keep=0):
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self._idx = 0

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v):
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self.offset:
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self.keys.shape[2]

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def _update_in_place(self, keys, values):
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self.offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self.offset += S
        self._idx += S

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def size(self):
        return min(self.offset, self.max_size)

    @property
    def state(self):
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        else:
            return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.keep, self.max_size, self.offset, self._idx)))

    @meta_state.setter
    def meta_state(self, v):
        self.keep, self.max_size, self.offset, self._idx = map(
            int,
            v,
        )

    def is_trimmable(self):
        return self.offset < self.max_size

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._idx -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        raise NotImplementedError("RotatingKVCache Quantization NYI")

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        if N > 1:
            window_size = window_size or self.max_size
            offset = min(self.max_size - 1, self.offset)
            if offset + N > window_size or return_array:
                return create_causal_mask(N, offset, window_size=window_size)
            else:
                return "causal"
        else:
            if window_size is None:
                return None
            # May need a mask for when window_size < max_size
            if self.offset >= window_size and self.max_size > window_size:
                idx = self._idx
                if idx >= self.max_size:
                    idx = 0
                if self.offset < self.max_size:
                    mask_size = self.offset + 1
                else:
                    mask_size = self.max_size
                mask = mx.arange(mask_size) >= (mask_size - window_size)
                mask = mx.roll(mask, shift=idx + 1)
                return mask

    @classmethod
    def merge(_, caches):
        return BatchRotatingKVCache.merge(caches)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class ArraysCache(_BaseCache):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._left_padding = None
        instance._left_padding_advance = 0
        instance._lengths = None
        instance._lengths_advance = 0
        return instance

    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        if left_padding:
            self.left_padding = mx.array(left_padding)

    @property
    def left_padding(self):
        if self._left_padding is None:
            return None
        if self._left_padding_advance == 0:
            return self._left_padding
        return self._left_padding - self._left_padding_advance

    @left_padding.setter
    def left_padding(self, value):
        self._left_padding = value
        self._left_padding_advance = 0

    @property
    def lengths(self):
        if self._lengths is None:
            return None
        if self._lengths_advance == 0:
            return self._lengths
        return self._lengths - self._lengths_advance

    @lengths.setter
    def lengths(self, value):
        self._lengths = value
        self._lengths_advance = 0

    @property
    def batch_size(self):
        for c in self.cache:
            if c is not None:
                return c.shape[0]
        if self._left_padding is not None:
            return self._left_padding.size
        elif self._lengths is not None:
            return self._lengths.size
        else:
            return 1

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, v):
        self.cache = v

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """

        a_batch = self.batch_size
        b_batch = other.batch_size

        def cat(a, b):
            shape = dtype = None
            if a is not None:
                shape = a.shape
                dtype = a.dtype
            if b is not None:
                shape = b.shape
                dtype = b.dtype

            if shape is None:
                return None

            if a is None:
                a = mx.zeros((a_batch,) + shape[1:], dtype=dtype)
            if b is None:
                b = mx.zeros((b_batch,) + shape[1:], dtype=dtype)

            return mx.concatenate([a, b])

        self.cache = [cat(c, o) for c, o in zip(self.cache, other.cache)]
        self.left_padding = cat(self.left_padding, other.left_padding)
        self.lengths = cat(self.lengths, other.lengths)

    def extract(self, idx):
        cache = ArraysCache(len(self.cache))
        cache.cache = [c[idx : idx + 1] for c in self.cache]
        return cache

    def prepare(self, lengths=None, **kwargs):
        self.lengths = mx.array(lengths)

    def finalize(self):
        self.lengths = None
        self.left_padding = None

    def advance(self, N):
        if self._lengths is not None:
            self._lengths_advance += N
        if self._left_padding is not None:
            self._left_padding_advance += N

    def make_mask(self, N: int):
        if self.left_padding is not None:
            pos = mx.arange(N)
            return pos >= self.left_padding[:, None]
        elif self.lengths is not None:
            pos = mx.arange(N)
            return pos < self.lengths[:, None]
        else:
            return None

    @classmethod
    def merge(cls, caches):
        n_state = len(caches[0].cache)
        B = len(caches)
        cache = cls(n_state)

        # All caches are empty so return early
        if all(c.empty() for c in caches):
            cache.left_padding = mx.array([0] * B)
            return cache

        for e in range(n_state):
            c_init = next(iter(c[e] for c in caches if c[e] is not None))
            shape = list(c_init.shape)
            shape[0] = B
            cache[e] = mx.zeros(shape, c_init.dtype)
            for i in range(B):
                if caches[i][e] is None:
                    continue
                cache[e][i : i + 1] = caches[i][e]
        return cache

    def empty(self):
        return self.cache[0] is None

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.cache if c is not None)


class ChunkedKVCache(_BaseCache):
    step = 256

    def __init__(self, chunk_size):
        self.keys = None
        self.values = None
        self.offset = 0
        self.chunk_size = chunk_size
        self.start_position = 0

    def maybe_trim_front(self):
        # Maintain the cache below the chunk size
        if self.keys is not None and self.keys.shape[2] >= self.chunk_size:
            self.start_position += self.keys.shape[2] - self.chunk_size
            self.keys = self.keys[..., -self.chunk_size :, :]
            self.values = self.values[..., -self.chunk_size :, :]

    def update_and_fetch(self, keys, values):
        prev = self.offset - self.start_position
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        end = self.offset - self.start_position
        self.keys[..., prev:end, :] = keys
        self.values[..., prev:end, :] = values
        return self.keys[..., :end, :], self.values[..., :end, :]

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset - self.start_position, n)
        self.offset -= n
        return n

    @property
    def meta_state(self):
        return tuple(map(str, (self.chunk_size, self.start_position)))

    @meta_state.setter
    def meta_state(self, v):
        self.chunk_size, self.start_position = map(int, v)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class CacheList(_BaseCache):
    def __init__(self, *caches):
        self.caches = caches

    def __getitem__(self, idx):
        return self.caches[idx]

    def is_trimmable(self):
        return all(c.is_trimmable() for c in self.caches)

    def trim(self, n):
        for c in self.caches:
            m = c.trim(n)
        return m

    @property
    def state(self):
        return [c.state for c in self.caches]

    @state.setter
    def state(self, v):
        for c, s in zip(self.caches, v):
            c.state = s

    @property
    def meta_state(self):
        return (
            [type(c).__name__ for c in self.caches],
            [c.meta_state for c in self.caches],
        )

    @meta_state.setter
    def meta_state(self, v):
        for c, m in zip(self.caches, v[1]):
            c.meta_state = m

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        for c in self.caches:
            c.filter(batch_indices)

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        for c, o in zip(self.caches, other.caches):
            c.extend(o)

    @classmethod
    def merge(cls, caches):
        cache = cls()
        cache.caches = tuple(
            caches[0].caches[i].merge([c.caches[i] for c in caches])
            for i in range(len(caches[0].caches))
        )
        return cache

    def extract(self, idx):
        return CacheList(*(c.extract(idx) for c in self.caches))

    def prepare(self, **kwargs):
        for c in self.caches:
            c.prepare(**kwargs)

    def finalize(self):
        for c in self.caches:
            c.finalize()

    def size(self):
        return max(c.size() for c in self.caches)

    def empty(self):
        return self.caches[0].empty()

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.caches)

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.caches = [
            globals()[c].from_state(s, m) for s, c, m in zip(state, *meta_state)
        ]
        return obj


def dynamic_roll(x, shifts, axis):
    n = x.shape[axis]
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    idx = (mx.arange(n)[expand_indices] - shifts[expand_shifts]) % n
    rolled = mx.take_along_axis(x, idx, axis=axis)
    return rolled


class BatchKVCache(_BaseCache):
    step = 256

    def __init__(self, left_padding: List[int]):
        """
        The BatchKV cache expects inputs to be left-padded.

        E.g. the following prompts:

            [1, 3, 5]
            [7]
            [2, 6, 8, 9]

        Should be padded like so:

            [0, 1, 3, 5]
            [0, 0, 0, 7]
            [2, 6, 8, 9]

        And ``left_padding`` specifies the amount of padding for each.
        In this case, ``left_padding = [1, 3, 0]``.
        """
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-pad for pad in left_padding])
        self._idx = 0

        self._right_padding = None

    def update_and_fetch(self, keys, values):
        prev = self._idx
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self._idx += keys.shape[2]
        self.keys[..., prev : self._idx, :] = keys
        self.values[..., prev : self._idx, :] = values
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self.keys = dynamic_roll(self.keys, padding[:, None], axis=2)
            self.values = dynamic_roll(self.values, padding[:, None], axis=2)
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        k, v = self.keys, self.values
        if self._idx < k.shape[2]:
            k = k[..., : self._idx, :]
            v = v[..., : self._idx, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        self._idx = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        if self._right_padding is not None:
            self._right_padding = self._right_padding[batch_indices]

        # Shift left to reduce padding
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = self.keys[..., min_left_pad:, :]
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            B, H, L1, D = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            B, H, L2, D = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)

        # Pad the keys and values so they are right-justified
        # with the index and the same size
        def pad(c):
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.array([]).reshape(Bc, H, 0, D)
                v = mx.array([]).reshape(Bc, H, 0, M)
            left = max_idx - c._idx
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx

    def extract(self, idx):
        cache = KVCache()
        padding = self.left_padding[idx].item()
        cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding : self._idx])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding : self._idx])
        cache.offset = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return BatchKVCache([0] * len(caches))

        padding = [max_length - length for length in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + c.offset] = c.keys[..., : c.offset, :]
            values[i : i + 1, :, p : p + c.offset] = c.values[..., : c.offset, :]

        cache = cls(padding)
        cache.keys = keys
        cache.values = values
        cache.offset += keys.shape[2]
        cache._idx = keys.shape[2]

        return cache

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def batch_size(self):
        if self.keys is not None:
            return int(self.keys.shape[0])
        return int(self.left_padding.shape[0])

    def is_single_row(self):
        return self.batch_size == 1

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class BatchRotatingKVCache(_BaseCache):
    step = 256

    def __init__(self, max_size, left_padding: List[int]):
        self.keys = None
        self.values = None

        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-pad for pad in left_padding])

        self.max_size = max_size
        self._idx = 0
        self._offset = 0
        self.rotated = False

        # Lengths for right_padded inputs to make sure that padding tokens do
        # not evict valid tokens.
        self._lengths = None

    def _trim(self, trim_size, v, append=None):
        if trim_size > 0:
            v = v[..., trim_size:, :]
        if append is not None:
            return mx.concatenate([v, append], axis=2)
        return v

    def _temporal_order(self):
        """
        Rearrange the cache into temporal order.
        """
        if self.rotated:
            self.keys = mx.roll(self.keys, -self._idx, axis=2)
            self.values = mx.roll(self.values, -self._idx, axis=2)
            self._idx = self.keys.shape[2]
            self.rotated = False

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self._temporal_order()

            # Slice off the end if needed
            if self.keys.shape[2] > self._idx:
                self.keys = self.keys[..., : self._idx, :]
                self.values = self.values[..., : self._idx, :]

            # Roll right sequences that are padded to make sure that we don't
            # trim valid cache entries
            if self._lengths is not None:
                roll = mx.maximum(0, self.offset - self._lengths)
                self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
                self.values = dynamic_roll(self.values, roll[:, None], axis=2)
                self.left_padding += roll
                self.offset -= roll

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            if trim_size > 0:
                self.left_padding -= trim_size
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._offset += keys.shape[2]
        self._idx = self.keys.shape[2]

        # Make sure left_padding and offset are evaluated
        self.keys = mx.depends(self.keys, (self.left_padding, self.offset))

        return self.keys, self.values

    def _update_in_place(self, keys, values):
        if self._lengths is not None:
            raise RuntimeError(
                "finalize() should be called before deocoding with BatchRotatingKVCache"
            )

        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
            self.left_padding -= trim_size

        # Rotate
        if self._idx == self.max_size:
            self.rotated = True
            self._idx = 0
        if self.rotated:
            self.left_padding -= S

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self._offset += S
        self.offset += S
        self._idx += S

        # Make sure left_padding and offset are evaluated
        self.keys = mx.depends(self.keys, (self.left_padding, self.offset))

        # If the buffer is not full, slice off the end
        if self._offset < self.max_size:
            return (
                self.keys[..., : self._offset, :],
                self.values[..., : self._offset, :],
            )
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        # Single-token updates normally use the in-place decode path. While
        # right-padding bookkeeping is active (_lengths set by prepare()), the
        # last prefill step may still be S=1; that must go through concat so
        # pad tokens are tracked until finalize() rolls them out.
        if keys.shape[2] == 1 and self._lengths is None:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchRotatingKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._lengths = mx.array(lengths) + self.offset

    def finalize(self):
        if self._lengths is not None:
            roll = mx.maximum(0, self.offset - self._lengths)
            self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
            self.values = dynamic_roll(self.values, roll[:, None], axis=2)
            self.left_padding += roll
            self.offset -= roll
            self._lengths = None

    @property
    def state(self):
        k, v = self.keys, self.values
        if self._offset < k.shape[2]:
            k, v = k[..., : self._offset, :], v[..., : self._offset, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self._offset, self._idx, self.rotated)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self._offset, self._idx = map(
            int,
            v[:3],
        )
        self.rotated = bool(v[3])

    def is_trimmable(self):
        return self._offset < self.max_size

    def trim(self, n):
        n = min(self._offset, n)
        self._offset -= n
        self._idx -= n
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        raise NotImplementedError("BatchRotatingKVCache Quantization NYI")

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        left_padding = self.left_padding
        window_size = window_size or self.max_size
        offset = min(self.max_size - 1, self._offset)
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask &= linds < rinds + window_size
        if (trim_size := self._idx - self.max_size + int(N > 1)) > 0:
            left_padding = left_padding - trim_size

        rotated = N == 1 and (self.rotated or self._idx >= self.max_size)
        if rotated:
            left_padding = left_padding - 1

        mask = mask & (rinds >= mx.expand_dims(left_padding, (1, 2, 3)))

        if rotated:
            idx = self._idx
            if idx >= self.max_size:
                idx = 0
            mask = mx.roll(mask, shift=idx + 1, axis=-1)

        return mask

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        if self._lengths is not None:
            self._lengths = self._lengths[batch_indices]

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        if (self.rotated != other.rotated) or self._idx != other._idx:
            self._temporal_order()
            other._temporal_order()

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            B, H, L1, D = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            B, H, L2, D = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)

        def pad(c):
            left = max_idx - c._idx
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.array([]).reshape(Bc, H, 0, D)
                v = mx.array([]).reshape(Bc, H, 0, M)
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx
        self._offset = max(self._offset, other._offset)

    def extract(self, idx):
        mx.eval(self.left_padding, self.offset)
        cache = RotatingKVCache(self.max_size)
        padding = max(0, self.left_padding.tolist()[idx])
        offset = self.offset.tolist()[idx]
        cache.keys = self.keys[idx : idx + 1]
        cache.values = self.values[idx : idx + 1]
        cache._idx = self._idx
        if self.rotated:
            cache.keys = mx.roll(cache.keys, -self._idx, axis=2)
            cache.values = mx.roll(cache.values, -self._idx, axis=2)
            cache._idx = self.max_size
        cache.keys = mx.contiguous(cache.keys[:, :, padding : cache._idx])
        cache.values = mx.contiguous(cache.values[:, :, padding : cache._idx])
        cache.offset = offset
        cache._idx = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        if not all(c.max_size == caches[0].max_size for c in caches):
            raise ValueError(
                "BatchRotatingKVCache can only merge caches with the same maximum size"
            )

        offsets = [c.offset for c in caches]
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return cls(caches[0].max_size, [0] * len(caches))

        padding = [max_length - length for length in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, length, c) in enumerate(zip(padding, lengths, caches)):
            if c.keys is None or length == 0:
                continue
            keys[i : i + 1, :, p : p + length] = c._temporal_order(c.keys)[
                ..., -length:, :
            ]
            values[i : i + 1, :, p : p + length] = c._temporal_order(c.values)[
                ..., -length:, :
            ]

        cache = cls(caches[0].max_size, padding)
        cache.keys = keys
        cache.values = values
        cache.offset = mx.array(offsets)
        cache._idx = keys.shape[2]
        cache._offset = keys.shape[2]

        return cache

    def size(self):
        return min(self._offset, self.max_size)

    def empty(self):
        return self.keys is None

    @property
    def batch_size(self):
        if self.keys is not None:
            return int(self.keys.shape[0])
        return int(self.left_padding.shape[0])

    def is_single_row(self):
        return self.batch_size == 1

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


# MLX-VLM cache extensions.


class BufferedRotatingKVCache(RotatingKVCache):
    """Temporal sliding-window cache with rollback slack for speculative blocks."""

    def __init__(self, max_size: int, keep: int = 0, buffer_size: int = 64):
        super().__init__(max_size=max_size, keep=keep)
        self.buffer_size = max(0, int(buffer_size))
        self.start_position = 0

    @classmethod
    def from_cache(
        cls, other: RotatingKVCache, buffer_size: int = 64
    ) -> "BufferedRotatingKVCache":
        cache = cls(other.max_size, other.keep, buffer_size=buffer_size)
        cache.offset = int(other.offset)
        if other.keys is None:
            return cache

        keys = other._temporal_order(other.keys)
        values = other._temporal_order(other.values)
        keep = min(int(keys.shape[2]), other.max_size)
        if keep < int(keys.shape[2]):
            keys = keys[..., -keep:, :]
            values = values[..., -keep:, :]

        cache.start_position = cache.offset - keep
        cache._idx = keep
        capacity = cache._capacity_for(keep)
        k_shape = (*keys.shape[:2], capacity, keys.shape[-1])
        v_shape = (*values.shape[:2], capacity, values.shape[-1])
        cache.keys = mx.zeros(k_shape, dtype=keys.dtype)
        cache.values = mx.zeros(v_shape, dtype=values.dtype)
        cache.keys[..., :keep, :] = keys
        cache.values[..., :keep, :] = values
        return cache

    def _target_size(self, incoming: int = 0) -> int:
        return self.max_size + max(self.buffer_size, incoming)

    def _capacity_for(self, needed: int) -> int:
        target = max(needed, self._target_size())
        return ((target + self.step - 1) // self.step) * self.step

    def _planned_drop(self, incoming: int) -> int:
        needed = self._idx + incoming
        target_size = self._target_size(incoming)
        if needed <= target_size:
            return 0
        target_start = max(0, self.offset - self.max_size + 1)
        return min(max(0, target_start - self.start_position), self._idx)

    def _compact(self, drop: int) -> None:
        if drop <= 0:
            return
        keep = self._idx - drop
        keys = self.keys[..., drop : self._idx, :]
        values = self.values[..., drop : self._idx, :]
        capacity = self.keys.shape[2]
        new_keys = mx.zeros((*keys.shape[:2], capacity, keys.shape[-1]), keys.dtype)
        new_values = mx.zeros(
            (*values.shape[:2], capacity, values.shape[-1]), values.dtype
        )
        new_keys[..., :keep, :] = keys
        new_values[..., :keep, :] = values
        self.keys = new_keys
        self.values = new_values
        self.start_position += drop
        self._idx = keep

    def _ensure_capacity(self, keys: mx.array, values: mx.array, needed: int) -> None:
        if self.keys is not None and needed <= self.keys.shape[2]:
            return

        capacity = self._capacity_for(needed)
        k_shape = (*keys.shape[:2], capacity, keys.shape[-1])
        v_shape = (*values.shape[:2], capacity, values.shape[-1])
        new_keys = mx.zeros(k_shape, dtype=keys.dtype)
        new_values = mx.zeros(v_shape, dtype=values.dtype)
        if self.keys is not None and self._idx > 0:
            new_keys[..., : self._idx, :] = self.keys[..., : self._idx, :]
            new_values[..., : self._idx, :] = self.values[..., : self._idx, :]
        self.keys = new_keys
        self.values = new_values

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if self.keep:
            return super().update_and_fetch(keys, values)

        incoming = int(keys.shape[2])
        drop = self._planned_drop(incoming)
        self._compact(drop)
        needed = self._idx + incoming
        self._ensure_capacity(keys, values, needed)

        self.keys[..., self._idx : needed, :] = keys
        self.values[..., self._idx : needed, :] = values
        self._idx = needed
        self.offset += incoming
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    def trim(self, n):
        n = min(int(n), self._idx, self.offset)
        self.offset -= n
        self._idx -= n
        return n

    def is_trimmable(self):
        return True

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        if self.keep:
            return super().make_mask(
                N, window_size=window_size, return_array=return_array
            )

        window_size = window_size or self.max_size
        drop = self._planned_drop(N)
        start = self.start_position + drop
        end = self.offset + N
        key_len = end - start
        if N == 1 and key_len <= window_size and not return_array:
            return None

        q_idx = mx.arange(self.offset, end)[:, None]
        k_idx = mx.arange(start, end)[None, :]
        mask = (q_idx >= k_idx) & (q_idx < k_idx + window_size)
        if N > 1 and key_len <= window_size and not return_array:
            return "causal"
        return mask

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self._idx = 0 if self.keys is None else int(self.keys.shape[2])
        self.start_position = max(0, int(self.offset) - self._idx)

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.keep,
                    self.max_size,
                    self.offset,
                    self._idx,
                    self.start_position,
                    self.buffer_size,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        vals = list(map(int, v))
        self.keep, self.max_size, self.offset, self._idx = vals[:4]
        self.start_position = vals[4] if len(vals) > 4 else 0
        self.buffer_size = vals[5] if len(vals) > 5 else 64


class BatchQuantizedKVCache(_BaseCache):
    """Batch-aware quantized KV cache for continuous batching.

    Mirrors ``BatchKVCache`` but stores keys/values in quantized form
    (packed uint32 + scales + biases), matching the format produced by
    ``mx.quantize``.  Supports ``extend()`` and ``filter()`` so that
    ``Batch.extend`` / ``Batch.filter`` work during continuous-batching.
    """

    step = 256

    def __init__(
        self,
        left_padding: List[int],
        group_size: int = 64,
        bits: int = 8,
    ):
        self.keys = None  # tuple (packed, scales, biases) or None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-lp for lp in left_padding])
        self._idx = 0
        self.group_size = group_size
        self.bits = bits
        # Set by prepare() for multi-row right-pad prefill; cleared in finalize().
        self._right_padding = None

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        """Batch pad lifecycle — same contract as :class:`BatchKVCache`."""
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchQuantizedKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        """Roll right-padding into left-padding after multi-row prefill."""
        if self._right_padding is None:
            return
        padding = self._right_padding
        if self.keys is not None:
            # Seq axis is 2 for packed/scales/biases (B, H, S, …) — same as BatchKVCache.
            self.keys = tuple(
                dynamic_roll(k, padding[:, None], axis=2) for k in self.keys
            )
            self.values = tuple(
                dynamic_roll(v, padding[:, None], axis=2) for v in self.values
            )
        self.offset -= padding
        self.left_padding += padding
        self._right_padding = None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Quantize incoming keys/values and append to the cache.

        Args:
            keys:   float [B, H, S, D_k]
            values: float [B, H, S, D_v]

        Returns:
            Quantized (keys_tuple, values_tuple) sliced to ``_idx``.
        """
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self._idx
        el_per_int = 8 * mx.uint32.size // self.bits

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def _init(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def _expand(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = tuple(k[..., :prev, :] for k in self.keys)
                    self.values = tuple(v[..., :prev, :] for v in self.values)
                self.keys = tuple(_expand(k) for k in self.keys)
                self.values = tuple(_expand(v) for v in self.values)
            else:
                self.keys = _init(k_head_dim)
                self.values = _init(v_head_dim)

        self.offset += num_steps
        self._idx += num_steps

        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self._idx, :] = q_keys[i]
            self.values[i][..., prev : self._idx, :] = q_values[i]

        return (
            tuple(k[..., : self._idx, :] for k in self.keys),
            tuple(v[..., : self._idx, :] for v in self.values),
        )

    def dequantize_for_apc(self):
        """Return raw float (keys, values) sliced to current _idx for APC storage.

        Returns (None, None) if the cache is empty.
        """
        if self.keys is None or self._idx == 0:
            return None, None
        return _dequantize_uniform(
            self.keys, self.values, self._idx, self.group_size, self.bits
        )

    def extract(self, idx):
        """Extract one batch row as a single-sequence QuantizedKVCache."""
        cache = QuantizedKVCache(group_size=self.group_size, bits=self.bits)
        if self.keys is None or self._idx == 0:
            return cache
        padding = int(self.left_padding[idx].item())
        end = self._idx
        cache.keys = tuple(
            mx.contiguous(k[idx : idx + 1, :, padding:end, :]) for k in self.keys
        )
        cache.values = tuple(
            mx.contiguous(v[idx : idx + 1, :, padding:end, :]) for v in self.values
        )
        cache.offset = int(cache.keys[0].shape[2])
        return cache

    def filter(self, batch_indices: mx.array):
        """Keep only the sequences at *batch_indices*."""
        if self.keys is not None:
            self.keys = tuple(k[batch_indices] for k in self.keys)
            self.values = tuple(v[batch_indices] for v in self.values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        if self._right_padding is not None:
            self._right_padding = self._right_padding[batch_indices]

        min_lp = self.left_padding.min().item()
        if min_lp > 0:
            if self.keys is not None:
                self.keys = tuple(k[..., min_lp:, :] for k in self.keys)
                self.values = tuple(v[..., min_lp:, :] for v in self.values)
            self._idx -= min_lp
            self.left_padding -= min_lp

    def extend(self, other: "BatchQuantizedKVCache"):
        """Concatenate *other* batch into this cache along the batch dim."""
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        self_len = 0 if self.keys is None else self.keys[0].shape[-2]
        other_len = 0 if other.keys is None else other.keys[0].shape[-2]
        max_size = max(max_idx, self_len, other_len)
        ref_keys = self.keys if self.keys is not None else other.keys
        ref_values = self.values if self.values is not None else other.values

        def _empty_like(parts, batch_size: int):
            out = []
            for part in parts:
                shape = list(part.shape)
                shape[0] = batch_size
                shape[-2] = 0
                out.append(mx.zeros(tuple(shape), dtype=part.dtype))
            return tuple(out)

        def _pad_quant(cache_obj):
            if cache_obj.keys is None:
                batch_size = int(cache_obj.offset.shape[0])
                keys = _empty_like(ref_keys, batch_size)
                values = _empty_like(ref_values, batch_size)
            else:
                keys = cache_obj.keys
                values = cache_obj.values
            left = max_idx - cache_obj._idx
            right = max_size - keys[0].shape[-2] - left
            if right < 0:
                trimmed_keys = tuple(k[..., : max_size - left, :] for k in keys)
                trimmed_values = tuple(v[..., : max_size - left, :] for v in values)
                right = 0
            else:
                trimmed_keys = keys
                trimmed_values = values

            if left != 0 or right != 0:
                pad_spec = [(0, 0)] * len(trimmed_keys[0].shape)
                pad_spec[-2] = (left, right)
                padded_keys = tuple(mx.pad(k, pad_spec) for k in trimmed_keys)
                padded_values = tuple(mx.pad(v, pad_spec) for v in trimmed_values)
            else:
                padded_keys = trimmed_keys
                padded_values = trimmed_values

            lp = cache_obj.left_padding + left
            return padded_keys, padded_values, cache_obj.offset, lp

        r_self = _pad_quant(self)
        r_other = _pad_quant(other)

        sk, sv, so, slp = r_self
        ok, ov, oo, olp = r_other

        self.keys = tuple(mx.concatenate([s, o]) for s, o in zip(sk, ok))
        self.values = tuple(mx.concatenate([s, o]) for s, o in zip(sv, ov))
        self.offset = mx.concatenate([so, oo])
        self.left_padding = mx.concatenate([slp, olp])
        self._idx = max_idx

    @property
    def state(self):
        if self.keys is None:
            return None, None, self.offset, self.left_padding
        k = tuple(x[..., : self._idx, :] for x in self.keys)
        v = tuple(x[..., : self._idx, :] for x in self.values)
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, val):
        self.keys, self.values, self.offset, self.left_padding = val
        if self.keys is not None:
            self._idx = self.keys[0].shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self._idx, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self._idx, self.group_size, self.bits = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def empty(self):
        return self.keys is None

    @property
    def batch_size(self):
        if self.keys is not None:
            return int(self.keys[0].shape[0])
        return int(self.left_padding.shape[0])

    def is_single_row(self):
        return self.batch_size == 1

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )


class PoolingCache(_BaseCache):
    """Cache for pooled (compressed) KV tokens with a remainder buffer.

    Stores two things:
      1. A growing pool of compressed tokens (step-allocated).
      2. A small remainder buffer of tokens not yet forming a full window.
    """

    def __init__(self, ratio: int):
        self.ratio = ratio

        self.buf_kv = None
        self.buf_gate = None
        self.remainder = 0

        self.pooled = None

    @property
    def offset(self):
        return 0 if self.pooled is None else self.pooled.shape[1]

    def accumulate_windows(self, kv: mx.array, gate: mx.array, offset):
        B, L, D1 = kv.shape
        _, _, D2 = gate.shape

        if self.buf_kv is None:
            self.buf_kv = mx.zeros((B, self.ratio, D1), dtype=kv.dtype)
            self.buf_gate = mx.zeros((B, self.ratio, D2), dtype=gate.dtype)

        # Prompt mode
        if L > 1:
            total = L + self.remainder
            usable = (total // self.ratio) * self.ratio
            new_remainder = total % self.ratio

            if usable > 0:
                r_kv = mx.concatenate(
                    [
                        self.buf_kv[:, : self.remainder],
                        kv[:, : (usable - self.remainder)],
                    ],
                    axis=1,
                )
                r_gate = mx.concatenate(
                    [
                        self.buf_gate[:, : self.remainder],
                        gate[:, : (usable - self.remainder)],
                    ],
                    axis=1,
                )
                r_base = offset - self.remainder
                self.remainder = 0
            else:
                r_kv = mx.zeros((B, 0, D1), dtype=kv.dtype)
                r_gate = mx.zeros((B, 0, D2), dtype=gate.dtype)
                r_base = 0

            if new_remainder > 0:
                self.buf_kv[:, self.remainder : new_remainder] = kv[:, -new_remainder:]
                self.buf_gate[:, self.remainder : new_remainder] = gate[
                    :, -new_remainder:
                ]
            self.remainder = new_remainder

            return r_kv, r_gate, r_base

        # Decode mode
        else:
            self.buf_kv[:, self.remainder : self.remainder + 1] = kv
            self.buf_gate[:, self.remainder : self.remainder + 1] = gate
            self.remainder = (self.remainder + 1) % self.ratio

            if self.remainder == 0:
                r_kv = self.buf_kv
                r_gate = self.buf_gate
                r_base = offset - self.ratio + 1
            else:
                r_kv = mx.zeros((B, 0, D1), dtype=kv.dtype)
                r_gate = mx.zeros((B, 0, D2), dtype=gate.dtype)
                r_base = 0

            return r_kv, r_gate, r_base

    def update_and_fetch(self, px: mx.array):
        if px.shape[1] == 0:
            if self.pooled is None:
                return mx.zeros((px.shape[0], 0, px.shape[-1]), dtype=px.dtype)
            return self.pooled

        if self.pooled is None:
            self.pooled = px
        else:
            self.pooled = mx.concatenate([self.pooled, px], axis=1)
        return self.pooled

    def make_mask(self, L: int = 1, offset: int = 0):
        """Build a causal validity mask for pooled positions.

        Query at absolute position ``offset + j`` can attend to pooled token
        ``i`` iff ``i < (offset + j) // ratio``.

        Returns ``(N, P)`` bool mask, or ``None`` when every pooled position
        is visible to every query (common during decode).
        """
        if self.pooled is None or L == 1:
            return None

        pool_idx = mx.arange(self.pooled.shape[1])
        query_idx = mx.arange(offset + 1, offset + L + 1)
        return pool_idx < query_idx[:, None] // self.ratio

    @property
    def state(self):
        buf_kv = self.buf_kv[:, : self.remainder] if self.remainder > 0 else None
        buf_gate = self.buf_gate[:, : self.remainder] if self.remainder > 0 else None
        return (buf_kv, buf_gate, self.pooled)

    @state.setter
    def state(self, v):
        buf_kv, buf_gate, pooled = v
        self.remainder = 0
        self.buf_kv = self.buf_gate = None
        if buf_kv is not None:
            self.accumulate_windows(buf_kv, buf_gate, 0)
        self.pooled = pooled

    @property
    def meta_state(self):
        return self.ratio

    @meta_state.setter
    def meta_state(self, v):
        self.ratio = v

    @classmethod
    def from_state(cls, state, meta_state):
        # Restoring buffered state calls ``accumulate_windows``, which needs
        # the compression ratio before the generic state setter can run.
        obj = cls(meta_state)
        obj.state = state
        return obj

    def is_trimmable(self):
        return self.pooled is None

    def trim(self, n):
        n = min(self.remainder, n)
        self.remainder -= n
        return n

    def size(self):
        return 0 if self.pooled is None else self.pooled.shape[1]

    def empty(self):
        return self.pooled is None and self.remainder == 0

    @property
    def nbytes(self):
        total = 0
        if self.buf_kv is not None:
            total += self.buf_kv.nbytes + self.buf_gate.nbytes
        if self.pooled is not None:
            total += self.pooled.nbytes
        return total

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        return BatchPoolingCache.merge(caches, prefix_lens)


class BatchPoolingCache(_BaseCache):
    """Batched pooling cache with per-element variable-length tracking."""

    def __init__(self, ratio: int, left_padding: List[int]):
        self.ratio = ratio

        if isinstance(left_padding, mx.array):
            left_padding = left_padding.tolist()
        self.left_padding = [int(p) for p in left_padding]

        batch_size = len(left_padding)

        self.buf_kv = None
        self.buf_gate = None
        self.remainder = [0] * batch_size

        self.pooled = None
        self._pool_lengths = [0] * batch_size

        self._lengths = [2**31] * batch_size
        self._processed = [0] * batch_size

    @property
    def offset(self):
        return mx.array(self._pool_lengths, dtype=mx.int32)

    def prepare(self, *, lengths=None, right_padding=None, left_padding=None):
        if left_padding is not None:
            if (
                self.buf_kv is not None
                or self.pooled is not None
                or any(self.remainder)
            ):
                raise ValueError(
                    "Left padding can only be added to an empty BatchPoolingCache"
                )
            if isinstance(left_padding, mx.array):
                left_padding = left_padding.tolist()
            if len(left_padding) != len(self.left_padding):
                raise ValueError("Left padding must match the cache batch size")
            self.left_padding = [
                current + int(pad)
                for current, pad in zip(self.left_padding, left_padding)
            ]
        if lengths is not None:
            self._lengths = [
                processed + length
                for processed, length in zip(self._processed, lengths)
            ]

    def finalize(self):
        self._lengths = [2**31] * len(self._pool_lengths)

    def accumulate_windows(self, kv: mx.array, gate: mx.array, offset):
        B, L, D1 = kv.shape
        _, _, D2 = gate.shape
        ratio = self.ratio

        if not (
            len(self.left_padding) == len(self._lengths) == len(self._processed) == B
        ):
            raise ValueError("Pooling cache metadata must match the input batch size")

        if self.buf_kv is None:
            self.buf_kv = mx.zeros((B, ratio, D1), dtype=kv.dtype)
            self.buf_gate = mx.zeros((B, ratio, D2), dtype=gate.dtype)

        # Each row can start at a different physical position because prompt
        # batches are left-padded. Only logical tokens participate in pooling.
        starts = [min(pad, L) for pad in self.left_padding]
        self.left_padding = [
            pad - start for pad, start in zip(self.left_padding, starts)
        ]
        valid_lengths = [
            max(0, min(length - processed, L - start))
            for length, processed, start in zip(self._lengths, self._processed, starts)
        ]
        for i in range(B):
            self._processed[i] += valid_lengths[i]

        totals = [vl + r for vl, r in zip(valid_lengths, self.remainder)]
        usable = [(t // ratio) * ratio for t in totals]
        max_usable = max(usable)
        new_remainder = [t % ratio for t in totals]

        # No sequence produced a full window yet
        if max_usable == 0:
            for i in range(B):
                r = self.remainder[i]
                vl = valid_lengths[i]
                start = starts[i]
                self.buf_kv[i, r : r + vl] = kv[i, start : start + vl]
                self.buf_gate[i, r : r + vl] = gate[i, start : start + vl]
            self.remainder = new_remainder

            r_kv = mx.zeros((B, 0, D1), dtype=kv.dtype)
            r_gate = mx.zeros((B, 0, D2), dtype=gate.dtype)
            r_base = 0
            return r_kv, r_gate, r_base

        # At least one sequence completed a window
        r_kv = mx.zeros((B, max_usable, D1), dtype=kv.dtype)
        r_gate = mx.zeros((B, max_usable, D2), dtype=gate.dtype)
        r_base = [0] * B

        new_buf_kv = mx.zeros_like(self.buf_kv)
        new_buf_gate = mx.zeros_like(self.buf_gate)

        for i in range(B):
            r = self.remainder[i]
            vl = valid_lengths[i]
            u = usable[i]
            nr = new_remainder[i]
            start = starts[i]

            if u > 0:
                # Tokens from the buffer (the leftover from last call)
                if r > 0:
                    r_kv[i, :r] = self.buf_kv[i, :r]
                    r_gate[i, :r] = self.buf_gate[i, :r]

                # Tokens from the new input that complete full windows
                consume = u - r
                r_kv[i, r : r + consume] = kv[i, start : start + consume]
                r_gate[i, r : r + consume] = gate[i, start : start + consume]

                r_base[i] = (
                    offset[i] + start - r
                    if isinstance(offset, mx.array)
                    else offset + start - r
                )

            # Fill new remainder buffer from the tail of the input
            if nr > 0:
                if u > 0:
                    # Old remainder was consumed into usable output;
                    # new remainder is purely from the tail of new input.
                    new_buf_kv[i, :nr] = kv[i, start + vl - nr : start + vl]
                    new_buf_gate[i, :nr] = gate[i, start + vl - nr : start + vl]
                else:
                    # No full window produced: carry over old buffer and
                    # append any new valid tokens.
                    if r > 0:
                        new_buf_kv[i, :r] = self.buf_kv[i, :r]
                        new_buf_gate[i, :r] = self.buf_gate[i, :r]
                    if vl > 0:
                        new_buf_kv[i, r : r + vl] = kv[i, start : start + vl]
                        new_buf_gate[i, r : r + vl] = gate[i, start : start + vl]

        self.buf_kv = new_buf_kv
        self.buf_gate = new_buf_gate
        self.remainder = new_remainder

        r_base = mx.array(r_base)
        return r_kv, r_gate, r_base

    def update_and_fetch(self, px: mx.array):
        B, N, D = px.shape

        if N == 0:
            if self.pooled is None:
                return mx.zeros((B, 0, D), dtype=px.dtype)
            return self.pooled

        # Derive how many new pooled tokens each sequence actually produced.
        new_counts = [
            (self._processed[i] - self.remainder[i]) // self.ratio
            - self._pool_lengths[i]
            for i in range(B)
        ]
        max_new = max(new_counts)
        if max_new == 0:
            if self.pooled is None:
                return mx.zeros((B, 0, D), dtype=px.dtype)
            return self.pooled

        max_pool = max(self._pool_lengths) + max_new

        if self.pooled is None:
            self.pooled = mx.zeros((B, max_pool, D), dtype=px.dtype)
        elif self.pooled.shape[1] < max_pool:
            pad = mx.zeros((B, max_pool - self.pooled.shape[1], D), dtype=px.dtype)
            self.pooled = mx.concatenate([self.pooled, pad], axis=1)

        for i in range(B):
            nc = new_counts[i]
            if nc > 0:
                pl = self._pool_lengths[i]
                self.pooled[i, pl : pl + nc] = px[i, :nc]
                self._pool_lengths[i] = pl + nc

        return self.pooled

    def make_mask(self, L: int = 1, offset=0):
        if self.pooled is None:
            return None

        B, P, _ = self.pooled.shape
        pool_lengths = mx.array(self._pool_lengths)

        # Length based mask
        pool_idx = mx.arange(P)[None, None, :]
        valid = pool_idx < pool_lengths[:, None, None]

        # Decode so no need for causal masking
        if L == 1:
            if all(pl == P for pl in self._pool_lengths):
                return None
            return valid

        # Prompt so we need to combine with causal
        if isinstance(offset, mx.array):
            query_pos = offset[:, None] + mx.arange(1, L + 1)
        else:
            query_pos = offset + mx.arange(offset + 1, offset + L + 1)[None]

        causal = pool_idx < (query_pos[..., None] // self.ratio)
        mask = causal & valid
        return mask

    @property
    def state(self):
        return (self.buf_kv, self.buf_gate, self.pooled)

    @state.setter
    def state(self, v):
        self.buf_kv, self.buf_gate, self.pooled = v

    @property
    def meta_state(self):
        return (
            self.ratio,
            self.remainder,
            self._pool_lengths,
            self._processed,
            self.left_padding,
        )

    @meta_state.setter
    def meta_state(self, v):
        if len(v) == 4:
            self.ratio, self.remainder, self._pool_lengths, self._processed = v
            self.left_padding = [0] * len(self.remainder)
        else:
            (
                self.ratio,
                self.remainder,
                self._pool_lengths,
                self._processed,
                self.left_padding,
            ) = v
        self._lengths = [2**31] * len(self.remainder)

    def is_trimmable(self):
        return self.pooled is None

    def trim(self, n):
        n = min(min(self.remainder), n)
        for i in range(len(self.remainder)):
            self.remainder[i] -= n
            self._processed[i] -= n
        return n

    def size(self):
        return 0 if self.pooled is None else self.pooled.shape[1]

    def empty(self):
        return self.pooled is None and all(r == 0 for r in self.remainder)

    @property
    def nbytes(self):
        total = 0
        if self.buf_kv is not None:
            total += self.buf_kv.nbytes + self.buf_gate.nbytes
        if self.pooled is not None:
            total += self.pooled.nbytes
        return total

    def filter(self, batch_indices):
        if isinstance(batch_indices, mx.array):
            idx_list = batch_indices.tolist()
        else:
            idx_list = list(batch_indices)

        if self.buf_kv is not None:
            self.buf_kv = self.buf_kv[batch_indices]
            self.buf_gate = self.buf_gate[batch_indices]
        if self.pooled is not None:
            self.pooled = self.pooled[batch_indices]

        self.remainder = [self.remainder[i] for i in idx_list]
        self._pool_lengths = [self._pool_lengths[i] for i in idx_list]
        self._lengths = [self._lengths[i] for i in idx_list]
        self._processed = [self._processed[i] for i in idx_list]
        self.left_padding = [self.left_padding[i] for i in idx_list]

    def extend(self, other):
        # Merge the remainder buffers
        if self.buf_kv is None and other.buf_kv is None:
            pass
        elif self.buf_kv is not None and other.buf_kv is not None:
            self.buf_kv = mx.concatenate([self.buf_kv, other.buf_kv], axis=0)
            self.buf_gate = mx.concatenate([self.buf_gate, other.buf_gate], axis=0)
        elif self.buf_kv is None:
            B = len(self.remainder)
            D1 = other.buf_kv.shape[2]
            D2 = other.buf_gate.shape[2]
            self.buf_kv = mx.concatenate(
                [mx.zeros((B, self.ratio, D1), dtype=other.buf_kv.dtype), other.buf_kv],
                axis=0,
            )
            self.buf_gate = mx.concatenate(
                [
                    mx.zeros((B, self.ratio, D2), dtype=other.buf_gate.dtype),
                    other.buf_gate,
                ],
                axis=0,
            )
        else:
            B2 = len(other.remainder)
            D1 = self.buf_kv.shape[2]
            D2 = self.buf_gate.shape[2]
            self.buf_kv = mx.concatenate(
                [self.buf_kv, mx.zeros((B2, self.ratio, D1), dtype=self.buf_kv.dtype)],
                axis=0,
            )
            self.buf_gate = mx.concatenate(
                [
                    self.buf_gate,
                    mx.zeros((B2, self.ratio, D2), dtype=self.buf_gate.dtype),
                ],
                axis=0,
            )

        # Merge the pooled buffers
        if self.pooled is None and other.pooled is None:
            pass
        else:
            B1 = len(self.remainder)
            B2 = len(other.remainder)
            P1 = 0 if self.pooled is None else self.pooled.shape[1]
            P2 = 0 if other.pooled is None else other.pooled.shape[1]
            max_P = max(P1, P2)

            if max_P > 0:
                if self.pooled is not None:
                    D = self.pooled.shape[2]
                else:
                    D = other.pooled.shape[2]
                dt = (self.pooled if self.pooled is not None else other.pooled).dtype

                def pad_pool(pooled, B, P):
                    if pooled is None:
                        return mx.zeros((B, max_P, D), dtype=dt)
                    if P < max_P:
                        pad = mx.zeros((pooled.shape[0], max_P - P, D), dtype=dt)
                        return mx.concatenate([pooled, pad], axis=1)
                    return pooled

                self.pooled = mx.concatenate(
                    [pad_pool(self.pooled, B1, P1), pad_pool(other.pooled, B2, P2)],
                    axis=0,
                )

        self.remainder = self.remainder + other.remainder
        self._pool_lengths = self._pool_lengths + other._pool_lengths
        self._lengths = self._lengths + other._lengths
        self._processed = self._processed + other._processed
        self.left_padding = self.left_padding + other.left_padding

    def extract(self, idx):
        cache = PoolingCache(self.ratio)
        pl = self._pool_lengths[idx]
        r = self.remainder[idx]

        if self.pooled is not None and pl > 0:
            cache.pooled = mx.contiguous(self.pooled[idx : idx + 1, :pl])

        if self.buf_kv is not None and r > 0:
            cache.buf_kv = mx.contiguous(self.buf_kv[idx : idx + 1])
            cache.buf_gate = mx.contiguous(self.buf_gate[idx : idx + 1])
            cache.remainder = r

        return cache

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        """Merge a list of PoolingCache instances into a BatchPoolingCache."""
        # APC passes prefix lengths to custom merge implementations. Pooling
        # caches derive progress from each row's pooled length and remainder.
        del prefix_lens
        B = len(caches)
        if not all(c.ratio == caches[0].ratio for c in caches):
            raise ValueError(
                "BatchPoolingCache can only merge caches with the same ratio"
            )
        ratio = caches[0].ratio
        batch_cache = cls(ratio, [0] * B)

        # Check if all caches are empty
        if all(c.empty() for c in caches):
            return batch_cache

        # Merge pooled buffers
        pool_sizes = [c.size() for c in caches]
        max_pool = max(pool_sizes)
        if max_pool > 0:
            D = next(c.pooled.shape[2] for c in caches if c.pooled is not None)
            dt = next(c.pooled.dtype for c in caches if c.pooled is not None)
            pooled = mx.zeros((B, max_pool, D), dtype=dt)
            for i, c in enumerate(caches):
                if c.pooled is not None:
                    ps = c.pooled.shape[1]
                    pooled[i, :ps] = c.pooled[0]
            batch_cache.pooled = pooled

        batch_cache._pool_lengths = pool_sizes
        batch_cache.remainder = [c.remainder for c in caches]
        batch_cache._processed = [
            c.remainder + ps * ratio for c, ps in zip(caches, pool_sizes)
        ]

        # Merge remainder buffers
        has_buf = any(c.buf_kv is not None for c in caches)
        if has_buf:
            D1 = next(c.buf_kv.shape[2] for c in caches if c.buf_kv is not None)
            D2 = next(c.buf_gate.shape[2] for c in caches if c.buf_gate is not None)
            dt = next(c.buf_kv.dtype for c in caches if c.buf_kv is not None)
            buf_kv = mx.zeros((B, ratio, D1), dtype=dt)
            buf_gate = mx.zeros((B, ratio, D2), dtype=dt)
            for i, c in enumerate(caches):
                if c.buf_kv is not None and c.remainder > 0:
                    buf_kv[i, : c.remainder] = c.buf_kv[0, : c.remainder]
                    buf_gate[i, : c.remainder] = c.buf_gate[0, : c.remainder]
            batch_cache.buf_kv = buf_kv
            batch_cache.buf_gate = buf_gate

        return batch_cache


class SimpleKVCache:
    """A simple key-value cache for transformer attention layers.

    Stores and concatenates key/value tensors along sequence dimension.
    """

    def __init__(self):
        self.keys = None
        self.values = None
        self.cache_length = 0

    def update_and_fetch(self, keys, values):
        """Update cache with new key/value tensors and return full cache.

        Args:
            keys: New key tensor to add [batch, heads, seq_len, head_dim]
            values: New value tensor to add [batch, heads, seq_len, head_dim]

        Returns:
            Tuple of (cached_keys, cached_values) containing full cache history
        """
        if self.cache_length == 0:
            # First update - just store tensors
            self.keys = keys
            self.values = values
        else:
            # Concatenate with existing cache along sequence dimension
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self.cache_length += keys.shape[2]
        return self.keys, self.values

    def fetch(self):
        return self.keys, self.values

    def update(self, keys, values):
        """Update cache with new key/value tensors without returning.

        Args:
            keys: New key tensor to store
            values: New value tensor to store
        """
        self.keys = keys
        self.values = values
        self.cache_length += keys.shape[2]


class StaticPrefixKVCache(_BaseCache):
    """Fixed-capacity KV cache with prefix fetch semantics.

    ``update_and_fetch`` returns only the populated prefix so normal encoder
    prefill attention sees the same shapes as a dynamic cache. Decoder code can
    use ``decoder_state`` to attend over the full fixed allocation together with
    an attention mask that hides unpopulated entries.
    """

    def __init__(self, max_size: int, step: int = 256, read_only: bool = False):
        self.max_size = int(max_size)
        self.step = int(step)
        self.keys = None
        self.values = None
        self.offset = 0
        self.read_only = read_only

    @classmethod
    def from_prefix(cls, other: "StaticPrefixKVCache") -> "StaticPrefixKVCache":
        cache = cls(other.max_size, other.step, read_only=True)
        cache.reset_from_prefix(other)
        return cache

    def reset_from_prefix(self, other: "StaticPrefixKVCache") -> None:
        self.keys = other.keys
        self.values = other.values
        self.offset = other.offset
        self.max_size = other.max_size
        self.step = other.step
        self.read_only = True

    def _capacity_for(self, needed: int) -> int:
        if needed <= self.max_size:
            return self.max_size
        return ((needed + self.step - 1) // self.step) * self.step

    def _ensure_capacity(self, keys: mx.array, values: mx.array, needed: int) -> None:
        if self.keys is not None and needed <= self.keys.shape[2]:
            return

        B, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        capacity = self._capacity_for(needed)
        new_keys = mx.zeros((B, n_kv_heads, capacity, k_head_dim), dtype=keys.dtype)
        new_values = mx.zeros((B, n_kv_heads, capacity, v_head_dim), dtype=values.dtype)
        if self.keys is not None and self.offset > 0:
            new_keys[..., : self.offset, :] = self.keys[..., : self.offset, :]
            new_values[..., : self.offset, :] = self.values[..., : self.offset, :]
        self.keys = new_keys
        self.values = new_values
        self.max_size = capacity

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        if self.read_only:
            if self.keys is None:
                return keys, values
            return (
                mx.concatenate([self.keys[..., : self.offset, :], keys], axis=2),
                mx.concatenate([self.values[..., : self.offset, :], values], axis=2),
            )

        seq_len = keys.shape[2]
        prev = self.offset
        end = prev + seq_len
        self._ensure_capacity(keys, values, end)
        self.keys[..., prev:end, :] = keys
        self.values[..., prev:end, :] = values
        self.offset = end
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @state.setter
    def state(self, v):
        if v is not None and len(v) == 2:
            self.keys, self.values = v
            if self.keys is not None:
                self.offset = self.keys.shape[2]
                self.max_size = self.keys.shape[2]

    @property
    def decoder_state(self):
        if self.keys is None:
            return None, None
        return self.keys, self.values

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self.step, self.offset)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self.step, self.offset = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(int(n), self.offset)
        self.offset -= n
        return n

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes
