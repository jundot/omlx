# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV cache — thin wrapper around mlx_vlm.turboquant.

Core implementation (codecs, Metal kernels, TurboQuantKVCache) lives in
mlx-vlm.  This module re-exports the public API and adds
BatchTurboQuantKVCache for omlx's continuous-batching scheduler.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import mlx.core as mx
from mlx_lm.models.cache import (
    KVCache,
    _BaseCache,
    create_attention_mask,
    create_causal_mask,
    dynamic_roll,
)
from mlx_vlm.turboquant import (
    TurboQuantKVCache,
    TurboQuantMSEState,
    TurboQuantProdState,
    TurboQuantPolarState,
    TurboQuantPolarProdState,
    TurboQuantSplitState,
    _build_codec,
    _concat_state,
    _slice_state,
    _slice_state_range,
    _state_length,
    _state_nbytes,
    _allocate_state_like,
    _write_state,
    _reserve_state_capacity,
    _QuantizedStateProxy,
    _validate_bits,
    turboquant_enabled,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TurboQuantKVCache",
    "BatchTurboQuantKVCache",
    "turboquant_enabled",
]


# ---------------------------------------------------------------------------
# Batch-level state helpers (axis-0 operations)
# ---------------------------------------------------------------------------

def _filter_state(state, indices):
    """Index-select along batch dimension (axis 0)."""
    if state is None:
        return None
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(
            state.norms[indices],
            state.indices[indices],
        )
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            state.norms[indices],
            state.mse_indices[indices],
            state.residual_norms[indices],
            state.qjl_signs[indices],
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            state.radii[indices],
            tuple(level[indices] for level in state.level_indices),
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            state.norms[indices],
            _filter_state(state.polar_state, indices),
            state.residual_norms[indices],
            state.qjl_signs[indices],
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _filter_state(state.low, indices),
            _filter_state(state.high, indices),
        )
    raise TypeError(f"Unsupported state type: {type(state)!r}")


def _concat_state_batch(states):
    """Concatenate a list of states along batch dimension (axis 0)."""
    if not states:
        return None
    first = states[0]
    if isinstance(first, TurboQuantMSEState):
        return TurboQuantMSEState(
            mx.concatenate([s.norms for s in states], axis=0),
            mx.concatenate([s.indices for s in states], axis=0),
        )
    if isinstance(first, TurboQuantProdState):
        return TurboQuantProdState(
            mx.concatenate([s.norms for s in states], axis=0),
            mx.concatenate([s.mse_indices for s in states], axis=0),
            mx.concatenate([s.residual_norms for s in states], axis=0),
            mx.concatenate([s.qjl_signs for s in states], axis=0),
        )
    if isinstance(first, TurboQuantPolarState):
        return TurboQuantPolarState(
            mx.concatenate([s.radii for s in states], axis=0),
            tuple(
                mx.concatenate([states[j].level_indices[i] for j in range(len(states))], axis=0)
                for i in range(len(first.level_indices))
            ),
        )
    if isinstance(first, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            mx.concatenate([s.norms for s in states], axis=0),
            _concat_state_batch([s.polar_state for s in states]),
            mx.concatenate([s.residual_norms for s in states], axis=0),
            mx.concatenate([s.qjl_signs for s in states], axis=0),
        )
    if isinstance(first, TurboQuantSplitState):
        return TurboQuantSplitState(
            _concat_state_batch([s.low for s in states]),
            _concat_state_batch([s.high for s in states]),
        )
    raise TypeError(f"Unsupported state type: {type(first)!r}")


def _pad_state_left(state, pad_length: int):
    """Prepend zeros along the token dimension (axis 2) of a state."""
    if state is None or pad_length <= 0:
        return state
    pad = _allocate_state_like(state, pad_length)
    return _concat_state(pad, state)


# ---------------------------------------------------------------------------
# BatchTurboQuantKVCache
# ---------------------------------------------------------------------------

class BatchTurboQuantKVCache(_BaseCache):
    """Batched TurboQuant KV cache for omlx continuous-batching scheduler.

    Stores fp16 during prefill (identical to BatchKVCache), then quantizes
    on the first decode token using mlx-vlm's TurboQuantKVCache codecs.
    """

    step = 256

    def __init__(self, left_padding: List[int], bits: float = 4.0, seed: int = 0):
        self.bits = _validate_bits(bits)
        self.seed = seed
        # Prevent AttributeError in mlx-lm's base.py SDPA which checks
        # hasattr(cache, "bits") and then accesses cache.group_size
        self.group_size = 0

        # fp16 storage (prefill phase)
        self._fp16_keys = None
        self._fp16_values = None

        # Quantized NamedTuple storage (decode phase)
        self._key_state = None
        self._value_state = None
        self._key_codec = None
        self._value_codec = None
        self._quantized = False

        # Batch tracking
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self._right_padding = None

    # ---- codec management --------------------------------------------------

    def _ensure_codecs(self, keys: mx.array, values: mx.array):
        if self._key_codec is None:
            key_bits = (
                math.floor(self.bits)
                if not math.isclose(self.bits, round(self.bits), abs_tol=1e-6)
                else self.bits
            )
            self._key_codec = _build_codec(keys, key_bits, mode="prod", seed=self.seed)
        if self._value_codec is None:
            val_bits = (
                math.ceil(self.bits)
                if not math.isclose(self.bits, round(self.bits), abs_tol=1e-6)
                else self.bits
            )
            self._value_codec = _build_codec(
                values, val_bits, mode="mse", seed=self.seed + 1
            )

    # ---- quantize fp16 buffer -> NamedTuple states -------------------------

    def _quantize_buffer(self):
        """Convert fp16 KV to quantized NamedTuple states. Called on first decode."""
        if self._quantized or self._fp16_keys is None:
            return
        k = self._fp16_keys[..., : self._idx, :]
        v = self._fp16_values[..., : self._idx, :]
        self._ensure_codecs(k, v)
        self._key_state = self._key_codec.quantize(k)
        self._value_state = self._value_codec.quantize(v)
        mx.eval(self._key_state, self._value_state)
        self._quantized = True
        self._fp16_keys = None
        self._fp16_values = None

    # ---- update_and_fetch --------------------------------------------------

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, T_new, D = keys.shape

        if T_new > 1:
            # Prefill: fp16 storage (same as BatchKVCache)
            prev = self._idx
            if self._fp16_keys is None or (prev + T_new) > self._fp16_keys.shape[2]:
                n_steps = (self.step + T_new - 1) // self.step
                k_shape = (B, H, n_steps * self.step, D)
                v_shape = (B, H, n_steps * self.step, values.shape[3])
                new_k = mx.zeros(k_shape, keys.dtype)
                new_v = mx.zeros(v_shape, values.dtype)
                if self._fp16_keys is not None:
                    if prev % self.step != 0:
                        self._fp16_keys = self._fp16_keys[..., :prev, :]
                        self._fp16_values = self._fp16_values[..., :prev, :]
                    self._fp16_keys = mx.concatenate(
                        [self._fp16_keys, new_k], axis=2
                    )
                    self._fp16_values = mx.concatenate(
                        [self._fp16_values, new_v], axis=2
                    )
                else:
                    self._fp16_keys, self._fp16_values = new_k, new_v

            self.offset += T_new
            self._idx += T_new
            self._fp16_keys[..., prev : self._idx, :] = keys
            self._fp16_values[..., prev : self._idx, :] = values
            return (
                self._fp16_keys[..., : self._idx, :],
                self._fp16_values[..., : self._idx, :],
            )
        else:
            # Decode: quantize on first token, then append
            if not self._quantized:
                self._quantize_buffer()
            self._ensure_codecs(keys, values)

            new_k = self._key_codec.quantize(keys)
            new_v = self._value_codec.quantize(values)
            new_end = self._idx + 1

            if self._key_state is None:
                self._key_state = _allocate_state_like(new_k, new_end)
                self._value_state = _allocate_state_like(new_v, new_end)
            else:
                self._key_state = _reserve_state_capacity(
                    self._key_state, self._idx, new_end, self.step
                )
                self._value_state = _reserve_state_capacity(
                    self._value_state, self._idx, new_end, self.step
                )

            _write_state(self._key_state, new_k, self._idx)
            _write_state(self._value_state, new_v, self._idx)
            self.offset += 1
            self._idx = new_end

            # Return proxied states for model mask slicing
            ks = _slice_state(self._key_state, self._idx)
            vs = _slice_state(self._value_state, self._idx)
            n_heads = keys.shape[1]
            return (
                _QuantizedStateProxy(ks, self._idx, n_heads),
                _QuantizedStateProxy(vs, self._idx, n_heads),
            )

    # ---- attention ---------------------------------------------------------

    def decode_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask=None,
    ) -> mx.array:
        """Delegate to TurboQuantKVCache's decode_attention via a temp cache."""
        tmp = TurboQuantKVCache(bits=self.bits, seed=self.seed)
        tmp.key_codec = self._key_codec
        tmp.value_codec = self._value_codec
        ks = _slice_state(self._key_state, self._idx) if keys_state is None else (
            keys_state._state if isinstance(keys_state, _QuantizedStateProxy) else keys_state
        )
        vs = _slice_state(self._value_state, self._idx) if values_state is None else (
            values_state._state if isinstance(values_state, _QuantizedStateProxy) else values_state
        )
        tmp.keys = ks
        tmp.values = vs
        tmp.offset = self._idx
        return tmp.decode_attention(
            queries, keys_state=ks, values_state=vs, scale=scale, mask=mask
        )

    def prefill_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask=None,
    ) -> Optional[mx.array]:
        if not self._quantized or self._key_codec is None:
            return None
        tmp = TurboQuantKVCache(bits=self.bits, seed=self.seed)
        tmp.key_codec = self._key_codec
        tmp.value_codec = self._value_codec
        ks = _slice_state(self._key_state, self._idx) if keys_state is None else (
            keys_state._state if isinstance(keys_state, _QuantizedStateProxy) else keys_state
        )
        vs = _slice_state(self._value_state, self._idx) if values_state is None else (
            values_state._state if isinstance(values_state, _QuantizedStateProxy) else values_state
        )
        tmp.keys = ks
        tmp.values = vs
        tmp.offset = self._idx
        return tmp.prefill_attention(
            queries, keys_state=ks, values_state=vs, scale=scale, mask=mask
        )

    def dequantize(self, keys_state=None, values_state=None):
        if not self._quantized:
            # Still in fp16 mode
            k = self._fp16_keys[..., : self._idx, :]
            v = self._fp16_values[..., : self._idx, :]
            return k, v
        ks = _slice_state(self._key_state, self._idx) if keys_state is None else (
            keys_state._state if isinstance(keys_state, _QuantizedStateProxy) else keys_state
        )
        vs = _slice_state(self._value_state, self._idx) if values_state is None else (
            values_state._state if isinstance(values_state, _QuantizedStateProxy) else values_state
        )
        keys = self._key_codec.dequantize(ks).astype(mx.float32)
        values = self._value_codec.dequantize(vs).astype(mx.float32)
        return keys, values

    # ---- batch operations --------------------------------------------------

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self._fp16_keys is not None or self._quantized:
                raise ValueError(
                    "Left padding can only be added to an empty BatchTurboQuantKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None and not self._quantized:
            padding = self._right_padding
            self._fp16_keys = dynamic_roll(self._fp16_keys, padding[:, None], axis=2)
            self._fp16_values = dynamic_roll(
                self._fp16_values, padding[:, None], axis=2
            )
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    def filter(self, batch_indices):
        if self._quantized:
            self._key_state = _filter_state(self._key_state, batch_indices)
            self._value_state = _filter_state(self._value_state, batch_indices)
        else:
            if self._fp16_keys is not None:
                self._fp16_keys = self._fp16_keys[batch_indices]
                self._fp16_values = self._fp16_values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        # Remove common left padding (fp16 only)
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0 and not self._quantized:
            self._fp16_keys = self._fp16_keys[..., min_left_pad:, :]
            self._fp16_values = self._fp16_values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other: "BatchTurboQuantKVCache"):
        # Force both to quantized if mixed
        if self._quantized != other._quantized:
            if not self._quantized:
                self._quantize_buffer()
            if not other._quantized:
                other._quantize_buffer()

        if self._quantized:
            max_idx = max(self._idx, other._idx)

            def _pad_and_trim(c):
                ks = _slice_state(c._key_state, c._idx)
                vs = _slice_state(c._value_state, c._idx)
                left = max_idx - c._idx
                if left > 0:
                    ks = _pad_state_left(ks, left)
                    vs = _pad_state_left(vs, left)
                return ks, vs, c.offset, c.left_padding + left

            s_ks, s_vs, s_off, s_lp = _pad_and_trim(self)
            o_ks, o_vs, o_off, o_lp = _pad_and_trim(other)
            self._key_state = _concat_state_batch([s_ks, o_ks])
            self._value_state = _concat_state_batch([s_vs, o_vs])
            self.offset = mx.concatenate([s_off, o_off])
            self.left_padding = mx.concatenate([s_lp, o_lp])
            self._idx = max_idx
            # Share codecs
            if self._key_codec is None:
                self._key_codec = other._key_codec
                self._value_codec = other._value_codec
        else:
            # fp16 extend (same as BatchKVCache)
            max_idx = max(self._idx, other._idx)
            max_size = max(self._fp16_keys.shape[2], other._fp16_keys.shape[2])

            def pad(c):
                left = max_idx - c._idx
                right = max_size - c._fp16_keys.shape[2] - left
                k, v = c._fp16_keys, c._fp16_values
                if right < 0:
                    k = k[..., :right, :]
                    v = v[..., :right, :]
                    right = 0
                if left != 0 or right != 0:
                    p = [(0, 0), (0, 0), (left, right), (0, 0)]
                    k = mx.pad(k, p)
                    v = mx.pad(v, p)
                return k, v, c.offset, c.left_padding + left

            self._fp16_keys, self._fp16_values, self.offset, self.left_padding = map(
                mx.concatenate, zip(*(pad(self), pad(other)))
            )
            self._idx = max_idx

    def extract(self, idx: int) -> TurboQuantKVCache:
        if not self._quantized:
            self._quantize_buffer()

        padding = self.left_padding[idx].item()
        end = self._idx
        tq = TurboQuantKVCache(bits=self.bits, seed=self.seed)

        ks = _slice_state_range(self._key_state, padding, end)
        vs = _slice_state_range(self._value_state, padding, end)
        tq.keys = _filter_state(ks, slice(idx, idx + 1))
        tq.values = _filter_state(vs, slice(idx, idx + 1))
        tq.offset = end - padding
        tq.key_codec = self._key_codec
        tq.value_codec = self._value_codec
        return tq

    @classmethod
    def merge(cls, caches: List[TurboQuantKVCache]) -> "BatchTurboQuantKVCache":
        bits = caches[0].bits
        seed = caches[0].seed
        lengths = [c.offset for c in caches]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]

        batch = cls(padding, bits=bits, seed=seed)

        # Share codecs from first cache that has them
        for c in caches:
            if c.key_codec is not None:
                batch._key_codec = c.key_codec
                batch._value_codec = c.value_codec
                break

        # Collect per-request states, left-pad to max_length
        key_states = []
        value_states = []
        for p, c in zip(padding, caches):
            ks, vs = c.state
            if ks is None:
                continue
            ks = ks._state if isinstance(ks, _QuantizedStateProxy) else ks
            vs = vs._state if isinstance(vs, _QuantizedStateProxy) else vs
            if p > 0:
                ks = _pad_state_left(ks, p)
                vs = _pad_state_left(vs, p)
            key_states.append(ks)
            value_states.append(vs)

        if key_states:
            batch._key_state = _concat_state_batch(key_states)
            batch._value_state = _concat_state_batch(value_states)
            mx.eval(batch._key_state, batch._value_state)

        batch.offset += max_length
        batch._idx = max_length
        batch._quantized = True
        return batch

    # ---- state / properties ------------------------------------------------

    @property
    def state(self):
        if self._quantized:
            ks = _slice_state(self._key_state, self._idx)
            vs = _slice_state(self._value_state, self._idx)
            n_heads = ks.norms.shape[1] if hasattr(ks, "norms") else (
                ks.low.norms.shape[1] if isinstance(ks, TurboQuantSplitState) else 1
            )
            return (
                _QuantizedStateProxy(ks, self._idx, n_heads),
                _QuantizedStateProxy(vs, self._idx, n_heads),
                self.offset,
                self.left_padding,
            )
        if self._fp16_keys is None:
            return None, None, self.offset, self.left_padding
        k = self._fp16_keys[..., : self._idx, :]
        v = self._fp16_values[..., : self._idx, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        if v is None:
            self._fp16_keys = self._fp16_values = None
            self._key_state = self._value_state = None
            self._quantized = False
            self._idx = 0
            return
        if len(v) == 4:
            first = v[0]
            if isinstance(first, _QuantizedStateProxy) or (
                first is not None and not isinstance(first, mx.array)
            ):
                # Quantized state
                ks = first._state if isinstance(first, _QuantizedStateProxy) else first
                vs = v[1]._state if isinstance(v[1], _QuantizedStateProxy) else v[1]
                self._key_state = ks
                self._value_state = vs
                self.offset = v[2]
                self.left_padding = v[3]
                self._idx = _state_length(ks) if ks is not None else 0
                self._quantized = True
            else:
                self._fp16_keys = v[0]
                self._fp16_values = v[1]
                self.offset = v[2]
                self.left_padding = v[3]
                self._idx = self._fp16_keys.shape[2] if self._fp16_keys is not None else 0
                self._quantized = False

    @property
    def meta_state(self):
        return tuple(map(str, (self._idx, self.bits, self.seed)))

    @meta_state.setter
    def meta_state(self, v):
        self._idx = int(v[0])
        self.bits = float(v[1])
        self.seed = int(v[2])

    def size(self):
        return self.offset

    def empty(self):
        return self._fp16_keys is None and self._key_state is None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    @property
    def nbytes(self):
        if self._quantized:
            return _state_nbytes(self._key_state) + _state_nbytes(self._value_state)
        if self._fp16_keys is not None:
            return self._fp16_keys.nbytes + self._fp16_values.nbytes
        return 0
