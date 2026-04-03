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

    Quantizes immediately on every update_and_fetch call (both prefill and
    decode), so the full-size fp16 KV buffer never exists.  This reduces peak
    memory during prefill by ~60-75% compared to the old approach that stored
    fp16 during prefill and quantized only on the first decode token.
    """

    step = 256

    def __init__(self, left_padding: List[int], bits: float = 4.0, seed: int = 0):
        self.bits = _validate_bits(bits)
        self.seed = seed
        # Prevent AttributeError in mlx-lm's base.py SDPA which checks
        # hasattr(cache, "bits") and then accesses cache.group_size
        self.group_size = 0

        # Quantized NamedTuple storage (always used)
        self._key_state = None
        self._value_state = None
        self._key_codec = None
        self._value_codec = None

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

    # ---- update_and_fetch --------------------------------------------------

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, T_new, D = keys.shape
        self._ensure_codecs(keys, values)

        new_k = self._key_codec.quantize(keys)
        new_v = self._value_codec.quantize(values)

        # Append via concat (no in-place _write_state which can hang mx.eval)
        if self._key_state is None:
            self._key_state = new_k
            self._value_state = new_v
        else:
            self._key_state = _concat_state(self._key_state, new_k)
            self._value_state = _concat_state(self._value_state, new_v)

        self.offset += T_new
        self._idx += T_new

        if T_new > 1:
            return keys, values
        else:
            return (
                _QuantizedStateProxy(self._key_state, self._idx, H),
                _QuantizedStateProxy(self._value_state, self._idx, H),
            )

    # ---- attention ---------------------------------------------------------

    def _make_tmp_cache(self):
        """Create a temporary TurboQuantKVCache sharing our codecs and state."""
        tmp = TurboQuantKVCache(bits=self.bits, seed=self.seed)
        tmp.key_codec = self._key_codec
        tmp.value_codec = self._value_codec
        tmp.keys = self._key_state
        tmp.values = self._value_state
        tmp.offset = self._idx
        return tmp

    def decode_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask=None,
    ) -> mx.array:
        tmp = self._make_tmp_cache()
        return tmp.decode_attention(queries, scale=scale, mask=mask)

    def prefill_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask=None,
    ) -> Optional[mx.array]:
        # TODO: re-enable once Metal kernel hang with D=256 is resolved
        return None

    def dequantize(self, keys_state=None, values_state=None):
        keys = self._key_codec.dequantize(self._key_state).astype(mx.float32)
        values = self._value_codec.dequantize(self._value_state).astype(mx.float32)
        return keys, values

    # ---- batch operations --------------------------------------------------

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self._key_state is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchTurboQuantKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is None:
            return
        padding = self._right_padding
        if self._key_state is not None:
            # Dequantize → roll → re-quantize (one-time cost, merge path only)
            k_fp16, v_fp16 = self.dequantize()
            k_rolled = dynamic_roll(k_fp16, padding[:, None], axis=2)
            v_rolled = dynamic_roll(v_fp16, padding[:, None], axis=2)
            self._key_state = self._key_codec.quantize(k_rolled)
            self._value_state = self._value_codec.quantize(v_rolled)
            mx.eval(self._key_state, self._value_state)
        self.offset -= padding
        self.left_padding += padding
        self._right_padding = None

    def filter(self, batch_indices):
        if self._key_state is not None:
            self._key_state = _filter_state(self._key_state, batch_indices)
            self._value_state = _filter_state(self._value_state, batch_indices)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

    def extend(self, other: "BatchTurboQuantKVCache"):
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

    def extract(self, idx: int) -> TurboQuantKVCache:
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
        return batch

    # ---- state / properties ------------------------------------------------

    @property
    def state(self):
        if self._key_state is not None:
            return self._key_state, self._value_state, self.offset, self.left_padding
        return None, None, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        if v is None:
            self._key_state = self._value_state = None
            self._idx = 0
            return
        if len(v) == 4:
            first = v[0]
            if first is None:
                self._key_state = self._value_state = None
                self.offset = v[2]
                self.left_padding = v[3]
                self._idx = 0
            else:
                self._key_state = first
                self._value_state = v[1]
                self.offset = v[2]
                self.left_padding = v[3]
                self._idx = _state_length(first) if first is not None else 0

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
        return self._key_state is None

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
        if self._key_state is not None:
            return _state_nbytes(self._key_state) + _state_nbytes(self._value_state)
        return 0
