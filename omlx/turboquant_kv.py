# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV cache — thin wrapper around mlx_vlm.turboquant.

Core implementation (codecs, Metal kernels, TurboQuantKVCache) lives in
mlx-vlm.  This module re-exports the public API and adds
BatchTurboQuantKVCache (inherits TurboQuantKVCache) for omlx's
continuous-batching scheduler.
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
# Codec rebuild for SSD cache reconstruction
# ---------------------------------------------------------------------------


def _infer_head_dim(state, bits: int) -> int:
    """Infer head_dim from a TQ quantized state's packed tensor width.

    MSEState.indices has shape (..., packed_width) where
    packed_width = ceil(head_dim * bits / 32).
    """
    if isinstance(state, TurboQuantMSEState):
        packed_width = state.indices.shape[-1]
    elif isinstance(state, TurboQuantProdState):
        packed_width = state.mse_indices.shape[-1]
        bits = max(bits - 1, 1)
    else:
        raise TypeError(f"Cannot infer head_dim from state type: {type(state).__name__}")
    return packed_width * 32 // bits


def _rebuild_codecs(tq_cache: TurboQuantKVCache, key_state, value_state) -> None:
    """Rebuild TQ codecs deterministically from (head_dim, bits, seed).

    TQ codecs (rotation matrices, codebooks) are fully determined by
    (head_dim, bits, seed) for integer bit-widths — no data dependency.
    This allows rebuilding codecs without the original fp16 tensors,
    which is needed when reconstructing from SSD cache.
    """
    bits = tq_cache.bits
    seed = tq_cache.seed
    fractional = not math.isclose(bits, round(bits), abs_tol=1e-6)
    key_bits = int(math.floor(bits) if fractional else bits)
    val_bits = int(math.ceil(bits) if fractional else bits)

    head_dim = _infer_head_dim(key_state, key_bits)

    dummy = mx.zeros((1, 1, 1, head_dim))
    tq_cache.key_codec = _build_codec(dummy, key_bits, mode="mse", seed=seed)
    tq_cache.value_codec = _build_codec(dummy, val_bits, mode="mse", seed=seed + 1)


# ---------------------------------------------------------------------------
# Batch-level state helpers (axis-0 operations)
# ---------------------------------------------------------------------------

def _filter_state(state, indices):
    """Index-select along batch dimension (axis 0)."""
    if state is None:
        return None
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(state.norms[indices], state.indices[indices])
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            state.norms[indices], state.mse_indices[indices],
            state.residual_norms[indices], state.qjl_signs[indices],
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            state.radii[indices],
            tuple(level[indices] for level in state.level_indices),
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            state.norms[indices], _filter_state(state.polar_state, indices),
            state.residual_norms[indices], state.qjl_signs[indices],
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _filter_state(state.low, indices), _filter_state(state.high, indices),
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
# BatchTurboQuantKVCache — inherits TurboQuantKVCache
# ---------------------------------------------------------------------------

class BatchTurboQuantKVCache(TurboQuantKVCache):
    """TurboQuantKVCache with batch operations for continuous batching.

    Inherits update_and_fetch, decode_attention, _ensure_codecs, state,
    and all decode logic from TurboQuantKVCache with ZERO overhead.
    Only adds batch-specific methods (merge/extract/extend/filter) and
    overrides make_mask for per-request left_padding support.
    """

    def __init__(self, left_padding: List[int], bits: float = 4.0, seed: int = 0):
        super().__init__(bits=bits, seed=seed)
        self.group_size = 0
        self.left_padding = mx.array(left_padding)
        self._batch_size = len(left_padding)
        # B=1: offset is int (parent-compatible, zero overhead decode)
        # B>1: offset is mx.array (per-request, needs override)
        if self._batch_size > 1:
            self.offset = mx.array([-l for l in left_padding])
        else:
            self.offset = -left_padding[0]
        self._right_padding = None

    # ---- update_and_fetch override for B>1 only ----------------------------

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if isinstance(self.offset, int):
            # B=1: parent's method directly (zero overhead)
            return super().update_and_fetch(keys, values)
        # B>1: track per-request offset separately from state offset
        T_new = keys.shape[2]
        # Use int offset for state management
        int_offset = self.offset.max().item()
        self.offset += T_new
        saved_offset = self.offset
        self.offset = int_offset
        result = super().update_and_fetch(keys, values)
        self.offset = saved_offset
        return result

    # ---- state override for B>1 (offset is mx.array) -----------------------

    @property
    def state(self):
        if isinstance(self.offset, int):
            return super().state
        # B>1: use keys length directly (offset is mx.array, can't compare with int)
        if self.keys is None:
            return None, None
        length = _state_length(self.keys)
        return _slice_state(self.keys, length), _slice_state(self.values, length)

    @state.setter
    def state(self, value):
        TurboQuantKVCache.state.fset(self, value)

    # ---- make_mask override (batch-aware) ----------------------------------

    def make_mask(self, *args, **kwargs):
        if isinstance(self.offset, int):
            return create_attention_mask(*args, offset=self.offset, **kwargs)
        return create_causal_mask(
            args[0], offset=self.offset, left_padding=self.left_padding, **kwargs
        )

    # prefill_attention and dequantize inherited from TurboQuantKVCache

    # ---- batch operations --------------------------------------------------

    def _ensure_array_offset(self):
        if isinstance(self.offset, int):
            self.offset = mx.array([self.offset])

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchTurboQuantKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding if isinstance(self.offset, mx.array) else left_padding[0].item()
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is None:
            return
        padding = self._right_padding
        if self.keys is not None:
            k_fp16, v_fp16 = self.dequantize()
            k_rolled = dynamic_roll(k_fp16, padding[:, None], axis=2)
            v_rolled = dynamic_roll(v_fp16, padding[:, None], axis=2)
            self.keys = self.key_codec.quantize(k_rolled)
            self.values = self.value_codec.quantize(v_rolled)
            mx.eval(self.keys, self.values)
        self.offset -= padding if isinstance(self.offset, mx.array) else padding[0].item()
        self.left_padding += padding
        self._right_padding = None

    def filter(self, batch_indices):
        self._ensure_array_offset()
        if self.keys is not None:
            self.keys = _filter_state(self.keys, batch_indices)
            self.values = _filter_state(self.values, batch_indices)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        self._cached_state = None
        self._cached_state_offset = -1

    def extend(self, other: "BatchTurboQuantKVCache"):
        self._ensure_array_offset()
        other._ensure_array_offset()
        max_off = max(self.offset.max().item(), other.offset.max().item())
        # Use the underlying int offset (total tokens) for state operations
        s_idx = _state_length(self.keys) if self.keys is not None else 0
        o_idx = _state_length(other.keys) if other.keys is not None else 0
        max_idx = max(s_idx, o_idx)

        def _pad_and_trim(c, idx):
            ks = _slice_state(c.keys, idx) if c.keys is not None else None
            vs = _slice_state(c.values, idx) if c.values is not None else None
            left = max_idx - idx
            if left > 0 and ks is not None:
                ks = _pad_state_left(ks, left)
                vs = _pad_state_left(vs, left)
            return ks, vs, c.offset, c.left_padding + left

        s_ks, s_vs, s_off, s_lp = _pad_and_trim(self, s_idx)
        o_ks, o_vs, o_off, o_lp = _pad_and_trim(other, o_idx)

        if s_ks is not None and o_ks is not None:
            self.keys = _concat_state_batch([s_ks, o_ks])
            self.values = _concat_state_batch([s_vs, o_vs])
        elif o_ks is not None:
            self.keys = o_ks
            self.values = o_vs

        self.offset = mx.concatenate([s_off, o_off])
        self.left_padding = mx.concatenate([s_lp, o_lp])
        # Parent's offset is used for state length — set to max
        # (state property uses self.offset for slicing)
        self._cached_state = None
        self._cached_state_offset = -1

        if self.key_codec is None:
            self.key_codec = other.key_codec
            self.value_codec = other.value_codec

    def extract(self, idx: int) -> TurboQuantKVCache:
        padding = self.left_padding[idx].item()
        total = self.offset[idx].item() if isinstance(self.offset, mx.array) else self.offset
        end = padding + total

        tq = TurboQuantKVCache(bits=self.bits, seed=self.seed)
        if self.keys is not None:
            ks = _slice_state_range(self.keys, padding, end)
            vs = _slice_state_range(self.values, padding, end)
            tq.keys = _filter_state(ks, slice(idx, idx + 1))
            tq.values = _filter_state(vs, slice(idx, idx + 1))
            tq.offset = total
        tq.key_codec = self.key_codec
        tq.value_codec = self.value_codec
        return tq

    @classmethod
    def merge(cls, caches: List[TurboQuantKVCache]) -> "BatchTurboQuantKVCache":
        bits = caches[0].bits
        seed = caches[0].seed
        lengths = [c.offset for c in caches]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]

        batch = cls(padding, bits=bits, seed=seed)

        for c in caches:
            if c.key_codec is not None:
                batch.key_codec = c.key_codec
                batch.value_codec = c.value_codec
                break

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
            batch.keys = _concat_state_batch(key_states)
            batch.values = _concat_state_batch(value_states)
            mx.eval(batch.keys, batch.values)

        batch.offset += max_length
        return batch
