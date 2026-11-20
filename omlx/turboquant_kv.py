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
    TurboQuantPolarProdState,
    TurboQuantPolarState,
    TurboQuantProdState,
    TurboQuantSplitState,
    _build_codec,
    _concat_state,
    _QuantizedStateProxy,
    _reserve_state_capacity,
    _slice_state,
    _slice_state_range,
    _state_length,
    _state_nbytes,
    _validate_bits,
    _write_state,
    turboquant_enabled,
)

logger = logging.getLogger(__name__)


@classmethod
def _from_cache(cls, cache, bits, seed=0):
    result = cls(bits=bits, seed=seed)
    if callable(getattr(cache, "keys_and_values", None)):
        keys, values = (
            cache.keys_and_values() if cache.keys is not None else (None, None)
        )
    else:
        keys, values = cache.state
    if keys is not None:
        result.update_and_fetch(keys, values)
    return result


TurboQuantKVCache.from_cache = _from_cache


__all__ = [
    "TurboQuantKVCache",
    "BatchTurboQuantKVCache",
    "turboquant_enabled",
]


def quantized_cache_class(scheme: str = "turboquant"):
    """Resolve the packed cache format without loading native kernels."""
    if scheme == "turboquant":
        return TurboQuantKVCache
    if scheme in ("affine4", "affine8"):
        from .affine4 import Affine4KVCache, Affine8KVCache

        return Affine8KVCache if scheme == "affine8" else Affine4KVCache
    raise ValueError(f"Unknown KV cache quantization scheme: {scheme}")


def _quantization_config(cache):
    return (getattr(cache, "quantization_scheme", "turboquant"), cache.bits, cache.seed)


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
        raise TypeError(
            f"Cannot infer head_dim from state type: {type(state).__name__}"
        )
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


def _concat_state_token_axis(states):
    """Concatenate TurboQuant states along token axis with a low-churn fast path."""
    if not states:
        return None
    if len(states) == 1:
        state = states[0]
        return state._state if isinstance(state, _QuantizedStateProxy) else state

    unwrapped = [
        state._state if isinstance(state, _QuantizedStateProxy) else state
        for state in states
    ]
    first = unwrapped[0]
    if isinstance(first, TurboQuantMSEState) and all(
        isinstance(state, TurboQuantMSEState) for state in unwrapped
    ):
        return TurboQuantMSEState(
            mx.concatenate([state.norms for state in unwrapped], axis=2),
            mx.concatenate([state.indices for state in unwrapped], axis=2),
        )

    result = first
    for state in unwrapped[1:]:
        result = _concat_state(result, state)
    return result


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
                mx.concatenate(
                    [states[j].level_indices[i] for j in range(len(states))], axis=0
                )
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


# Each growth copies the whole packed state, so long caches grow by 1/64 of
# their length instead of the stock 256 tokens.
_MIN_GROWTH_STEP = 256


def _growth_step(tokens: int) -> int:
    """Capacity step of a quantized cache that holds ``tokens``."""
    return max(_MIN_GROWTH_STEP, (tokens >> 6) // _MIN_GROWTH_STEP * _MIN_GROWTH_STEP)


# Room for a decode step or a verify block without a growth copy.
_MIN_SPARE_TOKENS = 64


def _spare_capacity(tokens: int) -> int:
    """Next growth-step boundary past ``tokens + _MIN_SPARE_TOKENS``."""
    step = _growth_step(tokens)
    return (tokens + _MIN_SPARE_TOKENS + step - 1) // step * step


def _state_rows(state) -> int:
    """Batch rows of a packed state."""
    first = state[0]
    return int(first.shape[0]) if isinstance(first, mx.array) else _state_rows(first)


def _allocate_state_rows(state, rows: int, capacity: int):
    """Zero state shaped like ``state`` with ``rows`` rows and ``capacity`` tokens."""
    fields = [
        (
            mx.zeros((rows, a.shape[1], capacity, *a.shape[3:]), dtype=a.dtype)
            if isinstance(a, mx.array)
            else _allocate_state_rows(a, rows, capacity)
        )
        for a in state
    ]
    return type(state)(*fields) if hasattr(state, "_fields") else tuple(fields)


def _write_state_block(dst, src, row: int, start: int) -> None:
    """Write ``src`` into ``dst`` at batch row ``row`` and token column ``start``."""
    for dst_field, src_field in zip(dst, src):
        if isinstance(dst_field, mx.array):
            rows, length = src_field.shape[0], src_field.shape[2]
            dst_field[row : row + rows, :, start : start + length] = src_field
        else:
            _write_state_block(dst_field, src_field, row, start)


def _place_states(reference, placements, rows: int, capacity: int):
    """Write ``(state, row, start)`` placements into one zero buffer."""
    out = _allocate_state_rows(reference, rows, capacity)
    for state, row, start in placements:
        if state is not None:
            _write_state_block(out, state, row, start)
    return out


def _join_state_token_axis(states, capacity: int):
    """Concatenate states along the token axis into a buffer of ``capacity``."""
    placements = []
    start = 0
    for state in states:
        placements.append((state, 0, start))
        start += _state_length(state)
    return _place_states(states[0], placements, _state_rows(states[0]), capacity)

def _write_state_rows(dst, src, starts):
    """Write row ``b`` of ``src`` at token column ``starts[b]`` of ``dst``."""
    length = _state_length(src)
    for dst_field, src_field in zip(dst, src):
        if isinstance(dst_field, mx.array):
            for row, start in enumerate(starts):
                dst_field[row : row + 1, :, start : start + length] = src_field[
                    row : row + 1
                ]
        else:
            _write_state_rows(dst_field, src_field, starts)




def _shift_state_rows(state, moves, end: int):
    """Shift each ``(row, start, shift)`` row's live tokens right to end at ``end``."""
    for field in state:
        if isinstance(field, mx.array):
            for row, start, shift in moves:
                # A real copy of the source window: it overlaps the target.
                window = mx.array(field[row : row + 1, :, start : end - shift]) + 0
                field[row : row + 1, :, start + shift : end] = window
        else:
            _shift_state_rows(field, moves, end)


def _roll_state(state, shifts, axis: int = 2):
    """Roll a quantized state along its token axis by per-row ``shifts``.

    Mirrors dynamic_roll's per-batch-row dynamic shift (same axis/shifts
    convention as _slice_state/_concat_state above), applied directly to the
    packed representation instead of dequantize -> roll -> requantize: roll
    is pure reindexing along the token axis, so applying it independently to
    every field that shares that axis (norms, packed indices, radii, ...)
    preserves their per-token pairing exactly, with no fp16 materialization
    and no re-rounding through the codec on the way back in.
    """
    if state is None:
        return None
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(
            dynamic_roll(state.norms, shifts, axis=axis),
            dynamic_roll(state.indices, shifts, axis=axis),
        )
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            dynamic_roll(state.norms, shifts, axis=axis),
            dynamic_roll(state.mse_indices, shifts, axis=axis),
            dynamic_roll(state.residual_norms, shifts, axis=axis),
            dynamic_roll(state.qjl_signs, shifts, axis=axis),
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            dynamic_roll(state.radii, shifts, axis=axis),
            tuple(
                dynamic_roll(level, shifts, axis=axis)
                for level in state.level_indices
            ),
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            dynamic_roll(state.norms, shifts, axis=axis),
            _roll_state(state.polar_state, shifts, axis=axis),
            dynamic_roll(state.residual_norms, shifts, axis=axis),
            dynamic_roll(state.qjl_signs, shifts, axis=axis),
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _roll_state(state.low, shifts, axis=axis),
            _roll_state(state.high, shifts, axis=axis),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


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

    _left_padded: bool | None = None

    # Reassign, never mutate in place: the setter clears the cached flag.
    @property
    def left_padding(self):
        return self._left_padding

    @left_padding.setter
    def left_padding(self, value):
        self._left_padding = value
        self._left_padded = None

    def _has_left_padding(self) -> bool:
        # Cached because a device read waits for all queued GPU work.
        if self._left_padded is None:
            self._left_padded = bool(mx.any(self._left_padding).item())
        return self._left_padded

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
        # Written physical column count (B>1 only; B=1 uses the parent's int
        # offset). Rows are end-aligned, but this must NOT be derived from
        # offset.max(): once filter() removes the last zero-left-padding row,
        # every logical offset is short of the written end, and an
        # offset-derived position writes INSIDE the survivors' live KV. It
        # also must not be derived from _state_length(self.keys): that is the
        # step-allocated capacity, not the written end.
        self._phys_end = 0
        # Per-row written ends after trim_rows (host ints); None: all end at
        # _phys_end.
        self._row_ends: list[int] | None = None

    @property
    def _idx(self):
        # mlx-vlm uses this cursor for rollback and padded attention masks.
        return self.offset if isinstance(self.offset, int) else self._phys_end

    @property
    def cache_step(self):
        return _growth_step(_state_length(self.keys) if self.keys is not None else 0)

    def trim(self, n):
        if self._row_ends is not None:
            return self.trim_rows([n] * len(self._row_ends))
        position = self._idx
        n = min(position, n)
        self.offset -= n
        self._phys_end = position - n
        self._cached_state = None
        self._cached_state_offset = -1
        return n

    # Rows drop different counts in place; the mask hides columns past each end.
    _omlx_ragged_trim = True

    def _ends(self) -> list[int]:
        if self._row_ends is not None:
            return list(self._row_ends)
        return [self._phys_end] * int(self.offset.shape[0])

    def _set_ends(self, ends: list[int]) -> None:
        self._phys_end = max(ends) if ends else 0
        self._row_ends = None if len(set(ends)) <= 1 else list(ends)
        self._cached_state = None
        self._cached_state_offset = -1

    def trim_rows(self, counts) -> int:
        """Drop the newest ``counts[b]`` written tokens of each row ``b``."""
        counts = [max(0, int(n)) for n in counts]
        if isinstance(self.offset, int) or (
            self._row_ends is None and len(set(counts)) == 1
        ):
            return self.trim(counts[0])
        ends = self._ends()
        counts = [min(n, end) for n, end in zip(counts, ends)]
        self.offset = self.offset - mx.array(counts, dtype=self.offset.dtype)
        self._set_ends([end - n for end, n in zip(ends, counts)])
        return max(counts)

    def _update_rows(self, keys: mx.array, values: mx.array):
        """Append ``keys``/``values`` row by row at each row's own end."""
        self._ensure_codecs(keys, values)
        new_keys, new_values = self._try_fused_kv_quantize(keys, values)
        if new_keys is None:
            new_keys = self.key_codec.quantize(keys)
            new_values = self.value_codec.quantize(values)
        ends = self._ends()
        n_new = keys.shape[2]
        used = self._phys_end
        needed = max(ends) + n_new
        self.keys = _reserve_state_capacity(self.keys, used, needed, self.cache_step)
        self.values = _reserve_state_capacity(
            self.values, used, needed, self.cache_step
        )
        _write_state_rows(self.keys, new_keys, ends)
        _write_state_rows(self.values, new_values, ends)
        self.offset = self.offset + n_new
        self._set_ends([end + n_new for end in ends])
        if n_new > 16:
            mx.eval(self.keys, self.values)
        ks, vs = self.state
        return (
            _QuantizedStateProxy(ks, self._phys_end, keys.shape[1]),
            _QuantizedStateProxy(vs, self._phys_end, keys.shape[1]),
        )

    def _align_row_ends(self) -> None:
        """Shift short rows right so every row ends at ``_phys_end`` again."""
        if self._row_ends is None:
            return
        end = self._phys_end
        shifts = [end - row_end for row_end in self._row_ends]
        if self.keys is not None:
            # Only short rows move, in place; their vacated columns become padding.
            mx.eval(self.left_padding)
            starts = [int(p) for p in self.left_padding.tolist()]
            moves = [
                (row, start, shift)
                for row, (start, shift) in enumerate(zip(starts, shifts))
                if shift > 0
            ]
            _shift_state_rows(self.keys, moves, end)
            _shift_state_rows(self.values, moves, end)
            mx.eval(self.keys, self.values)
        self.left_padding = self.left_padding + mx.array(shifts)
        self._row_ends = None
        self._cached_state = None
        self._cached_state_offset = -1

    # ---- update_and_fetch override for B>1 only ----------------------------

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if isinstance(self.offset, int):
            # B=1: parent's method directly (zero overhead)
            return super().update_and_fetch(keys, values)
        if self._row_ends is not None:
            return self._update_rows(keys, values)
        # B>1: track per-request offset separately from state offset
        T_new = keys.shape[2]
        # Append at the written physical end (see __init__._phys_end note).
        int_offset = self._phys_end
        self.offset += T_new
        saved_offset = self.offset
        self.offset = int_offset
        result = super().update_and_fetch(keys, values)
        self.offset = saved_offset
        self._phys_end = int_offset + T_new
        return result

    # ---- state override for B>1 (offset is mx.array) -----------------------

    @property
    def state(self):
        if isinstance(self.offset, int):
            return super().state
        # B>1: slice to the written end — _state_length(self.keys) is the
        # step-allocated capacity and would expose unwritten columns to
        # attention after the buffer grows.
        if self.keys is None:
            return None, None
        length = self._phys_end
        return _slice_state(self.keys, length), _slice_state(self.values, length)

    @state.setter
    def state(self, value):
        TurboQuantKVCache.state.fset(self, value)
        self._row_ends = None
        # The parent fset resets to int-offset (B=1) bookkeeping, where the
        # parent's offset is the write cursor (and trim() may move it back).
        # Drop any stale batch-mode value so _ensure_array_offset re-derives
        # _phys_end from that cursor at the B>1 switch.
        self._phys_end = 0

    # ---- make_mask override (batch-aware) ----------------------------------

    def make_mask(
        self,
        N: int,
        return_array: bool = False,
        window_size: Optional[int] = None,
    ):
        offset = self.offset
        if isinstance(offset, int):
            return create_attention_mask(N, offset, return_array, window_size)
        if self._row_ends is not None:
            # Row b's queries sit at its own end.
            ends = mx.array(self._row_ends)[:, None, None]
            queries = ends + mx.arange(N)[None, :, None]
            columns = mx.arange(max(self._row_ends) + N)[None, None, :]
            mask = (columns <= queries) & (columns >= self.left_padding[:, None, None])
            if window_size is not None:
                mask = mask & (columns > queries - window_size)
            return mask[:, None]
        if (
            isinstance(offset, mx.array)
            and offset.size == 1
            and not self._has_left_padding()
        ):
            return create_attention_mask(N, offset.item(), return_array, window_size)
        # B>1 (or a left-padded survivor after filter()): delegate to mlx-lm's
        # create_causal_mask with the physical column count + per-request
        # left_padding, exactly like BatchKVCache. The old hand-rolled term
        # compared each request's sequence length (offset) against the column
        # index, which masked out valid left-padded tokens — so left-padded
        # requests attended to ~nothing and decoded garbage. The column count
        # is the WRITTEN end, not offset.max(): after the zero-left-padding
        # row departs, offset.max() undercounts and blinds the survivors to
        # their own tail context.
        phys = self._phys_end
        return create_causal_mask(
            N, offset=phys, window_size=window_size, left_padding=self.left_padding
        )

    # prefill_attention and dequantize inherited from TurboQuantKVCache

    # ---- batch operations --------------------------------------------------

    def _ensure_array_offset(self):
        if isinstance(self.offset, int):
            # B=1 tracks written columns in the parent's int offset (plus any
            # left padding); sync the physical end before switching to
            # per-request array offsets, where the parent no longer maintains
            # it.
            lp0 = int(self.left_padding[0].item()) if self.left_padding is not None else 0
            self._phys_end = max(self._phys_end, self.offset + lp0)
            self.offset = mx.array([self.offset])

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchTurboQuantKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= (
                left_padding
                if isinstance(self.offset, mx.array)
                else left_padding[0].item()
            )
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is None:
            return
        self._align_row_ends()
        padding = self._right_padding
        if self.keys is not None:
            # Roll the already-quantized state directly (_roll_state) rather
            # than dequantize -> dynamic_roll -> requantize: roll only
            # reindexes along the token axis, so it's exact on the packed
            # representation and skips both the fp16 materialization and the
            # requantization codec's re-rounding of an already-quantized
            # value.
            shifts = padding[:, None]
            self.keys = _roll_state(self.keys, shifts, axis=2)
            self.values = _roll_state(self.values, shifts, axis=2)
            mx.eval(self.keys, self.values)
        self.offset -= (
            padding if isinstance(self.offset, mx.array) else padding[0].item()
        )
        self.left_padding += padding
        self._right_padding = None

    def filter(self, batch_indices):
        self._ensure_array_offset()
        ends = None
        if self._row_ends is not None:
            ends = mx.array(self._row_ends)[batch_indices].reshape(-1).tolist()
        if self.keys is not None:
            self.keys = _filter_state(self.keys, batch_indices)
            self.values = _filter_state(self.values, batch_indices)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        # Shift left to drop shared padding, mirroring BatchKVCache.filter().
        # The turboquant_skip_last layer is a stock BatchKVCache that compacts
        # here, and the model feeds every attention layer ONE mask built from
        # the first (TQ) layer's _phys_end. Skipping the same compaction
        # leaves that mask min_left_pad columns wider than the dense layer's
        # keys and crashes the next decode step (#2237).
        min_left_pad = int(self.left_padding.min().item())
        if min_left_pad > 0:
            if self.keys is not None:
                # Keep the spare columns past the written end.
                capacity = _state_length(self.keys)
                self.keys = _slice_state_range(self.keys, min_left_pad, capacity)
                self.values = _slice_state_range(self.values, min_left_pad, capacity)
            self._phys_end -= min_left_pad
            self.left_padding = self.left_padding - min_left_pad
        if ends is not None:
            self._set_ends([end - min_left_pad for end in ends])
        self._cached_state = None
        self._cached_state_offset = -1

    def extend(self, other: "BatchTurboQuantKVCache"):
        if not isinstance(other, BatchTurboQuantKVCache):
            raise TypeError(
                "BatchTurboQuantKVCache.extend expected BatchTurboQuantKVCache, "
                f"got {type(other).__name__}"
            )
        if _quantization_config(self) != _quantization_config(other):
            raise ValueError("Cannot extend caches with mixed quantization configs")
        self._ensure_array_offset()
        other._ensure_array_offset()
        self._align_row_ends()
        other._align_row_ends()
        # Align on the WRITTEN ends: _state_length is step-allocated capacity,
        # and padding a joining row by capacity difference would bury its
        # content behind unwritten columns.
        s_idx = self._phys_end if self.keys is not None else 0
        o_idx = other._phys_end if other.keys is not None else 0
        max_idx = max(s_idx, o_idx)
        ref_keys = self.keys if self.keys is not None else other.keys
        ref_values = self.values if self.values is not None else other.values

        def _trim(c, idx):
            if c.keys is None:
                return None, None
            return _slice_state(c.keys, idx), _slice_state(c.values, idx)

        s_rows = int(self.offset.shape[0])
        o_rows = int(other.offset.shape[0])
        s_off, s_lp = self.offset, self.left_padding + (max_idx - s_idx)
        o_off, o_lp = other.offset, other.left_padding + (max_idx - o_idx)
        s_ks, s_vs = _trim(self, s_idx)
        o_ks, o_vs = _trim(other, o_idx)

        if max_idx > 0 and ref_keys is not None:
            # Both sides go end-aligned into one buffer with room to grow.
            capacity = _spare_capacity(max_idx)
            rows = s_rows + o_rows
            self.keys = _place_states(
                ref_keys,
                [(s_ks, 0, max_idx - s_idx), (o_ks, s_rows, max_idx - o_idx)],
                rows,
                capacity,
            )
            self.values = _place_states(
                ref_values,
                [(s_vs, 0, max_idx - s_idx), (o_vs, s_rows, max_idx - o_idx)],
                rows,
                capacity,
            )
        elif s_ks is not None and o_ks is not None:
            # Nothing written on either side yet.
            self.keys = _concat_state_batch([s_ks, o_ks])
            self.values = _concat_state_batch([s_vs, o_vs])
        elif o_ks is not None:
            self.keys = o_ks
            self.values = o_vs

        self.offset = mx.concatenate([s_off, o_off])
        self.left_padding = mx.concatenate([s_lp, o_lp])
        self._phys_end = max_idx
        self._cached_state = None
        self._cached_state_offset = -1

        if self.key_codec is None:
            self.key_codec = other.key_codec
            self.value_codec = other.value_codec

    def _new_single_cache(self):
        return TurboQuantKVCache(bits=self.bits, seed=self.seed)

    def extract(self, idx: int) -> TurboQuantKVCache:
        padding = self.left_padding[idx].item()
        total = (
            self.offset[idx].item()
            if isinstance(self.offset, mx.array)
            else self.offset
        )
        end = padding + total

        tq = self._new_single_cache()
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
        for cache in caches:
            if not isinstance(cache, TurboQuantKVCache):
                raise TypeError(
                    "BatchTurboQuantKVCache.merge expected TurboQuantKVCache "
                    f"entries, got {type(cache).__name__}"
                )
        bits = caches[0].bits
        seed = caches[0].seed
        configs = {_quantization_config(c) for c in caches}
        if len(configs) > 1 or _quantization_config(caches[0])[0] != getattr(
            cls, "quantization_scheme", "turboquant"
        ):
            # Packed state width is ceil(head_dim * bits / 32) and codecs are
            # rebuilt from (head_dim, bits, seed), so members quantized under
            # different configs cannot share a batch. Without this guard the
            # mismatch surfaces as a raw mx.concatenate shape error (or, for
            # equal widths, silent garbage decode) deep in
            # _concat_state_batch (#2045).
            raise ValueError(
                "Cannot batch TurboQuant caches with mixed quantization "
                f"configs (bits, seed): {sorted(configs)}. A request restored "
                "from cache blocks written at another turboquant_kv_bits "
                "depth cannot share a batch with fresh requests; clear the "
                "paged SSD cache for this model if this persists."
            )
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
        reference_key_state = None
        reference_value_state = None
        for c in caches:
            ks, vs = c.state
            if ks is not None:
                reference_key_state = (
                    ks._state if isinstance(ks, _QuantizedStateProxy) else ks
                )
                reference_value_state = (
                    vs._state if isinstance(vs, _QuantizedStateProxy) else vs
                )
                break

        # An empty request keeps a zero row.
        rows = 0
        for p, c in zip(padding, caches):
            ks, vs = c.state
            if ks is None:
                if max_length > 0 and reference_key_state is not None:
                    rows += 1
                continue
            ks = ks._state if isinstance(ks, _QuantizedStateProxy) else ks
            vs = vs._state if isinstance(vs, _QuantizedStateProxy) else vs
            key_states.append((ks, rows, p))
            value_states.append((vs, rows, p))
            rows += _state_rows(ks)

        if rows:
            capacity = _spare_capacity(max_length)
            batch.keys = _place_states(reference_key_state, key_states, rows, capacity)
            batch.values = _place_states(
                reference_value_state, value_states, rows, capacity
            )
            mx.eval(batch.keys, batch.values)

        batch.offset += max_length
        batch._phys_end = max_length
        return batch
