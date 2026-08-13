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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
    _allocate_state_like,
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

__all__ = [
    "TURBOQUANT_CONVERSION_SLICE_TOKENS",
    "TURBOQUANT_PREFILL_KEY_CHUNK_TOKENS",
    "TURBOQUANT_PREFILL_QUERY_BLOCK_TOKENS",
    "TurboQuantKVCache",
    "BatchTurboQuantKVCache",
    "convert_kv_cache_sliced",
    "estimate_turboquant_conversion_peak_bytes",
    "estimate_turboquant_prefill_attention_workspace_bytes",
    "turboquant_mse_bytes_per_element",
    "turboquant_enabled",
]


TURBOQUANT_CONVERSION_SLICE_TOKENS = 8192
_CONVERSION_WORKSPACE_ARRAYS_PER_SOURCE = 4
TURBOQUANT_PREFILL_QUERY_BLOCK_TOKENS = 256
TURBOQUANT_PREFILL_KEY_CHUNK_TOKENS = 16384


@dataclass(frozen=True, slots=True)
class TurboQuantConversionStats:
    """Observed layer and slice counts for one cache-list conversion."""

    converted_layers: int
    skipped_dense_layers: int
    slices: int


def _turboquant_family_indices(cache_list: list[Any]) -> list[int]:
    """Return full-attention layer indices used by the skip-last rule."""
    return [
        index
        for index, cache_obj in enumerate(cache_list)
        if isinstance(cache_obj, (KVCache, TurboQuantKVCache))
    ]


def _turboquant_target_indices(
    cache_list: list[Any], *, skip_last: bool
) -> tuple[list[int], int | None]:
    """Return conversion targets and the optional dense skip-last layer."""
    target_indices = _turboquant_family_indices(cache_list)
    skipped_index = (
        target_indices.pop() if skip_last and len(target_indices) > 1 else None
    )
    return target_indices, skipped_index


def _turboquant_mse_bit_widths(bits: float) -> tuple[int, int]:
    """Return the integer key/value widths used by the MSE codec."""
    validated_bits = float(_validate_bits(bits))
    if math.isclose(validated_bits, round(validated_bits), abs_tol=1e-6):
        width = int(round(validated_bits))
        return width, width
    return int(math.floor(validated_bits)), int(math.ceil(validated_bits))


def _quantized_mse_vector_bytes(head_dim: int, bits: int) -> int:
    """Return one packed MSE vector's norm and uint32 index bytes."""
    packed_words = (head_dim * bits + 31) // 32
    return int(mx.float16.size + packed_words * mx.uint32.size)


def turboquant_mse_bytes_per_element(head_dim: int, bits: float) -> float:
    """Return the average packed MSE K/V resident width per element."""
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
        raise ValueError("head_dim must be a positive integer")
    key_bits, value_bits = _turboquant_mse_bit_widths(bits)
    key_vector_bytes = _quantized_mse_vector_bytes(head_dim, key_bits)
    value_vector_bytes = _quantized_mse_vector_bytes(head_dim, value_bits)
    return (key_vector_bytes + value_vector_bytes) / (2 * head_dim)

def estimate_turboquant_prefill_attention_workspace_bytes(
    *,
    query_tokens: int,
    kv_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    bits: float,
    compute_dtype_size: float = 2,
    causal: bool = True,
) -> int:
    """Bound the first chunked Q8-style TurboQuant prefill attention call.

    The long-prefill route retains all completed query-block outputs while it
    evaluates one 256-query by 16384-key block at a time. This structural
    bound prices those retained outputs, the active score/softmax tensors,
    unpack/cast/codebook tensors for K and V, packed state slices, and the
    caller-owned query input. It does not rely on allocator fusion or on a
    prior TurboQuant transient sample.
    """
    dimensions = (
        query_tokens,
        kv_len,
        num_query_heads,
        num_kv_heads,
        head_dim,
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in dimensions
    ):
        return 0
    if (
        not isinstance(bits, (int, float))
        or isinstance(bits, bool)
        or not math.isfinite(float(bits))
        or float(bits) <= 0
    ):
        return 0
    if (
        not isinstance(compute_dtype_size, (int, float))
        or isinstance(compute_dtype_size, bool)
        or not math.isfinite(float(compute_dtype_size))
        or float(compute_dtype_size) <= 0
    ):
        return 0

    q_block = min(query_tokens, TURBOQUANT_PREFILL_QUERY_BLOCK_TOKENS)
    k_block = min(kv_len, TURBOQUANT_PREFILL_KEY_CHUNK_TOKENS)
    key_bits, value_bits = _turboquant_mse_bit_widths(float(bits))
    key_words = (head_dim * key_bits + 31) // 32
    value_words = (head_dim * value_bits + 31) // 32
    compute_bytes = float(compute_dtype_size)

    # Full-query buffers: caller input, scaled queries, final compute cast,
    # retained float32 blocks, and the concatenated float32 result.
    total = (3 * compute_bytes + 8) * num_query_heads * query_tokens * head_dim
    # Nine active float32 query/value/accumulator stages.
    total += 36 * num_query_heads * q_block * head_dim
    # Dots, scaled scores, softmax subtraction, and weights.
    total += 16 * num_query_heads * q_block * k_block
    # K and V uint32 unpack, int32 cast, and float32 codebook-take tensors.
    total += 24 * num_kv_heads * k_block * head_dim
    # K/V norm casts plus packed state-slice materialization.
    total += 8 * num_kv_heads * k_block
    total += (
        num_kv_heads
        * k_block
        * (4 * (key_words + value_words) + 4)
    )
    # Per-query max/denominator and online-softmax state.
    total += 48 * num_query_heads * q_block

    if causal:
        # One additional masked score result, the causal bool tile, and its
        # query/key index vectors.
        total += 4 * num_query_heads * q_block * k_block
        total += q_block * k_block
        total += 8 * (q_block + k_block)

    return int(math.ceil(total))


def _quantized_state_shape_bytes(
    *,
    batch_size: int,
    num_heads: int,
    num_tokens: int,
    head_dim: int,
    bits: int,
) -> int:
    """Return packed MSE-state bytes for one key or value tensor."""
    vectors = batch_size * num_heads * num_tokens
    return vectors * _quantized_mse_vector_bytes(head_dim, bits)


def _layer_conversion_peak_bytes(
    keys: mx.array,
    values: mx.array,
    *,
    bits: float,
    slice_tokens: int,
) -> int:
    """Bound incremental bytes while converting one dense cache layer.

    After the first slice establishes the state type, the destination reserves
    final capacity before later slices run. The peak therefore holds the final
    state, one quantized slice, and bounded quantization workspace. The
    workspace charges four fp32-sized arrays per key/value source tensor.
    """
    batch_size = int(keys.shape[0])
    num_heads = int(keys.shape[1])
    num_tokens = int(keys.shape[2])
    key_dim = int(keys.shape[3])
    value_dim = int(values.shape[3])
    key_bits, value_bits = _turboquant_mse_bit_widths(bits)
    bounded_tokens = min(num_tokens, slice_tokens)

    def _state_bytes(tokens: int) -> int:
        return _quantized_state_shape_bytes(
            batch_size=batch_size,
            num_heads=num_heads,
            num_tokens=tokens,
            head_dim=key_dim,
            bits=key_bits,
        ) + _quantized_state_shape_bytes(
            batch_size=batch_size,
            num_heads=num_heads,
            num_tokens=tokens,
            head_dim=value_dim,
            bits=value_bits,
        )

    final_state = _state_bytes(num_tokens)
    slice_state = _state_bytes(bounded_tokens)
    source_elements = batch_size * num_heads * bounded_tokens * (key_dim + value_dim)
    workspace = (
        source_elements * mx.float32.size * _CONVERSION_WORKSPACE_ARRAYS_PER_SOURCE
    )
    codec_tables = 2 * (key_dim * key_dim + value_dim * value_dim) * mx.float32.size
    return int(final_state + slice_state + workspace + codec_tables)


def _validate_dense_kv_state(keys: mx.array, values: mx.array) -> int:
    """Validate the dense source shape and return its logical token count."""
    if keys.ndim != 4 or values.ndim != 4:
        raise ValueError("TurboQuant conversion requires 4-D K/V state")
    if keys.shape[:3] != values.shape[:3]:
        raise ValueError("TurboQuant conversion requires matching K/V batch shapes")
    return int(keys.shape[2])


def _append_turboquant_slice(
    cache_obj: TurboQuantKVCache,
    keys: mx.array,
    values: mx.array,
    *,
    start: int,
    end: int,
    stream: Any | None,
) -> None:
    """Quantize and materialize one bounded token slice.

    ``TurboQuantKVCache.state`` caches a lazy prefix slice. The converter
    ignores the returned proxies, so clear its private prefix immediately
    before the dependency constructs each slice update and again after eval.
    No converter-owned proxy then survives to the next donation decision.
    """
    cache_obj._cached_state = None
    cache_obj._cached_state_offset = -1
    try:
        if stream is None:
            cache_obj.update_and_fetch(
                keys[:, :, start:end, :],
                values[:, :, start:end, :],
            )
            mx.eval(cache_obj.keys, cache_obj.values)
            return
        with mx.stream(stream):
            cache_obj.update_and_fetch(
                keys[:, :, start:end, :],
                values[:, :, start:end, :],
            )
            mx.eval(cache_obj.keys, cache_obj.values)
    finally:
        cache_obj._cached_state = None
        cache_obj._cached_state_offset = -1


def _validate_converted_layer(
    cache_obj: TurboQuantKVCache,
    *,
    expected_tokens: int,
    expected_bits: float,
) -> None:
    """Reject an incomplete or incompatible converted cache candidate."""
    if cache_obj.offset != expected_tokens:
        raise RuntimeError(
            f"TurboQuant candidate converted {cache_obj.offset} "
            f"of {expected_tokens} tokens"
        )
    if not math.isclose(cache_obj.bits, expected_bits, abs_tol=1e-6):
        raise RuntimeError(
            f"TurboQuant candidate uses {cache_obj.bits} bits, "
            f"expected {expected_bits}"
        )
    key_state, value_state = cache_obj.state
    for label, state in (("key", key_state), ("value", value_state)):
        if not isinstance(state, TurboQuantMSEState):
            raise RuntimeError(
                f"TurboQuant {label} candidate has unsupported "
                f"{type(state).__name__} state"
            )
        if state.norms.dtype != mx.float16 or state.indices.dtype != mx.uint32:
            raise RuntimeError(f"TurboQuant {label} candidate has invalid state dtypes")
        if int(state.norms.shape[2]) != expected_tokens:
            raise RuntimeError(
                f"TurboQuant {label} candidate has incomplete logical state"
            )


def estimate_turboquant_conversion_peak_bytes(
    cache_list: list[Any],
    *,
    bits: float,
    skip_last: bool,
    slice_tokens: int = TURBOQUANT_CONVERSION_SLICE_TOKENS,
) -> int:
    """Return a conservative incremental peak for layer-wise conversion."""
    if slice_tokens <= 0:
        raise ValueError("slice_tokens must be positive")
    _validate_bits(bits)
    target_indices, _ = _turboquant_target_indices(cache_list, skip_last=skip_last)
    peak = 0
    for index in target_indices:
        cache_obj = cache_list[index]
        if isinstance(cache_obj, TurboQuantKVCache):
            if not math.isclose(cache_obj.bits, bits, abs_tol=1e-6):
                raise ValueError(
                    f"TurboQuant layer {index} uses {cache_obj.bits} bits, "
                    f"expected {bits}"
                )
            continue
        if not isinstance(cache_obj, KVCache) or cache_obj.empty():
            continue
        keys, values = cache_obj.state
        _validate_dense_kv_state(keys, values)
        peak = max(
            peak,
            _layer_conversion_peak_bytes(
                keys,
                values,
                bits=bits,
                slice_tokens=slice_tokens,
            ),
        )
    return peak


def convert_kv_cache_sliced(
    cache_list: list[Any],
    *,
    bits: float,
    skip_last: bool,
    slice_tokens: int = TURBOQUANT_CONVERSION_SLICE_TOKENS,
    stream: Any | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> TurboQuantConversionStats:
    """Convert dense full-attention layers in bounded token slices.

    Each destination layer is fully evaluated before replacing its dense
    source. Prior converted layers are released before the next layer starts.
    If a callback raises, the current dense layer remains in place; callers
    must discard the whole cache because earlier layers may already be swapped.
    """
    from .utils.metal_sync import _sync_and_clear_cache

    if slice_tokens <= 0:
        raise ValueError("slice_tokens must be positive")
    validated_bits = float(_validate_bits(bits))
    target_indices, skipped_index = _turboquant_target_indices(
        cache_list, skip_last=skip_last
    )
    converted_layers = 0
    skipped_dense_layers = int(
        skipped_index is not None and isinstance(cache_list[skipped_index], KVCache)
    )
    slices = 0

    for index in target_indices:
        cache_obj: Any | None = cache_list[index]
        keys: mx.array | None = None
        values: mx.array | None = None
        turbo_cache: TurboQuantKVCache | None = None
        converted_current_layer = False
        try:
            if isinstance(cache_obj, TurboQuantKVCache):
                if not math.isclose(cache_obj.bits, validated_bits, abs_tol=1e-6):
                    raise ValueError(
                        f"TurboQuant layer {index} uses {cache_obj.bits} bits, "
                        f"expected {validated_bits}"
                    )
                continue
            if not isinstance(cache_obj, KVCache):
                continue
            if check_cancelled is not None:
                check_cancelled()

            turbo_cache = TurboQuantKVCache(bits=validated_bits)
            if cache_obj.empty():
                cache_list[index] = turbo_cache
                converted_layers += 1
                continue

            keys, values = cache_obj.state
            num_tokens = _validate_dense_kv_state(keys, values)

            first_end = min(slice_tokens, num_tokens)
            _append_turboquant_slice(
                turbo_cache,
                keys,
                values,
                start=0,
                end=first_end,
                stream=stream,
            )
            slices += 1
            _sync_and_clear_cache(stream)
            if check_cancelled is not None:
                check_cancelled()

            if first_end < num_tokens:
                if stream is None:
                    turbo_cache.keys = _reserve_state_capacity(
                        turbo_cache.keys,
                        turbo_cache.offset,
                        num_tokens,
                        num_tokens,
                    )
                    turbo_cache.values = _reserve_state_capacity(
                        turbo_cache.values,
                        turbo_cache.offset,
                        num_tokens,
                        num_tokens,
                    )
                    turbo_cache._cached_state = None
                    turbo_cache._cached_state_offset = -1
                    mx.eval(turbo_cache.keys, turbo_cache.values)
                else:
                    with mx.stream(stream):
                        turbo_cache.keys = _reserve_state_capacity(
                            turbo_cache.keys,
                            turbo_cache.offset,
                            num_tokens,
                            num_tokens,
                        )
                        turbo_cache.values = _reserve_state_capacity(
                            turbo_cache.values,
                            turbo_cache.offset,
                            num_tokens,
                            num_tokens,
                        )
                        turbo_cache._cached_state = None
                        turbo_cache._cached_state_offset = -1
                        mx.eval(turbo_cache.keys, turbo_cache.values)
                _sync_and_clear_cache(stream)
                if check_cancelled is not None:
                    check_cancelled()

            for start in range(first_end, num_tokens, slice_tokens):
                if check_cancelled is not None:
                    check_cancelled()
                end = min(start + slice_tokens, num_tokens)
                _append_turboquant_slice(
                    turbo_cache,
                    keys,
                    values,
                    start=start,
                    end=end,
                    stream=stream,
                )
                slices += 1
                _sync_and_clear_cache(stream)
                if check_cancelled is not None:
                    check_cancelled()

            _validate_converted_layer(
                turbo_cache,
                expected_tokens=num_tokens,
                expected_bits=validated_bits,
            )
            cache_list[index] = turbo_cache
            converted_layers += 1
            converted_current_layer = True
        finally:
            cache_obj = None
            keys = None
            values = None
            turbo_cache = None
        if converted_current_layer:
            _sync_and_clear_cache(stream)

    return TurboQuantConversionStats(
        converted_layers=converted_layers,
        skipped_dense_layers=skipped_dense_layers,
        slices=slices,
    )


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


def _pad_state_left(state, pad_length: int):
    """Prepend zeros along the token dimension (axis 2) of a state."""
    if state is None or pad_length <= 0:
        return state
    pad = _allocate_state_like(state, pad_length)
    return _concat_state(pad, state)


def _empty_state_batch_like(state, batch_size: int):
    """Allocate an empty token state with the requested batch size."""
    if state is None:
        return None
    row = _filter_state(_allocate_state_like(state, 0), slice(0, 1))
    if batch_size == 1:
        return row
    return _concat_state_batch([row] * batch_size)


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

    def __init__(
        self, left_padding: list[int], bits: float = 4.0, seed: int = 0
    ) -> None:
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
        # step-allocated capacity, not the written end. (Deliberately not
        # named `_idx` — mlx-vlm's rollback_speculative_cache changes
        # behavior on that attribute.)
        self._phys_end = 0

    # ---- update_and_fetch override for B>1 only ----------------------------

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if isinstance(self.offset, int):
            # B=1: parent's method directly (zero overhead)
            return super().update_and_fetch(keys, values)
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
        window_size: int | None = None,
    ) -> str | mx.array | None:
        offset = self.offset
        if isinstance(offset, int):
            return create_attention_mask(N, offset, return_array, window_size)
        if (
            isinstance(offset, mx.array)
            and offset.size == 1
            and int(self.left_padding.max().item()) == 0
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
        padding = self._right_padding
        if self.keys is not None:
            k_fp16, v_fp16 = self.dequantize()
            k_rolled = dynamic_roll(k_fp16, padding[:, None], axis=2)
            v_rolled = dynamic_roll(v_fp16, padding[:, None], axis=2)
            self.keys = self.key_codec.quantize(k_rolled)
            self.values = self.value_codec.quantize(v_rolled)
            mx.eval(self.keys, self.values)
        self.offset -= (
            padding if isinstance(self.offset, mx.array) else padding[0].item()
        )
        self.left_padding += padding
        self._right_padding = None

    def filter(self, batch_indices):
        self._ensure_array_offset()
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
                self.keys = _slice_state_range(
                    self.keys, min_left_pad, self._phys_end
                )
                self.values = _slice_state_range(
                    self.values, min_left_pad, self._phys_end
                )
            self._phys_end -= min_left_pad
            self.left_padding = self.left_padding - min_left_pad
        self._cached_state = None
        self._cached_state_offset = -1

    def extend(self, other: "BatchTurboQuantKVCache"):
        if not isinstance(other, BatchTurboQuantKVCache):
            raise TypeError(
                "BatchTurboQuantKVCache.extend expected BatchTurboQuantKVCache, "
                f"got {type(other).__name__}"
            )
        self._ensure_array_offset()
        other._ensure_array_offset()
        max_off = max(self.offset.max().item(), other.offset.max().item())
        # Align on the WRITTEN ends: _state_length is step-allocated capacity,
        # and padding a joining row by capacity difference would bury its
        # content behind unwritten columns. _pad_and_trim also slices each
        # side down to its written end, normalizing any over-allocation.
        s_idx = self._phys_end if self.keys is not None else 0
        o_idx = other._phys_end if other.keys is not None else 0
        max_idx = max(s_idx, o_idx)
        ref_keys = self.keys if self.keys is not None else other.keys
        ref_values = self.values if self.values is not None else other.values

        def _pad_and_trim(c, idx):
            batch_size = int(c.offset.shape[0])
            if c.keys is None:
                if max_idx > 0 and ref_keys is not None:
                    ks = _empty_state_batch_like(ref_keys, batch_size)
                    vs = _empty_state_batch_like(ref_values, batch_size)
                else:
                    ks = None
                    vs = None
            else:
                ks = _slice_state(c.keys, idx)
                vs = _slice_state(c.values, idx)
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
        self._phys_end = max_idx
        self._cached_state = None
        self._cached_state_offset = -1

        if self.key_codec is None:
            self.key_codec = other.key_codec
            self.value_codec = other.value_codec

    def extract(self, idx: int) -> TurboQuantKVCache:
        padding = self.left_padding[idx].item()
        total = (
            self.offset[idx].item()
            if isinstance(self.offset, mx.array)
            else self.offset
        )
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
    def merge(cls, caches: list[TurboQuantKVCache]) -> BatchTurboQuantKVCache:
        for cache in caches:
            if not isinstance(cache, TurboQuantKVCache):
                raise TypeError(
                    "BatchTurboQuantKVCache.merge expected TurboQuantKVCache "
                    f"entries, got {type(cache).__name__}"
                )
        bits = caches[0].bits
        seed = caches[0].seed
        configs = {(c.bits, c.seed) for c in caches}
        if len(configs) > 1:
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

        for p, c in zip(padding, caches):
            ks, vs = c.state
            if ks is None:
                if max_length > 0 and reference_key_state is not None:
                    key_states.append(
                        _allocate_state_like(reference_key_state, max_length)
                    )
                    value_states.append(
                        _allocate_state_like(reference_value_state, max_length)
                    )
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
        batch._phys_end = max_length
        return batch
