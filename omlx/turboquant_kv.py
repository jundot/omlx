# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV cache — thin wrapper around mlx_vlm.turboquant.

Core implementation (codecs, Metal kernels, TurboQuantKVCache) lives in
mlx-vlm.  This module re-exports the public API and adds
BatchTurboQuantKVCache for omlx's continuous-batching scheduler.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
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
    _metal_available,
    turboquant_enabled,
    _multi_query_prod_score_kernel,
    _TurboQuantProdCodec,
    _TurboQuantMSECodec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime-loop value weighted-sum kernel (replaces mlx-vlm's unrolled version)
# ---------------------------------------------------------------------------

_VALUE_KERNEL_CHUNK = 16  # registers per thread for repeat accumulation

@lru_cache(maxsize=None)
def _chunked_value_weighted_sum_kernel(bits: int):
    """Value weighted sum with register-chunked loop over repeats.

    mlx-vlm's original kernel unrolls RepeatCount at compile time (one
    float per repeat), which hangs the Metal compiler for large counts.

    This version processes repeats in fixed-size register chunks (16).
    Value unpacking happens once per token per chunk-pass, keeping memory
    bandwidth close to the original while using only 16 registers.

    Performance: ~896x fewer value unpacks than pure runtime loop,
    only ~16x more than fully unrolled (which can't compile for large R).
    """
    if not _metal_available():
        return None

    C = _VALUE_KERNEL_CHUNK
    val_mask = (1 << bits) - 1
    source = f"""
        auto dim = thread_position_in_grid.x;
        auto n = thread_position_in_grid.z;
        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;
        auto bh = n / num_tok_tiles;
        auto tok_tile = n % num_tok_tiles;
        int t_start = tok_tile * TokTileSize;
        int t_end = min(t_start + TokTileSize, (int)token_count);

        if (dim >= Dim) return;

        auto wt = weights + bh * RepeatCount * token_count;
        auto nm = norms + bh * token_count;
        auto pk = packed + bh * token_count * PackedWidth;

        int bo = dim * {bits};
        int v_word = bo / 32;
        int v_shift = bo % 32;
        bool v_spill = (bo % 32 + {bits}) > 32;

        auto out_base = out + (bh * num_tok_tiles + tok_tile) * RepeatCount * Dim + dim;

        for (int r_base = 0; r_base < RepeatCount; r_base += {C}) {{
            int r_end = min(r_base + {C}, (int)RepeatCount);
            int chunk_len = r_end - r_base;
            float acc[{C}] = {{}};

            for (int t = t_start; t < t_end; t++) {{
                auto pt = pk + t * PackedWidth;
                uint vv = (pt[v_word] >> v_shift);
                if (v_spill) vv |= pt[v_word+1] << ({bits} - (v_shift+{bits}-32));
                float val = codebook[vv & {val_mask}u] * static_cast<float>(nm[t]);

                auto wt_row = wt + r_base * token_count + t;
                for (int i = 0; i < chunk_len; i++) {{
                    acc[i] += wt_row[i * token_count] * val;
                }}
            }}

            for (int i = 0; i < chunk_len; i++) {{
                out_base[(r_base + i) * Dim] = acc[i];
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"omlx_chunked_value_wsum_{bits}",
        input_names=["weights", "norms", "packed", "codebook"],
        output_names=["out"],
        source=source,
    )


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
    _BATCH_QUANTIZE_SIZE = 32  # buffer N decode tokens before batch-quantizing

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

        # fp16 decode buffer (batch-quantized every _BATCH_QUANTIZE_SIZE tokens)
        self._decode_buf_k = None  # (B, H, _BATCH_QUANTIZE_SIZE, D)
        self._decode_buf_v = None
        self._decode_buf_count = 0

        # Batch tracking
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0  # quantized token count
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

    # ---- decode buffer management -------------------------------------------

    def _flush_decode_buffer(self):
        """Batch-quantize buffered decode tokens and append to quantized state."""
        if self._decode_buf_count == 0:
            return
        buf_k = self._decode_buf_k[..., : self._decode_buf_count, :]
        buf_v = self._decode_buf_v[..., : self._decode_buf_count, :]
        new_k = self._key_codec.quantize(buf_k)
        new_v = self._value_codec.quantize(buf_v)
        if self._key_state is None:
            self._key_state = new_k
            self._value_state = new_v
        else:
            new_end = self._idx + self._decode_buf_count
            self._key_state = _reserve_state_capacity(
                self._key_state, self._idx, new_end, self.step
            )
            self._value_state = _reserve_state_capacity(
                self._value_state, self._idx, new_end, self.step
            )
            _write_state(self._key_state, new_k, self._idx)
            _write_state(self._value_state, new_v, self._idx)
        self._idx += self._decode_buf_count
        self._decode_buf_count = 0

    @property
    def total_tokens(self):
        """Total token count: quantized + buffered fp16."""
        return self._idx + self._decode_buf_count

    # ---- update_and_fetch --------------------------------------------------

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, T_new, D = keys.shape
        self._ensure_codecs(keys, values)

        if T_new > 1:
            # Prefill: quantize and append via concat
            new_k = self._key_codec.quantize(keys)
            new_v = self._value_codec.quantize(values)
            if self._key_state is None:
                self._key_state = new_k
                self._value_state = new_v
            else:
                self._key_state = _concat_state(self._key_state, new_k)
                self._value_state = _concat_state(self._value_state, new_v)
            self.offset += T_new
            self._idx += T_new
            return keys, values
        else:
            # Decode: buffer fp16, batch-quantize every N tokens
            if self._decode_buf_k is None or self._decode_buf_k.shape[0] != B:
                if self._decode_buf_k is not None:
                    self._flush_decode_buffer()
                N = self._BATCH_QUANTIZE_SIZE
                self._decode_buf_k = mx.zeros((B, H, N, D), dtype=keys.dtype)
                self._decode_buf_v = mx.zeros((B, H, N, D), dtype=values.dtype)

            idx = self._decode_buf_count
            self._decode_buf_k[..., idx : idx + 1, :] = keys
            self._decode_buf_v[..., idx : idx + 1, :] = values
            self._decode_buf_count += 1
            self.offset += 1

            if self._decode_buf_count >= self._BATCH_QUANTIZE_SIZE:
                self._flush_decode_buffer()

            # Return proxy with total token count for mask creation
            total = self.total_tokens
            return (
                _QuantizedStateProxy(None, total, H),
                _QuantizedStateProxy(None, total, H),
            )

    # ---- attention ---------------------------------------------------------

    def _make_tmp_cache(self):
        """Create a temporary TurboQuantKVCache sharing our codecs and state."""
        tmp = TurboQuantKVCache(bits=self.bits, seed=self.seed)
        tmp.key_codec = self._key_codec
        tmp.value_codec = self._value_codec
        # Slice to actual length (state may be over-allocated during decode)
        ks = self._key_state
        vs = self._value_state
        if _state_length(ks) > self._idx:
            ks = _slice_state(ks, self._idx)
            vs = _slice_state(vs, self._idx)
        tmp.keys = ks
        tmp.values = vs
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
        B, n_q_heads, L, D = queries.shape
        buf_count = self._decode_buf_count

        if buf_count == 0 and self._key_state is not None:
            # All tokens quantized — use TQ decode directly
            tmp = self._make_tmp_cache()
            return tmp.decode_attention(queries, scale=scale, mask=mask)

        if self._key_state is None and buf_count > 0:
            # Only fp16 buffer (no quantized tokens yet) — standard SDPA
            buf_k = self._decode_buf_k[..., :buf_count, :]
            buf_v = self._decode_buf_v[..., :buf_count, :]
            return mx.fast.scaled_dot_product_attention(
                queries, buf_k, buf_v, scale=scale, mask=mask,
            )

        # Hybrid: quantized old tokens + fp16 recent tokens
        n_kv_heads = self._decode_buf_k.shape[1]
        n_repeats = n_q_heads // n_kv_heads
        T_q = self._idx  # quantized token count

        # 1. Score quantized keys (TQ Metal kernel)
        grouped = (queries * scale).reshape(B, n_kv_heads, n_repeats, L, D)
        prepared = self._key_codec.prepare_queries(grouped)
        ks = self._key_state
        if _state_length(ks) > T_q:
            ks = _slice_state(ks, T_q)
        tq_scores = self._key_codec.score_prepared(prepared, ks)
        # shape: (B, H, R, 1, T_q)

        # 2. Score fp16 keys (standard dot product with GQA expansion)
        buf_k = self._decode_buf_k[..., :buf_count, :]  # (B, H_kv, N, D)
        buf_v = self._decode_buf_v[..., :buf_count, :]  # (B, H_kv, N, D)
        # grouped: (B, H_kv, R, 1, D), buf_k: (B, H_kv, N, D)
        # Expand buf_k to (B, H_kv, 1, N, D) for broadcasting with grouped
        fp16_scores = mx.sum(
            grouped * buf_k[:, :, None, :, :],  # (B, H_kv, R, N, D)
            axis=-1,
        )[:, :, :, None, :]  # (B, H_kv, R, 1, N)

        # 3. Concat scores → softmax
        all_scores = mx.concatenate([tq_scores, fp16_scores], axis=-1)
        all_weights = mx.softmax(all_scores, axis=-1)
        tq_weights = all_weights[..., :T_q]
        fp16_weights = all_weights[..., T_q:]  # (B, H_kv, R, 1, N)

        # 4. TQ value weighted sum
        vs = self._value_state
        if _state_length(vs) > T_q:
            vs = _slice_state(vs, T_q)
        tq_output = self._value_codec.weighted_sum(
            tq_weights, vs
        )  # (B, H_kv, R, 1, D)

        # 5. fp16 value weighted sum
        # fp16_weights: (B, H_kv, R, 1, N) → (B, H_kv, R, N) after squeeze
        # buf_v: (B, H_kv, N, D) → expand to (B, H_kv, 1, N, D)
        w = fp16_weights.squeeze(3)  # (B, H_kv, R, N)
        fp16_output = mx.einsum("bhrl,bhld->bhrd", w, buf_v)  # (B, H_kv, R, D)
        fp16_output = fp16_output[:, :, :, None, :]  # (B, H_kv, R, 1, D)

        # 6. Combine and reshape
        output = tq_output + fp16_output
        output = output.reshape(B, n_q_heads, L, D)
        return output.astype(queries.dtype)

    def prefill_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask=None,
    ) -> Optional[mx.array]:
        # Prefill (L>1) uses dequantize+SDPA fallback. The TQ value kernel
        # requires repeat_count = n_repeats * L accumulators — too large for
        # any practical kernel approach (unrolling hangs compiler, runtime
        # loop too slow). Decode (L=1) uses decode_attention() with efficient
        # Metal kernels where repeat_count = n_repeats (small).
        return None

    def dequantize(self, keys_state=None, values_state=None):
        # Flush any buffered decode tokens first
        self._flush_decode_buffer()
        ks = self._key_state
        vs = self._value_state
        if _state_length(ks) > self._idx:
            ks = _slice_state(ks, self._idx)
            vs = _slice_state(vs, self._idx)
        keys = self._key_codec.dequantize(ks).astype(mx.float32)
        values = self._value_codec.dequantize(vs).astype(mx.float32)
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
        self._flush_decode_buffer()
        if self._key_state is not None:
            self._key_state = _filter_state(self._key_state, batch_indices)
            self._value_state = _filter_state(self._value_state, batch_indices)
        if self._decode_buf_k is not None:
            self._decode_buf_k = self._decode_buf_k[batch_indices]
            self._decode_buf_v = self._decode_buf_v[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

    def extend(self, other: "BatchTurboQuantKVCache"):
        self._flush_decode_buffer()
        other._flush_decode_buffer()
        # Reset decode buffers — batch size changes after extend
        self._decode_buf_k = None
        self._decode_buf_v = None
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
        self._flush_decode_buffer()
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
        # Flush decode buffer so state includes all tokens
        self._flush_decode_buffer()
        if self._key_state is not None:
            ks = self._key_state
            vs = self._value_state
            if _state_length(ks) > self._idx:
                ks = _slice_state(ks, self._idx)
                vs = _slice_state(vs, self._idx)
            return ks, vs, self.offset, self.left_padding
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
        return self._key_state is None and self._decode_buf_count == 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self.total_tokens, left_padding=self.left_padding, **kwargs
        )

    @property
    def nbytes(self):
        if self._key_state is not None:
            return _state_nbytes(self._key_state) + _state_nbytes(self._value_state)
        return 0
