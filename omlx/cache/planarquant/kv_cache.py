# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803, N806
"""PlanarQuantKVCache — KV cache with packed PlanarQuant3 storage.

Three major features vs the old implementation:
  1. **Packed 3-bit storage** matching upstream block_planar3_0 layout:
     norm(fp16,2B) + qs(D/4B) + signs(D/8B) = 50B per 128-elem block
     → 0.39 bytes/elem → 5.1x compression vs FP16
  2. **Deferred quantization**: K/V stored as FP16 during prefill,
     bulk-converted to PlanarQuant3 after prefill completes. This avoids
     error compounding through the prefill — upstream claims 3x better PPL.
  3. **Asymmetric K/V**: V can remain FP16 while K is quantized, giving
     zero PPL loss at 5.1x K-compression (upstream's best config).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlx.core as mx

from .constants import PLANAR_D
from .metal_kernels import dequantize_fused, quantize_fused
from .reference import dequantize_block, quantize_block

logger = logging.getLogger(__name__)


def _has_metal() -> bool:
    try:
        return mx.metal.is_available()
    except Exception:
        return False


def _quantize(x: mx.array) -> tuple[mx.array, mx.array]:
    """Quantize using Metal kernel if available, else Python fallback."""
    if _has_metal():
        return quantize_fused(x)
    return quantize_block(x)


try:
    from mlx_lm.models.cache import _BaseCache, create_attention_mask
except ImportError:
    _BaseCache = object
    def create_attention_mask(*args, **kwargs):
        raise ImportError("mlx_lm.models.cache not available")


# ---------------------------------------------------------------------------
# Quantized-state proxy
# ---------------------------------------------------------------------------

@dataclass
class PlanarQuantState:
    """Packed PlanarQuant3 K or V state.

    Layout matches upstream block_planar3_0 per token-row:
      packed: (..., T, qs_size + signs_size)  uint8
      norms:  (..., T, 1)                     float16
    """
    packed: mx.array  # (B, H, T, qs_size+signs_size) uint8
    norms: mx.array   # (B, H, T, 1) float16

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.packed.shape)

    @property
    def dtype(self):
        return self.packed.dtype

    def __len__(self) -> int:
        return self.packed.shape[0]


# ---------------------------------------------------------------------------
# FP16 state proxy (for deferred prefill V or unquantized side)
# ---------------------------------------------------------------------------

@dataclass
class FP16State:
    """Plain FP16 K or V state (no quantization)."""
    tensor: mx.array  # (B, H, T, D) float16

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)

    @property
    def dtype(self):
        return self.tensor.dtype

    def __len__(self) -> int:
        return self.tensor.shape[0]


# ---------------------------------------------------------------------------
# PlanarQuantKVCache
# ---------------------------------------------------------------------------

class PlanarQuantKVCache(_BaseCache):
    """KV cache with packed PlanarQuant3 storage and deferred quantization.

    Storage modes:
      - ``deferred`` (prefill): K/V stored as FP16, quantized on finalize
      - ``quantized`` (decode): K and optionally V in packed PlanarQuant3
      - ``mixed`` (asymmetric): K=PlanarQuant3, V=FP16

    The cache starts in deferred mode and transitions to quantized after
    :meth:`finalize_prefill` is called. During decode, new tokens are
    quantized on insertion.
    """

    bits: float = 3.0
    cache_step: int = 256

    def __init__(
        self,
        bits: float = 3.0,
        quantize_v: bool = True,
    ):
        self.bits = float(bits)
        if self.bits != 3.0:
            raise ValueError("PlanarQuantKVCache only supports 3-bit PlanarQuant3")
        self.quantize_v = quantize_v
        self.group_size = PLANAR_D  # block size matches upstream
        self.offset: int = 0
        self._cap: int = 0
        self._finalized: bool = False  # True after prefill→quantize conversion

        # Deferred-mode FP16 buffers (used during prefill)
        self._k_fp16: mx.array | None = None
        self._v_fp16: mx.array | None = None

        # Quantized-mode packed buffers (used after finalize_prefill)
        self._k_packed: mx.array | None = None  # (B, H_k, cap, packed_last) uint8
        self._k_norms: mx.array | None = None    # (B, H_k, cap, 1) float16
        self._v_packed: mx.array | None = None
        self._v_norms: mx.array | None = None

        # Cached dequantized K/V for fast decode (avoids re-dequant per step
        # and avoids per-token quantization during decode). The cache is the
        # authoritative FP16 source after finalize; _k_packed/_v_packed hold
        # the prefill portion plus any lazily-flushed decode rows.
        self._k_dequant_cache: mx.array | None = None  # (B, H_k, cap, D_k) fp16
        self._k_dequant_offset: int = 0  # how many rows are in the cache
        self._v_dequant_cache: mx.array | None = None  # (B, H_v, cap, D_v) fp16
        self._v_dequant_offset: int = 0

        # Range [start, end) of decode rows that have NOT been packed yet.
        # state.getter calls _flush_unpacked() before serializing.
        self._k_unpacked_start: int | None = None
        self._k_unpacked_end: int | None = None
        self._v_unpacked_start: int | None = None
        self._v_unpacked_end: int | None = None

        # Shape memo
        self._B: int | None = None
        self._H_k: int | None = None
        self._H_v: int | None = None
        self._D_k: int | None = None
        self._D_v: int | None = None
        self._packed_last_k: int | None = None
        self._packed_last_v: int | None = None

        # Tiled decode configuration. When ``tile_size`` is not None,
        # :meth:`decode_attention` routes through
        # :meth:`decode_attention_tiled` with online softmax, decompressing
        # ``tile_size`` tokens at a time. This keeps peak memory O(tile_size)
        # instead of O(offset). ``memory_pressure`` enables an eager
        # per-token quantization path in update_and_fetch so dequant caches
        # are never allocated.
        self.tile_size: int | None = None
        self.memory_pressure: bool = False

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _invalidate_dequant_cache(self) -> None:
        self._k_dequant_cache = None
        self._k_dequant_offset = 0
        self._v_dequant_cache = None
        self._v_dequant_offset = 0
        self._k_unpacked_start = None
        self._k_unpacked_end = None
        self._v_unpacked_start = None
        self._v_unpacked_end = None

    def _init_buffers(self, keys: mx.array, values: mx.array) -> None:
        B, H_k, _, D_k = keys.shape
        _, H_v, _, D_v = values.shape
        if D_k % 8 != 0 or D_v % 8 != 0:
            raise ValueError(
                f"PlanarQuantKVCache requires head_dim divisible by 8; "
                f"got K head_dim={D_k}, V head_dim={D_v}"
            )
        cap = self.cache_step
        self._B = B
        self._H_k = H_k
        self._H_v = H_v
        self._D_k = D_k
        self._D_v = D_v
        self._packed_last_k = D_k // 4 + D_k // 8
        self._packed_last_v = D_v // 4 + D_v // 8
        self._cap = cap

        # Start in deferred mode — allocate FP16 buffers
        self._k_fp16 = mx.zeros((B, H_k, cap, D_k), dtype=mx.float16)
        self._v_fp16 = mx.zeros((B, H_v, cap, D_v), dtype=mx.float16)

    def _grow_fp16(self, new_end: int) -> None:
        if new_end <= self._cap:
            return
        grow_by = max(self.cache_step, new_end - self._cap)
        new_cap = self._cap + grow_by

        def _pad(arr: mx.array) -> mx.array:
            shape = list(arr.shape)
            shape[2] = new_cap - arr.shape[2]
            pad = mx.zeros(tuple(shape), dtype=arr.dtype)
            return mx.concatenate([arr, pad], axis=2)

        assert self._k_fp16 is not None
        assert self._v_fp16 is not None
        self._k_fp16 = _pad(self._k_fp16)
        self._v_fp16 = _pad(self._v_fp16)
        self._cap = new_cap

    def _grow_packed(self, new_end: int) -> None:
        if new_end <= self._cap:
            return
        grow_by = max(self.cache_step, new_end - self._cap)
        new_cap = self._cap + grow_by

        def _pad(arr: mx.array) -> mx.array:
            shape = list(arr.shape)
            shape[2] = new_cap - arr.shape[2]
            pad = mx.zeros(tuple(shape), dtype=arr.dtype)
            return mx.concatenate([arr, pad], axis=2)

        for attr in ("_k_packed", "_k_norms", "_v_packed", "_v_norms"):
            arr = getattr(self, attr)
            if arr is not None:
                setattr(self, attr, _pad(arr))
        if self._v_fp16 is not None and self._v_fp16.shape[2] < new_cap:
            self._v_fp16 = _pad(self._v_fp16)
        self._cap = new_cap

    @staticmethod
    def _write_slice(buf: mx.array, new: mx.array, start: int) -> mx.array:
        L = new.shape[2]
        end = start + L
        buf[..., start:end, :] = new.astype(buf.dtype)
        return buf

    # ------------------------------------------------------------------
    # Dequant-cache helpers (FP16 staging for decode / MPS SDPA)
    # ------------------------------------------------------------------

    def _ensure_k_dequant_cache(self) -> None:
        """One-time: dequant the packed prefill K into the FP16 cache."""
        if self._k_dequant_cache is not None and self._k_dequant_offset == self.offset:
            return
        assert self._B is not None and self._H_k is not None and self._D_k is not None
        cache = mx.zeros((self._B, self._H_k, self._cap, self._D_k), dtype=mx.float16)
        if self.offset > 0:
            assert self._k_packed is not None
            assert self._k_norms is not None
            k_dq = dequantize_fused(
                self._k_packed[..., :self.offset, :],
                self._k_norms[..., :self.offset, :],
                out_dtype=mx.float16,
            )
            cache[..., :self.offset, :] = k_dq.astype(mx.float16)
        self._k_dequant_cache = cache
        self._k_dequant_offset = self.offset

    def _ensure_v_dequant_cache(self) -> None:
        """One-time: dequant the packed prefill V into the FP16 cache."""
        if self._v_dequant_cache is not None and self._v_dequant_offset == self.offset:
            return
        assert self._B is not None and self._H_v is not None and self._D_v is not None
        cache = mx.zeros((self._B, self._H_v, self._cap, self._D_v), dtype=mx.float16)
        if self.offset > 0 and self._v_packed is not None and self._v_norms is not None:
            v_dq = dequantize_fused(
                self._v_packed[..., :self.offset, :],
                self._v_norms[..., :self.offset, :],
                out_dtype=mx.float16,
            )
            cache[..., :self.offset, :] = v_dq.astype(mx.float16)
        self._v_dequant_cache = cache
        self._v_dequant_offset = self.offset

    def _grow_k_dequant_cache(self, new_end: int) -> None:
        assert self._k_dequant_cache is not None
        if self._k_dequant_cache.shape[2] >= new_end:
            return
        B, H_k, _, D_k = self._k_dequant_cache.shape
        new_cache = mx.zeros((B, H_k, self._cap, D_k), dtype=mx.float16)
        logical_len = self._logical_len()
        new_cache[..., :logical_len, :] = self._k_dequant_cache[
            ..., :logical_len, :
        ]
        self._k_dequant_cache = new_cache

    def _grow_v_dequant_cache(self, new_end: int) -> None:
        assert self._v_dequant_cache is not None
        if self._v_dequant_cache.shape[2] >= new_end:
            return
        B, H_v, _, D_v = self._v_dequant_cache.shape
        new_cache = mx.zeros((B, H_v, self._cap, D_v), dtype=mx.float16)
        logical_len = self._logical_len()
        new_cache[..., :logical_len, :] = self._v_dequant_cache[
            ..., :logical_len, :
        ]
        self._v_dequant_cache = new_cache

    def _flush_unpacked(self) -> None:
        """Lazy-pack any unpacked decode rows into _k_packed / _v_packed.

        Called before :attr:`state` returns the packed buffers for
        serialization. No-op if there are no unpacked rows.
        """
        if self._k_unpacked_start is not None and self._k_unpacked_end is not None:
            start, end = self._k_unpacked_start, self._k_unpacked_end
            if end > start and self._k_dequant_cache is not None:
                k_rows = self._k_dequant_cache[..., start:end, :]
                k_packed, k_norms = _quantize(k_rows)
                assert self._k_packed is not None
                assert self._k_norms is not None
                self._k_packed = self._write_slice(self._k_packed, k_packed, start)
                self._k_norms = self._write_slice(self._k_norms, k_norms, start)
            self._k_unpacked_start = None
            self._k_unpacked_end = None

        if (self.quantize_v
                and self._v_unpacked_start is not None
                and self._v_unpacked_end is not None):
            start, end = self._v_unpacked_start, self._v_unpacked_end
            if end > start and self._v_dequant_cache is not None:
                v_rows = self._v_dequant_cache[..., start:end, :]
                v_packed, v_norms = _quantize(v_rows)
                assert self._v_packed is not None
                assert self._v_norms is not None
                self._v_packed = self._write_slice(self._v_packed, v_packed, start)
                self._v_norms = self._write_slice(self._v_norms, v_norms, start)
            self._v_unpacked_start = None
            self._v_unpacked_end = None

    # ------------------------------------------------------------------
    # Deferred quantization: finalize prefill
    # ------------------------------------------------------------------

    def finalize_prefill(self) -> None:
        """Bulk-convert FP16 prefill buffers to packed PlanarQuant3.

        Called after prefill completes. Converts the entire FP16 K (and
        optionally V) cache to packed PlanarQuant3 in one pass.
        """
        if self._finalized:
            return
        if self._k_fp16 is None:
            return

        assert self._D_k is not None
        assert self._packed_last_k is not None

        # Quantize K
        k_packed, k_norms = _quantize(self._k_fp16[..., :self.offset, :])
        # Reshape: _quantize returns (B, H, T, packed_last) and (B, H, T, 1)
        cap = self._cap
        B, H_k = self._B, self._H_k
        self._k_packed = mx.zeros((B, H_k, cap, self._packed_last_k), dtype=mx.uint8)
        self._k_norms = mx.zeros((B, H_k, cap, 1), dtype=mx.float16)
        self._k_packed[..., :self.offset, :] = k_packed.astype(mx.uint8)
        self._k_norms[..., :self.offset, :] = k_norms.astype(mx.float16)

        if self.quantize_v:
            assert self._v_fp16 is not None
            assert self._packed_last_v is not None
            _, H_v = self._B, self._H_v
            v_packed, v_norms = _quantize(self._v_fp16[..., :self.offset, :])
            self._v_packed = mx.zeros((B, H_v, cap, self._packed_last_v), dtype=mx.uint8)
            self._v_norms = mx.zeros((B, H_v, cap, 1), dtype=mx.float16)
            self._v_packed[..., :self.offset, :] = v_packed.astype(mx.uint8)
            self._v_norms[..., :self.offset, :] = v_norms.astype(mx.float16)
            self._v_fp16 = None  # Free FP16 V buffer
        else:
            # Asymmetric: V stays FP16, just trim to offset
            v_fp16 = self._v_fp16[..., :self.offset, :]
            self._v_packed = None
            self._v_norms = None
            self._v_fp16 = mx.zeros((B, self._H_v, cap, self._D_v), dtype=mx.float16)
            self._v_fp16[..., :self.offset, :] = v_fp16

        self._k_fp16 = None  # Free FP16 K buffer
        self._finalized = True
        self._invalidate_dequant_cache()
        logger.info("PlanarQuant: finalized prefill, converted to packed layout")

    # ------------------------------------------------------------------
    # mlx-lm cache contract
    # ------------------------------------------------------------------

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple:
        """Insert new K/V and return current state.

        During prefill (before finalize_prefill): stores FP16.
        During decode (after finalize_prefill): appends to FP16 dequant
        caches only. Per-token quantization is deferred; decode rows are
        lazily packed on state serialization via :meth:`_flush_unpacked`.
        This eliminates the per-step quantize overhead (0.3ms × n_layers)
        and routes decode attention through Apple's MPS SDPA.
        """
        L = keys.shape[2]
        new_end = self.offset + L

        if self._k_fp16 is None and self._k_packed is None:
            self._init_buffers(keys, values)

        if not self._finalized:
            # Deferred mode: store as FP16
            self._grow_fp16(new_end)
            assert self._k_fp16 is not None
            assert self._v_fp16 is not None
            self._k_fp16 = self._write_slice(self._k_fp16, keys, self.offset)
            self._v_fp16 = self._write_slice(self._v_fp16, values, self.offset)
            self.offset = new_end

            # Return FP16 states
            return (
                FP16State(self._k_fp16[..., :self.offset, :]),
                FP16State(self._v_fp16[..., :self.offset, :]),
            )

        # Quantized mode (post-finalize).
        # Keep packed buffer sized to match, but DO NOT per-token quantize —
        # the dequant caches are authoritative during decode. Lazy-pack on
        # state save via _flush_unpacked().
        self._grow_packed(new_end)
        assert self._k_packed is not None
        assert self._k_norms is not None

        # Memory-pressure mode: eagerly quantize new rows, skip dequant cache
        if self.memory_pressure:
            k_packed_new, k_norms_new = _quantize(keys.astype(mx.float16))
            self._k_packed = self._write_slice(self._k_packed, k_packed_new, self.offset)
            self._k_norms = self._write_slice(self._k_norms, k_norms_new, self.offset)
            if self.quantize_v:
                assert self._v_packed is not None
                assert self._v_norms is not None
                v_packed_new, v_norms_new = _quantize(values.astype(mx.float16))
                self._v_packed = self._write_slice(self._v_packed, v_packed_new, self.offset)
                self._v_norms = self._write_slice(self._v_norms, v_norms_new, self.offset)
            else:
                assert self._v_fp16 is not None
                self._v_fp16 = self._write_slice(
                    self._v_fp16, values.astype(mx.float16), self.offset
                )
            self.offset = new_end
            # Return packed states directly — tiled attention will dequant
            # per tile on demand.
            k_state = PlanarQuantState(
                self._k_packed[..., :self.offset, :],
                self._k_norms[..., :self.offset, :],
            )
            if self.quantize_v:
                v_state = PlanarQuantState(
                    self._v_packed[..., :self.offset, :],
                    self._v_norms[..., :self.offset, :],
                )
            else:
                v_state = FP16State(self._v_fp16[..., :self.offset, :])
            return k_state, v_state

        # Ensure K dequant cache covers the prefill portion (one-time dequant)
        self._ensure_k_dequant_cache()
        # Grow K cache buffer if needed, then append new FP16 K rows
        self._grow_k_dequant_cache(new_end)
        k_fp16 = keys.astype(mx.float16)
        self._k_dequant_cache[..., self.offset:new_end, :] = k_fp16
        self._k_dequant_offset = new_end

        # Track unpacked K range for lazy packing
        if self._k_unpacked_start is None:
            self._k_unpacked_start = self.offset
        self._k_unpacked_end = new_end

        if self.quantize_v:
            assert self._v_packed is not None
            assert self._v_norms is not None

            # Ensure V dequant cache covers the prefill portion
            self._ensure_v_dequant_cache()
            self._grow_v_dequant_cache(new_end)
            v_fp16 = values.astype(mx.float16)
            self._v_dequant_cache[..., self.offset:new_end, :] = v_fp16
            self._v_dequant_offset = new_end

            # Track unpacked V range
            if self._v_unpacked_start is None:
                self._v_unpacked_start = self.offset
            self._v_unpacked_end = new_end

            self.offset = new_end
            return (
                FP16State(self._k_dequant_cache[..., :self.offset, :]),
                FP16State(self._v_dequant_cache[..., :self.offset, :]),
            )

        # Asymmetric: V stays FP16 (no quantization at all for V)
        assert self._v_fp16 is not None
        self._v_fp16 = self._write_slice(self._v_fp16, values, self.offset)
        self.offset = new_end
        return (
            FP16State(self._k_dequant_cache[..., :self.offset, :]),
            FP16State(self._v_fp16[..., :self.offset, :]),
        )

    # ------------------------------------------------------------------
    # Dequant + attention
    # ------------------------------------------------------------------

    def _current_state(self) -> tuple:
        logical_len = self._logical_len()
        if not self._finalized:
            assert self._k_fp16 is not None
            assert self._v_fp16 is not None
            return (
                FP16State(self._k_fp16[..., :logical_len, :]),
                FP16State(self._v_fp16[..., :logical_len, :]),
            )
        k_state = PlanarQuantState(
            self._k_packed[..., :logical_len, :],
            self._k_norms[..., :logical_len, :],
        )
        if self.quantize_v:
            v_state = PlanarQuantState(
                self._v_packed[..., :logical_len, :],
                self._v_norms[..., :logical_len, :],
            )
        else:
            v_state = FP16State(self._v_fp16[..., :logical_len, :])
        return k_state, v_state

    def dequantize(
        self,
        keys_state=None,
        values_state=None,
        out_dtype: mx.Dtype = mx.float16,
    ) -> tuple[mx.array, mx.array]:
        """Return ``(keys, values)`` as float arrays."""
        if keys_state is None or values_state is None:
            keys_state, values_state = self._current_state()

        if isinstance(keys_state, FP16State):
            keys = keys_state.tensor.astype(out_dtype)
        elif isinstance(keys_state, PlanarQuantState):
            if out_dtype == mx.float32:
                keys = dequantize_block(keys_state.packed, keys_state.norms)
            else:
                keys = dequantize_fused(keys_state.packed, keys_state.norms, out_dtype=out_dtype)
        else:
            raise TypeError(f"Unknown key state type: {type(keys_state)}")

        if isinstance(values_state, FP16State):
            values = values_state.tensor.astype(out_dtype)
        elif isinstance(values_state, PlanarQuantState):
            if out_dtype == mx.float32:
                values = dequantize_block(values_state.packed, values_state.norms)
            else:
                values = dequantize_fused(values_state.packed, values_state.norms, out_dtype=out_dtype)
        else:
            raise TypeError(f"Unknown value state type: {type(values_state)}")

        return keys, values

    def _get_dequant_k(self, out_dtype: mx.Dtype = mx.float16) -> mx.array:
        """Get dequantized K, using cache if available to avoid re-dequant."""
        self._ensure_k_dequant_cache()
        assert self._k_dequant_cache is not None
        return self._k_dequant_cache[..., :self.offset, :].astype(out_dtype)

    def _get_dequant_v(self, out_dtype: mx.Dtype = mx.float16) -> mx.array:
        """Get dequantized V from cache (quantize_v=True only)."""
        self._ensure_v_dequant_cache()
        assert self._v_dequant_cache is not None
        return self._v_dequant_cache[..., :self.offset, :].astype(out_dtype)

    def enable_memory_pressure_mode(self, tile_size: int = 4096) -> None:
        """Switch to memory-pressure mode for very long contexts.

        Effects:
          - Sets ``self.tile_size`` so subsequent ``decode_attention`` calls
            route through ``decode_attention_tiled`` (online softmax).
          - Frees ``_k_dequant_cache`` / ``_v_dequant_cache``.
          - Future ``update_and_fetch`` calls eagerly quantize new rows
            directly into ``_k_packed`` / ``_v_packed`` (no dequant caching).

        Peak memory for the KV cache drops from O(offset × head_dim × fp16)
        to O(packed_bytes_per_token × offset + tile_size × head_dim × fp16),
        at the cost of per-step dequant of tile-sized slices. Reviewer
        reports this keeps throughput flat from 1K→100K context where the
        non-tiled path OOMs or degrades 2.1×.
        """
        self.tile_size = int(tile_size)
        self.memory_pressure = True
        # Ensure any lazily-unpacked decode rows are in _k_packed before we
        # drop the dequant cache (otherwise we'd lose data).
        self._flush_unpacked()
        self._invalidate_dequant_cache()

    def decode_attention_tiled(
        self,
        queries: mx.array,
        scale: float = 1.0,
        mask: mx.array | None = None,
        tile_size: int | None = None,
    ) -> mx.array:
        """Tiled decode attention with online softmax accumulation.

        For each tile of ``tile_size`` tokens:
          1. Dequantize the packed K (and V if ``quantize_v``) tile via
             the fused Metal kernel.
          2. Compute attention scores Q·Kᵀ·scale.
          3. Update running (m, l, o) with the flash-attention recurrence:
             m_new = max(m, scores.max)
             α = exp(m − m_new)
             p = exp(scores − m_new)
             l_new = α·l + p.sum
             o_new = α·o + p·V

        Returns ``(o / l)`` cast back to the query dtype. Produces
        bit-equivalent output to monolithic MPS SDPA within fp32
        accumulation precision.

        Memory: O(tile_size · head_dim · fp32) regardless of context length.
        Requires ``self._finalized`` — callers must ensure finalize_prefill
        has run. Packs any unpacked decode rows via ``_flush_unpacked``.
        """
        assert self._finalized, "decode_attention_tiled requires finalized cache"
        if tile_size is None:
            tile_size = self.tile_size or 4096

        # Ensure packed buffers cover all rows (finalize any lazy decode rows)
        self._flush_unpacked()
        assert self._k_packed is not None
        assert self._k_norms is not None

        t = self.offset
        if t == 0:
            return mx.zeros_like(queries)

        compute_dtype = queries.dtype
        b = queries.shape[0]
        h_q = queries.shape[1]
        q_len = queries.shape[2]
        # Use the value head_dim for output (GQA: d_q may equal d_v or d_k)
        d_v_full = self._D_v or queries.shape[-1]

        queries_f32 = queries.astype(mx.float32)

        # Online softmax state (fp32 for numerical stability)
        m = mx.full((b, h_q, q_len, 1), -float("inf"), dtype=mx.float32)
        sum_exp = mx.zeros((b, h_q, q_len, 1), dtype=mx.float32)
        out = mx.zeros((b, h_q, q_len, d_v_full), dtype=mx.float32)

        h_k = self._H_k or self._k_packed.shape[1]
        n_rep = h_q // h_k if h_k > 0 else 1

        n_tiles = (t + tile_size - 1) // tile_size
        for ti in range(n_tiles):
            start = ti * tile_size
            end = min(start + tile_size, t)

            # K tile — always packed after finalize
            k_tile = dequantize_fused(
                self._k_packed[..., start:end, :],
                self._k_norms[..., start:end, :],
                out_dtype=mx.float32,
            )  # (B, H_k, tile_len, D_k)

            # V tile — packed if quantize_v else raw fp16
            if self.quantize_v and self._v_packed is not None:
                assert self._v_norms is not None
                v_tile = dequantize_fused(
                    self._v_packed[..., start:end, :],
                    self._v_norms[..., start:end, :],
                    out_dtype=mx.float32,
                )
            else:
                assert self._v_fp16 is not None
                v_tile = self._v_fp16[..., start:end, :].astype(mx.float32)

            # GQA expansion: repeat K/V heads if queries have more heads
            if n_rep > 1:
                k_tile = mx.repeat(k_tile, n_rep, axis=1)
                v_tile = mx.repeat(v_tile, n_rep, axis=1)

            # scores: (B, H_q, Q, tile_len)
            scores = mx.matmul(queries_f32, k_tile.transpose(0, 1, 3, 2)) * scale

            if mask is not None:
                # Slice the mask column-range matching this tile
                # mask shape is typically (Q, T) or broadcast-compatible
                mask_tile = mask[..., start:end]
                scores = scores + mask_tile.astype(mx.float32)

            # Online softmax update
            tile_max = scores.max(axis=-1, keepdims=True)
            m_new = mx.maximum(m, tile_max)
            alpha = mx.exp(m - m_new)
            p = mx.exp(scores - m_new)
            sum_new = alpha * sum_exp + p.sum(axis=-1, keepdims=True)
            out_new = alpha * out + mx.matmul(p, v_tile)

            m = m_new
            sum_exp = sum_new
            out = out_new

        normalized = out / mx.maximum(sum_exp, mx.array(1e-20, dtype=mx.float32))
        return normalized.astype(compute_dtype)

    def decode_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Decode-path attention.

        When ``self.tile_size`` is set (memory-pressure mode), routes
        through :meth:`decode_attention_tiled` with online softmax — peak
        memory O(tile_size) instead of O(offset).

        Otherwise routes through Apple's MPS-backed SDPA via the FP16
        dequant caches. The fused quantized Metal kernel is retained in
        :func:`fused_quantized_sdpa` for research/reference but is ~103x
        slower than MPS on Apple Silicon and is no longer on the hot path.
        """
        if self.tile_size is not None and self._finalized:
            return self.decode_attention_tiled(
                queries, scale=scale, mask=mask, tile_size=self.tile_size
            )
        if keys_state is None or values_state is None:
            keys_state, values_state = self._current_state()

        out_dtype = queries.dtype if queries.dtype in (mx.float16, mx.float32) else mx.float16

        # Both PlanarQuant → dequant caches + MPS SDPA
        if (isinstance(keys_state, PlanarQuantState)
                and isinstance(values_state, PlanarQuantState)):
            keys = self._get_dequant_k(out_dtype=out_dtype)
            values = self._get_dequant_v(out_dtype=out_dtype)
            if queries.dtype != out_dtype:
                keys = keys.astype(queries.dtype)
                values = values.astype(queries.dtype)
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=scale, mask=mask
            )

        # Mixed: K=PlanarQuant, V=FP16 → dequant K cache + MPS SDPA
        if isinstance(keys_state, PlanarQuantState) and isinstance(values_state, FP16State):
            keys = self._get_dequant_k(out_dtype=out_dtype)
            values = values_state.tensor
            if queries.dtype != out_dtype:
                keys = keys.astype(queries.dtype)
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values.astype(queries.dtype), scale=scale, mask=mask
            )

        # Both FP16 (deferred mode, or states returned from update_and_fetch
        # after the decode-quantization deferral) → plain SDPA
        if isinstance(keys_state, FP16State) and isinstance(values_state, FP16State):
            return mx.fast.scaled_dot_product_attention(
                queries,
                keys_state.tensor.astype(queries.dtype),
                values_state.tensor.astype(queries.dtype),
                scale=scale,
                mask=mask,
            )

        # Fallback: dequantize everything
        keys, values = self.dequantize(keys_state, values_state, out_dtype=out_dtype)
        if queries.dtype != out_dtype:
            keys = keys.astype(queries.dtype)
            values = values.astype(queries.dtype)
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )

    def prefill_attention(
        self,
        queries: mx.array,
        scale: float = 1.0,
        mask: mx.array | None = None,
    ) -> mx.array | None:
        return None  # signal fallback

    # ------------------------------------------------------------------
    # _BaseCache contract
    # ------------------------------------------------------------------

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return (self._k_fp16 is None and self._k_packed is None) or self.offset == 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, max(0, int(n)))
        self.offset -= n
        self._invalidate_dequant_cache()
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def _logical_len(self) -> int:
        if isinstance(self.offset, mx.array):
            return max(0, int(self.offset.max().item()))
        return max(0, int(self.offset))

    @property
    def nbytes(self) -> int:
        total = 0
        logical_len = self._logical_len()
        if self._k_fp16 is not None:
            total += int(self._k_fp16[..., :logical_len, :].nbytes)
        if self._v_fp16 is not None:
            total += int(self._v_fp16[..., :logical_len, :].nbytes)
        if self._k_packed is not None:
            total += int(self._k_packed[..., :logical_len, :].nbytes)
            total += int(self._k_norms[..., :logical_len, :].nbytes)
        if self._v_packed is not None:
            total += int(self._v_packed[..., :logical_len, :].nbytes)
            total += int(self._v_norms[..., :logical_len, :].nbytes)
        if self._k_dequant_cache is not None:
            total += int(self._k_dequant_cache[..., :self._k_dequant_offset, :].nbytes)
        if self._v_dequant_cache is not None:
            total += int(self._v_dequant_cache[..., :self._v_dequant_offset, :].nbytes)
        return total

    @property
    def state(self):
        logical_len = self._logical_len()
        if self._k_fp16 is not None and not self._finalized:
            return (self._k_fp16[..., :logical_len, :],
                    self._v_fp16[..., :logical_len, :])
        if self._k_packed is not None:
            # Pack any deferred decode rows before serializing
            self._flush_unpacked()
            k_state = PlanarQuantState(
                self._k_packed[..., :logical_len, :],
                self._k_norms[..., :logical_len, :],
            )
            if self.quantize_v and self._v_packed is not None:
                v_state = PlanarQuantState(
                    self._v_packed[..., :logical_len, :],
                    self._v_norms[..., :logical_len, :],
                )
            elif self._v_fp16 is not None:
                v_state = FP16State(self._v_fp16[..., :logical_len, :])
            else:
                v_state = None
            return _pack_state(k_state), _pack_state(v_state) if v_state else None
        return None, None

    @state.setter
    def state(self, value) -> None:
        if value is None:
            self._k_fp16 = None
            self._v_fp16 = None
            self._k_packed = None
            self._k_norms = None
            self._v_packed = None
            self._v_norms = None
            self.offset = 0
            self._finalized = True
            self._invalidate_dequant_cache()
            return
        k_tensor, v_tensor = value
        if k_tensor is None:
            self.offset = 0
            return
        # Unpack requires meta_state
        k_idx, k_norm = _unpack_state(k_tensor, self._D_k, self._packed_last_k)
        B, H_k, T, pl_k = k_idx.shape
        self._B = B
        self._H_k = H_k
        self._D_k = pl_k * 8 // 3
        self._packed_last_k = pl_k
        self._k_packed = k_idx
        self._k_norms = k_norm
        self.offset = T
        self._cap = T
        self._finalized = True

        if v_tensor is not None and self.quantize_v:
            v_idx, v_norm = _unpack_state(v_tensor, self._D_v, self._packed_last_v)
            self._H_v = v_idx.shape[1]
            self._D_v = v_idx.shape[-1] * 8 // 3
            self._packed_last_v = v_idx.shape[-1]
            self._v_packed = v_idx
            self._v_norms = v_norm
            self.quantize_v = True
        elif v_tensor is not None:
            self._H_v = v_tensor.shape[1]
            self._D_v = v_tensor.shape[-1]
            self._packed_last_v = self._D_v // 4 + self._D_v // 8
            self._v_fp16 = v_tensor.astype(mx.float16)
            self._v_packed = None
            self._v_norms = None
            self.quantize_v = False
        else:
            self.quantize_v = False

    @property
    def meta_state(self) -> tuple[str, ...]:
        return tuple(map(str, (
            self.offset,
            self.bits,
            int(self.quantize_v),
            self._D_k or 0,
            self._D_v or 0,
            self._packed_last_k or 0,
            self._packed_last_v or 0,
        )))

    @meta_state.setter
    def meta_state(self, value) -> None:
        if not value:
            return
        vals = list(value)
        self.offset = int(vals[0])
        self.bits = float(vals[1])
        self.quantize_v = bool(int(vals[2]))
        if len(vals) >= 7:
            self._D_k = int(vals[3]) or None
            self._D_v = int(vals[4]) or None
            self._packed_last_k = int(vals[5]) or None
            self._packed_last_v = int(vals[6]) or None


# ---------------------------------------------------------------------------
# State packing for safetensors round-trip
# ---------------------------------------------------------------------------

def _pack_state(state) -> mx.array | None:
    if state is None:
        return None
    if isinstance(state, FP16State):
        return state.tensor
    if isinstance(state, PlanarQuantState):
        # Concatenate packed (uint8→uint16) + norms (fp16→uint16 view)
        idx_u16 = state.packed.astype(mx.uint16)
        norm_u16 = state.norms.astype(mx.float16).view(mx.uint16)
        return mx.concatenate([idx_u16, norm_u16], axis=-1)
    return None


def _unpack_state(packed: mx.array, D: int | None, packed_last: int | None):
    if D is not None and packed_last is not None:
        # packed_last indices + 1 norm scalar
        idx = packed[..., :packed_last].astype(mx.uint8)
        norm_u16 = packed[..., packed_last:]
        norms = norm_u16.view(mx.float16)
        return idx, norms
    # Fallback: assume 1 norm at end
    packed_last = packed.shape[-1] - 1
    idx = packed[..., :packed_last].astype(mx.uint8)
    norm_u16 = packed[..., packed_last:]
    norms = norm_u16.view(mx.float16)
    return idx, norms


# ---------------------------------------------------------------------------
# Batch variant
# ---------------------------------------------------------------------------

class BatchPlanarQuantKVCache(PlanarQuantKVCache):
    """Batch-aware PlanarQuant3 KV cache for continuous batching."""

    def __init__(
        self,
        left_padding: list[int] | None = None,
        bits: float = 3.0,
        quantize_v: bool = True,
    ):
        super().__init__(bits=bits, quantize_v=quantize_v)
        self.left_padding = left_padding or [0]
        self._batch_size = len(self.left_padding)
        self._right_padding: mx.array | None = None
        if self._batch_size > 1:
            self.offset = mx.array([-lp for lp in self.left_padding])
        else:
            self.offset = -self.left_padding[0]

    def make_mask(self, *args, **kwargs):
        try:
            from mlx_lm.models.cache import create_causal_mask
        except ImportError:
            create_causal_mask = None

        if isinstance(self.offset, int):
            return create_attention_mask(*args, offset=self.offset, **kwargs)
        if create_causal_mask is None:
            return create_attention_mask(*args, offset=0, **kwargs)
        return create_causal_mask(
            args[0],
            offset=self.offset,
            left_padding=mx.array(self.left_padding),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Batch-aware overrides for array offset
    # ------------------------------------------------------------------

    def _max_offset(self) -> int:
        """Return max offset across batch (int)."""
        if isinstance(self.offset, mx.array):
            return int(self.offset.max().item())
        return self.offset

    def _ensure_k_dequant_cache(self) -> None:
        max_off = self._max_offset()
        if self._k_dequant_cache is not None and self._k_dequant_offset == max_off:
            return
        assert self._B is not None and self._H_k is not None and self._D_k is not None
        cache = mx.zeros((self._B, self._H_k, self._cap, self._D_k), dtype=mx.float16)
        if max_off > 0:
            assert self._k_packed is not None
            assert self._k_norms is not None
            k_dq = dequantize_fused(
                self._k_packed[..., :max_off, :],
                self._k_norms[..., :max_off, :],
                out_dtype=mx.float16,
            )
            cache[..., :max_off, :] = k_dq.astype(mx.float16)
        self._k_dequant_cache = cache
        self._k_dequant_offset = max_off

    def _ensure_v_dequant_cache(self) -> None:
        max_off = self._max_offset()
        if self._v_dequant_cache is not None and self._v_dequant_offset == max_off:
            return
        assert self._B is not None and self._H_v is not None and self._D_v is not None
        cache = mx.zeros((self._B, self._H_v, self._cap, self._D_v), dtype=mx.float16)
        if max_off > 0 and self._v_packed is not None and self._v_norms is not None:
            v_dq = dequantize_fused(
                self._v_packed[..., :max_off, :],
                self._v_norms[..., :max_off, :],
                out_dtype=mx.float16,
            )
            cache[..., :max_off, :] = v_dq.astype(mx.float16)
        self._v_dequant_cache = cache
        self._v_dequant_offset = max_off

    def finalize_prefill(self) -> None:
        if self._finalized:
            return
        if self._k_fp16 is None:
            return
        max_off = self._max_offset()
        assert self._D_k is not None
        assert self._packed_last_k is not None

        B, H_k = self._B, self._H_k
        cap = self._cap

        # Quantize K — use max_off for the packed slice
        k_packed, k_norms = _quantize(self._k_fp16[..., :max_off, :])
        self._k_packed = mx.zeros((B, H_k, cap, self._packed_last_k), dtype=mx.uint8)
        self._k_norms = mx.zeros((B, H_k, cap, 1), dtype=mx.float16)
        self._k_packed[..., :max_off, :] = k_packed.astype(mx.uint8)
        self._k_norms[..., :max_off, :] = k_norms.astype(mx.float16)

        if self.quantize_v:
            assert self._v_fp16 is not None
            assert self._packed_last_v is not None
            _, H_v = self._B, self._H_v
            v_packed, v_norms = _quantize(self._v_fp16[..., :max_off, :])
            self._v_packed = mx.zeros((B, H_v, cap, self._packed_last_v), dtype=mx.uint8)
            self._v_norms = mx.zeros((B, H_v, cap, 1), dtype=mx.float16)
            self._v_packed[..., :max_off, :] = v_packed.astype(mx.uint8)
            self._v_norms[..., :max_off, :] = v_norms.astype(mx.float16)
            self._v_fp16 = None
        else:
            v_fp16 = self._v_fp16[..., :max_off, :]
            self._v_packed = None
            self._v_norms = None
            self._v_fp16 = mx.zeros((B, self._H_v, cap, self._D_v), dtype=mx.float16)
            self._v_fp16[..., :max_off, :] = v_fp16

        self._k_fp16 = None
        self._finalized = True
        self._invalidate_dequant_cache()
        logger.info("PlanarQuant batch: finalized prefill, converted to packed layout")

    # ------------------------------------------------------------------
    # Packed-state batch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_packed_state(
        state: PlanarQuantState, indices
    ) -> PlanarQuantState:
        packed = state.packed[indices]
        norms = state.norms[indices]
        return PlanarQuantState(packed, norms)

    @staticmethod
    def _concat_packed_batch(
        states: list[PlanarQuantState],
    ) -> PlanarQuantState:
        packed = mx.concatenate([s.packed for s in states], axis=0)
        norms = mx.concatenate([s.norms for s in states], axis=0)
        return PlanarQuantState(packed, norms)

    @staticmethod
    def _pad_packed_left(
        state: PlanarQuantState, pad: int
    ) -> PlanarQuantState:
        if pad == 0:
            return state
        B, H, T, C = state.packed.shape
        packed_pad = mx.zeros((B, H, pad, C), dtype=mx.uint8)
        norms_pad = mx.zeros((B, H, pad, state.norms.shape[-1]), dtype=mx.float16)
        packed = mx.concatenate([packed_pad, state.packed], axis=2)
        norms = mx.concatenate([norms_pad, state.norms], axis=2)
        return PlanarQuantState(packed, norms)

    @staticmethod
    def _slice_packed_range(
        state: PlanarQuantState, start: int, end: int
    ) -> PlanarQuantState:
        packed = state.packed[..., start:end, :]
        norms = state.norms[..., start:end, :]
        return PlanarQuantState(packed, norms)

    @staticmethod
    def _packed_state_length(state: PlanarQuantState) -> int:
        return state.packed.shape[2]

    @staticmethod
    def _write_batch_rows(
        buf: mx.array,
        new: mx.array,
        offsets: mx.array,
    ) -> tuple[mx.array, int | None, int | None]:
        """Write per-row chunks when offsets may start inside left padding."""
        min_start: int | None = None
        max_end: int | None = None
        B = new.shape[0]
        L = new.shape[2]
        for b in range(B):
            off = int(offsets[b].item())
            src_start = max(0, -off)
            if src_start >= L:
                continue
            dst_start = max(0, off)
            n = L - src_start
            dst_end = dst_start + n
            buf[b, :, dst_start:dst_end, :] = new[
                b : b + 1, :, src_start:, :
            ].astype(buf.dtype)[0]
            min_start = dst_start if min_start is None else min(min_start, dst_start)
            max_end = dst_end if max_end is None else max(max_end, dst_end)
        return buf, min_start, max_end

    # ------------------------------------------------------------------
    # update_and_fetch (batch override)
    # ------------------------------------------------------------------

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple:
        B = keys.shape[0]
        L = keys.shape[2]

        if self._k_fp16 is None and self._k_packed is None:
            self._init_buffers(keys, values)
            # Override offset to array for B>1
            if self._batch_size > 1 and not isinstance(self.offset, mx.array):
                # Already set as array in __init__
                pass
            elif B > 1 and isinstance(self.offset, int):
                self.offset = mx.array([self.offset] * B)

        max_off = self._max_offset()
        new_max = max_off + L

        if not self._finalized:
            self._grow_fp16(new_max)
            assert self._k_fp16 is not None
            assert self._v_fp16 is not None
            if isinstance(self.offset, mx.array):
                self._k_fp16, _, _ = self._write_batch_rows(
                    self._k_fp16, keys, self.offset
                )
                self._v_fp16, _, _ = self._write_batch_rows(
                    self._v_fp16, values, self.offset
                )
                self.offset = self.offset + L
            else:
                self._k_fp16 = self._write_slice(self._k_fp16, keys, self.offset)
                self._v_fp16 = self._write_slice(self._v_fp16, values, self.offset)
                self.offset = self.offset + L

            # Return full buffer up to max valid position
            max_valid = max(0, self._max_offset())
            return (
                FP16State(self._k_fp16[..., :max_valid, :]),
                FP16State(self._v_fp16[..., :max_valid, :]),
            )

        # Quantized mode — batch variant
        self._grow_packed(new_max)
        self._ensure_k_dequant_cache()
        self._grow_k_dequant_cache(new_max)

        assert self._k_dequant_cache is not None
        if isinstance(self.offset, mx.array):
            self._k_dequant_cache, k_start, k_end = self._write_batch_rows(
                self._k_dequant_cache, keys, self.offset
            )
            if k_start is not None:
                if self._k_unpacked_start is None:
                    self._k_unpacked_start = k_start
                else:
                    self._k_unpacked_start = min(self._k_unpacked_start, k_start)
                self._k_unpacked_end = max(self._k_unpacked_end or 0, k_end or k_start)
        else:
            k_fp16 = keys.astype(mx.float16)
            self._k_dequant_cache[..., self.offset:new_max, :] = k_fp16
            if self._k_unpacked_start is None:
                self._k_unpacked_start = self.offset
            self._k_unpacked_end = new_max

        self._k_dequant_offset = new_max

        if self.quantize_v:
            self._ensure_v_dequant_cache()
            self._grow_v_dequant_cache(new_max)
            assert self._v_dequant_cache is not None
            if isinstance(self.offset, mx.array):
                self._v_dequant_cache, v_start, v_end = self._write_batch_rows(
                    self._v_dequant_cache, values, self.offset
                )
                if v_start is not None:
                    if self._v_unpacked_start is None:
                        self._v_unpacked_start = v_start
                    else:
                        self._v_unpacked_start = min(self._v_unpacked_start, v_start)
                    self._v_unpacked_end = max(
                        self._v_unpacked_end or 0, v_end or v_start
                    )
            else:
                v_fp16 = values.astype(mx.float16)
                self._v_dequant_cache[..., self.offset:new_max, :] = v_fp16
                if self._v_unpacked_start is None:
                    self._v_unpacked_start = self.offset
                self._v_unpacked_end = new_max
            self._v_dequant_offset = new_max

            if isinstance(self.offset, mx.array):
                self.offset = self.offset + L
            else:
                self.offset = new_max

            max_valid = max(0, self._max_offset())
            return (
                FP16State(self._k_dequant_cache[..., :max_valid, :]),
                FP16State(self._v_dequant_cache[..., :max_valid, :]),
            )

        # Asymmetric V
        assert self._v_fp16 is not None
        if isinstance(self.offset, mx.array):
            self._v_fp16, _, _ = self._write_batch_rows(
                self._v_fp16, values, self.offset
            )
            self.offset = self.offset + L
        else:
            self._v_fp16 = self._write_slice(self._v_fp16, values, self.offset)
            self.offset = new_max

        max_valid = max(0, self._max_offset())
        return (
            FP16State(self._k_dequant_cache[..., :max_valid, :]),
            FP16State(self._v_fp16[..., :max_valid, :]),
        )

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def prepare(self, left_padding=None, right_padding=None) -> None:
        if left_padding is not None:
            left_padding = mx.array(left_padding)
            cur_max = self._max_offset() if isinstance(self.offset, mx.array) else self.offset
            if cur_max > 0:
                raise ValueError("left_padding prepare only allowed on empty cache")
            self.left_padding = left_padding
            self._batch_size = len(left_padding)
            self.offset = -left_padding
        if right_padding is not None:
            if isinstance(right_padding, (list, tuple)):
                self._right_padding = mx.array(right_padding)
            else:
                self._right_padding = right_padding

    def finalize(self) -> None:
        if self._right_padding is not None:
            rp = self._right_padding

            def _roll_b(t: mx.array, b: int, n: int, T: int) -> mx.array:
                # Roll [0:T] left by n, preserve unused tail [T:cap]
                cap = t.shape[2]
                return mx.concatenate([
                    t[b, :, n:T, :],
                    t[b, :, :n, :],
                    t[b, :, T:cap, :],
                ], axis=1)

            def _off(b: int) -> int:
                return (
                    int(self.offset[b].item())
                    if isinstance(self.offset, mx.array)
                    else int(self.offset)
                )

            if not self._finalized:
                # Deferred mode: roll FP16 buffers
                if self._k_fp16 is not None and self._v_fp16 is not None:
                    B = self._k_fp16.shape[0]
                    for b in range(B):
                        n = int(rp[b].item())
                        if n > 0:
                            T = _off(b)
                            self._k_fp16[b] = _roll_b(self._k_fp16, b, n, T)
                            self._v_fp16[b] = _roll_b(self._v_fp16, b, n, T)
                    self.left_padding = mx.array(self.left_padding) + rp
            else:
                # Quantized mode: roll packed+norms
                B = self._k_packed.shape[0] if self._k_packed is not None else 0
                for b in range(B):
                    n = int(rp[b].item())
                    if n > 0:
                        T = _off(b)
                        if self._k_packed is not None and self._k_norms is not None:
                            self._k_packed[b] = _roll_b(self._k_packed, b, n, T)
                            self._k_norms[b] = _roll_b(self._k_norms, b, n, T)
                        if self.quantize_v and self._v_packed is not None and self._v_norms is not None:
                            self._v_packed[b] = _roll_b(self._v_packed, b, n, T)
                            self._v_norms[b] = _roll_b(self._v_norms, b, n, T)
                        elif not self.quantize_v and self._v_fp16 is not None:
                            self._v_fp16[b] = _roll_b(self._v_fp16, b, n, T)
                self.left_padding = mx.array(self.left_padding) + rp
                self._invalidate_dequant_cache()
            self._right_padding = None
        else:
            # No right padding — just finalize prefill if needed
            if not self._finalized:
                self.finalize_prefill()

    def filter(self, indices: list[int]) -> None:
        if not self._finalized:
            # Deferred mode: filter FP16 buffers
            idx = mx.array(indices)
            if self._k_fp16 is not None:
                self._k_fp16 = self._k_fp16[idx]
            if self._v_fp16 is not None:
                self._v_fp16 = self._v_fp16[idx]
        else:
            # Quantized mode: filter packed buffers
            idx = mx.array(indices)
            if self._k_packed is not None:
                self._k_packed = self._k_packed[idx]
            if self._k_norms is not None:
                self._k_norms = self._k_norms[idx]
            if self.quantize_v:
                if self._v_packed is not None:
                    self._v_packed = self._v_packed[idx]
                if self._v_norms is not None:
                    self._v_norms = self._v_norms[idx]
            else:
                if self._v_fp16 is not None:
                    self._v_fp16 = self._v_fp16[idx]

        # Filter dequant caches
        if self._k_dequant_cache is not None:
            idx_mx = mx.array(indices)
            self._k_dequant_cache = self._k_dequant_cache[idx_mx]
        if self._v_dequant_cache is not None:
            idx_mx = mx.array(indices)
            self._v_dequant_cache = self._v_dequant_cache[idx_mx]

        # Reset unpacked ranges
        self._k_unpacked_start = None
        self._k_unpacked_end = None
        self._v_unpacked_start = None
        self._v_unpacked_end = None

        # Update offset, left_padding, batch_size
        if isinstance(self.offset, mx.array):
            self.offset = self.offset[idx]
        if not isinstance(self.left_padding, mx.array):
            self.left_padding = mx.array(self.left_padding)
        self.left_padding = self.left_padding[idx]
        self._batch_size = len(indices)
        if self._B is not None:
            self._B = self._batch_size

    def extend(self, other: BatchPlanarQuantKVCache) -> None:
        def _pad_cap(a: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
            # Pad axis=2 (seq/cap dim) to max(a, b) with zeros so axis=0 concat works
            ca, cb = a.shape[2], b.shape[2]
            if ca == cb:
                return a, b
            target = max(ca, cb)

            def _pad(t: mx.array) -> mx.array:
                if t.shape[2] == target:
                    return t
                shp = list(t.shape)
                shp[2] = target - t.shape[2]
                return mx.concatenate([t, mx.zeros(shp, dtype=t.dtype)], axis=2)

            return _pad(a), _pad(b)

        def _cat0(a: mx.array, b: mx.array) -> mx.array:
            a2, b2 = _pad_cap(a, b)
            return mx.concatenate([a2, b2], axis=0)

        if not self._finalized and not other._finalized:
            # Both deferred — concat FP16 buffers
            if self._k_fp16 is not None and other._k_fp16 is not None:
                self._k_fp16 = _cat0(self._k_fp16, other._k_fp16)
            if self._v_fp16 is not None and other._v_fp16 is not None:
                self._v_fp16 = _cat0(self._v_fp16, other._v_fp16)
        else:
            # At least one finalized — ensure both are finalized
            if not self._finalized:
                self.finalize_prefill()
            if not other._finalized:
                other.finalize_prefill()
            # Concat packed buffers
            if self._k_packed is not None and other._k_packed is not None:
                self._k_packed = _cat0(self._k_packed, other._k_packed)
            if self._k_norms is not None and other._k_norms is not None:
                self._k_norms = _cat0(self._k_norms, other._k_norms)
            if self.quantize_v:
                if self._v_packed is not None and other._v_packed is not None:
                    self._v_packed = _cat0(self._v_packed, other._v_packed)
                if self._v_norms is not None and other._v_norms is not None:
                    self._v_norms = _cat0(self._v_norms, other._v_norms)
            else:
                if self._v_fp16 is not None and other._v_fp16 is not None:
                    self._v_fp16 = _cat0(self._v_fp16, other._v_fp16)

        # Extend dequant caches
        if self._k_dequant_cache is not None and other._k_dequant_cache is not None:
            self._k_dequant_cache = mx.concatenate(
                [self._k_dequant_cache, other._k_dequant_cache], axis=0
            )
        if self._v_dequant_cache is not None and other._v_dequant_cache is not None:
            self._v_dequant_cache = mx.concatenate(
                [self._v_dequant_cache, other._v_dequant_cache], axis=0
            )

        # Merge offsets
        if isinstance(self.offset, mx.array) and isinstance(other.offset, mx.array):
            self.offset = mx.concatenate([self.offset, other.offset])
        elif isinstance(self.offset, int) and isinstance(other.offset, int):
            self.offset = mx.array([self.offset, other.offset])
        elif isinstance(self.offset, int) and isinstance(other.offset, mx.array):
            self.offset = mx.concatenate([mx.array([self.offset]), other.offset])
        elif isinstance(self.offset, mx.array) and isinstance(other.offset, int):
            self.offset = mx.concatenate([self.offset, mx.array([other.offset])])

        # Merge left_padding
        lp1 = self.left_padding if isinstance(self.left_padding, mx.array) else mx.array(self.left_padding)
        lp2 = other.left_padding if isinstance(other.left_padding, mx.array) else mx.array(other.left_padding)
        self.left_padding = mx.concatenate([lp1, lp2])
        self._batch_size = len(self.left_padding)
        if self._B is not None:
            self._B = self._batch_size

        # Reset unpacked ranges
        self._k_unpacked_start = None
        self._k_unpacked_end = None
        self._v_unpacked_start = None
        self._v_unpacked_end = None

    @classmethod
    def merge(
        cls,
        caches: list[PlanarQuantKVCache],
    ) -> BatchPlanarQuantKVCache:
        if not caches:
            raise ValueError("Cannot merge empty list of caches")
        # Auto-finalize any deferred caches
        for c in caches:
            if not c._finalized:
                c.finalize_prefill()
        # Find max length
        max_len = max(c.offset for c in caches)
        # Build merged batch
        merged = cls(
            left_padding=[0] * len(caches),
            bits=caches[0].bits,
            quantize_v=caches[0].quantize_v,
        )
        merged._finalized = True
        first = caches[0]
        merged._B = len(caches)
        merged._H_k = first._H_k
        merged._H_v = first._H_v
        merged._D_k = first._D_k
        merged._D_v = first._D_v
        merged._packed_last_k = first._packed_last_k
        merged._packed_last_v = first._packed_last_v
        merged._cap = max_len
        merged._batch_size = len(caches)

        # Concatenate K packed states
        k_packed_list = []
        k_norms_list = []
        offsets = []
        left_pads = []
        for c in caches:
            left_pad = max_len - c.offset
            left_pads.append(left_pad)
            offsets.append(c.offset)
            if left_pad > 0 and c._k_packed is not None:
                state = PlanarQuantState(
                    c._k_packed[..., :c.offset, :],
                    c._k_norms[..., :c.offset, :],
                )
                state = cls._pad_packed_left(state, left_pad)
                k_packed_list.append(state.packed)
                k_norms_list.append(state.norms)
            elif c._k_packed is not None:
                k_packed_list.append(c._k_packed[..., :c.offset, :])
                k_norms_list.append(c._k_norms[..., :c.offset, :])

        merged._k_packed = mx.concatenate(k_packed_list, axis=0)
        merged._k_norms = mx.concatenate(k_norms_list, axis=0)

        # V state
        if first.quantize_v:
            v_packed_list = []
            v_norms_list = []
            for c in caches:
                left_pad = max_len - c.offset
                if left_pad > 0 and c._v_packed is not None:
                    state = PlanarQuantState(
                        c._v_packed[..., :c.offset, :],
                        c._v_norms[..., :c.offset, :],
                    )
                    state = cls._pad_packed_left(state, left_pad)
                    v_packed_list.append(state.packed)
                    v_norms_list.append(state.norms)
                elif c._v_packed is not None:
                    v_packed_list.append(c._v_packed[..., :c.offset, :])
                    v_norms_list.append(c._v_norms[..., :c.offset, :])
            merged._v_packed = mx.concatenate(v_packed_list, axis=0)
            merged._v_norms = mx.concatenate(v_norms_list, axis=0)
        else:
            v_fp16_list = []
            for c in caches:
                left_pad = max_len - c.offset
                if left_pad > 0 and c._v_fp16 is not None:
                    pad_shape = list(c._v_fp16.shape)
                    pad_shape[2] = left_pad
                    v_pad = mx.zeros(tuple(pad_shape), dtype=mx.float16)
                    v_fp16_list.append(mx.concatenate(
                        [v_pad, c._v_fp16[..., :c.offset, :]], axis=2
                    ))
                elif c._v_fp16 is not None:
                    v_fp16_list.append(c._v_fp16[..., :c.offset, :])
            merged._v_fp16 = mx.concatenate(v_fp16_list, axis=0)

        merged.left_padding = mx.array(left_pads)
        merged.offset = mx.array(offsets)

        # Carry dequant caches
        k_dq_list = [c._k_dequant_cache for c in caches if c._k_dequant_cache is not None]
        if k_dq_list:
            merged._k_dequant_cache = mx.concatenate(k_dq_list, axis=0)
            merged._k_dequant_offset = max_len
        v_dq_list = [c._v_dequant_cache for c in caches if c._v_dequant_cache is not None]
        if v_dq_list:
            merged._v_dequant_cache = mx.concatenate(v_dq_list, axis=0)
            merged._v_dequant_offset = max_len

        return merged

    def extract(self, index: int) -> PlanarQuantKVCache:
        single = PlanarQuantKVCache(bits=self.bits, quantize_v=self.quantize_v)
        single._finalized = self._finalized
        single._B = 1
        single._H_k = self._H_k
        single._H_v = self._H_v
        single._D_k = self._D_k
        single._D_v = self._D_v
        single._packed_last_k = self._packed_last_k
        single._packed_last_v = self._packed_last_v

        if isinstance(self.offset, mx.array):
            single.offset = int(self.offset[index].item())
        else:
            single.offset = self.offset

        lp = int(self.left_padding[index].item()) if isinstance(self.left_padding, mx.array) else 0
        T = single.offset

        if self._k_packed is not None:
            # Extract row, removing left padding
            k_p = self._k_packed[index:index + 1]
            k_n = self._k_norms[index:index + 1]
            if lp > 0:
                k_p = k_p[:, :, lp:, :]
                k_n = k_n[:, :, lp:, :]
            single._k_packed = k_p
            single._k_norms = k_n
            single._cap = k_p.shape[2]
        else:
            single._cap = T

        if self.quantize_v and self._v_packed is not None:
            v_p = self._v_packed[index:index + 1]
            v_n = self._v_norms[index:index + 1]
            if lp > 0:
                v_p = v_p[:, :, lp:, :]
                v_n = v_n[:, :, lp:, :]
            single._v_packed = v_p
            single._v_norms = v_n
        elif self._v_fp16 is not None:
            v_f = self._v_fp16[index:index + 1]
            if lp > 0:
                v_f = v_f[:, :, lp:, :]
            single._v_fp16 = mx.zeros((1, self._H_v, single._cap, self._D_v), dtype=mx.float16)
            single._v_fp16[..., :T, :] = v_f[:, :, :T, :]

        return single

    def evict_dequant_caches(self) -> int:
        freed = 0
        if self._k_dequant_cache is not None:
            freed += int(self._k_dequant_cache.nbytes)
            self._k_dequant_cache = None
            self._k_dequant_offset = 0
        if self._v_dequant_cache is not None:
            freed += int(self._v_dequant_cache.nbytes)
            self._v_dequant_cache = None
            self._v_dequant_offset = 0
        return freed

    def _check_invariants(self) -> list[str]:
        violations = []
        if self._k_packed is not None and self._k_norms is not None:
            if self._k_packed.shape[2] != self._k_norms.shape[2]:
                violations.append(
                    f"K: packed T={self._k_packed.shape[2]} vs norms T={self._k_norms.shape[2]}"
                )
            if self._k_packed.shape[0] != self._batch_size:
                violations.append(
                    f"K: packed B={self._k_packed.shape[0]} vs batch_size={self._batch_size}"
                )
        if self.quantize_v and self._v_packed is not None and self._v_norms is not None:
            if self._v_packed.shape[2] != self._v_norms.shape[2]:
                violations.append(
                    f"V: packed T={self._v_packed.shape[2]} vs norms T={self._v_norms.shape[2]}"
                )
        if isinstance(self.offset, mx.array) and self.offset.shape[0] != self._batch_size:
            violations.append(
                f"offset len={self.offset.shape[0]} vs batch_size={self._batch_size}"
            )
        if isinstance(self.left_padding, mx.array) and self.left_padding.shape[0] != self._batch_size:
            violations.append(
                f"left_padding len={self.left_padding.shape[0]} vs batch_size={self._batch_size}"
            )
        return violations

    def decode_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask: mx.array | None = None,
    ) -> mx.array:
        if keys_state is not None or values_state is not None:
            keys, values = self.dequantize(
                keys_state=keys_state,
                values_state=values_state,
                out_dtype=queries.dtype,
            )
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=scale, mask=mask
            )

        self._ensure_k_dequant_cache()
        keys = self._k_dequant_cache[..., :self._k_dequant_offset, :].astype(queries.dtype)
        if self.quantize_v:
            self._ensure_v_dequant_cache()
            values = self._v_dequant_cache[..., :self._v_dequant_offset, :].astype(queries.dtype)
        else:
            assert self._v_fp16 is not None
            if isinstance(self.offset, mx.array):
                max_off = int(self.offset.max().item())
            else:
                max_off = self.offset
            values = self._v_fp16[..., :max_off, :].astype(queries.dtype)
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )


# ---------------------------------------------------------------------------
# Module-level helper aliases (exported for test access)
# ---------------------------------------------------------------------------

_concat_packed_batch = BatchPlanarQuantKVCache._concat_packed_batch
_filter_packed_state = BatchPlanarQuantKVCache._filter_packed_state
_pad_packed_left = BatchPlanarQuantKVCache._pad_packed_left
_packed_state_length = BatchPlanarQuantKVCache._packed_state_length
_slice_packed_range = BatchPlanarQuantKVCache._slice_packed_range


__all__ = [
    "PlanarQuantKVCache",
    "BatchPlanarQuantKVCache",
    "PlanarQuantState",
    "FP16State",
    "_concat_packed_batch",
    "_filter_packed_state",
    "_pad_packed_left",
    "_packed_state_length",
    "_slice_packed_range",
]
