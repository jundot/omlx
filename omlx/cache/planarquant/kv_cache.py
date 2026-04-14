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

from .constants import PLANAR_D, PLANAR_QS_SIZE, PLANAR_SIGNS_SIZE, PLANAR_BLOCK_BYTES
from .metal_kernels import dequantize_fused, fused_quantized_sdpa
from .reference import dequantize_block, quantize_block

logger = logging.getLogger(__name__)

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

        # Shape memo
        self._B: int | None = None
        self._H_k: int | None = None
        self._H_v: int | None = None
        self._D_k: int | None = None
        self._D_v: int | None = None
        self._packed_last_k: int | None = None
        self._packed_last_v: int | None = None

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _init_buffers(self, keys: mx.array, values: mx.array) -> None:
        B, H_k, _, D_k = keys.shape
        _, H_v, _, D_v = values.shape
        if D_k % 2 != 0 or D_v % 2 != 0:
            raise ValueError(
                f"PlanarQuantKVCache requires even head_dim; "
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
        self._cap = new_cap

    @staticmethod
    def _write_slice(buf: mx.array, new: mx.array, start: int) -> mx.array:
        L = new.shape[2]
        end = start + L
        buf[..., start:end, :] = new.astype(buf.dtype)
        return buf

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
        k_packed, k_norms = quantize_block(self._k_fp16[..., :self.offset, :])
        # Reshape: quantize_block returns (B, H, T, packed_last) and (B, H, T, 1)
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
            v_packed, v_norms = quantize_block(self._v_fp16[..., :self.offset, :])
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
        logger.info("PlanarQuant: finalized prefill, converted to packed layout")

    # ------------------------------------------------------------------
    # mlx-lm cache contract
    # ------------------------------------------------------------------

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple:
        """Insert new K/V and return current state.

        During prefill (before finalize_prefill): stores FP16.
        During decode (after finalize_prefill): quantizes on insertion.
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

        # Quantized mode: pack on insertion
        self._grow_packed(new_end)
        assert self._k_packed is not None
        assert self._k_norms is not None

        k_packed, k_norm = quantize_block(keys)
        self._k_packed = self._write_slice(self._k_packed, k_packed, self.offset)
        self._k_norms = self._write_slice(self._k_norms, k_norm, self.offset)

        if self.quantize_v:
            assert self._v_packed is not None
            assert self._v_norms is not None
            v_packed, v_norm = quantize_block(values)
            self._v_packed = self._write_slice(self._v_packed, v_packed, self.offset)
            self._v_norms = self._write_slice(self._v_norms, v_norm, self.offset)
            self.offset = new_end
            return (
                PlanarQuantState(
                    self._k_packed[..., :self.offset, :],
                    self._k_norms[..., :self.offset, :],
                ),
                PlanarQuantState(
                    self._v_packed[..., :self.offset, :],
                    self._v_norms[..., :self.offset, :],
                ),
            )

        # Asymmetric: V stays FP16
        assert self._v_fp16 is not None
        self._v_fp16 = self._write_slice(self._v_fp16, values, self.offset)
        self.offset = new_end
        return (
            PlanarQuantState(
                self._k_packed[..., :self.offset, :],
                self._k_norms[..., :self.offset, :],
            ),
            FP16State(self._v_fp16[..., :self.offset, :]),
        )

    # ------------------------------------------------------------------
    # Dequant + attention
    # ------------------------------------------------------------------

    def _current_state(self) -> tuple:
        if not self._finalized:
            assert self._k_fp16 is not None
            assert self._v_fp16 is not None
            return (
                FP16State(self._k_fp16[..., :self.offset, :]),
                FP16State(self._v_fp16[..., :self.offset, :]),
            )
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

    def decode_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Decode-path attention with fused inline dequant."""
        if keys_state is None or values_state is None:
            keys_state, values_state = self._current_state()

        # Both PlanarQuant → fused SDPA
        if (isinstance(keys_state, PlanarQuantState)
                and isinstance(values_state, PlanarQuantState)
                and mask is None
                and queries.shape[-2] == 1):
            return fused_quantized_sdpa(
                queries,
                k_packed=keys_state.packed,
                k_norms=keys_state.norms[..., 0],  # squeeze last dim
                v_packed=values_state.packed,
                v_norms=values_state.norms[..., 0],
                scale=scale,
            )

        # Mixed: K=PlanarQuant, V=FP16 → dequant K only, then SDPA
        if isinstance(keys_state, PlanarQuantState) and isinstance(values_state, FP16State):
            out_dtype = queries.dtype if queries.dtype in (mx.float16, mx.float32) else mx.float16
            keys, _ = self.dequantize(keys_state, values_state, out_dtype=out_dtype)
            values = values_state.tensor
            if queries.dtype != out_dtype:
                keys = keys.astype(queries.dtype)
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values.astype(queries.dtype), scale=scale, mask=mask
            )

        # Both FP16 (deferred mode) → plain SDPA
        if isinstance(keys_state, FP16State) and isinstance(values_state, FP16State):
            return mx.fast.scaled_dot_product_attention(
                queries,
                keys_state.tensor.astype(queries.dtype),
                values_state.tensor.astype(queries.dtype),
                scale=scale,
                mask=mask,
            )

        # Fallback: dequantize everything
        out_dtype = queries.dtype if queries.dtype in (mx.float16, mx.float32) else mx.float16
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
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @property
    def nbytes(self) -> int:
        total = 0
        if self._k_fp16 is not None:
            total += int(self._k_fp16[..., :self.offset, :].nbytes)
        if self._v_fp16 is not None:
            total += int(self._v_fp16[..., :self.offset, :].nbytes)
        if self._k_packed is not None:
            total += int(self._k_packed[..., :self.offset, :].nbytes)
            total += int(self._k_norms[..., :self.offset, :].nbytes)
        if self._v_packed is not None:
            total += int(self._v_packed[..., :self.offset, :].nbytes)
            total += int(self._v_norms[..., :self.offset, :].nbytes)
        return total

    @property
    def state(self):
        if self._k_fp16 is not None and not self._finalized:
            return (self._k_fp16[..., :self.offset, :],
                    self._v_fp16[..., :self.offset, :])
        if self._k_packed is not None:
            k_state = PlanarQuantState(
                self._k_packed[..., :self.offset, :],
                self._k_norms[..., :self.offset, :],
            )
            if self.quantize_v and self._v_packed is not None:
                v_state = PlanarQuantState(
                    self._v_packed[..., :self.offset, :],
                    self._v_norms[..., :self.offset, :],
                )
            elif self._v_fp16 is not None:
                v_state = FP16State(self._v_fp16[..., :self.offset, :])
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

        if v_tensor is not None:
            v_idx, v_norm = _unpack_state(v_tensor, self._D_v, self._packed_last_v)
            self._H_v = v_idx.shape[1]
            self._D_v = v_idx.shape[-1] * 8 // 3
            self._packed_last_v = v_idx.shape[-1]
            self._v_packed = v_idx
            self._v_norms = v_norm
            self.quantize_v = True
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


__all__ = [
    "PlanarQuantKVCache",
    "BatchPlanarQuantKVCache",
    "PlanarQuantState",
    "FP16State",
]
