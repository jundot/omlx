# SPDX-License-Identifier: Apache-2.0
"""Pure-MLX reference implementation of PlanarQuant3 quantize/dequantize.

Packed storage layout matches upstream block_planar3_0:
  norm:  fp16, 2 bytes
  qs:    D/4 bytes, 4 lower-2-bit indices per byte
  signs: D/8 bytes, 8 upper-1-bit signs per byte
  Total: 50 bytes per D=128 block = 0.39 bytes/elem = 5.1x compression vs fp16

Port of quantize_row_planar3_0_ref / dequantize_row_planar3_0 from:
https://github.com/johndpope/llama-cpp-turboquant/blob/feature/planarquant-kv-cache/ggml/src/ggml-planar-quant.c
"""

from __future__ import annotations

import mlx.core as mx

from .constants import centroids_mx, cos_sin_mx, midpoints_mx


def quantize_block(x: mx.array) -> tuple[mx.array, mx.array]:
    """Quantize a tensor along the last dim via PlanarQuant3.

    Returns packed storage matching upstream block_planar3_0 layout.

    Args:
        x: shape ``(..., D)`` where ``D`` is even, any float dtype.

    Returns:
        packed: shape ``(..., packed_last)``, dtype ``uint8``.
            Layout: [qs[0..D/4-1], signs[0..D/8-1]] = D/4 + D/8 bytes.
        norms:  shape ``(..., 1)``, dtype ``float16``. Corrected per-block norm.
    """
    d = x.shape[-1]
    if d % 8 != 0:
        raise ValueError(f"Last dim {d} must be divisible by 8 for PlanarQuant3")
    n_pairs = d // 2
    qs_size = d // 4
    signs_size = d // 8

    x32 = x.astype(mx.float32)

    # Per-block L2 norm over the last axis — shape (..., 1)
    norm_sq = mx.sum(x32 * x32, axis=-1, keepdims=True)
    grp_norm = mx.sqrt(mx.maximum(norm_sq, mx.array(1e-20, dtype=mx.float32)))
    inv_norm = mx.where(
        grp_norm > 1e-10,
        mx.array(1.0, dtype=mx.float32) / grp_norm,
        mx.array(0.0, dtype=mx.float32),
    )

    x_norm = x32 * inv_norm  # (..., D)

    # Split into pairs — (..., n_pairs, 2)
    x_pairs = x_norm.reshape((*x.shape[:-1], n_pairs, 2))
    v0 = x_pairs[..., 0]
    v1 = x_pairs[..., 1]

    # Forward Givens: r0 = c*v0 - s*v1,  r1 = s*v0 + c*v1
    cos_tab, sin_tab = cos_sin_mx(n_pairs)
    r0 = cos_tab * v0 - sin_tab * v1
    r1 = sin_tab * v0 + cos_tab * v1

    # Fast nearest-centroid lookup via midpoints (7 comparisons vs 8 LUT)
    midpoints = midpoints_mx()  # (7,)
    r0_exp = r0[..., None]
    r1_exp = r1[..., None]
    cmp0 = (r0_exp > midpoints).astype(mx.int32)
    cmp1 = (r1_exp > midpoints).astype(mx.int32)
    idx0 = mx.sum(cmp0, axis=-1)
    idx1 = mx.sum(cmp1, axis=-1)

    # Interleave idx0, idx1 back to (..., D) order [v0,v1,v0,v1,...]
    idx_pairs = mx.stack([idx0, idx1], axis=-1)  # (..., n_pairs, 2)
    indices = idx_pairs.reshape((*x.shape[:-1], d))  # (..., D) int32

    # Corrected norm
    centroids = centroids_mx()
    recon0 = mx.take(centroids, idx0, axis=0)
    recon1 = mx.take(centroids, idx1, axis=0)
    recon_sq = mx.sum(recon0 * recon0 + recon1 * recon1, axis=-1, keepdims=True)
    recon_norm = mx.sqrt(mx.maximum(recon_sq, mx.array(1e-20, dtype=mx.float32)))
    corrected = mx.where(
        recon_norm > 1e-10,
        grp_norm / recon_norm,
        grp_norm,
    )

    # Pack 3-bit indices: lower 2 bits into qs[], upper 1 bit into signs[]
    # Same as upstream: qs[j/4] |= (idx & 0x3) << ((j%4)*2)
    #                    signs[j/8] |= ((idx >> 2) & 1) << (j%8)
    # Since MLX lacks bitwise_or.reduce and int<<array, we use
    # precomputed power-of-2 shift tables and element-wise multiply+sum.

    # Precompute shift multipliers: 2^shift for each position
    qs_shift_powers = mx.array([1, 4, 16, 64], dtype=mx.uint16)  # 2^[0,2,4,6]
    signs_shift_powers = mx.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=mx.uint16)  # 2^[0..7]

    idx_u8 = indices.astype(mx.uint8)
    lower2 = idx_u8 & mx.array(3, dtype=mx.uint8)  # (..., D) values 0-3
    upper1 = (idx_u8 >> 2).astype(mx.uint8) & mx.array(1, dtype=mx.uint8)  # (..., D) values 0-1

    # Pack lower2: 4 values per byte
    # Position j within its group of 4 → multiplier qs_shift_powers[j%4]
    j = mx.arange(d, dtype=mx.int32)
    pos_in_group = j % 4  # (D,) 0-3
    lower2_weighted = lower2.astype(mx.uint16) * qs_shift_powers[pos_in_group]  # (..., D)

    batch_shape = tuple(x.shape[:-1])
    lower2_3d = lower2_weighted.reshape((*batch_shape, qs_size, 4))
    qs = mx.sum(lower2_3d, axis=-1).astype(mx.uint8)  # (..., qs_size)

    # Pack upper1: 8 values per byte
    pos_in_byte = j % 8  # (D,) 0-7
    upper1_weighted = upper1.astype(mx.uint16) * signs_shift_powers[pos_in_byte]
    upper1_3d = upper1_weighted.reshape((*batch_shape, signs_size, 8))
    signs = mx.sum(upper1_3d, axis=-1).astype(mx.uint8)

    # Concatenate qs + signs into packed tensor
    packed = mx.concatenate([qs, signs], axis=-1)  # (..., packed_last)

    return packed, corrected.astype(mx.float16)


def _unpack_indices(packed: mx.array, d: int) -> mx.array:
    """Unpack qs+signs into per-element 3-bit indices.

    Args:
        packed: shape ``(..., qs_size + signs_size)``, dtype uint8
        d: the original last-dim width

    Returns:
        indices: shape ``(..., d)``, dtype int32, values in [0, 7]
    """
    qs_size = d // 4

    qs = packed[..., :qs_size]
    signs = packed[..., qs_size:]

    j = mx.arange(d, dtype=mx.int32)
    byte_idx = j // 4  # which qs byte
    bit_shift = (j % 4) * 2  # shift within byte

    # Gather the right byte from qs for each j
    # qs has shape (..., qs_size), j has shape (d,)
    # We need to index qs[..., byte_idx[j]] for each j
    lower2 = (qs[..., byte_idx].astype(mx.int32) >> bit_shift) & 3  # (..., d)

    # Same for signs
    sign_byte_idx = j // 8
    sign_bit_shift = j % 8
    upper1 = (signs[..., sign_byte_idx].astype(mx.int32) >> sign_bit_shift) & 1  # (..., d)

    indices = lower2 | (upper1 << 2)  # (..., d) int32 values 0-7
    return indices


def dequantize_block(packed: mx.array, norms: mx.array) -> mx.array:
    """Inverse of :func:`quantize_block`.

    Args:
        packed: shape ``(..., packed_last)``, dtype uint8 — packed qs+signs.
        norms:  shape ``(..., 1)`` float16 — one scalar per block (last dim).

    Returns:
        x_hat: shape ``(..., D)``, dtype ``float32``.
    """
    # Infer D from packed_last = D/4 + D/8 = 3D/8
    packed_last = packed.shape[-1]
    d = packed_last * 8 // 3
    if d % 8 != 0:
        raise ValueError(
            f"Inferred D={d} from packed_last={packed_last} is not divisible by 8"
        )
    n_pairs = d // 2

    indices = _unpack_indices(packed, d)  # (..., d) int32

    centroids = centroids_mx()
    q_flat = mx.take(centroids, indices, axis=0)  # (..., d) float32

    q_pairs = q_flat.reshape((*packed.shape[:-1], n_pairs, 2))
    q0 = q_pairs[..., 0]
    q1 = q_pairs[..., 1]

    # Inverse Givens: f0 = c*q0 + s*q1,  f1 = -s*q0 + c*q1
    cos_tab, sin_tab = cos_sin_mx(n_pairs)
    f0 = cos_tab * q0 + sin_tab * q1
    f1 = -sin_tab * q0 + cos_tab * q1

    f_pairs_interleaved = mx.stack([f0, f1], axis=-1)
    x_hat = f_pairs_interleaved.reshape((*packed.shape[:-1], d))

    # norms is fp16; promote to fp32 for the multiply
    return x_hat * norms.astype(mx.float32)


def roundtrip(x: mx.array) -> mx.array:
    """Convenience: quantize then dequantize. Used in tests."""
    packed, norms = quantize_block(x)
    return dequantize_block(packed, norms)
