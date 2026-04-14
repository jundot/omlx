# SPDX-License-Identifier: Apache-2.0
"""Fused Metal kernels for PlanarQuant3 with packed block_planar3_0 layout.

Storage layout matches upstream block_planar3_0:
  norm:  fp16, 2 bytes
  qs:    D/4 bytes, 4 lower-2-bit indices per byte
  signs: D/8 bytes, 8 upper-1-bit signs per byte
  Total: 50 bytes per D=128 block

Three kernels:
  1. dequantize_fused — packed dequant for materialization
  2. fused_qk_matmul — Q·K^T with inline dequant, T-tiled
  3. fused_av_matmul — probs·V with inline dequant, T-tiled

Kernels 2+3 form the fused SDPA decode path that never materializes K/V.
"""

from __future__ import annotations

import mlx.core as mx

from .constants import centroids_mx, cos_sin_mx

_DEQUANT_KERNEL = None
_QK_KERNEL = None
_AV_KERNEL = None

# ---------------------------------------------------------------------------
# 1. Fused dequant kernel (packed layout)
# ---------------------------------------------------------------------------

_DEQUANT_SOURCE = """
    // Packed layout: [qs[0..qs_size-1], signs[0..signs_size-1]]
    // qs[j/4] holds lower 2 bits of index j at shift (j%4)*2
    // signs[j/8] holds upper 1 bit of index j at shift (j%8)
    //
    // Grid: (N * n_pairs, 1, 1), threadgroup (n_pairs, 1, 1)
    // Each thread handles one rotation pair (2 elements of D).

    uint gid = thread_position_in_grid.x;
    uint pair = gid % n_pairs;
    uint token = gid / n_pairs;
    uint j0 = pair * 2;
    uint j1 = j0 + 1;

    uint base = token * packed_last;

    // Unpack index for j0
    uint byte0 = j0 / 4;
    uint shift0 = (j0 % 4) * 2;
    uint low0 = (uint(packed[base + byte0]) >> shift0) & 3u;
    uint sign_byte0 = j0 / 8;
    uint sign_shift0 = j0 % 8;
    uint hi0 = (uint(packed[base + qs_size + sign_byte0]) >> sign_shift0) & 1u;
    uint idx0 = low0 | (hi0 << 2u);

    // Unpack index for j1
    uint byte1 = j1 / 4;
    uint shift1 = (j1 % 4) * 2;
    uint low1 = (uint(packed[base + byte1]) >> shift1) & 3u;
    uint sign_byte1 = j1 / 8;
    uint sign_shift1 = j1 % 8;
    uint hi1 = (uint(packed[base + qs_size + sign_byte1]) >> sign_shift1) & 1u;
    uint idx1 = low1 | (hi1 << 2u);

    float q0 = centroids[idx0];
    float q1 = centroids[idx1];

    float c = cos_tab[pair];
    float s = sin_tab[pair];

    // Inverse Givens
    float f0 = c * q0 + s * q1;
    float f1 = -s * q0 + c * q1;

    float norm = float(norms[token]);
    uint out_base = token * D;
    out[out_base + j0] = (T)(f0 * norm);
    out[out_base + j1] = (T)(f1 * norm);
"""


def _build_dequant_kernel():
    global _DEQUANT_KERNEL
    if _DEQUANT_KERNEL is None:
        _DEQUANT_KERNEL = mx.fast.metal_kernel(
            name="planarquant3_dequant_packed",
            input_names=["packed", "norms", "cos_tab", "sin_tab", "centroids"],
            output_names=["out"],
            source=_DEQUANT_SOURCE,
        )
    return _DEQUANT_KERNEL


# ---------------------------------------------------------------------------
# 2. Fused Q·K^T with inline dequant — T-tiled
# ---------------------------------------------------------------------------

_QK_SOURCE = """
    // Grid: (TILE_SIZE, n_tiles, BH_q)
    // Threadgroup: (TILE_SIZE, 1, 1)
    //
    // Each threadgroup processes one (bh_q, k_tile) tile.
    // TILE_SIZE threads cooperatively dequant one K row and compute
    // partial dot products. The Q vector is loaded once per threadgroup.
    //
    // K_packed:  (BH_kv, T, packed_last)   uint8
    // K_norms:   (BH_kv, T)                float32
    // Q:         (BH_q, D)                 float16

    threadgroup float q_shared[128];  // D=128 max

    uint tid = thread_position_in_threadgroup.x;
    uint k_tile = thread_position_in_grid.y;
    uint bh_q = thread_position_in_grid.z;

    uint b = bh_q / n_q_heads;
    uint q_head = bh_q % n_q_heads;
    uint kv_head = q_head / gqa_ratio;
    uint bh_kv = b * n_kv_heads + kv_head;

    uint k_start = k_tile * TILE_SIZE;
    uint k_end = min(k_start + TILE_SIZE, uint(T_dim));

    // Load Q into shared memory (n_pairs threads, each loads 2 elements)
    for (uint p = tid; p < n_pairs; p += TILE_SIZE) {
        q_shared[p * 2]     = float(Q[bh_q * D + p * 2]);
        q_shared[p * 2 + 1] = float(Q[bh_q * D + p * 2 + 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process each k position in this tile
    for (uint k_pos = k_start + tid; k_pos < k_end; k_pos += TILE_SIZE) {
        float score = 0.0f;
        float norm = K_norms[bh_kv * T_dim + k_pos];
        uint k_base = (bh_kv * T_dim + k_pos) * packed_last;

        for (uint p = 0; p < n_pairs; p++) {
            uint j0 = p * 2;
            uint j1 = j0 + 1;

            // Unpack idx0
            uint byte0 = j0 / 4;
            uint shift0 = (j0 % 4) * 2;
            uint low0 = (uint(K_packed[k_base + byte0]) >> shift0) & 3u;
            uint sb0 = j0 / 8;
            uint ss0 = j0 % 8;
            uint hi0 = (uint(K_packed[k_base + qs_size + sb0]) >> ss0) & 1u;
            uint idx0 = low0 | (hi0 << 2u);

            // Unpack idx1
            uint byte1 = j1 / 4;
            uint shift1 = (j1 % 4) * 2;
            uint low1 = (uint(K_packed[k_base + byte1]) >> shift1) & 3u;
            uint sb1 = j1 / 8;
            uint ss1 = j1 % 8;
            uint hi1 = (uint(K_packed[k_base + qs_size + sb1]) >> ss1) & 1u;
            uint idx1 = low1 | (hi1 << 2u);

            float k0 = (cos_tab[p] * centroids[idx0] + sin_tab[p] * centroids[idx1]) * norm;
            float k1 = (-sin_tab[p] * centroids[idx0] + cos_tab[p] * centroids[idx1]) * norm;

            score += q_shared[j0] * k0 + q_shared[j1] * k1;
        }

        // Store partial score — each (bh_q, k_pos) has exactly one writer
        scores[bh_q * T_dim + k_pos] = score;
    }
"""


def _build_qk_kernel():
    global _QK_KERNEL
    if _QK_KERNEL is None:
        _QK_KERNEL = mx.fast.metal_kernel(
            name="planarquant3_qk_tiled",
            input_names=["Q", "K_packed", "K_norms", "cos_tab", "sin_tab", "centroids"],
            output_names=["scores"],
            source=_QK_SOURCE,
        )
    return _QK_KERNEL


# ---------------------------------------------------------------------------
# 3. Fused probs·V with inline dequant — T-tiled
# ---------------------------------------------------------------------------

_AV_SOURCE = """
    // Grid: (n_pairs, 1, BH_q)
    // Threadgroup: (n_pairs, 1, 1)
    //
    // Each threadgroup handles one (bh_q, pair).
    // The pair thread loops over T in tiles, accumulating:
    //   out_even = sum_k probs[k] * (c*q0 + s*q1) * norm
    //   out_odd  = sum_k probs[k] * (-s*q0 + c*q1) * norm
    //
    // After the T-loop, write 2 output elements.

    uint pair = thread_position_in_threadgroup.x;
    uint bh_q = thread_position_in_grid.z;

    uint b = bh_q / n_q_heads;
    uint q_head = bh_q % n_q_heads;
    uint kv_head = q_head / gqa_ratio;
    uint bh_kv = b * n_kv_heads + kv_head;

    float cs = cos_tab[pair];
    float sn = sin_tab[pair];
    uint j0 = pair * 2;
    uint j1 = j0 + 1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint k_pos = 0; k_pos < T_dim; k_pos++) {
        float p = probs[bh_q * T_dim + k_pos];
        float norm = V_norms[bh_kv * T_dim + k_pos];
        uint v_base = (bh_kv * T_dim + k_pos) * packed_last;

        // Unpack idx0
        uint byte0 = j0 / 4;
        uint shift0 = (j0 % 4) * 2;
        uint low0 = (uint(V_packed[v_base + byte0]) >> shift0) & 3u;
        uint sb0 = j0 / 8;
        uint ss0 = j0 % 8;
        uint hi0 = (uint(V_packed[v_base + qs_size + sb0]) >> ss0) & 1u;
        uint idx0 = low0 | (hi0 << 2u);

        // Unpack idx1
        uint byte1 = j1 / 4;
        uint shift1 = (j1 % 4) * 2;
        uint low1 = (uint(V_packed[v_base + byte1]) >> shift1) & 3u;
        uint sb1 = j1 / 8;
        uint ss1 = j1 % 8;
        uint hi1 = (uint(V_packed[v_base + qs_size + sb1]) >> ss1) & 1u;
        uint idx1 = low1 | (hi1 << 2u);

        float q0 = centroids[idx0];
        float q1 = centroids[idx1];

        acc0 += p * (cs * q0 + sn * q1) * norm;
        acc1 += p * (-sn * q0 + cs * q1) * norm;
    }

    out[bh_q * D + j0] = (T)acc0;
    out[bh_q * D + j1] = (T)acc1;
"""


def _build_av_kernel():
    global _AV_KERNEL
    if _AV_KERNEL is None:
        _AV_KERNEL = mx.fast.metal_kernel(
            name="planarquant3_av_tiled",
            input_names=["probs", "V_packed", "V_norms", "cos_tab", "sin_tab", "centroids"],
            output_names=["out"],
            source=_AV_SOURCE,
        )
    return _AV_KERNEL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dequantize_fused(
    packed: mx.array,
    norms: mx.array,
    out_dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Fused Metal kernel dequant for packed PlanarQuant3.

    Args:
        packed: shape ``(..., packed_last)``, dtype ``uint8``.
        norms:  shape ``(..., 1)``, any float dtype.
        out_dtype: desired output dtype.

    Returns:
        Tensor of shape ``(..., D)``, dtype ``out_dtype``.
    """
    packed_last = packed.shape[-1]
    d = packed_last * 8 // 3
    if d % 2 != 0:
        raise ValueError(f"Last dim {packed_last} doesn't correspond to even D")
    n_pairs = d // 2
    qs_size = d // 4

    batch_shape = tuple(packed.shape[:-1])
    n = 1
    for s in batch_shape:
        n *= int(s)

    packed_flat = packed.astype(mx.uint8).reshape((n, packed_last))
    norms_flat = norms.astype(mx.float32).reshape((n, 1))

    cos_tab, sin_tab = cos_sin_mx(n_pairs)
    centroids = centroids_mx()

    kernel = _build_dequant_kernel()
    tg_size = n_pairs
    grid_x = n * n_pairs

    result = kernel(
        inputs=[packed_flat, norms_flat, cos_tab, sin_tab, centroids],
        template=[
            ("T", out_dtype),
            ("D", d),
            ("n_pairs", n_pairs),
            ("qs_size", qs_size),
            ("packed_last", packed_last),
        ],
        grid=(grid_x, 1, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[(n, d)],
        output_dtypes=[out_dtype],
    )[0]

    return result.reshape((*batch_shape, d))


def fused_quantized_sdpa(
    queries: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    scale: float,
) -> mx.array:
    """Fused decode-path attention with inline dequant of packed K and V.

    Args:
        queries:    ``(B, H_q, 1, D)`` float16/32
        k_packed:  ``(B, H_kv, T, packed_last)`` uint8
        k_norms:   ``(B, H_kv, T)`` float
        v_packed:  ``(B, H_kv, T, packed_last)`` uint8
        v_norms:   ``(B, H_kv, T)`` float
        scale:      attention scale

    Returns:
        ``(B, H_q, 1, D)``
    """
    if queries.shape[-2] != 1:
        raise ValueError("fused_quantized_sdpa only supports L_q=1 (decode path)")

    b, h_q, _, d = queries.shape
    packed_last = k_packed.shape[-1]
    _, h_kv, t, _ = k_packed.shape
    n_pairs = d // 2
    qs_size = d // 4
    if h_q % h_kv != 0:
        raise ValueError(f"n_q_heads ({h_q}) must be divisible by n_kv_heads ({h_kv})")
    gqa_ratio = h_q // h_kv
    bh_q = b * h_q

    q_flat = (queries * float(scale)).reshape((bh_q, d))
    q_half = q_flat.astype(mx.float16)

    k_pack_flat = k_packed.reshape((b * h_kv, t, packed_last)).astype(mx.uint8)
    k_norm_flat = k_norms.reshape((b * h_kv, t)).astype(mx.float32)
    v_pack_flat = v_packed.reshape((b * h_kv, t, packed_last)).astype(mx.uint8)
    v_norm_flat = v_norms.reshape((b * h_kv, t)).astype(mx.float32)

    cos_tab, sin_tab = cos_sin_mx(n_pairs)
    centroids = centroids_mx()

    # Kernel A: scores = Q·K^T
    tile_size = min(64, t)
    n_tiles = (t + tile_size - 1) // tile_size

    qk_kernel = _build_qk_kernel()
    scores = qk_kernel(
        inputs=[q_half, k_pack_flat, k_norm_flat, cos_tab, sin_tab, centroids],
        template=[
            ("D", d),
            ("n_pairs", n_pairs),
            ("n_q_heads", h_q),
            ("n_kv_heads", h_kv),
            ("gqa_ratio", gqa_ratio),
            ("T_dim", t),
            ("TILE_SIZE", tile_size),
            ("qs_size", qs_size),
            ("packed_last", packed_last),
        ],
        grid=(tile_size, n_tiles, bh_q),
        threadgroup=(tile_size, 1, 1),
        output_shapes=[(bh_q, t)],
        output_dtypes=[mx.float32],
    )[0]

    # Softmax
    probs = mx.softmax(scores, axis=-1, precise=True)

    # Kernel B: out = probs·V
    out_dtype = queries.dtype if queries.dtype in (mx.float16, mx.float32) else mx.float16
    av_kernel = _build_av_kernel()
    out_flat = av_kernel(
        inputs=[probs.astype(mx.float32), v_pack_flat, v_norm_flat, cos_tab, sin_tab, centroids],
        template=[
            ("T", out_dtype),
            ("D", d),
            ("n_pairs", n_pairs),
            ("n_q_heads", h_q),
            ("n_kv_heads", h_kv),
            ("gqa_ratio", gqa_ratio),
            ("T_dim", t),
            ("qs_size", qs_size),
            ("packed_last", packed_last),
        ],
        grid=(n_pairs, 1, bh_q),
        threadgroup=(n_pairs, 1, 1),
        output_shapes=[(bh_q, d)],
        output_dtypes=[out_dtype],
    )[0]

    return out_flat.reshape((b, h_q, 1, d)).astype(queries.dtype)
