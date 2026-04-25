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

import logging

import mlx.core as mx

from .constants import centroids_mx, cos_sin_mx, midpoints_mx

logger = logging.getLogger(__name__)

_DEQUANT_KERNEL = None
_QK_KERNEL = None
_AV_KERNEL = None
_QUANT_KERNEL = None
_WARMED_DIMS: set[int] = set()

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
# 4. Fused quantize kernel (packed layout)
# ---------------------------------------------------------------------------

_QUANT_SOURCE = """
    // Grid: (n_pairs, 1, N),  Threadgroup: (n_pairs, 1, 1)
    // Each threadgroup handles one row (token). Each thread handles one pair.
    //
    // Pipeline:
    //   Phase 1: Each thread computes pair_sq = v0^2 + v1^2. Thread 0 reduces.
    //   Phase 2: Thread 0 writes inv_norm to shared. Barrier.
    //   Phase 3: Each thread normalizes, applies Givens, does midpoint lookup.
    //   Phase 4: Each thread writes idx0, idx1 to shared. Thread 0 packs + writes norm.

    threadgroup uint idx_shared[256];     // max D=256
    threadgroup float inv_norm_shared[1];

    uint pair = thread_position_in_threadgroup.x;
    uint row = thread_position_in_grid.z;
    uint j0 = pair * 2;
    uint j1 = j0 + 1;

    float v0 = float(input_row[row * D + j0]);
    float v1 = float(input_row[row * D + j1]);

    // --- Phase 1: Compute L2 norm (reduction) ---
    // Each thread computes its pair's contribution
    float pair_sq = v0 * v0 + v1 * v1;
    // Store in idx_shared as float (reuse memory — it'll be overwritten)
    // Actually, we need shared float slots. Use a different approach:
    // Thread 0 does a serial reduction after barrier.

    // Write pair_sq to shared as uint (bit_cast) — too tricky.
    // Instead: just have thread 0 loop over all input values.
    // This is D reads which is fast (128 or 256 floats).

    if (pair == 0) {
        float total_norm_sq = 0.0f;
        for (uint j = 0; j < D; j++) {
            float v = float(input_row[row * D + j]);
            total_norm_sq += v * v;
        }
        float grp_norm = sqrt(max(total_norm_sq, 1e-20f));
        inv_norm_shared[0] = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 2: Read inv_norm, normalize ---
    float inv_norm = inv_norm_shared[0];
    float v0n = v0 * inv_norm;
    float v1n = v1 * inv_norm;

    // --- Phase 3: Forward Givens rotation on normalized vector ---
    float c = cos_tab[pair];
    float s = sin_tab[pair];
    float r0 = c * v0n - s * v1n;
    float r1 = s * v0n + c * v1n;

    // 7-comparison midpoint lookup for idx0
    uint idx0 = 0;
    if (r0 > midpoints[0]) idx0++;
    if (r0 > midpoints[1]) idx0++;
    if (r0 > midpoints[2]) idx0++;
    if (r0 > midpoints[3]) idx0++;
    if (r0 > midpoints[4]) idx0++;
    if (r0 > midpoints[5]) idx0++;
    if (r0 > midpoints[6]) idx0++;

    // 7-comparison midpoint lookup for idx1
    uint idx1 = 0;
    if (r1 > midpoints[0]) idx1++;
    if (r1 > midpoints[1]) idx1++;
    if (r1 > midpoints[2]) idx1++;
    if (r1 > midpoints[3]) idx1++;
    if (r1 > midpoints[4]) idx1++;
    if (r1 > midpoints[5]) idx1++;
    if (r1 > midpoints[6]) idx1++;

    // Write indices to shared memory
    idx_shared[j0] = idx0;
    idx_shared[j1] = idx1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 4: Thread 0 computes corrected norm, packs, writes output ---
    if (pair == 0) {
        float grp_norm = (inv_norm_shared[0] > 1e-10f) ? 1.0f / inv_norm_shared[0] : 0.0f;

        // Compute recon norm from indices
        float total_recon_sq = 0.0f;
        for (uint p2 = 0; p2 < n_pairs; p2++) {
            uint i0 = idx_shared[p2 * 2];
            uint i1 = idx_shared[p2 * 2 + 1];
            total_recon_sq += centroids[i0] * centroids[i0] + centroids[i1] * centroids[i1];
        }
        float recon_norm = sqrt(max(total_recon_sq, 1e-20f));
        float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

        // Pack indices: qs[] + signs[]
        uint out_base = row * packed_last;
        for (uint b4 = 0; b4 < qs_size; b4++) {
            uint byte_val = 0;
            uint base_j = b4 * 4;
            byte_val |= (idx_shared[base_j + 0] & 3u) << 0u;
            byte_val |= (idx_shared[base_j + 1] & 3u) << 2u;
            byte_val |= (idx_shared[base_j + 2] & 3u) << 4u;
            byte_val |= (idx_shared[base_j + 3] & 3u) << 6u;
            packed_out[out_base + b4] = byte_val;
        }
        for (uint b8 = 0; b8 < signs_size; b8++) {
            uint byte_val = 0;
            uint base_j = b8 * 8;
            byte_val |= ((idx_shared[base_j + 0] >> 2u) & 1u) << 0u;
            byte_val |= ((idx_shared[base_j + 1] >> 2u) & 1u) << 1u;
            byte_val |= ((idx_shared[base_j + 2] >> 2u) & 1u) << 2u;
            byte_val |= ((idx_shared[base_j + 3] >> 2u) & 1u) << 3u;
            byte_val |= ((idx_shared[base_j + 4] >> 2u) & 1u) << 4u;
            byte_val |= ((idx_shared[base_j + 5] >> 2u) & 1u) << 5u;
            byte_val |= ((idx_shared[base_j + 6] >> 2u) & 1u) << 6u;
            byte_val |= ((idx_shared[base_j + 7] >> 2u) & 1u) << 7u;
            packed_out[out_base + qs_size + b8] = byte_val;
        }

        norms_out[row] = (NT)(corrected_norm);
    }
"""


def _build_quant_kernel():
    global _QUANT_KERNEL
    if _QUANT_KERNEL is None:
        _QUANT_KERNEL = mx.fast.metal_kernel(
            name="planarquant3_quantize_packed",
            input_names=["input_row", "cos_tab", "sin_tab", "centroids", "midpoints"],
            output_names=["packed_out", "norms_out"],
            source=_QUANT_SOURCE,
        )
    return _QUANT_KERNEL


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
    if d % 8 != 0:
        raise ValueError(
            f"Last dim {packed_last} doesn't correspond to PlanarQuant3 D"
        )
    n_pairs = d // 2
    qs_size = d // 4

    batch_shape = tuple(packed.shape[:-1])
    n = 1
    for s in batch_shape:
        n *= int(s)

    packed_flat = packed if packed.dtype == mx.uint8 else packed.astype(mx.uint8)
    packed_flat = packed_flat.reshape((n, packed_last))
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


# ---------------------------------------------------------------------------
# Public API: quantize_fused
# ---------------------------------------------------------------------------


def quantize_fused(
    x: mx.array,
    out_dtype: mx.Dtype = mx.float16,
) -> tuple[mx.array, mx.array]:
    """Fused Metal kernel quantize for PlanarQuant3.

    Args:
        x: shape ``(..., D)``, any float dtype.
        out_dtype: norm dtype (fp16 or fp32).

    Returns:
        packed: shape ``(..., packed_last)``, dtype ``uint8``.
        norms:  shape ``(..., 1)``, dtype ``out_dtype``.
    """
    d = x.shape[-1]
    if d % 8 != 0:
        raise ValueError(f"Last dim {d} must be divisible by 8 for PlanarQuant3")
    n_pairs = d // 2
    qs_size = d // 4
    signs_size = d // 8
    packed_last = qs_size + signs_size

    batch_shape = tuple(x.shape[:-1])
    n = 1
    for s in batch_shape:
        n *= int(s)

    input_flat = x.astype(mx.float32).reshape((n, d))

    cos_tab, sin_tab = cos_sin_mx(n_pairs)
    centroids = centroids_mx()
    midpoints = midpoints_mx()

    kernel = _build_quant_kernel()

    packed_out, norms_out = kernel(
        inputs=[input_flat, cos_tab, sin_tab, centroids, midpoints],
        template=[
            ("NT", out_dtype),
            ("D", d),
            ("n_pairs", n_pairs),
            ("qs_size", qs_size),
            ("signs_size", signs_size),
            ("packed_last", packed_last),
        ],
        grid=(n_pairs, 1, n),
        threadgroup=(n_pairs, 1, 1),
        output_shapes=[(n, packed_last), (n, 1)],
        output_dtypes=[mx.uint8, out_dtype],
    )

    packed = packed_out.reshape((*batch_shape, packed_last))
    norms = norms_out.reshape((*batch_shape, 1))
    return packed, norms


def warm_planarquant_kernels(d: int = 128) -> bool:
    """Compile PlanarQuant quantize/dequant kernels before first decode."""
    if d in _WARMED_DIMS:
        return True
    try:
        x = mx.zeros((1, 1, 1, d), dtype=mx.float16)
        packed, norms = quantize_fused(x)
        dequant = dequantize_fused(packed, norms, out_dtype=mx.float16)
        mx.eval(packed, norms, dequant)
        try:
            mx.synchronize()
        except Exception:
            pass
        _WARMED_DIMS.add(d)
        return True
    except Exception as exc:
        logger.debug("PlanarQuant kernel warmup skipped: %s", exc)
        return False


# ---------------------------------------------------------------------------
# 5. Flash-attention-style fused SDPA (reads packed K/V, online softmax)
# ---------------------------------------------------------------------------

_FLASH_KERNEL = None

_FLASH_SOURCE = """
    // One threadgroup per (batch, query_head). Grid: (1, 1, BH_q).
    // Threadgroup: (n_pairs, 1, 1). Each thread handles rotation pair `pair`
    // → 2 elements of the D-dim per K/V row.
    //
    // Online-softmax flash attention for L_q=1 (decode):
    //   Running max M, running softmax sum S, running output accumulator O.
    //   For each K row k:
    //     1. Dequant K[k] for this pair.
    //     2. Compute partial dot q·k for this pair; sum across threads → full score.
    //     3. Update (M, S, correction = exp(old_M - new_M)).
    //     4. Scale O by correction, dequant V[k], accumulate O += exp_score * V_dq.
    //   Finally: out = O / S.

    threadgroup float q_shared[256];
    threadgroup float o_shared[256];
    threadgroup float partials[128];     // score partials across pairs (max n_pairs)
    threadgroup float state[4];          // [M, S, exp_score, correction]

    uint pair = thread_position_in_threadgroup.x;
    uint bh_q = thread_position_in_grid.z;
    uint b = bh_q / n_q_heads;
    uint q_head = bh_q % n_q_heads;
    uint kv_head = q_head / gqa_ratio;
    uint bh_kv = b * n_kv_heads + kv_head;

    uint j0 = pair * 2;
    uint j1 = j0 + 1;

    // Load Q into shared memory (pre-scaled by scale[0])
    float s_scale = float(scale_buf[0]);
    q_shared[j0] = float(Q[bh_q * D + j0]) * s_scale;
    q_shared[j1] = float(Q[bh_q * D + j1]) * s_scale;
    o_shared[j0] = 0.0f;
    o_shared[j1] = 0.0f;
    if (pair == 0) {
        state[0] = -1e30f;  // M
        state[1] = 0.0f;    // S
    }
    float c = cos_tab[pair];
    float s = sin_tab[pair];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < T_dim; k++) {
        // -------- Dequant K[k] for this pair --------
        uint k_base = (bh_kv * T_dim + k) * packed_last;
        uint byte0 = j0 >> 2;
        uint shift0 = (j0 & 3u) * 2u;
        uint low0 = (uint(K_packed[k_base + byte0]) >> shift0) & 3u;
        uint sb0 = j0 >> 3;
        uint ss0 = j0 & 7u;
        uint hi0 = (uint(K_packed[k_base + qs_size + sb0]) >> ss0) & 1u;
        uint k_idx0 = low0 | (hi0 << 2u);

        uint byte1 = j1 >> 2;
        uint shift1 = (j1 & 3u) * 2u;
        uint low1 = (uint(K_packed[k_base + byte1]) >> shift1) & 3u;
        uint sb1 = j1 >> 3;
        uint ss1 = j1 & 7u;
        uint hi1 = (uint(K_packed[k_base + qs_size + sb1]) >> ss1) & 1u;
        uint k_idx1 = low1 | (hi1 << 2u);

        float k_norm = K_norms[bh_kv * T_dim + k];
        float k0 = (c * centroids[k_idx0] + s * centroids[k_idx1]) * k_norm;
        float k1 = (-s * centroids[k_idx0] + c * centroids[k_idx1]) * k_norm;

        // Partial dot Q·K for this pair
        partials[pair] = q_shared[j0] * k0 + q_shared[j1] * k1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0: sum partials, update online softmax state
        if (pair == 0) {
            float full_score = 0.0f;
            for (uint p = 0; p < n_pairs; p++) full_score += partials[p];

            float new_M = max(state[0], full_score);
            float correction = exp(state[0] - new_M);
            float exp_score = exp(full_score - new_M);

            state[1] = state[1] * correction + exp_score;
            state[0] = new_M;
            state[2] = exp_score;
            state[3] = correction;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float exp_score = state[2];
        float correction = state[3];

        // Apply correction to O (scale prior accumulations by exp(old_M - new_M))
        o_shared[j0] *= correction;
        o_shared[j1] *= correction;

        // -------- Dequant V[k] for this pair --------
        uint v_base = (bh_kv * T_dim + k) * packed_last;
        uint v_low0 = (uint(V_packed[v_base + byte0]) >> shift0) & 3u;
        uint v_hi0 = (uint(V_packed[v_base + qs_size + sb0]) >> ss0) & 1u;
        uint v_idx0 = v_low0 | (v_hi0 << 2u);
        uint v_low1 = (uint(V_packed[v_base + byte1]) >> shift1) & 3u;
        uint v_hi1 = (uint(V_packed[v_base + qs_size + sb1]) >> ss1) & 1u;
        uint v_idx1 = v_low1 | (v_hi1 << 2u);

        float v_norm = V_norms[bh_kv * T_dim + k];
        float v0 = (c * centroids[v_idx0] + s * centroids[v_idx1]) * v_norm;
        float v1 = (-s * centroids[v_idx0] + c * centroids[v_idx1]) * v_norm;

        // Accumulate O += exp_score * V_dq
        o_shared[j0] += exp_score * v0;
        o_shared[j1] += exp_score * v1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    float inv_S = 1.0f / state[1];
    out[bh_q * D + j0] = (T)(o_shared[j0] * inv_S);
    out[bh_q * D + j1] = (T)(o_shared[j1] * inv_S);
"""


def _build_flash_kernel():
    global _FLASH_KERNEL
    if _FLASH_KERNEL is None:
        _FLASH_KERNEL = mx.fast.metal_kernel(
            name="planarquant3_flash_sdpa_decode",
            input_names=[
                "Q", "K_packed", "K_norms", "V_packed", "V_norms",
                "cos_tab", "sin_tab", "centroids", "scale_buf",
            ],
            output_names=["out"],
            source=_FLASH_SOURCE,
        )
    return _FLASH_KERNEL


def fused_flash_sdpa(
    queries: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    scale: float,
) -> mx.array:
    """Flash-attention-style fused decode SDPA over packed K/V.

    Reads packed K/V directly from global memory (5x less bandwidth vs FP16),
    dequantizes inline in registers, uses online softmax — never materializes
    the T×D score matrix or the dequantized K/V.

    Only supports decode (L_q=1) with no mask. For prefill or masked paths,
    fall back to dequant + MPS SDPA.

    Args:
        queries:   ``(B, H_q, 1, D)`` float16/32
        k_packed:  ``(B, H_kv, T, packed_last)`` uint8
        k_norms:   ``(B, H_kv, T, 1)`` or ``(B, H_kv, T)`` float
        v_packed:  ``(B, H_kv, T, packed_last)`` uint8
        v_norms:   ``(B, H_kv, T, 1)`` or ``(B, H_kv, T)`` float
        scale:     attention scale
    """
    if queries.shape[-2] != 1:
        raise ValueError("fused_flash_sdpa only supports L_q=1 (decode path)")

    b, h_q, _, d = queries.shape
    packed_last = k_packed.shape[-1]
    _, h_kv, t, _ = k_packed.shape
    n_pairs = d // 2
    qs_size = d // 4
    if h_q % h_kv != 0:
        raise ValueError(f"n_q_heads ({h_q}) must be divisible by n_kv_heads ({h_kv})")
    gqa_ratio = h_q // h_kv
    bh_q = b * h_q

    # Strip trailing dim from norms if present (B, H, T, 1) → (B, H, T)
    if k_norms.ndim == 4 and k_norms.shape[-1] == 1:
        k_norms = k_norms[..., 0]
    if v_norms.ndim == 4 and v_norms.shape[-1] == 1:
        v_norms = v_norms[..., 0]

    q_flat = queries.reshape((bh_q, d)).astype(mx.float16)
    k_pack_flat = k_packed.reshape((b * h_kv, t, packed_last)).astype(mx.uint8)
    k_norm_flat = k_norms.reshape((b * h_kv, t)).astype(mx.float32)
    v_pack_flat = v_packed.reshape((b * h_kv, t, packed_last)).astype(mx.uint8)
    v_norm_flat = v_norms.reshape((b * h_kv, t)).astype(mx.float32)

    cos_tab, sin_tab = cos_sin_mx(n_pairs)
    centroids = centroids_mx()

    out_dtype = queries.dtype if queries.dtype in (mx.float16, mx.float32) else mx.float16
    scale_buf = mx.array([float(scale)], dtype=mx.float32)
    kernel = _build_flash_kernel()
    out_flat = kernel(
        inputs=[
            q_flat, k_pack_flat, k_norm_flat, v_pack_flat, v_norm_flat,
            cos_tab, sin_tab, centroids, scale_buf,
        ],
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
