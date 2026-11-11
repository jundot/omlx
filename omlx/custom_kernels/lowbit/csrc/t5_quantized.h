// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// Bonsai t5 (base-3 ternary) decode and prefill kernel bodies.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;

// ---------------------------------------------------------------------------
// Realization 2.3 (I-K): Granlund-Montgomery magic divmod-3, exact for v in [0,255].
// floor(v/3) = (v*171) >> 9  - no integer divide instruction, pure multiply + shift.
// Then v mod 3 = v - 3q = v - (q<<1) - q.
METAL_FUNC uint t5_div3(uint v)  { return (v * 171u) >> 9u; }

// Parallel trit extraction helpers (I-L): all operate on the ORIGINAL byte v,
// so ALL four calls can be issued simultaneously with no serial data dependency.
// Exact for v in [0, 242] (max valid t5 byte = 3^5 - 1 = 242).
METAL_FUNC uint t5_div9(uint v)  { return (v * 228u) >> 11u; }  // floor(v/9)
METAL_FUNC uint t5_div27(uint v) { return (v * 76u)  >> 11u; }  // floor(v/27)
// floor(v/81) in {0,1,2}: two independent comparisons, no multiply needed.
// Precondition: v in [0, 242] (max valid t5 byte = 3^5 - 1 = 242).
// v=243 cannot appear in a validly-encoded t5 stream, so v/81 <= 2 always.
METAL_FUNC uint t5_div81(uint v) { return uint(v >= 81u) + uint(v >= 162u); }

// T5_TO_B4: radix-conversion LUT - maps each t5 byte (0..242) to its 5 trits
// repacked as 2-bit fields in a uint.  Field k occupies bits [2k+1 : 2k]:
//   bits[1:0] = t0,  bits[3:2] = t1,  bits[5:4] = t2,
//   bits[7:6] = t3,  bits[9:8] = t4,  t_k in {0,1,2}.
//
// This enables the same pre-scaling trick as 2-bit affine (I-L):
//   x_pre[k] = x[k] x 4^{-k}
//   x_pre[k] x (T5_TO_B4[v] & (3 << 2k))  =  x[k] x t_k   (no division)
// The -1 offset is absorbed via x_sum: result = s x (sum x[k]t_k - sum x[k]).
// 256 entries x 4 bytes = 1 KB; permanently L1-resident during GEMV.
constant constexpr uint T5_TO_B4[256] = {
    0x000, 0x001, 0x002, 0x004, 0x005, 0x006, 0x008, 0x009,
    0x00A, 0x010, 0x011, 0x012, 0x014, 0x015, 0x016, 0x018,
    0x019, 0x01A, 0x020, 0x021, 0x022, 0x024, 0x025, 0x026,
    0x028, 0x029, 0x02A, 0x040, 0x041, 0x042, 0x044, 0x045,
    0x046, 0x048, 0x049, 0x04A, 0x050, 0x051, 0x052, 0x054,
    0x055, 0x056, 0x058, 0x059, 0x05A, 0x060, 0x061, 0x062,
    0x064, 0x065, 0x066, 0x068, 0x069, 0x06A, 0x080, 0x081,
    0x082, 0x084, 0x085, 0x086, 0x088, 0x089, 0x08A, 0x090,
    0x091, 0x092, 0x094, 0x095, 0x096, 0x098, 0x099, 0x09A,
    0x0A0, 0x0A1, 0x0A2, 0x0A4, 0x0A5, 0x0A6, 0x0A8, 0x0A9,
    0x0AA, 0x100, 0x101, 0x102, 0x104, 0x105, 0x106, 0x108,
    0x109, 0x10A, 0x110, 0x111, 0x112, 0x114, 0x115, 0x116,
    0x118, 0x119, 0x11A, 0x120, 0x121, 0x122, 0x124, 0x125,
    0x126, 0x128, 0x129, 0x12A, 0x140, 0x141, 0x142, 0x144,
    0x145, 0x146, 0x148, 0x149, 0x14A, 0x150, 0x151, 0x152,
    0x154, 0x155, 0x156, 0x158, 0x159, 0x15A, 0x160, 0x161,
    0x162, 0x164, 0x165, 0x166, 0x168, 0x169, 0x16A, 0x180,
    0x181, 0x182, 0x184, 0x185, 0x186, 0x188, 0x189, 0x18A,
    0x190, 0x191, 0x192, 0x194, 0x195, 0x196, 0x198, 0x199,
    0x19A, 0x1A0, 0x1A1, 0x1A2, 0x1A4, 0x1A5, 0x1A6, 0x1A8,
    0x1A9, 0x1AA, 0x200, 0x201, 0x202, 0x204, 0x205, 0x206,
    0x208, 0x209, 0x20A, 0x210, 0x211, 0x212, 0x214, 0x215,
    0x216, 0x218, 0x219, 0x21A, 0x220, 0x221, 0x222, 0x224,
    0x225, 0x226, 0x228, 0x229, 0x22A, 0x240, 0x241, 0x242,
    0x244, 0x245, 0x246, 0x248, 0x249, 0x24A, 0x250, 0x251,
    0x252, 0x254, 0x255, 0x256, 0x258, 0x259, 0x25A, 0x260,
    0x261, 0x262, 0x264, 0x265, 0x266, 0x268, 0x269, 0x26A,
    0x280, 0x281, 0x282, 0x284, 0x285, 0x286, 0x288, 0x289,
    0x28A, 0x290, 0x291, 0x292, 0x294, 0x295, 0x296, 0x298,
    0x299, 0x29A, 0x2A0, 0x2A1, 0x2A2, 0x2A4, 0x2A5, 0x2A6,
    0x2A8, 0x2A9, 0x2AA, 0x000, 0x000, 0x000, 0x000, 0x000,
    0x000, 0x000, 0x000, 0x000, 0x000, 0x000, 0x000, 0x000,
};
// ---------------------------------------------------------------------------
// qmv_fast_t5_impl - optimized (O1-O5)
//
// [O1] packed_uchar4 weight loads (alignment 1, 7 loads vs 26 per row-group)
// [O2] packed_half4 activation loads (2 loads vs 5 per 5-trit byte)
// [O3] 20-trit chunk structure (4 bytes x 5 trits, fully unrolled)
// [O4] USE_SIGMA: skip x_sum + pre-scale muls (sigma precomputed elsewhere)
// [O5] row clamping instead of duplicated boundary loop
// ---------------------------------------------------------------------------
template <typename T, int group_size, bool USE_SIGMA = false>
METAL_FUNC void qmv_fast_t5_impl(
    const device uint8_t* w,
    const device T* scales,
    const device T* x,            // pre-scaled by 4^{-(j%5)} when USE_SIGMA
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const device float* sigma,    // [n_groups] group sums; used iff USE_SIGMA
    threadgroup const uint* lut,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int bytes_per_group = (group_size + 4) / 5;
  constexpr int full_bytes = group_size / 5;         // 25 for gs=128
  constexpr int chunk_bytes = full_bytes & ~3;       // 24: packed_uchar4 part
  constexpr int tail_full   = full_bytes - chunk_bytes; // 1 full byte in tail
  constexpr int rem_trits   = group_size % 5;        // 3 for gs=128
  constexpr int num_simdgroups = 4;
  constexpr int results_per_simdgroup = 4;

  typedef float U;
  constexpr U PS0 = 1.0f, PS1 = 0.25f, PS2 = 0.0625f,
              PS3 = 0.015625f, PS4 = 0.00390625f;

  thread U result[results_per_simdgroup] = {0};
  const int n_groups = in_vec_size / group_size;
  const int out_row  = tid.y * (num_simdgroups * results_per_simdgroup) +
                       simd_gid * results_per_simdgroup;
  const int ng_bpg   = n_groups * bytes_per_group;

  int rows[results_per_simdgroup];
  bool row_ok[results_per_simdgroup];
  for (int r = 0; r < results_per_simdgroup; r++) {
    row_ok[r] = (out_row + r) < out_vec_size;
    rows[r]   = row_ok[r] ? (out_row + r) : 0;
  }

  const device T* x_batch = x + tid.x * in_vec_size;

  for (int g = simd_lid; g < n_groups; g += SIMD_SIZE) {
    const device T* xg = x_batch + g * group_size;
    const device uint8_t* wg0 = w + rows[0]*ng_bpg + g*bytes_per_group;
    const device uint8_t* wg1 = w + rows[1]*ng_bpg + g*bytes_per_group;
    const device uint8_t* wg2 = w + rows[2]*ng_bpg + g*bytes_per_group;
    const device uint8_t* wg3 = w + rows[3]*ng_bpg + g*bytes_per_group;
    const U s0 = U(scales[rows[0]*n_groups + g]);
    const U s1 = U(scales[rows[1]*n_groups + g]);
    const U s2 = U(scales[rows[2]*n_groups + g]);
    const U s3 = U(scales[rows[3]*n_groups + g]);

    U a0 = 0, a1 = 0, a2 = 0, a3 = 0, x_sum = 0;

#pragma clang loop unroll(full)
    for (int c = 0; c < chunk_bytes / 4; c++) {
      const packed_uchar4 q0 = *((const device packed_uchar4*)(wg0 + c*4));
      const packed_uchar4 q1 = *((const device packed_uchar4*)(wg1 + c*4));
      const packed_uchar4 q2 = *((const device packed_uchar4*)(wg2 + c*4));
      const packed_uchar4 q3 = *((const device packed_uchar4*)(wg3 + c*4));
#pragma clang loop unroll(full)
      for (int bb = 0; bb < 4; bb++) {
        const uint p0 = lut[q0[bb]], p1 = lut[q1[bb]];
        const uint p2 = lut[q2[bb]], p3 = lut[q3[bb]];
        const int base = (c*4 + bb) * 5;
        const packed_half4 xv4 = *((const device packed_half4*)(xg + base));
        const U xv0 = U(xv4[0]), xv1 = U(xv4[1]), xv2 = U(xv4[2]), xv3 = U(xv4[3]);
        const U xv4s = U(xg[base + 4]);
        U xp1, xp2, xp3, xp4;
        if constexpr (USE_SIGMA) { xp1=xv1; xp2=xv2; xp3=xv3; xp4=xv4s; }
        else { x_sum += xv0+xv1+xv2+xv3+xv4s; xp1=xv1*PS1; xp2=xv2*PS2; xp3=xv3*PS3; xp4=xv4s*PS4; }
        a0+=xv0*U(p0&0x003u)+xp1*U(p0&0x00Cu)+xp2*U(p0&0x030u)+xp3*U(p0&0x0C0u)+xp4*U(p0&0x300u);
        a1+=xv0*U(p1&0x003u)+xp1*U(p1&0x00Cu)+xp2*U(p1&0x030u)+xp3*U(p1&0x0C0u)+xp4*U(p1&0x300u);
        a2+=xv0*U(p2&0x003u)+xp1*U(p2&0x00Cu)+xp2*U(p2&0x030u)+xp3*U(p2&0x0C0u)+xp4*U(p2&0x300u);
        a3+=xv0*U(p3&0x003u)+xp1*U(p3&0x00Cu)+xp2*U(p3&0x030u)+xp3*U(p3&0x0C0u)+xp4*U(p3&0x300u);
      }
    }
    if constexpr (tail_full > 0 || rem_trits > 0) {
      ushort t0, t1, t2, t3;
      if constexpr (tail_full > 0) {
        t0 = *((const device ushort*)(wg0 + chunk_bytes));
        t1 = *((const device ushort*)(wg1 + chunk_bytes));
        t2 = *((const device ushort*)(wg2 + chunk_bytes));
        t3 = *((const device ushort*)(wg3 + chunk_bytes));
      } else {
        t0 = ushort(wg0[chunk_bytes]) << 8; t1 = ushort(wg1[chunk_bytes]) << 8;
        t2 = ushort(wg2[chunk_bytes]) << 8; t3 = ushort(wg3[chunk_bytes]) << 8;
      }
      if constexpr (tail_full > 0) {
        const uint p0=lut[t0&0xFF], p1=lut[t1&0xFF], p2=lut[t2&0xFF], p3=lut[t3&0xFF];
        const int base = chunk_bytes * 5;
        const packed_half4 xv4 = *((const device packed_half4*)(xg + base));
        const U xv0=U(xv4[0]), xv1=U(xv4[1]), xv2=U(xv4[2]), xv3=U(xv4[3]), xv4s=U(xg[base+4]);
        U xp1,xp2,xp3,xp4;
        if constexpr (USE_SIGMA) { xp1=xv1;xp2=xv2;xp3=xv3;xp4=xv4s; }
        else { x_sum+=xv0+xv1+xv2+xv3+xv4s; xp1=xv1*PS1;xp2=xv2*PS2;xp3=xv3*PS3;xp4=xv4s*PS4; }
        a0+=xv0*U(p0&0x003u)+xp1*U(p0&0x00Cu)+xp2*U(p0&0x030u)+xp3*U(p0&0x0C0u)+xp4*U(p0&0x300u);
        a1+=xv0*U(p1&0x003u)+xp1*U(p1&0x00Cu)+xp2*U(p1&0x030u)+xp3*U(p1&0x0C0u)+xp4*U(p1&0x300u);
        a2+=xv0*U(p2&0x003u)+xp1*U(p2&0x00Cu)+xp2*U(p2&0x030u)+xp3*U(p2&0x0C0u)+xp4*U(p2&0x300u);
        a3+=xv0*U(p3&0x003u)+xp1*U(p3&0x00Cu)+xp2*U(p3&0x030u)+xp3*U(p3&0x0C0u)+xp4*U(p3&0x300u);
      }
      if constexpr (rem_trits > 0) {
        const uint p0=lut[t0>>8], p1=lut[t1>>8], p2=lut[t2>>8], p3=lut[t3>>8];
        const int base = full_bytes * 5;
#pragma clang loop unroll(full)
        for (int k = 0; k < rem_trits; k++) {
          const U xv=U(xg[base+k]); U xp;
          if constexpr (USE_SIGMA) { xp=xv; }
          else { x_sum+=xv; xp=xv*(k==0?PS0:k==1?PS1:k==2?PS2:k==3?PS3:PS4); }
          const uint m=0x3u<<(2*k);
          a0+=xp*U(p0&m); a1+=xp*U(p1&m); a2+=xp*U(p2&m); a3+=xp*U(p3&m);
        }
      }
    }
    const U xs = USE_SIGMA ? U(sigma[g]) : x_sum;
    result[0] += s0*(a0-xs); result[1] += s1*(a1-xs);
    result[2] += s2*(a2-xs); result[3] += s3*(a3-xs);
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && row_ok[row])
      y[tid.x * out_vec_size + out_row + row] = T(result[row]);
  }
}


// qmv_wide_t5: small-batch decode (M = vecs_per_tg = 2..5).
//
// Amortises the weight stream across all M vectors (Identity I-C):
// decode each t5 byte once, multiply with all M activations.
// k_lanes threads stride over groups, results_per_simdgroup=SIMD_SIZE/k_lanes.
template <typename T, int group_size, int vecs_per_tg, int k_lanes>
METAL_FUNC void qmv_wide_t5_impl(
    const device uint8_t* w,
    const device T* scales,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& M,
    threadgroup const uint* lut,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int bytes_per_group = (group_size + 4) / 5;
  constexpr int full_bytes = group_size / 5;
  constexpr int rem_trits  = group_size % 5;
  constexpr int num_simdgroups = 4;
  constexpr int results_per_simdgroup = SIMD_SIZE / k_lanes;

  typedef float U;
  const short k_lane = simd_lid % k_lanes;
  const short sg_row = simd_lid / k_lanes;

  const int out_row = tid.y * (results_per_simdgroup * num_simdgroups) +
                      results_per_simdgroup * simd_gid + sg_row;
  const int vec0 = tid.x * vecs_per_tg;
  const int n_groups = in_vec_size / group_size;

  const int row = min(out_row, out_vec_size - 1);
  const device uint8_t* wrow = w + row * (n_groups * bytes_per_group);
  const device T* srow = scales + row * n_groups;

  const device T* xv[vecs_per_tg];
  for (int v = 0; v < vecs_per_tg; v++) {
    xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
  }

  U result[vecs_per_tg] = {0};

  for (int g = k_lane; g < n_groups; g += k_lanes) {
    const device uint8_t* wg = wrow + g * bytes_per_group;
    const U s = U(srow[g]);
    const int k0 = g * group_size;

    // Full bytes: T5_TO_B4 LUT replaces serial divmod chain; all 5 dq values
    // are independent (no v=q reassignment), hiding decode latency.
    for (int b = 0; b < full_bytes; b++) {
      const uint p = T5_TO_B4[wg[b]];
      const int base = k0 + b * 5;
      const U dq0 = s * (U(p & 0x003u)        - 1.0f);
      const U dq1 = s * (U((p >>  2u) & 0x3u) - 1.0f);
      const U dq2 = s * (U((p >>  4u) & 0x3u) - 1.0f);
      const U dq3 = s * (U((p >>  6u) & 0x3u) - 1.0f);
      const U dq4 = s * (U((p >>  8u) & 0x3u) - 1.0f);
#pragma unroll
      for (int vi = 0; vi < vecs_per_tg; vi++) {
        result[vi] += U(xv[vi][base + 0]) * dq0
                    + U(xv[vi][base + 1]) * dq1
                    + U(xv[vi][base + 2]) * dq2
                    + U(xv[vi][base + 3]) * dq3
                    + U(xv[vi][base + 4]) * dq4;
      }
    }
    // Last partial byte via T5_TO_B4 LUT.
    if constexpr (rem_trits > 0) {
      const uint p = T5_TO_B4[wg[full_bytes]];
      const int base = k0 + full_bytes * 5;
      U dq0 = U(0), dq1 = U(0), dq2 = U(0), dq3 = U(0);
      if constexpr (rem_trits >= 1) { dq0 = s * (U(p & 0x003u) - 1.0f); }
      if constexpr (rem_trits >= 2) { dq1 = s * (U((p >> 2u) & 0x3u) - 1.0f); }
      if constexpr (rem_trits >= 3) { dq2 = s * (U((p >> 4u) & 0x3u) - 1.0f); }
      if constexpr (rem_trits >= 4) { dq3 = s * (U((p >> 6u) & 0x3u) - 1.0f); }
#pragma unroll
      for (int vi = 0; vi < vecs_per_tg; vi++) {
        if constexpr (rem_trits >= 1) result[vi] += U(xv[vi][base + 0]) * dq0;
        if constexpr (rem_trits >= 2) result[vi] += U(xv[vi][base + 1]) * dq1;
        if constexpr (rem_trits >= 3) result[vi] += U(xv[vi][base + 2]) * dq2;
        if constexpr (rem_trits >= 4) result[vi] += U(xv[vi][base + 3]) * dq3;
      }
    }
  }

  // k_lane shuffle reduction (same ladder as qmv_wide_impl).
  for (int v = 0; v < vecs_per_tg; v++) {
    if constexpr (k_lanes >= 32) result[v] += simd_shuffle_down(result[v], 16);
    if constexpr (k_lanes >= 16) result[v] += simd_shuffle_down(result[v], 8);
    if constexpr (k_lanes >= 8)  result[v] += simd_shuffle_down(result[v], 4);
    if constexpr (k_lanes >= 4)  result[v] += simd_shuffle_down(result[v], 2);
    if constexpr (k_lanes >= 2)  result[v] += simd_shuffle_down(result[v], 1);
  }

  if (k_lane == 0 && out_row < out_vec_size) {
    for (int v = 0; v < vecs_per_tg; v++) {
      if (vec0 + v < M) {
        y[(vec0 + v) * out_vec_size + out_row] = T(result[v]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// qmm_t5_impl: simdgroup MMA GEMM for t5 ternary weights (Identity I-M).
//
// Computes out[M, N] = x[M, K] @ t5_decode(w[N, K]).T  (float32 accumulate).
//
// Architecture (adapted from MetalTile mt_qmm_mma_int2):
//   BM=BN=32, BK=group_size (one complete t5 group per K-block iteration)
//   128 threads = 4 SG x 32 lanes; 2x2 SG tile layout (sm=sg/2, sn=sg%2)
//   xs[BM][BK+4] and ws[BN][BK+4] threadgroup tiles
//   BK/8 simdgroup_multiply_accumulate calls per K-block
//
// Thread (lane_in_tg = sg_id*32 + lane):
//   X loading :  row = lane_in_tg/4, k_sub = lane_in_tg%4 -> loads BK/4 x-elems
//   W dequant :  row = lane_in_tg/4, sub  = lane_in_tg%4 -> decodes BK/4 trits
//                (sub 0..3 each decode a contiguous BK/4 K-positions from t5 bytes)
//
// Grid: (ceil(N/32), ceil(M/32), B)  Threadgroup: (32, 4, 1)
// ---------------------------------------------------------------------------
template <typename T, int group_size>
METAL_FUNC void qmm_t5_impl(
    const device uint8_t* w,    // (N, n_groups * bpg) t5 bytes
    const device T* scales,     // (N, n_groups)
    const device T* x,          // (M, K)
    device T* out,              // (M, N)
    const constant int& M_c,
    const constant int& N_c,
    const constant int& K_c,
    threadgroup T* xs,          // BM*(group_size+4) - declared by caller
    threadgroup T* ws,          // BN*(group_size+4) - declared by caller
    uint2 tgid,                  // (n_tile, m_tile)
    uint  lane,                  // thread_index_in_simdgroup  (0..31)
    uint  sg_id)                 // simdgroup_index_in_threadgroup (0..3)
{
  constexpr int bpg    = (group_size + 4) / 5;   // bytes per group
  constexpr int xs_ld  = group_size + 4;          // padded stride for xs
  constexpr int ws_ld  = group_size + 4;          // padded stride for ws
  constexpr int BM = 32, BN = 32;
  (void)BM; (void)BN;

  // Float32 accumulators: 2x2 SG tile -> 4 frags per SG (each covers 8x8 out elements)
  simdgroup_matrix<float, 8, 8> c00, c01, c10, c11;
  c00.thread_elements()[0] = 0.f;  c00.thread_elements()[1] = 0.f;
  c01.thread_elements()[0] = 0.f;  c01.thread_elements()[1] = 0.f;
  c10.thread_elements()[0] = 0.f;  c10.thread_elements()[1] = 0.f;
  c11.thread_elements()[0] = 0.f;  c11.thread_elements()[1] = 0.f;

  const uint sm = sg_id >> 1u;       // SG M-index (0..1)
  const uint sn = sg_id & 1u;        // SG N-index (0..1)
  const uint lane_in_tg = sg_id * 32u + lane;

  // Fragment indices within an 8x8 sub-tile (same derivation as mt_qmm_mma_int2)
  const uint qid = lane >> 2u;
  const uint fm  = (qid & 4u) + ((lane >> 1u) & 3u);    // frag M offset in [0,7]
  const uint fn0 = ((qid & 2u) << 1u) + ((lane & 1u) << 1u);  // frag N offset
  const uint fn1 = fn0 + 1u;

  const uint m_base = tgid.y * (uint)BM;
  const uint n_base = tgid.x * (uint)BN;

  const int M = M_c, N = N_c, K = K_c;
  const int n_groups = K / group_size;

  // Per-block tile-loading assignment
  const uint tl_row     = lane_in_tg >> 2u;    // tile row 0..31 (both X and W)
  const uint tl_sub     = lane_in_tg & 3u;     // K-quarter 0..3
  const uint k_sub_size = (uint)(group_size >> 2);  // BK/4 per sub (32 for gs=128)

  // Precomputed threadgroup-memory read indices for MMA (invariant per K-block)
  const uint xs_m0 = (sm * 16u + fm) * xs_ld;
  const uint xs_m1 = (sm * 16u + 8u + fm) * xs_ld;
  const uint ws_n00 = (sn * 16u       + fn0) * ws_ld;
  const uint ws_n01 = (sn * 16u       + fn1) * ws_ld;
  const uint ws_n10 = (sn * 16u + 8u  + fn0) * ws_ld;
  const uint ws_n11 = (sn * 16u + 8u  + fn1) * ws_ld;

  for (int g = 0; g < n_groups; g++) {
    // ---- Load X tile into xs ----
    {
      const uint m_row = m_base + tl_row;
      const bool m_ok  = m_row < (uint)M;
      const uint k_off = (uint)(g * group_size) + tl_sub * k_sub_size;
      const device T* xp = x + min(m_row, (uint)(M - 1)) * K + k_off;
      const uint xs_dst  = tl_row * xs_ld + tl_sub * k_sub_size;
      for (uint i = 0; i < k_sub_size; i++) {
        xs[xs_dst + i] = m_ok ? xp[i] : T(0);
      }
    }

    // ---- Dequant W tile into ws ----
    {
      const uint n_row  = n_base + tl_row;
      const bool n_ok   = n_row < (uint)N;
      const float sc    = n_ok ? float(scales[n_row * n_groups + g]) : 0.f;
      const uint w_base = n_row * (uint)(n_groups * bpg) + (uint)(g * bpg);
      const uint ks     = tl_sub * k_sub_size;   // K-start within group (0,32,64,96)
      const uint ke     = ks + k_sub_size;        // K-end within group
      const uint b_lo   = ks / 5u;               // first byte index (inclusive)
      const uint b_hi   = (ke - 1u) / 5u;        // last byte index  (inclusive)

      for (uint b = b_lo; b <= b_hi; b++) {
        // Read t5 byte; pad with 1 (neutral trit=1->dq=0) if out of bounds.
        // T5_TO_B4 LUT replaces 4 divmod chains: trit k = (p >> 2k) & 3.
        const uint bval = (n_ok && b < (uint)bpg) ? (uint)w[w_base + b] : 1u;
        const uint p  = T5_TO_B4[bval];
        const uint bk = b * 5u;  // K-index of trit 0 of this byte
        // Write only the trits whose K-position falls in [ks, ke)
        if (bk     >= ks && bk     < ke) ws[tl_row * ws_ld + bk    ] = T(sc * (float( p        & 0x3u) - 1.f));
        if (bk+1u  >= ks && bk+1u  < ke) ws[tl_row * ws_ld + bk+1u ] = T(sc * (float((p >> 2u) & 0x3u) - 1.f));
        if (bk+2u  >= ks && bk+2u  < ke) ws[tl_row * ws_ld + bk+2u ] = T(sc * (float((p >> 4u) & 0x3u) - 1.f));
        if (bk+3u  >= ks && bk+3u  < ke) ws[tl_row * ws_ld + bk+3u ] = T(sc * (float((p >> 6u) & 0x3u) - 1.f));
        if (bk+4u  >= ks && bk+4u  < ke) ws[tl_row * ws_ld + bk+4u ] = T(sc * (float((p >> 8u) & 0x3u) - 1.f));
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- BK/8 simdgroup MMA steps ----
    {
      simdgroup_matrix<T, 8, 8> a0, a1, b0, b1;
      for (uint ki = 0; ki < (uint)(group_size >> 3); ki++) {
        const uint ko = ki * 8u;
        a0.thread_elements()[0] = xs[xs_m0 + fn0 + ko];
        a0.thread_elements()[1] = xs[xs_m0 + fn1 + ko];
        a1.thread_elements()[0] = xs[xs_m1 + fn0 + ko];
        a1.thread_elements()[1] = xs[xs_m1 + fn1 + ko];
        b0.thread_elements()[0] = ws[ws_n00 + fm + ko];
        b0.thread_elements()[1] = ws[ws_n01 + fm + ko];
        b1.thread_elements()[0] = ws[ws_n10 + fm + ko];
        b1.thread_elements()[1] = ws[ws_n11 + fm + ko];
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
        simdgroup_barrier(mem_flags::mem_none);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---- Write output ----
  const uint om0  = m_base + sm * 16u + fm;
  const uint om1  = m_base + sm * 16u + 8u + fm;
  const uint on0  = n_base + sn * 16u + fn0;
  const uint on1  = n_base + sn * 16u + fn1;
  const uint on80 = n_base + sn * 16u + 8u + fn0;
  const uint on81 = n_base + sn * 16u + 8u + fn1;

  if (om0 < (uint)M) {
    if (on0  < (uint)N) out[om0 * N + on0 ] = T(c00.thread_elements()[0]);
    if (on1  < (uint)N) out[om0 * N + on1 ] = T(c00.thread_elements()[1]);
    if (on80 < (uint)N) out[om0 * N + on80] = T(c01.thread_elements()[0]);
    if (on81 < (uint)N) out[om0 * N + on81] = T(c01.thread_elements()[1]);
  }
  if (om1 < (uint)M) {
    if (on0  < (uint)N) out[om1 * N + on0 ] = T(c10.thread_elements()[0]);
    if (on1  < (uint)N) out[om1 * N + on1 ] = T(c10.thread_elements()[1]);
    if (on80 < (uint)N) out[om1 * N + on80] = T(c11.thread_elements()[0]);
    if (on81 < (uint)N) out[om1 * N + on81] = T(c11.thread_elements()[1]);
  }
}
