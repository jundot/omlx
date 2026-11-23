// Adapted from TensorFold zig/kernels/cuda/minimax_m3_attn.cu,
// tf_mm3_attn_prompt_tc (Apache-2.0, Copyright 2026 TensorFold contributors).
// Changes: rewritten as an mx.fast.cuda_kernel body; queries and outputs are
// [heads, rows, 128] and the caches [kv heads, keys, 128]; the sizes come
// from the input shapes; the fixed draft window is removed; a block of 128
// keys is scored whole before its softmax weights are formed; each weight
// enters the value product as two bfloat16 parts (its rounding and the
// remainder), so the product keeps float32 precision; with PARTIAL each
// block's normalized output rounds to bfloat16 and the blocks combine by
// their log-sum-exp, as oMLX's MSA prefill kernels do on Metal.
//
// Inputs: q [16 * kv, rows, 128] bf16, k and v [kv, keys, 128] bf16,
// bits [rows, kv, words] uint32 (the blocks of 128 keys each (row, kv head)
// keeps), pos int32 [1] (position of row 0), scale float32 [1].
// Output: o [16 * kv, rows, 128] bf16, the causal attention of each row over
// the keys of its kept blocks.
// Grid: one 256-thread block per (tile of 8 rows, kv head); a warp owns one
// row's 16 heads. Dynamic shared memory: 2 * 128 * 136 bfloat16.
constexpr int D = 128, KB = 128, RPB = 8, KP = D + 8, VP = KB + 8;
constexpr float LN2 = 0.6931471805599453f;
extern __shared__ __align__(16) unsigned char smem[];
mm3_bf16 (*Ks)[KP] = reinterpret_cast<mm3_bf16 (*)[KP]>(smem);
mm3_bf16 (*Vt)[VP] = reinterpret_cast<mm3_bf16 (*)[VP]>(smem + KB * KP * sizeof(mm3_bf16));
const int rows = q_shape[1], keys = k_shape[1];
const int hk = bits_shape[1], words = bits_shape[2];
const int start = pos[0];
const int kvh = blockIdx.y;
const int r0 = blockIdx.x * RPB;
const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
const int row = r0 + w;
const bool live = row < rows;
const int last = start + (live ? row : r0);
const int maxlast = start + min(rows, r0 + RPB) - 1;
const float sl2 = scale[0] * 1.4426950408889634f;
// This warp's 16 heads' queries as A fragments, 8 k-steps of 16 dims.
uint32_t qa[8][4];
{
  const size_t at = live ? row : r0;
  const mm3_bf16* q0 = q + ((size_t)(kvh * 16 + g) * rows + at) * D;
  const mm3_bf16* q1 = q + ((size_t)(kvh * 16 + g + 8) * rows + at) * D;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    qa[s][0] = *reinterpret_cast<const uint32_t*>(q0 + 16 * s + 2 * t);
    qa[s][1] = *reinterpret_cast<const uint32_t*>(q1 + 16 * s + 2 * t);
    qa[s][2] = *reinterpret_cast<const uint32_t*>(q0 + 16 * s + 8 + 2 * t);
    qa[s][3] = *reinterpret_cast<const uint32_t*>(q1 + 16 * s + 8 + 2 * t);
  }
}
// acc sums P @ V of heads g (0, 1) and g + 8 (2, 3) at dims 8 n + 2 t on: over
// every block, or with PARTIAL over one block while tot holds the blocks
// combined so far.
float acc[16][4], tot[16][4];
#pragma unroll
for (int n = 0; n < 16; ++n)
#pragma unroll
  for (int c = 0; c < 4; ++c) acc[n][c] = tot[n][c] = 0.f;
// Running max and weight sum of each head: of the scores, or with PARTIAL of
// the blocks' log-sum-exp.
float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.f, l1 = 0.f;
const unsigned* kept = live ? bits + ((size_t)row * hk + kvh) * words : nullptr;
const int nb = maxlast / KB + 1;
const mm3_bf16* kc = k + (size_t)kvh * keys * D;
const mm3_bf16* vc = v + (size_t)kvh * keys * D;
for (int b = 0; b < nb; ++b) {
  const bool mine = live && b * KB <= last && ((kept[b / 32] >> (b % 32)) & 1u);
  if (!__syncthreads_or(mine)) continue;
  // Stage the block: key kk = tid / 2, dims (tid % 2) * 64 on; keys past the
  // tile's last row are zeroed.
  {
    const int kk = threadIdx.x >> 1, d0 = (threadIdx.x & 1) * 64;
    const int key = b * KB + kk;
    const bool ok = key <= maxlast;
    const uint4* ks = reinterpret_cast<const uint4*>(kc + (size_t)key * D + d0);
    const uint4* vs = reinterpret_cast<const uint4*>(vc + (size_t)key * D + d0);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint4 kv = ok ? ks[i] : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(&Ks[kk][d0 + 8 * i]) = kv;
      const uint4 vv = ok ? vs[i] : make_uint4(0, 0, 0, 0);
      const mm3_bf16* ve = reinterpret_cast<const mm3_bf16*>(&vv);
#pragma unroll
      for (int e = 0; e < 8; ++e) Vt[d0 + 8 * i + e][kk] = ve[e];
    }
  }
  __syncthreads();
  if (mine) {
    // Scores of keys 8 j + 2 t (+1) for heads g (0, 1) and g + 8 (2, 3).
    float sc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&Ks[8 * j + g][16 * s + 2 * t]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&Ks[8 * j + g][16 * s + 8 + 2 * t]);
        mm3_mma16816(sc[j], qa[s][0], qa[s][1], qa[s][2], qa[s][3], b0, b1);
      }
    }
    float mx0 = PARTIAL ? -CUDART_INF_F : m0, mx1 = PARTIAL ? -CUDART_INF_F : m1;
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const bool ok = b * KB + 8 * j + 2 * t + e <= last;
        sc[j][e] = ok ? sc[j][e] * sl2 : -CUDART_INF_F;
        sc[j][2 + e] = ok ? sc[j][2 + e] * sl2 : -CUDART_INF_F;
        mx0 = fmaxf(mx0, sc[j][e]);
        mx1 = fmaxf(mx1, sc[j][2 + e]);
      }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
    }
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        sc[j][e] = sc[j][e] == -CUDART_INF_F ? 0.f : exp2f(sc[j][e] - mx0);
        sc[j][2 + e] = sc[j][2 + e] == -CUDART_INF_F ? 0.f : exp2f(sc[j][2 + e] - mx1);
        s0 += sc[j][e];
        s1 += sc[j][2 + e];
      }
    if (!PARTIAL) {
      // Masked keys add 0 and an unmoved max rescales by 1 (no -inf - -inf).
      const float f0 = m0 == mx0 ? 1.f : exp2f(m0 - mx0);
      const float f1 = m1 == mx1 ? 1.f : exp2f(m1 - mx1);
      m0 = mx0;
      m1 = mx1;
      l0 = l0 * f0 + s0;
      l1 = l1 * f1 + s1;
#pragma unroll
      for (int n = 0; n < 16; ++n) {
        acc[n][0] *= f0;
        acc[n][1] *= f0;
        acc[n][2] *= f1;
        acc[n][3] *= f1;
      }
    }
    // A weight splits into its bfloat16 rounding and the rest, so the value
    // product keeps float32 precision on the bfloat16 tensor units.
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      uint32_t hi[4], lo[4];
#pragma unroll
      for (int f = 0; f < 4; ++f) {
        const float p0 = sc[2 * s + (f >> 1)][(f & 1) * 2];
        const float p1 = sc[2 * s + (f >> 1)][(f & 1) * 2 + 1];
        const __nv_bfloat162 h = __floats2bfloat162_rn(p0, p1);
        hi[f] = *reinterpret_cast<const uint32_t*>(&h);
        lo[f] = mm3_pack_bf16(p0 - __bfloat162float(h.x), p1 - __bfloat162float(h.y));
      }
#pragma unroll
      for (int n = 0; n < 16; ++n) {
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&Vt[8 * n + g][16 * s + 2 * t]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&Vt[8 * n + g][16 * s + 8 + 2 * t]);
        mm3_mma16816(acc[n], hi[0], hi[1], hi[2], hi[3], b0, b1);
        mm3_mma16816(acc[n], lo[0], lo[1], lo[2], lo[3], b0, b1);
      }
    }
    if (PARTIAL) {
      // The block's own softmax: its output rounds to bfloat16 and it weighs
      // exp(lse) against the other blocks, lse = ln of its summed weights.
#pragma unroll
      for (int off = 1; off < 4; off <<= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, off);
        s1 += __shfl_xor_sync(0xffffffffu, s1, off);
      }
      const float lse0 = (mx0 + log2f(s0)) * LN2, lse1 = (mx1 + log2f(s1)) * LN2;
      const float top0 = fmaxf(m0, lse0), top1 = fmaxf(m1, lse1);
      const float f0 = m0 == top0 ? 1.f : expf(m0 - top0);
      const float f1 = m1 == top1 ? 1.f : expf(m1 - top1);
      const float w0 = lse0 == top0 ? 1.f : expf(lse0 - top0);
      const float w1 = lse1 == top1 ? 1.f : expf(lse1 - top1);
      m0 = top0;
      m1 = top1;
      l0 = l0 * f0 + w0;
      l1 = l1 * f1 + w1;
#pragma unroll
      for (int n = 0; n < 16; ++n) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const float part = __bfloat162float(__float2bfloat16_rn(acc[n][c] / (c < 2 ? s0 : s1)));
          tot[n][c] = tot[n][c] * (c < 2 ? f0 : f1) + (c < 2 ? w0 : w1) * part;
          acc[n][c] = 0.f;
        }
      }
    }
  }
  __syncthreads();
}
if (!live) return;
if (!PARTIAL) {
#pragma unroll
  for (int off = 1; off < 4; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
}
mm3_bf16* out0 = o + ((size_t)(kvh * 16 + g) * rows + row) * D;
mm3_bf16* out1 = o + ((size_t)(kvh * 16 + g + 8) * rows + row) * D;
#pragma unroll
for (int n = 0; n < 16; ++n) {
  const float* sum = PARTIAL ? tot[n] : acc[n];
  *reinterpret_cast<uint32_t*>(out0 + 8 * n + 2 * t) = mm3_pack_bf16(sum[0] / l0, sum[1] / l0);
  *reinterpret_cast<uint32_t*>(out1 + 8 * n + 2 * t) = mm3_pack_bf16(sum[2] / l1, sum[3] / l1);
}
