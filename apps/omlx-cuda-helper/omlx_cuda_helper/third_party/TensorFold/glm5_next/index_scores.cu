// Adapted from TensorFold zig/kernels/cuda/glm_index_scores.cuh,
// tf_glm_index_scores (Apache-2.0, Copyright 2026 TensorFold contributors).
// Changes: rewritten as an mx.fast.cuda_kernel body; the sizes come from the
// input shapes and template arguments; the head weights arrive scaled; a
// score is rounded once to bf16 and a window that is not complete at the
// query position gets the bf16 value of -1e30; paged keys are dropped.
//
// Inputs: qi [rows, heads, 128] bf16 index queries, wts [rows, heads] bf16
// head weights, pk [windows, 128] bf16 pooled keys, pos int32 [1] position of
// row 0. Output: out [rows, windows] bf16, sum over heads of
// weight * max(query . key, 0), accumulated in float32.
// Grid: (ceil(rows / 8), ceil(tiles / LOOP)) blocks of 256 threads; a warp is
// one query row and a block walks LOOP tiles of 64 windows, staged once each.
constexpr int D = 128, BP = 64, RB = 8, KQ = D / 16;
__shared__ __align__(16) __nv_bfloat16 tile[2][BP * D];
const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
const long rows = qi_shape[0], heads = qi_shape[1], np = pk_shape[0], nb = (np + BP - 1) / BP;
const long r = (long)blockIdx.x * RB + warp, P = pos[0];
const bool live = r < rows;
const long npool = live ? (P + r + 1) / KP : 0;
const long last = (rows < (long)(blockIdx.x + 1) * RB ? rows : (long)(blockIdx.x + 1) * RB) - 1;
const long nmax = (P + last + 1) / KP;
const __nv_bfloat16 hidden = __float2bfloat16_rn(-1e30f);
// This row's query fragments for the 8 k16 steps and 2 head tiles (ldmatrix's
// A layout), and a lane's 4 head weights.
uint32_t aq[KQ][2][4];
#pragma unroll
for (int kq = 0; kq < KQ; ++kq)
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int h0 = i * 16 + g, h1 = h0 + 8;
    const __nv_bfloat16* b0 = qi + ((live ? r : 0) * heads + (h0 < heads ? h0 : 0)) * D + kq * 16 + 2 * q;
    const __nv_bfloat16* b1 = qi + ((live ? r : 0) * heads + (h1 < heads ? h1 : 0)) * D + kq * 16 + 2 * q;
    const bool o0 = live && h0 < heads, o1 = live && h1 < heads;
    aq[kq][i][0] = o0 ? *reinterpret_cast<const uint32_t*>(b0) : 0u;
    aq[kq][i][1] = o1 ? *reinterpret_cast<const uint32_t*>(b1) : 0u;
    aq[kq][i][2] = o0 ? *reinterpret_cast<const uint32_t*>(b0 + 8) : 0u;
    aq[kq][i][3] = o1 ? *reinterpret_cast<const uint32_t*>(b1 + 8) : 0u;
  }
float wv[4];
for (int i = 0; i < 4; ++i) {
  const long hh = g + 8 * i;
  wv[i] = live && hh < heads ? __bfloat162float(wts[r * heads + hh]) : 0.0f;
}
// The block's 256 threads copy tile pb's keys (windows past the block's
// visible ones read as 0).
auto stage = [&](int buf, long pb) {
  for (int idx = threadIdx.x; idx < BP * (D / 8); idx += RB * 32) {
    const int row = idx / (D / 8), u = idx % (D / 8);
    const long p = pb * BP + row;
    const bool ok = p < nmax;
    tfi_cp16(&tile[buf][tfi_tile_at(row, u)], pk + (ok ? p : 0) * D + u * 8, ok);
  }
};
const long first = (long)blockIdx.y * LOOP;
auto any = [&](int j) { return j < LOOP && first + j < nb && (first + j) * BP < nmax; };
if (any(0)) stage(0, first);
tfi_commit();
for (int j = 0; j < LOOP; ++j) {
  const long pb = first + j, p0 = pb * BP;
  if (pb >= nb) break;
  tfi_wait<0>();
  __syncthreads();  // tile j is in, and every warp is done with tile j - 1
  if (any(j + 1)) stage((j + 1) & 1, pb + 1);
  tfi_commit();
  if (!live) continue;
  if (p0 >= npool) {  // the row sees none of these windows
    for (int e = lane; e < BP; e += 32)
      if (p0 + e < np) out[r * np + p0 + e] = hidden;
    continue;
  }
  for (int gi = 0; gi < BP / 16; ++gi) {
    float acc[2][2][4] = {};
    const __nv_bfloat16* keys = tile[j & 1] + gi * 16 * D;
    #pragma unroll
    for (int kq = 0; kq < KQ; ++kq) {
      uint32_t b[4];
      tfi_ldm4(b, keys + tfi_tile_at((lane >> 4) * 8 + (lane & 7), kq * 2 + ((lane >> 3) & 1)));
      for (int nt = 0; nt < 2; ++nt)
        for (int i = 0; i < 2; ++i) tfi_mma(acc[i][nt], aq[kq][i], b + 2 * nt);
    }
    for (int nt = 0; nt < 2; ++nt)
      for (int e = 0; e < 2; ++e) {
        const float v0 = fmaxf(acc[0][nt][e], 0.0f), v1 = fmaxf(acc[0][nt][e + 2], 0.0f);
        const float v2 = fmaxf(acc[1][nt][e], 0.0f), v3 = fmaxf(acc[1][nt][e + 2], 0.0f);
        float s = __fmul_rn(wv[1], v1);
        s = __fmaf_rn(wv[0], v0, s);
        s = __fmaf_rn(wv[2], v2, s);
        s = __fmaf_rn(wv[3], v3, s);
        for (int o = 16; o >= 4; o >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, o);
        const long p = p0 + gi * 16 + nt * 8 + q * 2 + e;
        if (g == 0 && p < np) out[r * np + p] = p < npool ? __float2bfloat16_rn(s) : hidden;
      }
  }
}
