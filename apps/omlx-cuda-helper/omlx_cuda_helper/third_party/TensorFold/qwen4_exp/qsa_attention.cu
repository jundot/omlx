// Adapted from TensorFold zig/kernels/cuda/flashnext_attn.cu, attn_chunks as
// fn_attn_fused runs it (Apache-2.0, Copyright 2026 TensorFold contributors).
// Changes: rewritten as an mx.fast.cuda_kernel body; queries are
// [heads, rows, D], the caches [kv heads, keys, D] and the output
// [rows, heads, D]; a row's keys come from its QSA block picks and causal
// tail instead of a gathered id list; keys and values have their own tile,
// so the values load while the scores are taken; each 16-product score step
// is added with a correctly rounded float add; each softmax weight enters
// the value product as two bf16 parts (its rounding and the remainder), so
// that product keeps float32 precision; the output is o / l in float32.
//
// Inputs: q [KV * G, rows, D] bf16, k and v [KV, keys, D] bf16,
// sel [rows, TOPK] int32 (each row's picked blocks in key order),
// first int32 [1] (keys row 0 sees). Output: out [rows, KV * G, D] bf16.
// Grid: one 128-thread block per (row, kv head); warp w scores keys
// [8 w, 8 w + 8) of each 32-key tile and sums dims [D / 4 * w, D / 4 * (w + 1)).
constexpr int TK = 32, QR = 16, NW = 4, DP = D + 8, PP = TK + 8, NQ = D / 32;
extern __shared__ __align__(16) unsigned char smem[];
qsa_bf16 (*qs)[DP] = reinterpret_cast<qsa_bf16 (*)[DP]>(smem);
qsa_bf16 (*ks)[DP] = qs + QR;
qsa_bf16 (*vs)[DP] = ks + TK;
qsa_bf16 (*ph)[PP] = reinterpret_cast<qsa_bf16 (*)[PP]>(vs + TK);
qsa_bf16 (*pl)[PP] = ph + QR;
__shared__ int stok[TK];
__shared__ float red_m[NW][QR], red_l[NW][QR];
const int row = blockIdx.x / KV, kvh = blockIdx.x % KV;
const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
const int R = q_shape[1], n_tok = k_shape[1];
const int vis = first[0] + row;
const int complete = vis / RATIO;
const int count = complete < TOPK ? complete : TOPK;
const int n_valid = count * RATIO + (vis - complete * RATIO);
const int* picks = sel + size_t(row) * TOPK;
for (int e = tid; e < QR * D / 8; e += blockDim.x) {
  const int i = e / (D / 8), c = e % (D / 8);
  const int head = kvh * G + (i < G ? i : 0);
  qsa_cp16(&qs[i][8 * c], q + (size_t(head) * R + row) * D + 8 * c, i < G);
}
const qsa_bf16* kh = k + size_t(kvh) * n_tok * D;
const qsa_bf16* vh = v + size_t(kvh) * n_tok * D;
float m[2] = {-CUDART_INF_F, -CUDART_INF_F}, l[2] = {0.f, 0.f};
float o[NQ][4];
#pragma unroll
for (int n = 0; n < NQ; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
for (int t0 = 0; t0 < n_valid; t0 += TK) {
  if (tid < TK) {
    const int j = t0 + tid;
    int tok = -1;
    if (j < count * RATIO) tok = picks[j / RATIO] * RATIO + j % RATIO;
    else if (j < n_valid) tok = complete * RATIO + (j - count * RATIO);
    stok[tid] = tok;
  }
  __syncthreads();
  for (int e = tid; e < TK * D / 8; e += blockDim.x) {
    const int kk = e / (D / 8), c = e % (D / 8), tok = stok[kk];
    qsa_cp16(&ks[kk][8 * c], kh + size_t(tok < 0 ? 0 : tok) * D + 8 * c, tok >= 0);
  }
  qsa_commit();
  for (int e = tid; e < TK * D / 8; e += blockDim.x) {
    const int kk = e / (D / 8), c = e % (D / 8), tok = stok[kk];
    qsa_cp16(&vs[kk][8 * c], vh + size_t(tok < 0 ? 0 : tok) * D + 8 * c, tok >= 0);
  }
  qsa_commit();
  qsa_wait<1>();
  __syncthreads();
  float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
  for (int step = 0; step < D / 16; step += 2) {
    uint32_t a0[4], a1[4], b[4];
    qsa_ldm4(a0, &qs[lane % 16][step * 16 + (lane / 16) * 8]);
    qsa_ldm4(a1, &qs[lane % 16][(step + 1) * 16 + (lane / 16) * 8]);
    qsa_ldm4(b, &ks[warp * 8 + lane % 8][step * 16 + (lane / 8) * 8]);
    qsa_mma_rn(s, a0, b[0], b[1]);
    qsa_mma_rn(s, a1, b[2], b[3]);
  }
  float tile_m[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const bool valid = stok[warp * 8 + 2 * t + (e & 1)] >= 0;
    s[e] = valid ? __fdiv_rn(s[e], SQRT_D) : -CUDART_INF_F;
    tile_m[e >> 1] = fmaxf(tile_m[e >> 1], s[e]);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    tile_m[h] = fmaxf(tile_m[h], __shfl_xor_sync(0xffffffffu, tile_m[h], 1));
    tile_m[h] = fmaxf(tile_m[h], __shfl_xor_sync(0xffffffffu, tile_m[h], 2));
    if (t == 0) red_m[warp][g + 8 * h] = tile_m[h];
  }
  __syncthreads();
  // Every tile holds a valid key, so each row's maximum is finite.
  float next_m[2], alpha[2], sums[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = g + 8 * h;
    const float tm = fmaxf(fmaxf(red_m[0][r], red_m[1][r]), fmaxf(red_m[2][r], red_m[3][r]));
    next_m[h] = fmaxf(m[h], tm);
    alpha[h] = m[h] == -CUDART_INF_F ? 0.f : expf(m[h] - next_m[h]);
    sums[h] = 0.f;
  }
  float p[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    p[e] = s[e] == -CUDART_INF_F ? 0.f : expf(s[e] - next_m[e >> 1]);
    sums[e >> 1] += p[e];
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    __nv_bfloat162 high, low;
    qsa_split(p[2 * h], p[2 * h + 1], high, low);
    *reinterpret_cast<__nv_bfloat162*>(&ph[g + 8 * h][warp * 8 + 2 * t]) = high;
    *reinterpret_cast<__nv_bfloat162*>(&pl[g + 8 * h][warp * 8 + 2 * t]) = low;
    sums[h] += __shfl_xor_sync(0xffffffffu, sums[h], 1);
    sums[h] += __shfl_xor_sync(0xffffffffu, sums[h], 2);
    if (t == 0) red_l[warp][g + 8 * h] = sums[h];
  }
  qsa_wait<0>();
  __syncthreads();
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = g + 8 * h;
    const float sum = ((red_l[0][r] + red_l[1][r]) + red_l[2][r]) + red_l[3][r];
    l[h] = l[h] * alpha[h] + sum;
    m[h] = next_m[h];
  }
#pragma unroll
  for (int n = 0; n < NQ; ++n) {
    o[n][0] *= alpha[0];
    o[n][1] *= alpha[0];
    o[n][2] *= alpha[1];
    o[n][3] *= alpha[1];
  }
#pragma unroll
  for (int kq = 0; kq < TK / 16; ++kq) {
    uint32_t ah[4], al[4];
    qsa_ldm4(ah, &ph[lane % 16][kq * 16 + (lane / 16) * 8]);
    qsa_ldm4(al, &pl[lane % 16][kq * 16 + (lane / 16) * 8]);
#pragma unroll
    for (int n = 0; n < NQ; n += 2) {
      uint32_t b[4];
      qsa_ldm4t(b, &vs[kq * 16 + lane % 16][warp * (D / 4) + n * 8 + (lane / 16) * 8]);
      qsa_mma(o[n], ah, b[0], b[1]);
      qsa_mma(o[n], al, b[0], b[1]);
      qsa_mma(o[n + 1], ah, b[2], b[3]);
      qsa_mma(o[n + 1], al, b[2], b[3]);
    }
  }
  __syncthreads();
}
#pragma unroll
for (int h = 0; h < 2; ++h) {
  const int i = g + 8 * h;
  if (i >= G) continue;
  qsa_bf16* y = out + (size_t(row) * KV * G + kvh * G + i) * D + warp * (D / 4) + 2 * t;
#pragma unroll
  for (int n = 0; n < NQ; ++n) {
    *reinterpret_cast<__nv_bfloat162*>(y + n * 8) = __floats2bfloat162_rn(
        __fdiv_rn(o[n][2 * h], l[h]), __fdiv_rn(o[n][2 * h + 1], l[h]));
  }
}
