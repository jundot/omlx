// Adapted from TensorFold zig/kernels/cuda/gdn_prefill.cu, chain_kernel
// (Apache-2.0, Copyright 2026 TensorFold contributors). Changes: rewritten as
// an mx.fast.cuda_kernel body with the sizes from the input shapes; bf16
// queries, keys and update rates only.
//
// Inputs: q, k [1, W, hk, 128] bf16, v [1, W, hv, 128] bf16, g [1, W, hv]
// float32 (decay), beta [1, W, hv] bf16, state [1, hv, 128 (v), 128 (k)]
// float32. Outputs: y [1, W, hv, 128] bf16, last (the state after W steps).
// Grid: (hv, 128 / ROWS) blocks of 2 * ROWS threads; two threads hold one
// value row of the state, 64 key dims each.
constexpr int THREADS = 2 * ROWS;
__shared__ float4 ks[STEPS][GDN_DK / 4], qs[STEPS][GDN_DK / 4];
__shared__ float gs[STEPS], bs[STEPS];
__shared__ __align__(16) __nv_bfloat16 vs[STEPS][ROWS];
const int W = q_shape[1], hk = q_shape[2], hv = v_shape[2];
const int head = blockIdx.x, row = blockIdx.y * ROWS + (threadIdx.x >> 1), half = threadIdx.x & 1;
const int key_head = head / (hv / hk);
float s[64];
const float* s0 = state + (size_t(head) * GDN_DV + row) * GDN_DK;
#pragma unroll
for (int j = 0; j < 16; ++j) {
  const float4 t = *reinterpret_cast<const float4*>(s0 + 8 * j + 4 * half);
  s[4 * j] = t.x;
  s[4 * j + 1] = t.y;
  s[4 * j + 2] = t.z;
  s[4 * j + 3] = t.w;
}
GdnPrefetch<THREADS, ROWS, STEPS> pf;
pf.fetch(q, k, v, g, beta, 0, W, hk, hv, head, key_head, blockIdx.y * ROWS);
for (int t0 = 0; t0 < W; t0 += STEPS) {
  const int n = min(STEPS, W - t0);
  __syncthreads();
  pf.store(ks, qs, gs, bs, vs);
  __syncthreads();
  if (t0 + STEPS < W) pf.fetch(q, k, v, g, beta, t0 + STEPS, W, hk, hv, head, key_head, blockIdx.y * ROWS);
  for (int tt = 0; tt < n; ++tt) {
    const float gt = gs[tt], bt = bs[tt];
    const size_t vat = (size_t(t0 + tt) * hv + head) * GDN_DV + row;
    const float vt = __bfloat162float(vs[tt][threadIdx.x >> 1]);
    float m[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float4 kk = ks[tt][2 * j + half];
      s[4 * j] = s[4 * j] * gt;
      s[4 * j + 1] = s[4 * j + 1] * gt;
      s[4 * j + 2] = s[4 * j + 2] * gt;
      s[4 * j + 3] = s[4 * j + 3] * gt;
      m[0] = __fmaf_rn(s[4 * j], kk.x, m[0]);
      m[1] = __fmaf_rn(s[4 * j + 1], kk.y, m[1]);
      m[2] = __fmaf_rn(s[4 * j + 2], kk.z, m[2]);
      m[3] = __fmaf_rn(s[4 * j + 3], kk.w, m[3]);
    }
    float mem = (m[0] + m[1]) + (m[2] + m[3]);
    mem = mem + __shfl_xor_sync(0xffffffffu, mem, 1);
    const float delta = (vt - mem) * bt;
    float o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float4 kk = ks[tt][2 * j + half], qq = qs[tt][2 * j + half];
      s[4 * j] = __fmaf_rn(kk.x, delta, s[4 * j]);
      s[4 * j + 1] = __fmaf_rn(kk.y, delta, s[4 * j + 1]);
      s[4 * j + 2] = __fmaf_rn(kk.z, delta, s[4 * j + 2]);
      s[4 * j + 3] = __fmaf_rn(kk.w, delta, s[4 * j + 3]);
      o[0] = __fmaf_rn(s[4 * j], qq.x, o[0]);
      o[1] = __fmaf_rn(s[4 * j + 1], qq.y, o[1]);
      o[2] = __fmaf_rn(s[4 * j + 2], qq.z, o[2]);
      o[3] = __fmaf_rn(s[4 * j + 3], qq.w, o[3]);
    }
    float out = (o[0] + o[1]) + (o[2] + o[3]);
    out = out + __shfl_xor_sync(0xffffffffu, out, 1);
    if (half == 0) y[vat] = __float2bfloat16_rn(out);
  }
}
float* s1 = last + (size_t(head) * GDN_DV + row) * GDN_DK;
#pragma unroll
for (int j = 0; j < 16; ++j)
  *reinterpret_cast<float4*>(s1 + 8 * j + 4 * half) =
      make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
