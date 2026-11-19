// moe_prefill_gate_up.cu on the MX tile: x as E4M3 codes and E8M0 scales,
// mxfp4 banks, and the SwiGLU epilogue's FP8 round trip written as codes and
// scales for the MX down projection. A pass covers MT tiles of 64 routes.
extern __shared__ __align__(128) uint8_t smem[];
__shared__ int range[2];
const uint32_t e = blockIdx.y;
msv_expert_range(inds, inds_shape[0], e, range);
const int begin = range[0], end = range[1];
const long n0 = (long)e * I + blockIdx.x * 64;
const uint8_t* gcodes = reinterpret_cast<const uint8_t*>(gw) + n0 * (K / 2);
const uint8_t* ucodes = reinterpret_cast<const uint8_t*>(uw) + n0 * (K / 2);
const uint8_t* gscales = gs + n0 * (K / 32);
const uint8_t* uscales = us + n0 * (K / 32);
const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3;
const float limit = __uint_as_float(LIMIT);
__nv_bfloat16* tile = reinterpret_cast<__nv_bfloat16*>(smem);
for (int r0 = begin + (int)blockIdx.z * 64 * MT; r0 < end; r0 += (int)gridDim.z * 64 * MT) {
  const int mv = end - r0 < 64 * MT ? end - r0 : 64 * MT;
  float acc[MT][2][4][4];
  msv_expert_tile_mx<K, BDEP, STG, MT>(smem, mv,
      [&](int row) { return xc + (long)rows[r0 + row] * K; },
      [&](int row) { return xs + (long)rows[r0 + row] * (K / 32); },
      [&](int r) { return (r < 64 ? gcodes : ucodes) + (long)(r & 63) * (K / 2); },
      [&](int r) { return (r < 64 ? gscales : uscales) + (long)(r & 63) * (K / 32); }, acc);
  // Stage both projections as bf16: columns 0-63 gate, 64-127 up.
  #pragma unroll
  for (int m = 0; m < MT; ++m)
    #pragma unroll
    for (int a = 0; a < 2; ++a)
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        #pragma unroll
        for (int hlf = 0; hlf < 2; ++hlf) {
          const int row = m * 64 + wm * 32 + a * 16 + hlf * 8 + (lane >> 2);
          const int col = wn * 32 + j * 8 + (lane & 3) * 2;
          *reinterpret_cast<__nv_bfloat162*>(tile + row * 128 + col) = __floats2bfloat162_rn(acc[m][a][j][2 * hlf], acc[m][a][j][2 * hlf + 1]);
        }
  __syncthreads();
  // One thread per row and group of 32 output values.
  for (int q = tid; q < 128 * MT; q += 256) {
    const int row = q >> 1, c0 = (q & 1) * 32;
    if (row < mv) {
      const float weight = rw[r0 + row];
      float v[32];
      #pragma unroll
      for (int i = 0; i < 32; ++i) {
        float g = v41_float(tile[row * 128 + c0 + i]);
        float u = v41_float(tile[row * 128 + 64 + c0 + i]);
        if (LIMIT) {
          g = fminf(g, limit);
          u = fminf(fmaxf(u, -limit), limit);
        }
        const float ex = 1.0f / (1.0f + expf(fabsf(g)));
        const float value = __fmul_rn(__fmul_rn(g, g < 0.0f ? ex : 1.0f - ex), u);
        v[i] = v41_round(__fmul_rn(value, weight), __nv_bfloat16());
      }
      const long at = (long)(r0 + row) * I + blockIdx.x * 64 + c0;
      ascales[at / 32] = (uint8_t)v41_quantize_codes<FLOOR, INV_LIMIT>(v, acodes + at);
    }
  }
  __syncthreads();
}
