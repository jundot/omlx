// Adapted from TensorFold zig/kernels/cuda/minimax_m3_attn.cu,
// tf_mm3_index_scores_tc (Apache-2.0, Copyright 2026 TensorFold contributors).
// Changes: rewritten as an mx.fast.cuda_kernel body; the index queries are
// read as [heads, rows, 128] and the sizes come from the input shapes; a
// block that no row of the tile can select is skipped.
//
// Inputs: iq [heads, rows, 128] bf16, ik [keys, 128] bf16, pos int32 [1].
// Output: scores [rows, heads, blocks] float32, the highest product of a
// (row, head) with the keys of a block that are not after the row. Only the
// blocks before a row's own block are written.
// Grid: one 128-thread block per (key block, tile of 64 (row, head) pairs).
constexpr int D = 128, KB = 128, KP = D + 8;
__shared__ __align__(16) mm3_bf16 Ks[KB][KP];
const int heads = iq_shape[0], rows = iq_shape[1];
const int blocks = (ik_shape[0] + KB - 1) / KB;
const int start = pos[0];
const int j = blockIdx.x;
const int pairs = rows * heads;
const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
const int tile = blockIdx.y * 64;
if (tile >= pairs) return;
// The own block is always kept, so only earlier blocks need a score.
const int tile_last = min(tile + 63, pairs - 1) / heads;
if (j >= (start + tile_last) / KB) return;
const int maxlast = start + rows - 1;
for (int i = threadIdx.x; i < KB * (D / 8); i += blockDim.x) {
  const int kk = i / (D / 8), d0 = (i % (D / 8)) * 8;
  const int key = j * KB + kk;
  const uint4 v = key <= maxlast
      ? *reinterpret_cast<const uint4*>(ik + (size_t)key * D + d0)
      : make_uint4(0, 0, 0, 0);
  *reinterpret_cast<uint4*>(&Ks[kk][d0]) = v;
}
__syncthreads();
const int p0 = tile + w * 16;
if (p0 >= pairs) return;
const int pa = min(p0 + g, pairs - 1), pb = min(p0 + g + 8, pairs - 1);
const mm3_bf16* q0 = iq + ((size_t)(pa % heads) * rows + pa / heads) * D;
const mm3_bf16* q1 = iq + ((size_t)(pb % heads) * rows + pb / heads) * D;
uint32_t qa[8][4];
#pragma unroll
for (int s = 0; s < 8; ++s) {
  qa[s][0] = *reinterpret_cast<const uint32_t*>(q0 + 16 * s + 2 * t);
  qa[s][1] = *reinterpret_cast<const uint32_t*>(q1 + 16 * s + 2 * t);
  qa[s][2] = *reinterpret_cast<const uint32_t*>(q0 + 16 * s + 8 + 2 * t);
  qa[s][3] = *reinterpret_cast<const uint32_t*>(q1 + 16 * s + 8 + 2 * t);
}
const int last0 = start + pa / heads, last1 = start + pb / heads;
float mx0 = -CUDART_INF_F, mx1 = -CUDART_INF_F;
#pragma unroll 4
for (int n = 0; n < KB / 8; ++n) {
  float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&Ks[8 * n + g][16 * s + 2 * t]);
    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&Ks[8 * n + g][16 * s + 8 + 2 * t]);
    mm3_mma16816(c, qa[s][0], qa[s][1], qa[s][2], qa[s][3], b0, b1);
  }
  const int key = j * KB + 8 * n + 2 * t;
  if (key <= last0) mx0 = fmaxf(mx0, c[0]);
  if (key + 1 <= last0) mx0 = fmaxf(mx0, c[1]);
  if (key <= last1) mx1 = fmaxf(mx1, c[2]);
  if (key + 1 <= last1) mx1 = fmaxf(mx1, c[3]);
}
#pragma unroll
for (int off = 1; off < 4; off <<= 1) {
  mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
  mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
}
if (t == 0) {
  if (p0 + g < pairs) scores[(size_t)(p0 + g) * blocks + j] = mx0;
  if (p0 + g + 8 < pairs) scores[(size_t)(p0 + g + 8) * blocks + j] = mx1;
}
