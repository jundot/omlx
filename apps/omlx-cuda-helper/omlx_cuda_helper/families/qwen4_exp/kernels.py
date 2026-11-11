# SPDX-License-Identifier: Apache-2.0
"""CUDA kernels for Qwen3.8 Flash-Next prefill.

A block takes a kernel only after ``enable_kernels`` matched it with the
block's plain-ops reference on the block's own geometry. The QSA attention
and GDN recurrence sources live under ``third_party/TensorFold/qwen4_exp``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"


def _borrowed(name: str) -> str:
    return (_THIRD_PARTY / "TensorFold" / "qwen4_exp" / name).read_text()


def available() -> bool:
    return mx.cuda.is_available()


def close(got: mx.array, want: mx.array, what: str) -> bool:
    """Whether a kernel result equals its reference up to float32 rounding.

    A 16-bit output may round the other way in a few places when the float32
    sums take another order; anything more is a wrong kernel.
    """
    a, b = got.astype(mx.float32), want.astype(mx.float32)
    error = (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-30)).item()
    differ = (got != want).astype(mx.float32).mean().item()
    limit = 1e-5 if want.dtype == mx.float32 else 1e-3
    if error <= limit and (want.dtype == mx.float32 or differ <= 0.02):
        return True
    logger.warning(
        "%s kernel differs from its reference (relative error %.2e, %.1f%% of "
        "values); using the reference",
        what,
        error,
        100 * differ,
    )
    return False


# Conversions the kernels share; the 16-bit types round to nearest even.
_HEADER = """
__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ inline float to_float(__half x) { return __half2float(x); }
__device__ inline void from_float(float x, float* out) { *out = x; }
__device__ inline void from_float(float x, __nv_bfloat16* out) {
  *out = __float2bfloat16(x);
}
__device__ inline void from_float(float x, __half* out) { *out = __float2half(x); }
// A float rounded to the dtype of ``like`` and widened again.
template <typename T> __device__ inline float rounded(float x, const T* like) {
  T value;
  from_float(x, &value);
  return to_float(value);
}
"""


# MLX's own element functors, so a kernel rounds like the MLX ops it replaces.
_FUNCTORS = """
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include <cuda_bf16.h>
"""

_FUNCTOR_KERNELS: dict = {}


def _functor_kernel(name: str, inputs: list[str], outputs: list[str], source: str):
    if name not in _FUNCTOR_KERNELS:
        _FUNCTOR_KERNELS[name] = mx.fast.cuda_kernel(
            name=f"qwen4_{name}",
            input_names=inputs,
            output_names=outputs,
            source=source,
            header=_FUNCTORS,
        )
    return _FUNCTOR_KERNELS[name]


class SparseAttention:
    """Attention of contiguous query rows over their QSA-selected keys.

    Keys and values are read in place by token id on the tensor cores: bf16
    products with float32 sums, the softmax weights split into two bf16 parts
    so the value sums keep float32 precision, and the output rounds once.
    """

    # Query rows of one launch; bounds the selection the caller builds for it.
    rows_per_launch = 2048

    @staticmethod
    def fits(heads: int, kv_heads: int, dim: int, dtype: mx.Dtype) -> bool:
        """Whether a geometry fits the 16-row query tile and 48 KiB of tiles."""
        groups = heads // kv_heads
        return (
            dtype == mx.bfloat16
            and heads % kv_heads == 0
            and groups <= 16
            and dim % 64 == 0
            and dim <= 256
        )

    def __init__(self, heads: int, kv_heads: int, dim: int, ratio: int, top_k: int):
        self.heads, self.kv_heads, self.dim = heads, kv_heads, dim
        self.template = [
            ("KV", kv_heads),
            ("G", heads // kv_heads),
            ("D", dim),
            ("RATIO", ratio),
            ("TOPK", top_k),
        ]
        source = _borrowed("qsa_attention.cu")
        self.kernel = mx.fast.cuda_kernel(
            name=f"qwen4_qsa_mma_{heads}_{kv_heads}_{dim}_{ratio}_{top_k}",
            input_names=["q", "k", "v", "sel", "first"],
            output_names=["out"],
            source=source.replace("SQRT_D", f"{float(dim) ** 0.5!r}f"),
            header=_borrowed("qsa_header.cuh"),
            # Query tile, key and value tiles, and the two weight parts.
            shared_memory=(16 + 2 * 32) * (dim + 8) * 2 + 2 * 16 * 40 * 2,
        )

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        selected: mx.array,
        first_visible: int,
    ) -> mx.array:
        """``queries`` [1, heads, rows, dim] -> [1, rows, heads, dim].

        ``selected`` [rows, top_k] holds each row's block ids in key order and
        row ``i`` sees the first ``first_visible + i`` keys.
        """
        rows = queries.shape[2]
        first = mx.array([first_visible], dtype=mx.int32)
        out = self.kernel(
            inputs=[queries[0], keys[0], values[0], selected, first],
            template=self.template,
            grid=(rows * self.kv_heads * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(rows, self.heads, self.dim)],
            output_dtypes=[queries.dtype],
        )[0]
        return out[None]


# ``Attention.prepare`` of one (row, head) a warp: the RMS norm with its
# float32 scale rounded once, then the rotation of the first 2 * HALF dims by
# the row's position, each product and sum rounded as the plain ops round
# them. Heads in order: queries, keys, values (copied), index queries, the
# raw index key (copied). A lane holds width / 32 consecutive dims.
_QSA_PREP = """
  using namespace mlx::core::cu;
  constexpr int UNITS = H + 2 * KV + IH + 1;
  const int lane = threadIdx.x & 31;
  const size_t unit = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int rows = qo_shape[1];
  const size_t row = unit / UNITS;
  if (row >= size_t(rows)) return;
  const int u = int(unit % UNITS);
  const __nv_bfloat16* src;
  const float* w = nullptr;
  __nv_bfloat16* dst;
  int width = D;
  if (u < H) {
    src = qo + (row * H + u) * 2 * D;
    w = qs;
    dst = q + (size_t(u) * rows + row) * D;
  } else if (u < H + KV) {
    src = ko + (row * KV + (u - H)) * D;
    w = ks;
    dst = k + (size_t(u - H) * rows + row) * D;
  } else if (u < H + 2 * KV) {
    src = vo + (row * KV + (u - H - KV)) * D;
    dst = v + (size_t(u - H - KV) * rows + row) * D;
  } else {
    const int i = u - H - 2 * KV;
    width = ID;
    src = io + (row * (IH + 1) + i) * ID;
    w = i < IH ? iqs : nullptr;
    dst = i < IH ? iq + (row * IH + i) * ID : ik + row * ID;
  }
  const int per = width / 32, base = lane * per;
  float x[8];
  #pragma unroll
  for (int e = 0; e < 8; ++e) x[e] = e < per ? __bfloat162float(src[base + e]) : 0.f;
  if (w != nullptr) {
    float ss = 0.f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) ss = fmaf(x[e], x[e], ss);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float inv = rsqrtf(__fadd_rn(__fdiv_rn(ss, float(width)), eps[0]));
    #pragma unroll
    for (int e = 0; e < 8; ++e)
      x[e] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(x[e], inv), w[base + e])));
    // The partner of dim d < HALF is d + HALF, per / 32 * HALF lanes away.
    const int hop = HALF / per;
    const float pos = float(pos0[0] + int(row));
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float other = __shfl_xor_sync(0xffffffffu, x[e], hop);
      const int d = base + e;
      if (e < per && d < 2 * HALF) {
        const int i = d < HALF ? d : d - HALF;
        const float angle = __fmul_rn(pos, freq[i]);
        const float c = Cos{}(angle), sn = Sin{}(angle);
        x[e] = d < HALF ? __fsub_rn(__fmul_rn(x[e], c), __fmul_rn(other, sn))
                        : __fadd_rn(__fmul_rn(x[e], c), __fmul_rn(other, sn));
      }
    }
  }
  #pragma unroll
  for (int e = 0; e < 8; ++e)
    if (e < per) dst[base + e] = __float2bfloat16_rn(x[e]);
"""


def qsa_prep_fits(heads, kv_heads, dim, index_heads, index_dim, half, dtype) -> bool:
    """Whether ``qsa_prep`` covers a QSA geometry: whole lanes and rotations."""
    return (
        dtype == mx.bfloat16
        and dim in (128, 256)
        and index_dim in (128, 256)
        and half > 0
        and half % (dim // 32) == 0
        and half % (index_dim // 32) == 0
        and 2 * half <= min(dim, index_dim)
        and all(
            (half // (w // 32)) & (half // (w // 32) - 1) == 0 for w in (dim, index_dim)
        )
    )


def qsa_prep(att, q_out, k_out, v_out, index, first: int) -> tuple[mx.array, ...]:
    """``att.prepare`` outputs but the gate: (queries, keys, values, raw index
    keys, index queries) for rows at positions ``first`` on."""
    rows, dim = q_out.shape[1], att.head_dim
    heads, kv, ih, idim = att.heads, att.kv_heads, att.index_heads, att.index_dim
    units = heads + 2 * kv + ih + 1
    threads = rows * units * 32
    kernel = _functor_kernel(
        "qsa_prep",
        ["qo", "ko", "vo", "io", "qs", "ks", "iqs", "freq", "pos0", "eps"],
        ["q", "k", "v", "iq", "ik"],
        _QSA_PREP,
    )
    q, k, v, iq, ik = kernel(
        inputs=[
            q_out,
            k_out,
            v_out,
            index,
            att.q_scale,
            att.k_scale,
            att.index_q_scale,
            att.inv_freq,
            mx.array([first], dtype=mx.int32),
            mx.array([att.eps], dtype=mx.float32),
        ],
        template=[
            ("H", heads),
            ("KV", kv),
            ("D", dim),
            ("IH", ih),
            ("ID", idim),
            ("HALF", att.inv_freq.shape[0]),
        ],
        grid=(-(-threads // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (1, heads, rows, dim),
            (1, kv, rows, dim),
            (1, kv, rows, dim),
            (rows, ih, idim),
            (1, rows, idim),
        ],
        output_dtypes=[q_out.dtype] * 5,
    )
    return q, k, v, ik, iq


# Block scores [rows, blocks] float32: bf16 products on the tensor cores,
# each head's relu added in head order and divided by sqrt(DI) as
# ``Attention._block_scores`` does; blocks a row does not see completely
# score -inf. A launch block keeps 16 query rows and walks tiles of 64 blocks
# gridDim.x apart, loading the next tile while it scores one; warp w takes
# blocks [16 w, 16 w + 16) of a tile.
_BLOCK_SCORES = """
  constexpr int QT = 16, BT = 64, DP = DI + 8;
  extern __shared__ __align__(16) unsigned char smem[];
  qsa_bf16 (*qa)[QT][DP] = reinterpret_cast<qsa_bf16 (*)[QT][DP]>(smem);
  qsa_bf16 (*kb)[BT][DP] = reinterpret_cast<qsa_bf16 (*)[BT][DP]>(smem + HI * QT * DP * 2);
  const int rows = iq_shape[0], nb = pooled_shape[0], tiles = (nb + BT - 1) / BT;
  const int r0 = blockIdx.y * QT;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int seen = first[0];
  for (int e = tid; e < HI * QT * DI / 8; e += blockDim.x) {
    const int h = e / (QT * DI / 8), i = (e / (DI / 8)) % QT, c = e % (DI / 8);
    const int r = r0 + i;
    qsa_cp16(&qa[h][i][8 * c], iq + (size_t(r < rows ? r : 0) * HI + h) * DI + 8 * c, r < rows);
  }
  auto load = [&](int tile, int buf) {
    for (int e = tid; e < BT * DI / 8; e += blockDim.x) {
      const int j = e / (DI / 8), c = e % (DI / 8), b = tile * BT + j;
      qsa_cp16(&kb[buf][j][8 * c], pooled + size_t(b < nb ? b : 0) * DI + 8 * c, b < nb);
    }
    qsa_commit();
  };
  int buf = 0;
  if (int(blockIdx.x) < tiles) load(blockIdx.x, 0);
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x, buf ^= 1) {
    const bool ahead = tile + int(gridDim.x) < tiles;
    if (ahead) load(tile + gridDim.x, buf ^ 1);
    if (ahead) qsa_wait<1>(); else qsa_wait<0>();
    __syncthreads();
    float acc[HI][2][4];
    #pragma unroll
    for (int h = 0; h < HI; ++h)
      #pragma unroll
      for (int n = 0; n < 2; ++n) acc[h][n][0] = acc[h][n][1] = acc[h][n][2] = acc[h][n][3] = 0.f;
    #pragma unroll
    for (int step = 0; step < DI / 16; step += 2) {
      uint32_t b[2][4];
      #pragma unroll
      for (int n = 0; n < 2; ++n)
        qsa_ldm4(b[n], &kb[buf][warp * 16 + n * 8 + lane % 8][step * 16 + (lane / 8) * 8]);
      #pragma unroll
      for (int h = 0; h < HI; ++h) {
        uint32_t a0[4], a1[4];
        qsa_ldm4(a0, &qa[h][lane % 16][step * 16 + (lane / 16) * 8]);
        qsa_ldm4(a1, &qa[h][lane % 16][(step + 1) * 16 + (lane / 16) * 8]);
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
          qsa_mma_rn(acc[h][n], a0, b[n][0], b[n][1]);
          qsa_mma_rn(acc[h][n], a1, b[n][2], b[n][3]);
        }
      }
    }
    #pragma unroll
    for (int n = 0; n < 2; ++n)
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int r = r0 + g + 8 * (e >> 1), b = tile * BT + warp * 16 + n * 8 + 2 * t + (e & 1);
        if (r >= rows || b >= nb) continue;
        float total = fmaxf(acc[0][n][e], 0.f);
        #pragma unroll
        for (int h = 1; h < HI; ++h) total = __fadd_rn(total, fmaxf(acc[h][n][e], 0.f));
        const bool visible = b < (seen + r) / RATIO;
        scores[size_t(r) * nb + b] = visible ? __fdiv_rn(total, SQRT_DI) : -CUDART_INF_F;
      }
    // The next load reuses this buffer.
    __syncthreads();
  }
"""

# Each row's TOPK best blocks in key order, one 256-thread launch block a row:
# a radix select on order-preserving keys of the scores, ties at the cut going
# to the higher block ids as with MLX's stable top-k; -0 counts as +0. A row
# that sees at most TOPK complete blocks takes blocks 0 to TOPK - 1.
_TOP_BLOCKS = """
  constexpr int NT = 256;
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
  const int nb = scores_shape[1];
  const int complete = (first[0] + row) / RATIO;
  int* o = sel + size_t(row) * TOPK;
  if (complete <= TOPK) {
    for (int i = tid; i < TOPK; i += NT) o[i] = i;
    return;
  }
  const int n = complete < nb ? complete : nb;
  const float* s = scores + size_t(row) * nb;
  auto key = [](float x) -> unsigned {
    const unsigned u = __float_as_uint(x == 0.f ? 0.f : x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  };
  __shared__ unsigned hist[256], scan_a[NT], scan_e[NT];
  __shared__ unsigned cut[2];
  unsigned prefix = 0, mask = 0, need = TOPK;
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    hist[tid] = 0;
    __syncthreads();
    // Eight loads in flight a thread; the row stays in L2 between passes.
    for (int base = 0; base < n; base += 8 * NT) {
      unsigned k8[8];
      #pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int i = base + u * NT + tid;
        k8[u] = i < n ? key(s[i]) : 0u;
      }
      #pragma unroll
      for (int u = 0; u < 8; ++u) {
        const bool in = base + u * NT + tid < n && (k8[u] & mask) == prefix;
        const unsigned bin = in ? (k8[u] >> shift) & 255u : 256u;
        const unsigned peers = __match_any_sync(0xffffffffu, bin);
        if (in && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], __popc(peers));
      }
    }
    __syncthreads();
    if (tid == 0) {
      unsigned more = 0;
      int b = 255;
      while (more + hist[b] < need) more += hist[b--];
      cut[0] = prefix | (unsigned(b) << shift);
      cut[1] = need - more;
    }
    __syncthreads();
    prefix = cut[0];
    need = cut[1];
    mask |= 255u << shift;
  }
  // A thread owns a contiguous run of blocks, so the output keeps key order.
  const int seg = (n + NT - 1) / NT, a = tid * seg, z = a + seg < n ? a + seg : n;
  unsigned above = 0, equal = 0;
  for (int i0 = a; i0 < z; i0 += 8) {
    float v8[8];
    #pragma unroll
    for (int u = 0; u < 8; ++u) v8[u] = i0 + u < z ? s[i0 + u] : 0.f;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const unsigned k = key(v8[u]);
      above += i0 + u < z && k > prefix;
      equal += i0 + u < z && k == prefix;
    }
  }
  scan_a[tid] = above;
  scan_e[tid] = equal;
  __syncthreads();
  for (int step = 1; step < NT; step <<= 1) {
    const unsigned pa = tid >= step ? scan_a[tid - step] : 0u;
    const unsigned pe = tid >= step ? scan_e[tid - step] : 0u;
    __syncthreads();
    scan_a[tid] += pa;
    scan_e[tid] += pe;
    __syncthreads();
  }
  // Of the blocks equal to the cut, the first skip ones are left out.
  const unsigned skip = scan_e[NT - 1] - need;
  unsigned rank = scan_e[tid] - equal;
  unsigned at = scan_a[tid] - above + (rank > skip ? rank - skip : 0u);
  for (int i0 = a; i0 < z; i0 += 8) {
    float v8[8];
    #pragma unroll
    for (int u = 0; u < 8; ++u) v8[u] = i0 + u < z ? s[i0 + u] : 0.f;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      const unsigned k = key(v8[u]);
      if (i0 + u >= z) break;
      if (k > prefix) {
        o[at++] = i0 + u;
      } else if (k == prefix) {
        if (rank >= skip) o[at++] = i0 + u;
        ++rank;
      }
    }
  }
"""


# Launch blocks that share a row tile's block tiles.
_SCORE_WALKERS = 8


class BlockSelection:
    """``Attention._select`` for rows past the dense prefix, as two kernels."""

    def __init__(self, heads: int, dim: int, ratio: int, top_k: int):
        self.heads, self.dim, self.top_k = heads, dim, top_k
        self.template = [("HI", heads), ("DI", dim), ("RATIO", ratio)]
        self.scores = mx.fast.cuda_kernel(
            name=f"qwen4_qsa_block_scores_{heads}_{dim}_{ratio}",
            input_names=["iq", "pooled", "first"],
            output_names=["scores"],
            source=_BLOCK_SCORES.replace("SQRT_DI", f"{float(dim) ** 0.5!r}f"),
            header=_borrowed("qsa_header.cuh"),
            shared_memory=(heads * 16 + 2 * 64) * (dim + 8) * 2,
        )
        self.top = mx.fast.cuda_kernel(
            name=f"qwen4_qsa_top_blocks_{ratio}_{top_k}",
            input_names=["scores", "first"],
            output_names=["sel"],
            source=_TOP_BLOCKS,
            header=_HEADER,
        )

    @staticmethod
    def fits(heads: int, dim: int, dtype: mx.Dtype) -> bool:
        """Whether the query tile and two block tiles fit 96 KiB of shared memory."""
        tiles = (heads * 16 + 2 * 64) * (dim + 8) * 2
        return dtype == mx.bfloat16 and dim % 32 == 0 and tiles <= 96 * 1024

    def block_scores(
        self, index_queries: mx.array, pooled: mx.array, first_visible: int
    ) -> mx.array:
        """Masked scores [rows, blocks] of ``index_queries`` [rows, heads, dim]."""
        rows, blocks = index_queries.shape[0], pooled.shape[1]
        first = mx.array([first_visible], dtype=mx.int32)
        return self.scores(
            inputs=[index_queries, pooled[0], first],
            template=self.template,
            grid=(min(-(-blocks // 64), _SCORE_WALKERS) * 128, -(-rows // 16), 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(rows, blocks)],
            output_dtypes=[mx.float32],
        )[0]

    def __call__(
        self, index_queries: mx.array, pooled: mx.array, first_visible: int
    ) -> mx.array:
        """Selected blocks [rows, top_k] int32 in key order."""
        scores = self.block_scores(index_queries, pooled, first_visible)
        first = mx.array([first_visible], dtype=mx.int32)
        return self.top(
            inputs=[scores, first],
            template=self.template[2:] + [("TOPK", self.top_k)],
            grid=(scores.shape[0] * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(scores.shape[0], self.top_k)],
            output_dtypes=[mx.int32],
        )[0]


# GDN prework of one (row, 128-channel head) a warp, 4 channels a lane: the
# causal conv in float32 rounded once, SiLU and, for the query and key heads,
# the L2 norm with ``ops.row_sum``'s bf16 lane sums, every op through MLX's
# functors as the plain ops run them.
_GDN_PREWORK = """
  using namespace mlx::core::cu;
  constexpr int DIM = 128;
  const int lane = threadIdx.x & 31;
  const size_t unit = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int rows = qkv_shape[1], channels = qkv_shape[2], units = channels / DIM;
  const size_t row = unit / units;
  if (row >= size_t(rows)) return;
  const int head = int(unit % units), ch = head * DIM + lane * 4;
  float acc[4];
  #pragma unroll
  for (int tap = 0; tap < TAPS; ++tap) {
    // Row row + tap of the conv input: the window rows, then the chunk.
    const long src = long(row) + tap - (TAPS - 1);
    const __nv_bfloat16* xp = src < 0 ? state + (src + TAPS - 1) * channels + ch
                                      : qkv + src * channels + ch;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float term = __fmul_rn(__bfloat162float(xp[e]), taps[(ch + e) * TAPS + tap]);
      acc[e] = tap == 0 ? term : __fadd_rn(acc[e], term);
    }
  }
  __nv_bfloat16 act[4];
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    const __nv_bfloat16 c = __float2bfloat16_rn(acc[e]);
    act[e] = Multiply{}(c, Sigmoid{}(c));
  }
  const int section = head < KH ? 0 : (head < 2 * KH ? 1 : 2);
  if (section < 2) {
    __nv_bfloat16 part = Square{}(act[0]);
    #pragma unroll
    for (int e = 1; e < 4; ++e) part = Add{}(part, Square{}(act[e]));
    float total = __bfloat162float(part);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(0xffffffffu, total, o);
    const __nv_bfloat16 inv = Rsqrt{}(Add{}(__float2bfloat16_rn(total), __float2bfloat16_rn(1e-6f)));
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      act[e] = Multiply{}(act[e], inv);
      // The query also takes 1 / sqrt(128).
      if (section == 0) act[e] = Multiply{}(act[e], __float2bfloat16_rn(0.08838834764831845f));
    }
  }
  const int at = section == 0 ? head : (section == 1 ? head - KH : head - 2 * KH);
  const int heads = section == 2 ? units - 2 * KH : KH;
  __nv_bfloat16* dst = (section == 0 ? q : (section == 1 ? k : v)) + (row * heads + at) * DIM + lane * 4;
  #pragma unroll
  for (int e = 0; e < 4; ++e) dst[e] = act[e];
"""

# The recurrence output's RMS norm, its weight and the output gate of one
# (row, head) a warp, as ``GatedDeltaNet.finish``: the norm rounds to bf16
# before its weight and the gate product is float32.
_GDN_FINISH = """
  using namespace mlx::core::cu;
  constexpr int DIM = 128;
  const int lane = threadIdx.x & 31;
  const size_t unit = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  if (unit >= size_t(y_shape[1]) * y_shape[2]) return;
  const size_t at = unit * DIM + lane * 4;
  float f[4], ss = 0.f;
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    f[e] = __bfloat162float(y[at + e]);
    ss = __fadd_rn(ss, __fmul_rn(f[e], f[e]));
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = Rsqrt{}(__fadd_rn(__fmul_rn(ss, 1.0f / DIM), eps[0]));
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    const __nv_bfloat16 t = Multiply{}(__float2bfloat16_rn(__fmul_rn(f[e], inv)), w[lane * 4 + e]);
    const float zf = __bfloat162float(z[at + e]);
    const float g = SIGMOID ? Sigmoid{}(zf) : __fmul_rn(zf, Sigmoid{}(zf));
    out[at + e] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(t), g));
  }
"""


def gdn_glue_fits(gdn) -> bool:
    """Whether the prework and finish kernels cover a GDN block's shapes."""
    return (
        gdn.key_dim == gdn.value_dim == 128
        and gdn.qkv.weight.dtype == gdn.norm_weight.dtype == mx.bfloat16
    )


def gdn_prework(
    qkv: mx.array, state: mx.array, taps: mx.array, key_heads: int
) -> tuple[mx.array, mx.array, mx.array]:
    """(q, k, v) [1, rows, heads, 128] of ``GatedDeltaNet.prework`` in one launch.

    ``state`` [1, taps - 1, C] is the conv window before the chunk.
    """
    rows, channels = qkv.shape[1], qkv.shape[2]
    value_heads = channels // 128 - 2 * key_heads
    threads = rows * (channels // 128) * 32
    return _functor_kernel(
        "gdn_prework", ["qkv", "state", "taps"], ["q", "k", "v"], _GDN_PREWORK
    )(
        inputs=[qkv, state, taps],
        template=[("TAPS", taps.shape[1]), ("KH", key_heads)],
        grid=(-(-threads // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (1, rows, key_heads, 128),
            (1, rows, key_heads, 128),
            (1, rows, value_heads, 128),
        ],
        output_dtypes=[qkv.dtype] * 3,
    )


def gdn_finish(
    y: mx.array, z: mx.array, weight: mx.array, eps: float, sigmoid: bool
) -> mx.array:
    """``GatedDeltaNet.finish`` of y [1, rows, heads, 128], [1, rows, heads * 128]."""
    rows, heads = y.shape[1], y.shape[2]
    threads = rows * heads * 32
    out = _functor_kernel("gdn_finish", ["y", "z", "w", "eps"], ["out"], _GDN_FINISH)(
        inputs=[y, z, weight, mx.array([eps], dtype=mx.float32)],
        template=[("SIGMOID", sigmoid)],
        grid=(-(-threads // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[y.shape],
        output_dtypes=[weight.dtype],
    )[0]
    return out.reshape(1, rows, -1)


_GDN_KERNEL: list = []
# Value rows of the state one launch block holds, and steps it stages at once.
_GDN_ROWS, _GDN_STEPS = 64, 32


def gdn_fits(q: mx.array, v: mx.array, gamma: mx.array, beta: mx.array) -> bool:
    """Whether ``gated_delta`` covers these inputs: batch one, 128-wide heads."""
    return (
        q.shape[0] == 1
        and q.shape[-1] == v.shape[-1] == 128
        and q.dtype == v.dtype == beta.dtype == mx.bfloat16
        and gamma.dtype == mx.float32
        and gamma.ndim == 3
        and v.shape[2] % q.shape[2] == 0
    )


def gated_delta(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    gamma: mx.array,
    beta: mx.array,
    state: mx.array | None,
) -> tuple[mx.array, mx.array]:
    """``mx.fast.gated_delta_update`` with the steps staged in shared memory.

    Two threads hold a value row of the state and the next steps' inputs load
    while the current ones run, so a step waits on no global load.
    """
    if not _GDN_KERNEL:
        _GDN_KERNEL.append(
            mx.fast.cuda_kernel(
                name="qwen4_gdn_chain",
                input_names=["q", "k", "v", "g", "beta", "state"],
                output_names=["y", "last"],
                source=_borrowed("gdn_chain.cu"),
                header=_borrowed("gdn_header.cuh"),
            )
        )
    heads = v.shape[2]
    if state is None:
        state = mx.zeros((1, heads, 128, 128), dtype=mx.float32)
    y, last = _GDN_KERNEL[0](
        inputs=[q, k, v, gamma, beta, state.astype(mx.float32)],
        template=[("ROWS", _GDN_ROWS), ("STEPS", _GDN_STEPS)],
        grid=(heads * 2 * _GDN_ROWS, 128 // _GDN_ROWS, 1),
        threadgroup=(2 * _GDN_ROWS, 1, 1),
        output_shapes=[v.shape, (1, heads, 128, 128)],
        output_dtypes=[v.dtype, mx.float32],
    )
    return y, last


_DEPTHWISE_CONV = """
  const size_t elem = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t rows = x_shape[1] - (TAPS - 1) * DILATION;
  const size_t channels = x_shape[2];
  if (elem >= rows * channels) return;
  const size_t t = elem / channels;
  const size_t c = elem % channels;
  // Multiply and add separately: a fused multiply-add would round differently.
  float acc = __fmul_rn(to_float(x[t * channels + c]), w[c * TAPS]);
  for (int i = 1; i < TAPS; ++i) {
    const float tap = to_float(x[(t + size_t(i) * DILATION) * channels + c]);
    acc = __fadd_rn(acc, __fmul_rn(tap, w[c * TAPS + i]));
  }
  from_float(acc, y + elem);
"""
_CONV_KERNEL: list = []


def depthwise_conv(x: mx.array, taps: mx.array, dilation: int) -> mx.array:
    """``ops.depthwise_conv`` in one launch, with the same float32 rounding."""
    if not _CONV_KERNEL:
        _CONV_KERNEL.append(
            mx.fast.cuda_kernel(
                name="qwen4_depthwise_conv",
                input_names=["x", "w"],
                output_names=["y"],
                source=_DEPTHWISE_CONV,
                header=_HEADER,
            )
        )
    rows = x.shape[1] - (taps.shape[1] - 1) * dilation
    channels = x.shape[2]
    threads = -(-rows * channels // 256) * 256
    return _CONV_KERNEL[0](
        inputs=[x, taps],
        template=[("TAPS", taps.shape[1]), ("DILATION", dilation)],
        grid=(threads, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, rows, channels)],
        output_dtypes=[x.dtype],
    )[0]


_SQUARE_SUMS = """
  const size_t width = x_shape[2] / STREAMS;
  const auto* xp = x + size_t(blockIdx.x) * width;
  float acc = 0.0f;
  for (size_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float value = to_float(xp[i]);
    acc = fmaf(value, value, acc);
  }
  extern __shared__ char smem[];
  float* part = reinterpret_cast<float*>(smem);
  part[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) part[threadIdx.x] += part[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) sums[blockIdx.x] = part[0];
"""
_SQUARE_SUMS_KERNEL: list = []


def stream_square_sums(x: mx.array, streams: int) -> mx.array:
    """Float32 sum of squares of each residual stream: [1, rows, streams, 1]."""
    if not _SQUARE_SUMS_KERNEL:
        _SQUARE_SUMS_KERNEL.append(
            mx.fast.cuda_kernel(
                name="qwen4_stream_square_sums",
                input_names=["x"],
                output_names=["sums"],
                source=_SQUARE_SUMS,
                header=_HEADER,
                shared_memory=256 * 4,
            )
        )
    rows = x.shape[1]
    return _SQUARE_SUMS_KERNEL[0](
        inputs=[x],
        template=[("STREAMS", streams)],
        grid=(rows * streams * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, rows, streams, 1)],
        output_dtypes=[mx.float32],
    )[0]


# Normalized residual streams and their float32 products with the inject
# weights, one 256-thread launch block a row. Each stream's sum of squares
# adds in ``stream_square_sums``'s order and its inverse RMS goes through
# MLX's functors as in ``HyperConnection``; each value is
# bf16((x * inverse) * scale) as ``ops.stream_norm`` forms it, and the
# products of the rounded values are summed in float32.
_HC_NORM = """
  using namespace mlx::core::cu;
  constexpr int NT = 256;
  const int row = blockIdx.x, tid = threadIdx.x;
  const int width = x_shape[2], hidden = width / S;
  const __nv_bfloat16* xr = x + size_t(row) * width;
  __nv_bfloat16* yr = y + size_t(row) * width;
  __shared__ float part[S][NT];
  __shared__ float inverse[S];
  #pragma unroll
  for (int t = 0; t < S; ++t) {
    float acc = 0.f;
    for (int i = tid; i < hidden; i += NT) {
      const float value = __bfloat162float(xr[t * hidden + i]);
      acc = fmaf(value, value, acc);
    }
    part[t][tid] = acc;
  }
  __syncthreads();
  for (int stride = NT / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      #pragma unroll
      for (int t = 0; t < S; ++t) part[t][tid] += part[t][tid + stride];
    }
    __syncthreads();
  }
  if (tid < S) {
    inverse[tid] = Rsqrt{}(Add{}(Divide{}(part[tid][0], float(hidden)), eps[0]));
  }
  __syncthreads();
  float dots[S];
  #pragma unroll
  for (int t = 0; t < S; ++t) dots[t] = 0.f;
  for (int j = tid * 8; j < width; j += NT * 8) {
    const float inv = inverse[j / hidden];
    const uint4 raw = *reinterpret_cast<const uint4*>(xr + j);
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&raw);
    uint4 packed;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&packed);
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
      ob[e] = __float2bfloat16_rn(
          __fmul_rn(__fmul_rn(__bfloat162float(xb[e]), inv), scale[j + e]));
      const float v = __bfloat162float(ob[e]);
      #pragma unroll
      for (int t = 0; t < S; ++t) dots[t] = fmaf(v, inject[size_t(t) * width + j + e], dots[t]);
    }
    *reinterpret_cast<uint4*>(yr + j) = packed;
  }
  const int lane = tid & 31, warp = tid >> 5;
  #pragma unroll
  for (int t = 0; t < S; ++t) {
    float v = dots[t];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) part[t][warp] = v;
  }
  __syncthreads();
  if (tid < S) {
    float total = 0.f;
    for (int w = 0; w < NT / 32; ++w) total += part[tid][w];
    dots_out[size_t(row) * S + tid] = total;
  }
"""

# The block input of one hyper-connection: sigmoid(up) in float32 rounded to
# bf16, times each stream, the streams added one by one and scaled by 1 / S,
# every step in bf16 as ``layers._stream_mixer`` does. A thread per 8 dims.
_HC_MIX = """
  using namespace mlx::core::cu;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int width = up_shape[2], hidden = width / S;
  const size_t row = i / (hidden / 8);
  const int d = int(i % (hidden / 8)) * 8;
  if (row >= size_t(up_shape[1])) return;
  const __nv_bfloat16 share = __float2bfloat16_rn(1.0f / S);
  __nv_bfloat16 acc[8];
  #pragma unroll
  for (int s = 0; s < S; ++s) {
    const size_t at = row * width + size_t(s) * hidden + d;
    const uint4 ru = *reinterpret_cast<const uint4*>(up + at);
    const uint4 rn = *reinterpret_cast<const uint4*>(normed + at);
    const __nv_bfloat16* ub = reinterpret_cast<const __nv_bfloat16*>(&ru);
    const __nv_bfloat16* nb = reinterpret_cast<const __nv_bfloat16*>(&rn);
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
      const __nv_bfloat16 gate = __float2bfloat16_rn(Sigmoid{}(__bfloat162float(ub[e])));
      const __nv_bfloat16 term = Multiply{}(gate, nb[e]);
      acc[e] = s == 0 ? term : Add{}(acc[e], term);
    }
  }
  uint4 packed;
  __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&packed);
  #pragma unroll
  for (int e = 0; e < 8; ++e) ob[e] = Multiply{}(acc[e], share);
  *reinterpret_cast<uint4*>(out + row * hidden + d) = packed;
"""
# The residual write h + branch * gate of every stream, in bf16 like
# ``layers.write``. A thread per 8 dims of one row, every stream.
_HC_WRITE = """
  using namespace mlx::core::cu;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int hidden = branch_shape[2];
  const size_t row = i / (hidden / 8);
  const int d = int(i % (hidden / 8)) * 8;
  if (row >= size_t(branch_shape[1])) return;
  const uint4 rb = *reinterpret_cast<const uint4*>(branch + row * hidden + d);
  const __nv_bfloat16* bb = reinterpret_cast<const __nv_bfloat16*>(&rb);
  #pragma unroll
  for (int s = 0; s < S; ++s) {
    const __nv_bfloat16 gate = gates[row * S + s];
    const size_t at = (row * S + s) * hidden + d;
    const uint4 rh = *reinterpret_cast<const uint4*>(h + at);
    const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&rh);
    uint4 packed;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&packed);
    #pragma unroll
    for (int e = 0; e < 8; ++e) ob[e] = Add{}(hb[e], Multiply{}(bb[e], gate));
    *reinterpret_cast<uint4*>(out + at) = packed;
  }
"""


def hc_fits(streams: int, hidden: int, dtype: mx.Dtype) -> bool:
    return dtype == mx.bfloat16 and hidden % 8 == 0 and streams <= 8


def hc_norm(
    x: mx.array, scale: mx.array, inject: mx.array, eps: float
) -> tuple[mx.array, mx.array]:
    """(normed streams [1, L, width] bf16, float32 inject products [1, L, S]).

    ``scale`` [S, hidden] and ``inject`` [S, width] are float32.
    """
    streams, rows = scale.shape[0], x.shape[1]
    normed, dots = _functor_kernel(
        "hc_norm", ["x", "scale", "inject", "eps"], ["y", "dots_out"], _HC_NORM
    )(
        inputs=[x, scale, inject, mx.array([eps], dtype=mx.float32)],
        template=[("S", streams)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (1, rows, streams)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return normed, dots


def hc_mix(up: mx.array, normed: mx.array, streams: int) -> mx.array:
    """The block input [1, L, hidden] from the up projection and the streams."""
    rows, width = up.shape[1], up.shape[2]
    threads = rows * (width // streams // 8)
    return _functor_kernel("hc_mix", ["up", "normed"], ["out"], _HC_MIX)(
        inputs=[up, normed],
        template=[("S", streams)],
        grid=(-(-threads // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, rows, width // streams)],
        output_dtypes=[up.dtype],
    )[0]


def hc_write(h: mx.array, branch: mx.array, gates: mx.array) -> mx.array:
    """``layers.write`` in one launch: h [1, L, S * hidden], gates [1, L, S]."""
    streams, (rows, hidden) = gates.shape[-1], branch.shape[1:]
    threads = rows * (hidden // 8)
    return _functor_kernel("hc_write", ["h", "branch", "gates"], ["out"], _HC_WRITE)(
        inputs=[h, branch, gates],
        template=[("S", streams)],
        grid=(-(-threads // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[h.shape],
        output_dtypes=[h.dtype],
    )[0]


# silu(gate) * up of each routed row, gate and up the two halves of its
# gate/up output, with MLX's bf16 functors as ``nn.silu(gate) * up`` runs.
_SWIGLU = """
  using namespace mlx::core::cu;
  const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  const size_t inner = both_shape[2] / 2;
  if (i >= both_shape[0] * inner) return;
  const size_t row = i / inner, d = i % inner;
  const uint4 rg = *reinterpret_cast<const uint4*>(both + row * 2 * inner + d);
  const uint4 ru = *reinterpret_cast<const uint4*>(both + row * 2 * inner + inner + d);
  const __nv_bfloat16* gb = reinterpret_cast<const __nv_bfloat16*>(&rg);
  const __nv_bfloat16* ub = reinterpret_cast<const __nv_bfloat16*>(&ru);
  uint4 packed;
  __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&packed);
  #pragma unroll
  for (int e = 0; e < 8; ++e) ob[e] = Multiply{}(Multiply{}(gb[e], Sigmoid{}(gb[e])), ub[e]);
  *reinterpret_cast<uint4*>(out + i) = packed;
"""


def swiglu(both: mx.array) -> mx.array:
    """``nn.silu(gate) * up`` of [n, 1, 2 * inner] in one launch: [n, 1, inner]."""
    rows, inner = both.shape[0], both.shape[2] // 2
    threads = rows * inner // 8
    return _functor_kernel("swiglu", ["both"], ["out"], _SWIGLU)(
        inputs=[both],
        template=[],
        grid=(-(-threads // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, 1, inner)],
        output_dtypes=[both.dtype],
    )[0]


# Row r of the output is row src[r] of x, 16 bytes a thread.
_GATHER_ROWS = """
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t chunks = x_shape[1] * sizeof(T) / 16;
  if (i >= src_shape[0] * chunks) return;
  const size_t r = i / chunks, c = i % chunks;
  reinterpret_cast<uint4*>(y)[i] =
      reinterpret_cast<const uint4*>(x)[size_t(src[r]) * chunks + c];
"""
_GATHER_KERNEL: list = []


def gather_rows(x: mx.array, src: mx.array) -> mx.array:
    """``x[src]`` for x [n, width] with 16-byte rows; src [m] uint32."""
    if not _GATHER_KERNEL:
        _GATHER_KERNEL.append(
            mx.fast.cuda_kernel(
                name="qwen4_gather_rows",
                input_names=["x", "src"],
                output_names=["y"],
                source=_GATHER_ROWS,
            )
        )
    rows, width = src.shape[0], x.shape[1]
    threads = rows * (width * x.dtype.size // 16)
    return _GATHER_KERNEL[0](
        inputs=[x, src],
        template=[("T", x.dtype)],
        grid=(-(-threads // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[x.dtype],
    )[0]


# Each token's expert outputs weighted and added in the order ops.stacked_sum
# adds them, rounding to the output dtype after every product and sum.
_COMBINE = """
  const size_t elem = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t width = y_shape[2];
  const size_t token = elem / width;
  if (token >= w_shape[0]) return;
  const size_t d = elem % width;
  float part[K];
  for (int i = 0; i < K; ++i) {
    const float value = to_float(y[size_t(pos[token * K + i]) * width + d]);
    part[i] = rounded(value * to_float(w[token * K + i]), out);
  }
  const int groups = K < 8 ? K : 8;
  for (int first = 0; first < groups; ++first) {
    for (int row = first + 8; row < K; row += 8) {
      part[first] = rounded(part[row] + part[first], out);
    }
  }
  float total = part[0];
  for (int i = 1; i < groups; ++i) total = rounded(part[i] + total, out);
  from_float(total, out + elem);
"""
_COMBINE_KERNEL: list = []


def combine_experts(y: mx.array, positions: mx.array, weights: mx.array) -> mx.array:
    """Weighted sum of each token's expert rows: [tokens, width].

    ``y`` [n, 1, width] holds one row per (token, expert) in any order,
    ``positions`` [tokens * k] the row of each token's i-th expert and
    ``weights`` [tokens, k] its weight.
    """
    if not _COMBINE_KERNEL:
        _COMBINE_KERNEL.append(
            mx.fast.cuda_kernel(
                name="qwen4_combine_experts",
                input_names=["y", "pos", "w"],
                output_names=["out"],
                source=_COMBINE,
                header=_HEADER,
            )
        )
    tokens, k = weights.shape
    width = y.shape[-1]
    threads = -(-tokens * width // 256) * 256
    return _COMBINE_KERNEL[0](
        inputs=[y, positions.astype(mx.uint32), weights],
        template=[("K", k)],
        grid=(threads, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tokens, width)],
        output_dtypes=[y.dtype],
    )[0]
