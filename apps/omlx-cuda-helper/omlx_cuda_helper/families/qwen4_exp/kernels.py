# SPDX-License-Identifier: Apache-2.0
"""CUDA kernels for Qwen3.8 Flash-Next prefill.

A block takes a kernel only after ``enable_kernels`` matched it with the
block's plain-ops reference on the block's own geometry.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


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

# The row a launch block serves and the token id of its j-th attended key:
# the selected blocks in key order, then the incomplete tail. A row with at
# most TOPK complete blocks takes them all.
_ROW = """
  const int row = blockIdx.x / KV;
  const int kvh = blockIdx.x % KV;
  const int tid = threadIdx.x;
  const int n_tok = CACHE_shape[1];
  const int vis = first[0] + row;
  const int complete = vis / RATIO;
  const int count = complete < TOPK ? complete : TOPK;
  const int n_valid = count * RATIO + (vis - complete * RATIO);
  const int* picks = sel + size_t(row) * TOPK;
  auto token = [&](int j) {
    if (j < count * RATIO) return picks[j / RATIO] * RATIO + j % RATIO;
    return j < n_valid ? complete * RATIO + (j - count * RATIO) : -1;
  };
"""

# Float32 scores [rows, KV, G, W]; one warp per query head, one key per lane.
# Keys pass through shared memory 32 at a time as [dim][key] with row stride
# 33, so neither the fill nor the dot products share a bank. The tiles keep
# the input dtype so four blocks fit on one multiprocessor.
_SCORES = _ROW.replace("CACHE", "k") + """
  extern __shared__ char smem[];
  int* stok = reinterpret_cast<int*>(smem);
  T* sq = reinterpret_cast<T*>(smem + 32 * sizeof(int));
  T* sk = sq + G * D;
  for (int i = tid; i < G * D; i += blockDim.x) {
    const int head = kvh * G + i / D;
    sq[i] = q[(size_t(head) * q_shape[1] + row) * D + i % D];
  }
  const int lane = tid & 31;
  const T* qh = sq + (tid >> 5) * D;
  float* out = scores + ((size_t(row) * KV + kvh) * G + (tid >> 5)) * W;
  for (int base = 0; base < W; base += 32) {
    if (tid < 32) stok[tid] = token(base + tid);
    __syncthreads();
    for (int e = tid; e < 32 * D; e += blockDim.x) {
      const int tok = stok[e / D];
      const int d = e % D;
      if (tok >= 0) sk[d * 33 + e / D] = k[(size_t(kvh) * n_tok + tok) * D + d];
    }
    __syncthreads();
    float acc = 0.0f;
    for (int d = 0; d < D; ++d) {
      acc = fmaf(to_float(qh[d]), to_float(sk[d * 33 + lane]), acc);
    }
    const int j = base + lane;
    if (j < W) out[j] = stok[lane] >= 0 ? acc / SQRT_D : -3.4028234664e38f;
    __syncthreads();
  }
"""

# Softmax-weighted value sums, one thread per head dim. The weights of 64
# keys sit in shared memory, where all lanes of a warp read the same word.
_FOLD = _ROW.replace("CACHE", "v") + """
  extern __shared__ char smem[];
  int* stok = reinterpret_cast<int*>(smem);
  float* sp = reinterpret_cast<float*>(smem + 64 * sizeof(int));
  const float* pr = p + (size_t(row) * KV + kvh) * G * W;
  float acc[G];
  for (int h = 0; h < G; ++h) acc[h] = 0.0f;
  for (int base = 0; base < n_valid; base += 64) {
    const int keys = n_valid - base < 64 ? n_valid - base : 64;
    for (int j = tid; j < keys; j += blockDim.x) stok[j] = token(base + j);
    for (int e = tid; e < G * 64; e += blockDim.x) {
      if (e % 64 < keys) sp[e] = pr[size_t(e / 64) * W + base + e % 64];
    }
    __syncthreads();
    for (int j = 0; j < keys; ++j) {
      const float vd = to_float(v[(size_t(kvh) * n_tok + stok[j]) * D + tid]);
      for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h * 64 + j], vd, acc[h]);
    }
    __syncthreads();
  }
  for (int h = 0; h < G; ++h) {
    from_float(acc[h], out + (size_t(row) * KV * G + kvh * G + h) * D + tid);
  }
"""


class SparseAttention:
    """Attention of contiguous query rows over their QSA-selected keys.

    Keys and values are read in place by token id, not gathered. Scores,
    softmax and value sums are float32; the output rounds once.
    """

    # Query rows of one launch pair; bounds the float32 score sheet.
    rows_per_launch = 2048

    def __init__(
        self,
        heads: int,
        kv_heads: int,
        dim: int,
        ratio: int,
        top_k: int,
        dtype: mx.Dtype,
    ):
        self.heads, self.kv_heads, self.dim = heads, kv_heads, dim
        self.width = top_k * ratio + ratio - 1
        self.template = [
            ("KV", kv_heads),
            ("G", heads // kv_heads),
            ("D", dim),
            ("RATIO", ratio),
            ("TOPK", top_k),
            ("W", self.width),
        ]
        name = f"qwen4_qsa_{heads}_{kv_heads}_{dim}_{ratio}_{top_k}"
        self.scores = mx.fast.cuda_kernel(
            name=name + "_scores",
            input_names=["q", "k", "sel", "first"],
            output_names=["scores"],
            source=_SCORES.replace("SQRT_D", f"{float(dim) ** 0.5!r}f"),
            header=_HEADER,
            shared_memory=32 * 4 + (heads // kv_heads + 33) * dim * dtype.size,
        )
        self.fold = mx.fast.cuda_kernel(
            name=name + "_fold",
            input_names=["p", "v", "sel", "first"],
            output_names=["out"],
            source=_FOLD,
            header=_HEADER,
            shared_memory=64 * 4 + heads // kv_heads * 64 * 4,
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
        groups = self.heads // self.kv_heads
        first = mx.array([first_visible], dtype=mx.int32)
        blocks = rows * self.kv_heads
        scores = self.scores(
            inputs=[queries[0], keys[0], selected, first],
            template=[("T", queries.dtype), *self.template],
            grid=(blocks * groups * 32, 1, 1),
            threadgroup=(groups * 32, 1, 1),
            output_shapes=[(rows, self.kv_heads, groups, self.width)],
            output_dtypes=[mx.float32],
        )[0]
        out = self.fold(
            inputs=[mx.softmax(scores, axis=-1), values[0], selected, first],
            template=self.template,
            grid=(blocks * self.dim, 1, 1),
            threadgroup=(self.dim, 1, 1),
            output_shapes=[(rows, self.heads, self.dim)],
            output_dtypes=[queries.dtype],
        )[0]
        return out[None]


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
