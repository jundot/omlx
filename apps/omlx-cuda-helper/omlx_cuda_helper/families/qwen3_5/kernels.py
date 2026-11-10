# SPDX-License-Identifier: Apache-2.0
"""CUDA kernels for Qwen3.5 prefill.

A block takes a kernel only after ``enable_kernels`` matched it with the
block's plain-ops reference.
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


_CAUSAL_WEIGHTS = """
  const int tid = threadIdx.x;
  const int total = s_shape[1];
  const int visible = geometry[0] + blockIdx.x % geometry[1];
  const float* sr = s + size_t(blockIdx.x) * total;
  float* wr = w + size_t(blockIdx.x) * total;
  extern __shared__ char smem[];
  float* part = reinterpret_cast<float*>(smem);

  float top = -3.4028234664e38f;
  for (int j = tid; j < visible; j += blockDim.x) top = fmaxf(top, sr[j]);
  part[tid] = top;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) part[tid] = fmaxf(part[tid], part[tid + stride]);
    __syncthreads();
  }
  top = part[0];
  __syncthreads();

  float acc = 0.0f;
  for (int j = tid; j < total; j += blockDim.x) {
    const float e = j < visible ? expf(sr[j] - top) : 0.0f;
    wr[j] = e;
    acc += e;
  }
  part[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) part[tid] += part[tid + stride];
    __syncthreads();
  }
  if (tid == 0) z[blockIdx.x] = part[0];
"""
_CAUSAL_WEIGHTS_KERNEL: list = []
_THREADS = 256


def causal_weights(scores: mx.array, first: int, rows: int) -> tuple[mx.array, ...]:
    """Unnormalized causal softmax of float32 scores [n, total] in one launch.

    Row ``i`` attends to the first ``first + i % rows`` keys. Returns the
    weights [n, total] (zero on hidden keys) and their row sums [n, 1].
    """
    if not _CAUSAL_WEIGHTS_KERNEL:
        _CAUSAL_WEIGHTS_KERNEL.append(
            mx.fast.cuda_kernel(
                name="qwen35_causal_weights",
                input_names=["s", "geometry"],
                output_names=["w", "z"],
                source=_CAUSAL_WEIGHTS,
                shared_memory=_THREADS * 4,
            )
        )
    n = scores.shape[0]
    weights, norm = _CAUSAL_WEIGHTS_KERNEL[0](
        inputs=[scores, mx.array([first, rows], dtype=mx.int32)],
        grid=(n * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[scores.shape, (n, 1)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return weights, norm


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
"""

_CAUSAL_CONV = """
  const size_t elem = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t rows = x_shape[1] - (TAPS - 1);
  const size_t channels = x_shape[2];
  if (elem >= rows * channels) return;
  const size_t t = elem / channels;
  const size_t c = elem % channels;
  // Multiply and add separately: a fused multiply-add would round differently.
  float acc = __fmul_rn(to_float(x[t * channels + c]), w[c * TAPS]);
  for (int i = 1; i < TAPS; ++i) {
    const float tap = to_float(x[(t + i) * channels + c]);
    acc = __fadd_rn(acc, __fmul_rn(tap, w[c * TAPS + i]));
  }
  from_float(acc, y + elem);
"""
_CAUSAL_CONV_KERNELS: dict = {}


def causal_conv(x: mx.array, taps: mx.array) -> mx.array:
    """``layers.causal_conv`` in one launch, with the same float32 rounding."""
    # A kernel object compiles for the dtypes of its first call only.
    kernel = _CAUSAL_CONV_KERNELS.get(x.dtype)
    if kernel is None:
        kernel = _CAUSAL_CONV_KERNELS[x.dtype] = mx.fast.cuda_kernel(
            name=f"qwen35_causal_conv_{str(x.dtype).rsplit('.', 1)[-1]}",
            input_names=["x", "w"],
            output_names=["y"],
            source=_CAUSAL_CONV,
            header=_HEADER,
        )
    rows = x.shape[1] - taps.shape[1] + 1
    channels = x.shape[2]
    threads = -(-rows * channels // _THREADS) * _THREADS
    return kernel(
        inputs=[x, taps],
        template=[("TAPS", taps.shape[1])],
        grid=(threads, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(1, rows, channels)],
        output_dtypes=[x.dtype],
    )[0]
