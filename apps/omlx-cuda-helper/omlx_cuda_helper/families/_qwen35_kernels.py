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


def attention_matches(got: mx.array, want: mx.array) -> bool:
    """Whether fused attention agrees with the plain-ops reference.

    cuDNN's fused attention moves 16-bit outputs by about 3e-3 relative to
    the float32 reference; a wrong mask or layout moves them far more.
    """
    a, b = got.astype(mx.float32), want.astype(mx.float32)
    error = (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-30)).item()
    if error <= 1e-2:
        return True
    logger.warning(
        "Fused attention differs from its reference (relative error %.2e); "
        "using the reference",
        error,
    )
    return False


_THREADS = 256

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
