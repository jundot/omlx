# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm kernels for the Muse Spark decoder layer.

A Muse Spark layer runs six RMSNorms and two residual gates around its
attention and MoE blocks. The FP32 reference norms split into several small
kernels each, which dominates the non-matmul part of decode. These kernels do
the same math in one launch per group:

* ``scaled_rms``: ``T((x * inv) * w)`` in FP32, ``w`` or ``T(1 + w)``.
* ``post_attention``: scaleless norm of the attention output, the residual
  gate, then the centered pre-feedforward norm.
* ``moe_tail``: the input-scaled norm of every expert row, then the FP32
  routing-weighted sum in expert order.
* ``post_moe``: the input-scaled post-feedforward norm and the residual gate.

Every value is rounded to the activation dtype where the reference rounds it,
and rows are read like MLX's RMSNorm kernels (4 values per thread, looping
past 4096 values). The reductions and the residual gate can still round
differently from the composed reference ops, by one unit in the last place.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

_MAX_THREADS = 1024

# One threadgroup per row; every thread returns the row sum.
_HEADER = r"""
inline float muse_row_sum(
    float acc,
    threadgroup float* local_sums,
    uint simd_lane_id,
    uint simd_group_id) {
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = simd_sum(local_sums[simd_lane_id]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

template <typename W>
inline float muse_weight(const device W* w, uint i, bool centered) {
  if (centered) {
    return static_cast<float>(static_cast<W>(1.0f + static_cast<float>(w[i])));
  }
  return static_cast<float>(w[i]);
}
"""

# Thread lid holds the values at loop * STRIDE + lid * 4 + i (MLX rms_looped).
_PROLOGUE = r"""
  constexpr int N_READS = 4 * LOOPS;
  threadgroup float local_sums[32];
  uint row = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint simd_lane_id = thread_index_in_simdgroup;
  uint simd_group_id = simdgroup_index_in_threadgroup;
  const size_t base = size_t(row) * AXIS;
  uint col[N_READS];
  for (int i = 0; i < N_READS; i++) {
    col[i] = (i / 4) * STRIDE + lid * 4 + (i % 4);
  }
"""

_SCALED_RMS_SOURCE = _PROLOGUE + r"""
  const float eps = eps_in[0];
  float v[N_READS];
  float acc = 0;
  for (int i = 0; i < N_READS; i++) {
    v[i] = col[i] < AXIS ? static_cast<float>(x[base + col[i]]) : 0.0f;
    acc += v[i] * v[i];
  }
  float sum = muse_row_sum(acc, local_sums, simd_lane_id, simd_group_id);
  float inv = metal::precise::rsqrt(sum / AXIS + eps);
  for (int i = 0; i < N_READS; i++) {
    if (col[i] < AXIS) {
      out[base + col[i]] = static_cast<T>(
          (v[i] * inv) * muse_weight(w, col[i], CENTERED));
    }
  }
"""

_POST_ATTENTION_SOURCE = _PROLOGUE + r"""
  const float eps_a = eps_in[0];
  const float eps_n = eps_in[1];
  float v[N_READS];
  float acc = 0;
  for (int i = 0; i < N_READS; i++) {
    v[i] = col[i] < AXIS ? static_cast<float>(a[base + col[i]]) : 0.0f;
    acc += v[i] * v[i];
  }
  float sum = muse_row_sum(acc, local_sums, simd_lane_id, simd_group_id);
  float inv = metal::precise::rsqrt(sum / AXIS + eps_a);
  acc = 0;
  for (int i = 0; i < N_READS; i++) {
    if (col[i] < AXIS) {
      uint c = col[i];
      float h = static_cast<float>(static_cast<T>(v[i] * inv));
      T xv = static_cast<T>(alpha[c] * static_cast<float>(x[base + col[i]]) + beta[c] * h);
      x_out[base + col[i]] = xv;
      v[i] = static_cast<float>(xv);
    } else {
      v[i] = 0.0f;
    }
    acc += v[i] * v[i];
  }
  sum = muse_row_sum(acc, local_sums, simd_lane_id, simd_group_id);
  inv = metal::precise::rsqrt(sum / AXIS + eps_n);
  for (int i = 0; i < N_READS; i++) {
    if (col[i] < AXIS) {
      n_out[base + col[i]] = static_cast<T>((v[i] * inv) * muse_weight(w, col[i], true));
    }
  }
"""

_MOE_TAIL_SOURCE = _PROLOGUE + r"""
  const float eps = eps_in[0];
  float wv[N_READS];
  float total[N_READS];
  for (int i = 0; i < N_READS; i++) {
    wv[i] = col[i] < AXIS ? muse_weight(w, col[i], true) : 0.0f;
    total[i] = 0.0f;
  }
  for (int k = 0; k < TOPK; k++) {
    const size_t ybase = (size_t(row) * TOPK + k) * AXIS;
    float v[N_READS];
    float acc = 0;
    for (int i = 0; i < N_READS; i++) {
      v[i] = col[i] < AXIS
          ? static_cast<float>(static_cast<T>(static_cast<float>(y[ybase + col[i]]) * wv[i]))
          : 0.0f;
      acc += v[i] * v[i];
    }
    float sum = muse_row_sum(acc, local_sums, simd_lane_id, simd_group_id);
    float inv = metal::precise::rsqrt(sum / AXIS + eps);
    float s = scores[size_t(row) * TOPK + k];
    for (int i = 0; i < N_READS; i++) {
      float p = static_cast<float>(static_cast<T>(v[i] * inv)) * s;
      total[i] = p + total[i];
    }
  }
  for (int i = 0; i < N_READS; i++) {
    if (col[i] < AXIS) {
      out[base + col[i]] = static_cast<T>(total[i]);
    }
  }
"""

_POST_MOE_SOURCE = _PROLOGUE + r"""
  const float eps = eps_in[0];
  float v[N_READS];
  float acc = 0;
  for (int i = 0; i < N_READS; i++) {
    v[i] = col[i] < AXIS
        ? static_cast<float>(static_cast<T>(
              static_cast<float>(m[base + col[i]]) * muse_weight(w, col[i], true)))
        : 0.0f;
    acc += v[i] * v[i];
  }
  float sum = muse_row_sum(acc, local_sums, simd_lane_id, simd_group_id);
  float inv = metal::precise::rsqrt(sum / AXIS + eps);
  for (int i = 0; i < N_READS; i++) {
    if (col[i] < AXIS) {
      uint c = col[i];
      float h = static_cast<float>(static_cast<T>(v[i] * inv));
      x_out[base + col[i]] =
          static_cast<T>(alpha[c] * static_cast<float>(x[base + col[i]]) + beta[c] * h);
    }
  }
"""


@lru_cache(maxsize=None)
def _kernel(name, inputs, outputs, source):
    return mx.fast.metal_kernel(
        name=f"omlx_muse_spark_{name}",
        input_names=list(inputs),
        output_names=list(outputs),
        source=source,
        header=_HEADER,
    )


def _layout(axis: int) -> tuple[int, int]:
    """(loops, threads) for one row of ``axis`` values, as MLX's RMSNorm."""
    threads = min(_MAX_THREADS, 32 * -(-axis // (4 * 32)))
    return -(-axis // (4 * threads)), threads


@lru_cache(maxsize=None)
def _eps(*values: float):
    return mx.array(values, dtype=mx.float32)


def _launch(name, inputs, outputs, source, arrays, shape, dtype, template):
    axis = int(shape[-1])
    rows = 1
    for size in shape[:-1]:
        rows *= int(size)
    loops, threads = _layout(axis)
    return _kernel(name, inputs, outputs, source)(
        inputs=arrays,
        template=[
            ("T", dtype),
            ("AXIS", axis),
            ("LOOPS", loops),
            ("STRIDE", 4 * threads),
            *template,
        ],
        grid=(threads * rows, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[shape] * len(outputs),
        output_dtypes=[dtype] * len(outputs),
    )


def scaled_rms(x, w, eps: float, centered: bool):
    """``T((x * inv) * w)`` in FP32, with ``w`` or ``T(1 + w)``."""
    (out,) = _launch(
        "scaled_rms",
        ("x", "w", "eps_in"),
        ("out",),
        _SCALED_RMS_SOURCE,
        [x, w, _eps(float(eps))],
        x.shape,
        x.dtype,
        [("W", w.dtype), ("CENTERED", bool(centered))],
    )
    return out


def post_attention(x, attn, alpha, beta, w, attn_eps: float, norm_eps: float):
    """Residual gate of the normed attention output, then the centered norm.

    Returns ``(x, norm(x))``.
    """
    return _launch(
        "post_attention",
        ("x", "a", "alpha", "beta", "w", "eps_in"),
        ("x_out", "n_out"),
        _POST_ATTENTION_SOURCE,
        [x, attn, alpha, beta, w, _eps(float(attn_eps), float(norm_eps))],
        x.shape,
        x.dtype,
        [("W", w.dtype)],
    )


def moe_tail(y, scores, w, eps: float):
    """Input-scaled norm of every expert row, then the weighted FP32 sum.

    ``y`` is ``(..., K, D)`` and ``scores`` ``(..., K)``; returns ``(..., D)``.
    """
    (out,) = _launch(
        "moe_tail",
        ("y", "scores", "w", "eps_in"),
        ("out",),
        _MOE_TAIL_SOURCE,
        [y, scores.astype(mx.float32), w, _eps(float(eps))],
        y.shape[:-2] + y.shape[-1:],
        y.dtype,
        [("W", w.dtype), ("TOPK", int(y.shape[-2]))],
    )
    return out


def post_moe(x, mlp_out, alpha, beta, w, eps: float):
    """Input-scaled norm of the MoE output, then the residual gate."""
    (out,) = _launch(
        "post_moe",
        ("x", "m", "alpha", "beta", "w", "eps_in"),
        ("x_out",),
        _POST_MOE_SOURCE,
        [x, mlp_out, alpha, beta, w, _eps(float(eps))],
        x.shape,
        x.dtype,
        [("W", w.dtype)],
    )
    return out
