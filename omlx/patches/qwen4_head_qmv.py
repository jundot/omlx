# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp vocabulary head: stock ``quantized_matmul`` bits, fewer weight passes.

The head is the largest single projection of a decode step (248320 x 2560,
8-bit: 644 MiB). Two launches replace stock ones without changing a bit:

* One row: stock ``quantized_matmul`` runs ``qmv_fast`` with 8 output columns
  per threadgroup. ``row_exact_qmv.OneRowQmv`` runs the same tile with 2
  columns per threadgroup (each column's arithmetic is ``qmv_fast``'s), so
  more threadgroups stream the weights.
* Several rows: on GPUs where MLX 0.32 dispatches affine ``qmv_wide`` (Apple
  GPU family 15 and newer, rows below the ``qmv`` batch limit), each
  threadgroup reuses every decoded weight group for at most five rows, so 6-8
  rows stream the head twice and 11 rows three times. A row's arithmetic in
  ``qmv_wide`` does not depend on how many rows share its threadgroup (each
  row accumulates its own lane-strided groups and 8-value sub-chunks, then
  the same shuffle-down ladder), so the kernel below -- ``qmv_wide_impl``
  transcribed from MLX 0.32.4 -- serves up to 8 rows per weight pass with the
  stock bits.

Every multi-row shape is checked once against stock ``quantized_matmul`` on
this GPU before it serves; a shape that differs keeps the stock call.
"""

from __future__ import annotations

import logging
import re

import mlx.core as mx
import mlx.nn as nn

from .row_exact_qmv import one_row_qmv

logger = logging.getLogger(__name__)

# Rows per threadgroup of the single-pass kernel: past 8 the per-row
# accumulators and input pointers spill (11 rows in one pass ran 2.6x slower
# than two passes on M3 Ultra).
_MAX_ROWS_PER_PASS = 8
# Stock qmv_wide's cap (MLX 0.32.4 ``qmv_wide``: ceil(M / 5) tiles).
_STOCK_ROWS_PER_PASS = 5

_HEADER = r"""
using namespace metal;

// MLX 0.32.4 quantized.h ``dequantize`` (affine; 4/5/6/8 bits), verbatim.
template <typename U, int N, int bits, typename W>
inline void wide_dequantize(const device uint8_t* w, U scale, U bias, W w_local) {
  const float s = float(scale);
  const float b = float(bias);
  if (bits == 4) {
    float sc[2] = {s, s / 16.0f};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = static_cast<U>(sc[0] * (w[i] & 0x0f) + b);
      w_local[2 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0xf0) + b);
    }
  } else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;
      w_local[0] = static_cast<U>((w[0] & 0x1f) * s + b);
      w_local[1] =
          static_cast<U>((((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * s + b);
      w_local[2] = static_cast<U>(((w[1] & 0x7c) >> 2) * s + b);
      w_local[3] =
          static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * s + b);
      w_local[4] =
          static_cast<U>((((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * s + b);
      w_local[5] = static_cast<U>(((w[3] & 0x3e) >> 1) * s + b);
      w_local[6] =
          static_cast<U>((((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * s + b);
      w_local[7] = static_cast<U>(((w[4] & 0xf8) >> 3) * s + b);
    }
  } else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = static_cast<U>((w[0] & 0x3f) * s + b);
      w_local[1] =
          static_cast<U>((((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * s + b);
      w_local[2] =
          static_cast<U>((((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * s + b);
      w_local[3] = static_cast<U>(((w[2] >> 2) & 0x3f) * s + b);
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = static_cast<U>(s * w[i] + b);
    }
  }
}
"""

# MLX 0.32.4 ``qmv_wide_impl`` (affine, k_lanes = 8, two simdgroups) with the
# sizes and the per-threadgroup row count as template parameters.
_SOURCE = r"""
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 32 / KL;
  constexpr int sub = 8;
  typedef float U;
  const uint3 tid = threadgroup_position_in_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const short k_lane = simd_lid % KL;
  const short sg_row = simd_lid / KL;
  const int out_row = tid.y * (results_per_simdgroup * num_simdgroups) +
      results_per_simdgroup * simd_gid + sg_row;
  const int vec0 = tid.x * NV;
  const int row = min(out_row, N_OUT - 1);
  const int in_vec_size_w = K_IN * BITS / 8;
  const int in_vec_size_g = K_IN / GS;
  const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
  const device T* srow = scales + row * in_vec_size_g;
  const device T* brow = biases + row * in_vec_size_g;
  const device T* xv[NV];
  for (int v = 0; v < NV; v++) {
    xv[v] = x + min(vec0 + v, M_ROWS - 1) * K_IN;
  }
  U result[NV] = {0};
  for (int g = k_lane; g < in_vec_size_g; g += KL) {
    U scale = srow[g];
    U bias = brow[g];
#pragma unroll
    for (int sc = 0; sc < GS / sub; sc++) {
      const int k0 = g * GS + sc * sub;
      const device uint8_t* wc = wrow + k0 * BITS / 8;
      U w_dq[sub];
      wide_dequantize<U, sub, BITS>(wc, scale, bias, w_dq);
#pragma unroll
      for (int v = 0; v < NV; v++) {
        const device T* xc = xv[v] + k0;
        U acc = 0;
#pragma unroll
        for (int i = 0; i < sub; i++) {
          acc += static_cast<U>(xc[i]) * w_dq[i];
        }
        result[v] += acc;
      }
    }
  }
  for (int v = 0; v < NV; v++) {
    if (KL >= 32) result[v] += simd_shuffle_down(result[v], 16);
    if (KL >= 16) result[v] += simd_shuffle_down(result[v], 8);
    if (KL >= 8) result[v] += simd_shuffle_down(result[v], 4);
    if (KL >= 4) result[v] += simd_shuffle_down(result[v], 2);
    if (KL >= 2) result[v] += simd_shuffle_down(result[v], 1);
  }
  if (k_lane == 0 && out_row < N_OUT) {
    for (int v = 0; v < NV; v++) {
      if (vec0 + v < M_ROWS) {
        y[(vec0 + v) * N_OUT + out_row] = static_cast<T>(result[v]);
      }
    }
  }
"""

_KERNEL = None
_VERDICTS: dict[tuple, bool] = {}


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen4_head_qmv_wide",
            input_names=["w", "scales", "biases", "x"],
            output_names=["y"],
            header=_HEADER,
            source=_SOURCE,
        )
    return _KERNEL


def _gpu() -> tuple[int, str]:
    """(family generation, size letter) of an ``applegpu_g15d``-style name."""
    match = re.match(
        r"applegpu_g(\d+)([a-z])", str(mx.device_info().get("architecture", ""))
    )
    return (int(match.group(1)), match.group(2)) if match else (0, "")


def _qmv_batch_limit(k: int, n: int, gen: int, size: str) -> int:
    """MLX 0.32.4 ``get_qmv_batch_limit`` for the families that run qmv_wide."""
    small, medium = k <= 2048 and n <= 2048, k <= 4096 and n <= 4096
    if gen >= 17 and size != "d":
        return 33 if small else (25 if medium else 13)
    if gen >= 15 and size != "d":
        return 13 if small else (15 if medium else 13)
    if size == "d":
        return 32 if small else (18 if medium else 12)
    return 14 if small else (10 if medium else 6)


def stock_qmv_wide_passes(rows: int, k: int, n: int, bits: int, mode: str) -> int:
    """Weight passes stock ``quantized_matmul`` makes over ``rows`` affine rows
    with ``qmv_wide`` (MLX 0.32), or 0 when it runs another kernel."""
    gen, size = _gpu()
    if (
        not mx.__version__.startswith("0.32.")
        or mode != "affine"
        or gen < 15
        or rows < 2
        or rows >= _qmv_batch_limit(k, n, gen, size)
        or (k in (64, 128) and bits in (4, 8))
    ):
        return 0
    return -(-rows // _STOCK_ROWS_PER_PASS)


def _rows_per_pass(rows: int, k: int, n: int, bits: int, mode: str) -> int:
    """Rows per threadgroup of the fewer-pass launch, or 0 to keep stock."""
    stock = stock_qmv_wide_passes(rows, k, n, bits, mode)
    passes = -(-rows // _MAX_ROWS_PER_PASS)
    if not stock or passes >= stock:
        return 0
    return -(-rows // passes)


def _wide(linear, x2d: mx.array, per_pass: int) -> mx.array:
    rows, k = x2d.shape
    n = int(linear.weight.shape[0])
    return _kernel()(
        inputs=[linear.weight, linear.scales, linear.biases, x2d],
        template=[
            ("T", x2d.dtype),
            ("GS", int(linear.group_size)),
            ("BITS", int(linear.bits)),
            ("NV", per_pass),
            ("KL", 8),
            ("K_IN", k),
            ("N_OUT", n),
            ("M_ROWS", rows),
        ],
        grid=(32 * (-(-rows // per_pass)), 2 * (-(-n // 8)), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x2d.dtype],
    )[0]


def _verified(linear, rows: int, k: int, per_pass: int, dtype) -> bool:
    """Whether the fewer-pass launch reproduces stock ``quantized_matmul`` for
    this shape on this GPU (checked once per shape on a fixed probe)."""
    n = int(linear.weight.shape[0])
    key = (int(linear.bits), int(linear.group_size), k, n, rows, per_pass, dtype)
    verdict = _VERDICTS.get(key)
    if verdict is None:
        probe = (mx.random.normal((rows, k), key=mx.random.key(rows)) * 0.5).astype(
            dtype
        )
        want = mx.quantized_matmul(
            probe,
            linear.weight,
            scales=linear.scales,
            biases=linear.biases,
            transpose=True,
            group_size=linear.group_size,
            bits=linear.bits,
        )
        got = _wide(linear, probe, per_pass)
        mx.eval(want, got)
        verdict = _VERDICTS[key] = bool(mx.array_equal(want, got).item())
        if not verdict:
            logger.warning(
                "Qwen4 head qmv_wide launch %s differs from stock quantized_matmul "
                "on this GPU; it keeps the stock call",
                key,
            )
    return verdict


class _Qwen4HeadLinear(nn.QuantizedLinear):
    """The vocabulary head with the launches above (stock bits)."""

    def __call__(self, x):
        k = x.shape[-1]
        rows = x.size // k if k else 0
        mode = getattr(self, "mode", "affine")
        if (
            "bias" in self
            or self.get("biases") is None
            or x.dtype not in (mx.bfloat16, mx.float16)
            or self.scales.dtype != x.dtype
            or self.weight.ndim != 2
        ):
            return super().__call__(x)
        if rows == 1 and k not in (64, 128):  # stock routes K = 64/128 to qmv_quad
            plan = self.__dict__.get("_omlx_head_one_row")
            if plan is None or plan[0] is not self.weight or plan[1] != x.dtype:
                plan = (
                    self.weight,
                    x.dtype,
                    one_row_qmv(
                        self.weight,
                        self.scales,
                        self.biases,
                        int(self.bits),
                        int(self.group_size),
                        mode,
                        x.dtype,
                        1,
                    ),
                )
                self.__dict__["_omlx_head_one_row"] = plan
            if plan[2] is not None:
                return plan[2](x)
            return super().__call__(x)
        per_pass = _rows_per_pass(
            rows, k, int(self.weight.shape[0]), int(self.bits), mode
        )
        if per_pass and _verified(self, rows, k, per_pass, x.dtype):
            return _wide(self, x.reshape(rows, k), per_pass).reshape(*x.shape[:-1], -1)
        return super().__call__(x)


def apply_qwen4_head_qmv_patch(model) -> bool:
    """Route a loaded Qwen4-Exp language model's quantized head through
    ``_Qwen4HeadLinear``; False when there is none to route."""
    language = getattr(model, "language_model", model)
    if not type(language).__module__.startswith("mlx_vlm.models.qwen4_exp."):
        return False
    head = language.get("lm_head") if isinstance(language, nn.Module) else None
    if type(head) is not nn.QuantizedLinear:
        return False
    head.__class__ = _Qwen4HeadLinear
    return True


__all__ = ["apply_qwen4_head_qmv_patch", "stock_qmv_wide_passes"]
