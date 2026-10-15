# SPDX-License-Identifier: Apache-2.0
"""Decode / short-verify routed-expert kernels for MXFP4 experts (MiMo V2).

For a forward of a few token rows, MLX runs every (token, expert) pair of a
``SwitchGLU`` as its own ``gather_qmv``: gate and up are two passes over the
same activations, SwiGLU is a third kernel, and an expert picked by several
rows of a verify forward is streamed from memory once per row.  The kernels
here compute exactly the same numbers with fewer passes:

* ``gate_up_swiglu``: one kernel for the gate and up mat-vecs of all pairs,
  with the SwiGLU epilogue, and one weight pass per distinct expert (every
  row that picked it reuses the loaded weights).
* ``down_proj``: the down mat-vec of all pairs, one weight pass per distinct
  expert.

Bit-exactness: each output row is the reduction of MLX's
``fp_qmv_fast_impl`` (``fp_quantized.h``): lane ``l`` of a simdgroup owns the
16 inputs ``[512 b + 16 l, 512 b + 16 l + 16)`` of every 512-wide block ``b``,
accumulates ``scale * sum_i(((x0 w0 + x1 w1) + x2 w2) + x3 w3)`` block by
block, and the lanes are combined with ``simd_sum``; the result is rounded to
the activation dtype once.  MXFP4 products are exact in float32 (bf16/fp16
inputs times 2-bit-mantissa weights) and the e8m0 scale is a power of two
(exact for normal results), so the same additions in the same order give the same bits whatever the
compiler contracts.  The SwiGLU epilogue is MLX's compiled
``silu(gate) * up`` in the activation dtype, op for op.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

BLOCK = 512  # K values per simdgroup step (16 per lane)

_HEADER = r"""
// MXFP4 e2m1 nibble -> float exactly like MLX's fp4_e2m1 (via half).
inline float omlx_fp4(uint v) {
  uint b = v & 0xF;
  half c = as_type<half>(ushort((b & 7) << 9));
  c *= 16384.0;
  return static_cast<float>((b & 8) ? -c : c);
}

// e8m0 scale -> float like MLX's fp8_e8m0.
inline float omlx_e8m0(uint8_t s) {
  uint out = (s == 0 ? 0x400000u : (static_cast<uint>(s) << 23));
  return as_type<float>(out);
}

// MLX fp_quantized.h qdot<float, 16, 4> on 8 packed bytes (4 x uint16).
inline float omlx_qdot16(uint2 w, thread const float* x, float scale) {
  ushort ws[4] = {ushort(w.x & 0xFFFF), ushort(w.x >> 16), ushort(w.y & 0xFFFF), ushort(w.y >> 16)};
  float accum = 0;
  for (int i = 0; i < 4; i++) {
    accum +=
        (x[4 * i] * omlx_fp4(ws[i]) + x[4 * i + 1] * omlx_fp4(ws[i] >> 4) +
         x[4 * i + 2] * omlx_fp4(ws[i] >> 8) +
         x[4 * i + 3] * omlx_fp4(ws[i] >> 12));
  }
  return scale * accum;
}

// MLX unary_ops.h Sigmoid, applied in the activation dtype.
struct OmlxSigmoid {
  template <typename U>
  U operator()(U x) thread {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
"""

# Pairs p = row * TOPK + k use expert inds[p].  The threadgroup of the first
# pair that picked an expert computes every pair that picked it; later pairs'
# threadgroups exit.  Grid: x = pair, y = block of NSG * RPS output rows.
_GATE_UP_SOURCE = r"""
  constexpr int VPT = 16;
  uint pid = threadgroup_position_in_grid.x;
  uint rb = threadgroup_position_in_grid.y;
  uint sgid = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;
  uint e = inds[pid];
  for (uint q = 0; q < pid; q++) {
    if (inds[q] == e) {
      return;
    }
  }
  uint members[MAXDUP];
  int nm = 0;
  for (uint q = pid; q < NPAIRS && nm < MAXDUP; q++) {
    if (inds[q] == e) {
      members[nm++] = q;
    }
  }
  const int n0 = (rb * NSG + sgid) * RPS;
  constexpr int KW = KDIM / 8;   // uint32 per weight row
  constexpr int KS = KDIM / 32;  // scales per row
  const device uint32_t* gw = wg + (size_t(e) * GROWS + n0) * KW + lane * 2;
  const device uint8_t* gs = sg + (size_t(e) * GROWS + n0) * KS + lane / 2;
  const device uint32_t* uw = wu + (size_t(e) * UROWS + UOFF + n0) * KW + lane * 2;
  const device uint8_t* us = su + (size_t(e) * UROWS + UOFF + n0) * KS + lane / 2;

  float rg[MAXDUP][RPS];
  float ru[MAXDUP][RPS];
  for (int j = 0; j < MAXDUP; j++) {
    for (int r = 0; r < RPS; r++) {
      rg[j][r] = 0;
      ru[j][r] = 0;
    }
  }
  for (int b = 0; b < KDIM / 512; b++) {
    uint2 wgv[RPS];
    uint2 wuv[RPS];
    float sgv[RPS];
    float suv[RPS];
    for (int r = 0; r < RPS; r++) {
      wgv[r] = *(const device uint2*)(gw + r * KW + b * 64);
      wuv[r] = *(const device uint2*)(uw + r * KW + b * 64);
      sgv[r] = omlx_e8m0(gs[r * KS + b * 16]);
      suv[r] = omlx_e8m0(us[r * KS + b * 16]);
    }
    for (int j = 0; j < MAXDUP; j++) {
      if (j < nm) {
        const device T* xp = x + size_t(members[j] / TOPK) * KDIM + b * 512 + lane * VPT;
        float xt[VPT];
        for (int i = 0; i < VPT; i++) {
          xt[i] = static_cast<float>(xp[i]);
        }
        for (int r = 0; r < RPS; r++) {
          rg[j][r] += omlx_qdot16(wgv[r], xt, sgv[r]);
          ru[j][r] += omlx_qdot16(wuv[r], xt, suv[r]);
        }
      }
    }
  }
  for (int j = 0; j < MAXDUP; j++) {
    if (j < nm) {
      for (int r = 0; r < RPS; r++) {
        float g = simd_sum(rg[j][r]);
        float u = simd_sum(ru[j][r]);
        if (lane == 0) {
          T gt = static_cast<T>(g);
          T ut = static_cast<T>(u);
          T sig = OmlxSigmoid{}(gt);
          T silu = gt * sig;
          act[size_t(members[j]) * NOUT + n0 + r] = silu * ut;
        }
      }
    }
  }
"""

_DOWN_SOURCE = r"""
  constexpr int VPT = 16;
  uint pid = threadgroup_position_in_grid.x;
  uint rb = threadgroup_position_in_grid.y;
  uint sgid = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;
  uint e = inds[pid];
  for (uint q = 0; q < pid; q++) {
    if (inds[q] == e) {
      return;
    }
  }
  uint members[MAXDUP];
  int nm = 0;
  for (uint q = pid; q < NPAIRS && nm < MAXDUP; q++) {
    if (inds[q] == e) {
      members[nm++] = q;
    }
  }
  const int n0 = (rb * NSG + sgid) * RPS;
  constexpr int KW = KDIM / 8;
  constexpr int KS = KDIM / 32;
  const device uint32_t* dw = w + (size_t(e) * NOUT + n0) * KW + lane * 2;
  const device uint8_t* ds = s + (size_t(e) * NOUT + n0) * KS + lane / 2;

  float acc[MAXDUP][RPS];
  for (int j = 0; j < MAXDUP; j++) {
    for (int r = 0; r < RPS; r++) {
      acc[j][r] = 0;
    }
  }
  for (int b = 0; b < KDIM / 512; b++) {
    uint2 wv[RPS];
    float sv[RPS];
    for (int r = 0; r < RPS; r++) {
      wv[r] = *(const device uint2*)(dw + r * KW + b * 64);
      sv[r] = omlx_e8m0(ds[r * KS + b * 16]);
    }
    for (int j = 0; j < MAXDUP; j++) {
      if (j < nm) {
        const device T* xp = a + size_t(members[j]) * KDIM + b * 512 + lane * VPT;
        float xt[VPT];
        for (int i = 0; i < VPT; i++) {
          xt[i] = static_cast<float>(xp[i]);
        }
        for (int r = 0; r < RPS; r++) {
          acc[j][r] += omlx_qdot16(wv[r], xt, sv[r]);
        }
      }
    }
  }
  for (int j = 0; j < MAXDUP; j++) {
    if (j < nm) {
      for (int r = 0; r < RPS; r++) {
        float v = simd_sum(acc[j][r]);
        if (lane == 0) {
          y[size_t(members[j]) * NOUT + n0 + r] = static_cast<T>(v);
        }
      }
    }
  }
"""


@lru_cache(maxsize=None)
def _gate_up_kernel():
    return mx.fast.metal_kernel(
        name="omlx_mxfp4_gate_up_swiglu",
        input_names=["x", "inds", "wg", "sg", "wu", "su"],
        output_names=["act"],
        header=_HEADER,
        source=_GATE_UP_SOURCE,
    )


@lru_cache(maxsize=None)
def _down_kernel():
    return mx.fast.metal_kernel(
        name="omlx_mxfp4_down",
        input_names=["a", "inds", "w", "s"],
        output_names=["y"],
        header=_HEADER,
        source=_DOWN_SOURCE,
    )


def supported(linear, k: int, n: int) -> bool:
    """A ``QuantizedSwitchLinear`` these kernels handle (MXFP4, gs 32)."""
    return (
        getattr(linear, "mode", None) == "mxfp4"
        and getattr(linear, "bits", None) == 4
        and getattr(linear, "group_size", None) == 32
        and linear.get("biases") is None
        and linear.get("bias") is None
        and k % BLOCK == 0
        and linear.weight.dtype == mx.uint32
        and linear.scales.dtype == mx.uint8
    )


def gate_up_swiglu(x, inds, wg, sg, wu, su, *, n_out, up_offset=0, nsg=2, rps=4):
    """``swiglu(gate(x), up(x))`` per (row, expert) pair, shape ``(*inds.shape, n_out)``.

    ``x (..., K)`` rows matching ``inds (..., TOPK)``; ``wg``/``wu`` packed
    MXFP4 ``(E, rows, K / 8)`` with up rows starting at ``up_offset`` (a
    fused ``[gate; up]`` tensor passes itself twice with ``up_offset=n_out``).
    """
    K = int(x.shape[-1])
    topk = int(inds.shape[-1])
    pairs = int(inds.size)
    rows = pairs // topk
    assert n_out % (nsg * rps) == 0
    (act,) = _gate_up_kernel()(
        inputs=[x, inds, wg, sg, wu, su],
        template=[
            ("T", x.dtype),
            ("KDIM", K),
            ("NOUT", int(n_out)),
            ("GROWS", int(wg.shape[1])),
            ("UROWS", int(wu.shape[1])),
            ("UOFF", int(up_offset)),
            ("TOPK", topk),
            ("NPAIRS", pairs),
            ("MAXDUP", rows),
            ("NSG", nsg),
            ("RPS", rps),
        ],
        grid=(32 * pairs, nsg * (n_out // (nsg * rps)), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(*inds.shape, n_out)],
        output_dtypes=[x.dtype],
    )
    return act


def down_proj(a, inds, w, s, *, nsg=2, rps=4):
    """Per-pair down projection: ``a (..., TOPK, K)`` -> ``(..., TOPK, N)``."""
    K = int(a.shape[-1])
    topk = int(inds.shape[-1])
    pairs = int(inds.size)
    rows = pairs // topk
    n_out = int(w.shape[1])
    assert n_out % (nsg * rps) == 0
    (y,) = _down_kernel()(
        inputs=[a, inds, w, s],
        template=[
            ("T", a.dtype),
            ("KDIM", K),
            ("NOUT", n_out),
            ("TOPK", topk),
            ("NPAIRS", pairs),
            ("MAXDUP", rows),
            ("NSG", nsg),
            ("RPS", rps),
        ],
        grid=(32 * pairs, nsg * (n_out // (nsg * rps)), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(*inds.shape, n_out)],
        output_dtypes=[a.dtype],
    )
    return y
