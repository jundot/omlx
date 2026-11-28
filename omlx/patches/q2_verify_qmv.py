# SPDX-License-Identifier: Apache-2.0
#
# Ternary biases == -scales fast path and M2+ threadgroup geometry adapted from
# MTPLX (mtplx/kernels/ternary_qmv.py, Copyright 2026 Youssof Altoukhi,
# Apache-2.0, https://github.com/youssofal/mtplx).
#
# Copyright (c) 2026 David Dalcu. The half2 exponent-magic 2-bit decode in
# decode_w2 is adapted from ddalcu/mlx-serve's MIT-licensed
# src/kernels/quant/qmv2_decode.metal (https://github.com/ddalcu/mlx-serve).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from collections.abc import Callable, Sequence

import mlx.core as mx
import mlx.nn as nn

from ..utils.hardware import get_apple_gpu_generation

MIN_ROWS = 2
MAX_ROWS = 5
_BITS = 2
_VPT = 16
_BLOCK = _VPT * 32
_M1_NSG = 2
_M1_WIDE_N = 8192
_WIDE_TG_MIN_N = 4096
_HUGE_N_THRESHOLD = 65536
_MIN_N = 1024
_GROUP_SIZES = (32, 64, 128)

_KernelFn = Callable[..., list[mx.array]]
_SINGLE_KERNELS: dict[tuple[int, int, int, bool], _KernelFn] = {}
_GROUP_KERNELS: dict[tuple[int, int, int, tuple[int, ...], bool], _KernelFn] = {}
_SWIGLU_KERNELS: dict[tuple[int, int, int, bool], _KernelFn] = {}

_HEADER = """
#include <metal_simdgroup>

template <bool NEG_SCALE>
METAL_FUNC void decode_w2(uint32_t u, thread half2* q) {
  uint32_t u6 = u >> 6u;
  constexpr half off0 = NEG_SCALE ? 1025.0h : 1024.0h;
  constexpr half off1 = NEG_SCALE ? 257.0h : 256.0h;
  constexpr half off2 = NEG_SCALE ? 65.0h : 64.0h;
  constexpr half off3 = NEG_SCALE ? 17.0h : 16.0h;
  constexpr half off4 = NEG_SCALE ? 5.0h : 4.0h;
  q[0] = as_type<half2>((u & 0x00030003u) | 0x64006400u) - half2(off0);
  q[1] = as_type<half2>((u & 0x000C000Cu) | 0x5C005C00u) - half2(off1);
  q[2] = as_type<half2>((u & 0x00300030u) | 0x54005400u) - half2(off2);
  q[3] = as_type<half2>((u & 0x00C000C0u) | 0x4C004C00u) - half2(off3);
  q[4] = as_type<half2>((u & 0x03000300u) | 0x44004400u) - half2(off4);
  q[5] = as_type<half2>((u6 & 0x00300030u) | 0x54005400u) - half2(off2);
  q[6] = as_type<half2>((u6 & 0x00C000C0u) | 0x4C004C00u) - half2(off3);
  q[7] = as_type<half2>((u6 & 0x03000300u) | 0x44004400u) - half2(off4);
}
"""


def metal_ready() -> bool:
    return bool(mx.metal.is_available() and mx.default_device() == mx.gpu)


def mark_ternary(
    linear: nn.QuantizedLinear,
    *,
    is_ternary: bool | None = None,
) -> bool:
    """Record whether ``linear`` satisfies ``biases == -scales`` keyed by array identity."""
    biases = linear.biases
    scales = linear.scales
    if biases is None or biases.shape != scales.shape or biases.dtype != scales.dtype:
        object.__setattr__(linear, "_omlx_ternary_pair", None)
        return False
    flag = (
        bool(mx.array_equal(biases, -scales).item())
        if is_ternary is None
        else is_ternary
    )
    object.__setattr__(linear, "_omlx_ternary_pair", (scales, biases) if flag else None)
    return flag


def is_ternary_linear(linear: object) -> bool:
    """Return True when ``linear`` was proven ternary for its current ``(scales, biases)``."""
    if not isinstance(linear, nn.QuantizedLinear):
        return False
    pair = getattr(linear, "_omlx_ternary_pair", None)
    return (
        isinstance(pair, tuple)
        and len(pair) == 2
        and pair[0] is linear.scales
        and pair[1] is linear.biases
    )


def _geometry(rows: int, cols: int) -> tuple[int, int]:
    if get_apple_gpu_generation() == 13:
        rpsg = 2 if (rows == MAX_ROWS or cols < _M1_WIDE_N) else 4
        return rpsg, _M1_NSG
    nsg = 8 if (_WIDE_TG_MIN_N <= cols < _HUGE_N_THRESHOLD and cols % 16 == 0) else 4
    return 2, nsg


def _group_geometry(rows: int, out_dims: tuple[int, ...]) -> tuple[int, int]:
    total_cols = sum(out_dims)
    if get_apple_gpu_generation() == 13:
        rpsg = 2 if (rows == MAX_ROWS or total_cols < _M1_WIDE_N) else 4
        return rpsg, _M1_NSG
    can_nsg8 = _WIDE_TG_MIN_N <= total_cols < _HUGE_N_THRESHOLD and all(
        c % 16 == 0 for c in out_dims
    )
    return 2, (8 if can_nsg8 else 4)


def _swiglu_geometry(rows: int, cols: int) -> tuple[int, int]:
    if get_apple_gpu_generation() == 13:
        return (1 if rows == MAX_ROWS else 2), _M1_NSG
    return 1, (8 if cols >= _WIDE_TG_MIN_N else 4)


def eligible(
    rows: int,
    width: int,
    cols: int,
    bits: int,
    group_size: int,
    dtype: mx.Dtype,
) -> bool:
    return (
        int(bits) == _BITS
        and MIN_ROWS <= int(rows) <= MAX_ROWS
        and dtype in (mx.float16, mx.bfloat16)
        and int(group_size) in _GROUP_SIZES
        and int(width) % _BLOCK == 0
        and int(cols) % 8 == 0
        and int(cols) >= _MIN_N
    )


def group_eligible(linears: Sequence[object], x: mx.array) -> bool:
    if len(linears) < 2 or x.ndim != 3 or x.dtype not in (mx.float16, mx.bfloat16):
        return False
    rows = int(x.shape[0] * x.shape[1])
    width = int(x.shape[2])
    if not (MIN_ROWS <= rows <= MAX_ROWS and width % _BLOCK == 0):
        return False
    first = linears[0]
    if not isinstance(first, nn.QuantizedLinear):
        return False
    group_size = int(first.group_size)
    if group_size not in _GROUP_SIZES:
        return False
    max_cols = 0
    for item in linears:
        if not isinstance(item, nn.QuantizedLinear):
            return False
        cols = int(item.scales.shape[0])
        if (
            getattr(item, "mode", "affine") != "affine"
            or "bias" in item
            or int(item.bits) != _BITS
            or int(item.group_size) != group_size
            or int(item.weight.shape[1]) * _VPT != width
            or cols % 8 != 0
            or item.scales.dtype != x.dtype
            or item.biases is None
            or item.biases.dtype != x.dtype
            or item.scales.shape != (cols, width // group_size)
            or item.biases.shape != item.scales.shape
        ):
            return False
        if cols > max_cols:
            max_cols = cols
    return max_cols >= _MIN_N


def swiglu_eligible(gate: object, up: object, x: mx.array) -> bool:
    if not (
        isinstance(gate, nn.QuantizedLinear)
        and isinstance(up, nn.QuantizedLinear)
        and x.ndim == 3
        and getattr(gate, "mode", "affine") == "affine"
        and getattr(up, "mode", "affine") == "affine"
        and "bias" not in gate
        and "bias" not in up
        and gate.weight.shape == up.weight.shape
        and int(gate.bits) == _BITS
        and int(up.bits) == _BITS
        and int(gate.group_size) == int(up.group_size)
        and gate.scales.dtype == x.dtype
        and gate.biases is not None
        and gate.biases.dtype == x.dtype
        and up.scales.dtype == x.dtype
        and up.biases is not None
        and up.biases.dtype == x.dtype
    ):
        return False
    rows = int(x.shape[0] * x.shape[1])
    width = int(x.shape[2])
    cols = int(gate.scales.shape[0])
    group_size = int(gate.group_size)
    if (
        int(gate.weight.shape[1]) * _VPT != width
        or gate.scales.shape != (cols, width // group_size)
        or gate.biases.shape != gate.scales.shape
        or up.scales.shape != gate.scales.shape
        or up.biases.shape != gate.scales.shape
    ):
        return False
    return eligible(rows, width, cols, _BITS, group_size, x.dtype)


def _tile_body(rows: int, rpsg: int, nsg: int, neg_scale: bool) -> str:
    neg_lit = "true" if neg_scale else "false"
    bi_ptr = (
        "" if neg_scale else "    const device T* bi = bi_mat + long(out_row) * Kg;\n"
    )
    br_decl = "" if neg_scale else "      float br[RPSG];\n"
    br_load = "" if neg_scale else "        br[r] = float(bi[long(r) * Kg + g]);\n"
    if neg_scale:
        xsum_code = ""
        accum_line = "          result[m][r] += sr[r] * (a.x + a.y);"
    else:
        xsum_code = (
            "        float2 xsum2 = xf[0];\n"
            "        #pragma clang loop unroll(full)\n"
            "        for (int j = 1; j < 8; j++) {\n"
            "          xsum2 += xf[j];\n"
            "        }\n"
            "        float xsum = xsum2.x + xsum2.y;\n"
        )
        accum_line = "          result[m][r] += sr[r] * (a.x + a.y) + br[r] * xsum;"
    return f"""
    constexpr int M = {rows};
    constexpr int RPSG = {rpsg};
    constexpr int NSG = {nsg};

    const int K = x_shape[1];
    const int Kw = K / 16;
    const int Kg = K / GS;
    const uint simd_gid = thread_position_in_threadgroup.y;
    const uint simd_lid = thread_position_in_threadgroup.x;
    const int out_row = local_tile * (NSG * RPSG) + int(simd_gid) * RPSG;

    const device uint32_t* ws = w_mat + long(out_row) * Kw + int(simd_lid);
    const device T* sc = sc_mat + long(out_row) * Kg;
{bi_ptr}    const device T* xp = x + int(simd_lid) * 16;

    float result[M][RPSG];
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; m++) {{
      #pragma clang loop unroll(full)
      for (int r = 0; r < RPSG; r++) {{
        result[m][r] = 0.0f;
      }}
    }}

    for (int k = 0; k < K; k += 512) {{
      const int g = (k + int(simd_lid) * 16) / GS;
      const int kw = k / 16;
      half2 qw[RPSG][8];
      float sr[RPSG];
{br_decl}      #pragma clang loop unroll(full)
      for (int r = 0; r < RPSG; r++) {{
        decode_w2<{neg_lit}>(ws[long(r) * Kw + kw], qw[r]);
        sr[r] = float(sc[long(r) * Kg + g]);
{br_load}      }}
      #pragma clang loop unroll(full)
      for (int m = 0; m < M; m++) {{
        const device vec<T, 4>* xr = (const device vec<T, 4>*)(xp + long(m) * K + k);
        vec<T, 4> xl0 = xr[0], xl1 = xr[1], xh0 = xr[2], xh1 = xr[3];
        float2 xf[8] = {{
            float2(float(xl0.x), float(xh0.x)),
            float2(float(xl0.y), float(xh0.y)),
            float2(float(xl0.z), float(xh0.z)),
            float2(float(xl0.w), float(xh0.w)),
            float2(float(xl1.x), float(xh1.x)),
            float2(float(xl1.y), float(xh1.y)),
            float2(float(xl1.z), float(xh1.z)),
            float2(float(xl1.w), float(xh1.w)),
        }};
{xsum_code}        #pragma clang loop unroll(full)
        for (int r = 0; r < RPSG; r++) {{
          float2 a = float2(qw[r][0]) * xf[0];
          #pragma clang loop unroll(full)
          for (int j = 1; j < 8; j++) {{
            a = fma(float2(qw[r][j]), xf[j], a);
          }}
{accum_line}
        }}
      }}
    }}

    #pragma clang loop unroll(full)
    for (int m = 0; m < M; m++) {{
      #pragma clang loop unroll(full)
      for (int r = 0; r < RPSG; r++) {{
        float v = simd_sum(result[m][r]);
        if (simd_lid == 0) {{
          yp[long(m) * n_out + out_row + r] = static_cast<T>(v);
        }}
      }}
    }}
"""


def _single_source(rows: int, rpsg: int, nsg: int, neg_scale: bool) -> str:
    bi_line = "" if neg_scale else "    const device T* bi_mat = biases;\n"
    setup = (
        "    const device uint32_t* w_mat = w;\n"
        "    const device T* sc_mat = scales;\n"
        f"{bi_line}"
        "    device T* yp = y;\n"
        "    const int n_out = w_shape[0];\n"
        "    const int local_tile = int(threadgroup_position_in_grid.y);\n"
    )
    return setup + _tile_body(rows, rpsg, nsg, neg_scale)


def _group_source(
    rows: int,
    rpsg: int,
    nsg: int,
    out_dims: tuple[int, ...],
    neg_scale: bool,
) -> str:
    cols_per_tg = nsg * rpsg
    branches: list[str] = []
    end = 0
    for idx, cols in enumerate(out_dims):
        start = end
        end += cols // cols_per_tg
        head = "if" if idx == 0 else "} else if"
        bi_assign = "" if neg_scale else f"      bi_mat = b{idx};\n"
        branches.append(
            f"    {head} (tile < {end}) {{\n"
            f"      w_mat = w{idx};\n"
            f"      sc_mat = s{idx};\n"
            f"{bi_assign}"
            f"      yp = y{idx};\n"
            f"      n_out = {cols};\n"
            f"      local_tile = tile - {start};"
        )
    branches.append("    }")
    bi_decl = "" if neg_scale else "    const device T* bi_mat = nullptr;\n"
    setup = (
        "    const int tile = int(threadgroup_position_in_grid.y);\n"
        "    const device uint32_t* w_mat = nullptr;\n"
        "    const device T* sc_mat = nullptr;\n"
        f"{bi_decl}"
        "    device T* yp = nullptr;\n"
        "    int n_out = 0;\n"
        "    int local_tile = 0;\n" + "\n".join(branches)
    )
    return setup + _tile_body(rows, rpsg, nsg, neg_scale)


def _swiglu_source(rows: int, rpsg: int, nsg: int, neg_scale: bool) -> str:
    neg_lit = "true" if neg_scale else "false"
    bi_ptrs = (
        ""
        if neg_scale
        else (
            "    const device T* gbi = gate_biases + long(out_row) * Kg;\n"
            "    const device T* ubi = up_biases + long(out_row) * Kg;\n"
        )
    )
    bi_decl = "" if neg_scale else "      float gb[RPSG], ub[RPSG];\n"
    bi_loads = (
        ""
        if neg_scale
        else (
            "        gb[r] = float(gbi[long(r) * Kg + g]);\n"
            "        ub[r] = float(ubi[long(r) * Kg + g]);\n"
        )
    )
    if neg_scale:
        xsum_code = ""
        accum_lines = (
            "          g_res[m][r] += gs[r] * (ga.x + ga.y);\n"
            "          u_res[m][r] += us[r] * (ua.x + ua.y);"
        )
    else:
        xsum_code = (
            "        float2 xsum2 = xf[0];\n"
            "        #pragma clang loop unroll(full)\n"
            "        for (int j = 1; j < 8; j++) {\n"
            "          xsum2 += xf[j];\n"
            "        }\n"
            "        float xsum = xsum2.x + xsum2.y;\n"
        )
        accum_lines = (
            "          g_res[m][r] += gs[r] * (ga.x + ga.y) + gb[r] * xsum;\n"
            "          u_res[m][r] += us[r] * (ua.x + ua.y) + ub[r] * xsum;"
        )
    return f"""
    constexpr int M = {rows};
    constexpr int RPSG = {rpsg};
    constexpr int NSG = {nsg};

    const int K = x_shape[1];
    const int n_out = gate_w_shape[0];
    const int Kw = K / 16;
    const int Kg = K / GS;
    const uint simd_gid = thread_position_in_threadgroup.y;
    const uint simd_lid = thread_position_in_threadgroup.x;
    const int local_tile = int(threadgroup_position_in_grid.y);
    const int out_row = local_tile * (NSG * RPSG) + int(simd_gid) * RPSG;

    const device uint32_t* gws = gate_w + long(out_row) * Kw + int(simd_lid);
    const device T* gsc = gate_scales + long(out_row) * Kg;
    const device uint32_t* uws = up_w + long(out_row) * Kw + int(simd_lid);
    const device T* usc = up_scales + long(out_row) * Kg;
{bi_ptrs}    const device T* xp = x + int(simd_lid) * 16;

    float g_res[M][RPSG];
    float u_res[M][RPSG];
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; m++) {{
      #pragma clang loop unroll(full)
      for (int r = 0; r < RPSG; r++) {{
        g_res[m][r] = 0.0f;
        u_res[m][r] = 0.0f;
      }}
    }}

    for (int k = 0; k < K; k += 512) {{
      const int g = (k + int(simd_lid) * 16) / GS;
      const int kw = k / 16;
      half2 gw_tile[RPSG][8];
      half2 uw_tile[RPSG][8];
      float gs[RPSG], us[RPSG];
{bi_decl}      #pragma clang loop unroll(full)
      for (int r = 0; r < RPSG; r++) {{
        decode_w2<{neg_lit}>(gws[long(r) * Kw + kw], gw_tile[r]);
        decode_w2<{neg_lit}>(uws[long(r) * Kw + kw], uw_tile[r]);
        gs[r] = float(gsc[long(r) * Kg + g]);
        us[r] = float(usc[long(r) * Kg + g]);
{bi_loads}      }}
      #pragma clang loop unroll(full)
      for (int m = 0; m < M; m++) {{
        const device vec<T, 4>* xr = (const device vec<T, 4>*)(xp + long(m) * K + k);
        vec<T, 4> xl0 = xr[0], xl1 = xr[1], xh0 = xr[2], xh1 = xr[3];
        float2 xf[8] = {{
            float2(float(xl0.x), float(xh0.x)),
            float2(float(xl0.y), float(xh0.y)),
            float2(float(xl0.z), float(xh0.z)),
            float2(float(xl0.w), float(xh0.w)),
            float2(float(xl1.x), float(xh1.x)),
            float2(float(xl1.y), float(xh1.y)),
            float2(float(xl1.z), float(xh1.z)),
            float2(float(xl1.w), float(xh1.w)),
        }};
{xsum_code}        #pragma clang loop unroll(full)
        for (int r = 0; r < RPSG; r++) {{
          float2 ga = float2(gw_tile[r][0]) * xf[0];
          float2 ua = float2(uw_tile[r][0]) * xf[0];
          #pragma clang loop unroll(full)
          for (int j = 1; j < 8; j++) {{
            ga = fma(float2(gw_tile[r][j]), xf[j], ga);
            ua = fma(float2(uw_tile[r][j]), xf[j], ua);
          }}
{accum_lines}
        }}
      }}
    }}

    #pragma clang loop unroll(full)
    for (int m = 0; m < M; m++) {{
      #pragma clang loop unroll(full)
      for (int r = 0; r < RPSG; r++) {{
        float g_sum = simd_sum(g_res[m][r]);
        float u_sum = simd_sum(u_res[m][r]);
        if (simd_lid == 0) {{
          float gv = float(static_cast<T>(g_sum));
          float uv = float(static_cast<T>(u_sum));
          y[long(m) * n_out + out_row + r] = static_cast<T>((gv / (1.0f + exp(-gv))) * uv);
        }}
      }}
    }}
"""


def _single_kernel(rows: int, rpsg: int, nsg: int, neg_scale: bool) -> _KernelFn:
    key = (rows, rpsg, nsg, neg_scale)
    kernel = _SINGLE_KERNELS.get(key)
    if kernel is None:
        tag = "neg" if neg_scale else "aff"
        inputs = ["x", "w", "scales"] if neg_scale else ["x", "w", "scales", "biases"]
        built = mx.fast.metal_kernel(
            name=f"omlx_q2_verify_qmv_m{rows}_r{rpsg}_g{nsg}_{tag}",
            input_names=inputs,
            output_names=["y"],
            source=_single_source(rows, rpsg, nsg, neg_scale),
            header=_HEADER,
        )
        assert callable(built)
        kernel = built
        _SINGLE_KERNELS[key] = kernel
    return kernel


def _group_kernel(
    rows: int,
    rpsg: int,
    nsg: int,
    out_dims: tuple[int, ...],
    neg_scale: bool,
) -> _KernelFn:
    key = (rows, rpsg, nsg, out_dims, neg_scale)
    kernel = _GROUP_KERNELS.get(key)
    if kernel is None:
        inputs = ["x"]
        for idx in range(len(out_dims)):
            inputs += (
                [f"w{idx}", f"s{idx}"]
                if neg_scale
                else [f"w{idx}", f"s{idx}", f"b{idx}"]
            )
        dim_tag = "_".join(str(d) for d in out_dims)
        tag = "neg" if neg_scale else "aff"
        built = mx.fast.metal_kernel(
            name=f"omlx_q2_verify_group_m{rows}_r{rpsg}_g{nsg}_{tag}_{dim_tag}",
            input_names=inputs,
            output_names=[f"y{idx}" for idx in range(len(out_dims))],
            source=_group_source(rows, rpsg, nsg, out_dims, neg_scale),
            header=_HEADER,
        )
        assert callable(built)
        kernel = built
        _GROUP_KERNELS[key] = kernel
    return kernel


def _swiglu_kernel(rows: int, rpsg: int, nsg: int, neg_scale: bool) -> _KernelFn:
    key = (rows, rpsg, nsg, neg_scale)
    kernel = _SWIGLU_KERNELS.get(key)
    if kernel is None:
        tag = "neg" if neg_scale else "aff"
        inputs = (
            ["x", "gate_w", "gate_scales", "up_w", "up_scales"]
            if neg_scale
            else [
                "x",
                "gate_w",
                "gate_scales",
                "gate_biases",
                "up_w",
                "up_scales",
                "up_biases",
            ]
        )
        built = mx.fast.metal_kernel(
            name=f"omlx_q2_verify_swiglu_m{rows}_r{rpsg}_g{nsg}_{tag}",
            input_names=inputs,
            output_names=["y"],
            source=_swiglu_source(rows, rpsg, nsg, neg_scale),
            header=_HEADER,
        )
        assert callable(built)
        kernel = built
        _SWIGLU_KERNELS[key] = kernel
    return kernel


def q2_qmm(
    x2: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array | None = None,
    *,
    group_size: int,
    neg_scale: bool = False,
) -> mx.array:
    """Compute (rows, width) @ dequant(w_q).T -> (rows, cols) for 2-bit weights."""
    if x2.ndim != 2:
        raise ValueError(f"q2_qmm expects 2D input, got shape {x2.shape}")
    rows = int(x2.shape[0])
    width = int(x2.shape[1])
    cols = int(w_q.shape[0])
    gs = int(group_size)
    if (
        not eligible(rows, width, cols, _BITS, gs, x2.dtype)
        or w_q.shape != (cols, width // _VPT)
        or scales.shape != (cols, width // gs)
        or scales.dtype != x2.dtype
        or (
            not neg_scale
            and (
                biases is None
                or biases.shape != scales.shape
                or biases.dtype != x2.dtype
            )
        )
        or (neg_scale and biases is not None and biases.shape != scales.shape)
    ):
        raise ValueError(
            f"q2_qmm incompatible shapes: x={x2.shape}, w={w_q.shape}, scales={scales.shape}"
        )
    rpsg, nsg = _geometry(rows, cols)
    if neg_scale:
        inputs = [x2, w_q, scales]
    else:
        assert biases is not None
        inputs = [x2, w_q, scales, biases]
    (out,) = _single_kernel(rows, rpsg, nsg, neg_scale)(
        inputs=inputs,
        template=[("T", x2.dtype), ("GS", gs)],
        grid=(32, (cols // (rpsg * nsg)) * nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[x2.dtype],
    )
    return out


def q2_group_qmv(
    x2: mx.array,
    linears: Sequence[nn.QuantizedLinear],
) -> tuple[mx.array, ...]:
    rows = int(x2.shape[0])
    out_dims = tuple(int(lin.scales.shape[0]) for lin in linears)
    total_cols = sum(out_dims)
    rpsg, nsg = _group_geometry(rows, out_dims)
    neg_scale = all(is_ternary_linear(lin) for lin in linears)
    args: list[mx.array] = [x2]
    for lin in linears:
        if neg_scale:
            args += [lin.weight, lin.scales]
        else:
            biases = lin.biases
            assert biases is not None
            args += [lin.weight, lin.scales, biases]
    outputs = _group_kernel(rows, rpsg, nsg, out_dims, neg_scale)(
        inputs=args,
        template=[("T", x2.dtype), ("GS", int(linears[0].group_size))],
        grid=(32, (total_cols // (rpsg * nsg)) * nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(rows, cols) for cols in out_dims],
        output_dtypes=[x2.dtype] * len(out_dims),
    )
    return tuple(outputs)


def q2_swiglu(
    x2: mx.array,
    gate: nn.QuantizedLinear,
    up: nn.QuantizedLinear,
) -> mx.array:
    """Compute silu(T(x2 @ gate.T)) * T(x2 @ up.T) in a single Metal launch."""
    rows = int(x2.shape[0])
    cols = int(gate.scales.shape[0])
    rpsg, nsg = _swiglu_geometry(rows, cols)
    neg_scale = is_ternary_linear(gate) and is_ternary_linear(up)
    if neg_scale:
        inputs = [x2, gate.weight, gate.scales, up.weight, up.scales]
    else:
        gate_biases = gate.biases
        up_biases = up.biases
        assert gate_biases is not None and up_biases is not None
        inputs = [
            x2,
            gate.weight,
            gate.scales,
            gate_biases,
            up.weight,
            up.scales,
            up_biases,
        ]
    (out,) = _swiglu_kernel(rows, rpsg, nsg, neg_scale)(
        inputs=inputs,
        template=[("T", x2.dtype), ("GS", int(gate.group_size))],
        grid=(32, (cols // (rpsg * nsg)) * nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[x2.dtype],
    )
    return out
