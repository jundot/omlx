# SPDX-License-Identifier: Apache-2.0

import math
from functools import cache

import mlx.core as mx

from ..fp4_expert_rows import Activation
from ..mlx_vlm_glm5_next_compat.decode_kernels import moe_router_logits

_ROW_BLOCK = 8

_HEADER = r"""
template <typename U>
inline U ernie_sigmoid(U x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

# Invariant: a row's sums never depend on the other rows of its block.
# - Each mix dot and the sum of squares: 32 lanes split each of SLICES equal slices.
# - The finish kernel adds the slice partials in slice order.
_HC_PARTS_SOURCE = r"""
  constexpr int K = HC * D;
  constexpr int SLICE = K / SLICES;
  constexpr int PER = SLICE / 32;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int slice = int(threadgroup_position_in_grid.x);
  int nrows = 1;
  for (int i = 0; i < x_ndim - 2; i++) {
    nrows *= x_shape[i];
  }
  const int r0 = int(threadgroup_position_in_grid.y) * RB;
  const int rows = min(RB, nrows - r0);
  const int k0 = slice * SLICE + int(lane) * PER;
  for (int m = int(sg); m <= MIX; m += NSG) {
    for (int r = 0; r < rows; r++) {
      const device T* xr = x + size_t(r0 + r) * K + k0;
      float acc = 0;
      if (m < MIX) {
        const device W* f = fn + size_t(m) * K + k0;
        for (int i = 0; i < PER; i++) {
          acc += static_cast<float>(f[i]) * static_cast<float>(xr[i]);
        }
      } else {
        for (int i = 0; i < PER; i++) {
          const float v = static_cast<float>(xr[i]);
          acc += v * v;
        }
      }
      acc = simd_sum(acc);
      if (lane == 0) {
        parts[(size_t(r0 + r) * (MIX + 1) + m) * SLICES + slice] = acc;
      }
    }
  }
"""

_HC_PRE_SOURCE = r"""
  constexpr int K = HC * D;
  const uint lid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int row = int(threadgroup_position_in_grid.y);
  const float rms_eps = consts[0];
  const float hc_eps = consts[1];
  threadgroup float total[MIX + 1];
  threadgroup float pre[HC];
  threadgroup float partial[32];
  threadgroup float inv;
  if (int(lid) <= MIX) {
    const device float* p = parts + (size_t(row) * (MIX + 1) + lid) * SLICES;
    float t = p[0];
    for (int s = 1; s < SLICES; s++) {
      t += p[s];
    }
    total[lid] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(lid) < MIX) {
    const float denom = metal::precise::sqrt(total[MIX] / float(K)) + rms_eps;
    const float m = total[lid] / denom;
    mixes[size_t(row) * MIX + lid] = m;
    if (int(lid) < HC) {
      pre[lid] = ernie_sigmoid(m * scale[0] + base[lid]) + hc_eps;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device T* xr = x + size_t(row) * K;
  device T* out = normed + size_t(row) * D;
  float acc = 0;
  for (int d = int(lid); d < D; d += 1024) {
    float c = pre[0] * static_cast<float>(xr[d]);
    for (int s = 1; s < HC; s++) {
      c += pre[s] * static_cast<float>(xr[s * D + d]);
    }
    const T rounded = static_cast<T>(c);
    out[d] = rounded;
    const float v = static_cast<float>(rounded);
    acc += v * v;
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    partial[sg] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float sum = simd_sum(partial[lane]);
    if (lane == 0) {
      inv = metal::precise::rsqrt(sum / float(D) + rms_eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int d = int(lid); d < D; d += 1024) {
    out[d] = norm_weight[d] * static_cast<T>(static_cast<float>(out[d]) * inv);
  }
"""

_HC_EXPAND_SOURCE = r"""
  const uint lid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int row = int(threadgroup_position_in_grid.y);
  const float hc_eps = consts[1];
  threadgroup float post[4];
  threadgroup float comb[16];
  if (sg == 0) {
    const device float* m = mixes + size_t(row) * MIX;
    const bool active = lane < 4;
    const int i = min(int(lane), 3);
    const float post_v = 2 * ernie_sigmoid(m[4 + i] * scale[1] + base[4 + i]);
    float4 c;
    for (int jj = 0; jj < 4; jj++) {
      c[jj] = m[8 + i * 4 + jj] * scale[2] + base[8 + i * 4 + jj];
    }
    const float top = metal::max(metal::max(c.x, c.y), metal::max(c.z, c.w));
    float4 e = metal::precise::exp(c - top);
    c = e / (e.x + e.y + e.z + e.w) + hc_eps;
    c = active ? c : float4(0);
    float4 cols = float4(simd_sum(c.x), simd_sum(c.y), simd_sum(c.z), simd_sum(c.w));
    c = c / (cols + hc_eps);
    for (int it = 1; it < ITERS; it++) {
      c = c / ((c.x + c.y + c.z + c.w) + hc_eps);
      cols = float4(simd_sum(c.x), simd_sum(c.y), simd_sum(c.z), simd_sum(c.w));
      c = c / (cols + hc_eps);
    }
    if (active) {
      post[i] = post_v;
      for (int jj = 0; jj < 4; jj++) {
        comb[i * 4 + jj] = c[jj];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int d = (int(threadgroup_position_in_grid.x) * 256 + int(lid)) * 4;
  if (d >= D) {
    return;
  }
  const device T* xr = x + size_t(row) * HC * D;
  const device T* yr = y + size_t(row) * D;
  device T* outr = out + size_t(row) * HC * D;
  for (int t = 0; t < 4; t++) {
    const float yv = static_cast<float>(yr[d + t]);
    float res[4];
    for (int s = 0; s < 4; s++) {
      res[s] = static_cast<float>(xr[s * D + d + t]);
    }
    for (int i = 0; i < 4; i++) {
      float acc = comb[i] * res[0];
      for (int s = 1; s < 4; s++) {
        acc += comb[s * 4 + i] * res[s];
      }
      outr[i * D + d + t] = static_cast<T>(post[i] * yv + acc);
    }
  }
"""

# mlx: argpartition returns the top-k by a stable sort, ties to the lower id.
_SELECT_SOURCE = r"""
  threadgroup float biased[E];
  threadgroup float picked[TOPK];
  const uint row = threadgroup_position_in_grid.x;
  const uint e = thread_position_in_threadgroup.x;
  float score = s0[size_t(row) * E + e];
  score += s1[size_t(row) * E + e];
  const float b = score + bias[e];
  biased[e] = b;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint rank = 0;
  for (uint o = 0; o < E; o++) {
    const float v = biased[o];
    rank += (v > b || (v == b && o < e)) ? 1 : 0;
  }
  if (rank < TOPK) {
    inds[size_t(row) * TOPK + rank] = e;
    picked[rank] = score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (rank < TOPK) {
    float w = score;
    if (NORM) {
      float total = picked[0];
      for (int k = 1; k < TOPK; k++) {
        total += picked[k];
      }
      w = w / metal::max(total, 1e-20f);
    }
    weights[size_t(row) * TOPK + rank] = w * routed_scale[e];
  }
"""

_VHA_SOURCE = r"""
  const int d = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  const int part = int(thread_position_in_grid.z);
  const int length = o_shape[2];
  const int rows = o_shape[0] * length;
  const int b = r / length;
  const device T* src = o + (size_t(b) * H * length + (r - b * length)) * D + d;
  float v[H];
  for (int g = 0; g < H; g++) {
    v[g] = static_cast<float>(src[size_t(g) * length * D]);
  }
  for (int i = 0; i < H / PARTS; i++) {
    const int h = part * (H / PARTS) + i;
    float acc = 0;
    for (int g = 0; g < H; g++) {
      acc += static_cast<float>(mix[h * H + g]) * v[g];
    }
    const int group = h / HPG;
    out[(size_t(group) * rows + r) * (HPG * D) + (h - group * HPG) * D + d] =
        static_cast<T>(acc);
  }
"""

SITU = Activation(
    name="ernie_situ",
    source=r"""
                    const float gv = float(T(g));
                    const float uv = float(T(u));
                    const float ga = (situ[0] * metal::precise::tanh(gv / situ[0])) * fp4_sigmoid(gv);
                    const float ua = situ[1] * metal::precise::tanh(uv / situ[1]);
                    act[out] = T(ga * ua);""",
    inputs=("situ",),
)


@cache
def _hc_parts_kernel():
    return mx.fast.metal_kernel(
        name="ernie_hc_parts",
        input_names=["x", "fn"],
        output_names=["parts"],
        source=_HC_PARTS_SOURCE,
    )


@cache
def _hc_pre_kernel():
    return mx.fast.metal_kernel(
        name="ernie_hc_pre",
        input_names=["parts", "x", "scale", "base", "norm_weight", "consts"],
        output_names=["normed", "mixes"],
        header=_HEADER,
        source=_HC_PRE_SOURCE,
    )


@cache
def _hc_expand_kernel():
    return mx.fast.metal_kernel(
        name="ernie_hc_expand",
        input_names=["mixes", "y", "x", "scale", "base", "consts"],
        output_names=["out"],
        header=_HEADER,
        source=_HC_EXPAND_SOURCE,
    )


@cache
def _select_kernel():
    return mx.fast.metal_kernel(
        name="ernie_router_select",
        input_names=["s0", "s1", "bias", "routed_scale"],
        output_names=["inds", "weights"],
        source=_SELECT_SOURCE,
    )


@cache
def _vha_kernel():
    return mx.fast.metal_kernel(
        name="ernie_vha",
        input_names=["o", "mix"],
        output_names=["out"],
        source=_VHA_SOURCE,
    )


def vha(o, mix, groups):
    batch, heads, length, width = o.shape
    rows = batch * length
    parts = 1 if rows > _ROW_BLOCK else math.gcd(_ROW_BLOCK, heads)
    return _vha_kernel()(
        inputs=[o, mix],
        template=[
            ("T", o.dtype),
            ("H", heads),
            ("D", width),
            ("HPG", heads // groups),
            ("PARTS", parts),
        ],
        grid=(width, rows, parts),
        threadgroup=(min(width, 256), 1, 1),
        output_shapes=[(groups, rows, heads // groups * width)],
        output_dtypes=[o.dtype],
    )[0]


def hc_pre(x, fn, scale, base, norm_weight, consts):
    *lead, hc, width = x.shape
    rows = x.size // (hc * width)
    mix = fn.shape[0]
    slices = math.gcd(hc * width // 32, 32)
    parts = _hc_parts_kernel()(
        inputs=[x, fn],
        template=[
            ("T", x.dtype),
            ("W", fn.dtype),
            ("HC", hc),
            ("D", width),
            ("MIX", mix),
            ("SLICES", slices),
            ("RB", _ROW_BLOCK),
            ("NSG", 8),
        ],
        grid=(256 * slices, -(-rows // _ROW_BLOCK), 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, mix + 1, slices)],
        output_dtypes=[mx.float32],
    )[0]
    return _hc_pre_kernel()(
        inputs=[parts, x, scale, base, norm_weight, consts],
        template=[
            ("T", x.dtype),
            ("HC", hc),
            ("D", width),
            ("MIX", mix),
            ("SLICES", slices),
        ],
        grid=(1024, rows, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(*lead, width), (*lead, mix)],
        output_dtypes=[x.dtype, mx.float32],
    )


def hc_expand(mixes, y, x, scale, base, consts, iters):
    *lead, hc, width = x.shape
    rows = x.size // (hc * width)
    return _hc_expand_kernel()(
        inputs=[mixes, y, x, scale, base, consts],
        template=[
            ("T", x.dtype),
            ("HC", hc),
            ("D", width),
            ("MIX", mixes.shape[-1]),
            ("ITERS", iters),
        ],
        grid=(256 * -(-width // 1024), rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _zeros(size):
    return mx.zeros((size,), dtype=mx.float32)


def route(x, weight, weight_1, bias, routed_scale, top_k, normalize):
    experts = weight.shape[0]
    first = moe_router_logits(x, weight, _zeros(experts))
    if first is None:
        return None
    second = moe_router_logits(x, weight_1, _zeros(experts))
    rows = x.shape[0]
    return _select_kernel()(
        inputs=[first[0], second[0], bias, routed_scale],
        template=[("E", experts), ("TOPK", top_k), ("NORM", int(normalize))],
        grid=(experts * rows, 1, 1),
        threadgroup=(experts, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
