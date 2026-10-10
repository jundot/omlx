"""Experimental packed EXL3 routed projection; no cache or attention changes.

Format/math adapted from beamivalice/sushi commit
27ca1c8684c01d1970bf576d8cef80ebea617928 and Sushi v1.0.5
162c044702d7ed3095fb85868e6cce005c93ab34 (MIT), expert_exl3_kernels.zig.
Copyright (c) 2026 Theinruj Toranavikrai and David Dalcu.
See docs/exl3-backend.md for provenance and prototype limitations.
"""

import math
import os
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class Exl3Spec:
    halfwords: int
    window: int

    def __post_init__(self):
        if (
            type(self.halfwords) is not int
            or not 32 <= self.halfwords <= 64
            or self.halfwords % 2
            or type(self.window) is not int
            or not 8 <= self.window <= 16
        ):
            raise ValueError("Unsupported EXL3 trellis rate/window")

    @classmethod
    def from_config(cls, config):
        q = config.get("expert_quant", {})
        if (
            q.get("format") != "exl3"
            or q.get("codebook") != "mcg"
            or q.get("out_scales") != "svh"
        ):
            raise ValueError(
                "EXL3 prototype requires mcg codebook and svh output scales"
            )
        k = q.get("k")
        n = (
            float(k) * 16
            if isinstance(k, (float, int)) and not isinstance(k, bool)
            else 0.0
        )
        w = q.get("window", 16)
        if (
            not math.isfinite(n)
            or not n.is_integer()
            or not 32 <= n <= 64
            or int(n) % 2
            or not isinstance(w, int)
            or isinstance(w, bool)
            or not 8 <= w <= 16
        ):
            raise ValueError("Unsupported EXL3 trellis rate/window")
        return cls(int(n), w)


def hadamard128(x):
    shape = x.shape
    y = x.astype(mx.float32).reshape(-1, 128)
    stride = 1
    while stride < 128:
        z = y.reshape(-1, 128 // (stride * 2), 2, stride)
        a, b = z[:, :, 0, :], z[:, :, 1, :]
        y = mx.stack([a + b, a - b], axis=2).reshape(-1, 128)
        stride *= 2
    return (y * (128**-0.5)).reshape(shape)


_HAD_SOURCE = r"""
uint col=thread_position_in_grid.x,row=thread_position_in_grid.y;
uint lane=thread_index_in_threadgroup;
threadgroup float values[128];
uint si=uint(ids[row])*DIM+col;
float a=float(x[row*DIM+col]);
if(SCALE_FIRST) a*=float(scales[si]);
values[lane]=a;threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint step=1;step<128;step<<=1){
 float b=values[lane^step];a=(lane&step)?b-a:a+b;
 threadgroup_barrier(mem_flags::mem_threadgroup);values[lane]=a;
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
a*=0.08838834764831845f;
if(!SCALE_FIRST) a*=float(scales[si]);
y[row*DIM+col]=half(a);
"""

_LEGACY_COOP_SOURCE = r"""
threadgroup float partial[4 * 256];
uint ot = uint(threadgroup_position_in_grid.x);
uint slot = uint(threadgroup_position_in_grid.y);
uint split = uint(threadgroup_position_in_grid.z);
uint sg = uint(simdgroup_index_in_threadgroup);
uint lane = uint(thread_index_in_simdgroup);
uint lid = uint(thread_index_in_threadgroup);
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_HW = N;
constexpr uint PACKED_W = N / 2u;
constexpr uint IT = uint(IDIM) / TILE;
constexpr uint OT = uint(ODIM) / TILE;
constexpr uint SGS = 4u;
constexpr uint SPLITS = 1u;
const uint eid = uint(slots[slot]);
const uint tiles_per_split = (IT + SPLITS - 1u) / SPLITS;
const uint tk0 = split * tiles_per_split;
const uint tk1 = min(tk0 + tiles_per_split, IT);
const uint prow = (lane & 3u) * 2u;
const uint pcol = lane >> 2u;
uint pos[8];
pos[0] = prow * 16u + pcol;
pos[1] = (prow + 1u) * 16u + pcol;
pos[2] = (prow + 8u) * 16u + pcol;
pos[3] = (prow + 9u) * 16u + pcol;
pos[4] = prow * 16u + pcol + 8u;
pos[5] = (prow + 1u) * 16u + pcol + 8u;
pos[6] = (prow + 8u) * 16u + pcol + 8u;
pos[7] = (prow + 9u) * 16u + pcol + 8u;
const uint row0 = pos[0] >> 4u;
const uint row1 = pos[1] >> 4u;
const uint row2 = pos[2] >> 4u;
const uint row3 = pos[3] >> 4u;
float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
const size_t xb = (size_t)slot * (size_t)(IDIM);
const size_t expert_stride = (size_t)IT * (size_t)OT * (size_t)PACKED_HW;
const device uint* trellis_e = (const device uint*)(trellis + (size_t)eid * expert_stride);
if (N == 64u) {
for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
  const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
  const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
  const float in0 = float(x[xb + tk * TILE + row0]);
  const float in1 = float(x[xb + tk * TILE + row1]);
  const float in2 = float(x[xb + tk * TILE + row2]);
  const float in3 = float(x[xb + tk * TILE + row3]);
  const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
  const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
  for (uint p = 0u; p < 4u; p++) {
    const uint2 cw = uint2(uint(merged >> sh[p * 2u]), uint(merged >> sh[p * 2u + 1u])) & uint2(0xffffu);
    const float2 w = exl3_decode2(cw,(1u << WINDOW)-1u);
    acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
    acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
  }
}
} else {
  uint w0[4];
  uint w1[4];
  uint shv[4];
  uint frv[4];
  for (uint p = 0u; p < 4u; p++) {
    const exl3_win wv = exl3_pair_window(lane * 8u + p * 2u, N);
    w0[p] = wv.i0;
    w1[p] = wv.i1;
    shv[p] = wv.sh;
    frv[p] = wv.fresh;
  }
  for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * PACKED_W;
    const float in0 = float(x[xb + tk * TILE + row0]);
    const float in1 = float(x[xb + tk * TILE + row1]);
    const float in2 = float(x[xb + tk * TILE + row2]);
    const float in3 = float(x[xb + tk * TILE + row3]);
    const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    for (uint p = 0u; p < 4u; p++) {
      const ulong merged = ((ulong)words[w0[p]] << 32) | (ulong)words[w1[p]];
      const uint funnel = uint(merged >> shv[p]);
      const uint2 cw = uint2((funnel >> frv[p]) & 0xffffu, funnel & 0xffffu);
      const float2 w = exl3_decode2(cw,(1u << WINDOW)-1u);
      acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
      acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
    }
  }
}
for (uint si = 0u; si < 8u; si++) {
  partial[sg * 256u + pos[si]] = acc[si];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 16u) {
  float sum = 0.0f;
  for (uint r = 0u; r < 16u; r++) {
    const uint p = r * 16u + lid;
    for (uint g = 0u; g < SGS; g++) {
      sum += partial[g * 256u + p];
    }
  }
  y[(size_t)slot * (size_t)(ODIM) + ot * TILE + lid] = half(sum);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_COOP_SOURCE = r"""
threadgroup float partial[uint(OTPT) * 4 * 256];
uint ot = uint(threadgroup_position_in_grid.x) * uint(OTPT);
uint slot = uint(threadgroup_position_in_grid.y);
uint split = uint(threadgroup_position_in_grid.z);
uint sg = uint(simdgroup_index_in_threadgroup);
uint lane = uint(thread_index_in_simdgroup);
uint lid = uint(thread_index_in_threadgroup);
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_HW = N;
constexpr uint PACKED_W = N / 2u;
constexpr uint IT = uint(IDIM) / TILE;
constexpr uint OT = uint(ODIM) / TILE;
constexpr uint SGS = 4u;
constexpr uint SPLITS = 1u;
const uint eid = uint(slots[slot]);
const uint tiles_per_split = (IT + SPLITS - 1u) / SPLITS;
const uint tk0 = split * tiles_per_split;
const uint tk1 = min(tk0 + tiles_per_split, IT);
const uint prow = (lane & 3u) * 2u;
const uint pcol = lane >> 2u;
uint pos[8];
pos[0] = prow * 16u + pcol;
pos[1] = (prow + 1u) * 16u + pcol;
pos[2] = (prow + 8u) * 16u + pcol;
pos[3] = (prow + 9u) * 16u + pcol;
pos[4] = prow * 16u + pcol + 8u;
pos[5] = (prow + 1u) * 16u + pcol + 8u;
pos[6] = (prow + 8u) * 16u + pcol + 8u;
pos[7] = (prow + 9u) * 16u + pcol + 8u;
const uint row0 = pos[0] >> 4u;
const uint row1 = pos[1] >> 4u;
const uint row2 = pos[2] >> 4u;
const uint row3 = pos[3] >> 4u;
float acc[OTPT][8] = {};
const size_t xb = (size_t)slot * (size_t)(IDIM);
const size_t expert_stride = (size_t)IT * (size_t)OT * (size_t)PACKED_HW;
const device uint* trellis_e = (const device uint*)(trellis + (size_t)eid * expert_stride);
if (N == 64u) {
for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
  const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
  const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
  const float in0 = float(x[xb + tk * TILE + row0]);
  const float in1 = float(x[xb + tk * TILE + row1]);
  const float in2 = float(x[xb + tk * TILE + row2]);
  const float in3 = float(x[xb + tk * TILE + row3]);
  const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
  const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
  for (uint p = 0u; p < 4u; p++) {
    const uint2 cw = uint2(uint(merged >> sh[p * 2u]), uint(merged >> sh[p * 2u + 1u])) & uint2(0xffffu);
    const float2 w = exl3_decode2(cw,(1u << WINDOW)-1u);
    acc[0][p * 2u] = fma(ins[p * 2u], w.x, acc[0][p * 2u]);
    acc[0][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[0][p * 2u + 1u]);
  }
}
} else {
  const device uint *wp = trellis_e + ((size_t)(tk0 + sg) * OT + ot) * PACKED_W;
  const device half *pp = x + xb + (tk0 + sg) * TILE;
  const uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u), exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
  // IT is a whole H128 block, hence the second k-tile is always in range.
  // Preserve the legacy per-accumulator k order and final reduction order.
  for (uint tk = tk0 + sg; tk < tk1; tk += 2u * SGS) {
    ulong merged[2][OTPT];
    for (uint u = 0u; u < 2u; u++) {
      for (uint o = 0u; o < uint(OTPT); o++) {
        merged[u][o] = exl3_lane<N>(wp + u * SGS * OT * PACKED_W + o * PACKED_W, lane);
      }
    }
    for (uint u = 0u; u < 2u; u++) {
      const float in0 = float(pp[u * SGS * TILE + row0]);
      const float in1 = float(pp[u * SGS * TILE + row1]);
      const float in2 = float(pp[u * SGS * TILE + row2]);
      const float in3 = float(pp[u * SGS * TILE + row3]);
      const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
      for (uint o = 0u; o < uint(OTPT); o++) {
        for (uint p = 0u; p < 4u; p++) {
          const uint2 cw = uint2(uint(merged[u][o] >> sh[p * 2u]), uint(merged[u][o] >> sh[p * 2u + 1u])) & uint2(0xffffu);
          const float2 w = exl3_decode2(cw, (1u << WINDOW)-1u);
          acc[o][p * 2u] = fma(ins[p * 2u], w.x, acc[o][p * 2u]);
          acc[o][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[o][p * 2u + 1u]);
        }
      }
    }
    wp += 2u * SGS * OT * PACKED_W;
    pp += 2u * SGS * TILE;
  }
}
for (uint o = 0u; o < uint(OTPT); o++) {
  for (uint si = 0u; si < 8u; si++) {
    partial[o * SGS * 256u + sg * 256u + pos[si]] = acc[o][si];
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 16u * uint(OTPT)) {
  const uint o = lid >> 4u;
  float sum = 0.0f;
  for (uint r = 0u; r < 16u; r++) {
    const uint p = r * 16u + (lid & 15u);
    for (uint g = 0u; g < SGS; g++) {
      sum += partial[o * SGS * 256u + g * 256u + p];
    }
  }
  y[(size_t)slot * ODIM + (ot + o) * TILE + (lid & 15u)] = half(sum);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_COOP_HEADER = r"""
static inline float2 exl3_decode2(uint2 cw, uint mask) {
 cw &= uint2(mask);
 uint2 r = ((cw * uint2(0xCBAC1FEDu)) & uint2(0x8FFF8FFFu)) ^ uint2(0x3B603B60u);
 half4 h=as_type<half4>(r); return float2(half2(h.x+h.y,h.z+h.w));
}

static inline constexpr uint exl3_end(uint N, uint j) { return ((j + 1u) * N) >> 4u; }
static inline ulong exl3_window(const device uint *words, uint end, uint nwords, bool full) {
  const uint last = (end - 1u) >> 5u;
  const uint prev = last == 0u ? nwords - 1u : last - 1u;
  const uint shift = (0u - end) & 31u;
  ulong bits = (((ulong)words[prev] << 32u) | (ulong)words[last]) >> shift;
  if (full) bits |= ((ulong)words[prev == 0u ? nwords - 1u : prev - 1u] << 32u) << (32u - shift);
  return bits;
}
static inline constexpr uint exl3_lane_sh(uint N, uint j) { return N / 2u - exl3_end(N, j); }
template<uint N>
static inline ulong exl3_lane(const device uint *words, uint lane) {
  constexpr uint W = N / 2u;
  constexpr uint low = W & (0u - W);
  return exl3_window(words, W * lane + W, W, exl3_lane_sh(N, 0u) + 16u > 32u + (low < 32u ? low : 32u));
}
struct exl3_win { uint i0; uint i1; uint sh; uint fresh; };
static inline exl3_win exl3_pair_window(uint t0,uint n) {
 uint e0=((t0+1u)*n)>>4u,e1=((t0+2u)*n)>>4u;
 uint j0=(e0+16u*n-16u)>>5u,j1=(e1+16u*n-1u)>>5u;
 exl3_win w;w.i0=j0%(n>>1u);w.i1=j1%(n>>1u);
 w.sh=(j1+1u)*32u-(e1+16u*n);w.fresh=e1-e0;return w;
}
"""

_HAD_KERNEL = None
_KERNELS = {}
_FAST_READERS = os.environ.get("OMLX_EXL3_FAST_READERS", "1") != "0"


def _scaled_hadamard(x, scales, ids, first):
    global _HAD_KERNEL
    if _HAD_KERNEL is None:
        _HAD_KERNEL = mx.fast.metal_kernel(
            name="omlx_exl3_scaled_hadamard",
            input_names=["x", "scales", "ids"],
            output_names=["y"],
            source=_HAD_SOURCE,
        )
    rows, dim = x.shape
    return _HAD_KERNEL(
        inputs=[x, scales, ids],
        template=[("DIM", dim), ("SCALE_FIRST", first)],
        grid=(dim, rows, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float16],
    )[0]


def _inner(x, ids, trellis, spec, n, sorted_indices=False):
    if x.shape[0] >= 32:
        from .exl3_prefill import prefill_inner

        return prefill_inner(
            x, ids, trellis, spec, n, _COOP_HEADER, sorted_indices,
            fast_readers=_FAST_READERS,
        )
    fast = _FAST_READERS and spec.halfwords < 64
    kernel = _KERNELS.get(fast)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="omlx_exl3_funnel_gather" if fast else "omlx_exl3_coop_gather",
            input_names=["x", "trellis", "slots"],
            output_names=["y"],
            source=_COOP_SOURCE if fast else _LEGACY_COOP_SOURCE,
            header=_COOP_HEADER,
        )
        _KERNELS[fast] = kernel
    rows, k = x.shape
    tiles = 2 if fast else 1
    return kernel(
        inputs=[x, trellis, ids],
        template=[
            ("IDIM", k),
            ("ODIM", n),
            ("NHW", spec.halfwords),
            ("WINDOW", spec.window),
            ("OTPT", tiles),
        ],
        grid=(n // 16 // tiles * 128, rows, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.float16],
    )[0]


class Exl3SwitchLinear(nn.Module):
    """Drop-in gathered linear projection; all state is ordinary MLX arrays."""

    def __init__(self, trellis, suh, svh, spec):
        super().__init__()
        if (
            trellis.ndim != 4
            or trellis.dtype != mx.uint16
            or trellis.shape[-1] != spec.halfwords
        ):
            raise ValueError("Invalid packed trellis shape/dtype")
        e, kt, nt, _ = trellis.shape
        if min(e, kt, nt) <= 0:
            raise ValueError("Empty EXL3 bank or dimensions")
        if (
            suh.shape != (e, kt * 16)
            or svh.shape != (e, nt * 16)
            or suh.dtype != mx.float16
            or svh.dtype != mx.float16
            or kt % 8
            or nt % 8
        ):
            raise ValueError("Invalid EXL3 scales or Hadamard dimensions")
        self.trellis, self.suh, self.svh = trellis, suh, svh
        self._spec = spec
        self.freeze()

    @property
    def input_dims(self):
        return self.suh.shape[-1]

    @property
    def output_dims(self):
        return self.svh.shape[-1]

    @property
    def num_experts(self):
        return self.trellis.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        if x.shape[-1] != self.input_dims or not mx.issubdtype(
            indices.dtype, mx.integer
        ):
            raise ValueError("Invalid EXL3 input width/expert indices")
        # Mirror gather_mm broadcasting of [batch..., M, K] against rhs ids.
        batch = mx.broadcast_shapes(x.shape[:-2], indices.shape)
        z = mx.broadcast_to(x, (*batch, x.shape[-2], self.input_dims))
        ids = (
            mx.broadcast_to(indices[..., None], (*batch, x.shape[-2]))
            .reshape(-1)
            .astype(mx.uint32)
        )
        valid = ids < self.num_experts
        safe_ids = mx.minimum(ids, self.num_experts - 1)
        z = z.reshape(-1, self.input_dims).astype(mx.float16)
        prepared = _scaled_hadamard(z, self.suh, safe_ids, True)
        inner = _inner(
            prepared,
            safe_ids,
            self.trellis,
            self._spec,
            self.output_dims,
            sorted_indices,
        )
        y = _scaled_hadamard(inner, self.svh, safe_ids, False)
        y = mx.where(valid[:, None], y, mx.array(float("nan"), dtype=y.dtype))
        return y.reshape(*batch, x.shape[-2], self.output_dims).astype(x.dtype)


def install_packed_experts(model, config):
    """Replace only routed SwitchGLU projections before strict native loading."""
    import os

    if os.environ.get("OMLX_EXL3_ENABLED", "0") != "1":
        raise ValueError("EXL3 support is opt-in: set OMLX_EXL3_ENABLED=1")
    spec = Exl3Spec.from_config({"expert_quant": config.expert_quant})
    layers = list(model.language_model.model.layers)
    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        layers.extend(mtp.layers)
    for layer in layers:
        mlp = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if mlp is None:
            raise ValueError("EXL3 checkpoint requires routed experts in every layer")
        for name in ("gate_proj", "up_proj", "down_proj"):
            old = getattr(mlp, name)
            e, n, k = old.weight.shape
            setattr(
                mlp,
                name,
                Exl3SwitchLinear(
                    mx.zeros((e, k // 16, n // 16, spec.halfwords), dtype=mx.uint16),
                    mx.zeros((e, k), dtype=mx.float16),
                    mx.zeros((e, n), dtype=mx.float16),
                    spec,
                ),
            )


def validate_checkpoint_headers(model_path, config, *, include_mtp=False):
    """Reject mismatched packed banks before allocating any checkpoint arrays."""
    import json
    import struct
    from pathlib import Path

    spec = Exl3Spec.from_config(config)
    text = config["text_config"]
    hidden = text["hidden_size"]
    intermediate = text["moe_intermediate_size"]
    experts = text["num_experts"]
    expected = {}
    banks = [
        (f"language_model.model.layers.{layer}.mlp.switch_mlp", experts)
        for layer in range(text["num_hidden_layers"])
    ]
    if include_mtp:
        if text.get("mtp_num_hidden_layers", 1) != 1:
            raise ValueError("EXL3 MTP supports the native one-layer draft head")
        banks.append(
            ("mtp.layers.0.mlp.switch_mlp", text.get("mtp_num_experts") or experts)
        )
    for prefix, bank_experts in banks:
        for proj, k, n in [
            ("gate_proj", hidden, intermediate),
            ("up_proj", hidden, intermediate),
            ("down_proj", intermediate, hidden),
        ]:
            base = f"{prefix}.{proj}"
            expected[base + ".trellis"] = (
                "U16",
                [bank_experts, k // 16, n // 16, spec.halfwords],
            )
            expected[base + ".suh"] = ("F16", [bank_experts, k])
            expected[base + ".svh"] = ("F16", [bank_experts, n])
    seen = set()
    for path in Path(model_path).glob("*.safetensors"):
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError("Truncated safetensors header")
            size = struct.unpack("<Q", raw)[0]
            if size > min(16 * 1024**2, path.stat().st_size - 8):
                raise ValueError("Invalid safetensors header size")
            header = json.loads(f.read(size))
        for key, entry in header.items():
            if include_mtp:
                for prefix in (
                    "model.language_model.mtp.",
                    "language_model.mtp.",
                    "model.mtp.",
                ):
                    if key.startswith(prefix):
                        key = "mtp." + key[len(prefix) :]
                        break
            if key not in expected:
                continue
            if key in seen:
                raise ValueError("Duplicate packed tensor: " + key)
            dtype, shape = expected[key]
            start, end = entry.get("data_offsets", [-1, -1])
            if (
                entry.get("dtype") != dtype
                or entry.get("shape") != shape
                or start < 0
                or end - start != math.prod(shape) * 2
                or size + 8 + end > path.stat().st_size
            ):
                raise ValueError("Invalid EXL3 checkpoint tensor: " + key)
            seen.add(key)
    missing = set(expected) - seen
    if missing:
        raise ValueError("Missing EXL3 tensors: " + ", ".join(sorted(missing)[:3]))
