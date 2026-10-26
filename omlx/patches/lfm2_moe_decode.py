# SPDX-License-Identifier: Apache-2.0
"""Fused decode for LFM2-MoE (``lfm2_moe``, LFM2.5-24B-A2B).

A decode step of the stock mlx-lm layer runs about fifteen launches for
each sparse MoE feed-forward (RMSNorm, router linear, sigmoid, the fp32
bias add, ``argpartition``, ``take_along_axis``, the renormalization, three
``gather_qmm`` and the SwiGLU, the score product and sum, the residual add)
and about ten for each short-conv mixer (split, ``B * x``, concatenate with
the state, a depthwise ``conv1d``, the state slice, ``C * conv``, ...). At
batch one these launches are tiny and serially dependent, so the GPU spends
much of the step between kernels: on an M4 Max the MoE block took 121 us
for 26 MB of expert weights (about 68 us at memory bandwidth), and the
router chain alone 37 us.

This patch runs a one-token MoE block in three launches:

1. ``router``: one threadgroup per expert folds the ffn RMSNorm (MLX
   ``rms_norm`` arithmetic) and computes that expert's router logit; the
   weight row is loaded before the norm's reduction so its DRAM latency
   overlaps it. Threadgroup 0 also stores the normalized activation.
2. ``gate_up``: every simdgroup recomputes the sigmoid + expert-bias top-k
   selection from the 64 logits (two values per lane, a few ``simd_max``
   rounds), then runs gate and up rows of its selected expert with MLX's
   ``qmv`` arithmetic (``qwen35_moe_routed_decode.qmv_rows``) and the
   SwiGLU epilogue. The first threadgroup stores the selection and the
   renormalized scores.
3. ``down``: one simdgroup per selected expert computes down rows, then the
   score-weighted sum and the residual add.

and a short-conv mixer in two launches after its RMSNorm: the ``in_proj``
rows B, C and x of a channel block plus the 3-tap conv step and state
shift in one, ``out_proj`` plus the residual add in the other.

Experts are selected exactly as mlx-lm does (sigmoid in the activation
dtype, fp32 bias add, top-k; ties to the lowest index). The quantized dot
products are MLX's own arithmetic; renormalization, the expert sum and the
conv taps accumulate in fp32, so outputs can differ from the composed path
by an ulp of the activation dtype. Teacher-forced on LFM2.5-24B-A2B the
fused logits sit as close to the stock decode as the stock decode sits to
the stock prefill (mean KL 2.3e-3 vs 2.5e-3, top-1 agreement 99.0% vs
98.8%); like any two kernel paths, free-running greedy text can part at an
expert near-tie (fp16 sigmoid scores are often within an ulp). Stacked
expert weights are bound through a one-expert view
(``qwen35_moe_routed_decode._expert_view``) so the launches do not end a
command buffer each.

Attention layers run their projections, norms and RoPE as before, then
append to the copy-free KV decode tail (``kv_decode_tail``) and attend with
its two-segment kernel; out_proj and the residual add share one launch.

Rows of one sequence (``B == 1``) with up to ``MAX_ROWS`` positions take
these launches: a decode token, or an MTP verify block (for which the conv
launch also snapshots the state after the confirmed prefix into
``cache.rollback_state``). Longer chunks, batched rows, masks, ragged
lengths, left padding and anything outside the supported layout keep the
stock layer. If building a launch fails the patch disables itself and the
stock layer runs.
"""

from __future__ import annotations

import logging
from functools import cache
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from . import kv_decode_tail
from .module_cache import cached_per_module
from .qwen35_moe_routed_decode import (
    _COMMON,
    _expert_view,
    _Format,
    _format_header,
    _mlx_format,
    _name,
    _quantized_ok,
)

logger = logging.getLogger(__name__)

_APPLIED = False
_DISABLED = False
_ROUTER_SIMDGROUPS = 4
_GATE_UP_ROWS = 2
_DOWN_ROWS = 4
_CONV_ROWS = 2
_CONV_SIMDGROUPS = 2


def _float(v: float) -> str:
    return repr(float(v)) + "f"


# ---------------------------------------------------------------- MoE
_ROUTER_SRC = r"""
    constexpr int NT = NSG * 32;
    constexpr int CH = H / (NT * 8);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int tid = int(sg) * 32 + int(lane);
    const int e = int(threadgroup_position_in_grid.x);
    const int row = int(threadgroup_position_in_grid.y);
    h += row * H;
    xn += row * H;
    logits += row * int(threadgroups_per_grid.x);
    threadgroup float red[NSG];
    const device T* wr = gw + size_t(e) * H;
    T wv[CH][8];
    for (int c = 0; c < CH; c++) {
      const int base = (c * NT + tid) * 8;
      for (int t = 0; t < 8; t++) {
        wv[c][t] = wr[base + t];
      }
    }
    float hv[CH][8];
    float ss = 0;
    for (int c = 0; c < CH; c++) {
      const int base = (c * NT + tid) * 8;
      for (int t = 0; t < 8; t++) {
        hv[c][t] = float(h[base + t]);
        ss += hv[c][t] * hv[c][t];
      }
    }
    ss = simd_sum(ss);
    if (lane == 0) {
      red[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (int s = 0; s < NSG; s++) {
      tot += red[s];
    }
    const float inv = metal::precise::rsqrt(tot / H + EPS);
    float acc = 0;
    for (int c = 0; c < CH; c++) {
      const int base = (c * NT + tid) * 8;
      for (int t = 0; t < 8; t++) {
        const T xv = nw[base + t] * static_cast<T>(hv[c][t] * inv);
        if (e == 0) {
          xn[base + t] = xv;
        }
        acc += float(wv[c][t]) * float(xv);
      }
    }
    acc = simd_sum(acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      red[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float a = 0;
      for (int s = 0; s < NSG; s++) {
        a += red[s];
      }
      logits[e] = static_cast<T>(a);
    }
"""

# Lane l holds experts l + 32 k. Key = float(sigmoid_T(logit)) + bias, the
# fp32 ``scores_for_routing`` of mlx-lm; ties resolve to the lowest index.
_SELECT_HEADER = r"""
template <typename T, int NE, int TOPK>
METAL_FUNC void lfm2_select(
    const device T* logits, const device float* bias, uint lane,
    thread uint* sel_i, thread float* sel_p) {
  constexpr int PER = NE / 32;
  float key[PER];
  float prob[PER];
  for (int k = 0; k < PER; k++) {
    const int e = k * 32 + int(lane);
    const T p = omlx_mlx_sigmoid<T>(logits[e]);
    prob[k] = float(p);
    key[k] = float(p) + bias[e];
  }
  for (int j = 0; j < TOPK; j++) {
    float best = -INFINITY;
    int bk = 0;
    for (int k = 0; k < PER; k++) {
      if (key[k] > best) {
        best = key[k];
        bk = k;
      }
    }
    const float g = simd_max(best);
    const uint mine_i = uint(bk * 32 + int(lane));
    const uint gi = simd_min(best == g ? mine_i : 0xffffffffu);
    float mine_p = 0;
    if (mine_i == gi) {
      key[bk] = -INFINITY;
      mine_p = prob[bk];
    }
    sel_p[j] = simd_max(mine_p);
    sel_i[j] = gi;
  }
}
"""

_GATE_UP_SRC = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int ROWS = NSG * RPS;
    const int row = int(threadgroup_position_in_grid.y) % M;
    int b = int(threadgroup_position_in_grid.y) / M;
    x += row * K;
    logits += row * NE;
    y += size_t(row) * TOPK * NI;
    indices += row * TOPK;
    scores += row * TOPK;
    uint sel_i[TOPK];
    float sel_p[TOPK];
    lfm2_select<T, NE, TOPK>(logits, ebias, simd_lid, sel_i, sel_p);
    if (b == 0) {
      if (simd_gid == 0 && simd_lid == 0) {
        float total = 0;
        for (int j = 0; j < TOPK; j++) {
          total += sel_p[j];
        }
        for (int j = 0; j < TOPK; j++) {
          const float p = NORM_TOPK ? sel_p[j] / (total + 1e-6f) : sel_p[j];
          indices[j] = sel_i[j];
          scores[j] = static_cast<T>(p * RSF);
        }
      }
      return;
    }
    b -= 1;
    const int slot = b / (NI / ROWS);
    const int out_row = (b % (NI / ROWS)) * ROWS + int(simd_gid) * RPS;
    const size_t expert = size_t(sel_i[slot]);
    float result[2 * RPS];
    rt::qmv_rows<T, K, RPS, RPS>(
        (const device uint8_t*)w, scales, biases, expert * (2 * NI) + out_row,
        (const device uint8_t*)w, scales, biases, expert * (2 * NI) + NI + out_row,
        x, simd_lid, result);
    swiglu_store<T, RPS>(result, y + size_t(slot) * NI + out_row, simd_lid);
"""

_DOWN_SRC = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int row = int(tid.y) % M;
    const int out_row = int(tid.y) / M * RPS;
    const int slot = int(simd_gid);
    x += size_t(row) * TOPK * K;
    indices += row * TOPK;
    scores += row * TOPK;
    resid += row * N;
    y += row * N;
    threadgroup float part[TOPK * RPS];
    float result[RPS];
    const size_t expert = size_t(indices[slot]);
    rd::qmv_rows<T, K, RPS, 0>(
        (const device uint8_t*)w, scales, biases, expert * N + out_row,
        (const device uint8_t*)w, scales, biases, expert * N + out_row,
        x + slot * K, simd_lid, result);
    if (simd_lid == 0) {
      for (int r = 0; r < RPS; r++) {
        part[slot * RPS + r] = float(static_cast<T>(result[r]) * scores[slot]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0 && int(simd_lid) < RPS) {
      const int r = int(simd_lid);
      float acc = 0;
      for (int j = 0; j < TOPK; j++) {
        acc += part[j * RPS + r];
      }
      y[out_row + r] = resid[out_row + r] + static_cast<T>(acc);
    }
"""

# ---------------------------------------------------------------- short conv
# Simdgroup rows d.., H + d.., 2H + d.. of in_proj (B, C, x of a channel
# block) for each of the M input rows, then the 3-tap depthwise conv steps in
# row order:
#   bx = B * x; conv = w0 s0 + w1 s1 + w2 bx; y = C * conv; (s0, s1) = (s1, bx)
# new_state is the state after all M rows, conf_state after the first NC
# (an MTP verify block's confirmed prefix; NC = M otherwise).
_CONV_SRC = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int d = (int(threadgroup_position_in_grid.y) * NSG + int(simd_gid)) * RPS;
    float rb[M][2 * RPS];
    float rx[M][RPS];
    for (int m = 0; m < M; m++) {
      ct::qmv_rows<T, H, RPS, RPS>(
          (const device uint8_t*)w, scales, biases, d,
          (const device uint8_t*)w, scales, biases, H + d,
          x + m * H, simd_lid, rb[m]);
      ct::qmv_rows<T, H, RPS, 0>(
          (const device uint8_t*)w, scales, biases, 2 * H + d,
          (const device uint8_t*)w, scales, biases, 2 * H + d,
          x + m * H, simd_lid, rx[m]);
    }
    if (simd_lid == 0) {
      for (int r = 0; r < RPS; r++) {
        const int c = d + r;
        T s0 = state[c];
        T s1 = state[H + c];
        for (int m = 0; m < M; m++) {
          const T bv = static_cast<T>(rb[m][r]);
          const T cv = static_cast<T>(rb[m][RPS + r]);
          const T xv = static_cast<T>(rx[m][r]);
          const T bx = bv * xv;
          const float acc = float(cw[c * 3]) * float(s0) +
              float(cw[c * 3 + 1]) * float(s1) + float(cw[c * 3 + 2]) * float(bx);
          y[m * H + c] = cv * static_cast<T>(acc);
          s0 = s1;
          s1 = bx;
          if (m == NC - 1) {
            conf_state[c] = s0;
            conf_state[H + c] = s1;
          }
        }
        new_state[c] = s0;
        new_state[H + c] = s1;
      }
    }
"""

_OUT_SRC = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int d = (int(threadgroup_position_in_grid.y) * NSG + int(simd_gid)) * RPS;
    for (int m = 0; m < M; m++) {
      float r[RPS];
      ot::qmv_rows<T, K, RPS, 0>(
          (const device uint8_t*)w, scales, biases, d,
          (const device uint8_t*)w, scales, biases, d,
          x + m * K, simd_lid, r);
      if (simd_lid == 0) {
        for (int i = 0; i < RPS; i++) {
          y[m * K + d + i] = resid[m * K + d + i] + static_cast<T>(r[i]);
        }
      }
    }
"""


@cache
def _router_kernel(eps: float):
    return mx.fast.metal_kernel(
        name=f"omlx_lfm2_moe_router_norm_{abs(hash(eps)):x}",
        input_names=["h", "nw", "gw"],
        output_names=["xn", "logits"],
        header="using namespace metal;\n",
        source=_ROUTER_SRC.replace("EPS", _float(eps)),
    )


@cache
def _gate_up_kernel(fmt: _Format, norm_topk: bool, rsf: float):
    source = _GATE_UP_SRC.replace("NORM_TOPK", "true" if norm_topk else "false")
    return mx.fast.metal_kernel(
        name=f"omlx_lfm2_moe_gate_up_topk_{_name(fmt)}_{int(norm_topk)}_{abs(hash(rsf)):x}",
        input_names=["x", "w", "scales", "biases", "logits", "ebias"],
        output_names=["y", "indices", "scores"],
        header=_COMMON + _SELECT_HEADER + _format_header("rt", fmt),
        source=source.replace("RSF", _float(rsf)),
    )


@cache
def _down_kernel(fmt: _Format):
    return mx.fast.metal_kernel(
        name=f"omlx_lfm2_moe_down_combine_{_name(fmt)}",
        input_names=["x", "w", "scales", "biases", "indices", "scores", "resid"],
        output_names=["y"],
        header=_COMMON + _format_header("rd", fmt),
        source=_DOWN_SRC,
    )


@cache
def _conv_kernel(fmt: _Format):
    return mx.fast.metal_kernel(
        name=f"omlx_lfm2_conv_in_step_{_name(fmt)}",
        input_names=["x", "w", "scales", "biases", "cw", "state"],
        output_names=["y", "new_state", "conf_state"],
        header=_COMMON + _format_header("ct", fmt),
        source=_CONV_SRC,
    )


@cache
def _out_kernel(fmt: _Format):
    return mx.fast.metal_kernel(
        name=f"omlx_lfm2_out_resid_{_name(fmt)}",
        input_names=["x", "w", "scales", "biases", "resid"],
        output_names=["y"],
        header=_COMMON + _format_header("ot", fmt),
        source=_OUT_SRC,
    )


# ---------------------------------------------------------------- plans
def _operands(layer, view: bool = False) -> tuple:
    keys = ("weight", "scales", "biases")
    return tuple(_expert_view(layer[k]) if view else layer[k] for k in keys)


class _MoEPlan(NamedTuple):
    hidden: int
    inter: int
    top_k: int
    num_experts: int
    dtype: mx.Dtype
    eps: float
    norm_weight: mx.array
    gate_weight: mx.array
    expert_bias: mx.array
    gate_up_fmt: _Format
    gate_up_operands: tuple
    down_fmt: _Format
    down_operands: tuple
    norm_topk: bool
    rsf: float


class _ConvPlan(NamedTuple):
    hidden: int
    dtype: mx.Dtype
    in_fmt: _Format
    in_operands: tuple
    conv_weight: mx.array
    out_fmt: _Format
    out_operands: tuple


def _build_moe_plan(layer) -> _MoEPlan | None:
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwiGLU

    block = layer.get("feed_forward")
    norm = layer.get("ffn_norm")
    if block is None or type(norm) is not nn.RMSNorm or "switch_mlp" not in block:
        return None
    switch_mlp = block["switch_mlp"]
    gate_up = switch_mlp.get("gate_up_proj")
    down = switch_mlp.get("down_proj")
    gate = block.get("gate")
    if not (
        _quantized_ok(gate_up, QuantizedSwitchLinear)
        and _quantized_ok(down, QuantizedSwitchLinear)
        and type(switch_mlp.get("activation")) is SwiGLU
        and type(gate) is nn.Linear
        and "bias" not in gate
    ):
        return None
    dtype = gate_up["scales"].dtype
    hidden = down["weight"].shape[1]
    inter = down["weight"].shape[-1] * 32 // down.bits
    num_experts = gate["weight"].shape[0]
    top_k = int(block.top_k)
    gu_fmt = _mlx_format(gate_up, hidden, 2 * inter)
    d_fmt = _mlx_format(down, inter, hidden)
    nt = _ROUTER_SIMDGROUPS * 32 * 8
    if (
        gu_fmt is None
        or d_fmt is None
        or not gu_fmt.fast
        or gate["weight"].shape != (num_experts, hidden)
        or gate["weight"].dtype != dtype
        or norm["weight"].dtype != dtype
        or down["scales"].dtype != dtype
        or gate_up["weight"].shape[0] != num_experts
        or num_experts % 32
        or not 0 < top_k <= min(num_experts, 32)
        or hidden % nt
        or inter % _GATE_UP_ROWS
        or hidden % _DOWN_ROWS
    ):
        return None
    if getattr(block, "use_expert_bias", False):
        bias = block["expert_bias"].astype(mx.float32)
    else:
        bias = mx.zeros((num_experts,), dtype=mx.float32)
    mx.eval(bias)
    return _MoEPlan(
        hidden=hidden,
        inter=inter,
        top_k=top_k,
        num_experts=num_experts,
        dtype=dtype,
        eps=float(norm.eps),
        norm_weight=norm["weight"],
        gate_weight=gate["weight"],
        expert_bias=bias,
        gate_up_fmt=gu_fmt,
        gate_up_operands=_operands(gate_up, view=True),
        down_fmt=d_fmt,
        down_operands=_operands(down, view=True),
        norm_topk=bool(getattr(block, "norm_topk_prob", True)),
        rsf=float(getattr(block, "routed_scaling_factor", 1.0)),
    )


def _build_conv_plan(layer) -> _ConvPlan | None:
    conv = layer.get("conv")
    if conv is None:
        return None
    in_proj, out_proj, conv1d = (
        conv.get("in_proj"),
        conv.get("out_proj"),
        conv.get("conv"),
    )
    if not (
        _quantized_ok(in_proj, nn.QuantizedLinear)
        and _quantized_ok(out_proj, nn.QuantizedLinear)
        and type(conv1d) is nn.Conv1d
        and "bias" not in conv1d
        and getattr(conv, "L_cache", None) == 3
    ):
        return None
    dtype = in_proj["scales"].dtype
    hidden = out_proj["weight"].shape[0]
    in_fmt = _mlx_format(in_proj, hidden, 3 * hidden)
    out_fmt = _mlx_format(out_proj, hidden, hidden)
    weight = conv1d["weight"]
    if (
        in_fmt is None
        or out_fmt is None
        or out_proj["scales"].dtype != dtype
        or weight.shape != (hidden, 3, 1)
        or weight.dtype != dtype
        or hidden % (_CONV_ROWS * _CONV_SIMDGROUPS)
    ):
        return None
    return _ConvPlan(
        hidden=hidden,
        dtype=dtype,
        in_fmt=in_fmt,
        in_operands=_operands(in_proj),
        conv_weight=weight,
        out_fmt=out_fmt,
        out_operands=_operands(out_proj),
    )


# ---------------------------------------------------------------- launches
MAX_ROWS = 4  # one decode token, or an MTP verify block of up to 4 rows


def fused_moe(plan: _MoEPlan, h: mx.array) -> mx.array:
    """``h + feed_forward(ffn_norm(h))`` for ``h`` [1, M, H]."""
    hidden, dtype = plan.hidden, plan.dtype
    rows = h.shape[1]
    xn, logits = _router_kernel(plan.eps)(
        inputs=[h, plan.norm_weight, plan.gate_weight],
        template=[("T", dtype), ("H", hidden), ("NSG", _ROUTER_SIMDGROUPS)],
        grid=(32 * _ROUTER_SIMDGROUPS * plan.num_experts, rows, 1),
        threadgroup=(32 * _ROUTER_SIMDGROUPS, 1, 1),
        output_shapes=[(rows, hidden), (rows, plan.num_experts)],
        output_dtypes=[dtype, dtype],
    )
    top_k, inter = plan.top_k, plan.inter
    act, indices, scores = _gate_up_kernel(plan.gate_up_fmt, plan.norm_topk, plan.rsf)(
        inputs=[xn, *plan.gate_up_operands, logits, plan.expert_bias],
        template=[
            ("T", dtype),
            ("K", hidden),
            ("NI", inter),
            ("RPS", _GATE_UP_ROWS),
            ("NSG", 1),
            ("TOPK", top_k),
            ("NE", plan.num_experts),
            ("M", rows),
        ],
        grid=(32, rows * (1 + top_k * inter // _GATE_UP_ROWS), 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k * inter), (rows, top_k), (rows, top_k)],
        output_dtypes=[dtype, mx.uint32, dtype],
    )
    return _down_kernel(plan.down_fmt)(
        inputs=[act, *plan.down_operands, indices, scores, h],
        template=[
            ("T", dtype),
            ("K", inter),
            ("N", hidden),
            ("TOPK", top_k),
            ("RPS", _DOWN_ROWS),
            ("M", rows),
        ],
        grid=(32, rows * top_k * hidden // _DOWN_ROWS, 1),
        threadgroup=(32, top_k, 1),
        output_shapes=[h.shape],
        output_dtypes=[dtype],
    )[0]


def _out_proj(plan, y: mx.array, resid: mx.array) -> mx.array:
    """``resid + out_proj(y)`` for ``y`` [M, H] (MLX qmv arithmetic)."""
    hidden = resid.shape[-1]
    return _out_kernel(plan.out_fmt)(
        inputs=[y, *plan.out_operands, resid],
        template=[
            ("T", plan.dtype),
            ("K", hidden),
            ("RPS", _CONV_ROWS),
            ("NSG", _CONV_SIMDGROUPS),
            ("M", resid.shape[1]),
        ],
        grid=(32, hidden // _CONV_ROWS, 1),
        threadgroup=(32, _CONV_SIMDGROUPS, 1),
        output_shapes=[resid.shape],
        output_dtypes=[plan.dtype],
    )[0]


def fused_conv(
    plan: _ConvPlan, x: mx.array, xn: mx.array, cache, n_confirmed: int = 0
) -> mx.array:
    """``x + conv(xn)`` for ``x`` [1, M, H], advancing the conv ``cache``.

    With ``0 < n_confirmed < M`` (an MTP verify block) the state after the
    confirmed prefix goes to ``cache.rollback_state``."""
    hidden, dtype = plan.hidden, plan.dtype
    rows = x.shape[1]
    state = cache[0]
    if state is None:
        state = mx.zeros((1, 2, hidden), dtype=dtype)
    split = 0 < n_confirmed < rows
    y, new_state, conf_state = _conv_kernel(plan.in_fmt)(
        inputs=[xn, *plan.in_operands, plan.conv_weight, state],
        template=[
            ("T", dtype),
            ("H", hidden),
            ("RPS", _CONV_ROWS),
            ("NSG", _CONV_SIMDGROUPS),
            ("M", rows),
            ("NC", n_confirmed if split else rows),
        ],
        grid=(32, hidden // _CONV_ROWS, 1),
        threadgroup=(32, _CONV_SIMDGROUPS, 1),
        output_shapes=[(rows, hidden), (1, 2, hidden), (1, 2, hidden)],
        output_dtypes=[dtype, dtype, dtype],
    )
    out = _out_proj(plan, y, x)
    # Only after every launch was built, so a failure leaves the cache as is.
    cache[0] = new_state
    if split:
        cache.rollback_state = (conf_state,)
    cache.advance(rows)
    return out


class _AttnPlan(NamedTuple):
    hidden: int
    dtype: mx.Dtype
    out_fmt: _Format
    out_operands: tuple


def _build_attn_plan(layer) -> _AttnPlan | None:
    attn = layer.get("self_attn")
    if attn is None:
        return None
    out_proj = attn.get("out_proj")
    if not _quantized_ok(out_proj, nn.QuantizedLinear):
        return None
    hidden = out_proj["weight"].shape[0]
    out_fmt = _mlx_format(out_proj, hidden, hidden)
    if out_fmt is None or hidden % (_CONV_ROWS * _CONV_SIMDGROUPS):
        return None
    return _AttnPlan(hidden, out_proj["scales"].dtype, out_fmt, _operands(out_proj))


def fused_attention(plan: _AttnPlan, attn, x: mx.array, xn: mx.array, cache):
    """``x + self_attn(xn)`` for ``x`` [1, M, H] (causal over the M rows)
    through the KV decode tail, or None (nothing touched) when the layout is
    not served."""
    rows = x.shape[1]
    n_heads, n_kv = attn.n_heads, attn.n_kv_heads
    q = attn.q_layernorm(attn.q_proj(xn).reshape(1, rows, n_heads, -1)).transpose(
        0, 2, 1, 3
    )
    k = attn.k_layernorm(attn.k_proj(xn).reshape(1, rows, n_kv, -1)).transpose(
        0, 2, 1, 3
    )
    if not kv_decode_tail.attend_supported(q, k):
        return None
    v = attn.v_proj(xn).reshape(1, rows, n_kv, -1).transpose(0, 2, 1, 3)
    _out_kernel(plan.out_fmt)  # build before the cache is touched
    offset = int(cache.offset)
    q = attn.rope(q, offset=offset)
    k = attn.rope(k, offset=offset)
    segments = kv_decode_tail.decode_append(cache, k, v)
    if segments is None:
        keys, values = cache.update_and_fetch(k, v)
        o = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=attn.scale, mask="causal" if rows > 1 else None
        )
    else:
        k1, v1, n1, k2, v2, n2 = segments
        o = kv_decode_tail.attend(q, k1, v1, n1, k2, v2, n2, attn.scale)
    o = o.transpose(0, 2, 1, 3).reshape(rows, -1)
    return _out_proj(plan, o, x)


def _attn_cache_ok(cache, mask) -> bool:
    from mlx_lm.models.cache import KVCache

    return (
        (mask is None or (isinstance(mask, str) and mask == "causal"))
        and type(cache) is KVCache
        and kv_decode_tail._APPLIED
        and cache.offset > 0
    )


def _served_rows(x: mx.array) -> bool:
    return (
        x.ndim == 3
        and x.shape[0] == 1
        and 1 <= x.shape[1] <= MAX_ROWS
        and x.dtype in (mx.float16, mx.bfloat16)
    )


def _conv_cache_ok(cache, mask) -> bool:
    return (
        mask is None
        and cache is not None
        and getattr(cache, "lengths", None) is None
        and getattr(cache, "left_padding", None) is None
        and (cache[0] is None or cache[0].shape[0] == 1)
    )


def _make_patched_call(original):
    def patched(self, x, mask=None, cache=None, n_confirmed=0, **kwargs):
        global _DISABLED
        if kwargs or _DISABLED or self.training or not _served_rows(x):
            if n_confirmed:
                kwargs["n_confirmed"] = n_confirmed
            return original(self, x, mask, cache, **kwargs)
        moe = cached_per_module(self, "_omlx_lfm2_moe_plan", _build_moe_plan, depth=3)
        conv = None
        if not self.is_attention_layer and _conv_cache_ok(cache, mask):
            conv = cached_per_module(
                self, "_omlx_lfm2_conv_plan", _build_conv_plan, depth=3
            )
        attn = None
        if self.is_attention_layer and _attn_cache_ok(cache, mask):
            attn = cached_per_module(
                self, "_omlx_lfm2_attn_plan", _build_attn_plan, depth=3
            )
        if moe is None and conv is None and attn is None:
            if n_confirmed:
                return original(self, x, mask, cache, n_confirmed=n_confirmed)
            return original(self, x, mask, cache)
        h = None
        try:
            if conv is not None and conv.dtype == x.dtype:
                h = fused_conv(conv, x, self.operator_norm(x), cache, n_confirmed)
            elif attn is not None and attn.dtype == x.dtype:
                h = fused_attention(
                    attn, self.self_attn, x, self.operator_norm(x), cache
                )
        except Exception:
            # Raised while building the launches (fused_conv touches the
            # cache only after its launches are built; fused_attention before
            # its append): keep serving the stock layer.
            _DISABLED = True
            logger.warning("LFM2 fused decode disabled", exc_info=True)
        if h is None:
            if self.is_attention_layer:
                h = x + self.self_attn(self.operator_norm(x), mask=mask, cache=cache)
            elif n_confirmed:
                h = x + self.conv(
                    self.operator_norm(x),
                    mask=mask,
                    cache=cache,
                    n_confirmed=n_confirmed,
                )
            else:
                h = x + self.conv(self.operator_norm(x), mask=mask, cache=cache)
        if moe is not None and moe.dtype == x.dtype and not _DISABLED:
            try:
                return fused_moe(moe, h)
            except Exception:
                _DISABLED = True
                logger.warning("LFM2 fused decode disabled", exc_info=True)
        return h + self.feed_forward(self.ffn_norm(h))

    patched._omlx_lfm2_fused_decode = True
    return patched


def apply_lfm2_moe_decode_patch() -> bool:
    """Install the fused one-token decode on mlx-lm's ``Lfm2DecoderLayer``
    (``lfm2_moe``). Idempotent; returns whether it is installed."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from mlx_lm.models import lfm2_moe
    except ImportError:
        return False
    cls = lfm2_moe.Lfm2DecoderLayer
    if not getattr(cls.__call__, "_omlx_lfm2_fused_decode", False):
        cls.__call__ = _make_patched_call(cls.__call__)
    _APPLIED = True
    logger.info("LFM2-MoE fused decode installed")
    return True


__all__ = ["MAX_ROWS", "apply_lfm2_moe_decode_patch", "fused_conv", "fused_moe"]
