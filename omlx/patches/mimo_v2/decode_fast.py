# SPDX-License-Identifier: Apache-2.0
"""MiMo V2 decode and short-verify forward (a few token rows per step).

A MiMo-V2.6-Flash decode step used to dispatch ~2.8k kernels, most of them
small element-wise ops between the weight-streaming mat-vecs, and a 2-4 row
verify forward (Lightning MTP) paid for float32 router GEMMs, one expert
pass per (row, expert) pair and unfused attention on the full-attention
layers.  For forwards of at most ``MAX_ROWS`` token rows this module runs the
same math with fewer, cheaper dispatches:

* q/k/v: one quantized mat-vec over the row-concatenated q, k and v weights
  (the separate projections become views into that buffer, so no weight
  memory is duplicated); each output row is the same quantized dot product.
* one kernel splits q/k/v, applies the partial RoPE to q and k and the value
  scale to v (MLX's rope / multiply arithmetic, element for element).
* residual add + RMSNorm, and expert combine + residual + the next layer's
  RMSNorm, as one kernel each (``mx.fast.rms_norm``'s reduction tree, experts
  summed in MLX's order).
* router: float32 logits by a multi-row kernel with MLX's M=1 ``gemv``
  arithmetic on the bf16 weights widened on load (the reference casts the
  weights to float32 every step), and one kernel for sigmoid / bias / top-k /
  normalisation.
* routed experts: ``moe_decode`` (fused gate/up/SwiGLU and down mat-vecs,
  one weight pass per distinct expert of the forward).
* attention of verify forwards whose rows x GQA factor exceed MLX's vector
  SDPA kernel (the full-attention layers at 3+ rows) runs as row chunks that
  fit it, instead of MLX's unfused matmul / softmax fallback.
* long KV caches: from ``_FLASH_MIN_KEYS`` keys every forward's attention
  runs the split-key matrix kernel ``sdpa_flash`` (float32 simdgroup MMA, all
  rows and heads of a KV head sharing each K/V tile: 2-3x MLX's vector kernel,
  which is ALU-bound at GQA 16).

Exactness: a one-row (decode) forward is bit-identical to the reference
below ``_FLASH_MIN_KEYS`` keys (or with ``OMLX_MIMO_DECODE_FLASH=0``).
For verify forwards (L > 1) two reductions run in another order than the
reference: every row's router logits are the M=1 gemv a decode step computes
(MLX's batched float32 matmul sums differently) and chunked attention rows
use the vector SDPA kernel; everything else is bit-identical.  ``sdpa_flash``
computes MLX's float32 attention in another summation order (within one bf16
ULP of MLX's kernel, closer to float64) for one-row and verify forwards
alike, so a verify row's attention matches the one-row decode's at the same
position bit for bit.  (Whole verify rows can still differ from decode steps
by rounding, here as on the reference path: MLX runs quantized matmuls of 2+
rows as ``qmv_wide`` and one-row ones as ``qmv``, and a window layer's
rotating cache hands a verify its keys in time order but a decode step in
ring order.)  KV-cache updates, the SDPA calls themselves and lm_head are the
reference code; forwards outside the fast path's contract run the reference
layer loop.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
from functools import lru_cache
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from omlx.custom_kernels.nax import is_nax_available
from omlx.patches.mimo_v2 import moe_decode, sdpa_flash

logger = logging.getLogger(__name__)

# Token rows (batch x length) the fast path handles.  Keeps the fused qkv
# projection in MLX's mat-vec regime (qmv / qmv_wide), where every output row
# is computed independently of the output width.
MAX_ROWS = 8
_RMS_MAX_AXIS = 4096  # mx.fast.rms_norm's single-row kernel limit
_SELECT_MAX_EXPERTS = 1024
_SELECT_MAX_TOPK = 32


def _env_on(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in ("0", "false", "off", "no")


def enabled() -> bool:
    """On by default on M5 (NAX) GPUs, where the fused path is validated.

    OMLX_MIMO_DECODE_FAST=1 forces it on elsewhere, =0 turns it off.
    """
    value = os.environ.get("OMLX_MIMO_DECODE_FAST", "").strip().lower()
    if value in ("0", "false", "off", "no"):
        return False
    if value in ("1", "true", "on", "yes"):
        return True
    return _nax_available()


def _nax_available() -> bool:
    try:
        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


def sdpa_chunks_enabled() -> bool:
    """Verify rows beyond the vector SDPA limit run as row chunks; on by default."""
    return _env_on("OMLX_MIMO_DECODE_SDPA_CHUNKS")


def experts_enabled() -> bool:
    """Decode-time MXFP4 expert kernels (``moe_decode``); on by default."""
    return _env_on("OMLX_MIMO_DECODE_EXPERTS")


def sdpa_flash_enabled() -> bool:
    """Long-context attention through the split-key matrix kernel
    (``sdpa_flash``: MLX's float32 math in another summation order); on by
    default."""
    return _env_on("OMLX_MIMO_DECODE_FLASH")


# Key count from which ``sdpa_flash`` serves decode / verify attention.
_FLASH_MIN_KEYS = 4096


def _flash_min_keys() -> int:
    value = os.environ.get("OMLX_MIMO_DECODE_FLASH_MIN_KEYS", "")
    try:
        return max(1, int(value)) if value else _FLASH_MIN_KEYS
    except ValueError:
        return _FLASH_MIN_KEYS


# ---------------------------------------------------------------------------
# RMSNorm reduction (transcribed from MLX rms_norm.metal ``rms_single_row``:
# RMS_N_READS = 4 values per thread, simd_sum, one threadgroup per row).
# ---------------------------------------------------------------------------

_RMS_TAIL = r"""
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_ADD_RMS_SOURCE = r"""
  constexpr int N_READS = 4;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  uint gid = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint simd_lane_id = thread_index_in_simdgroup;
  uint simd_group_id = simdgroup_index_in_threadgroup;
  const uint axis_size = AXIS;
  const float eps = static_cast<float>(eps_in[0]);
  size_t base = size_t(gid) * axis_size + lid * N_READS;

  float acc = 0;
  float thread_x[N_READS];
  for (int i = 0; i < N_READS; i++) {
    if (lid * N_READS + i < axis_size) {
      T hv = x[base + i] + y[base + i];
      h_out[base + i] = hv;
      thread_x[i] = hv;
    } else {
      thread_x[i] = 0;
    }
    acc += thread_x[i] * thread_x[i];
  }
""" + _RMS_TAIL + r"""
  for (int i = 0; i < N_READS; i++) {
    if (lid * N_READS + i < axis_size) {
      T nv = w[lid * N_READS + i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
      n_out[base + i] = nv;
      F32_STORE
    }
  }
"""

# Expert combine + residual + RMSNorm.  The reference is
#   y = (y * scores[..., None]).sum(axis=-2).astype(T); h = h + y
# i.e. float32 products (each rounded) summed k = 0, 1, ... (MLX
# col_reduce_small keeps one partial per expert row and folds them in row
# order), rounded to T once, then a T add.
_COMBINE_RMS_SOURCE = r"""
  constexpr int N_READS = 4;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  uint gid = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint simd_lane_id = thread_index_in_simdgroup;
  uint simd_group_id = simdgroup_index_in_threadgroup;
  const uint axis_size = AXIS;
  const float eps = static_cast<float>(eps_in[0]);
  size_t base = size_t(gid) * axis_size + lid * N_READS;
  size_t ybase = size_t(gid) * TOPK * axis_size + lid * N_READS;

  float acc = 0;
  float thread_x[N_READS];
  for (int i = 0; i < N_READS; i++) {
    if (lid * N_READS + i < axis_size) {
      float total = 0;
      for (int k = 0; k < TOPK; k++) {
        float p = static_cast<float>(y[ybase + size_t(k) * axis_size + i]) *
            scores[gid * TOPK + k];
        total = p + total;
      }
      T yv = static_cast<T>(total);
      T hv = h[base + i] + yv;
      h_out[base + i] = hv;
      thread_x[i] = hv;
    } else {
      thread_x[i] = 0;
    }
    acc += thread_x[i] * thread_x[i];
  }
""" + _RMS_TAIL + r"""
  for (int i = 0; i < N_READS; i++) {
    if (lid * N_READS + i < axis_size) {
      n_out[base + i] =
          w[lid * N_READS + i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
    }
  }
"""

# Router: sigmoid, + correction bias, top-k by biased score (descending,
# lower expert id first on exact ties, like MLX's stable sort behind
# argpartition), gather the unbiased scores, normalise, scale.
_SELECT_SOURCE = r"""
  threadgroup float biased[NE];
  threadgroup float picked[TOPK];
  uint row = threadgroup_position_in_grid.x;
  uint e = thread_position_in_threadgroup.x;
  float g = logits[size_t(row) * NE + e];
  float y = 1 / (1 + metal::exp(metal::abs(g)));
  float sig = (g < 0) ? y : 1 - y;
  float b = sig + bias[e];
  biased[e] = b;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint rank = 0;
  for (uint j = 0; j < NE; j++) {
    float o = biased[j];
    rank += (o > b || (o == b && j < e)) ? 1 : 0;
  }
  if (rank < TOPK) {
    inds[size_t(row) * TOPK + rank] = e;
    picked[rank] = sig;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (e < TOPK) {
    float s = picked[e];
    if (NORM) {
      float total = 0;
      for (int k = 0; k < TOPK; k++) {
        total = picked[k] + total;
      }
      s = s / (total + 1e-20f);
    }
    scores[size_t(row) * TOPK + e] = s * rsf[0];
  }
"""

# q/k/v split + partial RoPE (MLX rope.metal, non-traditional, forward) +
# value scale.  grid: x = lane (head_dim / 2), y = head (q heads, k heads,
# v heads), z = token row (b * L + t).
_ROPE_SPLIT_SOURCE = r"""
  uint lane = thread_position_in_grid.x;
  uint head = thread_position_in_grid.y;
  uint row = thread_position_in_grid.z;
  uint b = row / SEQ;
  uint t = row % SEQ;
  const device T* src = qkv + size_t(row) * NTOT;
  if (head < NQ + NKV) {
    bool is_q = head < NQ;
    uint hh = is_q ? head : head - NQ;
    const device T* in = src + (is_q ? 0 : NQ * HD) + hh * HD;
    device T* out = is_q
        ? q_out + ((size_t(b) * NQ + hh) * SEQ + t) * HD
        : k_out + ((size_t(b) * NKV + hh) * SEQ + t) * HD;
    if (lane < HALF) {
      float d = static_cast<float>(lane) / static_cast<float>(HALF);
      float inv_freq = metal::exp2(-d * base_log2[0]);
      // uint + int like MLX's rope kernel (pos.y + batch_offset).
      float L = rope_scale[0] * static_cast<float>(t + offsets[b * OFF_STRIDE]);
      float theta = L * inv_freq;
      float costheta = metal::fast::cos(theta);
      float sintheta = metal::fast::sin(theta);
      float x1 = static_cast<float>(in[lane]);
      float x2 = static_cast<float>(in[lane + HALF]);
      float rx1 = x1 * costheta - x2 * sintheta;
      float rx2 = x1 * sintheta + x2 * costheta;
      out[lane] = static_cast<T>(rx1);
      out[lane + HALF] = static_cast<T>(rx2);
    } else {
      uint c = 2 * HALF + 2 * (lane - HALF);
      if (c < HD) {
        out[c] = in[c];
        out[c + 1] = in[c + 1];
      }
    }
  } else {
    uint hh = head - NQ - NKV;
    uint c = 2 * lane;
    if (c < VD) {
      const device T* in = src + (NQ + NKV) * HD + hh * VD;
      device T* out = v_out + ((size_t(b) * NKV + hh) * SEQ + t) * VD;
      if (HAS_VSCALE) {
        T vs = vscale[0];
        out[c] = in[c] * vs;
        out[c + 1] = in[c + 1] * vs;
      } else {
        out[c] = in[c];
        out[c + 1] = in[c + 1];
      }
    }
  }
"""


# Router logits: float32 ``x @ W.T`` for W (N, K) in bf16/fp16/fp32, every row
# computed exactly like MLX's M=1 ``gemv`` (float32 matrix, K >= 16 * N:
# bm1 bn8 sm1 sn32 tm4 tn4 -- per-thread K-strided products, simd shuffle-down
# tree, then simdgroups 1..7 folded into simdgroup 0 in order).  Several rows
# share one pass over the weights.  The weights and x are widened to float32
# on load, which is exact, so this equals ``x.astype(f32) @ W.astype(f32).T``
# of the reference router for each row, and reads half the bytes.
_ROUTER_GEMV_SOURCE = r"""
  constexpr int TM = 4;
  constexpr int TN = 4;
  constexpr int SN = 32;
  constexpr int BN = 8;
  constexpr int blockM = 4;
  constexpr int blockN = BN * SN * TN;
  threadgroup float tgp_memory[ROWS * BN * (blockM + TM)];
  uint tid = threadgroup_position_in_grid.x;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;
  int thrN = simd_lid;
  int sgN = simd_gid;
  int bn = (SN * sgN + thrN) * TN;
  int out_row = tid * blockM;
  const device W* matp = mat + size_t(out_row) * KDIM;

  float result[ROWS][TM];
  for (int r = 0; r < ROWS; r++) {
    for (int tm = 0; tm < TM; tm++) {
      result[r][tm] = 0;
    }
  }
  for (int i = 0; i < KDIM / blockN; ++i) {
    float v_coeff[ROWS][TN];
    for (int r = 0; r < ROWS; r++) {
      for (int tn = 0; tn < TN; tn++) {
        v_coeff[r][tn] = static_cast<float>(vec[size_t(r) * KDIM + bn + tn]);
      }
    }
    int mat_offset = 0;
    for (int tm = 0; tm < TM; tm++) {
      float inter[TN];
      for (int tn = 0; tn < TN; tn++) {
        inter[tn] = static_cast<float>(matp[mat_offset + bn + tn]);
      }
      for (int r = 0; r < ROWS; r++) {
        for (int tn = 0; tn < TN; tn++) {
          result[r][tm] += inter[tn] * v_coeff[r][tn];
        }
      }
      mat_offset += KDIM;
    }
    bn += blockN;
  }
  for (int r = 0; r < ROWS; r++) {
    for (int tm = 0; tm < TM; tm++) {
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        result[r][tm] += simd_shuffle_down(result[r][tm], sn);
      }
    }
  }
  threadgroup float* tgp_results = tgp_memory + sgN * (blockM + TM);
  if (thrN == 0) {
    for (int r = 0; r < ROWS; r++) {
      for (int tm = 0; tm < TM; tm++) {
        tgp_results[r * BN * (blockM + TM) + tm] = result[r][tm];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgN == 0 && thrN == 0) {
    for (int r = 0; r < ROWS; r++) {
      for (int sgn = 1; sgn < BN; sgn++) {
        for (int tm = 0; tm < TM; tm++) {
          result[r][tm] += tgp_memory[r * BN * (blockM + TM) + sgn * (blockM + TM) + tm];
        }
      }
      for (int tm = 0; tm < TM; tm++) {
        out[size_t(r) * NOUT + out_row + tm] = result[r][tm];
      }
    }
  }
"""


@lru_cache(maxsize=None)
def _router_gemv_kernel():
    return mx.fast.metal_kernel(
        name="omlx_mimo_router_gemv",
        input_names=["vec", "mat"],
        output_names=["out"],
        source=_ROUTER_GEMV_SOURCE,
    )


_GEMV_BLOCK_N = 8 * 32 * 4


def router_gemv_supported(k: int, n: int) -> bool:
    # MLX picks the bm1/bn8 gemv for K >= 16 * N; K must fill whole blocks.
    return k % _GEMV_BLOCK_N == 0 and n % 4 == 0 and k >= 16 * n


def router_logits(x, weight):
    """float32 ``x @ weight.T`` per row, bit-identical to MLX's M=1 gemv on
    the float32-cast operands (see ``_ROUTER_GEMV_SOURCE``)."""
    K = int(x.shape[-1])
    N = int(weight.shape[0])
    rows = x.size // K
    (out,) = _router_gemv_kernel()(
        inputs=[x, weight],
        template=[
            ("T", x.dtype),
            ("W", weight.dtype),
            ("ROWS", int(rows)),
            ("KDIM", K),
            ("NOUT", N),
        ],
        grid=(32 * (N // 4), 8, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(*x.shape[:-1], N)],
        output_dtypes=[mx.float32],
    )
    return out


@lru_cache(maxsize=None)
def _add_rms_kernel(want_f32: bool):
    return mx.fast.metal_kernel(
        name="omlx_mimo_add_rms" + ("_f32" if want_f32 else ""),
        input_names=["x", "y", "w", "eps_in"],
        output_names=["h_out", "n_out", "f_out"] if want_f32 else ["h_out", "n_out"],
        source=_ADD_RMS_SOURCE.replace(
            "F32_STORE",
            "f_out[base + i] = static_cast<float>(nv);" if want_f32 else "",
        ),
    )


@lru_cache(maxsize=None)
def _combine_rms_kernel():
    return mx.fast.metal_kernel(
        name="omlx_mimo_combine_rms",
        input_names=["h", "y", "scores", "w", "eps_in"],
        output_names=["h_out", "n_out"],
        source=_COMBINE_RMS_SOURCE,
    )


@lru_cache(maxsize=None)
def _select_kernel():
    return mx.fast.metal_kernel(
        name="omlx_mimo_router_select",
        input_names=["logits", "bias", "rsf"],
        output_names=["inds", "scores"],
        source=_SELECT_SOURCE,
    )


@lru_cache(maxsize=None)
def _rope_split_kernel():
    return mx.fast.metal_kernel(
        name="omlx_mimo_qkv_rope_split",
        input_names=["qkv", "offsets", "base_log2", "rope_scale", "vscale"],
        output_names=["q_out", "k_out", "v_out"],
        source=_ROPE_SPLIT_SOURCE,
    )


def _rms_threads(axis: int) -> int:
    needed = (axis + 3) // 4
    return 32 * ((needed + 31) // 32)


@lru_cache(maxsize=None)
def _eps_array(eps: float):
    return mx.array([eps], dtype=mx.float32)


def add_rms(x, y, w, eps: float, want_f32: bool = False):
    """``h = x + y; (h, rms_norm(h, w, eps)[, float32 copy])`` in one kernel."""
    D = int(x.shape[-1])
    rows = x.size // D
    threads = _rms_threads(D)
    shapes = [x.shape, x.shape] + ([x.shape] if want_f32 else [])
    dtypes = [x.dtype, x.dtype] + ([mx.float32] if want_f32 else [])
    return _add_rms_kernel(want_f32)(
        inputs=[x, y, w, _eps_array(float(eps))],
        template=[("T", x.dtype), ("AXIS", D)],
        grid=(threads * rows, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )


def combine_rms(h, y, scores, w, eps: float):
    """Weighted expert sum + residual + RMSNorm.

    ``h`` (..., D), ``y`` (..., K, D) expert rows, ``scores`` (..., K) float32.
    Returns ``(h + sum_k y_k * s_k, rms_norm(that, w, eps))``.
    """
    D = int(h.shape[-1])
    K = int(y.shape[-2])
    rows = h.size // D
    threads = _rms_threads(D)
    return _combine_rms_kernel()(
        inputs=[h, y, scores.astype(mx.float32), w, _eps_array(float(eps))],
        template=[("T", h.dtype), ("AXIS", D), ("TOPK", K)],
        grid=(threads * rows, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[h.shape, h.shape],
        output_dtypes=[h.dtype, h.dtype],
    )


def router_select(logits, bias, top_k: int, norm_topk_prob: bool, scale: float):
    """noaux_tc routing for ``n_group == 1`` from float32 router logits."""
    NE = int(logits.shape[-1])
    rows = logits.size // NE
    inds, scores = _select_kernel()(
        inputs=[logits, bias, _eps_array(float(scale))],
        template=[("NE", NE), ("TOPK", int(top_k)), ("NORM", int(bool(norm_topk_prob and top_k > 1)))],
        grid=(NE * rows, 1, 1),
        threadgroup=(NE, 1, 1),
        output_shapes=[(*logits.shape[:-1], top_k), (*logits.shape[:-1], top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return inds, scores


@lru_cache(maxsize=None)
def _log2f(value: float) -> float:
    """``std::log2(float)`` as MLX's rope dispatch computes the base."""
    lib = ctypes.CDLL(ctypes.util.find_library("m") or None)
    fn = lib.log2f
    fn.restype = ctypes.c_float
    fn.argtypes = [ctypes.c_float]
    return float(fn(ctypes.c_float(value)))


@lru_cache(maxsize=None)
def _f32_scalar(value: float):
    return mx.array([value], dtype=mx.float32)


@lru_cache(maxsize=None)
def _typed_scalar(value: float, dtype):
    return mx.array([value], dtype=dtype)


def offsets_array(offset):
    """``(int32 offsets, stride)`` for the rope kernel from a cache offset."""
    if isinstance(offset, int):
        return mx.array([offset], dtype=mx.int32), 0
    offsets = offset.astype(mx.int32) if offset.dtype != mx.int32 else offset
    offsets = offsets.reshape(-1)
    return offsets, (0 if offsets.size == 1 else 1)


def qkv_rope_split(
    qkv,
    offset,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    rope_dims: int,
    rope_base: float,
    rope_scale: float = 1.0,
    v_scale: Optional[float],
):
    """Split a fused ``(B, L, H*D + Hkv*D + Hkv*Dv)`` projection into roped
    ``q (B, H, L, D)``, roped ``k (B, Hkv, L, D)`` and scaled ``v``.

    ``offset`` is the cache offset (int, or an int array with one entry per
    batch row), or an ``(array, stride)`` pair from ``offsets_array``.
    """
    B, L, NT = qkv.shape
    offsets, off_stride = (
        offset if isinstance(offset, tuple) else offsets_array(offset)
    )
    half = rope_dims // 2
    lanes = max(head_dim // 2, (v_head_dim + 1) // 2)
    heads = n_heads + 2 * n_kv_heads
    dtype = qkv.dtype
    vs = _typed_scalar(float(v_scale), dtype) if v_scale is not None else _typed_scalar(1.0, dtype)
    return _rope_split_kernel()(
        inputs=[
            qkv,
            offsets,
            _f32_scalar(_log2f(float(rope_base))),
            _f32_scalar(float(rope_scale)),
            vs,
        ],
        template=[
            ("T", dtype),
            ("SEQ", int(L)),
            ("NTOT", int(NT)),
            ("NQ", int(n_heads)),
            ("NKV", int(n_kv_heads)),
            ("HD", int(head_dim)),
            ("VD", int(v_head_dim)),
            ("HALF", int(half)),
            ("OFF_STRIDE", int(off_stride)),
            ("HAS_VSCALE", int(v_scale is not None)),
        ],
        grid=(lanes, heads, B * L),
        threadgroup=(min(lanes, 128), 1, 1),
        output_shapes=[
            (B, n_heads, L, head_dim),
            (B, n_kv_heads, L, head_dim),
            (B, n_kv_heads, L, v_head_dim),
        ],
        output_dtypes=[dtype, dtype, dtype],
    )


# ---------------------------------------------------------------------------
# Model-level fast path.
# ---------------------------------------------------------------------------


class _FusedQKV:
    """Row-concatenated q/k/v quantized weights of one attention module."""

    __slots__ = ("weight", "scales", "biases", "group_size", "bits", "mode", "views")

    def __init__(self, weight, scales, biases, group_size, bits, mode, views):
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.views = views  # the projections' weight arrays (views of `weight`)

    def current(self, attn) -> bool:
        """Whether the projections still hold the views of this buffer (a
        later weight load would replace them).  Plain dict lookups: this runs
        for every layer of every forward."""
        v = self.views
        return (
            attn["q_proj"].get("weight") is v[0]
            and attn["k_proj"].get("weight") is v[1]
            and attn["v_proj"].get("weight") is v[2]
        )


class _GateCache:
    __slots__ = ("bias32", "gemv", "source")

    def __init__(self, gate):
        n_experts, hidden = gate.weight.shape
        self.source = gate.e_score_correction_bias
        self.bias32 = self.source.astype(mx.float32)
        mx.eval(self.bias32)
        self.gemv = router_gemv_supported(int(hidden), int(n_experts))


def _gate_cache(gate) -> _GateCache:
    gc = gate.__dict__.get("_omlx_gate")
    if gc is None or gc.source is not gate.get("e_score_correction_bias"):
        gc = gate.__dict__["_omlx_gate"] = _GateCache(gate)
    return gc


def _fuse_qkv(attn) -> Optional[_FusedQKV]:
    """Concatenate q/k/v weights once and rebind the projections to views.

    The fused buffer replaces the three separate ones (the projections keep
    working unchanged on row slices of it), so memory is not duplicated.
    """
    projs = (attn.q_proj, attn.k_proj, attn.v_proj)
    if not all(isinstance(p, nn.QuantizedLinear) for p in projs):
        return None
    if any("bias" in p for p in projs):
        return None
    q = projs[0]
    mode = getattr(q, "mode", "affine")
    for p in projs[1:]:
        if (
            p.bits != q.bits
            or p.group_size != q.group_size
            or getattr(p, "mode", "affine") != mode
        ):
            return None
    has_biases = [p.get("biases") is not None for p in projs]
    if any(has_biases) and not all(has_biases):
        return None
    rows = [int(p.weight.shape[0]) for p in projs]
    weight = mx.concatenate([p.weight for p in projs], axis=0)
    scales = mx.concatenate([p.scales for p in projs], axis=0)
    biases = (
        mx.concatenate([p.biases for p in projs], axis=0) if all(has_biases) else None
    )
    mx.eval(weight, scales, biases) if biases is not None else mx.eval(weight, scales)
    start = 0
    views = []
    for p, n in zip(projs, rows):
        p.weight = weight[start : start + n]
        p.scales = scales[start : start + n]
        if biases is not None:
            p.biases = biases[start : start + n]
        views.extend(a for a in (p.weight, p.scales, p.get("biases")) if a is not None)
        start += n
    mx.eval(views)
    return _FusedQKV(
        weight, scales, biases, q.group_size, q.bits, mode, tuple(p.weight for p in projs)
    )


_SWIGLU_MODULES = ("mlx_lm.models.switch_layers", "omlx.patches.glm_moe_dsa.switch_layers")


def _expert_kind(switch_mlp) -> Optional[str]:
    """``"split"`` / ``"fused"`` when the decode expert kernels reproduce this
    ``SwitchGLU`` (MXFP4 gs32 experts, SwiGLU activation), else ``None``."""
    if (
        type(switch_mlp).__name__ != "SwitchGLU"
        or type(switch_mlp).__module__ not in _SWIGLU_MODULES
    ):
        return None  # e.g. the expert-offload wrapper
    act = getattr(switch_mlp, "activation", None)
    if type(act).__name__ != "SwiGLU" or type(act).__module__ not in _SWIGLU_MODULES:
        return None
    down = getattr(switch_mlp, "down_proj", None)
    if down is None:
        return None
    inter = int(down.weight.shape[-1]) * 8
    hidden = int(down.weight.shape[1])
    if not moe_decode.supported(down, inter, hidden):
        return None
    if "gate_up_proj" in switch_mlp:
        gu = switch_mlp.gate_up_proj
        if int(gu.weight.shape[1]) != 2 * inter or not moe_decode.supported(gu, hidden, inter):
            return None
        return "fused"
    gate = getattr(switch_mlp, "gate_proj", None)
    up = getattr(switch_mlp, "up_proj", None)
    if gate is None or up is None:
        return None
    for p in (gate, up):
        if int(p.weight.shape[1]) != inter or not moe_decode.supported(p, hidden, inter):
            return None
    return "split"


def _expert_layout(mlp) -> Optional[str]:
    """``_expert_kind`` of ``mlp.switch_mlp``, cached against the identity of
    the modules it was derived from (a later regroup re-derives it)."""
    sw = mlp["switch_mlp"]
    get = getattr(sw, "get", None)
    key = (
        (id(sw), id(get("gate_up_proj")), id(get("gate_proj")), id(get("up_proj")), id(get("down_proj")))
        if get is not None
        else (id(sw),)
    )
    cached = mlp.__dict__.get("_omlx_experts")
    if cached is None or cached[1] != key:
        cached = mlp.__dict__["_omlx_experts"] = (_expert_kind(sw), key)
    return cached[0]


def _experts(switch_mlp, kind, x, inds):
    """Per-(row, expert) down-projected rows, ``(..., top_k, hidden)``."""
    down = switch_mlp.down_proj
    inter = int(down.weight.shape[-1]) * 8
    if kind == "fused":
        gu = switch_mlp.gate_up_proj
        gw = uw = gu.weight
        gs = us = gu.scales
        up_offset = inter
    else:
        gw, gs = switch_mlp.gate_proj.weight, switch_mlp.gate_proj.scales
        uw, us = switch_mlp.up_proj.weight, switch_mlp.up_proj.scales
        up_offset = 0
    act = moe_decode.gate_up_swiglu(x, inds, gw, gs, uw, us, n_out=inter, up_offset=up_offset)
    return moe_decode.down_proj(act, inds, down.weight, down.scales)


def _prepare(model) -> bool:
    """One-time per model: fuse q/k/v weights, check the RoPE / router /
    expert layouts the fast path supports and cache the float32 router bias."""
    state = model.__dict__.get("_omlx_decode_fast_state")
    if state is not None:
        return state
    ok = True
    try:
        for layer in model.layers:
            attn = layer.self_attn
            rope = attn.rope
            if type(rope) is not nn.RoPE or rope.traditional:
                ok = False
                break
            if attn.__dict__.get("_omlx_qkv") is None:
                attn.__dict__["_omlx_qkv"] = _fuse_qkv(attn) or False
            mlp = layer.mlp
            gate = getattr(mlp, "gate", None)
            if gate is None:
                continue
            n_experts, hidden = gate.weight.shape
            if (
                getattr(mlp, "sharding_group", None) is not None
                or gate.n_group != 1
                or gate.top_k > _SELECT_MAX_TOPK
                or n_experts > _SELECT_MAX_EXPERTS
                or n_experts % 32 != 0
            ):
                ok = False
                break
            _expert_layout(mlp)
            _gate_cache(gate)
    except Exception:  # noqa: BLE001 - never break the reference forward
        logger.warning("MiMo decode fast path disabled", exc_info=True)
        ok = False
    model.__dict__["_omlx_decode_fast_state"] = ok
    if ok:
        logger.info("MiMo decode fast path armed (fused qkv, fused norms, router, experts)")
    return ok


# MLX's vector SDPA kernel serves query rows x GQA factor <= 32; beyond that
# (MiMo's full-attention layers, GQA 16, at 3+ verify rows) it falls back to
# unfused matmul + softmax + matmul, several kernels per layer.
_SDPA_VECTOR_ROWS = 32


def _sdpa_row_chunks(sdpa, q, k, v, cache, scale, mask, sinks, rows):
    """Attention of ``q (B, H, L, D)`` in chunks of ``rows`` query rows, each
    within the vector kernel's limit; returns ``(B, L, H * Dv)``.

    Every chunk keeps the full forward's causal structure: a ``"causal"``
    mask becomes a key prefix ending at the chunk's last row, an array mask
    is sliced to the chunk's rows.
    """
    L = q.shape[2]
    S = k.shape[2]
    outs = []
    for r0 in range(0, L, rows):
        r1 = min(L, r0 + rows)
        kc, vc, mc = k, v, mask
        if isinstance(mask, str):
            end = S - (L - r1)
            kc, vc = k[:, :, :end], v[:, :, :end]
        elif mask is not None and mask.ndim >= 2 and mask.shape[-2] == L:
            mc = mask[..., r0:r1, :]
        o = sdpa(q[:, :, r0:r1], kc, vc, cache=cache, scale=scale, mask=mc, sinks=sinks)
        outs.append(o.swapaxes(1, 2))
    out = mx.concatenate(outs, axis=1)
    return out.reshape(out.shape[0], L, -1)


# Attention functions that compute MLX's fused SDPA for the fast path's calls
# (at most MAX_ROWS query rows, unquantized caches): mlx_lm's own and oMLX's
# routing wrappers, which pass such calls through to it unchanged.
_MLX_SDPA = {
    "mlx_lm.models.base": ("scaled_dot_product_attention",),
    "omlx.patches.sdpa256_attention": ("patched_sdpa",),
    "omlx.patches.qwen35_fa256_attention": ("patched_lm_sdpa",),
    "omlx.patches.turboquant_attention": ("patched_sdpa",),
}


def _is_mlx_sdpa(fn) -> bool:
    names = _MLX_SDPA.get(getattr(fn, "__module__", None) or "")
    return bool(names) and getattr(fn, "__name__", "") in names


def _vector_attention(q, k, v, cache, scale, mask, sinks, kernels):
    """``(B, L, H * Dv)`` attention of a short forward from the one-pass
    kernels, or ``None`` (the caller keeps MLX's SDPA).

    ``kernels`` is ``(flash, flash_min_keys)``.  From ``flash_min_keys`` keys
    ``sdpa_flash`` serves every row count, so a verify row's attention matches
    the one-row decode's.
    """
    flash_on, flash_min = kernels
    if not flash_on or cache is None or hasattr(cache, "bits"):
        return None
    inner = getattr(cache, "_cache", None)
    if inner is not None and hasattr(inner, "bits"):
        return None
    if not isinstance(k, mx.array) or not isinstance(v, mx.array) or k.ndim != 4:
        return None
    if k.shape[2] < flash_min:
        return None
    return sdpa_flash.sdpa_flash(q, k, v, scale, mask, sinks)


def _attention(attn, x, mask, cache, sdpa, offsets_memo, row_chunks, kernels=(False, 0)):
    B, L, _ = x.shape
    fused = attn.__dict__.get("_omlx_qkv")
    if fused and not fused.current(attn):
        fused = attn.__dict__["_omlx_qkv"] = _fuse_qkv(attn) or False
    offset = cache.offset
    if fused:
        # Full-attention and sliding-window caches share offsets per step:
        # build each int offset's device array once per forward.
        if isinstance(offset, int):
            rope_offset = offsets_memo.get(offset)
            if rope_offset is None:
                rope_offset = offsets_memo[offset] = offsets_array(offset)
        else:
            rope_offset = offsets_array(offset)
        qkv = mx.quantized_matmul(
            x,
            fused.weight,
            fused.scales,
            fused.biases,
            transpose=True,
            group_size=fused.group_size,
            bits=fused.bits,
            mode=fused.mode,
        )
        rope = attn.rope
        queries, keys, values = qkv_rope_split(
            qkv,
            rope_offset,
            n_heads=attn.n_heads,
            n_kv_heads=attn.n_kv_heads,
            head_dim=attn.head_dim,
            v_head_dim=attn.v_head_dim,
            rope_dims=rope.dims,
            rope_base=rope.base,
            rope_scale=rope.scale,
            v_scale=attn.v_scale,
        )
    else:
        queries = attn.q_proj(x).reshape(B, L, attn.n_heads, attn.head_dim).swapaxes(1, 2)
        keys = attn.k_proj(x).reshape(B, L, attn.n_kv_heads, attn.head_dim).swapaxes(1, 2)
        values = (
            attn.v_proj(x).reshape(B, L, attn.n_kv_heads, attn.v_head_dim).swapaxes(1, 2)
        )
        if attn.v_scale is not None:
            values = values * attn.v_scale
        queries = attn.rope(queries, offset=offset)
        keys = attn.rope(keys, offset=offset)
    keys, values = cache.update_and_fetch(keys, values)
    output = _vector_attention(
        queries, keys, values, cache, attn.scale, mask, attn.attention_sink_bias, kernels
    )
    if output is not None:
        return attn.o_proj(output)
    n_rep = max(1, attn.n_heads // attn.n_kv_heads)
    if row_chunks and L > 1 and L * n_rep > _SDPA_VECTOR_ROWS and n_rep <= _SDPA_VECTOR_ROWS:
        output = _sdpa_row_chunks(
            sdpa,
            queries,
            keys,
            values,
            cache,
            attn.scale,
            mask,
            attn.attention_sink_bias,
            _SDPA_VECTOR_ROWS // n_rep,
        )
        return attn.o_proj(output)
    output = sdpa(
        queries,
        keys,
        values,
        cache=cache,
        scale=attn.scale,
        mask=mask,
        sinks=attn.attention_sink_bias,
    )
    return attn.o_proj(output.swapaxes(1, 2).reshape(B, L, -1))


def _max_rows(model) -> int:
    rows = model.__dict__.get("_omlx_decode_fast_rows")
    if rows is None:
        top_k = max(
            (
                layer.mlp.gate.top_k
                for layer in model.layers
                if hasattr(layer.mlp, "gate")
            ),
            default=1,
        )
        # Keep SwitchGLU on its unsorted path (indices.size < 64), whose
        # expert rows the combine kernel sums in the reference order.
        rows = max(0, min(MAX_ROWS, (64 - 1) // max(1, top_k)))
        model.__dict__["_omlx_decode_fast_rows"] = rows
    return rows


def run_layers(model, h, cache, full_mask, swa_mask):
    """Decoder stack for a short forward; ``(h, norm(h))`` or ``None``.

    ``model`` is the inner ``MiMoV2Model``; ``h`` the input embeddings.
    Returns ``None`` (caller runs the reference loop) for anything outside
    the fast path's contract.
    """
    if not enabled() or h.ndim != 3:
        return None
    B, L, D = h.shape
    if B * L > _max_rows(model) or h.dtype not in (mx.bfloat16, mx.float16):
        return None
    if D % 4 != 0 or D > _RMS_MAX_AXIS:
        return None
    if any(c is None or hasattr(c, "bits") for c in cache):
        return None
    # The fused q/k/v path reads RoPE parameters directly, so a wrapped rope
    # (SpecPrefill's position-mapped or offset-adjusted RoPE) runs the reference.
    if any(type(layer.self_attn.rope) is not nn.RoPE for layer in model.layers):
        return None
    if not _prepare(model):
        return None
    layers = model.layers
    # The attention function the reference Attention.__call__ resolves (SDPA
    # patches rebind the model module's global, not only mlx_lm's base).
    sdpa = getattr(
        sys.modules.get(type(layers[0].self_attn).__module__),
        "scaled_dot_product_attention",
        None,
    )
    if sdpa is None:
        return None

    first = layers[0].input_layernorm
    x = mx.fast.rms_norm(h, first.weight, first.eps)
    n = len(layers)
    offsets_memo = {}
    use_experts = experts_enabled()
    row_chunks = sdpa_chunks_enabled()
    # sdpa_flash replaces MLX's own SDPA only; a patched attention function
    # bound in the model module keeps serving its calls.
    kernels = (False, 0)
    if _is_mlx_sdpa(sdpa):
        kernels = (sdpa_flash_enabled(), _flash_min_keys())
    for i, layer in enumerate(layers):
        nxt = layers[i + 1].input_layernorm if i + 1 < n else model.norm
        mask = swa_mask if layer.is_sliding_window else full_mask
        a = _attention(
            layer.self_attn, x, mask, cache[i], sdpa, offsets_memo, row_chunks, kernels
        )
        post = layer.post_attention_layernorm
        mlp = layer.mlp
        gate = getattr(mlp, "gate", None)
        if gate is None:
            h, xm = add_rms(h, a, post.weight, post.eps)
            h, x = add_rms(h, mlp(xm), nxt.weight, nxt.eps)
            continue
        h, xm = add_rms(h, a, post.weight, post.eps)
        gc = _gate_cache(gate)
        if gc.gemv:
            # Each row's logits equal the reference router's M=1 (decode) gemv.
            logits = router_logits(xm, gate.weight)
        else:
            logits = xm.astype(mx.float32) @ gate.weight.astype(mx.float32).T
        inds, scores = router_select(
            logits, gc.bias32, gate.top_k, gate.norm_topk_prob, gate.routed_scaling_factor
        )
        kind = _expert_layout(mlp) if use_experts else None
        if kind:
            y = _experts(mlp.switch_mlp, kind, xm, inds)
        else:
            y = mlp.switch_mlp(xm, inds)
        h, x = combine_rms(h, y, scores, nxt.weight, nxt.eps)
    return h, x
