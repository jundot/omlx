# SPDX-License-Identifier: Apache-2.0
"""Split-key ("flash decoding") vector SDPA with simdgroup matrix products.

For decode / MTP-verify forwards of a few query rows over a long KV cache.
MLX's vector kernel computes every (query row, head, key) score with a
32-lane ``simd_sum`` and runs the online softmax on all 32 lanes; at GQA 16
that is ALU-bound at ~1/3 of DRAM bandwidth for one row, and each extra row
costs as much again.  Here the ``G x L`` query rows of a KV head ("qrows",
``qrow = row * G + head``) are processed as 8-qrow bands with 8x8 float32
simdgroup matrix products:

* q is scaled in float32 exactly as MLX does (``float(scale) * float(q)``),
  K and V are widened from bf16/fp16 exactly; every product and sum is float32
  (the M5 GPUs' float32 simdgroup MMA is full precision).
* each threadgroup owns a contiguous chunk of keys (a split) for one KV head,
  stages 8-key K/V tiles in threadgroup memory for all bands, and runs the
  online softmax per tile (``fast::exp``, float32); the per-split partial
  output stays float32 (MLX rounds its per-block partials to the input dtype).
  One-row forwards give each band two simdgroups (head- and value-dim halves,
  scores exchanged through threadgroup memory) for more threads per tile.
* a second pass merges the splits (MLX's ``sdpa_vector_2pass_2`` recipe).
* chunk size and split count are runtime values: the kernels compile once
  per (row count, mask kind), not per key count.

Arithmetic vs MLX: the same float32 math in another summation order -- the
dot products are summed by the matrix unit as (first head-dim half) +
(second half) instead of lane partials + ``simd_sum``, the running max is
taken per 8-key tile instead of per key, the key splits are contiguous chunks
instead of MLX's strided blocks, and the partials are not rounded to bf16.
Every qrow's sequence of operations depends only on the key count (through
the chunk size), never on the number of query rows, the band layout or the
other qrows, so a verify row equals the one-row decode at the same position
bit for bit, except where the two key counts straddle a chunk-size step.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Optional

import mlx.core as mx

MAX_ROWS = 4
TILE_KEYS = 8

# Pass 1: grid (KV heads, batch, splits) threadgroups of one simdgroup per
# 8-qrow band.  Per TILE-key tile: stage K and V, S = Q' K^T (MMA over the
# head dim), mask, online softmax, O += P V (MMA).  Writes the unnormalized
# float32 O, the running max and the running sum per (qrow, split).
_PASS1_SOURCE = r"""
  constexpr int TK = TILE;
  constexpr int QB = (G * ROWS) / 8;
  constexpr int DK = D / 8;
  constexpr int DVB = V / 8;
  constexpr int DKH = DK / 2;     // head-dim blocks per half
  constexpr int DKQ = DK / HS;    // head-dim blocks held by this simdgroup
  constexpr int DVH = DVB / HS;   // value-dim blocks of this simdgroup
  constexpr int KLD = D + 8;
  constexpr int VLD = V + 8;
  constexpr int NT = QB * HS * 32;
  typedef float U;

  threadgroup T ktile[TK * KLD];
  threadgroup T vtile[TK * VLD];
  threadgroup float sx[HS > 1 ? QB * 2 * (TK / 8) * 64 : 1];

  const int kh = threadgroup_position_in_grid.x;
  const int b = threadgroup_position_in_grid.y;
  const int split = threadgroup_position_in_grid.z;
  const int band = simdgroup_index_in_threadgroup % QB;
  const int hs = simdgroup_index_in_threadgroup / QB;
  const int lane = thread_index_in_simdgroup;
  const int tid = thread_index_in_threadgroup;
  const int N = keys_shape[2];
  const int nsplit = threadgroups_per_grid.z;
  const int chunk = params[0];
  const int s0 = split * chunk;
  const int s1 = min(N, s0 + chunk);
  const int r = (band * 8) / G;
  const int g0 = (band * 8) % G;
  // 8x8 fragment element owned by this lane: row fm, columns fn and fn + 1.
  const int qid = lane / 4;
  const int fm = (qid & 4) + ((lane / 2) % 4);
  const int fn = (qid & 2) * 2 + (lane % 2) * 2;
  const int H = NKV * G;

  // Q band (rows: heads g0 .. g0 + 7 of query row r), q' = scale * q.
  simdgroup_matrix<U, 8, 8> Qf[DKQ];
  const U sc = scale[0];
  {
    const int h = kh * G + g0 + fm;
    const device T* qp = queries + b * queries_strides[0] + h * queries_strides[1] +
        r * queries_strides[2];
    const int64_t q3 = queries_strides[3];
    _Pragma("clang loop unroll(full)")
    for (int kd = 0; kd < DKQ; kd++) {
      const int d0 = (hs * DKQ + kd) * 8;
      U a0 = static_cast<U>(sc) * qp[(d0 + fn) * q3];
      U a1 = static_cast<U>(sc) * qp[(d0 + fn + 1) * q3];
      reinterpret_cast<thread vec<U, 2>&>(Qf[kd].thread_elements()) = vec<U, 2>(a0, a1);
    }
  }

  U m_run = Limits<U>::finite_min;
  U l_run = 0;
  if (HAS_SINKS && split == 0) {
    m_run = static_cast<U>(sinks[kh * G + g0 + fm]);
    l_run = 1;
  }
  simdgroup_matrix<U, 8, 8> Of[DVH];
  _Pragma("clang loop unroll(full)")
  for (int dv = 0; dv < DVH; dv++) {
    Of[dv] = simdgroup_matrix<U, 8, 8>(0);
  }

  const int64_t ks = keys_strides[2];
  const int64_t vs = values_strides[2];
  const int64_t k3 = keys_strides[3];
  const int64_t v3 = values_strides[3];
  const device T* kbase = keys + b * keys_strides[0] + kh * keys_strides[1];
  const device T* vbase = values + b * values_strides[0] + kh * values_strides[1];
  auto mrow = mask + (MASK_KIND ? (size_t(b) * ROWS + r) * N : 0);
  const int causal_limit = N - ROWS + r;

  const bool kvec = k3 == 1 && (ks % 8) == 0 && (keys_strides[1] % 8) == 0 &&
      (keys_strides[0] % 8) == 0;
  const bool vvec = v3 == 1 && (vs % 8) == 0 && (values_strides[1] % 8) == 0 &&
      (values_strides[0] % 8) == 0;
  const bool vecload = kvec && vvec;
  constexpr int KCH = TK * (D / 8);
  constexpr int VCH = TK * (V / 8);
  constexpr int CPT = (KCH + VCH + NT - 1) / NT;

  // This thread's 16-byte staging chunks: fixed tile slots, source pointers
  // advanced by one tile per step.
  const device T* csrc[CPT];
  threadgroup T* cdst[CPT];
  int crow[CPT];
  int64_t cstep[CPT];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < CPT; j++) {
    const int c = tid + j * NT;
    const bool isk = c < KCH;
    const int cc = isk ? c : c - KCH;
    const int per = isk ? (D / 8) : (V / 8);
    const int row = cc / per;
    const int col = (cc % per) * 8;
    crow[j] = c < KCH + VCH ? row : TK;  // TK: no chunk
    cdst[j] = isk ? (ktile + row * KLD + col) : (vtile + row * VLD + col);
    csrc[j] = isk ? (kbase + (s0 + row) * ks + col * k3) : (vbase + (s0 + row) * vs + col * v3);
    cstep[j] = TK * (isk ? ks : vs);
  }

  for (int t0 = s0; t0 < s1; t0 += TK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool full = t0 + TK <= s1;
    if (vecload) {
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < CPT; j++) {
        if (crow[j] < TK) {
          if (full || t0 + crow[j] < s1) {
            *reinterpret_cast<threadgroup vec<T, 8>*>(cdst[j]) =
                *reinterpret_cast<const device vec<T, 8>*>(csrc[j]);
          } else {
            *reinterpret_cast<threadgroup vec<T, 8>*>(cdst[j]) = vec<T, 8>(0);
          }
          csrc[j] += cstep[j];
        }
      }
    } else {
      for (int c = tid; c < KCH + VCH; c += NT) {
        const bool isk = c < KCH;
        const int cc = isk ? c : c - KCH;
        const int per = isk ? (D / 8) : (V / 8);
        const int row = cc / per;
        const int col = (cc % per) * 8;
        const int key = t0 + row;
        threadgroup T* dst = isk ? (ktile + row * KLD + col) : (vtile + row * VLD + col);
        const device T* src = isk ? (kbase + key * ks + col * k3) : (vbase + key * vs + col * v3);
        const int64_t e3 = isk ? k3 : v3;
        for (int e = 0; e < 8; e++) {
          dst[e] = key < s1 ? src[e * e3] : T(0);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q' K^T for the tile: TK / 8 fragments (rows: qrows, cols: keys),
    // summed as (head-dim first half) + (second half) whether one simdgroup
    // holds both halves (HS 1) or two simdgroups one each (HS 2).
    simdgroup_matrix<U, 8, 8> S[TK / 8];
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < TK / 8; kk++) {
      S[kk] = simdgroup_matrix<U, 8, 8>(0);
    }
    if (HS == 1) {
      simdgroup_matrix<U, 8, 8> S2[TK / 8];
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < TK / 8; kk++) {
        S2[kk] = simdgroup_matrix<U, 8, 8>(0);
      }
      _Pragma("clang loop unroll(full)")
      for (int kd = 0; kd < DKH; kd++) {
        _Pragma("clang loop unroll(full)")
        for (int kk = 0; kk < TK / 8; kk++) {
          simdgroup_matrix<T, 8, 8> Kt;
          simdgroup_load(Kt, ktile + kk * 8 * KLD + kd * 8, KLD, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(S[kk], Qf[kd], Kt, S[kk]);
          simdgroup_matrix<T, 8, 8> Kt2;
          simdgroup_load(Kt2, ktile + kk * 8 * KLD + (DKH + kd) * 8, KLD, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(S2[kk], Qf[DKH + kd], Kt2, S2[kk]);
        }
      }
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < TK / 8; kk++) {
        reinterpret_cast<thread vec<U, 2>&>(S[kk].thread_elements()) +=
            reinterpret_cast<thread vec<U, 2>&>(S2[kk].thread_elements());
      }
    } else {
      _Pragma("clang loop unroll(full)")
      for (int kd = 0; kd < DKH; kd++) {
        _Pragma("clang loop unroll(full)")
        for (int kk = 0; kk < TK / 8; kk++) {
          simdgroup_matrix<T, 8, 8> Kt;
          simdgroup_load(Kt, ktile + kk * 8 * KLD + (hs * DKH + kd) * 8, KLD, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(S[kk], Qf[kd], Kt, S[kk]);
        }
      }
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < TK / 8; kk++) {
        simdgroup_store(S[kk], sx + ((band * HS + hs) * (TK / 8) + kk) * 64, 8);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < TK / 8; kk++) {
        vec<U, 2> acc = *reinterpret_cast<const threadgroup vec<U, 2>*>(
            sx + ((band * HS) * (TK / 8) + kk) * 64 + fm * 8 + fn);
        acc += *reinterpret_cast<const threadgroup vec<U, 2>*>(
            sx + ((band * HS + 1) * (TK / 8) + kk) * 64 + fm * 8 + fn);
        reinterpret_cast<thread vec<U, 2>&>(S[kk].thread_elements()) = acc;
      }
    }

    // Mask (MLX's rules), tile max of row fm (lanes 1 and 8 apart share it).
    U rmax = Limits<U>::finite_min;
    const bool clean = MASK_KIND == 0 && (t0 + TK <= s1) &&
        (!CAUSAL || t0 + TK - 1 <= causal_limit);
    if (clean) {
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < TK / 8; kk++) {
        thread vec<U, 2>& e = reinterpret_cast<thread vec<U, 2>&>(S[kk].thread_elements());
        rmax = max(rmax, e[0]);
        rmax = max(rmax, e[1]);
      }
    } else {
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < TK / 8; kk++) {
        thread vec<U, 2>& e = reinterpret_cast<thread vec<U, 2>&>(S[kk].thread_elements());
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < 2; j++) {
          const int key = t0 + kk * 8 + fn + j;
          bool use_key = key < s1;
          if (CAUSAL) {
            use_key = use_key && key <= causal_limit;
          } else if (MASK_KIND == 1) {
            use_key = use_key && mrow[key];
          } else if (MASK_KIND == 2) {
            use_key = use_key && (mrow[key] >= Limits<T>::finite_min);
          }
          U s = e[j];
          if (MASK_KIND == 2 && use_key) {
            s += mrow[key];
          }
          s = use_key ? s : -INFINITY;
          e[j] = s;
          rmax = max(rmax, s);
        }
      }
    }
    rmax = max(rmax, simd_shuffle_xor(rmax, 1));
    rmax = max(rmax, simd_shuffle_xor(rmax, 8));
    const U m_new = max(m_run, rmax);
    const U factor = fast::exp(m_run - m_new);
    U rsum = 0;
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < TK / 8; kk++) {
      thread vec<U, 2>& e = reinterpret_cast<thread vec<U, 2>&>(S[kk].thread_elements());
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 2; j++) {
        U p = fast::exp(e[j] - m_new);
        e[j] = p;
        rsum += p;
      }
    }
    rsum += simd_shuffle_xor(rsum, 1);
    rsum += simd_shuffle_xor(rsum, 8);
    // A row whose max did not move has factor 1: skipping its rescale is exact.
    const bool moved = simd_any(m_new != m_run);
    l_run = l_run * factor + rsum;
    m_run = m_new;
    if (moved) {
      _Pragma("clang loop unroll(full)")
      for (int dv = 0; dv < DVH; dv++) {
        thread vec<U, 2>& e = reinterpret_cast<thread vec<U, 2>&>(Of[dv].thread_elements());
        e *= factor;
      }
    }

    // O += P V.
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < TK / 8; kk++) {
      _Pragma("clang loop unroll(full)")
      for (int dv = 0; dv < DVH; dv++) {
        simdgroup_matrix<T, 8, 8> Vf;
        simdgroup_load(Vf, vtile + kk * 8 * VLD + (hs * DVH + dv) * 8, VLD);
        simdgroup_multiply_accumulate(Of[dv], S[kk], Vf, Of[dv]);
      }
    }
  }

  const int prow0 = ((b * ROWS + r) * H + kh * G + g0);
  device U* op = o_part + (size_t(prow0) * nsplit + split) * V + hs * DVH * 8;
  _Pragma("clang loop unroll(full)")
  for (int dv = 0; dv < DVH; dv++) {
    simdgroup_store(Of[dv], op + dv * 8, ulong(nsplit) * V);
  }
  if (hs == 0 && (lane & 9) == 0) {
    const size_t idx = size_t(prow0 + fm) * nsplit + split;
    m_part[idx] = m_run;
    l_part[idx] = l_run;
  }
"""

# Pass 2: merge the splits of each (batch, row, head): one thread per output
# element, splits folded in order.
_PASS2_SOURCE = r"""
  typedef float U;
  const int prow = threadgroup_position_in_grid.x;   // (b * ROWS + r) * H + h
  const int d = thread_position_in_threadgroup.x;
  const int nsplit = m_part_shape[1];
  const device U* mp = m_part + size_t(prow) * nsplit;
  const device U* lp = l_part + size_t(prow) * nsplit;
  const device U* op = o_part + size_t(prow) * nsplit * V + d;
  U m = Limits<U>::finite_min;
  for (int s = 0; s < nsplit; s++) {
    m = max(m, mp[s]);
  }
  U l = 0;
  U o = 0;
  for (int s = 0; s < nsplit; s++) {
    U w = fast::exp(mp[s] - m);
    l += w * lp[s];
    o += w * op[s * V];
  }
  o = l == 0 ? o : (o / l);
  out[size_t(prow) * V + d] = static_cast<T>(o);
"""


@lru_cache(maxsize=None)
def _pass1_kernel():
    return mx.fast.metal_kernel(
        name="omlx_sdpa_flash_pass1",
        input_names=["queries", "keys", "values", "scale", "params", "mask", "sinks"],
        output_names=["o_part", "m_part", "l_part"],
        source=_PASS1_SOURCE,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=None)
def _pass2_kernel():
    return mx.fast.metal_kernel(
        name="omlx_sdpa_flash_pass2",
        input_names=["o_part", "m_part", "l_part"],
        output_names=["out"],
        source=_PASS2_SOURCE,
    )


def chunk_size(n_keys: int) -> int:
    """Keys per split: ``n_keys / 128`` rounded up to a power of two, in
    [256, 1024] (measured best on M5 Ultra from 8k to 1M keys).  The same for
    every row count, so a verify row matches the one-row decode at its
    position unless the two key counts straddle a power-of-two step."""
    env = os.environ.get("OMLX_SDPA_FLASH_CHUNK", "")
    if env:
        try:
            return max(TILE_KEYS, int(env) // TILE_KEYS * TILE_KEYS)
        except ValueError:
            pass
    c = 1 << max(0, math.ceil(math.log2(max(1, n_keys) / 128)))
    return max(256, min(1024, c))


def _head_split(rows: int) -> int:
    """Simdgroups per 8-qrow band: one-row forwards split each band's head
    and value dims over two simdgroups (more threads per key tile), multi-row
    forwards keep one; the arithmetic is the same either way."""
    env = os.environ.get("OMLX_SDPA_FLASH_HS", "")
    if env in ("1", "2"):
        return int(env)
    return 2 if rows == 1 else 1


@lru_cache(maxsize=None)
def _params_array(chunk: int):
    return mx.array([chunk], dtype=mx.int32)


@lru_cache(maxsize=None)
def _scale_array(scale: float):
    return mx.array([scale], dtype=mx.float32)


_DUMMY = {}


def _dummy(dtype):
    arr = _DUMMY.get(dtype)
    if arr is None:
        arr = _DUMMY[dtype] = mx.zeros((1,), dtype=dtype)
    return arr


def sdpa_flash(q, k, v, scale: float, mask, sinks) -> Optional[mx.array]:
    """Attention of ``q (B, H, L, D)`` over ``k (B, Hk, S, D)`` /
    ``v (B, Hk, S, Dv)`` for ``L <= MAX_ROWS`` rows, returned as
    ``(B, L, H * Dv)``.

    ``mask`` is ``None``, ``"causal"`` or a bool / additive array broadcastable
    to ``(B, 1, L, S)``; ``sinks`` per query head, like MLX's SDPA.  Returns
    ``None`` when the call is outside this kernel's contract.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    B, H, L, D = q.shape
    Hk, S = k.shape[1], k.shape[2]
    Dv = v.shape[3]
    if not (1 <= L <= MAX_ROWS) or k.shape[0] != B or v.shape[0] != B:
        return None
    if v.shape[1] != Hk or v.shape[2] != S or k.shape[3] != D or Hk == 0 or H % Hk:
        return None
    if D % 16 or Dv % 16 or D > 256 or Dv > 256 or L > S:
        return None
    dtype = q.dtype
    if dtype not in (mx.bfloat16, mx.float16) or k.dtype != dtype or v.dtype != dtype:
        return None
    gqa = H // Hk
    if gqa % 8:
        return None
    causal = isinstance(mask, str)
    if causal and mask != "causal":
        return None
    mask_kind = 0
    mask_arr = _dummy(mx.bool_)
    if mask is not None and not causal:
        if not isinstance(mask, mx.array) or mask.ndim < 1 or mask.ndim > 4:
            return None
        m = mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))
        if m.shape[1] != 1:
            return None
        try:
            m = mx.broadcast_to(m, (B, 1, L, S))
        except ValueError:
            return None
        if m.dtype == mx.bool_:
            mask_kind = 1
        else:
            mask_kind = 2
            m = m.astype(dtype)
        mask_arr = mx.contiguous(m.reshape(B, L, S))
    sinks_arr = _dummy(dtype)
    if sinks is not None:
        if sinks.ndim != 1 or sinks.shape[0] != H:
            return None
        sinks_arr = sinks.astype(dtype)

    chunk = chunk_size(S)
    splits = -(-S // chunk)
    qb = gqa * L // 8
    hs = _head_split(L)
    o_part, m_part, l_part = _pass1_kernel()(
        inputs=[q, k, v, _scale_array(float(scale)), _params_array(int(chunk)), mask_arr, sinks_arr],
        template=[
            ("T", dtype),
            ("D", int(D)),
            ("V", int(Dv)),
            ("G", int(gqa)),
            ("NKV", int(Hk)),
            ("ROWS", int(L)),
            ("TILE", TILE_KEYS),
            ("HS", hs),
            ("CAUSAL", bool(causal and L > 1)),
            ("MASK_KIND", int(mask_kind)),
            ("HAS_SINKS", sinks is not None),
        ],
        grid=(32 * qb * hs * Hk, B, splits),
        threadgroup=(32 * qb * hs, 1, 1),
        output_shapes=[(B * L * H, splits, Dv), (B * L * H, splits), (B * L * H, splits)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = _pass2_kernel()(
        inputs=[o_part, m_part, l_part],
        template=[("T", dtype), ("V", int(Dv))],
        grid=(Dv * B * L * H, 1, 1),
        threadgroup=(Dv, 1, 1),
        output_shapes=[(B, L, H * Dv)],
        output_dtypes=[dtype],
    )
    return out
