# SPDX-License-Identifier: Apache-2.0
"""Copy-free one-token KV appends for pipelined decode.

mlx-lm's ``KVCache.update_and_fetch`` writes each decoded token into the
preallocated buffer with a slice assignment and hands SDPA a view of that
buffer. MLX performs the assignment in place only when the buffer has no
other live reference. A pipelined decode loop (``async_eval`` of step n + 1
while step n runs, as ``BatchGenerator`` does) encodes step n + 1's
assignment while step n's graph, which reads a view of the same buffer, is
still in flight, so the assignment copies the whole buffer: every layer,
every token, growing with context. Whether it happens depends on how far
the host runs ahead of the GPU. A decode step slow to encode (hundreds of
small launches) tends to find step n finished and donates; a fused,
fast-to-encode step does not. On LFM2.5-24B-A2B with fused decode launches
this cost ~0.8 ms per token at 8k context (10 attention layers).

This module keeps the stock layout and semantics and adds a decode-only
side path:

* ``decode_append(cache, k, v)`` writes one token into a small per-cache
  tail buffer (``TAIL`` tokens). Only the tail is copied when its buffer is
  still referenced (a few hundred KB). Every ``TAIL`` tokens the tail is
  flushed into the main buffer with one slice assignment, so the main buffer
  is copied at most once per ``TAIL`` tokens.
* ``attend(q, segments, scale)`` is a split-K (flash-decoding) attention for
  one query row over the main buffer's first ``n_main`` rows plus the tail,
  reading every KV head once for its whole GQA group. It matches or beats
  MLX's vector SDPA for head_dim 64 (M4 Max, 32 q / 8 kv heads: 54 vs 69 us
  at 8k, 205 vs 208 us at 32k).

``KVCache.keys`` and ``KVCache.values`` become properties: reading either
flushes a pending tail first, so ``state``, ``update_and_fetch``, prefix
cache stores, batch merges and every other reader see the stock buffer with
all tokens, and assigning either drops a pending tail (a wholesale state
replacement). Caches that never take ``decode_append`` behave exactly as
before. ``offset`` stays the token count; external changes to it (trim,
rollback) are honored at the next append or flush.
"""

from __future__ import annotations

import logging
import math
from functools import cache

import mlx.core as mx

logger = logging.getLogger(__name__)

TAIL = 256
_APPLIED = False

_K, _V = "_omlx_keys", "_omlx_values"
_TK, _TV = "_omlx_tail_keys", "_omlx_tail_values"
_TN, _TB = "_omlx_tail_n", "_omlx_tail_base"


# ---------------------------------------------------------------- cache
def _flush(cache_obj) -> None:
    d = cache_obj.__dict__
    n = d.get(_TN, 0)
    if not n:
        return
    d[_TN] = 0
    base = d[_TB]
    valid = min(n, max(0, int(cache_obj.offset) - base))
    if valid == 0:
        return
    keys, values = d[_K], d[_V]
    need = base + valid
    if need > keys.shape[2]:
        step = getattr(cache_obj, "step", 256)
        extra = ((need - keys.shape[2] + step - 1) // step) * step
        b, h, _, dk = keys.shape
        keys = mx.concatenate([keys, mx.zeros((b, h, extra, dk), keys.dtype)], axis=2)
        values = mx.concatenate(
            [values, mx.zeros((b, h, extra, values.shape[3]), values.dtype)], axis=2
        )
    keys[..., base:need, :] = d[_TK][..., :valid, :]
    values[..., base:need, :] = d[_TV][..., :valid, :]
    d[_K], d[_V] = keys, values


def _get(name):
    def getter(self):
        d = self.__dict__
        if d.get(_TN):
            _flush(self)
        return d.get(name)

    return getter


def _set(name):
    def setter(self, value):
        d = self.__dict__
        d[_TN] = 0
        d[name] = value

    return setter


def decode_append(cache_obj, keys: mx.array, values: mx.array):
    """Append ``keys``/``values`` ([1, H, M, D], M decode or verify rows) and
    return ``(main_k, main_v, n_main, tail_k, tail_v, n_tail)``, or None when
    the cache has no main buffer yet (the stock path must run)."""
    d = cache_obj.__dict__
    main_k = d.get(_K)
    rows = keys.shape[2]
    if main_k is None or keys.shape[0] != 1 or not 0 < rows <= TAIL:
        return None
    if main_k.dtype != keys.dtype or main_k.shape[1] != keys.shape[1]:
        return None
    offset = int(cache_obj.offset)
    n = d.get(_TN, 0)
    if n:
        base = d[_TB]
        if base <= offset <= base + n:
            # A trim (MTP rejection) that stays inside the tail just drops
            # the trimmed rows; the main buffer is untouched.
            n = offset - base
            d[_TN] = n
        if offset != base + n or n + rows > TAIL:
            _flush(cache_obj)
            n = 0
    if n == 0:
        if offset > d[_K].shape[2]:
            return None
        # The kernel reads the main buffer row-contiguous: settle a sliced
        # view (a restored state) into its own buffer once per tail rather
        # than letting every launch copy it.
        d[_K], d[_V] = mx.contiguous(d[_K]), mx.contiguous(d[_V])
        d[_TB] = offset
        tk = d.get(_TK)
        if tk is None or tk.shape[:2] != keys.shape[:2] or tk.dtype != keys.dtype:
            b, h, _, dk = keys.shape
            d[_TK] = mx.zeros((b, h, TAIL, dk), keys.dtype)
            d[_TV] = mx.zeros((b, h, TAIL, values.shape[3]), values.dtype)
    tk, tv = d[_TK], d[_TV]
    tk[..., n : n + rows, :] = keys
    tv[..., n : n + rows, :] = values
    d[_TK], d[_TV] = tk, tv
    d[_TN] = n + rows
    cache_obj.offset = offset + rows
    return d[_K], d[_V], d[_TB], tk, tv, n + rows


def apply_kv_decode_tail_patch() -> bool:
    """Turn mlx-lm ``KVCache.keys``/``values`` into tail-flushing properties.
    Idempotent. Must run before caches are created (instances created
    earlier keep their plain attributes and are migrated on first access)."""
    global _APPLIED
    if _APPLIED:
        return True
    from mlx_lm.models.cache import KVCache

    for public, private in (("keys", _K), ("values", _V)):
        getter, setter = _get(private), _set(private)

        def get(self, _g=getter, _p=public, _q=private):
            d = self.__dict__
            if _p in d and _q not in d:
                d[_q] = d.pop(_p)
            return _g(self)

        setattr(KVCache, public, property(get, setter))
    _APPLIED = True
    logger.info("KV decode tail installed")
    return True


# ---------------------------------------------------------------- attention
# Pass 1: threadgroup (block b, kv head h) runs keys [b C, (b + 1) C) of the
# main buffer's first n1 rows followed by the tail's n2 rows, for the G
# query heads of h and their M query rows (row r sees the first
# N - (M - 1 - r) keys: causal verify rows). Simdgroups take 32-key tiles:
# lane = key for the scores (q pre-scaled by scale * log2 e, online softmax
# in base 2), lane = value dims for the weighted sum. Partials (unnormalized
# output, max, sum) go out per (query row, block); pass 2 merges blocks.
_PASS1 = r"""
    constexpr int DPL = D / 32;
    constexpr int J = G * M;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int hk = int(threadgroup_position_in_grid.y);
    const int b = int(threadgroup_position_in_grid.x);
    const int NB = int(threadgroups_per_grid.x);
    const int n1 = lens[0];
    const int n2 = lens[1];
    const int N = n1 + n2;
    const int cap1 = k1_shape[2];
    const int cap2 = k2_shape[2];
    const int C = ((N + NB - 1) / NB + 31) / 32 * 32;
    const int start = b * C;
    const int end = min(start + C, N);

    threadgroup float qs[J * D];
    threadgroup float red_m[NSG * J];
    threadgroup float red_l[NSG * J];
    threadgroup float red_o[NSG * J * D];
    const int tid = int(sg) * 32 + int(lane);
    // q is [NQ, M, D]: query j = g * M + r of this kv head.
    for (int i = tid; i < J * D; i += NSG * 32) {
      qs[i] = float(q[(hk * G) * M * D + i]) * SCALE_LOG2E;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m[J], l[J], acc[J][DPL];
    for (int j = 0; j < J; j++) {
      m[j] = -INFINITY;
      l[j] = 0;
      for (int i = 0; i < DPL; i++) {
        acc[j][i] = 0;
      }
    }
    const device T* kb1 = k1 + size_t(hk) * cap1 * D;
    const device T* vb1 = v1 + size_t(hk) * cap1 * D;
    const device T* kb2 = k2 + size_t(hk) * cap2 * D;
    const device T* vb2 = v2 + size_t(hk) * cap2 * D;

    for (int tu = start + int(sg) * 32; tu < end; tu += NSG * 32) {
      const int k = tu + int(lane);
      const bool in_range = k < end;
      float s[J];
      for (int j = 0; j < J; j++) {
        s[j] = 0;
      }
      if (in_range) {
        const device T* kr = k < n1 ? kb1 + size_t(k) * D : kb2 + size_t(k - n1) * D;
        for (int d = 0; d < D; d += 8) {
          const uint4 raw = *(const device uint4*)(kr + d);
          const vec<T, 2> a0 = as_type<vec<T, 2>>(raw.x);
          const vec<T, 2> a1 = as_type<vec<T, 2>>(raw.y);
          const vec<T, 2> a2 = as_type<vec<T, 2>>(raw.z);
          const vec<T, 2> a3 = as_type<vec<T, 2>>(raw.w);
          const float kk[8] = {float(a0.x), float(a0.y), float(a1.x), float(a1.y),
                               float(a2.x), float(a2.y), float(a3.x), float(a3.y)};
          for (int j = 0; j < J; j++) {
            float t = 0;
            for (int e = 0; e < 8; e++) {
              t += kk[e] * qs[j * D + d + e];
            }
            s[j] += t;
          }
        }
      }
      float p[J];
      for (int j = 0; j < J; j++) {
        const int r = j % M;
        const bool valid = in_range && k < N - (M - 1 - r);
        const float sv = valid ? s[j] : -INFINITY;
        const float mn = max(m[j], simd_max(sv));
        const float corr = (mn == -INFINITY) ? 1.0f : fast::exp2(m[j] - mn);
        p[j] = valid ? fast::exp2(sv - mn) : 0.0f;
        l[j] = l[j] * corr + simd_sum(p[j]);
        m[j] = mn;
        for (int i = 0; i < DPL; i++) {
          acc[j][i] *= corr;
        }
      }
      const int nk = min(32, end - tu);
      const bool seg1 = tu + nk <= n1;
      const bool seg2 = tu >= n1;
      if (nk == 32 && (seg1 || seg2)) {
        const device T* vr0 =
            (seg1 ? vb1 + size_t(tu) * D : vb2 + size_t(tu - n1) * D) + int(lane) * DPL;
        for (int j0 = 0; j0 < 32; j0 += VB) {
          vec<T, DPL> vv[VB];
          for (int jj = 0; jj < VB; jj++) {
            vv[jj] = *(const device vec<T, DPL>*)(vr0 + (j0 + jj) * D);
          }
          for (int jj = 0; jj < VB; jj++) {
            for (int j = 0; j < J; j++) {
              const float pj = simd_shuffle(p[j], ushort(j0 + jj));
              for (int i = 0; i < DPL; i++) {
                acc[j][i] += pj * float(vv[jj][i]);
              }
            }
          }
        }
      } else {
        for (int kj = 0; kj < nk; kj++) {
          const int key = tu + kj;
          const device T* vr = key < n1 ? vb1 + size_t(key) * D : vb2 + size_t(key - n1) * D;
          const vec<T, DPL> vv = *(const device vec<T, DPL>*)(vr + int(lane) * DPL);
          for (int j = 0; j < J; j++) {
            const float pj = simd_shuffle(p[j], ushort(kj));
            for (int i = 0; i < DPL; i++) {
              acc[j][i] += pj * float(vv[i]);
            }
          }
        }
      }
    }
    if (lane == 0) {
      for (int j = 0; j < J; j++) {
        red_m[sg * J + j] = m[j];
        red_l[sg * J + j] = l[j];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < J; j++) {
      float Mx = -INFINITY;
      for (int s2 = 0; s2 < NSG; s2++) {
        Mx = max(Mx, red_m[s2 * J + j]);
      }
      const float f = (m[j] == -INFINITY) ? 0.0f : fast::exp2(m[j] - Mx);
      for (int i = 0; i < DPL; i++) {
        red_o[(sg * J + j) * D + int(lane) * DPL + i] = acc[j][i] * f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      for (int j = 0; j < J; j++) {
        float Mx = -INFINITY;
        for (int s2 = 0; s2 < NSG; s2++) {
          Mx = max(Mx, red_m[s2 * J + j]);
        }
        float L = 0;
        for (int s2 = 0; s2 < NSG; s2++) {
          const float ms = red_m[s2 * J + j];
          L += (ms == -INFINITY) ? 0.0f : red_l[s2 * J + j] * fast::exp2(ms - Mx);
        }
        // output row: query head hk * G + j / M, query row j % M
        const int row = (hk * G + j / M) * M + j % M;
        for (int i = 0; i < DPL; i++) {
          float o = 0;
          for (int s2 = 0; s2 < NSG; s2++) {
            o += red_o[(s2 * J + j) * D + int(lane) * DPL + i];
          }
          po[(size_t(row) * NB + b) * D + int(lane) * DPL + i] = o;
        }
        if (lane == 0) {
          pm[row * NB + b] = Mx;
          pl[row * NB + b] = L;
        }
      }
    }
"""

# Pass 2: one simdgroup per (query head, query row) merges the blocks.
_PASS2 = r"""
    constexpr int DPL = D / 32;
    const uint lane = thread_index_in_simdgroup;
    const int hq = int(threadgroup_position_in_grid.y) * NSG + int(simdgroup_index_in_threadgroup);
    const int NB = nb[0];
    float Mx = -INFINITY;
    for (int b = int(lane); b < NB; b += 32) {
      Mx = max(Mx, pm[hq * NB + b]);
    }
    Mx = simd_max(Mx);
    float L = 0;
    for (int b = int(lane); b < NB; b += 32) {
      const float mb = pm[hq * NB + b];
      L += (mb == -INFINITY) ? 0.0f : pl[hq * NB + b] * fast::exp2(mb - Mx);
    }
    L = simd_sum(L);
    float o[DPL];
    for (int i = 0; i < DPL; i++) {
      o[i] = 0;
    }
    for (int b = 0; b < NB; b++) {
      const float mb = pm[hq * NB + b];
      if (mb == -INFINITY) {
        continue;
      }
      const float f = fast::exp2(mb - Mx);
      for (int i = 0; i < DPL; i++) {
        o[i] += po[(size_t(hq) * NB + b) * D + int(lane) * DPL + i] * f;
      }
    }
    for (int i = 0; i < DPL; i++) {
      out[hq * D + int(lane) * DPL + i] = static_cast<T>(o[i] / L);
    }
"""

_P1_SIMDGROUPS = 8
_P1_SIMDGROUPS_ROWS = 4  # keeps red_o within 32 KB for 16 query rows
MAX_ROWS = 4
_P2_SIMDGROUPS = 4
_V_BATCH = 8


@cache
def _pass1(scale: float):
    return mx.fast.metal_kernel(
        name=f"omlx_decode_attention_2seg_p1_{abs(hash(scale)):x}",
        input_names=["q", "k1", "v1", "k2", "v2", "lens"],
        output_names=["po", "pm", "pl"],
        source=_PASS1.replace("SCALE_LOG2E", repr(scale * math.log2(math.e)) + "f"),
    )


@cache
def _pass2():
    return mx.fast.metal_kernel(
        name="omlx_decode_attention_2seg_p2",
        input_names=["po", "pm", "pl", "nb"],
        output_names=["out"],
        source=_PASS2,
    )


def attend_supported(q: mx.array, k: mx.array) -> bool:
    """Whether ``attend`` serves these query rows and KV layout."""
    if q.ndim != 4 or q.shape[0] != 1 or not 1 <= q.shape[2] <= MAX_ROWS:
        return False
    nq, d = q.shape[1], q.shape[3]
    nk = k.shape[1]
    return (
        q.dtype in (mx.float16, mx.bfloat16)
        and d % 32 == 0
        and 32 <= d <= 128
        and k.shape[3] == d
        and nq % nk == 0
        and (nq // nk) * q.shape[2] <= 16
        and (nq * q.shape[2]) % _P2_SIMDGROUPS == 0
    )


def _blocks(n: int) -> int:
    return max(8, min(32, n // 2048))


def attend(q, k1, v1, n1: int, k2, v2, n2: int, scale: float) -> mx.array:
    """Causal softmax attention of the M query rows ``q`` [1, NQ, M, D] (the
    last M positions) over the first ``n1`` rows of ``k1``/``v1`` followed by
    the first ``n2`` rows of ``k2``/``v2`` (each [1, NK, capacity, D],
    contiguous)."""
    _, nq, rows, d = q.shape
    nk = k1.shape[1]
    nb = _blocks(n1 + n2)
    nsg = _P1_SIMDGROUPS if rows == 1 else _P1_SIMDGROUPS_ROWS
    lens = mx.array([n1, n2], dtype=mx.int32)
    po, pm, pl = _pass1(float(scale))(
        inputs=[q, k1, v1, k2, v2, lens],
        template=[
            ("T", q.dtype),
            ("D", d),
            ("G", nq // nk),
            ("M", rows),
            ("NSG", nsg),
            ("VB", _V_BATCH),
        ],
        grid=(32 * nsg * nb, nk, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(nq * rows * nb * d,), (nq * rows * nb,), (nq * rows * nb,)],
        output_dtypes=[mx.float32] * 3,
    )
    return _pass2()(
        inputs=[po, pm, pl, mx.array([nb], dtype=mx.int32)],
        template=[("T", q.dtype), ("D", d), ("NSG", _P2_SIMDGROUPS)],
        grid=(32, nq * rows, 1),
        threadgroup=(32, _P2_SIMDGROUPS, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


__all__ = [
    "MAX_ROWS",
    "TAIL",
    "apply_kv_decode_tail_patch",
    "attend",
    "attend_supported",
    "decode_append",
]
