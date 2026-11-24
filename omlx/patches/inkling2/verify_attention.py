# SPDX-License-Identifier: Apache-2.0
"""Row-exact verify attention for Inkling 2 in one launch per visit.

Each row must keep the bits of its token's one-row decode step, which MLX
0.32.4 runs through ``sdpa_vector_2pass``. Pass 1 and pass 2 below repeat that
kernel's arithmetic for all rows at once, so rerun
tests/test_inkling2_verify_attention.py after an MLX pin change.
"""

from __future__ import annotations

import os
from functools import cache

import mlx.core as mx

# Lightning MTP copy blocks verify up to 16 rows.
MAX_ROWS = 16
# Rows one simdgroup carries; more cost occupancy on M1-class GPUs.
SHARED_MAX_ROWS = 4
_TK = 4

_PASS1 = r"""
    constexpr int BD = 32;
    constexpr int QK = D / BD;
    constexpr int VP = V / BD;
    const int kv_head = int(threadgroup_position_in_grid.x);
    const int batch = int(threadgroup_position_in_grid.y);
    const int block = int(threadgroup_position_in_grid.z);
    const int g = int(thread_position_in_threadgroup.y);
    const int row = int(thread_position_in_threadgroup.z);
    const uint lane = thread_index_in_simdgroup;
    const int qh = GQA * kv_head + g;

    const int n_keys = rows_n[row];
    const int lo = rows_lo[row];

    const device T* qp = q + batch * q_strides[0] + qh * q_strides[1] +
        row * q_strides[2] + int(lane) * QK * q_strides[3];
    const device T* keys = k + batch * k_strides[0] + kv_head * k_strides[1] +
        (lo + block) * k_strides[2] + int(lane) * QK * k_strides[3];
    const device T* values = v + batch * v_strides[0] + kv_head * v_strides[1] +
        (lo + block) * v_strides[2] + int(lane) * VP * v_strides[3];
    const device T* relp = rel + batch * rel_strides[0] + row * rel_strides[1] +
        qh * rel_strides[2];
    const int rel_step = rel_strides[3];

    const int o_row = (batch * HQ + qh) * L + row;
    device T* out = partials + o_row * BLOCKS * V + block * V + int(lane) * VP;

    float qv[QK];
    float o[VP] = {0};
    for (int i = 0; i < QK; i++) {
      qv[i] = scale[0] * float(qp[i * q_strides[3]]);
    }

    float max_score = Limits<float>::finite_min;
    float sum_exp_score = 0;
    if (block == 0) {
      max_score = static_cast<float>(sink[batch * sink_strides[0] +
          row * sink_strides[1] + qh * sink_strides[2]]);
      sum_exp_score = 1;
    }

    const int stride_k_tok = BLOCKS * int(k_strides[2]);
    const int stride_v_tok = BLOCKS * int(v_strides[2]);

    int i = block;
    for (; i + (TK - 1) * BLOCKS < n_keys; i += BLOCKS * TK) {
      T k_reg[TK][QK];
      T v_reg[TK][VP];
      for (int t = 0; t < TK; t++) {
        const device T* kp = keys + t * stride_k_tok;
        for (int j = 0; j < QK; j++) {
          k_reg[t][j] = kp[j * k_strides[3]];
        }
      }
      for (int t = 0; t < TK; t++) {
        const device T* vp = values + t * stride_v_tok;
        for (int j = 0; j < VP; j++) {
          v_reg[t][j] = vp[j * v_strides[3]];
        }
      }
      for (int t = 0; t < TK; t++) {
        const int ii = i + t * BLOCKS;
        const int d = n_keys - 1 - ii;
        const T m = d < R ? relp[d * rel_step] : T(0);
        if (m >= Limits<T>::finite_min) {
          float score = 0;
          for (int j = 0; j < QK; j++) {
            score += qv[j] * static_cast<float>(k_reg[t][j]);
          }
          score = simd_sum(score);
          score += static_cast<float>(m);
          float new_max = max(max_score, score);
          float factor = fast::exp(max_score - new_max);
          float exp_score = fast::exp(score - new_max);
          max_score = new_max;
          sum_exp_score = sum_exp_score * factor + exp_score;
          for (int j = 0; j < VP; j++) {
            o[j] = o[j] * factor + exp_score * static_cast<float>(v_reg[t][j]);
          }
        }
      }
      keys += TK * stride_k_tok;
      values += TK * stride_v_tok;
    }
    for (; i < n_keys; i += BLOCKS) {
      const int d = n_keys - 1 - i;
      const T m = d < R ? relp[d * rel_step] : T(0);
      if (m >= Limits<T>::finite_min) {
        float score = 0;
        for (int j = 0; j < QK; j++) {
          score += qv[j] * keys[j * k_strides[3]];
        }
        score = simd_sum(score);
        score += m;
        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < VP; j++) {
          o[j] = o[j] * factor + exp_score * values[j * v_strides[3]];
        }
      }
      keys += stride_k_tok;
      values += stride_v_tok;
    }

    if (lane == 0) {
      sums[o_row * BLOCKS + block] = sum_exp_score;
      maxs[o_row * BLOCKS + block] = max_score;
    }
    for (int j = 0; j < VP; j++) {
      out[j] = static_cast<T>(o[j]);
    }
"""

# Full-attention visits: one simdgroup per query head takes RPS rows and loads
# each key once for all of them. Per row it repeats pass 1 step for step, so
# the partials keep the same bits.
_PASS1_SHARED = r"""
    constexpr int BD = 32;
    constexpr int QK = D / BD;
    constexpr int VP = V / BD;
    const int kv_head = int(threadgroup_position_in_grid.x);
    const int batch = int(threadgroup_position_in_grid.y);
    const int block = int(threadgroup_position_in_grid.z);
    const int g = int(thread_position_in_threadgroup.y);
    const uint lane = thread_index_in_simdgroup;
    const int qh = GQA * kv_head + g;
    const int base = int(thread_position_in_threadgroup.z) * RPS;
    const int count = min(RPS, L - base);

    const device T* keys = k + batch * k_strides[0] + kv_head * k_strides[1] +
        block * k_strides[2] + int(lane) * QK;
    const device T* values = v + batch * v_strides[0] + kv_head * v_strides[1] +
        block * v_strides[2] + int(lane) * VP;
    const int stride_k_tok = BLOCKS * int(k_strides[2]);
    const int stride_v_tok = BLOCKS * int(v_strides[2]);

    float qv[RPS][QK];
    float o[RPS][VP];
    float max_score[RPS];
    float sum_exp_score[RPS];
    int n_keys[RPS];
    const device T* relp[RPS];
    for (int r = 0; r < RPS; r++) {
      const int row = min(base + r, L - 1);
      const device T* qp = q + batch * q_strides[0] + qh * q_strides[1] +
          row * q_strides[2] + int(lane) * QK;
      for (int i = 0; i < QK; i++) {
        qv[r][i] = scale[0] * float(qp[i]);
      }
      for (int j = 0; j < VP; j++) {
        o[r][j] = 0;
      }
      max_score[r] = Limits<float>::finite_min;
      sum_exp_score[r] = 0;
      if (block == 0) {
        max_score[r] = static_cast<float>(sink[batch * sink_strides[0] +
            row * sink_strides[1] + qh * sink_strides[2]]);
        sum_exp_score[r] = 1;
      }
      // Padding slots of a short last group get no keys.
      n_keys[r] = base + r < L ? rows_n[row] : 0;
      relp[r] = rel + batch * rel_strides[0] + row * rel_strides[1] +
          qh * rel_strides[2];
    }
    const int n_max = n_keys[count - 1];

#define OMLX_ROWS_UPDATE(KR, VR, II)                                         \
      for (int r = 0; r < RPS; r++) {                                       \
        if ((II) < n_keys[r]) {                                             \
          const int d = n_keys[r] - 1 - (II);                               \
          const T m = d < R ? relp[r][d] : T(0);                            \
          if (m >= Limits<T>::finite_min) {                                 \
            float score = 0;                                                \
            for (int j = 0; j < QK; j++) {                                  \
              score += qv[r][j] * static_cast<float>(KR[j]);                \
            }                                                               \
            score = simd_sum(score);                                        \
            score += static_cast<float>(m);                                 \
            float new_max = max(max_score[r], score);                       \
            float factor = fast::exp(max_score[r] - new_max);               \
            float exp_score = fast::exp(score - new_max);                   \
            max_score[r] = new_max;                                         \
            sum_exp_score[r] = sum_exp_score[r] * factor + exp_score;       \
            for (int j = 0; j < VP; j++) {                                  \
              o[r][j] = o[r][j] * factor + exp_score * static_cast<float>(VR[j]); \
            }                                                               \
          }                                                                 \
        }                                                                   \
      }

    for (int i = block; i < n_max; i += BLOCKS) {
      T kr[QK];
      T vr[VP];
      for (int j = 0; j < QK; j++) {
        kr[j] = keys[j];
      }
      for (int j = 0; j < VP; j++) {
        vr[j] = values[j];
      }
      OMLX_ROWS_UPDATE(kr, vr, i)
      keys += stride_k_tok;
      values += stride_v_tok;
    }
#undef OMLX_ROWS_UPDATE

    for (int r = 0; r < count; r++) {
      const int o_row = (batch * HQ + qh) * L + base + r;
      if (lane == 0) {
        sums[o_row * BLOCKS + block] = sum_exp_score[r];
        maxs[o_row * BLOCKS + block] = max_score[r];
      }
      device T* out = partials + o_row * BLOCKS * V + block * V + int(lane) * VP;
      for (int j = 0; j < VP; j++) {
        out[j] = static_cast<T>(o[r][j]);
      }
    }
"""


# sdpa_vector_2pass_2 over (batch * head, row).
_PASS2 = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int EPT = D / BD;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    float o[EPT] = {0};
    threadgroup float outputs[BN * BD];

    const int head_idx = int(threadgroup_position_in_grid.x);
    const int q_seq_idx = int(threadgroup_position_in_grid.y);
    const int q_offset = head_idx * L + q_seq_idx;
    const device T* pp = partials + q_offset * BLOCKS * D + simd_gid * D +
        simd_lid * EPT;
    const device float* sp = sums + q_offset * BLOCKS;
    const device float* mp = maxs + q_offset * BLOCKS;
    device T* op = out + (q_seq_idx * HQ + head_idx) * D + simd_gid * EPT;

    float sum_exp_score = 0.0;
    float max_score = Limits<float>::finite_min;
    for (int b = 0; b < BLOCKS / BN; ++b) {
      max_score = max(max_score, mp[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);
    for (int b = 0; b < BLOCKS / BN; ++b) {
      float factor = fast::exp(mp[simd_lid + BN * b] - max_score);
      sum_exp_score += factor * sp[simd_lid + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);
    for (int b = 0; b < BLOCKS / BN; ++b) {
      float factor = fast::exp(mp[simd_gid] - max_score);
      for (int i = 0; i < EPT; i++) {
        o[i] += factor * static_cast<float>(pp[i]);
      }
      mp += BN;
      sp += BN;
      pp += BN * D;
    }
    for (int i = 0; i < EPT; i++) {
      outputs[simd_lid * BD + simd_gid] = o[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
      for (int i = 0; i < EPT; i++) {
        op[i] = static_cast<T>(o[i]);
      }
    }
"""


@cache
def _pass1():
    return mx.fast.metal_kernel(
        name="omlx_inkling2_verify_sdpa_pass1",
        input_names=["q", "k", "v", "rel", "sink", "rows_n", "rows_lo", "scale"],
        output_names=["partials", "sums", "maxs"],
        source=_PASS1,
        ensure_row_contiguous=False,
    )


@cache
def _pass1_shared():
    return mx.fast.metal_kernel(
        name="omlx_inkling2_verify_sdpa_pass1_shared",
        input_names=["q", "k", "v", "rel", "sink", "rows_n", "scale"],
        output_names=["partials", "sums", "maxs"],
        source=_PASS1_SHARED,
        ensure_row_contiguous=False,
    )


@cache
def _pass2():
    return mx.fast.metal_kernel(
        name="omlx_inkling2_verify_sdpa_pass2",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=_PASS2,
    )


@cache
def _arch_suffix() -> str:
    return str(mx.device_info().get("architecture", ""))[-1:]


def _two_pass(n_keys: int, gqa: int) -> bool:
    """MLX 0.32.4's route to ``sdpa_vector_2pass`` for one decode row."""
    devc = _arch_suffix()
    return (devc in ("d", "s") and n_keys >= 1024) or (gqa > 1 and n_keys >= 4096)


def _blocks(n_keys: int, gqa: int) -> int:
    """MLX 0.32.4's block count for a one-row (``n_simds == gqa``) call."""
    devc = _arch_suffix()
    n_simds = gqa
    if devc == "s":
        blocks = 64
        if n_keys > 1024 and n_simds > 4:
            if n_keys <= 8192:
                blocks = 128
            elif n_keys <= 32768:
                blocks = 256
            elif n_keys <= 65536:
                blocks = 512
            else:
                blocks = 1024
    elif devc == "d":
        blocks = 128
        if n_simds <= 2 and n_keys > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n_keys < 65536:
                blocks = 512
            elif n_keys >= 65536:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    env = int(os.environ.get("MLX_SDPA_BLOCKS", "0") or 0)
    if env > 0:
        blocks = ((env + 31) // 32) * 32
    return blocks


def rows_attention(q, k, v, rel, sink, *, scale: float, window: int):
    """The row-exact verify rows of ``banded_sink_attention`` in one launch.

    Returns ``[1, Hq, L, D]``, or None when some row would not take MLX's
    two-pass kernel, so the caller keeps its per-row loop.
    """
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    if (
        mx.default_device() != mx.gpu
        or b != 1
        or not 1 < length <= MAX_ROWS
        or hq % hkv
        or d % 32
        or v.shape[-1] != d
        or q.dtype not in (mx.float16, mx.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or rel.dtype != q.dtype
    ):
        return None
    gqa = hq // hkv
    n_rows, lo_rows = [], []
    for r in range(length):
        sr = s - length + 1 + r
        lo = max(0, sr - window) if window else 0
        n_rows.append(sr - lo)
        lo_rows.append(lo)
    if not all(_two_pass(n, gqa) for n in n_rows):
        return None
    blocks = {_blocks(n, gqa) for n in n_rows}
    if len(blocks) != 1:
        return None
    blocks = blocks.pop()
    windowed = any(lo_rows)
    if windowed and 32 * gqa * length > 1024:
        # One simdgroup per (head, row) would not fit a threadgroup.
        return None
    dt = q.dtype
    out_shapes = [
        (b * hq * length * blocks * d,),
        (b * hq * length * blocks,),
        (b * hq * length * blocks,),
    ]
    common = [
        ("T", dt),
        ("D", d),
        ("V", d),
        ("L", length),
        ("HQ", hq),
        ("GQA", gqa),
        ("R", int(rel.shape[-1])),
        ("BLOCKS", blocks),
    ]
    if not windowed:
        # Past ~16K keys two rows per simdgroup measured fastest (M1 Max).
        rps = min(length, SHARED_MAX_ROWS if s < 16384 else 2)
        groups = (length + rps - 1) // rps
        partials, sums, maxs = _pass1_shared()(
            inputs=[
                q,
                k,
                v,
                rel,
                sink.astype(dt),
                mx.array(n_rows, dtype=mx.int32),
                mx.array([scale], dtype=mx.float32),
            ],
            template=common + [("RPS", rps)],
            grid=(32 * hkv, gqa * b, groups * blocks),
            threadgroup=(32, gqa, groups),
            output_shapes=out_shapes,
            output_dtypes=[dt, mx.float32, mx.float32],
        )
    else:
        partials, sums, maxs = _pass1()(
            inputs=[
                q,
                k,
                v,
                rel,
                sink.astype(dt),
                mx.array(n_rows, dtype=mx.int32),
                mx.array(lo_rows, dtype=mx.int32),
                mx.array([scale], dtype=mx.float32),
            ],
            template=common + [("TK", _TK)],
            grid=(32 * hkv, gqa * b, length * blocks),
            threadgroup=(32, gqa, length),
            output_shapes=out_shapes,
            output_dtypes=[dt, mx.float32, mx.float32],
        )
    out = _pass2()(
        inputs=[partials, sums, maxs],
        template=[("T", dt), ("D", d), ("L", length), ("HQ", hq), ("BLOCKS", blocks)],
        grid=(1024 * b * hq, length, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(b, length, hq, d)],
        output_dtypes=[dt],
    )[0]
    return out.transpose(0, 2, 1, 3)
