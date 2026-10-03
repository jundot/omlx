// SPDX-License-Identifier: Apache-2.0
// Copyright © 2026 OpenAI

#pragma once

#include "mlx/backend/metal/kernels/steel/attn/attn.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "steel_qwen4_qsa_sparse_gqa.h"

// Packed-slot geometry for MSE bitstreams: rows pack `bits`-wide fields
// LSB-first across uint32 words, and a slot of 32/gcd(bits,32) dims spans
// exactly slot*bits/32 whole words, so slot loads stay word-aligned and
// every field offset is a compile-time constant.
constexpr int tq_gcd(int a, int b) { return b == 0 ? a : tq_gcd(b, a % b); }

template <int BITS>
constexpr int tq_slot_dims() {
  return 32 / tq_gcd(BITS, 32);
}

template <int BITS>
constexpr int tq_slot_words() {
  return tq_slot_dims<BITS>() * BITS / 32;
}

// Field J of one packed slot held in w0..w2 (words past the slot's own stay
// zero and are never selected: a slot spans at most three words).
template <int BITS, int J>
METAL_FUNC uint32_t tq_field(uint32_t w0, uint32_t w1, uint32_t w2) {
  constexpr int P = BITS * J;
  constexpr int A = P >> 5;
  constexpr int S = P & 31;
  constexpr uint32_t M = (uint32_t(1) << BITS) - 1u;
  if constexpr (A == 0) {
    if constexpr (S + BITS <= 32) {
      return (w0 >> S) & M;
    } else {
      return ((w0 >> S) | (w1 << (32 - S))) & M;
    }
  } else if constexpr (A == 1) {
    if constexpr (S + BITS <= 32) {
      return (w1 >> S) & M;
    } else {
      return ((w1 >> S) | (w2 << (32 - S))) & M;
    }
  } else {
    static_assert(A == 2 && S + BITS <= 32,
                  "slot field escapes the third word");
    return (w2 >> S) & M;
  }
}

// Compile-time-recursed scatters of one unpacked slot. K staging is
// transposed (element j of row k lands at KVs[k + (d+j)*ld]); V staging is
// row-major with the token norm folded in.
template <int BITS, int SLOT_DIMS, int J, typename T>
METAL_FUNC void tq_scatter_k(uint32_t w0, uint32_t w1, uint32_t w2,
                             const threadgroup float* cb, threadgroup T* dst,
                             int row, int ld, int d) {
  if constexpr (J < SLOT_DIMS) {
    dst[row + (d + J) * ld] = T(cb[tq_field<BITS, J>(w0, w1, w2)]);
    tq_scatter_k<BITS, SLOT_DIMS, J + 1, T>(w0, w1, w2, cb, dst, row, ld, d);
  }
}

template <int BITS, int SLOT_DIMS, int J, typename T>
METAL_FUNC void tq_scatter_v(uint32_t w0, uint32_t w1, uint32_t w2,
                             const threadgroup float* cb, float vn,
                             threadgroup T* dst, int base, int d) {
  if constexpr (J < SLOT_DIMS) {
    dst[base + d + J] = T(cb[tq_field<BITS, J>(w0, w1, w2)] * vn);
    tq_scatter_v<BITS, SLOT_DIMS, J + 1, T>(w0, w1, w2, cb, vn, dst, base, d);
  }
}

// Exact Qwen4 QSA main attention over TurboQuant MSE-packed K/V at any
// instantiated bit width (BITS_K/BITS_V independently from {2, 3, 4, 6, 8}).
//
// Clone of qwen4_qsa_sparse_gqa_attention with the dense K/V row staging
// replaced by in-threadgroup MSE unpacking (LSB-first `bits`-wide fields,
// matching _gen_unrolled_extract's layout). A slot of 32/gcd(bits,32) dims
// spans exactly slot*bits/32 whole uint32 words — one word per 8 dims at
// 4 bits, three words per 16 dims at 6 — so the staging loop keeps the
// dense kernel's access pattern: load the slot's words, look up the
// 2^bits-entry codebook held in threadgroup memory, scatter the slot's
// elements at constant-folded offsets.
//
// Queries arrive pre-rotated into the key codec's RHT frame on the host, so
// QK dots run directly against codebook values; the per-token fp16 key norm
// scales each score column in the epilogue before the online FP32 softmax.
// Value norms fold into the staged V rows, leaving the output accumulator in
// the value codec's rotated frame — the host applies one inverse RHT.
// Selection expansion, causal tail, masking, and the Steel MMA pipeline are
// identical to the dense kernel.
//
// Packed states are capacity-sliced views in production, so norms/packed
// carry explicit (B, H[, T]) strides instead of deriving them from kL.
template <
    typename T,
    int BITS_K,
    int BITS_V,
    int BK,
    int DC,
    int GQA,
    int H_PAD,
    int D,
    int WM,
    typename IndexT,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]] void
qwen4_qsa_sparse_gqa_attention_tq(
    const device T* Q [[buffer(0)]],
    const device half* KNorms [[buffer(1)]],
    const device uint32_t* KPacked [[buffer(2)]],
    const device half* VNorms [[buffer(3)]],
    const device uint32_t* VPacked [[buffer(4)]],
    const device float* CBK [[buffer(5)]],
    const device float* CBV [[buffer(6)]],
    const device IndexT* Topk [[buffer(7)]],
    device float* O [[buffer(8)]],
    const constant Qwen4QSATQParams* params [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on

  constexpr short kFragSize = 8;
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ = DC + padQ;
  constexpr short LDK = BK + padK;
  constexpr short LDV = DC + padV;

  constexpr int kNWarps = WM;
  constexpr int TQ = H_PAD / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TDC = DC / kFragSize;
  constexpr int D_CHUNKS = D / DC;

  static_assert(GQA <= H_PAD, "Qwen GQA heads must fit the padded MMA tile.");
  static_assert(TQ == 1, "Qwen sparse GQA expects one query-head tile.");
  static_assert(H_PAD % (kNWarps * kFragSize) == 0,
                "Padded query heads must divide evenly across simdgroups.");
  static_assert(BK % kFragSize == 0, "BK must be a multiple of eight.");
  static_assert(DC % kFragSize == 0, "DC must be a multiple of eight.");
  static_assert(D % DC == 0, "Head dimension must divide DC.");
  constexpr int kCbK = 1 << BITS_K;
  constexpr int kCbV = 1 << BITS_V;
  constexpr int kSlotK = tq_slot_dims<BITS_K>();
  constexpr int kWordsK = tq_slot_words<BITS_K>();
  constexpr int kSlotV = tq_slot_dims<BITS_V>();
  constexpr int kWordsV = tq_slot_words<BITS_V>();
  constexpr int kSlotsK = DC / kSlotK;
  constexpr int kSlotsV = DC / kSlotV;
  static_assert(kWordsK <= 3 && kWordsV <= 3,
                "One packed slot must fit three uint32 words.");
  static_assert(DC % kSlotK == 0 && DC % kSlotV == 0,
                "Dimension chunk must hold whole packed slots.");
  static_assert((DC * BITS_K) % 32 == 0 && (DC * BITS_V) % 32 == 0,
                "Dimension chunks must start on a packed-word boundary.");

  constexpr int tgp_size = WM * 32;
  const int lane = int(simd_group_id * 32 + simd_lane_id);
  const int q_pos = int(tid.x);
  const int kv_head = int(tid.y);
  const int b = int(tid.z);

  threadgroup T Qs[H_PAD * LDQ];
  threadgroup T KVs[(BK * LDV > DC * LDK) ? BK * LDV : DC * LDK];
  threadgroup int selected[BK];
  threadgroup float sel_kn[BK];
  threadgroup float sel_vn[BK];
  threadgroup float cbk_t[kCbK];
  threadgroup float cbv_t[kCbV];

  // One-time codebook preload (2^bits entries per side).
  for (int i = lane; i < kCbK; i += tgp_size) {
    cbk_t[i] = CBK[i];
  }
  for (int i = lane; i < kCbV; i += tgp_size) {
    cbv_t[i] = CBV[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  using MMAFragAcc = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  MMATile<AccumType, TQ, 1, MMAFragAcc> Qtile;
  MMATile<AccumType, 1, TK, MMAFragAcc> Ktile;
  MMATile<AccumType, TQ, TK, MMAFragAcc> Stile;
  MMATile<AccumType, 1, 1, MMAFragAcc> Vtile;
  MMATile<AccumType, TQ, D_CHUNKS * TDC, MMAFragAcc> Otile;
  Otile.clear();

  const short2 simd_coord = MMAFragAcc::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;
  const short Qs_offset = (tm + sm) * LDQ + sn;
  const short Ks_offset = sm * LDK + sn;
  const short Vs_offset = sm * LDV + sn;

  const AccumType scale = AccumType(params->scale * M_LOG2E_F);
  constexpr short rows_per_thread = decltype(Stile)::kRowsPerThread;
  AccumType max_score[rows_per_thread];
  AccumType sum_score[rows_per_thread] = {0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < rows_per_thread; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  const int query_head_base = kv_head * GQA;
  const device T *q_base = Q + size_t(b) * params->Q_strides[0] +
                           size_t(query_head_base) * params->Q_strides[1] +
                           size_t(q_pos) * params->Q_strides[2];
  const device half *k_nm = KNorms + size_t(b) * params->KNorm_strides[0] +
                            size_t(kv_head) * params->KNorm_strides[1];
  const device uint32_t *k_pk =
      KPacked + size_t(b) * params->KPacked_strides[0] +
      size_t(kv_head) * params->KPacked_strides[1];
  const device half *v_nm = VNorms + size_t(b) * params->VNorm_strides[0] +
                            size_t(kv_head) * params->VNorm_strides[1];
  const device uint32_t *v_pk =
      VPacked + size_t(b) * params->VPacked_strides[0] +
      size_t(kv_head) * params->VPacked_strides[1];
  const int64_t kp_tstride = params->KPacked_strides[2];
  const int64_t vp_tstride = params->VPacked_strides[2];
  const device IndexT *topk_base = Topk + size_t(b) * params->Topk_strides[0] +
                                   size_t(q_pos) * params->Topk_strides[2];

  const int q_abs = params->q_offset + q_pos;
  constexpr int kCompressRatio = 4;
  constexpr int kTail = kCompressRatio - 1;
  const int selected_tokens = params->topk * kCompressRatio + kTail;
  const int complete_blocks = (q_abs + 1) / kCompressRatio;
  const int valid_blocks = metal::min(params->topk, complete_blocks);
  const int n_tiles = (selected_tokens + BK - 1) / BK;

  for (int ktile = 0; ktile < n_tiles; ++ktile) {
    const int topk_off = ktile * BK;
    for (int k = lane; k < BK; k += tgp_size) {
      const int slot = topk_off + k;
      int k_pos = -1;
      if (slot < params->topk * kCompressRatio) {
        const int block_slot = slot / kCompressRatio;
        if (block_slot < valid_blocks) {
          const IndexT raw_block = topk_base[block_slot];
          const ulong candidate = ulong(raw_block) * ulong(kCompressRatio) +
                                  ulong(slot % kCompressRatio);
          if (candidate < ulong(params->kL) && candidate <= ulong(q_abs)) {
            k_pos = int(candidate);
          }
        }
      } else if (slot < selected_tokens) {
        const int tail_offset = slot - params->topk * kCompressRatio;
        const int candidate = complete_blocks * kCompressRatio + tail_offset;
        if (candidate < params->kL && candidate <= q_abs) {
          k_pos = candidate;
        }
      }
      selected[k] = k_pos;
      sel_kn[k] = k_pos >= 0 ? float(k_nm[k_pos]) : 0.0f;
      sel_vn[k] = k_pos >= 0 ? float(v_nm[k_pos]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short dchunk = 0; dchunk < D_CHUNKS; ++dchunk) {
      const int dbase = int(dchunk) * DC;
      for (int elem = lane; elem < H_PAD * (DC / 8); elem += tgp_size) {
        const int h = elem / (DC / 8);
        const int d8 = elem - h * (DC / 8);
        uint4 word = uint4(0);
        if (h < GQA) {
          word = *((const device uint4 *)(q_base +
                                          size_t(h) * params->Q_strides[1] +
                                          dbase) +
                   d8);
        }
        *((threadgroup uint4 *)(Qs + h * LDQ) + d8) = word;
      }
      // Unpack K rows: one packed slot of kSlotK dims per kWordsK word(s),
      // transposed scatter identical to the dense kernel's access pattern.
      for (int elem = lane; elem < BK * kSlotsK; elem += tgp_size) {
        const int k = elem / kSlotsK;
        const int slot = elem - k * kSlotsK;
        const int k_pos = selected[k];
        uint32_t words[3] = {0u, 0u, 0u};
        if (k_pos >= 0) {
          const device uint32_t* src =
              k_pk + size_t(k_pos) * size_t(kp_tstride) +
              size_t(dbase * BITS_K / 32) + size_t(slot * kWordsK);
          STEEL_PRAGMA_UNROLL
          for (int w = 0; w < kWordsK; ++w) {
            words[w] = src[w];
          }
        }
        tq_scatter_k<BITS_K, kSlotK, 0, T>(words[0], words[1], words[2], cbk_t,
                                           KVs, k, LDK, slot * kSlotK);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TDC; ++dd) {
        simdgroup_barrier(mem_flags::mem_none);
        Qtile.template load<T, 1, 1, LDQ, 1>(&Qs[Qs_offset + dd * kFragSize]);
        Ktile.template load<T, 1, 1, LDK, 1>(
            &KVs[Ks_offset + dd * kFragSize * LDK]);
        simdgroup_barrier(mem_flags::mem_none);
        tile_matmad(Stile, Qtile, Ktile, Stile);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      // Scale by attention scale (log2 domain) and the per-token key norm —
      // the dot ran against unit-norm codebook values in the rotated frame.
      // Invalid slots take true -INFINITY so exp2(-inf - max) == 0 (the
      // dense kernel's finite_min denominator bug does not apply here).
      constexpr auto neg_inf = selem_t(-INFINITY);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; ++i) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; ++j) {
          const short col_pos = sn + j * stile_t::kFragCols;
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < stile_t::MMAFrag_t::kElemCols; ++e) {
            const int col = col_pos + e;
            if (selected[col] < 0) {
              Stile.frag_at(i, j)[e] = neg_inf;
            } else {
              Stile.frag_at(i, j)[e] *= scale * AccumType(sel_kn[col]);
            }
          }
        }
      }
    }

    AccumType new_max[rows_per_thread];
    AccumType factor[rows_per_thread];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<Qwen4SparseMaxOp>(new_max);
    Stile.template row_bin_op<Qwen4SparseExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    AccumType sum_score_tmp[rows_per_thread] = {0};
    Stile.template row_reduce<Qwen4SparseSumOp>(sum_score_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }
    Otile.template row_bin_op<Qwen4SparseMulOp>(factor);

    STEEL_PRAGMA_UNROLL
    for (short vchunk = 0; vchunk < D_CHUNKS; ++vchunk) {
      const int dbase = int(vchunk) * DC;
      // Unpack V rows with the token norm folded in; the accumulator stays
      // in the value codec's rotated frame for the host-side inverse RHT.
      for (int elem = lane; elem < BK * kSlotsV; elem += tgp_size) {
        const int k = elem / kSlotsV;
        const int slot = elem - k * kSlotsV;
        const int k_pos = selected[k];
        uint32_t words[3] = {0u, 0u, 0u};
        if (k_pos >= 0) {
          const device uint32_t* src =
              v_pk + size_t(k_pos) * size_t(vp_tstride) +
              size_t(dbase * BITS_V / 32) + size_t(slot * kWordsV);
          STEEL_PRAGMA_UNROLL
          for (int w = 0; w < kWordsV; ++w) {
            words[w] = src[w];
          }
        }
        tq_scatter_v<BITS_V, kSlotV, 0, T>(words[0], words[1], words[2], cbv_t,
                                           sel_vn[k], KVs, k * LDV,
                                           slot * kSlotV);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; ++iq) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TDC; ++id) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ++ik) {
            const short kk = ik * kFragSize;
            const short dd = id * kFragSize;
            Vtile.template load<T, 1, 1, LDV, 1>(
                &KVs[Vs_offset + kk * LDV + dd]);
            MMAFragAcc::mma(Otile.frag_at(iq, vchunk * TDC + id),
                            Stile.frag_at(iq, ik), Vtile.frag_at(0, 0),
                            Otile.frag_at(iq, vchunk * TDC + id));
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  Otile.template row_bin_op<Qwen4SparseDivOp>(sum_score);
  device float *out = O + size_t(b) * params->O_strides[0] +
                      size_t(query_head_base + tm + sm) * params->O_strides[1] +
                      size_t(q_pos) * params->O_strides[2] + sn;
  const short rows_left = short(GQA - (tm + sm));
  if (rows_left > 0) {
    Otile.template store_safe<float, 1, 1>(out, params->O_strides[1],
                                           short2(D - sn, rows_left));
  }
}
