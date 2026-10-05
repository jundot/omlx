// Sorted-expert gather_qmm (x @ w[idx].T) on the M5 tensor units.
//
// This is MLX 0.32.2's affine_gather_qmm_rhs_nax (quantized_nax.h) with one
// fix: the per-simdgroup row bound is clamped in int before narrowing to
// short. MLX computes short(max(0, M - (y_row + tm))) first, which wraps once
// a call has more than 32767 rows, so rows go missing (ml-explore/mlx#3856,
// fixed upstream after 0.32.2). oMLX previously worked around that by slicing
// every large MoE prefill call and concatenating the outputs; this kernel
// lets one call cover all rows. Alignment flags are template parameters here
// instead of MLX's function constants, and the per-row arithmetic is
// otherwise unchanged, so each output row matches the sliced MLX result.

#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"
// clang-format on

template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool kAlignN,
    bool kAlignK>
[[kernel]] void qwen35_gather_qmm_rhs_t_nax(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    const device uint32_t* indices [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      true,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Ws[BN * BK_padded];

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Rows always use the bounded path: M is data dependent.
  const short tgp_bm = short(min(BM, M - y_row));
  const short tgp_bn = kAlignN ? BN : short(min(BN, N - y_col));

  const int k_remain = K - K_it * BK;
  const short2 tile_w = short2(k_remain, tgp_bn);

  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += y_col_long * K_w;
  scales += y_col_long * K_g;
  biases += y_col_long * K_g;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  // The fix: clamp in int, then narrow. Both results fit in short.
  const short sgp_sm = short(min(int(SM), max(0, M - (y_row + tm))));
  const short sgp_sn =
      kAlignN ? SN : short(min(int(SN), max(0, N - (y_col + tn))));

  const bool is_unaligned_sm = (sgp_sm != SM);
  const bool is_unaligned_bn = kAlignN ? false : (tgp_bn != BN);

  constexpr short BR = TN;
  constexpr short BC = TK;

  using AccumType = float;

  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
    const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));
    const bool sg_active = m_hi_lim > m_lo_lim;

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();

    const device T* xn = x + tm * K;

    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        biases + index * stride_s,
        K,
        Ws,
        simd_group_id,
        simd_lane_id);

    dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(kAlignN || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            loader_w.load_unsafe();
          } else {
            loader_w.load_safe(short2(BK, tgp_bn));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

              volatile int compiler_barrier;

              if constexpr (kAlignedM.value) {
                Atile.load(xn + kk1, K);
              } else {
                Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
              }

              Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<true>{});

              (void)compiler_barrier;
            }
          }

          xn += BK;
          loader_w.next();
        }

        if (!kAlignK) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_w.load_safe(tile_w);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

              volatile int compiler_barrier;

              const short psk = min(int(SK), max(0, (BK - kk1)));
              Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

              Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<true>{});

              (void)compiler_barrier;
            }
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if constexpr (kAlignedN.value) {
          if (m_lo_lim == 0 && m_hi_lim == SM) {
            Dtile.store(y + tm * N + tn, N);
          } else {
            Dtile.store_slice(
                y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
          }
        } else {
          Dtile.store_slice(
              y + tm * N + tn,
              N,
              short2(0, m_lo_lim),
              short2(sgp_sn, m_hi_lim));
        }
      });
    });
  }
}

// Only the fully aligned N/K layout the Qwen MoE prefill path produces
// (N and K multiples of 64) is instantiated; the C++ op validates it.
#define instantiate_qwen35_gather_qmm_rhs_t_nax(type, gs, bits)               \
  instantiate_kernel(                                                        \
      "qwen35_gather_qmm_rhs_t_nax_" #type "_gs_" #gs "_b_" #bits            \
      "_bm_64_bn_64_bk_64_wm_2_wn_2",                                        \
      qwen35_gather_qmm_rhs_t_nax,                                           \
      type,                                                                  \
      gs,                                                                    \
      bits,                                                                  \
      64,                                                                    \
      64,                                                                    \
      64,                                                                    \
      2,                                                                     \
      2,                                                                     \
      true,                                                                  \
      true)

#define instantiate_qwen35_gather_qmm_rhs_t_nax_bits(type, gs)                \
  instantiate_qwen35_gather_qmm_rhs_t_nax(type, gs, 4);                      \
  instantiate_qwen35_gather_qmm_rhs_t_nax(type, gs, 5);                      \
  instantiate_qwen35_gather_qmm_rhs_t_nax(type, gs, 6);                      \
  instantiate_qwen35_gather_qmm_rhs_t_nax(type, gs, 8)

instantiate_qwen35_gather_qmm_rhs_t_nax_bits(float16_t, 64);
instantiate_qwen35_gather_qmm_rhs_t_nax_bits(bfloat16_t, 64);
instantiate_qwen35_gather_qmm_rhs_t_nax_bits(float16_t, 128);
instantiate_qwen35_gather_qmm_rhs_t_nax_bits(bfloat16_t, 128);

#endif // __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
