#pragma once

#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::qwen35_prefill_kernels {

// Mirror of mlx::core::metal::is_nax_available() (not exported from libmlx):
// macOS >= 26.2 and an applegpu generation with tensor units (gen >= 17, or
// >= 18 for 'p'-suffix parts).
bool is_nax_available();

// True when the NAX metallib was built next to the extension. Kernel launch
// still degrades to the classic kernels if loading it fails at runtime.
bool nax_qmm_kernels_built();

// False once a NAX kernel launch failed and the op fell back to the classic
// kernels for the rest of the process (diagnostics for the M5 sweep).
bool nax_qmm_runtime_active();

// dispatch_budget bounds the per-dispatch work (B * H * qL * keys) by
// splitting the key axis into separately dispatched chunks combined with
// logsumexp weights; 0 keeps the single-dispatch behavior (issue #2225).
mx::array qwen35_fa256_attention(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    float scale,
    bool causal = true,
    int q_block = 32,
    int k_block = 8,
    int64_t dispatch_budget = 0,
    mx::StreamOrDevice s = {});

mx::array qwen35_q2_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    int group_size = 64,
    mx::StreamOrDevice s = {});

mx::array qwen35_q4_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    int group_size = 64,
    mx::StreamOrDevice s = {});

mx::array qwen35_q5_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    int group_size = 64,
    mx::StreamOrDevice s = {});

mx::array qwen35_q6_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    int group_size = 64,
    mx::StreamOrDevice s = {});

mx::array qwen35_q8_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    int group_size = 64,
    mx::StreamOrDevice s = {});

// oQ mixed-bit QxA8 path (Q4/Q5, GS64, affine) on the M5 tensor units.

// True when the INT8 NAX GEMM can actually run here: tensor units present,
// the NAX metallib built next to the extension, and no earlier launch failure.
bool oq_a8_kernels_available();

// Stage A. Returns {Qa int8 [..., K], Sa float32, Ra int16 [..., K/64]}.
// act_mode 0 gives one scale per row (Sa is [M]); act_mode 1 gives one scale
// per group of 64 (Sa is [..., K/64]).
std::vector<mx::array> qwen35_oq_a8_quantize(
    const mx::array& x,
    int act_mode = 0,
    mx::StreamOrDevice s = {});

// INT8 x INT8 -> INT32 GEMM against packed affine Q4/Q5 weights, with the
// affine correction applied at every GS64 boundary. Output dtype follows
// `scales`.
mx::array qwen35_oq_a8_qmm_t(
    const mx::array& qa,
    const mx::array& sa,
    const mx::array& ra,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int bits,
    int act_mode = 0,
    int variant = 800,
    mx::StreamOrDevice s = {});

// Test helper: unpack Q4/Q5 codes to INT8 [N, group_count * 64]. Production
// code never materializes this.
mx::array qwen35_oq_a8_decode_weights(
    const mx::array& weight,
    int bits,
    int group_count,
    mx::StreamOrDevice s = {});

mx::array qwen35_moe_weighted_sum(
    const mx::array& x_sorted,
    const mx::array& inv_order,
    const mx::array& scores,
    mx::StreamOrDevice s = {});

} // namespace omlx::qwen35_prefill_kernels
