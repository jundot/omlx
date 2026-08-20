// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// NVFP4 (E4M3) Metal kernel dispatch, ported from
// Layr-Labs/mlxfast-challenge (Sources/MLXFastModel/LagunaRuntimeModel.swift).
//
// Metal kernel sources live in laguna_nvfp4.metal, compiled into
// omlx_laguna_nvfp4_kernels.metallib by CMake. The metallib is loaded lazily
// on the first dispatch call and cached.
//
// MLX 0.32+ requires Metal dispatch to occur inside Primitive::eval_gpu.
// All public API functions return an unevaluated array whose Primitive drives
// the actual Metal dispatch at eval time.

#include "laguna_nvfp4_kernels.h"

#include <dlfcn.h>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <string>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace omlx::laguna_nvfp4 {

namespace {

using namespace mlx::core;
using namespace mlx::core::metal;

// ---------------------------------------------------------------------------
// Metallib loader
// ---------------------------------------------------------------------------

constexpr const char* kMetallibName = "omlx_laguna_nvfp4_kernels";

std::string binary_dir() {
    static std::string dir = []() {
        Dl_info info;
        if (!dladdr(reinterpret_cast<void*>(&binary_dir), &info)) {
            throw std::runtime_error(
                "laguna_nvfp4: unable to resolve binary dir.");
        }
        return std::filesystem::path(info.dli_fname).parent_path().string();
    }();
    return dir;
}

MTL::ComputePipelineState* get_laguna_kernel(
    metal::Device& d,
    const std::string& kernel_name) {
    auto* lib = d.get_library(kMetallibName, binary_dir());
    return d.get_kernel(kernel_name, lib);
}

// ---------------------------------------------------------------------------
// Shared-expert fused gate/up NVFP4 QMV primitive
// ---------------------------------------------------------------------------

// SharedSwiGLUQmvPrimitive:
//   inputs[0] = x      [K] bf16 activations
//   inputs[1] = w      [2N, K*4/32] uint8 fused gate/up NVFP4 codes
//   inputs[2] = scales [2N, K/16] uint8 E4M3 group scales
//   output[0] = out    [N] bf16 silu(gate) * up
class SharedSwiGLUQmvPrimitive : public Primitive {
 public:
    explicit SharedSwiGLUQmvPrimitive(Stream s) : Primitive(s) {}

 private:
    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 SharedSwiGLUQmvPrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));

        const auto& x = inputs[0];
        const auto& w = inputs[1];
        const auto& scales = inputs[2];

        auto kernel = get_laguna_kernel(
            d, "laguna_shared_nvfp4_swiglu_qmv_bf16_v1");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);

        int c = 0;
        enc.set_input_array(x, c++);
        enc.set_input_array(w, c++);
        enc.set_input_array(scales, c++);
        enc.set_output_array(out, c++);

        // 512 output rows; each threadgroup (2 simdgroups x 32 lanes) owns 4
        // rows. 128 tiles x 4 rows = 512.
        MTL::Size group_dims(64, 1, 1);
        MTL::Size grid_dims(128, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(SharedSwiGLUQmvPrimitive)
};

// SharedDownResidualPrimitive:
//   inputs[0] = activated [K2] bf16 swiglu output (K2 = 512)
//   inputs[1] = down_weight [N, K2/2] uint8 NVFP4 codes (N = 2048)
//   inputs[2] = down_scales [N, K2/16] uint8 E4M3 scales
//   inputs[3] = routed [N] bf16
//   inputs[4] = residual [N] bf16
//   output[0] = out [N] bf16 = residual + (routed + shared)
class SharedDownResidualPrimitive : public Primitive {
 public:
    explicit SharedDownResidualPrimitive(Stream s) : Primitive(s) {}

 private:
    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 SharedDownResidualPrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));

        auto kernel = get_laguna_kernel(
            d, "laguna_shared_nvfp4_down_residual_bf16_v1");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);

        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        enc.set_output_array(out, c++);

        // 2048 output rows; each threadgroup (2 simdgroups x 32 lanes) owns
        // 8 rows (2 simdgroups x 4 rows). 256 groups x 8 = 2048.
        MTL::Size group_dims(64, 1, 1);
        MTL::Size grid_dims(256, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(SharedDownResidualPrimitive)
};

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

array shared_nvfp4_swiglu_qmv(
    const array& x,
    const array& w,
    const array& scales,
    StreamOrDevice s) {
    if (x.ndim() != 1 || x.dtype() != bfloat16) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 shared_nvfp4_swiglu_qmv: expected 1-D bf16 input "
               "of shape [2048], got shape ["
            << x.shape() << "] dtype " << x.dtype();
        throw std::invalid_argument(msg.str());
    }
    int64_t K = x.shape(0);
    int64_t N = w.shape(0) / 2;
    // The kernel reads each row as K/2 bytes of 4-bit nibbles. mlx-standard
    // packing stores those in uint32 (8 nibbles per word) — K/8 uint32 per
    // row — and the challenge's fused plane uses the same byte layout.
    int64_t dtype_bytes = (w.dtype() == uint8) ? 1 : 4;
    int64_t row_bytes = w.shape(1) * dtype_bytes;
    if (w.ndim() != 2 || scales.ndim() != 2 ||
        scales.shape(0) != 2 * N ||
        scales.shape(1) * 16 != K ||
        row_bytes * 2 != K) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 shared_nvfp4_swiglu_qmv: shape mismatch — "
               "x " << x.shape() << ", w " << w.shape()
            << " (row_bytes " << row_bytes << "), scales " << scales.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<SharedSwiGLUQmvPrimitive>(s_stream);
    Shape out_shape{static_cast<ShapeElem>(N)};
    return array(out_shape, bfloat16, prim, {x, w, scales});
}

array shared_nvfp4_down_residual(
    const array& activated,
    const array& down_weight,
    const array& down_scales,
    const array& routed,
    const array& residual,
    StreamOrDevice s) {
    // activated [K2], down_weight [N, K2/2], down_scales [N, K2/16],
    // routed/residual [N]; N = 2048, K2 = 512.
    int64_t N = routed.shape(0);
    int64_t K2 = activated.shape(0);
    if (activated.ndim() != 1 || activated.dtype() != bfloat16 ||
        routed.ndim() != 1 || residual.ndim() != 1 ||
        routed.dtype() != bfloat16 || residual.dtype() != bfloat16 ||
        down_weight.ndim() != 2 || down_scales.ndim() != 2 ||
        down_weight.shape(0) != N || down_scales.shape(0) != N ||
        down_scales.shape(1) * 16 != K2 ||
        down_weight.shape(1) * 8 != K2 ||
        residual.shape(0) != N) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 shared_nvfp4_down_residual: shape mismatch — "
               "activated " << activated.shape() << ", down_weight "
            << down_weight.shape() << ", down_scales " << down_scales.shape()
            << ", routed " << routed.shape() << ", residual "
            << residual.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<SharedDownResidualPrimitive>(s_stream);
    Shape out_shape{static_cast<ShapeElem>(N)};
    return array(
        out_shape,
        bfloat16,
        prim,
        {activated, down_weight, down_scales, routed, residual});
}

// RoutedSwiGLUQmvPrimitive:
//   inputs[0] = input [K] bf16
//   inputs[1] = fused_weight [E, 2N, K/8] uint32 pair-interleaved planes
//   inputs[2] = fused_scales [E, 2N, K/16] uint8 E4M3 scales
//   inputs[3] = indices [R] uint32 routed expert ids
//   output[0] = activated [R*N] bf16
class RoutedSwiGLUQmvPrimitive : public Primitive {
 public:
    explicit RoutedSwiGLUQmvPrimitive(Stream s) : Primitive(s) {}

 private:
    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 RoutedSwiGLUQmvPrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));

        auto kernel = get_laguna_kernel(
            d, "laguna_routed_nvfp4_swiglu_qmv_bf16_v2");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);

        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        enc.set_output_array(out, c++);

        // 1024 groups: 8 expert slots x 128 tiles; threadgroup 2 simdgroups.
        MTL::Size group_dims(64, 1, 1);
        MTL::Size grid_dims(1024, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(RoutedSwiGLUQmvPrimitive)
};

array routed_nvfp4_swiglu_qmv(
    const array& input,
    const array& fused_weight,
    const array& fused_scales,
    const array& indices,
    StreamOrDevice s) {
    int64_t K = input.shape(0);
    int64_t E = fused_weight.shape(0);
    int64_t N = fused_weight.shape(1) / 2;
    int64_t R = indices.shape(0);
    if (input.ndim() != 1 || input.dtype() != bfloat16 ||
        fused_weight.ndim() != 3 || fused_scales.ndim() != 3 ||
        fused_weight.shape(1) != 2 * N ||
        fused_scales.shape(0) != E ||
        fused_scales.shape(1) != 2 * N ||
        fused_scales.shape(2) * 16 != K ||
        fused_weight.shape(2) * 8 != K ||
        indices.ndim() != 1 || indices.dtype() != uint32) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 routed_nvfp4_swiglu_qmv: shape mismatch — "
               "input " << input.shape() << ", fused_weight "
            << fused_weight.shape() << ", fused_scales "
            << fused_scales.shape() << ", indices " << indices.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<RoutedSwiGLUQmvPrimitive>(s_stream);
    Shape out_shape{static_cast<ShapeElem>(R * N)};
    return array(
        out_shape, bfloat16, prim,
        {input, fused_weight, fused_scales, indices});
}

// RoutedDownReducePrimitive:
//   inputs[0] = activated [R*K2] bf16 per-slot swiglu outputs (R=8, K2=512)
//   inputs[1] = down_weight [E, N, K2/8] uint32 per-expert NVFP4 planes
//   inputs[2] = down_scales [128 + E*N*16] uint8 halved group-32 planes
//   inputs[3] = indices [R] uint32 routed expert ids
//   inputs[4] = router_weights [R] float32 routed scores
//   output[0] = routed [N] bf16 = sum_slots(act * w) * 2.5
class RoutedDownReducePrimitive : public Primitive {
 public:
    explicit RoutedDownReducePrimitive(Stream s) : Primitive(s) {}

 private:
    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 RoutedDownReducePrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));

        auto kernel = get_laguna_kernel(
            d, "laguna_routed_nvfp4_down_reduce_bf16_v2");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);

        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        enc.set_output_array(out, c++);

        // 512 tiles x 256 threads (8 simdgroups = 8 expert slots).
        MTL::Size group_dims(256, 1, 1);
        MTL::Size grid_dims(512, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(RoutedDownReducePrimitive)
};

array routed_nvfp4_down_reduce(
    const array& activated,
    const array& down_weight,
    const array& down_scales,
    const array& indices,
    const array& router_weights,
    StreamOrDevice s) {
    int64_t N = 2048;
    int64_t K2 = 512;
    int64_t R = indices.shape(0);
    int64_t E = down_weight.shape(0);
    int64_t scale_rows = down_scales.shape(0) - 128;
    if (activated.ndim() != 1 || activated.dtype() != bfloat16 ||
        activated.shape(0) != R * K2 ||
        down_weight.ndim() != 3 || down_weight.dtype() != uint32 ||
        down_weight.shape(1) != N || down_weight.shape(2) != K2 / 8 ||
        down_scales.ndim() != 1 || down_scales.dtype() != uint8 ||
        down_scales.shape(0) != 128 + E * N * (K2 / 32) ||
        indices.ndim() != 1 || indices.dtype() != uint32 ||
        indices.shape(0) != R ||
        router_weights.ndim() != 1 || router_weights.dtype() != float32 ||
        router_weights.shape(0) != R) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 routed_nvfp4_down_reduce: shape mismatch — "
               "activated " << activated.shape() << ", down_weight "
            << down_weight.shape() << ", down_scales " << down_scales.shape()
            << ", indices " << indices.shape() << ", router_weights "
            << router_weights.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<RoutedDownReducePrimitive>(s_stream);
    Shape out_shape{static_cast<ShapeElem>(N)};
    return array(
        out_shape, bfloat16, prim,
        {activated, down_weight, down_scales, indices, router_weights});
}

// QkNormRopePrimitive: fused Q/K RMSNorm + RoPE (full-YaRN or sliding).
//   inputs[0] = raw_queries [QH*128] bf16
//   inputs[1] = raw_keys [KH*128] bf16
//   inputs[2] = query_weight [128] bf16
//   inputs[3] = key_weight [128] bf16
//   inputs[4] = angles [rotary_pairs*2] float32
//   output[0] = queries [QH*128] bf16
//   output[1] = keys [KH*128] bf16
class QkNormRopePrimitive : public Primitive {
 public:
    QkNormRopePrimitive(Stream s, bool full_yarn)
        : Primitive(s), full_yarn_(full_yarn) {}

 private:
    bool full_yarn_;

    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 QkNormRopePrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        for (auto& out : outputs) {
            out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        }

        const char* kname = full_yarn_
            ? "laguna_full_qk_norm_yarn_bf16_128_v4"
            : "laguna_sliding_qk_norm_rope_bf16_128_v1";
        auto kernel = get_laguna_kernel(d, kname);
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);

        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        for (auto& out : outputs) {
            enc.set_output_array(out, c++);
        }

        int groups = full_yarn_ ? 56 : 72;
        MTL::Size group_dims(32, 1, 1);
        MTL::Size grid_dims(groups, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(QkNormRopePrimitive)
};

std::pair<array, array> full_qk_norm_yarn(
    const array& raw_queries,
    const array& raw_keys,
    const array& query_weight,
    const array& key_weight,
    const array& angles,
    StreamOrDevice s) {
    if (raw_queries.ndim() != 1 || raw_queries.dtype() != bfloat16 ||
        raw_keys.ndim() != 1 || raw_keys.dtype() != bfloat16 ||
        raw_queries.shape(0) != 48 * 128 ||
        raw_keys.shape(0) != 8 * 128 ||
        query_weight.ndim() != 1 || key_weight.ndim() != 1 ||
        query_weight.shape(0) != 128 || key_weight.shape(0) != 128 ||
        angles.ndim() != 1 || angles.dtype() != float32 ||
        angles.shape(0) != 64) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 full_qk_norm_yarn: shape mismatch — "
               "raw_queries " << raw_queries.shape() << ", raw_keys "
            << raw_keys.shape() << ", angles " << angles.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<QkNormRopePrimitive>(s_stream, true);
    Shape q_shape{static_cast<ShapeElem>(48 * 128)};
    Shape k_shape{static_cast<ShapeElem>(8 * 128)};
    auto outs = array::make_arrays(
        {q_shape, k_shape}, {bfloat16, bfloat16}, prim,
        {raw_queries, raw_keys, query_weight, key_weight, angles});
    return {outs[0], outs[1]};
}

std::pair<array, array> sliding_qk_norm_rope(
    const array& raw_queries,
    const array& raw_keys,
    const array& query_weight,
    const array& key_weight,
    const array& angles,
    StreamOrDevice s) {
    if (raw_queries.ndim() != 1 || raw_queries.dtype() != bfloat16 ||
        raw_keys.ndim() != 1 || raw_keys.dtype() != bfloat16 ||
        raw_queries.shape(0) != 64 * 128 ||
        raw_keys.shape(0) != 8 * 128 ||
        query_weight.ndim() != 1 || key_weight.ndim() != 1 ||
        query_weight.shape(0) != 128 || key_weight.shape(0) != 128 ||
        angles.ndim() != 1 || angles.dtype() != float32 ||
        angles.shape(0) != 128) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 sliding_qk_norm_rope: shape mismatch — "
               "raw_queries " << raw_queries.shape() << ", raw_keys "
            << raw_keys.shape() << ", angles " << angles.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<QkNormRopePrimitive>(s_stream, false);
    Shape q_shape{static_cast<ShapeElem>(64 * 128)};
    Shape k_shape{static_cast<ShapeElem>(8 * 128)};
    auto outs = array::make_arrays(
        {q_shape, k_shape}, {bfloat16, bfloat16}, prim,
        {raw_queries, raw_keys, query_weight, key_weight, angles});
    return {outs[0], outs[1]};
}

// PrefillQkNormRopePrimitive: batched (multi-row) fused Q/K RMSNorm + RoPE.
//   inputs[0] = raw_queries [rows*QH*128] bf16
//   inputs[1] = raw_keys [rows*KH*128] bf16
//   inputs[2] = query_weight [128] bf16
//   inputs[3] = key_weight [128] bf16
//   inputs[4] = angles [atlas*2*rotary_pairs] float32
//   inputs[5] = offsets [1] int32 (per-row angle atlas base)
//   output[0] = queries [QH*rows*128] bf16 (head-major, row inner)
//   output[1] = keys [KH*rows*128] bf16
// Grid: ((QH+KH)/hpg, rows) threadgroups; hpg = 4 (128 threads) or 1 (32).
class PrefillQkNormRopePrimitive : public Primitive {
 public:
    PrefillQkNormRopePrimitive(Stream s, bool full_yarn, bool h1)
        : Primitive(s), full_yarn_(full_yarn), h1_(h1) {}

 private:
    bool full_yarn_;
    bool h1_;

    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 PrefillQkNormRopePrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        for (auto& out : outputs) {
            out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        }

        const char* kname = full_yarn_
            ? (h1_ ? "laguna_prefill_full_qk_norm_yarn_bf16_128_h1_v2"
                   : "laguna_prefill_full_qk_norm_yarn_bf16_128_v2")
            : (h1_ ? "laguna_prefill_sliding_qk_norm_rope_bf16_128_h1_v2"
                   : "laguna_prefill_sliding_qk_norm_rope_bf16_128_v2");
        auto kernel = get_laguna_kernel(d, kname);
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        for (auto& out : outputs) {
            enc.set_output_array(out, c++);
        }

        int heads = full_yarn_ ? 48 : 64;
        int rows = static_cast<int>(inputs[0].shape(0) / (heads * 128));
        int heads_per_group = h1_ ? 1 : 4;
        // Swift dispatch is in THREADS: (heads+kv)/hpg * (32*hpg) threads in
        // x, rows in y -> MTL threadgroups = threads / group size.
        MTL::Size group_dims(heads_per_group * 32, 1, 1);
        MTL::Size grid_dims((heads + 8) / heads_per_group, rows, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(PrefillQkNormRopePrimitive)
};

std::pair<array, array> prefill_full_qk_norm_yarn(
    const array& raw_queries,
    const array& raw_keys,
    const array& query_weight,
    const array& key_weight,
    const array& angles,
    const array& offsets,
    bool h1,
    StreamOrDevice s) {
    constexpr int heads = 48;
    constexpr int kv_heads = 8;
    constexpr int rotary_pairs = 32;
    int rows = static_cast<int>(raw_queries.shape(0) / (heads * 128));
    if (raw_queries.ndim() != 1 || raw_queries.dtype() != bfloat16 ||
        raw_queries.shape(0) % (heads * 128) != 0 ||
        raw_keys.ndim() != 1 || raw_keys.dtype() != bfloat16 ||
        raw_keys.shape(0) != rows * kv_heads * 128 ||
        query_weight.ndim() != 1 || key_weight.ndim() != 1 ||
        query_weight.shape(0) != 128 || key_weight.shape(0) != 128 ||
        angles.ndim() != 1 || angles.dtype() != float32 ||
        angles.shape(0) % (2 * rotary_pairs) != 0 ||
        offsets.ndim() != 1 || offsets.dtype() != int32 ||
        offsets.shape(0) < 1) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 prefill_full_qk_norm_yarn: shape mismatch — "
               "raw_queries " << raw_queries.shape() << ", raw_keys "
            << raw_keys.shape() << ", angles " << angles.shape()
            << ", offsets " << offsets.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<PrefillQkNormRopePrimitive>(s_stream, true, h1);
    Shape q_shape{static_cast<ShapeElem>(heads * rows * 128)};
    Shape k_shape{static_cast<ShapeElem>(kv_heads * rows * 128)};
    auto outs = array::make_arrays(
        {q_shape, k_shape}, {bfloat16, bfloat16}, prim,
        {raw_queries, raw_keys, query_weight, key_weight, angles, offsets});
    return {outs[0], outs[1]};
}

std::pair<array, array> prefill_sliding_qk_norm_rope(
    const array& raw_queries,
    const array& raw_keys,
    const array& query_weight,
    const array& key_weight,
    const array& angles,
    const array& offsets,
    bool h1,
    StreamOrDevice s) {
    constexpr int heads = 64;
    constexpr int kv_heads = 8;
    constexpr int rotary_pairs = 64;
    int rows = static_cast<int>(raw_queries.shape(0) / (heads * 128));
    if (raw_queries.ndim() != 1 || raw_queries.dtype() != bfloat16 ||
        raw_queries.shape(0) % (heads * 128) != 0 ||
        raw_keys.ndim() != 1 || raw_keys.dtype() != bfloat16 ||
        raw_keys.shape(0) != rows * kv_heads * 128 ||
        query_weight.ndim() != 1 || key_weight.ndim() != 1 ||
        query_weight.shape(0) != 128 || key_weight.shape(0) != 128 ||
        angles.ndim() != 1 || angles.dtype() != float32 ||
        angles.shape(0) % (2 * rotary_pairs) != 0 ||
        offsets.ndim() != 1 || offsets.dtype() != int32 ||
        offsets.shape(0) < 1) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 prefill_sliding_qk_norm_rope: shape mismatch — "
               "raw_queries " << raw_queries.shape() << ", raw_keys "
            << raw_keys.shape() << ", angles " << angles.shape()
            << ", offsets " << offsets.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<PrefillQkNormRopePrimitive>(s_stream, false, h1);
    Shape q_shape{static_cast<ShapeElem>(heads * rows * 128)};
    Shape k_shape{static_cast<ShapeElem>(kv_heads * rows * 128)};
    auto outs = array::make_arrays(
        {q_shape, k_shape}, {bfloat16, bfloat16}, prim,
        {raw_queries, raw_keys, query_weight, key_weight, angles, offsets});
    return {outs[0], outs[1]};
}

// DecodeQkvR1Primitive: fused Q/K/V NVFP4 projection for one token.
//   inputs[0] = normalized [2048] bf16
//   inputs[1] = weight_codes [rows][1024] uint8
//   inputs[2] = weight_scales [rows][128] uint8
//   output[0] = projected [rows] bf16
class DecodeQkvR1Primitive : public Primitive {
 public:
    DecodeQkvR1Primitive(Stream s, int heads)
        : Primitive(s), heads_(heads) {}

 private:
    int heads_;

    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 DecodeQkvR1Primitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));

        std::string kname = "laguna_decode_nvfp4_qkv_h"
            + std::to_string(heads_) + "_r1_v1_se1_sd1";
        auto kernel = get_laguna_kernel(d, kname);
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);

        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        enc.set_output_array(out, c++);

        int rows = (heads_ + 2 * 8) * 128;
        MTL::Size group_dims(64, 1, 1);
        MTL::Size grid_dims(rows / 2, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(DecodeQkvR1Primitive)
};

array decode_nvfp4_qkv_r1(
    const array& normalized,
    const array& weight_codes,
    const array& weight_scales,
    int heads,
    StreamOrDevice s) {
    int rows = (heads + 2 * 8) * 128;
    if (normalized.ndim() != 1 || normalized.dtype() != bfloat16 ||
        normalized.shape(0) != 2048 ||
        weight_codes.ndim() != 2 || weight_codes.dtype() != uint8 ||
        weight_codes.shape(0) != rows ||
        weight_codes.shape(1) != 1024 ||
        weight_scales.ndim() != 2 || weight_scales.dtype() != uint8 ||
        weight_scales.shape(0) != rows || weight_scales.shape(1) != 128) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 decode_nvfp4_qkv_r1: shape mismatch — "
               "normalized " << normalized.shape() << ", weight_codes "
            << weight_codes.shape() << ", weight_scales "
            << weight_scales.shape() << ", heads " << heads;
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<DecodeQkvR1Primitive>(s_stream, heads);
    Shape out_shape{static_cast<ShapeElem>(rows)};
    return array(out_shape, bfloat16, prim,
                 {normalized, weight_codes, weight_scales});
}

// OProjActPrimitive: gated affine o_proj (pre-activated per-head gate).
//   inputs[0] = attention_output [heads*128] bf16
//   inputs[1] = gate_values [heads] bf16
//   inputs[2] = weight_codes [2048, heads*16] uint32
//   inputs[3] = weight_scales [2048, heads*8] uint8
//   output[0] = projected [2048] bf16
class OProjActPrimitive : public Primitive {
 public:
    OProjActPrimitive(Stream s, int heads)
        : Primitive(s), heads_(heads) {}

 private:
    int heads_;

    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 OProjActPrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));

        std::string kname = "laguna_oproj_act_h"
            + std::to_string(heads_) + "_v1_sc1_se1";
        auto kernel = get_laguna_kernel(d, kname);
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);

        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        enc.set_output_array(out, c++);

        int out_vec = 2048;
        MTL::Size group_dims(64, 1, 1);
        MTL::Size grid_dims((out_vec / 8) * 64 / 64, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(OProjActPrimitive)
};

array oproj_act(
    const array& attention_output,
    const array& gate_values,
    const array& weight_codes,
    const array& weight_scales,
    int heads,
    StreamOrDevice s) {
    int in_vec = heads * 128;
    if (attention_output.ndim() != 1 || attention_output.dtype() != bfloat16 ||
        attention_output.shape(0) != in_vec ||
        gate_values.ndim() != 1 || gate_values.dtype() != bfloat16 ||
        gate_values.shape(0) != heads ||
        weight_codes.ndim() != 2 || weight_codes.dtype() != uint32 ||
        weight_codes.shape(0) != 2048 || weight_codes.shape(1) != in_vec / 8 ||
        weight_scales.ndim() != 2 || weight_scales.dtype() != uint8 ||
        weight_scales.shape(0) != 2048 || weight_scales.shape(1) != in_vec / 16) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 oproj_act: shape mismatch — attention_output "
            << attention_output.shape() << ", gate_values "
            << gate_values.shape() << ", weight_codes "
            << weight_codes.shape() << ", weight_scales "
            << weight_scales.shape() << ", heads " << heads;
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<OProjActPrimitive>(s_stream, heads);
    Shape out_shape{static_cast<ShapeElem>(2048)};
    return array(out_shape, bfloat16, prim,
                 {attention_output, gate_values, weight_codes, weight_scales});
}

// ResidualRmsPrimitive: fused residual add + RMSNorm (2 outputs).
class ResidualRmsPrimitive : public Primitive {
 public:
    explicit ResidualRmsPrimitive(Stream s) : Primitive(s) {}

 private:
    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 ResidualRmsPrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        for (auto& out : outputs) {
            out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        }
        auto kernel = get_laguna_kernel(d, "laguna_residual_rms_bf16_2048_v1");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        for (auto& out : outputs) {
            enc.set_output_array(out, c++);
        }
        int rows = static_cast<int>(inputs[0].shape(0) / 2048);
        MTL::Size group_dims(512, 1, 1);
        MTL::Size grid_dims(rows, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(ResidualRmsPrimitive)
};

std::pair<array, array> residual_rms(
    const array& residual,
    const array& branch,
    const array& weight,
    StreamOrDevice s) {
    if (residual.ndim() != 1 || residual.dtype() != bfloat16 ||
        branch.ndim() != 1 || branch.dtype() != bfloat16 ||
        weight.ndim() != 1 || weight.dtype() != bfloat16 ||
        residual.shape(0) != 2048 || branch.shape(0) != 2048 ||
        weight.shape(0) != 2048) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 residual_rms: shape mismatch — residual "
            << residual.shape() << ", branch " << branch.shape()
            << ", weight " << weight.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<ResidualRmsPrimitive>(s_stream);
    Shape shape{static_cast<ShapeElem>(2048)};
    auto outs = array::make_arrays(
        {shape, shape}, {bfloat16, bfloat16}, prim,
        {residual, branch, weight});
    return {outs[0], outs[1]};
}

// DecodeRouterTop8Primitive: 256-lane bitonic top-8 router (2 outputs).
class DecodeRouterTop8Primitive : public Primitive {
 public:
    DecodeRouterTop8Primitive(Stream s, bool normalizing)
        : Primitive(s), normalizing_(normalizing) {}

 private:
    bool normalizing_;

    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 DecodeRouterTop8Primitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        for (auto& out : outputs) {
            out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        }
        const char* kname = normalizing_
            ? "laguna_decode_router_top8_norm_v2"
            : "laguna_decode_router_top8_v3";
        auto kernel = get_laguna_kernel(d, kname);
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        for (auto& out : outputs) {
            enc.set_output_array(out, c++);
        }
        MTL::Size group_dims(256, 1, 1);
        MTL::Size grid_dims(1, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(DecodeRouterTop8Primitive)
};

std::pair<array, array> decode_router_top8(
    const array& logits,
    const array& correction_bias,
    bool normalizing,
    StreamOrDevice s) {
    if (logits.ndim() != 1 || logits.shape(0) != 256 ||
        correction_bias.ndim() != 1 || correction_bias.shape(0) != 256 ||
        correction_bias.dtype() != float32) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 decode_router_top8: shape mismatch — logits "
            << logits.shape() << ", correction_bias "
            << correction_bias.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<DecodeRouterTop8Primitive>(s_stream, normalizing);
    Shape idx_shape{static_cast<ShapeElem>(8)};
    Shape score_shape{static_cast<ShapeElem>(8)};
    auto outs = array::make_arrays(
        {idx_shape, score_shape}, {uint32, bfloat16}, prim,
        {logits, correction_bias});
    return {outs[0], outs[1]};
}

// SlidingFusedAttnRingPrimitive: fused sliding-attention decode (ring).
//   inputs[0] raw_queries [64*128], [1] raw_keys [8*128], [2] raw_values
//   [3] query_weight [128], [4] key_weight [128], [5] angles [128] fp32
//   [6] k_cache [8][512][128], [7] v_cache [8][512][128]
//   [8] params [1] uint32, [9] scale_arr [1] fp32
//   output[0] attended [64*128] bf16
class SlidingFusedAttnRingPrimitive : public Primitive {
 public:
    explicit SlidingFusedAttnRingPrimitive(Stream s) : Primitive(s) {}

 private:
    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 SlidingFusedAttnRingPrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));

        auto kernel = get_laguna_kernel(
            d, "laguna_sliding_fused_attn_ring_v1");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        enc.set_output_array(out, c++);

        MTL::Size group_dims(1024, 1, 1);
        MTL::Size grid_dims(64 / 2, 1, 1);  // 64 sliding heads, 2 per group
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(SlidingFusedAttnRingPrimitive)
};

array sliding_fused_attn_ring(
    const array& raw_queries,
    const array& raw_keys,
    const array& raw_values,
    const array& query_weight,
    const array& key_weight,
    const array& angles,
    const array& k_cache,
    const array& v_cache,
    const array& params,
    const array& scale_arr,
    StreamOrDevice s) {
    if (raw_queries.ndim() != 1 || raw_queries.dtype() != bfloat16 ||
        raw_queries.shape(0) != 64 * 128 ||
        raw_keys.shape(0) != 8 * 128 || raw_values.shape(0) != 8 * 128 ||
        query_weight.shape(0) != 128 || key_weight.shape(0) != 128 ||
        angles.dtype() != float32 || angles.shape(0) != 128 ||
        k_cache.ndim() != 3 || k_cache.dtype() != bfloat16 ||
        k_cache.shape(0) != 8 || k_cache.shape(1) != 512 ||
        k_cache.shape(2) != 128 ||
        v_cache.ndim() != 3 || v_cache.shape(0) != 8 ||
        v_cache.shape(1) != 512 || v_cache.shape(2) != 128 ||
        params.ndim() != 1 || params.dtype() != uint32 ||
        scale_arr.ndim() != 1 || scale_arr.dtype() != float32) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 sliding_fused_attn_ring: shape mismatch — "
               "raw_queries " << raw_queries.shape() << ", k_cache "
            << k_cache.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<SlidingFusedAttnRingPrimitive>(s_stream);
    Shape out_shape{static_cast<ShapeElem>(64 * 128)};
    return array(out_shape, bfloat16, prim,
                 {raw_queries, raw_keys, raw_values, query_weight, key_weight,
                  angles, k_cache, v_cache, params, scale_arr});
}

// ResidualRmsRouterPrimitive: fused residual add + RMSNorm + router GEMV.
class ResidualRmsRouterPrimitive : public Primitive {
 public:
    explicit ResidualRmsRouterPrimitive(Stream s) : Primitive(s) {}

 private:
    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 ResidualRmsRouterPrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        for (auto& out : outputs) {
            out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        }
        auto kernel = get_laguna_kernel(
            d, "laguna_residual_rms_router_bf16_2048_rpg8");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        for (auto& out : outputs) {
            enc.set_output_array(out, c++);
        }
        // 32 tiles x 512 threads; 2048 is distributed over 32 tiles (64 rows
        // per tile x 8 = ... actually: 256 experts/8 = 32 tiles).
        MTL::Size group_dims(512, 1, 1);
        MTL::Size grid_dims(32, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(ResidualRmsRouterPrimitive)
};

std::vector<array> residual_rms_router(
    const array& residual,
    const array& branch,
    const array& weight,
    const array& router_weight,
    const array& correction_bias,
    StreamOrDevice s) {
    if (residual.ndim() != 1 || residual.dtype() != bfloat16 ||
        residual.shape(0) != 2048 || branch.shape(0) != 2048 ||
        weight.shape(0) != 2048 || weight.ndim() != 1 ||
        router_weight.ndim() != 2 || router_weight.dtype() != bfloat16 ||
        router_weight.shape(0) != 256 || router_weight.shape(1) != 2048 ||
        correction_bias.ndim() != 1 || correction_bias.dtype() != float32 ||
        correction_bias.shape(0) != 256) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 residual_rms_router: shape mismatch — residual "
            << residual.shape() << ", router_weight "
            << router_weight.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<ResidualRmsRouterPrimitive>(s_stream);
    Shape s2048{static_cast<ShapeElem>(2048)};
    Shape s256{static_cast<ShapeElem>(256)};
    auto outs = array::make_arrays(
        {s2048, s2048, s256, s256},
        {bfloat16, bfloat16, bfloat16, uint32}, prim,
        {residual, branch, weight, router_weight, correction_bias});
    return outs;
}

// PrefillMoeTailPrimitive: weighted expert combine + shared + residual.
class PrefillMoeTailPrimitive : public Primitive {
 public:
    explicit PrefillMoeTailPrimitive(Stream s) : Primitive(s) {}

 private:
    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 PrefillMoeTailPrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        auto kernel = get_laguna_kernel(d, "laguna_prefill_moe_tail_bf16_v1");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        enc.set_output_array(out, c++);
        // The Swift wrapper's grid is in THREADS: (hidden/4, rows) threads
        // with a 256-thread group -> (hidden/1024, rows) threadgroups.
        int rows = static_cast<int>(inputs[0].shape(1));
        MTL::Size group_dims(256, 1, 1);
        MTL::Size grid_dims(2048 / 4 / 256, rows, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(PrefillMoeTailPrimitive)
};

array prefill_moe_tail(
    const array& expert_outputs,
    const array& router_weights,
    const array& shared_output,
    const array& residual,
    StreamOrDevice s) {
    // expert_outputs [1, rows, 8, 2048]; router_weights [1, rows, 8];
    // shared_output/residual [1, rows, 2048] (the Swift batched layout).
    int rows = static_cast<int>(expert_outputs.shape(1));
    if (expert_outputs.ndim() != 4 || expert_outputs.dtype() != bfloat16 ||
        expert_outputs.shape(0) != 1 ||
        expert_outputs.shape(2) != 8 || expert_outputs.shape(3) != 2048 ||
        router_weights.ndim() != 3 || router_weights.dtype() != float32 ||
        router_weights.shape(0) != 1 || router_weights.shape(1) != rows ||
        router_weights.shape(2) != 8 ||
        shared_output.ndim() != 3 || shared_output.dtype() != bfloat16 ||
        shared_output.shape(0) != 1 || shared_output.shape(1) != rows ||
        shared_output.shape(2) != 2048 ||
        residual.ndim() != 3 || residual.dtype() != bfloat16 ||
        residual.shape(0) != 1 || residual.shape(1) != rows ||
        residual.shape(2) != 2048) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 prefill_moe_tail: shape mismatch — "
               "expert_outputs " << expert_outputs.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<PrefillMoeTailPrimitive>(s_stream);
    Shape out_shape{static_cast<ShapeElem>(rows * 2048)};
    return array(out_shape, bfloat16, prim,
                 {expert_outputs, router_weights, shared_output, residual});
}

// PrefillSortedMoeTailPrimitive: weighted expert combine + shared + residual
// with an inverse-order permutation gather over the sorted expert plane.
class PrefillSortedMoeTailPrimitive : public Primitive {
 public:
    explicit PrefillSortedMoeTailPrimitive(Stream s) : Primitive(s) {}

 private:
    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 PrefillSortedMoeTailPrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        auto kernel =
            get_laguna_kernel(d, "laguna_prefill_sorted_moe_tail_bf16_v1");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        enc.set_output_array(out, c++);
        // Swift grid is in THREADS: (hidden/4, rows) with a 256-thread
        // group -> (hidden/4/256, rows) threadgroups.
        int rows = static_cast<int>(inputs[0].shape(1));
        MTL::Size group_dims(256, 1, 1);
        MTL::Size grid_dims(2048 / 4 / 256, rows, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(PrefillSortedMoeTailPrimitive)
};

array prefill_sorted_moe_tail(
    const array& sorted_expert_outputs,
    const array& inverse_order,
    const array& router_weights,
    const array& shared_output,
    const array& residual,
    StreamOrDevice s) {
    // sorted_expert_outputs [1, rows, 8, 2048]; inverse_order [rows*8]
    // uint32; router_weights [1, rows, 8]; shared_output/residual
    // [1, rows, 2048] (the Swift batched layout).
    int rows = static_cast<int>(sorted_expert_outputs.shape(1));
    if (sorted_expert_outputs.ndim() != 4 ||
        sorted_expert_outputs.dtype() != bfloat16 ||
        sorted_expert_outputs.shape(0) != 1 ||
        sorted_expert_outputs.shape(2) != 8 ||
        sorted_expert_outputs.shape(3) != 2048 ||
        inverse_order.ndim() != 1 || inverse_order.dtype() != uint32 ||
        inverse_order.shape(0) != rows * 8 ||
        router_weights.ndim() != 3 || router_weights.dtype() != float32 ||
        router_weights.shape(0) != 1 || router_weights.shape(1) != rows ||
        router_weights.shape(2) != 8 ||
        shared_output.ndim() != 3 || shared_output.dtype() != bfloat16 ||
        shared_output.shape(0) != 1 || shared_output.shape(1) != rows ||
        shared_output.shape(2) != 2048 ||
        residual.ndim() != 3 || residual.dtype() != bfloat16 ||
        residual.shape(0) != 1 || residual.shape(1) != rows ||
        residual.shape(2) != 2048) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 prefill_sorted_moe_tail: shape mismatch — "
               "sorted_expert_outputs " << sorted_expert_outputs.shape()
            << ", inverse_order " << inverse_order.shape()
            << ", router_weights " << router_weights.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<PrefillSortedMoeTailPrimitive>(s_stream);
    Shape out_shape{static_cast<ShapeElem>(rows * 2048)};
    return array(out_shape, bfloat16, prim,
                 {sorted_expert_outputs, inverse_order, router_weights,
                  shared_output, residual});
}

// PrefillRouterTournamentPrimitive: batched 2-phase bitonic router.
class PrefillRouterTournamentPrimitive : public Primitive {
 public:
    explicit PrefillRouterTournamentPrimitive(Stream s) : Primitive(s) {}

 private:
    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error(
            "laguna_nvfp4 PrefillRouterTournamentPrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        for (auto& out : outputs) {
            out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        }
        auto kernel = get_laguna_kernel(
            d, "laguna_prefill_router_tournament_v1");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) {
            enc.set_input_array(in, c++);
        }
        for (auto& out : outputs) {
            enc.set_output_array(out, c++);
        }
        int rows = static_cast<int>(inputs[0].shape(0) / 256);
        MTL::Size group_dims(256, 1, 1);
        MTL::Size grid_dims(1, rows, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    DEFINE_NAME(PrefillRouterTournamentPrimitive)
};

std::pair<array, array> prefill_router_tournament(
    const array& logits,
    const array& correction_bias,
    StreamOrDevice s) {
    int rows = static_cast<int>(logits.shape(0) / 256);
    if (logits.ndim() != 1 || logits.shape(0) % 256 != 0 ||
        correction_bias.ndim() != 1 || correction_bias.dtype() != float32 ||
        correction_bias.shape(0) != 256) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 prefill_router_tournament: shape mismatch — "
               "logits " << logits.shape() << ", correction_bias "
            << correction_bias.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<PrefillRouterTournamentPrimitive>(s_stream);
    Shape idx_shape{static_cast<ShapeElem>(rows * 8)};
    Shape score_shape{static_cast<ShapeElem>(rows * 8)};
    auto outs = array::make_arrays(
        {idx_shape, score_shape}, {uint32, bfloat16}, prim,
        {logits, correction_bias});
    return {outs[0], outs[1]};
}

// ── LM-head prune primitives ───────────────────────────────────────────

class LmCoarsePrimitive : public Primitive {
 public:
    explicit LmCoarsePrimitive(Stream s) : Primitive(s) {}
 private:
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("laguna_nvfp4 LmCoarsePrimitive has no CPU path.");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        for (auto& out : outputs) out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        auto kernel = get_laguna_kernel(d, "laguna_lmhead_int5_coarse_ratio_bound_delta_bf16_v6");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) enc.set_input_array(in, c++);
        for (auto& out : outputs) enc.set_output_array(out, c++);
        int vocab = static_cast<int>(inputs[1].shape(0));
        // The Swift launch is (vocab/16 * 512, 1, 1) with a 512-thread group
        // = 16 simdgroups; each simdgroup computes ONE row (row = tg*16 + sg).
        MTL::Size group_dims(512, 1, 1);
        MTL::Size grid_dims(vocab / 16, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }
    DEFINE_NAME(LmCoarsePrimitive)
};

class LmArgmaxStage1Primitive : public Primitive {
 public:
    explicit LmArgmaxStage1Primitive(Stream s) : Primitive(s) {}
 private:
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("laguna_nvfp4 LmArgmaxStage1Primitive has no CPU path.");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        for (auto& out : outputs) out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        auto kernel = get_laguna_kernel(d, "laguna_lmhead_coarse_argmax_stage1_v5");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) enc.set_input_array(in, c++);
        for (auto& out : outputs) enc.set_output_array(out, c++);
        int vocab = static_cast<int>(inputs[0].shape(0));
        int partials = vocab / 784;  // 128
        MTL::Size group_dims(224, 1, 1);  // 224 threads per threadgroup
        MTL::Size grid_dims(1, partials, 1);  // row = grid.y
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }
    DEFINE_NAME(LmArgmaxStage1Primitive)
};

class LmThresholdPrimitive : public Primitive {
 public:
    explicit LmThresholdPrimitive(Stream s) : Primitive(s) {}
 private:
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("laguna_nvfp4 LmThresholdPrimitive has no CPU path.");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        auto kernel = get_laguna_kernel(d, "laguna_lmhead_exact_winner_bf16_midpoint_threshold_v1");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) enc.set_input_array(in, c++);
        enc.set_output_array(out, c++);
        MTL::Size group_dims(32, 1, 1);
        MTL::Size grid_dims(1, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }
    DEFINE_NAME(LmThresholdPrimitive)
};

class LmInlineExactPrimitive : public Primitive {
 public:
    explicit LmInlineExactPrimitive(Stream s) : Primitive(s) {}
 private:
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("laguna_nvfp4 LmInlineExactPrimitive has no CPU path.");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        auto kernel = get_laguna_kernel(d, "laguna_lmhead_exact_inline_mask_block_delta_bf16_lane0_mask_v1");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) enc.set_input_array(in, c++);
        enc.set_output_array(out, c++);
        int vocab = static_cast<int>(inputs[0].shape(0));
        MTL::Size group_dims(256, 1, 1);
        MTL::Size grid_dims(vocab / 32, 1, 1);  // 4 rows per simdgroup x 8
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }
    DEFINE_NAME(LmInlineExactPrimitive)
};

array lm_head_prune(
    const array& x,
    const array& codes_lo,
    const array& codes_hi,
    const array& scales,
    const array& lm_head,
    StreamOrDevice s) {
    int vocab = static_cast<int>(codes_lo.shape(0));
    auto s_stream = to_stream(s);

    // 1. coarse + delta
    auto coarse_prim = std::make_shared<LmCoarsePrimitive>(s_stream);
    Shape cv{vocab};
    auto coarse_out = array::make_arrays(
        {cv, cv}, {float32, bfloat16}, coarse_prim, {x, codes_lo, codes_hi, scales});
    auto& coarse = coarse_out[0];
    auto& delta = coarse_out[1];

    // 2. argmax stage 1 -> 128 partials
    int partials = vocab / 784;
    auto arg_prim = std::make_shared<LmArgmaxStage1Primitive>(s_stream);
    Shape pv{partials};
    auto partials_out = array::make_arrays(
        {pv, pv}, {float32, uint32}, arg_prim, {coarse});
    auto& pmax = partials_out[0];
    auto& pidx = partials_out[1];

    // 3. winner threshold
    auto thr_prim = std::make_shared<LmThresholdPrimitive>(s_stream);
    Shape tv{1};
    auto thr = array(tv, float32, thr_prim, {pmax, pidx, lm_head, x});

    // 4. inline exact
    auto inline_prim = std::make_shared<LmInlineExactPrimitive>(s_stream);
    Shape av{vocab};
    return array(av, bfloat16, inline_prim, {coarse, delta, thr, lm_head, x});
}

// DenseGateUpSwiGLUPrimitive: bf16 dense gate/up fused + SwiGLU.
class DenseGateUpSwiGLUPrimitive : public Primitive {
 public:
    explicit DenseGateUpSwiGLUPrimitive(Stream s) : Primitive(s) {}
 private:
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("laguna_nvfp4 DenseGateUpSwiGLUPrimitive has no CPU path.");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        auto kernel = get_laguna_kernel(d, "laguna_dense_gate_up_swiglu_bf16_v1");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) enc.set_input_array(in, c++);
        enc.set_output_array(out, c++);
        MTL::Size group_dims(64, 1, 1);       // 2 simdgroups
        MTL::Size grid_dims(8192 / 64, 1, 1); // 128 groups
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }
    DEFINE_NAME(DenseGateUpSwiGLUPrimitive)
};

array dense_gate_up_swiglu(
    const array& input,
    const array& fused_weight,
    StreamOrDevice s) {
    if (input.ndim() != 1 || input.dtype() != bfloat16 ||
        input.shape(0) != 2048 ||
        fused_weight.ndim() != 2 || fused_weight.dtype() != bfloat16 ||
        fused_weight.shape(0) != 2 * 8192 || fused_weight.shape(1) != 2048) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 dense_gate_up_swiglu: shape mismatch — input "
            << input.shape() << ", fused_weight " << fused_weight.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<DenseGateUpSwiGLUPrimitive>(s_stream);
    Shape out_shape{static_cast<ShapeElem>(8192)};
    return array(out_shape, bfloat16, prim, {input, fused_weight});
}

// DenseDownResidualPrimitive: bf16 dense down + residual.
class DenseDownResidualPrimitive : public Primitive {
 public:
    explicit DenseDownResidualPrimitive(Stream s) : Primitive(s) {}
 private:
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("laguna_nvfp4 DenseDownResidualPrimitive has no CPU path.");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));
        auto kernel = get_laguna_kernel(d, "laguna_dense_down_residual_bf16_v1");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        int c = 0;
        for (const auto& in : inputs) enc.set_input_array(in, c++);
        enc.set_output_array(out, c++);
        MTL::Size group_dims(64, 1, 1);
        MTL::Size grid_dims(2048 / 16, 1, 1); // 128 groups, 16 rows each
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }
    DEFINE_NAME(DenseDownResidualPrimitive)
};

array dense_down_residual(
    const array& activated,
    const array& down_weight,
    const array& residual,
    StreamOrDevice s) {
    if (activated.ndim() != 1 || activated.dtype() != bfloat16 ||
        activated.shape(0) != 8192 ||
        down_weight.ndim() != 2 || down_weight.dtype() != bfloat16 ||
        down_weight.shape(0) != 2048 || down_weight.shape(1) != 8192 ||
        residual.ndim() != 1 || residual.dtype() != bfloat16 ||
        residual.shape(0) != 2048) {
        std::ostringstream msg;
        msg << "laguna_nvfp4 dense_down_residual: shape mismatch — activated "
            << activated.shape() << ", down_weight " << down_weight.shape();
        throw std::invalid_argument(msg.str());
    }
    auto s_stream = to_stream(s);
    auto prim = std::make_shared<DenseDownResidualPrimitive>(s_stream);
    Shape out_shape{static_cast<ShapeElem>(2048)};
    return array(out_shape, bfloat16, prim, {activated, down_weight, residual});
}

int64_t abi_probe(const array& a) {
    return static_cast<int64_t>(a.size());
}

}  // namespace omlx::laguna_nvfp4
