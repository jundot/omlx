// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// Bonsai 1-bit / 2-bit Metal kernel dispatch.
//
// Metal kernel sources live in bonsai_quantized.metal (qmv_fast / qmv_wide)
// and spec_decode.metal, compiled into omlx_bonsai_kernels.metallib by CMake.
// The metallib is loaded lazily on the first dispatch call and cached.

#include "bonsai_kernels.h"

#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <sstream>
#include <string>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::bonsai_kernels {

namespace {

using namespace mlx::core;
using namespace mlx::core::metal;

// ---------------------------------------------------------------------------
// Metallib loader
// ---------------------------------------------------------------------------

constexpr const char* kMetallibName = "omlx_bonsai_kernels";

static std::atomic<bool> metallib_ok{true};

std::string binary_dir() {
    static std::string dir = []() {
        Dl_info info;
        if (!dladdr(reinterpret_cast<void*>(&binary_dir), &info)) {
            throw std::runtime_error("bonsai: unable to resolve binary dir.");
        }
        return std::filesystem::path(info.dli_fname).parent_path().string();
    }();
    return dir;
}

MTL::ComputePipelineState* get_bonsai_kernel(
    Device& d,
    const std::string& kernel_name) {
    std::string lib_path =
        binary_dir() + "/" + kMetallibName + ".metallib";
    return d.get_kernel(kernel_name, lib_path);
}

// ---------------------------------------------------------------------------
// Type string helper
// ---------------------------------------------------------------------------

std::string type_str(Dtype dt) {
    if (dt == float16)  return "float16_t";
    if (dt == bfloat16) return "bfloat16_t";
    if (dt == float32)  return "float";
    std::ostringstream msg;
    msg << "bonsai: unsupported dtype " << dt;
    throw std::invalid_argument(msg.str());
}

// ---------------------------------------------------------------------------
// Contiguity helpers
// ---------------------------------------------------------------------------

array ensure_row_contiguous(const array& x, const Stream& s) {
    if (x.flags().row_contiguous) return x;
    array c = contiguous_copy_gpu(x, s);
    metal::get_command_encoder(s).add_temporary(c);
    return c;
}

// ---------------------------------------------------------------------------
// Kernel name construction
// ---------------------------------------------------------------------------

// affine_qmv_fast_<type>_gs_<gs>_b_<bits>_batch_<0|1>
std::string qmv_fast_kname(
    const std::string& type, int group_size, int bits, bool batched) {
    return "affine_qmv_fast_" + type
        + "_gs_" + std::to_string(group_size)
        + "_b_"  + std::to_string(bits)
        + (batched ? "_batch_1" : "_batch_0");
}

// affine_qmv_wide_<type>_gs_<gs>_b_<bits>_nv_<nv>_kl_<kl>_batch_<0|1>
std::string qmv_wide_kname(
    const std::string& type, int group_size, int bits,
    int vecs_per_tg, int k_lanes, bool batched) {
    return "affine_qmv_wide_" + type
        + "_gs_" + std::to_string(group_size)
        + "_b_"  + std::to_string(bits)
        + "_nv_" + std::to_string(vecs_per_tg)
        + "_kl_" + std::to_string(k_lanes)
        + (batched ? "_batch_1" : "_batch_0");
}

// ---------------------------------------------------------------------------
// Group size derivation: K*bits/8 = w.shape(-1), K/group_size = scales.shape(-1)
// ---------------------------------------------------------------------------

int derive_group_size(const array& w, const array& scales, int bits) {
    int64_t K = static_cast<int64_t>(w.shape(-1)) * 8 / bits;
    int64_t n_groups = scales.shape(-1);
    if (n_groups <= 0) return 64;
    return static_cast<int>(K / n_groups);
}

// ---------------------------------------------------------------------------
// qmv_fast dispatch (1-bit single-row)
// ---------------------------------------------------------------------------

void dispatch_qmv_fast(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int M, int N, int K,
    int group_size, int bits,
    Device& d,
    const Stream& s) {

    int B = static_cast<int>(out.size()) / M / N;
    bool batched = B > 1;
    bool fast_aligned = (N % 8 == 0) && (K % 512 == 0);
    std::string variant = fast_aligned ? "qmv_fast" : "qmv";
    std::string kname = (fast_aligned
        ? qmv_fast_kname(type_str(x.dtype()), group_size, bits, batched)
        : ("affine_qmv_" + type_str(x.dtype())
            + "_gs_" + std::to_string(group_size)
            + "_b_"  + std::to_string(bits)
            + (batched ? "_batch_1" : "_batch_0")));

    auto kernel = get_bonsai_kernel(d, kname);
    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);

    // Buffer layout: w, scales, biases, x, out, K, N
    int c = 0;
    enc.set_input_array(w,      c++);
    enc.set_input_array(scales, c++);
    enc.set_input_array(biases, c++);
    enc.set_input_array(x,      c++);
    enc.set_output_array(out,   c++);
    enc.set_bytes(K, c++);
    enc.set_bytes(N, c++);

    int bn = 8, bk = 32;
    MTL::Size group_dims(bk, 2, 1);
    MTL::Size grid_dims(M, (N + bn - 1) / bn, B);
    enc.dispatch_threadgroups(grid_dims, group_dims);
}

// ---------------------------------------------------------------------------
// qmv_wide dispatch (2-bit small-batch)
// ---------------------------------------------------------------------------

void dispatch_qmv_wide(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int M, int N, int K,
    int group_size, int bits,
    Device& d,
    const Stream& s) {

    int B = static_cast<int>(out.size()) / M / N;
    bool batched = B > 1;

    // Tile size: ceil(M/ceil(M/5)), capped at 5
    int n_tiles = (M + 4) / 5;
    int vecs_per_tg = (M + n_tiles - 1) / n_tiles;
    // affine mode uses k_lanes=8 (more rows/simdgroup)
    int k_lanes = 8;
    int num_simdgroups = 2;
    int rows_per_tg = (32 / k_lanes) * num_simdgroups;

    std::string kname = qmv_wide_kname(
        type_str(x.dtype()), group_size, bits, vecs_per_tg, k_lanes, batched);

    auto kernel = get_bonsai_kernel(d, kname);
    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);

    int c = 0;
    enc.set_input_array(w,      c++);
    enc.set_input_array(scales, c++);
    enc.set_input_array(biases, c++);
    enc.set_input_array(x,      c++);
    enc.set_output_array(out,   c++);
    enc.set_bytes(K, c++);
    enc.set_bytes(N, c++);
    enc.set_bytes(M, c++);

    MTL::Size group_dims(32, num_simdgroups, 1);
    MTL::Size grid_dims(
        (M + vecs_per_tg - 1) / vecs_per_tg,
        (N + rows_per_tg - 1) / rows_per_tg,
        B);
    enc.dispatch_threadgroups(grid_dims, group_dims);
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

array bonsai_q1_affine_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto& d = metal::device(s.device);

    auto x_c = ensure_row_contiguous(x, s);
    int64_t K = static_cast<int64_t>(w.shape(-1)) * 8;  // 1-bit: 8 per byte
    int N = static_cast<int>(w.shape(-2));
    int M = static_cast<int>(x_c.size()) / static_cast<int>(K);
    int group_size = derive_group_size(w, scales, 1);

    auto out_shape = x_c.shape();
    out_shape.back() = N;
    array out(out_shape, x_c.dtype(), nullptr, {});
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    dispatch_qmv_fast(x_c, w, scales, biases, out, M, N, static_cast<int>(K),
                      group_size, 1, d, s);
    return out;
}

array bonsai_q2_affine_qmv_wide(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto& d = metal::device(s.device);

    auto x_c = ensure_row_contiguous(x, s);
    int64_t K = static_cast<int64_t>(w.shape(-1)) * 4;  // 2-bit: 4 per byte
    int N = static_cast<int>(w.shape(-2));
    int M = static_cast<int>(x_c.size()) / static_cast<int>(K);
    int group_size = derive_group_size(w, scales, 2);

    auto out_shape = x_c.shape();
    out_shape.back() = N;
    array out(out_shape, x_c.dtype(), nullptr, {});
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    dispatch_qmv_wide(x_c, w, scales, biases, out, M, N, static_cast<int>(K),
                      group_size, 2, d, s);
    return out;
}

std::pair<array, array> bonsai_spec_decode_verify(
    const array& draft,
    const array& target,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto& d = metal::device(s.device);

    int B = draft.shape(0);
    int K = draft.shape(1);

    auto n_accepted = array({B},     mlx::core::int32, nullptr, {});
    auto committed  = array({B, K+1}, mlx::core::int32, nullptr, {});
    n_accepted.set_data(mlx::core::allocator::malloc(n_accepted.nbytes()));
    committed.set_data(mlx::core::allocator::malloc(committed.nbytes()));

    auto kernel = get_bonsai_kernel(d, "spec_decode_verify");
    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);

    enc.set_input_array(draft,      0);
    enc.set_input_array(target,     1);
    enc.set_output_array(n_accepted, 2);
    enc.set_output_array(committed,  3);
    enc.set_bytes(K, 4);
    enc.set_bytes(B, 5);

    int tgroup = std::min(B, 256);
    MTL::Size grid_dims(B, 1, 1);
    MTL::Size group_dims(tgroup, 1, 1);
    enc.dispatch_threads(grid_dims, group_dims);

    return {n_accepted, committed};
}

bool is_nax_available() {
    try {
        auto& d = metal::device(mlx::core::Device::gpu);
        // Require gen >= 18 (gen-17 computes wrong results with NAX qmm/gemm,
        // see Bonsai MLX fork commit 4446b4e6).
        return d.get_architecture_gen() >= 18;
    } catch (...) {
        return false;
    }
}

} // namespace omlx::bonsai_kernels
