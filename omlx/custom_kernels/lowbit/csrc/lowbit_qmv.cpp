// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// Bonsai t5 (base-3 ternary) Metal kernel dispatch.
//
// Metal kernel sources live in lowbit_qmv.metal (qmv_fast / qmv_wide / qmm),
// compiled into omlx_lowbit_kernels.metallib.
// The metallib is loaded lazily on the first dispatch call and cached.
//
// MLX 0.32+ requires Metal dispatch to occur inside Primitive::eval_gpu.
// All public API functions return an unevaluated array whose Primitive drives
// the actual Metal dispatch at eval time.

#include "lowbit_qmv.h"
#include "omlx/binary_dir.h"

#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <sstream>
#include <string>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace omlx::lowbit {

namespace {

using namespace mlx::core;
using namespace mlx::core::metal;

// ---------------------------------------------------------------------------
// Metallib loader
// ---------------------------------------------------------------------------

constexpr const char* kMetallibName = "omlx_lowbit_kernels";

MTL::ComputePipelineState* get_lowbit_kernel(
    metal::Device& d,
    const std::string& kernel_name) {
    auto* lib = d.get_library(kMetallibName, binary_dir());
    return d.get_kernel(kernel_name, lib);
}

// ---------------------------------------------------------------------------
// Type string helper
// ---------------------------------------------------------------------------

std::string type_str(Dtype dt) {
    if (dt == float16)  return "float16_t";
    if (dt == bfloat16) return "bfloat16_t";
    if (dt == float32)  return "float";
    std::ostringstream msg;
    msg << "lowbit: unsupported dtype " << dt;
    throw std::invalid_argument(msg.str());
}

// ---------------------------------------------------------------------------
// Contiguity helper (used in public API before Primitive is created)
// ---------------------------------------------------------------------------

array ensure_row_contiguous(const array& x, const Stream& s) {
    return contiguous(x, /*allow_col_major=*/false, s);
}

// t5 weight tensor: (N, n_groups * bytes_per_group) uint8.
// bytes_per_group = ceil(group_size / 5): 26 for gs=128, 13 for gs=64.
int derive_t5_group_size(const array& w, const array& scales) {
    int64_t n_groups = scales.shape(-1);
    if (n_groups <= 0)
        throw std::invalid_argument("t5: scales has 0 groups");
    int64_t bpg = w.shape(-1) / n_groups;
    if (bpg == 26) return 128;
    if (bpg == 13) return 64;
    std::ostringstream msg;
    msg << "t5: unrecognised bytes_per_group=" << bpg
        << " (expected 26 for gs=128 or 13 for gs=64)";
    throw std::invalid_argument(msg.str());
}

// ---------------------------------------------------------------------------
// t5 kernel name helpers
// ---------------------------------------------------------------------------

// affine_qmv_fast_t5_<type>_gs_<gs>
std::string qmv_fast_t5_kname(const std::string& type, int group_size) {
    return "affine_qmv_fast_t5_" + type
        + "_gs_" + std::to_string(group_size);
}

// affine_qmv_wide_t5_<type>_gs_<gs>_nv_<nv>_kl_<kl>
std::string qmv_wide_t5_kname(
    const std::string& type, int group_size, int vecs_per_tg, int k_lanes) {
    return "affine_qmv_wide_t5_" + type
        + "_gs_" + std::to_string(group_size)
        + "_nv_" + std::to_string(vecs_per_tg)
        + "_kl_" + std::to_string(k_lanes);
}

// ---------------------------------------------------------------------------
// t5 dispatch functions
// ---------------------------------------------------------------------------

// Buffer layout for t5 kernels: w(0), scales(1), x(2), y(3), K(4), N(5)
// (no biases — t5 is always symmetric)
void dispatch_qmv_fast_t5(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int M, int N, int K,
    int group_size,
    metal::Device& d,
    const Stream& s) {

    std::string kname = qmv_fast_t5_kname(type_str(x.dtype()), group_size);

    auto kernel = get_lowbit_kernel(d, kname);
    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);

    int c = 0;
    enc.set_input_array(w,      c++);
    enc.set_input_array(scales, c++);
    enc.set_input_array(x,      c++);
    enc.set_output_array(out,   c++);
    enc.set_bytes(K, c++);
    enc.set_bytes(N, c++);

    int bn = 16, bk = 32;
    MTL::Size group_dims(bk, 4, 1);
    MTL::Size grid_dims(M, (N + bn - 1) / bn, 1);
    enc.dispatch_threadgroups(grid_dims, group_dims);
}

void dispatch_qmv_wide_t5(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int M, int N, int K,
    int group_size,
    metal::Device& d,
    const Stream& s) {

    int n_tiles = (M + 4) / 5;
    int vecs_per_tg = (M + n_tiles - 1) / n_tiles;
    int k_lanes = 8;
    int num_simdgroups = 4;
    int rows_per_tg = (32 / k_lanes) * num_simdgroups;

    std::string kname = qmv_wide_t5_kname(
        type_str(x.dtype()), group_size, vecs_per_tg, k_lanes);

    auto kernel = get_lowbit_kernel(d, kname);
    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);

    int c = 0;
    enc.set_input_array(w,      c++);
    enc.set_input_array(scales, c++);
    enc.set_input_array(x,      c++);
    enc.set_output_array(out,   c++);
    enc.set_bytes(K, c++);
    enc.set_bytes(N, c++);
    enc.set_bytes(M, c++);

    MTL::Size group_dims(32, num_simdgroups, 1);
    MTL::Size grid_dims(
        (M + vecs_per_tg - 1) / vecs_per_tg,
        (N + rows_per_tg - 1) / rows_per_tg,
        1);
    enc.dispatch_threadgroups(grid_dims, group_dims);
}

// T5QmvPrimitive: t5 base-3 ternary decode (Identity I-D).
//   inputs[0] = x      (row-contiguous activations)
//   inputs[1] = w      (uint8 t5 weight bytes, (N, n_groups*bytes_per_group))
//   inputs[2] = scales ((N, n_groups) scale per group; no biases)
class T5QmvPrimitive : public Primitive {
 public:
    explicit T5QmvPrimitive(Stream s, bool wide)
        : Primitive(s), wide_(wide) {}

 private:
    bool wide_;

    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error("T5QmvPrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));

        const auto& x      = inputs[0];
        const auto& w      = inputs[1];
        const auto& scales = inputs[2];

        int group_size = derive_t5_group_size(w, scales);
        int N          = static_cast<int>(w.shape(-2));
        int n_groups   = static_cast<int>(scales.shape(-1));
        int K          = n_groups * group_size;
        int M          = static_cast<int>(x.size()) / K;

        if (wide_) {
            dispatch_qmv_wide_t5(x, w, scales, out,
                                 M, N, K, group_size, d, s);
        } else {
            dispatch_qmv_fast_t5(x, w, scales, out,
                                 M, N, K, group_size, d, s);
        }
    }

    DEFINE_NAME(T5QmvPrimitive)
};

// ---------------------------------------------------------------------------
// T5QmmPrimitive: t5 MMA GEMM for prefill (Identity I-M).
//   inputs[0] = x      (M, K) row-contiguous activations
//   inputs[1] = w      (N, n_groups*bpg) uint8 t5 bytes
//   inputs[2] = scales (N, n_groups)
// ---------------------------------------------------------------------------

// affine_qmm_t5_<type>_gs_<gs>
static std::string qmm_t5_kname(const std::string& type, int group_size) {
    return "affine_qmm_t5_" + type + "_gs_" + std::to_string(group_size);
}

static void dispatch_qmm_t5(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int M, int N, int K,
    int group_size,
    metal::Device& d,
    const Stream& s) {

    std::string kname = qmm_t5_kname(type_str(x.dtype()), group_size);
    auto kernel = get_lowbit_kernel(d, kname);
    auto& enc   = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);

    int c = 0;
    enc.set_input_array(w,      c++);
    enc.set_input_array(scales, c++);
    enc.set_input_array(x,      c++);
    enc.set_output_array(out,   c++);
    enc.set_bytes(M, c++);
    enc.set_bytes(N, c++);
    enc.set_bytes(K, c++);

    // Grid: (ceil(N/32), ceil(M/32))  TG: (32, 4, 1)
    MTL::Size group_dims(32, 4, 1);
    MTL::Size grid_dims((N + 31) / 32, (M + 31) / 32, 1);
    enc.dispatch_threadgroups(grid_dims, group_dims);
}

class T5QmmPrimitive : public Primitive {
 public:
    explicit T5QmmPrimitive(Stream s) : Primitive(s) {}

 private:
    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error("T5QmmPrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s  = stream();
        auto& d  = metal::device(s.device);
        auto& out = outputs[0];
        out.set_data(mlx::core::allocator::malloc(out.nbytes()));

        const auto& x      = inputs[0];
        const auto& w      = inputs[1];
        const auto& scales = inputs[2];

        int group_size = derive_t5_group_size(w, scales);
        int N          = static_cast<int>(w.shape(-2));
        int n_groups   = static_cast<int>(scales.shape(-1));
        int K          = n_groups * group_size;
        int M          = static_cast<int>(x.size()) / K;

        dispatch_qmm_t5(x, w, scales, out, M, N, K, group_size, d, s);
    }

    DEFINE_NAME(T5QmmPrimitive)
};

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Helper: ensure scales and biases match x's dtype so the Metal kernel
// (which reads them as T = x's dtype) gets correct data.
static array ensure_dtype(const array& a, Dtype dt, const Stream& s) {
    return (a.dtype() == dt) ? a : astype(a, dt, s);
}

array t5_qmv(
    const array& x,
    const array& w,
    const array& scales,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto x_c = ensure_row_contiguous(x, s);
    auto sc  = ensure_dtype(scales, x_c.dtype(), s);
    int N = static_cast<int>(w.shape(-2));
    auto out_shape = x_c.shape();
    out_shape.back() = N;
    return array(out_shape, x_c.dtype(),
        std::make_shared<T5QmvPrimitive>(s, /*wide=*/false),
        {x_c, w, sc});
}

array t5_qmv_wide(
    const array& x,
    const array& w,
    const array& scales,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto x_c = ensure_row_contiguous(x, s);
    auto sc  = ensure_dtype(scales, x_c.dtype(), s);
    int N = static_cast<int>(w.shape(-2));
    auto out_shape = x_c.shape();
    out_shape.back() = N;
    return array(out_shape, x_c.dtype(),
        std::make_shared<T5QmvPrimitive>(s, /*wide=*/true),
        {x_c, w, sc});
}

array t5_qmm(
    const array& x,
    const array& w,
    const array& scales,
    StreamOrDevice s_) {
    auto s   = to_stream(s_);
    auto x_c = ensure_row_contiguous(x, s);
    auto sc  = ensure_dtype(scales, x_c.dtype(), s);
    int N = static_cast<int>(w.shape(-2));
    // Output shape: same as x but last dim replaced with N
    auto out_shape = x_c.shape();
    out_shape.back() = N;
    return array(out_shape, x_c.dtype(),
        std::make_shared<T5QmmPrimitive>(s),
        {x_c, w, sc});
}

} // namespace omlx::lowbit
