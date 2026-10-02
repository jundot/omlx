// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// Bonsai 1-bit / 2-bit Metal kernel dispatch.
//
// Metal kernel sources live in bonsai_quantized.metal (qmv_fast / qmv_wide)
// and spec_decode.metal, compiled into omlx_bonsai_kernels.metallib by CMake.
// The metallib is loaded lazily on the first dispatch call and cached.
//
// MLX 0.32+ requires Metal dispatch to occur inside Primitive::eval_gpu.
// All public API functions return an unevaluated array whose Primitive drives
// the actual Metal dispatch at eval time.

#include "bonsai_kernels.h"

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

namespace omlx::bonsai_kernels {

namespace {

using namespace mlx::core;
using namespace mlx::core::metal;

// ---------------------------------------------------------------------------
// Metallib loader
// ---------------------------------------------------------------------------

constexpr const char* kMetallibName = "omlx_bonsai_kernels";

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
    msg << "bonsai: unsupported dtype " << dt;
    throw std::invalid_argument(msg.str());
}

// ---------------------------------------------------------------------------
// Contiguity helper (used in public API before Primitive is created)
// ---------------------------------------------------------------------------

array ensure_row_contiguous(const array& x, const Stream& s) {
    if (x.flags().row_contiguous) return x;
    return contiguous(x, /*allow_col_major=*/false, s);
}

// ---------------------------------------------------------------------------
// Kernel name construction
// ---------------------------------------------------------------------------

// affine_qmv_fast_[sym_]<type>_gs_<gs>_b_<bits>_batch_<0|1>
std::string qmv_fast_kname(
    const std::string& type, int group_size, int bits, bool batched,
    bool symmetric = false) {
    return std::string(symmetric ? "affine_qmv_fast_sym_" : "affine_qmv_fast_")
        + type
        + "_gs_" + std::to_string(group_size)
        + "_b_"  + std::to_string(bits)
        + (batched ? "_batch_1" : "_batch_0");
}

// affine_qmv_wide_[sym_]<type>_gs_<gs>_b_<bits>_nv_<nv>_kl_<kl>_batch_<0|1>
std::string qmv_wide_kname(
    const std::string& type, int group_size, int bits,
    int vecs_per_tg, int k_lanes, bool batched,
    bool symmetric = false) {
    return std::string(symmetric ? "affine_qmv_wide_sym_" : "affine_qmv_wide_")
        + type
        + "_gs_" + std::to_string(group_size)
        + "_b_"  + std::to_string(bits)
        + "_nv_" + std::to_string(vecs_per_tg)
        + "_kl_" + std::to_string(k_lanes)
        + (batched ? "_batch_1" : "_batch_0");
}

// ---------------------------------------------------------------------------
// Group size derivation
// ---------------------------------------------------------------------------

// MLX packs quantized weights as uint32 (32/bits values per element).
// Exception: Bonsai 1-bit uses uint8 packing (8 values per byte).
int derive_group_size(const array& w, const array& scales, int bits) {
    // Bonsai 1-bit: 32 values per uint32 (vs stock MLX's 8 per uint8).
    int64_t pack = (bits == 1) ? 32 : (32 / bits);
    int64_t K = static_cast<int64_t>(w.shape(-1)) * pack;
    int64_t n_groups = scales.shape(-1);
    if (n_groups <= 0) return 64;
    return static_cast<int>(K / n_groups);
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

    auto kernel = get_bonsai_kernel(d, kname);
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

    auto kernel = get_bonsai_kernel(d, kname);
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

// ---------------------------------------------------------------------------
// qmv_fast dispatch (called from eval_gpu)
// ---------------------------------------------------------------------------

void dispatch_qmv_fast(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int M, int N, int K,
    int group_size, int bits,
    metal::Device& d,
    const Stream& s,
    bool symmetric = false) {

    // out shape is x.shape[:-1] + [N], so out.size() == M * N always and
    // the batched kernel variants are never dispatched (not instantiated in
    // the metallib).  Guard the invariant so a future shape change fails
    // loudly here instead of on a missing kernel lookup.
    int B = static_cast<int>(out.size()) / M / N;
    if (B != 1) {
        throw std::runtime_error(
            "[bonsai] batched qmv dispatch is not supported (B=" +
            std::to_string(B) + ")");
    }
    bool batched = false;
    bool fast_aligned = (N % 8 == 0) && (K % 512 == 0);
    // Symmetric fallback: only symmetric fast kernel exists; fall through to
    // affine_qmv (non-fast) for unaligned shapes when symmetric is true, or
    // use the affine path entirely when the shape fails the fast_aligned gate.
    std::string kname = (fast_aligned
        ? qmv_fast_kname(type_str(x.dtype()), group_size, bits, batched, symmetric)
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

    int bn = 16, bk = 32;
    MTL::Size group_dims(bk, 4, 1);
    MTL::Size grid_dims(M, (N + bn - 1) / bn, B);
    enc.dispatch_threadgroups(grid_dims, group_dims);
}

// ---------------------------------------------------------------------------
// qmv_wide dispatch (called from eval_gpu)
// ---------------------------------------------------------------------------

void dispatch_qmv_wide(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int M, int N, int K,
    int group_size, int bits,
    metal::Device& d,
    const Stream& s,
    bool symmetric = false) {

    // out shape is x.shape[:-1] + [N], so out.size() == M * N always and
    // the batched kernel variants are never dispatched (not instantiated in
    // the metallib).  Guard the invariant so a future shape change fails
    // loudly here instead of on a missing kernel lookup.
    int B = static_cast<int>(out.size()) / M / N;
    if (B != 1) {
        throw std::runtime_error(
            "[bonsai] batched qmv dispatch is not supported (B=" +
            std::to_string(B) + ")");
    }
    bool batched = false;

    // Tile size: ceil(M/ceil(M/5)), capped at 5
    int n_tiles = (M + 4) / 5;
    int vecs_per_tg = (M + n_tiles - 1) / n_tiles;
    // affine mode uses k_lanes=8 (more rows/simdgroup)
    int k_lanes = 8;
    int num_simdgroups = 4;
    int rows_per_tg = (32 / k_lanes) * num_simdgroups;

    std::string kname = qmv_wide_kname(
        type_str(x.dtype()), group_size, bits, vecs_per_tg, k_lanes, batched, symmetric);

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

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

// BonsaiQmvPrimitive: wraps qmv_fast and qmv_wide dispatch.
//   inputs[0] = x  (row-contiguous activations)
//   inputs[1] = w  (packed quantized weights)
//   inputs[2] = scales
//   inputs[3] = biases
class BonsaiQmvPrimitive : public Primitive {
 public:
    BonsaiQmvPrimitive(Stream s, int bits, bool wide, bool symmetric = false)
        : Primitive(s), bits_(bits), wide_(wide), symmetric_(symmetric) {}

 private:
    int bits_;
    bool wide_;
    bool symmetric_;

    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error("BonsaiQmvPrimitive has no CPU path.");
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
        const auto& biases = inputs[3];

        // Bonsai 1-bit: 32 values per uint32 (vs stock MLX's 8 values per uint8).
        int64_t pack = (bits_ == 1) ? 32 : (32 / bits_);
        int64_t K = static_cast<int64_t>(w.shape(-1)) * pack;
        int N = static_cast<int>(w.shape(-2));
        int M = static_cast<int>(x.size()) / static_cast<int>(K);
        int group_size = derive_group_size(w, scales, bits_);

        if (wide_) {
            dispatch_qmv_wide(x, w, scales, biases, out,
                              M, N, static_cast<int>(K),
                              group_size, bits_, d, s, symmetric_);
        } else {
            dispatch_qmv_fast(x, w, scales, biases, out,
                              M, N, static_cast<int>(K),
                              group_size, bits_, d, s, symmetric_);
        }
    }

    DEFINE_NAME(BonsaiQmvPrimitive)
};

// BonsaiSpecDecodePrimitive: wraps spec_decode_verify kernel.
//   inputs[0] = draft  [B, K] int32
//   inputs[1] = target [B, K+1] int32 (argmax token ids; caller argmaxes logits)
//   outputs[0] = n_accepted [B] int32
//   outputs[1] = committed  [B, K+1] int32
class BonsaiSpecDecodePrimitive : public Primitive {
 public:
    explicit BonsaiSpecDecodePrimitive(Stream s) : Primitive(s) {}

    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error("BonsaiSpecDecodePrimitive has no CPU path.");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& s = stream();
        auto& d = metal::device(s.device);

        auto& n_accepted = outputs[0];
        auto& committed  = outputs[1];
        n_accepted.set_data(mlx::core::allocator::malloc(n_accepted.nbytes()));
        committed.set_data(mlx::core::allocator::malloc(committed.nbytes()));

        const auto& draft  = inputs[0];
        const auto& target = inputs[1];

        int B = draft.shape(0);
        int K = draft.shape(1);

        auto kernel = get_bonsai_kernel(d, "spec_decode_verify");
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);

        enc.set_input_array(draft,       0);
        enc.set_input_array(target,      1);
        enc.set_output_array(n_accepted, 2);
        enc.set_output_array(committed,  3);
        enc.set_bytes(K, 4);
        enc.set_bytes(B, 5);

        int tgroup = std::min(B, 256);
        MTL::Size grid_dims(B, 1, 1);
        MTL::Size group_dims(tgroup, 1, 1);
        enc.dispatch_threads(grid_dims, group_dims);
    }

    DEFINE_NAME(BonsaiSpecDecodePrimitive)
};

// BonsaiT5QmvPrimitive: t5 base-3 ternary decode (Identity I-D).
//   inputs[0] = x      (row-contiguous activations)
//   inputs[1] = w      (uint8 t5 weight bytes, (N, n_groups*bytes_per_group))
//   inputs[2] = scales ((N, n_groups) scale per group; no biases)
class BonsaiT5QmvPrimitive : public Primitive {
 public:
    explicit BonsaiT5QmvPrimitive(Stream s, bool wide)
        : Primitive(s), wide_(wide) {}

 private:
    bool wide_;

    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error("BonsaiT5QmvPrimitive has no CPU path.");
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

    DEFINE_NAME(BonsaiT5QmvPrimitive)
};

// ---------------------------------------------------------------------------
// BonsaiT5QmmPrimitive: t5 MMA GEMM for prefill (Identity I-M).
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
    auto kernel = get_bonsai_kernel(d, kname);
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

class BonsaiT5QmmPrimitive : public Primitive {
 public:
    explicit BonsaiT5QmmPrimitive(Stream s) : Primitive(s) {}

 private:
    void eval_cpu(
        const std::vector<array>& /* inputs */,
        std::vector<array>& /* outputs */) override {
        throw std::runtime_error("BonsaiT5QmmPrimitive has no CPU path.");
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

    DEFINE_NAME(BonsaiT5QmmPrimitive)
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

array bonsai_q1_affine_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto x_c = ensure_row_contiguous(x, s);
    auto sc  = ensure_dtype(scales, x_c.dtype(), s);
    auto bi  = ensure_dtype(biases, x_c.dtype(), s);
    int N = static_cast<int>(w.shape(-2));
    auto out_shape = x_c.shape();
    out_shape.back() = N;
    return array(out_shape, x_c.dtype(),
        std::make_shared<BonsaiQmvPrimitive>(s, 1, /*wide=*/false),
        {x_c, w, sc, bi});
}

array bonsai_q2_affine_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto x_c = ensure_row_contiguous(x, s);
    auto sc  = ensure_dtype(scales, x_c.dtype(), s);
    auto bi  = ensure_dtype(biases, x_c.dtype(), s);
    int N = static_cast<int>(w.shape(-2));
    auto out_shape = x_c.shape();
    out_shape.back() = N;
    return array(out_shape, x_c.dtype(),
        std::make_shared<BonsaiQmvPrimitive>(s, 2, /*wide=*/false),
        {x_c, w, sc, bi});
}

array bonsai_q1_affine_qmv_wide(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto x_c = ensure_row_contiguous(x, s);
    auto sc  = ensure_dtype(scales, x_c.dtype(), s);
    auto bi  = ensure_dtype(biases, x_c.dtype(), s);
    int N = static_cast<int>(w.shape(-2));
    auto out_shape = x_c.shape();
    out_shape.back() = N;
    return array(out_shape, x_c.dtype(),
        std::make_shared<BonsaiQmvPrimitive>(s, 1, /*wide=*/true),
        {x_c, w, sc, bi});
}

array bonsai_q2_affine_qmv_wide(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto x_c = ensure_row_contiguous(x, s);
    auto sc  = ensure_dtype(scales, x_c.dtype(), s);
    auto bi  = ensure_dtype(biases, x_c.dtype(), s);
    int N = static_cast<int>(w.shape(-2));
    auto out_shape = x_c.shape();
    out_shape.back() = N;
    return array(out_shape, x_c.dtype(),
        std::make_shared<BonsaiQmvPrimitive>(s, 2, /*wide=*/true),
        {x_c, w, sc, bi});
}

array bonsai_q1_affine_qmv_sym(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto x_c = ensure_row_contiguous(x, s);
    auto sc  = ensure_dtype(scales, x_c.dtype(), s);
    auto bi  = ensure_dtype(biases, x_c.dtype(), s);
    int N = static_cast<int>(w.shape(-2));
    auto out_shape = x_c.shape();
    out_shape.back() = N;
    return array(out_shape, x_c.dtype(),
        std::make_shared<BonsaiQmvPrimitive>(s, 1, /*wide=*/false, /*symmetric=*/true),
        {x_c, w, sc, bi});
}

array bonsai_q2_affine_qmv_sym(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto x_c = ensure_row_contiguous(x, s);
    auto sc  = ensure_dtype(scales, x_c.dtype(), s);
    auto bi  = ensure_dtype(biases, x_c.dtype(), s);
    int N = static_cast<int>(w.shape(-2));
    auto out_shape = x_c.shape();
    out_shape.back() = N;
    return array(out_shape, x_c.dtype(),
        std::make_shared<BonsaiQmvPrimitive>(s, 2, /*wide=*/false, /*symmetric=*/true),
        {x_c, w, sc, bi});
}

array bonsai_q1_affine_qmv_wide_sym(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto x_c = ensure_row_contiguous(x, s);
    auto sc  = ensure_dtype(scales, x_c.dtype(), s);
    auto bi  = ensure_dtype(biases, x_c.dtype(), s);
    int N = static_cast<int>(w.shape(-2));
    auto out_shape = x_c.shape();
    out_shape.back() = N;
    return array(out_shape, x_c.dtype(),
        std::make_shared<BonsaiQmvPrimitive>(s, 1, /*wide=*/true, /*symmetric=*/true),
        {x_c, w, sc, bi});
}

array bonsai_q2_affine_qmv_wide_sym(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    auto x_c = ensure_row_contiguous(x, s);
    auto sc  = ensure_dtype(scales, x_c.dtype(), s);
    auto bi  = ensure_dtype(biases, x_c.dtype(), s);
    int N = static_cast<int>(w.shape(-2));
    auto out_shape = x_c.shape();
    out_shape.back() = N;
    return array(out_shape, x_c.dtype(),
        std::make_shared<BonsaiQmvPrimitive>(s, 2, /*wide=*/true, /*symmetric=*/true),
        {x_c, w, sc, bi});
}

array bonsai_t5_qmv(
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
        std::make_shared<BonsaiT5QmvPrimitive>(s, /*wide=*/false),
        {x_c, w, sc});
}

array bonsai_t5_qmv_wide(
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
        std::make_shared<BonsaiT5QmvPrimitive>(s, /*wide=*/true),
        {x_c, w, sc});
}

array bonsai_t5_qmm(
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
        std::make_shared<BonsaiT5QmmPrimitive>(s),
        {x_c, w, sc});
}

std::pair<array, array> bonsai_spec_decode_verify(
    const array& draft,
    const array& target,
    StreamOrDevice s_) {
    auto s = to_stream(s_);
    int B = draft.shape(0);
    int K = draft.shape(1);

    if (draft.dtype() != mlx::core::int32 || target.dtype() != mlx::core::int32) {
        throw std::invalid_argument(
            "[bonsai_spec_decode_verify] draft and target must be int32 token ids "
            "(argmax target logits before calling).");
    }
    if (target.ndim() != 2 || target.shape(0) != B || target.shape(1) != K + 1) {
        throw std::invalid_argument(
            "[bonsai_spec_decode_verify] target must have shape [B, K+1].");
    }

    // Sibling outputs must be created through make_arrays so eval_gpu
    // receives both in one outputs vector.
    auto primitive = std::make_shared<BonsaiSpecDecodePrimitive>(s);
    auto outs = array::make_arrays(
        {{B}, {B, K + 1}},
        {mlx::core::int32, mlx::core::int32},
        primitive,
        {draft, target});
    return {outs[0], outs[1]};
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
