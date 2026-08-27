// SPDX-License-Identifier: Apache-2.0
// Fused sparse decode attention (windowed local KV + selected pooled KV +
// attention sinks) for DeepSeek-V4-style single-KV-head MLA layers. Replaces
// the composed rowwise-GEMM + logsumexp glue of
// omlx.patches.deepseek_v4.deepseek_v4_model._dspark_sparse_exact_attention
// with one Metal dispatch.
//
// Layout contract matches the other decode_fast ops: realized inputs must be
// row-contiguous (true for omlx caches and decode-time queries). A
// misaligned realized input is a hard error, since neither a nested eval nor
// an immediate contiguous copy is legal inside eval_gpu; the wrapper
// pre-filters and callers keep the composed fallback for exotic layouts.

#include "sparse_attn_decode.h"

#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace omlx::decode_fast_kernels {

namespace {

using namespace mlx::core;

constexpr int kSparseAttnMaxS = 1024;  // mirrors OMLX_SPARSE_MAX_S

std::string sparse_type_name(Dtype dtype) {
  if (dtype == float32) {
    return "float";
  }
  if (dtype == float16) {
    return "float16_t";
  }
  if (dtype == bfloat16) {
    return "bfloat16_t";
  }
  std::ostringstream msg;
  msg << "Unsupported sparse_attn_decode dtype: " << dtype << ".";
  throw std::invalid_argument(msg.str());
}

std::string current_binary_dir_sparse() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir_sparse), &info)) {
      throw std::runtime_error("Unable to get omlx_decode_fast binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

class SparseAttnDecodePrimitive : public Primitive {
 public:
  explicit SparseAttnDecodePrimitive(Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("SparseAttnDecodePrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);

    auto& q = inputs[0];         // [B, H, 1, D]
    auto& local_kv = inputs[1];  // [B, 1, W, D]
    auto& pooled = inputs[2];    // [B, 1, P, D]
    auto& sinks = inputs[3];     // [H] float32
    auto& o = outputs[0];        // [B, H, 1, D]

    if (q.strides(3) != 1 || local_kv.strides(3) != 1 ||
        pooled.strides(3) != 1 || sinks.strides(-1) != 1 ||
        q.strides(0) % 8 != 0 || q.strides(1) % 8 != 0 ||
        local_kv.strides(0) % 8 != 0 || local_kv.strides(2) % 8 != 0 ||
        pooled.strides(0) % 8 != 0 || pooled.strides(2) % 8 != 0) {
      throw std::runtime_error(
          "[omlx_decode_fast.sparse_attn_decode] realized input layout is "
          "not compatible with the vectorized kernel (unit last stride and "
          "8-element-aligned batch/row strides required); use the composed "
          "sparse-attention path for this layout.");
    }

    o.set_data(allocator::malloc(o.nbytes()));

    const int H = q.shape(1);
    const int B = q.shape(0);
    const int W = local_kv.shape(2);
    const int P = pooled.shape(2);
    constexpr int kHeadsPerGroup = 4;  // mirrors the HPG template arg

    std::string kname;
    kname.reserve(64);
    concatenate(kname, "omlx_sparse_attn_decode_", sparse_type_name(q.dtype()),
                "_", q.shape(-1), "_", kHeadsPerGroup);

    auto lib =
        d.get_library("omlx_decode_fast_kernels", current_binary_dir_sparse());
    auto& compute_encoder = metal::get_command_encoder(s);
    auto kernel = d.get_kernel(kname, lib);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(local_kv, 1);
    compute_encoder.set_input_array(pooled, 2);
    compute_encoder.set_input_array(sinks, 3);
    compute_encoder.set_output_array(o, 4);
    compute_encoder.set_bytes(W, 5);
    compute_encoder.set_bytes(P, 6);
    compute_encoder.set_bytes(H, 7);
    const size_t q_b_stride = q.strides()[0];
    const size_t q_h_stride = q.strides()[1];
    const size_t kv_b_stride = local_kv.strides()[0];
    const size_t kv_row_stride = local_kv.strides()[2];
    const size_t pooled_b_stride = pooled.strides()[0];
    const size_t pooled_row_stride = pooled.strides()[2];
    compute_encoder.set_bytes(q_b_stride, 8);
    compute_encoder.set_bytes(q_h_stride, 9);
    compute_encoder.set_bytes(kv_b_stride, 10);
    compute_encoder.set_bytes(kv_row_stride, 11);
    compute_encoder.set_bytes(pooled_b_stride, 12);
    compute_encoder.set_bytes(pooled_row_stride, 13);

    compute_encoder.dispatch_threadgroups(
        MTL::Size(H / kHeadsPerGroup, B, 1), MTL::Size(256, 1, 1));
  }

  DEFINE_NAME(OMLXSparseAttnDecode)

  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const SparseAttnDecodePrimitive*>(&other) != nullptr;
  }
};

} // namespace

bool sparse_attn_decode_supported(
    const mx::array& q,
    const mx::array& local_kv,
    const mx::array& pooled,
    const mx::array& sinks,
    mx::StreamOrDevice s_) {
  auto s = to_stream(s_);
  if (s.device == Device::cpu) {
    return false;
  }
  if (q.ndim() != 4 || local_kv.ndim() != 4 || pooled.ndim() != 4) {
    return false;
  }
  auto t = q.dtype();
  if (t != float32 && t != float16 && t != bfloat16) {
    return false;
  }
  if (local_kv.dtype() != t || pooled.dtype() != t ||
      sinks.dtype() != float32) {
    return false;
  }
  const int D = q.shape(-1);
  if (D != 512) {
    return false;
  }
  const int B = q.shape(0);
  if (B < 1 || B > 8 || q.shape(2) != 1) {
    return false;
  }
  if (local_kv.shape(0) != B || pooled.shape(0) != B ||
      local_kv.shape(1) != 1 || pooled.shape(1) != 1 ||
      local_kv.shape(-1) != D || pooled.shape(-1) != D) {
    return false;
  }
  const int W = local_kv.shape(2);
  const int P = pooled.shape(2);
  if (W + P < 1 || W + P > kSparseAttnMaxS) {
    return false;
  }
  if (sinks.ndim() != 1 || sinks.shape(0) != q.shape(1)) {
    return false;
  }
  if (q.shape(1) % 4 != 0) {
    return false;  // kernel processes 4 heads per threadgroup
  }
  return true;
}

mx::array sparse_attn_decode(
    const mx::array& q,
    const mx::array& local_kv,
    const mx::array& pooled,
    const mx::array& sinks,
    mx::StreamOrDevice s_) {
  if (!sparse_attn_decode_supported(q, local_kv, pooled, sinks, s_)) {
    throw std::invalid_argument(
        "[omlx_decode_fast.sparse_attn_decode] unsupported shapes/dtypes for "
        "the fused sparse decode attention kernel.");
  }
  auto s = to_stream(s_);
  std::vector<array> inputs = {q, local_kv, pooled, sinks};
  auto primitive = std::make_shared<SparseAttnDecodePrimitive>(s);
  Shape out_shape{q.shape(0), q.shape(1), q.shape(2), q.shape(3)};
  return array(
      std::move(out_shape), q.dtype(), std::move(primitive), std::move(inputs));
}

} // namespace omlx::decode_fast_kernels
