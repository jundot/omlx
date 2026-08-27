// SPDX-License-Identifier: Apache-2.0
// Fused RoPE(K) + key-cache append (single-token decode) — host side of the
// ml-explore/mlx#4297 port. Single-output primitive so the K-cache buffer
// can be donated and updated in place; the V cache is appended with a plain
// slice_update at the call site (which donates in place as well).

#include "rope_append.h"

#include <dlfcn.h>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace omlx::decode_fast_kernels {

namespace {

using namespace mlx::core;

std::string current_binary_dir_rope() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir_rope), &info)) {
      throw std::runtime_error("Unable to get omlx_decode_fast binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

// Local replica of mlx's (unexported) get_block_dims for 2-D grids:
// powers-of-two block sides, at most 1024 threads.
MTL::Size block_dims_2d(uint32_t dim0, uint32_t dim1) {
  auto pow2ceil = [](uint32_t v) {
    uint32_t p = 1;
    while (p < v) {
      p <<= 1;
    }
    return p;
  };
  uint32_t w0 = std::min(pow2ceil(dim0), 32u);
  uint32_t w1 = std::min(pow2ceil(dim1), 1024u / w0);
  return MTL::Size(w0, w1, 1);
}

class RoPEAppendPrimitive : public Primitive {
 public:
  RoPEAppendPrimitive(
      Stream stream,
      int dims,
      bool traditional,
      float base,
      float scale,
      int offset)
      : Primitive(stream),
        dims_(dims),
        traditional_(traditional),
        base_(base),
        scale_(scale),
        offset_(offset) {}

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("RoPEAppendPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& k = inputs[0];
    auto& k_cache_in = inputs[1];
    auto& k_cache = outputs[0];

    auto& s = stream();
    auto& d = metal::device(s.device);

    auto lib =
        d.get_library("omlx_decode_fast_kernels", current_binary_dir_rope());
    auto& compute_encoder = metal::get_command_encoder(s);

    // Share the cache buffer when it is uniquely referenced (in-place
    // append, the hot path); otherwise allocate a fresh buffer and copy the
    // old contents with the flat copy kernel (cache is row-contiguous by
    // the entry gate, so a flat copy is exact).
    if (k_cache_in.is_donatable()) {
      k_cache.copy_shared_buffer(k_cache_in);
    } else {
      k_cache.set_data(
          allocator::malloc(k_cache_in.nbytes()),
          k_cache_in.data_size(),
          k_cache_in.strides(),
          k_cache_in.flags());
      std::string cname;
      concatenate(cname, "omlx_flat_copy_", type_to_name(k_cache_in));
      auto copy_kernel = d.get_kernel(cname, lib);
      int64_t n = k_cache_in.data_size();
      uint32_t threads = std::min<int64_t>(1024, n);
      uint32_t groups = std::min<int64_t>(4096, (n + threads - 1) / threads);
      compute_encoder.set_compute_pipeline_state(copy_kernel);
      compute_encoder.set_input_array(k_cache_in, 0);
      compute_encoder.set_output_array(k_cache, 1);
      compute_encoder.set_bytes(n, 2);
      compute_encoder.dispatch_threads(
          MTL::Size(groups * threads, 1, 1), MTL::Size(threads, 1, 1));
    }

    int ndim = k.ndim();
    int k_dims = k.shape(-1);
    // Rows are densely packed (checked when the primitive is built)
    int64_t rows = k.size() / (k.shape(-2) * k_dims);
    int64_t k_cache_mat_stride = inputs[1].strides()[ndim - 3];

    bool with_freqs = inputs.size() == 3;
    bool large =
        k_cache.data_size() > INT32_MAX || k_cache.size() > INT32_MAX;

    std::string kname;
    concatenate(
        kname,
        "omlx_rope_append_",
        with_freqs ? "freqs_" : "",
        large ? "large_" : "",
        type_to_name(k));
    std::string hash_name;
    concatenate(hash_name, kname, traditional_ ? "_traditional" : "");
    metal::MTLFCList func_consts = {
        {&traditional_, MTL::DataType::DataTypeBool, 2}};

    auto kernel = d.get_kernel(kname, lib, hash_name, func_consts);

    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(k, 0);
    compute_encoder.set_output_array(k_cache, 1);
    compute_encoder.set_bytes(offset_, 2);
    compute_encoder.set_bytes(scale_, 3);
    compute_encoder.set_bytes(dims_, 4);
    compute_encoder.set_bytes(k_dims, 5);
    compute_encoder.set_bytes(k_cache_mat_stride, 6);
    if (with_freqs) {
      auto& freqs = inputs[2];
      compute_encoder.set_input_array(freqs, 7);
      auto freq_stride = freqs.strides()[0];
      compute_encoder.set_bytes(freq_stride, 8);
    } else {
      float base = std::log2(base_);
      compute_encoder.set_bytes(base, 7);
    }

    uint32_t dim0 = k_dims / 2;
    uint32_t dim1 = rows;
    auto group_dims = block_dims_2d(dim0, dim1);
    MTL::Size grid_dims(dim0, dim1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  DEFINE_NAME(OMLXRoPEAppend)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const RoPEAppendPrimitive&>(other);
    return dims_ == rhs.dims_ && base_ == rhs.base_ && scale_ == rhs.scale_ &&
        traditional_ == rhs.traditional_ && offset_ == rhs.offset_;
  }

 private:
  int dims_;
  bool traditional_;
  float base_;
  float scale_;
  int offset_;
};

} // namespace

bool rope_kv_append_supported(
    const mx::array& keys,
    const mx::array& values,
    const mx::array& key_cache,
    const mx::array& value_cache,
    int offset,
    int dims,
    mx::StreamOrDevice s_) {
  auto s = to_stream(s_);
  if (s.device == Device::cpu) {
    return false;
  }
  if (keys.ndim() != 4 || values.ndim() != 4 || key_cache.ndim() != 4 ||
      value_cache.ndim() != 4) {
    return false;
  }
  if (keys.shape(-2) != 1 || values.shape(-2) != 1) {
    return false; // single-token decode only
  }
  auto t = keys.dtype();
  if ((t != float32 && t != float16 && t != bfloat16) ||
      values.dtype() != t || key_cache.dtype() != t ||
      value_cache.dtype() != t) {
    return false;
  }
  for (int i = 0; i < 2; ++i) {
    if (keys.shape(i) != key_cache.shape(i) ||
        values.shape(i) != value_cache.shape(i) ||
        keys.shape(i) != values.shape(i)) {
      return false;
    }
  }
  if (key_cache.shape(-1) != keys.shape(-1) ||
      value_cache.shape(-1) != values.shape(-1)) {
    return false;
  }
  if (offset < 0 || offset + 1 > key_cache.shape(-2) ||
      offset + 1 > value_cache.shape(-2)) {
    return false;
  }
  if (dims <= 0 || dims % 2 != 0 || dims > keys.shape(-1)) {
    return false;
  }
  // NOTE: as with the other decode_fast ops, row-contiguity of lazy arrays
  // is not reliably knowable here; the realized caches in omlx are dense by
  // construction, and the primitive assumes dense rows.
  if (!key_cache.flags().row_contiguous) {
    return false;
  }
  return true;
}

std::vector<mx::array> rope_kv_append(
    const mx::array& keys,
    const mx::array& values,
    const mx::array& key_cache,
    const mx::array& value_cache,
    int offset,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const std::optional<mx::array>& freqs,
    mx::StreamOrDevice s_) {
  if (freqs.has_value() && base.has_value()) {
    throw std::invalid_argument(
        "[omlx_decode_fast.rope_kv_append] Only one of base or freqs can "
        "have a value.");
  }
  if (!freqs.has_value() && !base.has_value()) {
    throw std::invalid_argument(
        "[omlx_decode_fast.rope_kv_append] Neither base nor freqs has a "
        "value.");
  }
  if (!rope_kv_append_supported(
          keys, values, key_cache, value_cache, offset, dims, s_)) {
    throw std::invalid_argument(
        "[omlx_decode_fast.rope_kv_append] unsupported shapes/dtypes for "
        "the fused kernel.");
  }

  auto s = to_stream(s_);
  std::vector<array> inputs{keys, key_cache};
  if (freqs.has_value()) {
    inputs.push_back(astype(*freqs, float32, s));
  }
  auto k_updated = array(
      key_cache.shape(),
      key_cache.dtype(),
      std::make_shared<RoPEAppendPrimitive>(
          s, dims, traditional, base.value_or(1.0), scale, offset),
      std::move(inputs));

  Shape starts(4, 0);
  Shape v_stops(value_cache.shape());
  starts[2] = offset;
  v_stops[2] = offset + 1;
  auto v_updated = slice_update(value_cache, values, starts, v_stops, s);
  return {k_updated, v_updated};
}

} // namespace omlx::decode_fast_kernels
