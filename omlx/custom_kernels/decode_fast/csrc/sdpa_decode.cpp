// SPDX-License-Identifier: Apache-2.0
// Decode-mode SDPA (query length <= 8) using the omlx_sdpa_decode kernels:
// a port of ml-explore/mlx#4294 (closed unmerged upstream) — tiled online
// softmax, vectorized KV loads, context-scaled 2-pass split on 'd'-class
// GPUs, fp32 partials. Host dispatch mirrors mlx's
// scaled_dot_product_attention.cpp vector paths, with kernel names and the
// metallib swapped for the omlx ones.
//
// Layout contract matches the other decode_fast ops: realized inputs must
// satisfy the stride predicates below (KV caches and decode-time queries in
// omlx always do). A misaligned realized input is a hard error, since
// neither a nested eval nor an immediate contiguous copy is legal inside
// eval_gpu; use the mx.fast fallback for exotic layouts.

#include "sdpa_decode.h"

#include <dlfcn.h>
#include <algorithm>
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

std::string sdpa_type_name(Dtype dtype) {
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
  msg << "Unsupported sdpa_decode dtype: " << dtype << ".";
  throw std::invalid_argument(msg.str());
}

std::string current_binary_dir_sdpa() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir_sdpa), &info)) {
      throw std::runtime_error("Unable to get omlx_decode_fast binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

// Layout predicates mirrored from ScaledDotProductAttention::eval_gpu.
bool q_layout_ok(const array& arr) {
  if (arr.flags().row_contiguous) {
    return true;
  }
  auto& strides = arr.strides();
  auto& shape = arr.shape();
  if (shape[0] == 1 || shape[1] == 1) {
    auto bidx = shape[0] == 1 ? 1 : 0;
    return (strides[3] == 1) && (strides[2] == shape[3] * shape[bidx]) &&
        (strides[bidx] == shape[3]);
  }
  return false;
}

bool kv_layout_ok(const array& arr) {
  auto& strides = arr.strides();
  auto& shape = arr.shape();
  if (strides.back() != 1) {
    return false;
  }
  auto per_thread = shape[3] / 32;
  if ((shape[2] > 1 && strides[2] % per_thread != 0) ||
      (shape[1] > 1 && strides[1] % per_thread != 0)) {
    return false;
  }
  if (shape[0] == 1 || shape[1] == 1) {
    return true;
  }
  return (strides[0] == strides[1] * shape[1]);
}

// Mirrors mlx::core::metal::check_kernel_threadgroup_size (mlx backend/metal
// utils.h), which upstream calls at every SDPA dispatch site: Metal may drop
// a dispatch whose threadgroup exceeds the compiled pipeline residency limit
// without any error, leaving the output buffer uninitialized (#3660).
void check_kernel_threadgroup_size(
    const MTL::ComputePipelineState* kernel,
    const MTL::Size& group_dims,
    const std::string& name) {
  auto max_size = kernel->maxTotalThreadsPerThreadgroup();
  auto requested_size =
      group_dims.width * group_dims.height * group_dims.depth;
  if (max_size < requested_size) {
    std::ostringstream msg;
    msg << "Maximum threads per threadgroup is " << max_size
        << " but requested " << requested_size << " for kernel " << name
        << ".";
    throw std::runtime_error(msg.str());
  }
}

// Routing decision for vector decode SDPA: which pass to use and how many
// blocks the two-pass split selects. The eval_gpu router, the two-pass
// encoder, and sdpa_decode_supported() share it so the pipeline probed for
// residency is always the pipeline that gets dispatched.
struct SdpaDecodePlan {
  bool use_2pass;
  int blocks;
};

SdpaDecodePlan sdpa_decode_plan(int gqa_factor, int qL, int N, char devc) {
  const bool gqa = gqa_factor > 1;
  SdpaDecodePlan plan;
  if (devc == 'd') {
    plan.use_2pass = gqa ? (N >= 1024) : (N >= 32768);
  } else {
    plan.use_2pass =
        (devc == 's' && N >= 1024) || (gqa && N >= 4096);
  }

  const int n_simds = gqa_factor * qL;
  if (devc == 's') {
    plan.blocks = 64;
    if (N > 1024 && n_simds > 4) {
      if (N <= 8192) {
        plan.blocks = 128;
      } else if (N <= 32768) {
        plan.blocks = 256;
      } else if (N <= 65536) {
        plan.blocks = 512;
      } else {
        plan.blocks = 1024;
      }
    }
  } else if (devc == 'd') {
    // Split the KV sequence so that each threadgroup processes a contiguous
    // chunk of ~256 keys, while keeping at least 32-64 blocks for parallelism
    // and capping at 256 to bound the partials traffic. Tuned on M3 Ultra.
    int b = (((N + 255) / 256 + 31) / 32) * 32;
    plan.blocks = std::min(256, std::max(N >= 4096 ? 64 : 32, b));
  } else {
    plan.blocks = n_simds >= 4 ? 64 : 32;
  }
  return plan;
}

void sdpa_decode_1pass(
    const Stream& s,
    metal::Device& d,
    MTL::Library* lib,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      "omlx_sdpa_decode_",
      sdpa_type_name(q.dtype()),
      "_",
      q.shape(-1),
      "_",
      v.shape(-1));

  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];

  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(q.shape(0) * q.shape(1), q.shape(2), 1);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks = sinks.has_value();
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks, MTL::DataType::DataTypeBool, 25},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks ? "_sinks" : "_nosinks";

  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = d.get_kernel(kname, lib, hash_name, func_consts);
  check_kernel_threadgroup_size(kernel, group_dims, hash_name);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(gqa_factor, 4);
  compute_encoder.set_bytes(N, 5);
  compute_encoder.set_bytes(k_head_stride, 6);
  compute_encoder.set_bytes(k_seq_stride, 7);
  compute_encoder.set_bytes(v_head_stride, 8);
  compute_encoder.set_bytes(v_seq_stride, 9);
  compute_encoder.set_bytes(scale, 10);
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 11 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 13);
    compute_encoder.set_bytes(q_seq_stride, 14);
    compute_encoder.set_bytes(head_stride, 15);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 16);
    compute_encoder.set_bytes(q.shape(1), 17);
  }

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_decode_2pass(
    const Stream& s,
    metal::Device& d,
    MTL::Library* lib,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      "omlx_sdpa_decode_2pass_1_",
      sdpa_type_name(q.dtype()),
      "_",
      q.shape(-1),
      "_",
      v.shape(-1));

  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  int blocks = sdpa_decode_plan(
      gqa_factor, q.shape(2), N, d.get_architecture().back()).blocks;

  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];
  MTL::Size group_dims(32, gqa_factor, q.shape(2));
  MTL::Size grid_dims(k.shape(1), q.shape(0), blocks);

  // Partials in float32 for accuracy (PR #4294); tiny vs the KV read.
  Shape intermediate_shape;
  intermediate_shape.reserve(out.ndim() + 1);
  intermediate_shape.insert(
      intermediate_shape.end(), out.shape().begin(), out.shape().end() - 1);
  intermediate_shape.push_back(blocks);
  intermediate_shape.push_back(out.shape().back());
  array intermediate(intermediate_shape, float32, nullptr, {});
  intermediate_shape.pop_back();
  array sums(intermediate_shape, float32, nullptr, {});
  array maxs(std::move(intermediate_shape), float32, nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  maxs.set_data(allocator::malloc(maxs.nbytes()));
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.add_temporary(intermediate);
  compute_encoder.add_temporary(sums);
  compute_encoder.add_temporary(maxs);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks = sinks.has_value();
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks, MTL::DataType::DataTypeBool, 25},
      {&blocks, MTL::DataType::DataTypeInt, 26},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks ? "_sinks_" : "_nosinks_";
  hash_name += std::to_string(blocks);

  auto kernel = d.get_kernel(kname, lib, hash_name, func_consts);
  check_kernel_threadgroup_size(kernel, group_dims, hash_name);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(intermediate, 3);
  compute_encoder.set_output_array(sums, 4);
  compute_encoder.set_output_array(maxs, 5);
  compute_encoder.set_bytes(N, 7);
  compute_encoder.set_bytes(k_head_stride, 8);
  compute_encoder.set_bytes(k_seq_stride, 9);
  compute_encoder.set_bytes(v_head_stride, 10);
  compute_encoder.set_bytes(v_seq_stride, 11);
  compute_encoder.set_bytes(scale, 12);
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 13 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 15);
    compute_encoder.set_bytes(q_seq_stride, 16);
    compute_encoder.set_bytes(head_stride, 17);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 18);
  }

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Final reduction pass.
  kname.clear();
  concatenate(
      kname,
      "omlx_sdpa_decode_2pass_2_",
      sdpa_type_name(q.dtype()),
      "_",
      v.shape(-1));

  kernel = d.get_kernel(kname, lib);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(intermediate, 0);
  compute_encoder.set_input_array(sums, 1);
  compute_encoder.set_input_array(maxs, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(blocks, 4);

  group_dims = MTL::Size(1024, 1, 1);
  grid_dims = MTL::Size(q.shape(0) * q.shape(1), q.shape(2), 1);
  check_kernel_threadgroup_size(kernel, group_dims, kname);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

class SdpaDecodePrimitive : public Primitive {
 public:
  SdpaDecodePrimitive(Stream stream, float scale, bool do_causal)
      : Primitive(stream), scale_(scale), do_causal_(do_causal) {}

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("SdpaDecodePrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);

    auto& q = inputs[0];
    auto& k = inputs[1];
    auto& v = inputs[2];
    auto& o = outputs[0];

    std::optional<array> sinks = std::nullopt;
    std::optional<array> mask = std::nullopt;
    if (has_sinks_) {
      sinks = inputs[3];
      if (inputs.size() > 4) {
        mask = inputs[4];
      }
    } else if (inputs.size() > 3) {
      mask = inputs[3];
    }

    if (!q_layout_ok(q) || !kv_layout_ok(k) || !kv_layout_ok(v) ||
        (mask && !(*mask).flags().row_contiguous &&
         !(q.shape(0) == 1 || q.shape(1) == 1 ||
           (*mask).strides(0) == (*mask).strides(1) * (*mask).shape(1))) ||
        (sinks && (*sinks).strides(-1) != 1)) {
      throw std::runtime_error(
          "[omlx_decode_fast.sdpa_decode] realized input layout is not "
          "compatible with the vectorized decode kernels; use "
          "mx.fast.scaled_dot_product_attention for this layout.");
    }

    o.set_data(allocator::malloc(o.nbytes()));

    bool do_causal = do_causal_ && q.shape(2) > 1;
    auto plan = sdpa_decode_plan(
        q.shape(1) / k.shape(1), q.shape(2), k.shape(2),
        d.get_architecture().back());

    auto lib = d.get_library(
        "omlx_decode_fast_kernels", current_binary_dir_sdpa());
    if (plan.use_2pass) {
      sdpa_decode_2pass(
          s, d, lib, q, k, v, o, scale_, do_causal, mask, sinks);
    } else {
      sdpa_decode_1pass(s, d, lib, q, k, v, o, scale_, do_causal, mask, sinks);
    }
  }

  DEFINE_NAME(OMLXSdpaDecode)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const SdpaDecodePrimitive&>(other);
    return scale_ == rhs.scale_ && do_causal_ == rhs.do_causal_ &&
        has_sinks_ == rhs.has_sinks_;
  }

  bool has_sinks_ = false;

 private:
  float scale_;
  bool do_causal_;
};

} // namespace

bool sdpa_decode_supported(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    mx::StreamOrDevice s_) {
  auto s = to_stream(s_);
  if (s.device == Device::cpu) {
    return false;
  }
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    return false;
  }
  auto t = q.dtype();
  if (t != float32 && t != float16 && t != bfloat16) {
    return false;
  }
  if (k.dtype() != t || v.dtype() != t) {
    return false;
  }
  const int qk_dim = q.shape(-1);
  const int v_dim = v.shape(-1);
  const bool head_dim_ok =
      (qk_dim == v_dim &&
       (qk_dim == 64 || qk_dim == 96 || qk_dim == 128 || qk_dim == 256)) ||
      (qk_dim == 192 && v_dim == 128);
  if (!head_dim_ok || k.shape(-1) != qk_dim) {
    return false;
  }
  const int qL = q.shape(2);
  const int kL = k.shape(2);
  if (qL < 1 || qL > 8 || qL > kL) {
    return false;
  }
  if (k.shape(0) != q.shape(0) || v.shape(0) != q.shape(0) ||
      v.shape(1) != k.shape(1) || v.shape(2) != kL) {
    return false;
  }
  const int gqa_factor = q.shape(1) / k.shape(1);
  if (q.shape(1) % k.shape(1) != 0 || qL * gqa_factor > 32) {
    return false;
  }
  // Residency gate (jundot/omlx#3660): Metal may silently drop a dispatch
  // whose threadgroup exceeds the compiled pipeline residency limit, leaving
  // the output unwritten (observed on heavy D=256 instantiations; the router
  // picks one pass at gathered QSA widths whenever the architecture gate does
  // not force two). Probe the exact pipeline the plan selects and fail closed
  // so callers fall back to portable SDPA. The probe mirrors the decode
  // contract (causal=False, no mask, no sinks); other FC variants of a
  // kernel share its register footprint.
  auto& d = mlx::core::metal::device(s.device);
  MTL::Library* lib = nullptr;
  try {
    lib = d.get_library("omlx_decode_fast_kernels", current_binary_dir_sdpa());
  } catch (...) {
    return false;
  }
  if (lib == nullptr) {
    return false;
  }

  const auto plan = sdpa_decode_plan(
      gqa_factor, qL, kL, d.get_architecture().back());
  const bool query_transposed = !q.flags().row_contiguous;
  const bool no_causal = false;
  const bool ffalse = false;
  mlx::core::metal::MTLFCList consts = {
      {&ffalse, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&no_causal, MTL::DataType::DataTypeBool, 22},
      {&ffalse, MTL::DataType::DataTypeBool, 23},
      {&ffalse, MTL::DataType::DataTypeBool, 24},
      {&ffalse, MTL::DataType::DataTypeBool, 25},
  };

  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      "omlx_sdpa_decode",
      plan.use_2pass ? "_2pass_1" : "",
      "_",
      sdpa_type_name(t),
      "_",
      qk_dim,
      "_",
      v_dim);
  std::string hash_name = kname;
  hash_name += "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += "_nc_nosinks";

  auto residency_ok = [&](const std::string& name,
                          const std::string& hash,
                          const mlx::core::metal::MTLFCList& fc,
                          size_t threads) -> bool {
    try {
      auto* kernel = d.get_kernel(name, lib, hash, fc);
      return kernel != nullptr &&
          kernel->maxTotalThreadsPerThreadgroup() >= threads;
    } catch (...) {
      return false;
    }
  };

  if (!plan.use_2pass) {
    // group_dims is (1024, 1, 1) for every one-pass instantiation.
    return residency_ok(kname, hash_name, consts, 1024);
  }

  hash_name += "_" + std::to_string(plan.blocks);
  consts.push_back({&plan.blocks, MTL::DataType::DataTypeInt, 26});
  if (!residency_ok(kname, hash_name, consts,
          32 * static_cast<size_t>(gqa_factor) * qL)) {
    return false;
  }

  // Aggregation pass: host dispatch takes the no-function-constants path.
  kname.clear();
  concatenate(
      kname, "omlx_sdpa_decode_2pass_2_", sdpa_type_name(t), "_", v_dim);
  return residency_ok(kname, "", {}, 1024);
}

mx::array sdpa_decode(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    float scale,
    bool causal,
    const std::optional<mx::array>& mask,
    const std::optional<mx::array>& sinks,
    mx::StreamOrDevice s_) {
  if (!sdpa_decode_supported(q, k, v, s_)) {
    throw std::invalid_argument(
        "[omlx_decode_fast.sdpa_decode] unsupported shapes/dtypes for the "
        "decode kernels.");
  }
  auto s = to_stream(s_);
  std::vector<array> inputs = {q, k, v};
  auto primitive = std::make_shared<SdpaDecodePrimitive>(s, scale, causal);
  if (sinks.has_value()) {
    primitive->has_sinks_ = true;
    inputs.push_back(*sinks);
  }
  if (mask.has_value()) {
    inputs.push_back(*mask);
  }
  Shape out_shape{q.shape(0), q.shape(1), q.shape(2), v.shape(-1)};
  return array(
      std::move(out_shape), q.dtype(), std::move(primitive), std::move(inputs));
}

} // namespace omlx::decode_fast_kernels
