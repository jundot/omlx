#include "fused_moe.h"

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <string>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get omlx_glm_kernels binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

bool row_contiguous(const array& arr) {
  return arr.flags().row_contiguous && arr.strides(-1) == 1;
}

struct Mxfp4BlocksVariant {
  int bm;
  int bn;
  int bk;
  int wm;
  int wn;
};

Mxfp4BlocksVariant mxfp4_blocks_variant(int variant) {
  switch (variant) {
    case 0:
      return {/* bm = */ 8, /* bn = */ 32, /* bk = */ 32, /* wm = */ 1, /* wn = */ 2};
    case 1:
      return {/* bm = */ 16, /* bn = */ 32, /* bk = */ 32, /* wm = */ 1, /* wn = */ 2};
    case 2:
      return {/* bm = */ 32, /* bn = */ 32, /* bk = */ 32, /* wm = */ 1, /* wn = */ 2};
    case 3:
      return {/* bm = */ 16, /* bn = */ 64, /* bk = */ 32, /* wm = */ 1, /* wn = */ 2};
    case 4:
      return {/* bm = */ 32, /* bn = */ 64, /* bk = */ 32, /* wm = */ 1, /* wn = */ 2};
    default: {
      std::ostringstream msg;
      msg << "Unsupported DeepSeek MXFP4 block-list variant " << variant << ".";
      throw std::invalid_argument(msg.str());
    }
  }
}

int affine_pack_factor(int bits) {
  switch (bits) {
    case 2:
      return 4;
    case 3:
      return 8;
    case 4:
      return 2;
    case 8:
      return 1;
    default:
      return 0;
  }
}

int affine_bytes_per_pack(int bits) {
  switch (bits) {
    case 2:
      return 1;
    case 3:
      return 3;
    case 4:
      return 1;
    case 8:
      return 1;
    default:
      return 0;
  }
}

bool supported_deepseek_affine(int group_size, int bits) {
  return group_size == 64 && (bits == 2 || bits == 3);
}

int affine_packed_row_bytes(int K, int bits) {
  const int pack_factor = affine_pack_factor(bits);
  const int bytes_per_pack = affine_bytes_per_pack(bits);
  if (pack_factor == 0 || bytes_per_pack == 0 || K % pack_factor != 0) {
    return -1;
  }
  return K * bytes_per_pack / pack_factor;
}

std::string glm_type_name(Dtype dtype) {
  if (dtype == float16) {
    return "float16_t";
  }
  if (dtype == bfloat16) {
    return "bfloat16_t";
  }
  std::ostringstream msg;
  msg << "Unsupported GLM fused kernel dtype: " << dtype << ".";
  throw std::invalid_argument(msg.str());
}

class GlmDsaQ8VupFlatPrimitive : public Primitive {
 public:
  explicit GlmDsaQ8VupFlatPrimitive(Stream stream) : Primitive(stream) {}

  static bool unsupported(
      const array& x,
      const array& weight,
      const array& scales,
      const array& biases,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (x.dtype() != float16 && x.dtype() != bfloat16) {
      return true;
    }
    if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
        biases.dtype() != x.dtype()) {
      return true;
    }
    if (x.ndim() != 4 || weight.ndim() != 3 || scales.ndim() != 3 ||
        biases.ndim() != 3) {
      return true;
    }
    if (!row_contiguous(x) || !row_contiguous(weight) ||
        !row_contiguous(scales) || !row_contiguous(biases)) {
      return true;
    }

    constexpr int bits = 8;
    constexpr int group_size = 64;
    constexpr int pack_factor = 32 / bits;
    const int H = x.shape(1);
    const int K = x.shape(3);
    const int N = weight.shape(1);
    if (H != 64 || K != 512 || N != 256) {
      return true;
    }
    if (weight.shape(0) != H || scales.shape(0) != H ||
        biases.shape(0) != H || scales.shape(1) != N ||
        biases.shape(1) != N || weight.shape(2) * pack_factor != K ||
        scales.shape(2) != K / group_size ||
        biases.shape(2) != K / group_size) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("GlmDsaQ8VupFlatPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const auto& biases = inputs[3];

    out.set_data(allocator::malloc(out.nbytes()));

    constexpr int group_size = 64;
    constexpr int bits = 8;
    constexpr int bm = 32;
    constexpr int bn = 32;

    const int B = x.shape(0);
    const int H = x.shape(1);
    const int M = x.shape(2);
    const int K = x.shape(3);
    const int N = weight.shape(1);

    std::string kname;
    concatenate(
        kname,
        "affine_qmm_t_head_flat_",
        glm_type_name(x.dtype()),
        "_gs_",
        group_size,
        "_b_",
        bits,
        "_alN_true");

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(weight, 0);
    compute_encoder.set_input_array(scales, 1);
    compute_encoder.set_input_array(biases, 2);
    compute_encoder.set_input_array(x, 3);
    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_bytes(K, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(M, 7);
    compute_encoder.set_bytes(H, 8);

    MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B * H);
    MTL::Size group_dims(32, 2, 2);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(GlmDsaQ8VupFlatPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }

};

class GlmMoeWeightedSumPrimitive : public Primitive {
 public:
  explicit GlmMoeWeightedSumPrimitive(Stream stream) : Primitive(stream) {}

  static bool unsupported(
      const array& x_sorted,
      const array& inv_order,
      const array& scores,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (x_sorted.dtype() != float16 && x_sorted.dtype() != bfloat16) {
      return true;
    }
    if (scores.dtype() != float32 || inv_order.dtype() != uint32) {
      return true;
    }
    if (x_sorted.ndim() != 3 || x_sorted.shape(-2) != 1 ||
        scores.ndim() < 2 || inv_order.ndim() != 1) {
      return true;
    }
    if (!row_contiguous(x_sorted) || !row_contiguous(inv_order) ||
        !row_contiguous(scores)) {
      return true;
    }
    const int topk = scores.shape(-1);
    if ((topk != 6 && topk != 8) || x_sorted.shape(0) != scores.size() ||
        inv_order.size() != scores.size()) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("GlmMoeWeightedSumPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x_sorted = inputs[0];
    const auto& inv_order = inputs[1];
    const auto& scores = inputs[2];

    out.set_data(allocator::malloc(out.nbytes()));

    const int topk = scores.shape(-1);
    const int tokens = scores.size() / topk;
    const int D = x_sorted.shape(-1);

    const bool use_tiled = true;
    const int tiled_threads = 256;
    const int vec = (D % 4 == 0) ? 4 : 1;

    std::string kname;
    if (use_tiled) {
      concatenate(
          kname,
          "moe_weighted_sum_tiled_",
          glm_type_name(x_sorted.dtype()),
          "_score_float_topk_",
          topk,
          "_t_",
          tiled_threads);
    } else {
      concatenate(
          kname,
          vec == 1 ? "moe_weighted_sum_" : "moe_weighted_sum_vec",
          vec == 1 ? "" : std::to_string(vec),
          vec == 1 ? "" : "_",
          glm_type_name(x_sorted.dtype()),
          "_score_float_topk_",
          topk);
    }

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x_sorted, 0);
    compute_encoder.set_input_array(inv_order, 1);
    compute_encoder.set_input_array(scores, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(tokens, 4);
    compute_encoder.set_bytes(D, 5);

    const int threads = use_tiled ? tiled_threads : 256;
    const int total = vec == 1 ? tokens * D : tokens * ((D + vec - 1) / vec);
    MTL::Size group_dims(threads, 1, 1);
    MTL::Size grid_dims(
        use_tiled ? tokens : (total + threads - 1) / threads, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(GlmMoeWeightedSumPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }

};

class DeepseekMxfp4GatherBlocksPrimitive : public Primitive {
 public:
  explicit DeepseekMxfp4GatherBlocksPrimitive(Stream stream, int variant)
      : Primitive(stream), variant_(variant) {
    (void)mxfp4_blocks_variant(variant_);
  }

  static bool unsupported(
      const array& x,
      const array& weight,
      const array& scales,
      const array& block_meta,
      const array& block_count,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (x.dtype() != float16 && x.dtype() != bfloat16) {
      return true;
    }
    if (weight.dtype() != uint32 || scales.dtype() != uint8 ||
        block_meta.dtype() != int32 || block_count.dtype() != int32) {
      return true;
    }
    if (x.ndim() != 3 || x.shape(1) != 1 || weight.ndim() != 3 ||
        scales.ndim() != 3 || block_meta.ndim() != 2 ||
        block_meta.shape(1) != 3 || block_count.size() != 1) {
      return true;
    }
    if (!row_contiguous(x) || !row_contiguous(weight) ||
        !row_contiguous(scales) || !row_contiguous(block_meta) ||
        !row_contiguous(block_count)) {
      return true;
    }

    constexpr int bits = 4;
    constexpr int group_size = 32;
    constexpr int values_per_uint32 = 32 / bits;
    const int K = x.shape(2);
    const int E = weight.shape(0);
    const int N = weight.shape(1);
    if (x.shape(0) <= 0 || K <= 0 || N <= 0 || E <= 0 ||
        block_meta.shape(0) <= 0) {
      return true;
    }
    if (weight.shape(2) * values_per_uint32 != K || scales.shape(0) != E ||
        scales.shape(1) != N || scales.shape(2) != K / group_size) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "DeepseekMxfp4GatherBlocksPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const auto& block_meta = inputs[3];
    const auto& block_count = inputs[4];

    out.set_data(allocator::malloc(out.nbytes()));

    const auto cfg = mxfp4_blocks_variant(variant_);
    const int max_blocks = block_meta.shape(0);
    const int M = x.shape(0);
    const int K = x.shape(2);
    const int N = weight.shape(1);

    std::string kname;
    concatenate(
        kname,
        "deepseek_mxfp4_gather_blocks_rhs_",
        glm_type_name(x.dtype()),
        "_bm_",
        cfg.bm,
        "_bn_",
        cfg.bn,
        "_bk_",
        cfg.bk,
        "_wm_",
        cfg.wm,
        "_wn_",
        cfg.wn);

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(weight, 1);
    compute_encoder.set_input_array(scales, 2);
    compute_encoder.set_input_array(block_meta, 3);
    compute_encoder.set_input_array(block_count, 4);
    compute_encoder.set_output_array(out, 5);
    compute_encoder.set_bytes(max_blocks, 6);
    compute_encoder.set_bytes(M, 7);
    compute_encoder.set_bytes(N, 8);
    compute_encoder.set_bytes(K, 9);

    MTL::Size grid_dims((N + cfg.bn - 1) / cfg.bn, max_blocks, 1);
    MTL::Size group_dims(cfg.wm * cfg.wn * 32, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(DeepseekMxfp4GatherBlocksPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const DeepseekMxfp4GatherBlocksPrimitive&>(other);
    return variant_ == rhs.variant_;
  }
  auto state() const {
    return std::make_tuple(variant_);
  }

 private:
  int variant_;
};

class DeepseekMxfp4GatherPairBlocksPrimitive : public Primitive {
 public:
  explicit DeepseekMxfp4GatherPairBlocksPrimitive(
      Stream stream,
      int variant,
      bool concat_output = false)
      : Primitive(stream), variant_(variant), concat_output_(concat_output) {
    (void)mxfp4_blocks_variant(variant_);
  }

  static bool unsupported(
      const array& x,
      const array& weight0,
      const array& scales0,
      const array& weight1,
      const array& scales1,
      const array& block_meta,
      const array& block_count,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (x.dtype() != float16 && x.dtype() != bfloat16) {
      return true;
    }
    if (weight0.dtype() != uint32 || scales0.dtype() != uint8 ||
        weight1.dtype() != uint32 || scales1.dtype() != uint8 ||
        block_meta.dtype() != int32 || block_count.dtype() != int32) {
      return true;
    }
    if (x.ndim() != 3 || x.shape(1) != 1 || weight0.ndim() != 3 ||
        scales0.ndim() != 3 || weight1.ndim() != 3 || scales1.ndim() != 3 ||
        block_meta.ndim() != 2 || block_meta.shape(1) != 3 ||
        block_count.size() != 1) {
      return true;
    }
    if (!row_contiguous(x) || !row_contiguous(weight0) ||
        !row_contiguous(scales0) || !row_contiguous(weight1) ||
        !row_contiguous(scales1) || !row_contiguous(block_meta) ||
        !row_contiguous(block_count)) {
      return true;
    }

    constexpr int bits = 4;
    constexpr int group_size = 32;
    constexpr int values_per_uint32 = 32 / bits;
    const int K = x.shape(2);
    const int E = weight0.shape(0);
    const int N = weight0.shape(1);
    if (x.shape(0) <= 0 || K <= 0 || N <= 0 || E <= 0 ||
        block_meta.shape(0) <= 0) {
      return true;
    }
    if (weight1.shape() != weight0.shape() || scales1.shape() != scales0.shape()) {
      return true;
    }
    if (weight0.shape(2) * values_per_uint32 != K ||
        scales0.shape(0) != E || scales0.shape(1) != N ||
        scales0.shape(2) != K / group_size) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "DeepseekMxfp4GatherPairBlocksPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x = inputs[0];
    const auto& weight0 = inputs[1];
    const auto& scales0 = inputs[2];
    const auto& weight1 = inputs[3];
    const auto& scales1 = inputs[4];
    const auto& block_meta = inputs[5];
    const auto& block_count = inputs[6];

    out.set_data(allocator::malloc(out.nbytes()));

    const auto cfg = mxfp4_blocks_variant(variant_);
    const int max_blocks = block_meta.shape(0);
    const int M = x.shape(0);
    const int K = x.shape(2);
    const int N = weight0.shape(1);

    std::string kname;
    concatenate(
        kname,
        concat_output_ ? "deepseek_mxfp4_gather_pair_concat_blocks_rhs_"
                       : "deepseek_mxfp4_gather_pair_blocks_rhs_",
        glm_type_name(x.dtype()),
        "_bm_",
        cfg.bm,
        "_bn_",
        cfg.bn,
        "_bk_",
        cfg.bk,
        "_wm_",
        cfg.wm,
        "_wn_",
        cfg.wn);

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(weight0, 1);
    compute_encoder.set_input_array(scales0, 2);
    compute_encoder.set_input_array(weight1, 3);
    compute_encoder.set_input_array(scales1, 4);
    compute_encoder.set_input_array(block_meta, 5);
    compute_encoder.set_input_array(block_count, 6);
    compute_encoder.set_output_array(out, 7);
    compute_encoder.set_bytes(max_blocks, 8);
    compute_encoder.set_bytes(M, 9);
    compute_encoder.set_bytes(N, 10);
    compute_encoder.set_bytes(K, 11);

    const int grid_n = concat_output_ ? 2 * N : N;
    MTL::Size grid_dims(
        (grid_n + cfg.bn - 1) / cfg.bn,
        max_blocks,
        concat_output_ ? 1 : 2);
    MTL::Size group_dims(cfg.wm * cfg.wn * 32, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(DeepseekMxfp4GatherPairBlocksPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const DeepseekMxfp4GatherPairBlocksPrimitive&>(other);
    return variant_ == rhs.variant_ && concat_output_ == rhs.concat_output_;
  }
  auto state() const {
    return std::make_tuple(variant_, concat_output_);
  }

 private:
  int variant_;
  bool concat_output_;
};

class DeepseekAffineGatherBlocksPrimitive : public Primitive {
 public:
  explicit DeepseekAffineGatherBlocksPrimitive(
      Stream stream,
      int group_size,
      int bits,
      int variant)
      : Primitive(stream),
        group_size_(group_size),
        bits_(bits),
        variant_(variant) {
    (void)mxfp4_blocks_variant(variant_);
    if (!supported_deepseek_affine(group_size_, bits_)) {
      throw std::invalid_argument(
          "Unsupported DeepSeek affine block-list quantization.");
    }
  }

  static bool unsupported(
      const array& x,
      const array& weight,
      const array& scales,
      const array& biases,
      const array& block_meta,
      const array& block_count,
      int group_size,
      int bits,
      Stream s) {
    if (s.device == Device::cpu || !supported_deepseek_affine(group_size, bits)) {
      return true;
    }
    if (x.dtype() != float16 && x.dtype() != bfloat16) {
      return true;
    }
    if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
        biases.dtype() != x.dtype() || block_meta.dtype() != int32 ||
        block_count.dtype() != int32) {
      return true;
    }
    if (x.ndim() != 3 || x.shape(1) != 1 || weight.ndim() != 3 ||
        scales.ndim() != 3 || biases.ndim() != 3 || block_meta.ndim() != 2 ||
        block_meta.shape(1) != 3 || block_count.size() != 1) {
      return true;
    }
    if (!row_contiguous(x) || !row_contiguous(weight) ||
        !row_contiguous(scales) || !row_contiguous(biases) ||
        !row_contiguous(block_meta) || !row_contiguous(block_count)) {
      return true;
    }

    const int K = x.shape(2);
    const int E = weight.shape(0);
    const int N = weight.shape(1);
    const int packed_bytes = affine_packed_row_bytes(K, bits);
    if (x.shape(0) <= 0 || K <= 0 || N <= 0 || E <= 0 ||
        block_meta.shape(0) <= 0 || packed_bytes <= 0) {
      return true;
    }
    if (weight.shape(2) * static_cast<int>(sizeof(uint32_t)) != packed_bytes ||
        scales.shape(0) != E || scales.shape(1) != N ||
        scales.shape(2) != K / group_size || biases.shape() != scales.shape()) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "DeepseekAffineGatherBlocksPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const auto& biases = inputs[3];
    const auto& block_meta = inputs[4];
    const auto& block_count = inputs[5];

    out.set_data(allocator::malloc(out.nbytes()));

    const auto cfg = mxfp4_blocks_variant(variant_);
    const int max_blocks = block_meta.shape(0);
    const int M = x.shape(0);
    const int K = x.shape(2);
    const int N = weight.shape(1);

    std::string kname;
    concatenate(
        kname,
        "deepseek_affine_gather_blocks_rhs_",
        glm_type_name(x.dtype()),
        "_gs_",
        group_size_,
        "_b_",
        bits_,
        "_bm_",
        cfg.bm,
        "_bn_",
        cfg.bn,
        "_bk_",
        cfg.bk,
        "_wm_",
        cfg.wm,
        "_wn_",
        cfg.wn);

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(weight, 1);
    compute_encoder.set_input_array(scales, 2);
    compute_encoder.set_input_array(biases, 3);
    compute_encoder.set_input_array(block_meta, 4);
    compute_encoder.set_input_array(block_count, 5);
    compute_encoder.set_output_array(out, 6);
    compute_encoder.set_bytes(max_blocks, 7);
    compute_encoder.set_bytes(M, 8);
    compute_encoder.set_bytes(N, 9);
    compute_encoder.set_bytes(K, 10);

    MTL::Size grid_dims((N + cfg.bn - 1) / cfg.bn, max_blocks, 1);
    MTL::Size group_dims(cfg.wm * cfg.wn * 32, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(DeepseekAffineGatherBlocksPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const DeepseekAffineGatherBlocksPrimitive&>(other);
    return group_size_ == rhs.group_size_ && bits_ == rhs.bits_ &&
        variant_ == rhs.variant_;
  }
  auto state() const {
    return std::make_tuple(group_size_, bits_, variant_);
  }

 private:
  int group_size_;
  int bits_;
  int variant_;
};

class DeepseekAffineGatherPairBlocksPrimitive : public Primitive {
 public:
  explicit DeepseekAffineGatherPairBlocksPrimitive(
      Stream stream,
      int group_size,
      int bits,
      int variant)
      : Primitive(stream),
        group_size_(group_size),
        bits_(bits),
        variant_(variant) {
    (void)mxfp4_blocks_variant(variant_);
    if (!supported_deepseek_affine(group_size_, bits_)) {
      throw std::invalid_argument(
          "Unsupported DeepSeek affine pair block-list quantization.");
    }
  }

  static bool unsupported(
      const array& x,
      const array& weight0,
      const array& scales0,
      const array& biases0,
      const array& weight1,
      const array& scales1,
      const array& biases1,
      const array& block_meta,
      const array& block_count,
      int group_size,
      int bits,
      Stream s) {
    if (s.device == Device::cpu || !supported_deepseek_affine(group_size, bits)) {
      return true;
    }
    if (x.dtype() != float16 && x.dtype() != bfloat16) {
      return true;
    }
    if (weight0.dtype() != uint32 || weight1.dtype() != uint32 ||
        scales0.dtype() != x.dtype() || scales1.dtype() != x.dtype() ||
        biases0.dtype() != x.dtype() || biases1.dtype() != x.dtype() ||
        block_meta.dtype() != int32 || block_count.dtype() != int32) {
      return true;
    }
    if (x.ndim() != 3 || x.shape(1) != 1 || weight0.ndim() != 3 ||
        scales0.ndim() != 3 || biases0.ndim() != 3 || weight1.ndim() != 3 ||
        scales1.ndim() != 3 || biases1.ndim() != 3 || block_meta.ndim() != 2 ||
        block_meta.shape(1) != 3 || block_count.size() != 1) {
      return true;
    }
    if (!row_contiguous(x) || !row_contiguous(weight0) ||
        !row_contiguous(scales0) || !row_contiguous(biases0) ||
        !row_contiguous(weight1) || !row_contiguous(scales1) ||
        !row_contiguous(biases1) || !row_contiguous(block_meta) ||
        !row_contiguous(block_count)) {
      return true;
    }

    const int K = x.shape(2);
    const int E = weight0.shape(0);
    const int N = weight0.shape(1);
    const int packed_bytes = affine_packed_row_bytes(K, bits);
    if (x.shape(0) <= 0 || K <= 0 || N <= 0 || E <= 0 ||
        block_meta.shape(0) <= 0 || packed_bytes <= 0) {
      return true;
    }
    if (weight1.shape() != weight0.shape() || scales1.shape() != scales0.shape() ||
        biases1.shape() != biases0.shape()) {
      return true;
    }
    if (weight0.shape(2) * static_cast<int>(sizeof(uint32_t)) != packed_bytes ||
        scales0.shape(0) != E || scales0.shape(1) != N ||
        scales0.shape(2) != K / group_size ||
        biases0.shape() != scales0.shape()) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "DeepseekAffineGatherPairBlocksPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x = inputs[0];
    const auto& weight0 = inputs[1];
    const auto& scales0 = inputs[2];
    const auto& biases0 = inputs[3];
    const auto& weight1 = inputs[4];
    const auto& scales1 = inputs[5];
    const auto& biases1 = inputs[6];
    const auto& block_meta = inputs[7];
    const auto& block_count = inputs[8];

    out.set_data(allocator::malloc(out.nbytes()));

    const auto cfg = mxfp4_blocks_variant(variant_);
    const int max_blocks = block_meta.shape(0);
    const int M = x.shape(0);
    const int K = x.shape(2);
    const int N = weight0.shape(1);

    std::string kname;
    concatenate(
        kname,
        "deepseek_affine_gather_pair_concat_blocks_rhs_",
        glm_type_name(x.dtype()),
        "_gs_",
        group_size_,
        "_b_",
        bits_,
        "_bm_",
        cfg.bm,
        "_bn_",
        cfg.bn,
        "_bk_",
        cfg.bk,
        "_wm_",
        cfg.wm,
        "_wn_",
        cfg.wn);

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(weight0, 1);
    compute_encoder.set_input_array(scales0, 2);
    compute_encoder.set_input_array(biases0, 3);
    compute_encoder.set_input_array(weight1, 4);
    compute_encoder.set_input_array(scales1, 5);
    compute_encoder.set_input_array(biases1, 6);
    compute_encoder.set_input_array(block_meta, 7);
    compute_encoder.set_input_array(block_count, 8);
    compute_encoder.set_output_array(out, 9);
    compute_encoder.set_bytes(max_blocks, 10);
    compute_encoder.set_bytes(M, 11);
    compute_encoder.set_bytes(N, 12);
    compute_encoder.set_bytes(K, 13);

    MTL::Size grid_dims((2 * N + cfg.bn - 1) / cfg.bn, max_blocks, 1);
    MTL::Size group_dims(cfg.wm * cfg.wn * 32, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(DeepseekAffineGatherPairBlocksPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const DeepseekAffineGatherPairBlocksPrimitive&>(other);
    return group_size_ == rhs.group_size_ && bits_ == rhs.bits_ &&
        variant_ == rhs.variant_;
  }
  auto state() const {
    return std::make_tuple(group_size_, bits_, variant_);
  }

 private:
  int group_size_;
  int bits_;
  int variant_;
};

class DeepseekMxfp4GatherExpertPrimitive : public Primitive {
 public:
  explicit DeepseekMxfp4GatherExpertPrimitive(Stream stream, int variant)
      : Primitive(stream), variant_(variant) {
    (void)mxfp4_blocks_variant(variant_);
  }

  static bool unsupported(
      const array& x,
      const array& weight,
      const array& scales,
      const array& indices,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (x.dtype() != float16 && x.dtype() != bfloat16) {
      return true;
    }
    if (weight.dtype() != uint32 || scales.dtype() != uint8 ||
        (indices.dtype() != uint32 && indices.dtype() != int32)) {
      return true;
    }
    if (x.ndim() != 3 || x.shape(1) != 1 || weight.ndim() != 3 ||
        scales.ndim() != 3 || indices.ndim() != 1 ||
        indices.size() != x.shape(0)) {
      return true;
    }
    if (!row_contiguous(x) || !row_contiguous(weight) ||
        !row_contiguous(scales) || !row_contiguous(indices)) {
      return true;
    }

    constexpr int bits = 4;
    constexpr int group_size = 32;
    constexpr int values_per_uint32 = 32 / bits;
    const int K = x.shape(2);
    const int E = weight.shape(0);
    const int N = weight.shape(1);
    if (x.shape(0) <= 0 || K <= 0 || N <= 0 || E <= 0) {
      return true;
    }
    if (weight.shape(2) * values_per_uint32 != K || scales.shape(0) != E ||
        scales.shape(1) != N || scales.shape(2) != K / group_size) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "DeepseekMxfp4GatherExpertPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const auto& indices = inputs[3];

    out.set_data(allocator::malloc(out.nbytes()));

    const auto cfg = mxfp4_blocks_variant(variant_);
    const int M = x.shape(0);
    const int K = x.shape(2);
    const int E = weight.shape(0);
    const int N = weight.shape(1);

    std::string kname;
    concatenate(
        kname,
        "deepseek_mxfp4_gather_expert_rhs_",
        glm_type_name(x.dtype()),
        "_bm_",
        cfg.bm,
        "_bn_",
        cfg.bn,
        "_bk_",
        cfg.bk,
        "_wm_",
        cfg.wm,
        "_wn_",
        cfg.wn);

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(weight, 1);
    compute_encoder.set_input_array(scales, 2);
    compute_encoder.set_input_array(indices, 3);
    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_bytes(M, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(K, 7);
    compute_encoder.set_bytes(E, 8);

    MTL::Size grid_dims((N + cfg.bn - 1) / cfg.bn, E, 1);
    MTL::Size group_dims(cfg.wm * cfg.wn * 32, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(DeepseekMxfp4GatherExpertPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const DeepseekMxfp4GatherExpertPrimitive&>(other);
    return variant_ == rhs.variant_;
  }
  auto state() const {
    return std::make_tuple(variant_);
  }

 private:
  int variant_;
};

} // namespace

array glm_dsa_q8_vup_flat(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    StreamOrDevice s /* = {} */) {
  if (x.ndim() != 4 || weight.ndim() != 3 || scales.ndim() != 3 ||
      biases.ndim() != 3) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_q8_vup_flat] expected x rank 4 and "
        << "quantized weights rank 3, got " << x.shape() << ", "
        << weight.shape() << ", " << scales.shape() << ", " << biases.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }

  const int B = x.shape(0);
  const int H = x.shape(1);
  const int L = x.shape(2);
  constexpr int bits = 8;
  constexpr int group_size = 64;
  constexpr int pack_factor = 32 / bits;
  const int V = weight.shape(1);
  const int K = weight.shape(2) * pack_factor;
  if (H != weight.shape(0) || H != scales.shape(0) ||
      H != biases.shape(0) || V != scales.shape(1) ||
      V != biases.shape(1) || x.shape(3) != K ||
      scales.shape(2) != K / group_size ||
      biases.shape(2) != K / group_size) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_q8_vup_flat] incompatible shapes: "
        << x.shape() << ", " << weight.shape() << ", " << scales.shape()
        << ", " << biases.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_q8_vup_flat] expected float16 or "
        << "bfloat16 input, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
      biases.dtype() != x.dtype()) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_q8_vup_flat] expected uint32 weight and "
        << "scale/bias dtype " << x.dtype() << ", got " << weight.dtype()
        << ", " << scales.dtype() << ", " << biases.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {x, weight, scales, biases};
  if (GlmDsaQ8VupFlatPrimitive::unsupported(x, weight, scales, biases, stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.glm_dsa_q8_vup_flat] unsupported M3 GLM shape.");
  }

  Shape out_shape{B, L, H * V};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<GlmDsaQ8VupFlatPrimitive>(stream),
      std::move(inputs));
}

array glm_moe_weighted_sum(
    const array& x_sorted,
    const array& inv_order,
    const array& scores,
    StreamOrDevice s /* = {} */) {
  if (x_sorted.ndim() != 3 || x_sorted.shape(-2) != 1) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_moe_weighted_sum] expected x_sorted shape "
        << "[N, 1, D], got " << x_sorted.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (scores.ndim() < 2) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_moe_weighted_sum] expected scores rank >= 2, "
        << "got " << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (inv_order.ndim() != 1 || inv_order.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_moe_weighted_sum] expected uint32 inv_order "
        << "rank 1, got " << inv_order.shape() << " dtype "
        << inv_order.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  const int topk = scores.shape(-1);
  const int64_t routed_rows = scores.size();
  const int D = x_sorted.shape(-1);
  if (x_sorted.shape(0) != routed_rows || inv_order.size() != routed_rows) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_moe_weighted_sum] incompatible shapes: "
        << x_sorted.shape() << ", " << inv_order.shape() << ", "
        << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (topk <= 0 || D <= 0) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_moe_weighted_sum] invalid topk or hidden "
        << "dim: topk=" << topk << ", D=" << D << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(x_sorted.dtype(), floating)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_moe_weighted_sum] expected floating "
        << "x_sorted, got " << x_sorted.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {x_sorted, inv_order, scores};
  Shape out_shape = scores.shape();
  out_shape.pop_back();
  out_shape.push_back(D);
  if (GlmMoeWeightedSumPrimitive::unsupported(
          x_sorted, inv_order, scores, stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.glm_moe_weighted_sum] unsupported M3 GLM shape.");
  }
  return array(
      std::move(out_shape),
      x_sorted.dtype(),
      std::make_shared<GlmMoeWeightedSumPrimitive>(stream),
      std::move(inputs));
}

array deepseek_mxfp4_gather_qmm_blocks(
    const array& x,
    const array& weight,
    const array& scales,
    const array& block_meta,
    const array& block_count,
    int variant,
    StreamOrDevice s /* = {} */) {
  (void)mxfp4_blocks_variant(variant);

  if (x.ndim() != 3 || x.shape(1) != 1 || weight.ndim() != 3 ||
      scales.ndim() != 3 || block_meta.ndim() != 2 ||
      block_meta.shape(1) != 3 || block_count.size() != 1) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_blocks] expected "
        << "x [M,1,K], weight [E,N,K/8], scales [E,N,K/32], "
        << "block_meta [B,3], block_count [1], got " << x.shape() << ", "
        << weight.shape() << ", " << scales.shape() << ", "
        << block_meta.shape() << ", " << block_count.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  constexpr int bits = 4;
  constexpr int group_size = 32;
  constexpr int values_per_uint32 = 32 / bits;
  const int K = x.shape(2);
  const int E = weight.shape(0);
  const int N = weight.shape(1);
  if (weight.shape(2) * values_per_uint32 != K || scales.shape(0) != E ||
      scales.shape(1) != N || scales.shape(2) != K / group_size) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_blocks] "
        << "incompatible shapes: " << x.shape() << ", " << weight.shape()
        << ", " << scales.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_blocks] expected "
        << "float16 or bfloat16 input, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32 || scales.dtype() != uint8 ||
      block_meta.dtype() != int32 || block_count.dtype() != int32) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_blocks] expected "
        << "uint32 weight, uint8 scales, int32 block_meta/count, got "
        << weight.dtype() << ", " << scales.dtype() << ", "
        << block_meta.dtype() << ", " << block_count.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {x, weight, scales, block_meta, block_count};
  if (DeepseekMxfp4GatherBlocksPrimitive::unsupported(
          x, weight, scales, block_meta, block_count, stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_blocks] unsupported shape.");
  }

  Shape out_shape{x.shape(0), 1, N};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<DeepseekMxfp4GatherBlocksPrimitive>(stream, variant),
      std::move(inputs));
}

array deepseek_mxfp4_gather_qmm_pair_blocks(
    const array& x,
    const array& weight0,
    const array& scales0,
    const array& weight1,
    const array& scales1,
    const array& block_meta,
    const array& block_count,
    int variant,
    StreamOrDevice s /* = {} */) {
  (void)mxfp4_blocks_variant(variant);

  if (x.ndim() != 3 || x.shape(1) != 1 || weight0.ndim() != 3 ||
      scales0.ndim() != 3 || weight1.ndim() != 3 || scales1.ndim() != 3 ||
      block_meta.ndim() != 2 || block_meta.shape(1) != 3 ||
      block_count.size() != 1) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_pair_blocks] "
        << "expected x [M,1,K], two weights [E,N,K/8], two scales "
        << "[E,N,K/32], block_meta [B,3], block_count [1], got "
        << x.shape() << ", " << weight0.shape() << ", " << scales0.shape()
        << ", " << weight1.shape() << ", " << scales1.shape() << ", "
        << block_meta.shape() << ", " << block_count.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  constexpr int bits = 4;
  constexpr int group_size = 32;
  constexpr int values_per_uint32 = 32 / bits;
  const int K = x.shape(2);
  const int E = weight0.shape(0);
  const int N = weight0.shape(1);
  if (weight1.shape() != weight0.shape() || scales1.shape() != scales0.shape() ||
      weight0.shape(2) * values_per_uint32 != K || scales0.shape(0) != E ||
      scales0.shape(1) != N || scales0.shape(2) != K / group_size) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_pair_blocks] "
        << "incompatible shapes: " << x.shape() << ", " << weight0.shape()
        << ", " << scales0.shape() << ", " << weight1.shape() << ", "
        << scales1.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_pair_blocks] expected "
        << "float16 or bfloat16 input, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight0.dtype() != uint32 || scales0.dtype() != uint8 ||
      weight1.dtype() != uint32 || scales1.dtype() != uint8 ||
      block_meta.dtype() != int32 || block_count.dtype() != int32) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_pair_blocks] expected "
        << "uint32 weights, uint8 scales, int32 block_meta/count, got "
        << weight0.dtype() << ", " << scales0.dtype() << ", "
        << weight1.dtype() << ", " << scales1.dtype() << ", "
        << block_meta.dtype() << ", " << block_count.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {
      x, weight0, scales0, weight1, scales1, block_meta, block_count};
  if (DeepseekMxfp4GatherPairBlocksPrimitive::unsupported(
          x, weight0, scales0, weight1, scales1, block_meta, block_count, stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_pair_blocks] unsupported shape.");
  }

  Shape out_shape{2, x.shape(0), 1, N};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<DeepseekMxfp4GatherPairBlocksPrimitive>(stream, variant),
      std::move(inputs));
}

array deepseek_mxfp4_gather_qmm_pair_concat_blocks(
    const array& x,
    const array& weight0,
    const array& scales0,
    const array& weight1,
    const array& scales1,
    const array& block_meta,
    const array& block_count,
    int variant,
    StreamOrDevice s /* = {} */) {
  const auto cfg = mxfp4_blocks_variant(variant);
  auto stream = to_stream(s);
  if (DeepseekMxfp4GatherPairBlocksPrimitive::unsupported(
          x, weight0, scales0, weight1, scales1, block_meta, block_count, stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_pair_concat_blocks] unsupported shape.");
  }
  const int N = weight0.shape(1);
  if (N % cfg.bn != 0) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_pair_concat_blocks] "
        "output dimension must be divisible by the block N.");
  }
  std::vector<array> inputs = {
      x, weight0, scales0, weight1, scales1, block_meta, block_count};
  Shape out_shape{x.shape(0), 1, 2 * N};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<DeepseekMxfp4GatherPairBlocksPrimitive>(
          stream, variant, true),
      std::move(inputs));
}

array deepseek_affine_gather_qmm_blocks(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    const array& block_meta,
    const array& block_count,
    int group_size,
    int bits,
    int variant,
    StreamOrDevice s /* = {} */) {
  (void)mxfp4_blocks_variant(variant);

  if (x.ndim() != 3 || x.shape(1) != 1 || weight.ndim() != 3 ||
      scales.ndim() != 3 || biases.ndim() != 3 || block_meta.ndim() != 2 ||
      block_meta.shape(1) != 3 || block_count.size() != 1) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_affine_gather_qmm_blocks] expected "
        << "x [M,1,K], weight [E,N,packed_words], scales/biases "
        << "[E,N,K/group_size], block_meta [B,3], block_count [1], got "
        << x.shape() << ", " << weight.shape() << ", " << scales.shape()
        << ", " << biases.shape() << ", " << block_meta.shape() << ", "
        << block_count.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  if (!supported_deepseek_affine(group_size, bits)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_affine_gather_qmm_blocks] unsupported "
        << "affine quantization group_size=" << group_size << " bits=" << bits
        << ".";
    throw std::invalid_argument(msg.str());
  }

  const int K = x.shape(2);
  const int E = weight.shape(0);
  const int N = weight.shape(1);
  const int packed_bytes = affine_packed_row_bytes(K, bits);
  if (packed_bytes <= 0 ||
      weight.shape(2) * static_cast<int>(sizeof(uint32_t)) != packed_bytes ||
      scales.shape(0) != E || scales.shape(1) != N ||
      scales.shape(2) != K / group_size || biases.shape() != scales.shape()) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_affine_gather_qmm_blocks] "
        << "incompatible shapes: " << x.shape() << ", " << weight.shape()
        << ", " << scales.shape() << ", " << biases.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_affine_gather_qmm_blocks] expected "
        << "float16 or bfloat16 input, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
      biases.dtype() != x.dtype() || block_meta.dtype() != int32 ||
      block_count.dtype() != int32) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_affine_gather_qmm_blocks] expected "
        << "uint32 weight, scales/biases matching input dtype, int32 "
        << "block_meta/count, got " << weight.dtype() << ", "
        << scales.dtype() << ", " << biases.dtype() << ", "
        << block_meta.dtype() << ", " << block_count.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {
      x, weight, scales, biases, block_meta, block_count};
  if (DeepseekAffineGatherBlocksPrimitive::unsupported(
          x,
          weight,
          scales,
          biases,
          block_meta,
          block_count,
          group_size,
          bits,
          stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.deepseek_affine_gather_qmm_blocks] unsupported shape.");
  }

  Shape out_shape{x.shape(0), 1, N};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<DeepseekAffineGatherBlocksPrimitive>(
          stream, group_size, bits, variant),
      std::move(inputs));
}

array deepseek_affine_gather_qmm_pair_concat_blocks(
    const array& x,
    const array& weight0,
    const array& scales0,
    const array& biases0,
    const array& weight1,
    const array& scales1,
    const array& biases1,
    const array& block_meta,
    const array& block_count,
    int group_size,
    int bits,
    int variant,
    StreamOrDevice s /* = {} */) {
  const auto cfg = mxfp4_blocks_variant(variant);
  auto stream = to_stream(s);
  if (DeepseekAffineGatherPairBlocksPrimitive::unsupported(
          x,
          weight0,
          scales0,
          biases0,
          weight1,
          scales1,
          biases1,
          block_meta,
          block_count,
          group_size,
          bits,
          stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.deepseek_affine_gather_qmm_pair_concat_blocks] unsupported shape.");
  }
  const int N = weight0.shape(1);
  if (N % cfg.bn != 0) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.deepseek_affine_gather_qmm_pair_concat_blocks] "
        "output dimension must be divisible by the block N.");
  }
  std::vector<array> inputs = {
      x,
      weight0,
      scales0,
      biases0,
      weight1,
      scales1,
      biases1,
      block_meta,
      block_count};
  Shape out_shape{x.shape(0), 1, 2 * N};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<DeepseekAffineGatherPairBlocksPrimitive>(
          stream, group_size, bits, variant),
      std::move(inputs));
}

array deepseek_mxfp4_gather_qmm_expert(
    const array& x,
    const array& weight,
    const array& scales,
    const array& indices,
    int variant,
    StreamOrDevice s /* = {} */) {
  (void)mxfp4_blocks_variant(variant);

  if (x.ndim() != 3 || x.shape(1) != 1 || weight.ndim() != 3 ||
      scales.ndim() != 3 || indices.ndim() != 1 ||
      indices.size() != x.shape(0)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_expert] expected "
        << "x [M,1,K], weight [E,N,K/8], scales [E,N,K/32], "
        << "indices [M], got " << x.shape() << ", " << weight.shape() << ", "
        << scales.shape() << ", " << indices.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  constexpr int bits = 4;
  constexpr int group_size = 32;
  constexpr int values_per_uint32 = 32 / bits;
  const int K = x.shape(2);
  const int E = weight.shape(0);
  const int N = weight.shape(1);
  if (weight.shape(2) * values_per_uint32 != K || scales.shape(0) != E ||
      scales.shape(1) != N || scales.shape(2) != K / group_size) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_expert] "
        << "incompatible shapes: " << x.shape() << ", " << weight.shape()
        << ", " << scales.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_expert] expected "
        << "float16 or bfloat16 input, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32 || scales.dtype() != uint8 ||
      (indices.dtype() != uint32 && indices.dtype() != int32)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_expert] expected "
        << "uint32 weight, uint8 scales, uint32/int32 indices, got "
        << weight.dtype() << ", " << scales.dtype() << ", "
        << indices.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {x, weight, scales, indices};
  if (DeepseekMxfp4GatherExpertPrimitive::unsupported(
          x, weight, scales, indices, stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_expert] unsupported shape.");
  }

  Shape out_shape{x.shape(0), 1, N};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<DeepseekMxfp4GatherExpertPrimitive>(stream, variant),
      std::move(inputs));
}

} // namespace omlx::glm_kernels
