#include "gather_qmm.h"
#include "omlx/binary_dir.h"

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

namespace omlx::quant_matmul {

namespace {

using namespace mlx::core;

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
    case 6:
      return 4;
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
    case 6:
      return 3;
    case 8:
      return 1;
    default:
      return 0;
  }
}

bool supported_affine_gather(int group_size, int bits) {
  return group_size == 64 &&
      (bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8);
}

int affine_packed_row_bytes(int K, int bits) {
  const int pack_factor = affine_pack_factor(bits);
  const int bytes_per_pack = affine_bytes_per_pack(bits);
  if (pack_factor == 0 || bytes_per_pack == 0 || K % pack_factor != 0) {
    return -1;
  }
  return K * bytes_per_pack / pack_factor;
}

std::string metal_type_name(Dtype dtype) {
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

class Mxfp4GatherBlocksPrimitive : public Primitive {
 public:
  explicit Mxfp4GatherBlocksPrimitive(Stream stream, int variant)
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
        "Mxfp4GatherBlocksPrimitive has no CPU path.");
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
        metal_type_name(x.dtype()),
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

    auto lib = d.get_library("omlx_quant_matmul_kernels", binary_dir());
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

  DEFINE_NAME(Mxfp4GatherBlocksPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const Mxfp4GatherBlocksPrimitive&>(other);
    return variant_ == rhs.variant_;
  }
  auto state() const {
    return std::make_tuple(variant_);
  }

 private:
  int variant_;
};

class Mxfp4GatherPairBlocksPrimitive : public Primitive {
 public:
  explicit Mxfp4GatherPairBlocksPrimitive(
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
        "Mxfp4GatherPairBlocksPrimitive has no CPU path.");
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
        metal_type_name(x.dtype()),
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

    auto lib = d.get_library("omlx_quant_matmul_kernels", binary_dir());
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

  DEFINE_NAME(Mxfp4GatherPairBlocksPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const Mxfp4GatherPairBlocksPrimitive&>(other);
    return variant_ == rhs.variant_ && concat_output_ == rhs.concat_output_;
  }
  auto state() const {
    return std::make_tuple(variant_, concat_output_);
  }

 private:
  int variant_;
  bool concat_output_;
};

class AffineGatherBlocksPrimitive : public Primitive {
 public:
  explicit AffineGatherBlocksPrimitive(
      Stream stream,
      int group_size,
      int bits,
      int variant)
      : Primitive(stream),
        group_size_(group_size),
        bits_(bits),
        variant_(variant) {
    (void)mxfp4_blocks_variant(variant_);
    if (!supported_affine_gather(group_size_, bits_)) {
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
    if (s.device == Device::cpu || !supported_affine_gather(group_size, bits)) {
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
        "AffineGatherBlocksPrimitive has no CPU path.");
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
        metal_type_name(x.dtype()),
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

    auto lib = d.get_library("omlx_quant_matmul_kernels", binary_dir());
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

  DEFINE_NAME(AffineGatherBlocksPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const AffineGatherBlocksPrimitive&>(other);
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

class AffineGatherPairBlocksPrimitive : public Primitive {
 public:
  explicit AffineGatherPairBlocksPrimitive(
      Stream stream,
      int group_size,
      int bits,
      int variant)
      : Primitive(stream),
        group_size_(group_size),
        bits_(bits),
        variant_(variant) {
    (void)mxfp4_blocks_variant(variant_);
    if (!supported_affine_gather(group_size_, bits_)) {
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
    if (s.device == Device::cpu || !supported_affine_gather(group_size, bits)) {
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
        "AffineGatherPairBlocksPrimitive has no CPU path.");
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
        metal_type_name(x.dtype()),
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

    auto lib = d.get_library("omlx_quant_matmul_kernels", binary_dir());
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

  DEFINE_NAME(AffineGatherPairBlocksPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const AffineGatherPairBlocksPrimitive&>(other);
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

} // namespace

array mxfp4_gather_qmm_blocks(
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
    msg << "[quant_matmul.mxfp4_gather_qmm_blocks] expected "
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
    msg << "[quant_matmul.mxfp4_gather_qmm_blocks] "
        << "incompatible shapes: " << x.shape() << ", " << weight.shape()
        << ", " << scales.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[quant_matmul.mxfp4_gather_qmm_blocks] expected "
        << "float16 or bfloat16 input, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32 || scales.dtype() != uint8 ||
      block_meta.dtype() != int32 || block_count.dtype() != int32) {
    std::ostringstream msg;
    msg << "[quant_matmul.mxfp4_gather_qmm_blocks] expected "
        << "uint32 weight, uint8 scales, int32 block_meta/count, got "
        << weight.dtype() << ", " << scales.dtype() << ", "
        << block_meta.dtype() << ", " << block_count.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {x, weight, scales, block_meta, block_count};
  if (Mxfp4GatherBlocksPrimitive::unsupported(
          x, weight, scales, block_meta, block_count, stream)) {
    throw std::invalid_argument(
        "[quant_matmul.mxfp4_gather_qmm_blocks] unsupported shape.");
  }

  Shape out_shape{x.shape(0), 1, N};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<Mxfp4GatherBlocksPrimitive>(stream, variant),
      std::move(inputs));
}

array mxfp4_gather_qmm_pair_blocks(
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
    msg << "[quant_matmul.mxfp4_gather_qmm_pair_blocks] "
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
    msg << "[quant_matmul.mxfp4_gather_qmm_pair_blocks] "
        << "incompatible shapes: " << x.shape() << ", " << weight0.shape()
        << ", " << scales0.shape() << ", " << weight1.shape() << ", "
        << scales1.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[quant_matmul.mxfp4_gather_qmm_pair_blocks] expected "
        << "float16 or bfloat16 input, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight0.dtype() != uint32 || scales0.dtype() != uint8 ||
      weight1.dtype() != uint32 || scales1.dtype() != uint8 ||
      block_meta.dtype() != int32 || block_count.dtype() != int32) {
    std::ostringstream msg;
    msg << "[quant_matmul.mxfp4_gather_qmm_pair_blocks] expected "
        << "uint32 weights, uint8 scales, int32 block_meta/count, got "
        << weight0.dtype() << ", " << scales0.dtype() << ", "
        << weight1.dtype() << ", " << scales1.dtype() << ", "
        << block_meta.dtype() << ", " << block_count.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {
      x, weight0, scales0, weight1, scales1, block_meta, block_count};
  if (Mxfp4GatherPairBlocksPrimitive::unsupported(
          x, weight0, scales0, weight1, scales1, block_meta, block_count, stream)) {
    throw std::invalid_argument(
        "[quant_matmul.mxfp4_gather_qmm_pair_blocks] unsupported shape.");
  }

  Shape out_shape{2, x.shape(0), 1, N};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<Mxfp4GatherPairBlocksPrimitive>(stream, variant),
      std::move(inputs));
}

array mxfp4_gather_qmm_pair_concat_blocks(
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
  if (Mxfp4GatherPairBlocksPrimitive::unsupported(
          x, weight0, scales0, weight1, scales1, block_meta, block_count, stream)) {
    throw std::invalid_argument(
        "[quant_matmul.mxfp4_gather_qmm_pair_concat_blocks] unsupported shape.");
  }
  const int N = weight0.shape(1);
  if (N % cfg.bn != 0) {
    throw std::invalid_argument(
        "[quant_matmul.mxfp4_gather_qmm_pair_concat_blocks] "
        "output dimension must be divisible by the block N.");
  }
  std::vector<array> inputs = {
      x, weight0, scales0, weight1, scales1, block_meta, block_count};
  Shape out_shape{x.shape(0), 1, 2 * N};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<Mxfp4GatherPairBlocksPrimitive>(
          stream, variant, true),
      std::move(inputs));
}

array affine_gather_qmm_blocks(
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
    msg << "[quant_matmul.affine_gather_qmm_blocks] expected "
        << "x [M,1,K], weight [E,N,packed_words], scales/biases "
        << "[E,N,K/group_size], block_meta [B,3], block_count [1], got "
        << x.shape() << ", " << weight.shape() << ", " << scales.shape()
        << ", " << biases.shape() << ", " << block_meta.shape() << ", "
        << block_count.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  if (!supported_affine_gather(group_size, bits)) {
    std::ostringstream msg;
    msg << "[quant_matmul.affine_gather_qmm_blocks] unsupported "
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
    msg << "[quant_matmul.affine_gather_qmm_blocks] "
        << "incompatible shapes: " << x.shape() << ", " << weight.shape()
        << ", " << scales.shape() << ", " << biases.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[quant_matmul.affine_gather_qmm_blocks] expected "
        << "float16 or bfloat16 input, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
      biases.dtype() != x.dtype() || block_meta.dtype() != int32 ||
      block_count.dtype() != int32) {
    std::ostringstream msg;
    msg << "[quant_matmul.affine_gather_qmm_blocks] expected "
        << "uint32 weight, scales/biases matching input dtype, int32 "
        << "block_meta/count, got " << weight.dtype() << ", "
        << scales.dtype() << ", " << biases.dtype() << ", "
        << block_meta.dtype() << ", " << block_count.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {
      x, weight, scales, biases, block_meta, block_count};
  if (AffineGatherBlocksPrimitive::unsupported(
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
        "[quant_matmul.affine_gather_qmm_blocks] unsupported shape.");
  }

  Shape out_shape{x.shape(0), 1, N};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<AffineGatherBlocksPrimitive>(
          stream, group_size, bits, variant),
      std::move(inputs));
}

array affine_gather_qmm_pair_concat_blocks(
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
  if (AffineGatherPairBlocksPrimitive::unsupported(
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
        "[quant_matmul.affine_gather_qmm_pair_concat_blocks] unsupported shape.");
  }
  const int N = weight0.shape(1);
  if (N % cfg.bn != 0) {
    throw std::invalid_argument(
        "[quant_matmul.affine_gather_qmm_pair_concat_blocks] "
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
      std::make_shared<AffineGatherPairBlocksPrimitive>(
          stream, group_size, bits, variant),
      std::move(inputs));
}

} // namespace omlx::quant_matmul
