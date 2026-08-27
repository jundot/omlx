#include "dspark_gemm.h"

#include <algorithm>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;
using namespace mlx::steel;

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to locate DSpark kernel binary.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

int next_power_of_two(int value) {
  int result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

class DSparkRowwiseGemmPrimitive : public Primitive {
 public:
  DSparkRowwiseGemmPrimitive(Stream stream, bool transpose_rhs)
      : Primitive(stream), transpose_rhs_(transpose_rhs) {}

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("DSpark rowwise GEMM has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    const auto& lhs = inputs[0];
    const auto& rhs = inputs[1];
    auto& output = outputs[0];

    const bool fp32 = lhs.dtype() == float32;
    const int bm = fp32 && transpose_rhs_ ? 32 : 64;
    const int bn = fp32 ? (transpose_rhs_ ? 64 : 32)
                        : (transpose_rhs_ ? 32 : 64);
    const int bk = fp32 ? (transpose_rhs_ ? 16 : 32)
                        : (transpose_rhs_ ? 32 : 16);
    const int wm = fp32 ? (transpose_rhs_ ? 1 : 2)
                        : (transpose_rhs_ ? 2 : 1);
    constexpr int wn = 2;
    const int rows = lhs.shape(0);
    const int M = lhs.shape(1);
    const int K = lhs.shape(2);
    const int N = transpose_rhs_ ? rhs.shape(1) : rhs.shape(2);
    const int ldb = transpose_rhs_ ? K : N;

    output.set_data(allocator::malloc(output.nbytes()));
    auto library = device.get_library("omlx_glm_kernels", current_binary_dir());

    // Match the split-K choice made by an independent MLX M=1 matmul. The
    // row dimension is encoded only in grid.z, so every verify row retains
    // exactly the same partition and accumulation order as ordinary decode.
    const int tile_m = (M + 15) / 16;
    const int tile_n = (N + 15) / 16;
    const int tile_k = K / 16;
    const bool use_split_k = tile_m * tile_n <= 2048 && tile_k >= 8 &&
        K >= std::max(M, N);
    if (use_split_k) {
      constexpr int split_bm = 32;
      constexpr int split_bn = 32;
      constexpr int split_bk = 16;
      constexpr int split_wm = 2;
      constexpr int split_wn = 2;
      const int split_tiles_m = (M + split_bm - 1) / split_bm;
      const int split_tiles_n = (N + split_bn - 1) / split_bn;
      const int split_ratio =
          (K / split_bk) / (split_tiles_m * split_tiles_n);
      const int partitions =
          std::min(std::max(2, next_power_of_two(split_ratio)), 32);
      const int partition_stride = M * N;
      const int gemm_k_iterations = (K / split_bk) / partitions;
      const int partition_size = gemm_k_iterations * split_bk;

      array split(
          {rows, partitions, M, N},
          float32,
          nullptr,
          {});
      split.set_data(allocator::malloc(split.nbytes()));

      const bool mn_aligned = M % split_bm == 0 && N % split_bn == 0;
      const bool k_aligned = K % split_bk == 0;
      std::ostringstream split_name;
      split_name << "omlx_dspark_gemm_splitk_" << type_to_name(lhs)
                 << "_nt" << (transpose_rhs_ ? "true" : "false") << "_MN_"
                 << (mn_aligned ? "taligned" : "naligned") << "_K_"
                 << (k_aligned ? "taligned" : "naligned");

      auto split_kernel = device.get_kernel(split_name.str(), library);
      auto& encoder = metal::get_command_encoder(stream);
      encoder.set_compute_pipeline_state(split_kernel);

      GEMMSpiltKParams params{
          M,
          N,
          K,
          K,
          ldb,
          N,
          split_tiles_n,
          split_tiles_m,
          partitions,
          partition_stride,
          partition_size,
          0,
          gemm_k_iterations};
      const int64_t batch_stride_a = static_cast<int64_t>(M) * K;
      const int64_t batch_stride_b =
          static_cast<int64_t>(rhs.shape(1)) * rhs.shape(2);

      encoder.set_input_array(lhs, 0);
      encoder.set_input_array(rhs, 1);
      encoder.set_output_array(split, 2);
      encoder.set_bytes(params, 3);
      encoder.set_bytes(batch_stride_a, 4);
      encoder.set_bytes(batch_stride_b, 5);
      encoder.dispatch_threadgroups(
          MTL::Size(split_tiles_n, split_tiles_m, rows * partitions),
          MTL::Size(32, split_wn, split_wm));

      const std::string accum_name =
          "omlx_dspark_gemm_splitk_accum_" + type_to_name(output) +
          "_float32";
      auto accum_kernel = device.get_kernel(accum_name, library);
      encoder.set_compute_pipeline_state(accum_kernel);
      encoder.set_input_array(split, 0);
      encoder.set_output_array(output, 1);
      encoder.set_bytes(partitions, 2);
      encoder.set_bytes(partition_stride, 3);
      encoder.set_bytes(N, 4);
      encoder.dispatch_threads(
          MTL::Size(N, M, rows), MTL::Size(32, 8, 1));
      encoder.add_temporary(std::move(split));
      return;
    }

    const bool has_batch = false;
    const bool use_out_source = false;
    const bool do_axpby = false;
    const bool align_m = M % bm == 0;
    const bool align_n = N % bn == 0;
    const bool align_k = K % bk == 0;
    metal::MTLFCList constants = {
        {&has_batch, MTL::DataType::DataTypeBool, 10},
        {&use_out_source, MTL::DataType::DataTypeBool, 100},
        {&do_axpby, MTL::DataType::DataTypeBool, 110},
        {&align_m, MTL::DataType::DataTypeBool, 200},
        {&align_n, MTL::DataType::DataTypeBool, 201},
        {&align_k, MTL::DataType::DataTypeBool, 202},
    };

    std::string base_name = "omlx_dspark_gemm_" + type_to_name(lhs) +
        "_nt" + (transpose_rhs_ ? "true" : "false");
    std::ostringstream hash;
    hash << base_name << "_am" << align_m << "_an" << align_n << "_ak"
         << align_k;

    auto kernel = device.get_kernel(base_name, library, hash.str(), constants);
    auto& encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);

    constexpr int swizzle_log = 0;
    const int tiles_n = (N + bn - 1) / bn;
    const int tiles_m = (M + bm - 1) / bm;
    GEMMParams params{
        M,
        N,
        K,
        K,
        ldb,
        N,
        tiles_n,
        tiles_m,
        static_cast<int64_t>(M) * K,
        static_cast<int64_t>(rhs.shape(1)) * rhs.shape(2),
        static_cast<int64_t>(M) * N,
        swizzle_log,
        K / bk,
        1};

    encoder.set_input_array(lhs, 0);
    encoder.set_input_array(rhs, 1);
    encoder.set_output_array(output, 3);
    encoder.set_bytes(params, 4);

    const int swizzle = 1 << swizzle_log;
    const int grid_n = tiles_n * swizzle;
    const int grid_m = (tiles_m + swizzle - 1) / swizzle;
    encoder.dispatch_threadgroups(
        MTL::Size(grid_n, grid_m, rows), MTL::Size(32, wn, wm));
  }

  DEFINE_NAME(OMLXDSparkRowwiseGemm)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    return transpose_rhs_ ==
        static_cast<const DSparkRowwiseGemmPrimitive&>(other).transpose_rhs_;
  }
  auto state() const {
    return std::make_tuple(nullptr, transpose_rhs_);
  }

 private:
  bool transpose_rhs_;
};

class DSparkRingGemmPrimitive : public Primitive {
 public:
  DSparkRingGemmPrimitive(Stream stream, bool transpose_rhs)
      : Primitive(stream), transpose_rhs_(transpose_rhs) {}

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("DSpark physical-ring GEMM has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    const auto& lhs = inputs[0];
    const auto& source = inputs[1];
    const auto& indices = inputs[2];
    auto& output = outputs[0];

    constexpr int M = 64;
    constexpr int ring_size = 128;
    constexpr int head_dim = 512;
    const int rows = lhs.shape(0);
    const int64_t batch_stride_a =
        static_cast<int64_t>(lhs.shape(1)) * lhs.shape(2);
    const int source_ld = source.shape(1);

    output.set_data(allocator::malloc(output.nbytes()));
    auto library = device.get_library("omlx_glm_kernels", current_binary_dir());
    auto& encoder = metal::get_command_encoder(stream);

    if (transpose_rhs_) {
      constexpr int N = ring_size;
      constexpr int K = head_dim;
      constexpr int bm = 32;
      constexpr int bn = 32;
      constexpr int bk = 16;
      constexpr int wm = 2;
      constexpr int wn = 2;
      constexpr int tiles_m = M / bm;
      constexpr int tiles_n = N / bn;
      constexpr int partitions = 4;
      constexpr int partition_stride = M * N;
      constexpr int gemm_k_iterations = 8;
      constexpr int partition_size = gemm_k_iterations * bk;

      array split({rows, partitions, M, N}, float32, nullptr, {});
      split.set_data(allocator::malloc(split.nbytes()));

      const std::string kernel_name =
          "omlx_dspark_ring_scores_" + type_to_name(lhs);
      auto kernel = device.get_kernel(kernel_name, library);
      encoder.set_compute_pipeline_state(kernel);

      GEMMSpiltKParams params{
          M,
          N,
          K,
          K,
          head_dim,
          N,
          tiles_n,
          tiles_m,
          partitions,
          partition_stride,
          partition_size,
          0,
          gemm_k_iterations};

      encoder.set_input_array(lhs, 0);
      encoder.set_input_array(source, 1);
      encoder.set_input_array(indices, 2);
      encoder.set_output_array(split, 3);
      encoder.set_bytes(params, 4);
      encoder.set_bytes(batch_stride_a, 5);
      encoder.set_bytes(source_ld, 6);
      encoder.dispatch_threadgroups(
          MTL::Size(tiles_n, tiles_m, rows * partitions),
          MTL::Size(32, wn, wm));

      const std::string accum_name =
          "omlx_dspark_gemm_splitk_accum_" + type_to_name(output) +
          "_float32";
      auto accum_kernel = device.get_kernel(accum_name, library);
      encoder.set_compute_pipeline_state(accum_kernel);
      encoder.set_input_array(split, 0);
      encoder.set_output_array(output, 1);
      encoder.set_bytes(partitions, 2);
      encoder.set_bytes(partition_stride, 3);
      encoder.set_bytes(N, 4);
      encoder.dispatch_threads(
          MTL::Size(N, M, rows), MTL::Size(32, 8, 1));
      encoder.add_temporary(std::move(split));
      return;
    }

    constexpr int N = head_dim;
    constexpr int K = ring_size;
    constexpr int bm = 64;
    constexpr int bn = 64;
    constexpr int bk = 16;
    constexpr int wm = 1;
    constexpr int wn = 2;
    constexpr int tiles_m = M / bm;
    constexpr int tiles_n = N / bn;
    constexpr int gemm_k_iterations = K / bk;
    constexpr int64_t batch_stride_d = static_cast<int64_t>(M) * N;

    const std::string kernel_name =
        "omlx_dspark_ring_values_" + type_to_name(lhs);
    auto kernel = device.get_kernel(kernel_name, library);
    encoder.set_compute_pipeline_state(kernel);

    GEMMParams params{
        M,
        N,
        K,
        K,
        N,
        N,
        tiles_n,
        tiles_m,
        batch_stride_a,
        0,
        batch_stride_d,
        0,
        gemm_k_iterations,
        1};

    encoder.set_input_array(lhs, 0);
    encoder.set_input_array(source, 1);
    encoder.set_input_array(indices, 2);
    encoder.set_output_array(output, 3);
    encoder.set_bytes(params, 4);
    encoder.set_bytes(batch_stride_a, 5);
    encoder.set_bytes(source_ld, 6);
    encoder.dispatch_threadgroups(
        MTL::Size(tiles_n, tiles_m, rows), MTL::Size(32, wn, wm));
  }

  DEFINE_NAME(OMLXDSparkRingGemm)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    return transpose_rhs_ ==
        static_cast<const DSparkRingGemmPrimitive&>(other).transpose_rhs_;
  }
  auto state() const {
    return std::make_tuple(nullptr, transpose_rhs_);
  }

 private:
  bool transpose_rhs_;
};

} // namespace

mx::array dspark_rowwise_gemm(
    const mx::array& lhs,
    const mx::array& rhs,
    bool transpose_rhs,
    mx::StreamOrDevice s) {
  auto stream = to_stream(s);
  if (!metal::is_available() || stream.device == Device::cpu) {
    throw std::invalid_argument("DSpark rowwise GEMM requires Metal.");
  }
  if (lhs.ndim() != 3 || rhs.ndim() != 3 || lhs.shape(0) != rhs.shape(0) ||
      lhs.shape(1) < 1 || lhs.shape(1) > 64 || lhs.dtype() != rhs.dtype() ||
      (lhs.dtype() != float16 && lhs.dtype() != bfloat16 &&
       lhs.dtype() != float32) ||
      (lhs.dtype() == float32 && !transpose_rhs) ||
      !lhs.flags().row_contiguous || !rhs.flags().row_contiguous) {
    throw std::invalid_argument("Unsupported DSpark rowwise GEMM layout.");
  }
  const int K = lhs.shape(2);
  if ((transpose_rhs && rhs.shape(2) != K) ||
      (!transpose_rhs && rhs.shape(1) != K)) {
    throw std::invalid_argument("DSpark rowwise GEMM K dimensions differ.");
  }
  const int N = transpose_rhs ? rhs.shape(1) : rhs.shape(2);
  Shape shape{lhs.shape(0), lhs.shape(1), N};
  return array(
      std::move(shape),
      lhs.dtype(),
      std::make_shared<DSparkRowwiseGemmPrimitive>(stream, transpose_rhs),
      std::vector<array>{lhs, rhs});
}

mx::array dspark_ring_gemm(
    const mx::array& lhs,
    const mx::array& source,
    const mx::array& indices,
    bool transpose_rhs,
    mx::StreamOrDevice s) {
  auto stream = to_stream(s);
  if (!metal::is_available() || stream.device == Device::cpu) {
    throw std::invalid_argument("DSpark physical-ring GEMM requires Metal.");
  }
  if (lhs.ndim() != 3 || source.ndim() != 2 || indices.ndim() != 2 ||
      lhs.shape(0) != indices.shape(0) || lhs.shape(0) < 1 ||
      lhs.shape(0) > 6 || lhs.shape(1) != 64 || source.shape(0) < 128 ||
      source.shape(1) != 512 || indices.shape(1) != 128 ||
      lhs.dtype() != source.dtype() ||
      (lhs.dtype() != float16 && lhs.dtype() != bfloat16) ||
      indices.dtype() != uint32 || !lhs.flags().row_contiguous ||
      !source.flags().row_contiguous || !indices.flags().row_contiguous) {
    throw std::invalid_argument("Unsupported DSpark physical-ring GEMM layout.");
  }
  if ((transpose_rhs && lhs.shape(2) != 512) ||
      (!transpose_rhs && lhs.shape(2) != 128)) {
    throw std::invalid_argument("Unsupported DSpark physical-ring GEMM shape.");
  }

  Shape shape{lhs.shape(0), lhs.shape(1), transpose_rhs ? 128 : 512};
  return array(
      std::move(shape),
      lhs.dtype(),
      std::make_shared<DSparkRingGemmPrimitive>(stream, transpose_rhs),
      std::vector<array>{lhs, source, indices});
}

} // namespace omlx::glm_kernels
