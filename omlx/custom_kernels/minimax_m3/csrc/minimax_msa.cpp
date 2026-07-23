#include "minimax_msa.h"

#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::minimax_m3_kernels {

namespace {

using namespace mlx::core;

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error(
          "Unable to get omlx_minimax_m3_kernels binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

bool row_contiguous(const array& arr) {
  return arr.flags().row_contiguous && arr.strides(-1) == 1 &&
      arr.offset() == 0;
}

array ensure_row_contiguous(const array& arr, Stream stream) {
  return contiguous(arr, false, stream);
}

class MinimaxMSATopKPrimitive : public Primitive {
 public:
  MinimaxMSATopKPrimitive(
      Stream stream,
      int q_start,
      float scale,
      int block_size,
      int topk,
      int init_blocks,
      int local_blocks)
      : Primitive(stream),
        q_start_(q_start),
        scale_(scale),
        block_size_(block_size),
        topk_(topk),
        init_blocks_(init_blocks),
        local_blocks_(local_blocks) {}

  static bool unsupported(
      const array& idx_queries,
      const array& idx_keys,
      int block_size,
      int topk,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (idx_queries.dtype() != idx_keys.dtype()) {
      return true;
    }
    if (idx_queries.dtype() != float32 && idx_queries.dtype() != float16 &&
        idx_queries.dtype() != bfloat16) {
      return true;
    }
    if (idx_queries.ndim() != 4 || idx_keys.ndim() != 4 ||
        idx_keys.shape(1) != 1) {
      return true;
    }
    if (!row_contiguous(idx_queries) || !row_contiguous(idx_keys)) {
      return true;
    }
    if (idx_queries.shape(0) != idx_keys.shape(0) ||
        idx_queries.shape(3) != idx_keys.shape(3)) {
      return true;
    }

    return block_size != 128 || topk != 16 || idx_queries.shape(3) != 128 ||
        idx_queries.shape(2) <= 0 || idx_keys.shape(2) <= 0;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("MinimaxMSATopKPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& q = inputs[0];
    auto& k = inputs[1];
    auto& out = outputs[0];

    constexpr int bm = 64;
    constexpr int bn = 128;
    constexpr int bk = 16;
    constexpr int wm = 2;
    constexpr int wn = 2;
    constexpr int topk_threads = 256;

    const int B = q.shape(0);
    const int H = q.shape(1);
    const int M = q.shape(2);
    const int N = k.shape(2);
    const int D = q.shape(3);
    const int num_blocks = (N + block_size_ - 1) / block_size_;
    const int tiles_m = (M + bm - 1) / bm;

    mlx::steel::GEMMParams params{
        /* const int M = */ M,
        /* const int N = */ N,
        /* const int K = */ D,
        /* const int lda = */ D,
        /* const int ldb = */ D,
        /* const int ldd = */ num_blocks,
        /* const int tiles_n = */ num_blocks,
        /* const int tiles_m = */ tiles_m,
        /* const int64_t batch_stride_a = */ int64_t(H) * M * D,
        /* const int64_t batch_stride_b = */ int64_t(N) * D,
        /* const int64_t batch_stride_d = */ int64_t(H) * M * num_blocks,
        /* const int swizzle_log = */ 0,
        /* const int gemm_k_iterations_aligned = */ D / bk,
        /* const int batch_ndim = */ 1};

    array block_scores({B, H, M, num_blocks}, float32, nullptr, {});
    block_scores.set_data(allocator::malloc(block_scores.nbytes()));
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.add_temporary(block_scores);

    out.set_data(allocator::malloc(out.nbytes()));

    std::string block_kernel_name;
    concatenate(
        block_kernel_name,
        "minimax_msa_block_scores_",
        type_to_name(q),
        "_bm",
        bm,
        "_bn",
        bn,
        "_bk",
        bk,
        "_wm",
        wm,
        "_wn",
        wn);

    auto lib = d.get_library("omlx_minimax_m3_kernels", current_binary_dir());
    auto block_kernel = d.get_kernel(block_kernel_name, lib);
    compute_encoder.set_compute_pipeline_state(block_kernel);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_output_array(block_scores, 2);
    compute_encoder.set_bytes(params, 3);
    compute_encoder.set_bytes(H, 4);
    compute_encoder.set_bytes(q_start_, 5);
    compute_encoder.set_bytes(block_size_, 6);
    compute_encoder.set_bytes(scale_, 7);
    compute_encoder.set_bytes(num_blocks, 8);

    MTL::Size block_grid(num_blocks, tiles_m, B * H);
    MTL::Size block_group(wm * wn * 32, 1, 1);
    compute_encoder.dispatch_threadgroups(block_grid, block_group);

    auto topk_kernel =
        d.get_kernel("minimax_msa_topk_select_topk16_t256", lib);
    compute_encoder.set_compute_pipeline_state(topk_kernel);
    compute_encoder.set_input_array(block_scores, 0);
    compute_encoder.set_output_array(out, 1);
    const int rows = B * H * M;
    compute_encoder.set_bytes(rows, 2);
    compute_encoder.set_bytes(H, 3);
    compute_encoder.set_bytes(M, 4);
    compute_encoder.set_bytes(num_blocks, 5);
    compute_encoder.set_bytes(q_start_, 6);
    compute_encoder.set_bytes(block_size_, 7);
    compute_encoder.set_bytes(init_blocks_, 8);
    compute_encoder.set_bytes(local_blocks_, 9);

    MTL::Size topk_grid(rows, 1, 1);
    MTL::Size topk_group(topk_threads, 1, 1);
    compute_encoder.dispatch_threadgroups(topk_grid, topk_group);
  }

  DEFINE_NAME(OMLXMinimaxMSATopK)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const MinimaxMSATopKPrimitive&>(other);
    return q_start_ == rhs.q_start_ && scale_ == rhs.scale_ &&
        block_size_ == rhs.block_size_ && topk_ == rhs.topk_ &&
        init_blocks_ == rhs.init_blocks_ && local_blocks_ == rhs.local_blocks_;
  }
  auto state() const {
    return std::make_tuple(
        q_start_, scale_, block_size_, topk_, init_blocks_, local_blocks_);
  }

 private:
  int q_start_;
  float scale_;
  int block_size_;
  int topk_;
  int init_blocks_;
  int local_blocks_;
};

} // namespace

array minimax_msa_topk(
    const array& idx_queries,
    const array& idx_keys,
    int q_start,
    float scale,
    int block_size,
    int topk,
    int init_blocks,
    int local_blocks,
    StreamOrDevice s) {
  if (idx_queries.ndim() != 4 || idx_keys.ndim() != 4) {
    std::ostringstream msg;
    msg << "[omlx_minimax_m3.minimax_msa_topk] expected rank-4 idx "
        << "query/key arrays, got " << idx_queries.shape() << " and "
        << idx_keys.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (idx_queries.shape(0) != idx_keys.shape(0) || idx_keys.shape(1) != 1 ||
      idx_queries.shape(3) != idx_keys.shape(3)) {
    std::ostringstream msg;
    msg << "[omlx_minimax_m3.minimax_msa_topk] incompatible idx query/key "
        << "shapes " << idx_queries.shape() << " and " << idx_keys.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (block_size <= 0 || topk <= 0 || init_blocks < 0 || local_blocks < 0) {
    throw std::invalid_argument(
        "[omlx_minimax_m3.minimax_msa_topk] block_size/topk must be "
        "positive and forced block counts must be non-negative.");
  }

  auto stream = to_stream(s);
  auto final_type = result_type(idx_queries, idx_keys);
  if (final_type != float32 && final_type != float16 && final_type != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_minimax_m3.minimax_msa_topk] expected floating idx "
        << "query/key arrays, got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto q = ensure_row_contiguous(astype(idx_queries, final_type, stream), stream);
  auto k = ensure_row_contiguous(astype(idx_keys, final_type, stream), stream);
  std::vector<array> inputs = {q, k};
  if (MinimaxMSATopKPrimitive::unsupported(q, k, block_size, topk, stream)) {
    throw std::invalid_argument(
        "[omlx_minimax_m3.minimax_msa_topk] unsupported MiniMax M3 MSA "
        "top-k shape.");
  }

  Shape out_shape{q.shape(0), q.shape(1), q.shape(2), topk};
  return array(
      std::move(out_shape),
      int32,
      std::make_shared<MinimaxMSATopKPrimitive>(
          stream,
          q_start,
          scale,
          block_size,
          topk,
          init_blocks,
          local_blocks),
      std::move(inputs));
}

} // namespace omlx::minimax_m3_kernels
