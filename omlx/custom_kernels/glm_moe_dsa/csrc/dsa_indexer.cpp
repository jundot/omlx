#include "dsa_indexer.h"

#include "kernels/mma_dsa_indexer_score.h"

#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
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

struct DSATopKParams {
  int rows;
  int L;
  int K;
  int topk;
  bool causal_valid_prefix;
};

bool row_contiguous(const array& arr) {
  return arr.flags().row_contiguous && arr.strides(-1) == 1 &&
      arr.offset() == 0;
}

array ensure_row_contiguous(const array& arr, Stream stream) {
  return contiguous(arr, false, stream);
}

class DSAIndexerScoresPrimitive : public Primitive {
 public:
  DSAIndexerScoresPrimitive(
      Stream stream,
      bool causal,
      bool weights_lh,
      int unused_causal_prefix_topk,
      bool skip_causal_future_store,
      int causal_q_offset,
      int mask_ratio,
      int mask_q_offset)
      : Primitive(stream),
        causal_(causal),
        weights_lh_(weights_lh),
        unused_causal_prefix_topk_(unused_causal_prefix_topk),
        skip_causal_future_store_(skip_causal_future_store),
        causal_q_offset_(causal_q_offset),
        mask_ratio_(mask_ratio),
        mask_q_offset_(mask_q_offset) {}

  static bool unsupported(
      const array& q,
      const array& k,
      const array& weights,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (q.dtype() != k.dtype() || q.dtype() != weights.dtype()) {
      return true;
    }
    if (q.dtype() != float16 && q.dtype() != bfloat16) {
      return true;
    }
    if (!row_contiguous(q) || !row_contiguous(k) ||
        !row_contiguous(weights)) {
      return true;
    }
    if (q.ndim() != 4 || k.ndim() != 4 ||
        (weights.ndim() != 3 && weights.ndim() != 4)) {
      return true;
    }
    const bool weights_lh = weights.ndim() == 3;
    if ((q.shape(1) != 32 && q.shape(1) != 64) || k.shape(1) != 1) {
      return true;
    }
    if (weights_lh) {
      if (weights.shape(1) != q.shape(2) || weights.shape(2) != q.shape(1)) {
        return true;
      }
    } else {
      if (weights.shape(1) != q.shape(1) || weights.shape(2) != q.shape(2) ||
          weights.shape(3) != 1) {
        return true;
      }
    }
    if (q.shape(3) != 128 || k.shape(3) != 128) {
      return true;
    }
    if (q.shape(3) % 16 != 0) {
      return true;
    }
    return k.shape(2) < 64;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("DSAIndexerScoresPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& weights = inputs[2];

    out.set_data(allocator::malloc(out.nbytes()));

    constexpr int bm = 64;
    constexpr int bk = 16;

    const int B = q.shape(0);
    const int H = q.shape(1);
    const int M = q.shape(2);
    const int N = k.shape(2);
    const int D = q.shape(3);

    // bm/bn/wm/wn do not enter the per-element K-reduction order (bk=16 and
    // the MMA fragment K-layout are unchanged), so the tile config only
    // affects scheduling. bn=128 (paired with wm=2,wn=4) was measured on
    // M3 Ultra (L=2048, bf16, H=64): it ties bn=64 at P=25k but is ~9%
    // slower at P=125k and ~11% slower at P=2.5k — the kernel is
    // compute/barrier-bound (Q and pooled-K panels are largely
    // L2-resident), so the traffic reduction does not pay. bn=64/wm2/wn2
    // is the fixed configuration.
    const int bn = 64;
    const int wm = 2;
    const int wn = 2;

    const int tiles_m = (M + bm - 1) / bm;
    const int tiles_n = (N + bn - 1) / bn;

    mlx::steel::GEMMParams params{
        /* const int M = */ M,
        /* const int N = */ N,
        /* const int K = */ D,
        /* const int lda = */ D,
        /* const int ldb = */ D,
        /* const int ldd = */ N,
        /* const int tiles_n = */ tiles_n,
        /* const int tiles_m = */ tiles_m,
        /* const int64_t batch_stride_a = */ int64_t(H) * M * D,
        /* const int64_t batch_stride_b = */ int64_t(N) * D,
        /* const int64_t batch_stride_d = */ int64_t(M) * N,
        /* const int swizzle_log = */ 0,
        /* const int gemm_k_iterations_aligned = */ D / bk,
        /* const int batch_ndim = */ 1};

    bool do_causal = causal_;
    bool use_weights_lh = weights_lh_;
    metal::MTLFCList func_consts = {
        {&do_causal, MTL::DataType::DataTypeBool, 300},
        {&use_weights_lh, MTL::DataType::DataTypeBool, 301},
    };

    std::string base_name;
    concatenate(
        base_name,
        "steel_dsa_indexer_score_",
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

    std::string hash_name;
    concatenate(
        hash_name,
        base_name,
        "_causal_",
        (do_causal ? 't' : 'n'),
        "_wlh_",
        (use_weights_lh ? 't' : 'n'));

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto& compute_encoder = metal::get_command_encoder(s);
    auto kernel = d.get_kernel(base_name, lib, hash_name, func_consts);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(weights, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(params, 4);
    compute_encoder.set_bytes(H, 5);
    compute_encoder.set_bytes(unused_causal_prefix_topk_, 6);
    compute_encoder.set_bytes(skip_causal_future_store_, 7);
    compute_encoder.set_bytes(causal_q_offset_, 8);
    compute_encoder.set_bytes(mask_ratio_, 9);
    compute_encoder.set_bytes(mask_q_offset_, 10);

    MTL::Size group_dims = MTL::Size(wm * wn * 32, 1, 1);
    MTL::Size grid_dims = MTL::Size(tiles_n, tiles_m, B);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(OMLXDSAIndexerScores)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const DSAIndexerScoresPrimitive&>(other);
    return causal_ == rhs.causal_ && weights_lh_ == rhs.weights_lh_ &&
        unused_causal_prefix_topk_ == rhs.unused_causal_prefix_topk_ &&
        skip_causal_future_store_ == rhs.skip_causal_future_store_ &&
        causal_q_offset_ == rhs.causal_q_offset_ &&
        mask_ratio_ == rhs.mask_ratio_ &&
        mask_q_offset_ == rhs.mask_q_offset_;
  }
  auto state() const {
    return std::make_tuple(
        causal_,
        weights_lh_,
        unused_causal_prefix_topk_,
        skip_causal_future_store_,
        causal_q_offset_,
        mask_ratio_,
        mask_q_offset_);
  }

 private:
  bool causal_;
  bool weights_lh_;
  int unused_causal_prefix_topk_;
  bool skip_causal_future_store_;
  int causal_q_offset_;
  int mask_ratio_;
  int mask_q_offset_;
};

class Qwen4QSAIndexerScoresPrimitive : public Primitive {
 public:
  Qwen4QSAIndexerScoresPrimitive(
      Stream stream,
      int mask_ratio,
      int mask_q_offset)
      : Primitive(stream),
        mask_ratio_(mask_ratio),
        mask_q_offset_(mask_q_offset) {}

  static bool unsupported(const array& q, const array& k, Stream s) {
    if (s.device == Device::cpu || q.dtype() != k.dtype()) {
      return true;
    }
    if (q.dtype() != float16 && q.dtype() != bfloat16) {
      return true;
    }
    if (!row_contiguous(q) || !row_contiguous(k)) {
      return true;
    }
    if (q.ndim() != 4 || k.ndim() != 4) {
      return true;
    }
    // This is intentionally the production Qwen4-Exp geometry only. A stale
    // config or generalized caller must stay on qsa_fast's fp32 MLX path.
    return q.shape(0) != 1 || k.shape(0) != 1 || q.shape(1) != 4 ||
        k.shape(1) != 1 || q.shape(2) <= 0 || k.shape(2) <= 0 ||
        q.shape(3) != 128 || k.shape(3) != 128;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "Qwen4QSAIndexerScoresPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];
    const auto& q = inputs[0];
    const auto& k = inputs[1];

    out.set_data(allocator::malloc(out.nbytes()));

    constexpr int bm = 64;
    constexpr int bn = 64;
    constexpr int bk = 16;
    constexpr int wm = 2;
    constexpr int wn = 2;
    const int B = q.shape(0);
    const int H = q.shape(1);
    const int M = q.shape(2);
    const int N = k.shape(2);
    const int D = q.shape(3);
    const int tiles_m = (M + bm - 1) / bm;
    const int tiles_n = (N + bn - 1) / bn;

    mlx::steel::GEMMParams params{
        /* const int M = */ M,
        /* const int N = */ N,
        /* const int K = */ D,
        /* const int lda = */ D,
        /* const int ldb = */ D,
        /* const int ldd = */ N,
        /* const int tiles_n = */ tiles_n,
        /* const int tiles_m = */ tiles_m,
        /* const int64_t batch_stride_a = */ int64_t(H) * M * D,
        /* const int64_t batch_stride_b = */ int64_t(N) * D,
        /* const int64_t batch_stride_d = */ int64_t(M) * N,
        /* const int swizzle_log = */ 0,
        /* const int gemm_k_iterations_aligned = */ D / bk,
        /* const int batch_ndim = */ 1};

    std::string base_name;
    concatenate(
        base_name,
        "qwen4_qsa_indexer_score_",
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

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = d.get_kernel(base_name, lib);
    auto& encoder = metal::get_command_encoder(s);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_output_array(out, 2);
    encoder.set_bytes(params, 3);
    encoder.set_bytes(mask_ratio_, 4);
    encoder.set_bytes(mask_q_offset_, 5);
    const float score_divisor = std::sqrt(static_cast<float>(D));
    encoder.set_bytes(score_divisor, 6);
    encoder.dispatch_threadgroups(
        MTL::Size(tiles_n, tiles_m, B),
        MTL::Size(wm * wn * 32, 1, 1));
  }

  DEFINE_NAME(OMLXQwen4QSAIndexerScores)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const Qwen4QSAIndexerScoresPrimitive&>(other);
    return mask_ratio_ == rhs.mask_ratio_ &&
        mask_q_offset_ == rhs.mask_q_offset_;
  }
  auto state() const {
    return std::make_tuple(mask_ratio_, mask_q_offset_);
  }

 private:
  int mask_ratio_;
  int mask_q_offset_;
};

// ── v25 M2 MMA score kernel (mma_dsa_indexer_score.h) ───────────────────────
// Split dispatch: the interior instantiation runs the unmodified hot loop on
// fully-interior tiles; the boundary instantiation handles partial edge tiles
// with clamped loads. Both write disjoint regions of ONE output allocation.
// M/N/mask offsets are runtime params — a recompile per chunk would otherwise
// stall prefill (N grows and mask_q_offset changes every chunk).
// The kernel source is compiled ONCE at runtime by the macOS Metal compiler
// (get_library builder path) instead of shipping in the metallib: the Xcode
// CLI toolchain's codegen for this kernel measures 3.4 %-points slower (see
// the header comment).
class MMADSAIndexerScoresPrimitive : public Primitive {
 public:
  static constexpr int kBM = 64;
  static constexpr int kBN = 64;
  static constexpr int kThreads = 128; // WM=2, WN=2
  static constexpr int kSwizzleLog = 2;

  MMADSAIndexerScoresPrimitive(Stream stream, int mask_ratio, int mask_q_offset)
      : Primitive(stream),
        mask_ratio_(mask_ratio),
        mask_q_offset_(mask_q_offset) {}

  static bool unsupported(
      const array& q,
      const array& k,
      const array& weights,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    // The kernel is instantiated for bf16 / H=64 / D=128 / weights-LH only.
    if (q.dtype() != bfloat16 || k.dtype() != bfloat16 ||
        weights.dtype() != bfloat16) {
      return true;
    }
    if (!row_contiguous(q) || !row_contiguous(k) ||
        !row_contiguous(weights)) {
      return true;
    }
    if (q.ndim() != 4 || k.ndim() != 4 || weights.ndim() != 3) {
      return true;
    }
    if (q.shape(1) != 64 || k.shape(1) != 1) {
      return true;
    }
    if (weights.shape(1) != q.shape(2) || weights.shape(2) != q.shape(1)) {
      return true;
    }
    if (q.shape(3) != 128 || k.shape(3) != 128) {
      return true;
    }
    return k.shape(2) < 64;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("MMADSAIndexerScoresPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& weights = inputs[2];

    out.set_data(allocator::malloc(out.nbytes()));

    const int B = q.shape(0);
    const int M = q.shape(2);
    const int N = k.shape(2);

    const int tiles_m_full = M / kBM;
    const int tiles_n_full = N / kBN;
    const int tiles_m = (M + kBM - 1) / kBM;
    const int tiles_n = (N + kBN - 1) / kBN;

    OMLXMMADSAScoreParamsHost params{
        /* int M = */ M,
        /* int N = */ N,
        /* int mask_ratio = */ mask_ratio_,
        /* int mask_q_offset = */ mask_q_offset_};

    // Swizzled threadgroup grid identical to the measured harness form.
    const int tg_x = tiles_n << kSwizzleLog;
    const int tg_y =
        (tiles_m + (1 << kSwizzleLog) - 1) >> kSwizzleLog;
    MTL::Size group_dims = MTL::Size(kThreads, 1, 1);
    MTL::Size grid_dims = MTL::Size(tg_x, tg_y, B);

    auto lib = d.get_library("omlx_glm_mma_dsa_v25", []() {
      return std::string(kMMADSAScoreKernelSource);
    });
    auto& compute_encoder = metal::get_command_encoder(s);

    auto dispatch = [&](const char* name) {
      auto kernel = d.get_kernel(name, lib);
      compute_encoder.set_compute_pipeline_state(kernel);
      compute_encoder.set_input_array(q, 0);
      compute_encoder.set_input_array(k, 1);
      compute_encoder.set_input_array(weights, 2);
      compute_encoder.set_output_array(out, 3);
      compute_encoder.set_bytes(params, 4);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    };

    if (tiles_m_full > 0 && tiles_n_full > 0) {
      dispatch("mma_dsa_indexer_score_bfloat16_interior");
    }
    if (tiles_m > tiles_m_full || tiles_n > tiles_n_full) {
      dispatch("mma_dsa_indexer_score_bfloat16_boundary");
    }
  }

  DEFINE_NAME(OMLXMMADSAIndexerScores)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const MMADSAIndexerScoresPrimitive&>(other);
    return mask_ratio_ == rhs.mask_ratio_ &&
        mask_q_offset_ == rhs.mask_q_offset_;
  }
  auto state() const {
    return std::make_tuple(mask_ratio_, mask_q_offset_);
  }

 private:
  int mask_ratio_;
  int mask_q_offset_;
};

class DSATopKIndicesPrimitive : public Primitive {
 public:
  DSATopKIndicesPrimitive(
      Stream stream,
      int topk,
      bool bucketed,
      bool causal_valid_prefix)
      : Primitive(stream),
        topk_(topk),
        bucketed_(bucketed),
        causal_valid_prefix_(causal_valid_prefix) {}

  static bool unsupported(const array& scores, int topk, Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (scores.dtype() != float16 && scores.dtype() != bfloat16) {
      return true;
    }
    if (!row_contiguous(scores)) {
      return true;
    }
    if (scores.ndim() != 4 || scores.shape(1) != 1) {
      return true;
    }
    if (topk != 512 && topk != 2048) {
      return true;
    }
    return scores.shape(-1) < topk;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("DSATopKIndicesPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& scores = inputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    constexpr int threads = 1024;

    const int B = scores.shape(0);
    const int L = scores.shape(2);
    const int K = scores.shape(3);
    const int rows = B * L;

    std::string base_name;
    concatenate(
        base_name,
        "steel_dsa_topk_indices_",
        type_to_name(scores),
        "_topk",
        topk_,
        "_t",
        threads);

    bool bucketed = bucketed_;
    metal::MTLFCList func_consts = {
        {&bucketed, MTL::DataType::DataTypeBool, 302},
    };

    std::string hash_name;
    concatenate(
        hash_name,
        base_name,
        "_bucketed_",
        (bucketed ? 't' : 'n'));

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto& compute_encoder = metal::get_command_encoder(s);
    auto kernel = d.get_kernel(base_name, lib, hash_name, func_consts);
    compute_encoder.set_compute_pipeline_state(kernel);

    DSATopKParams params{
        /* int rows = */ rows,
        /* int L = */ L,
        /* int K = */ K,
        /* int topk = */ topk_,
        /* bool causal_valid_prefix = */ causal_valid_prefix_};

    compute_encoder.set_input_array(scores, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_bytes(params, 2);

    MTL::Size group_dims = MTL::Size(threads, 1, 1);
    MTL::Size grid_dims = MTL::Size(rows, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(OMLXDSATopKIndices)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const DSATopKIndicesPrimitive&>(other);
    return topk_ == rhs.topk_ && bucketed_ == rhs.bucketed_ &&
        causal_valid_prefix_ == rhs.causal_valid_prefix_;
  }
  auto state() const {
    return std::make_tuple(topk_, bucketed_, causal_valid_prefix_);
  }

 private:
  int topk_;
  bool bucketed_;
  bool causal_valid_prefix_;
};

class DSparkFP32TopKIndicesPrimitive : public Primitive {
 public:
  explicit DSparkFP32TopKIndicesPrimitive(Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("DSpark FP32 top-k has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    const auto& scores = inputs[0];
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    constexpr int topk = 512;
    constexpr int threads = 256;
    const int rows = scores.shape(0);
    DSATopKParams params{
        /* int rows = */ rows,
        /* int L = */ 1,
        /* int K = */ scores.shape(1),
        /* int topk = */ topk,
        /* bool causal_valid_prefix = */ false};

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel =
        d.get_kernel("dspark_fp32_topk_indices_topk512_t256", lib);
    auto& encoder = metal::get_command_encoder(s);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(scores, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(params, 2);
    encoder.dispatch_threadgroups(
        MTL::Size(rows, 1, 1), MTL::Size(threads, 1, 1));
  }

  DEFINE_NAME(OMLXDSparkFP32TopKIndices)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }
};

class Qwen4QSAFP32TopKIndicesPrimitive : public Primitive {
 public:
  explicit Qwen4QSAFP32TopKIndicesPrimitive(Stream stream)
      : Primitive(stream) {}

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("Qwen4 QSA FP32 top-k has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    const auto& scores = inputs[0];
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    constexpr int topk = 512;
    constexpr int threads = 256;
    const int rows = scores.shape(1);
    DSATopKParams params{
        /* int rows = */ rows,
        /* int L = */ rows,
        /* int K = */ scores.shape(2),
        /* int topk = */ topk,
        /* bool causal_valid_prefix = */ false};

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel =
        d.get_kernel("qwen4_qsa_fp32_topk_indices_topk512_t256", lib);
    auto& encoder = metal::get_command_encoder(s);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(scores, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(params, 2);
    encoder.dispatch_threadgroups(
        MTL::Size(rows, 1, 1), MTL::Size(threads, 1, 1));
  }

  DEFINE_NAME(OMLXQwen4QSAFP32TopKIndices)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }
};
// ── DC-1: fused decode indexer scan ─────────────────────────────────────────
// One kernel computes the head-summed indexer scores for a single query position
// (s == 1) directly into [B,1,1,S] with fp32 accumulation, replacing the decode
// chain q@k^T -> relu -> *w -> head-sum that materializes four S-sized tensors
// per layer per token. K is addressed by STRIDES: capacity-backed cache slices
// are consumed in place (no ensure_row_contiguous copy). Scores come out in the
// input dtype by default (feeding the native 16-bit radix top-k) or fp32 when
// fp32_scores is set (selection then matches fp32 ground truth exactly).
struct OMLXDSADecodeParamsHost {
  int S;
  int64_t k_batch_stride;
  int64_t k_row_stride;
};

class DSADecodeScoresPrimitive : public Primitive {
 public:
  DSADecodeScoresPrimitive(Stream stream, bool fp32_scores)
      : Primitive(stream), fp32_scores_(fp32_scores) {}

  static bool unsupported(
      const array& q,
      const array& k,
      const array& w,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (q.dtype() != k.dtype() || q.dtype() != w.dtype()) {
      return true;
    }
    if (q.dtype() != float16 && q.dtype() != bfloat16) {
      return true;
    }
    if (!row_contiguous(q) || !row_contiguous(w)) {
      return true;
    }
    if (q.ndim() != 4 || k.ndim() != 4 || w.ndim() != 2) {
      return true;
    }
    // q [B,32,1,128] contiguous; k [B,1,S,128] with contiguous rows only
    // (capacity-backed slices allowed); rows must stay 16B-aligned for the
    // vec4 loads: row stride % 8 elements == 0.
    if (q.shape(1) != 32 || q.shape(2) != 1 || q.shape(3) != 128) {
      return true;
    }
    if (k.shape(0) != q.shape(0) || k.shape(1) != 1 || k.shape(3) != 128) {
      return true;
    }
    if (k.strides(3) != 1 || (k.strides(2) % 8) != 0) {
      return true;
    }
    if (w.shape(0) != q.shape(0) || w.shape(1) != 32) {
      return true;
    }
    return k.shape(2) < 1024;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("DSADecodeScoresPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& w = inputs[2];

    out.set_data(allocator::malloc(out.nbytes()));

    constexpr int threads = 256;
    const int B = q.shape(0);
    const int S = k.shape(2);
    const int blocks = (S + threads - 1) / threads;

    std::string base_name;
    concatenate(
        base_name,
        "dsa_decode_scores_",
        type_to_name(q),
        fp32_scores_ ? "_of32" : "_osame",
        "_h32_d128_t",
        threads);

    OMLXDSADecodeParamsHost params{
        /* int S = */ S,
        /* int64_t k_batch_stride = */ k.shape(0) == 1 ? 0 : k.strides(0),
        /* int64_t k_row_stride = */ k.strides(2)};

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto& compute_encoder = metal::get_command_encoder(s);
    auto kernel = d.get_kernel(base_name, lib);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(w, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(params, 4);

    MTL::Size group_dims = MTL::Size(threads, 1, 1);
    MTL::Size grid_dims = MTL::Size(blocks, 1, B);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(OMLXDSADecodeScores)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const DSADecodeScoresPrimitive&>(other);
    return fp32_scores_ == rhs.fp32_scores_;
  }
  auto state() const {
    return std::make_tuple(fp32_scores_);
  }

 private:
  bool fp32_scores_;
};

array dsa_topk_indices_impl(
    const array& scores,
    int topk,
    bool bucketed,
    bool causal_valid_prefix,
    StreamOrDevice s) {
  if (scores.ndim() != 4 || scores.shape(1) != 1) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_topk_indices] expected scores with shape "
        << "[B, 1, L, K], got " << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (topk <= 0 || topk > scores.shape(-1)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_topk_indices] invalid topk " << topk
        << " for scores with shape " << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  auto scores_contiguous = ensure_row_contiguous(scores, stream);
  std::vector<array> inputs = {scores_contiguous};
  if (DSATopKIndicesPrimitive::unsupported(scores_contiguous, topk, stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.dsa_topk_indices] unsupported M3 GLM shape.");
  }

  Shape out_shape{
      scores_contiguous.shape(0), 1, scores_contiguous.shape(2), topk};
  return array(
      std::move(out_shape),
      uint32,
      std::make_shared<DSATopKIndicesPrimitive>(
          stream, topk, bucketed, causal_valid_prefix),
      std::move(inputs));
}

} // namespace

array dsa_indexer_scores(
    const array& queries,
    const array& keys,
    const array& weights,
    bool causal,
    int unused_causal_prefix_topk,
    bool skip_causal_future_store,
    int causal_q_offset,
    int mask_ratio,
    int mask_q_offset,
    StreamOrDevice s) {
  if (queries.ndim() != 4 || keys.ndim() != 4 ||
      (weights.ndim() != 3 && weights.ndim() != 4)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_indexer_scores] expected q/k rank 4 and "
        << "weights rank 3 or 4, got " << queries.shape() << ", "
        << keys.shape() << ", " << weights.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (keys.shape(1) != 1) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.dsa_indexer_scores] keys must have a singleton "
        "indexer head axis.");
  }
  const bool weights_lh = weights.ndim() == 3;
  bool weights_match = false;
  if (weights_lh) {
    weights_match = weights.shape(1) == queries.shape(2) &&
        weights.shape(2) == queries.shape(1);
  } else {
    weights_match = weights.shape(1) == queries.shape(1) &&
        weights.shape(2) == queries.shape(2) && weights.shape(3) == 1;
  }
  if (queries.shape(0) != keys.shape(0) ||
      queries.shape(0) != weights.shape(0) || !weights_match ||
      queries.shape(3) != keys.shape(3)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_indexer_scores] incompatible q, k, "
        << "weights shapes: " << queries.shape() << ", " << keys.shape()
        << ", " << weights.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto final_type = result_type(queries, keys, weights);
  if (final_type != float16 && final_type != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_indexer_scores] expected float16 or "
        << "bfloat16 inputs, got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }
  if (unused_causal_prefix_topk < 0) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_indexer_scores] "
        << "unused_causal_prefix_topk must be non-negative, got "
        << unused_causal_prefix_topk << ".";
    throw std::invalid_argument(msg.str());
  }
  if (causal_q_offset < -1) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_indexer_scores] causal_q_offset must be "
        << "-1 or non-negative, got " << causal_q_offset << ".";
    throw std::invalid_argument(msg.str());
  }
  if (mask_ratio < 0) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_indexer_scores] mask_ratio must be "
        << "non-negative (0 disables the fused pooled-causal mask), got "
        << mask_ratio << ".";
    throw std::invalid_argument(msg.str());
  }
  if (mask_ratio > 0 && mask_q_offset < 0) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_indexer_scores] mask_q_offset must be "
        << "non-negative when mask_ratio > 0, got " << mask_q_offset << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  auto q = ensure_row_contiguous(astype(queries, final_type, stream), stream);
  auto k = ensure_row_contiguous(astype(keys, final_type, stream), stream);
  auto w = ensure_row_contiguous(astype(weights, final_type, stream), stream);

  std::vector<array> inputs = {q, k, w};
  if (DSAIndexerScoresPrimitive::unsupported(q, k, w, stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.dsa_indexer_scores] unsupported M3 GLM shape.");
  }

  Shape out_shape{q.shape(0), 1, q.shape(2), k.shape(2)};
  return array(
      std::move(out_shape),
      final_type,
      std::make_shared<DSAIndexerScoresPrimitive>(
          stream,
          causal,
          weights_lh,
          unused_causal_prefix_topk,
          skip_causal_future_store,
          causal_q_offset,
          mask_ratio,
          mask_q_offset),
      std::move(inputs));
}

array qwen4_qsa_indexer_scores(
    const array& queries,
    const array& pooled_keys,
    int mask_ratio,
    int mask_q_offset,
    StreamOrDevice s) {
  if (queries.ndim() != 4 || pooled_keys.ndim() != 4) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.qwen4_qsa_indexer_scores] expected q/k rank "
        << "4, got " << queries.shape() << " and " << pooled_keys.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (queries.dtype() != pooled_keys.dtype() ||
      (queries.dtype() != float16 && queries.dtype() != bfloat16)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.qwen4_qsa_indexer_scores] q/k must have matching "
        "float16 or bfloat16 dtype.");
  }
  if (mask_ratio != 4 || mask_q_offset < 0) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.qwen4_qsa_indexer_scores] requires "
        << "mask_ratio=4 and a non-negative mask_q_offset, got "
        << mask_ratio << " and " << mask_q_offset << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  auto q = ensure_row_contiguous(queries, stream);
  auto k = ensure_row_contiguous(pooled_keys, stream);
  if (Qwen4QSAIndexerScoresPrimitive::unsupported(q, k, stream)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.qwen4_qsa_indexer_scores] unsupported shape "
        << "or layout; expected q [1,4,M,128] and k [1,1,N,128], got "
        << q.shape() << " and " << k.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  Shape out_shape{1, q.shape(2), k.shape(2)};
  return array(
      std::move(out_shape),
      float32,
      std::make_shared<Qwen4QSAIndexerScoresPrimitive>(
          stream, mask_ratio, mask_q_offset),
      std::vector<array>{q, k});
}

array dsa_indexer_scores_mma(
    const array& queries,
    const array& keys,
    const array& weights,
    int mask_ratio,
    int mask_q_offset,
    StreamOrDevice s) {
  if (queries.ndim() != 4 || keys.ndim() != 4 || weights.ndim() != 3) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_indexer_scores_mma] expected q/k rank 4 "
        << "and weights rank 3 ([B, L, H]), got " << queries.shape() << ", "
        << keys.shape() << ", " << weights.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (keys.shape(1) != 1) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.dsa_indexer_scores_mma] keys must have a "
        "singleton indexer head axis.");
  }
  if (mask_ratio < 0) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_indexer_scores_mma] mask_ratio must be "
        << "non-negative (0 disables the fused pooled-causal mask), got "
        << mask_ratio << ".";
    throw std::invalid_argument(msg.str());
  }
  if (mask_ratio > 0 && mask_q_offset < 0) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_indexer_scores_mma] mask_q_offset must "
        << "be non-negative when mask_ratio > 0, got " << mask_q_offset
        << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  auto q = ensure_row_contiguous(queries, stream);
  auto k = ensure_row_contiguous(keys, stream);
  auto w = ensure_row_contiguous(weights, stream);

  std::vector<array> inputs = {q, k, w};
  if (MMADSAIndexerScoresPrimitive::unsupported(q, k, w, stream)) {
    // Deliberately a hard error, not a silent Steel fallback: the caller's
    // gate must already have routed unsupported configurations (fp16, H!=64,
    // causal, weights rank 4, GLM shapes) to dsa_indexer_scores.
    throw std::invalid_argument(
        "[omlx_glm_kernels.dsa_indexer_scores_mma] unsupported shape/dtype "
        "(kernel serves bf16, H=64, D=128, weights [B, L, H] only).");
  }

  Shape out_shape{q.shape(0), 1, q.shape(2), k.shape(2)};
  return array(
      std::move(out_shape),
      bfloat16,
      std::make_shared<MMADSAIndexerScoresPrimitive>(
          stream, mask_ratio, mask_q_offset),
      std::move(inputs));
}

array dsa_topk_indices(
    const array& scores,
    int topk,
    bool bucketed,
    bool causal_valid_prefix,
    StreamOrDevice s) {
  return dsa_topk_indices_impl(scores, topk, bucketed, causal_valid_prefix, s);
}

array dspark_fp32_topk_indices(
    const array& scores,
    int topk,
    StreamOrDevice s) {
  if (scores.ndim() != 2 || scores.dtype() != float32 || topk != 512 ||
      scores.shape(1) < topk) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dspark_fp32_topk_indices] expected FP32 "
        << "scores [rows, K>=512] and topk=512, got " << scores.shape()
        << ", topk=" << topk << ".";
    throw std::invalid_argument(msg.str());
  }
  auto stream = to_stream(s);
  if (stream.device == Device::cpu) {
    throw std::invalid_argument("DSpark FP32 top-k requires Metal.");
  }
  auto contiguous_scores = ensure_row_contiguous(scores, stream);
  Shape out_shape{contiguous_scores.shape(0), topk};
  return array(
      std::move(out_shape),
      uint32,
      std::make_shared<DSparkFP32TopKIndicesPrimitive>(stream),
      std::vector<array>{contiguous_scores});
}

array qwen4_qsa_topk_indices(
    const array& scores,
    int topk,
    StreamOrDevice s) {
  if (scores.ndim() != 3 || scores.shape(0) != 1 ||
      scores.shape(1) < 1 || scores.dtype() != float32 || topk != 512 ||
      scores.shape(2) < topk) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.qwen4_qsa_topk_indices] expected FP32 "
        << "scores [1, M>=1, N>=512] and topk=512, got " << scores.shape()
        << ", topk=" << topk << ".";
    throw std::invalid_argument(msg.str());
  }
  auto stream = to_stream(s);
  if (stream.device == Device::cpu) {
    throw std::invalid_argument("Qwen4 QSA FP32 top-k requires Metal.");
  }
  auto contiguous_scores = ensure_row_contiguous(scores, stream);
  Shape out_shape{1, contiguous_scores.shape(1), topk};
  return array(
      std::move(out_shape),
      uint32,
      std::make_shared<Qwen4QSAFP32TopKIndicesPrimitive>(stream),
      std::vector<array>{contiguous_scores});
}
array dsa_decode_scores(
    const array& queries,
    const array& keys,
    const array& weights,
    bool fp32_scores,
    StreamOrDevice s) {
  if (queries.ndim() != 4 || keys.ndim() != 4 ||
      (weights.ndim() != 2 && weights.ndim() != 3)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_decode_scores] expected q/k rank 4 and "
        << "weights rank 2 or 3, got " << queries.shape() << ", "
        << keys.shape() << ", " << weights.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (queries.shape(2) != 1) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.dsa_decode_scores] decode kernel expects a single "
        "query position (q shape [B,H,1,D]).");
  }

  auto final_type = result_type(queries, keys, weights);
  if (final_type != float16 && final_type != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.dsa_decode_scores] expected float16 or bfloat16 "
        << "inputs, got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  auto q = ensure_row_contiguous(astype(queries, final_type, stream), stream);
  // K is consumed via strides — capacity-backed cache slices stay in place.
  // (astype is a no-op on the cache's native dtype; a dtype-mismatched call
  // would still copy, which the row-stride guard then re-checks.)
  auto k = astype(keys, final_type, stream);
  auto w = astype(weights, final_type, stream);
  if (w.ndim() == 3) {
    // accept [B, 1, H]
    w = reshape(w, {w.shape(0), w.shape(2)}, stream);
  }
  w = ensure_row_contiguous(w, stream);

  std::vector<array> inputs = {q, k, w};
  if (DSADecodeScoresPrimitive::unsupported(q, k, w, stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.dsa_decode_scores] unsupported shape/dtype/layout.");
  }

  Shape out_shape{q.shape(0), 1, 1, k.shape(2)};
  return array(
      std::move(out_shape),
      fp32_scores ? float32 : final_type,
      std::make_shared<DSADecodeScoresPrimitive>(stream, fp32_scores),
      std::move(inputs));
}

} // namespace omlx::glm_kernels
