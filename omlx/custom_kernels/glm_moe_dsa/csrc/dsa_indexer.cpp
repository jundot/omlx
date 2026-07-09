#include "dsa_indexer.h"

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
      int causal_q_offset)
      : Primitive(stream),
        causal_(causal),
        weights_lh_(weights_lh),
        unused_causal_prefix_topk_(unused_causal_prefix_topk),
        skip_causal_future_store_(skip_causal_future_store),
        causal_q_offset_(causal_q_offset) {}

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
    if (q.shape(2) % 64 != 0 || k.shape(2) % 64 != 0 ||
        q.shape(3) % 16 != 0) {
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
        causal_q_offset_ == rhs.causal_q_offset_;
  }
  auto state() const {
    return std::make_tuple(
        causal_,
        weights_lh_,
        unused_causal_prefix_topk_,
        skip_causal_future_store_,
        causal_q_offset_);
  }

 private:
  bool causal_;
  bool weights_lh_;
  int unused_causal_prefix_topk_;
  bool skip_causal_future_store_;
  int causal_q_offset_;
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
          causal_q_offset),
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

} // namespace omlx::glm_kernels
