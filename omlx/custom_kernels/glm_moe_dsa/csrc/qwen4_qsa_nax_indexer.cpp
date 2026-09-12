// SPDX-License-Identifier: Apache-2.0
#include "qwen4_qsa_nax_indexer.h"

#include <dlfcn.h>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace omlx::glm_kernels {
namespace {
using namespace mlx::core;

constexpr const char* kNaxMetallibName = "omlx_glm_kernels_nax";

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

bool row_contiguous(const array& a) {
  return a.flags().row_contiguous;
}

// Mirror of mlx::core::metal::is_nax_available() (not exported by libmlx).
bool nax_gpu() {
  static bool available = []() {
    if (!metal::is_available()) {
      return false;
    }
    bool os_ok = false;
    if (__builtin_available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      os_ok = true;
    }
    if (!os_ok) {
      return false;
    }
    auto& d = metal::device(Device::gpu);
    const auto& arch = d.get_architecture();
    if (arch.empty()) {
      return false;
    }
    const char suffix = arch.back();
    const int gen = d.get_architecture_gen();
    return gen >= (suffix == 'p' ? 18 : 17);
  }();
  return available;
}

bool nax_lib_built() {
  static bool built = std::filesystem::exists(
      std::filesystem::path(current_binary_dir()) /
      (std::string(kNaxMetallibName) + ".metallib"));
  return built;
}

class Qwen4QSANaxIndexerScoresPrimitive : public Primitive {
 public:
  Qwen4QSANaxIndexerScoresPrimitive(
      Stream stream, int mask_ratio, int mask_q_offset, int block_rows, int block_cols)
      : Primitive(stream),
        mask_ratio_(mask_ratio),
        mask_q_offset_(mask_q_offset),
        block_rows_(block_rows),
        block_cols_(block_cols) {}

  static bool unsupported(const array& q, const array& k, int block_rows, int block_cols, Stream s) {
    if (s.device == Device::cpu || q.dtype() != k.dtype()) {
      return true;
    }
    if (q.dtype() != float16 && q.dtype() != bfloat16) {
      return true;
    }
    if (q.ndim() != 4 || k.ndim() != 4) {
      return true;
    }
    if (block_rows != 64 || (block_cols != 64 && block_cols != 128)) {
      return true;
    }
    return q.shape(0) != 1 || k.shape(0) != 1 || q.shape(1) != 4 ||
        k.shape(1) != 1 || q.shape(2) <= 0 || k.shape(2) <= 0 ||
        q.shape(3) != 128 || k.shape(3) != 128;
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("Qwen4QSANaxIndexerScoresPrimitive has no CPU path.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    if (!row_contiguous(q) || !row_contiguous(k)) {
      throw std::runtime_error(
          "[omlx_glm_kernels.qwen4_qsa_nax_indexer_scores] inputs must be row-contiguous");
    }
    out.set_data(allocator::malloc(out.nbytes()));
    const int bm = block_rows_;
    const int bn = block_cols_;
    constexpr int bk = 64;
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
        /* M = */ M,
        /* N = */ N,
        /* K = */ D,
        /* lda = */ D,
        /* ldb = */ D,
        /* ldd = */ N,
        /* tiles_n = */ tiles_n,
        /* tiles_m = */ tiles_m,
        /* batch_stride_a = */ int64_t(H) * M * D,
        /* batch_stride_b = */ int64_t(N) * D,
        /* batch_stride_d = */ int64_t(M) * N,
        /* swizzle_log = */ 0,
        /* gemm_k_iterations_aligned = */ D / bk,
        /* batch_ndim = */ 1};
    std::string name;
    concatenate(name, "qwen4_qsa_nax_indexer_score_", type_to_name(q), "_bm", bm,
                "_bn", bn, "_bk", bk, "_wm", wm, "_wn", wn);
    auto lib = d.get_library(kNaxMetallibName, current_binary_dir());
    auto kernel = d.get_kernel(name, lib);
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
    encoder.dispatch_threadgroups(MTL::Size(tiles_n, tiles_m, B), MTL::Size(wm * wn * 32, 1, 1));
  }

  DEFINE_NAME(OMLXQwen4QSANaxIndexerScores)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Qwen4QSANaxIndexerScoresPrimitive&>(other);
    return mask_ratio_ == rhs.mask_ratio_ && mask_q_offset_ == rhs.mask_q_offset_ &&
        block_rows_ == rhs.block_rows_ && block_cols_ == rhs.block_cols_;
  }

 private:
  int mask_ratio_;
  int mask_q_offset_;
  int block_rows_;
  int block_cols_;
};
} // namespace

bool qwen4_qsa_nax_indexer_available() {
  return nax_gpu() && nax_lib_built();
}

array qwen4_qsa_nax_indexer_scores(
    const array& queries, const array& pooled_keys, int mask_ratio, int mask_q_offset,
    int block_rows, int block_cols, StreamOrDevice s) {
  auto stream = to_stream(s);
  if (Qwen4QSANaxIndexerScoresPrimitive::unsupported(queries, pooled_keys, block_rows, block_cols, stream) ||
      mask_ratio <= 0 || mask_q_offset < 0 || !qwen4_qsa_nax_indexer_available()) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.qwen4_qsa_nax_indexer_scores] expected q=[1,4,M,128], "
        << "k=[1,1,N,128] bf16/fp16, mask_ratio>0, block_rows == 64, block_cols in {64,128}, a NAX GPU and "
        << kNaxMetallibName << ".metallib; got " << queries.shape() << ", "
        << pooled_keys.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  // Lazy views carry no final flags; contiguous() is elided on realized row-major inputs.
  array q = contiguous(queries, false, s);
  array k = contiguous(pooled_keys, false, s);
  Shape out_shape{q.shape(0), q.shape(2), k.shape(2)};
  return array(
      std::move(out_shape), float32,
      std::make_shared<Qwen4QSANaxIndexerScoresPrimitive>(stream, mask_ratio, mask_q_offset, block_rows, block_cols),
      std::vector<array>{q, k});
}
} // namespace omlx::glm_kernels
