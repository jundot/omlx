// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// oQh kernel dispatch. Metal sources: oqh.metal, compiled into
// omlx_oqh_kernels.metallib and loaded on first use.

#include "oqh.h"
#include "omlx/binary_dir.h"

#include <sstream>
#include <string>
#include <string_view>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace omlx::oqh {

namespace {

using namespace mlx::core;

constexpr const char* kMetallibName = "omlx_oqh_kernels";

MTL::ComputePipelineState* get_kernel(metal::Device& d, const std::string& name) {
  auto* lib = d.get_library(kMetallibName, binary_dir());
  return d.get_kernel(name, lib);
}

std::string type_str(Dtype dt) {
  if (dt == float16) {
    return "float16_t";
  }
  if (dt == bfloat16) {
    return "bfloat16_t";
  }
  if (dt == float32) {
    return "float";
  }
  std::ostringstream msg;
  msg << "oqh: unsupported dtype " << dt;
  throw std::invalid_argument(msg.str());
}

void check_bits(int bits) {
  if (bits != 2 && bits != 3 && bits != 4 && bits != 5 && bits != 6 && bits != 8) {
    throw std::invalid_argument("oqh: bits must be 2, 3, 4, 5, 6 or 8");
  }
}

void check_float(const array& x, const char* what) {
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "oqh: " << what << " must be float16 or bfloat16";
    throw std::invalid_argument(msg.str());
  }
}

// Rotation blocks are 128 or 256 inputs (oqh.format.MAX_BLOCK).
void check_block(int width, int block, int min_block, const char* what) {
  const bool pow2 = block > 0 && (block & (block - 1)) == 0;
  if (!pow2 || block < min_block || block > 256 || width % block != 0) {
    std::ostringstream msg;
    msg << "oqh: " << what << " block " << block << " must be a power of two in [" << min_block
        << ", 256] dividing width " << width;
    throw std::invalid_argument(msg.str());
  }
}

int in_width(const array& w, int bits) {
  return static_cast<int>(w.shape(-1)) * 32 / bits;
}

array as_dtype(const array& a, Dtype dt, const Stream& s) {
  return a.dtype() == dt ? a : astype(a, dt, s);
}


// Inputs: x, w, rscale, [indices], [w_b, rscale_b]. A pair writes [.., 2N].
class OqhQmv : public Primitive {
 public:
  OqhQmv(Stream s, int bits, bool gather, bool shared_x, bool pair)
      : Primitive(s), bits_(bits), gather_(gather), shared_x_(shared_x), pair_(pair) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("OqhQmv has no CPU path.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const int K = in_width(w, bits_);
    const int N = static_cast<int>(w.shape(-2));
    std::string name = (gather_ ? "oqh_gather_qmv_" : "oqh_qmv_") + type_str(x.dtype()) +
        "_b" + std::to_string(bits_);
    if (gather_) {
      name += shared_x_ ? "_s1" : "_s0";
    }
    if (pair_) {
      name += "_pair";
    }
    const int rows = static_cast<int>(x.size() / K);
    // x rows per threadgroup: each weight block is read once for all of them.
    const int R = gather_ || rows == 1 ? 1 : rows <= 4 ? 4 : 8;
    if (!gather_) {
      name += "_r" + std::to_string(R);
    }
    auto& enc = metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(get_kernel(d, name));
    int c = 0;
    enc.set_input_array(w, c++);
    enc.set_input_array(inputs[2], c++);
    enc.set_input_array(x, c++);
    enc.set_output_array(out, c++);
    if (gather_) {
      enc.set_input_array(inputs[3], c++);
    }
    enc.set_bytes(K, c++);
    enc.set_bytes(N, c++);
    // Single kernels never read the second pair of buffers.
    const size_t b = gather_ ? 4 : 3;
    enc.set_input_array(pair_ ? inputs[b] : w, c++);
    enc.set_input_array(pair_ ? inputs[b + 1] : inputs[2], c++);
    if (!gather_) {
      enc.set_bytes(rows, c++);
    }
    const int out_rows = pair_ ? 2 * N : N;
    MTL::Size group_dims(32, 2, 1);
    MTL::Size grid_dims = gather_
        ? MTL::Size(1, out_rows / 8, static_cast<int>(inputs[3].size()))
        : MTL::Size((rows + R - 1) / R, out_rows / 8, 1);
    enc.dispatch_threadgroups(grid_dims, group_dims);
  }

  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const OqhQmv&>(other);
    return bits_ == o.bits_ && gather_ == o.gather_ && shared_x_ == o.shared_x_ &&
        pair_ == o.pair_;
  }

  DEFINE_NAME(OqhQmv)

 private:
  int bits_;
  bool gather_;
  bool shared_x_;
  bool pair_;
};

enum class Producer { Norm, Rotate, Act, ActSplit, GatedNorm };

class OqhProducer : public Primitive {
 public:
  OqhProducer(Stream s, Producer kind, int block, int mode, float eps)
      : Primitive(s), kind_(kind), block_(block), mode_(mode), eps_(eps) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("OqhProducer has no CPU path.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    for (auto& o : outputs) {
      o.set_data(allocator::malloc(o.nbytes()));
    }
    auto& out = outputs[0];
    const std::string t = type_str(out.dtype());
    const std::string k = "_k" + std::to_string(block_);
    auto& enc = metal::get_command_encoder(s);
    int c = 0;
    if (kind_ == Producer::Norm) {
      // Inputs: (x, w, signs). Outputs: (rotated, plain).
      const int D = static_cast<int>(inputs[0].shape(-1));
      enc.set_compute_pipeline_state(get_kernel(d, "oqh_norm_" + t));
      enc.set_input_array(inputs[0], c++);
      enc.set_input_array(inputs[1], c++);
      enc.set_input_array(inputs[2], c++);
      enc.set_output_array(outputs[0], c++);
      enc.set_output_array(outputs[1], c++);
      enc.set_bytes(D, c++);
      enc.set_bytes(eps_, c++);
      const int rows = static_cast<int>(inputs[0].size() / D);
      enc.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(D / 8, 1, 1));
      return;
    }
    const int L = static_cast<int>(out.shape(-1));
    if (kind_ == Producer::GatedNorm) {
      // Inputs: (x, z, w, signs). Outputs: (rotated, plain).
      const std::string tin = type_str(inputs[0].dtype());
      enc.set_compute_pipeline_state(get_kernel(d, "oqh_gnorm_" + tin + "_" + t + k));
      const int Ls = static_cast<int>(inputs[3].size());
      enc.set_input_array(inputs[0], c++);
      enc.set_input_array(inputs[1], c++);
      enc.set_input_array(inputs[2], c++);
      enc.set_input_array(inputs[3], c++);
      enc.set_output_array(out, c++);
      enc.set_bytes(Ls, c++);
      enc.set_bytes(eps_, c++);
      enc.set_output_array(outputs[1], c++);
    } else {
      // Inputs: (y, signs) for Rotate (y alone) and ActSplit (its two halves),
      // (a, b, signs) for Act.
      const bool split = kind_ == Producer::ActSplit;
      const bool two = kind_ == Producer::Act;
      enc.set_compute_pipeline_state(
          get_kernel(d, "oqh_act_" + t + k + "_m" + std::to_string(mode_) + (split ? "_s" : "")));
      enc.set_input_array(inputs[0], c++);
      enc.set_input_array(inputs[two ? 1 : 0], c++);
      enc.set_input_array(inputs.back(), c++);
      enc.set_output_array(out, c++);
      enc.set_bytes(L, c++);
      // Clamp limit of mode 2; the other modes ignore it.
      enc.set_bytes(eps_, c++);
    }
    const int tgs = static_cast<int>(out.size() / block_);
    enc.dispatch_threadgroups(MTL::Size(tgs, 1, 1), MTL::Size(block_ / 4, 1, 1));
  }

  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const OqhProducer&>(other);
    return kind_ == o.kind_ && block_ == o.block_ && mode_ == o.mode_ && eps_ == o.eps_;
  }

  DEFINE_NAME(OqhProducer)

 private:
  Producer kind_;
  int block_;
  int mode_;
  float eps_;
};

// Whether a is row-contiguous when it is evaluated: evaluated inputs report it,
// the oQh primitives, MLX's quantized matmuls and custom Metal kernels always
// write fresh dense outputs, and a reshape or added / removed unit axis of a
// dense array stays dense.
bool known_dense(const array& a, int depth = 0) {
  if (a.is_available()) {
    return a.flags().row_contiguous;
  }
  if (!a.has_primitive() || depth > 4) {
    return false;
  }
  // MLX does not export its primitives' type info, so match them by name.
  const std::string_view name = a.primitive().name();
  if (name == "OqhQmv" || name == "OqhProducer" || name == "QuantizedMatmul" ||
      name == "GatherQMM" || name == "CustomKernel") {
    return true;
  }
  const bool view = name == "Reshape" || name == "ExpandDims" || name == "Squeeze";
  return view && a.inputs().size() == 1 && known_dense(a.inputs()[0], depth + 1);
}

// The kernels index every input as a dense row-major array; any other input (a
// slice, a tensor-parallel view) gets a contiguous node, which copies at run
// time only if it is strided.
array dense(const array& a, const Stream& s) {
  return known_dense(a) ? a : contiguous(a, false, s);
}

array dense(const array& a, Dtype dt, const Stream& s) {
  return dense(as_dtype(a, dt, s), s);
}

void check_signs(const array& signs, int width, const char* what) {
  if (signs.size() != static_cast<size_t>(width)) {
    std::ostringstream msg;
    msg << "oqh: " << what << " needs " << width << " signs, got " << signs.size();
    throw std::invalid_argument(msg.str());
  }
}

} // namespace

namespace {

void check_pair(
    const array& w, const array& rscale, const std::optional<array>& w_b,
    const std::optional<array>& rscale_b) {
  if (w_b.has_value() != rscale_b.has_value() ||
      (w_b && (w_b->shape() != w.shape() || w_b->dtype() != w.dtype() ||
               rscale_b->shape() != rscale.shape()))) {
    throw std::invalid_argument("oqh: w_b and rscale_b must both be given and match w and rscale");
  }
}

} // namespace

array qmv(
    const array& x, const array& w, const array& rscale, int bits,
    const std::optional<array>& w_b, const std::optional<array>& rscale_b, StreamOrDevice s_) {
  auto s = to_stream(s_);
  check_bits(bits);
  check_float(x, "x");
  check_pair(w, rscale, w_b, rscale_b);
  const int K = in_width(w, bits);
  const int N = static_cast<int>(w.shape(-2));
  if (w.ndim() != 2 || x.shape(-1) != K || N % 8 != 0 || K % 16 != 0) {
    throw std::invalid_argument("oqh qmv: expected w [N, K * bits / 32], N % 8 == 0, K % 16 == 0");
  }
  auto xc = dense(x, s);
  auto out_shape = xc.shape();
  out_shape.back() = w_b ? 2 * N : N;
  std::vector<array> inputs = {xc, dense(w, s), dense(rscale, xc.dtype(), s)};
  if (w_b) {
    inputs.push_back(dense(*w_b, s));
    inputs.push_back(dense(*rscale_b, xc.dtype(), s));
  }
  return array(
      out_shape, xc.dtype(), std::make_shared<OqhQmv>(s, bits, false, false, w_b.has_value()),
      std::move(inputs));
}

array gather_qmv(
    const array& x, const array& w, const array& rscale, const array& indices, int bits,
    const std::optional<array>& w_b, const std::optional<array>& rscale_b, StreamOrDevice s_) {
  auto s = to_stream(s_);
  check_bits(bits);
  check_float(x, "x");
  check_pair(w, rscale, w_b, rscale_b);
  const int K = in_width(w, bits);
  const int N = static_cast<int>(w.shape(-2));
  const int P = static_cast<int>(indices.size());
  const int R = static_cast<int>(x.size() / K);
  if (w.ndim() != 3 || x.shape(-1) != K || N % 8 != 0 || K % 16 != 0 || (R != 1 && R != P)) {
    throw std::invalid_argument("oqh gather_qmv: expected w [E, N, K * bits / 32] and x with 1 or P rows");
  }
  auto xc = dense(x, s);
  auto idx = dense(indices, uint32, s);
  std::vector<array> inputs = {xc, dense(w, s), dense(rscale, xc.dtype(), s), idx};
  if (w_b) {
    inputs.push_back(dense(*w_b, s));
    inputs.push_back(dense(*rscale_b, xc.dtype(), s));
  }
  return array(
      {P, w_b ? 2 * N : N}, xc.dtype(),
      std::make_shared<OqhQmv>(s, bits, true, R == 1, w_b.has_value()), std::move(inputs));
}

std::vector<array> rmsnorm_had(
    const array& x, const array& weight, const array& signs, float eps, StreamOrDevice s_) {
  auto s = to_stream(s_);
  check_float(x, "x");
  const int D = static_cast<int>(x.shape(-1));
  // One threadgroup per row: D / 8 threads, a 256 block per simdgroup.
  if (D % 256 != 0 || D > 8192) {
    std::ostringstream msg;
    msg << "oqh: norm width " << D << " must be a multiple of 256 and at most 8192";
    throw std::invalid_argument(msg.str());
  }
  check_signs(signs, D, "rmsnorm_had");
  auto xc = dense(x, s);
  return array::make_arrays(
      {xc.shape(), xc.shape()}, {xc.dtype(), xc.dtype()},
      std::make_shared<OqhProducer>(s, Producer::Norm, 256, 0, eps),
      {xc, dense(weight, xc.dtype(), s), dense(signs, xc.dtype(), s)});
}

array rotate(const array& x, const array& signs, int block, StreamOrDevice s_) {
  auto s = to_stream(s_);
  check_float(x, "x");
  const int L = static_cast<int>(x.shape(-1));
  check_block(L, block, 128, "rotate");
  check_signs(signs, L, "rotate");
  auto xc = dense(x, s);
  return array(
      xc.shape(), xc.dtype(), std::make_shared<OqhProducer>(s, Producer::Rotate, block, 3, 0.0f),
      {xc, dense(signs, xc.dtype(), s)});
}

array act_split_had(
    const array& y,
    const array& signs,
    int block,
    std::optional<float> limit,
    StreamOrDevice s_) {
  auto s = to_stream(s_);
  check_float(y, "y");
  const int L = static_cast<int>(signs.size());
  if (y.shape(-1) != 2 * L) {
    throw std::invalid_argument("oqh act_split_had: last dim must be 2 * signs.size()");
  }
  check_block(L, block, 128, "act_split_had");
  auto yc = dense(y, s);
  auto shape = yc.shape();
  shape.back() = L;
  return array(
      std::move(shape), yc.dtype(),
      std::make_shared<OqhProducer>(
          s, Producer::ActSplit, block, limit ? 2 : 0, limit.value_or(0.0f)),
      {yc, dense(signs, yc.dtype(), s)});
}

array act_pair_had(
    const array& a,
    const array& b,
    const array& signs,
    int block,
    std::optional<float> limit,
    StreamOrDevice s_) {
  auto s = to_stream(s_);
  check_float(a, "a");
  const int L = static_cast<int>(signs.size());
  if (a.shape() != b.shape() || a.shape(-1) != L) {
    throw std::invalid_argument("oqh act_pair_had: a and b must match and end in signs.size()");
  }
  check_block(L, block, 128, "act_pair_had");
  auto ac = dense(a, s);
  return array(
      ac.shape(), ac.dtype(),
      std::make_shared<OqhProducer>(s, Producer::Act, block, limit ? 2 : 0, limit.value_or(0.0f)),
      {ac, dense(b, ac.dtype(), s), dense(signs, ac.dtype(), s)});
}

std::vector<array> gnorm_had(
    const array& x, const array& z, const array& weight, const array& signs, float eps, int block,
    StreamOrDevice s_) {
  auto s = to_stream(s_);
  check_float(z, "z");
  if (x.shape(-1) != 128 || z.size() != x.size()) {
    throw std::invalid_argument("oqh gnorm_had: expected 128-wide heads and z shaped like x");
  }
  const int L = static_cast<int>(signs.size());
  if (static_cast<int>(z.shape(-1)) * static_cast<int>(z.shape(-2)) != L) {
    throw std::invalid_argument("oqh gnorm_had: signs must cover heads x 128");
  }
  check_block(L, block, 128, "gnorm_had");
  auto xc = dense(x, s);
  auto zc = dense(z, s);
  return array::make_arrays(
      {zc.shape(), zc.shape()}, {zc.dtype(), zc.dtype()},
      std::make_shared<OqhProducer>(s, Producer::GatedNorm, block, 0, eps),
      {xc, zc, dense(weight, zc.dtype(), s), dense(signs, zc.dtype(), s)});
}

} // namespace omlx::oqh
