#include "kv_roll.h"
#include "omlx/binary_dir.h"

#include <sstream>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::attention {

namespace {

using namespace mlx::core;

constexpr int kRollBlock = 256;
constexpr int kRollTile = 16;
constexpr int kRollThreads = 256;

const char* width_name(int itemsize) {
  switch (itemsize) {
    case 1:
      return "b1";
    case 2:
      return "b2";
    case 4:
      return "b4";
    default:
      throw std::invalid_argument(
          "[attention.roll_rows] expected a 1, 2 or 4 byte dtype.");
  }
}

class RollRowsPrimitive : public Primitive {
 public:
  RollRowsPrimitive(Stream stream, int axis, int length, int max_shift)
      : Primitive(stream),
        axis_(axis),
        length_(length),
        max_shift_(max_shift) {}

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("RollRowsPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    const auto& x = inputs[0];
    const auto& shifts = inputs[1];
    auto& out = outputs[0];

    int rows = 1;
    for (int i = 0; i < axis_; ++i) {
      rows *= x.shape(i);
    }
    const int capacity = x.shape(axis_);
    int inner = 1;
    for (int i = axis_ + 1; i < x.ndim(); ++i) {
      inner *= x.shape(i);
    }
    const std::string width = width_name(x.itemsize());
    auto lib = d.get_library("omlx_attention_kernels", binary_dir());
    auto& encoder = metal::get_command_encoder(s);

    if (!x.is_donatable()) {
      out.set_data(allocator::malloc(out.nbytes()));
      encoder.set_compute_pipeline_state(
          d.get_kernel("kv_roll_copy_" + width, lib));
      encoder.set_input_array(x, 0);
      encoder.set_input_array(shifts, 1);
      encoder.set_output_array(out, 2);
      encoder.set_bytes(capacity, 3);
      encoder.set_bytes(length_, 4);
      encoder.set_bytes(inner, 5);
      encoder.dispatch_threads(
          MTL::Size(inner, capacity, rows),
          MTL::Size(std::min(inner, kRollThreads), 1, 1));
      return;
    }

    out.copy_shared_buffer(x);
    if (max_shift_ == 0) {
      return;
    }
    const int blocks = (length_ + kRollBlock - 1) / kRollBlock;
    array halo({rows, blocks, max_shift_, inner}, x.dtype(), nullptr, {});
    halo.set_data(allocator::malloc(halo.nbytes()));
    encoder.add_temporary(halo);

    encoder.set_compute_pipeline_state(
        d.get_kernel("kv_roll_halo_" + width, lib));
    encoder.set_input_array(x, 0);
    encoder.set_input_array(shifts, 1);
    encoder.set_output_array(halo, 2);
    encoder.set_bytes(capacity, 3);
    encoder.set_bytes(length_, 4);
    encoder.set_bytes(inner, 5);
    encoder.set_bytes(max_shift_, 6);
    const int halo_width = max_shift_ * inner;
    encoder.dispatch_threads(
        MTL::Size(halo_width, blocks, rows),
        MTL::Size(std::min(halo_width, kRollThreads), 1, 1));

    // The apply pass writes positions the halo pass reads.
    encoder.barrier();

    encoder.set_compute_pipeline_state(
        d.get_kernel("kv_roll_apply_" + width, lib));
    encoder.set_output_array(out, 0);
    encoder.set_input_array(shifts, 1);
    encoder.set_input_array(halo, 2);
    encoder.set_bytes(capacity, 3);
    encoder.set_bytes(length_, 4);
    encoder.set_bytes(inner, 5);
    encoder.set_bytes(max_shift_, 6);
    encoder.dispatch_threadgroups(
        MTL::Size((inner + kRollTile - 1) / kRollTile, blocks, rows),
        MTL::Size(kRollThreads, 1, 1));
  }

  DEFINE_NAME(OMLXRollRows)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const RollRowsPrimitive&>(other);
    return axis_ == rhs.axis_ && length_ == rhs.length_ &&
        max_shift_ == rhs.max_shift_;
  }
  auto state() const {
    return std::make_tuple(axis_, length_, max_shift_);
  }

 private:
  int axis_;
  int length_;
  int max_shift_;
};

} // namespace

array roll_rows(
    const array& x,
    const array& shifts,
    int axis,
    int length,
    int max_shift,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  if (stream.device == Device::cpu) {
    throw std::invalid_argument("[attention.roll_rows] requires a GPU stream.");
  }
  if (axis < 0) {
    axis += x.ndim();
  }
  if (axis < 0 || axis >= x.ndim()) {
    throw std::invalid_argument("[attention.roll_rows] axis out of range.");
  }
  if (length <= 0 || length > x.shape(axis) || max_shift < 0 ||
      max_shift >= length) {
    std::ostringstream msg;
    msg << "[attention.roll_rows] invalid length " << length
        << " or max_shift " << max_shift << " for axis size " << x.shape(axis)
        << ".";
    throw std::invalid_argument(msg.str());
  }
  width_name(x.itemsize());
  int rows = 1;
  for (int i = 0; i < axis; ++i) {
    rows *= x.shape(i);
  }
  if (shifts.size() != static_cast<size_t>(rows)) {
    throw std::invalid_argument(
        "[attention.roll_rows] expected one shift per row before the axis.");
  }
  auto row_shifts =
      contiguous(reshape(astype(shifts, int32, stream), {rows}, stream), false,
                 stream);
  // Only a row-contiguous bank can be updated in place; others are copied.
  auto bank = contiguous(x, false, stream);
  return array(
      bank.shape(),
      bank.dtype(),
      std::make_shared<RollRowsPrimitive>(stream, axis, length, max_shift),
      {bank, row_shifts});
}

} // namespace omlx::attention
