// SPDX-License-Identifier: Apache-2.0
#include "command_buffer_caps.h"

#include "mlx/backend/metal/device.h"

namespace omlx::quant_matmul {

namespace metal = mlx::core::metal;

namespace {

// metal::Device keeps its buffer caps private and has no setter. An explicit
// instantiation may name a private member, so these tags hand out pointers to
// the two fields; a renamed field fails to compile instead of misbehaving.
template <typename Tag, typename Tag::type Member>
struct DeviceField {
  friend typename Tag::type field(Tag) {
    return Member;
  }
};

struct OpsPerBuffer {
  using type = int metal::Device::*;
  friend type field(OpsPerBuffer);
};

struct MbPerBuffer {
  using type = int metal::Device::*;
  friend type field(MbPerBuffer);
};

template struct DeviceField<OpsPerBuffer, &metal::Device::max_ops_per_buffer_>;
template struct DeviceField<MbPerBuffer, &metal::Device::max_mb_per_buffer_>;

} // namespace

std::tuple<int, int> set_command_buffer_caps(int ops, int mb) {
  auto& d = metal::device(mlx::core::Device::gpu);
  auto previous = d.get_max_ops_mb_per_buffer();
  d.*field(OpsPerBuffer{}) = ops;
  d.*field(MbPerBuffer{}) = mb;
  return previous;
}


} // namespace omlx::quant_matmul
