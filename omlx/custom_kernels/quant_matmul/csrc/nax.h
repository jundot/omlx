// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"

namespace omlx::quant_matmul {

// Mirror of mlx::core::metal::is_nax_available() (mlx v0.32.2 device.cpp),
// which libmlx does not export: macOS >= 26.2 and applegpu gen >= 17
// ('p'-suffix parts need gen >= 18).
inline bool is_nax_available() {
  static bool available = []() {
    if (!mlx::core::metal::is_available()) {
      return false;
    }
    bool os_ok = false;
    if (__builtin_available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      os_ok = true;
    }
    if (!os_ok) {
      return false;
    }
    auto& d = mlx::core::metal::device(mlx::core::Device::gpu);
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

} // namespace omlx::quant_matmul
