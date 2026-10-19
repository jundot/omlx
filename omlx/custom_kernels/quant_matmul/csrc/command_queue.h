#pragma once

#include <cstdint>

#include "mlx/utils.h"

namespace omlx::quant_matmul {

// Moves the calling thread's encoder for `stream` onto one process-wide
// MTLCommandQueue and returns the encoder's queue address afterward (0 for a
// CPU stream). Call it before the stream encodes any work on this thread.
uintptr_t share_command_queue(mlx::core::StreamOrDevice stream);

} // namespace omlx::quant_matmul
