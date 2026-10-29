#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::attention {

// Rotate each row's first ``length`` positions along ``axis`` right by its shift.
// The rotation is in place when ``x`` is donatable, and later positions keep their values.
// ``shifts`` holds one int per row (the product of dims before ``axis``).
// Every shift is in [0, max_shift], and max_shift < length.
mx::array roll_rows(
    const mx::array& x,
    const mx::array& shifts,
    int axis,
    int length,
    int max_shift,
    mx::StreamOrDevice s = {});

} // namespace omlx::attention
