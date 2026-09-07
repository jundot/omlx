// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "sdpa_decode.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native exact decode SDPA for Qwen4 QSA";
  m.def(
      "abi_probe",
      [](const mlx::core::array& a) { return static_cast<int64_t>(a.size()); },
      "a"_a);
  m.def(
      "sdpa_decode_supported",
      &omlx::decode_fast_kernels::sdpa_decode_supported,
      "q"_a,
      "k"_a,
      "v"_a,
      "stream"_a = nb::none());
  m.def(
      "sdpa_decode",
      &omlx::decode_fast_kernels::sdpa_decode,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "causal"_a,
      "mask"_a = nb::none(),
      "sinks"_a = nb::none(),
      "stream"_a = nb::none());
}
