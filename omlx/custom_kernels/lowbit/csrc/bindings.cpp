#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "lowbit_qmv.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native T5 ternary kernels for oMLX";

  // ABI canary: when the extension is built with a nanobind whose ABI tag
  // differs from the one the mlx wheel was built with, the NB_DOMAIN is
  // isolated and every mx.array argument is rejected with "incompatible
  // function arguments" (issue #2139). fast.py calls this probe once at
  // import and disables the native symbols when it fails.
  m.def(
      "abi_probe",
      [](const mlx::core::array& a) {
        return static_cast<int64_t>(a.size());
      },
      "a"_a);
  m.def(
      "t5_qmv",
      &omlx::lowbit::t5_qmv,
      "x"_a, "w"_a, "scales"_a,
      "stream"_a = nb::none());
  m.def(
      "t5_qmv_wide",
      &omlx::lowbit::t5_qmv_wide,
      "x"_a, "w"_a, "scales"_a,
      "stream"_a = nb::none());
  m.def(
      "t5_qmm",
      &omlx::lowbit::t5_qmm,
      "x"_a, "w"_a, "scales"_a,
      "stream"_a = nb::none());
}
