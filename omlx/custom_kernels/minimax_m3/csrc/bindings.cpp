#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "minimax_msa.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native MiniMax M3 kernels for oMLX";

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
      "minimax_msa_topk",
      &omlx::minimax_m3_kernels::minimax_msa_topk,
      "idx_queries"_a,
      "idx_keys"_a,
      "q_start"_a,
      "scale"_a,
      "block_size"_a,
      "topk"_a,
      "init_blocks"_a,
      "local_blocks"_a,
      "stream"_a = nb::none());
}
