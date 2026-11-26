#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "oqh.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "oQh kernels for oMLX";

  // ABI canary checked once by fast.py (see custom_kernels/_loader.py).
  m.def(
      "abi_probe",
      [](const mlx::core::array& a) { return static_cast<int64_t>(a.size()); },
      "a"_a);
  m.def(
      "qmv",
      &omlx::oqh::qmv,
      "x"_a, "w"_a, "rscale"_a, "bits"_a, "w_b"_a = nb::none(), "rscale_b"_a = nb::none(),
      "stream"_a = nb::none());
  m.def(
      "gather_qmv",
      &omlx::oqh::gather_qmv,
      "x"_a, "w"_a, "rscale"_a, "indices"_a, "bits"_a, "w_b"_a = nb::none(),
      "rscale_b"_a = nb::none(), "stream"_a = nb::none());
  m.def(
      "rmsnorm_had",
      &omlx::oqh::rmsnorm_had,
      "x"_a, "weight"_a, "signs"_a, "eps"_a, "stream"_a = nb::none());
  m.def("rotate", &omlx::oqh::rotate, "x"_a, "signs"_a, "block"_a, "stream"_a = nb::none());
  m.def(
      "act_split_had",
      &omlx::oqh::act_split_had,
      "y"_a, "signs"_a, "block"_a, "limit"_a = nb::none(), "stream"_a = nb::none());
  m.def(
      "act_pair_had",
      &omlx::oqh::act_pair_had,
      "a"_a, "b"_a, "signs"_a, "block"_a, "limit"_a = nb::none(), "stream"_a = nb::none());
  m.def(
      "gnorm_had",
      &omlx::oqh::gnorm_had,
      "x"_a, "z"_a, "weight"_a, "signs"_a, "eps"_a, "block"_a, "stream"_a = nb::none());
}
