// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "decode_fast.h"
#include "rope_append.h"
#include "sdpa_decode.h"
#include "sparse_attn_decode.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native decode fast-path kernels for oMLX";

  // ABI canary: see omlx.custom_kernels.qwen35_prefill (issue #2139).
  m.def(
      "abi_probe",
      [](const mlx::core::array& a) {
        return static_cast<int64_t>(a.size());
      },
      "a"_a);

  m.def(
      "rms_norm_residual_supported",
      &omlx::decode_fast_kernels::rms_norm_residual_supported,
      "x"_a,
      "weight"_a,
      "residual"_a,
      "stream"_a = nb::none());

  m.def(
      "rms_norm_residual",
      &omlx::decode_fast_kernels::rms_norm_residual,
      "x"_a,
      "weight"_a,
      "residual"_a,
      "eps"_a,
      "stream"_a = nb::none());

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

  m.def(
      "rope_kv_append_supported",
      &omlx::decode_fast_kernels::rope_kv_append_supported,
      "keys"_a,
      "values"_a,
      "key_cache"_a,
      "value_cache"_a,
      "offset"_a,
      "dims"_a,
      "stream"_a = nb::none());

  m.def(
      "rope_kv_append",
      &omlx::decode_fast_kernels::rope_kv_append,
      "keys"_a,
      "values"_a,
      "key_cache"_a,
      "value_cache"_a,
      "offset"_a,
      "dims"_a,
      "traditional"_a,
      "base"_a = nb::none(),
      "scale"_a = 1.0f,
      "freqs"_a = nb::none(),
      "stream"_a = nb::none());

  m.def(
      "sparse_attn_decode_supported",
      &omlx::decode_fast_kernels::sparse_attn_decode_supported,
      "q"_a,
      "local_kv"_a,
      "pooled"_a,
      "sinks"_a,
      "stream"_a = nb::none());

  m.def(
      "sparse_attn_decode",
      &omlx::decode_fast_kernels::sparse_attn_decode,
      "q"_a,
      "local_kv"_a,
      "pooled"_a,
      "sinks"_a,
      "stream"_a = nb::none());
}
