# ds4-metal kernel provenance

This package contains the capability and fallback contract for optional
DeepSeek V4 Metal primitives. It does not vendor the DwarfStar runtime or
enable a production dispatch.

## Upstream reference

| Field | Value |
|---|---|
| Project | `ivanfioravanti/ds4-metal` (DwarfStar) |
| Source | https://github.com/ivanfioravanti/ds4-metal |
| Commit | `78269ce7ca0f8fd4deff15b803ea4bc87fc6b99e` |
| Commit date | 2026-08-22 |
| License | MIT; see `LICENSE.ds4-metal` in this package |
| Notices | Copyright 2026 the ds4.c authors; copyright 2023-2026 the ggml authors |

DwarfStar acknowledges that its GGUF quant layouts and tables, CPU quant/dot
logic, and certain kernels are retained or adapted from llama.cpp/GGML under
the MIT license. The ggml copyright is therefore retained here even though
this first seam adds no GGUF runtime.

## Adapted-code inventory

| oMLX path | Upstream reference | Status |
|---|---|---|
| `benchmarks/prototypes/ds4_tp_prefill_moe_phase_a.metal` | `metal/moe.metal` MXFP4 pair-plus-SwiGLU family | From-scratch MLX/Steel adaptation; Apache-2.0; no upstream source copied |
| `omlx/custom_kernels/glm_moe_dsa/csrc/ds4_prefill_moe.{metal,cpp,h}` | The same pair-plus-SwiGLU scheduling idea | Callable but isolated fixed-shape MLX implementation derived from the prototype; Apache-2.0; no upstream source copied |
| `omlx/custom_kernels/glm_moe_dsa/csrc/dsa_indexer_nax.metal` | `metal/dsv4_misc.metal::kernel_dsv4_indexer_scores_nax` | Modified port to oMLX's BF16 MLX-array ABI; upstream MIT notice retained by this package; oMLX modifications marked Apache-2.0 |

The phase-A implementations principally adapt the scheduling idea behind these
upstream `metal/moe.metal` symbols:

- `kernel_mul_mm_id_pair_swiglu_f16_impl`
- `kernel_mul_mm_id_mxfp4_pair_swiglu_f16`

The oMLX pair-plus-SwiGLU code consumes MLX's separate packed-weight and
E8M0-scale arrays; DwarfStar consumes GGUF `block_mxfp4` records that
interleave one scale byte with sixteen code bytes. No DwarfStar Metal source
is copied into the prototype or its callable fixed-shape implementation.

The NAX file identifies the ds4-metal kernel it was ported from in its source
comment. This inventory and `LICENSE.ds4-metal` provide the retained MIT and
ggml notices for its distribution without changing that concurrently developed
kernel file.

If a later change copies or modifies upstream source, that file must retain
the MIT notice, identify the upstream path and pinned commit, and describe the
modification here. Model weights and GGUF artifacts have their own licenses
and are not covered by this code notice.
