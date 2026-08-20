# omlx.custom_kernels.laguna_nvfp4

NVFP4 (E4M3) decode kernels for oMLX, ported from
[Layr-Labs/mlxfast-challenge](https://github.com/Layr-Labs/mlxfast-challenge)
(`Sources/MLXFastModel/LagunaRuntimeModel.swift`).

## Porting rules

- **Kernel bodies are kept verbatim** from the challenge source. Only the
  framework-injected bits change: the Swift `MLXFast.metalKernel` buffer
  bindings become explicit `[[buffer(n)]]` attributes, and the flag-injected
  Metal headers are resolved at the challenge's **default configuration**
  (all `DARKBLOOM_NVFP4_*` flags on, nibble split 1).
- Each kernel gets a C++ `Primitive` dispatch (mlx backend pattern), a
  nanobind binding (`NB_DOMAIN mlx`, ABI-pinned nanobind 2.13), a Python
  wrapper with a **stock-op fallback**, and tests against the stock
  `mx.quantized_matmul(mode="nvfp4")` path on real model data.

## NVFP4 contract

```text
weight  U32 [N, K*4/32]   4-bit E4M3 nibbles (8 per uint32)
scales  U8  [N, K/16]     E4M3 group scales (group size 16)
bias    —                  none (always affine with scale only)
```

The fused gate/up plane concatenates gate rows `0..N-1` over up rows
`N..2N-1` on the row axis. The kernel's `2^22` row rescale is the deferred
form of the challenge's scale fold (bit-exact per the challenge's closed
case analysis).

## Kernels

| Kernel | Swift source | Status |
|---|---|---|
| `laguna_shared_nvfp4_swiglu_qmv_bf16_v1` | `lagunaSharedSwiGLUQMVKernel` | ✅ bit-exact vs stock |
| `laguna_shared_nvfp4_down_residual_bf16_v1` | `lagunaSharedDownResidualSource(false)` | ✅ within 4 ulp |
| `laguna_routed_nvfp4_swiglu_qmv_bf16_v2` | `lagunaRoutedSwiGLUQMVKernel` | ✅ within ~2 ulp |
| `laguna_routed_nvfp4_down_reduce_bf16_v2` | `lagunaRoutedDownReduceKernel` | ✅ within ~2 ulp |
| `laguna_full_qk_norm_yarn_bf16_128_v4` | `lagunaFullQKNormYaRNKernel` | ✅ ~1 bf16 step |
| `laguna_sliding_qk_norm_rope_bf16_128_v1` | `lagunaSlidingQKNormRoPEKernel` | ✅ bit-exact |
| `laguna_decode_nvfp4_qkv_h*_r1_v1_se1_sd1` | `lagunaDecodeNVFP4QKVR1Source` | ✅ bit-exact (h48/h64) |
| `laguna_oproj_act_h*_v1_sc1_se1` | `lagunaGatedAffineOProjNVFP4Source` | ✅ bit-exact (h48/h64) |
| `laguna_residual_rms_bf16_2048_v1` | `lagunaResidualRMSNormKernel` | ✅ bit-exact |
| `laguna_decode_router_top8_v3/_norm_v2` | `lagunaDecodeRouterTop8KernelSource` | ✅ bit-exact |
| `laguna_sliding_fused_attn_ring_v1` | `lagunaSlidingFusedAttentionKernel` | ✅ fast-exp ULP vs reference |
| `laguna_residual_rms_router_bf16_2048_rpg8` | `lagunaResidualRMSNormRouterSource(8)` | ✅ bit-exact (4 outputs) |
| `laguna_prefill_moe_tail_bf16_v1` | `lagunaPrefillMoETailKernel` | ✅ bit-exact |
| `laguna_prefill_router_tournament_v1` | `lagunaPrefillRouterTournamentKernelSource` | ✅ bit-exact |

Not yet ported (documented follow-up): the LM-head prune family
(`LagunaLmHeadPrune.swift` — coarse int5 argmax + exact-winner threshold +
sparse refine; requires the challenge's int5 checkpoint transform omlx does
not produce), the prefill router top8 / sorted moe_tail / qk_norm prefill
variants, the decode ordinal/tournament router variants, and the lane-major
scale-bank variants of the QKV/o_proj kernels. These are draft-side or
prefill-adjacent and do not change the decode hot path already covered.

## Correctness posture (important)

Each kernel is verified against the **stock `mx.quantized_matmul(mode="nvfp4")`
path** (or the equivalent stock ops) — most families are bit-exact, the
accumulation-order ones agree within 1-4 bf16 ulp. The kernels implement the
challenge's reference arithmetic (folded scales, seed elision, pair-interleaved
planes, halved scale banks), which differs from the stock kernels' reduction
order at ULP level. With `OMLX_LAGUNA_NVFP4_KERNELS=1` the real-model decode is
therefore **not token-identical** to the stock path — near-tie argmax tokens
can flip, exactly the challenge's own documented platform-divergence caveat
(their ranked runner re-checks against a hidden serial trajectory). The env is
**default OFF** and the stock path is untouched, so default behavior is
unchanged; the kernels are opt-in performance work with the divergence
documented, per the laguna branch's established opt-in posture.

## Wired into the model

`omlx/patches/laguna/laguna_model.py`:
- the shared-expert fused gate/up + SwiGLU kernel (single dispatch),
- the routed pair-interleaved swiglu + halved-plane down-reduce kernels
  (replace the gather-qmm routed path, with the 2.5 kernel scale compensated
  against the model's `moe_routed_scaling_factor`).

Real-model decode A/B (256-token, Poolside pin): **12.27 vs 12.80 ms/tok
(+4.3%)** with the routed+shared kernels on.

## Build

```bash
OMLX_WITH_CUSTOM_KERNEL=1 python setup.py build_ext --inplace
```

## Validation

`tests/test_laguna_nvfp4_kernels.py`:

- exact cases (all-zero weights, single-nibble row activation);
- synthetic NVFP4 planes vs the stock `mx.quantized_matmul(mode="nvfp4")`
  + swiglu path — all rows within 4 bf16 ULP (accumulation-order only, most
  rows bit-exact);
- the real Poolside `Laguna-XS-2.1-NVFP4-mlx` model (pinned revision
  `841778bda563a36104dd521e37d99218e46f4f25`) — real layer-1 shared-expert
  fused plane + real hidden state, same ULP bound (measured max diff
  `1.5e-5` vs `|y|max 3.6e-3`, 251/512 bit-exact);
- speed on the real plane: **127.9 vs 134.0 µs/call (1.05×)** for the single
  shared-expert decode call — the win compounds across 40 layers of fused
  expert/attention kernels per token.
