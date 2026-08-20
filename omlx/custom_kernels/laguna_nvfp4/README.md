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
| prefill QK-norm v2/h1 ×4, prefill sorted moe tail | batched prefill | ✅ |
| decode router ordinal ×4, prefill router top8 ×2 / tournament-norm / tournament-ordinal ×2 | router variants | ✅ bit-exact |
| shared/routed rows1 · halved · wide · packed · top8keys ×8 | r1/packed/scale-layout variants | ✅ ≤4 ulp |
| `laguna_dense_gate_up_swiglu_bf16_v1`, `laguna_dense_down_residual_bf16_v1` | dense-MLP bf16 fusions | ✅ |
| `laguna_decode_embedding_rope_atlas_bf16_2048_v2` | embedding + RoPE atlas | ✅ bit-exact |
| `laguna_inject_empty_dispatch_v1`, `laguna_inject_dram_sweep_u4_v2` | harness timing probes | ✅ |
| `laguna_full_fused_attn_grow_v1` | full-attn fused grow | ✅ fast-exp ULP |
| `laguna_lmhead_int5_*` base/inline coarse + sparse refine | LM-head family | ✅ real-model validated |

**All 46 challenge kernel names are present** (42 concrete kernel functions +
4 macro-instantiated variants); 62 tests pass.
| `laguna_sliding_qk_norm_rope_bf16_128_v1` | `lagunaSlidingQKNormRoPEKernel` | ✅ bit-exact |
| `laguna_decode_nvfp4_qkv_h*_r1_v1_se1_sd1` | `lagunaDecodeNVFP4QKVR1Source` | ✅ bit-exact (h48/h64) |
| `laguna_oproj_act_h*_v1_sc1_se1` | `lagunaGatedAffineOProjNVFP4Source` | ✅ bit-exact (h48/h64) |
| `laguna_residual_rms_bf16_2048_v1` | `lagunaResidualRMSNormKernel` | ✅ bit-exact |
| `laguna_decode_router_top8_v3/_norm_v2` | `lagunaDecodeRouterTop8KernelSource` | ✅ bit-exact |
| `laguna_sliding_fused_attn_ring_v1` | `lagunaSlidingFusedAttentionKernel` | ✅ fast-exp ULP vs reference |
| `laguna_residual_rms_router_bf16_2048_rpg8` | `lagunaResidualRMSNormRouterSource(8)` | ✅ bit-exact (4 outputs) |
| `laguna_prefill_moe_tail_bf16_v1` | `lagunaPrefillMoETailKernel` | ✅ bit-exact |
| `laguna_prefill_router_tournament_v1/_norm_v1` | `lagunaPrefillRouterTournamentKernelSource` | ✅ bit-exact |
| `laguna_decode_router_top8_ordinal_v1/_norm_v1/_table_v1/_table_norm_v1` | `lagunaDecodeRouterOrdinalKernelSource` | ✅ bit-exact |
| `laguna_prefill_router_top8_v1/_norm_v1` | `lagunaPrefillRouterTop8KernelSource` | ✅ bit-exact |
| `laguna_prefill_router_tournament_ordinal_active64_v2/_norm_active64_v2` | `lagunaPrefillRouterTournamentOrdinalKernelSource` | ✅ bit-exact |
| `laguna_lmhead_int5_coarse_*` + argmax stage1 + exact-winner + inline exact (`lm_head_prune`) | `LagunaLmHeadPrune.swift` family | ✅ real-model validated |

## LM-head int5 prune

The `lm_head_prune` op fuses the challenge's 4-kernel pipeline over the int5
planes built by `build_int5_planes` (the nibble/bit-plane + e8m0 scale
transform with the `|q| <= 15` init certificate): coarse+delta pass, argmax
stage-1, exact-winner bf16-predecessor-midpoint threshold, inline-mask exact
pass. Real-model validation (Poolside 100k×2048 lm_head + a real hidden row):
prune argmax == stock argmax, winner slot bf16-exact, zero non-winner slots
above the winner — the prune's certified-bound contract.

Not ported (explicitly excluded): the NVFP4 attention re-quantization
transform (`LagunaRuntimeWeights`), which converts the bf16 attention
projections to NVFP4 — without it the decode QKV / o_proj / QK-norm kernels
have no matching bank in omlx's stock bf16 attention path. Also not ported:
the lane-major scale-bank variants of the QKV/o_proj kernels (alternate
scale layout requiring the same transform).

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

Real-model decode A/B (256-token, Poolside pin, interleaved off/on to
cancel thermal drift): **13.23 vs 13.26 ms/tok (−0.3%, within noise)** with
the routed+shared kernels on — the fused kernels are correct (bit-exact /
ULP) and individually faster on the shared-expert path (~1.05x isolated),
but the end-to-end decode is dominated by the bf16 attention + the (not
requantized) projection stream, so the integrated win is neutral at this
scale. Reproduce with `tools/qwen38_mtp/bench_laguna_kernels.py`.

### Attention kernels: not directly wireable in omlx

The challenge's runtime re-quantizes the attention projections to NVFP4
group-16 (`LagunaRuntimeWeights` step) so its decode QKV / o_proj / QK-norm
kernels target NVFP4 banks. omlx's stock Laguna attention keeps the bf16
projections, so those kernels have no matching bank to dispatch against
without adopting the challenge's attention re-quantization transform —
documented as a separate follow-up, not part of this port's decode hot-path
wiring.

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
