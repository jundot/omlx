# DS4 Q-A/KV/compressor projection bundle

This is the source contract for an oMLX-native port inspired by ds4-metal's
`kernel_dsv4_qkv_pair_quad_compressor_store_q8_0`. It changes no production
dispatch.

## Exact bundle

Every projection below reads the same BF16 output of the layer's attention
RMSNorm. The real 43-layer checkpoint contains two local layers, 21 ratio-4
layers, and 20 ratio-128 layers.

| projection | local rows | ratio-4 rows | ratio-128 rows | oMLX storage |
|---|---:|---:|---:|---|
| `wq_a` | 1024 | 1024 | 1024 | packed MXFP8 U32 + group-32 U8 scales |
| raw `wkv` | 512 | 512 | 512 | packed MXFP8 U32 + group-32 U8 scales |
| compressor `wkv` / `wgate` | - | 1024 each | 512 each | BF16 |
| index-compressor `wkv` / `wgate` | - | 256 each | - | BF16 |

The released MLX checkpoint already stores the runtime packed layout. Source
and runtime bytes are 6,488,064 per local layer, 27,459,584 per ratio-4 layer,
and 14,876,672 per ratio-128 layer: 887,160,832 bytes (846.0625 MiB) over all
layers. APE and norm tensors are not projection weights; all APE tensors add
another 5,672,960 checkpoint bytes.

The current path launches 210 projection kernels per model pass and rank:
`2*2 + 21*6 + 20*4`. A full bundle is 43 dispatches, saving 167 (79.5%). TP2
duplicates those launches on both ranks. It shards downstream `wq_b` and query
heads, but leaves every tensor in this bundle and both compressor caches
replicated. Therefore the bundle takes no group argument, changes no
collective, and runs identically on each rank.

`indexer.weights_proj` also reads the same input, but it is deliberately not in
the first bundle. Cold M=1024 ratio-4 prefill produces only 256 pooled rows, so
that conditional projection is not used; decode uses it only after the pool
exceeds top-k 512. It also has a different small-output reduction schedule.

## Rounding frontier

The DwarfStar kernel cannot be copied verbatim. It reads Q8_0 and FP32 input,
stores FP32 projection outputs, adds APE into FP32 compressor state, and later
narrows its cache. oMLX reads BF16 input, uses MXFP8 Q-A/raw-KV weights and BF16
compressor weights, accumulates each projection independently in FP32, and
stores every projection output once to BF16.

More importantly, oMLX caches the raw BF16 compressor gate. Ratio-4 later casts
APE to BF16 before adding it; ratio-128 promotes the BF16 gate to FP32 and adds
F32 APE. Moving DwarfStar's APE/state-store epilogue into the projection kernel
would cross both rounding frontiers. The first primitive must return projection
views only and let the existing `Compressor.consume` mutate caches.

## First native ABI

Implement only this fixed symbol first:

```text
deepseek_v4_qkv_compressor_bundle_b1(
    x_bf16[1,4096],
    wq_a_u32[1024,1024], wq_a_scale_u8[1024,128],
    wkv_u32[512,1024],   wkv_scale_u8[512,128],
    compressor_wkv_bf16[1024,4096],
    compressor_wgate_bf16[1024,4096],
    index_compressor_wkv_bf16[256,4096],
    index_compressor_wgate_bf16[256,4096],
) -> packed_bf16[1,4096]
```

Packed output slices are `[0:1024]` Q-A, `[1024:1536]` raw KV,
`[1536:2560]` compressor KV, `[2560:3584]` compressor gate,
`[3584:3840]` index KV, and `[3840:4096]` index gate. There is no APE,
position, cache, or distributed argument. Each bank must retain stock MLX's
per-row K walk/reduction and one BF16 store; only dispatch packaging and shared
activation loads may change.

The first isolated implementation uses one MLX primitive and command encoder
but queues two Metal dispatches: an unequal-row MXFP8 Q-A/raw-KV bank and a
four-buffer BF16 dense bank. Both write directly into the packed output; there
is no host boundary or intermediate concatenation. The native diagnostic
`deepseek_v4_qkv_compressor_bundle_b1_dispatches()` reports `2`. No production
model path references either symbol.

### Fixed B1 result (2026-08-22)

All six packed slices are array-equal to both separate stock projections and
the exact grouped-stock baseline on M3 Ultra and M5 Max. The implementation
does not pass the 1.05x performance gate:

| Host | Native (2 dispatches) | Grouped stock (2) | Separate (6) | Speedup vs faster baseline |
|---|---:|---:|---:|---:|
| M3 Ultra | 0.2474 ms | 0.2372 ms | 0.2435 ms | 0.959x |
| M5 Max | 0.3891 ms | 0.3707 ms | 0.3838 ms | 0.953x |

The symbol therefore remains isolated and unused. Machine-readable results
are in `ds4_qkv_compressor_bundle_b1_results_2026-08-22.json`.

## Gates

The B1 gate compares all six slices with `mx.array_equal`, then compares
ratio-4 cache remainder, previous-window state, pooled rows, layer output, and
logits across positions modulo 4 = 0, 1, 2, 3. Run reference/candidate/
candidate/reference (ABBA) against the faster of separate stock projections
and the safe grouped stock bound. Require at least 1.05x on M3 Ultra and M5
independently. Repeat local and TP; both ranks must agree and collective count
and order must be unchanged.

The next gate is fixed B=1, M=1024. It again requires six array-equal projection
boundaries, full cold-prefill cache/logit equality, and at least 1.05x on each
machine. Do not assume arbitrary concatenation is exact: on M3 Ultra the
ratio-4 1024-row pairs and 256-row pairs are array-equal when grouped, but the
ratio-128 512-to-1024 BF16 concat changes MLX's GEMM reduction geometry.

### Lossless M=1024 candidate

The ratio-4 M=1024 candidate now exposes the only grouping already proven
array-equal: original packed MXFP8 Q-A plus raw-KV, original BF16 main
compressor KV/gate, and original BF16 index-compressor KV/gate. M3 performs
three stock-arithmetic dispatches. M5 performs four because its main 1024-row
BF16 pair must remain two GEMMs: concatenating that pair changes its reduction
geometry, while Q/KV and the index pair remain exact. Both restore the same
six views. The candidate never
dequantizes, converts to INT8, or requantizes an affine suffix.

The M5 Q/KV bank uses the qualified NAX variant 5 tile
(BM128/BK64/BN64/WM4/WN2); all original U32 codes and U8 scales pass directly
to the same MLX MXFP8 implementation. If that symbol, NAX metallib, or device
capability is absent, the route fails closed before packing any weights.

On first qualified use, each source module is rebound to a row view of its
canonical bank. Consequently there is no steady-state duplicate weight copy;
the normal fallback still reads the exact source rows. The route is restricted
to BF16 `[1,1024,4096]`, the exact released DS4-Flash ratio-4 fingerprint,
Apple M3 Ultra or M5 Max, and either single-node execution or a rank-local TP2
model. It adds no collective. `OMLX_DSV4_QKV_BUNDLE_PREFILL=0` is the rollback
and remains the cluster default until full cold-prefill gates clear on both
machines.

Run the checkpoint audit and available stock-grouping ABBA probe with:

```bash
python benchmarks/bench_ds4_qkv_compressor_bundle.py \
  --model /path/to/DeepSeek-V4-Flash-0731-FP8 --layer 2 --rows 1,1024
```

The machine-readable ABI, byte ledger, dispatch ledger, and gate are in that
benchmark module and pinned by
`tests/test_ds4_qkv_compressor_bundle_contract.py`.
