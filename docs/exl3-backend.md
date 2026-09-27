# Experimental packed EXL3 projection

Base oMLX: f0d8428acd3220c364177d1ea9593e4e15f94107.

The tile permutation, tail-biting pair funnel, MCG codebook and normalized
Hadamard stages follow Sushi commit 27ca1c8684c01d1970bf576d8cef80ebea617928,
`src/expert_exl3.zig` and `src/expert_exl3_kernels.zig`. Sushi describes these
as its own implementation of the ExLlamaV3 format. Adaptations here implement
MLX module integration and gathered SIMD projection. Retained Sushi notices:
[MIT licence](licenses/sushi-MIT.txt). No checkpoint Python is executed.

## Licence scope

Sushi's own EXL3 implementation is MIT. This permits modification and distribution
with its copyright and permission notices retained. We are not incorporating the
other Apache/BSD components enumerated in Sushi's NOTICE. If additional code is
ported, audit its individual provenance and preserve the applicable notices.

The inspected local Qwen3.8 Flash-Next pack carries Qwen Community License 1.0.
It explicitly permits use, modification, deployment and derivative works, with
notice retention and additional commercial MaaS/AI-work-assistant conditions.
Sam's noncommercial friends bot does not appear to trigger those commercial
conditions. Model weights are not included in this source contribution.
This is a licence-scope review of the supplied files, not a blanket certification
of any future use or third-party hosting service.

## Status and gates

This is NOT a production loader. MCG/svh packs, 128-wide Hadamard blocks,
and integer halfword rates from 32 through 64 are the supported prototype scope.
Native attention and cache types are unchanged. No full expert bank is expanded.
Invalid expert ids produce NaN without out-of-bounds scale access.

The independent CPU oracle passes synthetic rate/window fixtures and one real
640-to-2560 expert. The original serial kernel measured 5.312 ms/projection;
this is too slow for deployment. The SIMD prototype measured 3.584 ms/projection with real-expert cosine approximately 1.0 and maximum absolute difference 3.05e-5. It changes FP32 summation order. These are tiny projection microbenchmarks, not full-model performance.

Remaining gates: efficient many-token prefill, exact mixed trunk loading,
full-model PLE configuration and lifecycle,
vision and tool regression tests, memory admission and cache restoration.
Do not claim Sushi-pack support or open a support PR before these pass.

No live Herman service or model settings were changed by this prototype.

## Flat n-gram adapter results

A metadata adapter exposes the flat weight/scales/biases file as virtual row
shards to the existing DiskBackedShardedEmbedding implementation. It reuses
the native mmap/prefetch/affine row gather and close path. No table copy or
resident dequantization is performed. Format, geometry, dtype, bounds,
overlap, configuration agreement and directory containment are checked.

Eight unit tests pass (packed projections and flat-table validation). An
opt-in tiny native reader check confirms exact selected-row parity across
shards with repeated ids, and idempotent resource cleanup. The real table's
384 virtual descriptors validate from its header; tensor contents were not
read during this check. These results do not establish full-model correctness.

Reproduce with an Apple Silicon MLX Python environment:

```sh
python -m unittest -v tests.test_flat_ngram_views tests.test_exl3_reference
python -m tests.check_flat_ngram_native
python -m tests.check_real_expert
```

The last command requires Sam's local checkpoint path and copies just one
expert. Do not include its hardcoded local model choice in an upstream suite.

## Timebox conclusion

Licence review and arithmetic/storage feasibility succeeded. Complete oMLX
Sushi-pack support did not. EXL3 projection optimization, full strict mixed
loader integration, prefill benchmarking and native cache/vision/tool checks
are substantial remaining work. No load dispatch is installed, no full model
was loaded, and no support PR was opened. Do not turn this prototype on live.
