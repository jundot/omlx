# Qwen4-Exp FP16 compute-path work

Status: implementation candidate for review on
[#3873](https://github.com/jundot/omlx/pull/3873), reimplemented from oMLX
`v0.7.0rc1` and then rebased onto current main. Component coverage is complete
for the PLE, hyper-connection (HC), and fused GDN seams described below. This is
not a claim that the candidate has replaced an installed app or passed
whole-model serving and performance acceptance.

## Intent and release identity

A Qwen3.8-Flash-Next oQ4e checkpoint can keep its packed quantized weights while
using FP16 floating tensors. Several Qwen4-Exp serving paths previously assumed
BF16: PLE row decoding cast back to BF16, HC optimized dispatch rejected FP16,
and fused GDN prefill/decode/verify required a uniform-dtype and numerical audit.
Changing only `text_config.dtype` did not change those engine operations.

The intended result is dtype-preserving FP16 compute with packed weights, PLE
mmap, HC, QSA/KV state and Lightning MTP retained. FP32 accumulation and
recurrent state remain FP32 where the model requires them. Whole-model
acceptance targets an M2 Ultra with 128 GB.

This rewrite uses the following immutable source and donor identities:

- Source tag: `v0.7.0rc1`
- Source commit: `35be079d8a86a44dc2c6d485fbfbf43754e66298`
- PR base at verification: `f0d8428acd3220c364177d1ea9593e4e15f94107`
- Official donor: `oMLX-0.7.0rc1-macos26-27.dmg`, app build 2849
- Donor SHA-256:
  `82c1ea4d882153bb2da5cd2793e950620b2d2eb81b2e90695272e878be79b83a`

The donor image is used only to supply release runtime layers for a separately
built candidate. It does not authorize replacing or restarting an installed
oMLX app.

## Implemented boundary

### PLE

PLE uses the loaded shared `weight_scale.dtype` as the output compute dtype for
resident and mmap rows, including prefetched rows, heterogeneous shard layouts,
duplicate indices, and empty requests. FP8 rows decode directly at that dtype.
Packed weights and affine quantization metadata are not rewritten.

When an older checkpoint omits the shared scale, sanitization now creates the
unit scale in the configured text dtype: FP16 for `float16`/`fp16`/`f16`, and
the existing BF16 default otherwise. A converted checkpoint's real runtime
scale remains authoritative and is not overwritten.

### Hyper-connections

HC fused decode and prefill eligibility accepts uniform FP16 or BF16
activations, norm weights, and affine scales/biases, including
`QuantizedLinear` subclasses. Exact-hybrid model dispatch now admits FP16 as
well as BF16. FP16 kernel errors propagate; the request is not silently retried
through another implementation after native execution was intended.

Canonical multi-token prefill, target verification, MTP-enabled calls, and
non-production geometries retain their existing routes. An integration test
asserts that production-geometry FP16 decode actually enters the native hybrid
projection rather than merely passing a helper-level compatibility check.

### Fused GDN

Qwen4 fused GDN prefill, single-token decode, and multi-token L2 verification
accept a uniform FP16 or BF16 compute dtype across the relevant input,
projection, normalization, parameter, affine-scale, and convolution-cache
boundaries. The rewrite preserves current main's FP16/BF16 Qwen3.5 RMS verify
and fused replay path introduced after rc1.

The implementation preserves FP32 recurrent state intentionally. FP16
projection comparisons use a bounded `5e-4` tolerance where MLX operation
ordering produces expected rounding differences; cache transitions and the
remaining outputs are checked exactly. Real Qwen4 FP16 calls that meet the
operation shape but fail a fused contract raise a bounded error rather than
falling back after partial work. Multi-token prefill is kept distinct from
single-token decode so it cannot be misclassified by the old broad predicate.

## Verification on this rewrite

Environment: macOS/Apple Silicon, Python 3.11.15, MLX 0.32.2, using the isolated
development environment already associated with the PR worktree.

The focused command covers PLE, HC fused/projection, GDN prework, and the
vendored Qwen4-Exp compatibility overlay:

```bash
python -m pytest -q \
  tests/test_qwen4_ple_compute_dtype.py \
  tests/test_qwen4_hc_fused.py \
  tests/test_qwen4_hc_projection.py \
  tests/test_qwen35_gdn_prework.py \
  tests/test_mlx_vlm_qwen4_exp_compat.py
```

On the untouched rc1 source, two pre-existing
`test_qwen4_small_hyper_connection_fusion_fails_closed[False/True]` exact-FP32
equality assertions reproduce with approximately `1e-6` differences. Current
main has since corrected that baseline. The focused result after rebasing this
rewrite onto current main is **327 passed**.

The default repository suite, run after moving build-in-place optional native
binaries out of the source tree to match a clean upstream checkout, is
**15,045 passed, 384 skipped, 86 deselected**.

### Candidate packaging

An isolated `0.7.0rc1 (2870)` candidate was built from the rebased source with
the verified official rc1 donor and `--with-custom-kernel`. The build used a
CPython 3.11 environment matching the donor ABI. The build-time bundled-MLX ABI
probe passed, all five packaged custom-kernel extension modules imported from
the candidate bundle, and `codesign --verify --deep --strict` passed. The
embedded GDN source SHA-256 matched the committed source.

Running the repository suite while the build-in-place binaries were present
executed optional native tests that a clean checkout skips. Three decode-fast
SDPA cases and one GLM native exactness case failed; none of the affected source
or test files differs from current main. The generated binaries were moved to a
recoverable artifact directory before the clean full-suite run. This is an
explicit current-main/custom-kernel acceptance gap, not evidence that those
optional kernels passed functional validation.

The candidate was not launched, installed, or copied over `/Applications/oMLX.app`.
Whole-model runtime acceptance remains a separate layer.

## Historical motivation, not current performance claims

A local 2026-09-18 experiment based on `v0.7.0.dev2` (`b390b31e`, app build
2657) compared BF16 and FP16 compute checkpoints on M2 Ultra / 128 GB, MLX
0.32.2. The FP16 copy retained the original packed oQ checkpoint. Its recorded
tensor inventory was 2,725 FP16, 1,020 U32, three I64, including 76 MTP tensors.

The workload was the oMLX external-endpoint throughput benchmark, Code/Python
corpus, 128 output tokens, adaptive Lightning MTP, and cold prefix cache.

| Requested prompt tokens | BF16 actual / FP16 actual | BF16 prefill tok/s | FP16 prefill tok/s | BF16 decode tok/s | FP16 decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 4,096 | 4,407 / 4,406 | 451.7 | 527.7 | 42.5 | 38.6 |
| 32,768 | 30,950 / 30,951 | 585.3 | 805.9 | 36.6 | 42.1 |
| 65,536 | 62,029 / 62,029 | 578.9 | 796.8 | 33.9 | 38.3 |

Those records motivate the work but are not a fresh balanced A/B of this
rewrite. Actual token lengths differed by one in two rows, MTP acceptance
varied, 64K runs parked MTP adaptively, and 4K FP16 decode regressed. They do
not establish reproducible performance acceptance for this PR.

## Remaining whole-model acceptance

Before claiming production-ready Flash-Next FP16 serving:

- Run text and image requests against the target packed checkpoint through an
  isolated candidate server, without replacing the installed app.
- Record actual PLE, HC, GDN, QSA, cache, and MTP engagement for prefill,
  single-token decode, multi-token verification, partial/rejected drafts,
  restored prefixes, and rollback on failure.
- Run a balanced BF16/FP16 comparison on identical source, corpus, lengths,
  cache state, output length, and MTP policy. Report prefill, decode, TTFT,
  end-to-end latency, memory, and MTP acceptance, including regressions.

No installed app replacement, consumer migration, or current-branch full-model
benchmark is claimed by this document.
