# Qwen4-Exp FP16 compute-path work

Status: draft, partial implementation. This is **not** a declaration that an
unmodified oMLX release can serve Flash-Next FP16 with all fused paths and MTP.

## Problem and intended outcome

A Qwen3.8-Flash-Next oQ4e checkpoint can retain its packed quantized weights while
using FP16 floating tensors. Several Qwen4-Exp serving paths still assume BF16:
PLE row decoding casts back to BF16, HC optimized-path eligibility checks reject
FP16, and fused GDN decode/verify require a separate dtype and numerical audit.
Changing only `text_config.dtype` does not change those engine operations.

The intended outcome is dtype-preserving FP16 serving with packed weights, PLE
mmap, hyper-connections (HC), QSA/KV state and Lightning MTP retained. FP32
accumulation and recurrent state should remain FP32 where the model requires it.
The target hardware for whole-model acceptance is M2 Ultra, 128 GB.

## Current implementation boundary

The PLE change uses the **loaded shared `weight_scale.dtype`** as the output
compute dtype for mmap rows, including prefetched rows, heterogeneous shard
layouts and empty requests. FP8 rows decode directly at that dtype in both
resident and mmap storage. Packed weights and affine quantization metadata are
not rewritten. Existing BF16 defaults remain unchanged.

The caller must load the intended shared scale. Inferring or synthesizing an
FP16 scale when a checkpoint omits it is not implemented here. This narrow
change does not establish uniform dtype across the entire model, nor does it
convert a checkpoint automatically.

The test fixture saves small real safetensors and compares selected rows with
MLX dequantization, covering BF16 and FP16, dense and affine storage, heterogeneous
group sizes, prefetch hits, duplicate indices and empty input. It also checks
that source tensors remain unchanged and exercises resident/mmap FP8 decoding.

HC fused decode and prefill eligibility now accepts uniform FP16 or BF16
activations, norm weights and affine scales/biases, including QuantizedLinear
subclasses. FP16 kernel errors propagate instead of silently continuing through
another implementation. The exact hybrid projection helper also accepts FP16;
its model-level `language.py` dispatch remains BF16-only in this draft. No broad
FP16 rejection has been added to canonical prefill or target verification.

## Verification on this branch

Base: `4e5308cc` (current main when the branch was created), tested on 2026-09-23
with Python 3.11.15, MLX 0.32.2, mlx-vlm `ea79808c`, mlx-lm `872ae88d`, and
transformers 5.17.0 in an isolated environment installed from `.[dev]`.

- PLE regression file: **16 passed**.
- HC fused/projection files: **115 passed**. Fused comparisons use stated
  tolerances against canonical FP16 operations; they do not claim bit-exactness.
  Exact hybrid helper tests compare exact results at 4/5/6/8 bits.
- Existing compatibility file: **67 passed, 2 failed**. Both failing cases are
  `test_qwen4_small_hyper_connection_fusion_fails_closed[False/True]`, whose tiny
  FP32 results differ at the exact-equality assertion. The same two failures
  reproduced on untouched base source with the same dependencies.
- New PLE tests against untouched base: **7 passed, 9 failed**. Seven failures
  expose existing dtype behavior (FP32 promotion or BF16 output); two exercise
  the newly introduced FP8 dtype argument, which is absent on the base.
- Python syntax, new PLE test lint and whitespace checks passed.

HC tests run the MLX Metal kernels. These are component tests, not a full native
app build or a full-model serving benchmark. The complete repository suite has
not been run.

## Why the old GDN implementation is not included

A local dev2 experiment changed HC, PLE and GDN for FP16. The dev4 migration did
not achieve real-model acceptance: requests encountered an overly broad decode
predicate, reclassified projections, and finally a fused GDN verify rejection.
Unit tests and successful app packaging did not establish serving correctness.

Current upstream work must be preserved:

- [#3760](https://github.com/jundot/omlx/pull/3760) accepts quantized-linear
  subclasses introduced by the prefill projection wrapper.
- [#3776](https://github.com/jundot/omlx/pull/3776) implements the Qwen4 **L2** verify
  prework with BF16 rounding semantics, independently of Qwen3.5 RMS normalization.
- Commit `d1889871` resolves Qwen4 normalization lazily, avoiding registration of
  the upstream module before the vendored module can be installed.

The old FP16 GDN code used the earlier RMS-based verify seam and retained exact
class checks. One decode gate comparison allowed `rtol=atol=5e-4`; it was not
bit-exact evidence. Replaying that diff would reintroduce known defects and
cannot establish FP16 L2 correctness on current main.

[#3853](https://github.com/jundot/omlx/pull/3853) is complementary work for
Qwen3.5/3.6 FP16 decode on M1 Max. Its measurements do not cover Flash-Next on
M2 Ultra or Qwen4 MTP verification.

## Historical motivation, not current-branch performance claims

A local 2026-09-18 experiment based on `v0.7.0.dev2` (`b390b31e`, app build 2657)
compared BF16 and FP16 compute checkpoints on M2 Ultra / 128 GB, MLX 0.32.2.
The FP16 copy was built from `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` using the
repository's `tools/clone_mlx_model_fp16.py` workflow, retaining the original.
Its recorded tensor inventory was 2,725 FP16, 1,020 U32, three I64, including
76 MTP tensors. This was a packed oQ checkpoint, not a dense FP16 dequantization.

The recorded workload was the oMLX external-endpoint throughput benchmark,
Code/Python corpus, 128 output tokens, adaptive Lightning MTP and cold prefix
cache (`cached_tokens=0`).

| Requested prompt tokens | BF16 actual / FP16 actual | BF16 prefill tok/s | FP16 prefill tok/s | BF16 decode tok/s | FP16 decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 4,096 | 4,407 / 4,406 | 451.7 | 527.7 | 42.5 | 38.6 |
| 32,768 | 30,950 / 30,951 | 585.3 | 805.9 | 36.6 | 42.1 |
| 65,536 | 62,029 / 62,029 | 578.9 | 796.8 | 33.9 | 38.3 |

Limitations: these are historical local records, not a fresh balanced A/B of
this PR. Actual token lengths differ by one at two points, MTP acceptance varied,
and 64K runs parked MTP adaptively. The 4K decode result regressed. Raw benchmark
artifacts and immutable model hashes are not included, so the table establishes
motivation only; it cannot serve as reproducible acceptance for this branch.
The earlier successful text/image smoke also applies only to the dev2 experiment.

## Reproduction and completion criteria

For the current PLE regression tests on Apple Silicon:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_qwen4_ple_compute_dtype.py -q
python -m pytest tests/test_mlx_vlm_qwen4_exp_compat.py -q
python -m pytest tests/test_qwen4_hc_fused.py tests/test_qwen4_hc_projection.py -q
```

Before representing complete Flash-Next FP16 support as ready:

- Establish a current-main baseline and record exact source/dependency revisions,
  checkpoint configuration, tensor hashes and conversion command.
- Audit missing-scale synthesis and loaded text/image embeddings without assuming
  every floating tensor should be FP16.
- Validate HC outputs against current canonical FP16 operations and retain BF16
  regression coverage, including wrapped projections and kernel failure behavior.
- Implement FP16 GDN decode and Qwen4 L2 verify against the current verifier, with
  intentional FP32 recurrent state. Test numerical output and cache transitions,
  rather than only widening dtype or class eligibility predicates.
- Exercise prefill, single-token decode, multi-token verify, batch execution,
  partial/rejected drafts, restored prefix caches and failure rollback.
- Build matching native kernels and complete real target-model text/image requests
  through a candidate server. Record actual PLE/HC/GDN/QSA/MTP engagement.
- Run a reproducible balanced whole-server BF16/FP16 comparison on the same source,
  corpus, prompt lengths, output length, cache state and MTP policy. Report PP,
  decode, TTFT, end-to-end latency, memory and MTP acceptance, including regressions.

No candidate app has been deployed for this PR, and no current-branch full-model
benchmark or end-to-end FP16/MTP acceptance is claimed.
