# Upstream packaging verification

The first two sections retain the historical packaging and legacy-runtime
records. Native current-main validation is recorded separately below.

- The vendored bridge archive: 264 tests pass.
- Six added transport regressions fail against the uncorrected diagnostic, then
  pass with the already supplied negative-stride/UINT64 fix. Restrictions are
  enforced by a CPU test double; no GPU execution is claimed.
- Eight default-installer tests fail before implementation and pass afterwards.
  These check path quoting, shell syntax, forwarding, safe symlink replacement,
  independent service identity and installer help without MLX.
- Complete contributed suite: 289 tests pass; no skips, failures or errors.
- 99 Python files compile and parse using Python 3.11 grammar.
- Production bridge sources are byte-identical to the vendored bridge archive
  except the files listed in `SOURCE-MANIFEST.json`
  (`runtime_changes_from_archive`): `device_selftest.py` (both previously
  confirmed fixes), `scheduler_compat.py` (non-JANGH2 VLM-native batch
  conversion plus `_wrap_caches` mixed-list passthrough) and
  `tests/test_scheduler_cache_compat.py` (regression tests for that change).
- The scheduler fallback is verified by CPU test doubles that mirror
  mlx-lm 0.31.2 `to_batch_cache()` and the mlx_vlm 0.7.8 batch cache classes;
  live mixed-model serving still needs Mac hardware acceptance.
- The optional installer does not upgrade packages, rewrite active bridge code,
  move model files, read API keys or automatically start/stop a process.

The ordinary Git diff is add-only and is round-trip checked in a fresh local Git
repository, then the contributed tests are rerun there. A full network clone of
upstream could not be obtained in this container; no claim is made that its full
existing test suite ran. Current-main interfaces were inspected via upstream
source, and they differ from the legacy private hooks. This is an RFC, not a
verified native current-main integration.

No macOS launchd process was started here. The installer emits a plist whose
schema and payload were round-tripped with plistlib; actual login service and
mixed-model inference need verification on the working Mac.

The source manifest records the two input artifacts and contributed source
hashes. Earlier user-supplied real-Mac results are useful reported evidence, but
not a substitute for these explicit test boundaries.

## Mac regression verification

Executed on Apple M5 Max with Python 3.11.17, MLX 0.32.4, MLX-LM 0.31.2,
MLX-VLM 0.7.8 and the existing PR #364 oMLX 0.3.0rc1 runtime.

- Complete contributed CPU suite: 295 tests pass.
- The original Allosaurus failure was reproduced at relative L2
  0.056308435 and maximum absolute logit error 1.817502141. Disabling device
  caching reproduced the same error; repeated reference execution was exact.
- Matching FP32 expert scale placement and summation order, and using MLX's
  precise exponential, made all three fixed-prefix/cached-continuation cases
  exact: relative L2, maximum error and KL divergence are zero, with 100%
  top-token agreement over 36 sampled positions. Acceptance limits are unchanged.
- The complete `prepare.py --long-profile` workflow passed qualification,
  tuning, whole-model comparison, and fresh-cache 1024/4096-token profiling.
  Qualification includes 24 exact FP32 reference-alignment cases across
  decode, sorted prefill, grouped tiles, row tuning, mixed bits and rotations.
- Standalone long profiling measured about 249/212 prefill tokens per second
  and 32/30 decode steps per second for 1024/4096-token prompts respectively.
  These are standalone timings, not concurrent HTTP-server throughput.
- Real Qwen3.6-35B-A3B-4bit and Qwen3.8-27B-MLX-4bit checkpoints passed through
  the PR VLM adapter and boundary-snapshot batch generator with bridge hooks
  active: chunked singleton prefill/decode, unequal-length batching/filtering,
  and cached continuation. Generated token counts matched and log probabilities
  were finite. The native hybrid-cache fix preserves offset arrays and cache
  list identity.
- A real BatchGenerator smoke check reproduced the cleanup stream ownership
  failure before the lifecycle fix. Closing on its MLX worker during scheduler
  shutdown cleared the wired-memory marker and allowed later main-thread
  destruction without the stream error.

This acceptance run did not test launchd, HTTP tool calling, image inputs,
mixed-model concurrent serving or every discovered checkpoint.


## Native oMLX integration (25 October 2026)

Executed on Apple M5 Max (128 GiB) with oMLX 0.7.3.dev1, MLX 0.32.4,
MLX-LM commit `0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d` and MLX-VLM
commit `4319ef2c837b7226b758c3dd69edda51b7079785`, in an isolated runtime.
The existing PR #364 runtime and its packages were retained.

- Native dispatch uses the existing custom-quantization loading seam and VLM
  engine. No bridge scheduler, stream or chat hooks are installed. Strict
  checkpoint audit precedes the vendor's exact projection fusion. PLE mappings
  and MTP selection are model-local.
- All three fixed 64-token prefix / four-token cached-continuation cases match
  the FP32 expert reference exactly at all 36 sampled positions: zero relative
  L2, maximum absolute error and KL divergence, with 100% top-token agreement.
  The acceptance thresholds were not changed.
- The normal VLM engine passed direct generation, unequal concurrent requests,
  thinking-enabled streaming and teardown. The ordinary `omlx serve` HTTP path
  passed a direct answer, two concurrent requests, reasoning/content streaming,
  deferred unload/reload and clean shutdown on an isolated port. Exactly two
  native loads occurred across that test, confirming request reuse. A fresh-server MTP auto-default bug
  found by this check was corrected at the existing capability/default seams.
- Saved MTP flags are normalized on effective runtime settings and engine reuse
  signatures. JANGH2 cannot advertise or enable unsupported packed MTP through
  the admin API. Ordinary Qwen4 MTP behavior has dedicated regressions.
- Normal generation_config.json EOS/sampling defaults are loaded without allowing
  the sidecar merge to override audited architecture or quantization settings.
- The final combined regression run passed 850 tests covering native loading,
  model settings, engine-pool admission/reuse, cache handling, MTP and optional
  thinking. Separately, all 297 contributed CPU tests passed. Eleven opt-in
  template tests passed against the actual local Allosaurus tokenizer.
- The wheel contains all 14 native package files, including its MIT license and
  provenance notice. Its format detector imports without MLX or contrib files.

The packed-kernel optimization reuses identical rotated inputs and hoists row
scales out of the column loop without changing FP32 multiplication, accumulation
or reduction order. Nine paired real 512-expert component cases were bit-exact.
At default 4/4 row geometry, preliminary component speedups were 7–13%.

The following full-model measurements were taken with the existing server
stopped. Each row used the same native model, five interleaved before/after
pairs, fresh caches and 32 fixed teacher-token decode steps. The before version
is the packed kernel immediately before the scale/input reuse optimization;
these are not comparisons between different oMLX releases.

| Prompt tokens | Prefill before (tokens/s) | Prefill after (tokens/s) | Change | Decode before / after (steps/s) |
| --- | ---: | ---: | ---: | ---: |
| 256 | 289.1 | 315.5 | +9.2% | 32.1 / 31.9 |
| 1024 | 254.2 | 279.5 | +9.9% | 26.5 / 25.8 |

Outputs were identical in every pair. Decode medians varied by a few percent
within the observed sample ranges, so no decode speedup is claimed. These are
standalone timings, not concurrent HTTP throughput. Native image inputs, HTTP
tool calling, mixed-model concurrency and cancellation under sustained load
were not part of this acceptance run.


The legacy `prepare.py --long-profile` workflow was rerun against the final
shared source after stopping port 8500: qualification, tuning, all three exact
whole-model comparisons and long profiling passed. It selected 4/2 row geometry
with legacy prefill. The 1024/4096-token profiles measured 275.1/228.7 prefill
tokens/s and 32.8/30.7 decode steps/s respectively. Existing reports were backed
up and replaced by genuine new runs; no fingerprints or acceptance limits were
edited to bypass checks.
