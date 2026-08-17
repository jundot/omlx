# Qwen3.5/3.6/3.8 ANE/GPU Prefill (Experimental)

This source-build experiment uses private AppleNeuralEngine APIs to split one
fixed-shape Qwen3.5/3.6/3.8 prompt across both ANEs and the GPU. Two INT8 programs,
pinned to physical ANE instances 1 and 2, compute disjoint output-channel
slices while Metal computes the remaining quantized channels. It is disabled
by default.

At the default 53% MLP request, alignment gives the two ANEs 26.5% of gate and
up channels each (52.9% total) and leaves 47.1% on GPU. A native merge applies
SwiGLU without materializing the full gate/up result. The GDN z+qkv input
projection is split 50/50 between the ANEs and GPU. The MLP down projection,
GDN recurrence, b/a and output projections, normalization, embeddings, and
logits remain on GPU.

## Requirements and limits

- Apple silicon with the private `AppleNeuralEngine.framework` runtime present.
- The dual path is intended for M3 Ultra, where the two dies expose physical
  ANE instances 1 and 2.
- The oMLX native custom kernels must be built (`OMLX_WITH_CUSTOM_KERNEL=1`).
- Dense Qwen3.5/3.6/3.8 affine q4 gate/up linears with group size 64 or 128.
  The down projection may use compatible affine q2/q4/q5/q6/q8 weights and
  remains on the GPU.
- Optional GDN acceleration accepts affine q4/q5 projections with group size
  64 or 128. Mixed q4/q5 layouts are supported when the ANE prefix covers the
  full z projection, leaving a homogeneous qkv suffix on the GPU.
- An MLP prefill call whose flattened token count exactly matches the fixed
  configured sequence length. Decode, target verification, short chunks, and
  unsupported layers automatically use the existing path.
- Fixed-shape ANE programs and their combined q4 suffixes are prepared eagerly
  on the MLX executor while the model starts. For the 64-layer 27B target this
  adds a substantial startup phase, but the first matching prompt no longer
  pays the compilation cost. Programs are cached for the model's lifetime.

The implementation uses undocumented APIs and can stop working after a macOS
update. It also requantizes the selected weights to per-output-channel INT8,
so it is an approximate acceleration path rather than bit-exact inference.

## Per-model settings

```json
{
  "qwen35_ane_prefill_enabled": true,
  "qwen35_ane_prefill_sequence_length": 2048,
  "qwen35_ane_prefill_fraction": 0.53,
  "qwen35_ane_prefill_max_layers": 64,
  "qwen35_ane_prefill_dual_ane": true,
  "qwen35_ane_prefill_gdn": true,
  "qwen35_ane_prefill_gdn_fraction": 0.50,
  "qwen35_ane_prefill_gdn_max_layers": 48
}
```

The private runtime accepted 121 resident model handles in a focused probe.
The current dual path packages every fixed-shape slice as a procedure inside
one model per physical ANE instance. The measured 64 MLP and 48 GDN layout
therefore exposes 112 procedures from only two resident programs, instead of
stopping at 60 dual MLPs. Extensions predating procedure banks retain the
120-program fallback budget. Other sequence lengths require separately
compiled fixed-shape banks and should be benchmarked before use.

The macOS app exposes the same controls under **Models → model settings →
Advanced → Experimental → Qwen ANE Prefill** for detected Qwen3.5/3.6/3.8
models. Enabling or changing a control reloads a resident model when the
working profile is applied. The editor starts from the measured 2,048-token,
53% MLP / 50% GDN, dual-ANE, 64/48-layer configuration above; the feature
itself stays off until explicitly enabled.

When the feature is active, oMLX aligns the scheduler's prompt chunk size with
the configured fixed ANE shape. This also overrides the wider Qwen prefill
floor used on high-memory systems. A 4,096-token ANE shape is supported, but a
4K benchmark request prefills only 4,095 tokens because the final token is
reserved for generation kickoff. The default remains 2,048 so 4K prompts still
route one full chunk through ANE.

The throughput-benchmark screen also offers a **Full · 2,048** warm-up. The
scheduler reserves the last prompt token for the first decode step, so this
mode builds a 2,049-token prompt to execute one genuine 2,048-token prefill
before timing begins. **Quick · 32** retains the previous low-latency warm-up.

Every native throughput-benchmark trial emits INFO-level comparison traces to
`server.log`. `[benchmark-prefill]` records every scheduler chunk (token count,
cache offset, model/cache evaluation time, and non-model overhead), while
`[benchmark-ane-profile]` records actual native MLP/GDN operation counts and
the same input-ready, ANE-evaluation, GPU-QMM, gap, and duty-cycle counters used
by the offline benchmark. `[benchmark-ane-summary]` explicitly reports the
expected fixed-shape calls and residual GPU tail. Ordinary inference requests
do not enable these counters or emit the per-chunk trace.

Qwen's configured padding token (`<|endoftext|>` for the tested checkpoint) is
a normal learned token, not a state-neutral null token. Appending it changes
the logits and advances both the KV cache and Gated DeltaNet recurrent state.
Padding can only be made semantically inert by carrying a mask through cache
positions, RoPE, and every recurrent update; the normal single-request path
does not provide that guarantee, so synthetic padding is not used to force ANE
shapes.

For the combined-only path, the native bridge directly merges the planar ANE
prefix and row-major GPU suffix while applying SwiGLU. This avoids materializing
the full raw gate/up result and two subsequent concatenations before the q4 down
projection. Extensions built before this fused primitive retain the compatible
raw merge path automatically.

The combined GPU suffix is retained alongside the original gate/up tensors so
decode and every fallback remain unchanged. The dual path also owns two input
and two output surfaces per accelerated layer. This deliberately spends memory
to avoid per-request weight preparation and to keep both ANEs ready.

## Qwen3.8-27B-oQ4e validation

The group-size-64 and mixed q4/q5 path was validated on an M3 Ultra with
`Qwen3.8-27B-oQ4e-mtp`, a 128-token generation tail, and a 2,048-token ANE
prompt block. The 4K row is a matched current-revision recheck. Its gain ranged
from 1.7% to 3.4% across matched fixed prompts because only one 2,048-token ANE
chunk runs before the 2,047-token GPU tail. The 16K and 32K GPU baselines are
the mean of two deterministic runs; their ANE/GPU values are single
scheduler-aligned rechecks.

| Prompt | GPU PP | ANE/GPU PP | PP change | TTFT change | End-to-end change |
|---:|---:|---:|---:|---:|---:|
| 4K | 445.2 tok/s | 460.4 tok/s | +3.4% | -3.3% | -2.3% |
| 16K | 439.1 tok/s | 517.0 tok/s | +17.8% | -15.1% | -13.6% |
| 32K | 408.9 tok/s | 486.0 tok/s | +18.9% | -15.9% | -15.0% |

The 16K and 32K output hashes matched the GPU path exactly. The 4K output was
stable across ANE rechecks but differed from GPU, which is consistent with the
approximate INT8 ANE prefix. Peak memory increased by about 4.15 GB, and eager
load time increased from 3.35 to about 27-29 seconds on the test system.
Token-generation throughput was unchanged because decode remains on the GPU.

## M3 Ultra reference result

On `True2456/Qwen3.8-27B-AWQ-4.85bpw`, sequence length 2,048:

| Measurement | GPU path | ANE/GPU path | Result |
|---|---:|---:|---:|
| Complete layer-0 MLP | 61.12 ms | 48.45 ms | 1.26x |
| Full 64-layer language body | 6.00 s | 5.28 s | 1.136x |

The complete body result is 341 to 388 prompt tokens/s. Eager preparation of
all 64 combined programs took about 15.1 seconds on the reference run; this is
paid during startup rather than by the first request. Combining gate+up also
reduced the accelerated layer-0 MLP from 49.78 ms to 48.45 ms (1.027x versus
the earlier two-program hybrid).

With all 64 MLPs enabled in the combined validation run, final hidden-state
cosine similarity was 0.99993 and last-token logit cosine similarity was
0.99975; the top token was unchanged. These measurements are workload-specific
and are not a substitute for downstream quality evaluation.

Fusing the ANE/GPU output merge with SwiGLU reduced the same combined-only
64-layer body from 5.2799 s to 5.2243 s (392.0 prompt tokens/s), a further
1.06% improvement. The matching GPU run was 5.9991 s, making the fused path
1.148x faster overall. Eager preparation took 16.5 seconds and remained outside
request timing. Final hidden-state and last-token logit cosine similarity
against GPU were 0.99995 and 0.99997; the top token was unchanged.

Packing the complete dual workload into one 112-procedure program per ANE
removed the residency tradeoff: all 64 MLP and 48 GDN slices fit in two
resident programs. Eager compilation takes about 37-40 seconds because each
large bank is compiled monolithically.

An instrumentation pass found that older benchmark builds compiled and
counted the GDN procedures but did not dispatch them: the installed mlx-vlm
version lacked the anticipated backend-registration function. Those older
4.80-4.96 second "GDN" figures therefore measured the MLP-only path and are
superseded. The compatibility hook now intercepts mlx-vlm's projection helper,
and the benchmark verifies 64 MLP plus 48 GDN procedure dispatches per prompt.

With the corrected hook and the retuned 53% MLP / 50% GDN request, the final
deterministic paired run measured:

| Path | Median body time | Prompt throughput | Versus GPU |
|---|---:|---:|---:|
| GPU only | 6.1149 s | 334.9 tok/s | 1.000x |
| Dual ANE/GPU | 4.5084 s | 454.3 tok/s | 1.356x |

Final hidden-state cosine similarity was 0.999200, last-token logit cosine
similarity was 0.998522, and top-1 was unchanged. These are single-prompt
numerical checks, not downstream quality validation.

That layout issues two ANE evaluations for each accelerated operation: one
request pinned to each physical ANE. Across 64 MLP and 48 GDN operations this
is 224 evaluations per 2,048-token prompt, or 112 sequential evaluations on
each ANE. The two evaluations belonging to an operation are launched in
parallel. Gate and up are already combined in each MLP evaluation, and z and
qkv are already combined in each GDN evaluation, so splitting either group
would increase dispatch count.

A single unpinned procedure containing the same 55% MLP slice took 57.90 ms
for a representative layer, versus 41.51 ms for the two pinned evaluations.
The one-call form was therefore 39.5% slower in latency (the dual form was
28.3% faster), showing that this driver does not effectively stripe one
procedure across both ANEs. Replacing the two short-lived dispatch threads
with persistent high-priority workers also regressed throughput, so the
existing paired launch was retained.

Profiling identified 53% MLP / 50% GDN as the best measured split. At 50% GDN,
the ANE and GPU GDN portions take about 10.1 ms and 9.95 ms respectively. The
53% MLP point measured 4.4768 s in its seven-run tuning pass, versus 4.5172 s
at 54% and 4.5370 s at 55%. Larger 60% banks were slower, and a monolithic
60%/60% bank exceeded the compiler's model-verification or weight-blob limit.

Set `OMLX_ANE_PROFILE=1` when running the benchmark to collect opt-in phase
timings. In the final paired run, ANE0 and ANE1 were executing requests for
38.81% each of total body time. Request launch delay was only 29-37 us. The
dominant downtime was dependency/input readiness: 25.1 ms before each MLP
request and 23.8 ms before each GDN request, primarily queued GPU work required
to produce the next input rather than ANE submission overhead. MLP ANE and GPU
suffix work averaged 20.37 ms and 19.19 ms; GDN averaged 10.15 ms and 9.95 ms.

The runtime's completion-handler async path was also tested. Its convenience
form measured 4.9474 s versus 4.9071 s for the threaded submit path. Calling
`doEvaluateWithModel:options:request:qos:completionEvent:error:` directly with
reused completion events removed that allocation overhead but still measured
4.9089 s. The existing submit path was retained.

The blocking input-pack wait is intentional on the tested M3 Ultra driver.
Moving it to a worker, replacing it with a private ANE wait event, or launching
ANE from the Metal completion callback all delayed ANE until after the queued
GPU suffix and destroyed device overlap. The completion-callback version
increased a fused layer from about 47.5 ms to 71.0 ms and the 64-layer body from
5.2243 s to 6.3535 s (322.3 prompt tokens/s), 5.6% slower than GPU-only. The
blocking version was therefore retained.
