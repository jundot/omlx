# Qwen3.5/3.6/3.8 ANE/GPU Prefill (Experimental)

This source-build experiment uses private AppleNeuralEngine APIs to split one
fixed-shape Qwen3.5/3.6/3.8 prompt across both ANEs and the GPU. Two INT8 programs,
pinned to physical ANE instances 1 and 2, compute disjoint output-channel
slices while Metal computes the remaining quantized channels. It is disabled
by default.

At the default 53% MLP request, alignment gives the two ANEs 26.5% of gate and
up channels each (52.9% total) and leaves 47.1% on GPU. A native merge applies
SwiGLU without materializing the full gate/up result. The GDN z+qkv input
projection uses a precision-aware split: ANE computes only the token-local z
gate, while every recurrent qkv row stays off the approximate ANE path and
normally uses checkpoint-precision GPU projection. The configured GDN fraction
is a ceiling; a wider request is capped at the model's z boundary (37.5% on the
validated Qwen3.6/3.8 27B layout). Optional FP16 CPU sharing can take
independent gate/up, down-projection, and residual GDN qkv slices; all three
branches run in parallel and are merged natively.
GDN recurrence, b/a and output projections, normalization, embeddings, and
logits remain on GPU.

## Requirements and limits

- Apple silicon with the private `AppleNeuralEngine.framework` runtime present.
- The dual path is intended for M3 Ultra, where the two dies expose physical
  ANE instances 1 and 2.
- The oMLX native custom kernels must be built (`OMLX_WITH_CUSTOM_KERNEL=1`).
- Dense Qwen3.5/3.6/3.8 affine q4/q5/q6/q8 gate/up linears with group size 64
  or 128. The optimized fused q4 path remains unchanged; compatible quantized
  weights are retained for every GPU suffix. The down projection may use
  compatible affine q2/q4/q5/q6/q8 weights.
- Optional GDN acceleration accepts affine q4/q5/q6/q8 projections with group
  size 64 or 128. The z width must align exactly to the selected single- or
  dual-ANE output granularity. Mixed q4/q5/q6/q8 layouts are supported because
  ANE covers exactly z and leaves a homogeneous qkv suffix on the GPU.
- CPU sharing requires a separately preprocessed FP16 clone of the model. It
  does not modify or dequantize the source checkpoint in place. The CPU GDN
  slice applies only to residual qkv outputs after the ANE prefix; z remains
  wholly on ANE so mixed-quantization checkpoints keep a valid GPU suffix.
  The clone utility rejects non-finite tensors and BF16 values outside the
  FP16 range before writing any checkpoint files.
- An MLP prefill call whose flattened token count matches the fixed configured
  sequence length, or a single-prompt call that is wider: wide chunks are tiled
  internally into fixed-shape blocks. A tuner-calibrated suffix may zero-pad a
  sufficiently large residual intermediate activation to one fixed ANE tile;
  smaller tails, decode, target verification, and unsupported layers use the
  existing path.
- Fixed-shape ANE programs and their combined affine suffixes are prepared eagerly
  on the MLX executor while the model starts. For the 64-layer 27B target this
  adds a substantial startup phase, but the first matching prompt no longer
  pays the compilation cost. Programs are cached for the model's lifetime.

The implementation uses undocumented APIs and can stop working after a macOS
update. It also requantizes the selected weights to per-output-channel INT8,
so ANE results are approximate rather than bit-exact. The approximation is
kept out of recurrent GDN qkv state specifically to prevent long-context error
accumulation; the MLP and token-local GDN z branches remain approximate.

On NAX GPUs (the M5 family) the hybrid GPU suffix runs on dedicated NAX
qmm kernels (group sizes 64 and 128), which resolves the prefill regression
that early field testing saw when the suffix competed with tensor-unit
prefill. The optimal ANE/GPU balance sits well below the classic ~50%
optimum there, so use the Tune ANE Split utility in the model settings to
measure the split for the specific machine before enabling. If the NAX
metallib is missing at runtime, the suffix quietly falls back to the
classic Metal kernels, and `OMLX_QWEN35_QMM_NAX=0` forces that fallback.
`OMLX_QWEN35_ANE_PREFILL=0` keeps the whole feature off everywhere
regardless of the per-model setting.

The ANE GDN dispatch runs through the mlx-lm prefill linear patch, so
`OMLX_QWEN35_Q4_LM_LINEAR=0` disables ANE GDN acceleration as well as the
standalone GPU qmm routing. GDN b/a suffix projections follow the same q8
token threshold as that patch: below `OMLX_QWEN35_Q8_LINEAR_MIN_TOKENS`
(default 16384, which covers every fixed ANE shape) q8 b/a use stock MLX,
where the native q8 tile is not profitable.

## Per-model settings

```json
{
  "qwen35_ane_prefill_enabled": true,
  "qwen35_ane_prefill_sequence_length": 2048,
  "qwen35_ane_prefill_tail_padding_min_tokens": 0,
  "qwen35_ane_prefill_fraction": 0.53,
  "qwen35_ane_prefill_max_layers": 64,
  "qwen35_ane_prefill_dual_ane": true,
  "qwen35_ane_prefill_gdn": true,
  "qwen35_ane_prefill_gdn_fraction": 0.50,
  "qwen35_ane_prefill_gdn_max_layers": 48,
  "qwen35_ane_prefill_cpu_enabled": false,
  "qwen35_ane_prefill_cpu_fraction": 0.135,
  "qwen35_ane_prefill_cpu_down_fraction": 0.0,
  "qwen35_ane_prefill_cpu_gdn_fraction": 0.0,
  "qwen35_ane_prefill_cpu_threads": 8,
  "qwen35_ane_prefill_cpu_shared_resource": true
}
```

The private runtime accepted 121 resident model handles in a focused probe.
The current dual path packages every fixed-shape slice as a procedure inside
one model per physical ANE instance. The measured 64 MLP and 48 GDN layout
therefore exposes 112 procedures from only two resident programs, instead of
stopping at 60 dual MLPs. Extensions predating procedure banks retain the
120-program fallback budget. Other sequence lengths require separately
compiled fixed-shape banks and should be benchmarked before use.

Loading a bank maps its entire weight blob into the owning ANE's device
address window at program-create. That window is about 4 GiB per ANE
instance, so the dual 53%/50% Qwen3.8-27B layout at roughly 3.75 GiB per
bank fits one bank per die on M3 Ultra but cannot host both banks on a
single-die chip such as M3 Max, where the load fails with 0x20004. When a
bank fails to load, oMLX first retries with two near-half banks per
instance and then with progressively smaller split banks before falling
back to per-layer programs; `OMLX_QWEN35_ANE_BANK_MAX_BYTES`
forces an initial per-bank cap for testing, counted on the source weights
handed to the bank compiler (about four times the compiled INT8 program
size). An interleaved M3 Ultra A/B measured split banks about 1% faster at
prefill with a slightly shorter eager load, but the monolithic bank was
bit-stable across five repeated greedy runs while split runs occasionally
diverged at a greedy tie, so the monolithic bank remains the first attempt
and splitting stays a load-failure fallback. The per-layer fallback
prioritizes MLPs within its 120-program budget and logs when GDN layers are
dropped instead of leaving them silently on the GPU, and benchmark traces
report the compiled MLP and GDN counts alongside the configured ones.

The macOS app exposes the same controls under **Models → model settings →
Advanced → Experimental → Qwen ANE Prefill** for detected Qwen3.5/3.6/3.8
models. Enabling or changing a control reloads a resident model when the
working profile is applied. The editor starts from the measured 2,048-token,
53% MLP / 50% GDN ceiling, dual-ANE, 64/48-layer configuration above; the
feature itself stays off until explicitly enabled. The runtime reports when it
caps that requested GDN fraction at the model-specific z boundary.

The split tuner calibrates five workload controls: MLP gate/up work on ANE,
MLP gate/up work on CPU, MLP down-projection work on CPU, GDN work on ANE, and
GDN qkv work on CPU. It packages several widths from one real MLP and GDN layer
into a small temporary procedure bank, measures the production native paths,
and then eagerly compiles only the predicted full-model candidate. Timings
from that application-level run rebalance the ANE, CPU, and GPU branch rates
once before a final verification. This avoids a five-dimensional full-model
grid while still making end-to-end prompt throughput the recommendation
criterion. CPU dimensions are skipped automatically when the checkpoint or
native extension does not support FP16 CPU sharing. Zero is always a valid CPU
GDN candidate, so the tuner can retain GPU-only residual qkv when CPU sharing
does not pay off.

The tuner preserves the model's single- or dual-ANE execution setting. In
single-ANE mode it compiles one unpinned calibration bank and tunes ANE/GPU
MLP and GDN splits normally. CPU gate/up, down-projection, and GDN sharing are
all calibrated in either mode when the checkpoint has the required eager FP16
rows and the matching native symbols are available.

After selecting the best full-model candidate, the tuner derives the first
profitable padded tail from measurements made by the current run. If `S` is
the fixed ANE sequence length, `G` is GPU-only prompt throughput, and `H` is
the winning hybrid throughput, the crossover is
`floor(S * G / H) + 1` tokens. This is the first integer tail for which the
estimated GPU time (`tail / G`) exceeds one padded hybrid tile (`S / H`). The
tuner writes zero when the hybrid candidate does not beat GPU-only, which
keeps padding disabled. Saved thresholds are cleared during each search so an
older calibration cannot influence a new result.

The scheduler keeps its normal prompt chunk width; chunks wider than the
compiled ANE shape are tiled internally. Chunks narrower than the compiled
shape can use a padded intermediate tile only when they meet the calibrated
threshold; otherwise they stay on the ordinary GPU path. With boundary caching
on, delivered chunks are cut at the 2,048-token cache block edge, so 2,048
remains the safe default everywhere.
A 4K benchmark request prefills only 4,095 tokens because the final token is
reserved for generation kickoff; the benchmark screen's ANE alignment option
adds one token so an exact multiple of the fixed shape is prefilled.

The throughput-benchmark screen also offers a **Full · 2,048** warm-up. The
scheduler reserves the last prompt token for the first decode step, so this
mode builds a 2,049-token prompt to execute one genuine 2,048-token prefill
before timing begins. **Quick · 32** retains the previous low-latency warm-up.

Every native throughput-benchmark trial emits INFO-level comparison traces to
`server.log`. `[benchmark-prefill]` records every scheduler chunk (token count,
cache offset, model/cache evaluation time, and non-model overhead), while
`[benchmark-ane-profile]` records actual native MLP/GDN operation counts and
the same input-ready, ANE-evaluation, GPU-QMM, gap, and duty-cycle counters used
by the offline benchmark. `[benchmark-ane-summary]` reports the observed
fixed-shape tiles and residual tail when the scheduler trace is available,
falling back to a prompt-length estimate otherwise. Ordinary inference requests
do not enable these counters or emit the per-chunk trace.

Qwen's configured padding token (`<|endoftext|>` for the tested checkpoint) is
a normal learned token, not a state-neutral null token. Appending it changes
the logits and advances both the KV cache and Gated DeltaNet recurrent state.
Padding can only be made semantically inert by carrying a mask through cache
positions, RoPE, and every recurrent update; the normal single-request path
does not provide that guarantee, so synthetic token padding is not used to
force ANE shapes. Intermediate tail padding is different: it adds zero rows
only around the tokenwise MLP and GDN input projections, slices those rows from
the projection result before GDN recurrence or later model stages, and never
alters the token sequence, positions, or cache state.

For the combined-only path, the native bridge directly merges the planar ANE
prefix and row-major GPU suffix while applying SwiGLU. This avoids materializing
the full raw gate/up result and two subsequent concatenations before the q4 down
projection. Extensions built before this fused primitive retain the compatible
raw merge path automatically.

The combined GPU suffix is retained alongside the original gate/up tensors so
decode and every fallback remain unchanged. The dual path also owns two input
and two output surfaces per accelerated layer. This deliberately spends memory
to avoid per-request weight preparation and to keep both ANEs ready.
When CPU sharing is enabled, model admission additionally reserves the eager
FP16 gate/up and GDN rows, the down-projection GPU suffix, and bounded
materialization scratch. This projected size participates in the normal memory
guard before the model begins loading.

## Recurrent-safe GDN validation

The current z-only policy was selected from a controlled 32K comparison on
`Qwen3.6-27B-oQ4e-mtp`. A 50% ANE slice that included 2,048 of 10,240 recurrent
qkv rows deterministically failed ordering and exact-format summarization tasks
that distinguished the GPU baseline. Capping ANE at all 6,144 z rows (37.5% of
z+qkv) restored all four baseline-discriminating retrieval, ordering, code, and
summary checks. Six varied generations all ended normally without a suspected
loop, and their output hashes repeated exactly in a second pass.

The precision cap preserved nearly all of the useful acceleration. A direct
2,048-token GDN projection measured 9.61 ms for z-only ANE/GPU versus 14.73 ms
on GPU (1.53x). In an isolated real-server 32K cold/cache-hit pair, subtracting
the hit time from the cold time gave about 507 prompt tok/s, approximately 22%
above the matched 416 tok/s GPU baseline and slightly above the earlier 50%
full-GDN result. These figures are specific to the tested M3 Ultra and model,
but the recurrent-versus-token-local boundary is enforced for every model.

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
The throughput-only GDN measurements below predate the recurrent-safe z-only
policy and are retained as implementation history, not current tuning advice.

With the corrected hook and the retuned 53% MLP / 50% GDN request, the final
deterministic paired run measured:

| Path | Median body time | Prompt throughput | Versus GPU |
|---|---:|---:|---:|
| GPU only | 6.1149 s | 334.9 tok/s | 1.000x |
| Dual ANE/GPU | 4.5084 s | 454.3 tok/s | 1.356x |

Final hidden-state cosine similarity was 0.999200, last-token logit cosine
similarity was 0.998522, and top-1 was unchanged. These are single-prompt
numerical checks, not downstream quality validation.

The optional CPU GDN branch was subsequently validated with the FP16 clone of
`Qwen3.8-27B-AWQ-4.85bpw`, 2,048 tokens, 53% MLP on ANE, 13.5% MLP gate/up on
CPU, and 20% MLP down projection on CPU. Moving 15% of qkv to CPU improved a
deliberately reduced 45% ANE GDN split from 471.6 to 480.1 tok/s, with the ANE,
CPU, and GPU GDN branches measuring 8.63, 8.86, and 9.18 ms per operation.
However, it did not beat the already balanced 50% ANE / 0% CPU GDN result:

| GDN on ANE | GDN on CPU | Prompt throughput | Versus 50% ANE / 0% CPU |
|---:|---:|---:|---:|
| 50% | 0% | 490.2 tok/s | reference |
| 50% | 5% | 475.2 tok/s | -3.1% |
| 45% | 12.5% | 478.7 tok/s | -2.3% |
| 45% | 15% | 480.1 tok/s | -2.0% |

That standalone sweep held the surrounding workload and most candidate widths
fixed, so it was useful for validating branch timing but did not predict the
best complete application split. The subsequent in-app five-way tuner jointly
selected 45% MLP on ANE, 45% GDN on ANE, 14% gate/up on CPU, 20% down on CPU,
and 13% GDN qkv on CPU. It measured 517.9 prompt tok/s, 45.8% above its GPU-only
baseline. That application result was authoritative for the earlier
throughput study; the current tuner fixes the ANE portion at z and only tunes
the residual CPU/GPU qkv split. The standalone table illustrates why CPU GDN
must be tuned jointly rather than accepted or rejected from an isolated
fixed-split sweep. Hidden-state and last-token logit cosine similarity at the
validated CPU GDN point were 0.999989 and 0.999999, and top-1 matched the GPU
path.

That layout issues two ANE evaluations for each accelerated operation: one
request pinned to each physical ANE. Across 64 MLP and 48 GDN operations this
is 224 evaluations per 2,048-token prompt, or 112 sequential evaluations on
each ANE. The two evaluations belonging to an operation are launched in
parallel. Gate and up are already combined in each MLP evaluation. In the
historical GDN layout, z and qkv were also combined in each evaluation; the
current layout compiles the same number of GDN procedures but ends each one at
the z boundary.

A single unpinned procedure containing the same 55% MLP slice took 57.90 ms
for a representative layer, versus 41.51 ms for the two pinned evaluations.
The one-call form was therefore 39.5% slower in latency (the dual form was
28.3% faster), showing that this driver does not effectively stripe one
procedure across both ANEs. Replacing the two short-lived dispatch threads
with persistent high-priority workers also regressed throughput, so the
existing paired launch was retained.

Historical profiling identified 53% MLP / 50% GDN as the best measured split.
At 50% GDN,
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
