# K2 Uno

Uno uses conditional adapter proposals and target verification for dense K2 models.
It is off by default. In model settings, choose the matching adapter in the **Uno**
selector under **Acceleration**. Choose **Off** for ordinary decoding. **Get adapter**
opens the matching Hugging Face repository when no local adapter is available.
The web and native downloaders recognize Uno helpers; helpers are excluded from
standalone inference choices. The loader validates the pair and tensor layout.
If adapter verification fails, the model card keeps its metadata and shows the
reason with **Retry**. Download remains disabled until verification succeeds;
an unverified adapter is not labeled unsupported.

| Base | Adapter | Context limit |
| --- | --- | --- |
| IFM/K2-Horizon-0.9B | IFM/K2-Horizon-0.9B-Uno | 131,072 tokens |
| IFM/K2-Horizon-7B | IFM/K2-Horizon-7B-Uno | 262,144 tokens |

The loader accepts compatible dense layouts without a model-size whitelist.
The limits include prompt and output tokens. Full-limit inference is not validated
by the small runtime checks described below. MoVA and gated attention are unsupported.

## Settings and lifecycle

Conflicts show the effective value, including inherited global sampling values.
**Apply required settings** explicitly edits the working model/profile settings:
repetition penalty 1.0, min-p and presence penalty 0, and conflicting accelerators,
Guided Grammar and Thinking Budget off. Save the working settings/profile to persist
these changes. Global defaults remain unchanged. Profiles and model loading
revalidate adapter availability; a stale profile cannot replace working settings.

Profile creation and updates validate the effective base-plus-profile settings.
A base save or profile application that would invalidate saved profiles is rejected
with their names and conflicts. A profile can explicitly set `uno_enabled: false`
to use ordinary decoding with its own sampling settings.
Already-invalid exposed profiles remain in `/v1/models` and `/v1/models/status`
with `invalid: true` and an `invalid_reason`; the server logs a warning. Requests
to those aliases return a 400 naming the conflict. Repairing the profile clears
the invalid state without changing the base model's settings.

Uno runs while one request is active. Overlap uses ordinary continuous batching;
unconsumed proposals are discarded before a merge, and Uno resumes when one request
remains. Snapshots contain emitted tokens only. EOS and output-budget boundaries
finish through the ordinary batch lifecycle, including the final KV row.

Temperature, top-p, top-k, stop strings, native K2 tools and SSD prefix reuse are
supported. Thinking budgets, structured output, distributed serving, nonneutral
penalties, min-p and XTC probability are rejected. `reasoning_effort` remains
available. Tools require the optional grammar backend; its absence does not prevent
ordinary Uno requests. Allocator cleanup uses the scheduler's existing policy.

## Performance evidence

Use the existing **Bench** facility for repeatable throughput measurements. It labels
Uno and includes `uno_enabled` and the adapter name in its settings snapshot. Local
result rows additionally contain a `uno` runtime snapshot from the loaded engine:
resolved base/adapter paths, compiled status, compiled ANE layer count, cycles,
proposals, accepted proposals, Uno/ordinary token counts and their host-observed
decode times. Token counts are processed rows, including initialization and cancelled
work; they need not equal delivered completion usage. Counters are cumulative for
that loaded model, including warmup;
subtract snapshots to isolate a workload. They are not per-request GPU profiler
measurements. Resolved Hugging Face snapshot paths identify revisions; locally
converted models need a separate revision/conversion manifest.

Compiled GPU regions share the native K2 attention and MLP math. They are enabled
only for full-head default RoPE: the released 0.9B layout does not qualify; the 7B
layout does. ANE prefill is optional. Compare ordinary, compiled ordinary, Uno and
Uno+ANE separately when attributing speedups. ANE/compiled arithmetic can change
outputs and uses separate prefix-cache namespaces where needed. A fixed seed does
not guarantee identical sequences between Uno and ordinary decoding.

K2 ANE prefill pads only near-full MLP tiles: at least 31/32 of the configured
tile width (1,984 rows for a 2,048-row tile). Shorter tails stay on the GPU.
Padding is internal to the MLP; it does not add prompt or KV-cache rows. Decode
and Uno proposals/verification remain on the GPU. This policy has its own prefix
cache namespace.

Local checks on 0.9B and a quantized 7B covered natural completion, native tool calls,
cache reuse, cancellation and overlap returning to singleton decoding. Uno was slower
on the sampled coding prompt. That prompt contained only 27 tokens, so the enabled
ANE configuration did not execute any ANE prefill tiles.

On an Apple M4 Max with 128 GiB unified memory, BF16 K2-Horizon-7B tool-prompt
replays measured the following median seconds from prefill through natural EOS:

| Prompt tokens | GPU ordinary | GPU Uno | Uno + ANE | Ordinary / Uno + ANE |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 15.047 | 6.870 | 6.861 | 2.19x |
| 8,192 | 25.637 | 18.693 | 16.684 | 1.54x |

These direct model calls exclude HTTP, tokenization, setup and response parsing.
All three modes use compiled GPU regions and retain ANE programs in memory; only
Uno + ANE activates ANE prefill, using `fraction=1/3`, `width=2048` and the near-full
policy above. Each mode uses
one warmup and three timed repetitions in alternating order, fresh KV caches,
greedy sampling, an eight-token Uno block and a 768-token output budget. The saved
migration-plan tool prompts finish at EOS with 290/297 output tokens at 2K/8K;
outputs match the historical token sequences. Other user inference and processing
jobs were paused, but OS/background activity was not controlled. The base revision
is `586b03f0fd1fbbf2f13eeafc33749e95ae34dd10`; the matching 7B Uno adapter revision
is `669f041aab04fad836e757ede9a028058b064996`.

Most of the gain comes from Uno. ANE adds negligible request-level benefit at 2K
and about 12% at 8K relative to GPU Uno in these replays. The incremental benefit
of near-full padding over full-tile-only ANE is small and remains provisional.
A separate 8K serving-engine comparison measured 24.088 s ordinary, 16.970 s Uno
and 17.016 s Uno + ANE: a 1.42x overall gain, with no additional ANE improvement.
That serving check used the earlier full-tile-only policy.

These small samples do not establish a general throughput or quality improvement.
Keep Uno experimental and off by default. For a useful comparison, hold checkpoint,
sampling, prompts, output budget and cache state constant; warm each mode, alternate
run order, and report finish reasons, acceptance, ordinary-decoding share and memory.
Store run outputs and checkpoint manifests outside the source tree.
