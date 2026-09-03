# Metal and exact prompt-tail keepwarm (experimental)

This opt-in latency stack makes cached follow-up turns feel immediate without
changing model math. It has three separately gated layers:

1. **Metal readiness** — a tiny asynchronous kernel keeps the command path
   responsive after idle time.
2. **Exact resident L0** — a bounded, exclusive handoff retains validated live
   cache state so a matching next turn avoids serialization/reconstruction.
3. **Prompt-tail materialization** — while fully idle, oMLX reconstructs an
   existing durable prefix, evaluates only a bounded uncached suffix through
   the target model, and atomically publishes an exact stable-prefix fallback.

The third layer handles a common agent/tool case: a client re-renders the prior
assistant transcript differently from the raw generated token stream. oMLX can
retain both the longer terminal state and the guaranteed input-prompt prefix
under one byte ceiling, then acquire the longest exact prefix that really
matches the next request. If several terminal branches exist, the newest
terminal and the shared stable boundary win; older branches use durable cache.

The request-boundary design is adapted from Jonathan Spangler's Apache-2.0
[ThunderMLX](https://github.com/jonathan308/ThunderMLX) work. The oMLX
implementation is re-engineered around continuous batching, per-engine MLX
executors, paged/SSD cache ownership, live settings, and unload/reload safety.

## Support boundary

| Path | Current support |
| --- | --- |
| Asynchronous Metal pulse | Local Batched and VLM engines |
| Exact resident L0 | Text-only non-speculative caches whose complete timeline validates |
| Immediate stable boundary | Exact, unwrapped B1 `KVCache` graphs; copied at the preceding durable block boundary |
| Immediate hybrid boundary | Finalized B1 exact `ArraysCache`/`SizedArraysCache` plus exact `KVCache`, copied at `N-1` |
| Prompt-tail materialization | Eligible text-only non-speculative models |
| Speculative decode | Fail closed unless that architecture proves an exact target-terminal transaction |
| Qwen3.5/3.6 Lightning MTP | Exact target offsets plus empty rollback/undo state publish directly; other terminal shapes reconcile to a fresh standard target cache before publication |
| Qwen4 speculative decode | Qualified in Fusion with the separate cached-suffix/terminal transaction stack; the standalone upstream PR remains fail closed until that dependency lands |
| Image/audio/video requests | Metal pulse only; media-keyed KV never enters text-only L0 or prompt-tail materialization |

A later genuinely media-free request can arm the text tier. If an earlier image
remains in the full conversation payload, the request remains multimodal and
continues on the existing media-keyed cache path.

## Metal readiness

After useful request activity, idle/post-response work on the engine's existing
one-worker executor lazily creates a dedicated thread-bound MLX stream and
compiles one safe `mx.fast.metal_kernel`. It dispatches one fp32 element with a
one-thread grid via `mx.async_eval`; the serving path performs no host read,
`mx.eval`, scalar access, or synchronization barrier. Retained pulse state is
one four-byte input and the latest four-byte output.

Request-start uses only an already-prepared pulse. If off-path preparation has
not happened, request-start performs zero Metal allocation or compilation.

## Exact prompt-tail path

- It is disabled unless the master toggle and prompt-tail subfeature are on.
- A completed B1 plain-`KVCache` prefill can publish its already-computed stable
  durable boundary immediately. The source cache remains untouched and the
  detached prefix is fully evaluated before deferred terminal publication.
- Plain chat templates may rewrite several generation-marker tokens, so the
  generic provider uses the preceding durable block boundary and reprocesses at
  most one block. Qwen4 keeps its more specific `N-1` provider only when the
  separate target-terminal capability marker is present; this standalone PR
  does not set that marker.
- Hybrid recurrent/KV models may publish exact `N-1` only when the complete
  graph consists of finalized B1 upstream `ArraysCache` (optionally the exact
  `SizedArraysCache` wrapper) plus plain `KVCache`. Every recurrent slot and
  logical KV prefix is independently copied and eagerly evaluated.
- A request must be text-only, cacheable, inside the token/byte limits, and
  have an independently durable reusable prefix.
- Only one hidden materializer runs process-wide.
- New admission invalidates the lifecycle epoch and aborts between bounded
  chunks; user work always wins.
- Hidden forwards are target-only: no sampler, output, BatchGenerator row, MTP
  draft, prompt-priming capture, or durable cache store is created.
- Every retained cache must prove exact tokens, logical offsets, recurrent
  counts, rollback cleanliness, and supported auxiliary state. QSA additionally
  proves K/V capacity, raw index keys, index offsets, and text MRoPE positions.
- Publication is atomic with admission and hot-cache clear.
- An already resident stable prefix suppresses redundant hidden materialization
  only when it covers the expected durable boundary; short/unrelated entries do
  not block useful work.
- The durable paged/SSD tier may be read/promoted but is never written by the
  hidden pass. Hidden reads do not inflate user cache-rate, phase, prefill, or
  decode telemetry.
- The UI toggle temporarily provides two resident slots under the existing byte
  ceiling. Turning it off restores the prior configured slot policy.
- Hot-cache clear cancels in-flight work and cannot be followed by stale
  repopulation. Unload drains maintenance before model/Metal teardown.

## Enable and observe

Enable **Settings → Advanced → Performance → Latent Metal keepwarm**. It applies
live to loaded engines and persists for later loads. It is off by default
because it can use additional idle power, GPU time, and resident cache memory.

Engine telemetry records pulse preparation/submission separately. Prompt-tail
telemetry reports scheduled, published, skipped, cancelled, and failed results
with bounded prompt/prefix/suffix counts. Hidden work is never reported as a
user request speed.

## Environment controls

| Variable | Default | Purpose |
| --- | ---: | --- |
| `OMLX_KEEPWARM` | `0` | Master switch |
| `OMLX_KEEPWARM_INTERVAL_SECONDS` | `2` | Periodic idle pulse cadence |
| `OMLX_KEEPWARM_IDLE_AFTER_SECONDS` | `2` | Idle gate before periodic pulse |
| `OMLX_KEEPWARM_REQUEST_START` | `1` | Reuse a prepared pulse at request start |
| `OMLX_KEEPWARM_REQUEST_START_IDLE_SECONDS` | `2` | Request-start idle gate |
| `OMLX_KEEPWARM_POST_RESPONSE` | `1` | Enable post-response pulse |
| `OMLX_KEEPWARM_POST_RESPONSE_DELAY_SECONDS` | `1` | Post-response pulse delay |
| `OMLX_KEEPWARM_LARGE_CACHE_TOKENS` | `8192` | Long-cache cadence threshold |
| `OMLX_KEEPWARM_LARGE_CACHE_INTERVAL_SECONDS` | `2` | Long-cache cadence |
| `OMLX_KEEPWARM_SLOW_THRESHOLD_SECONDS` | `1` | Slow-submission threshold |
| `OMLX_KEEPWARM_SLOW_BACKOFF_SECONDS` | `60` | Backoff after failure/slow work |
| `OMLX_KEEPWARM_PROMPT_TAIL` | `0` | Exact idle prompt-tail materialization |
| `OMLX_KEEPWARM_PROMPT_TAIL_DELAY_SECONDS` | `1` | Delay after completion |
| `OMLX_KEEPWARM_PROMPT_TAIL_MIN_TOKENS` | `256` | Minimum eligible prompt |
| `OMLX_KEEPWARM_PROMPT_TAIL_MAX_SUFFIX_TOKENS` | `4096` | Maximum hidden target suffix |
| `OMLX_KEEPWARM_PROMPT_TAIL_MAX_TOKENS` | `262144` | Maximum complete prompt |
| `OMLX_KEEPWARM_PROMPT_TAIL_CHUNK_SIZE` | `128` | Cancellation granularity |
| `OMLX_EXACT_RESIDENT_MAX_ENTRIES` | explicit `0`; UI uses `2` temporarily | Resident exact-prefix slots |
| `OMLX_EXACT_RESIDENT_MAX_BYTES` | `8 GiB` | Shared terminal/fallback ceiling |

An explicit zero resident-slot setting is a hard disable.

## Physical Fusion qualification

Hardware: M3 Ultra, 256 GB unified memory. Model:
Qwen3.8-Flash-Next oQ4e MTP. Prompt-tail Qwen measurements use the separately
qualified Fusion terminal/cached-suffix stack; they are integration evidence,
not a claim that this standalone PR enables speculative Qwen reuse by itself.

| Gate | Result |
| --- | --- |
| 18,174-token transcript divergence, OFF | 16,384 cached; 3.80s model / 4.17s visible TTFT |
| Same exact turn, ON | 18,152 cached; 0.23s model / 0.62s visible TTFT |
| Improvement | 6.7× visible TTFT; exact output parity |
| Qwen3.8-27B hybrid rapid turns | 1.72–1.78s model / 1.89–1.94s visible baseline; 0.24s model / 0.40s visible with exact `N-1` resident reuse |
| Hybrid clone overhead | 10.2–12.2ms; 0.42 GiB; hot and paged outputs byte-identical (`98d8670fc5f9e7f2c4b37ab13818cf46879cb5e8f914fb8642c317db4f7c24fe`) |
| Ordinary matching terminal | About 0.47s visible TTFT; no regression |
| Structured tool follow-up | 18,445/18,487 cached; 0.51s total; valid tool call/result flow |
| Cold sustained performance | 9,994 prompt tokens at 907.7 tok/s; 500-token decode at 74.36 tok/s; 98.2% MTP acceptance |
| Admission during hidden pass | Cancelled in 437ms; user request completed normally |
| Hot-cache clear during hidden pass | Cancelled in 488ms; zero stale L0 repopulation |
| Unload/reload | Unload completed in 1s; reload restored the durable 4,096-token prefix |
| Idle memory/SSD soak | RSS and 1.60 GB resident L0 flat; SSD manifest hash unchanged |
| B2/B4/B6 smoke | Every request completed with a unique exact marker and zero errors |
| Multimodal → fresh text | Image request completed correctly; no media L0 arm; fresh text completed in 0.33s |

The original pulse-only qualification also kept cold prefill and B1 throughput
within run-to-run noise, survived cancellation/load/unload, and left the SSD
manifest and repeated-idle memory samples unchanged.

## Release gate

Before enabling by default, compare OFF/ON across:

- ordinary exact-terminal turns and transcript-divergent agent/tool turns;
- 0/5/15/60-second idle gaps;
- B1/B2/B4/B6, active-prefill/decode overlap, and cancellation;
- hot clear, SSD restore, restart, unload/reload, and memory pressure;
- text, structured tools, image/audio/video exclusion, and a later media-free
  request;
- cold prefill, sustained 500-token decode, output hashes, and speculative
  acceptance.

No speed claim is valid unless prompt/completion counts, cache coverage,
hardware, quantization, and speculative acceptance are reported separately.
