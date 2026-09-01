# OoO-Spec in the regular batched engine

Status: experimental serial-exact functional MVP, disabled by default. It is
active only for an explicit loopback sidecar and the capability gates below.
Every declined hint uses ordinary target decoding. This phase makes no speed
or acceleration claim.

## Regular prefill and decode ownership

For a prompt `P = p0 ... pn`, `Scheduler._do_external_prefill()` runs
`p0 ... p(n-1)` through a request-local prompt cache and returns that cache plus
`pn`. `BatchGenerator.insert()` gives `pn` to a new `GenerationBatch`. Its
constructor processes `pn`, commits it to batched KV, and samples `y1`. The
first `next_generated()` call processes `y1`, samples `y2`, and returns `y1`.

An ordinary live row therefore owns more than KV:

1. batched KV through the last emitted token;
2. a sampled, not-yet-emitted token and its logprobs;
3. logits-processor token history and state;
4. generated-token count and stop-state-machine state.

The pending token and processor history are private `GenerationBatch` state.
The bounded lane does not attempt to adopt or synthesize those internals.

## The bounded verification boundary

The engine first freezes a call-free candidate containing the exact messages,
final tools, template tools, and template options. It does not start the
sidecar there. After scheduler capacity admission, prompt tokenization and
bounds, prefix-cache validation, memory preflight, actual logits-processor
construction, SpecPrefill exclusion, and model/cache capability checks, the
scheduler starts the sidecar immediately before target prefill. Ineligible or
rejected requests therefore make zero sidecar calls while eligible requests
can still overlap the provider with full or chunked prefill.

The verification boundary remains after external target prefill and
state-machine construction, but before ordinary batch insertion. At that
point the prompt cache is request-local and its exact offset must be
`len(P) - 1`.

The scheduler polls the mailbox once. A pending, failed, late, timed-out, or
malformed result is terminal and ordinary insertion continues unchanged. A
usable semantic tool call is rendered again with the target tokenizer and chat
template. The rendered base token ids must exactly equal the live
`Request.prompt_token_ids`, and the augmented rendering must be append-only at
both text and token level.

Verification then:

1. clones every dense `KVCache` layer into detached storage;
2. processes `pn` with a one-token target forward, exactly like ordinary
   greedy decode;
3. accepts only a matching candidate token, then processes that accepted token
   with another one-token target forward;
4. repeats one token at a time until mismatch or full acceptance;
5. queues the target correction or full-acceptance bonus with the exact target
   logprob row, then processes that token with one final one-token forward;
6. asserts that every layer offset is `len(P) + len(queue)`.

No multi-token target verification forward is used by the live lane. This
avoids sequence-shape-dependent arithmetic drift and intentionally gives up
the putative verifier speedup in this functional phase.

The baseline prefill cache is never mutated. Any error before activation drops
the hint and inserts that baseline normally.

## Output, early termination, and public handback

The verified queue owns its isolated cache and publishes exactly one token per
scheduler step. Each token goes through the regular request accounting,
detokenizer, output parser, stop-token, stop-string, length, and tool-call
finalization path. The queue cannot directly append visible text or tool calls.

If any regular path terminates midway, the scheduler trims the unexposed queue
tail and asserts the exact exposed offset. Matcher, length, parser, and
text-fallback termination all attach this trimmed target cache to the normal
completion-storage path. Semantic rows use private negative UIDs, so relying
on public-batch extraction for parser/text fallback would silently lose cache
storage. If an exact trim fails after exposure, the scheduler cold-replays
the last prompt token plus exposed outputs one token at a time from the
retained untouched `P[:-1]` cache.

When the queue drains without termination, no private continuation-adoption API
is needed. The scheduler trims the final exposed queue token, preserving cache
through `P + queue[:-1]`, and calls public `BatchGenerator.insert()` with that
final token as its input. The constructor replays it and recreates the ordinary
pending next token and logprobs. A small public state-machine adapter seeds the
already-advanced matcher state; request-local parser and stop-string buffers
continue unchanged. If trim or insertion violates an invariant, batch-one
ownership permits discarding the empty/partial generator, cold-replaying the
prefix, and retrying ordinary insertion.

## Capability and request gates

All of these are required:

- exact regular `BatchedEngine`, not VLM, DFlash, MTP, or a subclass;
- `max_num_seqs == 1` and `completion_batch_size == 1`;
- greedy temperature zero and positive `max_tokens`;
- no grammar, logits processor, repetition/presence/frequency penalty,
  thinking budget, or XTC;
- exact ordinary dense `mlx_lm.models.cache.KVCache` for every target layer;
- full-attention, non-recurrent/non-hybrid model metadata;
- no model-weight quantization metadata or loaded `QuantizedLinear`, and no
  TurboQuant KV; these variants remain unproven;
- no VLM inputs, SpecPrefill, VLM-MTP, mRoPE/MTP markers, or draft model;
- non-partial chat with tools, `parallel_tool_calls=false`, and `tool_choice`
  absent or `auto`.

The server trigger is currently limited to OpenAI `/v1/chat/completions`;
Anthropic Messages and OpenAI Responses remain on their unchanged baseline
paths. Direct `BatchedEngine` callers must pass the same explicit tool options.

The feature also requires:

```text
OMLX_OOO_SPEC_ENABLED=1
OMLX_OOO_SPEC_ENDPOINT=http://127.0.0.1:PORT/path
```

Only `127.0.0.1`, `::1`, and `localhost` endpoints are accepted. Optional
bounds are `OMLX_OOO_SPEC_TIMEOUT_S` (default `0.25`),
`OMLX_OOO_SPEC_MAX_PROMPT_TOKENS` (default `4096`), and
`OMLX_OOO_SPEC_MAX_SUFFIX_TOKENS` (default `128`). The protocol carries only a
tool index and JSON arguments; token ids and provider-rendered tool syntax are
never trusted. HTTPX environment proxy inheritance is disabled. Tool schemas
containing `$ref`, `$dynamicRef`, or `$recursiveRef` at any depth are rejected,
including internal references, so argument validation cannot retrieve a URL.

## Remaining upstream contract

The public trim/replay handback is sufficient only because this lane is
batch-one, greedy, and excludes all logits processors. A supported atomic
continuation-adoption and per-row rollback API is still required before this
can safely cover concurrent batching, stateful processors/constraints,
non-replayable cache families, or model-specific recurrent/rotating/quantized
state.

There is no live multi-token/batched verifier and no numerical-drift opt-in.
Any future batched target verifier must remain dormant and unwired until a
preregistered gate proves token decisions, full logprob vectors, all cache
arrays, end-to-end continuation, and a representative performance win across
the intended model/dtype/hardware matrix.

The CPU tests prove target-only full/partial/zero acceptance, correction and
bonus tokens, exact logprobs/cache offsets, ordinary output equivalence,
stop/parser/length termination and cache storage, cross-handback stop strings,
cancellation, late/timeout/malformed no-ops, zero provider calls for ineligible
requests, proxy bypass, recursive-reference rejection, and cold replay after a
trim mutant. A real two-layer dense Llama differential checks float32 and
bfloat16 tokens, full logprob vectors, and every valid key/value cache array
against public serial `BatchGenerator` decoding. This is functional exactness
evidence, not GPU, production-model, acceleration, memory-headroom, or
sidecar-quality evidence.
