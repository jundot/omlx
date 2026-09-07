# Affine4 SVG chat performance comparison

Apple M5 Pro, 48 GiB, 2026-09-07. Current oMLX `b8c3d1da`, original oMLX Affine4 `1ee0d24a`, VLM Affine4 source `10f9030d` for the isolated attention comparison.

The matched server requests do not reproduce a 30-to-22 tok/s regression in the current branch. The user's completed chat produced its actual SVG answer at approximately 33.4 tok/s; the roughly 22.7 tok/s phase was thinking. MTP yields fewer useful tokens per cycle on thinking text. Both revisions have nearly identical model-pass costs in the controlled short-context requests.

## Completed manual chat

The prompt was `draw dinosaur using plain svg`, with no system prompt, thinking enabled without a budget, MTP enabled, and an explicit 100,000-token chat limit. The actual prompt contained 57 tokens. The server completed 28,903 generated tokens in 1,125.01 seconds, reporting 25.7 tok/s overall.

The final browser statistics recorded approximately 18,286 thinking tokens over 806.5 seconds (22.7 tok/s). Subtracting those from the server total leaves approximately 10,617 answer tokens over 317.7 seconds (33.4 tok/s). These phase counts are approximate: the UI saves the latest server-polled token count while thinking, rather than retokenizing the finished phases. The chat template is byte-identical between the two oMLX revisions.

The completed request logged 10,260 MTP cycles, 2.82 emitted tokens/cycle, and 79.8% conditional draft acceptance. The earlier aborted attempt logged 2.24 tokens/cycle and 65.4% acceptance, consistent with its roughly 22 tok/s observed speed. Boundary capture/sync/save totaled about 350 ms over the 1,125-second completed request. This cache work does not explain the speed difference.

## Matched server conditions

Both revisions ran sequentially on port 8004 against the same local `Jundot/Qwen3.8-27B-oQ4e-mtp` weights. MLX 0.32.2, mlx-lm 0.31.3 and installed mlx-vlm 0.6.3 were shared. The manual dashboard stayed available on port 8003 with its idle model unloaded during measurement. No other model process was active. Swap usage remained 4,983.62 MiB between the initial and later environment snapshots; `pmset` recorded no thermal/performance warning. These observations do not measure GPU clock stability or recreate earlier desktop activity.

Requests explicitly used temperature 1.0, top-p 0.95, top-k 0, repetition penalty 1.0 and seed 17 or 29. They enabled Lightning MTP and 4-bit Affine4, retaining the final full-attention layer in native BF16. ANE, DFlash, SpecPrefill and VLM MTP were disabled. Prefix caching stayed enabled, with no prefix hits. Each process received the same 1,218-token/128-generation warmup, excluded from comparison.

Revision order was current/old/old/current across the two seeds; thinking-on/off order was reversed correspondingly. Each request allowed 4,096 generated tokens with normal EOS behavior. Temporary instrumentation recorded cumulative MTP statistics every 512 emitted tokens and at completion, without an additional GPU synchronization. Product source was unchanged.

| Revision | Seed | Mode | Generated | tok/s | Tokens/cycle | Acceptance | Backbone ms/cycle | Finish |
|---|---:|---|---:|---:|---:|---:|---:|---|
| current | 17 | thinking | 4,096 | 30.46 | 2.656 | 75.5% | 82.84 | length |
| current | 17 | thinking off | 4,096 | 38.84 | 3.377 | 91.3% | 82.41 | length |
| old | 17 | thinking off | 4,096 | 38.49 | 3.394 | 91.4% | 83.42 | length |
| old | 17 | thinking | 4,096 | 29.63 | 2.576 | 73.6% | 82.74 | length |
| old | 29 | thinking | 4,096 | 27.26 | 2.346 | 67.9% | 82.18 | length |
| old | 29 | thinking off | 95 | 42.45 | 3.429 | 95.7% | 83.33 | stop |
| current | 29 | thinking off | 198 | 43.74 | 3.618 | 96.6% | 82.94 | stop |
| current | 29 | thinking | 4,096 | 29.81 | 2.565 | 73.1% | 81.70 | length |

For thinking, the two-run pooled rate is **30.13 current versus 28.40 old**. Current emits 2.610 tokens/cycle versus 2.456 old; backbone time is 82.26 versus 82.45 ms/cycle. Including the logged MTP-head time gives approximately 85.89 versus 85.77 ms/cycle. The throughput difference closely follows accepted-token yield, not a faster backbone.

The sustained seed-17 SVG-writing control measured **38.84 current versus 38.49 old**, with 3.377 versus 3.394 emitted tokens/cycle and 82.41 versus 83.42 ms/backbone cycle. Across 512-token windows, old throughput ranged from 24.83 to 40.21 tok/s and current from 26.92 to 41.61, depending on output and acceptance.

The seed-29 thinking-off requests emitted tool-call text and stopped after 198 current/95 old tokens. No tools were provided or executed. These are retained as output outcomes and excluded from sustained SVG-speed claims. Thinking-off is a diagnostic variation, not a recommendation for equivalent answer quality. The four thinking-on samples reached the limit while still thinking; the server's final content fallback duplicates the reasoning text and must not be counted as a separate answer phase.

Identical seeds do not produce identical continuations: the initial matched outputs diverged after 18–55 tokens. The implementations use different KV rotations/scale precision and prefill quantization, and adaptive MTP changes random-number consumption. These are matched requests, not a teacher-forced text replay or a statistical quality evaluation.

## Additional long SVG control

Because the second seed did not produce sustained SVG, an additional matched request explicitly asked for a long, complete illustrated dinosaur SVG of at least 600 lines and prohibited tool calls/Markdown. Both revisions used seed 41, thinking off, the same sampler, an 8,192-token cap, and normal stopping. The full prompt and SSE data are preserved. This is a separate workload from the original short prompt.

| Revision | Generated | tok/s | Tokens/cycle | Acceptance | Backbone ms/cycle | Finish |
|---|---:|---:|---:|---:|---:|---|
| current | 8,192 | 39.41 | 3.489 | 93.2% | 83.26 | length |
| old | 4,490 | 39.39 | 3.481 | 93.2% | 83.45 | stop |

The additional request measured **39.41 current versus 39.39 old tok/s**. Old stopped naturally with a complete 4,490-token SVG; current reached the 8,192-token cap with its SVG still incomplete. Neither met the requested 600-line length within these outputs. Comparing the shared emitted-token interval 1–4,096 gives 39.75 current versus 39.45 old tok/s. These are performance samples, not evidence of equivalent output quality.

## Context-length attention isolation

Identical BF16 random Q/K/V tensors were encoded by each codec separately, with dense Qwen geometry: batch 1, 24 query heads, 4 KV heads, D256. Tests cover 4K/8K/16K/32K context and query lengths 1/2/4, including MTP verify widths. Each case received 10 warmups and 41 randomized interleaved measured rounds. Timings include query transformation and attention; KV conversion is outside the interval. The table shows four-row causal attention, median milliseconds per attention layer.

| Context | Original oMLX | Current | VLM PR |
|---|---:|---:|---:|
| 4,096 | 0.3995 | 0.3810 | 0.3600 |
| 8,192 | 0.4630 | 0.5209 | 0.4867 |
| 16,384 | 0.6908 | 0.7571 | 0.7300 |
| 32,768 | 1.1439 | 1.1428 | 1.0765 |

Current is faster or tied for all one-row cases, but some multi-row attention cases remain slower than original oMLX: +12.5% at 8K/four rows, +12.5% at 16K/two rows, +9.6% at 16K/four rows, and +26.0% at 32K/two rows (0.7828 versus 0.6214 ms per attention layer). At 32K/four rows, current and old are tied at about 1.143 ms. VLM remains somewhat faster than current across these shapes. The two-row 32K gap is an isolated tuning opportunity; its whole-model impact was not measured separately. No blanket performance-parity claim is established.

All recorded current native specializations use the bounded FP16 value accumulator. These measurements isolate attention rather than entire MTP cycles. The native oMLX/VLM dispatch checks are included in the raw results. The JSON also contains raw library BF16 SDPA timings; those do not include oMLX's separate verify-split optimization and are not an end-to-end native-KV benchmark.

## Interpretation and limits

The measured 22-versus-30 observation is not evidence by itself of an Affine4 regression: the completed current chat already writes its SVG at about 33 tok/s, and matched old/current requests have similar per-cycle cost. Thinking has lower draft acceptance and therefore emits fewer tokens for each expensive model pass. A longer thinking phase lowers the displayed whole-request average even when the eventual SVG-writing phase is fast.

The original old-patch SVG prompt, exact generated text and historical runtime state were not recovered. The controlled thinking samples stop at 4K and do not replay the full 18K-thought/29K-output manual conversation. Manual whole-run backbone time averaged about 102.4 ms/cycle versus roughly 82–83 in the short isolated controls; context and runtime conditions were not independently separated at whole-model level. The isolated context sweep bounds the attention-specific difference, but does not fully attribute that cross-run absolute cost change. The earlier documented 200K kernel/VLM gap also remains a separate result.

No runtime optimization was committed from this investigation. The evidence does not justify a speculative accumulator, MTP-policy or UI change. The manual branch server is restored after the tests.

[Structured results](affine4_svg_comparison.json), [reproduction archive](affine4_svg_reproduction.zip).
