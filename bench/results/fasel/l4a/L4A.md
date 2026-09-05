# Fase L4A — dual-tier memory diagnosis (8k HOBBIT, memtrace per-tier events)

Protocol: 8k single-request decode-48, budget 0, cold-tier 3, gate-tokens,
min-free-gb 16. Events per the plan: dual_tier.{hot,cold}.bank_ready,
dual_tier.{hot,cold}.qmm_submitted, dual_tier.mask_ready, dual_tier.add_submitted,
dual_tier.layer_exit, plus enter/exit and ctx/glu frames. Every record carries
layer, projection, positions, hot/cold positions + bank bytes (and the usual
active/cache/peak/footprint).

## Answers (hf=0.25 and hf=0.10)

1. WHICH EVENT CREATES THE PEAK? None of the dual-tier events.
   - The allocator peak reaches 14.232 GiB BEFORE the first ctx.ensure of
     the measured window and is equal at every later event: the plateau is
     the chunk-wide lazy graph (attention + dense + expert banks held until
     the chunk-end eval), not the per-layer tier assembly. Within one layer
     build, ACTIVE grows 5.09 -> 11.9 GiB monotonically; hot/cold bank
     READY->QMM->mask->add add ~0.2% on top (per-layer banks: mean 21 MiB
     both tiers, max 339 MiB at hf=0.25).
2. DOES THE PEAK SCALE WITH THE HOT/COLD RATIO OR WITH POSITIONS?
   - Position-driven, ratio-blind. Correlation of layer_exit peak with
     hot_positions 0.43-0.44, cold_positions 0.44-0.45, total positions
     0.44 — statistically identical. Moving hf 0.25 -> 0.10 leaves the
     14.232 GiB peak exactly the same; only the per-tier bank SPLIT moves
     (hot mean 8.2 -> 3.6 MiB, cold mean 13.0 -> 16.5 MiB; max hot 109 ->
     45.7 MiB, max cold 230 -> 281.6 MiB).
3. LAYER BOUNDARY RETENTION? 99.8-99.9% of ACTIVE memory survives
   layer_exit into the next layer's enter: the lazy graph retains every
   layer's banks until the chunk-end eval (the F4A plateau conclusion,
   now with per-tier attribution).
4. GATE/UP/DOWN PROFILES? Identical (each projection 24696 events, the
   same 14.232 GiB peak) — no projection-specific driver.

## Conclusion for L4B
- B1 (eval between tiers), B2 (mask-free reassembly) target tensors 1-2
  orders of magnitude below the plateau driver; both closed by this
  measurement without implementation.
- B3 (small-tier-first order) implemented + bit-identity test + measured
  (run3, small-first): see agg_hf25_smallfirst.txt — expected no peak
  change (the tier banks are ~20 MiB of an 11.9 GiB graph).
- B4 (phase policy: uniform tier for long prefill) stays a documented
  product option with the PPL gate at release time, as F4B already noted.
- The <=10 GiB target is only reachable via the chunk-end eval boundary
  (Fase I1 machinery) or chunk schedule changes — both Re-baseline items
  outside this phase's tier-lifetime variants. HOBBIT 8k stays within the
  28 GiB ceiling on this box (guard throttles 8k chunks to 1024; peak
  14.23 GiB, phys lifetime max 20.7 GiB, no swap).

## Runs
- run1_hf25.json / memtrace_hf25.jsonl: tok 2.745, ttft 98.2s, peak 14.232,
  decode metal 5.22, phys 20.7, tokens 48.
- run2_hf10.json / memtrace_hf10.jsonl: tok 2.899, ttft 91.5s, peak 14.232,
  decode metal 5.22, phys 19.6, tokens 48.
- run3_hf25_smallfirst.json: B3 arm.