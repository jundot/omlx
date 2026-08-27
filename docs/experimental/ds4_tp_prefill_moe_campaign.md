# DS4 TP prefill routed-MoE campaign

This is the lossless kernel campaign for the first fixed promotion point:
DS4 Flash, equal TP2 (the UI's TP4/4 split), `M=1024`, `top_k=6`, 256
experts, hidden width 4096, and local intermediate width 1024. It does not
change production dispatch.

## Current path and the high-Amdahl opportunity

`SwitchGLU` makes 6,144 routed rows. The current M3 block path sorts
them by expert, materializes a 48 MiB sorted input, dispatches pair-concat for
up and gate, materializes a 24 MiB pair, writes a 12 MiB LimitedSwiGLU mid,
dispatches down into a 48 MiB sorted route output, unsorts it, casts it back to
BF16, multiplies six BF16 route rows by BF16-cast scores, and reduces the six
slots. Pair-concat is useful launch packaging, but its `pair_id` selects one
projection per threadgroup; it does not share the activation tile between up
and gate.

At this shape each expert has 24 routes in the deterministic benchmark, so a
BM32 plan contains exactly 256 useful blocks. Per rank and layer the three
MXFP4 expert projections contain about 1,632 MiB of weights. The routed MoE
does 154.62 GFLOP per 1,024-token layer chunk, or 6.49 GFLOP/token/rank over
43 layers. A 1,000 tok/s target therefore asks for about 6.49 TFLOP/s per
rank from routed MoE; arithmetic capacity is not the obstacle.

## Phase 0: deterministic expert work map

The route identity is `route_id = token * 6 + slot`. The canonical order is
the stable sort key `(expert_id, route_id)`. Expert counts are converted to
BM32 block offsets by a deterministic exclusive prefix sum. Atomics may fill
neither equal-expert route order nor block order. The existing sort can remain
for the first kernel proof; a later indirect-X loader can consume the route-id
map directly and remove the 48 MiB sorted-X materialization.

The ds4-metal map builder is the structural reference, with one important
change: its token-centric atomic scatter explicitly leaves expert-local order
unspecified. oMLX needs stable route ids because the exact phase-B reduction
is keyed by original top-k slot.

There is a second non-transferable detail. ds4-metal folds the route weight
into the SwiGLU mid before down. Current oMLX applies it after the down result
has narrowed to FP16 and then cast to BF16. Moving the score across the down
matmul is equivalent over real numbers but not through these rounding
boundaries. Phase A therefore fuses LimitedSwiGLU but not the route score;
phase B fuses the score into the post-down deterministic epilogue. Pre-down
score folding may be tested as an explicitly non-promotable probe, but it
cannot enter the lossless path without array-equal proof.

## Phase A: shared-X gate/up plus LimitedSwiGLU

The proposed primitive ABI is frozen by
`benchmarks/bench_ds4_tp_prefill_moe_campaign.py` as
`deepseek_mxfp4_gather_qmm_pair_swiglu_blocks`.
The original fixed-shape sketch remains in
`benchmarks/prototypes/ds4_tp_prefill_moe_phase_a.metal`. Its phase-A kernel is
now exposed as an isolated native symbol by
`omlx/custom_kernels/glm_moe_dsa/csrc/ds4_prefill_moe.{cpp,h,metal}` for the
M3/M5 ABBA gate. No model, planner, or runtime path calls the symbol.

The kernel keeps BM32/BN32/BK32 and the existing Steel MMA K sequence. One
threadgroup stages X once, loads independent up and gate MXFP4 tiles, and
updates two independent FP32 accumulator sets. Before LimitedSwiGLU, each
accumulator is explicitly narrowed to FP16 exactly where the two current
projection outputs are stored. The epilogue then applies:

1. `gate = min(gate, 10)` in FP16;
2. `up = clamp(up, -10, 10)` in FP16;
3. the same FP16 SiLU and multiply sequence as `_limited_swiglu`; and
4. one FP16 activated output `[R, 1, 1024]`.

The first parity gate is `mx.array_equal` at that activated boundary. Final
logits alone are not sufficient. Promotion also requires cache/logit parity
through a real prompt and no regression of the current rollback path.

### Phase-A fixed-shape result (2026-08-22)

The callable isolated symbol is exact versus pair-concat on both Macs, after
replacing the sketch's algebraic sigmoid with MLX's stable `Sigmoid` functor.
It does not pass the performance gate:

| Host | Shared-X | Pair-concat | Stock | Speedup vs faster baseline |
|---|---:|---:|---:|---:|
| M3 Ultra rank 0 | 8.209 ms | 8.017 ms | 9.091 ms | 0.977x |
| M5 Max rank 1 | 9.369 ms | 9.306 ms | 7.541 ms | 0.805x |

On M5, stock NAX is itself not array-equal to the Steel pair-concat path
(maximum activated-boundary difference 0.0078125), so the requested equality
to both references is unsatisfiable there. The symbol remains isolated and
unused; there is no production feature flag or dispatch. Machine-readable
results are in `ds4_tp_prefill_moe_phase_a_results_2026-08-22.json`.

Run the same ABBA-style harness separately on both machines after the isolated
symbol exists:

```bash
python benchmarks/bench_ds4_tp_prefill_moe_campaign.py \
  --model /path/to/DS4-Flash --device-label m3-ultra --rank 0 --strict

python benchmarks/bench_ds4_tp_prefill_moe_campaign.py \
  --model /path/to/DS4-Flash --device-label m5-max --rank 1 --strict
```

The candidate must be array-equal to both references and beat both pair-concat
and stock; this matters on M5, where
NAX stock gather-QMM can be the better production baseline. The default gate
is exact parity and at least 1.05x versus the faster baseline. Record M3 and
M5 results independently; do not average away a straggler regression.

Phase A removes the 24 MiB gate/up materialization and halves the logical
gate/up X-tile loads. For the exact fixed shape, the analytical logical-byte
ceiling is about 1.33x for routed MoE and about 1.14x end to end if routed MoE
is half of prefill time. The absolute Amdahl ceiling, if all gate/up time
vanished, is 1.50x end to end. These are ceilings, not expected results.

## Phase B: down plus exact fixed-top-6 reduction

Phase B must preserve these otherwise easy-to-miss boundaries:

1. each down FP32 accumulator is stored/narrowed to FP16;
2. that FP16 value is cast to the original BF16 activation dtype;
3. the FP32 score is cast to BF16;
4. multiplication is BF16; and
5. accumulation is BF16 in original slot order 0, 1, 2, 3, 4, 5.

Unordered atomics are forbidden. Expert-major threadgroups complete in an
unspecified order, and BF16 addition is order-sensitive. The earlier fused
decode primitive is a warning: isolated layer parity is not enough when a
different cast or reduction topology can perturb later routing and caches.

Three schedules are valid experiments:

1. Existing expert-major down plus a deterministic fixed-slot second
   reduction. This is the exact oracle but retains the 48 MiB route buffer.
2. Six sequential per-slot expert maps writing/accumulating directly to the
   local `[L,4096]` row. This removes the route buffer and is exact, but the
   deterministic M=1024 fixture activates all 256 experts in every slot: the
   analytical down-weight read amplification is exactly 6.0x. Reject it unless
   measured cache behavior offsets that amplification.
3. A bounded output-tile scratch (384 KiB for `R x BN32`) followed by the
   deterministic slot reduction before reusing the tile. This preserves the
   expert-major weight schedule and removes the full route tensor, at the cost
   of more dispatch boundaries. Sweep wider supertiles to trade scratch for
   dispatch count.

Avoiding the 12 MiB mid as well requires an expert-persistent kernel that keeps
an activated BM tile on chip while walking down-output tiles. BM32 activation
alone is 64 KiB, before X/weight staging, so BM8/BM16 persistent variants must
be benchmarked rather than assumed faster. A production A+B path is not
allowed to claim the no-mid ceiling while silently allocating an internal
`[R,1024]` temporary.

The combined ideal removes 84 MiB of core persistent intermediates per layer
(pair + mid + routed down); indirect-X can remove another 48 MiB. Under the
documented perfect-on-chip-reuse byte model, A+B has about a 1.99x routed-MoE
and 1.33x end-to-end ceiling at a 50% MoE share. Its absolute infinite-speed
Amdahl ceiling is 2.0x end to end.

## Promotion order

1. Land the deterministic map and phase-A isolated symbol only.
2. Pass activated-boundary exactness and M3/M5 M=1024 ABBA gates.
3. Run full-layer and full-model cold-prefill parity/performance gates.
4. Promote phase A behind an independent rollback.
5. Establish the phase-B deterministic second-reduction oracle.
6. Benchmark per-slot and bounded-tile schedules; retain only a measured win.
7. Attempt BM8/BM16 persistent no-mid variants, again behind an independent
   rollback.
8. Only after local output is exact, place the existing TP `all_sum` directly
   after `[L,4096]`; collective order and count remain unchanged.
