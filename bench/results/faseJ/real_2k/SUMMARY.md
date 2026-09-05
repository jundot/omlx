# Fase J — real-model A/B (Qwen3.8-Flash-Next-oQ4e-mtp, qwen4_exp)

Protocol: `bench/bench_expert_streaming.py --model qwen --budget 0
--prompt-len 2k --decode 48 --single-request`. Prompt 1898 tokens, 48 decode
tokens, greedy (temperature 0). Baseline = `c1be4b98`; Fase J = `80bf9b9f`.
Both arms run the **same** instrumented bench script (copied into a scratch
worktree), so the comparison is apples-to-apples.

Machine: 48 GiB, ~27 GiB free at start.

## Headline: Etapa C fixes the memory; Etapa E spends it on chunk size

| arm | TTFT | decode tok/s | Metal peak | phys_footprint max | MLX pool max | output |
|---|---|---|---|---|---|---|
| baseline (c1be4b98) | 85.7 s | **3.059** | 7.01 GiB | 37.03 GiB | 30.46 GiB | **A** |
| Fase J (C+E on) | **32.9 s** | 1.991 | 8.33 GiB | 11.25 GiB | 2.51 GiB | B |
| Fase J, Etapa E off | 92.3 s | 2.143 | **6.68 GiB** | **11.09 GiB** | **2.37 GiB** | **A** |
| Fase J, per-layer eval off | 116.2 s | 1.825 | 7.01 GiB | 37.73 GiB | 30.35 GiB | A |

`mlx_peak` = `mx.get_peak_memory()` (active buffers only).
`phys_footprint max` = whole-process lifetime high-water mark (includes the
IOAccelerator-backed allocator pool). **This is the unit the Fase J problem was
reported in (~35.7 GiB), and it is the one that matters.**

## What each etapa actually does

**Etapa C (per-layer eval boundary) delivers the entire memory win, and is
bit-exact.** Turning it off returns phys_footprint to 37.73 GiB and the pool to
30.35 GiB — i.e. baseline. With it on and Etapa E off, the output is identical
to baseline (hash `1765b80737`) in 2/2 runs.

**Etapa E (guard accounting) converts the freed memory into larger prefill
chunks.** That is where the 62% TTFT win comes from — and it is exactly what
breaks bit-exactness: a different chunk size means different GEMM shapes, hence
a different reduction order, hence different logits, hence different tokens.
With `OMLX_STREAMING_BANK_BOUNDARY_ACCOUNT=0` the output returns to baseline
while the memory win is fully retained; the TTFT win is lost.

So the trade is: **bit-exact and 70% less memory (C only), or 62% faster TTFT
and a different greedy decode (C+E).** You cannot have both.

## The output difference is deterministic, not noise

Every arm is perfectly reproducible within itself. Across all 11 real runs there
are exactly two distinct outputs, and they partition perfectly on whether the
per-layer eval boundary + Etapa E accounting are active:

- output **A** (`1765b80737`) — baseline r1, baseline r2, Fase J per-layer-eval
  off, Fase J Etapa E off r1, Fase J Etapa E off r2, e43dcfff r1, e43dcfff r2
- output **B** (`40e84090dd`) — Fase J r1..r4, Fase J barrier=0, Fase J promote=0,0

They diverge at token 3 of 48:

- A: "We need **respond** to user. User **pasted** repeated sentence many times, ending truncated: ..."
- B: "We need **answer** to user. User **repeated** sentence many times, ending truncated: ..."

Both are coherent continuations of the same thought. This is a tie-break between
near-equal logits, not obviously a degradation — but it does fail the project's
`bit_exact_kind=tokens` gate (acceptance criterion 1), and the handoff never
flagged that Etapa E costs bit-exactness.

## Open: decode throughput regresses ~30%, and it predates the memory work

    baseline (c1be4b98)   3.059 tok/s   (3.047, 3.071)
    e43dcfff              2.349 tok/s   (2.526, 2.171)
    Fase J (80bf9b9f)     1.991 tok/s   (1.802, 2.103, 2.064)

The bisect says most of it came from the *earlier* commits
(`5ef31dd6..e43dcfff`: async seed, shared layer I/O, routing-plan reuse), not
from the prefill-memory etapas. Kill-switch arms on Fase J HEAD:

    barrier=0              2.283 tok/s  (2.180, 2.387)   partial recovery
    promote=0,0            2.100 tok/s  (2.005, 2.196)   no real effect (A1/A1b exonerated)
    Etapa E off            2.143 tok/s  (2.106, 2.181)   no recovery

So: A1b is **not** the cause. Etapa B accounts for ~11% of the ~35%. The rest is
in the async-seed / shared-layer-IO commits and is still unlocalised.

End-to-end for this 48-token workload Fase J still wins (58 s vs 101 s) because
TTFT dominates, but for generation-heavy loads the decode term dominates and
Fase J loses.

## Reproducing

```sh
PYTHONPATH=<worktree> <python> bench/bench_expert_streaming.py \
  --model qwen --budget 0 --prompt-len 2k --decode 48 --single-request \
  --min-free-gb 10 --out-dir <dir> --out <dir>/result.json
```

Kill switches used above:
`OMLX_EXPERT_STREAMING_PER_LAYER_EVAL=0` (Etapa C),
`OMLX_STREAMING_BANK_BOUNDARY_ACCOUNT=0` (Etapa E),
`OMLX_EXPERT_STREAMING_LAYER_BARRIER=0` (Etapa B),
`OMLX_EXPERT_STREAMING_BANK_PROMOTE=0` + `OMLX_EXPERT_STREAMING_BANK_PROMOTE_CTX=0` (A1/A1b).

---

# Decode regression: root cause

The headline table above shows decode falling 3.059 -> ~1.9 tok/s. The
handoff predicted this would come from the prefill work changing decode-path
tiles. It does not. **It is an I/O queue-depth collapse, and it took two
commits.**

## Attribution

| commit | decode tok/s | decode CPU% | decode disk GiB/s |
|---|---|---|---|
| c1be4b98 (baseline) | 3.06 | 85 | 0.92 |
| 5ef31dd6 | 2.35 | 59 | 0.72 |
| f799067e | 1.86 | 41 | 0.46 |

Both steps are the same mistake made twice.

**`5ef31dd6` introduced `read_expert_into()`.** It coalesces a demand set into
one buffer per projection and reads it with a single `preadv` per contiguous
run. Fewer syscalls and fewer allocations — genuinely the right idea. But the
runs are issued in a plain `for` loop, one blocking `preadv` at a time. The
old path submitted one job per run to a 16-thread pool. Queue depth went
~16 -> 1.

**`f799067e` added the rolling layer-context path** with
`_CTX_PREFETCH_AHEAD=1`, capping how many of those serialised bank reads can
be outstanding at once. Depth 1 confirmed.

## Why decode and not prefill

| phase | baseline | Fase J | delta |
|---|---|---|---|
| prefill disk GiB/s | 1.59 / 1.72 | 1.37–1.52 | **-9%** |
| decode disk GiB/s | 0.92 / 0.91 | 0.36–0.50 | **-50%** |

Prefill asks for hundreds of contiguous experts per call, so even a serial
reader streams well and the device readahead covers it. Decode asks for a
handful of scattered experts and gets no readahead benefit, so depth was all
it had.

The decisive number: **Fase J reads ~40% fewer bytes during decode than the
baseline (8.7 vs 14.5 GiB) and is still 45% slower.** The work got cheaper
and the pipe got narrower. That is a depth signature, not a volume one, and
it rules out "we are just reading more" as an explanation.

## Ruled out

- **Async hotness seed** (`maybe_seed`): `seed_done` is set in five places and
  awaited in exactly one — a test. Nothing in production joins it, so its SSD
  burst does overlap decode. Measured anyway: seed on 1.72/1.99, seed off
  1.86/1.70 tok/s. Noise (±8% on this box) exceeds the effect. Real code
  smell, not the cause.
- **Etapa A1/A1b** (bank promotion): `promote=0,0` -> 2.100, within Fase J noise.
- **Etapa E**: it changes the output, not the throughput.
- **Etapa B** (layer barrier): `barrier=0` -> 2.28 vs ~2.0.

## Fixes tried (2 reps each, same box)

| arm | tok/s | TTFT | phys_lifetime_max | CPU% | tokens vs baseline |
|---|---|---|---|---|---|
| control (AHEAD=1) | 1.98 / 1.75 | 92.0 / 97.4 | 11.06 / 11.02 | 42.8 / 39.5 | exact |
| `CTX_AHEAD=3` | 2.24 / 2.21 | 86.4 / 84.4 | 11.03 / 11.14 | 50.9 / 48.7 | **exact** |
| `BANK_MAX_BYTES=1` | 2.18 / 2.43 | 72.6 / 73.2 | **9.57 / 9.45** | 70.9 / 75.0 | **exact** |

`CTX_AHEAD=3`: **+19% decode, -9% TTFT, no memory change.** Shipped as the
default in `072e19e2`. 3 already covers every remaining projection, so the
knob is maxed out — it cannot go further.

`BANK_MAX_BYTES=1` forces the bank read to fail and falls back to the legacy
run-parallel path (QD 16). It beats the shipped fix on **all three axes**:
+23% decode, -23% TTFT, *and* 14% lower peak memory. Note it also does 12%
more cache misses (446,730 vs 398,451) and is still faster — the per-miss
cost dominates.

## Conclusion and remaining headroom

The coalesced bank read is not earning its place on this workload. It was
justified by `bench/prefill_mem_harness.py` and by the A1b synthetic
measurements, both of which measured allocation counts, not I/O depth. On the
real model it is slower *and* fatter.

Real queue depth has to be restored inside `read_expert_into` — issue the
per-run `preadv` calls concurrently. That keeps coalescing's allocation wins
while giving back depth, and it is where the remaining gap to the baseline's
3.06 tok/s lives. It needs a nested pool (`read_expert_into` is itself called
from `_EXPERT_IO_POOL` workers, so submitting to that same pool and waiting
would risk deadlock) and a byte-identity test against
`load_expert_slice`, which `test_backing_read_expert_into_matches_load_expert_slice`
already provides in single-threaded form.
