# Expert Streaming — Paper Survey and Gap Analysis (2026-08)

Study comparing recent MoE-offloading papers against this repo's expert-streaming
implementation, to identify further latency / tok/s levers **without quality loss**.
No code was changed for this analysis; it records what is already covered, what was
measured negative here, and what is actually worth building next.

Sources: a curated paper list provided by the maintainer (2026-08), read against
`docs/expert-streaming.md` (phases 1–G), `omlx/patches/expert_streaming/`, and the
model integration layer.

## TL;DR — the regime mismatch

Almost every paper on the list optimizes a **PCIe regime** (experts in CPU RAM,
cache in GPU VRAM, 8–25 GB/s transfers) where prefetch and compute overlap hide the
transfer because bandwidth is left over. Our regime is **saturated NVMe** (~2.3–2.5 GB/s
effective random reads of a 3.7 GB/s sequential drive): the disk saturates before the
GPU does. We have already implemented and measured the headline techniques of these
papers **twice**, and both times they lost:

- Router-lookahead staging prefetch (measured negative; removed — see main doc). The kernel-side F_RDADVISE readahead keeps the useful half.
  dropped unconsumed, 0.037 → 0.011 tok/s.
- Warm-only prefetch (previous token's next-layer experts, ~35% repeat): neutral to
  negative on the QD16 demand path.

The transferable value concentrates in three families, which independently match the
empirical conclusions of our own doc:

1. **Bytes/token reduction via precision tiers** (HOBBIT, MoBiLE) — our doc already
   names "requant of cold experts ≈ 2×" as the next real lever.
2. **Routing-consistency analytics before choosing cache policy** (LRC / SRP-SCH) —
   formalizes our measured Qwen 23–32% vs GLM 0% inter-token expert reuse.
3. **Route-affinity scheduling** (ExpertFlow's token scheduler) — explains why batch
   decode failed with distinct prompts (batch pays only when requests share routing).

## What the implementation already covers

- LRU expert cache (opt-in) — lost to **page-cache-only** default even at 14.5% hit
  rate (B2); big LRU budgets go negative past ~8 GiB on a 48 GiB box (A1).
- pread per expert slice + QD16 thread pool + run coalescing (10.6 GB/s cold vs 0.35
  for mmap+MADV_RANDOM; QD16 = +34% over QD8, E1).
- mlock pins + **learned pin store** persisted across runs (B3/E3: +15% decode,
  +6–10% with profile preload, −22% TTFT).
- Prefill-hotness cache seeding at the prefill→decode boundary (G3, ~720 slices on
  qwen 512) — the request-level version of FineMoE's expert map.
- Router-lookahead prefetch (Eliseev-style) — measured negative in this regime (above).
- Mixed quantization: oQ4e experts, 8-bit shared experts, per-tensor overrides.
- Adaptive top-k mass truncation (B4, opt-in approximate: +27% GLM at 0.85).
- MTP speculative decode, default-on for qwen4_exp with `mtp.*` (+27–37% at QD8,
  +4–5% at QD16, E2/B5).
- F_RDADVISE kernel readahead on predicted runs (G2, within noise at 512/2k scale).
- Dual-SSD striping, chunked-prefill memory guard with honest watermarks (F1),
  off-boundary MLX pool release (G1, near no-op — peaks are intra-chunk).
- Batch decode measured **negative** with distinct prompts (A3); streaming stays
  single-request.

Reference numbers (48 GiB Mac, 3.7 GB/s NVMe): Qwen 2.06 tok/s cold decode; GLM 0.697;
DSV4 1.22; 8k prefill ~60% GPU-bound (expert QMM + assemble); decode is I/O-bound
(GPU 8–35%).

## Paper-by-paper mapping

| Paper | Core technique | Status here |
|---|---|---|
| Eliseev & Mazur 2023 | LRU + next-layer router-logit prefetch + mixed quant | Covered/superseded (prefetch negative here) |
| MoE-Infinity | Activation traces → sparsity-aware cache + prefetch | Covered (G3 seeding + E3 learned pins) |
| HOBBIT | Low-precision substitution of cache-miss experts | **Not covered — top opportunity** |
| LRC (Not All Models…) | SRP/SCH routing-consistency metrics | Not covered as a tool; our findings are the phenomenon |
| ExpertFlow | Trained route predictor + token scheduler + predictive cache | Partially covered; predictor = P2 |
| FineMoE (fMoE) | Fine-grained expert maps + semantic hints → prefetch | Covered in essence (G3); GLM routing volatility caps the upside |
| SMoE (ISCA 26) | Substitute missed expert with similar resident one | Inferior alternative to a precision tier |
| MoBiLE | Big-little expert precision split | Reinforces the HOBBIT-style tier |
| LightMoE | Task-aware expert availability | Multi-tenant serving future |
| SiDA-MoE | Trained hash predicts experts pre-router | Covered/superseded; DSV4 ships hash routing natively (`tid2eid`, layers 0–2) |

Notes per paper:

- **HOBBIT** is the direct, structured version of our "bytes/token" lever: on a cache
  miss, serve a low-precision copy of the expert instead of stalling on the full read.
  Their copies live in RAM; ours cannot (Qwen 99G ≈ 17 GB at ~2-bit; GLM 190G ≈ 40+ GB),
  so the adaptation is a **low-precision tier on the SSD itself**: the hot set
  (pins/hotness) reads oQ4e, misses read oQ2.7 (oQ2/oQ2.5/oQ2.7/oQ3 already supported by
  `omlx/oq.py`). Expected ~2× on the I/O floor for GLM-class decode and on Qwen's miss
  path — and, by halving bytes, it re-opens headroom where prefetch may start to pay
  (sequence: bytes first, re-test prefetch after). Near-lossless, not bit-exact: needs a
  quality harness first (none exists in the repo today).
- **LRC** proposes SRP (can a fixed expert group cover a token segment) and SCH (hit
  rate of a cache given future information) and finds optimal cache ≈ 2× active experts,
  shared experts reduce consistency, domain specialization drives consistency. Our
  measured knee (Qwen top-10: ~32 experts/layer useful at 4 GiB; 8 GiB negative) is
  consistent with the ≈2× rule. A cheap **bit-exact offline analysis script** would turn
  pins/seed/threshold defaults into per-model measured decisions instead of hour-long
  benches, and pre-answers "does this new model suit streaming".
- **ExpertFlow**: the trained all-layer route predictor only pays with spare bandwidth
  (P2, revisit after the precision tier). The token scheduler cannot reorder causal
  prefill (attention order matters), but its multi-request form — co-scheduling requests
  with overlapping routes — is the path to re-opening batch for homogeneous traffic (A3).
- **SMoE** substitution reduces bytes/token without a second copy but silently changes
  outputs and needs an offline similarity index; a uniform, auditable, switchable oQ2.7
  tier dominates it on cost/benefit.
- **MoE-Infinity v3's** insight (batch=1 decode leaves a small reusable hot set) holds
  for Qwen-like routing, not GLM's (0% inter-token reuse at any budget up to 8 GiB).

## Measured: JANG 4S/4M SRP/SCH (routing-trace harness, budget 0, short prompt)

| Metric | 4S | 4M | Reading |
|---|---|---|---|
| SCH ceiling (S>=64/layer) | 77.8% | 77.5% | hit ceiling of any cache, Belady included |
| SCH(S=16) | 65.2% | 65.4% | the knee: 16 slots ~ 2/3 of the ceiling |
| SCH(S=32) | 74.2% | 74.2% | 42-64 slots ~ everything there is to gain |
| adjacent-call repeat (decode) | 38.5% | 39.3% | matches the ~35% band; GLM measured 0% — opposite families |
| distinct experts/layer per segment | 104/512 | 103/512 | real working set ~ 20% of the bank |
| top-10 demand share | 40.5% | 41.1% | a small fixed group covers 40% of demand |
| SRP(G=64, seg=128) | 89.6% | 89.8% | demand-weighted; a fixed group covers nearly all volume |
| prefill SCH (S>=256) | 50% | 50% | prefill is a ~199-uniq/call broadcast — the seeded burst is the right mechanism |

Conclusions:

1. **Pins: predicted by the knee, refuted by the matrix** — the pin-knee matrix tested the exact SCH knee (16 slots/layer; 3 interleaved reps, own v2 profiles) and both measured **null (-0.3% median)**: 4S 2.862->2.852, 4M 1.977->1.972 tok/s. On this box the page cache already serves the decode working set up to the oracle ceiling — mlock only converts evictable into wired. The SCH knee informs how many slots a *device-side* residency scheme should hold, not how much page-pin budget to buy; an LRU heap beyond ~64 slots/layer stays waste.
2. **4S ~ 4M routing**: routing is a property of the model (512 experts, top-10), not the quant — matching the 2x3090 finding that hit rate depends only on slot count. But learned-pin profiles do NOT transfer between them: the fingerprint gate (config sha + packing, 4S `oQ4e3b` vs 4M `oQ4e4b`) refuses by design.
3. **Prefill**: the oracle caps at 50% (total broadcast) — any prefill cache effort has a low ceiling.

## Measured: bytes/token levers on the JANGs (interleaved A/B, warmup discarded)

**topk 0.85 (4S)**: base median 2.817 -> 3.142 tok/s (**+11.5%**); cache-prior 2.0 is null at budget 0 (no LRU to rerank — the converter refuses it, consistent with the design). PPL cost: +3.9% (4S), +3.0% (4M). Explicit trade: ~+11% tok/s for ~3-4% ppl — an opt-in documented knob, never a default; the autotuner sweeps it only with `--sweep-topk`, and the ppl gate runs via `bench/ppl_expert_streaming.py --streaming` (deterministic to the 4th decimal).

**Cold tier (uniform low-precision requant)**: rejected as a *default* line by maintainer policy — defaults stay bit-exact. The tier ships dormant behind opt-in knobs (`expert_streaming_cold_tier` + the HOBBIT hot/cold split), with the quality/speed Pareto measured per model in the main doc.

## Prioritized opportunities

**P0 — bit-exact (zero quality risk):**

1. **G4 — per-layer `mx.eval + clear_cache` for qwen4_exp.** The DeepSeek loader's
   pattern is the known-good answer to intra-chunk lazy-graph accumulation (~29 GiB
   pool peaks); the doc marks it "not built for qwen4_exp". Bounds allocator churn,
   protects the page cache the run depends on — attacks the 341 s/8k shared-box case
   and budget-0 decode disk-boundness.
2. **Learned pin store server integration** (E3 follow-up, already "queued": save on
   unload, reload on load, per model). Measured +6–10% decode, −22% TTFT; zero output
   change (mlock'd file ranges only).
3. **SRP/SCH offline analysis per model** (script + doc section): per-model defaults for
   pins / seed size / top-k threshold, and a pre-flight "does streaming pay" check for
   new checkpoints.

**P1 — near-lossless (requires an explicit quality policy):**

4. **Dual-precision expert tier on SSD** (HOBBIT/MoBiLE): requant cold experts to oQ2.7
   (or oQ3) with the existing oQ quantizer; hot set stays oQ4e; misses read the cheap
   tier → ~½ miss bytes. Prerequisite: a **perplexity/quality harness** (gap — verified
   absent from bench/tools/scripts) plus tok/s & TTFT A/B. Afterwards, re-test
   F_RDADVISE readahead with the freed headroom.
5. **Top-k 0.85** is already implemented (B4, +27% GLM); today's opt-in default is
   correct — only a per-profile default decision remains.

**P2 — document why NOT to pursue (for now):**

- Trained route predictors (ExpertFlow/SiDA): per-model training cost, saturated disk;
  revisit only after the precision tier.
- Expert substitution (SMoE): silent output changes; the oQ2.7 tier dominates.
- Position-level prefetch maps (FineMoE): G3 + GLM routing volatility already cap it.
- More LRU/RAM cache: measured negative (B2/A1).

## Validation & risks

- Reuse the existing protocol: `bench/bench_expert_streaming.py` +
  `bench/resource_sampler.py` + `bench/cache_cool.py`, `--min-free-gb` / `--mem-ceiling-gib`
  sized to psutil *available* (post-F lessons), A/B only in idle windows (G-series
  numbers were dominated by machine state on the shared box).
- Main P1 risk is quality: build the perplexity harness before any requant decision.
- Sequencing matters: bytes/token first (P1-4), then re-test prefetch with headroom.

## Sources

- ExpertFlow (DAC'26): arxiv.org/abs/2410.17954
- LRC — Not All Models Suit Expert Offloading (v4): arxiv.org/abs/2505.16056
- HOBBIT: arxiv.org/abs/2411.01433
- FineMoE/fMoE: arxiv.org/abs/2502.05370
- MoE-Infinity (v3): arxiv.org/abs/2401.14361
- Eliseev & Mazur 2023: arxiv.org/abs/2312.17238
- SMoE (ISCA'26), MoBiLE (ASP-DAC'26), LightMoE (ACL'26 Findings), SiDA-MoE (MLSys'24):
  assessed from the maintainer's curated summaries (direct PDFs inaccessible during this
  study); treat their fine print as secondhand.
- A Survey on Mixture of Experts in LLMs (TKDE'25): positioning only.
