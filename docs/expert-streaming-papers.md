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

- PILOT staging prefetch (router-lookahead, one layer ahead): 94% of staged bundles
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
   decode failed with distinct prompts (Fase A3) and the condition under which it pays.

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

## Measured: JANG 4S/4M SRP/SCH (2026-09-04, traces in `bench/results/lrc/`)

Traces do protocolo congelado (budget 0, short prompt, 43-46 decode calls/layer,
`OMLX_EXPERT_STREAMING_TRACE`, 2 regimes separados por `positions`):

| métrica | 4S | 4M | leitura |
|---|---|---|---|
| SCH ceiling (S≥64/layer) | 77.8% | 77.5% | teto de hit de qualquer cache, até oráculo |
| SCH(S=16) | 65.2% | 65.4% | joelho da curva: 16 slots ≈ 2/3 do teto |
| SCH(S=32) | 74.2% | 74.2% | 42-64 slots ≈ tudo que existe para ganhar |
| repeat adjacente (decode) | 38.5% | 39.3% | bate o ~35% da doc; GLM era 0% — famílias opostas |
| distinct experts/layer no segmento | 104/512 | 103/512 | working set real ~20% do banco |
| top-10 demand share | 40.5% | 41.1% | um grupo fixo pequeno cobre 40% da demanda |
| SRP(G=64, seg=128) | 89.6% | 89.8% | demand-weighted; pin fixo cobre quase todo o volume |
| prefill SCH (S≥256) | 50% | 50% | prefill é broadcast de ~199 uniq/call (570 pos × top-10) — cache não ajuda, seeded burst é o caminho certo |

Conclusões:

1. **Pins: predito pelo joelho, refutado pela matriz** — a matriz pin-knee (`bench/results/lrc/matrix/`) testou o exato joelho SCH (16 slots/layer: 4S 1.5 GiB, 4M 2.0 GiB, 3 reps interleaved A/C, profiles v2 próprios) e ambos ficaram **null (−0.3% na mediana)**: 4S 2.862→2.852, 4M 1.977→1.972 tok/s. O L2-null do oQ4e reproduz no JANG: neste box o page cache já serve o working set de decode até o teto de oráculo — mlock só converte evictable em wired (custo sem ganho). O joelho SCH informa **quantos slots valem a pena se o residente for device-side (slot-bank)**, não quanto pin de página comprar. LRU heap além de ~64 slots/layer continua desperdício.
2. **Slot-bank é o alavanca certa pro hit path**: com hit 78% no teto, o custo per-use (load 43% + stack 28% do wall) cai sobre ~3/4 das chamadas — o bound do spike melhora. O invariante multi-token (#27861) já está no plano (docs/spike-slot-bank.md).
3. **4S ≈ 4M em routing**: routing é propriedade do modelo (512 experts, top-10), não do quant — igual ao achado do post 2x3090 ("hit rate não depende do quant, só de slot count"). MAS o learned-pin profile NÃO transfere entre eles: o fingerprint gate (config_sha + packing — 4S `oQ4e3b` vs 4M `oQ4e4b`) recusa por design (verificado: load do profile 4S no 4M loga `fingerprint mismatch — profile ignored`, pin degrada para observação in-run). Transferir exigiria remapear byte-ranges entre packings distintos; não é caminho.
4. **Prefill**: oráculo limita a 50% (broadcast total) — qualquer esforço de cache em prefill tem teto baixo; o seeded burst existente é o mecanismo certo.

## Measured: levers de bytes/token no JANG (2026-09-04, `bench/results/lever_matrix/` + `ppl_topk/`)

**topk 0.85 (4S)**: matriz interleaved v2 (warmup descartado, pares adjacentes ×3): base mediana 2.817 → tk85 **3.142 tok/s (+11.5%)**; o cache-prior 2.0 é **null** em budget 0 (3.147; sem LRU não há o que ranquear — recusa consistente com o design). **Custo ppl (4S)**: 1.4848 → 1.5106 (0.9, +1.7%) / 1.5422 (0.85, +3.9%). **Custo ppl (4M)**: 1.2075 → 1.2437 (0.85, **+3.0%**). Trade-off explícito: ~+11% tok/s por ~3–4% ppl — **knob opt-in documentado, NÃO default** (política do projeto: defaults bit-exact; autotuner só o varre com `--sweep-topk`; ppl gate separado via `bench/ppl_expert_streaming.py --streaming` — ppl determinístico, 3 reps idênticos ao 4º decimal).

**Cold tier 3-bit (4M) — REJEITADO POR POLÍTICA (decisão do mantenedor, 2026-09-04)**: requantizar 4→3 bits é perda de qualidade por construção, e a classe near-lossless **não é aceita** neste projeto — defaults permanecem bit-exact. O que a linha deixou pronto e **dormente** (atrás de knobs opt-in, nunca default):

- Fixes do requant tool p/ JANGs: chaves per-tensor sem sufixo `.weight` + packing que separa weight/scales/biases entre shards (agrupamento via index global; `--out-dir` p/ tier fora do dir do modelo).
- Runtime: `OMLX_EXPERT_STREAMING_COLD_ROOT` (tier em dir arbitrário; volumes read-only).
- Autotuner: sweep de cold_tier exige `--sweep-cold-tier` explícito + `expert_cold/` presente (nunca automático; testes travam esse contrato).
- Encontrado e NÃO corrigido (condição de reabertura): bug de wiring uniform-tier — a linear é construída com bits do source enquanto o backing serve bytes do tier → `gather_qmm` rejeita o shape. A medição ppl/tok/s **nunca foi completada**; a hipótese (−25% bytes de miss, teto ~+33% no trecho miss-bound) permanece não-testada.

Nota factual: o 4S JÁ É 3-bit no corpo (86×3b + 43×2b + 17×4b) — tier nele nunca fez sentido; o 4M é 4-bit uniforme (144 banks) e o requant funcional mediu 56.2→42.2 GiB (0.75×) antes do veto (artefatos removidos).

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
   F_RDADVISE/PILOT prefetch with the freed headroom.
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
