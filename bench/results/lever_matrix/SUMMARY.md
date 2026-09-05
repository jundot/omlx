# Lever matrix + cold tier (2026-09-04)

Protocolo congelado `bench/bench_expert_streaming.py --model qwen-jang|qwen-jang4m --budget 0 --decode 96 --prompt-len short --min-free-gb 12`, warmup descartado, pares adjacentes (drift-canceling) ×3 reps.

## 4S topk/prior tok/s (lever_matrix/4s_v2)

| arm | reps | mediana tok/s |
|---|---|---|
| base | 2.845/2.817/2.811 | **2.817** |
| tk85 (--topk 0.85) | 3.142/3.118/3.164 | **3.142** (+11.5%) |
| tk85+prior2.0 | 3.115/3.160/3.147 | 3.147 (prior null a budget 0) |

## 4S ppl gate (ppl_topk/, --streaming, 24 janelas pg1342)

| topk | ppl | Δ |
|---|---|---|
| none | 1.4848 | — |
| 0.90 | 1.5106 | +1.7% |
| 0.85 | 1.5422 | +3.9% |

(ppl determinístico: 3 reps × 3 massas idênticos ao 4º decimal)

Veredito: **+11.5% tok/s por +3.9% ppl (0.85)**. Knob opt-in documentado; **default permanece bit-exact** (política do projeto — levers de qualidade nunca automáticos).

## 4M ppl gate (topk)

| topk | ppl | Δ |
|---|---|---|
| none | 1.2075 | — |
| 0.85 | 1.2437 | +3.0% |

(3 reps idênticos ao 4º decimal, ambos os modelos — ppl determinístico no streaming path. Custo ppl do 4M < 4S: 3.0% vs 3.9%.)

## 4M cold tier 3-bit (cold_tier_4M/)

- 4M = corpo 4-bit uniforme (144 banks 4b/g64) → alvo do tier. 4S JÁ é 3-bit no corpo (86×3b+43×2b+17×4b) — tier nele cai (requant composto).
- Requant: 56.2 → 42.2 GiB (0.75×), requant_err ≤ 0.16, `_cold_tier_status_dir` = complete (144 banks).
- Fix do tool: chaves per-tensor sem sufixo `.weight` + packing JANG que separa weight/scales/biases entre shards (agrupamento pelo index global).
- Runtime: `OMLX_EXPERT_STREAMING_COLD_ROOT` aponta o tier a um dir arbitrário (este bench usa bench/results/cold_tier_4M/expert_cold).
- **REJEITADO POR POLÍTICA (2026-09-04)** antes do A/B: requant é perda por construção; classe near-lossless não é aceita. Tier (42.2 GiB) removido; tool/runtime ficam dormentes atrás de knobs opt-in. Bug uniform-tier conhecido (linear com bits do source + backing servindo tier) documentado como condição de reabertura — medição ppl/tok/s nunca completada.
