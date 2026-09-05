# Spike: Slot-bank + grouped-GEMM (Fase 3 backlog)

**Status**: IMPLEMENTADO atrás de env (`OMLX_EXPERT_STREAMING_SLOT_BANK=1`) — **kill-gate NÃO atingido (+0.4%, dentro do drift)**. Veredito medido 2026-09-04 abaixo; código retido (default off, 198/198 testes verdes, custo-zero com feature off) porque o fast path é bit-exact e o aprendizado está nos números.

## Veredito medido (bench/results/slotbank/ + replay)

A/B congelado no JANG 4S (budget 0, short, 3 reps interleaved): A mediana **2.762** tok/s vs B (S=16) **2.772** → **+0.4% = null**. O replay do trace real (`/tmp/replay2.py`, lógica lookup/insert/LRU idêntica à `SlotBank`) explica o porquê:

| S (slots/layer) | hit per-expert (LRU real) | **full-hit calls** (fast path dispara) | evictions |
|---|---|---|---|
| 8 | 27.9% | 0.0% | 15871 |
| 16 | 49.0% | 4.9% | 10744 |
| 32 | 64.3% | 12.7% | 6529 |
| 64 | 72.7% | 20.7% | 3123 |
| 128 | 77.6% | 27.0% | 197 |

1. **O fast path exige FULL-HIT da demand set inteira** (10 experts/call) — com churn LRU real, só 5% das chamadas a S=16 (27% a S=128, ~6.4 GiB wired). Hits parciais ainda pagam promote+stack do caminho legacy **mais** o insert row-assign — por isso o null não é anomalia, é a aritmética: o slot-bank elimina stack+promote apenas em full-hits, e eles são raros.
2. **Gap Belady vs LRU confirmado no trace**: SCH oráculo S=16 = 65% vs LRU real 49% (a doc do LRC paper já media ~2.7× gap) — o ceiling de 77.8% é inatingível por eviction policy; só a ordenação temporal do trace trava isso.
3. **Regime**: no 2x3090 (PCIe), o hit caro (RAM→VRAM por uso) faz o slot-bank pagar com 84-92% de hit. Aqui o hit já é pread de page-cache (barato) — o custo por uso é o promote+stack, que o full-hit elimina, mas os inserts (misses de churn) custam row-assign sem eliminação equivalente.
4. **Lição de design**: para este regime, o próximo experimento seria S=64-128 (full-hit 21-27%) com o insert **condicionado a churn observado** (skip insert se o expert não repetir) — mas o teto aritmético (~21% × 30% wall) ≈ +6% não justifica 6+ GiB wired. Reabrir somente se o working set de decode encolher (ex.: topk 0.85 combinado com prior — o trace desse regime teria menos experts/call e mais full-hits).

---

**Histórico do scoping original** (números GLM-JANG que motivaram o spike — o diagnóstico permanece válido, o gate é que não foi atingido):

Fundações já existentes:
`deepseek_mxfp4_gather_qmm_blocks` + variantes pair/concat e `glm_moe_weighted_sum` em `omlx/custom_kernels/glm_moe_dsa/csrc/fused_moe.{h,cpp,metal}`, expostos em `fast.py` e consumidos por `switch_layers.py`/`deepseek_v32.py`.

## Numbers from the GLM-JANG decode (budget 1 GiB, prior 2.0, PROFILE=1, 42 layers × 12,726 calls, wall 62.6 s)

| stage | total s | share of wall | ms/call |
|---|---|---|---|
| load (LRU take → promote np→mx) | 26.97 | **43.1%** | 2.12 |
| stack (mini-bank assembly + graph build) | 17.83 | **28.5%** | 1.40 |
| gate_eval (route eval + d2h copy) | 14.10 | 22.5% | 1.11 |
| unique (np.unique remap) | 0.11 | 0.2% | 0.01 |

Hit 28% (32,261 hits / 82,630 misses global). O caminho custa ~3.5 ms/call em load+stack sozinhos — é daí que vem o teto de 1.1–1.3 tok/s.

## Diagnosis

Todo token paga `load+stack` **por chamada** mesmo em hit: o hit path toma U slices numpy do LRU e os re-promove a mx + re-stacka o mini-bank a cada chamada. O load é dominado pelo take do LRU + np→mx copy; o stack, pela montagem (U,O,I) + graph build. Ou seja: o custo por uso é proporcional ao número de chamadas × experts, não à novidade (misses). Com hit 28%, ~72% do tráfego de montagem é redundante — o mesmo expert é re-promovido e re-stackado token após token.

## Proposal: slot-bank

1. **Layout**: alocar S slots por (layer, proj) — buffer persistente mx (S, O, I) + escala/bias. Slot = residência de um expert. Mapa expert_id→slot_idx em numpy no host; MRU evict quando cheio.
2. **Insert**: no miss, escrever o slice direto no slot (uma cópia np→device, substituindo take+stack).
3. **Gather**: usar `mx.take`/indices do slot no `gather_qmm` existente (remap muda de rank-of-uniq para slot_idx — mesmo shape, zero mudança nos kernels do caminho atual). O `deepseek_mxfp4_gather_qmm_blocks` com block_meta/block_count (o path Metal direto) fica como fase 2 do spike, pois o JANG é affine-int8 (group 64) e o kernel affine (`deepseek_affine_gather_qmm_blocks`) já existe com a assinatura certa.
4. **Evict**: invalidar slot (mapa host) + overwrite no próximo insert; sem free/alloc de mx arrays.

## Esperado (bound calculado)

- Elimina o stack per-call nos hits: −17.8s de 62.6s wall (−28.5%) → 44.8s se load e gate não mudarem.
- Load em hit degrada para custo O(1) de map-lookup (o slice já está no device): praticamente zera o load_ms nos hits → load 26.97s × 72% hit-path share... na verdade com hit 28%: load já é 43% do wall **já** majoritariamente em hits — o take + promote acontece também nos hits. Se o slot-bank cobre 100% dos hits, remove ~28% das chamadas de load mais barato... conservador: −10 a −15s adicionais.
- Estimativa total: 62.6s → ~35–40s decode wall ⇒ 1.3 → **~1.9–2.2 tok/s** (mesmo sem mexer em gate_eval).

## Kill-gate

+10% tok/s no protocolo congelado (3 reps prior 2.0 short vs 3 reps exact). Se o slot-bank não entregar, reverter sem drama — nenhum path compartilhado é alterado (o cache LRU numpy permanece para o cold path / page-cache-only).

## Wiring points (já mapeados)

- `streaming_switch.py:2011` `StreamingQuantizedSwitchLinear.__call__` — resolve demand set, chama `_promote_banks` (2011–2493):
  - `_split` (917) → LRU hit/miss
  - ctx fast path (2041–2054) + Etapa A1b bank promote (2196–2214)
  - tier_single promote direto (2216–2229, 2450–2473) — **ponto de inserção do slot-bank**: substituir `_stack_tier`/`mx.stack` por take no slot-bank
  - `gather_qmm` final (2462–2473) — remap troca para slot_idx
- `shard_bank.py:1273+` `load_expert*`/`read_expert_into` — leitura do disco; inalterado (miss ainda lê, mas escreve no slot)
- `ExpertLRUCache` (`streaming_switch.py:660+`) — budget/capacity; o slot-bank usa budget próprio em device (S × (O,I) bytes × dtypes) e contabiliza no mesmo `budget_gib`
- Kernels: `deepseek_affine_gather_qmm_blocks(x, weight, scales, biases, block_meta, block_count, group_size, bits, variant)` em `fused_moe.h:55` — assinatura casa perfeitamente com o slot-bank (weight/scales/biases = slots; block_meta/block_count = mapa de posição)

## Riscos

- **Fidelidade**: take/gather no slot é bit-identical por construção (mesmos bytes, mesma ordem) — mas a invalidação de slot tem que ser à prova de staleness (mapa expert→slot no host, uma entrada por expert; sem duplicatas).
- **Memória**: slots residem em wired/device memory — com 45 layers × 2 projs × S... para GLM: 204 experts/layer? Não — checar `n_routed_experts` do config JANG: com budget 1 GiB e (O,I) do MoE GLM, S ≈ 40–80 experts globais. `mx.metal.set_cache_limit` / budget accounting já existe (`ExpertLRUCache.capacity`).
- **Arena de slots vs promove a cada call**: o ganho real é eliminar re-promoção de hits. Se o hit rate for baixo (28%), o ganho limita-se a ~1/3 do tráfego — por isso o gate de +10% é o critério de vida.
- **Desafio prévio documentado**: single-promotion parcial (scoreboard) mostrou concat ≈ stack em wall — **o custo está no tráfego de montagem, não nas alocações**. O slot-bank ataca exatamente o tráfego (sem tráfego nos hits, só o gather final).

## Invariante multi-token (lição llama.cpp #27861)

O post 2x3090 Part 2 documenta um bug real no expert cache deles que o slot-bank **não pode repetir**: todo expert uncached era mapeado para **um único dummy slot**, mas os kernels `mul_mat_id` em batch assumem **ids distintos por token** → OOB writes. Só o path `mmvq` (experts quant até 8 tokens) era seguro, então o branch deles gata o cache em 8 tokens e o verify MTP em 4.

Nosso caminho atual já é seguro por construção e o slot-bank tem que preservar isso:

- `_build_plan_into` (`streaming_switch.py`) faz union do demand set **across positions** via `np.unique(flat)` + `searchsorted` remap — dois tokens que roteiam experts disjuntos nunca colapsam para o mesmo índice; o `remapped` tem o mesmo shape de `indices`.
- Slot-bank: o remap muda de rank-of-uniq para **slot_idx**, mas a cardinalidade tem que ser a do union (um slot distinto por expert distinto no batch). Nunca mapear dois uncached para o mesmo slot na mesma chamada; evict/insert no meio de um batch verify não pode corromper a outra posição do mesmo batch.
- Caso de regressão obrigatório: batch verify MTP (2 posições, expert sets distintos — o mesmo que custa 2.3x leituras no Gap-2) + batch 4/8 tokens com sets disjuntos: bit-idêntico vs path sem slot-bank, sem aliasing, sem OOB.
- Bateria de fidelidade do kill-gate passa a incluir greedy multi-token (verify-shaped), não só single-token decode.

## Estimativa de esforço

2–3 dias: (1) slot map + insert/evict no linear, atrás de env `OMLX_EXPERT_STREAMING_SLOT_BANK=1`; (2) remap slot_idx + take no caminho dual-tier e uniform; (3) bench A/B congelado + kill-gate; (4) se passar: testes de fidelidade (bateria 6 prompts greedy + ppl canário) + auditor + settings/ui plumbing se virar default.
