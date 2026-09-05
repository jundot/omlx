# Handoff — Fase J: Otimização de Memória do Prefill (MoE Expert Streaming)

> **Data:** 2026-08-30
> **Sessão:** Continuação da execução do Plano Fase J (otimizações de streaming de experts do SSD)
> **Estado:** Código implementado até C13; nova frente de otimização de memória do prefill identificada e planejada

---

## 1. Contexto da Sessão

### 1.1 O que foi feito nesta sessão

Esta sessão continuou a execução do **Plano Fase J** — um plano de 14 commits (C0→C13) para otimizar a latência de streaming de experts MoE do SSD no servidor de inferência `omlx` (baseado em MLX). O trabalho acumulado até o início desta sessão:

- **C0–C13:** Todos os commits foram implementados, testados e commitados. A documentação foi atualizada (`docs/expert-streaming.md`).
- **Benchmarks:** Executados com sucesso em Qwen (modelo canônico). GLM está bloqueado por hardware (prefill requer ~46 GiB, máquina tem 48 GiB total).
- **Diagnóstico de memória:** Confirmado que o N-gram está em `mmap` (streaming do SSD) e **não é o culpado** pelo alto uso de memória. O problema é o **Metal/MLX buffers** durante o prefill (~34,5 GiB de `IOAccelerator`).

### 1.2 Evidências coletadas

| Métrica | Valor |
|---------|-------|
| Physical footprint (pico) | 35,7 GiB |
| IOAccelerator (graphics) | 34,5 GiB |
| Mapped file virtual | 143,7 GiB |
| Mapped file resident | 60,5 MiB |
| MLX cache durante prefill | 30,36 GiB |
| MLX cache durante decode | 0,06 GiB |
| Swap usado (global) | 5,80 GiB |

**Conclusão do diagnóstico:** O pico de memória ocorre durante o **prefill**, não durante o decode. O N-gram (tabela de embeddings) está corretamente em streaming via `mmap` e não está carregado integralmente na RAM.

### 1.3 Arquivos-chave modificados

- `omlx/patches/expert_streaming/streaming_switch.py` — C2, C3, C4, C6, C10, C12
- `omlx/patches/expert_streaming/shard_bank.py` — C1, C2 (read_expert_into)
- `omlx/patches/expert_streaming/warmer.py` — C5, C7, C8
- `omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py` — C9
- `omlx/patches/expert_streaming/qwen35_stream_eval.py` — C9
- `omlx/scheduler.py` — C11
- `bench/bench_expert_streaming.py` — C0, single-request protocol, token-ID gate
- `omlx/engine/vlm.py` — Propagação de token IDs
- `docs/expert-streaming.md` — C13, documentação de resultados
- `tests/test_expert_streaming.py` — Testes para C1, C2, C3, C12
- `tests/test_vlm_engine.py` — Testes de propagação de tokens

### 1.4 Commits criados

```
70742ea7 bench: record optimized streaming measurements
7b787641 perf: add shared layer I/O kill switch
9cfb40a7 perf: reuse routing plan indices for bias gathers
e43dcfff perf: complete async seed and shared layer I/O
5ef31dd6 perf: execute Fase J streaming optimizations
26e124c7 bench: add single-request memory-aware protocol
7b10a94c docs: correct benchmark TTFT units
```

### 1.5 Resultados de benchmark (Qwen, estado otimizado combinado)

| Braço | TTFT | tok/s | hit_rate |
|-------|------|-------|----------|
| M0 A0 (baseline) | 198,83 s | 0,3004 | 0 |
| Otimizado A0 | 84,27 s | 0,4514 | 0 |
| M0 B3a (baseline) | 106,42 s | 0,3451 | 0,0877 |
| Otimizado B3 | 85,20 s | 0,4141 | 0,0323 |

**Nota:** O `hit_rate` do B3 mudou de 0,0877 para 0,0323 — precisa ser investigado antes de claims definitivos de performance.

### 1.6 Novo protocolo de benchmark

Foi adicionado `--single-request` ao `bench/bench_expert_streaming.py`:
- Usa uma única requisição `stream_chat` (não faz segundo prefill)
- Mede TTFT até o primeiro token
- Mede decode após o primeiro token
- Propaga token IDs reais (`bit_exact_kind=tokens`)

---

## 2. Problema Identificado: Pico de Memória no Prefill

### 2.1 Onde a memória fica retida (evidência no código)

**A. `_LayerLoadContext` mantém todos os bancos NumPy vivos simultaneamente**
- `streaming_switch.py:472-512` — `ensure()` preenche `bundles` para **todos** os lineares (gate + up + down) de uma vez
- Cada linha é uma *view* sobre o banco NumPy inteiro, mantendo-o vivo
- Bancos de gate + up + down ficam residentes do início da camada até o retorno: **~3× U × per_expert_bytes** de RAM por camada

**B. Grafo lazy do chunk inteiro (dominante em qwen4_exp)**
- `Qwen4ExpDecoderLayer.__call__` (`qwen4_exp/language.py:1496-1544`) **não tem** verificação de `_stream_eval`
- `qwen35_stream_eval.py` só envolve `Qwen3_5MoeDecoderLayer` — **inerte em qwen4_exp**
- O scheduler só faz `mx.eval` no final do chunk
- Cópias promovidas por linha e outputs de stack de **todas** as camadas MoE permanecem vivos como inputs não avaliados do grafo
- Pico F1 ~26 GB documentado: 48 layers × ~215 uniq × ~2.5 MB

**C. `mx.stack` gera double-buffer**
- Linhas (U cópias MLX) + saída do stack coexistem durante o kernel → **transient 2× bank** por projeção

**D. Linhas NumPy no LRU fixam bancos inteiros**
- `streaming_switch.py:1126-1130` armazena views `raw`; evictar uma entrada não libera até a última linha do banco ser evictada
- Intencional (F2) e limitado pelo orçamento LRU — **não mexer**

### 2.2 Por que TurboQuant 8-bit não resolve

- O patch foi ativado corretamente (`TurboQuant KV cache enabled: 8.0 bits`)
- O teste terminou com `rc=143` sem JSON válido
- KV estimado: 27,28 KB/token × 1.898 tokens ≈ **52 MB** (insignificante vs 34,5 GiB de Metal)
- O problema dominante é o **transient do prefill** (~62,49 GiB predito), não o KV

---

## 3. Plano de Otimização Proposto

### 3.1 Visão geral

Transformar o prefill em um **pipeline de memória limitada**, mantendo o plano de roteamento compartilhado mas eliminando a retenção simultânea de bancos de múltiplas projeções.

```
┌─────────────────────────────────────────────────────────────┐
│  Prompt completo → não materializar tudo de uma vez         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Chunk adaptativo: 512 → 256 → 128 tokens                   │
│  (reduz tamanho do temporário Metal)                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────┐    ┌─────────┐    ┌─────────┐
│  Gate   │ →  │   Up    │ →  │  Down   │
│ carrega │    │ libera  │    │ carrega │
│ e avalia│    │ o banco │    │ só então│
└─────────┘    └─────────┘    └─────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Limite de working set: Metal + ativações + experts ≤       │
│  orçamento                                                  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Etapas de implementação

#### Etapa A: Tilear o demand set em `StreamingQuantizedSwitchLinear.__call__`

**Objetivo:** Dividir `plan.uniq_list` em tiles de tamanho fixo para limitar o banco de experts por projeção.

**Implementação:**
- Dividir `plan.uniq_list` em tiles (padrão 32-64 experts)
- Limitar bytes do tile ≤ `_BANK_MAX_BYTES` (256 MiB)
- Por tile:
  - Fatia contígua de `x`/`remapped` (offset remapped pelo início do tile)
  - Carregamento via ladder existente
  - Montagem com **promoção única** quando all-miss: `mx.array(bank).view(...).reshape(U, *per_shape)` em vez de U cópias + `mx.stack`
  - `gather_qmm` na fatia
  - **`mx.eval(tile_out)`** (ou `async_eval` por flag) e drop refs
- Concatenar saídas em ordem ascendente
- Bias por tile via `self._bias[uniq_mx[tile]]`

**Ganho:** Elimina o double-buffer do `mx.stack` e reduz o footprint de montagem pela metade.

**Código atual relevante:** `streaming_switch.py:835-873` (`_load_expert_bank_np`), `streaming_switch.py:1140-1158` (promoção e stack)

#### Etapa B: Remover retenção union de `_LayerLoadContext`

**Objetivo:** Manter sobreposição de I/O sem reter bancos de todas as projeções simultaneamente.

**Implementação:**
- Substituir `_LayerLoadContext.ensure()` por prefetch assíncrono NumPy byte-capped (ex: 2 bancos)
- Disparar leitura de `down` enquanto `up`/`gate` promove/computa
- Linhas NumPy são promovidas e descartadas **por projeção**
- Kill-switch env espelha `_LAYER_BARRIER_ENV`
- Manter `prefill_bypass` exatamente onde está (nunca dentro de `_load_expert_bundle`)

**Ganho:** Elimina a retenção de ~2 bancos de projeção em RSS por camada.

**Código atual relevante:** `streaming_switch.py:472-512` (`_LayerLoadContext`), `streaming_switch.py:1269-1277` (anexação do ctx)

#### Etapa C: Definir pontos de boundary de eval

**Objetivo:** Forçar avaliação de resultados intermediários para liberar buffers do grafo lazy.

**Implementação:**
- Começar com **2 syncs por GLU** (após up+gate, após down) em vez de evals bloqueantes por tile
- Ou usar `mx.async_eval` + drop refs, que libera buffers conforme a GPU drena
- Qualquer `mx.clear_cache` deve passar por `_sync_and_clear_cache` (metal_sync.py:36-63)
- Aplicar o mesmo a `StreamingSwitchLinear` ou documentar escopo quantizado-only
- Opcionalmente patchar `Qwen4ExpDecoderLayer.__call__` para adicionar boundary `_stream_eval`

**Ganho:** Pool MLX estagna em ~1 camada de working set em vez de crescer 48×.

**Código atual relevante:** `qwen4_exp/language.py:1496-1544` (sem `_stream_eval`), `qwen35_stream_eval.py:88-96` (só Qwen3_5)

#### Etapa D: Threshold-gate limpezas de pool

**Objetivo:** Limpar o pool MLX apenas quando necessário, não incondicionalmente.

**Implementação:**
- Dentro do linear: `get_cache_memory() >= OMLX_EXPERT_STREAMING_CACHE_THRESH` (default 2 GiB) antes de `mx.clear_cache()`
- Limpezas de limite de chunk do scheduler tornam-se threshold-gated
- **Manter uma limpeza incondicional no final do prefill** para decode começar com pool limpo
- Todas as limpezas continuam em `_sync_and_clear_cache`
- Preservar fallback `get_cache_memory is None → clear`

**Ganho:** Evita overhead de limpezas desnecessárias mantendo controle de memória.

**Código atual relevante:** `scheduler.py:3544, 3750, 3797, 5242, 5665` (limpezas incondicionais atuais)

#### Etapa E: Atualizar contabilidade do guard (OFF por padrão — ver nota)

**Objetivo:** Fazer o scheduler refletir o novo pico de memória limitado por tile.

**Implementação:**
- Adicionar flag `boundary_active` a `streaming_guard_info` (setada em `__init__.py:642-646`)
- Fazer `_streaming_bank_bytes` cobrar `min(2, projections) × tile_bytes + ativação de uma camada` quando ativo
- Deixar o caminho medido por EWMA intocado

**Ganho:** O throttle/admission para de cobrar o pior caso de 26 GB e permite que o ganho apareça no comportamento.

> **Decisão de execução (2026-08-30): Etapa E fica OFF por padrão.** A medição
> prévia com E ativo divergiu do baseline em *token-ID bit-exactness* (primeira
> divergência no token 3 de 48), violando o critério de aceitação #1 (bit-exactness
> em primeiro lugar). Como a contabilidade do guard estruturalmente **não** deveria
> alterar saídas, se E for necessário ligar no futuro ele precisa ser re-verificado
> como *output-neutral*; se alterar os numerics, é um bug latente a corrigir à
> parte — não se liga E para contorná-lo. O ganho de comportamento (admission
> deixar de cobrar o pior caso de 26 GiB) fica condicionado a essa re-verificação.

**Código atual relevante:** `scheduler.py:3863-3882` (`_streaming_bank_bytes`), `scheduler.py:3952-3955` (`max()` no per_token)

---

## 4. Ganhos Adicionais Identificados

### G1: Montagem de banco com promoção única
No hot path de prefill (prefill_bypass on → all-miss), as linhas do tile vêm de um único banco NumPy fresco, então `w_bank` pode ser construído com **uma** chamada `mx.array(bank).view(...).reshape(U, *per_shape)` em vez de U cópias `mx.array` + `mx.stack`. Isso **divide pela metade** o footprint de montagem e remove U−1 alocações Metal por projeção.

### G2: Boundary de 6 linhas para qwen4_exp
Adicionar o boundary `_stream_eval` em `Qwen4ExpDecoderLayer.__call__` (antes do return em `language.py:1544`) obtém a maior parte do ganho cross-layer sem tocar nos lineares. Fazer ambos (boundary na camada + nos lineares) é redundância benéfica.

### G3: Atualizar termo estático de banco no guard (OFF por padrão — ver nota na Etapa E)
Sem isso, o scheduler continua rejeitando e sub-dimensionando como antes, e o ganho nunca aparece no comportamento.

### G4: Paridade no variant BF16
`StreamingSwitchLinear.__call__` (linhas 636-687) nunca usa `plan.ctx` e retém U arrays mx por projeção da mesma forma — aplicar o mesmo tratamento ou documentar escopo quantizado-only.

---

## 5. Riscos e Mitigações

| Risco | Descrição | Mitigação |
|-------|-----------|-----------|
| **R1** | Bit-exactness de `gather_qmm` em tiles | Gate de token-ids idênticos; concat em ordem ascendente é provadamente correta |
| **R2** | Stub de tracing OQ (`mx.eval` é noop) | Eval dentro do linear não é load-bearing para valores, apenas para lifetime |
| **R3** | Ladder de carregamento por tile | Manter cache hit → prefetch → bank read → fallback legacy; kill-switch |
| **R4** | Hooks e contratos alterados | `_warm_pins.on_layer_start`/`on_layer_plan`, `_trace_row`, `weighted_sum` — todos preservados |
| **R5** | Double boundary GLM | GLM já avalia por camada; evals extras são baratos; medir TTFT nos braços GLM |
| **R6** | Ruído no ledger de reclaim | Verificar que predictor não cobra em dobro após chunks com pool retido |
| **R7** | Gap DFlash (pré-existente) | DFlashEngine não tem termo de banco; verificar se serve modelo streaming |

---

## 6. Critérios de Aceitação

A mudança só deve ser considerada segura se:

1. ✅ Gate por token IDs continuar passando (`bit_exact_kind=tokens`)
2. ✅ Texto e 48 token IDs forem iguais ao baseline
3. ✅ Pico de `IOAccelerator` cair substancialmente (esperado: ~29 GiB → low single-digit GiB)
4. ✅ Swap delta por execução diminuir
5. ✅ Não houver crescimento de `load_ms` desproporcional
6. ✅ Throughput não cair mais do que o custo aceitável dos tiles
7. ✅ Caminho decode continuar usando tiles maiores/LRU, sem contenção agressiva do prefill
8. ✅ Hit-rate inalterado a budget fixo
9. ✅ RSS não aumentar mais de 5%

---

## 7. Testes Necessários

- `tests/test_expert_streaming.py` existentes devem passar
- Novos testes:
  - Tile concat vs single-shot `mx.array_equal`
  - Single-promotion all-miss vs stack por-linha
  - Fallback ladder por tile
  - Contagens LRU após puts tileados
- Benchmark de bit-exactness por token-ids
- Braços Qwen e GLM (se possível)

---

## 8. Ordem de Execução Recomendada

```
1. Instrumentar por camada e projeção (active, cache, footprint, IOAccelerator)
2. A/B do C6 (ligado vs desligado) para confirmar retenção do ctx
3. Implementar Etapa B (remover retenção union do ctx)
4. Implementar Etapa A (tiles de experts)
5. Implementar Etapa C (boundary de eval)
6. Implementar Etapa D (threshold-gate limpezas)
7. Implementar Etapa E (atualizar guard) — **OFF por padrão** (ver nota na Etapa E: conflita com bit-exactness)
8. Rodar testes e benchmarks de validação
9. Documentar resultados em docs/expert-streaming.md
```

---

## 9. Arquivos para Modificar

| Arquivo | Mudança |
|---------|---------|
| `omlx/patches/expert_streaming/streaming_switch.py` | Tiles, promoção única, boundary de eval, ctx sem retenção union |
| `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py` | Adicionar boundary `_stream_eval` |
| `omlx/scheduler.py` | Threshold-gate limpezas, atualizar `_streaming_bank_bytes` |
| `omlx/patches/expert_streaming/__init__.py` | Adicionar `boundary_active` flag |
| `tests/test_expert_streaming.py` | Novos testes de tile e promoção única |
| `bench/bench_expert_streaming.py` | Instrumentação de memória por fase |
| `docs/expert-streaming.md` | Documentar resultados |

---

## 10. Referências

- **Plano original:** `PLAN-fase-J-streaming-otimizacoes.md`
- **Documentação de resultados:** `docs/expert-streaming.md`
- **Log de memória:** `.workbuddy-ai/memory/2026-08-30.md`
- **Benchmarks:** `bench/results/faseJ/`
- **Código principal:** `omlx/patches/expert_streaming/streaming_switch.py`

---

## 11. Notas Finais

- O N-gram **já está em streaming via mmap** e não é o problema
- TurboQuant 8-bit **não resolve** o problema dominante de memória
- O problema é o **working set do prefill no Metal** (bancos de experts + grafo lazy)
- A solução é **limitar o working set por projeção/tile**, não reduzir o N-gram ou o KV
- Manter o protocolo `--single-request` para medir sem segundo prefill escondido
- GLM permanece **não mensurável** nesta máquina (prefill requer ~46 GiB, máquina tem 48 GiB total)

---

## 12. Resultados de Verificação (execução Fase J)

Estado ao fim da sessão 2026-08-31 (worktree `fork-feature-expert-streaming-61637226`, HEAD `6b020181` → `0a4d3c72`).

### 12.1 As 44 falhas da suíte completa são AMBIENTAIS (não o diff)
Isolado via `/tmp/iso_test.sh` (varia árvore vs. estilo de invocação em `test_paged_ssd_cache.py`):
- Árvore COM-mudanças, invocação limpa → **170 passed**.
- Árvore LIMPA, invocação estilo-com-mudanças → **27 failed, 143 passed**, todas com `PermissionError: EEXIST: file already exists, mkdir '.../ssd_cache/0'` (o broker de `mkdir` do sandbox não é idempotente para `makedirs(exist_ok=True)`).
- O mesmo gatilho dispara em código não modificado → **0% atribuível ao diff de `read_expert_into`/`RUN_QD`**. Suíte alvo `tests/test_expert_streaming.py` roda verde sob qualquer modo.

### 12.2 RUN_QD — committed em `0a4d3c72`
`perf: parallelize per-run preadv in read_expert_into (RUN_QD=16)`. 5 arquivos, +533.
- Pool aninhado `_RUN_IO_POOL` (separado de `_EXPERT_IO_POOL`) evita deadlock; plan-then-execute com scatter-after-barrier → bit-exact por construção.
- Varredura QD: 1→2.184 tok/s; 8→3.203/3.059; **16→3.324/3.419 (pico + mais estável, stdev 0.046 vs 0.390 em QD=8); 32→3.138 (regride)**. Default 16 = +55% vs profundidade 1, acima do baseline 3.06. `phys` plano ~10.9 GiB; 15 runs token-ID bit-exact.
- Corrigido gate vazio de teste (`_write_shard` escrevia payload zero → qualquer comparação de bytes passava): `_write_shard_filled` + `assert out1.any()`.

### 12.3 C6 (Etapa B / Layer Barrier) A/B — bit-exact-preserving
Knob: `OMLX_EXPERT_STREAMING_LAYER_BARRIER` (default ON, `streaming_switch.py:55,1720`). Quando ON, a GLU quantizada monta `_LayerLoadContext` compartilhado (retenção union das projeções); quando OFF, cada projeção carrega independente.
- **Barrier ON (default): 111 passed.**
- **Barrier OFF (legacy, +`CTX_ROLLING=0`): 108 passed, 3 failed** — e os 3 failures são EXATAMENTE os testes que ASSERTEM comportamento de barrier-ON: `test_glu_forward_emits_memtrace_events` (espera eventos `ctx.ensure.`), `test_ctx_bank_promote_is_bit_identical_on_default_path` (`assert ss._LAYER_BARRIER_ENV`), `test_ctx_bank_promote_declined_on_partial_demand` (espera gravação do caminho de promoção ctx). Nenhum é teste numérico.
- Todos os gates numéricos/bit-exact passam em AMBOS os modos (`test_streaming_quantized_glu_matches_reference`, `test_quantized_call_identical_with_and_without_bank_promote` [A1, L3089], `test_ctx_bank_promote_is_bit_identical_on_default_path` [A1b, L3252], `test_backing_read_expert_into_matches_load_expert_slice`). **Conclusão: a barrier é behavior-preserving / bit-exact-safe.**
- Mecânica A1 vs A1b: **A1 (`_BANK_PROMOTE_ENV`, L1444) é código morto na config default** — só dispara quando `len(missing) == len(plan.uniq_list)` (batch todo miss), quase nunca verdadeiro porque a ctx pré-particiona. **A1b (`_BANK_PROMOTE_CTX_ENV`, L670/695/1561) é o caminho vivo, e só dispara PORQUE a barrier monta `plan.ctx`.** Logo a barrier (C6/Etapa B) é o *habilitador* da promoção, não a A1.
- Ressalva: esta A/B prova equivalência de CORREÇÃO, NÃO que a barrier entrega o ganho de retenção de memória alegado (G2/G3). Um bench de RSS durante prefill seria necessário para confirmar; não medido nesta sessão.

### 12.4 Etapa E
OFF por padrão (decisão já registrada em §3 Etapa E + nota de execução). Medição prévia com E ativo divergiu do baseline em token-ID bit-exactness (token 3 de 48) → conflito com critério #1.

### 12.5 A2 (tiling / Etapa A) — IMPLEMENTADO (off by default)
- **Implementado e validado.** Flag `OMLX_EXPERT_STREAMING_TILE` (ler em runtime via `_expert_tile_size()`, default `0` = desligado). Quando `N > tile < len(uniq_list)` e `x.ndim >= 3`, `StreamingQuantizedSwitchLinear.__call__` desvia para `_call_tiled` + `_build_tile_bank` (isolados, SEM mexer no caminho non-tiled).
- **Bit-exact por construção — verificado por teste, não só por R1.** Produção sempre roteia via GLU, que expande `x` para 3D `(P, *, in)` e ordena posições por expert; nesse layout `gather_qmm` faz roteamento **por posição** (`out[b] = x[b] @ W[remapped[b]]`). Logo cada posição depende só de `x[b]` e da linha de peso selecionada por `remapped[b]` → tilear por expert + reabmontar em ordem de expert crescente é idêntico bit-a-bit. Confirmado empiricamente (probe: `mx.array_equal(full, subset_reassembled) == True`) e pelos 2 novos testes `test_quantized_call_identical_with_and_without_tiling` (linear, `x=(12,1,128)`, `TILE=3` vs `0`) e `test_glu_tiling_bit_identical_prefill` (GLU, `x=(1,200,128)`, `TILE=4` vs `0`) — ambos passam, com e sem a barreira da Etapa B.
- **Gotcha de implementação resolvido:** `gather_qmm` com `x` 2D degenera para layout all-pairs `(P,P,out)` (não é o caminho de produção); por isso `_call_tiled` só engaja para `x.ndim >= 3` e o teste linear usa `x` 3D. O caminho 2D continua no non-tiled (correto, sem benefício de tiling). Também contornados 3 limites do MLX deste build: sem boolean-indexing (`x[m]`), sem `argwhere`/`nonzero` (usado `mx.argsort(-m_int)[:count]`), e sem `mx.array.at[...].set` (reabmontagem via `concatenate` + `argsort(idx_pos_all)`, que é permutação das posições).
- **A2 NÃO endereça o pico de 34.5 GiB (Metal/IOAccelerator):** esse pico é o pool do allocator MLX (grafo lazy), conforme a própria nota de ganho da Etapa C. A2 limita o working set *host* por projeção (bank de `tile` experts por vez em vez de todos os demandados; elimina double-buffer do `mx.stack`) — ganho secundário. O pico real é Etapa C/D. Commit: `feat(expert-streaming): demand-set tiling of quantized gather_qmm (Etapa A, off by default)`.
- Suíte `tests/test_expert_streaming.py`: **113 passed** (111 prévios + 2 de A2), sem regressão.

### 13. Etapa C/D — JÁ IMPLEMENTADA E VERIFICADA (corrige framing do §12.5)
O §12.5 diz que "o pico real é Etapa C/D" como se C/D fosse trabalho futuro.
**Não é:** inspecionando o código atual (commit anterior a este), Etapa C e D
já estão implementadas, wireadas e **ON por padrão**:

- **Etapa C (boundary de eval por camada):** `omlx/patches/expert_streaming/qwen35_stream_eval.py`
  envolve `Qwen3_5MoeDecoderLayer` **e** `Qwen4ExpDecoderLayer` com
  `mx.eval(out)` + clear sincronizado quando `x.shape[1] > 1` (prefill). Flag
  `OMLX_EXPERT_STREAMING_PER_LAYER_EVAL` default **1** (ON). Aplicada em
  `omlx/patches/expert_streaming/__init__.py` (`layer._stream_eval = True` L550;
  `qse.apply_qwen35_moe_stream_eval()` L843; `boundary_active` L855). Testada por
  `test_qwen_stream_eval_*` (8 testes passando).
- **Etapa D (limpeza de pool threshold-gated):** helpers em `omlx/utils/metal_sync.py`
  (`should_clear_cache`, `cache_clear_threshold_bytes`, `_CACHE_CLEAR_THRESH_ENV`
  default 2 GiB, `_sync_and_clear_cache`). Já usado no wrapper per-layer
  (`qwen35_stream_eval.py:114`: só limpa se `get_cache_memory() >= threshold`) e
  em um site do scheduler (`scheduler.py:2322`). Os demais `~20` sites de
  `_sync_and_clear_cache` ainda limpam incondicionalmente — o que, na prática,
  mantém o pico BAIXO (é por isso que medimos 8.33 GiB abaixo). Gateá-los também
  é refinamento de cadência, não necessário para o pico.

**Evidência de que o pico de 34.5 GiB FOI RESOLVIDO** (bench real,
`bench/results/faseJ/real_faseJ_2k.json`, qwen 2k, `budget_gib=0` = pior caso
all-miss, C/D ON):
- `metal_peak_prefill_gib = 8.33` (medido via `mx.get_peak_memory()` — o mesmo
  metric de alta do critério de aceitação #3, "IOAccelerator peak").
- `phys_lifetime_max_gib = 11.28`; `mlx_active_gib_max (prefill) = 7.92`.
- 8.33 GiB = "low single-digit GiB", que é exatamente o alvo do critério #3
  ("~29 GiB → low single-digit GiB"). O 34.5 GiB era a medição PRÉ-C/D.

#### 13.1 Confirmação em 8k (worst-case all-miss) — PASSOU
Bench real `bench/results/faseJ/real_faseJ_8k.json`, qwen 8k, `budget_gib=0`
(all-miss, re-leitura do banco de experts = pior caso), `--single-request`,
C/D ON (`OMLX_EXPERT_STREAMING_PER_LAYER_EVAL=1`, `OMLX_EXPERT_STREAMING_CACHE_THRESH=2`),
rodado em 2026-08-31. Comparação direta com o 2k (mesmas condições):

| metric | 2k | 8k | veredito |
|---|---|---|---|
| `metal_peak_prefill_gib` (crit #3) | 8.33 | **6.95** | ✅ limitado, ainda menor em 8k |
| `phys_lifetime_max_gib` | 11.28 | 11.64 | ✅ plano, ≪ teto de 28 GiB |
| TTFT | 35.27 s | 253.84 s | ⚠️ degradado (latência, não memória) |
| cache | — | 0 hits / 1.108.182 misses | ✅ caminho all-miss exercido |

- O `metal_peak_prefill_gib` em 8k (6.95 GiB) ficou **abaixo** do de 2k (8.33) —
  o boundary de eval por camada + limpeza de pool threshold-gated (2 GiB)
  mantém o pool retido em ~1 working set de camada **independentemente do
  comprimento do prompt**. O pico não escala com o prompt.
- `phys_lifetime_max_gib` plano (11.64 vs 11.28) — bem abaixo do teto.
- A única regressão é **TTFT** (253.84 s, ~7.2× o de 2k). Causas observadas no log:
  (a) **amplificação de leitura** do all-miss (1.99 GB/s de disco no prefill,
  `mlx_cache_gib_max=2.13` disparando os clears); e (b) **throttle do scheduler
  super-conservador**: o log mostra predição de 67.46 GB mas o pico real foi
  11.64 GiB, então o scheduler encolheu o chunk de prefill `2048 → 512`
  (`adaptive_prefill_throttle`, `predicted=67.46GB` vs `target=23.94GB`). Isso
  estrangulou a latência desnecessariamente — é um ajuste de *scheduler tuning*
  separado, **não** um estouro de memória. Os critérios de aceitação de pico
  foram atingidos.
- `get_cache_memory()` existe neste build de MLX (caso contrário o clear
  rodaria toda camada incondicionalmente — comportamento idêntico entre 2k e 8k
  porque é o mesmo binário, então a comparação é válida).

**Conclusão:** o item "pivotar para Etapa C/D para o pico de 34.5 GiB" do
resumo de still-open está FECHADO pela implementação existente **e confirmado
em 8k** (§13.1). O alvo de pico do critério #3 está atingido em 2k e 8k.
Resta, se desejado (nenhum é bloqueador do critério de pico):
(a) ~FEITO — robustez em 8k confirmada; (b) gateá-los os ~20 sites restantes
do scheduler com `should_clear_cache` (refinamento de cadência, com risco de
subir o pico se feito mal — NÃO fazer sem medir); (c) ~FEITO — `adaptive_prefill_throttle`
investigado e corrigido (§14): a estimativa estática SDPA+KV era ~40× o medido
para o MoE 4-bit e dominava o MAX permanentemente; agora é limitada a
`_PREFILL_STATIC_MAX_OVER_MEASURED` (3×) do medido quando há amostras. Nada de
código de pico pendente.

---

### 14. `adaptive_prefill_throttle` — correção da superestimativa de memória (TTFT em 8k)
Item (c) da conclusão de §13.1, implementado e testado.

**Sintoma (§13.1):** no bench 8k all-miss, o scheduler previu `predicted=67.46 GB`
para o chunk de prefill mas o pico real foi `11.64 GiB` (`metal_peak 6.95`).
Como o alvo de throttle era `23.94 GB`, o scheduler encolheu o chunk
`2048 → 512` e pausou para LRU eviction — inflacionando o TTFT de ~35 s (2k)
para ~254 s (8k), ~7×. A causa NÃO é o pico de memória (critério #3 ok), é o
*scheduler tuning*.

**Raiz:** `Scheduler._predicted_chunk_transient` (em `omlx/scheduler.py`) fazia
`per_token = MAX(medido_last, EWMA, estática_SDPA+KV)` **incondicionalmente**.
Para Qwen3.8-Flash-Next-oQ4e (MoE 4-bit) a estimativa estática genérica
(~34.5 MB/token) era ~40× o medido (~0.9 MB/token). Por estar no `MAX`, o valor
errado **dominava permanentemente** — o sinal medido (`mx.get_peak_memory`,
ground truth) nunca sobrepunha. Resultado: todo chunk era estrangulado ao
tamanho de piso (512). Isso também contradizia o próprio docstring de
`_adaptive_chunk_size` ("Measured: once the per-scheduler EWMA has samples,
use its `bytes_per_token` ... First chunk: fall back to the static estimate").

**Correção:** a estimativa estática só entra no `MAX` quando NÃO há amostras
medidas (primeiro chunk de um modelo recém-carregado — fallback conservador).
Uma vez que o `PrefillTransientTracker` tem amostras, a estática pode apenas
*elevar* a predição por-token até `_PREFILL_STATIC_MAX_OVER_MEASURED` (default
3.0, env `OMLX_PREFILL_STATIC_MAX_OVER_MEASURED`) vezes a taxa medida:
- estática modestamente maior que o medido (crescimento de kv_len que o EWMA
  atrasa) ainda vence o MAX → backstop de segurança preservado;
- estática absurdamente maior (fórmula densa superestimando MoE 4-bit em ~40×)
  é truncada em 3× o medido → não domina mais.

A predição resultante nunca fica abaixo de `medido × 1.3` (safety factor),
então NÃO há risco de subestimar o pico (o ground truth medido sempre vale).
As chamadas ao `memory_monitor` e `_streaming_bank_bytes` foram envolvidas em
`try/except` — uma falha do monitor não quebra o dimensionamento do chunk; o
sinal medido (ou o fallback de watermark do caller) still aplica.

**Resultado esperado no bench 8k:** o primeiro chunk ainda usa o fallback
estático (1 pausa de eviction + 1 chunk de 512, seguro), mas após a 1ª medição
(`~0.9 MB/token`) os chunks seguintes rodam em `2048` (previsto 2048-chunk ≈
2–7 GiB ≪ alvo 23.94 GB). O prefill 8k passa de ~16 chunks de 512 para ~5
chunks → TTFT deve cair de ~254 s para a faixa de ~80–120 s (o residual vem da
amplificação de leitura all-miss do §13.1, não do throttle).

**CONFIRMADO por re-run (bench/results/faseJ/real_faseJ_8k_postthrottle.json,
mesmas condições do §13.1):**

| metric | pré-fix | pós-fix | delta |
|---|---|---|---|
| `ttft_s` | 253.84 | **119.61** | **−53% (2.12× mais rápido)** |
| `metal_peak_prefill_gib` | 6.95 | 8.40 | single-digit, critério #3 ok |
| `phys_lifetime_max_gib` | 11.64 | 12.14 | plano, ≪ teto 28 GiB |

O log pós-fix mostra o throttle ainda disparando uma vez no 1º chunk
(`2048 -> 512`, `per_token=34539.1KB` — fallback estático conservador, seguro),
mas após a 1ª medição (`~0.9 MB/token`) os chunks seguintes rodam em 2048
(sinal medido domina). O `metal_peak` subiu de 6.95→8.40 GiB (chunks maiores
carregam mais tokens por eval) — ainda single-digit e dentro do alvo. O ganho
de latência vem de eliminar os ~15 chunks de piso; o residual de ~120 s é a
amplificação de leitura all-miss (§13.1), não o throttle.

**Arquivos:** `omlx/scheduler.py` (`_predicted_chunk_transient`,
`_PREFILL_STATIC_MAX_OVER_MEASURED`), `tests/test_scheduler_chunked_prefill.py`
(3 testes de regressão: medido-vence-estática, fallback-estático-sem-amostras,
falha-do-monitor-é-segura), `tests/test_prefill_oom_graceful.py` (`_throttle_ctx`
ganha o novo atributo). Suíte: 168 passed.

---

### 15. Etapa D — cadência de `_sync_and_clear_cache` no scheduler: VALIDADA e REVERTIDA

Item (b) da conclusão de §13.1, investigado e **rejeitado** com medida.

**Tentativa:** gatear os ~11 sites restantes do scheduler que ainda chamavam
`_sync_and_clear_cache` **incondicionalmente** (sem `should_clear_cache()`),
usando o predicado de cadência `should_clear_cache()` (limpa só quando
`get_cache_memory() >= 2 GiB`, ou sempre se a medição for indisponível). Os
sites pré-gateados (L2324, e L12026/L12145 em `step`) ficaram de fora. Os 11
alvos:
- 2 *lambdas*: L8930 (`_try_specprefill_scoring`), L10365 (scoring de spec-prefill).
- 9 chamadas diretas: L9528 (teardown/error-recovery), L10398 (spec-prefill
  abort cleanup), L10478 (`_PrefillAbortedError`), L10505/L10529 (prefill
  evict/reject), L10545 (prefill *done*), L10583/L10588/L10610
  (capacity-reject / memory-exception / runtime-error), L11557
  (generation-overflow recovery).
- Deliberadamente **DEIXADOS incondicionais** (mantêm o pico BAIXO, aviso de
  §13): L3658/3768/3828 (`_do_external_prefill`), L5071
  (`_reclaim_prefill_headroom`), L5538 (`_step_prefill_chunk`), L5605
  (`_finalize_chunked_prefill_cache_for_insert`), L5746/5765/5778/5806
  (`_advance_chunked_prefills`).

**Validação (o passo "medir" que §13 exigia — "NÃO fazer sem medir"):** rodou a
suíte de regressão de prefill. **4 testes de `TestPrefillCleanupUsesEngineStream`
FALHARAM**:
`test_first_chunk_capacity_rejection_drains_engine_stream`,
`test_first_chunk_runtime_error_drains_engine_stream`,
`test_non_chunked_capacity_rejection_drains_engine_stream`,
`test_non_chunked_runtime_error_drains_engine_stream`.

**Causa-raiz:** o gate envolve o clear em `if should_clear_cache():`. No
ambiente de teste `mx.get_cache_memory()` retorna 0 (< limiar default de 2 GiB)
→ `should_clear_cache()` retorna `False` → o **dreno obrigatório do engine
stream do Metal** em erro/abort/evict/reject de prefill é PULADO. O teste
afirma que o engine stream FOI drenado → falha a asserção
(`assert streams, "prefill cleanup did not clear the Metal buffer cache"`).

**Não é artefato de test-env:** em produção `should_clear_cache()` *também*
retorna `False` sempre que o pool estiver abaixo do limiar no momento do erro
de prefill — então um erro de prefill pequeno/já-drenado vazaria os transients
(subiria o pico E quebraria o invariante documentado de "dreno em erro").
O aviso de §13 está, portanto, correto e foi confirmado pela medição.

**Decisão:** **REVERTIDO** o gateamento no scheduler
(`git checkout -- omlx/scheduler.py`). Após o revert, a suíte completa de
prefill = **168 passed** (restaurada). O comportamento de manter o pico baixo
é preservado pelos clears incondicionais do prefill-hot-path + reclaim
(L5538, L5071 etc.) e pelo eval por camada threshold-gated da Etapa C/D.
Nada de código de pico pendente; nenhum commit de `scheduler.py` necessário
(voltou ao estado limpo de `25e9016e`).

**Recomendação para um refinamento de cadência futuro e seguro:** desacoplar
"dreno obrigatório de error-recovery" (deve ser **INCONDICIONAL** — é
corretude/invariante, não otimização de cadência) de "cadência steady-state"
(a única coisa que pode ser gateada). Um predicado correto seria
`should_clear_cache() OR is_error_recovery_path`. Ou, melhor: gatear só sites
comprovadamente FORA do prefill E FORA de qualquer error-recovery (ex.: cadência
de decode-step, já tratada em `step` em L12026/L12145). Revisitar só com um
harness de medição de pico *por site*.

---

**Fim do handoff.**
