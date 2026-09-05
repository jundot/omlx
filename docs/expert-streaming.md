# Expert Streaming (SSD)

Run Mixture-of-Experts (MoE) models that are larger than your Mac's RAM by keeping only the hot experts resident and streaming the rest from SSD.

Inspired by [slipstream](https://github.com/dwijenpatel/slipstream) (Swift/Metal expert LRU from SSD) and [colibri](https://github.com/JustVugg/colibri) (learned pin store + multi-tier memory), ported to oMLX's Python/MLX stack.

## When to use

- **You have a large MoE** (e.g. `glm_moe_dsa` / Qwen3.6-35B-A3B) and a **16–24 GB Mac**. Without streaming the model needs `resident_bytes` (checkpoint × 1.05) — often above the wired limit. With streaming it needs `dense_bytes × 1.05` (page-cache-only default), so a 35B MoE that needs ~21 GB resident fits in ~5–8 GB.
- You care about **fitting, not single-stream speed**. Streaming is slower than fully resident and disables continuous batching for that model (one request at a time). Use the default resident mode when the model already fits.

## How it works

- **Dense stays resident**: attention, shared experts, embeddings, LM head — always in unified memory.
- **Experts live on SSD**: the stacked `switch_mlp.(gate|up|down)_proj` banks (`(E, O, I)` tensors) stay memory-mapped from the original safetensors files (MADV_RANDOM, like the Qwen4 PLE `DiskBackedShardedEmbedding`). No duplicate copy is made.
- **Per-layer demand loads**: on a batch, the union of routed experts is resolved per layer; hits in the (optional) LRU run immediately, misses fault the expert slice with coalesced `preadv` reads on the process-wide `_EXPERT_IO_POOL` (16 workers, QD16 — QD32 measured slower; never per-call pools) plus a separate run-read pool; the mini-bank is assembled on the inference thread. Quantized scales/biases ride along. The default cache policy is **page-cache only** (no LRU) — see Fase B below; `expert_streaming_budget_gib > 0` re-enables a bounded LRU. Fused `gate_up` models budget 2 projections/expert (split: 3), reconciled to the majority layout at convert time. With an active cold tier the estimate reports tier-aware effective bytes (`expert_bytes_effective`, `uniform`/`hobbit`/`none`) in the admin payload. Eviction is LRU by default, `S3-FIFO` behind `OMLX_EXPERT_STREAMING_CACHE=s3fifo` (A/B in progress). Per-request health (`lru_hit_rate`, `prefetch_precision`, `stash_hit_rate`, `ctx_fallbacks`) is logged as one `expert_streaming req` line.
- **One budget, auto-forced**: set `Expert Streaming` on in the per-model settings (or leave the budget empty for the page-cache-only default). Forced when `resident × 1.10 > ceiling ≥ streaming × 1.10 + one-layer-bank` (10% margin + streaming transient, so an exactly-at-ceiling load is refused), oMLX auto-enables it and shows an amber "auto-enabled" hint — the same pattern as `qwen4_ple_ssd_offload`.

## Supported models

| model_type | example | MoE | per-expert (oQ4e) |
|---|---|---|---|
| `glm5_next` / `glm5_next_text` | GLM-5.3-Flash-oQ4e (190G) | 42 layers × 288 routed | ~13 MB weight + scales |
| `qwen4_exp` / `qwen4_exp_text` | Qwen3.8-Flash-Next-oQ4e (99G) | 48 layers × 512 routed + PLE | ~2.7 MB weight + scales |
| `glm_moe_dsa` | GLM-5.2 MoE DSA | — | ~1.7 MB (BF16) / quantized packed |
| `deepseek_v32` | DeepSeek-V3.2 family | — | quantized packed |
| `deepseek_v4` / `deepseek_v4_mtp` | DeepSeek-V4-Flash-0731-oQ4e-mtp (166G) | 43 layers + 3 MTP stages × 256 routed | ~12.6 MB (mxfp4 gs32) |
| `qwen3_moe` | Qwen3-MoE family (128 experts, top-8) | sparse-step pattern¹ | stacked `switch_mlp` |
| `qwen2_moe` | Qwen2-MoE family (60 experts, top-4) | every layer is MoE² | stacked `switch_mlp` |
| `deepseek_v3` | DeepSeek-V3 (256 routed, top-8) | dense-first layers skipped³ | stacked `switch_mlp` |
| `glm4_moe` | GLM-4-MoE (128 routed, top-8) | dense-first layers skipped³ | stacked `switch_mlp` |

The allowlist (`SUPPORTED_TYPES` in `residency.py`, the single source of truth) is a hard gate: an unlisted `switch_mlp` family reports `supported = False` (fail closed) instead of forcing streaming without the lazy load — that combination used to materialize multi-hundred-GB banks before conversion (OOM/SIGKILL).

¹ `qwen3_moe`: a layer is MoE iff `(layer_idx + 1) % decoder_sparse_step == 0` and it is not in `mlp_only_layers` (mirrors mlx_lm; defaults resolve to all-MoE). ² `qwen2_moe`: the mlx decoder is unconditionally all-MoE. ³ `deepseek_v3` / `glm4_moe`: resolved from `first_k_dense_replace` (generic fallback). All four carry `moe_intermediate_size`, which the estimate reads instead of the 1407 fallback. Runtime conversion of the widened four is covered by fake-checkpoint walk tests; bit-exact validation on real weights is the Etapa L follow-up.

Loading a glm5_next / qwen4_exp checkpoint with `expert_streaming_enabled` uses the lazy loader (`lazy=True`) and converts to streaming **before** `materialize_lazy_state` — the multi-hundred-GB MoE banks are dropped as lazy arrays instead of ever being materialized. GLM decoders additionally get `compile_ffn` disabled and a per-layer `mx.eval(out)` + `mx.clear_cache()` so the per-layer expert mini-banks (~3.4 GB at prefill) do not accumulate in the lazy graph / allocator and swap the machine. Text-engine loads (BatchedEngine) apply the same lazy + convert-before-materialize order for streaming-supported model types — this is what makes `deepseek_v4` viable on 16 GB Macs.

### 0.6.4 optimized paths (QSA / affine tile)

Streaming only replaces `switch_mlp` (MoE experts) — attention is never touched, so the 0.6.4 fast paths stay engaged under streaming: Qwen4's exact QSA prefill/decode (`qsa_fast.py`), resident-PLE/`hc_projection` opts and sparse-native kernel (`qwen4_qsa_sparse_gqa`), and GLM-5.3's affine prefill tile (incl. the strided-input `contiguous` fix). Two deliberate interactions: (1) the scheduler gates *wide* Qwen4 prefill chunks on the sparse native kernel being built — without it, chunks stay at 2048 and streaming still works, just with smaller prefill steps; (2) PLE speculative-rollback snapshots use the simplified single-site capture (no `complete` two-phase protocol) with `ValueError` validation, and the lazy gate is fail-closed on `estimate.supported` — a checkpoint with no detectable MoE banks loads non-lazy even if its `model_type` is listed. The MoE weighted-sum in the streaming GLU routes to `glm_moe_weighted_sum` (native ext, `mx.fast` fallback inside) with scatter-unsort as the last resort.

### Decode latency: o que foi medido (Qwen3.8-JANG_4M, budget 1 GiB, short, 96 toks)

Base 1.79–1.82 tok/s, hit 9.2%, ~1.6 GB/token relidos (91% miss), 93% dos runs de leitura de tamanho 1, read p95 3.8 ms (p95 do SSD: 270 µs), pool QD16 saturado, GPU a 38%. Tentativas e resultado (kill criteria: +10% tok/s):

| variante | tok/s | hit | evictions | veredito |
|---|---|---|---|---|
| retenção np no ctx-path | 1.62 (−9%) | 14.5% | 47.9k | reverte: page cache já serve o re-read; todo uso re-paga promote |
| + admission filter | 1.71 (−4%) | 15.8% | 29.0k | reverte |
| + budget 3 GiB | 1.52 (−15%) | 32.0% | 25.9k | reverte: mais RAM retida, mais lento |
| retenção pós-promote (mx) | 1.70 (−5%) | 15.9% | 59.3k | reverte: churn de buffers MLX come a economia |
| stash prefetch (STASH=1) | 1.38 (−23%) | 5.3% | 63.1k | reverte: +52 GB especulativos roubam banda |
| sem seed (SEED=0) | 1.78 (≈) | 0.0% | 0 | seeder é a única fonte de residency |
| QD8 / QD24 (vs 16) | 1.85 / 1.78 | 9.2% | 0 | dentro do ruído; default 16 mantido |
| single-promotion parcial (1 promote do bank + concat + remap, sem stack) | 1.80 (≈) | 9.2% | 0 | reverte: probe mostrou 2433 engajamentos corretos mas concat ≈ stack em wall — o custo está no tráfego de montagem, não nas alocações |
| cold-aware retention (mincore: só retém repetido + não-residente) | 1.88 (≈, dentro do ruído) | 9.2% | 0 | reverte: prova que o LRU já é write-quiet no decode (puts ≈ 0; 125k misses com 0 evictions) — o gate não tinha o que filtrar; page cache (17% residente) + seeder bastam |
| cross-layer prefetch Fate-style (router l+1 sobre gate-input l, top-k+2, F_RDADVISE) | 1.82–1.86 (≈; OFF 1.96 no mesmo box) | 9.2% | 0 | reverte: recall 51% (bar 85%; Fate relata 97% noutros modelos — similaridade cross-layer menor neste checkpoint) com 2× advises (117k vs 63k) e zero ganho — o teto é entrega/overlap, não o sinal; auditoria achou ainda env-parse sem guarda e união multi-row sem budget (corrigidos antes do revert) |
| coalesce I/O (Fase 2) | sem bench | 9.2% base | 0 | nao executado: telemetria PROFILE mostra 92.6% runs singleton no decode (184518/199206, max 10) e consecutivos ja coalescidos; fundir singletons = ponte de gaps, morta em 2 regimes (34.0s vs 31.4s; 55.0 vs 47.5); macOS sem io_uring nao tem syscall multi-range com furos |
| cache-prior GLM-5.3-Flash-JANG-MTP (bonus 2.0, budget 1 GiB) | **1.141 (×3) vs 1.008 exact (×2) = +13.2%** | 8.5% → 25.8% | — | **LANDED opt-in** após fix do loader (export transformers cru nunca carregou antes: remaps de container/hc/conv/bias + drop do draft layer); kill-gate +10% superado; TTFT ~11s e saídas sãs em ambos os braços |
| topk 0.85 GLM-JANG (budget 1 GiB) | **1.532 (×3, spread ±1.2%) = +52.2%** | 8.6% | — | hook do `Glm5NextMoE` herdado do oQ4e, **primeira medição no JANG** (oQ4e deletado); ppl 5.9546 (+3.3%) — joelho: 0.70 dá +103% por +16.5% ppl (penhasco) |
| **topk 0.85 + prior 2.0 GLM-JANG** | **1.826 (×3) = +81.4%** | 30.9% | — | **Pareto recomendado**: knobs independentes e aditivos (truncagem de massa reduz o I/O; prior reordena o que sobra); custo de ppl é o do topk sozinho (+3.3%); saídas sãs; 0.70+prior chega a 2.23 (+121%) por +16.5% ppl — posição de velocidade máxima, não default |
| MTP nativo d1 GLM-JANG (budget 1 GiB, Gap-1 LANDED) | **1.303 (×3) vs 0.894 base = +45.7%** | 8.6% | — | port Lightning MTP → módulo vendored glm5_next; draft layer 45 attachado (antes: dropado); accept 80.8%, 1.85 tok/ciclo, draft = 3% do backbone; **32/32 tokens bit-idênticos** ao base (sampling-exact por construção: verify contra o trunk) |
| MTP nativo d2 / d3 GLM-JANG | 1.231 (+37.7%) / 1.149 (+28.6%) | 8.6% | — | d1 > d2 > d3 em prompt prose: o controlador adaptativo explora drafts profundos que falham além da posição 1; default do modelo 3, `--mtp-depth N` fixa o teto no bench |
| **MTP d3 + topk 0.85 + prior 2.0 GLM-JANG** | **1.786 (×3) = +99.8%** | 30.9% | — | **novo Pareto**: MTP empilha sobre o combo (draft corta ciclos, topk corta bytes, prior reordena o resto); todos os braços passam o kill-gate +10%; ordem dos runs: base→d1→d2→d3→combo (page cache aquece ao longo — base pode estar subestimado) |
| MTP novo 4M/4S JANG (Gap-2, single-stage) | 4M: 2.72 d1 vs 3.55 base (**−23%**); 4S: 2.79 d1 vs 4.02 base (**−31%**) | — | — | **MTP OFF recomendado sob streaming**: o MTP upstream virou 1 stage (`mtp.layers.[0]` + `mtp_draft/vmlx_mtp_proposal_head.safetensors`, só um lm_head 320-dim) — depth 2/3 não têm stage; d1 bit-exato (`tokens`) com accept 68–75% e 1.76–1.79 tok/ciclo, mas o forward de verify (2 posições, expert sets distintos) custa **2.3× as leituras SSD** do passo base — 1.79 < 2.3, derrota estrutural. Sem streaming (residente) o verify ≈ 1× e o MTP venceria; nesta caixa o streaming é obrigatório. Bateria `bench/results/mtp_gap2/` |
| VLM MTP adapter graceful fallback | — | — | 0 | sem o fix, toda rejeição parcial de draft crashava o request (`LanguageModel has no mtp_clamp_accept` — o adapter anunciava o hook opcional mas o delegate explodia; qwen4_exp não tem bound de replay); agora clamp ausente = sem clamp e rollback ausente = False (fallback para passo não-MTP, já previsto pelo chain). Sem mudança para modelos com o hook. Teste `tests/test_vlm_mtp_adapter.py` |

Conclusão: neste workload o teto é volume (1.6 GB/tok) + granularidade (200k preads singletons) + custo por uso (promote/stack/eval). Retenção e prefetch por prev-token perdem; alavanca restante seria QMM sem stack por expert (não tentado). QSA nativo, em contraste: 4.3 ms vs 18.0 ms portable (~4.1×).

### Protocolo A/B congelado (Fase 2: prefetch cross-layer)

```sh
.venv/bin/python bench/bench_expert_streaming.py --model qwen-jang4m --budget 1.0 --prompt-len short --out /tmp/<arm>.json
```

3 reps por braço (Fase 3: 3 reps prior + 2 exact — terceira exact abortada pelo guarda de RAM do box), page cache warm, checkpoint `Qwen3.8-Flash-Next-JANG_4M`.

### Governador dinâmico de residência (N1, LANDED opt-in)

Budget fixo é decidido no load e nunca revisitado; a memória livre do sistema continua
se movendo (observado na matriz 36-cell: pico de RSS ~25-27G em 51G com o decode rodando
a maior parte do tempo abaixo de 50% de uso). O governador revisita a capacidade do LRU
nos boundaries de request a partir da memória disponível (`psutil.available`, com
fallback `vm_stat`):

- livre < 10% da RAM → `clear()` do cache (desespero; páginas são re-legíveis do SSD)
- livre < 20% da RAM → reduz a capacidade à metade (piso `min_cap`)
- livre > 40% da RAM → dobra a capacidade (teto `max_budget`)
- entre 20% e 40% → estável (histerese; zero churn de resize)

Cooldown de 30s entre ações. Opt-in: `OMLX_EXPERT_STREAMING_DYNAMIC=1` com budget > 0
(um budget-0 é page-cache-only por escolha operacional e fica intocado). Teto do
crescimento: `OMLX_EXPERT_STREAMING_DYNAMIC_MAX_GIB` (default 6). Roda na thread de
inferência no mesmo ponto do summary por request — sem corrida com put/get. Estado no
`expert_streaming_summary` (`governor.actions/last_action/last_free_gib/capacity`);
ações logam uma linha `expert_streaming governor: ...`.

Nota honesta de calibragem: sob carga de bench com 4 requests concorrentes + page cache
sujo, `available` fica ~16G (entre 20% e 40% de 48G) e o governador deliberadamente
não age — histerese fazendo o trabalho dela. Em máquina descansada ele cresce (provado:
3077→6162 slots no primeiro boundary). Thresholds relativos à RAM total, não absolutos.

### Knob aproximado: cache-prior rerank (Fase 3, LANDED opt-in)

`expert_streaming_cache_prior` (per-model settings + WebUI + app macOS; env `OMLX_EXPERT_STREAMING_CACHE_PRIOR` como fallback; excluído de profiles como os demais knobs de hardware; entra na runtime signature). Default `0.0`/nulo = roteamento exato, bit-idêntico. Bônus em logit-space para experts residentes no LRU antes do top-k (Qualcomm 2412.00099). Aproximado por desenho — muda outputs; saídas inspecionadas sãs. Sweep no autotuner (`--sweep-prior`, candidatos `--priors`).

Calibragem (Qwen-JANG_4M, budget 1 GiB):

| bonus | short tok/s | short hit | 2k tok/s | 2k hit | qualidade |
|---|---|---|---|---|---|
| 0.0 (exato) | 1.88 (×2) | 9.2% | 1.43 | 1.0% | referência |
| 0.5 | 1.76 | 14.4% | — | — | sã |
| 1.0 | 2.08 (×3) | 19.3% | — | — | sã |
| 2.0 | **2.78 (×3)** | 38.0% | **1.76** | 17.6% | sã (short + 2k) |
| 4.0 | 1.01 | 17.7% | — | — | **degenerada** ("This just a response is a response") |

Penhasco de fidelidade entre 2.0 e 4.0; default recomendado `2.0` onde houver headroom de qualidade, `1.0` conservador. TTFT idêntico ao exato (hook só no decode).

Fidelidade (bateria 6 prompts, greedy, exato vs 2.0): divergência no token 0 em todos (rerank atua já no prefill) mas 6/6 saídas sãs e on-task (haiku, 17×23=391, Hamlet, HTTP/2); prefill-ppl idêntico (3.3144 exato e prior, 8 janelas canary) — o knob só morde no decode, onde o LRU tem o que dizer. **Condicional de orçamento**: com budget 0 (page-cache only) o resident set é vazio e o rerank é overhead puro (autotune b0: 1.60 vs 1.70) — o conversor recusa o prior sem LRU com warning.

**GLM-5.3-Flash-JANG-MTP A/B (Fase 3, LANDED)**: o checkpoint é um export transformers cru que nunca carregou neste repo (chaves `model.*` bare, params `hc_base/hc_fn/hc_scale`, convs 2D bare `q/k/v_conv1d`, bias do router no nível do MoE, camada 45 = draft MTP). Fix no `Model.sanitize` vendored: remap de container, params hc, conv reshape/fusão, `e_score_correction_bias` → `gate.`, drop do draft layer. Depois do fix: load 4.2s, output são. Cache-prior 2.0: **1.141 tok/s (×3) vs 1.008 exact (×2) = +13.2%**, hit 8.5%→25.8%, TTFT ~11s inalterado, saídas sãs em ambos os braços. O rerank GLM (bônus nos logits crus antes do `group_expert_select`) passa o kill-gate de +10%.

**GLM follow-ups (Fase 3)**: (1) PPL canário (ctx 512, 8 janelas, 4088 terms): **5.7644 exato e prior** — nll bit-idêntico (1.751704), o rerank não degrada fidelidade no checkpoint GLM, só reordena o I/O; (2) sweep de bônus (budget 1 GiB): 1.0→1.063 / **2.0→1.277** / 3.0→1.233 — 2.0 é o ótimo GLM também (3.0 degrada, como na calibração); (3) autotune GLM (--sweep-prior): b1 sem prior 0.934 vs b0 1.021 — **no GLM o LRU sozinho não paga; o prior é quem faz o budget render** (1.14–1.28 com prior). Autotune corrigido: braços prior em base b0 eram recusados por construção (budget-0 refusal re-media o baseline); agora carregam budget>0 clamped ao room.

**topk 0.85 no GLM-JANG (revisão de métodos não-testados)**: revisando a pesquisa da sessão contra o que rodou, três métodos tinham evidência no GLM-oQ4e (deletado do disco) mas nunca no JANG: o truncamento de massa topk (E5: +27% no oQ4e), o MTP (A2: noise no oQ4e sem draft — **o JANG-MTP tem o draft real**, layer 45), e o cold tier (I5/I6: `expert_cold/` gerado para o oQ4e, não existe para o JANG). O topk 0.85 mediu **+52.2% (1.532 ×3)** no JANG, e o combinado com prior 2.0 **+81.4% (1.826 ×3)** — knobs independentes e aditivos: o topk corta bytes/token (menos experts roteados), o prior reordena o I/O do que sobra (hit 8.6→30.9%). Custo de ppl do combinado = o do topk sozinho (+3.3%; o prior é grátis — o gate ppl é prefill-shaped e o prefill_bypass não enche o LRU, então o rerank nunca dispara lá, consistente com o prior-vs-exato bit-idêntico). Método: 2 falsos negativos capturados — (a) a primeira bateria do combo rodou contaminada por PPLs em paralelo (0.681 → 1.826 limpo); (b) o braço via env `OMLX_MOE_TOPK_THRESHOLD` não se auto-descreve no `effective_config` (disciplina M5) — o ppl harness ganhou `--topk` como ModelSettings (paridade 5.9546 confirmada). MTP no JANG **LANDED (Gap-1)**: o draft layer 45 agora attacha sob `--mtp` (seção acima) em vez de ser dropado; quando o MTP está inativo o drop permanece (comportamento stock). Resta o slot-bank como item de maior incerteza/upside. Métricas: tok/s (gate **+10%**), TTFT, hit_rate, read p50/p95, bytes/token, acurácia de proposta por camada (`prediction_totals.recall`, bar **≥85%** para preditores). Abaixo do gate: revert sem resíduo + linha no scoreboard. Cada fase: testes → subagente auditor (bit-exatidão do default, thread-safety MLX, custo zero com feature off) → bench → gate → commit.

### MTP nativo no GLM-5.3-Flash-JANG-MTP (Gap-1, LANDED)

O JANG guarda o draft como **uma camada trunk a mais** (`model.layers.45.*`, convenção nextn) em vez de chaves `mtp.*` reais. O port cobre 5 pontos:

1. **sanitize** (`gl5_next/language.py` vendored): remapa `model.layers.45.*` → `language_model.mtp.0.*`, com fusão draft de `kv_b_proj` → `embed_q`/`unembed_out` (o pai DSV32 só funde `model.layers.0..44`); `remap_mtp_quant_overrides` copia os overrides `layers.45` → `mtp.0.block`.
2. **quant no load** (`model_loading.expand_per_layer_quant_keys`): variants de runtime `language_model.mtp.<i>.[block.]<resto>` para chaves nextn — sem elas o `nn.quantize` cai no global 8-bit e o banco 2-bit gs64 do draft falha no strict load (era o 3º erro de shape).
3. **HC identidade** (`glm5_next_model._identity_hc`): o draft não traz pesos `attn_hc/ffn_hc`; HC sem parâmetros (residual puro, bit-exato quando os streams são broadcast idêntico).
4. **streaming do stage** (`expert_streaming`): descobre `language_model.mtp` mesmo quando `layers` resolve pelo root, aceita `mlp` além de `ffn`, chaves raw `model.layers.45.*` + remapeadas, e desliga `compile_ffn` no DecoderLayer interno (não no wrapper — `mx.eval` do plano é ilegal sob `mx.compile`). 43 banks convertidos (42 trunk + 1 draft).
5. **bench**: `--mtp --mtp-depth N` (teto de draft; default = modelo, 3), `--model glm-jang` detecta o path nativo pelo `config.json`.

Protocolo: `--model glm-jang --budget 1.0 --decode 96 --min-free-gb 6 --single-request`, 3 reps/braço, resultados em `bench/results/mtp_gap1/`. Com MTP inativo o patch é pass-through (só o corpo linear-attn é substituído, matematicamente idêntico a `n_confirmed=0`).

**Gap-2 fechado por inspeção (sem código)**: o `switch_mlp` do JANG **já é 2-bit gs64 em todas as 43 camadas** (gate/up/down `(288,2048,256)` + scales `(288,2048,64)`); o resto é 8-bit gs64 e o embed 6-bit. Não há tier abaixo de 2-bit no formato — cold tier para o JANG não existe (o `expert_cold/` era do oQ4e, deletado do disco).

### DeepSeek V4 Flash JANG (`DeepSeek-V4-Flash-0731-JANG`, sem draft heads)

> Nota: o checkpoint `DeepSeek-V4-Flash-0731-JANG` em disco **não tem draft heads** (censo das 102 shards: só `layers.0–42`, nenhuma chave `mtp/dspark/draft/nextn`) — MTP nativo é impossível nele; o trecho sobre oQ4e-mtp ao final descreve o antigo checkpoint (deletado).

**Spill-stacking (carga do checkpoint per-expert).** O JANG guarda os 256 experts/camada desagrupados (`layers.N.ffn.experts.{i}.w{1,2,3}.*`); o sanitize empilharia ~64 GiB de uma vez e o `mx.eval` do load morria com SIGKILL nesta caixa de 48 GiB. O sanitize agora empilha **uma camada por vez (~2 GiB transitórios), salva em `AI Models/.omlx_spill/<modelo>/spill_layer_*.safetensors` e recarrega via mmap** — o strict load vê as chaves `switch_mlp` empilhadas de sempre, e a conversão para streaming (que roda antes do `materialize_lazy_state`) descarta esses arrays antes que sejam lidos. Spill com manifest (sizes+mtimes das shards): segundo load em diante é *spill hit*, sem re-empilhar. Detalhes em `omlx/patches/deepseek_v4/spill.py`; kill-switch `OMLX_DSV4_SPILL=0`; spill fora do dir do checkpoint (o discovery varre `**/*.safetensors` recursivamente). Identidade byte-a-byte do spill vs empilhamento em RAM verificada (`/tmp/probe_spill_identity.py`: 9/9 bancos da camada 0 IDENTICAL).

**Bateria medida** (`bench/results/dsv4_stream/base_r{1,2,3}.json`, `--model dsv4-jang --budget 1.0 --decode 96 --min-free-gb 6 --single-request`, 48 GiB compartilhados): load **3.4–3.5 s**, **8.1–8.2 GiB** pós-load (pico vitalício ~17 GiB), TTFT 5.8–6.8 s, decode **1.48 / 2.61 / 2.31 tok/s** (rep 1 com page cache frio). Saída sã e coerente (`bit_exact_kind: text`). Gate de residente bit-exato indisponível por construção (o residente não cabe na máquina) — a garantia é identidade do spill + sanidade da saída.

### DeepSeek V4 Flash (oQ4e-mtp — checkpoint antigo, deletado)

`deepseek_v4` nests the MoE under `layer.ffn` (not `mlp`) and keeps one routed bank per **MTP/DSpark stage** under `mtp.<stage>[.block].ffn.switch_mlp`. The converter walks both: 43 main layers + 3 draft stages (layer ids `43..45` share the same LRU). Notes:

- **Residency**: the `mtp.<stage>` banks count as expert bytes in `expert_streaming_estimate`, so `resident_bytes`/`streaming_bytes` and the admin capability flags stay accurate for the `-mtp` checkpoints (~9.7 GB of draft-stage experts at oQ4e would otherwise sit resident). When the runtime MTP is inactive the converter simply converts fewer layers and the per-layer LRU split is rebalanced.
- **Activation**: DeepSeek V4's `SwitchGLU` uses `LimitedSwiGLU(swiglu_limit)` (fp32 on draft stages); the streaming GLU copies the original activation, keeping output bit-exact with the resident path.
- **Gate**: layers `0..2` are hash-routed (`tid2eid[input_ids]`); routing is untouched by streaming — only the expert banks are swapped.
- **Budget (measured estimate)**: dense 7.82 GiB resident + streaming total 8.21 GiB; per-expert 12.75 MiB × 256 experts × 46 banks (43 layers + 3 DSpark stages — the checkpoint ships 3, `num_nextn_predict_layers` notwithstanding). The default 1 GiB gives ~1 slot/layer (GLM-class numbers, ~0.07 tok/s on the measured baseline); 4–8 GiB is the sensible range if you have the RAM headroom.
- **First measured streaming run** (`dsv4_short_f.json`, 48 GiB shared box, budget 0 / page-cache only, QD16): load 2.9 s (dense only — the 150+ GiB expert banks stay on SSD), short-prompt TTFT 8.0 s, decode **1.223 tok/s** (16 tokens, disk 1.1 GB/s — page-cache hits from the prefill sweep cover most expert reads). Fastest short-TTFT of the three supported families (qwen ~10 s, GLM ~18.5 s); steady-state decode beats GLM-5.3's 0.697 tok/s despite the larger per-expert banks. Long prompts (2k/8k) are a different regime: at oQ4e a full 46-bank sweep is ~150 GiB, and the chunked guard re-sweeps per chunk when the page cache cannot hold the bank — measure on an idle machine before quoting numbers.

## Settings

| field | type | notes |
|---|---|---|
| `expert_streaming_enabled` | bool | hardware-specific; excluded from profiles/templates |
| `expert_streaming_budget_gib` | float? | `null` = automatic RAM-scaled cache (default, via `budget_auto`); `0` = **page-cache only** (no app-level LRU); `>0` = fixed LRU heap, clamp `0–64`. Any explicit value (including `0`) always wins over `budget_auto` |
| `expert_streaming_budget_auto` | bool? | Default-**on** RAM-scaled cache: `min((ceiling − streaming − 2 GiB) × 0.5, min(8 GiB, knee))`, whole MiB. More RAM = more cached experts. `null` follows the default (on); `false` opts back out to page-cache only. Machine-specific; excluded from profiles |
| `expert_streaming_topk_threshold` | float? | `null` / `>= 1.0` = exact routing (default, bit-exact); `0.05–0.95` = adaptive top-k mass truncation (approximate, changes outputs). Hooks exist only for `qwen4_exp` and `glm5_next` (+ `_text`); other types warn and stay exact |
| `expert_streaming_pin_sync` | bool? | Fase M1: apply learned pins synchronously at load (bench arms). Default off (background) |
| `expert_streaming_pin_regime` | str? | Which routing sample pins read: `decode` (default) or `prefill` |

All are load-time settings: toggling unloads (and re-loads if pinned) the model. The runtime signature includes the *effective* (forced-or-requested) value and the threshold when `< 1.0` (it changes outputs), so the `GET /admin/api/models` capability flags `expert_streaming_supported / forced / reason / *_bytes / moe_layers / per_expert_bytes` always reflect what would actually run.

## UI

**Tuning com evidência (autotune + knobs):** o `bench/autotune_expert_streaming.py` varre budget/QD/pilot/coalesce/readahead/seed (bit-exact, automáticos) + topk/prior/**cold_tier** apenas com flags explícitas (`--sweep-topk`, `--sweep-prior`, `--sweep-cold-tier` + `expert_cold/` presente) — one-factor-at-a-time + head-to-head + validação 8k, watchdog de RAM/swap — e `--apply` persiste o vencedor no per-model profile, o mesmo store que a UI edita. **Política do projeto: defaults bit-exact.** Levers de qualidade (topk, cold tier) mudam outputs, são opt-in documentados e têm gate próprio (`bench/ppl_expert_streaming.py --streaming`, corpus fixo; ppl determinístico). Medido no JANG 4S: topk 0.85 = **+11.5% tok/s, +3.9% ppl** (4M: +3.0% ppl) — knob opt-in, nunca default. Cold tier: near-lossless por construção, **rejeitado como linha de desenvolvimento** (decisão do mantenedor; ver papers doc — tool e runtime ficam dormentes atrás dos knobs).

- **WebUI**: card `Advanced → Expert Streaming (SSD)` → toggle + `Cache budget (GiB)` input + `Auto RAM-scaled cache` toggle + pins block (`Pin budget`, `Apply pins synchronously at load`, `Pin profile regime`). Visible only when `supported`; disabled amber hint when `forced`. Lives under the `Experimental Features` header, before the Qwen ANE block.
- **macOS app**: `Model Settings → Advanced` → same toggle + conditional `Expert LRU budget` and `Auto RAM-scaled cache` rows, pins block with sync toggle + regime field (auto-save, like `qwen4_ple_ssd_offload`). The `GiB` field clears to `null` when empty; values outside `0–64` are rejected; regime accepts `decode`/`prefill`, empty clears.

## Measured baseline (48 GiB Mac, external SSD, warm page cache)

Decode 48 tokens after a short prefill, `OMLX_EXPERT_STREAMING_PROFILE=1`:

| model | budget | hit rate | tok/s | notes |
|---|---|---|---|---|
| Qwen 99G | 0.5 GiB (8/layer) | 0 % | ~1.0 | flat — bottleneck is per-call serial load, not hits |
| Qwen 99G | 1–2 GiB (8–16/layer) | 0 % | ~0.85 | same |
| Qwen 99G | 4 GiB (32/layer) | 23 % | ~0.9 | knee — more memory buys nothing |
| Qwen 99G | 8 GiB (64/layer) | 32 % | 0.34 | **negative**: big cache evicts OS page cache → misses re-read SSD |
| GLM 190G | 1/2/4 GiB | 0 % | 0.065–0.072 | 13 MB/expert; ~120 ms/call serialized copies dominate |

Per-stage profile (per call): Qwen — `gate_eval 1.5 ms` + `load 16 ms`; GLM — `gate_eval 1.4 ms` + `load ~120 ms`. Physical footprint is bounded: Qwen 5–7 GiB, GLM ~14 GiB (dense materialized 10.5 GiB + cache). Conclusions:

- **The bottleneck is the synchronous per-layer path** (router eval → host round-trip → serial expert loads → stack → gather), not hit rate. Cache size saturates quickly (slipstream's 6.25 % finding holds at 48 GiB).
- **Bigger caches go negative** past the knee (8 GiB on Qwen): expert cache competes with the OS page cache that makes misses cheap.
- Default `1 GiB` is right; invest in overlap/prefetch (next phase), not more memory.

## Cold-cache baseline + PILOT A/B (phase 1 result)

Same bench with a cold page cache (SSD delivering ~320–390 MB/s sustained, `bench/resource_sampler.py` measuring GPU/CPU/disk/RSS per phase), decode 8–16, GLM 190G:

| config | tok/s | disk read (decode) | GPU util | sync load |
|---|---|---|---|---|
| GLM 4 GiB, PILOT off | 0.037 | 386 MB/s | 13 % | 10.9 ms/miss |
| GLM 4 GiB, PILOT on (staging) | 0.011 | 326 MB/s | 8 % | 13.1 ms/miss |
| GLM 8 GiB, PILOT off | 0.063 | 316 MB/s | 11 % | 9.2 ms/miss |

Key findings:

- **Decode is I/O-bound at the SSD, not latency-bound**: GPU idles at 8–13 %, disk read saturates at ~330 MB/s. The working set per token (42 layers × 10 experts × 13 MB ≈ 5.5 GB) ÷ 330 MB/s ≈ 16.6 s/token ≈ **0.06 tok/s — the measured 8 GiB result sits on the physical I/O floor**. No software overlap can beat this without reducing bytes per token.
- **PILOT (router-lookahead prefetch) is strictly negative in this regime.** Workers read numpy slices into a staging buffer (0.49 ms/staged hit vs 10.9 ms sync — the mechanics work), but the router predicts one layer ahead and the disk is already saturated: 94 % of staged bundles were dropped unconsumed (~380 GB of wasted reads) and the demand path slowed by contention. A/B: 0.037 → 0.011 tok/s.
- **No inter-token expert reuse on GLM**: hit rate stays 0 % even with 606 slots (14/layer, 63 % of one token's per-layer demand) at temperature 0. GLM routers change completely between tokens (unlike Qwen's 23–32 % inter-token hits at 4 GiB).
- PILOT is therefore **default OFF** (`OMLX_EXPERT_STREAMING_PILOT=1` to opt in). It only pays when I/O is *not* saturated (warm cache, budget ≥ working set) — the exact regime where slipstream found prefetch didn't pay either.

Strategic conclusion: for models ≫ RAM the levers that matter are (a) **batch decode** (amortizes the per-token working set across requests), (b) budgets ≥ the batch working set, and (c) reducing bytes/token (lower-bit requant of cold experts). Prefetch/overlap alone cannot. Fase A below tested (a) and (b) directly — both failed on a 48 GiB box; the levers that remain are bytes/token and bandwidth.

## Fase A — capacity, MTP, batch (decision experiments)

Protocol: cold page cache between runs (`bench/cache_cool.py` touches ~72 % of available anon memory), PILOT off, `bench/resource_sampler.py` on, GLM 190 G unless noted.

### A1 — capacity sweep (diagonal-reuse hypothesis, Patterns Ob2)

| budget | hit | tok/s | phys at end | note |
|---|---|---|---|---|
| 8 GiB (ref) | 0 % | 0.063 | 14.2 G | the I/O-floor result |
| 16 GiB | 15.9 % | 0.040 | 19.4 G | disk **writes** 340–580 MB/s during decode (swap) |
| 24 GiB | 19.5 % | 0.053 | 22.2 G | still swap-bound, below 8 GiB |

- Inter-token reuse on GLM **exists** (0 % → 16 % → 19.5 % as capacity grows — the diagonal from Patterns Ob2 is real), but capturing ≥ 2 tokens of working set needs ≥ 16 GiB of heap, which pushes the 48 GiB box into swap: the *capacity point sits below the reuse point* on this hardware. The budget sweet spot stays ≤ 8 GiB.

### A2 — MTP (SP-MoE amortization) — SUPERSEDED, see Fase B retest

- GLM-oQ4e 8 GiB + `--mtp`: 0.073 tok/s (+16 %) — within noise; the checkpoint has no `-mtp` suffix (draft weights likely stripped by the publisher).
- Qwen 4 GiB + `--mtp`: 0.326 tok/s ≈ the no-MTP cold single run. Draft steps fault *their own* experts → bytes/step rise faster than accepted tokens pay back, in this I/O-bound regime.
- Fase A verdict "MTP stays opt-in" was an **access-method artifact**: the mmap+LRU stack put wall/call at 30 ms, so the cycle's 3.4 forwards/token (164 calls / 48 layers) dominated. Retested after pread + page-cache-only (B2) — see Fase B below: MTP is now a **win (+27–37 %)**.

### A3 — concurrent (batched) decode, GLM 8 GiB, distinct prompts

| N | aggregate tok/s | per request | hit |
|---|---|---|---|
| 1 (ref) | 0.063 | 0.063 | 0 % |
| 2 | 0.053 | 0.026 | — |
| 4 | 0.043 | 0.011 | 0 % |

- Distinct prompts route to distinct experts: the per-step working set scales ~linearly with N while the SSD stays saturated — batch **amortizes nothing** here (the ≥ 1.8× criterion failed; aggregate declines monotonically). Batch would only pay when requests share routing (same domain/language) — worth re-testing with homogeneous traffic, not the default serving assumption.

### Fase A conclusion

On a 48 GiB Mac with one ~330 MB/s SSD, GLM-190G decode is pinned to the I/O floor and **every byte-adding strategy makes it worse** (MTP drafts, batch). The remaining levers are both physical: **bytes/token** (oQ2e requant of cold experts ≈ 2×) and **bandwidth** (multi-SSD striping, now supported — see below).

> **Correction (post-Fase A):** the "I/O floor" above was an artifact of the
> access method, not the hardware. The 4 TB external SSD is a ~3.7 GB/s
> NVMe; `mmap` + `MADV_RANDOM` cold page faults deliver only 0.2–0.4 GB/s.
> Switching to one `os.pread` per contiguous expert slice lifted decode from
> 0.063 to 0.336 tok/s (5.3×) — see the next section. Byte-adding strategies
> (MTP drafts) also flipped sign once the per-call cost dropped — see B5.

## pread + parallel demand loads (the real fix)

Micro-benchmark on a GLM oQ4e shard (40 × 4 MB expert slices, cold page cache):

| method | cold GB/s |
|---|---|
| mmap + MADV_RANDOM (page faults) | **0.35** |
| `os.pread` (single contiguous read) | **10.6** |

`_ShardReader.expert_slice` now uses one `os.pread` per expert slice, and the
quantized streaming linear resolves the whole per-layer demand set with a
thread pool (`_EXPERT_IO_POOL`, 8 workers, QD8) — workers return raw numpy
slices, MLX promotion stays on the inference thread.

GLM 190G oQ4e, 8 GiB budget, decode 16, cold cache, PILOT off:

| load path | tok/s | TTFT | disk (decode) | load ms/call | GPU |
|---|---|---|---|---|---|
| mmap faults (Fase 0/1) | 0.063 | ~200 s | 330 MB/s | ~190 ms | 11 % |
| pread, serial | 0.249 | ~40 s | 1.4 GB/s | 22.5 ms | 29 % |
| pread, 4 workers | 0.313 | 22.9 s | 1.85 GB/s | 15.4 ms | 35 % |
| pread, 8 workers | **0.336** | 22.8 s | 1.85–2.3 GB/s | 14.7 ms | 35 % |

Qwen 4 GiB cold single generation: 0.853 tok/s (was 0.30). PILOT remains
negative even on the fast path (38 ms vs 30 ms wall/call — 89 % of staged
bundles dropped), so it stays default-off. Batch decode still amortizes
nothing (see Fase A3). Remaining bottlenecks: disk random-read ceiling
(~2.3 GB/s of the 3.7 sequential spec) and the mini-bank assemble/promote
(~5 ms/call); bytes/token (oQ2e) is the next real lever.

## Fase B — FlashNext comparison (page-cache era)

Ported the transferable techniques from [macqwen-releases](https://github.com/1architect/macqwen-releases) ("FlashNext", MIT). Their README claims 1.9 tok/s on Qwen3.8-Flash-Next; their own code comments pin down the components: threshold-1.0 exact = 0.53, 0.85 mass truncation = 0.94 ("the cliff"), 1.9 = 0.85 truncation + mlock pins + warm file cache + steady state. Our bit-exact pread path already measured 0.853 cold where they measure 0.53 exact.

### B1 — fixed-cost cuts (bit-exact) — commit 7e958a6

- BF16 promoted via `mx.array(v).view(mx.bfloat16)` bit-reinterpret: ~9× faster on 4 MB slices, and it matches `mx.load` exactly — the old `shift → f32 → astype` roundtrip flushed bf16 subnormals to zero (Metal FTZ), i.e. the *old* path was the inexact one.
- One host sync per MoE layer: the first streaming linear builds a shared `_RemapPlan` (eval + unique + compact remap via `np.searchsorted`), gate/up/down reuse it. (MLX 0.32 refuses zero-copy CPU dlpack — `mx.from_dlpack(np, copy=False)` — so the view-reinterpret is the practical promote.)
- Demand reads sorted by ascending expert id (= ascending file offset, row-major banks).
- Profile (Qwen 4G cold): wall/call 21.8 → 9.2 ms; stack bucket 5.4 → 0.5 ms. Decode is now **disk-bound** — remaining levers are fewer bytes (truncation) and fewer disk reads (page cache / pins).

### B2 — page-cache-only is the new default — commit cd10501

Budget semantics: `null`/`0` = no app-level LRU — expert reuse rides the OS file cache (clean, evictable pages, never swapped). `>0` = fixed LRU heap (opt-in). Admission charges dense bytes only for the default (file cache is not committed memory → more load headroom).

Cold A/B on the 48 GiB Mac (16-token decode):

| config | tok/s | RSS (decode avg) |
|---|---|---|
| Qwen LRU 4G | 0.834 | 7.3 GB |
| Qwen **page-cache only** | **1.007** (warm 1.133) | 5.6–5.8 GB |
| GLM LRU 8G | 0.363 — 0% hit rate, 8 GB of bundles pinned in RSS | 11.7 GB |
| GLM **page-cache only** | **0.381** | 4.8 GB |

Even at a 14.5% LRU hit rate (Qwen) the heap loses: it pins RSS that the OS could have used for page cache. Matches FlashNext's finding that their app-level LRU always lost.

### B4 — adaptive top-k truncation (opt-in, approximate) — commit 15071667

`expert_streaming_topk_threshold`: after top-k selection, keep the smallest score-descending prefix whose relative mass reaches the threshold; dropped slots reuse the top expert (duplicates collapse in the streaming plan — no extra I/O); kept scores renormalize to the original total top-k mass. `null`/`>= 1.0` bypasses everything — bit-exact by construction (verified: identical generated text on the real checkpoint with `--topk 1.0`).

Cold sweep (16-token decode, page-cache only):

| threshold | Qwen | GLM |
|---|---|---|
| exact | ~1.0 | 0.381 |
| 0.85 | 1.091 | **0.485 (+27%)** |
| 0.70 | 1.197 | — |

Outputs diverge by design below 1.0 (FlashNext measured 7/10 identical tokens at 0.70); the WebUI/Swift hint carries that warning.

### B3 — mlock pins (opt-in) and warm-only prefetch (experimental)

- **Pins** (`OMLX_EXPERT_STREAMING_PIN=1`): observe routing for the first 8 decode calls, then `mlock` the page-aligned file ranges of the most frequent experts per layer within `OMLX_EXPERT_STREAMING_PIN_GIB` (default 0.25 GiB, `OMLX_EXPERT_STREAMING_PIN_TOKENS` for the window). Zero-copy — the locked pages ARE the file cache pages — but they become wired memory. The mapping address is obtained via `PyObject_GetBuffer` (read-only mmap cannot expose a writable buffer). Qwen 48-token decode: **1.538 → 1.764 tok/s (+15%)**; GLM short decode: neutral (13 MB experts → the budget covers ~2 experts/layer, and the 8-token observation eats half a 16-token run).
- **Warm-only prefetch** (`OMLX_EXPERT_STREAMING_WARM=1`): before a layer's demand loads, fire discarded reads for the previous token's next-layer experts (independent routing repeats ~35% across adjacent tokens). On the 48 GiB Mac: **neutral to negative** (1.531 vs 1.538 alone; drags pins 1.764 → 1.624) — the QD8 demand path already saturates the NVMe, and warming adds read traffic. It exists for the 16 GB-class case FlashNext targets (small page cache, less demand pressure); leave it off otherwise. The old PILOT staging prefetcher (default OFF) is superseded by this.

### B5 — MTP retest after the pread/page-cache stack: now a win

Upstream v0.6.3 fixed Qwen4 Lightning MTP weight detection (#3200: qwen4_exp binds only the embedded `mtp.*` head; the draft layer carries its own `switch_mlp` bank, so the draft pass streams its experts like any other layer). Retested on the Fase B stack — Qwen, 0G page-cache-only, 48-token decode, cold:

| config | tok/s | vs no-MTP (1.538) |
|---|---|---|
| MTP | 1.958 / 2.110 (with profile) | **+27–37 %** |
| MTP + pins | 2.058 | **+34 %** |

Why it flipped: the draft/verify cycle runs **3.4 target forwards per generated token** (164 MoE-layer calls / 48 layers — acceptance is modest), but the B1/B2 stack cut wall/call from 30 ms to **3.97 ms**. Per-token cost went from ~4.9 s (unpayable) to ~0.65 s of MoE I/O, and the wider verify reads reuse page-cache-resident experts from the draft rounds (`sync_ms_per_load` 0.16 ms). TTFT unchanged (~10 s cold). Conclusion: **enable Lightning MTP by default for streaming qwen4_exp checkpoints that ship `mtp.*` weights**; keep it opt-in elsewhere (GLM has no draft weights, DeepSeek untested on the new stack).

### Recommendation matrix

| hardware | budget | pins | threshold | MTP |
|---|---|---|---|---|
| 48 GB+, model >> RAM | default (0) | optional (+15% Qwen) | 1.0 exact | on (qwen4_exp with `mtp.*`) |
| 48 GB+, quality flexible | default | optional | 0.85 | on (qwen4_exp with `mtp.*`) |
| 16 GB-class | default | `PIN=1` | 0.85 (+ `WARM=1` to test) | on (qwen4_exp with `mtp.*`) |

### B5 — LRU at 6 GiB is net-negative (Fase J)

Measured on Qwen 8k prompt with page-cache only vs LRU (budget=6 GiB, per-layer cap ~47 slots): LRU hit 4.95% with 130k evictions and **0.649 tok/s vs 2.86 tok/s page-cache only**; swap write spikes 519 MB/s and `phys_lifetime` ~19 GiB vs 12 GiB. Insert-on-every-miss thrashes when working set >> capacity and rows pin the stacked bank until evicted (F2 contract). Operational default is **budget=0** for this model/box. If a workload needs a cache, enable the scan-resistant admission filter (`OMLX_EXPERT_STREAMING_ADMISSION=1` — only inserts experts seen >=2 times in the last 1024 accesses) and validate with `OMLX_EXPERT_STREAMING_TRACE=1` + `bench/lrc_analysis.py --cache-sizes 142 trace.jsonl` to read SCH(142); if SCH ~10-15% the cache will not pay.

## Fase E — bottleneck experiments (QD, coalescing, learned pins, MTP tuning, ANE)

All runs cold (28 GB `cache_cool`), Qwen 0G page-cache-only unless noted; run-to-run noise ~±5%.

### E1 — QD16 is the new pool default (+34%)

| QD | tok/s (48-tok decode) |
|---|---|
| 8 (old default) | 1.538 |
| **16** | **2.061** |
| 32 | 2.097 (plateau) |

Disk read max ~2.5 GB/s of the 3.7 GB/s sequential spec; short-prompt TTFT 10.4 → 8.9 s. `OMLX_EXPERT_STREAMING_QD` overrides.

**Run coalescing** (consecutive expert ids → one `pread` per bank key, `_ShardReader.expert_run`, verified bit-exact): +4% decode (adjacency ~2%), ~2% at 8k prefill (the union already covers ~the full bank). Kept — free and no-regression; `OMLX_EXPERT_STREAMING_COALESCE=0` disables.

**Long-prompt prefill finding**: 8k prompt = ~90 s TTFT streaming ~66 GB (full expert bank per layer) — but disk only sustains 1.85-1.89 GB/s of it and the GPU sits at 81-85%: **long-prompt prefill is ~60% GPU/CPU-bound** (expert QMM over ~7.4k positions + mini-bank assemble/promote). Decode immediately after a long prefill runs ~4.2 tok/s (page cache hot). The real long-prompt levers are chunked prefill / assemble cost — not bandwidth.

### E2 — MTP tuning on the QD16 stack

MTP's edge shrank from +27-37% (QD8) to +4-5% (QD16): the pool resize cut the per-forward I/O cost MTP was amortizing. Sweep (`--mtp-block`, no-MTP baseline 2.135):

| block | 2 | 3 | 4 | 8 |
|---|---|---|---|---|
| tok/s | 2.097 | 2.165 | 2.213 | 2.248 |

Pins no longer stack with deep drafting (block 8 + pins: 2.012/2.069, reproduced twice). Guidance: MTP stays default-off; when enabled use `draft_block_size` 4-8 with pins off — or QD16 alone, which beats MTP alone.

### E3 — learned pin store (persisted routing frequencies)

`OMLX_EXPERT_STREAMING_PIN_PROFILE=<path>`: PinController saves observed per-layer frequencies after pinning; the next load reloads them and wires the hot set from token 1 (no 8-call window). The mlock also populates the pages, so the reload doubles as a *targeted* warmup.

| arm | TTFT | decode |
|---|---|---|
| baseline | 8.3-8.9 s | 2.06-2.14 |
| (b) deterministic warmup 1.25 G post-load | 7.1-8.0 s | 1.71-1.93 — ~0/negative |
| (c) learned pins loaded at start | **6.9 s (−22%)** | **2.273 (+6-10%)** |

(b) confirms the prediction: warmup uncorrelated with the first request's routing is wasted I/O. (c) is the optimistic bound (profile from the same prompt); arbitrary first requests gain proportionally to overlap. Server follow-up: save on unload, reload on load, per model.

### E4 — ANE prefill: attaches to qwen4_exp, not the streaming lever

Required building the native extension (nanobind 2.13.0 ABI-matched to MLX 0.32.0; upstream `setup.py` cannot pass `-DPython_EXECUTABLE` through `CMAKE_ARGS` for paths with spaces — space-free venv symlink workaround). With it present, ANE attaches to the vendored qwen4_exp **without any port** ("48 MLP + 36 GDN procedures into 2 instance-pinned ANE programs").

Cold TTFT: 2k 25.7 → 26.3 s (no gain); 8k 89.7 → 87.1 s (~3%). Prefill is GPU-bound (81-85%) in both arms — time goes to expert QMMs and assembly, which ANE does not touch (dense MLPs + GDN projections only). Keep ANE for resident models; streaming TTFT needs chunked prefill / assemble work instead.

### E5 — GLM on the QD16 stack

GLM 0G decode 64: **0.697 tok/s vs 0.381 measured pre-E1 (+83%)** — large experts benefit more from deeper queues. TTFT 23 → 18.5 s. Pins 2.5 GiB: neutral (2.5 G = ~4-5 experts/layer of 288 — too sparse). GLM's levers remain QD16 (now default) + topk 0.85 (+27%).

## Trade-offs

- **Prefill is burst-heavy**: a long prompt touches almost every expert once → many faults on first prefill. Keep the existing **paged SSD KV cache** on — a repeated prefix is restored from disk in milliseconds instead of re-faulting experts. The first prefill still warms the LRU, so the following decode hits more.
- **Decode sync per layer**: the router's `top-k` indices are read back to the CPU to decide which experts to fault. With pread + the parallel demand loader this sync overlaps useful I/O (QD8); batch decode still multiplies the per-step working set and makes it *worse* (Fase A3), so streaming models effectively stay single-request.
- **Quantized path**: expert weight + scales + biases are restored as a co-located bundle. Fused `gate_up_proj` is handled as a single streaming bank when present; otherwise `gate/up/down` are three independent banks. The output is bit-exact versus the resident path (gather indices are remapped to a compact mini-bank).
- **GLM memory bombs are fixed but still bounded**: without the per-layer eval+`clear_cache`, decode/prefill pins 42 layers × ~3.4 GB of mini-banks and swaps the machine (40–50 GB observed).

## Measuring

- `GET /admin/api/models` returns `expert_streaming_bytes` vs `expert_resident_bytes` so you can size the budget before enabling.
- `OMLX_EXPERT_STREAMING_PROFILE=1` prints a per-layer, per-stage profile (gate_eval / unique / load hit+miss / staged vs sync split / stack / wall ms) with hit rate per layer.
- `python bench/bench_expert_streaming.py --model qwen|glm --budget N --decode M [--out file.json]` reproduces the tables above (TTFT, steady-state tok/s, cache stats, profile, prefetcher stats, phys) and samples GPU/CPU/disk/RSS per phase via `bench/resource_sampler.py` (per-phase means/max saved into the result JSON).
- `python bench/bench_expert_batch.py --model qwen|glm --budget N --concurrency K --decode M` measures batched decode (Fase A3).
- `python bench/cache_cool.py --gb N` evicts the page cache without root between runs.

## Limitations

- No learned pin store (colibri's `.coli_usage` heat) yet — the LRU is cold after a restart. A sidecar that persists routing heat per model is queued. Note: inter-token reuse is model-dependent (Qwen hits 23–32 %, GLM 0 %) — pins pay on Qwen-like routing, not GLM.
- Async prefetch (PILOT) exists behind `OMLX_EXPERT_STREAMING_PILOT=1` but is **negative by default**: on a saturated SSD it wastes bandwidth (see phase-1 A/B above).
- Dual-SSD striping is supported: copy alternating shards to a second disk (`python bench/stripe_model.py --model <dir> --target <dir2>`) and run with `OMLX_EXPERT_STREAMING_EXTRA_ROOTS=<dir2>`. Mirrored shards are served from the stripe root, the rest fall back to the primary — no RAID, original dir untouched. Note: only pays if the second disk is *fast* — in our setup the 4 TB NVMe (~3.7 GB/s) is the fast one and the 2 TB (~875 MB/s) would slow reads down.
- TTFT on GLM-class models is high (prefill faults ~all experts once); KV snapshot helps repeated prompts only.

## Fase F — long-prompt prefill: honest accounting, and what doesn't work

Investigation of the 8k-prompt TTFT bottleneck (GPU 81-85%, disk 1.85-1.9 GB/s) and the memory incidents it caused on the 48 GB dev machine.

### F1 — the transient was invisible to the guard, and the guard was double-charging

The prefill chunk forward is lazy: every MoE layer's assembled mini-bank stays live until the chunk-end eval, so one ~1.5k-token chunk commits ~32 GB of Metal transients (~17 MB/token measured; expert banks dominate). Four accounting bugs fell out, all fixed (`180f6c54`):

1. The guard's static transient model (SDPA + KV) never included the **streaming mini-bank term** — now charged as `uniq experts/layer ≈ 0.2 × chunk tokens` (measured 0.145 on qwen4_exp; saturating at experts/layer), via `backing.streaming_guard_info` attached at convert time.
2. `_current_usage_bytes` charged the **evictable file pages** of streamed experts as commitment (phys 21.5 GB while Metal active was 4.5 GB) — for streaming models only live Metal counts.
3. The per-chunk transient tracker learned the same page-cache poisoning (61 MB/token) — it now probes `mx.get_active_memory()` on streaming models.
4. Admission's `observed_max` floor assumed size-invariant transients; streaming bank transients scale with chunk size, so the floor is now discounted linearly.

The bench also gained `--min-free-gb` (abort when the machine is memory-starved — starved runs fragment prefill into many chunks, re-stream experts, and thrash) and `--mem-ceiling-gib` (propagate the watermarks the server's `ProcessMemoryEnforcer` would — the bench runs without an enforcer, so throttle/guard never engaged and the Metal pool rode to ~30 GiB).

**Residual truth**: with honest watermarks, chunk size controls peak, and peak × machine headroom caps throughput. On a machine with the user active (≈22 GB free), 8k TTFT is ~2.3-3× the idle-machine number — that's physics (compression + page-cache contention), not a bug.

### F2 — persistent bank: tested, reverted (post-mortem `170c003f`)

Two designs were measured, both reverted:

- **Promote-once** (mx bundles in the LRU): double-holds the weights — LRU (Metal) + the per-chunk stack are the same bytes twice. Metal active hit 37 GB on a 6 GiB budget run and the guard force-stopped the prefill.
- **Persistent per-linear bank** (stack surviving chunks): MLX's lazy chunk graph keeps every remount's stack live until the chunk-end eval, so the bank's Metal peak equals the per-chunk assembly it replaced — zero prefill win — plus a permanent 12 GiB+ hold between requests that starved the next prefill.

The F1 profile had already said so: per GLU call the CPU-side assembly is ~7 ms (gate 1.8 + load 4.8 + stack 0.1) against a 121 ms wall — assembly was never the lever. Kept from this round: the LRU slot-sizing fix (one slot = one projection slice = `per_expert/3`; a 12 GiB budget previously bought ~4 GiB of bank, hit rate 0.002).

### F3 — SpecPrefill and streaming are anti-synergistic (`70e348e3`)

SpecPrefill (draft-scored sparse prefill, keep 0.2) saves target *compute*; streaming's cost is expert *I/O per routing neighborhood*. Measured on Qwen 2k, draft Qwen3.5-2B-bf16:

| | TTFT | LRU misses | uniq experts/call |
|---|---|---|---|
| streaming only | 25.7 s | ~34k | ~34 |
| + SpecPrefill (cold) | 93.8 s | 466k | — |
| + SpecPrefill (warm) | 26.3 s | 57k | 132 |

Sparse selected tokens spread routing across the bank (4× unique experts per forward), destroying the locality dense chunks rely on; the QMM saving is eaten by expert loading. **Do not combine** — SpecPrefill is a resident-model lever.

### Where this leaves long-prompt TTFT

- The levers that survived: QD16, coalesced runs, learned pins (E3), and the honest chunk watermarks (F1) that trade chunk size for machine safety under pressure.
- The remaining prefill cost is the expert QMM itself and the one full-bank sweep — chunked prefill sizing amortizes assembly but cannot remove compute; only fewer/cheaper expert-token pairs can (none of the tested accelerators — ANE, SpecPrefill, MTP at prefill — reduce it for streaming).
- DeepSeek V4's per-layer `mx.eval` + `clear_cache` pattern (already in its loader) is the known-good answer for the lazy-graph accumulation; a qwen4_exp equivalent would bound intra-chunk peaks if ever needed.

### Post-F machine-state reality check (shared 48 GiB box, 2026-08)

Remeasuring the 8k idle TTFT after F1 on a machine whose owner was actively using it (`memory_pressure -Q` 88-90 % free, but **psutil available 21.9 GB**) — `qwen8k_f_idle.json`, budget 0, ceiling 28 GiB:

- TTFT **341.1 s** (vs the E4-era idle ~87 s and the F1 busy 296.9 s), decode **0.303 tok/s**. The prefill read **~546 GB from disk** — a full 66 GiB bank sweep re-executed ~8× — because the MLX allocator cache rode to 30.4 GiB under the honest watermark and evicted the page cache the run itself depends on; every later chunk then re-streamed the full bank from SSD.
- **Decode kept the squeeze**: MLX cache was still holding ~16 GiB during decode (the periodic clear gate — every 512 steps *and* above a threshold — never fires for a 96-token decode), so budget-0 decode was fully disk-bound. Candidate lever: `_sync_and_clear_cache()` (or a `get_cache_memory()`-aware throttle probe) at the prefill→decode boundary.
- **Two preflight lessons.** (1) `memory_pressure -Q` free-% is the wrong gate for streaming runs — it counts ~24 GiB of inactive page cache as free that the run can't actually commit without evicting itself; the bench's psutil-based `--min-free-gb` (default 22) is the right signal, and overriding it to 20 is what let the starved run through. (2) The ceiling default (28 GiB) presumes an idle machine: at 21.9 GB available, a 2k prompt fits a *single* ~34 GiB transient chunk under that watermark and wired memory hit 38 GiB into swap. Size `--mem-ceiling-gib` to the machine's *available*, not its capacity.
- **The E4-era ~87 s is not the post-F1 target — and ~40 GB available is not a realistic condition.** The 87 s was measured *pre*-F1 (no watermarks engaged: fixed 2048-token chunks); under honest watermarks even a 40 GB-available machine would chunk smaller and sweep more, so no settings knob buys it back. The realistic state of the shared box is ~22 GB available, and the honest post-F1 8k number there is the 341 s datapoint above.
- **KV is not what blows the envelope on these checkpoints.** qwen4_exp is a hybrid: 36×GDN layers carry O(1) recurrent state (not per-token) and only 12×full-attention layers keep KV — 2 KV heads × head_dim 256 bf16 ≈ 2 KB/token → **~0.17 GiB at 8k**. dsv4 uses MLA (1 KV head × 512 latent ≈ 1 KB/token/layer) → **~0.3 GiB at 8k**. The long-context memory that matters is (a) the expert mini-bank transients (~17 MB/token, charged by the guard) and (b) the **unreleased MLX allocator pool** (30.4 GiB above) — which is a code lever (release between chunks / at the prefill→decode boundary, the DeepSeek per-layer `eval+clear` pattern generalized), not a settings knob.

## Fase G — pool release, kernel readahead, hotness seeding (after a ds4 study)

Studied antirez/ds4 (Metal SSD streaming for the same checkpoint class). It independently reaches the same architecture conclusions as our E/F phases — pread + parallel demand loads, single-size-class slab (= F2's `per_slot = per_expert//3`), cache budget sized from the backend's recommended working set minus context (= F1's "size to available"), and the "short token-major prefill spends most of its time in VM/driver synchronization" wall we hit as page-cache eviction. Three of its ideas were adopted:

### G1 — off-boundary MLX pool release (`844cb4dc`)

`_should_release_streaming_pool()` fires for streaming models (guard info resolved from the backing) whenever `mx.get_cache_memory()` crosses `max(memory_limit/3, 2 GiB)`, off the 512-step periodic boundary — hooked into the chunked prefill chunk tail and the `step()` tail. **Measured reality on the bench's external prefill path: a no-op.** The release-arm log shows the tail sees pool ≈ 0 GiB: the ~29 GiB pool peaks are *intra*-chunk/step allocator high-water (freed lazy-graph intermediates, trimmed by MLX's own cache limit before the tail looks). Where it does engage: the chunked path's chunk tail holds the just-freed chunk — that is the multi-request server case. Bounding the intra-step churn itself is the per-layer `mx.eval + clear_cache` pattern (G4, not built for qwen4_exp).

### G2 — F_RDADVISE kernel readahead (`c33d820`)

`_ShardReader.advise_range()` issues `fcntl F_RDADVISE` (Darwin cmd 44 — not exported by Python's fcntl module; radvisory packed `=qi4x`, best-effort) and `ExpertBackingStore.advise_expert_run()` collapses contiguous expert ids into one advisory per run (row-major banks make runs contiguous). Rides the warmer's next-layer previous-token prediction flow in advise-only mode — the kernel pulls the predicted ranges into the page cache with zero userspace copy (`OMLX_EXPERT_STREAMING_RA`, default on, `=0` disables). Prefill keeps plain pread (prediction is useless there — F3; ds4 defaults the same way). Effect at 512/2k scale: within run noise.

### G3 — prefill-hotness cache seed (`c33d820`)

ds4's cache seeding. `PrefillHotnessRecorder` accumulates per-layer expert frequency during prefill-sized calls (decode rows excluded), then at the first decode-sized call swaps the cache to the prompt-wide hot set:

- budget > 0: `ExpertLRUCache.retain_hot()` evicts everything outside the hot set and missing hot bundles load from the backing — attacks F2's cold-start (hit_rate 0.002: the prefill demand path fills the LRU with the *last* chunks' experts, then decode misses).
- budget 0 (default): a bounded discarded-read burst into the page cache (`OMLX_EXPERT_STREAMING_SEED_GIB`, 2 GiB cap), async on the warm pool.

Measured live on qwen 512: 720 slices seeded at the prefill→decode boundary. `OMLX_EXPERT_STREAMING_SEED=0` disables.

### Machine honesty for the G-series benches

Two swap incidents during this phase (user visible). The bench runs at ceiling 20 GiB with ~24 GB available still over-commit: the intra-step pool peaks (~29 GiB wired) ride above any tail watermark, so on this shared box the G-series A/B numbers are dominated by machine state, not code — identical 2k workloads measured TTFT 118.3 / 96.8 / 112.2 s across the session as available memory moved 21.9 → 27.5 GB. Real lever A/B needs either an idle window or G4 first. Survival notes: preflight on psutil available (not `memory_pressure` free-%), ceiling sized to available, and treat the first `memory_pressure` complaint as a stop signal.

## Fase H — Autotune (per-machine parameter profiles)

`bench/autotune_expert_streaming.py` turns the hand-run bench A/Bs above into an automated, safety-railed search. It exists because the G-series showed this machine's numbers are dominated by memory state: the only honest way to pick parameters is a tool that (a) measures the machine first, (b) refuses to run when the box is loaded, and (c) never pushes it into swap.

**What it tunes.** The streaming knobs that measured meaningful deltas in E–G: `expert_streaming_budget_gib`, IO depth (QD), run coalescing, F_RDADVISE readahead, prefill-hotness seed, and the PILOT prefetcher. `topk` is opt-in (`--sweep-topk`) because it trades output fidelity.

**How a session runs** (~1.5–2 h for the default shape):

1. **Probe** (no model load): RAM/available/swap; the enforcer's static + Metal ceilings for the configured tier; sequential and random-expert-size read bandwidth on the model's own shard. A near-saturated random probe prunes the QD sweep.
2. **Calibration** (discarded): default config at the screening context (2k/32 tokens) — warms the page cache and measures the loaded runtime footprint used by every later preflight.
3. **Screening**: one-factor-at-a-time trials, each scored against the calibration reference (TTFT 50% + decode tok/s 50%, minus a penalty for any observed swap growth). Budget candidates are filtered by the memory actually left after load + reserve.
4. **Head-to-head + validation**: the screening winner reruns against the default, then both run at the long context (8k/96) — a winner that regresses there is not recommended.
5. **Recommendation**: `bench/results/autotune/<model>_<stamp>/recommendation.json` with every trial row, the winning config, and machine probe numbers.

**Memory safety is the design constraint, not a feature:**

- Per-trial ceiling = `min(static/metal cap, available − reserve)` — sized to the machine's *available* memory, never its capacity (the F1 lesson).
- A watchdog thread samples the trial process every 2 s: swap growth > 2 GiB (immediate) or available < 5 GiB for 2 consecutive samples → SIGKILL the trial, record a safe-failure, and raise the reserve for the remaining trials.
- Each trial is skipped (not failed) when available memory can't hold the loaded runtime + reserve + margin; the session aborts after two consecutive watchdog kills or if the machine fails to drain between trials.
- Nothing runs on its own — a human launches the tuner; `--dry-run` prints the probe and the trial plan without loading the model.

**Where the result lives.** `--apply` writes the winning knobs into the model's per-model settings (`ModelSettingsManager` → `~/.omlx/model_settings.json`) — the same configuration the app's model settings UI edits. The IO knobs are real per-model settings since Fase H (`expert_streaming_io_depth`, `_coalesce`, `_readahead`, `_seed`, `_pilot`; unset = env/default behavior), part of the engine runtime signature (changing one reloads the engine), and excluded from cross-model profile templates like the other hardware-specific streaming fields. Apply with the server stopped (or reload the model afterwards): a running server keeps its own in-memory settings manager and would overwrite the file on its next save.

```bash
# preview the machine profile and trial plan (no model load)
.venv/bin/python bench/autotune_expert_streaming.py --model qwen --dry-run
# full session; artifacts under bench/results/autotune/
.venv/bin/python bench/autotune_expert_streaming.py --model qwen
# persist the winner as the model's configuration
.venv/bin/python bench/autotune_expert_streaming.py --model qwen --apply
```

Machine-probe datapoint (2026-08-29, shared 48 GiB box under active use): static ceiling 42 GiB (balanced tier), Metal cap 37.4 GiB, SSD sequential ~16 GB/s with the bank's shards warm in page cache (random QD1 ~15 GB/s — the QD sweep was pruned as near-saturated; cold-cache random bandwidth is lower, so the prune is conservative for cached servers). The tuner refused a real session at 21.5 GB available, as designed.

## Fase I — qwen prefill eval boundary (G4)

Follow-up to the G series, prioritized by the paper survey
([expert-streaming-papers.md](expert-streaming-papers.md)). The qwen4_exp finding:
the installed `mlx_vlm` decoder ignored the converter's `_stream_eval` flag — only
the vendored GLM decoder honors it — so long prefill chunks accumulated one
streaming mini-bank per layer in the lazy graph until the chunk-end eval
(~17 MB/token, intra-chunk pool peaks ~29 GiB: the Fase G "machine honesty" root
cause), and the retained allocator pool could grow big enough to evict the page
cache the run itself depends on (the post-F 341 s/8k case).

### I1 — per-layer eval boundary for qwen4_exp (bit-exact)

`patches/expert_streaming/qwen35_stream_eval.py` wraps the installed
`Qwen3_5MoeDecoderLayer.__call__` (same class-patch mechanism as the adaptive
top-k patch): when the layer carries `_stream_eval`, the call is prefill-shaped
(`x.shape[1] > 1`; decode is `[B, 1, H]`) and not an MTP verify pass
(`target_verify`), the layer output is `mx.eval`'d and `mx.clear_cache()` trims
the allocator cache — the DeepSeek/GLM per-layer pattern. Bit-exact by
construction (`mx.eval` materializes what the next layer reads anyway). Decode
and verify stay lazy: 48 forced syncs/token would erode the QD16 win and their
graphs are small.

- Knob: `expert_streaming_per_layer_eval` (`None` = env
  `OMLX_EXPERT_STREAMING_PER_LAYER_EVAL`, default **on**) — a runtime-signature
  knob like the other IO overrides (a change rebuilds the engine); excluded from
  profiles. Toggle exposed in the WebUI Expert Streaming card and in the macOS
  app's Model Settings → Advanced (a stored null renders as on, the built-in
  default).
- GLM/DeepSeek decoders honor the boundary natively and are unaffected by the knob.
- Tests: boundary gating (prefill fires; decode / verify / un-flagged layer /
  knob-off skip), idempotent wrap, settings round-trip + profile exclusion +
  API persist.
- **Measured (idle window, Qwen 2k prompt, 96-token decode, warm page cache —
  the Fase G confound controlled by repeating the OFF arm after the ON arm)**:
  TTFT 119.2 s (off) vs 120.9 s (on) — a tie; the apparent 211 s → 121 s
  improvement in the first cold-cache run was page-cache warming, not the
  boundary. Decode: **0.405 (off) vs 0.580 tok/s (on) — the boundary is
  +43% decode**, releasing each layer's expert mini-bank from the lazy graph
  instead of accumulating 48 layers × uniq × 2.7 MB per chunk. Default **on**
  confirmed; artifacts `bench/results/qwen_2k_pleval_{off,on,off2}.json`.

### I2 — learned pin store server integration (E3 follow-up)

`PinController` takes a per-model profile path now:
`<model>/.omlx/expert_pin_profile.json` (env `OMLX_EXPERT_STREAMING_PIN_PROFILE`
stays the bench override and wins when set). Loaded at convert — the hot set is
wired from token 1 — and saved on engine `stop()` while the backing store is
still reachable (`save_expert_pin_profile`; BatchedEngine + VLM wrapper).
Settings `expert_streaming_pins` (None = env `OMLX_EXPERT_STREAMING_PIN`,
default off) + `expert_streaming_pin_gib` (None = env
`OMLX_EXPERT_STREAMING_PIN_GIB`, default 1.25): runtime-signature governed,
profile-excluded, toggles in both UIs. mlock only — zero output change.
Measured in E3: +6–10% decode, −22% TTFT with a same-prompt profile.

### I3 — routing trace + LRC analysis (SRP/SCH)

`OMLX_EXPERT_STREAMING_TRACE=<path>` appends one JSONL row per MoE layer call
(`{call, layer, positions, uniq}`); `bench/lrc_analysis.py` computes the
routing-consistency metrics of arXiv:2505.16056 — SCH (Belady oracle-cache hit
rate per cache size; the paper's ≈2× active-experts sweet spot is directly
sweepable) and SRP (fixed-group coverage per segment, demand-weighted and
distinct). Purpose: per-model defaults for pins/seed/top-k and a pre-flight
"does streaming pay" check. Offline only — no UI.

**Measured (Pride and Prejudice, streaming engine):**

- Qwen, continuous decode (96 tokens, short prompt; the bench workload):
  SCH(S=8..128) = 33.9 / 44.8 / 54.3 / 64.0 / 73.6%; the real LRU cache in
  the same run measured 26.8% hit rate — inside the known 23–32% reuse
  band, with Belady's ceiling ~2.7× above the LRU at the sweet spot.
- Same protocol on both models (24 disjoint 64-token windows — the
  cross-document regime a multi-user server actually sees):
  GLM SCH(S=128) = **76.0%**, Qwen = **73.2%** — essentially identical.

The old "GLM 0% inter-token reuse" calibration does NOT reproduce in the
disjoint-window regime: it was an artifact of single-text continuous decode.
Cross-document routing is highly reusable for both models — a pinned/cacheable
expert set can pay on GLM too, not only on Qwen. The remaining regime
difference is within one document (decode), where Qwen keeps reuse (SCH ~34–74%)
and GLM's continuous-decode reuse stays to be re-measured with this harness
before any GLM pin-budget default changes.

**Measured (JANG 4S/4M, 2026-09-04 — `bench/results/lrc/`):** same frozen
protocol, budget 0, short prompt, regimes split by `positions` (decode ≤ 64
rows). Both quants agree to within 0.3pp — routing is a property of the model
(512 experts, top-10), not of the quant, matching the 2x3090 Part-2 finding
("hit rate doesn't depend on the quant, only on slot count"):

- Decode SCH ceiling **77.5–77.8%** from S=64/layer (S=16 already 65%, S=32
  74%, S=128+ flat) — the knee is 16, everything saturates by 64 slots.
- Adjacent-call repeat 38.5%/39.3% (4S/4M) — the documented ~35% band, on the
  trace's own rows. Working set ~104 distinct experts/layer per segment.
- SRP(G=64) demand coverage 89.6–89.8% — a fixed 64-expert group/layer covers
  ~90% of decode demand; top-10 experts alone hold 40–41% of demand.
- Prefill: SCH ceiling **50%** — the prefill call is a near-broadcast union
  (~199 uniq/call at 570 positions); caching prefill is structurally capped,
  the seeded page-cache burst is the right mechanism (and it shows: TTFT 9-11s
  for a 58-token prompt with 1008 slices seeded).

Actionable reading (updated 2026-09-04 by the pin-knee matrix,
`bench/results/lrc/matrix/`): the SCH knee predicted a pin win, and the matrix
refuted it — 16-slot/layer pins (4S 1.5 GiB, 4M 2.0 GiB, 3 interleaved reps,
own v2 profiles) measured **null (−0.3%)** on both. The L2 finding reproduces
on the JANGs: on this box the page cache already serves the decode working
set up to the oracle ceiling, so mlock only trades evictable for wired. The
SCH curve is the sizing answer for **device-side residency (slot-bank)** —
how many slots a slot-bank must hold per layer to capture ~78% of the hit
ceiling (16 at the knee, 64 at saturation) — not a page-pin budget. Pin
profiles do NOT transfer 4S↔4M: the fingerprint gate (config_sha + packing
`oQ4e3b` vs `oQ4e4b`) rejects cross-model profiles by design.

### I4 — perplexity harness

`bench/ppl_expert_streaming.py`: token NLL / perplexity over a local corpus
via mlx_lm's resident path (disjoint ctx-token windows, context-only first
position). Streaming compute is bit-exact versus resident (test-pinned), so
the resident measurement represents the streamed path. This is the quality
gate for I5: compare the oQ4e checkpoint against its cold-tier variant on the
same corpus before trusting the tier.

`--streaming` loads through the omlx streaming engine instead — for the
production checkpoints (GLM 190 GB, Qwen 99 GB) the expert banks far exceed
RAM, so the resident path cannot load them at all. `--cold-tier none|2|3`
selects the arm; `--ctx 1000` is used for the GLM gate (see the affine-tile
bug below for why not 1024).

**Affine-tile race at T >= 1024 (found by this harness).** The first long
streaming run read ppl ~28k: uniform logits, NLL ≈ ln(vocab). A causal
differential (hidden states at positions 0..511 must be bit-identical between
a 512- and a 1024-token forward) traced the divergence to layer 0 of
glm5_next — before any MoE — where the `gated_delta_update` kernel input `a`
(f_b_proj output) diverged while q/k/v/b stayed bit-identical. The chain of
evidence: the real checkpoint is q8 (not the config-default q4), the q8
affine tile only routes at T >= 1024 (`min_tokens = 1024`), and a same-run
replay of the exact captured `fa_o` through the same `f_b_proj` weights
disagreed with what the live forward produced — but the replay was clean on
the next run, and isolated tile calls (dozens of shapes, real weights
included) never diverge. Verdict: an intermittent GPU-side race inside the
custom tile at T >= 1024, not deterministic wrong math. The production chat
path is unaffected — the scheduler's streaming bank guard steps prefill
chunks down to <= 512 tokens, and a 2.5k-token GLM chat completion verified
coherent — but any direct >= 1024-token forward (raw evals) intermittently
produced garbage. Guard in `patches/mlx_vlm_glm5_next_compat/.../glm5_next/
linear.py` (`_tile_corrupts_at_long_prefill`): the tile is blocked entirely
at T >= 1024 (falls back to `mx.quantized_matmul`); below 1024 the tile still
routes (q8 indexer pinned by test at T=1023). After the guard, ctx-1024
streaming ppl reads 3.32 (2 windows) versus 28,323 before. The underlying
tile race remains open — the guard is containment.

### I5 — cold precision tier (uniform)

`tools/requant_cold_tier.py` writes `<model>/expert_cold/`: the full switch_mlp
expert set requantized at `--bits 3` (or 2) with the source group size, same
shard filenames / key names, packing recorded in the shard `__metadata__`
(`omlx_cold_bits` / `omlx_cold_group_size`). Only affine banks with a
`.biases` key convert — the affine bias must ride along or the runtime's
dequantize reconstructs shifted values.

Runtime: `expert_streaming_cold_tier` ("2"/"3"; None = off) makes the backing
resolve expert-bank keys from `expert_cold/` first
(`ExpertBackingStore(cold_root=...)`) — slices, coalesced runs, pins, F_RDADVISE
and dtype reads all funnel through the same reader choke point — and the
converter overrides the streaming linears' bits/group size with the tier's
recorded packing, so the single gather_qmm per projection stays uniform.
Every expert reads the tier; the per-expert hot(4-bit)/cold(low) split from
HOBBIT is the recorded follow-up once a quality verdict exists. Bytes per
token drop 25% (3-bit) or ~50% (2-bit) — the direct lever on the I/O floor
that caps GLM decode.

- Partial tiers are refused (`cold_tier_status`): the uniform-packing
  assumption would silently break. The admin capability flag
  `expert_streaming_cold_tier_present` gates the UI input (both UIs + i18n).
- Runtime-signature governed; excluded from profiles.
- Tiers generated (3-bit, same group size as source): GLM-5.3-Flash-oQ4e
  141.8 → 106.3 GiB (0.75×, 126 banks / 33 shards, requant err ≤ ~0.03
  typical), Qwen3.8-Flash-Next 57.4 → 43.1 GiB (0.75×). 2-bit halves the
  4-bit bytes instead if the 3-bit quality delta proves too small to ship.
- **Quality gate measured (Pride and Prejudice, streaming engine)**:
  - GLM (24 × 1000-token windows, 23,976 tokens): oQ4e ppl **4.381** vs
    cold-3bit **5.435** — **+24.0% ppl / +0.216 NLL per token**.
  - Qwen (12 × 2048-token windows, 24,564 tokens): oQ4e ppl **1.312** vs
    cold-3bit **1.372** — **+4.6% ppl / +0.045 NLL per token**.
  Verdict: the uniform 3-bit tier is far from "near-lossless" on GLM (13 MB
  experts, 288/layer) and borderline on Qwen (2.7 MB experts, 512/layer) —
  the smaller-per-expert, wider-routing model tolerates it much better. As a
  default it is rejected for GLM; for Qwen it stays opt-in pending the
  tok/s A/B that decides whether 4.6% ppl buys a real decode win. The
  per-expert HOBBIT hot/cold split (only truly-cold experts get the low
  tier) is the recorded path to shrink both deltas.
- **tok/s / TTFT A/B measured (idle window, short prompt, 48-token decode;
  `--cold-tier` added to the bench)**:
  - GLM: base 0.452–0.479 vs cold-3bit 0.572–0.602 tok/s across two
    repetitions — **+26–33% decode**, TTFT 20.3 → 19.0 s. The 13 MB-expert
    I/O floor is exactly where the 25% byte cut pays.
  - Qwen: base 1.436 vs cold-3bit 1.575 tok/s — **+9.7%**, TTFT 10.8 → 9.5 s.
  Decision: cold tier stays **opt-in** everywhere (GLM +24% ppl / Qwen +4.6%
  ppl are real costs); the GLM decode win is large enough that GLM users on
  the I/O floor who accept the quality hit get a real option. Artifacts
  `bench/results/{qwen,glm}_i5_{base,cold3}*.json`.
- **G2/PILOT re-test with the tier (idle window)**: prefetch stays negative
  in the new byte regime — GLM cold-3bit + PILOT measured 0.370 tok/s vs
  0.572–0.602 without (−35%; `staged_dropped` 94% of 77.5k staged bundles).
  The saturated-NVMe verdict of Fase E/G holds: concurrent prefetch cannot
  buy what the disk cannot serve. Closed — do not revisit without a
  different disk. Artifact `bench/results/glm_i5_cold3_pilot.json`.

### I6 — HOBBIT per-expert hot/cold split

I5 made the cold tier uniform: EVERY expert reads `expert_cold/`, quality
be damned (+24% ppl on GLM at the time). HOBBIT's insight (from the paper's
hybrid offloading) is that routing is extremely skewed — a small set of
experts carries most of the token mass — so only the truly-cold experts need
the low-precision tier while the hot ones keep the source 4-bit packing.

**The hotness signal was wrong (discovery).** The learned pin profile ranked
experts by PRESENCE PER PLAN: `PinController.on_layer_plan` counted each
unique expert once per routing plan, so every recorded frequency was
identical (GLM profile: 42 layers × 64 entries, every count == 4). Worse,
`_PIN_PROFILE_KEEP=64` truncated the record list, so on a 288-expert model
`ceil(0.25 × 64) == 16` selected ids 0..15 — an arbitrary id prefix, not a
hot set. Both defects are fixed in I6:

- **Per-token usage counts**: the streaming switch computes
  `np.bincount(plan.flat_np, minlength=num_experts)` per layer call and
  passes it through `WarmPinHook.on_layer_plan` (new `counts` argument;
  `wants_usage_counts` gates the bincount so the readahead warmer never
  pays it). `PinController`/`PrefillHotnessRecorder` accept `counts=None`
  and fall back to the presence signal for old callers.
- **Profile coverage**: `_PIN_PROFILE_KEEP` is env-tunable
  (`OMLX_EXPERT_STREAMING_PIN_KEEP`, default **512** — the full 288-expert
  width fits; JSON format unchanged).
- **Real fraction denominator**: `load_hot_set_from_profile(path, fraction,
  num_experts)` computes `ceil(fraction × num_experts)` from the model
  estimate (288 for GLM), clamped to the available records; the old code
  used the record count, which the keep cap truncated.
- **Correct layer log**: the split log reports `42/42` layers
  (`estimate.num_moe_layers`), not the profile's layer count.

The regenerated GLM profile (24 × 1000-token windows, 23,976 tokens, pins
enabled, same corpus) now has 288 entries per layer with genuinely skewed
counts (layer 3: top expert 1,474 uses, median ~640, min 96 — 2,237 distinct
count values, not one).

**Runtime**: `expert_streaming_hot_fraction` (0–1, opt-in like the tier
itself; None/unset = uniform I5 — the env `OMLX_EXPERT_STREAMING_HOT_FRACTION`
is the bench override). With a cold tier active and a learned profile
present, the converter builds a dual-tier path per layer: hot experts keep
the SOURCE packing (4-bit oQ4e), everyone else reads `expert_cold/` (3-bit);
the backing (`set_hot_experts`) routes reads and the linear builds one
mini-bank per tier with a masked dual `gather_qmm` (positions are mutually
exclusive per tier; negative gather indices clamp to 0 with the -1 kept for
the keep mask — `gather_qmm` takes unsigned rows and nan × 0 stays nan).
The LRU keys are tier-suffixed (`bundle_key` appends `#c` for cold) so the
two packings of one expert never alias, and coalesced pread runs
(`_group_runs`) break at tier boundaries because a run reads ONE backing
reader resolved by its first id.

Three real-bugs-found-by-the-gate notes: (1) `io_ov` was resolved after the
backing block that needed it (RAM-dict fallback on every cold-tier load);
(2) `set_hobbit_split` was wired only in the fused-projection branch — GLM
uses split gate/up/down, so the linear stayed uniform while the backing
split, mixing packings in one mini-bank; (3) the raw-pread staging path put
unsuffixed LRU keys, serving cold bundles to hot slots. All three reproduced
as `mx.stack` shape errors or silent RAM fallback and are fixed + tested.

**Quality gate re-measured with the current harness** (the I5 numbers above
came from the pre-affine-guard era; streaming compute is bit-exact, so the
base/cold numbers were re-run for a fair comparison. GLM, 24 × 1000-token
windows, 23,976 tokens, budget 2 GiB):

| arm | ppl | Δ ppl vs base |
|---|---|---|
| base (oQ4e 4-bit) | **2.2225** | — |
| cold-uniform (3-bit) | 2.5552 | **+14.96%** |
| HOBBIT 0.25 | **2.2505** | **+1.26%** |

The split recovers ~92% of the uniform tier's quality penalty while keeping
most of its byte savings (75% of experts still read 3-bit). Artifacts
`bench/results/ppl_runs/{glm_base_i6, glm_cold3_i6, glm_hobbit25,
glm_profile_regen}.json`; profile backup
`expert_pin_profile.json.pre-i6.bak` beside the regenerated one.

**tok/s / TTFT A/B (idle window, short prompt, 48-token decode, pins off,
`--min-free-gb 22 --mem-ceiling-gib 21`; `--hot-fraction` added to the
bench)**:

| arm | tok/s (rep 1 / rep 2) | TTFT |
|---|---|---|
| base | 0.582 / 0.575 | 18.7 / 19.0 s |
| cold-uniform 3-bit | **0.708 / 0.748** | 15.9 / 16.7 s |
| HOBBIT 0.25 | **0.647 / 0.625** | 17.5 / 17.5 s |

Decode: cold-uniform **+22–30%** vs base; HOBBIT **+9–11%** vs base and
~84–88% of the uniform tier's speed — the hot experts ride 4-bit bytes on
the routed 25% of experts. Decision: the quality/speed Pareto point is
model-dependent — HOBBIT 0.25 gives GLM users the near-lossless option
(+1.3% ppl for +9–11% decode) while the uniform tier stays for maximum
speed. Fração 0.5 remains the recorded follow-up if the ppl headroom shows
up elsewhere (Qwen). Artifacts `bench/results/i6/glm_i6_{base,cold3,
hobbit25}.json` (rep 2).

### I6b — Post-I6 campaign: fraction sweep, Qwen, cold 2-bit, and the
### profile caveats

**GLM fraction sweep** (same harness as the I6 gate: 24 × 1000-token
windows, 3-bit cold tier, prefill-learned profile):

| fraction | hot/layer | ppl | Δ ppl vs base | penalty recovered |
|---|---|---|---|---|
| 0.125 | 36 | 2.3072 | +3.81% | 74.5% |
| **0.25** | **72** | **2.2505** | **+1.26%** | **91.6%** |
| 0.5 | 144 | 2.2317 | +0.42% | 96.7% |

0.25 is the knee: below it the penalty triples, above it the byte savings
shrink while quality is already ~97% recovered. Artifacts
`bench/results/ppl_runs/glm_hobbit{125,50}.json`.

**Qwen3.8-Flash-Next (48 layers, 512 experts, 10/tok)** — the skew is
flatter than GLM's (top-10 experts carry 17% of mass, 375 distinct count
values over 245,760 routed tokens), yet HOBBIT is even more effective:

| arm | ppl | Δ ppl vs base | tok/s (48-tok decode) | TTFT |
|---|---|---|---|---|
| base (oQ4e 4-bit) | 1.3119 | — | 1.622 | 10.7 s |
| cold3-uniform | 1.3718 | +4.57% | 2.068 (+27%) | 8.5 s |
| **HOBBIT 0.25 @ 3-bit** | **1.3147** | **+0.21%** | **1.804 (+11%)** | 8.4 s |
| cold2-uniform | 1.6998 | +29.6% | 2.322 (+43%) | — |
| **HOBBIT 0.25 @ 2-bit** | **1.3246** | **+0.97%** | **1.955 (+21%)** | — |

Two findings. (1) The 3-bit split is near-lossless on Qwen (+0.21% ppl,
95.4% of the tier's penalty recovered) — the cold tier + HOBBIT 0.25 is a
defensible Qwen default, not just an opt-in. (2) The **2-bit tier under
HOBBIT** turns a catastrophic uniform tier (+29.6% ppl) into a usable
extreme-speed point: +0.97% ppl for +21% decode vs base. The 2-bit tier
was generated (`tools/requant_cold_tier.py --bits 2`, 57.4 → 28.7 GiB of
expert banks, 0.50×) for this measurement and then parked
(`expert_cold_2bit_measured/`) with the 3-bit tier restored as
`expert_cold/` — swapping is a directory rename since the backing resolves
`<model>/expert_cold`. Artifacts `bench/results/ppl_runs/qwen_*.json`,
`bench/results/i6/qwen_i6_*.json`.

**Profile caveats (measured, not assumed).**

- **Regime mismatch**: a decode-dominant run (48-token decode, 18-token
  prompt, pins on, clean profile) learns hot sets that share only **35%**
  of their top-72 experts with the prefill-learned profile (overlap@10 =
  9%, top-1 expert identical in 1/42 layers); the prefill hot set covers
  ~40% of decode token mass. Caveat: the decode profile carries few tokens
  (336 routed tokens on layer 3 vs 192,000 for prefill), so treat this as
  directional — but the two regimes clearly diverge. A profile learned in
  long-prefill sessions under-serves decode-heavy chat; a per-regime
  profile (or a decode-weighted update) is the recorded follow-up.
- **Prefill vs decode tok/s**: pins themselves cost decode speed on GLM
  (0.522–0.526 tok/s with pins + pin warm traffic vs 0.575–0.582 pins
  off) — the mlock pass and seeding add ~10% decode overhead in the
  bench's cold-cache regime; production benefit comes from the WARM next
  load, not the current one.

**Bench parity**: `bench_expert_streaming.py --pins` now matches the ppl
harness (mlock the observed hot experts at 1.25 GiB, persist the learned
profile on unload) so decode-dominant profiles can be harvested from the
tok/s bench. The readahead warmer's `F_RDADVISE` runs also break at
HOBBIT tier boundaries now (`advise_expert_run` segments per resolved
reader), matching the demand path's run-grouping rule — a straddling run
previously advised one reader's byte range for both tiers' experts.

**Domain sensitivity (code corpus vs book)**. A 500 KB corpus of the
project's own .py/.swift/.ts sources (`bench/corpus/omlx_code.txt`) vs
pg1342: code is an EASIER domain for the split — HOBBIT 0.25 costs
**+0.39%** ppl on code (2.8072 vs 2.7963 base) versus +1.26% on the book
corpus, consistent with routing being more skewed on code (top expert
2,088 of 192,000 tokens on layer 3 vs 1,474 on the book). But the hot
sets are domain-specific: book×code top-72 overlap is only **22.4%**
(min 8.3%, max 33.3%; top-1 expert identical in 1/42 layers) — even lower
than the prefill×decode overlap. The cross-domain run (book-learned
profile applied to the code corpus) confirms the caveat quantitatively:
**+2.22%** ppl (2.8585) vs **+0.39%** with the code-learned profile — a
mismatched profile costs ~5× the penalty of a matched one (still far
below the uniform tier's +15%, but 3/4 of its hot set is wrong).
Practical rule: learn the profile on the workload you serve; a profile
imported from another domain leaves ~3/4 of the hot set cold. Artifacts
`bench/results/ppl_runs/glm_code_{regen, hobbit25, hobbit25_bookprofile}.json`.

## Fase K — I/O concurrency on expert streaming (branch feature/expert-streaming)

Fase K merges the two halves of the c1be4b98 divergence: the faseJ
I/O-concurrency pipeline (rolling layer-context prefetch, per-run read
parallelism, single-promotion banks, throttled pool clears, the eval boundary
for the vendored qwen4_exp decoder, the 3x-over-measured static cap) onto the
HOBBIT dual-tier + I6b ports. All Fase K timing-only changes are bit-exact
and the gate now proves it at the TOKEN-ID level (K8: 48/48 ids identical
across the pipeline, legacy and depth-1 arms, bench/results/fasek/tokgate2/,
every sidecar at bit_exact_kind=tokens); the throttle cap (F4) is a
re-base and rewrites the bench's chunk_schedule reference together.

### Corrections from the static review (K1-K8, signed off on this branch)

- K1/K7 (8cab5436): the O2 speculation state (stash ring, routing history,
  linears registry, advise stats, pending futures) moved from module
  globals into SpeculationState, one instance per conversion; backing.
  close() drains speculation workers with the readers; all ring mutations
  sit under one lock and the pending queue is bounded (speculation never
  blocks demand). Two engines with identical tensor keys can no longer
  share ring bytes.
- K3 (f31d0b8b + 1f998e1b): a restored prior is NOT measured signal —
  load_prior zeroes the raw deltas and _predicted_chunk_transient gates
  its measured MAX on tracker.samples > 0, so the first chunk prices the
  static estimate until the first real update() replaces the prior. The
  reclaim ledger stays independent (it charges releases, not samples).
- K2 (5644c2b3): one shared segmentation (segment_runs) for the demand
  planner, the stash and the advisor — contiguous ids within one reader;
  sparse stash speculation reads exact keys (the old reader-only segment-
  ation stored hole experts).
- K5 (2e16f40f): the F7 run-gap bridge reaches the C2 read path
  (read_expert_into, merge_gap); the scatter writes ONLY the demanded
  ids at (eid - first) rows. split4/ later measured the bridge as a net
  TTFT loss in both regimes, so RUN_MERGE_GAP now defaults to 0 (the
  knob stays for slower backends).
- K4 (dc980cc5): the PREFILL_QD regime pool now serves the rolling
  prefetch and union paths, not just the legacy fallback call.
- K6 (e4b402f9): tier-aware bank bytes under HOBBIT — the prefetch and
  bank caps measure hot segments at the source packing width.
- K8 (5fb3c23c): vlm engines propagate output_token_ids on
  GenerationOutput.tokens; the bench requires non-empty token lists
  under --gate-tokens instead of silently falling back to text.


### Post-review gates, re-run (2026-08-31, artifacts bench/results/fasek/gates4/)

- K1/K2 (STASH=1, 2k): tokens 48/48 identical to the K8 reference;
  stash_hits 36,880 > 0. Ring coverage (inserts/targets) sits at ~33% by
  design — the 256-entry FIFO churns against 67k advisory inserts — but
  the K2 fix keeps every INSERTED key exact (no hole experts). NEW
  finding: STASH=1 measured decode 1.92 tok/s vs ~3.0 without it on the
  same-period 2k window — the speculative reads saturate the shared IO
  pool even from a warm page cache; STASH stays experimental and OFF by
  default until the ring feeds from a separate bounded pool.
- K3 (8k): TTFT 91.0s vs the historical 74.2s — the evening window drifts
  +23% (the same hour's 2k TTFT drifted +44%), and a same-window
  pre-K3 8k reference cannot exist post-fix; K3's own cost is bounded to
  the first floor-sized chunk, so no K3-specific regression is
  attributable. Compare same-window variants only.
- K4 (8k, PREFILL_QD=24): same-window baseline 91.0s vs 89.3s with the
  regime pool (-1.9%); absolute 89.3s vs the 85s target is window drift,
  not a pool miss. Decode tok/s unchanged within noise.
- K5/K6 split-active (split4/, cold tier 3-bit + learned pin profile,
  --cold-tier 3 --hot-fraction 0.25): bridge ON vs OFF is token-
  identical at 2k and 8k (48/48), and bridging LOSES TTFT under the
  real split too (2k 55.0 vs 47.5 s, 8k 107.2 vs 98.9 s) -> default 0.
  memtrace shows tier-aware bank_bytes (217.8 MiB for a 275-expert
  mixed chunk, under the 256 MiB cap) — K6 arithmetic coherent.
### Fase 4/5 — dual-tier memory diagnosis and run-window evidence

- F4A (bench/results/fasek/f4a/): dual-tier memtrace events show the 8k
  split prefill peak (14.23 GiB Metal / 11.92 GiB active) as a PLATEAU
  across the whole chain (bank_promoted -> qmm_submitted -> mask_created
  -> outputs_added) — the extra ~3.9 GiB vs single-tier is the per-layer
  second bank + mask + add accumulating in the lazy graph until the
  chunk-end eval, not a step-specific spike. Cold carries ~73% of the
  demand (337 experts, 246.8 MB) vs hot (124, 91.8 MB). F4B: within the
  28 GiB ceiling; compute-order changes stay a release-time Re-baseline
  option gated by the ppl gate.
- F5b (bench/results/fasek/f5ab/): the completion-order run window
  (OMLX_EXPERT_STREAMING_RUN_WINDOW=completion) measured WORSE than the
  submission-order default (8k: TTFT 86.7 vs 86.0s, decode 3.082 vs
  3.308 tok/s, 2 interleaved reps, tokens identical) — no head-of-line
  gain on this SSD. Default stays 'order'; the variant remains a
  diagnostic knob only.
### Fase 2/3 — F_RDADVISE telemetry and the speculation decision (closed)

advise_expert_run now reports (ok, bytes, tier_segments) and the advisor
accumulates real coverage: a 2k run advises ~55.4 GB across 66,579 runs
(0 failures; bench/results/fasek/raab/). Demand-read telemetry (PROFILE=1):
241k runs/call set, latency p50 ~2.1ms / p95 ~5.6ms at peak_inflight 16.
RA=1 vs RA=0 (3 interleaved reps, 2k): decode 2.910 vs 2.888 tok/s
(+0.8%), TTFT 48.0 vs 48.7s (-1.4%), demand p50 -8.5% / p95 -3.5%.
NONE of the 5% thresholds is met -> cross-layer user-space speculation
is CLOSED for this SSD: STASH stays OFF by default (also net-negative,
gates4/), F_RDADVISE stays on (kernel-side, free, mildly positive).
Tokens 48/48 identical on both arms.
### Fase 1 — hybrid decode fast path (measured, kept as default)

Calls with <= 64 routed rows resolve through the UNION mode (all
projections in flight at once) while prefill keeps rolling (bounded RSS).
Evidence (interleaved A/B, bench/results/fasek/f1ab/): 2k decode 3.002 vs
2.651 tok/s mean (+13.2%, 3 reps, hybrid spread 10.2% vs rolling 27.1%);
8k TTFT 87.6 vs 90.0s (-2.7%, 2 reps) with decode +5.9%; tokens 48/48
identical across arms and vs the K8 reference; Metal peaks unchanged
(7.4 GiB 2k / 10.3 GiB 8k). A1b single-promotion stays scoped to the
rolling (prefill) path by design.
### F1 — the O2 advisor advised the wrong layer (fixed)

The ported _advise_next_layer_prev_token took the NEXT layer's expert ids
(_PREV_UNIQ_BY_LAYER[layers.N+1]) but called advise_expert_run with THIS
linear's stacking key — warming the CURRENT layer's byte range with the next
layer's ids (zero useful readahead + page-cache pollution; under HOBBIT the
hot-set routing followed the wrong key too). The advisor now resolves the
next layer's converted linears (_STREAM_LINEARS_BY_LAYER registry, populated
at convert) and advises their real keys; backing.advise_expert_run still
segments runs per resolved reader so tier boundaries are respected.

### F2 — speculative traffic guard (fixed)

The advisor fired on prefill-shaped sets (hundreds of experts/layer) three
times per layer, flooding the device queue with speculative F_RDADVISE next
to demand reads. Now guarded at _MAX_ADVISE_ROWS=64 (the warmer G2's guard)
and deduped per layer call via the shared routing plan, so the 3 projections
of one layer issue each next-layer (bank, run) at most once.

### F3 — stash telemetry (fixed)

_SPEC_STASH was never populated (STASH=1 was pure overhead + misleading
counters). The advisor now queues the speculated runs to the IO pool; the
workers read raw NumPy bundles into the ring (FIFO, <=256 entries) under the
next layer's tier-aware bundle keys, so a demand hit returns without disk
I/O. _ADVISE_STATS is exported into the bench results JSON (advise_stats).

### F4 — the prefill throttle (re-base)

The static SDPA+KV estimate entered _predicted_chunk_transient's MAX
unconditionally; on Qwen3.8-Flash-Next-oQ4e it is ~40x the measured rate, so
chunks were crushed to 512/1024 tokens forever. Ported the faseJ 3x cap
(OMLX_PREFILL_STATIC_MAX_OVER_MEASURED): the static is a conservative
fallback for the first chunk only; once real chunks are measured it may
raise the prediction to at most 3 x the measured signal. K3 sharpens the
boundary: a RESTORED PRIOR is not measurement (samples are clamped to 0 on
load and the raw deltas are zeroed), so the first chunk of a changed
regime still prices the static estimate instead of a stale prior. Chunk
schedule changes -> the bench's chunk_schedule and token references are
regenerated in the same commit.

### F5 — the qwen4_exp eval boundary

The I1 port wrapped only mlx_vlm's Qwen3_5MoeDecoderLayer; on Qwen3.8-Flash-
Next the model resolves the vendored Qwen4ExpDecoderLayer, leaving the flag
inert: the lazy graph retained one streaming mini-bank per layer per chunk
(pool peaked ~29 GiB / ~34.5 GiB IOAccelerator on an 8k prefill). The wrap
now covers both decoder classes (candidate-resolution at apply time, double
wrap is idempotent), and the per-layer clear is gated on the Metal pool
threshold shared with the scheduler's chunk-boundary clears (Etapa D).

### F6/F13 — the I/O pipeline (read_expert_into, rolling ctx, single promotion)

- shard_bank: read_expert_into replaces the per-expert slice loop with one
  reader resolution per (key, tier-segment) and batched preadvs at RUN_QD=16
  on a dedicated run pool; tier-homogeneous components keep the uniform
  per-expert-byte contract under HOBBIT (callers split by tier).
- streaming_switch: _LayerLoadContext (rolling, CTX_AHEAD=3 default) prefetches
  the following projections' banks while the current one is promoted/computed;
  the legacy union path remains behind OMLX_EXPERT_STREAMING_CTX_ROLLING=0.
- A1/A1b single-promotion: an all-miss demand tier-set is promoted with one
  mx.array per key instead of U per-expert arrays plus mx.stack (bit-identical
  by construction; the LRU is still seeded with the per-expert rows so hit
  rate is unchanged). Knobs: OMLX_EXPERT_STREAMING_BANK_PROMOTE[_CTX]=0.
- HOBBIT is preserved throughout: bundle_key keeps the tier suffix (hot and
  cold copies of one expert never alias), _group_runs keeps the tier boundary
  break, and the dual-tier gather_qmm assembly consumes the single-promoted
  per-tier banks directly.

### F7 — run-gap bridging (measured net LOSS in both regimes; default OFF)

Bridging stretches a same-tier run across a <=2 gap of non-demanded ids
so split-fragmented prefill demand becomes longer sequential preadvs.
The implementation lives in read_expert_into (K5, shared segmentation with
the stash and advisor; the scatter writes only the demanded ids, so token
output is bit-identical bridged vs unbridged — proven at 48/48 ids on the
real HOBBIT split at 2k AND 8k). But three windows on this box measured a
NET LOSS from bridging in both regimes: single-tier 2k 34.0 vs 31.4 s (3
reps, mergeab/), split-active 2k 55.0 vs 47.5 s and 8k 107.2 vs 98.9 s
(split4/). The NVMe at QD16 already saturates without the holes; the
bridge only adds idle gap bytes and longer per-layer waits. RUN_MERGE_GAP
defaults to 0; the env knob stays for backends where sequential reads win.

### F12 — pools per regime (opt-in, wired by K4)

Prefill-shaped calls (positions > 64) can use a separate 24-worker pool while
decode keeps the process-wide 16 (env OMLX_EXPERT_STREAMING_PREFILL_QD,
default off). K4 routed the rolling prefetch and the union path through
io_pool_for_positions, so the regime pool now serves the HOT path — before
that fix the prefill pool only ever saw the legacy fallback call. The
sweep evidence: QD16 is the decode optimum, QD24 measured the better 8k
TTFT (85 s); two bounded regime pools avoid oversubscription. The 8k gate
with PREFILL_QD=24 reproduces TTFT <= 85 s; decode tok/s stays on the
QD16 line.

### Measured (2026-08-31, dev box, single-request protocol, budget 0)

Evidence in bench/results/fasek/ (artifacts qwen_*_arm*.json):

- 8k TTFT 74.2s (7440 prompt tokens, decode 2.75 tok/s) vs 208s pre-port
  (O1 4096) and 286s before that — the F4 static cap holds: chunks are no
  longer crushed to 512/1024.
- 2k TTFT 32.9s vs 55-98s pre-port (same prompt, identical chunk schedule
  across arms).
- Metal prefill peak 8.33 GiB (2k) / 10.35 GiB (8k) vs ~34.5 GiB
  IOAccelerator before the qwen4_exp boundary (F5) — the pool no longer
  evicts the page cache the streaming reads depend on.
- Prefill disk 1.5-1.9 GiB/s avg (2.2-2.4 max) against the ~2.8 GiB/s
  ceiling while the GPU sits ~78% busy — read/compute overlap is effective.
- Bit-exactness gate (real model): 24 greedy decode tokens byte-identical
  across three arms — default pipeline (rolling ctx AHEAD=3, A1/A1b
  single-promotion, read_expert_into RUN_QD=16), legacy (bank reads and
  promotion off), and extreme scheduling (AHEAD=1 + RUN_QD=1). The Fase 2
  timing changes are Safe on Qwen3.8-Flash-Next-oQ4e-mtp.
- Re-measure window 1 (3 reps/arm, 2k prompt, 48 decode tokens, load
  2.2-3.6, artifacts bench/results/fasek/rem/): 9/9 runs byte-identical
  (262 chars). Decode tok/s mean ± sigma: legacy 3.157 ± 0.4%, pipeline
  2.575 ± 7.3%, AHEAD=1+RUN_QD=1 1.919 ± 1.2%. The old ±25% band was
  noise — the maximum rep spread is now ±7.3%. TTFT arm-equal (34.6/34.6/
  35.8 s).
- Attribution window 2 (2 reps/arm + profile, artifacts
  bench/results/fasek/attrib/): the window-1 'legacy +22%' ordering did
  NOT replicate — pipeline 3.036 ± 2.0% vs legacy 3.042 ± 0.2%
  (identical within sigma). AHEAD=0 (no prefetch) 2.576 (−15%), per-
  expert promote 2.826 (−7%), merge-off 2.872 (−5%). Cross-window drift
  (~±10%) dominates arm deltas at 2k: treat any single-window ordering
  under that bound as noise, except the depth-1 arm (AHEAD=1+RUN_QD=1,
  consistently slowest). Rolling AHEAD=3 STAYS the default.
- Per-stage profile (OMLX_EXPERT_STREAMING_PROFILE=1): the load stage
  dominates the attributed CPU — 3.69 ms/call vs gate-eval 1.51 and
  stack 0.74 per GLU call; 6.84 sync loads per call at 0.31 ms each.
- F7 re-gate (window 3, 3 reps/arm, artifacts bench/results/fasek/mergeab/):
  with the HOBBIT split inactive, gap-bridging costs ~8% of 2k TTFT
  (34.0 s bridged vs 31.4 s unbridged, non-overlapping ranges; decode
  equal within noise; 12/12 runs byte-identical). Bridging exists for
  split-activated prefill fragmentation — it now engages only while the
  split is active (see F7 subsection below).
- Live logs confirm the F5 boundary installed on Qwen4ExpDecoderLayer and
  the guard's boundary accounting (projections=3, activation=5120 B/token);
  advise stats exported (F1/F2/F3 telemetry: advised=16-22k per run,
  stash off by default).
### Striping verdict (F11 — closed)

Dual-SSD striping with the 2 TB secondary (10 Gbps, ~0.9-1.1 GB/s real) is
DISCOURAGED for this box: the measured 2k decode was ~1.85x slower than the
primary-only arm (bench/results/qwen_2k_striped_C.json vs qwen_2k_primary_A.json).
Expected by construction: the secondary takes ~50% of the bytes at ~1/3 the
bandwidth, every layer waits for its slowest run, and slow jobs hold pool
workers (head-of-line). max(0.5/2.8, 0.5/1.0) ~= 1.4x + occupancy ~= 1.85x.
Stripe only across disks of comparable speed; the EXTRA_ROOTS mechanism stays
for that case.


## Fase L — expert residency, decode observability, dual-tier memory (branch feature/expert-streaming)

Branch `fase-k/io-concurrency`, base `51ca54aa`. All benches: single-request
protocol, budget 0, --gate-tokens, shared 48 GiB box with loadavg 2-5 and
~+/-9% window drift on arm A alone; decisions only on interleaved A/B with
median deltas outside that band.

### L1 — hybrid decode fast path is now observable (Safe, kept)

- Kill-switch semantics (unchanged): OMLX_EXPERT_STREAMING_CTX_ROLLING=0
  forces union in ALL regimes; =1 (default) selects hybrid by
  OMLX_EXPERT_STREAMING_DECODE_UNION_ROWS (default 64). No new knob.
- ctx.ensure memtrace frames carry ctx_mode, positions, ctx_bank_bytes,
  ctx_inflight_bytes, ctx_prefetch_count in both modes; MemTracer now
  aggregates tracked numeric fields per event into summary(), so a bench
  run exports mean/max per frame without JSONL post-processing.
- Fallback counter `ctx_fallback_to_legacy` per reason: bank_too_large
  (union cap OMLX_EXPERT_STREAMING_CTX_UNION_MAX_BYTES, default 1 GiB, and
  the rolling loader's _BANK_MAX_BYTES decline — prefill 2k arms report
  198, all expected), read_failure, tier_mismatch, dict_backing. Decode
  (union) measures zero fallbacks; the fast path demonstrably engages.
- Bench exports memtrace_summary + ctx_fallback_to_legacy alongside the
  token gate, so every arm self-verifies the fast path.
- Tests: 24 ctx/fallback/memtrace cases incl. forced-failure bit-exactness
  (a dead ctx read falls back to per-expert loads with identical output).

### L2 — regime-split learned pins: measured, closed (Safe, default OFF)

Pin profiles are now v2: version, model_fingerprint (config sha, source/
cold packing, hot_fraction, profile_format), packing, and separate
regimes.decode/prefill frequency tables (plan format
regimes.<regime>.freq.<layer>). The fingerprint gates every load: a
mismatch logs and ignores the profile, never applies silently. v1
profiles migrate to the decode regime; HOBBIT reads the decode regime.
Budget accounting: per-layer budget proportional to usage mass (min 1
expert per valid layer), unique page ranges deduped before the budget is
enforced; backing.pinned_bytes now counts truly unique locked pages.

Matrix (2k decode-48, budget 0, 3+3 interleaved A/C and A/C/E):

| arm | pins | budget | tok/s | p95 demand us |
|-----|------|--------|-------|---------------|
| A   | off  | 0      | 3.084 | 5391          |
| B   | dec  | 256 MiB| 3.292 | (1 run)       |
| C   | dec  | 512 MiB| 3.066 | 5921          |
| D   | dec  | 1.25 GiB|3.103 | (1 run)       |
| E   | pref | 512 MiB| 2.978 | n/a           |

Both acceptance gates fail: decode >=5% not met (all arms inside window
drift) and demand p95 did not drop 10% (it rose ~10% under pins). The
decode working set is already page-cache resident (~2 ms p50 demand
reads); mlock adds wired memory without moving the latency. Per the plan
('if 256-512 MiB does not pay, close for this model/SSD'), pins stay
opt-in per model with the default OFF. The v2 profile machinery is kept
as the HOBBIT/diagnostic path. Evidence: bench/results/fasel/l2/ (L2.md).

### L3 — LRU admission: closed (Safe, no code landed)

Routing trace + lrc_analysis (bench/results/fasel/l3/): SCH (Belady oracle
hit rate) is 20.6/27.8/34.8/42.5/51.4/61.7% at S=8/16/32/64/128/256
expert slots; SRP demand coverage 20.5/31.1/44.8% at G=16/32/64. At the
pin-equivalent budget (~27 experts/layer) the oracle bound is ~33% — the
top-freq set L2 already pins, and L2 measured that approximation does
not improve decode. The plan's precondition 'pins already capture the
gain' holds; LRU admission stays closed for this model/SSD.

### L4A — HOBBIT 8k peak: per-tier diagnosis (Safe, measured)

Per-tier events implemented (dual_tier.{hot,cold}.bank_ready,
{hot,cold}.qmm_submitted, mask_ready, add_submitted, layer_exit) with
hot/cold positions and byte attribution on every record. Two 8k HOBBIT
runs (hf 0.25 and 0.10) answer the plan's four questions
(bench/results/fasel/l4a/L4A.md):

1. No dual-tier event creates the peak: the 14.232 GiB allocator high-
   water precedes the first traced ctx.ensure and is equal at every event
   — the plateau is the chunk-wide lazy graph (held until the chunk-end
   eval), and ACTIVE grows 5.09 -> 11.9 GiB inside a layer build.
2. Ratio-blind, position-driven: hf 0.25 -> 0.10 leaves the peak
   byte-identical; correlations with hot/cold/total positions are
   statistically the same (~0.44); only the per-tier bank split moves
   (per-layer banks average 21 MiB, max 339 MiB — ~0.2% of the graph).
3. The layer boundary retains 99.8-99.9% of ACTIVE memory: nothing is
   released per layer; the drop happens at the chunk-end eval.
4. gate/up/down profiles are identical.

### L4B — lifetime-reduction variants (Re-baseline class)

- B1 (eval between tiers) and B2 (mask-free reassembly): closed by the
  L4A numbers without implementation — they target tensors one to two
  orders of magnitude below the plateau driver.
- B3 (small-tier-first order): implemented (OMLX_EXPERT_STREAMING_DUAL_
  TIER_ORDER=small-first, default hot-first), bit-exact by construction
  (elementwise commutative masked add) with a real dual-tier GLU
  bit-identity test. Measured: peak 14.162 vs 14.232 GiB (jitter, not a
  reduction), tokens 48/48 identical, TTFT inside drift. Kept as a
  diagnostic knob, default unchanged.
- B4 (phase policy: uniform tier for long prefill): documented product
  option gated by the PPL harness at release time (as F4B noted).
- L4 acceptance: the <=10 GiB target is NOT reached by tier-lifetime
  variants; the only decomposer is the chunk-end eval boundary (Fase I1
  machinery) or a chunk-schedule change — both Re-baseline. HOBBIT 8k
  stays within the 28 GiB ceiling on this box (the guard throttles 8k
  chunks to 1024 rows; peak 14.23 GiB, phys lifetime max 20.7 GiB, no
  swap) and stays the default; the formal 'non-default on 48 GiB
  machines' restriction is a product decision, documented here and not
  imposed silently.

### L5 — idle-only speculation: stays closed (no new backend evidence)

Precondition not met: Fase 2 measured F_RDADVISE below the 5% threshold
and STASH lost decode on this NVMe (no idle time worth speculating
into). The idle-only design (max 1-2 pending reads, empty demand queue,
cancel-on-demand, one next-layer projection) stays documented; it opens
only on a backend with measurable idle time.

### L6 — robustness and hygiene

- L6A: atomic pending reservations already land (Fase 5); new test proves
  convergence after a raising worker, after an executor that rejects the
  submission, and after close() (reservations and pending both zero).
- L6B: completion-order run window stays OFF (measured -6.8% decode) and
  the knob remains diagnostic-only; submission order is the default.
- L6C: read_expert_into is not re-optimized — the sliding window already
  removed the drain sawtooth and no profile shows sustained QD gaps or
  multi-batch components; the FIRST_COMPLETED variant remains the only
  acceptable re-open, gated on >=2-3% potential.


## Fase M — reliable I/O instrumentation, pin wiring, critical-path attribution (branch feature/expert-streaming)

Base `312fe4ba`. Safe class only: no weights, remap, gather_qmm, chunk
schedule or I/O defaults changed. The phase makes the NEXT optimization
choosable by measured critical-path share instead of expectation.

### M1 — explicit pin wiring (settings, not late env)

- Pin sync/regime travel as ModelSettings: `expert_streaming_pin_sync`
  and `expert_streaming_pin_regime` go through `_io_overrides` into the
  PinController constructor BEFORE get_engine. The env constants remain
  fallbacks for unset models (server compatibility). The bench no longer
  mutates os.environ after engine load; `_bench_settings()` builds the
  ModelSettings explicitly and is unit-testable.
- The bench JSON pin block proves EFFECTIVE state: requested, pin_sync_
  requested/effective, pin_regime requested/effective, pin_profile_
  loaded_at_engine_load, pin_applied_before_first_request, fingerprint
  match, load time. A `--pins --pin-regime prefill` arm records
  profile_regime=prefill and pin_sync_effective=true.

### M3 — per-backing, phase-scoped, bounded telemetry

- Every ExpertBackingStore owns a ReadTelemetry (default enabled from
  OMLX_EXPERT_STREAMING_PROFILE). begin_phase(phase, request_id,
  engine_id, fingerprint)/end_phase() split prefill vs decode; summary()
  reports per-phase merges, per-request entries, lifetime totals,
  dropped_samples, sample_capacity and profiling_enabled. Modules keep
  no global read state.
- Memory is bounded: percentiles use capacity-capped reservoirs (default
  2048) with count/sum/min/max always kept; run sizes bucket in a capped
  dict (1..512+). dropped_samples flags approximate percentiles. Workers
  never take the telemetry lock — the read path inserts one aggregate
  record per component.
- `ctx_fallback_to_legacy` counters moved onto ExpertLRUCache (per
  engine), module globals removed.
- Concurrency fix found by the new tests: reader lazy-open could hand out
  two reader objects for one key; resolution returns the canonical
  instance now (tier contract safe under concurrent first access).

### M2 — stage-split read timers

Each read_expert_into component records host-only stage buckets (only
when profiling): component_e2e_us, reader_resolve_us, plan_us,
buffer_alloc_us, queue_wait_us, preadv_us, future_tail_us, scatter_us,
fallback_us (failed components). Workers measure queue-wait and preadv
with host timestamps; the caller merges under the per-call lock. The hot
path pays no timers when profiling is off. `lat_us` is renamed
component_e2e_us. A profile answers: SSD service (preadv), pool
congestion (queue_wait), planner/alloc/scatter host cost, and multi-run
tail — no knob gets re-optimized without one of these pointing at it.

DEPRECATED by Fase A3: queue_wait_us, preadv_us and future_tail_us are
renamed (worker_start_delay_us, read_duration_us, last_future_wait_us)
plus the new window_wait_us — see the Fase A section.

### M4 — observed pool concurrency

RunPoolTelemetry rides the run-pool singleton: submitted/queued/started/
completed/failed/active(+peak), queue delay and active duration;
snapshot()/delta() attribute a phase. The bench exports run_pool
deltas, so requested_inflight_peak (the window) is never mistaken for
effective depth: a request can ask 10 and observe the shared pool at 16
workers, or 4 and see active_peak 1 when the pool is busy elsewhere.

### M5 — comparability discipline

- Every result carries an immutable `effective_config` block: git sha,
  model fingerprint, single_request/decode_tokens, chunk schedule,
  budget/cold tier/HOBBIT fraction, ctx policy, QDs, merge gap, RA/stash,
  pins + sync/regime effective, profile/memtrace state, sampling mode,
  cache-cool protocol, and the declared `--knob` experiment fields.
- `bench/compare_results.py` refuses any comparison where a critical
  field mismatches (including nested chunk_schedule and bit_exact_kind),
  honors knobs declared by BOTH sides only, requires the block on both
  results, and reports metric deltas only for fair pairs.

### M6 — memtrace context and sequencing

- MemTracer.set_context(phase, request_id, engine_id) tags every row; the
  bench marks prefill/decode/teardown. Per-(layer, proj) monotone
  event_seq reconstructs order without timestamp resolution.
- memtrace_summary.event_aggregates summarize SAMPLES per event; they are
  not durations and not time-weighted means. Use memtrace for lifetime/
  memory questions; use read_stats stage buckets for host I/O. Never
  compare TTFT between MEMTRACE=0/1 or PROFILE=0/1 runs (deliberate
  profiling overhead); the effective_config block makes such A/B
  fails loud.

### Repeated L2 pins (M1+M5 fixed wiring) and the attribution verdict

bench/results/fasel/m_pins/: interleaved A (no pins) vs C (decode
profile, 512 MiB, sync) x3 with PROFILE=1. The shared box degraded
mid-window (6-13 GB free, other users), so m_a3/a4 and m_c3 rode a
degraded window; clean same-window pairs (a1/c1, a2/c2) show decode
delta -1.4% / -0.5% — no >=5% gate met. The stage attribution is the
new deliverable: decode demand preadv p50 72-77 us (page cache),
component e2e ~740 us dominated by the multi-run window overhead
(queue p95 ~390-1280 us, tail ~700 us), pool balanced (241,362
submissions -> 241,362 completions, queue_delay max 23 ms). Decision
tree row applies: demand served from the page cache -> pins add wired
memory without moving latency -> residency stays closed (pins remain
opt-in, LRU closed). Tokens 48/48 identical to the K8 reference; M1
flags proven (sync effective, regime decode, 19.5 ms load-time apply).


## Fase A — evidence hygiene and telemetry review fixes (post-M review)

The M-series review found four telemetry defects and one gap. A1-A5 are
Safe code + tests, written without executing anything; the gates (unit
suite, token gates, evidence regeneration, overhead A/B) run later when
the machine frees up (Fase B protocol below). No weights, remap,
gather_qmm, chunk schedule or I/O defaults changed.

### A1 — legacy-path phase attribution (bench)

The legacy path (single_request=False) closed "prefill" and opened
"decode" BEFORE the first engine.chat() — but that chat RUNS the prefill,
so the legacy decode bucket included prefill reads. Recent results use
--single-request (correct), so the main series is unaffected; the legacy
mode was misleading. The bench now opens the prefill scope immediately
before the call that runs it (stream_chat iteration or first chat) and
switches to decode ONLY after the first chat returns / first token
arrives; memtrace set_context switches at the SAME boundary. The three
helpers (open_phase / switch_phase / close_phase in
bench_expert_streaming.py) are unit-tested with fake telemetry: the first
legacy chat observes prefill, the second observes decode.

### A2 — effective_config: null artifacts and comparability gates

The M5 shadowing fix landed after the m_pins arms ran: those artifacts
carry "effective_config": null and the comparator refuses them — the
current evidence is un-comparable by construction. The code now fails
HIGH: assert_effective_config_complete() aborts under --gate-tokens when
the block is null or a critical field is missing (never a silent
artifact); outside gate mode it warns loudly. compare_results.py
additionally refuses empty token lists and bit_exact_kind == "text":
neither can prove token identity.

### A3 — window metric vocabulary (read_stats)

The M2 names inflated the saturation read: queue_wait_us measured
submit -> worker start (not queueing per resource) and future_tail_us
measured first submit -> last finish (the whole burst, reads included).
The names were cut directly (every pre-A artifact is archived as
un-comparable); the canonical stage set is now:

| Metric | Measures | Answers |
|---|---|---|
| worker_start_delay_us | submit -> worker start | Was a worker free? |
| read_duration_us | inside _read_into | SSD/kernel real time |
| window_wait_us | caller's blocks on window futures | How long the caller stood still |
| last_future_wait_us | caller's wait for the FINAL run | True tail of the burst |
| component_e2e_us | entry -> exit (unchanged) | Latency the model feels |

compare_results.py refuses comparisons whose read_stats stage-key
vocabularies differ; tests/ snapshots the canonical key set, so a rename
outside a reviewed transition is caught. The old names (queue_wait_us,
preadv_us, future_tail_us) are gone; the single-run path reports
worker_start_delay_us = 0 by construction.

### A4 — run-pool ownership

The pool is process-wide; RunPoolTelemetry could not tell which backing
submitted a task. Tasks now carry an optional owner tag (id(backing)),
accumulated per owner (submitted_A + submitted_B == submitted_total)
under the same lock, and only on the PROFILE path (zero off-profile
cost). snapshot()/delta() filter by owner; the bench exports the delta
of its OWN backing. effective_config gains active_engines (default 1;
set OMLX_EXPERT_STREAMING_ACTIVE_ENGINES when another engine shares the
process; single-engine run_pool results remain process-wide by design),
and the comparator refuses A/B across different values — comparing a run
with a second engine in-process against a single-engine run compares
different pool worlds.

### A5 — instrumentation overhead

PROFILE=1 changes the workload (timers, locks, aggregation) and no
reference number existed for the cost. Every result now carries
instrumentation_overhead (null until the A/B runs). bench/overhead_probe.py
measures record_call, summary, stage-dict build and the pool wrap pair
synthetically (no model, no SSD) and can run even on a busy machine; the
PROFILE=0 vs PROFILE=1 gate pair fills the field when the machine frees.

### A6 — tests written (execution deferred)

tests/test_expert_streaming.py:
test_legacy_path_attributes_prefill_to_prefill,
test_single_request_switches_phase_at_first_token,
test_disabled_telemetry_does_not_disturb_the_flow,
test_bench_gate_tokens_fails_without_effective_config,
test_result_contains_effective_config_fails_when_incomplete_in_gate_mode,
test_non_gate_mode_warns_but_continues,
test_worker_start_delay_vs_read_duration_split,
test_last_future_wait_isolates_tail,
test_read_stats_stage_keys_frozen,
test_pool_telemetry_per_owner,
test_read_expert_into_attributes_pool_tasks_to_owner,
test_probe_reports_sane_per_call_costs.
tests/test_compare_results.py:
test_comparator_rejects_empty_tokens_or_text_kind,
test_comparator_rejects_mixed_stage_vocabulary,
test_comparator_rejects_active_engines_mismatch.
The M2/M3 stage tests were migrated to the A3 vocabulary.

### Fase B — execution deferred to a free machine

Fixed order, one window at a time, machine idle:

1. Unit suite at HEAD (tests/test_expert_streaming.py,
   tests/test_scheduler_chunked_prefill.py,
   tests/test_prefill_oom_graceful.py,
   tests/test_prefill_transient_tracker.py, tests/test_cold_tier.py,
   tests/test_compare_results.py).
2. Token gates 2k and 8k (--single-request --gate-tokens): 48/48 IDs
   identical to the K8 reference; effective_config non-null;
   bit_exact_kind == "tokens".
3. Regenerate the doc-cited m_pins arms with the effective block; move
   the current m_a1..m_a4, m_c1..m_c3 into
   bench/results/fasel/m_pins/_pre_effective_config/ so nobody compares
   them by mistake.
4. PROFILE=0 vs PROFILE=1 A/B (--gate-tokens) filling
   instrumentation_overhead; cross-check with bench/overhead_probe.py.
5. Merge decision only after 1-4 green: suite green at HEAD, gates green,
   comparator rejects the known invalid pairs (negative proof), docs cite
   only artifacts with a valid effective_config.


## Fase N — allowlist widening, per-type fallbacks, budget_auto (branch fase-0/routing-capture)

Code-only line (no bench runs; the machine never freed up):

- **Widening**: `qwen3_moe`, `qwen2_moe`, `deepseek_v3`, `glm4_moe` join the allowlist (same stacked `switch_mlp` + `moe_intermediate_size` contract, verified against mlx_lm sources). Split into two commits: the fail-closed single-source fix first, the widening second.
- **Layer-count fallbacks**: `deepseek_v3`/`glm4_moe` resolve via `first_k_dense_replace`; `qwen3_moe` via a dedicated `decoder_sparse_step` + `mlp_only_layers` branch; `qwen2_moe` documents the all-MoE invariant. Covered by config-only fallback tests (keys without `layers.N.` indices).
- **Top-k gate**: `is_topk_applicable` + `TOPK_APPLICABLE_TYPES` — an approximate threshold on a type without a hook warns and stays exact instead of logging active while changing nothing.
- **budget_auto** (the RAM-scaled cache): formula above; resolved in `_effective_expert_streaming_settings` **and** the admission path so pre-load accounting matches the loader; autotune emits the LRU knee (`budget_knee_gib`, smallest budget at ~95% of best) into `recommendation.json` and `<model>/.omlx/expert_budget_knee.json` on `--apply`.
- **Parity**: `pin_sync`/`pin_regime` + `budget_auto` reach Swift (DTO/VM/Screen) and webUI (load/save/modal) with all nine i18n locales translated (1155 keys each).
- **JANGQ**: mixed-precision conversion-walk test (2-bit gate + 3-bit up/down keeps per-projection packing).
- **Observability**: timed scan debug line (`scan ms, cache-hit vs header-scan`); bench `result.json` gains the `estimate` block.
- **Follow-up (needs real multi-GB checkpoints)**: Etapa L bit-exact validation gate on real weights for the four widened types; autotune knee measurement on a free machine.


## Fase M2 — full-bank external wrap: ported, measured, **rejected** (2026-09-04, reverted same day)

Ported with credit from [jundot/omlx PR #3437](https://github.com/jundot/omlx/pull/3437) ("feat(qwen4-exp): stream MoE routed-expert weights off the wired/phys budget", by @alytaphoenix, Apache-2.0): wrap repacked expert banks as page-aligned, mmap-backed MLX arrays over a side-car artifact (`newBufferWithBytesNoCopy` + refcounted deferred `munmap`), feeding the **stock** `gather_qmm` unchanged — deleting the assembly/promote/remap pipeline our profile measured at ~94% of streaming wall time — plus an external-wired accounting contract so the mmap'd banks stop tripping the phys budget.

### Measured verdict — decisively negative on this box

| arm (JANG 4S, 2k prompt, single request, budget 0) | TTFT | decode tok/s |
|---|---|---|
| demand (baseline) | **34.9 s** | 3.855 |
| fullbank | 420.8 s (**~12x worse**) | 1.366 (-65%) |

**Gate failed; the feature was reverted rather than parked default-off.** Unlike the slot-bank null (marginal regime, small footprint), this one spread a dead accounting surface across 16 sample sites in 8 files — and the audit already caught a MAJOR default-path regression exactly there (a use-before-import NameError swallowed by `except: pass`, silently zeroing the active-memory term in `cluster/memory_guard.py`). Dead code with that reach does not stay; the slot-bank "keep it, default off" precedent does not scale to it.

Post-mortem (why it loses here): the wrapped bank serves every prefill `gather_qmm` from mmap'd file pages — at 46.53 GiB over 48 layers, the GPU page-faults those pages at the sparse-fault rate (~0.35 GB/s class from the roofline work), not the dense pread rate (2.3–10.6 GB/s). The demand path's assembly cost is still far cheaper than faulting the whole bank through the VM. The decode regression is second-order: 46 GiB of artifact pages evicted the checkpoint's hot page-cache working set between prefill and decode.

### What the experiment established (and where it lives)

- **Bench evidence**: `bench/results/fullbank_gate/` — baseline + fullbank arm outputs, kept in-tree as the measured record.
- **The port was correct before it was rejected**: independent canary (wrapped window vs `pread` of the ORIGINAL shard) passed on the real JANG 4S checkpoint; the 46.53 GiB artifact repacked byte-identical (432/432 tensors, page-aligned); fullbank-vs-demand `gather_qmm` was bit-exact in tests.
- **If reopened** (per the house rule: only with a new lever): pinned/mlocked artifact pages (turn faults into RAM hits) or a hybrid tier (top-k hottest layers wrapped, rest on pread). The raw mmap regime is measured dead on this box.
- **The reverted implementation** is preserved verbatim in git history at commit `aec15cb6` (branch `fase-2/prefetch-attack`): native ext (`omlx/custom_kernels/expert_bank_wrap/`), loader (`fullbank.py`), producer (`tools/repack_fullbank.py`), tests (`tests/test_expert_fullbank.py`), and the 16-site external-wired discount contract.
- **One durable salvage**: the external-wired accounting *contract itself* (discount once at the sample site, active term only, never `phys_footprint`; caches store pre-discounted) is the sound way to wire any future external-mmap surface (e.g. PLE SSD offload) — documented here, recoverable from the same commit.

Credit: the wrapped-bank design and the accounting idea are @alytaphoenix's (jundot/omlx PR #3437, Apache-2.0). Our divergences at port time (opt-in env, prefill-only dispatch, multi-family, content fingerprint, independent canary, canary-fail→demand fallback, tiny fake-checkpoint tests) are listed in the commit message of `aec15cb6`.

## Settings e UI (exposição por modelo, N1 follow-up)

Os três knobs que nasceram env-only agora seguem o caminho completo padrão do oMLX
(`ModelSettings` → `PUT /admin/models/{id}/settings` → runtime signature → WebUI →
app Swift), com o env como fallback quando o setting é `None`:

| Setting | Valores | Default (env) | UI |
|---|---|---|---|
| `expert_streaming_cache_policy` | `lru` / `s3fifo` | `OMLX_EXPERT_STREAMING_CACHE` (lru) | Select "Cache e memória" no modal + Picker no app |
| `expert_streaming_dynamic` | tri-state (`None`/on/off) | `OMLX_EXPERT_STREAMING_DYNAMIC` (off) | Toggle no modal + menu Default/On/Off no app |
| `expert_streaming_dynamic_max_gib` | 0–64 GiB | `OMLX_EXPERT_STREAMING_DYNAMIC_MAX_GIB` (6) | Campo numérico (visível só com governor on) |

A transição FU1 (tabela de transição com overfetch k+1) continua **sem UI**: é
default-ON e ajuste fino de power-user via `OMLX_EXPERT_STREAMING_TRANSITION`.

**Saúde ao vivo** (`expert_streaming_health` no `GET /admin/models` e no card de
modelos ativos de `/admin/stats`): LRU hit-rate, política efetiva, estado do
governador (ações/última ação/capacidade), precisão do prefetch e stash — os
mesmos números que os logs `expert_streaming req` imprimem por request, agora no
card "Active Models" da WebUI e do app Swift. `null`/ausente = modelo não faz
streaming.

Edge cases documentados: `dynamic=true` com `budget=0` não arma o governador
(page-cache-only é decisão operacional; o hint da UI avisa); mudança de política
com modelo carregado dispara reload via runtime signature (padrão dos IO knobs);
cold tier aceita 2–8 bits no PUT (o núcleo já aceitava, a UI alinhou).

## References

- slipstream thesis + measurements: per-layer cache slots, 6.25 % hot-expert locality, decode attention near roofline, per-layer CPU wake floor.
- colibri expert atlas: routing heat is measurably structured and therefore cacheable.
- Paper survey & gap analysis (2024–26 MoE-offloading literature vs this implementation, prioritized next levers): [expert-streaming-papers.md](expert-streaming-papers.md).
