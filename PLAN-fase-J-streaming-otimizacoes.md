# Plano Fase J — Otimizações de latência do streaming MoE (SSD) — v2 auditada

**Ação 0 (ao aprovar):** gravar este documento integralmente em
`PLAN-fase-J-streaming-otimizacoes.md`, na raiz do repo
(`/Users/errepe/WorkBuddy/Worktrees/omlx/fork-feature-expert-streaming-61637226/`).
Artefato de trabalho, não commitado. O conteúdo final com números medidos vai para
`docs/expert-streaming.md` no C13.

Esta v2 incorpora uma auditoria contra o código real. Os 13 commits originais
permanecem, mas **4 blocantes e 14 divergências** foram encontrados; o delta está
na seção 1. Tudo marcado como **[NOVO]** ou **[REVISADO]** difere do plano v1.

---

## 1. Auditoria: blocantes e divergências

### 1.1 Blocantes (resolvidos por decisão do usuário)

| # | Plano v1 dizia | Realidade verificada | Decisão |
| --- | --- | --- | --- |
| **B1** | `rtk .venv/bin/python …` | Worktree **não tem `.venv`**. O único venv utilizável é `/Volumes/SSD 4TB/DEV/omlx/.venv` (py 3.13.13, mlx 0.32.0, mlx_vlm 0.6.3, mlx_lm 0.31.3, pytest+ruff), e nele `omlx` está **editable apontando para o outro worktree** (`feat/ane-oproj-moe-swiglu`). Testei o `sys.path` exato de `python bench/bench_expert_streaming.py`: `import omlx` resolve para `/Volumes/SSD 4TB/DEV/omlx/omlx` — **o bench rodaria o branch errado**. | Reusar o venv com `PYTHONPATH=$PWD` + preflight que aborta se `omlx.__file__` não for deste worktree. (O finder editable é *appended* ao `sys.meta_path`, então cwd/PYTHONPATH vence — verificado.) |
| **B2** | Gate de bit-exatidão por `token_ids` | `bench/bench_expert_streaming.py:298` faz `getattr(out2,'completion_tokens') or getattr(out2,'token_ids')`; `completion_tokens` é um **int truthy**, então `isinstance(_ids, list)` falha e grava `"token_ids": null`. **Todos os 100+ arquivos em `bench/results/` têm `token_ids: null`** — o critério de sucesso #1 é vacuidade silenciosa. O campo real é `GenerationOutput.tokens: List[int]` (`omlx/engine/base.py:166`). | **C0 novo**, antes do M0: gravar `out2.tokens` + `text`, fail-high se nulo. |
| **B3** | A8: `mx.clear_cache()` por camada no qwen e no glm | O braço qwen é **`qwen4_exp`** (`config.json`: `model_type=qwen4_exp`, `Qwen4ExpForConditionalGeneration`; mlx_vlm 0.6.3 upstream **não** tem `qwen4_exp`, logo usa-se o vendor tree). `Qwen4ExpDecoderLayer` **não lê `_stream_eval`**; o wrapper `qwen35_stream_eval.py` patchea `Qwen3_5MoeDecoderLayer`, classe que o qwen4_exp não instancia. O lado glm (`glm5_next/language.py:863-866`) **é real**. | **C9 escopo revisado**: threshold-gate só no glm. No qwen mantém-se `mx.eval(out)` e documenta-se a inércia. Remove-se a expectativa "C9 → TTFT 2k". |
| **B4** | Gates: `available ≥ 22 GiB`, `--min-free-gb 22`, swap 0,8 GiB | `available ≈ 21,5 GiB` → **toda run aborta no preflight**. Swap em uso agora: **5,6 GiB**. O gate #3 precisa ser delta por run, não absoluto. `docs/expert-streaming.md:349` registra a mesma lição (override para 20 liberou a run). | `available ≥ 20 GiB`, `--min-free-gb 18`, `--mem-ceiling-gib = available − 6`, swap medido por run com abort em delta > +1 GiB. |

### 1.2 Mapa de paths corrigido

| v1 | Caminho real |
| --- | --- |
| `omlx/expert_streaming/` | **`omlx/patches/expert_streaming/`** |
| `streaming_switch.py` | `omlx/patches/expert_streaming/streaming_switch.py` (1118 linhas) |
| `shard_bank.py` | `omlx/patches/expert_streaming/shard_bank.py` (557) |
| `warmer.py` | `omlx/patches/expert_streaming/warmer.py` (545) |
| `prefetch.py` | `omlx/patches/expert_streaming/prefetch.py` (133) |
| `tools/qwen35_stream_eval.py:63-64` | `omlx/patches/expert_streaming/qwen35_stream_eval.py:63-64` (87) |
| `omlx/batch_generator.py:3282` | `omlx/patches/mlx_lm_mtp/batch_generator.py:3282` |
| `omlx/models/deepseek_v32.py:709-716` | `omlx/patches/glm_moe_dsa/deepseek_v32.py:704-716` |
| qwen4_exp `language.py:986` | `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py:986` |
| `glm5_next:863-866` | `omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py:863-866` |
| `omlx/scheduler.py` | ✅ correto (`_should_release_streaming_pool` em 2306) |

### 1.3 Divergências por achado

| ID | Veredito | Correção no plano |
| --- | --- | --- |
| **A2** | PARCIAL | `bytearray()` dupla-cópia existe **só** em `expert_slice` (179). `expert_run` (216-246) já usa `os.pread` + `frombuffer` com offset — uma cópia a menos. |
| **A12** | ✅ + | Metadados re-derivados confirmados; o loop `n_elements` (164-166) só alimenta a checagem de invariante (168). `_proj_keys` **não** está em `shard_bank.py` — é `warmer.py:68`. |
| **A3** | ✅ exato | Scan O(N) em 362-369 confirmado. Extras: `_global_cap` (323) é código morto; **não há lock** na classe; os pokes `cache._store.pop` em **779/781** não decrementam `_layer_counts` → drift de contagem (evicção prematura). |
| **A4** | ✅ + pior | LRU guarda o **numpy cru** (930-931) → um **hit** também paga `_promote_np` + `mx.stack`. Comentário in-repo (924-929) registra 37 GB Metal active com LRU(mx) — experimento já feito. |
| **A5** | ✅ exato | 1072-1081; cada linear termina com `list(io_pool.map(...))` = barreira cheia → 3 barreiras/camada (2 com gate_up fundido). |
| **A1** | ✅ | `maybe_seed` (418-441): budget>0 → `_seed_lru` (449-476) **síncrono** (3 preads seriais QD1 por expert); budget 0 → `_seed_page_cache` (já async). **O stall só existe em budget>0** → C5 mede no braço B. |
| **A6** | ✅ | `_submit` 117-173; o trabalho vai ao `_WARM_POOL` (145) — o custo na thread de inferência é jobs listcomp + closure + `submit`, 48×/token. |
| **A7** | ✅ | `_seed_page_cache` 478-515: 1 task serial, ~2 GiB, nunca usa `load_expert_run`. |
| **A11** | ✅ bug real | `from .kernels import fast` (1106) → `omlx.patches.expert_streaming.kernels`, **inexistente**; `except Exception: pass` (1110) engole. Fix: `omlx.patches.glm_moe_dsa.kernels` (existe, `fast` na linha 64). |
| **A9a** kwargs | ❌ | São **8** kwargs (`ProfileAccumulator.add`, 978-988), nenhum dict é construído, e `add()` já curto-circuita. **Remover do escopo.** |
| **A9b** perf_counter | ❌ | Só existe em `StreamingSwitchLinear` (bf16, 579/581). O caminho quantizado (o dos modelos oQ4e) não tem. **Remover do escopo.** |
| **A9c** triple-hash bf16 | ❌ | Não reproduz: `_slice_dtypes` é cached (666-673); `_promote_np` faz 1 cmp de str + 1 de dtype. **Remover.** |
| **A9d** hasattr | ✅ + bug | `_slice_dtypes_lazy` (667) custa 0,274 µs no miss. **E atribuir uma tupla faz o `nn.Module.__setattr__` colocá-la na árvore de parâmetros** → polui `parameters()`/`tree_flatten`. Promovido a correção + higiene. |
| **A9e** bias | ⚠️ | Mecanismo real (`mx.stack([self._bias[int(e)] …])`, 595-597/973-975), mas **`plan.uniq_mx` não existe** no repo — precisa ser criado em `_RemapPlan` (447-462). |
| **A13** | ✅ | `np.unique` em 483-487. |
| **A14** | ❌ inerte | `queue.put(timeout=0.05)` (73), `deque.remove` (84) e `task_done` (119) são reais, mas o **prefetcher nunca é alimentado**: `OMLX_EXPERT_STREAMING_PILOT` default 0 e **nenhum caller de `submit` existe no repo**. Rebaixado a higiene. |
| **A10a** scheduler | ⚠️ | O sentinel `_streaming_guard_info` **já existe** (3859) e já é usado em 3871/4520/4742 — só a linha **2324** re-resolve. E **`deep_reset()` (12149) não invalida** → após reload o valor congela. Callers quentes: 5394 (chunk) e 11982 (token). |
| **A10b** mx.eval MTP | ⚠️ | Confirmado em 3282, mas dentro de `_run_verify_cycle_legacy` (3216) que tem **zero cobertura** (testes só exercitam `_run_verify_cycle_chain`). Remover troca 1 sync batelada por 2 → pode ser **negativo**. Medir antes. |
| **A10c** qwen4_exp HC | ❌ | **Não existe**: nenhum símbolo `HC`, nenhum `shape[1] > 1` no qwen4_exp; `_target_verify_linear` vem do mlx-vlm upstream (fora do repo). **Remover do escopo.** |
| **A10d** PLE `copy=True` | ⚠️ | `np.array(x)` já é `copy=True` por default → remover o kwarg é literalmente **no-op**. A mudança que economiza é `np.asarray(...)` (segura: fancy indexing já devolve array novo). PLE = tabela de embedding N-gram, não experts. |
| **A10e** router_eligible | ❌ | É `omlx/patches/qwen35_moe_router.py:134` (qwen35, não qwen4_exp), aplicado sem gate de família a `Qwen3_5MoeSparseMoeBlock`; inerte em linhas de prefill. **Remover do escopo.** |

### 1.4 Fatos de ambiente (verificados)

- `pytest.ini`: `pythonpath=.`, `asyncio_mode=auto`, `addopts = -v --tb=short -m "not slow and not integration"`. `pyproject.toml:294` também define `[tool.pytest.ini_options]` sem markers/addopts → **`pytest.ini` vence**; invocações que o contornem perdem o filtro.
- `ruff`: line-length 88, `E,F,W,I,N,UP,B,SIM`, ignore `E501,B905`, **`extend-exclude = ["omlx/patches/*/vendor"]`** → edições em glm5_next/qwen4_exp **não são cobertas pelo lint gate**.
- `mx.get_cache_memory()` **existe** no mlx 0.32; docstring: *"memory not currently used that has not been returned to the system allocator"* = **free pool**. Pendência #1 do v1 resolvida.
- `--prompt-len` aceita `short|512|2k|8k`; `--model` aceita `qwen|glm|dsv4`; `--decode` default 96.
- `bench/` não tem `__init__.py` → invocar como script, não `-m`.

---

## 2. Ambiente e preflight obrigatório **[NOVO]**

```
cd /Users/errepe/WorkBuddy/Worktrees/omlx/fork-feature-expert-streaming-61637226
export VENV="/Volumes/SSD 4TB/DEV/omlx/.venv/bin/python"
export PYTHONPATH="$PWD"     # OBRIGATÓRIO: sem isso o import cai no branch errado

# preflight — aborta se não estiver rodando o código deste worktree
rtk "$VENV" -c "import omlx,sys;p=omlx.__file__;print(p);\
sys.exit(0 if 'fork-feature-expert-streaming-61637226' in p else 1)" || exit 1
```

Testes: `rtk "$VENV" -m pytest tests/test_expert_streaming.py tests/test_cold_tier.py -x -q`
Lint: `rtk "/Volumes/SSD 4TB/DEV/omlx/.venv/bin/ruff" check omlx/patches/expert_streaming/ omlx/scheduler.py`

---

## 3. C0 — corrigir o gate de bit-exatidão **[NOVO, entra antes do M0]**

`bench/bench_expert_streaming.py:296-306`:

- gravar `out2.tokens` (campo real `List[int]`) ao lado de `text`;
- **fail-high**: abortar a run se `tokens` vier nulo/vazio;
- usar `--out-dir` **distinto por braço** — os arquivos `{model}_{budget}g_output.json` e `_samples.json` colidem entre braços com mesmo modelo+budget.

Sem o C0, o critério de sucesso #1 não existe.

---

## 4. Commits

Cada commit entra somente com: testes verdes (mapa abaixo), `ruff check` limpo
(ressalva: vendor fora do ruff), gate de bit-exatidão (`tokens` idênticos ao baseline).

**Suítes por arquivo:** `expert_streaming/*` → `tests/test_expert_streaming.py` (1868 linhas, 61 testes) + `tests/test_cold_tier.py` (184) · `scheduler.py` → `tests/test_scheduler.py:3420/3434/3457` · `mlx_lm_mtp/batch_generator.py` → testes MTP · vendor qwen4_exp → `tests/test_mlx_vlm_qwen4_exp_compat.py` + `tests/test_vlm_qwen4_exp_loader.py` (⚠️ v1 citava `test_mlx_vlm_qwen4_exp_loader.py`, que **não existe**) · vendor glm5_next → `tests/test_mlx_vlm_glm5_next_compat.py` (único; **não cobre `_stream_eval`**).

### C1 — `fix(shard_bank): preadv zero-copy + per-key read params` (A2+A12)

`omlx/patches/expert_streaming/shard_bank.py`

- `_ReadParams` NamedTuple + `self._rp: dict[str,_ReadParams]` em `_ShardReader` (126). `header` é imutável pós-`__init__` → sem invalidação. Checagem de size-mismatch vai para o precompute (levanta `ValueError` uma vez por key).
- `_read_into(off, out_u8)`: `os.preadv` (disponível; short read devolve menos bytes, pós-EOF devolve 0) com fallback `pread`+`frombuffer`. **OSError preservado.**
- `expert_slice` (153-182) e `expert_run` (216-246) reescritos; `expert_run` segue devolvendo N views sem cópia (buffer único vivo pelas views).
- Colapsar resolves duplicados: `load_expert` (487-489), `pin_expert` (531-539, 2× `_reader_for_key`), `advise_expert_run` (513-529, 2× `expert_byte_range`).
- Guardas: `memoryview` exige buffer escrevível (usar `np.empty`, nunca `np.frombuffer` no caminho preadv); `per_expert_bytes % itemsize != 0` → fallback.
- Testes novos: bytes/shape idênticos ao caminho antigo; run == slices; size-mismatch → `ValueError`; short read → `OSError`.

### C2 — `perf(streaming): bank-first demand assembly` (A4; depende C1)

- **Decisão de design: manter numpy no LRU** (opção i). Rationale: memoizar o banco pela tupla de uniques (opção ii) tem hit rate ~0 em decode (a tupla muda todo token) e o cache de mx promovido (opção iii) já foi medido no repo: 37 GB Metal active, guard matou o 2º prefill (comentário em 924-929).
- Bank `(U, per_expert_bytes)` uint8 por projeção; workers `preadv` direto no slot (novo `ExpertBackingStore.read_expert_into` → `_ShardReader.expert_slice_into`); hits copiam para o slot; **1** promoção por projeção via `_np_to_mx`.
- `legacy = True` se qualquer bundle for `mx.array` (collapse 770-785, dict-backed 838-847, prefetch-staged) — nunca misturar caminhos. `U == 1` mantém `expand_dims`. Teto `U * per_expert_bytes > _BANK_MAX_BYTES` (default 256 MiB) → legacy (limita prefill U≈288).
- **Expectativa honesta:** miss path 4→1 cópia (vitória em b0); hit path 2→2 cópias, mas U−1 alocações Metal a menos (neutro/pequeno ganho em b3/b4).
- Testes: bank vs stack por-slice em safetensors sintético (`mx.array_equal`); fallback mx; roundtrip do cache. Preserva `test_np_to_mx_bf16_preserves_bits:699` (**não mexer em `_np_to_mx`**).

### C3 — `perf(cache): O(1) per-layer LRU eviction` (A3)

`streaming_switch.py`, `ExpertLRUCache` (301): OrderedDict por camada + OrderedDict global (ordem cross-layer) + `_layer_counts` (preserva assert em `tests/test_expert_streaming.py:1218`). `put` (350-378): eviction O(1) via `popitem(last=False)` da fila da camada.

- Novo `discard(key)` substitui os pokes `_store.pop` de 779/781 → **corrige o drift de `_layer_counts`**.
- `threading.Lock` (pré-req C5). `_global_cap` (323) morto → remover ou documentar. `retain_hot` (385-407) passa pelo lock e mantém o rebuild de counts (401-405).
- Testes novos: `test_expert_lru_per_layer_cap_evicts_oldest_of_same_layer` + `test_expert_lru_counts_bounded_after_raw_store_pop` (hoje **zero** cobertura de `_per_layer_cap`).

### C4 — `perf(cache): skip LRU fills during prefill` (A3b; depende C3)

Gate no `__call__` dos linears (`plan.positions > 64`), **NUNCA** dentro de `_load_expert_bundle` (o seeder depende dos puts). Só com seed ativo (`cache.prefill_bypass`); `misses` continua contado.

### C5 — `fix(warmer): async LRU hotness seed` (A1; depende C3)

`warmer.py`: `retain_hot` + coleta síncrona de missing; leituras async no `_WARM_POOL` com runs coalescidos (`load_expert_run`); puts destravados pelo lock. `async_seed=True` default; testes existentes com `False`.
⚠️ Só afeta budget>0: com budget 0, `maybe_seed` (427) já chama `_seed_page_cache`, que é async.

### C6 — `perf(glu): single pooled I/O batch per MoE layer` (A5; depende C2) — invasivo, último entre os estruturais

- `_LayerLoadContext` em `_RemapPlan` (campo `ctx`); a 1ª projeção resolve a união das 3 e emite **1** `io_pool.map` coalescido; as demais consomem `ctx.raw`.
- Preservar: ladder por projeção (`cache.get` → `pf.take`+`_promote_np` na thread de inferência → pooled read → `_load_expert_bundle` síncrono; **só o passo 3 é compartilhado**); inserção no LRU por projeção em ordem de `plan.uniq_list`, após a barreira; ordem dos hooks (`on_layer_start` 1059 antes, `on_layer_plan`/trace depois); atributos lidos em runtime pelo vendor glm5_next (`_expert_streaming_profile`, `gate_proj/up_proj/down_proj`, `bundle_key`, `_load_expert_np`); contrato weighted-sum (1104-1111). `p.add(load=…)` conta o map **uma vez**.
- Kill-switch `OMLX_EXPERT_STREAMING_LAYER_BARRIER=0`.
- **Plano B (abort isolado)** se: hit_rate mudar com budget fixo; teste de bit-exatidão falhar; RSS de prefill +>5%; `sync_ms_per_load` cair <15% em QD16.
- Testes: contar chamadas de `io_pool.map` (1, não 3); bit-exact vs legacy.

### C7 — `perf(warmer): off-thread submit path` (A6)

Precompute de `keys_by_layer` no `__init__` (`_proj_keys`, warmer.py:68, chamado em 124); 48 submits → 1/token. Modo read (`OMLX_EXPERT_STREAMING_WARM=1`): runs como tasks no pool. Modo advise: só precompute. Sub-ruído no default; efeito real no braço WARM.

### C8 — `perf(warmer): coalesced paced page-cache seed burst` (A7)

`_seed_page_cache` (478-515): eids → runs → `load_expert_run`; 1 task por camada (paralelismo 4); teto `SEED_GIB` mantido. É o caminho **budget 0** → medir no braço A.

### C9 — `perf(stream_eval): threshold-gated clear_cache` (A8) **[REVISADO — só glm]**

- `glm5_next/language.py:863-866`: manter `mx.eval(out)` (load-bearing); `mx.clear_cache()` só se `mx.get_cache_memory() >= THRESH`. Semântica confirmada: **free pool**. Default `max(memory_limit/3, 2 GiB)`; env nova `OMLX_EXPERT_STREAMING_CACHE_THRESH`. Cobrir **os dois** branches do FFN (compile guard em 851 e eager).
- `qwen35_stream_eval.py:63-64`: manter `mx.eval(out)`; **documentar a inércia** (wrapper patchea `Qwen3_5MoeDecoderLayer`; o braço qwen usa `Qwen4ExpDecoderLayer`). Sem expectativa de delta no qwen.
- Testes com monkeypatch de `get_cache_memory`. Métrica: **TTFT nos braços GLM**.

### C10 — `chore(streaming): micro-opts` (A9, A13, A14 reduzidos)

Escopo final: `_PROFILE_ENV` hoisted (nome real, linha 29) com gates nos call sites · `np.unique` size-switched (483-487) · bias via `plan.uniq_mx` (**criar** em `_RemapPlan`, 447-462) + `mx.take` (595-597, 973-975) · `_slice_dtypes` no `__init__` **com `object.__setattr__`** (evita poluir a árvore do `nn.Module`) · scratch `_RemapPlan`; `getattr profile` 1× · `_group_runs` max via `OMLX_EXPERT_STREAMING_RUN_MAX` (nova, default 16) · `prefetch.take` O(1) e `put_nowait`/`task_done` — **higiene**, o prefetcher não é alimentado hoje.
**Removidos** (não verificaram): 8 kwargs de profiling, perf_counter por expert, triple-hash bf16, hoist do swiglu. Split em 4 sub-commits.

### C11 — `chore: micro-opts (integração)` (A10 revisado)

- **scheduler**: usar o sentinel `_streaming_guard_info` na linha **2324** (igual a 3871) **e** adicionar `self._streaming_guard_info = None` em `deep_reset()` (12173) — sem isso o reload congela o valor. Sub-commit separado (production-critical). Testes: `tests/test_scheduler.py:3420/3434/3457`.
- **PLE**: `qwen4_exp/language.py:986` → `np.asarray(...)` (remover só o `copy=True` é no-op). Testes: `test_mlx_vlm_qwen4_exp_compat.py:636/671`.
- **MTP** `mlx_lm_mtp/batch_generator.py:3282`: **medir antes de remover** (1 sync batelada → 2). Se remover, adicionar teste — `_run_verify_cycle_legacy` (3216) tem zero cobertura hoje.
- **Removidos**: qwen4_exp HC e cache de `router_eligible`.

### C12 — `fix(glm_moe_dsa): streaming weighted-sum fast path` (A11 — corretude)

`streaming_switch.py:1106` → import absoluto `omlx.patches.glm_moe_dsa.kernels` (módulo existe, 67 linhas, `fast` na 64; hoje o relativo aponta para `omlx.patches.expert_streaming.kernels`, inexistente, e o `except Exception: pass` de 1110 engole).
⚠️ `_FastDispatch.__getattr__` cai para `mx.fast` e `has()` cai para `hasattr(mx.fast, name)` → consertar o import habilita o caminho mesmo sem a extensão nativa (mudança além do import). Caller `deepseek_v32.py:704-716` confia sem fallback de ndim.
Gate: **teste unitário novo**; manter `tests/test_glm_moe_dsa_patch.py:903` verde. Único commit que muda output de propósito, fora dos benches.

### C13 — `docs: audit + Fase J`

Inserir `## Fase J` **antes de `## References`** (linha 516 de `docs/expert-streaming.md`, 520 linhas). Corrigir snippet stale em **449-452**: `--prompt-tokens 2048` → `--prompt-len 2k`; `--min-free-gib 22` → `--min-free-gb 22`. Conteúdo: achados, commits, números medidos, **tabela de rejeitados** (v1 §3 + a tabela 1.3 desta v2), trade-off do C4, e o hazard venv/`PYTHONPATH`.

---

## 5. Campanha de benchmark

**Comando canônico** (sempre com `PYTHONPATH` e `--out-dir` próprio):

```
# A: qwen, budget 0
OMLX_EXPERT_STREAMING_PROFILE=1 rtk "$VENV" bench/bench_expert_streaming.py \
  --model qwen --budget 0 --prompt-len 2k --decode 48 \
  --min-free-gb 18 --mem-ceiling-gib <available-6> \
  --out bench/results/faseJ/A_<fase>.json --out-dir bench/results/faseJ/A
# B: idem --budget 3 --out-dir .../B
# GLM (marcos): --model glm --budget {0,4} --prompt-len 2k --decode 16 --out-dir .../glm_b{0,4}
# Braços: OMLX_EXPERT_STREAMING_WARM=1 ; OMLX_EXPERT_STREAMING_PILOT=1
```

**Gates de memória (revisados):** ① `psutil.available ≥ 20 GiB` antes de cada fase (era 22) — abaixo, PARAR e avisar. ② `--mem-ceiling-gib = available − 6`; `--min-free-gb 18`; budgets ≤3 (qwen) / ≤4 (glm). ③ `sysctl -n vm.swapusage` antes/depois de **cada** run, abortar em **delta > +1 GiB** (absoluto não serve: já há 5,6 GiB em uso). ④ Sequencial, page cache quente, conferir processos ao fim.

**Protocolo:**

- **Probe pré-M0 — resolvido por leitura do código:** `ttft_s` = `engine.chat(max_tokens=1)` completo → **inclui o 1º forward de decode**, então o stall do seed aparece em ttft ✅. `decode_s` **NÃO** é um chat cacheado: é um **segundo prefill completo** + N tokens, com `n = completion_tokens or decode` (truncado por EOS) → **`tok_s` é diluído** e o gate de no-regressão sobre ele é menos sensível. Greedy confirmado: `temperature=0.0` (271/277) → `mx.argmax`; MTP/ANE/SpecPrefill off.
- **M0:** qwen A,B,B,A (mediana) + glm b0/b4; congelar `tokens` (após C0).
- **Por commit:** testes+ruff → 1 run B; C4/C8/C9 também 1 run A (C9 só glm). Delta no ruído (~±5%) → repetir antes de claim. C10/C11: no-regressão por desenho, sem claim de vitória.

**Métrica→commit:** C1→`sync_ms_per_load` · C2→`stack_ms` · C3→evictions/TTFT b3 · C4→TTFT b3 · C5→ttft chat#1 (**braço B apenas**) · C6→`load_ms`/`sync_ms_per_load` · C7→braço WARM · C8→tokens iniciais (**braço A**) · C9→**TTFT GLM** · C10/C11→no-regressão · C12→teste unitário.
⚠️ `sync_ms_per_load` é alimentado **só** no caminho quantizado (`add_load_source` em 933; ausente em `StreamingSwitchLinear` 564-611) → confirmar no M0 qual classe os modelos oQ4e usam; se for a bf16, a métrica é 0.0 e C1/C6 precisam de proxy alternativo (`wall_ms`/`load_ms`).

**Custo:** ~22 runs qwen (1–4 min) + 4 glm (5–10 min) + 2 braços ≈ 1,5–2 h de bench + implementação/testes.

---

## 6. Ordem e dependências

`C0 → C1 → C2 → C3 → C4 → C5 → C7 → C8 → C9 → C10 → C12 → C11 → C6 → C13`

- C0 antes de tudo (senão não há gate). **C6 por último entre os estruturais** (plano B: abort isolado). C11 dividido em sub-commits por subsistema, scheduler por último. C12 depois do C10 (hoist). C13 fecha com números.
- Dependências: C2←C1 · C4/C5←C3 (lock) · C6←C2 · C8←C7 · C9/C10 independentes.

---

## 7. Critérios de sucesso

1. `tokens` idênticos em toda run — e o campo é **real** após o C0 (fail-high). C12 gated por teste unitário, fora dos benches.
2. Zero runs com delta de swap > +1 GiB.
3. Preflight de branch (`omlx.__file__`) verde em toda run.
4. Quedas mensuráveis: `sync_ms_per_load`/`stack_ms` (C1/C2), TTFT budget-3 (C3/C4), fim do stall do 1º token em b3 (C5), TTFT GLM (C9).
5. Nenhuma regressão >5% em `tok_s`/`ttft_s` — com a ressalva de que `tok_s` é diluído pelo re-prefill.
6. C13 publicado com números + tabela de rejeitados.

## 8. Riscos e pendências

- **C2 neutro em b3/b4** (hit copia para o slot) → aceitar se b0 ganhar e b3/b4 não regredirem; memo de bank fica como follow-up.
- **Box compartilhado**: números são machine-state-dominated (lição da Fase G) → delta <5% sempre repetir.
- **C12 muda comportamento** (habilita fast path) → se o teste mostrar divergência frente ao caminho não-streaming, é correção: documentar no C13.
- **Venv do outro worktree**: se for recriado via `uv sync`, mlx-lm/mlx-vlm (pins git) podem resolver revisões diferentes e o M0 perde comparabilidade. Smoke de import + 1 teste rápido antes do M0.
- **Pendência restante:** confirmar no M0 se os modelos oQ4e usam `StreamingQuantizedSwitchLinear` (proxy de métrica de C1/C6) e parametrizar `async_seed` sem quebrar os asserts existentes de `PrefillHotnessRecorder` (`maybe_seed` 418-441 é o único entry point).
