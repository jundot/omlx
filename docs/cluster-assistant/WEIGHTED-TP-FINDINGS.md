# Weighted Tensor Parallelism — verification report (2026-11-15)

Read-only verification of `omlx_weighted_tp_konzept_v1_2.md` and its PoC bundle
against **this** repository, the **installed** MLX stack, and the upstream
sources it cites. No production code was changed. Nothing here reopens a frozen
decision: R1 is PP only and the assistant still refuses `tensor_parallel_size != 1`
(`omlx/cluster/assistant/election.py:841`). Turning any of this into a shipped
feature needs an explicit R2 decision.

**Inputs verified**

| Input | Version / pin |
|---|---|
| Repo | `rebase/upstream-main`, HEAD `be4e980d` |
| MLX | `0.32.4` (`uv run python -c "import mlx.core"`) |
| MLX-LM | `0.32.3.dev8+g0d4d3d04f` |
| PoC bundle | `omlx_weighted_tp_poc_v1_2.zip` → `poc_weighted_tp.py`, `audit_weighted_tp.py` |
| Concept | `omlx_weighted_tp_konzept_v1_2.md` (444 lines, rev 1.2) |

**Evidence grades** (kept from the concept): *Verified* = read in source or
executed here. *Reported* = upstream/model-card claim, not reproduced.
*Design* = proposal, no implementation. *Open* = correctness on real hardware
still unproven.

---

## 1. Five findings that change the plan

### 1.1 The concept's upstream research is accurate. Its picture of *this* repo is not.

Every upstream claim I could check is correct, down to the merge date
(§2 below). What is wrong is the assumption that symmetric tensor parallelism
does not exist here yet, and therefore that weighted TP is greenfield work.

In this build, tensor parallelism is a **shipped, user-selectable product
feature**. The cluster wizard offers `auto | tensor | pipeline`
(`apps/omlx-web/omlx_web/static/js/cluster_v2.js:530`), the strategy card
already renders `1/{count} of every layer`
(`cluster_v2.js:266`), the planner builds real `tensor_parallel_rank` groups
(`omlx/cluster/planner.py:2721-2846`), the worker shards layer by layer while
streaming weights (`omlx/cluster/progressive_loading.py:247-296`), the runtime
validates TP groups and rejects incomplete ones
(`omlx/cluster/runtime.py:170-186`), and the collective backend is chosen
automatically (`jaccl` only if `rdma_ctl` is enabled on **all** members, else
`ring` — `omlx/cluster/backends.py:75-111`).

The consequence: **weighted TP is not a new subsystem, it is one allocator and
the removal of a handful of divisibility assertions.** The repo already does
contiguous, unequal, group-aligned splitting and sums it back correctly:

```python
# omlx/cluster/tensor_strategies.py:277-293
def _uneven_group_ranges(total_groups: int, size: int) -> list[tuple[int, int]]:
    """Contiguous per-rank ``[lo, hi)`` group ranges covering ``total_groups``.
    When ``total_groups`` is not divisible by ``size`` the first
    ``total_groups % size`` ranks receive one extra group."""
```

and its docstring on `_shard_switch_mlp_uneven` (line 299-322) states the exact
invariant weighted TP needs:

> The recombining `all_sum` in `_wrap_sharded_moe` is shape-agnostic, so unequal
> per-rank widths sum back to the full result exactly (verified to fp noise on
> mlx 0.31.x).

That is the mechanism. It currently distributes the **remainder** (+1 group to
low ranks), not a **weight ratio**, and it is wired into exactly one adapter —
`_shard_switch_mlp_uneven` is called only from `_shard_nemotron_h`
(`tensor_strategies.py:620`). `_shard_glm5_mlp` still calls plain
`shard_inplace` (line 675-680), i.e. an even split. Replacing
`_uneven_group_ranges(groups, size)` with a weight-driven
`_weighted_group_ranges(groups, weights)` and routing the GLM-5.3 MoE through it
is the whole idea in one function.

### 1.2 MLX's `segments=` looks like the uneven API and is not one — this is measured

MLX 0.32.4 does have an uneven-sharding knob, so the concept's line "`sizes=`
ist PR-Entwurf, nicht garantiert in Main" needs sharpening, but its conclusion
holds:

```
shard_linear(module, sharding, *, segments: Union[int, list] = 1, group=None)
shard_inplace(module, sharding, *, segments: Union[int, list] = 1, group=None)
```

`segments` is **group boundaries plus an even split inside every group**
(`_shard` → `_split` in `mlx/nn/layers/distributed.py`: a list of ints is passed
straight to `mx.split`, i.e. it is *split points*, and each resulting part is
then divided by `N`). Executed here with a two-rank group:

| `segments` on D=24 | rank 0 width | rank 1 width | ratio |
|---|---|---|---|
| `[12]` | 12 | 12 | 1.00 |
| `[4, 8]` | 12 | 12 | 1.00 |
| `[16]` | 12 | 12 | 1.00 |

Every configuration yields exactly `D/N`. `segments` can keep a fused QKV
projection inside head boundaries — this repo uses it that way
(`tensor_strategies.py:870`: `segments=[q_dim, q_dim + kv_dim, q_dim + 2*kv_dim]`)
— but it cannot give one rank twice the work.

The weighted `sizes=` **is** still open: MLX PR #4465 (state `open`, labels
`await verification` + `low priority`, last commit 2026-09-17 adding exactly
"a list of sizes in unpacked elements"). Independent confirmation from a third
party in the MLX-LM #1863 thread: a downstream fork reverted #1863 because
"sizes= … the MLX the fork runs on (0.32.x) doesn't take".

**So: weighted sharding must be done in oMLX (explicit contiguous slices), not
by waiting for MLX.** The good news is that oMLX already does it that way for
MoE and it works.

### 1.3 "Not divisible by three" is a category error — and it hides the real constraint

The document's headline ("Ein Modell muss nicht in jeder
Konfigurationsdimension durch 3 teilbar sein") argues against a filter nobody
has to write. The design has `world_size = 2` with *weights* `(2, 1)`
(§0.1 of the concept says so itself). Two physical ranks means the divisibility
question is `% 2`, which almost every modern dimension satisfies.

This repository's gate is exactly that, and it is correct:

```python
# omlx/cluster/planner.py:2780-2791
if tensor_parallel_size > 1:
    incompatible = tuple(
        value for value in model.tensor_parallel_divisors
        if value % tensor_parallel_size
    )
```

The **real** constraint is not divisibility but **alignment**: every cut must
land on a quantisation group / whole KV-head group / kernel-accepted width, and
that cut is then *unequal* on purpose. The document identifies this correctly
in §1.2 and §3.3; it just frames it as a thirds problem.

Practical value of the reframing: `Qwen3-32B` at TP=2 does not need
`25600 % 3`; it needs a cut at a multiple of 128 near `25600·2/3`, which is
`17024/8576` — the document's own number, reached via the correct criterion.

### 1.4 Two of the six PoC targets cannot be tensor-parallelised in this build at all

Measured against the real capability gates (`registered_model_types()` and
`native_shard_is_layer_local()`), today:

| `model_type` | oMLX adapter | native `shard()` passes the layer-locality proof | TP possible now |
|---|---|---|---|
| `qwen3` (Qwen3-32B) | — | yes | **yes** |
| `llama` (Llama 3.3 70B) | — | yes | **yes** |
| `qwen3_moe` (Qwen3-235B) | — | yes | **yes** |
| `glm5_next` (GLM-5.3-Flash) | yes | n/a | **yes** |
| `glm_moe_dsa` (GLM-5.3 big) | — | yes | **yes** |
| `qwen3_next`, `qwen3_next_moe` | yes | n/a | **yes** |
| `qwen3_5`, `qwen3_5_moe` | yes | n/a | **yes** |
| `nemotron_h`, `ministral3`, `muse_spark` | yes | n/a | **yes** |
| `deepseek_v41` | — | **no** — "native shard mutates model state outside its layer loop" | **no** |
| `qwen4_exp` (Qwen3.8) | **no** | **no** — the vendored patch ships `model.py` with no `shard()` | **no** |

This directly reorders the roadmap in the concept:

- **DeepSeek V4.1 (concept P4) is blocked by this repo's own guard, not by
  MoE offload.** `omlx/patches/deepseek_v41/language.py:944` keeps the tensor
  group, but the model's `shard()` mutates state outside the layer loop, so
  `apply_tensor_strategy` would refuse it. Making P4 real means writing a
  layer-local adapter (as `glm5_next` has), not "using PR #5086's optimised
  path" — #5086 optimises the collective/quant path *after* sharding, which is
  unreachable while sharding is refused.
- **Qwen3.8-Flash-Next (concept P5) has no TP path at all** in this build. It
  is PP-capable here (upstream #5033 applied) and it runs single-node, but it
  cannot be sharded. It is genuinely a *new adapter*, exactly as §3.2 warns,
  and it is the last step, not a benchmark you can lean on early.

The concept's own conclusion ("Qwen3.8 … keine geeignete schnelle Abkürzung")
survives; what changes is that DeepSeek V4.1 is in the same category, not one
step easier.

### 1.5 The `43/21` head split in §1.2 is wrong for both Dense references

The document contradicts itself, and §1.2 is the wrong half.

- §1.2 table: "64 Heads → 43/21" as a generic answer.
- §3.1/§3.2: Qwen3-32B and Llama 3.3 70B have 8 KV heads with 8 Q heads each,
  so an exact, layout-safe split is **whole KV groups 5/3 → Q 40/24**,
  attention share 62.5/37.5.

Both configs are verified: `num_attention_heads=64`, `num_key_value_heads=8`.
`43/21` is not expressible in whole KV groups and would break the local
head-to-KV mapping. **Use 40/24 for the Dense references.**

`43/21` is correct for the other two: GLM-5.3-Flash has
`num_attention_heads=64` and `num_key_value_heads=64` (one KV head per Q head),
and DeepSeek V4.1 uses MLA with `num_key_value_heads=1` and
`head_dim=512`, so heads are independent and 43/21 is fine.

---

## 2. Upstream findings: all confirmed, with one addition

Checked 2026-11-15 via the GitHub API, not by reading prose.

| Claim in concept | Verdict | Evidence |
|---|---|---|
| MLX #4465 uneven sharding **open** | **Verified** | `state=open`, `merged=false`, labels `await verification`, `low priority`; the `sizes=` commit is 2026-09-17 |
| MLX-LM #1863 depends on #4465, **open** | **Verified** | `state=open`, `merged=false`, description says "should not be merged until that PR is merged" |
| #1863 distributes *whole KV groups*, spreads the remainder, is **not** a weight planner | **Verified** | PR body: `_rank_sizes(dim, N, block=1)`, "distribute whole KV-head *groups* … rather than splitting q_proj's and k_proj's/v_proj's output features independently" |
| MLX-LM #1816 already merged | **Verified** | `merged=true`, `merged_at=2026-09-08T18:09:12Z` |
| oMLX #5086 merged 2026-11-10 | **Verified** | `merged=true`, `merged_at=2026-11-10T06:29:09Z` |
| exo #953 open | **Verified** | `state=open`, 16 comments, label `enhancement` |
| exo fork commit `884c867` real, asymmetric | **Verified** | "feat: asymmetric tensor parallelism for heterogeneous clusters", 2026-03-31, 6 files, `asymmetric_parallel.py` +346, tests +104 |
| `segments`/`sizes` not usable for weighting | **Verified** | §1.2, measured |
| JACCL = TB5 RDMA, multi-cable, `MLX_METAL_FAST_SYNCH` unreliable | **Verified** | current MLX 0.32.4 distributed docs, incl. the deadlock reference to mlx#3142 |

The addition matters for §9 of the concept ("any-to-any install and start"):
the MLX docs state RDMA must be enabled **in macOS Recovery** by running
`rdma_ctl enable`, and that this *cannot be done remotely even with sudo*.
`backends.py` already models this as a per-member `jaccl_ready` flag and falls
back to `ring` for the whole cluster — which is correct and fail-safe, but it
means "the assistant picks JACCL for you" is bound to a manual, one-time,
physical step on every Mac. Any plan that assumes a TP path will be JACCL
because the hardware is TB5 is optimistic by one recovery boot per machine.

### Model data: the numeric matrix is accurate

Verified against the live `config.json` files and the HF blob API.

| Checkpoint | Concept says | Measured | Verdict |
|---|---|---|---|
| `TensorFold/Qwen3.8-Flash-Next-MLX-4bit` | 111.58 GB / 103.91 GiB | 111.58 GB / 103.92 GiB, 22 shards | **Verified** |
| `mlx-community/GLM-5.3-Flash-4bit` | 190 GiB | 189.98 GiB / 203.99 GB, 62 shards | **Verified** |
| `mlx-community/Qwen3-32B-4bit` | 18.4 GB | 18.43 GB / 17.16 GiB | **Verified** |
| `mlx-community/Llama-3.3-70B-Instruct-4bit` | 39.7 GB | 39.69 GB | **Verified** |
| `mlx-community/Llama-3.3-70B-Instruct-bf16` | 141 GB | 141.11 GB / 131.42 GiB | **Verified** |
| `mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit` | 250 GB | 249.79 GB | **Verified** |
| `bearzi/DeepSeek-V4.1-Flash-1017-oQ4e-mtp` | 402.4 GiB payload | 402.36 GiB, 82 shards | **Verified** |
| DeepSeek "288 GiB resident" | 288 GiB | — | **Reported**, unverifiable offline (depends on Engram SSD offload actually working) |

Dimension claims also hold:

- **Qwen3-32B** — hidden 5120, 64 heads, 8 KV, MLP 25600, 64 layers. `%3`
  irrelevant; aligned 2:1 cut = **17024/8576** (133 + 67 groups of 128).
  *Verified.*
- **Llama 3.3 70B** — same head layout, MLP 28672 → aligned 2:1 =
  **19072/9600**. `Verified`.
- **GLM-5.3-Flash** (`glm5_next`) — hidden 4096, 64 heads, **64 KV heads**,
  dense 12288, `moe_intermediate_size` 2048, 288 experts, top-8, 45 layers +
  1 nextn, `q_lora_rank` 1536, KDA 64 heads. Dense 2:1 = 8192/4096 *exact*;
  MoE at group 64 = 21/11 → **1344/704**; experts 192/96 *exact*.
  `Verified`.
- **DeepSeek V4.1** — hidden 5120, 64 heads, **1 KV head** (MLA,
  `head_dim` 512), MoE 2304 → **1536/768 exact**, 384 experts, top-6,
  `q_lora_rank` 1280, `o_lora_rank` 1024, `o_groups` 8, 40 layers, Engram at
  layers `[1, 14]`, `num_nextn_predict_layers` 3, DSpark. `Verified`.
- **Qwen3-235B** — 64 heads, **4 KV heads**, dense 12288 → 8192/4096 exact,
  MoE 1536 → 1024/512 exact, 128 experts, 94 layers. GQA gives only 3:1
  (48/16) or 1:1 (32/32), as stated. `Verified`.
- **Qwen3.8-Flash-Next** (`qwen4_exp`) — 24 heads → 16/8, 48 linear-value
  heads → 32/16, 16 linear-key heads and **2 KV heads** not 2:1, 512 experts
  top-10, `moe_intermediate_size` 640 (group 32 → 416/224), 48 layers, 4:1
  linear:full attention ratio, `ple_layer_ids [2]`, `mtp` block declared in
  config with 1 layer. `Verified`.

One nuance on MTP: the `Qwen3.8` **config declares** an MTP block, while the
`TensorFold` Q4 conversion ships **no mtp-named file** (same for
`Qwen3-32B-4bit` and `GLM-5.3-Flash-4bit`, even though the latter's config sets
`num_nextn_predict_layers: 1`). So "this checkpoint has no MTP head" is
consistent with the file listing but not proven by it — MTP weights can live
inside the regular shards. Treat as **Reported**, and check the actual
`model.safetensors` key list before planning any MTP tuning on those
checkpoints.

---

## 3. What already exists here (the porting surface)

Everything the concept proposes to create mostly exists; what is missing is the
weighting and the head-level cut.

| Concept proposal | Reality in this repo | File |
|---|---|---|
| New `weighted_tp_plan.py` | Not needed as a separate object; there is already `ModelLayout` carrying `tensor_parallel_heads`, `tensor_parallel_kv_heads`, `tensor_parallel_divisors`, `kv_replicated_across_tp`, `supports_tensor_parallel` | `omlx/cluster/planner.py:97-140` |
| `apply_tensor_strategy(..., tp_plan=None)` | Signature confirmed: `(model, group, *, mx_module, progress)` — no plan object. An optional `weights=` keyword is the minimal extension | `tensor_strategies.py:886-892` |
| Adapter registry | Exists, 8 adapters, with a capability gate that refuses unsafe native `shard()` | `tensor_strategies.py:23-85`, `:780-812` |
| "GLM5_NEXT weighted path" | Adapter exists and is concrete; it is exactly where `rank * width` has to become a weighted offset | `tensor_strategies.py:683-762` |
| Generic quantised slice helper | Exists in practice, hand-rolled per adapter with scales/biases handled explicitly | `tensor_strategies.py:296-350` (MoE), `:569-578` (Mamba) |
| Quant-aware `segments` for fused tensors | Already used | `tensor_strategies.py:870`, `omlx/patches/deepseek_v4/deepseek_v4_model.py:2553` |
| Ring vs JACCL switchable, ring first-class | Exists, chosen per cluster from per-member `jaccl_ready` | `omlx/cluster/backends.py:75-111`, `omlx/cluster/rdma/` |
| Plan + rank mapping distributed identically | Exists: `tensor_parallel_size` / `tensor_parallel_rank` per assignment, validated for completeness and contiguity | `omlx/cluster/runtime.py:126-186` |
| Layer-at-a-time shard-then-free loading | Exists, with explicit peak-RAM reasoning | `omlx/cluster/progressive_loading.py:240-296` |

The `rank * width` sites that must become cumulative weighted offsets, all in
`omlx/cluster/tensor_strategies.py`:

| Lines | Adapter | What breaks under 2:1 |
|---|---|---|
| 394-414 | `qwen3_next` | fused QKV `mx.arange(rank*k, (rank+1)*k)`, `A_log`, `dt_bias` |
| 544-560, 587-609 | `nemotron_h` | attention/Mamba packed offsets, conv channel windows |
| 602, 620-621 | `nemotron_h` | `start = rank * heads`; MoE switch MLP |
| 704-730 | `glm5_next` linear layers | `_require_divisible(num_heads, size)`, `rank * width` conv window, `mx.split(A_log)[rank]`, `qkv_dim`, `conv_dim`, `groups` |
| 732-738 | `glm5_next` full attention | `_require_divisible`, `mx.split(embed_q/unembed_out)[rank]` |
| 801 | `_shard_vocab_head` | `_require_divisible(vocab_size, size)` |

Those two `_require_divisible(attention.num_heads, size, …)` calls at 704 and
732 are the hard blockers for 43/21 or 40/24 on GLM-5.3-Flash: with 64 heads
and size 2 they pass, so the adapter *runs* — but it silently produces a 32/32
split, which is the 50:50 assumption the concept wants to dissolve.

---

## 4. The actual code delta for 2:1

Four changes, in dependency order. Sizes are for a two-rank weighted plan.

**A. One allocator, one place.** Replace the remainder-spreading
`_uneven_group_ranges(total, size)` with a weight-driven sibling and make the
existing MoE path use it:

```python
def _weighted_group_ranges(total_groups, weights, align=1):
    """Contiguous per-rank [lo, hi) covering total_groups, sized by weights.

    Every boundary is a multiple of `align`; the residual (at most
    len(weights)*align/2 groups) goes to the rank with the most headroom.
    """
```

Then `_shard_switch_mlp_uneven` (line 328) becomes
`lo, hi = _weighted_group_ranges(groups, weights, align=1)[rank]`, and
`_shard_glm5_mlp` (line 675) is switched from `shard_inplace` to the same helper
(it already has a quantised `scales`/`biases` pair to slice). This single change
makes the MoE tensors 2:1 **correctly and completely**, because the recombining
`all_sum` is already shape-agnostic. Concrete effect for GLM-5.3-Flash:
32 groups → today `16/16`, weighted `21/11` → 1344/704 neurons.

**B. Thread the weights through.** `apply_tensor_strategy(model, group, *,
weights=None)`; `progressive_loading.py:274` passes
`assignment.tensor_parallel_weights` from the plan. `None` must reproduce
today's behaviour bit-for-bit — that is the regression contract for every
symmetric deployment.

**C. Head-level cuts.** For each adapter, replace
`_require_divisible(heads, size)` + `rank * width` with
`_weighted_group_ranges(heads_or_kv_groups, weights, align=1)` and keep the
GQA invariant: **split whole KV groups, never Q heads independently**
(`q_per_kv` constant on every rank). For the Dense references that is 5/3 KV
groups → 40/24 Q heads.

**D. Price memory per rank, not per average.** Today both the weight and the
KV estimate assume an even split:

```python
# omlx/cluster/planner.py:2875
per_member = stage_layer_bytes // tensor_parallel_size
# omlx/cluster/planner.py:2579 / 2594
return total // max(1, tensor_parallel_size)
```

Under 2:1 these over-reserve the small rank and under-reserve the large one —
which is precisely the failure mode the concept's §6.1 rollback criterion names
("Rang 1 benötigt mehr als sein praktisch verfügbares RAM-Budget"). Both
formulas need the per-rank weight, and `sharded_weight_bytes`
(`planner.py:2925`) becomes a per-assignment number.

The plan hash already covers `tensor_parallel_size` (`planner.py:2956`), so a
drifted weight vector must be added to it; otherwise `plan_mismatch` cannot
detect a changed ratio.

---

## 5. PoC bundle: reproduced, and shown to be blind to its own thesis

**Ran, on this machine, with the project venv** (2 ranks, one host):

```
$ mlx.launch --backend ring -n 2 -- python poc_weighted_tp.py
rank=1/2 slice=[256:384] width=128 ... max_abs_error=4.7683716e-07
rank=0/2 slice=[0:256]   width=256 ... max_abs_error=4.7683716e-07
```

So the algebraic claim — column-parallel `gate`/`up`, local activation,
row-parallel `down`, one `all_sum`, unequal widths, identical output shape — is
**Verified** at fp32. That is a real, reproducible P1 result and the document is
right to start there.

Two structural limits follow from the same run:

1. **It cannot test the alignment case.** Line 25 rejects
   `hidden % sum(weights) != 0` — it enforces the very `% 3` filter §3.3
   argues against — and it has no alignment parameter. Executed:
   `--hidden 99 --in-dim 8` slices `66/33`, which is unaligned to any
   `group_size` of 32/64/128, and still reports
   `max_abs_error=9.3132257e-10`. A green PoC therefore says nothing about
   quantised kernels, scale offsets, or head groups.
2. **It has no quantisation at all**, so P2 ("quantisierte, gemischt breite
   Projektionen") is entirely unwritten, and that is where the risk lives: the
   scales axis is the quantisation-group axis and must be sliced at group
   boundaries, as `_shard_switch_mlp_uneven` already does
   (`scales[..., lo:hi]`, `weight[..., plo:phi]` with
   `packed_per_group = weight.shape[-1] // groups`).

The companion `audit_weighted_tp.py` is honest about being dimension-only; note
it reports `num_key_value_heads` with `align=1`, so for Qwen3-32B it prints a
`5/3` KV cut *and separately* a naive head cut, and a reader can mistake the
latter for the answer. It should emit only the whole-group answer when
`num_attention_heads % num_key_value_heads == 0`.

---

## 6. Recommended order, reordered by what is actually shardable

The concept's P0/P1 are sound. From P2a onward the order has to follow §1.4.

| Step | Target | Why here | Status |
|---|---|---|---|
| P0 | Freeze pins, measure PP baseline at batch 1, MTP off; 1-link ring microbenchmarks at 4/16/64/256 KiB/1 MiB | Unchanged, and cheap | — |
| P1 | 2-rank `all_sum` algebra | **Done and reproduced above** (`4.8e-07`) | Verified |
| P2 | Quantised uneven slices: BF16/4/5/6/8-bit, group 32/64/128; assert scales/biases land on group boundaries and peak RAM really drops | This is the real gate; the algebra is already proven | Open |
| **P2a** | **`glm5_next` first, not the Dense references** | It is the only target with a concrete oMLX adapter *and* exact 2:1 dimensions (dense 12288→8192/4096, MoE 2048→1344/704, experts 288→192/96) *and* it already carries the whole unequal-MoE mechanism next door in `nemotron_h` — switching `_shard_glm5_mlp` to it is a few lines. | Design |
| P2b | Qwen3-32B Q4, then Llama 3.3 70B BF16 | Still the best *Dense* correctness references (whole KV groups 5/3 → 40/24), and they exercise the native `shard()` path rather than an adapter, which is a different code path from `glm5_next` | Design |
| P3 | Qwen3-235B (`qwen3_moe`, native shard) | MoE 1536→1024/512 exact; GQA 4 KV heads forces 3:1 or 1:1 for attention — accept and log that per-module ratios differ | Design |
| P4 | DeepSeek V4.1 | **First write a layer-local shard adapter**, or the current guard refuses it. Engram offload must work before the first 288 GiB load | Blocked |
| P5 | Qwen3.8-Flash-Next | **Write `shard()` from scratch**; PLE tables, Gated DeltaNet state, 2 KV heads, indexer, N-gram/hyperconnections. Last, as the concept says | Blocked |

Two additions I would insert:

- **P2c — KV-cache and long-context correctness under asymmetric heads.** The
  concept's test matrix has this as a row; under 2:1 it is a distinct failure
  mode (rank 1 holds fewer heads but the same sequence). `kv_replicated_across_tp`
  already distinguishes MLA from GQA — verify it still means what it says under
  weighting.
- **P0b — one measured collective baseline for `ring` and, if RDMA is enabled,
  `jaccl`.** Because `backends.py` falls back for the *whole cluster* when one
  member lacks RDMA, and because RDMA needs a recovery-mode boot per Mac, the
  planning assumption should be `ring` until proven otherwise.

---

## 7. Corrections to apply to the concept document

| § | Claim | Correction |
|---|---|---|
| Title / §3 | "Durch 3 teilbar" as the central obstacle | Not a constraint: `world_size=2`. The constraint is alignment, and it is already implemented |
| §0.5 / §4.1 | oMLX has 50:50 assumptions and no TP plan object; `apply_tensor_strategy` has no plan parameter | Half right: no plan parameter (correct), but symmetric TP is a shipped feature with 8 adapters, a planner, a runtime group validator and a JACCL/ring selector. The delta is an allocator, not a subsystem |
| §1.2 table | `64 Heads → 43/21` | Wrong for the Dense references; whole KV groups give **40/24**. `43/21` is right only for MHA/MLA models (GLM-5.3-Flash 64/64, DeepSeek V4.1 MLA) |
| §2.2 | `sizes=` is a PR draft, not in main | Still true, and now sharpened: 0.32.4 ships `segments=`, which is group-aware but **even** — measured 12/12 for every segment list. Do not mistake it for the weighted API |
| §3.1 | DeepSeek V4.1 is a "second/third end-to-end adapter" | It is **blocked** in this build: the native `shard()` fails the layer-locality guard. It needs an adapter first |
| §3.1 | Qwen3.8 "guter Betriebs-Benchmark, aber schwieriger erster TP-Adapter" | Understated: there is no TP path at all (`qwen4_exp` has no `shard()` in the vendored patch and no adapter) |
| §5 P2a | Dense references first | Keep them as *Dense* references, but the cheapest real 2:1 win is `glm5_next`, whose MoE machinery already exists |
| §4.3 | Backend is selectable and ring stays first-class | Confirmed, and stronger than stated: selection is automatic per cluster and fail-safe (`jaccl` only if every member has RDMA) |
| §4.3 / §9 | JACCL availability treated as a hardware property | Add: RDMA is enabled **in macOS Recovery** and cannot be done remotely; assume `ring` for planning until verified per Mac |
| §6 | Memory peaks "pro Rank nachweisen" | Correct, and currently impossible: the planner prices weight and KV per *average*, so a 2:1 plan would under-reserve rank 0 |
| §7 | `audit_weighted_tp.py` proposes candidate boundaries | Fine, but it prints a naive head cut next to the GQA-safe one; gate it so only the whole-group answer is shown for GQA models |
| PoC §P1 | The script demonstrates the 2:1 slice | It rejects non-3-divisible widths and ignores alignment; `--hidden 99` proves it cannot detect an unaligned cut |

---

## 8. What is still genuinely open

Unchanged from the concept, plus two new ones:

1. Do the real fused kernels accept 1344/704 and 416/224 with matching scale
   slices? (`_shard_switch_mlp_uneven` proves the *pattern* at group boundaries;
   it does not prove fused-MoE kernel width acceptance.)
2. GQA: whole KV groups (40/24) vs independently split Q heads with replicated
   KV. The repo's invariant is whole groups; the alternative is untested.
3. Engram / N-gram / PLE tables: replicate, SSD-map per node, or partition.
4. Fused kernels that bypass `ShardedLinear` and call quantised matmul directly
   — `_shard_glm5_next` already documents exactly this problem (line 690-691).
5. MTP: can reference and draft stages share one shard plan? (And: which
   checkpoints actually ship an MTP head — see §2.)
6. 2:1 is a starting point, not an optimum; per-module ratios may differ.
7. `all_sum` is the only collective needed for pure TP; uneven `all_gather`
   (vocab-split head, logits) is a separate audit — and `_shard_vocab_head`
   requires exact divisibility today.
8. Loading without duplicates: `progressive_loading` already shards a layer then
   frees it; **new** — that path must be re-timed for weighted shards, because
   the large rank materialises more per layer.
9. **New:** an asymmetric plan invalidates the "every member holds 1/N" story in
   the UI (`cluster.v2.split.tensor_share: '1/{count} of every layer'`) and in
   the memory card. A weighted plan needs to show *which* share each Mac holds,
   or the card lies.
10. **New:** the assistant refuses TP by design (`election.py:841-849`,
    `routes.py:1843`). Any weighted-TP product surface is an R2 decision, not an
    implementation detail.

---

## 9. Bottom line

The concept is upstream-accurate and its algebra is reproducible — I ran the PoC
and it holds. Its two weak points are (a) it treats this repository as if
symmetric TP did not exist, when in fact the unequal-group-plus-shape-agnostic-
`all_sum` mechanism weighted TP needs already runs here (for Nemotron-H's MoE,
one helper call away from GLM-5.3-Flash), and (b) it undersells how many of its
own targets are currently unshardable: DeepSeek V4.1 is refused by oMLX's own
layer-locality guard and Qwen3.8 has no `shard()` at all.

The honest one-sentence plan: **one weight-driven group allocator + one optional
`weights=` argument + per-rank memory pricing, proven first on
`glm5_next`'s MoE (which needs no new adapter), then on the two Dense references
with whole-KV-group 40/24 — and treat DeepSeek V4.1 and Qwen3.8 as adapter work,
not as benchmarks.**

No frozen decision was changed by this report. TP remains R2; R1 stays PP-only.

---

## 10. Sources

**Local, executed**
- `omlx/cluster/tensor_strategies.py:277-350, 683-762, 786-805, 886-892`
- `omlx/cluster/planner.py:97-140, 780-870, 2560-2600, 2721-2791, 2845-2960`
- `omlx/cluster/runtime.py:100-186`, `omlx/cluster/backends.py:1-111`
- `omlx/cluster/progressive_loading.py:240-296`
- `omlx/patches/qwen4_exp_mlx_lm/__init__.py:29` (registers the module, ships no `shard()`)
- `omlx/patches/deepseek_v41/language.py:944-964`
- `mlx 0.32.4`: `mlx/nn/layers/distributed.py` (`shard_linear`, `_shard`, `_split`)
- `mlx.launch --backend ring -n 2 -- python poc_weighted_tp.py` (and `--hidden 99`)

**Upstream, checked 2026-11-15**
- MLX #4465 — `open`, `await verification`, `low priority`
- MLX-LM #1863 — `open`, depends on #4465; #1816 — `merged 2026-09-08`
- oMLX #5086 — `merged 2026-11-10`
- exo #953 — `open`; fork commit `884c867` (2026-03-31)
- MLX distributed docs, 0.32.4 build: JACCL, recovery-mode `rdma_ctl enable`,
  multi-cable minimum rule, `MLX_METAL_FAST_SYNCH` deadlock (mlx#3142)

**Model data**
- HF `config.json`: `Qwen/Qwen3-32B`, `Qwen/Qwen3.8-Flash-Next`,
  `zai-org/GLM-5.3-Flash`, `deepseek-ai/DeepSeek-V4.1-Flash-1017`,
  `Qwen/Qwen3-235B-A22B-Instruct-2507`
- HF blob API sizes and file lists for the seven checkpoints in §2