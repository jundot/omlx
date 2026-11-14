# Upstream packages — how to offer this fork back (Phase I)

This is the **prep** for offering our work to upstream. It is a plan, not a
promise.

**Measurement basis (re-cut 2026-11-14):** `git diff --numstat upstream/main..HEAD`
on `rebase/upstream-main`, where `upstream/main` = `624cff0e` **is an ancestor of
HEAD**. The old pin `1e27576a` was *not* an ancestor of upstream/main, so the old
numbers mixed our work with upstream drift. They no longer do: this diff is
exactly our net work on top of current upstream, and every package below is
listed against the tree that would be pushed.

Numbers are the **committed** tree (`HEAD` = `1b8fc777`, 2026-11-14). Re-run the
same `numstat` command before cutting a PR; later commits are docs only.

Read with the rebase: [`REBASE-2026-11-14.md`](REBASE-2026-11-14.md).

Goal: **independent packages, each with standalone value**, offered cheapest-first
(**discussion → issues → PRs**). A package is worth opening alone only if it
helps an oMLX user who does not want the Cluster Assistant.

---

## 1. Measured areas

206 files, `+133 887 / −65 714` (199 601 changed lines) — but **132 873 of those
lines are reordering churn** in three generated files, not work:

- `apps/omlx-mac/Resources/Localizable.xcstrings` — 78 914 lines (`39 499/39 415`)
  because the catalogue is now **upstream-preferred** and our merge rewrote its
  key order. Real delta: **+28 string entries**.
- `apps/omlx-web/omlx_web/i18n/*.json` (11 locales) — 53 957 lines, same reason.
  Real delta: **+185 keys per locale**, and **no key removed** (every file is a
  strict superset of upstream).
- `tailwind.css` — 2 lines (`build_css.py` output).

Content (churn removed) is **≈66 728 lines**.

| Area | Files | + | − | Total | Notes |
|---|---|---|---|---|---|
| `omlx/cluster/assistant/` | 24 | 19 344 | 0 | 19 344 | the assistant backend, all new |
| `tests/cluster_assistant/` | 53 | 17 763 | 0 | 17 763 | fake-driven suite |
| `omlx/cluster/` (non-assistant) | 9 | 1 018 | 96 | 1 114 | planner, routes, discovery, inventory, staging, probe, deployment |
| `omlx/admin` + `engine_pool` + `server` + `settings` | 4 | 143 | 21 | 164 | `local_only`, router mount, probe switch |
| `apps/omlx-web` | 35 | 38 982 | 26 099 | 65 081 | −53 957 i18n churn → **11 122 content** |
| `apps/omlx-mac` | 24 | 41 151 | 39 437 | 80 588 | −78 914 xcstrings churn → **1 652 content** |
| `tests/` (non-assistant) | 17 | 1 180 | 63 | 1 243 | cluster + engine-pool tests |
| `docs/` | 33 | 12 544 | 0 | 12 544 | `cluster-assistant/**` |
| other | 6 | 1 071 | 0 | 1 071 | `AGENTS.md`, `CLUSTER-ASSISTANT.md`, `scripts/check_peer_versions.py`, dotfiles — split into P2/P3/P6 below |

---

## 2. The packages (by independence and risk)

Sizes are the **content** delta on the rebased tree (churn excluded), with the
old-pin measurement for comparison. "Tests" are counted in the package size.

| # | Package | Scope | New size | Old pin | Independent? | Why it has standalone value |
|---|---|---|---|---|---|---|
| P1 | **`local_only` model load** | `omlx/engine_pool.py`, `omlx/admin/routes.py`, `tests/{test_engine_pool,test_admin_load_errors,test_cluster_engine_pool}.py` | **293** (107 code, 164 tests, 22 deleted), 5 files | 293 (identical) | **Yes** | `POST /api/models/{id}/load?local_only=true` loads a model on this Mac alone: a registry cluster deployment is ignored, an already-loaded distributed engine is unloaded (or refused when busy) instead of being reported as this Mac's own, and the PLE/Engram table is priced into local admission. Generic fix for anyone who wants a single-Mac load. |
| P2 | **Test-infra fixes** | `tests/conftest.py`, `tests/_tokenizer_support.py`, `tests/test_glm5_next_mtp.py`, `tests/{test_scheduler,test_settings}.py`, `.editorconfig`, `.gitignore`, `.python-version` | **161**, 8 files | 159 | **Yes** | The `conftest` name-collision fix (`MockTokenizer` → uniquely named module) and the native-indexer skip keep the suite stable; no product behaviour. |
| P3 | **Peer version check** | `scripts/check_peer_versions.py`, `docs/cluster-assistant/PEER-VERSIONS.md` | **209** (141 tool, 68 doc), 2 files | 209 (identical) | **Yes** | A read-only tool that detects a mixed-version cluster (405/404 on the assistant operations/tuning routes). Useful even without the assistant; it sends no key and writes nothing. |
| P4 | **macOS helper / DMG fixes** | `apps/omlx-mac/**` (Helper plists + `main.swift`, `Scripts/{build,package_dmg,provision_runtime}.sh`, `PerformanceScreenVM`, `PythonRuntime`, `WelcomeWindow`, menubar, mac tests, +7 xcstrings entries), `docs/cluster-assistant/{HELPER,DISTRIBUTION,SIGNING,SLIM-RUNTIME}.md` | **2 380 + 7 xcstring entries**, 28 files | 2 402 | **Yes** | The SMJobBless helper could never install (launchd plist not embedded, trust requirements wrong) and the DMG was unsigned. Broadly useful to the app, independent of clustering. |
| P5 | **Cluster planner / resource-policy parity** | `omlx/cluster/{planner,routes,deployment,discovery,model_inventory,staging,performance_worker,discovery_routes,catalogue}.py` + 5 cluster test modules | **1 567** (1 114 core, 453 tests), 14 files | 1 253 | Mostly | Measured admission, memory tier, table (`engram`/Qwen4 PLE)-aware planning, **routed-expert (MoE) offload restored on upstream's refactored pipeline and now threaded through the coordinator plan path and the deployment decode**, discovery fallback that carries name/chip/RAM. Useful to the cluster core; the assistant consumes it but does not own it. |
| P6 | **Cluster Assistant** | `omlx/cluster/assistant/**` (19 344), `tests/cluster_assistant/**` (17 763), assistant web UI + its tests (11 122), `docs/cluster-assistant/**` minus P3/P4 docs (11 832), `AGENTS.md` + `CLUSTER-ASSISTANT.md` (908), `omlx/server.py` + `omlx/settings.py` mount (38), +185 i18n keys × 11 locales, +7 xcstrings entries | **≈63 070** (61 007 without the i18n/xcstring content), 132 files | 61 743 | No (feature) | The product feature. Needs a **discussion + design issue first**, then a staged PR series (measurement → election/plan → apply/start → tuning/run-local → UI). Depends on P1 (`local_only`) and P5. |

Notes:

- **P1, P2, P3 are the easy first offers**: small, self-contained, low-risk. P1
  and P3 replayed **byte-identically**; upstream's `engine_pool` refactor did not
  touch those lines.
- **P4** is app-maintainer territory; a clean bug-fix PR. Its delta *shrank* from
  the old pin (2 402 → 2 380 + 7 entries) because the string catalogue is now
  upstream-preferred: upstream added ~3 930 catalogue keys between the pins, so
  our fork-only entries are all that remain.
- **P5 grew 1 253 → 1 567 (+314)** — see §3. It is a real core improvement but may
  need a maintainer decision on the resource-policy model before a PR.
- **P6** is the reason for the fork, and P6 is also where the churn lives: raw
  60 868 added lines exclude ~2 063 lines of genuinely new localisation. Open it
  as a **discussion** first ("run a model that does not fit one Mac, 2–4 Macs over
  Thunderbolt, in the existing app"), then an issue with the design (measurement,
  election, PP-only, plan/apply/start, tuning presets, run-local), then PRs in
  that order. Do not open a single 60k-line PR.

---

## 3. What changed vs the old pin

| # | Change | Why it matters for the offer |
|---|---|---|
| P1 | **No diff change** (`97/16` engine pool, `10/3` admin routes — same as the old pin). | Upstream's `EnginePool` refactor did not touch the `local_only` seams; the replay was conflict-free. Still re-run its tests on the rebased tree (green). |
| P2 | +2 lines. | Trivial; the `conftest` fix still applies unchanged. |
| P3 | **No diff change.** | The tool probes `/admin/api/cluster/{stage,plan}` and `/admin/api/cluster/assistant/{operations/active,tuning-preset}`; nothing upstream moved. |
| P4 | xcstrings delta 84 lines → **+7 entries** (verified: keys only in ours; 0 changed, 0 removed); Swift/shell/docs unchanged (`+2 380`). | Do not send the 78 914-line catalogue diff — it is the same 1704 entries re-serialized. Cut P4 from our **7** entries on top of upstream's catalogue. |
| P5 | `planner.py` **109 → 253** added lines; `routes.py` 102/39 → 334/42; `deployment.py` +10. | Upstream **removed routed-expert (MoE) offload from the planner**. It is **restored on upstream's refactored pipeline** (`plan_hybrid`, `plan_unequal_pipeline`, `_finish_pipeline_plan`, `_held_layer_bytes`, `PipelineAssignment.moe_offload_*`) so the assistant's MoE lever and the context-vs-experts tradeoff work; the `()` default reproduces upstream exactly. The coordinator plan path is threaded too (`_saved_moe_offload`/`_moe_offload_layer_bytes`/`_moe_offload_for_plan` feed `_create_cluster_plan`, autoconfigure and `_build_performance_plan`; `deployment._assignment_from_dict` decodes the fields so `planned_weight_bytes` does not re-charge streamed experts). Also on top: Qwen4 PLE table detection (`_qwen4_ple_layer_ids`, `_is_qwen4_ple_tensor`), `_declared_context_tokens`, and `_saved_engram_ssd_offload` PLE parity in `routes.py`. |
| P6 | `cluster_v2.js` 3 532/9 → **3 962/132**; `policy.py` 1 553 → 1 574; assistant tests +39. | The assistant UI is our base again, with the upstream pieces the merged tests need ported in (`splitDevices`, `deviceHelperOnly`, `pairingStepOne`, `cudaGuide*`, helper-aware `restoreAccelerator`/`planNodes`/`inventoryHosts`/`versionMismatches`, helper pairing → `startAccelerator`, helper-aware `acceleratorRows`). Ports cut **both** ways, so a P6 PR must be split per upstream area, not dropped in whole. |
| P6 | **Not replayed: our own open PR #5033** (`omlx/patches/qwen4_exp_mlx_lm/`, `tests/test_qwen4_cluster_adapter.py`, `docs/experimental/qwen4-exp-cluster.md`). | These files never existed in upstream history; they came from our open PR #5033 via the old base pin. The rebase replays only our delta (`1e27576a..feat`), so they are gone. The Qwen4 **PLE table detection** survived in F5 (`_qwen4_ple_layer_ids`/`_is_qwen4_ple_tensor`), and upstream's `mlx_vlm_qwen4_exp_compat` is untouched. Consequence: no Qwen4-Exp **pipeline** across Macs on this branch (the Qwen3.8 path is single-node `run-local`). See §7. |

---

## 4. The new PRs (concrete cut)

Fixes first (independent), then the assistant as a **chain** — each step is green
and useful when it lands, but 2 and 3 need their predecessors. Note that the MoE
planner pricing the assistant needs lives in **open PR #5024**, not in F5 (see §7).

| # | PR | Contents | Code ≈ | Depends on |
|---|---|---|---:|---|
| F1 | `local_only` model load | `engine_pool`, `admin/routes`, its tests | 85 | — |
| F2 | Test infra | `tests/conftest.py` (MockTokenizer collision), `_tokenizer_support.py`, native-indexer skip, `.editorconfig`, `.gitignore`, `.python-version`. **Trimmable**: the two dotfiles are opinions upstream may decline; `test_settings.py`/`test_scheduler.py` belong to the feature PRs, not here. | 80 (tests) | — |
| F4 | macOS helper / DMG | Helper plists + `main.swift`, `Scripts/*`, `PerformanceScreenVM`, `PythonRuntime`, `WelcomeWindow`, menubar, mac tests, **+7 xcstring entries** | ~880 (+ tests) | — |
| F5 | **Cluster-core extras** (not in #5024) | `discovery` +183/−18 (fallback carries name/chip/RAM), `model_inventory` +86/−7, `staging` +71/−7, `performance_worker` +55/−7, `discovery_routes` +21, `catalogue` +5/−3, its tests | ~200 logical (421 raw) | — |
| 1 | **Assistant plans** | A1 measure (`measurement`, `state`, `calibration`, `models`, `/measure`, `/status`) + A2 recommend/plan (`recommendation`, `evaluation`, `election`, `policy`, `topology`, `/recommend`, `/plan`) + `search` + `DELETE /models/{node}/{model}` (the recommendation emits `free_disk_space`) + U1 (UI up to `planning`) + `CLUSTER-ASSISTANT.md` | ~5 600 | **#5024 (planner half), hard** |
| 2 | **Assistant runs** | A3 apply/start/stop (`provisioning`, `start`, `operations`, `/apply`, `/start`, `/stop`, `/session`, `/operations/*`) + A4 tuning (`tuning`, `tuning_store`, `redeploy_tuning`, `performance_store`, `/tune*`) + `diagnostics` (only start emits it), `network-preflight`, `performance-probe`, the read-only peer check (folded F3) + U2 (apply/ready/stop/tuning) | ~4 900 | PR 1 |
| 3 | **Run on one Mac** | `run-local`, `run-local/{id}/stop`, `tune-local` (+ statuses) and the small UI card | ~400 | PR 1 + 2 |

Order: **F1–F4 in any order → #5024 (rebased) and F5 → PR 1 → PR 2 → PR 3.** F2 may
ride along with F1. If F5 rides with the #5024 rebase, one track disappears.

Rules:

- Every PR must build and pass its own tests on `rebase/upstream-main`, alone.
- No PR may reference a route another PR adds (no dangling `error.action`): that is
  why `DELETE /models/...` is in PR 1 and `diagnostics` in PR 2.
- Never include generated churn: cut `Localizable.xcstrings` to its 28 entries,
  `i18n/*.json` to the keys a PR actually introduces, `tailwind.css` to its rebuild.
- `AGENTS.md` (705 lines) is fork-internal and is **not** offered. Most of
  `docs/cluster-assistant/**` is internal design documentation; offer only the
  user-facing part, with its feature PR.

PR 1 is a clean stop point: it measures, recommends, plans, shows the
context-vs-speed tradeoff and can free disk — it starts nothing.

## 5. Status (fill in as we go)

| # | Status | Upstream issue | Upstream PR | Notes |
|---|---|---|---|---|
| F1 | draft | — | — | re-cut on upstream/main; no conflict |
| F2 | draft | — | — | trimmable; two dotfiles optional |
| F4 | draft | — | — | app maintainers; cut the 7 xcstring entries only |
| F5 | draft | — | — | cluster-core extras (discovery/inventory/staging/probe/catalogue) |
| 1 | discuss | — | — | hard dependency on #5024's planner pricing |
| 2 | discuss | — | — | after 1 |
| 3 | discuss | — | — | after 1+2 |
| #5024 | updated `bbbf18d2` | — | #5024 | rebased; runtime half mirrored on the branch |
| #5033 | **open** (clean) | — | #5033 | re-apply onto the branch, or withdraw |

## 6. Hard prerequisites

- **No rebase needed any more.** `upstream/main` is an ancestor of HEAD, so the
  diffs in this file are directly comparable and PR branches can be cut from
  `rebase/upstream-main` as-is. Re-check `git merge-base --is-ancestor
  upstream/main HEAD` before cutting, and re-cut if upstream has moved.
- Each package must build and pass its own tests **on the rebased tree**, alone.
- Never include the generated churn: cut `Localizable.xcstrings`, `i18n/*.json`
  and `tailwind.css` down to their content delta (+7 entries, +185 keys per
  locale, rebuilt CSS), or the PR is unreviewable.
- Never include fork-only product decisions as if they were upstream defaults;
  call them out in the issue and let maintainers decide.

---

## 7. Our two open upstream PRs — updated 2026-11-14

Both were **open** in `jundot/omlx` and folded into the old base pin `1e27576a`.
Both are now **rebased onto `upstream/main` `624cff0e` and pushed**, so each PR
head is `upstream/main + PR content only` — no fork delta mixed in:

| PR | Verified state (2026-11-14) | What it contains | What our branch has | Action |
|---|---|---|---|---|
| **#5024** `feat(cluster-moe-offload)`<br>“carry MoE expert offload into pipeline ranks” | head **`bbbf18d2`**, 4 commits, **+619/−16, 16 files**, `mergeable: true` | planner +118, routes +127, deployment, autoconfigure, **inference_worker +23, progressive_loading +24, `deepseek_v41/loading.py`, `deepseek_v41_mlx_lm/model.py`, `moe_offload.py`**, docs, 6 test files | **Both halves.** Pricing was ported during the rebase; the runtime half was mirrored from `bbbf18d2` (commit `392aaccb`), so the branch matches the PR. | **Done** — rebased, pushed, mirrored. |
| **#5033** `feat/qwen4-exp-cluster`<br>“make Qwen4-Exp pipeline-parallel” | head **`25da70de`**, 8 commits, **+1689/−77, 15 files**, **`mergeable: true` / `clean`** | `qwen4_exp_mlx_lm` adapter (335), `pipeline_compat`, planner +233, routes, vendor `language.py`, `model_loading`, 3 tests (+804), doc | applied to `rebase/upstream-main` (the PR is the source of truth) | **Done** — ready to merge. |

Rebase evidence: `c74f1b18..624cff0e` contains **no** `omlx/cluster/` changes, so
both rebases were conflict-free. Focused tests in a clean worktree:
#5024 **358 passed**, #5033 **163 passed**.

Product decision recorded 2026-11-14: **PP only in R1 (Qwen4-Exp/Qwen3.8
included), general tensor-parallel support in R2.** The old note that the Qwen3.8
PLE VLM is “never planned as PP” described the pre-#5033 state and is withdrawn.

## 8. Prepared branches (local, committed, tested — ready to push)

Fixes are based directly on `upstream/main` `624cff0e` (`HEAD^ == 624cff0e`), so
each push is a clean single-PR branch. The assistant series is stacked on a local
integration base (§8.1). None is pushed yet.

| Branch | Commit | Files | +/− | Tests |
|---|---|---|---:|---|
| `pr/f1-local-only` | `bda778e0` | 5 | +271/−22 | 244 passed |
| `pr/f2-test-infra` | `c819ec74` | 3 | +74/−48 | 517 passed, 4 skipped; collection clean (19 969) |
| `pr/f4-mac-helper-dmg` | `be025eee` | 28 | +2380/−22 | 32 + 81 macOS, 13 python |
| `pr/f5-cluster-core` | `f62d50f0` | 11 | +888/−53 | 1614 cluster |
| `pr/a0-cluster-base` (not offered) | `34b2fc6c` | 47 | +4471/−258 | 295 focused (F1+F2+F5+#5024+#5033) |
| `pr/a1-assistant-plan` | `5b3cafae` | 62 | +22 884/−79 **vs a0** | 312 assistant + 473 web |
| `pr/a2-assistant-run` | `7e53f41b` | 83 | +23 315/−570 **vs a1** | 615 assistant + 547 web |
| `pr/a3-run-local` | `417ef95d` | 13 | +4268/−16 **vs a2** | 646 assistant + 559 web |

### 8.1 Why the assistant series has a local base

A fork PR can only be based on a branch in the **target** repo, so A1 cannot be
"based on A0" at GitHub. Instead:

- `pr/a0-cluster-base` exists only locally: `upstream/main` + F1 + F2 + F5 +
  #5024 + #5033, i.e. **all prerequisites** in one mergeable state. It is what
  the assistant needs to import and run today.
- A1/A2/A3 are stacked on it and contain **only assistant content** (verified:
  every diff is a subset of `omlx/cluster/assistant/**`, `tests/cluster_assistant/**`,
  `apps/omlx-web/**`, `CLUSTER-ASSISTANT.md`, `omlx/server.py`, `omlx/settings.py`
  and two assistant tests). No prerequisite code is duplicated or reverted.
- So they are opened **after** F5 + #5024 + #5033 land (the phased decision), at
  which point `pr/a0-cluster-base` is simply `upstream/main` and A1 rebases onto
  it unchanged in size. Opening them today would duplicate ~4.5k prerequisite
  lines in the PR diff.

### 8.2 Notes a reviewer needs

- F1 also carries the `force_lm`/VLM fallback guard (same bug: a cluster
  deployment leaking into a local load); its tests live in the same file.
- F2 is trimmed to the three real test-infra files (conftest collision,
  `_tokenizer_support.py`, native-indexer skip). The dotfiles (`.editorconfig`,
  `.python-version`) and the `.gitignore` change were dropped by decision; the
  root conftest still re-exports `MockTokenizer`, so existing call sites work.
- F3 (peer check) was **folded into A2** by decision: the script probes the
  assistant's operations/tuning routes, which only exist there. Its links to
  sibling cluster docs were removed so it stands alone inside A2.
- F4 adds **7** xcstrings entries to upstream's catalogue (the 78 914-line diff
  was a re-serialization of the same 1704 entries, not content).
- F5 extracts exactly one line from `omlx/cluster/routes.py` (the inventory call
  must use `DEFAULT_REMOTE_PYTHON`, not the hardcoded `~/omlx-distributed` path).
- A1 deliberately ends at the plan: the card has no start-setup button, no
  progress rows, no endpoint/session/stop/tuning — and
  `/admin/api/cluster/performance-probe` is allowlisted in the UI seam test with
  the reason that its caller ships with A2.
- A1 needs the F2 conftest fix because the nested
  `tests/cluster_assistant/conftest.py` shadows the root `conftest` module; that
  is why F2 is in the base and offered first.

### 8.3 Upload

```sh
# phase 1 — fixes, in any order
for b in pr/f1-local-only pr/f2-test-infra pr/f4-mac-helper-dmg; do
  git push origin "$b:$b"
done
# phase 2 — cluster core
git push origin pr/f5-cluster-core:pr/f5-cluster-core
# phase 3 (after F5, #5024, #5033 are merged) — rebase then push
git rebase --onto origin/main pr/a0-cluster-base pr/a1-assistant-plan
git push origin pr/a1-assistant-plan:pr/a1-assistant-plan
# then A2, A3 the same way (A2's base becomes A1, A3's base becomes A2;
# open all three as drafts and merge bottom-up)
```

## 9. Final PR count

- **7 new PRs**: F1, F2, F4, F5, A1, A2 (includes the folded peer check), A3.
- **2 existing PRs, both handled**:
  - **#5024** — rebased onto `624cff0e`, pushed (`bbbf18d2`), **both halves**
    mirrored on our branch (pricing in the rebase, runtime in `392aaccb`).
  - **#5033** — rebased onto `624cff0e`, pushed (`25da70de`), applied on our
    branch (`054f92e4`).
- Total tracks: **9** (7 new + 2 existing), plus the non-offered local base
  `pr/a0-cluster-base`.
