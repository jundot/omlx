You are continuing the oMLX Cluster Assistant project.

Working directory:
/Users/andreaskolb/Documents/PycharmProjects/omlx-cluster-assistant

Branch: feat/cluster-assistant
Base: 1e27576a (tag `cluster-assistant-base-2026-11-07`): #5024 + #5033 +
rank-0-over-SSH + launch reliability + the MoE fit hint. The branch was rebased
onto this base on 2026-11-07.
HEAD at handover: see `git log -1`; do not push.

Do not assume any prior conversation. Every decision and every constraint is in the repository.

---

## Read first, in this order

1. `AGENTS.md` §2 (resolved implementation decisions) and §10 (the 2026-11-07 product decisions).
2. `docs/cluster-assistant/ROADMAP-TO-CHAPTER-10.md` — the gate plan. In particular:
   - §0 recorded decisions,
   - §2 root cause of the resource-planning defect,
   - §3 gap inventory (R1–R7 are the first slice),
   - §4 Phase A (A1–A7),
   - §13 handover notes.
3. `docs/cluster-assistant/PROJECT-BRIEF.md` §5 (Screen 5/6), §8 (PP planning, “fewest Macs that fit”), §13 (contracts), §22 (resolved decisions).
4. `docs/cluster-assistant/SCREEN-5-HANDOFF.md`, `SCREEN-6-MANAGE.md`, and `PHASE-2B-RUN.md` §12–§13 (live evidence).
5. Code you will drive:
   - `omlx/cluster/assistant/election.py` → `elect`, `_plan_for_order`, `budget_for`, `_score`
   - `omlx/cluster/assistant/recommendation.py` → `_largest_context_plan`, the context ladder
   - `omlx/cluster/assistant/provisioning.py` → `plan_apply`, `build_deployment_request`
   - `omlx/cluster/assistant/measurement.py` / `state.py` → `NodeFacts`, `ModelLayout`
   - `omlx/cluster/planner.py` → `plan_hybrid`, `plan_unequal_pipeline`, `ModelLayout`
   - `omlx/cluster/routes.py` → `ClusterPlanNodeRequest`, `_saved_engram_ssd_offload`, `_moe_offload_for_plan`, `_build_performance_plan`
   - `omlx/model_settings.py` → `ModelSettings` (`qwen4_ple_ssd_offload`, `deepseek_v41_engram_ssd_offload`, `moe_expert_offload_enabled`, `moe_expert_offload_resident_fraction`, `max_context_window`)
   - `omlx/cluster/assistant/start.py` → `CoordinatorOps`, `observe_performance`, `ensure_chat_profile`
   - the card: `apps/omlx-web/omlx_web/static/js/cluster_v2.js` and `templates/dashboard/_cluster_v2.html`
6. Reference only (the “thin layer”): `~/Documents/PycharmProjects/localinferenz`.
   Read `console/planner.py`, `console/catalog.py`, `console/deploy.py`, `console/model_space.py`.
   Port its **planning logic and measured policy**; never its app, wizard, chat, profile system or German UI.

---

## Goal for this session

Execute **Phase A** of the gate roadmap: make the assistant’s resource planning
match the thin layer, then continue through the remaining P0/P1 gate items if
time permits.

**Do not start Chapter 10.** The gate is defined in
`ROADMAP-TO-CHAPTER-10.md` §1 and §11.

The two defects to close first:

1. **Too conservative resource planning.** The assistant’s election plans pure
   resident memory. It ignores:
   - Engram / ngram / PLE tables that can live on SSD,
   - MoE routed-expert offload and the resident fraction,
   - the memory-guard tier (safe / balanced / aggressive),
   - the lower context / smaller chunk that would make a model fit.

   The coordinator’s `/plan` already supports `engram_ssd_offload`,
   `moe_offload_layer_bytes`, `moe_expert_offload_resident_fraction` and
   `memory_guard_tier`. The election rejects the model before the coordinator is
   asked. Fix the election, not a second planner.

2. **The election always uses every measured Mac.** `elect()` treats the
   supplied list as the whole cluster. A 4-Mac measurement always becomes a
   4-rank plan, even when 2 Macs fit with one hop instead of three. Implement
   “fewest Macs that fit, then the best pair/set and least hops”, with the
   excluded Macs named.

---

## First slice — Phase A

### A1. Facts

- Add/extract the model facts the policy needs: `max_context_tokens`, per-layer
  table bytes (Engram/ngram/PLE), per-layer MoE expert bytes,
  `mtp_weight_bytes`, `supports_pipeline`, `pipeline_stage_starts`,
  `kv_bytes_per_token_per_layer`.
- Add each Mac’s saved `memory_guard_tier`, live admission p05/p50/p95 and
  `wired_limit_mb` to the measured node facts.
- Snapshot the saved per-model settings listed above.
- Fix the Qwen PLE / DeepSeek Engram parity in this checkout
  (`_saved_engram_ssd_offload` currently only honours the DeepSeek flag) or pass
  the flag explicitly so the coordinator does not have to re-derive it.

### A2. One policy object

Add a `ResourcePolicy` carried through measure → recommend → plan → apply →
start and into the plan signature:

```text
per node:  memory_guard_tier (safe|balanced|aggressive|custom)
per model: context_tokens
           chunk / prefill_step
           table_offload (engram | ple | none)
           moe_expert_offload: bool
           moe_resident_fraction: 0 < f <= 1
```

Rules (already decided):

- defaults come from the saved settings;
- every lever is **shown in the plan** and applied only on **Start setup**;
- never silently overwrite an explicit user setting;
- the same policy goes to plan/apply/start; a mismatch is `plan_mismatch`;
- the coordinator’s `/plan` and the election must use the same policy.

### A3. Election wiring and admission

- `budget_for(facts, rank, tier=...)` passes `memory_guard_tier` into
  `NodeBudget`.
- `_plan_for_order` passes `engram_ssd_offload`, `moe_offload_layer_bytes` and
  `moe_expert_offload_resident_fraction` into `plan_hybrid`.
- Compute the MoE layer bytes with the existing oMLX helper; do not duplicate
  the arithmetic.
- Admission: **p05 is the safety floor; p50/live admission plans by default.**
  p95 may be tried only for a huge model that cannot fit at p50, and only with
  an explicit risk confirmation. Expose safe/balanced/aggressive visibly.

### A4. The “bring this model up” solver

Subset-first, then policy:

```text
for subset_size in 2 .. N:                    # fewest Macs first
    for subset in combinations(measured_nodes, subset_size):
        for policy in saved → table offload → MoE 0.75/0.5/0.25 →
                      context ladder → tier → chunk:
            elect(model, subset, policy)
        keep feasible; rank by measured critical stage, headroom, fewer hops
    if this size has a feasible plan:
        choose the best set at this size and STOP   # never add a Mac
```

- A 150 GiB-class model on four Macs must plan on the two Macs that cover it
  (e.g. M5 Max + M2 Ultra, one hop), not all four with three hops.
- Extra Macs are `excluded_nodes` with a reason (“not needed; the selected set
  fits at this context/policy”).
- Only grow to 3/4 when the smaller set cannot fit, and keep the reason per
  size.
- Within a size: measured critical stage, then headroom, then fewer hops, then
  the coordinator preference. Without measured compute, prefer headroom and
  record `unmeasured_compute`; do not invent a performance ranking.
- A single-Mac plan remains unapplicable (`ClusterDeployment` needs 2+ hosts).
- Never exceed the model’s declared max context; never go below the documented
  floor silently.
- If nothing fits, one action from the near miss.

### A5. Apply only on the selected Macs

- Write the chosen per-model offload/MoE settings on the selected Macs through
  the existing admin seam; excluded Macs are not staged, written to or started.
- Pass `memory_guard_tier` per selected node in `ClusterPlanNodeRequest`.
- Add the wired-limit/admission preflight. A too-low limit is a preflight
  refusal with one explicit action; the full helper is Phase 5.
- Record one warning per lever (`table_offload_enabled`,
  `moe_offload_enabled`, `context_reduced`, `tier_raised`, `chunk_reduced`) and
  one `macs_not_needed` note naming the excluded Macs.

### A6. UI

Show the chosen levers and the selected Macs:

```text
Macs      2 of 4 — M5 Max + M2 Ultra (1 hop; .12 and .14 not needed)
Tier      Balanced (or Aggressive — proposed)
Context   32,768 tokens (reduced from 128k)
Offload   PLE table on SSD
Experts   75 % resident
```

An advanced “Use all Macs” option may force every Mac; it is not the default.
The user still presses **Start setup**.

### A7. Tests and live proof

Fake-driven first:

- every lever (offload, context, tier, MoE fraction, chunk) changes a refusal
  into a feasible plan;
- the subset loop returns the fewest Macs that fit and names the excluded ones;
- a model that needs 3 returns 3 and records why 2 failed;
- plan/apply/start use the same subset and policy; a drift is `plan_mismatch`;
- a failed setting write fails apply with one action;
- Qwen PLE and DeepSeek Engram flags are honoured.

Live, only after the fake tests pass:

- a DeepSeek V4.1 Engram and a Qwen4 PLE checkpoint that the thin layer runs are
  planned and started through the assistant;
- a model that only fits with aggressive / shorter context / expert offload
  starts and serves one completion;
- a 150 GiB-class model on the four-Mac line plans on the two Macs that cover
  it, with the other two recorded as not needed.

Acceptance for Phase A is the same as the roadmap §4 A7 and §6 C4.

---

## After Phase A

Continue in roadmap order:

- **Phase B:** synthetic probe after apply for the next plan; streaming
  TTFT/prefill after ready; live `observed` refresh + in-app probe toggle.
- **Phase C:** reload resume + stop lock; `copy_model` source-first flow;
  model-less measurement.
- **Phase D:** 3/4-node tests and live start; consented 2-Mac fallback for an
  incomplete selected topology.
- **Phase E:** only the remaining upstream tracking.

Do not start P2/Phase F (macOS app/tray, full helper, telemetry, TP, >4 Macs).

---

## Constraints that do not change

- No push. No Chapter 10. No second planner, app, wizard, chat or tab.
- No tensor parallelism, no German UI, no >4 Macs.
- No silent model/context/tier change, no local rank-0 fallback.
- No token-rate estimate where no measurement exists.
- Browser JS never launches, deploys, SSHes, stages, supervises or probes.
- No API key, cookie, SSH credential or model content in any file, fixture,
  test, log or commit.
- One tab, one component; the assistant stays inside `clusterV2Wizard()`.
- Port LocalInferenz’s planning logic, not its UI.
- If a new product decision is needed, ask once and record it in `AGENTS.md`
  §10 and `PROJECT-BRIEF.md` §22.

---

## Live state at handover

- `.11` (M5 Max) runs this branch on port 8099; no deployment is running.
- `~/.omlx/settings.json` is restored to its original bytes
  (`assistant_performance_probe` absent → off).
- `.14` is paired and still holds the staged `dev-1.5b` copy.
- `.12` is discovered but unpaired; `.13` was absent and has very little free
  disk.
- Start each server from a Terminal window on that Mac, never over SSH.
- Read `/admin/api/cluster/assistant/operations/active` before a live run; it
  must answer `{"operation": null}`.
- Unload a deployment before measuring (a loaded cluster drops the measured
  admission).

Verification:

```sh
uv run pytest tests/cluster_assistant -q
uv run pytest tests/test_cluster_*.py -q
uv run pytest apps/omlx-web/tests -q        # with node on PATH
uv run ruff check <touched python files>
```

Current baseline: `tests/cluster_assistant` 250 passed;
`tests/test_cluster_*.py` 1600 passed, 2 skipped;
`apps/omlx-web/tests` 493 passed with Node, 0 skipped.

---

## Report back at the end

- What was implemented, with file paths.
- The exact policy/levers and subset the solver chose for a concrete model.
- Tests added and the full verification counts.
- What is still open in the gate (by roadmap ID), and why.
- Any product decision you had to record.
- Whether any live run was performed; if not, the exact blocker.

Stop after the gate is met or you are blocked. Do not start Chapter 10.