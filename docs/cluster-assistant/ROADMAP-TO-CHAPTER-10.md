# Roadmap to Chapter 10 — close every gap first

Status: ready for handover (2026-11-07). Product decisions are closed; the
implementation phases are the next step. This is the gate plan, not an
implementation.

Purpose: close every open point recorded through Chapter 9 (Screen 5 + the
`observed` decision) and the Screen-6 slice, **including the new
resource-planning defect**, before any Chapter-10 work starts.

The reference for the resource-planning work is the LocalInferenz prototype
(“the thin layer”). Port its **planning logic and measured policy**, never its
app, wizard, chat, profile system or German UI (`AGENTS.md` §2).

---

## 0. Recorded decisions (2026-11-07)

These were closed before handover and are frozen in `AGENTS.md` §2/§10 and
`PROJECT-BRIEF.md` §22:

- **Name:** oMLX Cluster Assistant.
- **Fork/upstream/pin:** focused fork. The pin is `1e27576a`
  (`cluster-assistant-base`, tag `cluster-assistant-base-2026-11-07`): #5024 +
  #5033 + rank-0-over-SSH + launch reliability + the MoE fit hint. Do not
  rebase again without a decision.
- **Rank-0 rule / co-location:** best feasible plan; free admission is a strong
  preference; coordinator/rank 0/endpoint stay co-located.
- **RDMA:** ring over TCP only; RDMA/JACCL is advanced.
- **Default context/model:** recommendation decides; 32k target; no fixed
  model.
- **3/4-node topology:** a complete fabric is a **connected line** — every
  consecutive pair in the chosen rank order has a measured link. An incomplete
  line is refused with `topology_incomplete`, the missing pairs are named, and
  the 2-Mac plan is offered only with explicit confirmation; the consent is part
  of the plan signature. An unmeasured fabric keeps the nominal-ring warning.
  A 2-Mac plan is the normal minimal subset when it fits and is linked.
- **Helper:** full signed helper is Phase 5; before the gate, a too-low wired
  limit is a preflight refusal with one action.
- **Telemetry:** none; opt-in diagnostics/local logs only.
- **Resource levers:** shown in the plan; applied on **Start setup**; never
  silent.
- **Admission:** p05 floor; p50/live planning by default; p95 only with an
  explicit risk confirmation for a huge model.
- **Measured compute:** synthetic probe after apply, used for the next plan.
- **TTFT/prefill:** one streaming measurement after ready.
- **Gate scope:** P0/P1 only; P2 after Chapter 10.

---

## 1. The gate

Chapter 10 starts only when:

1. every **P0** and **P1** item in §3 is closed with tests, docs and, where it
   touches hardware, a live run;
2. the **model-availability acceptance test** in §6 passes: any checkpoint that
   the thin layer runs on the validation Macs can also be **planned and
   started** by the assistant on the same Macs, with the chosen context, memory
   tier and SSD/expert offload shown in the plan, and on the **fewest Macs that
   fit** (no unnecessary third/fourth hop);
3. the test suites are green:
   `tests/cluster_assistant`, `tests/test_cluster_*.py`,
   `apps/omlx-web/tests` with Node, plus `ruff` on touched files;
4. `AGENTS.md`, `PROJECT-BRIEF.md`, `README.md` and the Screen documents no
   longer contradict each other or the code.

**P2** items (macOS app/tray, signed helper, telemetry, tensor parallelism,
>4 Macs) are productization after Chapter 10 unless you explicitly pull them
into the gate. They are listed in §10 so nothing is forgotten.

---

## 2. Root cause of the “too conservative” defect

The assistant currently plans **pure resident memory**. The thin layer plans the
policy the Mac will actually launch with. Three concrete gaps:

### 2.1 The election ignores offload

`omlx/cluster/assistant/election.py::_plan_for_order` calls `plan_hybrid` with
only the model, budgets, context, microbatch and rank-0 extra. It does **not**
pass:

```python
engram_ssd_offload=...
moe_offload_layer_bytes=...
moe_expert_offload_resident_fraction=...
```

`plan_hybrid`/`plan_unequal_pipeline` already support all three and already
compute `ssd_weight_bytes` and `moe_offload_weight_bytes` correctly. The
coordinator’s `/plan` route **does** pass them (`_saved_engram_ssd_offload`,
`_moe_offload_for_plan`), but the assistant’s election rejects the model before
the coordinator is ever asked. A DeepSeek Engram / Qwen4 PLE / MoE checkpoint
that fits with tables on SSD is therefore reported `no_feasible_plan` even
though the launch would fit.

### 2.2 The budget ignores the tier

`election.budget_for` builds a `NodeBudget` with the default
`memory_guard_tier="balanced"`; `provisioning.build_deployment_request` hardcodes
`role="headless"` and never passes `memory_guard_tier`. The coordinator’s plan
request type (`ClusterPlanNodeRequest`) already accepts
`safe|balanced|aggressive|custom`, and the worker uses it. The assistant also
plans against `admission_p05` only; the p50/p95 measurements ride along unused.
The thin layer plans against the live admission and lets the tier change the
reserve.

### 2.3 There is no “bring this model up” solver

`recommend` bisects a context ladder through `elect`, and `plan` takes that one
fixed context. There is no joint search over
`tier × context × chunk × table-offload × MoE-offload`, so a user-selected model
that fails at the default context/tier is refused instead of being brought up
with a shorter context or an offload the thin layer would enable.

### 2.4 The election always uses every measured Mac

`elect(nodes, …)` treats the supplied list as the whole cluster: every candidate
gets `rest = all other nodes`, so a 4-Mac measurement always produces 4-rank
plans. There is no “try 2 first” pass. `_score` then sorts by predicted critical
stage (or `inf` when unmeasured) and headroom, so a 4-Mac plan can win merely
because it has more aggregate headroom — three pipeline hops instead of one.

**This is our extension, not a thin-layer port:** the LocalInferenz planner also
plans over every node it is given. The requirement is new: *fewest Macs that
fit, then the best pair/set and least hops.*

### 2.5 What the thin layer does instead

| Thin-layer piece | What it does | Port into the assistant |
|---|---|---|
| `console/planner.py` | Joint enumeration of model/context/chunk/layout; Pareto front; `balanced`/`ttft`/`decode`/`headroom` selection. | A resource-policy solver around `elect()` (not a second planner), plus the new subset loop. |
| `console/catalog.py` | Tier reserves (`safe` 20 %/6/16, `balanced` 8 %/3/8, `aggressive` 2 %/1.5/4), live admission, 8 GiB + 20 % wired guard, Engram/PLE table markers, smallest-largest-stage objective. | Reuse oMLX’s own guard/tier arithmetic; pass the tier; add a wired-limit preflight. |
| `console/deploy.py` | Writes `deepseek_v41_engram_ssd_offload` / `qwen4_ple_ssd_offload` on every node; aggressive tier when a single Mac owns >60 % of RAM; clamps context to the model maximum; per-stage KV/SSD bytes. | Write the same per-model settings through the admin seam; pass tier/offload to the plan; clamp context. |
| `console/model_space.py` | Safe source-first copy and deletion (keep the smoke model and one cluster copy). | The existing free-space relay already deletes; add source-first copy and the `copy_model` action. |
| `console/tuning.py` | Measured chunk/throughput presets, keyed by model + hardware. | Later, feeds `compute`/`observed` (Phase B). |

---

## 3. Gap inventory

Priorities: **P0** blocks the gate, **P1** closes a recorded capability gap,
**P2** productization after Chapter 10.

| ID | Gap | Source | Priority | Phase |
|---|---|---|---|---|
| **R1** | Election ignores Engram/PLE/MoE offload; too conservative | new defect + `election._plan_for_order` | **P0** | A |
| **R2** | Memory tier (safe/balanced/aggressive) not chosen or passed | new defect + `budget_for`/`build_deployment_request` | **P0** | A |
| **R3** | No joint tier × context × chunk × offload solver for a user-selected model | new defect + `recommend`/`plan` | **P0** | A |
| **R4** | Offload switches are not applied on every Mac before launch; Qwen PLE switch absent in this checkout | new defect + `routes._saved_engram_ssd_offload` | **P0** | A |
| **R5** | Wired-limit / admission preflight for large models | new defect + thin-layer `wired_limit_mib` | **P1 — implemented 2026-11-15**: `wired_limit_binds()` + action `raise_wired_limit` on the plan refusal | A |
| **R6** | Plan summary/UI does not show the chosen tier, context, offload or MoE fraction | new defect | **P0** | A |
| **R7** | The election always uses **every measured Mac**; no “fewest Macs that fit” / least-hop selection | new requirement | **P0** | A |
| **M1** | No measured compute for election (`unmeasured_compute`, `compute=None`) | audit | **P1** | B |
| **M2** | TTFT/prefill not measured on the distributed non-streaming path (decode only) | live §12.3 | **P1** | B |
| **M3** | `observed` not refreshed after the ready poll; no in-app probe toggle | audit + Screen-5 §3 | **P1** | B |
| **L1** | Assistant card cannot resume after reload; stop not under the operation lock | Screen-6 §1 | **P1** | C |
| **L2** | `copy_model` action is an info toast, not a flow | Screen-3 §11 | **P1** | C |
| **L3** | Model-less / hardware-only measurement missing | audit | **P1** | C |
| **N1** | 3/4-node start, `test_any_to_any.py`, `test_pp_plan.py`; a 3/4-node topology with an incomplete fabric offers a consented 2-Mac fallback | Phase 3 | **P1** | D |
| **D1** | Stale docs: §22 still “open”, AGENTS “observed stays null”, Screen-5 “Screen 6 later”, “three chapters” | audit | **closed 2026-11-07** | E |
| **D2** | Product decisions: name, fork/pin, RDMA default, default context/model, helper timing, telemetry | AGENTS §10 / Brief §22 | **closed 2026-11-07** | E |
| **U1** | PR #5024/#5033 waiting on upstream | UPSTREAM.md | dependency | E |
| **F1** | macOS app/menu/tray, signed helper, auto-update, network setup | Phase 4/5 | **P2** | F |
| **F2** | Tensor parallelism, >4 Macs, HA, multi-tenancy | out of scope | **P2** | F |

---

## 4. Phase A — Resource policy parity with the thin layer (the new defect)

**Goal:** the assistant’s plan uses the same resident/offload/tier arithmetic
the coordinator will launch with, and a user-selected model is brought up with
the least-invasive levers instead of being refused.

### A1. Measure the facts the policy needs

- Extend the measurement’s model facts (or expose what `ModelLayout` already
  has): `max_context_tokens`, per-layer table bytes (Engram/ngram/PLE), per-layer
  MoE expert bytes, `mtp_weight_bytes`, `supports_pipeline`,
  `pipeline_stage_starts`, `kv_bytes_per_token_per_layer`.
- Add each Mac’s saved `memory_guard_tier`, live admission p50/p95/p05 and
  `wired_limit_mb` to the measured node facts.
- Snapshot the saved per-model settings that matter:
  `deepseek_v41_engram_ssd_offload`, `qwen4_ple_ssd_offload`,
  `moe_expert_offload_enabled`, `moe_expert_offload_resident_fraction`,
  `max_context_window`.
- Fix the Qwen PLE gap in this checkout: `_saved_engram_ssd_offload` must honour
  either table switch (the PR-branch fix), or the assistant must pass the flag
  explicitly so the coordinator does not need to re-derive it.

**Acceptance:** a measurement JSON shows the table layers/bytes, MoE bytes,
model max context, saved tier and saved offload flags for a real DeepSeek/Qwen4
checkpoint.

### A2. Add one policy object, not scattered flags

Add a `ResourcePolicy` (name provisional) carried through
`measure → recommend → plan → apply → start` and into the plan signature:

```text
per node:  memory_guard_tier (safe|balanced|aggressive|custom)
per model: context_tokens
           chunk / prefill_step
           table_offload (engram | ple | none)   # derived from the checkpoint
           moe_expert_offload: bool
           moe_resident_fraction: 0 < f <= 1
```

Rules (decided 2026-11-07):

- defaults come from the saved settings, never from a hidden client default;
- a user’s explicit setting is honoured; the assistant may **propose** a change
  but applies it only through the plan the user starts. Every lever is shown in
  the plan and applied on **Start setup** — never silently;
- the same policy is sent to `plan`, `apply` and `start`; a mismatch is
  `plan_mismatch`, not a silent re-plan;
- the coordinator’s `/plan` and the assistant’s election must both use it, or
  the apply/start placement check will catch the drift.

### A3. Make the election use the policy

- `budget_for(facts, rank, tier=...)` passes `memory_guard_tier` into
  `NodeBudget`.
- `_plan_for_order` passes `engram_ssd_offload`,
  `moe_offload_layer_bytes` and `moe_expert_offload_resident_fraction` into
  `plan_hybrid`.
- Use **p05 as the safety floor and p50/live admission as the planning value**.
  The solver may try **p95 only as the last admission lever**, and only when the
  request is about a huge model that cannot fit at p50; that attempt must be
  surfaced as an explicit risk confirmation (“this plans against the optimistic
  ceiling; the load may be refused”), never selected silently.
- Expose the visible `safe / balanced / aggressive` choice; the saved tier is
  the default.
- Compute `moe_offload_layer_bytes` from the measured layout/config with the
  existing oMLX helper (`omlx.patches.deepseek_v41.moe_offload.offload_layer_bytes`)
  or a generic equivalent; do not duplicate the arithmetic.
- Use the p05 admission as the safety floor and p50/live admission (with the
  chosen tier) as the planning value, or expose a `safe`/`balanced` choice. The
  thin layer uses the live ceiling; document the exact rule in the brief.

**Acceptance:** for a checkpoint with table/offload bytes, `elect()` returns a
feasible plan that the coordinator’s `/plan` reproduces, and a plan without the
policy is refused as `plan_mismatch`.

### A4. Add the “bring this model up” solver

A small joint search around the existing election, not a second planner:

```text
for subset_size in 2 .. N:                    # fewest Macs first
    for subset in combinations(measured_nodes, subset_size):
        for policy in policies (saved → table offload → MoE 0.75/0.5/0.25 →
                                context ladder 1M/128k/32k/16k/8k/4k →
                                tier balanced→aggressive → chunk 2048/4096/8192):
            elect(model, subset, policy)
        keep feasible; Pareto/priority (quality, critical stage, headroom, hops)
    if this subset_size has a feasible plan:
        choose the best pair/set at this size and STOP   # do not add a Mac
        return the chosen subset, levers, alternatives, near misses
```

Rules:

- **Fewest Macs that fit wins.** A 150 GiB need on four Macs must plan on the
  two Macs that actually cover it (e.g. the M5 Max + M2 Ultra, one hop), not on
  all four with three hops. Extra Macs are recorded as `excluded_nodes` with
  the reason “not needed; the selected set fits at this context/policy”.
- **Within the minimal size, choose the best pair/set**: measured critical
  stage first, then headroom, then fewer hops/shorter fabric, then the existing
  coordinator preference. Without measured compute, prefer the feasible set
  with the most headroom and record `unmeasured_compute`; do not invent a
  performance ranking.
- **Only grow the set when the smaller set cannot fit.** Grow in size order
  (2 → 3 → 4) and keep the reason for every size that failed, so the UI can say
  “3 Macs were needed because the 2-Mac split lacks 7.4 GiB at 32k”.
- A **single-Mac plan is still not applicable** (`ClusterDeployment` needs 2+
  hosts); the smallest cluster subset is 2. A one-Mac product mode is a
  separate decision, not this phase.
- Never exceed the model’s `max_position_embeddings`; never go below a
  documented floor (e.g. 4k) silently.
- Prefer the least-invasive lever first: honour saved settings → enable an
  offload the checkpoint supports → reduce context → raise the tier → reduce
  the MoE resident fraction → smaller chunk. Subset size is the outer loop: a
  bigger context or a saved (non-offload) policy may make a 2-Mac set fit
  before a 4-Mac set is ever considered.
- If nothing fits, the one action comes from the near miss
  (`choose_smaller_model`, `add_mac`, `free_disk_space`,
  `adjust_model_or_context`).
- The user-selected model is never replaced silently.

The measurement still covers every paired Mac (it is read-only and is what
answers “what is available”); **plan/apply/start bind to the selected subset**.
The measurement check becomes “the selected nodes are a subset of the measured
nodes”, not “the measurement covers exactly the selected nodes”. The plan
signature and `PlanInputs` carry the selected `node_ids`/`rank_order` so a
re-election cannot silently grow back to all Macs.

**Acceptance:** a fake-driven test per lever (offload, context, tier, MoE
fraction, chunk) changes a refusal into a feasible plan; a 4-Mac measurement
with a 2-Mac-feasible model returns a 2-Mac plan and names the two excluded
Macs; a model that needs 3 returns 3 and says why 2 failed; the plan summary
shows the chosen policy. A live test starts a model that the current assistant
refuses.

### A5. Apply the policy only on the selected Macs

- Before apply/start, write the chosen per-model offload/MoE settings on every
  **selected** Mac through the existing admin seam (`MODEL_SETTINGS_PATH`),
  exactly as the thin layer writes them on every node. Never overwrite an
  explicit user value without showing it in the plan first. Excluded Macs are
  not staged, not written to and not started.
- Pass `memory_guard_tier` per selected node in `ClusterPlanNodeRequest`.
- Add a preflight for `iogpu.wired_limit_mb` / live admission: if the limit is
  too low, fail with one action (configure the Mac / use the helper) instead of
  letting oMLX refuse the load. The actual set needs the productization helper;
  until then the action is explicit.
- Record one warning per lever applied (`table_offload_enabled`,
  `moe_offload_enabled`, `context_reduced`, `tier_raised`,
  `chunk_reduced`) and one `macs_not_needed` note naming the excluded Macs, so
  Details explains why the plan differs from the saved settings and the full
  cluster.

**Acceptance:** the launched deployment’s per-node settings and assignment
tier match the plan; a node whose setting could not be written fails apply with
one action, never starts with a mixed policy; no excluded Mac receives a stage
or a setting write.

### A6. UI: show the levers and the selected Macs

The plan summary adds one compact block:

```text
Macs      2 of 4 — M5 Max + M2 Ultra (1 hop; .12 and .14 not needed)
Tier      Balanced (or Aggressive — proposed)
Context   32,768 tokens (reduced from 128k)
Offload   PLE table on SSD
Experts   75 % resident
```

Show a `Safe / Balanced / Aggressive` preference and a
`Quality / Balanced / Speed` priority only if they can change the answer;
otherwise keep the primary path to one recommendation. If the user insists on
using every Mac, an advanced “Use all Macs” option can force `node_selection =
all`, but it is not the default. The user still presses **Start setup**; that is
the confirmation for any proposed change. Screen 6 Details repeats the applied
policy and the excluded Macs.

### A7. Tests and live proof

Fake-driven:

- every lever changes feasibility; the plan signature includes the policy;
- the subset loop returns the **fewest** Macs that fit, names the excluded
  Macs, and does not grow to all Macs when a pair fits;
- a model that needs 3 returns 3 and records why 2 failed;
- apply/start use the same subset and policy; a drifted subset is
  `plan_mismatch`;
- no silent model/context downgrade; one action when nothing fits;
- a failed setting write fails apply with one action;
- the Qwen PLE and DeepSeek Engram flags are honoured.

Live (validation line `.11`+`.14`, then `.12`/`.13` when available):

- DeepSeek V4.1 Engram and Qwen4 PLE checkpoints that the thin layer runs are
  planned and started through the assistant;
- a model that only fits with `aggressive` / a shorter context / expert offload
  starts and serves one completion;
- a 150 GiB-class model on the four-Mac line plans on the **two** Macs that
  cover it (M5 Max + M2 Ultra) and serves one completion, with the other two
  recorded as not needed;
- the recorded plan shows the same levers as the thin layer’s proposal.

**Exit:** the model-availability acceptance test in §6 passes.

---

## 5. Phase B — performance and `observed`

### B1. Measured compute (`M1`)

**Decided 2026-11-07:** run the existing synthetic
`run_cluster_performance_probe` after `apply` (staged, before `start`) on the
coordinator, store the profiles, and use them for the **next** measurement/plan.
It never rewrites the current `predicted` and never gates the start.

Remove `unmeasured_compute` from a run only if a real profile exists; never fake
it.

### B2. TTFT and prefill (`M2`)

**Decided 2026-11-07:** add one streaming measurement after `ready` and read the
final usage chunk’s `time_to_first_token` / `prompt_tokens_per_second`; keep the
probe off by default. Do not derive TTFT from total/generation duration.

The existing non-streaming `observed` decode rate stays as it is.

### B3. Live `observed` and the probe switch (`M3`)

- Give Screen 6 one bounded “refresh status” read (or a manual Refresh button)
  so a probe that lands after the ready poll is visible without a reload.
- Add an in-app toggle for `assistant_performance_probe` (server setting) so it
  is not edit-JSON-only.

---

## 6. Phase C — lifecycle and model flow

### C1. Reload resume and the operation lock (`L1`)

- Persist a minimal “last assistant deployment” (measurement id, plan inputs,
  coordinator, deployment id, model/context) so a reloaded card can show
  Running/Stopped and offer Start again. Store no key, SSH target or secret.
- Refuse `stop` while another apply/start is running, or queue it behind the
  existing cluster-wide lock.

### C2. `copy_model` (`L2`)

Implement the source-first copy the thin layer already has
(`model_space.ensure_model` + safe eviction) behind the existing
`copy_model` action. Copy from a complete peer source, never download if a peer
has the bytes, never delete the last cluster copy, and never delete the
checkpoint the cluster is about to run. This is in the P1 gate; the message-only
fallback is removed once the flow exists.

### C3. Model-less measurement (`L3`)

Let `measure` accept `model: null` (hardware-only), so recommendation can run
before a model is selected; then measure the chosen model (Option A) before
`plan`. This is the original recommendation follow-up and removes the “measure
the wrong model first” step.

### C4. Model-availability acceptance test

One fake-driven test and one live checklist:

- for a checkpoint the thin layer can run on the same nodes, the assistant
  returns a feasible plan with the same **or better** levers, applies it, starts
  it, and serves one completion;
- when a smaller subset of the measured Macs fits, the plan uses that subset
  (fewest Macs, least hops) and names the excluded Macs; it does not use all
  available Macs “because they are there”;
- for a checkpoint neither can run, the assistant returns one action and does
  not start anything.

---

## 7. Phase D — 3/4-node and fabric (`N1`)

- Add `test_any_to_any.py` and `test_pp_plan.py` (3/4-node permutations, ring
  and line).
- **Decided 2026-11-07:** RDMA default is ring over TCP; RDMA/JACCL is the
  later advanced path. A 3/4-Mac plan still requires the full topology; if the
  selected fabric is incomplete, the assistant shows the missing cable/link and
  offers **“run with 2 Macs” only with explicit confirmation** — never a silent
  degrade. A 2-Mac plan is the normal minimal subset when it fits. The
  validation cluster runs as a line; a physical ring needs STP per bridge
  member.
- Live-run a 3- or 4-node start on a loss-free line and record it in
  `PHASE-2B-RUN.md`.
- Resolve the environment gaps: `.12` pairing, `.13` disk/availability, `.14`
  disk (97 %).

---

## 8. Phase E — documentation and product decisions

### E1. Reconcile the records (`D1`) — closed 2026-11-07

- `PROJECT-BRIEF.md` §22 now records the answers instead of “still open”; the
  rank-0 rule and co-location are marked frozen.
- `AGENTS.md` §2 now says the synthetic probe stays unwired while the real
  completion may fill `observed`.
- `SCREEN-5-HANDOFF.md` §9 no longer lists Screen 6 as later.
- `SCREEN-3-MINIMAL.md` §11 now says four slices.

### E2. Decisions (`D2`) — closed 2026-11-07

All answers are in §0 and in `AGENTS.md` §2/§10 / `PROJECT-BRIEF.md` §22. The
new agent does not need to re-ask them; if a new question appears, ask once and
record it in the same two places.

### E3. Upstream (`U1`)

Track #5024/#5033 as external dependencies; do not rebase the feature branch
before the #5024 decision. This does not block Chapter 10 unless the gate is
defined to require a merged upstream.

---

## 9. Phase F — post-Chapter-10 productization (P2)

**Confirmed 2026-11-07:** these are not gate items; they follow Chapter 10.
**Started 2026-11-07:** the macOS menu/tray entry (existing Cluster tab; no new
window/app/control plane) is implemented; the helper is next but its signing
credentials and privilege model are open.

- macOS app menu/tray entry, Welcome/Cluster Assistant entry;
- signed/notarized helper, wired-limit/network setup, auto-update;
- telemetry/diagnostics;
- tensor parallelism, >4 Macs, HA failover, multi-tenancy.
- **First-run onboarding should install the helper and set the hardware limits
  (noted 2026-11-15).** Today the wired-limit helper is installed from
  *Performance → “Apply with Admin…”* in the macOS app, and the web dashboard's
  *Settings → Performance* shows only the terminal command
  (`sudo sysctl iogpu.wired_limit_mb=…`) — the button is **app-only**, which a new
  user misses exactly when they need it (a 128 GB Mac runs with a 107.5 GB Metal
  cap until then). The Welcome window already runs the **runtime installer** for
  the dependencies, so it is the natural place for the same one-admin-dialog step:
  bless the helper, apply the recommended `iogpu.wired_limit_mb`, and show the
  resulting ceiling. One prompt, at the screen where the user is already
  installing things. The web page should then either deep-link into the app or
  spell out the two-step; it must not look like the command is the only option.

---

## 10. Sequencing

```text
Phase A (resource policy)  ─┬─> Phase B (compute/TTFT/observed)
                            ├─> Phase C (reload/copy/model-less)
                            └─> Phase D (3/4-node)
Phase E (docs/decisions)   ── can run in parallel, but D1 must close before the gate
Phase F (productization)   ── after Chapter 10
```

Suggested order inside Phase A:

1. A1 facts + the Qwen PLE/Engram parity;
2. A2 policy object + A3 election wiring;
3. A4 solver + A5 apply/start settings;
4. A6 UI + A7 tests/live.

Phase B can start once A3 is stable (the policy changes what `compute` is
applied to). Phase C and D can proceed in parallel with B.

---

## 11. Definition of done (checklist)

- [x] `R1–R7` closed in code, fake-driven tests and live runs (2026-11-07): the
      election/plan/apply/start share one `ResourcePolicy` (context, Engram/PLE
      table offload, MoE resident fraction, memory tier, admission, microbatch);
      the plan reports the levers and `excluded_nodes`; the subset solver uses
      the fewest Macs that fit; apply writes the levers on every selected Mac
      and fails with one action; the wired limit is a preflight refusal; the
      card shows Macs/Tier/Context/Offload/Experts with a "Use all Macs"
      override. **Live:** `dev-1.5b` full path; a 4-Mac measurement of
      `GLM-5.3-Flash-oQ2e-mtp` (107.8 GiB) planned on `.12`+`.11`, named `.14`
      and `.13` as not needed, started and served a completion; a DeepSeek V4.1
      Engram start (`.11`+`.12`) served a completion; a routed-expert 25 %
      start served a completion. **Qwen4 PLE:** its table is detected, but the
      checkpoint is not pipeline-capable, so plan now refuses it with
      `choose_pipeline_model` (a real bug fixed by enforcing
      `supports_pipeline` and applying the pre-load patches in
      `inspect_safetensors_layout`). A pipeline-capable 150 GiB-class
      checkpoint is not present on the validation cluster; the 4-Mac *start* of
      a >100 GiB model remains Phase D disk/`.13` environment work. See
      `PHASE-2B-RUN.md` §14.
- [x] `M1–M3` closed (2026-11-07): the synthetic compute probe runs after apply
      on the coordinator, its profiles are stored and used by the next
      measurement/plan; one streaming completion after ready fills TTFT and
      prefill (non-streaming decode stays); the ready card has a bounded Refresh
      and an in-app probe toggle. Live: apply with the probe stored both Macs'
      profiles and the next plan reported stage seconds with no
      `unmeasured_compute`; a probe-on start reported prefill 9.06 tok/s,
      decode 21.43 tok/s, TTFT 110 ms.
- [x] `L1–L3`: **L1 closed** (reload resume via `GET /assistant/session`, stop
      lock local + coordinator), **L2 closed offline** (`copy_model` runs the
      assistant flow; apply stages source-first from a complete peer; the
      last-copy guard is in the delete route) and **L3 closed** (hardware-only
      measurement, recommend, measure the chosen model before plan). The live
      copy/delete path stays in the deferred matrix.
- [x] `N1`: `test_any_to_any.py`, `test_pp_plan.py` and the topology tests are
      in place (3/4-node subsets, contiguous stages, rank-0 tail, MTP charge,
      origin independence, connected-line completeness). The live 3- and 4-node
      starts served completions on 2026-11-07. The consented 2-Mac fallback is
      implemented (2026-11-07): `topology_incomplete` names the missing pairs,
      “Run with 2 Macs” is an explicit click, and the consent salts the plan
      signature. The live cable-pull check remains in the deferred test matrix.
- [x] `D1` closed: no stale “open”/“later”/“observed stays null” text (2026-11-07).
- [x] `D2` recorded 2026-11-07 in `AGENTS.md` §10 and the brief §22.
- [x] All suites green: 314 assistant, 1614 cluster, 320 web; `ruff` clean on
      the assistant and touched test files (2026-11-07).
- [ ] Live evidence in `PHASE-2B-RUN.md` for every hardware-touching item; the
      cluster is offline, so this is the one open gate item and is covered by
      the deferred test matrix in the untracked verification note.
- [x] No key, cookie, SSH credential or model content in any file/fixture/log.

Only then start Chapter 10.

**2026-11-07:** the offline gate is closed; live evidence is deferred. Chapter 10
has started as the DAU acceptance/support chapter — see
[`CHAPTER-10.md`](CHAPTER-10.md). P2 stays out.

---

## 12. Non-goals for this gate

- A second planner, a second app, a second wizard or a second chat.
- Copying LocalInferenz code/UI wholesale; only the planning/measurement logic
  and the existing oMLX knobs are reused.
- Tensor parallelism, >4 Macs, Linux/CUDA, HA, multi-tenancy.
- Silent model/context/tier changes; every lever is shown in the plan and
  confirmed by **Start setup**.
- A token-rate estimate where no measurement exists.
---

## 13. Handover notes for the next agent

**Read first:** `AGENTS.md` §2 and §10, `PROJECT-BRIEF.md` §5, §8, §13, §22,
this roadmap §0–§4, `SCREEN-5-HANDOFF.md`, `SCREEN-6-MANAGE.md`, and
`PHASE-2B-RUN.md` §12–§13. The copy-paste starting prompt for this handover is
[`GATE-START-PROMPT.md`](GATE-START-PROMPT.md).

**First slice (do not start Chapter 10):**

1. A1 — add the model/node facts the policy needs, including the Qwen PLE /
   DeepSeek Engram parity in this checkout.
2. A2 — the `ResourcePolicy` object carried through plan/apply/start.
3. A3 — pass `memory_guard_tier`, `engram_ssd_offload`,
   `moe_offload_layer_bytes` and the resident fraction into the election’s
   `plan_hybrid` call; add the p05-floor/p50-planning/p95-consent rule.
4. A4 — the subset-first + lever solver, including `excluded_nodes`.
5. A7 — fake-driven tests before any hardware run.

**Live state at handover (2026-11-07):** the `.11` server runs this branch on
port 8099; no deployment is running; `~/.omlx/settings.json` is back to its
original bytes (`assistant_performance_probe` absent → off); `.14` still holds
the staged `dev-1.5b` copy; `.12` is discovered but unpaired; `.13` was absent
and has very little free disk. Start servers from a Terminal on the Mac, never
over SSH.

**Commands:**

```sh
uv run pytest tests/cluster_assistant -q
uv run pytest tests/test_cluster_*.py -q
uv run pytest apps/omlx-web/tests -q        # with node on PATH
uv run ruff check <touched python files>
```

**Rules that stay:** no push; no new app/wizard/chat; no silent model/context/
tier change; no local rank-0 fallback; no token-rate estimate without a
measurement; keys, cookies and SSH credentials never enter the repo.
