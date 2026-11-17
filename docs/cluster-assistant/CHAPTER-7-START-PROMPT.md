# Session 7 — First live end-to-end run (Phase 2b)

## CHAPTER 7 — Live acceptance on real Macs, and the Screen-4 progress brief

### WORKING DIRECTORY

```text
/Users/andreaskolb/Documents/PycharmProjects/omlx-cluster-assistant
```

BRANCH `feat/cluster-assistant`

Last commits:

```text
ac059445  feat(cluster): minimal Screen-3 assistant card in the Cluster tab
fc14e337  docs: define minimal Screen-3 UI contract
50cd16f5  feat(cluster): apply and start an approved plan on the elected coordinator
```

Do not push. Do not assume any prior conversation: every decision is in the
repository. Chapters 1–6 are finished; do not redo them. The assistant backend
and the minimal Screen-3 UI are already committed and tested against fakes. This
chapter is **the first run on real hardware**, plus the recorded decision that
unblocks Screen 4. It is not a feature chapter.

Today the validation cluster is expected to be online, but verify that before
relying on it.

---

### READ FIRST, IN THIS ORDER

1. `AGENTS.md` — §2 "Resolved implementation decisions" (the full list,
   including the Screen-3 entry decision, one-tab/`assistantState()`, endpoint
   visibility and the no-SSE polling rule), §3 read list, §4 "Current important
   constraint", §5 project layout, §8 hard prohibitions, §9 environment.
2. `CLUSTER-ASSISTANT.md` — current status: what exists and what deliberately
   does not.
3. `docs/cluster-assistant/SCREEN-3-MINIMAL.md` — §4.1 (entry decision A) and
   §11 (what the shipped slice does and does not do, and the deferred items).
4. `docs/cluster-assistant/SCREEN-4-PROGRESS.md` — the recorded Screen-4
   decision (Option A), the eight labels, the six observable rows, the two
   combined rows and the rendering rules. Do not re-open it.
5. `docs/cluster-assistant/CHAPTER-5.md` — §2.1 (remote start is the existing
   SSH control plane, same activation), §4 (operation contract), §6
   (performance-probe policy), §7 (no model-less measurement), §8 (UI policy).
6. `docs/cluster-assistant/PROJECT-BRIEF.md`
   - §5 Screens 3–6 (especially Screen 4 phases and Screen 5 gating),
   - §8.5/§8.6 (plan object; no legal plan and the one-action rule),
   - §9.4/§9.5 (endpoint ownership; origin independence),
   - §12.1 (English only), §12.4 (error/action table),
   - §13.2/§13.3/§13.3a/§13.4 (endpoints, plan object, recommendation object,
     persistence),
   - §17.2 (no promised throughput before measurement),
   - Appendix A.5 (rank 0 == local coordinator == endpoint owner),
   - Phase 2b (the acceptance criteria for this chapter).
7. `docs/cluster-assistant/UPSTREAM.md` — every patch row and the test that
   keeps it honest.
8. `docs/cluster-assistant/README.md` — note that its "Current milestone" is
   stale and is one of this chapter's small doc updates.

Then the code you will drive and, only if the live run proves a bug, touch:

| Area | File |
|---|---|
| Request/response schemas + builders | `omlx/cluster/assistant/models.py` |
| The seven routes | `omlx/cluster/assistant/routes.py` |
| Operation states/actions | `omlx/cluster/assistant/operations.py` |
| Provisioning / start | `omlx/cluster/assistant/provisioning.py`, `start.py` |
| Election / measurement / recommendation | `election.py`, `measurement.py`, `state.py`, `recommendation.py` |
| Cluster tab template | `apps/omlx-web/omlx_web/templates/dashboard/_cluster_v2.html` (`data-cluster-v2-assistant`) |
| Cluster tab component | `apps/omlx-web/omlx_web/static/js/cluster_v2.js` (`ASSISTANT_API`, `assistantState()`, the `// Assistant — the minimal Screen-3 slice` block) |
| Assistant tests | `tests/cluster_assistant/`, `apps/omlx-web/tests/ui/test_cluster_v2_wizard.py`, `apps/omlx-web/tests/test_cluster_ui_integration_ui.py` |
| Fixtures | `apps/omlx-web/tests/ui/fixtures/cluster_v2/assistant_*.json` |

Read the assistant block in `cluster_v2.js` and the assistant card in the
template before touching anything. Do not infer its behaviour from this prompt.

---

### STATE OF THE REPO (verified at `ac059445` — not guesses)

Backend:

- Exactly seven admin routes, mounted by the existing `omlx/server.py` mount:

```text
POST /admin/api/cluster/assistant/measure
GET  /admin/api/cluster/assistant/status?measurement_id=…
POST /admin/api/cluster/assistant/recommend
POST /admin/api/cluster/assistant/plan
POST /admin/api/cluster/assistant/apply
POST /admin/api/cluster/assistant/start
GET  /admin/api/cluster/assistant/operations/{operation_id}
```

- `measure` is a job and requires a model identity (`model_path` xor
  `model_size_bytes` + `layer_count`, plus `mtp`). Model-less measurement does
  not exist. `recommend` reports `hardware_only: false`.
- `plan` is synchronous, pure and strict about the measurement it reads
  (node set, model identity, `mtp`). `apply` re-elects and refuses a
  `plan_signature` it cannot reproduce. `start` only launches an `applied`
  operation and re-prepares the placement.
- Operation states: `queued | applying | applied | starting | loading | ready |
  failed`. `progress` is only `{phase, message, nodes_ready, nodes_total}`.
  `endpoint` is `{node, host, port, url, ready}` or `null`; `url` is only
  meaningful when `state == "ready"` and `endpoint.ready == true`.
- Failures carry exactly one `error.{code, message, action}`. The action set is
  `measure | plan | apply | start | status | add_mac | pair_mac | copy_model |
  configure_endpoint | retry_coordinator | diagnostics |
  adjust_model_or_context`.
- No token rate exists anywhere. `predicted` has stage seconds only,
  `predicted.measured` is false, `observed` is `null`. Never invent a rate.
- Tests at `ac059445`: `tests/cluster_assistant` = 193 passed;
  `tests/test_cluster_*.py` = 1593 passed, 2 skipped;
  `apps/omlx-web/tests` = 289 passed, 164 skipped.

UI (this is the surface you will actually exercise):

- One assistant card inside the Cluster tab plan section. `assistantState()`
  runs `idle → measuring → recommending → measuring_model → planning →
  applying → starting → loading → ready`, plus `failed`, each pinned with
  `data-assistant-state`. A computed plan rests in `planning` with
  `assistant.plan` set.
- Entry is **decision A**: measure the model the tab already selected
  (model-bound), poll it ready, `recommend` over the model list, and re-measure
  only when `selected_model`'s `model_path` differs from the measured one.
- `selected_model` / `selected_context_tokens` are bound from `recommend` and
  also sync the existing picker (`selectedModelPath`, `targetContextTokens`).
- **Start setup** posts `apply`, waits for the apply operation to reach
  `applied`, then posts `start` with that operation ID, then polls.
- `endpoint.url` is rendered (link, copy, Open Chat, Connect agent) only when
  `assistantState() === 'ready'` and `endpoint.ready === true`.
- One failure, one button, labelled from `error.action`.
  `retry_coordinator` shows "Open/configure oMLX on <coordinator>" and retries
  the same stage on the same coordinator. `pair_mac` / `add_mac` open the
  existing pairing affordance. No branch ever names this Mac as coordinator.
- `Open Chat` navigates to the existing `/chat` page **without** a model
  preselection. `Connect agent` is the endpoint card (base URL + model ID
  + copy). `copy_model` is not automated and falls back to a message.
- The i18n keys exist in `en.json` and in every existing locale file as the
  **English fallback** (the repo's `normalize_i18n.py` step of `make web`; the
  identical-key-set test requires this). No translation was added. Keep it that
  way: English only, no German.
- `node` is not installed in this environment, so the JS-executed tests skip.
  A working substitute for manual checks is macOS's JavaScriptCore shell:

```text
/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc
```

  It has `readFile`, `load` and `drainMicrotasks`, so the shipped component can
  be executed against the fixtures without node. Do not add this to CI.

---

### GOAL FOR THIS SESSION

(a) prove the committed Screen-3 slice on the real Macs, end to end, with
    evidence — not a faked cluster and not a unit test;
(b) fix only what the live run proves broken, keeping the frozen contracts;
    if a contract change is required, stop and record a decision instead;
(c) write the evidence and the pass/fail per Phase 2b exit criterion;
(d) **after the run passes**, implement the recorded Screen-4 progress surface
    (UI only) exactly as `SCREEN-4-PROGRESS.md` specifies; if the run fails,
    fix it first and leave Screen 4 for the next chapter;
(e) keep every existing test green, and update the stale README milestone.

This session is NOT:

- Screen-5 implementation, SSE/streaming, or a new progress source;
- hardware-only/model-less measurement;
- tensor parallelism;
- a new tab, page, wizard or control plane;
- macOS-app or tray work;
- a performance-rate estimate, or enabling the distributed performance probe;
- any change to `election.py`, `measurement.py`, `state.py`,
  `recommendation.py`, `omlx/cluster/routes.py` or `omlx/server.py` unless a
  live failure proves it necessary — and then only with a recorded reason.

---

### THE SCREEN-4 DECISION IS ALREADY RECORDED — DO NOT RE-OPEN IT

`docs/cluster-assistant/SCREEN-4-PROGRESS.md` records **Option A**: map the
existing assistant states, operation kind and
`progress.{phase,message,nodes_ready,nodes_total}` onto the eight Screen-4
labels. No backend change, no SSE.

The mapping in one glance:

```text
1 Checking Macs        <- assistantState measuring / measuring_model
2 Planning             <- assistantState recommending / planning
3+4 Preparing and copying the model
                       <- operation kind apply, state applying
5 Starting coordinator <- operation kind start, state starting
6+7 Loading the model and verifying the endpoint
                       <- operation kind start, state loading
8 Ready                <- operation state ready AND endpoint.ready
```

Only six labels are independently observable. "Preparing runtime" shares the
`staging` phase with "Copying model", and "Verifying endpoint" runs inside the
blocking activation call after `loading`. A row must never animate a phase the
backend did not report.

The real order is Checking → Planning → Preparing/Copying → Starting →
Loading/Verifying → Ready. The brief's label order (Preparing/Copying before
Planning) is corrected in the Screen-4 document because planning must precede
staging; treat that correction as recorded.

The live run is the evidence check: if the combined rows hide a wait long enough
to matter, record it in `PHASE-2B-RUN.md` and leave the B follow-up
(`preparing` phase, activation verification signal) for a later chapter. Do not
change the backend progress contract in this chapter.

Also settle and record without asking twice:

- **Validation is manual on the browser side.** The agent cannot click the UI.
  The user runs each scenario; the agent watches the measurement/operation JSON
  and the server logs through the local admin session. Never write a key,
  cookie, SSH credential or model path config into the repository.
- **No live cluster in unit tests.** Any new test stays fake-driven; the live
  evidence lives in the run document, not in the test suite.
- **Evidence over impressions.** Every scenario records the actual
  `measurement_id`, `plan_signature`, operation IDs, state transitions with
  timestamps, the endpoint URL and a real completion. "It looked fine" is not
  evidence.
- **The performance probe stays off.** A deployment will now exist, but the
  assistant request sets `auto_tune=False` and the probe is not wired. Record
  the probe as the next optional follow-up; do not enable it in this chapter.
- **The untracked prompt docs** `CHAPTER-5-START-PROMPT.md` and
  `SCREEN-3-START-PROMPT.md` are prior-session artifacts. Add this chapter's
  prompt beside them if you like, but do not delete or rewrite them silently.

---

### ACCEPTANCE MATRIX (run these, in this order)

Verify reachability first; if a node is offline, record which and run the
2-Mac subset that is available. Never hardcode an address into the app; the
pairing/discovery snapshot is the only source.

| # | Origin | Peers | Expected coordinator | What it proves |
|---|---|---|---|---|
| A | `.14` M1 Max 32 GB | `.11` M5 Max 128 GB | `.11` (feasibility, not origin) | a non-coordinator origin drives a remote coordinator |
| B | `.11` M5 Max 128 GB | `.13` M2 Ultra 64 GB | `.11` or `.13`, whichever the election picks | local-origin path; coordinator still elected, not assumed |
| C | any online Mac | `.11` + `.13` + `.14` | the election's choice | a 3-Mac plan, staging and start without manual ranks/IPs |
| D | `.14` | `.11` with oMLX stopped | none (failure) | `retry_coordinator`, same coordinator, no local fallback |
| E | an unpaired Mac | — | none (failure) | `pair_mac`, existing pairing flow, nothing started |
| F | the winner of A | — | already ready | `GET <endpoint>/v1/models` and one real completion via `/chat` |

For every scenario capture:

- the measurement: `measurement_id`, `measured_at`, node count, warning codes;
- the recommendation: `selected_model`, `selected_context_tokens`, feasibility
  and any `action`;
- the plan: `plan_signature`, `coordinator_node`, `rank_order`, stages, the
  endpoint (if any), warnings;
- the operations: apply and start IDs, every observed state
  (`applying → applied → starting → loading → ready|failed`), timings, and the
  final `endpoint.{url,ready}`;
- any failure: the exact `error.{code,message,action}` and which button the UI
  rendered;
- the proof that the origin did not become rank 0 when it was not elected.

Staging is tens of gigabytes: record what was already on each Mac and what was
copied, and do not re-run a full copy just to repeat a passing step.

---

### DELIVERABLES

#### 1. `docs/cluster-assistant/PHASE-2B-RUN.md` (new)

- Environment table: which Macs, chips, RAM, addresses, app version, commit.
- One section per scenario in the matrix: steps, evidence, pass/fail against
  the Phase 2b exit criteria, and the exact failure action when it failed.
- A "findings" list, split into:
  - **bug** (implementation does not do what the brief says),
  - **contract gap** (the brief asks for something the API cannot express —
    this feeds Screen 4/5, it is not a silent fix),
  - **environment** (a Mac, cable or permission problem).
- A short "what still blocks Screen 4 / Screen 5" section with evidence.

#### 2. `docs/cluster-assistant/SCREEN-4-PROGRESS.md` (already recorded)

- The decision is Option A and is **already written**. Do not re-open it.
- Update it only if the live run proves a mapping claim wrong (for example, the
  combined row hides a wait that matters). Record the evidence in
  `PHASE-2B-RUN.md` and leave any B follow-up for a later chapter.
- It must stay specific: eight labels, six observable rows, the two combined
  rows, the failure rule, the no-rate rule, and the tests.

#### 3. Screen-4 progress surface (UI only, after the run passes)

Implement exactly the mapping and rendering rules in `SCREEN-4-PROGRESS.md`:

- `_cluster_v2.html`: one progress surface inside the assistant card, visible
  from the first phase through `ready` (and the failed state). Stable markers
  for every row; the two combined rows carry both labels.
- `cluster_v2.js`: a pure `assistantProgressRows()` (or equivalent) that
  returns the rows from the existing `assistantState()`/`assistant.plan`/
  `assistant.operation` data. No new fetch, no new timer, no new backend call,
  no new assistant state.
- `en.json`: the new label keys (and the existing locale files via
  `normalize_i18n.py`, English fallback only).
- CSS only if the existing utilities are insufficient; rebuild and commit the
  built CSS.
- Do not show a percentage, byte count or token rate. Staging may show
  `nodes_ready/nodes_total` only.
- The endpoint keeps the existing ready-only rule.

#### 4. Small doc updates (same commit)

- `docs/cluster-assistant/README.md`: current milestone → Phase 2b validated /
  Screen-4 progress surface built; link the run document and the Screen-4 brief.
- `CLUSTER-ASSISTANT.md` "Current status": the live run result, the Screen-4
  surface, and what still blocks Screen 5. Keep it short; do not claim more than
  the evidence shows.
- `AGENTS.md` "Resolved implementation decisions": the Screen-4 decision is
  already recorded; add the rule that live evidence lives in
  `PHASE-2B-RUN.md`, not in unit tests.
- `UPSTREAM.md`: only if a live fix or the progress surface patches an upstream
  file. The progress surface does; add the rows with the reason and the test
  that keeps each honest.

#### 5. Code fixes (only if the run proves them)

- Keep them minimal and fake-tested.
- A fix that changes a route, a request/response shape, an operation state, an
  error code or an action string is a contract change: stop, record it in the
  run document, and ask once before changing it.
- The browser still may not launch, deploy, stage, SSH or supervise. The
  assistant routes remain the entire UI surface.

---

### MUST NOT CHANGE

- `election.py`, `measurement.py`, `state.py`, `recommendation.py`,
  `omlx/cluster/routes.py`, `omlx/server.py` without a live-proven reason.
- The seven assistant routes, their request/response shapes, the operation
  state machine, the error codes and the action strings.
- `selected_model` / `selected_context_tokens` binding, the endpoint visibility
  rule, and the one-action-per-failure rule.
- The existing 8 wizard states, their guards, the existing plan/activate
  controls, and the pairing flows.
- Anything under `apps/omlx-mac`.
- English only; no German and no translated assistant strings.

---

### IMPLEMENTATION RULES

- Run the real flow first; touch code second.
- The UI never computes a plan, rank order, coordinator, layer split or
  admission. It renders what `recommend`, `plan` and the operation say.
- The UI never selects a coordinator. `plan.coordinator_node` and
  `operation.coordinator_node` are display values.
- Exactly one action per failure, taken from `error.action`. No generic
  "try again" that hides a coordinator or pairing problem.
- No token rate anywhere. `predicted.measured=false` and `observed=null` mean
  the UI says nothing about speed.
- Never copy an API key, a cookie, an SSH target list or model data into the
  repository or into a document.
- Keep the diff reviewable: the run document, the small doc updates, the
  Screen-4 progress surface, and at most a small fake-tested fix.
- The progress surface is display-only. It reads the state the assistant
  machine already has and adds no request, timer, state or backend field.

---

### WORKFLOW

1. Read the docs and the assistant block; trace recommendation → plan → apply
   → start in `models.py`/`routes.py` before running anything.
2. Preflight: confirm each Mac is reachable, the app is running on the origin,
   and the devices/pairing snapshot is populated. Record the environment.
3. The Screen-4 decision is already recorded in `SCREEN-4-PROGRESS.md`. Do not
   re-open it; validate it against the real run.
4. Run scenarios A–F with the user driving the browser; capture the JSON and
   timings as you go. If a step fails, capture the exact payload before retrying.
5. Write `PHASE-2B-RUN.md` with the evidence and the pass/fail table.
6. Fix only live-proven bugs, with fake-driven tests; re-run the suites.
7. Only if the run passed: implement the Screen-4 progress surface and its
   tests per `SCREEN-4-PROGRESS.md`. If the run failed, stop after the fixes and
   leave Screen 4 for the next chapter.
8. Update the README/status/AGENTS/UPSTREAM docs named above.
9. Verification pass:

```sh
uv run pytest tests/cluster_assistant -q          # expect 193 passed (or more)
uv run pytest tests/test_cluster_*.py -q          # expect 1593 passed, 2 skipped
uv run pytest apps/omlx-web/tests -q              # expect 289 passed, 164 skipped
uv run ruff check <touched python files>
```

10. If the template/component changed, rebuild CSS and commit the built output:

```sh
cd apps/omlx-web && python build_css.py
```

11. Commit on `feat/cluster-assistant` with a focused message. Do not push.

---

### DEFINITION OF DONE

- A non-coordinator origin completed measure → recommend → plan → apply →
  start → ready on real Macs, with the endpoint served by the elected Mac.
- The same flow completed with origin == coordinator.
- The 3-Mac scenario ran (or its unavailability is recorded with the reason).
- `retry_coordinator` and `pair_mac` were observed on real failures, each with
  exactly one action and no local fallback.
- `GET <endpoint>/v1/models` and one real completion succeeded.
- `PHASE-2B-RUN.md` contains the evidence and the pass/fail per Phase 2b exit
  criterion, not prose impressions.
- If the run passed, the Cluster tab shows the Screen-4 progress surface:
  eight labels, six observable rows, the two combined rows, no invented phase,
  no percentage/byte/rate, endpoint still only when ready. The mapping tests are
  fake-driven and green.
- README / status / AGENTS / UPSTREAM are updated consistently.
- All pre-existing tests stay green; any new test is fake-driven.
- No backend contract changed silently; no token rate; no second tab; English
  only.

Report back: the environment used, the scenarios run and their pass/fail, the
exact evidence (measurement/plan/operation IDs, coordinator, endpoint), the
findings by kind (bug / contract gap / environment), whether the Screen-4
surface was built (and if not, why), every file changed, whether any backend
file changed (and why, if so), whether `server.py` changed (it should not), and
what still blocks Screen 4 and Screen 5.

STOP after the live run, the fixes it proves, and — only if the run passed — the
Screen-4 progress surface. Do not implement Screen 5, SSE/streaming, the
performance probe, hardware-only measurement, full chat, tray, macOS-app work or
tensor parallelism without a new instruction.