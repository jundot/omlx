# Session 6 — Minimal Screen-3 UI

## CHAPTER 6 — Minimal Screen-3 UI (the first UI slice)

### WORKING DIRECTORY

```text
/Users/andreaskolb/Documents/PycharmProjects/omlx-cluster-assistant
```

BRANCH `feat/cluster-assistant`

Last commits:

```text
fc14e337  docs: define minimal Screen-3 UI contract
50cd16f5  feat(cluster): apply and start an approved plan on the elected coordinator
b70b92c6  docs: define Chapter 5 apply and start plan
```

Do not push. Do not assume any prior conversation: every decision is in the
repository. Chapters 1–5 are finished; do not redo them. The backend is stable
and frozen for this chapter — this is a UI chapter.

---

### READ FIRST, IN THIS ORDER

1. `AGENTS.md` — §2 "Resolved implementation decisions" (especially the
   Screen-3 bullets and the frozen backend rules), §3 read list, §4 "Current
   important constraint", §5 project layout, §8 hard prohibitions.
2. `docs/cluster-assistant/SCREEN-3-MINIMAL.md` — the authoritative brief for
   this chapter (states, bindings, errors, acceptance, tests). If it and
   `PROJECT-BRIEF.md` disagree, ask; do not pick silently.
3. `docs/cluster-assistant/PROJECT-BRIEF.md`
   - §5 Screens 3–6 (what the user is supposed to see),
   - §8.5/§8.6 (plan object; "no legal plan" and the one-action rule),
   - §9.4/§9.5 (endpoint ownership; origin independence),
   - §12.1 (English only), §12.4 (error/action table),
   - §13.2/§13.3/§13.3a/§13.4 (endpoints, plan object, recommendation
     object, persistence),
   - §17.2 (no promised throughput before measurement),
   - Appendix A.5 (rank 0 == local coordinator == endpoint owner).
4. `docs/cluster-assistant/CHAPTER-5.md` — §2.1 (the recorded remote-start
   decision), §4 (operation contract), §6 (performance-probe policy),
   §7 (no model-less measurement), §8 (UI policy).
5. `docs/cluster-assistant/UPSTREAM.md` — this chapter patches upstream web
   files, so it must add rows there.
6. `CLUSTER-ASSISTANT.md` — current status.

Then the backend contract you must consume, exactly as committed:

| Area | File |
|---|---|
| Request/response schemas + builders | `omlx/cluster/assistant/models.py` |
| The seven routes | `omlx/cluster/assistant/routes.py` |
| Operation states/actions | `omlx/cluster/assistant/operations.py` |
| What apply/start send | `omlx/cluster/assistant/provisioning.py`, `start.py` |
| Backend tests to keep green | `tests/cluster_assistant/test_apply_api.py`, `test_start_api.py`, `test_operations.py`, `assistant_harness.py` |

Then the UI you will actually change (read before designing; do not infer):

| Area | File / symbol |
|---|---|
| Cluster tab template | `apps/omlx-web/omlx_web/templates/dashboard/_cluster_v2.html` (plan section ~L659, active ~L1010, error banner ~L121) |
| Cluster tab component | `apps/omlx-web/omlx_web/static/js/cluster_v2.js` — `clusterV2Wizard()`, `CLUSTER_V2_API`, `wizardState()`, `apiFetch`, `runPlan`, `loadModels`, `modelOptions`, `selectedModelPath`, `targetContextTokens`, `executionProfile`, `pollStagingJob` (single-flight + generation idiom), `notify`, `window.t(...)` |
| JS harness | `apps/omlx-web/tests/ui/test_cluster_v2_wizard.py` — `_run_wizard`, `_WIZARD_TWO_MACS`, `WIZARD_STATES`, `ALLOWED_ENDPOINTS`, `test_wizard_consumes_only_contract_endpoints` |
| Python-driven UI tests | `apps/omlx-web/tests/test_cluster_ui_integration_ui.py` |
| Fixtures | `apps/omlx-web/tests/ui/fixtures/cluster_v2/*.json` |
| i18n | `apps/omlx-web/omlx_web/i18n/en.json` (and `window.t` fallback) |
| CSS | `apps/omlx-web/build_css.py` |

---

### STATE OF THE REPO (verified facts — not guesses)

Backend (Chapter 5, commit `50cd16f5`), frozen for this chapter:

- The assistant router exposes exactly seven routes, all admin-gated by the
  existing mount in `omlx/server.py` (no `server.py` change this chapter):

```text
POST /admin/api/cluster/assistant/measure
GET  /admin/api/cluster/assistant/status?measurement_id=…
POST /admin/api/cluster/assistant/recommend
POST /admin/api/cluster/assistant/plan
POST /admin/api/cluster/assistant/apply
POST /admin/api/cluster/assistant/start
GET  /admin/api/cluster/assistant/operations/{operation_id}
```

- Tests: `tests/cluster_assistant` = 193 passed; `tests/test_cluster_*.py` =
  1593 passed, 2 skipped. `tests/test_cluster_seams.py` has no assistant entry
  in `allowed_uncalled`. `ruff`/`mypy` clean for the assistant module.
- `measure` requires a **model identity**: `model_path` xor
  `model_size_bytes` + `layer_count` (+ `mtp`). Model-less / `hardware_only`
  measurement does **not** exist; `recommend` reports `hardware_only: false`.
  This is the one thing that shapes the UI flow (see the design question).
- `measure` is a job: `{measurement_id, state:"running", poll}`; poll
  `/status` until `state == "ready"` or `"failed"` (failure carries
  `error.{code,message,action}`).
- `recommend` request: `measurement_id`, `candidates[{model_path,
  friendly_name?, declared_context_tokens?}]` (1–32), optional
  `context_tokens` (target), `execution_profile`. Response: `selected_model`
  (a model id), `selected_context_tokens`, ordered `candidates[]` (each with
  `model_path`, `label`, `feasible`, `max_context_tokens`, `nodes`, `reasons`,
  `warnings`, `action`, `critical_stage_seconds:null`, `measured_speed:false`).
  No feasible candidate = `409 no_feasible_candidate` with the same body plus
  `code`/`action`/`message`.
- `plan` request: `measurement_id`, `model_path` xor size identity,
  `context_tokens`, `mtp`, `rank0_extra_bytes`, `execution_profile`,
  `microbatch_size?`, `origin_node?`, optional `node_ids`. Response (§13.3):
  `plan_signature`, `coordinator_node`, `rank_order`, `stages`, `groups`,
  `predicted` (stage seconds only), `observed:null`, `endpoint` or `null`,
  `warnings`. Infeasible = `409 no_feasible_plan` (plan object with
  `feasible:false`, `code`, `action:"adjust_model_or_context"`).
- `apply` request: `measurement_id`, `plan_signature`, the same model
  identity, `context_tokens`, `rank0_extra_bytes`, `mtp`,
  `execution_profile`, `microbatch_size?`, `coordinator_node`, `rank_order`,
  `origin_node?`. Returns `{operation_id, state:"applying", poll}`.
- `start` request: `measurement_id`, `plan_signature`, `coordinator_node`,
  optional `operation_id`. Returns `{operation_id, state:"starting", poll}`.
- Operation status: `kind`, `state`
  (`queued|applying|applied|starting|loading|ready|failed`),
  `progress{phase,message,nodes_ready,nodes_total}`,
  `endpoint{node,host,port,url,ready}|null`, `warnings[]`,
  `error{code,message,action}|null`.
- **`endpoint.url` may be shown/clickable only when
  `state == "ready"` and `endpoint.ready == true`.** An unresolved endpoint is
  `null`, never a port-0 URL.
- Refusal shapes the UI must render without parsing prose: flat
  `{code,message,action}` (plus `reason`/`measurement_id` for
  `stale_measurement`, plus `operation_id` for `operation_in_progress`).
  Codes seen in this flow: `stale_measurement`, `no_feasible_candidate`,
  `no_feasible_plan`, `plan_mismatch`, `single_node_plan`, `node_not_paired`,
  `coordinator_unresolved`, `operation_not_applied`,
  `operation_in_progress`, `unsupported_parallelism`, and operation failures
  `staging_failed`, `coordinator_unreachable`, `coordinator_start_failed`,
  `readiness_unproven`, `placement_changed`, `operation_interrupted`.
  Actions: `measure`, `plan`, `apply`, `start`, `status`, `add_mac`,
  `pair_mac`, `copy_model`, `configure_endpoint`, `retry_coordinator`,
  `diagnostics`, `adjust_model_or_context`.
- No token rate exists anywhere (`predicted` has stage seconds only,
  `observed` is `null`); nothing may display one.

UI (verified):

- There is **no assistant UI at all** today: `grep -i assistant` in
  `_cluster_v2.html` and `cluster_v2.js` returns nothing.
- `apps/omlx-web/omlx_web/static/js/cluster_v2.js` is 3955 lines and exports the
  global factory `clusterV2Wizard()`; the template is 1652 lines and guards
  each section with `x-show="wizardState() === '<state>'"` plus a
  `data-cluster-v2-state="<state>"` marker. The existing 8 states are pinned by
  `WIZARD_STATES` / `test_each_state_section_is_guarded_by_its_state_machine_value`.
- `ALLOWED_ENDPOINTS` in `apps/omlx-web/tests/ui/test_cluster_v2_wizard.py` does
  **not** contain the assistant routes; `test_wizard_consumes_only_contract_endpoints`
  will fail until they are added there.
- `_run_wizard(body)` executes the shipped component under Node with
  `window.t` bound to the real `en.json`. **`node` is not installed in this
  environment**: those tests currently skip (48 passed, 82 skipped across
  `test_cluster_ui_integration_ui.py` + `test_cluster_v2_wizard.py`). Write the
  JS-executed tests anyway (they must pass where node exists) **and** static
  assertions that run here.
- The existing plan section already loads `modelOptions` from
  `POST /admin/api/cluster/models`, keeps the user's `selectedModelPath`,
  `targetContextTokens` (default 32768), `executionProfile`, and renders its
  own plan/activate controls. Do not remove or rewire those in this slice.
- There is an existing `/chat` page (`omlx_web/routes.py::chat_page`), and there
  is no deep link that preselects a model.

---

### THE ONE DESIGN QUESTION TO SETTLE BEFORE CODING

`/measure` still requires a model identity, but the Screen-3 flow in
`SCREEN-3-MINIMAL.md` §4 starts with `measure(hardware)` and then recommends.
Something has to give. Ask once (one question, three options), record the answer
in `SCREEN-3-MINIMAL.md` §4 and in `AGENTS.md` "Resolved implementation
decisions", then code:

- **A (recommended) — measure the model the tab already has selected.** The plan
  section already knows `selectedModelPath`. Measure that model (model-bound),
  poll to ready, then call `recommend` over the candidate list (legal: Chapter 4
  reads only the hardware half and may evaluate any model), render the
  recommended model and its `selected_context_tokens`, and **only if the
  recommended `model_path` differs** measure that model again (model-bound,
  second poll) before `plan`. No backend change; hardware-only measurement stays
  the later follow-up it already is.
- **B — add model-less measurement in this chapter** (`model: null`,
  `hardware_only: true`) so the flow truly starts with hardware. That reopens
  the frozen Chapter 3 measurement contract and `ModelIdentity`; `plan` must
  keep refusing a model-less record. Bigger than a UI chapter.
- **C — drop recommendation from the first slice** and drive
  `measure(model) → plan → apply → start` for the model the user picked. This
  contradicts the SCREEN-3 acceptance criteria that bind `selected_model` /
  `selected_context_tokens` to the recommendation; only pick it if you accept
  updating that document.

Also settle and record, without asking twice:

- **One tab, one component.** Extend `clusterV2Wizard()` and the existing
  Cluster tab. Add a separate `assistantState()` machine with
  `data-assistant-state="…"` markers; do **not** add entries to `WIZARD_STATES`
  or to the existing `wizardState()` chain (that would rewrite the 8-state
  contract tests for no reason).
- **Assistant states:** `idle → measuring → recommending → measuring_model →
  planning → applying → starting → loading → ready`, plus `failed` as an
  in-card error state. `applying/starting/loading` mirror the operation states
  one-for-one; never invent states the backend cannot report.
- **Polling, no SSE.** One single-flight poll loop per job (measurement and
  operation) with a generation counter, exactly like `pollStagingJob`; a late
  response from an old generation must not overwrite the current one. Stop on
  `ready`/`failed` (measurement) and `ready`/`failed` (operation).
- **Plan-time inputs are frozen once chosen.** Keep `rank0_extra_bytes` (0),
  `mtp`, `execution_profile`, `microbatch_size` and `context_tokens` in one
  state object and send the *same* values to `plan` and `apply`. A drift is a
  `409 plan_mismatch` the user did not cause.
- **`Start setup` = `apply` then `start`.** Never `start` before the apply
  operation is `applied`; single-flight button while either job runs; the
  operation is the only source of progress.
- **`Open Chat` / `Connect agent`:** links/buttons that appear only when ready.
  `Open Chat` may navigate to the existing `/chat` page; `Connect agent` shows
  the endpoint URL + model id with a copy action. Neither may call launch,
  deploy, stage or SSH; if a deep link needs new backend work, degrade to the
  endpoint card and say so in the chapter doc.
- **Advanced details collapse:** stage table, rank order, measurement ids,
  warnings and the raw operation body go behind a `<details>`/“Details”
  affordance. The primary path shows only model, context, coordinator, Mac
  count, progress, endpoint when ready, and one action on failure.

---

### GOAL FOR THIS SESSION

(a) one assistant card inside the existing Cluster tab plan section that drives
    the whole backend flow without the user typing an IP, SSH target, rank,
    layer or model path;
(b) the bindings from `SCREEN-3-MINIMAL.md` §3, including `selected_model` and
    `selected_context_tokens` from the recommendation;
(c) `Start setup` posts `apply` then `start` and polls the operation;
(d) progress for applying/starting/loading, endpoint only when ready,
    Chat/agent affordances only when ready;
(e) exactly one action on every failure, rendered from `error.action`
    (`retry_coordinator`, `pair_mac`, `measure`, `plan`, `add_mac`,
    `configure_endpoint`, `diagnostics`, …), with no local fallback in any
    branch;
(f) tests: JS-executed where node exists, static where it does not;
(g) the small doc updates, same commit.

This session is NOT:

- any backend change (routes, models, operations, provisioning, start,
  election, measurement, plan/recommend shapes);
- hardware-only measurement (unless the design question answers B);
- SSE/streaming, tray, full chat, agent handoff, macOS-app work;
- tensor parallelism;
- a second tab, page, wizard or control plane;
- German localization or new non-English locale entries;
- any estimated or displayed token rate.

---

### DELIVERABLES

#### 1. `_cluster_v2.html`

- One assistant section inside the existing plan area
  (`x-show="wizardState() === 'plan'"`); do not add a tab or a new wizard.
- `data-assistant-state="<state>"` on each guarded block so a static test can
  find every state.
- The required elements only: recommended model card, context value, plan
  summary (coordinator, Mac count, model, context), **Start setup**, progress
  line, endpoint URL + copy + **Open Chat** + **Connect agent** when ready,
  error banner with `error.message` and one button labelled from
  `error.action`.
- Endpoint markup must be unreachable unless `assistantReady()` is true
  (both `state === 'ready'` and `endpoint.ready === true`).
- A collapsed “Details” block for stages/rank order/ids/warnings.
- No inline JS logic beyond `x-*` bindings; no launch/stage/deploy markup.

#### 2. `cluster_v2.js`

- `ASSISTANT_API` with exactly the seven assistant paths (plus the existing
  `/admin/api/cluster/models` model list, reused, not duplicated).
- Assistant state, single-flight pollers, and the flow from the recorded design
  decision, with the frozen plan-time inputs.
- `assistantActionLabel(action)`, `assistantActionHandler(action)` (dispatch
  only: measure again, plan again, open pairing, open model picker, show
  diagnostics, retry coordinator). Unknown actions fall back to a plain
  message, never to a different behaviour.
- `pair_mac`: open the **existing** pairing flow (the join/approve machinery
  already in the component); do not implement pairing.
- `retry_coordinator`: show “Open/configure oMLX on <coordinator>” and a retry
  that re-posts `start` only; never change `coordinator_node`, never use the
  local Mac.
- No `fetch`/`XMLHttpRequest` outside `apiFetch`; no new timers that outlive
  the component.
- English strings through `window.t('cluster.assistant.…')`.

#### 3. Endpoint allowlist, fixtures, i18n

- Add the seven assistant routes to `ALLOWED_ENDPOINTS` in
  `apps/omlx-web/tests/ui/test_cluster_v2_wizard.py` (that list is the network
  contract for the tab).
- Add fixtures under `apps/omlx-web/tests/ui/fixtures/cluster_v2/` for at least:
  measurement running/ready, recommendation, plan, operation
  applying/loading/ready, and two failures (`retry_coordinator`,
  `pair_mac`). Each fixture must be a real backend payload shape (copy the
  field names from `models.py`).
- Add every new key to `omlx_web/i18n/en.json`; no other locale.

#### 4. CSS

- Extend the cluster styles where the existing tab already keeps them and run
  `cd apps/omlx-web && python build_css.py`. Commit the built CSS exactly as the
  repo does today.

#### 5. Tests

`apps/omlx-web/tests/test_cluster_ui_integration_ui.py` (Python-driven, mocked
`apiFetch`, node-gated) — at minimum:

1. recommendation response binds `selected_model` / `selected_context_tokens`
   into the rendered state;
2. the `plan` request carries the recommendation's model and context, and the
   `apply` request carries the same plan-time inputs (`plan_signature`,
   `rank0_extra_bytes`, `mtp`, `execution_profile`, `microbatch_size`);
3. `apply` is posted before `start`, and `start` is not posted while apply is
   unapplied or already failed;
4. operation polling stops at `ready` and at `failed`, and a stale generation
   cannot overwrite a newer one;
5. endpoint is not exposed before `ready` (and is exposed after);
6. `retry_coordinator` renders the coordinator action and posts nothing that
   changes the coordinator; the local Mac never becomes rank 0;
7. `pair_mac` opens the existing pairing affordance and does not start
   anything;
8. the assistant flow never calls `/deployments`, `/stage`, `/replan`,
   `/autoconfigure` or `/ssh-key*`.

`apps/omlx-web/tests/ui/test_cluster_v2_wizard.py` (static, runs here):

- every `data-assistant-state` from the state list exists in the template;
- `wizardState() === '<state>'` markers and `WIZARD_STATES` are unchanged;
- the assistant endpoints appear only in `ASSISTANT_API` and are allowlisted;
- every `window.t('cluster.assistant.…')` key exists in `en.json`;
- no `ssh`, `rank`, `layer`, `model_path` input in the assistant primary path;
- the existing 8-state and endpoint-contract tests stay green.

`tests/cluster_assistant` must stay green untouched (193 passed). If a UI test
needs a backend payload, take it from a real fixture — do not change the
backend to fit the UI.

#### 6. Docs (small, same commit)

- `SCREEN-3-MINIMAL.md`: status → implemented, record the design decision and
  the settled points above, and note anything deferred (e.g. `Connect agent`
  degradation).
- `AGENTS.md` "Resolved implementation decisions": the entry-measurement
  choice, one-tab/`assistantState()` decision, endpoint visibility rule, no-SSE
  polling rule.
- `CLUSTER-ASSISTANT.md` "Current status": the UI slice exists; what it does
  and does not do.
- `UPSTREAM.md`: new rows for the patched web files
  (`_cluster_v2.html`, `cluster_v2.js`, `en.json`, CSS) with the reason and the
  test that keeps each honest. This chapter *is* an upstream patch; record it.

---

### MUST NOT CHANGE

- `omlx/cluster/assistant/*` except where a test proves nothing else is needed.
  The seven routes, their request/response shapes, the operation state machine,
  the error codes and the action strings are frozen.
- `election.py`, `measurement.py`, `state.py`, `recommendation.py`,
  `omlx/cluster/routes.py`, `omlx/server.py`.
- The existing 8 wizard states, their guards, the existing plan/activate
  controls, and the pairing flows.
- `apps/omlx-web/omlx_web/templates/dashboard.html` structure beyond including
  nothing new (the tab already exists).
- Anything under `apps/omlx-mac`.

---

### IMPLEMENTATION RULES

- One tab, one component, one poll loop per job. Reuse `apiFetch`, `notify`,
  `window.t`, and the existing single-flight/generation idiom.
- The UI never computes a plan, rank order, coordinator, layer split or
  admission. It renders what `recommend`, `plan` and the operation say.
- The UI never selects a coordinator. `plan.coordinator_node` and
  `operation.coordinator_node` are display values.
- Exactly one action per failure, taken from `error.action`. No generic
  “try again” that hides a coordinator or pairing problem.
- No launch, deploy, stage, SSH or supervisor code in the browser. The
  assistant routes are the entire surface.
- English only: new strings live in `en.json`; no German, no other locale.
- No throughput number anywhere; when the backend says speed is unknown, the UI
  stays silent about speed.
- Keep the diff reviewable: a new section in the template, a new block in the
  component, one allowlist line, fixtures, tests, CSS.

---

### WORKFLOW

1. Read the docs and the files above; trace recommendation → plan → apply →
   start in `models.py`/`routes.py` before designing.
2. Ask the one design question (A/B/C) once; record the answer in
   `SCREEN-3-MINIMAL.md` before coding.
3. Write a 5–10 bullet plan (states, bindings, flow, error mapping, test
   matrix).
4. Implement template → component → allowlist/fixtures/i18n → CSS → tests.
5. `uv run pytest tests/cluster_assistant -q` (must stay 193 passed).
6. `uv run pytest apps/omlx-web/tests/test_cluster_ui_integration_ui.py
   apps/omlx-web/tests/ui/test_cluster_v2_wizard.py -q`
7. `uv run pytest tests/test_cluster_*.py -q` (one full pass)
8. `cd apps/omlx-web && python build_css.py`
9. `uv run ruff check` on touched Python files (web tests only; do not reformat
   unrelated files).
10. Commit on `feat/cluster-assistant` with a focused message. Do not push.

---

### DEFINITION OF DONE

- The Cluster tab renders `selected_model` and `selected_context_tokens` from
  the recommendation, and `plan` is requested with exactly those values.
- **Start setup** runs `apply` → `start` → operation polling, showing
  applying/starting/loading, and never starts inference from the browser.
- The endpoint URL only becomes visible/clickable when the operation is `ready`
  and `endpoint.ready` is true.
- `retry_coordinator` renders a coordinator-open/configure action; `pair_mac`
  renders a pairing action; neither ever selects the local Mac.
- Every failure shows exactly one action, derived from `error.action`.
- The primary path has no IP, SSH, rank, layer or model-path field.
- English only, no token rate, no second tab.
- `tests/cluster_assistant` unchanged and green; UI tests green where node
  exists and the static contract tests green in this environment; CSS built.
- `UPSTREAM.md`, `AGENTS.md`, `SCREEN-3-MINIMAL.md`, `CLUSTER-ASSISTANT.md`
  updated in the same commit.

Report back: what you built, which decision you recorded (and where), the
states/endpoints you bound, the tests you added and their pass/skip status, the
files changed, whether any backend file changed (it should not) and whether
`server.py` changed (it should not), and what still blocks a full Screen-4
progress view / Screen-5 handoff.

STOP after the minimal Screen-3 slice and its tests. Do not continue into
hardware-only measurement, SSE/streaming, full chat, tray, macOS-app work or
tensor parallelism without a new instruction.