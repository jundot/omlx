# Minimal Screen-3 UI

Status: implemented (2026-11-03) — see §11

Parent: `CHAPTER-5.md`

Scope: the smallest UI that can drive recommendation → plan → apply → start and show a ready endpoint.

---

## 1. What blocks it

**Only the UI itself.**

The backend contracts for measurement, recommendation, plan, apply, start and
operation status are already the ones the UI must consume. Do not add a second
wizard, a second control plane, or new launch logic in JavaScript.

The two backend facts below are not workarounds for the UI. They are normal
one-action outcomes that the UI must render faithfully:

| Backend outcome | UI meaning | UI action |
|---|---|---|
| `retry_coordinator` | The elected coordinator's oMLX app is closed or has no admin key. | Show “Open/configure oMLX on <coordinator>”, then retry. Never start the local Mac instead. |
| `node_not_paired` / `pair_mac` | The origin is a rank-1 peer without a paired SSH target on itself. | Show “Pair this Mac”, open the existing pairing flow, then retry. Never fall back to a local rank 0. |

---

## 2. Location

Use the existing oMLX Cluster v2 surface:

- `apps/omlx-web/omlx_web/templates/dashboard/_cluster_v2.html`
- `apps/omlx-web/omlx_web/static/js/cluster_v2.js`

Add the minimal assistant states to the existing tab; do not create a second
tab, page, or wizard. No macOS-app-specific screen is required for this slice.

---

## 3. Required data bindings

The UI must read these values from the backend:

- `selected_model`
- `selected_context_tokens`
- measurement status (`state`, `measurement_id`)
- recommendation candidates (at minimum the selected candidate)
- plan (`plan_signature`, `coordinator_node`, `stages`, `endpoint`)
- operation status (`applying`, `starting`, `loading`, `ready`, `failed`)
- `error.code`
- `error.message`
- `error.action`

The UI must not compute a plan, derive a rank order, or infer a coordinator.

---

## 4. Flow

```text
measure(hardware)
  → poll measurement status until ready
  → recommend
  → select recommended model + context
  → measure(model-bound)
  → poll measurement status until ready
  → plan
  → show selected model/context + plan summary
  → apply
  → start
  → poll operation status
  → ready: show endpoint.url
  → failed: show error.message + one action
```

If the backend returns a ready model-bound measurement and plan, the UI does
not need to expose every intermediate measurement detail.

### 4.1 Entry decision (recorded 2026-11-03)

`/measure` requires a model identity (Chapter 3); model-less measurement does
not exist. The flow above therefore starts from the model the Cluster tab
already has selected, not from bare hardware.

**Decision: Option A — measure the selected model, recommend, then re-measure
only if the recommendation names a different model.**

```text
model = selectedModelPath (or the first model in the list)
  → measure(model, model-bound) → poll until ready
  → recommend(measurement_id, candidates = the model list)
  → bind selected_model + selected_context_tokens
  → if recommended model_path == the measured model: plan
  → else: measure(model-bound, recommended model) → poll → plan
  → apply → start → poll operation → ready / failed
```

It is legal because `recommend` reads only the hardware half of a measurement
and may evaluate any candidate (§11.1a); `plan` keeps its strict measurement
match, which is why the recommended model is measured again before planning.
No backend change: hardware-only measurement stays the later follow-up.

Rejected: **B** (add `model: null` / `hardware_only: true` now — reopens the
Chapter 3 measurement contract and `ModelIdentity` for a UI chapter) and **C**
(drop `recommend` — contradicts §9 of this document and the
`selected_model`/`selected_context_tokens` acceptance criteria).

### 4.2 Entry decision superseded: Phase C3 (recorded 2026-11-13)

Phase C3 of `ROADMAP-TO-CHAPTER-10.md` §6 implemented what option B deferred:
`measure` accepts `model: null` (hardware-only, `hardware_only: true`), and
the UI now uses it for the entry measurement. The §4 flow above is the
current contract again:

```text
measure(hardware-only) → poll until ready
  → recommend(measurement_id, candidates = the model list)
  → bind selected_model + selected_context_tokens
  → measure(model-bound, chosen model) → poll until ready
  → plan → apply → start → poll operation → ready / failed
```

The chosen model is **always** measured model-bound before `plan` (no
shortcut when it happens to match a previous measurement); `plan` keeps its
strict node-set/model-identity/`mtp` match and refuses a hardware-only
measurement. The 2026-11-03 Option A entry above is history, not the current
behaviour. Confirmed by the review alignment on 2026-11-13
(`REVIEW-2026-11-13.md`, decision E1); the UI test
`test_assistant_recommendation_binds_selected_model_and_context` asserts the
hardware-only entry (`model_path: null`) plus the model-bound second
measurement.

---

## 5. Screen-3 states

| State | What the user sees | Primary action |
|---|---|---|
| Measuring hardware | “Checking your Macs…” | none / progress |
| Recommendation | Recommended model + context | Continue |
| Model measurement | “Preparing measurements for <model>…” | none / progress |
| Plan | Model, context, coordinator, stage summary | Start setup |
| Applying | “Preparing the model…” | progress |
| Starting | “Starting <coordinator>…” | progress |
| Loading | “Loading the model…” | progress |
| Ready | Model, endpoint URL, Chat / Connect agent | Open Chat |
| Failed | Plain message + exactly one action | Retry / Pair / Configure / Change model |

The endpoint URL is **not shown as a clickable link** until
`operation.state == "ready"` and `endpoint.ready == true`.

---

## 6. Required UI elements

Keep it minimal:

- one recommended model card,
- context selector or a read-only context value,
- a plan summary:
  - coordinator,
  - number of Macs,
  - model,
  - context,
- **Start setup** button,
- progress line,
- endpoint URL only when ready,
- **Open Chat** and **Connect agent** only when ready,
- error banner with `error.message` and one `error.action` button.

Advanced details may be collapsed behind “Details”:

- stage table,
- rank order,
- measurement IDs,
- warnings,
- logs.

---

## 7. Error rendering

Use `error.action` as the button identity. Do not invent a different action.

Examples:

```text
retry_coordinator
  → button: "Open oMLX on <coordinator>" or "Retry"
  → never select the local Mac as coordinator

pair_mac
  → button: "Pair this Mac"
  → open the existing oMLX pairing flow
  → never select the local Mac as rank 0

stale_measurement
  → button: "Measure again"
```

The UI must not:

- hide the coordinator failure behind a generic “try again”,
- silently retry with the local Mac,
- show a raw stack trace,
- show `endpoint.url` before ready.

---

## 8. Prohibited UI behavior

- No launch, deployment, SSH, staging, or supervisor code in browser JS.
- No local-first fallback.
- No second wizard or tab.
- No rank/layer/IP/SSH fields in the primary path.
- No German strings.
- No tensor-parallel controls.
- No token-rate estimate when the backend has no measured speed.
- No clickable endpoint before `ready=true`.

---

## 9. Acceptance criteria

- The UI renders `selected_model` and `selected_context_tokens` from the
  recommendation.
- The plan request uses those selected values.
- **Start setup** posts `apply`, then `start`, then polls operation status.
- Progress is visible through applying/starting/loading.
- Endpoint URL is hidden until `ready=true`.
- `retry_coordinator` renders a coordinator-open/configure action.
- `node_not_paired`/`pair_mac` renders a pairing action.
- Neither error starts the local Mac as fallback.
- Every failure has exactly one action.
- The primary flow has no IP, SSH, rank, layer, or model-path input.
- The UI remains English-only.

## 10. Test expectations

At minimum:

- recommendation response binds model/context;
- plan request carries the selected model/context;
- apply/start order is correct;
- operation polling stops at ready/failed;
- endpoint hidden until ready;
- `retry_coordinator` action;
- `pair_mac` action;
- no local fallback in either error path;
- no launch call from the browser.

Tests can use mocked fetch responses; no live cluster is required for the
minimal UI contract.

---

## 11. Implementation record (2026-11-03)

Entry decision: **A** (see §4.1), superseded by **Phase C3** on 2026-11-13
(see §4.2): the entry measurement is hardware-only and the chosen model is
measured model-bound before `plan`. `plan` still refuses a mismatched or
hardware-only measurement.

What was built:

- One assistant card inside the existing Cluster tab plan section, backed by
  `clusterV2Wizard()`. No second tab, page, wizard or control plane; no new
  entries in `WIZARD_STATES` or the existing `wizardState()` chain.
- A separate `assistantState()` machine with `data-assistant-state="…"`
  markers: `idle → measuring → recommending → measuring_model → planning →
  applying → starting → loading → ready`, plus `failed`. `applying`, `starting`
  and `loading` mirror the operation states one-for-one; no state exists that
  the backend cannot report.
- A computed plan rests in `planning` with `assistant.plan` set (the same guard
  renders the in-flight spinner while the request runs). This keeps the state
  list exactly as decided without inventing a backend state.
- `selected_model` and `selected_context_tokens` are bound from `recommend`;
  they also sync the existing picker (`selectedModelPath`,
  `targetContextTokens`) so the primary path and the advanced controls describe
  the same choice.
- Plan-time inputs are frozen into one object
  (`measurement_id`, `model_path`, `context_tokens`, `mtp`,
  `rank0_extra_bytes`, `execution_profile`, `microbatch_size`) and sent
  unchanged to both `plan` and `apply`; `apply` adds `plan_signature`,
  `coordinator_node` and `rank_order` from the plan object.
- `Start setup` posts `apply`, polls until the apply operation is `applied`,
  then posts `start` with that operation ID and polls the start operation.
  Progress is rendered through applying/starting/loading.
- One single-flight poll loop per job (measurement and operation) with a
  generation counter, mirroring `pollStagingJob`. Polling stops at
  `ready`/`failed`; a late response from an old generation cannot overwrite a
  newer one. No SSE.
- `endpoint.url` is rendered (link, copy, Open Chat, Connect agent) only when
  `assistantState() === 'ready'` **and** `endpoint.ready === true`.
- Every failure renders one button labelled from `error.action`. The dispatch
  is exactly: measure again, plan again, apply/prepare again, start again, poll
  the named operation, retry on the same coordinator, open the existing pairing
  flow (pair/add a Mac), open diagnostics, or reopen the picker. No branch
  selects a coordinator and no branch falls back to this Mac.
- `retry_coordinator` shows “Open/configure oMLX on <coordinator>” and retries
  the same stage: when the apply operation is `applied` it re-posts `start`,
  otherwise it re-runs `apply` for the same frozen plan. `coordinator_node` is
  never changed.
- `pair_mac` / `add_mac` open the existing pairing affordance
  (`beginPairing` on a discovered/pending Mac, else the device/pairing
  surface). Nothing is measured, applied or started by that action.
- Advanced evidence (stage table, rank order, measurement/plan ids, warnings,
  raw operation body) is collapsed behind “Details”. The primary path has no
  IP, SSH, rank, layer or model-path field.
- English strings live in `omlx_web/i18n/en.json`. The repo's own
  `normalize_i18n.py` (`make web`) mirrors the key set into the existing locale
  files with the English value as fallback, so no translation and no new locale
  is added; every `cluster.assistant.*` value is the English one.
  No token rate is displayed anywhere.

Earlier deferred items that have since landed:

- `copy_model` is a real flow now: the action selects the candidate and runs the
  assistant flow; apply stages from a complete peer copy, and the free-space
  delete refuses the model the cluster is about to run and the last measured
  copy.
- Hardware-only measurement (C3) is implemented: `measure` accepts no model,
  `recommend` reads the hardware half, and the chosen model is measured again
  before `plan`. `plan` refuses a hardware-only record with `measure`.
- Reload resume (C1): `GET /assistant/session` returns the last deployment and
  the card offers Start again/Stop; a stop is refused while another operation
  runs.
- Compute/TTFT (B1–B3): the synthetic probe after apply feeds the next plan,
  one streaming completion fills TTFT/prefill after ready, and the card has a
  bounded Refresh and an in-app probe toggle.

The consented 2-Mac fallback for an incomplete selected topology is implemented
(2026-11-07): the plan refuses with `topology_incomplete`, names the missing
pairs, and offers “Run with 2 Macs” only through an explicit click; the consent
salts the signature.

Still deliberately absent: SSE/streaming progress, tensor parallelism, and the
macOS app/tray/helper (P2).

Since this record, four slices landed on top of the same card:

- **Screen 4** (`SCREEN-4-PROGRESS.md` §8) — the eight labels are rendered as
  six observable rows from `assistantState()`, the operation kind/state and
  `progress.{phase,message,nodes_ready,nodes_total}`. No backend change, no SSE.
  The endpoint stays hidden until `assistantReady()`.
- **Free space** (`PHASE-2B-RUN.md` §9) — the one `free_disk_space` action opens
  a panel that lists the measured models of the Mac the user picked, with sizes,
  a two-step confirm, and a re-measure after a delete through the free-space
  assistant route.
- **Screen 5** (`SCREEN-5-HANDOFF.md`) — the two ready buttons now finish the
  job. `Open Chat` opens the existing `/chat` with the resolved serving id
  (relative for a local coordinator, the coordinator’s own page for a remote
  one), and `Connect agent` shows `<endpoint>/v1`, the serving id and where the
  key lives. `/chat` reads one `?model=` parameter, selects a known id and
  leaves its default alone for an unknown one; it adds no history entry and
  reads no credential. The operation gained one nullable `observed` block,
  filled only after `ready` by a real completion on the coordinator (server
  setting `assistant_performance_probe`, default off). No second page, chat or
  wizard was added.
- **Screen 6** (`SCREEN-6-MANAGE.md`) — the ready card carries **Stop cluster**
  (the tenth assistant route, relayed to the coordinator’s own unload) and, once
  stopped, **Start again**, which re-applies the same frozen plan inputs and
  starts. The endpoint and both handoff links disappear while stopped. No
  re-planning, no model/context edit, no Mac changes.