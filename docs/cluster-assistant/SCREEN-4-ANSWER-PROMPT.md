# Screen-4 decision — answer to the open question

**Paste this into the new session as its instruction. It answers the A/B/C
question; do not ask it again.** It is an addendum to
`docs/cluster-assistant/CHAPTER-7-START-PROMPT.md`, which still owns the live
Phase 2b acceptance run and the evidence document.

---

## The answer: Option A

Feed the eight Screen-4 labels from data that already exists:

- `assistantState()` — the assistant machine;
- `assistant.plan` — whether the user-facing plan exists;
- `assistant.operation` — the apply/start operation record, including
  `kind`, `state` and `progress.{phase,message,nodes_ready,nodes_total}`;
- `assistant.endpoint` — the final endpoint.

No backend change. No SSE. No new fetch, timer, state or progress source.

This decision is already recorded in
`docs/cluster-assistant/SCREEN-4-PROGRESS.md` and in `AGENTS.md`
("Resolved implementation decisions"). Do not re-open it.

---

## Why A, in one paragraph

The operation only ever reports `staging`, `starting`, `loading`, `ready`
(plus `queued`/`failed`). `run_apply_job` uses `staging` for **both** the
deployment preparation and the model copy, and `run_start_job` gets control
back only after the blocking activation call has already run the readiness
canary. So “Preparing runtime” cannot be separated from “Copying model”, and
“Verifying endpoint” cannot be separated from “Loading model” without a new
progress signal inside activation. Option B would therefore not be a vocabulary
rename for the second pair; Option C (SSE) changes transport, not the missing
signal. A is the only option that is truthful today.

---

## The mapping (real execution order)

| # | Screen-4 label | Source | Observable |
|---|---|---|---|
| 1 | Checking Macs | `assistantState()` is `measuring` or `measuring_model` | yes, indeterminate |
| 2 | Planning | `assistantState()` is `recommending` or `planning` | yes, indeterminate |
| 3+4 | Preparing and copying the model | operation `kind === 'apply'`, `state === 'applying'` (`progress.phase === 'staging'`) | yes, `nodes_ready` / `nodes_total` |
| 5 | Starting coordinator | operation `kind === 'start'`, `state === 'starting'` | yes |
| 6+7 | Loading the model and verifying the endpoint | operation `kind === 'start'`, `state === 'loading'` | yes |
| 8 | Ready | operation `state === 'ready'` **and** `endpoint.ready === true` | yes |

Two recorded corrections:

- **Order:** the brief lists Preparing/Copying before Planning. This
  architecture plans first because the signed placement authorises staging.
  Show Checking → Planning → Preparing/Copying → Starting → Loading/Verifying →
  Ready.
- **Rows:** eight labels, six independently progressing rows. The two merged
  labels are sub-steps of their row and must never animate on their own.

---

## Rendering rules

- A row is `done`, `active`, `pending` or `failed`. Only one row is `active`.
- Never claim a phase is done before its source says so.
- Indeterminate spinner where no count exists. Never a fake percentage.
- Staging may show `nodes_ready` / `nodes_total` only. No bytes, no tokens, no
  rate.
- On failure, the active row becomes failed, later rows stay pending, and the
  card shows the one action from `error.action`.
- The Ready row and the endpoint URL appear only when `state === 'ready'` and
  `endpoint.ready === true`.
- The surface is display-only. It adds no request, timer, state or backend
  field.

---

## What to implement (only after the live Phase 2b run passes)

### 1. `apps/omlx-web/omlx_web/static/js/cluster_v2.js`

Add one pure function next to the assistant block, for example:

```js
assistantProgressRows() {
  return [
    { key: 'checking', label: ..., state: ..., detail: ... },
    { key: 'planning', ... },
    { key: 'preparing_copying', ... },
    { key: 'starting', ... },
    { key: 'loading_verifying', ... },
    { key: 'ready', ... },
  ];
}
```

- Derive everything from `this.assistant.state`, `this.assistant.plan`,
  `this.assistant.operation`, `this.assistant.progress`,
  `this.assistant.endpoint`.
- `preparing_copying` detail: `nodes_ready` of `nodes_total` Macs ready, from
  `assistant.progress` (fall back to the operation's values).
- Do not add a new fetch, timer, generation, state field or API call.
- Do not parse `progress.message` prose to decide a phase.

### 2. `apps/omlx-web/omlx_web/templates/dashboard/_cluster_v2.html`

- Add one progress surface inside the assistant card, visible while
  `assistantState()` is a running state and in `ready`/`failed`.
- Mark it `data-cluster-v2-assistant-progress`.
- Each row: `data-assistant-progress-row="<key>"`, a label, a state marker, and
  the detail line where one exists.
- The two combined rows carry both labels in their text:
  “Preparing and copying the model” and
  “Loading the model and verifying the endpoint”.
- Keep the existing endpoint visibility rule untouched.

### 3. `apps/omlx-web/omlx_web/i18n/en.json`

Add the label keys (English). Mirror them into the existing locale files with
`cd apps/omlx-web && python normalize_i18n.py`; the repo's identical-key-set
test requires it. English fallbacks only, no translation.

### 4. Tests

Static (`apps/omlx-web/tests/ui/test_cluster_v2_wizard.py`):

- the progress surface and all six row keys exist in the template;
- both combined labels exist;
- no percentage/byte/rate text in the surface;
- the endpoint markup still sits behind `assistantReady()`.

Node-gated (`apps/omlx-web/tests/test_cluster_ui_integration_ui.py`; use the
`jsc` shell locally when `node` is absent):

- `measuring` / `recommending` → row 1/2 active, no operation row;
- apply applying → `preparing_copying` active with the Mac counts, later rows
  pending;
- apply applied → `preparing_copying` done; `starting` active only after the
  start operation exists;
- start loading → `loading_verifying` active, endpoint still hidden;
- ready → `ready` done and the endpoint exposed;
- failed at any point → active row failed, one action, later rows pending;
- a stale generation cannot advance a row.

Keep the runtime verification green:

```sh
uv run pytest tests/cluster_assistant -q
uv run pytest tests/test_cluster_*.py -q
uv run pytest apps/omlx-web/tests -q
```

### 5. CSS + docs

- Rebuild CSS only if the existing utilities are insufficient:
  `cd apps/omlx-web && python build_css.py`.
- Update `UPSTREAM.md` rows for the patched web files, `README.md` (current
  milestone), and `CLUSTER-ASSISTANT.md` (status).

---

## Guardrails

- **Validate first.** This surface is only built after the live Phase 2b run
  passes. If the run fails, fix the run, record it in
  `PHASE-2B-RUN.md`, and leave Screen 4 for the next chapter.
- **Do not change the backend** (`models.py`, `routes.py`, `operations.py`,
  `provisioning.py`, `start.py`, `election.py`, `measurement.py`, `state.py`,
  `recommendation.py`, `routes.py`, `server.py`) to make a phase observable.
  That is the deferred B follow-up.
- **Do not add SSE.**
- **Do not re-open the decision.** If the live run shows a combined row hides a
  meaningful wait, record the evidence in `PHASE-2B-RUN.md` and propose the B
  follow-up later; keep the surface truthful now.
- **No token rate, no percentage, no byte count.**
- **No second tab, no second state machine, no new endpoint.**
- **English only.**

---

## What this does not solve (record, do not fix here)

- Screen-5 handoff (`Open Chat` preselects no model; `Connect agent` stays the
  endpoint card).
- Splitting `preparing` from `staging` (small B follow-up).
- A real `verifying` phase during activation (needs an activation progress
  source; larger B follow-up).
- Performance probe / `observed` rates.

Report back: whether the live run passed, the rows you implemented, the tests
added and their pass/skip status, every file changed, whether any backend file
or `server.py` changed (they must not), and what remains for Screen 5.