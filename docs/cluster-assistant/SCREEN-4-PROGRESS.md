# Screen 4 — Progress: the recorded decision

Status: implemented (2026-11-07) — see §8. The decision in §1–§7 stands
unchanged; §8 records what shipped and the one place the card departs from the
letter of §5 (a reload has no plan-time inputs).

Parent: `PROJECT-BRIEF.md` §5 (Screen 4), §12.4 (one action per failure),
`SCREEN-3-MINIMAL.md` §5 (which already ships a reduced version of this).

---

## 1. The decision: Option A — map the data we already report

Screen 4 asks for eight named phases. The backend reports:

- the assistant's own state machine —
  `idle → measuring → recommending → measuring_model → planning → applying →
  starting → loading → ready`, plus `failed`;
- the operation kind (`apply` / `start`) and state
  (`queued | applying | applied | starting | loading | ready | failed`);
- `progress.{phase, message, nodes_ready, nodes_total}`, whose `phase` is one of
  exactly `staging | applied | starting | loading | ready | failed`;
- one `error.{code, message, action}` per failure;
- `endpoint.{node, host, port, url, ready}`.

**Decision: compose the eight labels from that data. No backend contract change,
no new endpoint, no SSE.**

Rejected:

- **B (enrich the backend phase vocabulary).** The operation `phase` already has
  one value per step the coordinator really performs (`staging`, `applied`,
  `starting`, `loading`, `ready`). A richer vocabulary would be new strings on a
  frozen contract to express distinctions the run cannot observe — see §3.
- **C (SSE/streaming).** Nothing here is a transport problem: the existing
  single-flight poll already delivers every transition, and polling a 4-Mac
  cluster's operation costs one request every few seconds. Streaming would add a
  second lifecycle (reconnect, resume, generation guard) for no new truth.

---

## 2. The eight labels and where each one comes from

The brief's list is **corrected** in one respect: planning must precede staging,
because the plan is what staging stages. The real order is:

| # | Label (en) | Label key | Source of truth | Shown while |
|---|---|---|---|---|
| 1 | Checking Macs | `cluster.assistant.progress.checking` | `assistantState() === 'measuring'` (entry measurement) or `'measuring_model'` | the measurement job runs |
| 2 | Planning | `cluster.assistant.progress.planning` | `assistantState() ∈ {'recommending', 'planning'}` | recommendation and the pure plan call |
| 3 | Preparing runtime **and** Copying model | `cluster.assistant.progress.preparing` | operation `kind='apply'`, `state ∈ {'applying'}`, `progress.phase='staging'` | the staging job runs |
| 4 | Starting coordinator | `cluster.assistant.progress.starting` | operation `kind='start'`, `progress.phase='starting'` | coordinator health + placement re-check |
| 5 | Loading model **and** Verifying endpoint | `cluster.assistant.progress.loading` | operation `kind='start'`, `progress.phase='loading'` | activation, which ends with the readiness canary |
| 6 | Ready | `cluster.assistant.progress.ready` | operation `state='ready'` **and** `endpoint.ready === true` | terminal |

Only **six** labels are independently observable. Two pairs are combined
because the backend cannot separate them:

- **Preparing runtime + Copying model.** `apply` prepares the coordinator and
  stages the model inside one job; there is no event between "runtime ready" and
  "first byte copied". Splitting them would require a new phase string that
  describes a step no code performs separately.
- **Loading model + Verifying endpoint.** The readiness canary runs inside the
  blocking activation call (`ops.activate`); the transition to `loading` is the
  last event before `ready`. There is no "loaded but not verified" state, and
  inventing one would let the UI claim a readiness the backend has not proven.

`applied` (the apply operation's terminal state) is **not** a Screen-4 phase: it
means "staged, nothing started". The assistant posts `start` immediately after
it, so the progress surface passes through it in one poll. If a future chapter
lets the user pause between apply and start, that needs its own label.

---

## 3. Rules the implementation must keep

1. **A phase may never animate without backend evidence.** Each row is driven by
   a value in §2. No timer, no optimistic advance, no "maybe it is loading now".
2. **No phase may lie about readiness.** `Ready` is rendered only when
   `operation.state === 'ready' && endpoint.ready === true`. `loading` is *not*
   "almost ready": between the canary and the state change nothing may be shown
   as ready.
3. **A failed phase stays failed.** The failure row replaces the phase list, shows
   `error.message`, and offers exactly one button labelled from `error.action`
   (SCREEN-3 §7). The failed row keeps the last phase the backend reported, so a
   user can say where it stopped.
4. **One row at a time is "current"; earlier rows become "done" only from the
   state that superseded them.** The poll's monotonic operation states make this
   safe; the measurement states are likewise terminal (`ready`/`failed`).
5. **No percentage, byte count or token rate.** `nodes_ready/nodes_total` may be
   shown as "2 of 3 Macs" (that is a count the backend reports). Progress is
   never converted into a percentage, and no token rate is displayed anywhere:
   `predicted.measured` is false and `observed` is null.
6. **No second tab, no new wizard state.** The surface lives in the existing
   Cluster tab, driven by `assistantState()` (SCREEN-3 §11).
7. **English only.**

## 4. Failure → action per phase

The action always comes from `error.action`; this table only records which
failures each phase can produce, so the next session can test them.

| Phase | Failures that can appear there | `error.action` |
|---|---|---|
| Checking Macs | `admission_probe_failed`, `budget_probe_failed`, `measurement_failed`, `measurement_refused`, `measurement_in_progress` | `measure` / `status` |
| Planning | `stale_measurement`, `no_feasible_candidate`, `no_feasible_plan`, `unsupported_parallelism` | `measure` / `adjust_model_or_context` |
| Preparing runtime + Copying model | `single_node_plan`, `node_not_paired`, `coordinator_unresolved`, `plan_mismatch`, `coordinator_unreachable`, `apply_failed` | `add_mac` / `pair_mac` / `configure_endpoint` / `plan` / `retry_coordinator` / `diagnostics` |
| Starting coordinator | `operation_not_applied`, `placement_changed`, `coordinator_unreachable`, `coordinator_start_failed` | `apply` / `plan` / `retry_coordinator` |
| Loading model + Verifying endpoint | `readiness_unproven`, `coordinator_start_failed`, `placement_changed` | `diagnostics` / `retry_coordinator` / `plan` |
| Ready | — (terminal; a later stop is Screen 6) | — |

`operation_in_progress` (`action: status`) can appear on any apply/start and
belongs to the phase that was requested.

---

## 5. What is shown when a phase cannot be distinguished

| Situation | What the row says |
|---|---|
| `progress.phase` is unknown to this build (a newer coordinator) | the previous known phase stays, with the raw `progress.message` underneath. Never a guess, never an empty row. |
| `progress.message` is present but `nodes_*` is absent | the message alone. The count line is omitted rather than shown as "0 of 0". |
| The operation is `applied` but the UI has not posted `start` yet | the Preparing row stays current; there is no "Staged" phase. |
| A measurement is `ready` but `plan` has not run | Planning stays current until the plan answer (or the failure) arrives. |
| The page is reloaded mid-operation | the card resumes from `operation_id` (`status` action) and renders the phase the stored record reports. |

---

## 6. Tests the next session must add

Fake-driven only; no live cluster in the suite.

1. **Static mapping.** For each of the six labels, assert exactly one
   `assistantState()`/`operation`/`progress.phase` predicate drives it, and that
   no label is bound to a timer, an interval or `setTimeout`.
2. **Ready gate.** `Ready` renders only for `state === 'ready' && endpoint.ready`;
   a `ready` operation with `endpoint: null` must render the failure/unknown
   state, never a URL (mirrors
   `test_assistant_endpoint_is_unreachable_until_ready`).
3. **Combined pairs.** A `progress.phase = 'staging'` event renders the
   Preparing/Copying row and never a separate "Copying" row; `'loading'` renders
   Loading/Verifying and never "Verifying" alone.
4. **Unknown phase.** A fixture with `progress.phase = 'future_phase'` keeps the
   previous row and shows the message.
5. **Failure.** One failure renders one button from `error.action` and keeps the
   last reported phase (`test_assistant_failure_renders_exactly_one_action_from_error_action`
   extended with the phase text).
6. **No rates.** No `tok/s`, `token rate`, `%` or byte total in the progress
   markup; `predicted.measured` and `observed` are not displayed.
7. **Fixtures.** Extend `apps/omlx-web/tests/ui/fixtures/cluster_v2/assistant_*.json`
   with one operation per phase plus one unknown-phase and one failure case;
   they must match the real `AssistantOperationStatus` shape.

No backend test changes are expected. If an implementation finds it needs a new
`progress.phase` value, that is decision B and must be re-asked, not slipped in.

---

## 7. Explicitly out of scope

- SSE/streaming, websockets, server push.
- A progress percentage, ETA, byte progress or throughput.
- A new route, a new tab, a new wizard state.
- Stopping/restarting a cluster (Screen 6, `Running / Manage`).
- The Screen 5 agent handoff.
- Tensor parallelism.

---

## 8. Implementation record (2026-11-07)

Built as decided: **no backend change for the rows**, no SSE, one component.

### 8.1 What the card renders

- One progress list inside the existing assistant card
  (`data-cluster-v2-assistant-progress`), rendered by
  `assistantProgressRows()` from the six-row table
  `ASSISTANT_PROGRESS_ROWS`. Each row carries
  `data-assistant-progress-row="<key>"` and
  `data-assistant-progress-state="done|current|pending|failed"`.
- The current row is `assistantProgressRow()`, which is
  `assistantReportedRow()` — one predicate per row, exactly the table in §2 —
  except in `failed`, where it is `assistant.failedRow`: the row the backend
  named before the failure arrived. `failAssistant()` captures it first, so a
  failure can say where it stopped.
- Rows before the current one are `done`; rows after it are `pending` and never
  animate. No timer, interval or `setTimeout` participates in the mapping.
- Under the current row: the backend's `progress.message`, and for the staging
  row (the only phase that reports per-Mac counts) `progress_nodes`
  (“2 of 3 Macs ready”). No percentage, byte total, ETA or token rate is
  rendered anywhere.
- The three per-state blocks (`applying`/`starting`/`loading`) keep their
  spinner and their one-sentence state line; the row list is the progress
  surface, so the backend message and the Mac count live there once.

### 8.2 The two pairs, the Ready gate and the failure row

- `progress.phase = 'staging'` (or the `applying` state) renders the combined
  **Preparing the runtime and copying the model** row; there is no separate
  “Copying”. `'loading'` renders the combined **Loading the model and verifying
  the endpoint** row; there is no separate “Verifying”.
- `Ready` is current only when `assistantReady()` is true — the same predicate
  that gates the endpoint link, `state === 'ready' && endpoint.ready === true`.
  A `ready` operation whose endpoint is not ready keeps the loading row (the
  canary has not been proven) and exposes no URL.
- A failure marks the current row `failed`, keeps its message, and leaves the
  failure card with its one `error.action` button.

### 8.3 Unknown phase

`assistantRowMessage()` shows `progress.message` when the phase is unknown to
this build (the row stays as it was) or when that phase owns the current row. A
known phase belonging to another row never labels this row, so a stale message
cannot describe the wrong step. `ASSISTANT_KNOWN_PHASES` is the operation
vocabulary; a coordinator that adds a phase simply gets its message shown
without a guessed row.

### 8.4 Reload mid-operation — the one correction to §5

§5 expects a reload to “resume from `operation_id`”. A reload genuinely has no
`operation_id`: the card keeps it only in memory, the browser may not call
`operations/active` (that is a server-to-server lock question), and the plan
inputs, model and context do not survive either — a resumed `ready` card would
have to invent the model label it no longer has.

What is implemented is the resume point the backend itself offers: the next
`apply`/`start` answers `409 operation_in_progress` carrying the running
`operation_id`, the action `status`, and one sentence. Pressing that one action
polls the named operation and renders exactly the phase its stored record
reports — no local fallback, no invented phase. Test:
`test_assistant_resumes_the_running_operation_the_backend_names`.

### 8.5 Tests

Static (`apps/omlx-web/tests/ui/test_cluster_v2_wizard.py`): the six-row table
and its order, one predicate per row with no timer, the combined pairs, the
Ready gate, the unknown-phase fallback, the failure row, and “no rate, no
percentage” on the rendered markup.

Behaviour under Node (`apps/omlx-web/tests/test_cluster_ui_integration_ui.py`):
the full walk `checking → planning → preparing → starting → loading → ready`
with the backend's own messages, Ready withheld without a ready endpoint, the
unknown phase keeping its row, a failure keeping its row, and the resume via
`status`.

Fixtures added: `assistant_operation_starting.json`,
`assistant_operation_unknown_phase.json`,
`assistant_error_free_disk_space.json`, `assistant_measure_inventory.json`.

### 8.6 What did not change

`election.py`, `measurement.py`, `recommendation.py`, `operations.py`, the
operation state machine, the error codes and the action strings. The only
backend addition is the free-space relay described in `PHASE-2B-RUN.md` §9
(the free-space assistant route); `state.py`/`models.py` gained the measured model list
the delete affordance shows, additively and optionally on disk.

**Live walk (2026-11-07):** the row mapping was exercised on real Macs after the
run that held the cluster ended — `PHASE-2B-RUN.md` §11 records the measurement,
plan, apply and start ids, the phase sequence with timings
(`applying/staging → applied → starting → loading(+canary) → ready` in 7.4 s for
a 2-Mac `dev-1.5b` plan), the endpoint and one real completion. The card's
rendering of those values stays covered by the Node-driven tests, because the
agent cannot click; §9.2 of the same document is the free-space relay's live
proof.