# Screen 5 — Ready: the handoff, and `observed`

Status: implemented (2026-11-07). This document records the decision and what
shipped; §6 is the `observed` decision that closes the row in
`PHASE-2B-RUN.md` §6.

Parent: `PROJECT-BRIEF.md` §5 (Screen 5), §12.4 (one action per failure),
`SCREEN-3-MINIMAL.md` §11 (the deferred items this chapter finishes),
`SCREEN-4-PROGRESS.md` §8 (the card this chapter extends).

The screen already existed. This chapter finishes its two buttons and answers
the one question that had been open since Chapter 5: **what `observed` means and
who may fill it.**

---

## 1. The link contract

Both links are built from values the backend reported. The component computes
nothing except a URL string.

```text
Chat:   <base>/chat?model=<serving model id>
Agent:  <endpoint>/v1                      (base URL)
        <serving model id>                 (model id)
        Settings → API key on <coordinator> (a sentence, never the key)
```

### 1.1 `<base>`

| Coordinator | Chat link | Why |
|---|---|---|
| this Mac | `/chat?model=<id>` (relative) | the browser already talks to this server, so its session and its injected key apply |
| another Mac | `<endpoint.url>/chat?model=<id>` | the endpoint **is** that Mac's page/API base; the origin cannot proxy because it never has that Mac's key and must not get it |

A remote coordinator therefore gets one English sentence under the button: the
chat page is that Mac's own page, that Mac's admin session applies (a session
cookie is per host, so the user may have to sign in there once), and that page
injects **that** Mac's own key into its own `localStorage` — which is exactly why
the origin needs none. The card does not pretend otherwise.

`assistantCoordinatorIsLocal()` compares `endpoint.node` (then the operation's
or plan's `coordinator_node`) with `selfDevice().node_id`. The endpoint is
`null` unless `assistantReady()` is true, so a missing coordinator can never
produce a link.

### 1.2 `<serving model id>`

The recommendation answers with the **planner path**
(`/Users/andreaskolb/Models/dev-1.5b`), while `/v1/models` advertises the
**serving id** (`dev-1.5b`). The card's `modelOptions` carry both, so:

```js
assistantServingModelId() {
    return modelOptions.find(m => m.model_path === selectedModel)?.id || '';
}
```

* resolved → the link carries `?model=<id>`;
* unresolved (or the catalogue has no serving id) → the link carries **no**
  `model` parameter. The chat page then selects its normal default. The agent
  card says the serving id could not be resolved and asks the user to pick the
  model in Chat. No id is invented and no different model is silently chosen.

### 1.3 The agent card

It shows the base URL (`<endpoint>/v1`), the serving model id, a copy button for
the id, and the sentence naming where the key lives on the coordinator. There is
**no key, token, cookie or credential** in the DOM, a URL, a fixture, a log or a
commit. The existing endpoint copy button is unchanged.

### 1.4 Ready-only

Everything above is inside the block gated by `assistantState() === 'ready'`
**and** `assistantReady()` (`state === 'ready' && endpoint.ready === true`). The
endpoint stays `null` otherwise; no port-0 URL is ever rendered. A failure keeps
its one `error.action` (SCREEN-3 §7). “Open details” is still the existing
Details panel.

---

## 2. `/chat` preselection

The chat page had no query-string handling. The change is deliberately small:

* `requestedModelFromUrl()` reads `model` exactly once from
  `new URLSearchParams(window.location.search)`;
* `applyRequestedModelFromUrl()` runs once, after `loadModels()` has a non-empty
  `availableModels`, and selects the id only when `isModelAvailableOnServer()`
  says it is present;
* an unknown value leaves the normal default selection untouched and marks the
  attempt used, so a later visibility refresh cannot override a user's choice;
* nothing is fetched, redirected, written to history or added as a new entry —
  the parameter is already in the URL, so a reload applies it again;
* the `endpoint` parameter is rejected by omission: the page reads only
  `model`. There is no redirect, no fetch and no key in the query string.

No streaming, upload, audio, theme, settings or transport code was touched.

---

## 3. What the card renders for a measured operation

If `assistant.observed` is a block, the ready surface shows one line:

```text
42.4 tok/s decode · 320.5 tok/s prefill · first token in 250 ms ·
measured 2026-11-07T23:40:12+00:00
```

Only fields the backend measured are shown; a missing prefill rate is omitted,
not invented. With `observed: null` the line is absent and the card says nothing
about speed (the existing `performance_unmeasured` warning in Details still
explains why).

The card's operation poll still stops at `ready` (AGENTS §2, SCREEN-4 §3). The
probe is written **after** that terminal state, so a probe that lands later is
not rendered live by this slice; it is part of the operation record returned by
`GET /admin/api/cluster/assistant/operations/{id}` and is rendered whenever a
ready status already carries it. Screen 6 (state/manage) reads the same record.

---

## 4. Where `observed` lives

One additive field, no state-machine change:

* `OperationRecord.observed: OperationObserved | None` in
  `omlx/cluster/assistant/operations.py`, persisted in
  `assistant-operations.json` (schema stays 1; a record without the field loads
  as `None`);
* `AssistantOperationStatus.observed: OperationObservedView | None` in
  `omlx/cluster/assistant/models.py`, so the status poll and the router expose
  the same nullable block;
* `OperationStore.observe()` attaches a measurement (or one warning) only to a
  `ready` record. It is not a transition and cannot move an operation; `failed`,
  `applying`, `starting` and `loading` are untouched.

The plan is **not** persisted and is not rewritten. `predicted` keeps its
meaning, and `plan.observed` stays `null`; a test pins that a plan re-run after a
measured operation is identical.

---

## 5. What may fill it, when, and how it is switched on

### 5.1 What fills it

A **real non-streaming completion** on the elected coordinator supplies the
decode rate, and one **streaming completion** supplies TTFT and prefill from its
final usage chunk. oMLX already computes the end-to-end timing fields
(`usage.time_to_first_token`, `usage.prompt_tokens_per_second`,
`usage.generation_tokens_per_second`), and the coordinator's existing
`engine/distributed.py::_record_strategy_benchmark` persists the same request as
a strategy benchmark.

The probe maps them to the recorded block:

| `observed` field | Source |
|---|---|
| `prefill_tok_per_s` | `usage.prompt_tokens_per_second` from the streaming final chunk (optional: a cached prefix is not prefill work) |
| `decode_tok_per_s` | `usage.generation_tokens_per_second` from the non-streaming reply (required) |
| `ttft_ms` | `usage.time_to_first_token × 1000` from the streaming final chunk (optional) |
| `measured_at` | the coordinator's timestamp for the completion |

The live check (2026-11-07) showed the distributed **non-streaming** path
reports `generation_tokens_per_second` but not TTFT or prefill. B2 therefore adds
one streaming completion after ready, with
`stream_options.include_usage=true`; without that option the final chunk carries
no usage. Live: prefill 9.06 tok/s, decode 21.43 tok/s, TTFT 110 ms. Nothing is
derived to fill a missing field.

What is **not** the source:

* the 2-token readiness canary — it proves readiness and cannot measure decode;
* the synthetic `launch.run_cluster_performance_probe` / `performance_worker` —
  it is a pre-launch synthetic benchmark of weight bytes per second and
  collective latency, not token rates; its profiles feed the **next** plan (B1)
  and never this `observed` block;
* any estimate derived from model size or the plan's predicted seconds.

### 5.2 When and where

After the operation is already `ready`, asynchronously, on the elected
coordinator, through the same local/remote seam apply/start use
(`CoordinatorOps.observe` for decode, `CoordinatorOps.stream_observe` for
TTFT/prefill):

* local coordinator — `_local_admin` posts the non-streaming completion and
  `_local_stream_observe` reads the streaming SSE final chunk, with the key that
  Mac stores;
* remote coordinator — the same bare-interpreter snippets authenticate the
  coordinator's own loopback call with its own key. The origin never sees or
  stores that key; pairing exchanges no new credential.

The probe is never inside the blocking activation, never called from the
browser, and `auto_tune` stays `false`, so the approved placement remains the
launched placement. The card has a bounded **Refresh** that re-reads the
operation once, so a block that lands after the ready poll is visible without a
reload.

### 5.3 The switch

**Decision: a server setting, default off, toggled in-app.**

```text
ServerSettings.assistant_performance_probe = false
GET/POST /admin/api/cluster/assistant/performance-probe
```

It is read on the Mac that posts `start` (the origin, whose operation record
receives `observed`). The card's `Performance probe` button writes the setting
through that route; no edit-JSON step and no per-operation flag, so the start
request shape is unchanged.

### 5.4 Failure

A probe that cannot answer records one warning (`performance_probe_failed` with
the reason) on the already-`ready` operation. It never unreadies a cluster, stops
a deployment, or fails the operation. A reply without `decode` and `ttft` is
“not measured”, never zero.

---

## 6. The Screen-5 row in `PHASE-2B-RUN.md` §6

**Closed 2026-11-07.**

* The handoff exists: `Open Chat` opens the existing `/chat` with the resolved
  serving id (the coordinator's own page for a remote coordinator), and
  `Connect agent` shows the base URL, the serving id and where the key lives.
* The speed question is answered: `observed` is a nullable block on the
  operation, filled only by the real post-ready completion described above, off
  by default, and `predicted` is never rewritten. With the probe off (the
  default) the block stays `null` and the card says nothing about speed.

Live evidence for this chapter is recorded in `PHASE-2B-RUN.md` §12; the
fake-driven tests are listed below.

---

## 7. Tests

Static (`apps/omlx-web/tests/ui/test_cluster_v2_wizard.py`):

* the two links are built from `endpoint.url` and the catalogue's serving id,
  both ready-gated, no id fallback, no secret in the card;
* the chat page’s preselection reads the parameter once, never writes history,
  fetches or carries a key, and an unknown value leaves the default alone;
* the operation fixtures carry `observed` (null and measured variants).

Node (`apps/omlx-web/tests/test_cluster_ui_integration_ui.py`):

* `Open Chat` targets `http://172.16.1.11:8099/chat?model=…` for a remote
  coordinator and `/chat?model=…` for a local one;
* without a resolved id it targets the page with no `model` parameter;
* the agent base URL and model id are the endpoint’s and the catalogue’s;
* a measured operation renders exactly the measured line, and `observed: null`
  renders none;
* the chat methods execute: a known `?model=` is selected, an unknown one leaves
  the default.

Backend (`tests/cluster_assistant/test_observed_api.py`, fake-driven):

* a normal start stores `observed: null` and runs no probe;
* a real completion fills the block and keeps the cluster `ready`;
* the record round-trips and an old record loads with `observed: null`;
* a failing probe (or a reply without timing) adds one warning and keeps
  `ready`;
* a re-run plan is unchanged (`predicted` identical, `observed: null`);
* the probe asks the coordinator for its **serving** model id.

---

## 8. What did not change

The ten assistant routes’ request/response shapes (except the additive
`observed`, `deployment_id`, `coordinator_local` and `stopped` fields on the
operation), the operation state machine, the error codes, the action strings,
`selected_model`/`selected_context_tokens` binding, endpoint visibility, the
Screen-4 rows and their sources, `election.py`, `measurement.py`,
`recommendation.py`, `omlx/cluster/routes.py`, `omlx/server.py`, the chat page’s
streaming/upload/audio/theme/settings behaviour, the eight wizard states, the
pairing flows and `apps/omlx-mac`.

The tenth route is `POST /admin/api/cluster/assistant/stop` (Screen 6); it adds
no new error code beyond the documented one-action refusals and no new state.

---

## 9. Still later (not this chapter)

* Rendering a probe that lands after the card’s ready poll live (Screen 6 now
  reads the same operation record; the gate roadmap adds the refresh).
* Model-less/hardware-only measurement.
* Tensor parallelism.

Screen 6 landed after this chapter (`SCREEN-6-MANAGE.md`).