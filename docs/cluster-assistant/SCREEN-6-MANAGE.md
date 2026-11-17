# Screen 6 — Running / Manage: state + stop/restart

Status: implemented (2026-11-07). Decision and implementation record.

Parent: `PROJECT-BRIEF.md` §5 (Screen 6), `AGENTS.md` §2 (frozen decision),
`SCREEN-5-HANDOFF.md` (the ready surface this extends).

The slice is deliberately small: **state + stop/restart, nothing more.**
Re-planning, editing the model or context of a running cluster, and
adding/removing a Mac are a later chapter.

---

## 1. What the user sees

When an operation is `ready` and `endpoint.ready` is true, the existing ready
card shows the model, context, endpoint, Chat/agent handoff and one new
**Stop cluster** button.

After a successful stop:

* the endpoint URL, `Open Chat` and `Connect agent` disappear;
* one line says the cluster is stopped and the endpoint is closed;
* **Start again** appears;
* the Ready progress row stays the last reached row (it does not regress to
  Loading).

**Start again** re-applies the frozen in-memory plan inputs (the same
`measurement_id`, model, context, MTP, profile and microbatch) and lets the
existing apply → start poll launch the deployment again. It does not measure,
recommend or plan again. If the measurement has gone stale, the backend answers
`stale_measurement` with the one action `measure` and the existing failure card
renders it.

After a page reload the card **resumes** (Phase C1, recorded in
`ROADMAP-TO-CHAPTER-10.md` §6 C1; superseded the original no-resume limitation
on 2026-11-13, see `REVIEW-2026-11-13.md` decision E1):
`GET /assistant/session` returns the newest ready start operation (else the
newest applied apply) with its frozen plan inputs, coordinator, deployment ID
and stopped flag — no key, SSH target or secret. The idle card shows a session
strip with Running/Stopped and offers **Start again** (re-applies the stored
inputs; apply re-proves the plan signature, so drift is still `plan_mismatch`)
and **Stop**. A running operation is polled, not resumed; failed records are
not a session. The existing deployments list remains a return surface, not
the only one.

---

## 2. Stop is a relay to the coordinator's own unload

Stop must work for a local **and** a remote coordinator without the browser
holding another Mac's key. The browser therefore calls one assistant route:

```text
POST /admin/api/cluster/assistant/stop
{"operation_id": "op-…-start-…"}

200 {"operation_id", "deployment_id", "stopped": true, "warnings": [...]}
```

The operation record is the only input. It carries the elected coordinator, the
endpoint port and the deployment ID the coordinator assigned, so the route never
re-elects and a stale measurement cannot block a stop.

| Coordinator | Seam |
|---|---|
| this Mac | `_local_admin` → `POST 127.0.0.1:<port>/admin/api/cluster/deployments/{id}/unload` with the key this Mac stores |
| another Mac | `remote_admin` (the existing SSH seam) → that Mac's own bare interpreter authenticates its loopback unload with its own key |

The key never crosses the assistant API and pairing exchanges no credential.
The remote target is resolved at request time from the paired device record
(`ssh_user`) and the measured endpoint host; an unpaired coordinator answers
`node_not_paired` / `pair_mac`, never a local fallback.

### 2.1 Responses and one action

| Situation | Code | Action |
|---|---|---|
| unknown operation | `unknown_operation` (404) | `status` |
| not a ready start / no deployment ID | `operation_not_ready` (409) | `status` |
| remote coordinator never paired | `node_not_paired` (409) | `pair_mac` |
| no answer (transport 0/401/403, OSError) | `coordinator_unreachable` (409) | `retry_coordinator` |
| the coordinator answered but did not stop it | `stop_failed` (409) | `diagnostics` |
| the coordinator no longer has that deployment (404) | success | — |

A second stop is idempotent: it returns the stored record and does not relay
again. A failed stop leaves `stopped` false, so the card keeps showing the
running cluster and Stop remains the retry.

---

## 3. The record

`OperationRecord` (and `AssistantOperationStatus`) gained three additive fields:

* `deployment_id` — the coordinator's ID, written when the start becomes
  `ready` from the activation report;
* `coordinator_local` — recorded when the operation is created, so stop knows
  which seam to use without guessing;
* `stopped` — set by a successful stop; `state` stays `ready`.

`state` deliberately gains **no** `stopped` value: the frozen state machine has
no such state, and the backend cannot report a transition it does not have.
`stopped` is the authoritative boolean. The endpoint stays historically ready
(it *was* proven ready); the card hides it while `stopped` is true, and
`assistantReady()` requires `!stopped`.

The stop adds the warning `deployment_stopped` to the ready record, visible in
Details. No API key, SSH target or host is stored.

The ready/stopped card also keeps the Phase A plan summary: `Macs` (selected
subset plus `excluded_nodes`), `Tier`, `Context`, `Offload` and `Experts` come
from the plan's `resource_policy`, and the operation status repeats the same
policy block. A reload cannot rebuild it (the operation ID and plan inputs still
do not survive), so the deployments list remains the return surface.

---

## 4. Tests

Backend (`tests/cluster_assistant/test_stop_api.py`, fake-driven):

* local unload path/method/port and `stopped: true` on the record;
* idempotent second stop (one relay);
* remote coordinator target (`user@host`) through `remote_admin`;
* an unpaired coordinator asks to pair and is never dialled;
* a non-ready operation and an unknown operation have one action;
* a vanished deployment counts as stopped;
* no answer keeps the operation running with `retry_coordinator`.

UI static (`apps/omlx-web/tests/ui/test_cluster_v2_wizard.py`): Stop/Start again
are ready-gated, the stop method calls only the assistant route (no
`/deployments`, `/unload`, `fetch`, SSH), Start again reuses the frozen inputs
and never plans/measures/recommends, and a stopped ready operation keeps the
Ready row.

UI Node (`apps/omlx-web/tests/test_cluster_ui_integration_ui.py`): stop posts
`{operation_id}` to `/assistant/stop`, clears the endpoint and shows the stopped
state; a backend `stopped` snapshot renders without an endpoint; Start again
posts `apply` with the same plan signature and measurement and makes no
plan/measure/recommend call.

---

## 5. Live evidence

`PHASE-2B-RUN.md` §13: a real 2-Mac start on `.11`, a stop that marked the
record `stopped: true` with `deployment_stopped`, an idempotent second stop, and
a start-again from the same frozen inputs that reached ready with the same
deployment ID and `stopped: false`.

---

## 6. Explicitly out of scope

* Re-planning or editing a running cluster's model/context.
* Adding or removing a Mac while running.
* A new wizard state, tab, page or SSE stream.
* A second manage surface: the existing deployments list stays the return
  surface after a reload.
* Tray or macOS-app work.