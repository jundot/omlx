# Chapter 5 — Apply / Start

Status: implementation brief

Parent: `PROJECT-BRIEF.md`

Scope: the first backend chapter that may touch staging, deployment, launch, supervisor and health.

---

## 1. Goal

Turn an **approved, model-bound plan** into a running cluster:

```text
approved plan
  → prepare/stage
  → start supervisor on the elected coordinator
  → wait for health
  → load model
  → smoke request
  → endpoint ready
```

This chapter is the only place where launch/deployment code may be wired.

## 2. What genuinely blocks Chapter 5

| Item | Blocks Chapter 5? | Rule |
|---|---|---|
| Apply/start orchestration | **Yes** | This is the chapter. Implement it here. |
| Distributed performance probe | **No** | Runs only after an approved deployment exists. Keep `measured=false` until then. |
| Model-less measurement | **No** | Recommendation follow-up. Do not relax `ModelIdentity`. |
| Screen-3 UI | **Only for end-to-end acceptance** | Build minimal bindings after the backend API and tests. |

Do not solve performance-probe availability or hardware-only measurement by
weakening this chapter’s contracts.

### 2.1 Remote start: the recorded decision

Rank 0 must stay local to the Mac that runs the process
(`ClusterDeployment.hosts[0].ssh == "127.0.0.1"`), and activation needs that
Mac’s own engine pool and readiness canary — it cannot run as a detached script
on another Mac. So “start the supervisor on the elected coordinator” means “the
coordinator’s own oMLX server runs the existing activation”.

The decision, taken once and recorded here, is **Option A: SSH control plane,
same activation**.

- The origin resolves the elected coordinator’s paired SSH target from the
  device registry at request time (never persisted), and drives the
  coordinator’s own admin routes over the existing SSH seam
  (`staging.run_remote_python`, which goes through `ssh_policy`):
  `/admin/api/cluster/plan` to sign the placement, `/admin/api/cluster/stage`
  to copy the model, `/admin/api/cluster/deployments` to activate.
- The loopback admin call on the coordinator authenticates with the key that
  Mac already stores locally. The key never reaches the origin, pairing
  exchanges no credential, and nothing is written to the Keychain for this.
- Staging and activation therefore run unchanged on the coordinator, with rank 0
  local to it. There is no new runtime, no second deployment registry, no second
  launch path, and no TCP listener on the origin.
- If the coordinator’s oMLX server is not running or refuses the admin call, the
  operation fails with one action (`retry_coordinator`, whose message says to
  open and configure oMLX on that Mac). Nothing starts on the origin.

Rejected: **B** (per-pair admin credential in the origin’s Keychain — reopens the
credential decision of §13.4/§14) and **C** (porting the excluded rank-0 SSH
draft `574f0460`, which is not in this base and changes how rank 0 is launched).

Two consequences are recorded rather than re-decided:

- **A single-Mac plan cannot be applied.** `ClusterDeployment` requires 2–64
  hosts, so a one-node election is refused with `409 single_node_plan` and the
  one action `add_mac`; no local-engine mode is invented.
- **The plan is not persisted.** `plan` stayed pure, so `apply` takes the
  plan-time inputs again (`context_tokens`, `rank0_extra_bytes`, `mtp`,
  `execution_profile`, `microbatch_size`), re-elects from the stored
  measurement, and refuses a `plan_signature` mismatch (`409 plan_mismatch`,
  one action `plan`). A client-supplied stage table is never obeyed.

### 2.2 The cluster-wide lock on a remote coordinator (recorded 2026-11-13)

`operations/active` reads the coordinator's own operation store. A remote
start drives the coordinator's **raw** cluster routes (`/plan`, `/stage`,
`/deployments`) through `_remote_ops`, so the origin's running record lives
only in the origin's store — the coordinator's store stays empty and answers
`operations/active` with `null` while it is actually working. A second origin
asking the same coordinator therefore sees "free" and both activate
(REVIEW-2026-11-13.md B5). The frozen rule is one apply/start at a time,
**cluster-wide**, not per origin.

The decision (E3, Option A) is to **mirror the operation record onto the
coordinator through the same admin seam** — no new control plane, no second
registry, no TCP listener on the origin:

- One new admin route on the existing assistant router,
  `POST /admin/api/cluster/assistant/operations/mirror`, accepts
  `{operation_id, kind, coordinator_node, state}` (plus the model/context for a
  sensible display) and calls a new `OperationStore.mirror(...)`. A running
  mirror is refused with `OperationInProgressError` when another record is
  already running; a terminal mirror (`applied`/`ready`/`failed`) frees the
  lock. The mirror reuses the existing record shape and the existing
  `active()` query, so `operations/active` works unchanged.
- The origin mirrors `running` to the coordinator **before** it activates
  (remote coordinator only; a local coordinator already shares the store),
  and mirrors the terminal state when the operation finishes or fails. A
  refused mirror is the existing `409 operation_in_progress` + action
  `status`. Release is best-effort: a coordinator that cannot be reached for
  the release logs a warning rather than failing an already-ready cluster.
- Mirror records are lock markers, not the coordinator's own assistant
  sessions: `latest_session()` skips them (the coordinator's deployments list
  remains its return surface). A stale mirror (origin crashed mid-operation)
  stops counting as running after a generous TTL so a wedged lock cannot
  block the cluster forever; the TTL is bounded well above a normal
  staging/launch.

Rejected: **B** (a separate lock route with its own state) — a second piece of
distributed state to keep consistent with the operation records, for no gain
over reusing the store. The mirror is the operation record, so there is one
source of truth.

Test (T1, red first): two origins, one coordinator. Origin A mirrors a
running apply/start; origin B's `_busy_on_coordinator` must see A's operation
and refuse with `operation_in_progress`; after A mirrors terminal, B proceeds.
Fake-driven through the admin seam, no real Mac.

## 3. Scope

### In scope

- Accept an approved plan from `plan`.
- Validate `measurement_id`, `plan_signature`, model identity and `mtp`.
- Prepare/stage the model through existing oMLX paths.
- Start the supervisor on the elected coordinator.
- Support local and remote coordinator start through the existing launch
  layer.
- Return an operation handle and status.
- Wait for health and model readiness.
- Run one smoke completion.
- Set `endpoint.ready = true` only after health/smoke.
- Surface one action on failure.
- Tests with fakes for launch/deployment; no real cluster in unit tests.

### Out of scope

- Distributed performance probe before an approved deployment.
- Model-less measurement / `hardware_only=true`.
- Recommendation UI.
- Full chat/agent UI.
- Tensor parallelism.
- A second runtime, a second deployment registry, or a second launch path.
- Silent local fallback when the elected coordinator fails.
- Estimated token rates in `predicted`.

## 4. Contracts

### 4.1 Apply

```text
POST /admin/api/cluster/assistant/apply
```

Input:

- `measurement_id`
- `plan_signature`
- approved plan identity:
  - `model_path` or model size/layer identity,
  - `context_tokens`,
  - `mtp`,
  - `coordinator_node`,
  - `rank_order`,
- optional `origin_node` for diagnostics only (never a rank-0 fallback).

Rules:

- The server recomputes/validates the plan and signature; do not trust a
  client-supplied stage table.
- Reject stale measurement, mismatched model identity, mismatched `mtp`, or a
  different node set.
- `apply` may stage/prepare; it must not start inference.
- Return an operation handle:

```json
{
  "operation_id": "op-...",
  "state": "applying",
  "poll": "/admin/api/cluster/assistant/operations/op-..."
}
```

### 4.2 Start

```text
POST /admin/api/cluster/assistant/start
```

Input:

- `measurement_id`
- `plan_signature`
- `coordinator_node`
- optional `operation_id` from apply.

Rules:

- Start only an applied/approved plan.
- Start the supervisor on the elected coordinator.
- Remote start uses the existing oMLX launch/SSH/deployment layer; no new
  runtime.
- Do not block the HTTP request on the long-running supervisor.
- Return an operation handle; do not report `ready` until health/smoke.
- If the elected coordinator cannot start, fail with one action. Do not start
  the local Mac instead.

Mechanism (recorded in §2.1, Option A):

- The origin calls the coordinator’s own `/admin/api/cluster/plan` and
  `/admin/api/cluster/deployments` over the existing SSH seam
  (`staging.run_remote_python`, `ssh_policy`); the coordinator’s loopback admin
  call uses the key that Mac already stores locally.
- When the coordinator *is* this Mac, the same pipeline runs in-process
  (`prepare_cluster_deployment`, `start_cluster_staging`,
  `activate_cluster_deployment`).
- `start` re-prepares and refuses a placement that changed since `apply`
  (`placement_changed`, one action `plan`) instead of launching an unapproved
  rank map.
- Operation states: `queued → starting → loading → ready`, or `failed` with one
  `error.action` (`retry_coordinator`, `diagnostics`, `plan`). `loading` is the
  activation itself, which ends with the readiness canary.

### 4.3 Operation status

```text
GET /admin/api/cluster/assistant/operations/{operation_id}
```

States:

```text
queued
applying
applied
starting
loading
ready
failed
```

Each state includes:

- `operation_id`,
- `measurement_id`,
- `plan_signature`,
- `coordinator_node`,
- `endpoint` (`ready=false` until health),
- `progress`,
- `warnings`,
- one `error.action` on failure.

Keep this separate from the measurement status endpoint.

## 5. Implementation sequence

1. Add operation/state types and a small persisted operation store.
2. Implement `apply` validation and staging through existing oMLX paths.
3. Implement `start` local path through existing launch/deployment code.
4. Implement `start` remote path through existing SSH/launch code.
5. Add operation status and progress.
6. Add health wait and model-ready check.
7. Add one smoke completion.
8. Set endpoint `ready` only after health/smoke.
9. Add failure actions and no-local-fallback tests.
10. Add minimal UI bindings only after the backend is stable.

## 6. Performance probe policy

`run_cluster_performance_probe` needs an approved `ClusterDeployment`. Until
that exists:

- `compute` is `None`,
- `predicted.measured` is `false`,
- `predicted.critical_stage_seconds` may be `null`,
- recommendation `measured_speed` is `false`.

This is not a Chapter 5 failure. Chapter 5 starts and verifies the endpoint
without claiming a rate. The probe may run after the approved deployment/start
and may fill `observed`; it must never fabricate a rate.

Do not:
- call the performance probe from `measure`, `recommend` or `plan`;
- create a fake deployment just to get a probe;
- fill `predicted` with token rates.

## 7. Model-less measurement policy

Model-less / hardware-only measurement is a recommendation follow-up, not part
of Chapter 5.

Chapter 5 requires:

- a ready,
- model-bound,
- matching-node-set,
- matching-model-identity,
- matching-`mtp`

measurement.

Do not relax `ModelIdentity` or the `measurement_refusal` contract in this
chapter.

## 8. UI policy

The backend is the first deliverable.

Detailed UI brief: [`SCREEN-3-MINIMAL.md`](SCREEN-3-MINIMAL.md).

After the backend and tests pass, add only minimal Screen-3 bindings:

- render `selected_model`,
- render `selected_context_tokens`,
- show plan/approve state,
- call `apply`/`start`,
- show operation progress,
- show endpoint when `ready=true`,
- show one action on failure.

The UI must not:
- launch inference itself,
- construct a plan,
- bypass validation,
- use the app origin as a coordinator fallback,
- add German strings.

## 9. Tests

Unit/fake tests:

1. apply rejects stale measurement.
2. apply rejects a different model identity.
3. apply rejects a different `mtp`.
4. apply rejects a different node set.
5. apply stages but never starts inference.
6. start rejects an unapplied/unapproved plan.
7. start uses the elected coordinator, not the origin.
8. remote coordinator start is attempted through the existing launch seam.
9. failed coordinator start does not start local rank 0.
10. endpoint stays `ready=false` until health/smoke.
11. operation status transitions are monotonic and carry one action.
12. performance probe is absent before approved deployment and does not block.
13. UI bindings are not required for backend tests.

Integration (when hardware is available):

- 2 Macs, approved plan, remote coordinator, start, health, smoke.
- M1 origin + M5 coordinator.
- coordinator start failure produces a clear action, not a local fallback.

## 10. Acceptance criteria

- Approved model-bound plan can be applied.
- Supervisor starts on the elected coordinator.
- Health and model readiness are verified.
- Endpoint is not `ready` before health/smoke.
- One smoke completion succeeds.
- Failure has one actionable next step.
- No local fallback.
- No performance-rate fabrication.
- No model-less measurement added to `plan`.
- Operation status is observable.
- Tests pass without a real cluster.

## 11. Stop conditions

Do not continue into full UI, performance optimization, hardware-only
measurement, or tensor parallelism in this chapter. Stop after the backend
apply/start path and its tests; then wire minimal UI separately.