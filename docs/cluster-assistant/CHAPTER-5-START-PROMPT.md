# oMLX Cluster Assistant — Session 5

## CHAPTER 5 — Apply / Start backend (still no UI)

### WORKING DIRECTORY

```
/Users/andreaskolb/Documents/PycharmProjects/omlx-cluster-assistant
```

**BRANCH** `feat/cluster-assistant`

Last commits:

- `b70b92c6` "docs: define Chapter 5 apply and start plan"
- `74e683ea` "feat(cluster): recommend a model from the persisted measurement"

Do not push. Do not assume any prior conversation: every decision is in the
repository. Chapters 1–4 are finished; do not redo them. **This is the first
chapter allowed to wire staging, deployment, launch, supervisor and health.**

---

### READ FIRST, IN THIS ORDER

1. `AGENTS.md` — especially "Resolved implementation decisions" in §2 (the
   Chapter 5 bullets), the "Current important constraint" in §4, and the
   prohibitions in §8.
2. `docs/cluster-assistant/CHAPTER-5.md` — **the authoritative chapter brief**
   (scope, contracts, tests, acceptance, stop conditions). If it and
   `PROJECT-BRIEF.md` disagree, ask; do not pick silently.
3. `docs/cluster-assistant/PROJECT-BRIEF.md`
   - §8.5 — the predicted/observed plan shape (no token rates in `predicted`),
   - §8.6 — "no legal plan" and the one-action rule,
   - §9.2/§9.3/§9.4/§9.5/§9.6 — flow, remote start, endpoint, origin
     independence, staging,
   - §11.1a — recommendation vs. planning (already resolved, Option A),
   - §12.4 — error/action table (Screen 3 feedback),
   - §13.2/§13.3/§13.3a/§13.4 — endpoints, plan object, recommendation object,
     persistence,
   - §17.2 — no promised throughput before measurement,
   - §22 — resolved and still-open decisions,
   - Appendix A.5 — "rank 0 == local coordinator == endpoint owner".
4. `docs/cluster-assistant/UPSTREAM.md` — what is and is not in this base.
5. `CLUSTER-ASSISTANT.md`.
6. `omlx/cluster/assistant/state.py` — module docstring is the persistence
   contract (TTL, one-job rule, what a measurement covers, no secrets).
7. `omlx/cluster/assistant/measurement.py` — the three measurement rules.
8. `omlx/cluster/assistant/election.py` — `NodeFacts`, `budget_for()`, `elect()`,
   `PlanCandidate.shard_plan`, the no-fallback rule.
9. `omlx/cluster/assistant/models.py`, `routes.py` — the current API contract
   (exactly four routes).
10. `omlx/cluster/assistant/recommendation.py` — its contract must stay inert
    (`measure`, `recommend` and `plan` never launch anything).
11. `tests/cluster_assistant/` — especially `test_measurement_state.py`,
    `test_measure_api.py`, `test_plan_api.py`, `test_recommend_api.py`.
12. `tests/test_cluster_seams.py` — `test_no_unreachable_functions_in_the_cluster_package`.

Then inspect the oMLX code you will actually reuse. **Read it before designing;
do not infer the seams.**

| Area | File / symbol |
|---|---|
| Activation pipeline | `omlx/cluster/routes.py`: `POST /deployments` → `_activate_and_report`, `_create_deployment`, `_placement_signature` |
| Staging job | `omlx/cluster/routes.py`: `POST /stage`, `_run_staging_job`, `_record_staging_job`, `_staging_job_snapshot`, `GET /stage/{job_id}` |
| Request shapes | `omlx/cluster/routes.py`: `ClusterDeploymentRequest`, `ClusterHostRequest`, `ClusterPlanNodeRequest`, `_node_budgets`, `_reserve_bytes_for` |
| Deployment invariants | `omlx/cluster/deployment.py`: `ClusterDeployment.__post_init__` (`hosts[0].ssh == "127.0.0.1"`, 2–64 hosts, 64-hex `plan_hash`, contiguous ranks, `path_map`) |
| Launch / SSH seam | `omlx/cluster/launch.py`: `DistributedJobSupervisor` (`start`, `stop`, `status`), `preflight_remote_hosts`, `build_mlx_launch_argv`, `stop_deployment_processes`, `_cluster_ssh_argv`, `run_cluster_performance_probe` |
| Staging primitives | `omlx/cluster/staging.py`: `plan_cluster_staging`, `stage_files_from_source`, `check_disk_for_staging`, `run_remote_python`, `free_disk_bytes` |
| Health | `omlx/cluster/liveness.py`: `check_peers`, `raise_if_peer_lost`, `describe_failure`, `read_remote_marker`; `omlx/cluster/runtime.py`: `read_runtime_markers` |
| Smoke | `omlx/cluster/supervisor.py`: `run_worker_smoke`; the readiness canary in `_activate_and_report` (`engine.generate("__omlx_cluster_readiness__", …)`) |
| Registries | `omlx/cluster/registry.py`: `get_cluster_registry`, `get_device_registry` (paired addresses, `http_port`; **no SSH targets and no API keys in measurements**) |
| Engine start | `omlx/engine/distributed.py` (`DistributedJobSupervisor` use) and the pool activation used by `_activate_and_report` |
| House style | `tests/test_cluster_plan_approval.py`, `tests/test_cluster_ui_integration.py`, `tests/test_cluster_routes.py`, `tests/test_cluster_launch.py`, `tests/test_cluster_staging.py` |

---

### STATE OF THE REPO (verified facts — not guesses)

**Chapter 1 is frozen** (`election.py`, 1033 lines, 16 tests): scoring order,
no-fallback rule, origin independence, `MAX_CANDIDATE_NODES`, TP refusal.

**Chapter 2 is frozen** (`measurement.py`, 601 lines): no invented spread,
`compute=None` when unmeasured, installed-RAM refusal.

**Chapter 3 is frozen** (`state.py`, 741 lines; commit `a8e5d12a`):
`MeasurementStore`, `MeasurementRecord`, `ModelIdentity`,
`measurement_refusal()`, one JSON file
`<base_path>/cluster/assistant-measurements.json`, schema 1, atomic write,
newest 8, `MEASUREMENT_TTL_SECONDS = 900`, fail-closed load, interrupted jobs
recovered as failed, no SSH targets or keys stored.

**Chapter 4 is frozen** (commit `74e683ea`; `recommendation.py`, 861 lines):
recommendation reads only the hardware half of a measurement (Option A), may
evaluate any model, `plan` keeps the strict node-set/model/`mtp` match, and
`election.budget_for()` is the single measured-admission → `NodeBudget` builder.
32 new tests; `models.py` 815, `routes.py` 1001.

The assistant router exposes **exactly four routes**, all admin-gated by the
existing `server.py` mount (`_register_cluster_routes`: one `include_router` +
`set_assistant_getters(...)`; the only server.py patch site):

```
POST /admin/api/cluster/assistant/measure
GET  /admin/api/cluster/assistant/status?measurement_id=…
POST /admin/api/cluster/assistant/recommend
POST /admin/api/cluster/assistant/plan
```

`/measure` is a job (409 `measurement_in_progress`, action `status`);
`/plan` and `/recommend` are pure and never probe, read settings or write.
Two machine-readable 409s exist: `stale_measurement` (flat body, action
`measure`) and `no_feasible_plan` / `no_feasible_candidate` (object + code +
action). TP stays 400.

Tests: `tests/cluster_assistant` = **140 passed**; `tests/test_cluster_*.py` =
**1593 passed, 2 skipped**. `ruff check`/`format` clean, `mypy
omlx/cluster/assistant` clean (repo-wide baseline is noisy).
`tests/test_cluster_seams.py` `allowed_uncalled` lists nothing from the
assistant — keep it that way.

**What does not exist yet:** no operation/apply/start type or store, no
apply/start/operations route, no bridge from an assistant plan to a
`ClusterDeploymentRequest`, no remote-coordinator activation, no
health/smoke/`endpoint.ready` transition. `run_cluster_performance_probe` is
never called, so `compute` is `None`, `predicted.measured` is false and
`measured_speed` is false — that is correct, not a bug.

The validation cluster (172.16.1.11/.12/.13/.14) may be **offline**. No test may
touch SSH, a model on disk, or a real Mac.

---

### THE ONE DESIGN QUESTION TO SETTLE BEFORE CODING

`ClusterDeployment` requires `hosts[0].ssh == "127.0.0.1"`: rank 0 runs on the
Mac that owns the process. The elected coordinator is frequently **not** the
origin (that is the whole point of Chapter 1), and the rank-0 SSH draft
(`574f0460`) is **not** in this base (see `UPSTREAM.md`). So: how does the
origin start the supervisor on a remote elected coordinator without removing
that invariant and without building a second launch path?

- **A (recommended) — SSH control-plane, same activation.** The origin resolves
  the coordinator's paired SSH target and runs the **existing** activation on
  the coordinator through the existing SSH seam (`run_remote_python` /
  `_cluster_ssh_argv`, all through `ssh_policy`), then polls the coordinator's
  runtime marker over SSH (`read_remote_marker` / `read_runtime_markers`) for
  progress. On the coordinator, rank 0 is local, the existing
  `_activate_and_report` and readiness canary run unchanged, and staging/launch
  use the code that already exists. No new runtime, no new registry, no new
  credential, no rank-0 topology change.
- **B — coordinator admin API.** The origin posts the signed activation to the
  coordinator's existing `/admin/api/cluster/deployments` over the paired
  address with a Bearer admin key (`require_admin` already accepts it). This
  requires deciding where the coordinator's key lives (no key is exchanged by
  pairing today; brief §13.4 says Keychain only), so it reopens a security
  decision.
- **C — port the excluded rank-0 SSH draft (`574f0460`).** Rejected: not in the
  base, larger refactor, and it changes how rank 0 is launched.

Also settle and record, without asking twice:

- **Single-Mac plans.** `ClusterDeployment` requires 2–64 hosts, so a one-node
  elected plan cannot become a cluster deployment. Refuse `apply` with one
  action (`add_mac`, §8.6) and record it; do not invent a local-engine mode.
- **The plan is not persisted.** `plan` stayed pure (Chapter 3). `apply`
  therefore takes the plan-time inputs (`context_tokens`, `rank0_extra_bytes`,
  `mtp`, `execution_profile`), re-elects from the stored measurement, and
  refuses a `plan_signature` mismatch. Never trust a client-supplied stage
  table.
- **No secrets in the operation record.** No SSH targets, no API keys, no
  command lines. Paired SSH targets are resolved at request time from
  `get_device_registry()` and are not persisted.

Ask once (one question, three options), record the answer in
`docs/cluster-assistant/CHAPTER-5.md` (§2/§4), in `PROJECT-BRIEF.md` §13.2 and
in `AGENTS.md` "Resolved implementation decisions", then code. Do not silently
pick.

---

### GAPS THIS SESSION CLOSES

1. There is no operation type, state machine or store.
2. There is no `POST /apply`, `POST /start` or operation status route.
3. Nothing converts an elected `PlanCandidate`/`ShardPlan` into the
   `ClusterDeploymentRequest` the existing activation expects.
4. No remote-coordinator start path exists at all.
5. `endpoint.ready` is hard-coded false and nothing can make it true.
6. Nothing proves "coordinator start failed → no local fallback" at the API
   boundary.
7. The staging runner is private in `omlx/cluster/routes.py`
   (`_run_staging_job`). It must be **reused**, not copied — a second staging
   implementation is a hard prohibition.

---

### GOAL FOR THIS SESSION

(a) operation types + a persisted operation store with a monotonic state machine;
(b) `apply`: validate `measurement_id` + recomputed `plan_signature`, stage
    through the existing oMLX staging seam, never start inference;
(c) `start`: supervisor on the **elected** coordinator, local and remote, never
    blocking the request on the supervisor;
(d) operation status with progress and one action on failure;
(e) wait for health + model readiness, run one smoke completion, and only then
    set `endpoint.ready = true`;
(f) tests with fakes — no real cluster, no SSH, no model on disk;
(g) the small doc updates, same commit.

This session is **NOT**:

- UI, web template, dashboard JS, macOS-app work (only after the backend, and
  only with a new instruction);
- tensor parallelism;
- the distributed performance probe as a requirement (see below);
- model-less / `hardware_only` measurement;
- chat, tray, client handoff, SSE/streaming;
- a second wizard, control plane, deployment registry, staging implementation or
  launch path;
- German localization;
- estimated token rates to fill the missing probe.

---

### DELIVERABLES

#### 1. `omlx/cluster/assistant/operations.py` (new)

- `OperationRecord` (operation_id, kind `apply|start`, measurement_id,
  plan_signature, coordinator_node, rank_order, context_tokens, mtp, state,
  created_at, updated_at, progress, endpoint, warnings, error).
- `OperationStore`: one JSON file
  `<base_path>/cluster/assistant-operations.json`, schema 1, mode 0600, atomic
  replacement, bounded (newest 8), fail-closed load, interrupted operations
  recovered as `failed` on restart (same reason as a measurement job cannot
  outlive its process).
- States exactly as `CHAPTER-5.md` §4.3:
  `queued → applying → applied → starting → loading → ready`, and `failed`.
  Transitions are forward-only; a late event may not move a record backwards.
  One apply/start at a time: a second request is `409 operation_in_progress`
  with the running `operation_id` and the one action `status` (the same rule and
  shape as the measurement job — do not join).
- No secrets: no SSH targets, no API keys, no argv, no model file lists.

#### 2. `omlx/cluster/assistant/provisioning.py` (new)

- Build the `ClusterDeploymentRequest` for the elected plan:
  - `hosts[0]` = the elected coordinator with `ssh="127.0.0.1"` (the invariant
    from `deployment.py`), peers with their paired SSH targets;
  - `ips` from the measured endpoint hosts / paired registry addresses;
  - `backend` from the coordinator's measured `NodeFacts.link.transport`
    (`ring` / `jaccl` / `jaccl-ring` — already validated by `LinkHints`);
  - `nodes` from the measurement's capacity/admission (`election.budget_for`
    semantics, `role` from `node_role`), never invented;
  - `path_map` only when a real per-node path is known, otherwise the shared
    coordinator path;
  - `approved_placement` from the **recomputed** plan, not from the request.
- Recompute the election from the stored measurement and the apply-time inputs;
  refuse when the recomputed `plan_hash` ≠ the submitted `plan_signature`
  (`409 plan_mismatch`, one action `plan`).
- Resolve paired SSH targets and addresses through `get_device_registry()`
  **read-only**; a Mac missing from the registry is a one-action refusal, not a
  guess.
- Staging through the existing seam (`plan_cluster_staging` /
  `stage_files_from_source` / `check_disk_for_staging`), or through a single
  documented extraction of the existing `_run_staging_job` runner. **Do not
  copy it.**

#### 3. `omlx/cluster/assistant/start.py` (new)

- Local coordinator: the existing activation path (`_activate_and_report`
  internals / engine pool activation + readiness canary), off the event loop.
- Remote coordinator: the option you recorded (recommended A), through the
  existing SSH seam. On the coordinator the activation is the same code with
  rank 0 local.
- Never block the HTTP request on the long-running supervisor; return the
  operation handle.
- Health first (`check_peers` + `raise_if_peer_lost` + `preflight_remote_hosts`),
  then `starting` → `loading` → readiness canary → `ready`.
- **No local fallback.** If the elected coordinator cannot start, the operation
  fails with one action (`retry_coordinator` / `choose_another_coordinator` /
  `diagnostics`) and nothing starts on the origin. There is no branch anywhere in
  these modules that names the origin as a coordinator.

#### 4. `routes.py` + `models.py`

```
POST /admin/api/cluster/assistant/apply
POST /admin/api/cluster/assistant/start
GET  /admin/api/cluster/assistant/operations/{operation_id}
```

- `apply` request: `measurement_id`, `plan_signature`, plan-time inputs
  (`model_path` xor `model_size_bytes`+`layer_count`, `context_tokens`,
  `rank0_extra_bytes`, `mtp`, `execution_profile`), `coordinator_node`,
  `rank_order` (for validation only), optional `origin_node` (diagnostics only).
- `start` request: `measurement_id`, `plan_signature`, `coordinator_node`,
  optional `operation_id` from apply.
- `extra="forbid"` everywhere; missing fields 422.
- Missing/unknown/running/failed/stale/mismatched measurement ⇒ the **same flat
  `409 stale_measurement`** body as `/plan` (`code`, `reason`, `message`,
  `action: "measure"`, `measurement_id`), with no operation object.
- A plan mismatch, an unapplied start or a single-node plan is a distinct
  409/400 with one action; a failed operation carries `error.action`.
- `/measure`, `/status`, `/recommend`, `/plan` must keep byte-identical
  behaviour and response shapes; their tests must not change except where the
  route list is asserted.
- `models.py` gets the request/response schemas + builders only; no decision
  logic.

#### 5. Tests

`tests/cluster_assistant/test_apply_api.py`, `test_start_api.py`,
`test_operations.py` (FastAPI `TestClient`, admin dependency stubbed as in
`tests/test_cluster_ui_integration.py`, launch/staging/health injected like
`MeasureProbes`):

1. `apply` rejects a stale, unknown, running or failed measurement with the same
   flat 409 as `/plan`.
2. `apply` rejects a different model identity, a different `mtp`, a different
   node set and a wrong `plan_signature` (each with `reason` + one action).
3. `apply` stages but never starts inference (no launch/pool call reaches the
   fakes).
4. `start` rejects an unapplied/unapproved plan.
5. `start` uses the **elected** coordinator; the origin never appears as rank 0.
6. A remote coordinator start is attempted through the existing launch/SSH seam
   (assert the seam was called with the coordinator, not the origin).
7. A failed coordinator start does **not** start local rank 0, and the operation
   ends `failed` with exactly one `error.action`.
8. `endpoint.ready` stays false until health **and** one smoke completion
   succeed; an endpoint that was never proven is not reported ready.
9. Operation states are monotonic (a late `applying` cannot follow `ready`), and
   a second apply/start while one runs is 409 `operation_in_progress` + `status`.
10. Operation status survives a restart as `failed`, and the store stays
    bounded/atomic/fail-closed.
11. A missing performance probe does not block apply/start/ready and no rate
    appears in any operation body (`predicted` untouched, `observed` null).
12. Every new public function has a production caller —
    `tests/test_cluster_seams.py` must pass with **no new `allowed_uncalled`
    entries**.

Update `tests/cluster_assistant/test_plan_api.py` (and `test_recommend_api.py`)
only where the exact-route-list assertion is asserted; every other assertion
must keep passing unchanged. That is the regression gate for this session.

#### 6. Docs (small, same commit)

- Record the remote-start decision in `CHAPTER-5.md` §2/§4 and in
  `PROJECT-BRIEF.md` §13.2 (endpoint semantics) and §13.4 (persistence).
- `AGENTS.md` "Resolved implementation decisions": the remote-start option, the
  single-node refusal, the "plan is not persisted; apply recomputes" rule, the
  operation-store location/one-at-a-time rule, and the confirmed
  performance-probe policy.
- `CLUSTER-ASSISTANT.md` "Current status" and
  `tests/cluster_assistant/README.md`.
- `UPSTREAM.md`: only if `server.py` gains a patch. It should not — the router is
  already mounted. If you extract the staging runner out of
  `omlx/cluster/routes.py`, that **is** an upstream file patch: record it there
  with its reason and the test that keeps it honest.

---

### MUST NOT CHANGE

- `election.py`: scoring order, no-fallback rule, origin independence,
  `MAX_CANDIDATE_NODES`, TP refusal. `budget_for` stays the only budget builder.
- Chapter 2 measurement rules; Chapter 3 state/job/plan contract, 409 shapes,
  TTL, one-job rule, endpoint port rules; Chapter 4 recommendation contract.
- `ClusterDeployment`'s `hosts[0].ssh == "127.0.0.1"` invariant. Start the
  coordinator **on** the elected Mac; never make a local controller drive a
  remote rank 0.
- `launch.py`, `staging.py`, `deployment.py`, `modelsync.py`, `supervisor.py`
  semantics. Reuse them; document any extraction.
- `omlx/cluster/registry.py` persistence shape and `omlx/server.py`'s single
  patch site.
- Anything under `apps/omlx-web` or `apps/omlx-mac`.

---

### IMPLEMENTATION RULES

- One staging implementation, one launch path, one deployment registry, one
  operation store. Reuse `_activate_and_report`'s pipeline and the readiness
  canary instead of writing a parallel one.
- `apply` may stage; it must not start inference. `start` may not stage a plan
  that `apply` did not approve.
- Validate by recomputation, never by trusting the client: re-elect from the
  stored measurement and compare `plan_hash`.
- Never persist SSH targets, addresses, API keys, argv or model file lists.
- Every failure has exactly one actionable next step (§8.6, §12.4). No raw stack
  traces in the response body.
- `endpoint.ready` is false until health **and** one smoke completion succeed.
  The endpoint's host/port come from the measurement, never invented, never 0.
- No token rates anywhere: `predicted` keeps stage seconds only, `observed`
  stays `null` until a real measurement exists. The distributed performance
  probe needs an approved deployment; if you wire it at all it must be explicit,
  non-blocking, and must never mint a rate. A missing probe is not a failure.
- No model-less measurement: `apply`/`start` require a ready, model-bound,
  matching-node-set, matching-identity, matching-`mtp` measurement. Do not relax
  `ModelIdentity` or `measurement_refusal`.
- English only; the suite must stay runnable off-cluster.

---

### WORKFLOW

1. Read the docs and code above; trace `POST /stage` and `POST /deployments`
   end to end before designing.
2. Ask the one design question (A/B/C) once; record the answer in
   `CHAPTER-5.md` before coding.
3. Write a 5–10 bullet plan (operation shape, state machine, apply/start
   sequence, error codes, test matrix).
4. Implement `operations.py`, then `provisioning.py`, then `start.py`, then
   routes/models, then tests.
5. `uv sync --extra cluster`
6. `uv run pytest tests/cluster_assistant -q`
7. `uv run pytest tests/test_cluster_plan_approval.py tests/test_cluster_ui_integration.py tests/test_cluster_seams.py`
8. `uv run pytest tests/test_cluster_*.py` (one full pass before committing)
9. `uv run ruff check` + `uv run ruff format` on touched files;
   `uv run mypy omlx/cluster/assistant`
10. Fix failures.
11. Commit on `feat/cluster-assistant` with a focused message. Do not push.

Useful extra targets if you touch the reused seams:

```sh
uv run pytest tests/test_cluster_routes.py tests/test_cluster_launch.py tests/test_cluster_staging.py
```

---

### DEFINITION OF DONE

- An approved, model-bound plan can be applied; a mismatched, stale or
  unapproved one is refused with a machine-readable reason and one action.
- `apply` stages and starts nothing; `start` starts on the **elected** Mac only.
- Health and one smoke completion are verified before `endpoint.ready` is true.
- A coordinator start failure produces one action and no local fallback.
- Operation status is observable, monotonic and survives a restart as `failed`.
- No rate appears anywhere; a missing performance probe never blocks.
- `tests/cluster_assistant`, `tests/test_cluster_seams.py` and the cluster
  suites are green; ruff and mypy clean for the assistant module.

Report back: what you implemented, what you tested, which files changed, whether
`server.py` changed (it should not) and whether any upstream file had to change,
the recorded design decision, the exact operation states/endpoints, and what
still blocks the minimal Screen-3 UI.

**STOP after the apply/start backend and its tests.** Do not continue into UI,
performance optimisation, hardware-only measurement, chat, tray or tensor
parallelism without a new instruction.