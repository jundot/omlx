# Implementation Plan: Cluster Diagnostics (B) and Fabric Doctor (C)

**Status:** Executable plan — derived from `docs/cluster-ux-design.md` Sections B and C.
**Tracking issue:** jundot/omlx#2820.
**Branch:** `feat/cluster-diagnostics-and-fabric-doctor` (currently at `c840909f`, which already ships `choose_fabric_subnet()` — C.8 step 1's subnet half).

This document turns the design's build orders B.7 (items **B1–B6**) and C.8 (items **C1–C5**) into checklists a developer executes. It does not re-argue the design; failure numbers (#1–#11) refer to the traceability table in `docs/cluster-ux-design.md` §0.

## Line-reference conventions

Two branches are in play and **`fix/cluster-tensor-and-poll-jitter` (PR #2819) is *not* an ancestor of this branch** — both fork from `main` (`db1423db`). Therefore:

- Unlabeled `file:line` anchors refer to **this branch @ `c840909f`** (verified by reading).
- Anchors tagged **[#2819]** refer to the fix branch (`22d129d3`); they arrive here only after PR #2819 merges and this branch rebases.
- `docs/cluster-ux-design.md` itself lives on the fix branch (commit `3be14fba`); until the rebase, read it via `git show 3be14fba:docs/cluster-ux-design.md`.

**Prerequisite P0 (before anything below that depends on it):** merge PR #2819 into `main`, rebase this branch. It supplies: `ClusterStatus.ssh_user` + `probe.local_login_name()` [#2819 `models.py`, `probe.py`], home-relative staging paths (`staging.home_relative_model_path`) [#2819 `staging.py`], the dashboard poll guards in `syncClusterNodesFromPeers()` / `normalizeClusterTensorParallelSize()` [#2819 `dashboard.js:1768`, `:3357`], and the VLM `supports_pipeline=False` fix [#2819 `planner.py:684`].

## Shared architectural conventions (read once, applied throughout)

- **Persistence pattern:** every new store copies `ClusterRegistry` (`omlx/cluster/registry.py:24-60`): `base_path / "cluster" / <name>.json`, `schema_version: 1`, `threading.RLock`, fail-closed on corrupt file (`load_error`, empty state), atomic write via temp-file + `os.replace` (see `enrollment.py:157-177`), module-level `configure_*/get_*` pair (`registry.py:167-176`) wired at `omlx/server.py:1934-1940`.
- **API prefix:** all new endpoints join `router = APIRouter(prefix="/admin/api/cluster")` (`omlx/cluster/routes.py:123`).
- **Redaction:** anything user-visible that carries probe/SSH output passes `_redact_diagnostic()` (`routes.py:291`) before leaving the server, exactly as `/diagnostics` does (`routes.py:1835`).
- **Library purity:** modules under `omlx/cluster/` other than `routes.py` do not import the incident store; they raise/return as today. The routes layer (and the job runner introduced in B2) converts outcomes into incidents. This keeps `launch.py`/`transport.py` importable in worker-only installs.
- **Tests:** one `tests/test_cluster_<module>.py` per new module, following the existing naming (e.g. `tests/test_cluster_fabric_subnet.py`). Route tests extend `tests/test_cluster_routes.py`; template smoke tests extend `tests/test_cluster_dashboard.py`.

---

# Section B — diagnostics, incident model, output

## B1 — Server-owned incidents + monotonic-merge poll rule

**Goal:** every abnormal end state produces a durable server-side record that a dashboard poll can add to but never erase. Closes failures **#7** (silent fallback becomes a WARN incident — completed in C5) and **#8** (poll-wiped error state; PR #2819 stopped the wiping, this makes error state server-owned so there is nothing left to wipe).

### Files & symbols

| Action | Path / symbol |
|---|---|
| **Create** | `omlx/cluster/incidents.py` — `Severity` (StrEnum), `Incident` (frozen dataclass), `IncidentStore`, `configure_cluster_incidents()`, `get_cluster_incidents()` |
| **Create** | `tests/test_cluster_incidents.py` |
| Modify | `omlx/server.py:1934-1940` — wire `configure_cluster_incidents(base_path)` beside `configure_cluster_registry` / `configure_cluster_enrollment` |
| Modify | `omlx/cluster/routes.py` — new endpoints `GET /incidents`, `POST /incidents/{id}/dismiss`; emitters in `activate_cluster_deployment` exception paths (`routes.py:3155-3172`), `_run_staging_job` failure paths (`routes.py:1494+`), `cluster_peer_health` unhealthy transitions (`routes.py:2172`) |
| Modify | `omlx/cluster/routes.py:1760-1835` — `/diagnostics` bundle gains `"incidents"` key (last 200) |
| Modify | `omlx/admin/static/js/dashboard.js` — new `loadClusterIncidents()` called from `refreshClusterExperience()` (`dashboard.js:1470-1480`); cursor `_clusterIncidentSeq`; merge-only state `clusterIncidents` |
| Modify | `omlx/admin/templates/dashboard/_cluster.html` — incident badge + reverse-chron feed panel (minimal render here; B4 does the full three-surface layout). The dismiss-clears-error button at `_cluster.html:25` starts posting dismissals instead of blanking client state |

### Data model / API

```python
class Incident:            # frozen dataclass
    seq: int               # store-assigned, strictly monotonic — the poll cursor
    id: str                # uuid4 hex
    ts: float
    severity: Severity     # info | warn | error
    source: str            # "coordinator" | "rank:<n>" | "peer:<node_id>" | "gui" | "doctor"
    state_code: str        # machine key, joins B3's guidance codes
    message: str           # already redacted
    guidance_code: str | None
    job_id: str | None     # B2 start-job / staging job correlation
    deployment_id: str | None
    bundle_ref: str | None # B6 slice anchor
    dismissed_at: float | None = None
    superseded_by: str | None = None   # job supersession keeps history (design B.3)
```

`IncidentStore(base_path)`: ring buffer capped at 200, **persisted** to `base_path/cluster/incidents.json` (design B.3 says "persisted by the coordinator"; survives server restart). Methods: `record(...) -> Incident` (assigns `seq`), `list(since_seq=0)`, `dismiss(incident_id)`, `supersede(job_id, by_incident_id)`.

Endpoints:
- `GET /admin/api/cluster/incidents?since=<seq>` → `{"incidents": [...], "latest_seq": N}`. The `since` cursor makes monotonic merge a *server-enforced* property, not client discipline.
- `POST /admin/api/cluster/incidents/{id}/dismiss` → `{"ok": true}`. Dismissal is the only removal path (plus supersession, which flags but keeps the record).

### Checklist

- [ ] Write `Incident`/`Severity`/`IncidentStore` in `omlx/cluster/incidents.py`, copying the store skeleton from `registry.py:24-60` and the atomic save from `enrollment.py:157-177`.
- [ ] Wire `configure_cluster_incidents(base_path)` at `server.py:1934-1940`.
- [ ] Add the two endpoints to `routes.py`; run every `message` through `_redact_diagnostic` (`routes.py:291`).
- [ ] Emit incidents from the three existing failure funnels: (a) each `except` arm of `activate_cluster_deployment` (`routes.py:3155-3172`) before re-raising the `HTTPException`; (b) `_run_staging_job`'s failure update paths; (c) `cluster_peer_health` when a previously-healthy deployment reports `healthy=False` (the "coordinator narrates deaths" rule — `check_peers`/`describe_failure`, `liveness.py:307`, `:379`, already produce the text).
- [ ] Add `"incidents"` to the `/diagnostics` payload (`routes.py:1824-1834`).
- [ ] `dashboard.js`: `loadClusterIncidents()` fetches with the cursor, merges into a `Map` keyed by `id`, never deletes locally; dismissal round-trips through the endpoint. Hook into `refreshClusterExperience()` (`dashboard.js:1470`) on the 2 s tick (payload is tiny after the first fetch thanks to `since`).
- [ ] `_cluster.html`: badge (unseen-incident count) + feed rows (severity, source, age, message, [Dismiss]).

### Tests

- `tests/test_cluster_incidents.py`: `seq` strictly increases across `record()` calls; ring cap evicts oldest-first (plain FIFO); `dismiss` persists across a store reload; corrupt `incidents.json` → empty store + `load_error` (fail-closed like `registry.py:33-39`); `list(since_seq)` excludes already-seen records; `supersede` marks but does not delete.
- `tests/test_cluster_routes.py`: activation failure (monkeypatched `_create_deployment` raising) records an incident whose `message` matches the HTTP detail; `GET /incidents?since=` cursor semantics; dismissed incident stays in `/diagnostics` bundle.

### Risks / backward-compat

- New endpoints only — no wire change to existing responses.
- Multi-writer safety: `record()` from thread-pool contexts (`asyncio.to_thread` paths) — the RLock pattern covers it, but keep `record()` free of I/O beyond the save.
- Do not emit incidents from library modules (`launch.py`, `transport.py`) directly — worker-only installs import them without a configured store. Routes-layer emission only until B2's job runner exists.
- Disk churn: one JSON rewrite per incident; acceptable at the 200 cap. If measured hot, batch dismissal writes.

---

## B2 — Start Cluster as a persisted server-owned job

**Goal:** clicking Start creates a server-side job whose state survives browser reloads; the button renders the server's record, never local component state. Closes failure **#8**'s dead-button class.

### Files & symbols

| Action | Path / symbol |
|---|---|
| **Create** | `omlx/cluster/start_job.py` — `StartJobPhase` (StrEnum), `StartJobStore`, `run_start_job()` orchestrator |
| **Create** | `tests/test_cluster_start_job.py` |
| Modify | `omlx/cluster/routes.py` — refactor the body of `cluster_autoconfigure` (`routes.py:1003+`) into a callable `_autoconfigure(request) -> dict` and the body of `activate_cluster_deployment` (`routes.py:2959-3195`) into `async _activate(request) -> dict`, each kept behind its existing endpoint; new endpoints `POST/GET /start-jobs`; reuse `_run_staging_job` (`routes.py:1494`) for the staging phase |
| Modify | `omlx/admin/static/js/dashboard.js` — `startCluster()` (`dashboard.js:5121-5190`) becomes "POST /start-jobs then poll"; `activateClusterProposal()` (`:5192-5229`) retained for the manual-plan path; on component init, `GET /start-jobs` re-attaches to a running job (reload safety) |
| Modify | `omlx/admin/templates/dashboard/_cluster.html` — Start buttons at `:139-140` and `:1854-1862` render `clusterStartJob.phase` / progress from the server record; manual activation block `:2567-2575` unchanged this pass |

### Data model / API

```python
class StartJobPhase(StrEnum):
    QUEUED = "queued"; LINK_SETUP = "link_setup"; AUTOCONFIGURE = "autoconfigure"
    STAGING = "staging"; ACTIVATING = "activating"; READY = "ready"; FAILED = "failed"

# StartJobStore record (dict, mirroring the _STAGING_JOBS shape at routes.py:1461-1492):
{ "job_id", "created_at", "updated_at", "phase", "model_path", "hosts": [...],
  "staging_job_id": str|None,          # joins the existing /stage/{job_id} record
  "attempt": int,                       # "#4 replaces #3"; superseded jobs kept
  "superseded_by": str|None,
  "incident_id": str|None,              # FAILED phase always carries one
  "result": {...}|None }                # the /deployments response on READY
```

**Persistence decision (explicit):** the job store is **in-memory** (dict + lock, like `_STAGING_JOBS`, `routes.py:1461-1463`) — that satisfies the design's stated requirement ("if the browser reloads, the job is still there", B.3). Durable history across *server* restarts is provided by B1: every phase failure and every READY records an Incident, and the deployment registry (`registry.py`) is the durable record of a successful start. Write this trade-off in the module docstring.

Endpoints:
- `POST /admin/api/cluster/start-jobs` (body = today's autoconfigure request shape, `ClusterAutoconfigureRequest`, `routes.py:781-807`) → `202 {"job_id"}`. Refuses (`409`) if a non-terminal job exists for the same model, or a legacy sync activation is in flight (shared `threading.Lock` with `_activate`).
- `GET /admin/api/cluster/start-jobs/{job_id}` → the record; `GET /admin/api/cluster/start-jobs` → recent records (newest first, cap 16).

`run_start_job` executes, as an `asyncio.Task` on the server loop, exactly the ladder `startCluster()` runs client-side today (`dashboard.js:5121-5190`): link status → `configure_link` (`transport.py:824`) when `setup_available` → `_autoconfigure` → stage when `staging.ready == False` (spawn `_run_staging_job`, record its `job_id`, await completion) → re-`_autoconfigure` → `_activate`. Each phase transition updates the store; each failure records an Incident (source `"coordinator"`, `job_id` set) and parks the job in FAILED.

### Checklist

- [ ] Extract `_autoconfigure(request)` and `_activate(request)` from their endpoints (pure refactor; endpoints keep byte-identical behavior — the sync `POST /deployments` remains for the manual "Activate manual plan" path and API users).
- [ ] Implement `StartJobStore` + `run_start_job` in `omlx/cluster/start_job.py`; incidents on every failure and on READY (severity info).
- [ ] New-attempt supersession: creating a job for a model with an existing terminal FAILED job links `superseded_by` and calls `IncidentStore.supersede`.
- [ ] Add the three endpoints; guard against concurrent activation with a lock shared with `_activate`.
- [ ] `dashboard.js`: rewrite `startCluster()` to POST + poll `GET /start-jobs/{id}` on the existing 750 ms activation timer (`startClusterActivationProgress`, `dashboard.js:1429-1448` — repoint `refresh` at the job record); derive `clusterActivationLoading` from `phase ∉ {ready, failed}` so the [#2819] poll guards in `syncClusterNodesFromPeers()` keep holding during a job.
- [ ] Init-time re-attach: in cluster tab setup (`dashboard.js:919-929` region) fetch `GET /start-jobs` and adopt any non-terminal job.
- [ ] `_cluster.html`: button labels at `:1854-1862` switch from the loading-flag ternary to server phase text (`preparing → staging (rank 1: 62%) → launching → ready | failed`); staging % read through `staging_job_id` → existing `/stage/{job_id}` (`routes.py:1707`).

### Tests

- `tests/test_cluster_start_job.py`: with monkeypatched `_autoconfigure`/`_run_staging_job`/`_activate` fakes — phase sequence QUEUED→…→READY; failure in each phase → FAILED + incident recorded with that phase's `state_code`; 409 on concurrent same-model job; supersession chain; GET returns the record after the task finished (no cleanup race).
- `tests/test_cluster_routes.py`: `POST /start-jobs` 202 shape; legacy `POST /deployments` still works and is mutually exclusive with a running job.

### Risks / backward-compat

- `_activate` is async and touches the engine pool — the job **must** run as an event-loop task (`asyncio.create_task`), not a thread; blocking sub-steps already use `asyncio.to_thread` internally.
- Server shutdown mid-job: task cancelled; registry rollback inside `_activate` (`routes.py:3129-3154`) already restores the previous deployment. Record a FAILED incident from a `finally` if cancelled.
- The legacy sync path stays until the manual-plan UI migrates — note as follow-up, do not break `activateClusterProposal()`.
- Keep the [#2819] guard semantics: a running job must suppress `syncClusterNodesFromPeers` node rebuilds exactly as `clusterActivationLoading` does today.

---

## B3 — TransportState readiness ladder + reason/remedy strings

**Goal:** every rung between "cable in" and "collective proven" is a named state carrying `reason` and `remedy` copy rendered verbatim in CLI and GUI. Closes failure **#5** (the `peer_linked_config_pending` ceiling and the unexplained stale 169.254).

### Files & symbols

| Action | Path / symbol |
|---|---|
| **Create** | `omlx/cluster/readiness.py` — `LADDER_ORDER`, `TransportReadiness` dataclass, `ladder_copy(state, **evidence) -> (reason, remedy)`, `link_ladder_state(...)` pure derivation |
| **Create** | `tests/test_cluster_readiness.py` |
| Modify | `omlx/cluster/models.py:14-25` — extend `TransportState` with `ADDRESSED`, `ROUTED`, `REACHABLE`, `FABRIC_VERIFIED`, `COLLECTIVE_OK`; delete the "This slice can report only through `peer_linked_config_pending`" sentence (`models.py:19`); `to_dict` (`:167-175`) gains `transport.readiness = {state, reason, remedy}` |
| Modify | `omlx/cluster/probe.py:483-490` — the node-local state decision extends past `PEER_LINKED_CONFIG_PENDING`: non-link-local IP on the active RDMA interface (compare against `transport._UNROUTABLE_NETWORKS`, `transport.py:1037-1040`) → `ADDRESSED`; `route.uses_rdma_interface` (`models.py:98`, warning already emitted at `probe.py:506-510`) → `ROUTED`. **Node-local status tops out at ROUTED** — `REACHABLE`+ require two-ended proof and belong to the link level |
| Modify | `omlx/cluster/transport.py:425-450` — `LinkStatus` gains `ladder: str`, `reason: str`, `remedy: str` fields (serialized in `to_dict`); `assess_link` (`:880-1020`) and `classify_link` (`:453-563`) populate them per branch (e.g. the `:984-1001` "addresses not reachable" branch → `ladder="routed"`, remedy "Run Fabric Doctor") |
| Modify | `omlx/cluster/guidance.py` — `Guidance` (`:18-37`) gains `code: str | None = None`; new `_BY_CODE: dict[str, Guidance]` and `explain_code(code, message=None) -> Guidance` that tries the structured lookup first and falls back to `explain()` (`:370-385`); each `_RULES` entry (`:82-358`) gets a `code` so message-regex and state-code lookups converge on the same copy objects |
| Modify | `omlx/cluster/probe.py:636` region — `format_cluster_status` prints `Transport: <state>` plus indented `reason:` / `remedy:` lines when below `COLLECTIVE_OK` |
| Modify | `omlx/admin/static/js/dashboard.js` / `_cluster.html` — render `transport.readiness.reason/remedy` beside the existing transport line (full status-strip layout waits for B4) |

### Data model / API

- `TransportState` gains five members after `PEER_LINKED_CONFIG_PENDING` (`models.py:25`); existing four keep their exact values.
- `ClusterStatus.to_dict()["transport"]["readiness"] = {"state", "reason", "remedy"}` — additive.
- `LinkStatus.to_dict()` gains `"ladder"`, `"reason"`, `"remedy"` — additive.
- `fabric_verified` (`models.py:141`) becomes derived: true iff the link ladder ≥ `FABRIC_VERIFIED`. Its plumbing (`routes.py:367`, `:566`) is untouched.
- Guidance copy migration seed: the regex table `_RULES` at `guidance.py:82` (entry point `explain()` at `guidance.py:370`) — every entry gains a `code`; new ladder states get fresh copy in `readiness.ladder_copy` (e.g. `ADDRESSED`-blocked-below: reason "The link has a self-assigned 169.254 address — macOS never finished configuring it", remedy "Run Fabric Doctor → Re-address link").

### Checklist

- [ ] Extend the enum + docstring in `models.py`; add `TransportReadiness` serialization.
- [ ] Write `readiness.py`: `link_ladder_state(link_status, shared_link, verify_result, collective_result) -> TransportState` as a pure function; copy table for every state.
- [ ] Extend the `probe.py:483-490` decision; keep it evidence-only (no SSH added to the local probe).
- [ ] Populate `LinkStatus.ladder/reason/remedy` in every `classify_link` and `assess_link` return (nine construction sites between `transport.py:453-1020` — grep `LinkStatus(`).
- [ ] Guidance codes: add `code=` to each `_RULES` entry; build `_BY_CODE`; `explain_code` preserves the never-raise/never-None contract (`guidance.py:373-375`).
- [ ] CLI output (`format_cluster_status`) and minimal dashboard render.
- [ ] The `/guidance` endpoint (`routes.py:1442`) accepts an optional `code` and prefers it over message-matching.

### Tests

- `tests/test_cluster_readiness.py`: table-driven — each evidence combination maps to exactly one rung; every state has non-empty reason and remedy ("states without copy don't ship", design appendix 3); ladder ordering is total.
- `tests/test_cluster_probe.py`: 169.254 on the RDMA interface stays `PEER_LINKED_CONFIG_PENDING` with the stale-address reason; routable IP without RDMA route → `ADDRESSED`; route via RDMA interface → `ROUTED`.
- `tests/test_cluster_link_status.py`: each `assess_link` branch carries the expected `ladder`; the `:984-1001` unreachable branch names the Doctor in `remedy`.
- `tests/test_cluster_guidance.py`: `explain_code("stale_link_address")` returns structured copy; unknown code falls back through the regex path; every `_RULES` entry has a unique code.

### Risks / backward-compat

- **Wire compat verified:** no Python code re-parses `TransportState(value)` from a remote payload (grepped; peers serialize to string at `models.py:168`, coordinators consume dicts), and the dashboard renders state strings as text — new values degrade gracefully on an older coordinator. `protocol_version` stays `"1.0"` (`models.py:10`); note the additive change in its docstring.
- The `_REMOTE_SYSTEM_PROBE` bootstrap payload hardcodes `"state": "enabled_no_peer" | "unavailable"` (`launch.py:1031`) — leave it; a pre-runtime host can never be past that anyway.
- Don't let node-local state claim `REACHABLE`: single-host evidence can't prove bidirectionality (that was failure #5's lie in miniature).

---

## B4 — "Why can't I start?" precondition panel

**Goal:** whenever Start is not green, a row-per-precondition panel with live evidence and a fix affordance renders in its place; CLI prints the same rows. Closes the presentation half of **#5/#8/#9** (composes B1–B3).

### Files & symbols

| Action | Path / symbol |
|---|---|
| **Create** | `omlx/cluster/preconditions.py` — `PreconditionRow` dataclass, `readiness_rows(...)` composer |
| **Create** | `tests/test_cluster_preconditions.py` |
| Modify | `omlx/cluster/routes.py` — new `GET /readiness?hosts=…&model=…`; built by restructuring what `_autoconfigure` already computes: `fabric_ready`/`fabric_blocker` (`routes.py:1245-1256`), the preflight summary (`:1379-1389`), staging readiness (`_staging_for`, `:840`), into structured rows instead of one prose blocker |
| Modify | `omlx/admin/templates/dashboard/_cluster.html` — the three-surface layout of design B.4: status strip (one line per node), incident badge/panel (from B1), and the precondition panel rendered when Start is disabled (today Start is a silently disabled button, `:139-140`, `:1854-1855` — "a disabled control without an adjacent explanation is a design defect") |
| Modify | `omlx/admin/static/js/dashboard.js` — `loadClusterReadiness()` on the 10 s discovery tick (`refreshClusterExperience`, `dashboard.js:1470-1480`, after the counter gate at `:1473`), **not** the 2 s tick — rows embed SSH-backed evidence |
| Modify | `omlx/cli.py` — `cluster status` parser (`cli.py:1301`) gains `--explain`; handler (`cli.py:800-816`) prints the same rows in text (design B.6: CLI and GUI can never tell different stories) |

### Data model / API

`GET /admin/api/cluster/readiness` →

```json
{ "ready": false,
  "rows": [
    {"id": "ssh",      "state": "pass|warn|fail", "evidence": "aphoenix@mbp.local ✓ verified 40 s ago (both directions)", "evidence_age_s": 40, "fix": {"kind": "reverify"}},
    {"id": "fabric",   "state": "...", "evidence": "reachable · 172.16.99.1 ⇄ .2 · TB5", "fix": {"kind": "doctor"}},
    {"id": "staging",  "state": "...", "evidence": "⟳ MacBook 8.4/17.2 GB", "fix": {"kind": "stage_details", "job_id": "…"}},
    {"id": "budget",   "state": "...", "evidence": "<from B5>", "fix": {"kind": "role_editor", "node_id": "…"}},
    {"id": "strategy", "state": "...", "evidence": "<from B5>", "fix": null}
  ] }
```

Cluster-level `ready` is the conjunction from design B.2: `all(ssh_ok, fabric ≥ reachable, model_staged, budget_fits_live, strategy_compatible)`. Every `evidence` string carries its age; the client renders stale-gray past 30 s (design A.6's live-data rule).

### Checklist

- [ ] `preconditions.py`: `readiness_rows()` takes already-collected evidence (peer health from `check_peers`, `liveness.py:307`; `LinkStatus` with B3 ladder; staging manifest; budgets; strategy flags) — pure over inputs so it tests without Macs.
- [ ] Endpoint assembles evidence with bounded timeouts, reusing `_autoconfigure`'s sub-steps rather than duplicating them; responses cached 10 s server-side so B2's job poll and this never stampede SSH.
- [ ] `_cluster.html`: status strip (state name in text, color secondary — "never color-only", design B.4); precondition panel replaces the disabled-button dead zone; fix affordances dispatch to existing actions (`peer-probe` `routes.py:2240`, Doctor C3, role editor via `/node-roles` `routes.py:2686` + `/node-budgets` `:2707`, staging details via `/stage/{job_id}` `:1707`).
- [ ] CLI `--explain` prints rows (reuse the same endpoint logic locally, not HTTP).
- [ ] Wire incident badge from B1 into the strip.

### Tests

- `tests/test_cluster_preconditions.py`: table-driven rows — each single failing input flips exactly its row and `ready`; evidence strings carry ages; the failing-budget row names the role (ties to B5).
- `tests/test_cluster_routes.py`: `/readiness` with fakes returns the five rows; server-side cache honors the 10 s TTL.
- `tests/test_cluster_dashboard.py`: template renders panel markup when `ready=false`.

### Risks / backward-compat

- Endpoint cost is the risk: five SSH-backed checks. Mitigate via the cache + reuse of the freshest `_autoconfigure` result; never call from the 2 s tick.
- Don't remove the existing blocker-string plumbing (`clusterAutoconfigureError`) until the panel proves out — render both for one release.

---

## B5 — Live memory-budget breakdown + plan-time parallelism gating

**Goal:** the memory row always shows the live ceiling *with its arithmetic* (physical − role reserve − in-use), and parallelism modes a model can't run are disabled with the reason inline. Closes failures **#9** and **#10**.

### Files & symbols

| Action | Path / symbol |
|---|---|
| Modify | `omlx/cluster/launch.py:1122-1229` — `probe_remote_admission_ceiling`: the remote script (`:1145-1158`) returns the full breakdown, not one int: `{"admission_ceiling_bytes", "breakdown": {"static", "dynamic", "metal_cap", "hard_limit"}}` from `memory_guard.ceiling_breakdown()` (`memory_guard.py:180-192` — those are its real keys). **Coordinate with C5**, which retargets this same script's health-probe port — land C5's port change first or in the same PR |
| Modify | `omlx/cluster/routes.py:2707-2760` — `/node-budgets` response: each node gains `"breakdown"`: physical (`ClusterStatus.physical_memory_bytes`), ceiling components (above), `role` + `reserve_bytes` + `usable_bytes` (already in `NodeBudgetSuggestion.to_dict()`, `node_role.py:308-316`), plus `binding: "role_reserve" | "dynamic_pressure" | "metal_cap"` naming which constraint binds |
| Modify | `omlx/cluster/node_role.py` — no arithmetic changes (the invariant is already enforced: `reserve_for` `:112-124`, `admission_bytes` `:135-152`, and the planner/guard agreement documented at `models.py:122-126`); only ensure `NodeBudgetSuggestion` exposes the pieces the row needs |
| Modify | `omlx/cluster/routes.py` — `/autoconfigure` and `/plan` responses gain `"strategies"`: `{"tensor": {"supported", "reason"}, "pipeline": {"supported", "reason"}}` from `ModelProfile.supports_tensor_parallel` / `supports_pipeline` (`planner.py:113`, `:116`; computed at `:583`, `:669`; VLM fix [#2819 `planner.py:684`]) |
| Modify | `omlx/admin/static/js/dashboard.js` — `clusterTensorParallelOptions()` (`dashboard.js:3329`) consumes `strategies` so an unsupported mode is never offered; budget row renderer |
| Modify | `omlx/admin/templates/dashboard/_cluster.html` — memory row + disabled-radio-with-inline-reason for parallelism (feeds B4's `budget` and `strategy` rows) |

### Data model / API

Memory row payload (per node, additive to `/node-budgets`):

```json
{"breakdown": {"physical_bytes": …, "static": …, "dynamic": …, "metal_cap": …,
               "hard_limit": …, "role": "workstation", "reserve_bytes": …,
               "usable_bytes": …, "binding": "role_reserve"}}
```

Failing-case copy comes verbatim from design B.5 ("The model fits in this Mac's memory, but not in its current budget…"), with the role name linked to the role editor.

**Parallelism authority (decision):** plan-time gating uses **`ModelProfile.supports_tensor_parallel`/`supports_pipeline`** — config-cheap, already computed at plan build, already the refusal source at `routes.py:748-757`. `pipeline_assignment_is_honored()` (`pipeline_compat.py:30`) — which the design's traceability #10 cites — answers a *different, narrower* question (does the installed hook consume an unequal assignment) and imports `mlx_lm` model modules; it **stays the load-time enforcement** inside the worker path. The plan-time flags close the GUI→400 route; the routes.py:748-757 `PlanningError` remains as the API backstop for non-GUI callers. Document this split in `preconditions.py`.

### Checklist

- [ ] Extend the remote breakdown script in `launch.py` (keep `admission_ceiling_bytes` as a top-level key — older coordinators reading a newer peer, and the reverse, must keep working; treat a missing `breakdown` as "peer too old, render ceiling only").
- [ ] Extend `/node-budgets` (`routes.py:2725-2760`) to pass the breakdown through `suggest_budget` context and name the binding constraint.
- [ ] Add `strategies` to `_autoconfigure` and `/plan` responses (profile is already in hand in `_model_and_nodes`, `routes.py:592`).
- [ ] Dashboard: memory row with arithmetic + age; strategy radios disabled with reason text; delete the client-side guesswork that [#2819 `dashboard.js:3357`] had to add heuristics for (options snap logic simplifies once the server says which modes exist).
- [ ] Feed both rows into B4's `readiness_rows`.

### Tests

- `tests/test_cluster_launch.py`: breakdown script payload parses; missing `breakdown` key degrades to the single-int path.
- `tests/test_cluster_routes.py`: `/node-budgets` breakdown arithmetic for a workstation-role 48 GB node names `role_reserve` as binding; `/autoconfigure` `strategies.pipeline.supported == False` with reason for a VLM config (fixture from [#2819 tests/test_cluster_planner.py]).
- `tests/test_cluster_node_role.py`: unchanged invariants still hold (regression net).

### Risks / backward-compat

- Version skew on the breakdown script — handled by the missing-key fallback above.
- `ceiling_breakdown()` respects operator memory settings (`memory_guard.py:206-210`); the row must label `dynamic` as "in use by other apps right now" honestly (it's reclaimable-based, not an app-by-app account).
- Keep `routes.py:748-757` errors intact: the UI gate is UX, not the safety boundary.

---

## B6 — Asset-version handshake + bundle-slice Details

**Goal:** a stale cached dashboard names itself and demands a reload; every incident row opens its evidence slice in one click. Closes failure **#8**'s cached-JS ghost and failure **#11**.

### Files & symbols

| Action | Path / symbol |
|---|---|
| Modify | `omlx/admin/routes.py:1103-1109` — `_static_version` already computes an mtime stamp for cache-busting (`dashboard.html:98` uses it via the `static()` template global). Reuse it: export `asset_version()` (mtime of `static/js/dashboard.js`, cached with 60 s TTL), inject `window.OMLX_ASSET_VERSION` into the dashboard template, and add an HTTP middleware (register beside the router) stamping `X-Omlx-Asset-Version` on `/admin/api/cluster/*` responses |
| Modify | `omlx/admin/static/js/dashboard.js` — one fetch wrapper for cluster API calls compares the header to `window.OMLX_ASSET_VERSION`; mismatch sets a non-dismissable `assetStale` flag |
| Modify | `omlx/admin/templates/dashboard/_cluster.html` — "Dashboard updated — reload" bar bound to `assetStale` |
| Modify | `omlx/cluster/routes.py:1760-1835` — `/diagnostics` gains `?incident=<id>`: returns that incident, its correlated staging/start-job record, the readiness snapshot, and the log window ±30 s (from launcher tails already in the runtime payload, `routes.py:1744-1756`) |
| Modify | `omlx/admin/static/js/dashboard.js:1337` — `downloadClusterDiagnostics` grows the per-incident slice fetch behind each feed row's [Details]; the existing global save (copy at `_cluster.html:824`) now includes the incident feed (done in B1) |

### Data model / API

- Response header `X-Omlx-Asset-Version: <mtime-int>` on cluster API responses; template global `window.OMLX_ASSET_VERSION` from the same source.
- `GET /admin/api/cluster/diagnostics?incident=<id>` → `{"incident": {...}, "job": {...}|null, "readiness": {...}, "log_window": [...]}` — all through `_redact_diagnostic`.

### Checklist

- [ ] `asset_version()` + middleware + template injection.
- [ ] Fetch wrapper + reload bar (non-dismissable by design B.3).
- [ ] Slice mode on `/diagnostics`; correlation by `incident.job_id` / `deployment_id`.
- [ ] [Details] affordance on every incident row (B1's feed) and on B4's failing rows that carry `bundle_ref`.

### Tests

- `tests/test_cluster_routes.py` (or a new `tests/test_admin_asset_version.py`): header present on `/admin/api/cluster/status`; slice endpoint filters to one incident and stays redacted.
- Template smoke: reload bar markup exists.

### Risks / backward-compat

- Packaged app (`apps/omlx-mac/...Stage/`) may normalize file mtimes — if two builds collide, switch the stamp to a content hash of `dashboard.js` (same call site; note in code).
- Middleware ordering: stamp only after auth so 401 redirects aren't misread as version churn.

---

# Section C — Fabric Doctor

## C1 — Bidirectional bound-connect check (completes C.8 step 1)

**Goal:** promotion to "reachable" requires a TCP connect *bound to the fabric source IP* to succeed in **both** directions — the check that separates "cable works" from "VPN firewall swallows inbound". Closes the remaining half of failure **#4** and the one-way case of **#6**.

### Files & symbols

| Action | Path / symbol |
|---|---|
| Modify | `omlx/cluster/transport.py:1544-1602` — `verify_link_reachability()`. Today the bound-connect script (`:1571-1583`) runs **only** as a fallback when `route -n get` is unavailable (Linux), and the positive proof is one ICMP ping per direction (`:1593-1601`). Change: after the route-interface check passes, **always** run the bound TCP connect (`socket.create_connection((peer, 22), timeout=3, source_address=(local_fabric_ip, 0))`) in both directions; ping stays as a diagnostic detail in the failure reason, never as the success criterion (a VPN firewall can pass ICMP and route lookups while dropping TCP — the incident's exact shape: LAN ping/SSH fine, peer→coordinator on the fabric refused) |
| Modify | same function — one-way failure reason names the direction and the likely cause, seeding design C.4's copy: `"{host} cannot accept connections on the Thunderbolt link (a firewall or VPN on that Mac applies to all interfaces)"` |
| **Create/extend** | `tests/test_cluster_link_reachability.py` (or extend `tests/test_cluster_link_status.py`) |

### Data model / API

None. `verify_link_reachability` keeps its `(bool, str)` contract and its injectable `runner` (`LinkCommandRunner`, `transport.py:1503-1531`); `assess_link` (`:939-1019`) picks the change up for free because `resolve_link_addresses` already threads `verify` (`:1488-1500`). When B3 lands, a passing bound connect is what earns the `REACHABLE` rung.

### Checklist

- [ ] Hoist the bound-connect script to a module constant; run it for each direction after (not instead of) the route check.
- [ ] Order per direction: route check → bound TCP connect → (on TCP failure only) ping, to distinguish "no route" / "firewall drops TCP" / "host down" in the reason string.
- [ ] Keep total budget bounded: `_run_link_command`'s 8 s cap (`:1522-1529`) × 2 extra commands ≈ worst-case +16 s on a dead link — acceptable for an explicit check; note it in the docstring.
- [ ] Direction-naming failure copy as above.

### Tests

With a fake `runner` (pattern proven in this file's existing tests):
- route ok + bound connect ok both ways → `(True, …)`; ping never consulted.
- route ok both ways + bound connect refused peer→coordinator → `(False, reason)` where reason names the peer host and "cannot accept connections".
- route command unavailable (Linux) + bound connect ok → still `(True, …)` (existing behavior preserved, `:1577-1583`).
- bound connect times out (simulated 255/"timed out") → reason mentions firewall/VPN.

### Risks / backward-compat

- Depends on the peer's sshd listening on the fabric IP (port 22 binds all interfaces by default) — already a cluster prerequisite; if a hardened sshd binds only the LAN, the check honestly reports the fabric unreachable, which is the correct conservative answer. Note in docstring.
- Slightly stricter than today: links that passed on ping alone may now report unreachable. That is the point (empirical probes outrank optimism), but flag it in the PR description.

---

## C2 — Durable addressing: `networksetup -setmanual` + recorded `fabric_subnet` intent

**Goal:** a good address assignment survives reboots and oMLX's own re-runs; link-setup treats recorded intent as authoritative instead of re-addressing. Closes failure **#4**'s "fix was ephemeral" tail and the reboot-loss case of **#5**.

### Files & symbols

| Action | Path / symbol |
|---|---|
| **Create** | `omlx/cluster/fabric_intent.py` — `FabricIntent` dataclass + `FabricIntentStore` (registry pattern), `configure_fabric_intent()`, `get_fabric_intent()` |
| **Create** | `tests/test_cluster_fabric_intent.py` |
| Modify | `omlx/server.py:1934-1940` — wire the store |
| Modify | `omlx/cluster/transport.py` — new `_authorized_networksetup(host, service, address, mask)` beside `_authorized_ifconfig` (`:747-821`), reusing the `_remote_gui_authorize` Aqua-session bridge (`:662-744`) unchanged; new pure parser `parse_hardware_ports(output)` for `networksetup -listallhardwareports` (maps BSD device enX → service name), styled after `parse_interface_addresses` (`:1210`) |
| Modify | `omlx/cluster/transport.py:824-877` — `configure_link()`: the address-selection block (`:857-869`) becomes a three-tier decision: (1) an existing subnet on either endpoint that passes the collision check is kept (today's `:860-864` reuse, now collision-checked — "user/Doctor choices are respected"); (2) a recorded `FabricIntent` that passes the collision check is re-applied; (3) `choose_fabric_subnet(_occupied_networks(hosts))` (`:1054`, `:1079`) picks fresh **and writes intent** after `assess_link` confirms (`:871-876`). Addressing itself goes through `_authorized_networksetup` with `ifconfig` (`_authorized_ifconfig`) as fallback when no service maps to the interface |
| Modify | `omlx/cluster/transport.py:771-772` — the hardcoded `netmask 255.255.255.0` becomes a parameter derived from the chosen network's prefix |

### Data model / API

```python
class FabricIntent:      # base_path/cluster/fabric-intent.json
    subnet: str          # "172.16.99.0/24"
    hosts: tuple[str, str]
    chosen_by: str       # "auto" | "doctor" | "user"
    reason: str          # e.g. "vpn_exclusion", "collision_free_default"
    recorded_at: float
    addressing: str      # "networksetup" | "ifconfig"  (what was actually applied)
```

**Design deviation, recorded:** design C.3/C.6 specifies an RFC1918 **/29** and mask `255.255.255.248`; the committed `choose_fabric_subnet` uses **/24** candidates (`transport.py:1049-1051`, tests in `tests/test_cluster_fabric_subnet.py` assert `172.16.99.0/24`). The shipped /24 supersedes the /29 language — same collision guarantees, one shipped implementation. The mask parameterization added here makes a later narrowing trivial if ever wanted.

**Scope limitation, recorded:** intent is recorded by the **coordinator** (which drives setup). Writing a mirror record on the peer's store waits for Section A's `PairedPeer` plumbing; until then, a coordinator swap loses intent (worst case: fresh collision-checked selection, i.e. today's behavior).

### Checklist

- [ ] `fabric_intent.py` store (registry skeleton).
- [ ] `parse_hardware_ports` parser + fixture from real `networksetup -listallhardwareports` output.
- [ ] `_authorized_networksetup`: validate service/address/mask exactly as `_authorized_ifconfig` validates its inputs (`:756-763`); one combined `shell_command` when a service must be created first (single admin prompt).
- [ ] Rework `configure_link`'s selection tiers; write intent only after the post-addressing `assess_link` succeeds.
- [ ] Fallback: no service maps to the RDMA interface (Thunderbolt Bridge membership cases) → keep `ifconfig` addressing, record `addressing="ifconfig"` so C5's watchdog knows this link will drift on reboot.
- [ ] Parameterize the netmask through both authorized paths.

### Tests

- `tests/test_cluster_fabric_intent.py`: store round-trip, corrupt-file fail-closed, provenance fields.
- `tests/test_cluster_transport.py` (extend): `parse_hardware_ports` fixtures (TB bridge, multiple ports, missing service); `configure_link` with monkeypatched `_authorized_*` + fake probes — tier 1 keeps a valid existing subnet; tier 2 re-applies intent verbatim; colliding intent falls through to tier 3 and rewrites intent; intent written only after `assess_link` reports ready.
- Existing `tests/test_cluster_fabric_subnet.py` untouched (regression net).

### Risks / backward-compat

- **The real risk is macOS service mapping**: the RDMA-active `enX` may be a member of "Thunderbolt Bridge" rather than owning its own service, and `-setmanual` on the bridge service may not address the member interface the way the incident's `ifconfig` did. First task of this item is a 30-minute empirical spike on the two-Mac rig; if service addressing proves unreliable, ship intent + `ifconfig` + C5's watchdog re-apply as the durability mechanism and record `addressing="ifconfig"` permanently. The plan works either way; only the "survives reboot without oMLX running" property differs.
- Admin-consent fatigue: at most one prompt per node per re-address (combined command).
- Interface renumbering (`en6`→`en4`, comment at `transport.py:1026-1031`): service identity survives it — that is the point of preferring `networksetup`; the parser must map by hardware port, not cached BSD name.

---

## C3 — The Doctor flow: C.2–C.4 checks with errno-mapped JACCL probe

**Goal:** one guided flow runs the ladder checks in order, stops at the first red rung, and every finding carries diagnosis + concrete fix (auto-applied with consent where safe). Closes failures **#5** (stale-address as a named check) and **#6** (error-60 becomes a named, pre-checked condition).

### Files & symbols

| Action | Path / symbol |
|---|---|
| **Create** | `omlx/cluster/doctor.py` — `DoctorCheck`/`DoctorFinding`/`DoctorReport` dataclasses; `run_fabric_doctor(hosts, *, probes=...) -> DoctorReport`; `ERRNO_DIAGNOSES` table; `run_fabric_collective_probe(hosts, addresses, rdma_matrix, timeout=10)` |
| **Create** | `tests/test_cluster_doctor.py` |
| Modify | `omlx/cluster/routes.py` — `POST /doctor` (runs as a job via B2's `StartJobStore` generalization or its own tiny job dict; the Doctor takes tens of seconds), `GET /doctor/{job_id}`; report appended to the incident feed (B1) and to `/diagnostics` |
| Modify | `omlx/cli.py` — `cluster` subparsers (`cli.py:1288-1301`) gain `doctor`. **Naming deviation, recorded:** design says `omlx fabric doctor`; the CLI's only cluster namespace is `omlx cluster …` (`cli.py:800`, `:1288`) — ship `omlx cluster doctor` for consistency |
| Modify | `omlx/admin/templates/dashboard/_cluster.html` + `dashboard.js` — Doctor panel using B4's row grammar (check · state · evidence · fix); every [Run Fabric Doctor] affordance from B3/B4 copy targets it |
| Reuse | `transport.py`: `probe_host_interfaces` (`:1305`), `_occupied_networks` (`:1079`), `choose_fabric_subnet` (`:1054`), `verify_link_reachability` (C1), `_UNROUTABLE_NETWORKS` (`:1037`), `configure_link`/C2 for fixes; `collective.py:104` `run_local_collective_smoke` as the launcher template for the cross-host probe; `deployment.py:316-333` for the RDMA-matrix contract the probe must honor |

### Data model / API

```python
class DoctorFinding:
    check_id: str          # "link_presence" | "address_sanity" | "subnet_collision"
                           # | "route_pinning" | "bound_connect" | "jaccl_probe"
                           # | "rdma_staleness" | "admin_port"
    state: str             # "pass" | "fail" | "skipped"   (skipped ⇒ an earlier check was red)
    evidence: str
    diagnosis: str         # plain-language, from design C.2–C.4 copy
    remedy: str
    fix_action: dict | None  # {"kind": "readdress" | "move_subnet", ...} → C2 machinery

ERRNO_DIAGNOSES = {
    60: ("ETIMEDOUT", "connection is being silently dropped — a firewall or VPN on the fabric path", "fix: subnet move / VPN exclusion (C.3/C.5 remedies)"),
    61: ("ECONNREFUSED", "the peer's worker isn't listening yet — a launch-order problem, not a network problem; retrying is correct", None),
}
```

- `run_fabric_collective_probe`: a minimal 2-rank handshake **outside the real collective** — same ports and the same RDMA-matrix contract deployment validates (`deployment.py:316-333`) — launched the way `run_local_collective_smoke` launches loopback ranks (`collective.py:104`) but across the two hosts on the fabric IPs, 10 s budget. Success records measured bandwidth and earns `COLLECTIVE_OK` (B3); failure output is scanned for `error: <n>` and mapped through `ERRNO_DIAGNOSES` — converting the silent exponential backoff of failure #6 into one named finding. The retry-narration rule (design B.3) also applies: the real launcher's backoff emits one incident per retry tier — wire that where `DistributedJobSupervisor` (`launch.py:1680`) surfaces rank stderr, as part of this item.
- `POST /admin/api/cluster/doctor {"hosts": [...]}` → `202 {"job_id"}`; `GET /admin/api/cluster/doctor/{job_id}` → `{"phase", "findings": [...], "verdict": "Fabric verified — 68 Gb/s measured…" | first-red summary}` (design C.7's output grammar).

### Checklist

- [ ] `doctor.py` check functions, each pure over injected probe results (the `assess_link` injection pattern, `transport.py:883-886`, is the template).
- [ ] Check order = ladder order; stop-at-first-red with later checks marked `skipped` ("later checks would only report consequences of the first failure", design C.1).
- [ ] Check 1 (link/address sanity): TB port enumeration via collected status (`ThunderboltPort`, `models.py:73-90`); 169.254 detection against `_UNROUTABLE_NETWORKS`; interface-renumber finding referencing the en6→en4 note (`transport.py:1026-1031`); [Re-address link] fix → C2.
- [ ] Check 2 (subnet collision): hostile-prefix set from `_occupied_networks` + C4's utun/exclusion enrichment; copy per design C.3 with the concrete "WARP routes 10.0.0.0/8 through utun4" style evidence; [Move link addresses] fix → C2, then re-run ladder.
- [ ] Check 3 (routes + bound connect): C1's `verify_link_reachability`, findings split into route-pinning vs bound-connect rows with per-direction evidence.
- [ ] Check 4 (JACCL probe): `run_fabric_collective_probe` + errno mapping; refuse to run while a deployment is active (`get_cluster_registry().list()` non-empty for these hosts) — a probe on live ports would perturb the collective.
- [ ] Check 5 (RDMA staleness + admin-port): devices present (`RDMACapability.devices`, `models.py:53-61`) but no fabric addresses → "addresses were lost (this happens after reboot…)"; admin-port health check per C5.
- [ ] Every completed run → one incident (severity by verdict) with `bundle_ref`; report embedded in `/diagnostics`.
- [ ] Endpoint + CLI + dashboard panel; wire B4's fabric row [Run Fabric Doctor] and B3's remedy strings to it; automatic invocation when the ladder regresses (subscribe in the same code path that records the regression incident).

### Tests

- `tests/test_cluster_doctor.py`: each check's pass/fail/finding branches with fake probes; stop-at-first-red ordering (a collision failure marks route/JACCL checks skipped); errno table — stderr containing `(error: 60)` → firewall diagnosis, `61` → launch-order diagnosis, unknown errno → generic + raw detail; probe refusal while a deployment is registered; verdict string formats (success carries bandwidth).
- `tests/test_cluster_routes.py`: doctor job lifecycle; report lands in incidents and diagnostics.
- CLI: `tests/test_cluster_cli.py` gains a `doctor` invocation smoke test with a stubbed runner.

### Risks / backward-compat

- The cross-host probe needs the worker runtime on the peer; when absent, degrade check 4 to `skipped` with "worker runtime missing" evidence (reuse the `_RUNTIME_MISSING`/`_RUNTIME_UNVERIFIED` distinction, `launch.py:1049-1050`) rather than failing the fabric.
- Port choice for the probe must not collide with a starting deployment — take ports from `_available_launch_ports` (`launch.py:126`) and hold the same activation lock B2 introduced.
- Doctor auto-fixes are consent-gated through the existing `_remote_gui_authorize` bridge — no new privilege surface.

---

## C4 — VPN pre-warning + exclusion enrichment

**Goal:** detect a hungry full-tunnel VPN *before* the user hits it, warn in plain language, and feed readable exclusion lists into subnet selection — while empirical probes (C1/C3) remain the only promotion authority. Closes the "before" half of failure **#4**.

### Files & symbols

| Action | Path / symbol |
|---|---|
| **Create** | `omlx/cluster/vpn.py` — `VPNProfile` dataclass; `detect_vpn(host, *, runner=...) -> VPNProfile`; pure parsers `parse_route_table(netstat_output)`, `parse_warp_settings(output)` |
| **Create** | `tests/test_cluster_vpn.py` |
| Modify | `omlx/cluster/transport.py:1054-1076` — `choose_fabric_subnet` gains `preferred: Sequence[IPv4Network] = ()`: candidates inside a detected VPN exclusion rank before the static preference order (systematizing the incident's 172.16.99.x trick, design C.3 selection order item 2) |
| Modify | `omlx/cluster/transport.py:1079-1098` — `_occupied_networks` (or a wrapper `hostile_networks(hosts)` in `vpn.py`) also counts every route pointing at a `utun*` interface as occupied, not just interface subnets |
| Modify | `omlx/cluster/routes.py` — `_autoconfigure` / B4 `/readiness` responses gain a `"vpn"` field per host: `{"present", "client", "full_tunnel", "exclusions": [...]}` |
| Modify | `_cluster.html` / `dashboard.js` — pre-warning banner before any fabric setup (design C.5 copy: "…is on a corporate VPN that captures all traffic. oMLX will pick link addresses the VPN ignores and verify the link end-to-end before use.") |

### Data model / API

```python
class VPNProfile:
    present: bool
    client: str           # "warp" | "tailscale" | "globalprotect" | "anyconnect" | "unknown"
    full_tunnel: bool     # default route or /1+/1 pair via a utun
    utun_interfaces: tuple[str, ...]
    exclusions: tuple[str, ...]   # readable split-tunnel exclusion CIDRs, may be empty
```

Heuristics (in order, all failure-tolerant): utun interfaces from `probe_host_interfaces` (`transport.py:1305`); default//1+/1 routes via `netstat -rn` through `_run_link_command` (`:1508`); client signatures — `warp-cli settings` / Cloudflare plist, Tailscale, GlobalProtect, AnyConnect. Any read failure → `client="unknown"`, empty exclusions. **No configuration read ever promotes the ladder** (design C.5 layer 2): `VPNProfile` only (a) warns and (b) enriches selection.

### Checklist

- [ ] Parsers + `detect_vpn` with fixture-driven tests (capture a real WARP full-tunnel `netstat -rn` and `warp-cli settings` output as fixtures).
- [ ] `hostile_networks` = interface subnets ∪ utun-routed prefixes; thread into `configure_link`'s tier-3 selection and the Doctor's check 2.
- [ ] `preferred=` plumbing in `choose_fabric_subnet` (exclusion-contained candidates first), keeping the existing candidate tuple (`transport.py:1049-1051`) as the base order.
- [ ] `"vpn"` field in autoconfigure/readiness; banner in the dashboard; the per-client copy-paste exclusion instruction strings from design C.4 live in `vpn.py` beside the detection (single source).
- [ ] Doctor check 2 evidence upgraded to name the client and captured range when known.

### Tests

- `tests/test_cluster_vpn.py`: WARP full-tunnel fixture → `present, full_tunnel, client="warp"`; split-tunnel exclusions parsed; garbage output → `unknown` without raising; utun-routed 10.0.0.0/8 marks the whole /8 hostile.
- `tests/test_cluster_fabric_subnet.py` (extend): exclusion-contained candidate wins over the static order; empty `preferred` preserves today's assertions byte-for-byte.

### Risks / backward-compat

- MDM-locked or absent `warp-cli` — by design tolerated (heuristic layer only).
- Client output format drift — parsers defensive, fixtures dated in comments.
- Do not let a *wrong* exclusion read poison selection: exclusion-preferred candidates still pass the same collision check.

---

## C5 — Drift watchdog + dead-port retirement via advertised admin port

**Goal:** the memory-ceiling fast path probes the peer's *real* admin port and confesses when it falls back; live addressing is compared against recorded intent and drift raises an incident or self-heals. Closes failure **#7** and the reboot-drift tail of **#4/#5**.

### Files & symbols

| Action | Path / symbol |
|---|---|
| Modify | `omlx/cluster/models.py` — `ClusterStatus` gains `admin_port: int = 0` (beside `fabric_verified`, `models.py:141`; serialized in the `node` dict, `:151-165`) — the same additive pattern [#2819] used for `ssh_user` |
| Modify | `omlx/cluster/probe.py:561+` — `collect_cluster_status` populates it from `get_settings().server.port` (`omlx/settings.py` docstring `:19`; live example `omlx/server.py:452`), fail-soft to 0 |
| Modify | `omlx/cluster/launch.py:1122-1229` — `probe_remote_admission_ceiling` gains `admin_port: int = 0`; the embedded script's hardcoded `http://127.0.0.1:9000/health` (`:1150`) becomes the passed port, and the silent `except Exception: pass` (`:1153-1154`) is replaced by returning `{"fast_probe_ok": false, "fast_probe_error": ...}` alongside the slow-path ceiling so the *caller* can confess |
| Modify | `omlx/cluster/routes.py:2725-2760` — the one caller (`/node-budgets`, probe call at `:2734`) threads the peer's advertised `admin_port` (from the freshest peer status it holds) and, on `fast_probe_ok == false`, records `Incident(WARN, "fast ceiling probe unreachable on port {n}; using slower local computation")` — the design's "fallbacks confess" rule, B1 dependency |
| **Create** | drift check: `omlx/cluster/fabric_intent.py` gains pure `detect_drift(intent, live_interfaces) -> DriftFinding | None`; invoked from the 10 s discovery tick's server path — concretely, inside B4's `/readiness` assembly (or `_autoconfigure`'s fabric step) where fresh `probe_host_interfaces` data is already in hand |
| Note | when Section A's `PairedPeer` lands, `admin_port` also becomes a stored field of the peer record (design A.6); until then the advertised-status value is the source |

**Explicit non-goal:** an event-driven wake/network-change watcher (macOS `SCNetworkReachability`/sleep-wake notifications). No such infrastructure exists in the repo (`supervisor.py` is worker lifecycle only); the poll-driven comparison above catches drift within one dashboard tick, which is when anyone can act on it anyway. Leave a future-work line in `fabric_intent.py`.

### Data model / API

- `ClusterStatus.node.admin_port` — additive.
- Remote ceiling script return: `{"admission_ceiling_bytes": int, "fast_probe_ok": bool, "fast_probe_error": str}` (coordinate with B5, which adds `"breakdown"` to the same script — **same PR or C5 first**).
- `DriftFinding {kind: "address_lost" | "address_changed" | "intent_collides", live, expected}` → incident copy: silent auto-restore when the collision check still passes **and** `addressing == "networksetup"` (re-apply needs no new consent because the service config persists — restoration is `networksetup` re-assert); otherwise `Incident(WARN, "The link's saved addresses now collide with a new VPN range — Fabric Doctor needs to pick new ones.")` or (for `ifconfig`-recorded links) an incident prompting a consented Doctor re-address rather than a silent privileged action.

### Checklist

- [ ] `admin_port` field + probe population + serialization.
- [ ] Rework the ceiling script: parameterized port, honest fast-probe result; keep `admission_ceiling_bytes` top-level (older-coordinator compat).
- [ ] Thread the port in `/node-budgets`; WARN incident on fallback; the same port feeds Doctor check 5 ("Fast memory probe target isn't answering on port {n} — Start will still work but planning will be slower", design C.4).
- [ ] `detect_drift` + invocation point + incident/auto-restore policy above.
- [ ] Grep for any other `9000` assumption (verified: `launch.py:1150` is the only one in `omlx/cluster/`).

### Tests

- `tests/test_cluster_probe.py`: `admin_port` advertised; settings-unavailable → 0.
- `tests/test_cluster_launch.py`: script receives the given port; fast-probe failure returns slow-path ceiling with `fast_probe_ok=false` (no exception, no silence); port 0 skips the fast path entirely.
- `tests/test_cluster_routes.py`: `/node-budgets` fallback records the WARN incident.
- `tests/test_cluster_fabric_intent.py` (extend): drift matrix — live==intent → None; missing address → `address_lost`; new collision → `intent_collides` with the design's copy.

### Risks / backward-compat

- Older peer advertising no `admin_port` → 0 → fast path skipped, slow path used, one-time info incident, not a failure.
- Auto-restore must never invoke a *new* privileged prompt without a user action — hence the `addressing`-conditional policy above.
- Both C5 and B5 edit the same embedded script — dependency graph orders them.

---

# Dependency graph & sequencing

```
P0 (merge #2819 + rebase)  ──────────────┐  (hard prerequisite for: B5's VLM flags,
                                         │   anything touching dashboard poll guards,
                                         │   home-relative staging assumptions, ssh_user)
C1 (bound connect) ── none               │
B3 (ladder+guidance) ── none             │
B1 (incidents) ── none                   ▼
B2 (start job) ──► B1                    (soft: job failures become incidents)
C2 (durable addressing) ── C1 (assess_link is its verifier); B1 soft (drift incidents live in C5)
C5a (admin_port + probe retarget) ── B1 (the WARN incident)          [small]
B5 (budget+strategies) ──► C5a (same embedded script), P0 (VLM flag)
B4 (panel) ──► B1 + B3 (+ B2 for button state, B5 for its budget/strategy rows)
B6 (asset handshake + slices) ──► B1 (bundle_ref); handshake half standalone
C3 (Doctor) ──► C1 + C2 + B3 (row grammar/ladder) + B1 (report→incident) + B2 (job runner)
C4 (VPN) ──► C1 (empirical-authority split); feeds C2/C3 selection
C5b (drift watchdog) ──► C2 (intent) + B1
```

No cycles. `C1`, `B3`, `B1` are the independent roots; `C3` is the deepest sink.

# PR / branch grouping (merge order)

| # | Branch (suggested) | Contents | Depends on |
|---|---|---|---|
| PR-0 | `fix/cluster-tensor-and-poll-jitter` | **already open (#2819)** — merge, then rebase this branch | — |
| PR-1 | `feat/cluster-diagnostics-and-fabric-doctor` (this branch, continued) | **C1** on top of the committed subnet selection — together they are C.8 step 1, one reviewable theme: "the fabric is only ready when proven" | PR-0 (rebase only) |
| PR-2 | `feat/cluster-readiness-ladder` | **B3** (enum, readiness module, LinkStatus fields, guidance codes, CLI line) | PR-1 (LinkStatus edits stack) |
| PR-3 | `feat/cluster-incidents` | **B1** (store, endpoints, emitters, feed) | — |
| PR-4 | `feat/cluster-start-job` | **B2** (refactor + job runner + button) | PR-3 |
| PR-5 | `feat/cluster-durable-fabric` | **C2** + **C5a** (intent store, networksetup, admin_port, honest ceiling probe) — one PR because both touch `configure_link`/probe surfaces reviewers must see together | PR-1, PR-3 |
| PR-6 | `feat/cluster-budget-strategies` | **B5** (breakdown + strategies) | PR-5 (shared script), PR-0 |
| PR-7 | `feat/cluster-start-panel` | **B4** + **B6** (panel, strip, handshake, slices) | PR-2, PR-3, PR-4, PR-6 |
| PR-8 | `feat/cluster-fabric-doctor` | **C3** + **C4** + **C5b** (doctor, VPN, drift) — the Doctor is the integration point for all three | PR-1, PR-2, PR-3, PR-4, PR-5 |

Each PR is independently shippable and behind no flag: every item degrades additively (new endpoints, new fields, stricter-but-honest checks).

# Effort sizing & first-two-weeks slice

| Item | Size | Basis |
|---|---|---|
| C1 | **S** | one function rework + fake-runner tests |
| B3 | **M** | enum + 9 LinkStatus sites + copy table + guidance codes |
| B1 | **L** | new store + endpoints + three emitter funnels + feed UI + poll rewrite |
| B2 | **M–L** | endpoint-body refactor is the risk; job runner itself is small |
| C2 | **M** | store S, networksetup empirics are the unknown (timeboxed spike) |
| C5 (a+b) | **S** | additive field + script rework + pure drift check |
| B5 | **M** | two response extensions + UI rows |
| B4 | **M** | composition + template work |
| B6 | **S–M** | middleware S; slice endpoint M-ish |
| C3 | **L** | cross-host probe + check suite + job wiring + UI |
| C4 | **S–M** | parsers + selection plumbing |

**Recommended first two weeks:** PR-1 (**C1**) → PR-2 (**B3**) → PR-3 (**B1**), while PR #2819 review/merge and the C2 `networksetup` spike run in parallel. That closes the ladder-honesty and silent-failure classes (failures #5, #8, and half of #4/#6) before any UI composition work, and everything after stacks on merged foundations.

---

# D — Model-sharding fixes (distinct workstream from B/C diagnostics)

These items make specific model *architectures* actually run under distributed inference. They are independent of the B/C fabric-doctor work and can ship on their own PR (target: the existing #2819 clustering-reliability PR). Root-cause analysis verified live on mlx 0.31.2/0.31.3; see `docs/vlm-distributed-design.md` for the VLM half.

## D1 — Nemotron-H TP=2 quantized-MoE sharding fix

**Problem.** `nemotron_h` (hybrid Mamba2/attention + MoE, 4-bit oQ4, group_size 64) crashes mid-shard at TP=2: `Array split does not result in sub arrays with equal size: 2 splits along axis -1 for shape (128,2688,29)`. The MoE experts' down-projection (`switch_mlp.fc2`) is row-parallel over the 1856-wide intermediate = `1856/64 = 29` quant groups; **29 is prime**, so no 2ⁿ TP degree splits it evenly. The planner is blind to the quant-group constraint and offers TP=2 anyway. A latent sibling bug slices quantized Mamba `in_proj.weight` without its `scales`/`biases`.

### Files & symbols
- `omlx/cluster/tensor_strategies.py` — `_shard_nemotron_h` (~400): Mamba branch (471) and MoE branch (507-521); `_wrap_sharded_moe` (237) needs **no** change (its `all_sum` is shape-agnostic).
- `omlx/cluster/planner.py` — `_tensor_parallel_divisors` (611), nemotron_h branch (625).
- Config: per-tensor quantization overrides live in `config.json["quantization"]` (some layers 8-bit gs=64).

### Checklist
- [ ] **D1a (mandatory latent-bug fix):** in the Mamba branch, after `mixer.in_proj.weight = ...[indices]` (line 471), apply the **same `indices`** to `mixer.in_proj.scales` and `mixer.in_proj.biases`, guarded by `hasattr(mixer.in_proj, "scales")`. Row-slicing is group-safe (groups run along the input axis). Without this, layer-0 Mamba shards into a silently-broken state (10304 scale rows vs 5152 weight rows).
- [ ] **D1b (the fix that makes it run):** replace the two `shard_inplace` calls on `switch_mlp.fc1`/`fc2` (508-509) with **uneven group-aligned manual slicing**. Compute per-rank group ranges over the N scale-groups (`base = groups // size`; first `groups % size` ranks get one extra → rank0=[0,15), rank1=[15,29) at TP=2). `fc1` (column-parallel, ReLU² non-gated so output = 1856, one group axis): slice **axis 1** of weight/scales/biases at `[glo*gs : ghi*gs]`. `fc2` (row-parallel): slice `weight` **axis -1** packed cols `[glo*(gs/8) : ghi*(gs/8)]` (4-bit → 8 values/uint32), `scales`/`biases` axis -1 at `[glo:ghi]`. Wrap slices in `mx.contiguous`. Verified numerically exact vs unsharded (1.5e-5 on signal magnitude 80). Keep `shared_experts` on `shard_inplace` (down_proj input 3712 → 58 groups, even).
- [ ] **D1c (planner honesty, ship regardless):** extend `_tensor_parallel_divisors` (or add a companion viability check) so quantized **row-split** tensors (o_proj, mamba out_proj, fc2, shared down_proj) contribute `input_dim / group_size` using the quant dict **including per-tensor overrides**. Require divisibility where `shard_inplace`/`shard_linear` still runs; require only `group_count >= size` for the custom-sliced fc1/fc2. (kv_heads=2 already caps this model at TP=2, so the guard only needs TP=2 honesty here.)

### Tests
- [ ] `tests/test_cluster_tensor_strategies.py`: per-rank group-range math (15/14 split) + fc2 packed-column arithmetic; assert reassembled scales cover [0,29) with no overlap.
- [ ] `tests/test_cluster_planner.py`: a synthetic nemotron_h-like quant config with an odd row-split group count is **not** rejected outright (custom path allows `>=size`), and a genuinely un-splittable `shard_inplace` dim **is** flagged.

### D1e/D1f — runtime forward-path fixes (found while verifying on hardware)
The sharding fixes above got the model to *load*; two more bugs blocked the first
inference. Both are in `omlx/cluster/pipeline_compat.py` (`_install_nemotron_h_pipeline`),
whose `pipeline_call` becomes `NemotronHModel.__call__` even for pure TP:
- [x] **D1e:** `pipeline_call` rejected the `n_confirmed` kwarg that the base MTP
  patch threads through every model `__call__` (`TypeError: … unexpected keyword
  argument 'n_confirmed'`). Accept `n_confirmed: int = 0, **_unused` and ignore it
  (MTP is inactive on the distributed path). Mirrors the mlx-vlm runtimes' `pop`.
- [x] **D1f:** `pipeline_call` read `pipeline_rank`/`pipeline_size`/`fa_idx`/`ssm_idx`,
  which only `pipeline()` sets — never called on the pure-TP path (`AttributeError:
  'NemotronHModel' object has no attribute 'pipeline_rank'`). Default to a single
  stage (`rank 0`, `size 1`, so send/recv/all_gather no-op) and recompute the
  first attention/SSM cache indices locally when `pipeline()` did not run.

### Verify (real hardware) — ✅ DONE 2026-08-19
- [x] Both fork servers pick up changes via the worker shim (workers re-import per launch); files synced to the peer worktree `/Users/aphoenix/repos/omlx-fork-test`.
- [x] Activation driver → `deployments: 200`; deployment `…-1899cb75cc02` live, jaccl, rank0=mini / rank1=peer.
- [x] `/v1/chat/completions` generated coherent tokens across both Macs (TP=2 `all_sum` succeeded → both ranks live).

### Fallbacks (do NOT implement unless D1b blocks)
- Requant fc2/down_proj family only to gs=32 into a **new** dir (58 groups → even). ~19GB re-stage; requant error bounded by original (not nil).
- Expert-parallel (shard the 128-expert axis): correct collectively but `gather_qmm` still computes all experts → no compute win without token compaction. Legitimate upstream project; defer.

## D2 — Qwen3.5/3.6 text-only distributed (VLM checkpoints)

**Problem.** `Qwen3.6-35B-A3B` / `Qwen3.8-27B` are VLM checkpoints (`*ForConditionalGeneration`, `vision_config` present). Activation is rejected with HTTP 400 ("…is a vlm model. Distributed cluster inference currently supports text LLM models only.") at `engine_pool.py:668-674`. But the pinned mlx-lm already loads these **text-only** (`qwen3_5*.Model.sanitize` drops `vision_tower.*`/`visual.*`; `qwen3_5.Model.shard()` passes oMLX's `native_shard_is_layer_local` proof). Only the gate + accounting block a working text-only cluster.

### Files & symbols
- `omlx/engine_pool.py` — the gate: `resolve_cluster_model_id` (668-674), `register_cluster_model` (706-711), `_distributed_deployment_for_entry` (186) / dispatch (2137-2141).
- `omlx/cluster/inference_worker.py` — `_validate_loaded_stage` (540-575) hard-codes `model.model`; qwen3_5 family Model wraps `language_model` → **latent crash on pure-TP text Qwen too**. Mirror the `_common_layer_owner` fallback (`tensor_strategies.py:96-119`).
- `omlx/cluster/planner.py` — `_LAYER_PATTERNS` (29) wrongly merges `vision_tower.blocks.*` (0.83 GiB) and `language_model.mtp.layers.0` (0.47 GiB) into decoder layers → over-counts layer sizes for VLM checkpoints.

### Checklist (tier a — text-only, oMLX-only, no upstream change)
- [ ] Behind an explicit **"deploy text-only"** flag (never silent vision-drop — see issues #1261/#1426): allow VLM checkpoints past the three `engine_pool` gates when the flag is set and a capability probe (`_supports_tensor_parallel`/`_supports_pipeline`) confirms shardability.
- [ ] Text-only layout filtering in the planner: exclude `vision_tower.*` and `*.mtp.*` from `_LAYER_PATTERNS` accounting; admission uses `text_only_size`.
- [ ] Fix `_validate_loaded_stage` to resolve the layer owner via the `language_model`/`model` fallback chain (also fixes pure-TP text Qwen3.5/3.6 today).
- [ ] Reject image requests at inference time with a clear error when a deployment was activated text-only.
- [ ] 2-rank TP smoke test for a qwen3_5 text-only shard.

### Tests
- [ ] `_validate_loaded_stage` accepts a `language_model`-wrapped model tree.
- [ ] planner text-only accounting excludes vision/mtp params (assert measured layer-0 ≈ 456 MiB, not 960 MiB on the 35B).
- [ ] gate allows VLM checkpoint only with the text-only flag + capability pass.

### Deferred (tier b — full VLM, vision on rank 0): L–XL, needs an image path in mlx-lm's server and TP-broadcast of visual embeddings (Qwen3.6 deepstack). Out of scope for this PR.
