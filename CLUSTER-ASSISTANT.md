# oMLX Cluster Assistant

This repository is a **focused fork/branch of oMLX** for the Cluster Assistant project.

It starts from oMLX commit:

```text
1e27576af5e10c8f4459e0fdcb05c5cf59245d44
```

Do not confuse this project with the older LocalInferenz prototype. LocalInferenz is a reference, not the base. The target is the existing oMLX app, web UI, cluster control plane, and runtime.

## Read first

1. [`docs/cluster-assistant/PROJECT-BRIEF.md`](docs/cluster-assistant/PROJECT-BRIEF.md)
2. [`AGENTS.md`](AGENTS.md)
3. [`docs/cluster-assistant/README.md`](docs/cluster-assistant/README.md)
4. [`docs/cluster-assistant/UPSTREAM.md`](docs/cluster-assistant/UPSTREAM.md)

The project brief is self-contained. Its **Appendix A** is the cold-start handoff for a new agent.

## What this project is

A DAU-friendly cluster mode inside oMLX:

- 2–4 Apple Silicon Macs over Thunderbolt.
- Automatic discovery/pairing.
- Any Mac can originate the setup.
- The coordinator/rank 0/endpoint is elected by a measured, feasibility-first plan.
- Pipeline parallelism only in the MVP.
- English-only product UI.
- The user reaches Chat or an agent without touching a terminal.

## What this project is not

- Not exo.
- Not a second LocalInferenz app.
- Not a second oMLX wizard or control plane.
- Not tensor parallelism yet.
- Not a German-localized product.

## Current status

Bootstrap, the election library, the measurement job, the measurement-driven
plan API, model recommendation, the apply/start backend, the minimal Screen-3
card, and the **first live run on real Macs** (Phase 2b):

- oMLX source checked out at the pinned baseline.
- Project brief copied in.
- Agent handoff rules in `AGENTS.md`.
- Coordinator-election library in `omlx/cluster/assistant/election.py`, with
  tests in `tests/cluster_assistant/test_election.py`.
- Measurement layer in `omlx/cluster/assistant/measurement.py`: turns oMLX's
  node-budget payloads, performance profiles and link measurements into
  `NodeFacts`. A missing measurement stays missing.
- Persisted measurements in `omlx/cluster/assistant/state.py`: one bounded,
  atomic JSON file under `<base_path>/cluster/`, keyed by `measurement_id`,
  with one documented staleness/mismatch rule.
- `POST /admin/api/cluster/assistant/measure` starts an asynchronous,
  read-only job and returns a `measurement_id` immediately;
  `GET .../status` polls it. One job at a time; a second is refused with 409.
- `POST /admin/api/cluster/assistant/plan` is synchronous and pure: it loads
  the stored measurement, never probes, and returns the brief's §13.3 plan
  object with a real endpoint port and `ready: false`.
- Model recommendation in `omlx/cluster/assistant/recommendation.py`:
  `POST /admin/api/cluster/assistant/recommend` ranks candidate models for the
  measured Macs from the hardware half of a stored measurement. Feasibility is
  `elect()` per candidate (PP only), the context is the largest ladder value
  that still plans, staging bytes are reported per Mac, and a model that needs
  tensor parallelism or cannot be read is refused with one action.
- Apply/start backend: `omlx/cluster/assistant/operations.py` (persisted,
  forward-only operation store, one apply/start at a time),
  `provisioning.py` (re-elect, verify `plan_signature`, resolve paired targets,
  build the `ClusterDeploymentRequest`) and `start.py` (stage through oMLX's
  staging job, then health → activation → readiness canary on the elected
  coordinator, locally or over the existing SSH seam). Endpoint `ready` becomes
  true only after one passed smoke completion; a failed coordinator start has
  exactly one action and never starts a local rank 0.
- `election.budget_for` is the single budget builder the election and the
  recommendation share, so admission cannot mean two things.
- Minimal Screen-3 UI inside the existing Cluster tab plan section: one
  assistant card drives measure → recommend → plan → apply → start through the
  assistant routes, binds `selected_model`/`selected_context_tokens`,
  shows applying/starting/loading, exposes `endpoint.url` only when the
  operation is `ready` and `endpoint.ready` is true, and renders exactly one
  action from `error.action`. Entry is decision A: measure the selected model,
  recommend, and re-measure only if the recommendation names another model.
- Screen-4 progress on that same card: six observable rows (Checking Macs,
  Planning, Preparing runtime + Copying model, Starting coordinator, Loading
  model + Verifying endpoint, Ready) mapped one-to-one onto the assistant state,
  the operation kind/state and `progress.{phase,message,nodes_ready,nodes_total}`
  (`omlx_web/static/js/cluster_v2.js`, `assistantProgressRows()`). Ready needs
  `state === 'ready'` and `endpoint.ready`; a failure keeps the row it stopped
  in, an unknown phase keeps the previous row and shows the raw message, and no
  percentage, ETA or token rate is rendered. No backend change, no SSE.
- Free space: when a copy cannot fit (`insufficient_disk`, action
  `free_disk_space`), the card lists the measured models **of the Mac the user
  picked**, with sizes, and deletes one after a two-step confirm through the free-space assistant route
  (`DELETE /admin/api/cluster/assistant/models/{node_id}/{model_id}`), which
  relays to that Mac's own delete route over the admin seam. It never deletes the
  model the measurement plans to run, and it measures again afterwards.
- Screen-5 handoff: the ready card opens the **existing** `/chat` with the
  cluster's serving model id (`/chat?model=…` for a local coordinator, that
  Mac's own `<endpoint>/chat?model=…` for a remote one), and `Connect agent`
  shows `<endpoint>/v1`, the serving id and where to get the key on the
  coordinator. `/chat` reads that one `?model=` parameter, selects a known id
  and leaves its default alone for an unknown one; it adds no history entry and
  reads no credential. No second page or chat.
- Measured performance: the operation carries one nullable `observed` block
  (`prefill_tok_per_s`, `decode_tok_per_s`, `ttft_ms`, `measured_at`). It is
  filled only **after** `ready`, asynchronously, on the elected coordinator, by
  one real non-streaming completion (decode) plus one streaming completion
  (TTFT/prefill from the final usage chunk), never by the 2-token readiness
  canary and never by an estimate. It is off by default
  (`ServerSettings.assistant_performance_probe`), toggled in-app, `predicted` is
  never rewritten, the ready card can refresh the operation once, and a failed
  probe only adds a warning (`SCREEN-5-HANDOFF.md`).
- Measured compute: after `apply`, the coordinator runs the existing synthetic
  distributed probe through `POST /admin/api/cluster/performance-probe`; the
  origin stores the per-Mac profiles (`AssistantPerformanceStore`) and the
  **next** measurement/plan uses them. The probe never rewrites `predicted`,
  never gates apply or start, and a failure is one `compute_probe_failed`
  warning.
- Hardware-only measurement (C3): `measure` accepts no model at all, so the
  first step can answer "what hardware is here?", `recommend` reads it, and the
  chosen model is measured again before `plan`. `plan` refuses a hardware-only
  record with `stale_measurement` / `measure`.
- Reload resume (C1): `GET /admin/api/cluster/assistant/session` returns the
  last ready/applied operation with its plan inputs, policy, deployment and
  endpoint; the card offers Start again or Stop without rebuilding a plan. Stop
  is refused with `operation_in_progress` while another operation runs.
- Source-first model placement (C2): `apply` stages from the Mac that owns a
  complete copy (never a download); the free-space delete refuses the model the
  cluster is about to run and refuses the last measured copy.
- Screen 6: the ready card carries **Stop cluster**, which calls the tenth
  assistant route (`POST /admin/api/cluster/assistant/stop`); the backend relays
  to the elected coordinator's existing unload route through the same local/SSH
  seam and marks the operation `stopped` (its `state` stays `ready`). A stopped
  card hides the endpoint and offers **Start again**, which re-applies the frozen
  plan inputs and starts. No re-planning, no model/context edit, no Mac changes
  (`SCREEN-6-MANAGE.md`).
- Topology line (2026-11-07): the measurement records the point-to-point links;
  a 3/4-Mac plan needs a connected line through its rank order. A missing pair
  is `topology_incomplete` with the exact cables named; the 2-Mac plan is
  offered only after explicit confirmation, and the confirmation is part of the
  plan signature. An unmeasured fabric keeps the nominal-ring warning.
- Redacted support bundle (2026-11-07): `GET
  /admin/api/cluster/assistant/diagnostics` exports versions, hardware,
  topology, measurement, plan/result, health, warnings and the explicit
  redaction list. It never re-elects, probes or launches, and it contains no
  key, token, SSH target, model path, prompt or completion. The card copies it
  and falls back to its local text block when the backend is down.
- Still absent: SSE progress streaming, tray/macOS-app work and the signed
  helper (P2); TP and >4 Macs remain out.

First live run (Phase 2b, `docs/cluster-assistant/PHASE-2B-RUN.md`): the card ran
measure → recommend → plan → apply → start → ready on real Macs over the
Thunderbolt bridge (26.9 Gbit/s measured), ended on a reachable endpoint that
answered `/v1/models` and a real completion, and produced one-action failures
(`pair_mac`, `retry_coordinator`) without a local fallback. Nine live defects
were found; eight are fixed with fake-driven tests (discovery naming, model
inventory, local endpoint, local coordinator, remote admin settings, origin as
worker, the unknown-action dead end, unpaired Macs as launch targets). The
assistant now also plans Lightning MTP when the checkpoint supports it and fills
a documented chat sampling profile on a model that has none. What is deliberately
recorded instead of fixed: the Workstation role starving a 32 GB Mac, the
recommendation's action vocabulary, a cluster-wide operation lock, and the
distributed performance probe.

The Screen-5 handoff is built: the decision and implementation record are
[`docs/cluster-assistant/SCREEN-5-HANDOFF.md`](docs/cluster-assistant/SCREEN-5-HANDOFF.md).
The Screen-6 stop/start-again slice is
[`docs/cluster-assistant/SCREEN-6-MANAGE.md`](docs/cluster-assistant/SCREEN-6-MANAGE.md),
with live evidence in `PHASE-2B-RUN.md` §13. The Screen-4 progress surface is
built: its decision and implementation record are
[`docs/cluster-assistant/SCREEN-4-PROGRESS.md`](docs/cluster-assistant/SCREEN-4-PROGRESS.md)
§8, and the free-space delete relay is
[`docs/cluster-assistant/PHASE-2B-RUN.md`](docs/cluster-assistant/PHASE-2B-RUN.md)
§9.

The UI card, its states and its deferred items are documented in
[`docs/cluster-assistant/SCREEN-3-MINIMAL.md`](docs/cluster-assistant/SCREEN-3-MINIMAL.md)
§11.

Chapter 5 was not blocked by the distributed performance probe, model-less
measurement, or the full Screen-3 UI; the minimal slice is built and has run on
hardware. Performance is unmeasured by default (`predicted.measured=false`,
`observed=null`); the post-ready measurement may fill `observed`, and the
synthetic compute probe may feed the next plan — no rate is ever estimated.

The minimal first UI slice is defined in
[`docs/cluster-assistant/SCREEN-3-MINIMAL.md`](docs/cluster-assistant/SCREEN-3-MINIMAL.md).

## Development

```sh
uv sync --extra cluster
uv run pytest tests/cluster_assistant
```

The oMLX web UI lives in `apps/omlx-web`; rebuild CSS there after template changes:

```sh
cd apps/omlx-web
python build_css.py
```

## Source of truth

The code and tests in this fork are the source of truth. The brief records decisions and scope; it does not freeze the upstream oMLX implementation.

Patch/PR status is tracked separately in
[`docs/cluster-assistant/UPSTREAM.md`](docs/cluster-assistant/UPSTREAM.md). Do
not assume `origin/main` equals this base or that every local patch branch is
present.

If upstream oMLX moves, re-audit the cluster routes and UI before copying assumptions.