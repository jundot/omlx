# oMLX Cluster Assistant

A DAU-friendly cluster mode inside oMLX: 2–4 Apple Silicon Macs over
Thunderbolt, automatic discovery/pairing, and a coordinator/rank 0/endpoint that
is **elected by a measured, feasibility-first plan** instead of being decided by
whichever Mac happened to open the dashboard.

This branch carries the whole flow up to a running cluster: the assistant
**measures** the Macs, **recommends** a model, **plans** the pipeline split,
**applies** (stages) it, **starts** it on the elected coordinator and reports
one endpoint — and can stop it again and tune it.

## What is in this branch

- **Coordinator election** — `omlx/cluster/assistant/election.py`. A pure
  decision over measured facts: capacity, admission, compute, links, model
  shape. The plan uses the **fewest Macs that fit**, ranks by measured critical
  stage (then headroom, then fewer hops, then the coordinator preference), and
  **never** falls back to the local Mac. Rank 0, coordinator and endpoint are
  co-located; pipeline parallelism only (tensor parallelism is refused).
- **Measurement** — `measurement.py` turns oMLX's own node budgets,
  performance profiles and link measurements into `NodeFacts`. A measurement
  that did not happen stays missing: no zero, no nominal rate, no estimate.
- **Persisted measurements** — `state.py`: one bounded, atomic JSON file under
  `<base_path>/cluster/`, keyed by `measurement_id`, with one documented
  staleness/mismatch rule (15 minutes, node set, model identity, MTP).
- **Model recommendation** — `recommendation.py` answers “which model should
  these Macs run?” from the hardware half of a measurement, over the whole
  catalogue, and prefers single-node inference when a model fits one Mac.
- **Resource policy** — `policy.py` carries context, table offload
  (Engram/PLE), routed-expert offload, memory-guard tier, microbatch and the
  admission percentile through measure → recommend → plan as **one object**. The
  plan shows every lever and the experts-vs-context trade as an explicit choice.
- **Topology** — `topology.py`: a 3/4-Mac plan needs a connected line through
  its rank order; the 2-Mac fallback needs explicit consent, salted into the
  plan signature.
- **Apply** — `provisioning.py` re-elects from the stored measurement, verifies
  the approved `plan_signature`, resolves each Mac's paired SSH target at
  request time, writes the chosen per-model levers on every selected Mac and
  builds the coordinator's deployment request. It stages; it never activates.
- **Start** — `start.py` drives the elected coordinator's own plan/stage/
  deployments routes: locally in-process when the coordinator is this Mac, over
  the existing SSH seam when it is another one. `endpoint.ready` becomes true
  only after health **and** one real smoke completion. A refused coordinator
  start has exactly one action (`retry_coordinator`) and never starts this Mac.
- **Operations** — `operations.py`: one bounded, atomic, forward-only record per
  apply/start with one operation at a time cluster-wide (`operations/active`,
  `operations/mirror`). A reloaded card resumes from `session`.
- **Tuning** — `tuning.py`/`tuning_store.py` sweep sampling knobs and the MTP
  on/off/depth configs with **real completions**, save the winner as a preset
  per (model, hardware), and replay it on the next Start (`redeploy_tuning.py`
  does the redeploy sweep for the per-model MTP settings).
- **Support bundle** — `diagnostics.py` exports a redacted report (no key, SSH
  target, model path, prompt or completion) and `network_preflight.py` returns
  read-only local commands; neither executes anything.
- **The card** — one assistant card inside the existing Cluster tab. It renders
  the backend's answer, shows exactly one action per refusal, exposes
  `endpoint.url` only when the operation is `ready` and its endpoint proved
  ready, and never derives a plan, a rank order or a coordinator itself.

## API

All routes live under `/admin/api/cluster/assistant` and are mounted with the
same two gates as the rest of the cluster surface (admin, experimental cluster
enabled):

```text
POST   /measure                          start a job, return a measurement_id
GET    /status?measurement_id=…          poll that job
POST   /recommend                        which model should we run?
POST   /plan                             plan from the stored facts
POST   /apply                            stage the approved plan
POST   /start                            launch it on the elected coordinator
POST   /stop                             stop the ready deployment
GET    /session                          the last deployment, after a reload
POST   /search                           measure up to N constellations, keep the best
GET    /search/{job_id}                  poll that search
POST   /tune                             fine-tune the winning deployment
POST   /tune-redeploy                    sweep the MTP switch/depth by redeploying
GET    /tune-redeploy/{job_id}           poll that sweep
GET    /tuning-preset                    the saved preset for a model + node set
GET    /performance-probe                read the post-ready probe switch
POST   /performance-probe                write it
GET    /diagnostics                      redacted support bundle
GET    /network-preflight                read-only local network plan
GET    /operations/active                is the coordinator busy? (server-to-server)
POST   /operations/mirror                register/release another origin's operation
GET    /operations/{operation_id}        poll that operation
DELETE /models/{node_id}/{model_id}      free space on a Mac that is short
```

`measure`, `apply`, `start`, `search`, `tune-redeploy` and `tune-local` are jobs
because they take seconds to minutes. `recommend` and `plan` are synchronous and
pure: they load the stored record and decide, so a plan can never be justified
by a number it measured itself. One measurement job and one apply/start run at a
time; a second is refused with `409` rather than joined.

## Refusals

Every refusal carries a machine-readable `code` and **exactly one** `action`:

`measure`, `plan`, `apply`, `start`, `status`, `add_mac`, `pair_mac`,
`copy_model`, `configure_endpoint`, `retry_coordinator`, `diagnostics`,
`adjust_model_or_context`, plus the recommendation's own model-choice actions
`choose_smaller_model`, `choose_pipeline_model`, `check_model_path` and
`free_disk_space`.

`free_disk_space` is answered by the delete route above, and `apply` uses the
same rule automatically when a model must be distributed: only a model that Mac
listed, never the checkpoint the cluster is about to run, never the last
complete copy, largest first.

## Measured, never promised

`predicted` carries stage seconds, not tokens/s. A token rate appears only in
the operation's `observed` block, filled **after** `ready` by one real
non-streaming completion (decode) plus one streaming completion (TTFT/prefill)
on the elected coordinator — off by default (`ServerSettings.
assistant_performance_probe`) and never an estimate. The synthetic distributed
compute probe runs after `apply` and feeds the **next** plan; it never rewrites
the approved placement and never gates a start.

## What this branch does not do

- No single-node “run on this Mac” flow (that is the next slice).
- No SSE/streaming; the card polls one job at a time with a generation guard.
- No tensor parallelism and no more than four Macs.
- No macOS-app, tray or signed-helper work, and no telemetry.

## Development

```sh
uv sync --extra cluster
uv run pytest tests/cluster_assistant
```

The web UI lives in `apps/omlx-web`; rebuild CSS there after template changes:

```sh
cd apps/omlx-web
python build_css.py
```

The assistant tests are fake-driven: no Mac, model, GPU or SSH is involved. Live
evidence from the validation cluster belongs in the project's run log, not in
the test suite.