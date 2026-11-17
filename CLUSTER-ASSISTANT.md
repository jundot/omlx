# oMLX Cluster Assistant

A DAU-friendly cluster mode inside oMLX: 2–4 Apple Silicon Macs over
Thunderbolt, automatic discovery/pairing, and a coordinator/rank 0/endpoint that
is **elected by a measured, feasibility-first plan** instead of being decided by
whichever Mac happened to open the dashboard.

This branch is the first slice of that feature: the assistant **measures** the
Macs, **recommends** a model, and **plans** the pipeline split. Staging,
launching and tuning arrive with the run backend; nothing here starts inference.

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
  catalogue. Feasibility is `elect()` per candidate, the context is the largest
  ladder value at or below the target that still plans, and staging bytes are
  reported per Mac.
- **Resource policy** — `policy.py` carries context, table offload
  (Engram/PLE), routed-expert offload, memory-guard tier, microbatch and the
  admission percentile through measure → recommend → plan as **one object**. The
  plan shows every lever and the experts-vs-context trade as an explicit choice.
- **Topology** — `topology.py`: a 3/4-Mac plan needs a connected line through
  its rank order. A missing pair is refused with the exact cables named; the
  2-Mac fallback is offered only with explicit consent, which salts the plan
  signature.
- **The card** — one assistant card inside the existing Cluster tab
  (`apps/omlx-web/omlx_web/static/js/cluster_v2.js`, `_cluster_v2.html`). It
  renders the backend's answer, shows exactly one action per refusal, and never
  derives a plan, a rank order or a coordinator itself.

## API

All routes live under `/admin/api/cluster/assistant` and are mounted with the
same two gates as the rest of the cluster surface (admin, experimental cluster
enabled):

```text
POST   /measure                          start a job, return a measurement_id
GET    /status?measurement_id=…          poll that job
POST   /recommend                        which model should we run?
POST   /plan                             plan from the stored facts
DELETE /models/{node_id}/{model_id}      free space on a Mac that is short
```

`measure` is a job because the probes read memory ceilings, model inventories
and links over SSH. `recommend` and `plan` are synchronous and pure: they load
the stored record and decide, so a slow probe can never look like a hung plan
and a plan can never be justified by a number it measured itself. One
measurement job runs at a time; a second is refused with
`409 measurement_in_progress` rather than joined.

A measurement records each Mac's own catalogue, so “already present” is answered
from measured facts, and a model that lives only on another Mac is still usable:
the Mac that holds it is asked for the checkpoint's layout.

## Refusals

Every refusal carries a machine-readable `code` and **exactly one** `action`:

`measure`, `plan`, `status`, `add_mac`, `pair_mac`, `copy_model`,
`configure_endpoint`, `retry_coordinator`, `diagnostics`,
`adjust_model_or_context`, plus the recommendation's own model-choice actions
`choose_smaller_model`, `choose_pipeline_model`, `check_model_path` and
`free_disk_space`.

`free_disk_space` is answered by the delete route above: it deletes at most one
model, only on a Mac the measurement covered, only one that Mac itself listed,
never the checkpoint the cluster is about to run and never the last complete
copy across the measured Macs.

## What this branch does not do

- No staging, launch, activation, stop or tuning: those are the run backend
  (`apply`/`start`/`stop`, the persisted operation record, the tuning presets
  and the post-`ready` measurement).
- No single-node “run on this Mac” flow.
- No SSE/streaming; the card polls one job at a time with a generation guard.
- No tensor parallelism and no more than four Macs.
- No macOS-app or helper work, and no telemetry.

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