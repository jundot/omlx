# Cluster test runbook — `rebase/upstream-main`

The exact steps to bring the cluster up on this branch, prove the flow end to end,
and then build the DMG. Written 2026-11-14 for the run on 2026-11-15.

Branch: **`rebase/upstream-main`**, based on `upstream/main` = `624cff0e`
(`upstream/main` is an ancestor of HEAD). Pre-rebase fallbacks:
`feat/cluster-assistant` (`28245931`) and `backup/pre-rebase-2026-11-14`.
Background: [`REBASE-2026-11-14.md`](REBASE-2026-11-14.md).

Everything below is the shipped product surface. Nothing here re-plans, invents a
rank, or writes a setting the plan did not show.

---

## 0. What "green" means

| # | Gate | Evidence |
|---|---|---|
| 1 | Suite on this branch | `uv run pytest tests -q` (19863 passed / 776 skipped on 2026-11-14) |
| 2 | Web suite | `apps/omlx-web/tests` with node on PATH (559 passed) |
| 3 | macOS python | `uv run pytest apps/omlx-mac/Tests/python -q` (27 passed) |
| 4 | Origin serves the assistant surface | `GET /health` 200 and the peer check (§2) |
| 5 | Flow reaches `ready` on the elected coordinator | §3, endpoint answers `/v1/models` |
| 6 | Run-local serves Qwen3.8 on this Mac | §5 |
| 7 | Stop hides the endpoint and `Start again` works | §6 |
| 8 | DMG builds | §7 |

Record 5–8 in [`PHASE-2B-RUN.md`](PHASE-2B-RUN.md) with measurement, plan,
operation and deployment IDs. Live evidence lives there, never in the test suite.

---

## 0.4 Running the suite in a git worktree

The native custom kernels are build artefacts (`omlx/custom_kernels/*/*.so`,
`*.metallib`, gitignored), so a fresh worktree has none and two tests fail for
environmental reasons:

```
tests/test_glm5_next_mtp.py::test_split_k_batched_decode_matches_rows_alone[lengths0]
tests/test_glm5_next_mtp.py::test_split_k_batched_decode_matches_rows_alone[lengths1]
```

Neither touches cluster code; they pass in the main checkout where `make kernels`
ran. In a worktree either run `make kernels PYTHON=<worktree>/.venv/bin/python`
or copy the four `.so` files and their `.metallib` from the main checkout:

```sh
for t in attention lowbit quant_matmul sparse_index; do
  mkdir -p <worktree>/omlx/custom_kernels/$t
  cp omlx/custom_kernels/$t/* <worktree>/omlx/custom_kernels/$t/
done
```

After that the file is green (98 passed, verified 2026-11-15).

## 0.5 Reset to a clean cluster state (done 2026-11-15)

Before a from-scratch run (find devices → pair → measure → plan → apply → start,
manually and through the assistant), the origin holds no cluster state:

- **Kept:** `~/.omlx/cluster/identity.json` — this Mac's node identity. Deleting it
  would give the Mac a new `node_id` and invalidate every pairing for no benefit.
- **Deleted (backed up first):** `devices.json`, `deployments.json`,
  `assistant-measurements.json`, `assistant-operations.json`,
  `assistant-performance.json`, `assistant-tuning-presets.json`, `incidents.json`,
  `join-session.json`, `pairing-keys.json`, `pairing-audit.jsonl`,
  `strategy-benchmarks.json`, and the whole `~/.omlx/cluster/runtime/` staging
  cache (27 GB).
- **Backup:** `~/.omlx/backup-cluster-2026-11-15/` holds the eleven JSON files.

To repeat it later:

```sh
BK=~/.omlx/backup-cluster-$(date +%Y%m%d-%H%M%S)
mkdir -p "$BK"
cd ~/.omlx/cluster
cp -a devices.json deployments.json assistant-*.json incidents.json \
      join-session.json pairing-*.json* strategy-benchmarks.json "$BK"/ 2>/dev/null
rm -f devices.json deployments.json assistant-*.json incidents.json \
      join-session.json pairing-*.json* strategy-benchmarks.json
rm -rf runtime                      # the staged model copies
```

Verify with the server running:

```sh
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8099/api/cluster/devices   # paired: 0
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8099/admin/api/cluster/deployments  # []
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8099/admin/api/cluster/assistant/session  # {"session":null}
```

Not reset: `~/.omlx/settings.json` (API key, model dirs, bind) and
`~/.omlx/model_settings.json` (per-model preferences such as MTP and table
offload). Reset the latter only deliberately — it is model state, not cluster
state, and a stale switch would otherwise influence a plan.

## 1. Preconditions

- Repo on `rebase/upstream-main`, clean tree, `uv sync --extra cluster` done.
- **Distributed inference is on**: Settings → Advanced → *Distributed inference*
  (`server.distributed_inference_enabled = true`). With it off, every
  `/admin/api/cluster/...` route — including the whole assistant surface —
  answers **404**. This is the one setting that silently makes the run look
  broken.
- Checked on this Mac on 2026-11-14: `server.port = 8099`,
  `distributed_inference_enabled = true`, `auth.skip_api_key_verification = false`
  and a main API key is set — so the curl calls below **do** need the Bearer key.
- Peers are **paired** (Cluster tab) and reachable over SSH. The assistant only
  ever targets paired Macs.
- Thunderbolt fabric is a **line**, not a ring. A ring of macOS software bridges
  is a layer-2 loop and macOS does not run STP by default (measured 65 % packet
  loss, `PHASE-2B-RUN.md` §10). If a 3–4 Mac plan reports `topology_incomplete`,
  the plan names the missing link; do not "fix" it by cabling a ring.

Peers as this Mac last saw them (`~/.omlx/cluster/devices.json`). **Confirm on the
Cluster tab** — the table in `PHASE-2B-RUNBOOK.md` for `.12`/`.13`/`.14` is older
than this file:

| Node ID (first 8) | Name | Last known TB address | Port |
|---|---|---|---|
| `e8a5fd55` | MacBook-Pro-von-Andreas (this Mac, M5 Max) | `127.0.0.1` | 8099 |
| `82b7d9cc` | MacStudio-von-Andreas-2 | `172.16.1.13` | 8099 |
| `a6c9260f` | Mac-Studio-M2Ultra | `172.16.1.12` | 8099 |
| `c84867fe` | MacStudio-von-Andreas | `172.16.1.14` | 8099 |

---

## 2. Start the origin and check the peer builds

### 2.1 Origin server

```sh
uv run omlx serve --port 8099 --headless
```

- `--headless` serves the inference and admin APIs **without the web UI**: the
  right mode for a curl/API run.
- Drop `--headless` when you want to drive the same flow from the **Cluster tab**
  (one component, `clusterV2Wizard()`, assistant states in the existing tab).
- The default bind is `127.0.0.1`. A non-loopback bind (`--host 0.0.0.0`) **requires
  `--api-key`**; the peers' own servers must be reachable on the LAN for the
  version check and for their endpoint/deployment routes.
- Loopback requests skip admin auth **only** when `skip_api_key_verification` is
  on (it is off on this Mac), so pass the main API key as a Bearer token:
  `Settings → API key`. Never paste the key into a log or a doc.

```sh
curl -s localhost:8099/health          # 200 before anything else
```

### 2.2 Peer version check (read-only)

```sh
uv run python scripts/check_peer_versions.py            # --port 8099 by default
```

It probes four routes per Mac and never sends a key, never writes anything:
`POST /admin/api/cluster/stage`, `POST /admin/api/cluster/plan`,
`POST /admin/api/cluster/assistant/operations/active`,
`POST /admin/api/cluster/assistant/tuning-preset`.
`200/401/403/422` = route present (it answered before auth/validation);
`404/405` = **absent** → `OLDER BUILD`.

**Recorded run, 2026-11-14 (before anything was started):**

```text
e8a5fd55 (this Mac)                        unreachable (this Mac's server is not running)
82b7d9cc MacStudio-von-Andreas-2           unreachable (health=None)
a6c9260f Mac-Studio-M2Ultra                unreachable (health=None)
c84867fe MacStudio-von-Andreas             unreachable (health=None)

Every reachable Mac exposes the assistant route surface.
exit=0
```

Read that honestly: **no Mac was running**, so the check proved nothing. Its
`exit=0` means "no evidence of skew", not "every Mac is current".

**What to expect tomorrow:** this Mac (origin, current build) prints `OK`; the
peers are older builds without the assistant operations/tuning routes and will
print `OLDER BUILD` with `missing (405/404)` for those two paths. Consequences:

- the **cluster-wide operation lock is only real when the coordinator is this
  Mac** — a peer coordinator cannot answer `operations/active`, so it is not a
  lock;
- `/tune`, `/tune-redeploy`, `run-local` and `tune-local` must be driven **on this
  Mac**;
- `stage`/`plan` on the peers are enough for them to be workers.

To make the lock real everywhere, update every Mac to this build
([`PEER-VERSIONS.md`](PEER-VERSIONS.md)).

---

## 3. The cluster flow

Shell helpers (loopback; drop `AUTH` when the key check is skipped):

```sh
export BASE=http://127.0.0.1:8099
export KEY='<main API key>'                       # Settings → API key, this Mac
AUTH=(-H "Authorization: Bearer $KEY")
post() { curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' -X POST "$BASE$1" -d "$2"; }
get()  { curl -fsS "${AUTH[@]}" "$BASE$1"; }
```

Poll cadence: the UI polls every 1000 ms; any human-paced loop is fine. A
measurement is minutes on a large checkpoint, a start is minutes.

### 3.1 Measure the hardware only (Phase C3 entry)

No `model_path` and no `model_size_bytes` = **hardware-only**. `mtp` must stay
`false` here (a hardware-only measurement cannot reserve an MTP head — 400).

```sh
post /admin/api/cluster/assistant/measure '{
  "nodes": [
    {"node_id": "<this Mac node_id>", "ssh": "127.0.0.1"},
    {"node_id": "82b7d9cc-…", "ssh": "andreaskolb@172.16.1.13"}
  ],
  "roles": {"<this Mac node_id>": "workstation", "82b7d9cc-…": "workstation"}
}'
```

- `ssh` is `127.0.0.1` for this Mac and the paired `user@host` for a peer (take it
  from the Cluster tab pairing record; a bare `fe80::` address has no route, so
  never use one).
- `roles`: `workstation`, or **`headless` for a Mac under 64 GB installed RAM** —
  the workstation reserve is ≥32 GiB and would leave ~1 GiB of admission. The
  backend enforces the same rule (`workstation_reserve_too_large`).
- Answer: `{measurement_id, state:"running", poll}`. One measurement at a time; a
  second is `409 measurement_in_progress`.

Poll until `state == "ready"`:

```sh
get "/admin/api/cluster/assistant/status?measurement_id=$M1"
```

`state` is `queued → running → ready|failed`; `error.action` carries the one next
step on failure.

### 3.2 Recommend a model

```sh
post /admin/api/cluster/assistant/recommend '{
  "measurement_id": "'$M1'",
  "candidates": [
    {"model_path": "/Users/andreaskolb/models/GLM-5.3-Flash-oQ2e-mtp"},
    {"model_path": "/Users/andreaskolb/models/Qwen3.8-Flash-Next-oQ5e-mtp"}
  ],
  "context_tokens": 32768,
  "execution_profile": "balanced"
}'
```

- 1–32 candidates; `declared_context_tokens` and `friendly_name` are optional.
- Answer: `selected_model`, `selected_context_tokens`, per-candidate verdicts,
  `warnings`, and `single_node` — the **local-engine answer** when the model fits
  one Mac (E6), `null` otherwise.
- No feasible candidate is `409 no_feasible_candidate` **with the same body plus
  `code`/`action`/`message`**, so the near misses come with the refusal.

### 3.3 Measure the chosen model bound to it

`plan` keeps a strict node-set/model-identity/`mtp` match, so the recommended
model is measured **again**, bound to the model:

```sh
post /admin/api/cluster/assistant/measure '{
  "nodes": [ … the same nodes as 3.1 … ],
  "roles": { … },
  "model_path": "/Users/andreaskolb/models/GLM-5.3-Flash-oQ2e-mtp",
  "mtp": true
}'
```

`mtp` is `true` only when the catalogue says the checkpoint has an MTP head
(`mtp_compatible`); a model without one is planned with `mtp: false` and never
reserves a draft head. Poll `…/status` again → `$M2`.

### 3.4 Plan

```sh
post /admin/api/cluster/assistant/plan '{
  "measurement_id": "'$M2'",
  "model_path": "/Users/andreaskolb/models/GLM-5.3-Flash-oQ2e-mtp",
  "context_tokens": 32768,
  "rank0_extra_bytes": 0,
  "mtp": true,
  "execution_profile": "balanced",
  "microbatch_size": null,
  "node_selection": "minimal",
  "admission": "p50",
  "confirm_two_mac_fallback": false
}'
```

Read, do not assume:

| Field | What it tells you |
|---|---|
| `plan_signature` | the only thing `apply`/`start` will re-prove |
| `coordinator_node`, `rank_order` | who runs rank 0 and the order; the origin is **not** a preference |
| `selected_node_ids`, `excluded_nodes` | the fewest Macs that fit, and why the others were left out |
| `resource_policy` | context, `table_offload` (`engram`/`ple`), MoE offload + resident fraction, tier, admission |
| `tradeoffs` | when experts are offloaded: the `label: "speed"` alternative (smaller context, experts resident) with its **own** `resource_policy` + `plan_signature` |
| `predicted.measured` / `critical_stage_seconds` | measured compute or honestly unmeasured (`unmeasured_compute` warning) |
| `topology` | `line`, `measured`, `complete`, `missing_links` |
| `endpoint` | `null` until ready — never a port-0 URL |

No feasible plan is `409 no_feasible_plan` with the same body plus `code`/`action`.
`plan` is synchronous, pure and repeatable; nothing is staged.

### 3.5 Apply (stages; starts nothing)

Repeat the plan inputs **verbatim**, add the signature, the coordinator, the rank
order, and the approved `resource_policy` values:

```sh
post /admin/api/cluster/assistant/apply '{
  "measurement_id": "'$M2'",
  "plan_signature": "<plan.plan_signature>",
  "model_path": "/Users/andreaskolb/models/GLM-5.3-Flash-oQ2e-mtp",
  "context_tokens": 32768,
  "rank0_extra_bytes": 0,
  "mtp": true,
  "execution_profile": "balanced",
  "microbatch_size": null,
  "coordinator_node": "<plan.coordinator_node>",
  "rank_order": ["<…>"],
  "target_context_tokens": "<plan.resource_policy.target_context_tokens>",
  "table_offload": "<plan.resource_policy.table_offload>",
  "table_kind": "<plan.resource_policy.table_kind>",
  "moe_expert_offload": false,
  "moe_resident_fraction": 1.0,
  "memory_guard_tier": null,
  "admission": "p50",
  "node_selection": "minimal",
  "confirm_two_mac_fallback": false
}'
```

The body is `extra="forbid"`: an unknown field is a 422, not silently ignored. A
placement that changed since the plan is `409 plan_mismatch` with the one action
`plan` — re-plan, never edit the signature.

Answer: `{operation_id, state, poll}`. Keep that id — `start`, `stop`, `tune` and
`tuning-preset` all take it:

```sh
export OP=<the applied operation_id>
get /admin/api/cluster/assistant/operations/$OP        # queued → applying → applied
```

Progress is `progress.{phase,message,nodes_ready,nodes_total}` — render it, do not
invent a percentage. A second apply/start anywhere is
`409 operation_in_progress` + action `status` naming the running operation
(**only a lock when the coordinator can answer `/operations/active`**, see §2.2).

### 3.6 Start and wait for ready

```sh
post /admin/api/cluster/assistant/start '{
  "measurement_id": "'$M2'",
  "plan_signature": "<plan.plan_signature>",
  "coordinator_node": "<plan.coordinator_node>",
  "operation_id": "'$OP'"
}'
```

Never post `start` from an unapplied or failed operation. States:
`queued → starting → loading → ready|failed`. `endpoint.ready` stays **false**
until health **and** one smoke completion succeed; show `endpoint.url` only then,
and only together with `state == "ready"`.

`start` returns its **own** `operation_id` — that is the ready operation, so
re-point `$OP` at it; `tune`, `tune-redeploy`, `stop` and `session` all take the
start operation, not the apply:

```sh
export OP=<the start operation_id>
get /admin/api/cluster/assistant/operations/$OP        # starting → loading → ready
```

Verify the endpoint itself, not the claim:

```sh
curl -s <endpoint.url>/v1/models
curl -s <endpoint.url>/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<serving id>","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

The serving id is the catalogue id for that `model_path`; when the coordinator is
another Mac, its own API key and session apply there.

### 3.7 Optional: constellation search

`POST /admin/api/cluster/assistant/search` with
`{measurement_id, context_tokens, priority, max_count (1–4), mtp, execution_profile}`
returns a `job_id`; poll `GET …/search/{job_id}`. It plans, starts and measures
several constellations, keeps the best **measured** and stops the losers — minutes,
and it holds the one apply/start lock. Run it only if the plain flow is green.

---

## 4. Tuning the running cluster

Both need a **ready start operation**.

```sh
post /admin/api/cluster/assistant/tune '{"operation_id": "'$OP'"}'
```

Synchronous (real completions on the winning coordinator): sweeps the sampler and
the MTP switch, returns `sampler`, `mtp`, the rate rows and a `preset_key`, and
saves the preset. `chunk_applicable` is `false` in this build — do not promise a
prefill-chunk sweep.

```sh
post /admin/api/cluster/assistant/tune-redeploy '{"operation_id": "'$OP'"}'
get  /admin/api/cluster/assistant/tune-redeploy/<job_id>
```

A background job (`queued → running → ready|failed`, `done`/`total`, `rows`): it
**stops the running cluster**, deploys each MTP config (off / adaptive at a depth),
measures a real rate, and merges the winner into the same preset. Minutes to tens
of minutes. The preset stores the measured values and is replayed by the next
Start **before activation** (reported as `tuning_preset_applied` /
`tuning_preset_skipped`). One preset per (model, node config).

```sh
get "/admin/api/cluster/assistant/tuning-preset?operation_id=$OP"
```

reads what is stored for that model + node set (`{"preset": null}` means "nothing
tuned for this model on these Macs yet" — never an error).

---

## 5. run-local: Qwen3.8 on this Mac

The cluster apply refuses a single-Mac plan (`single_node_plan`, action `add_mac`)
— a pipeline needs 2+ hosts. A model that **fits one Mac** is served by the local
engine instead, which is also faster: measured 254 tok/s single-node vs 53 tok/s
for the 4-Mac cluster on dev-1.5b, and **Qwen3.8-Flash-Next-oQ5e-mtp loaded on
`.11` alone at 32k context with the PLE table on SSD: 40.3 tok/s, TTFT 0.34 s**
(2026-11-14).

Precondition: the recommendation's `single_node` names **this** Mac (§3.2). The
card only offers "Run on this Mac" then; do not drive it from a cluster answer.

```sh
post /admin/api/cluster/assistant/run-local '{
  "measurement_id": "'$M2'",
  "model_path": "/Users/andreaskolb/models/Qwen3.8-Flash-Next-oQ5e-mtp",
  "layer_count": <measured layer count>,
  "context_tokens": 32768,
  "mtp": true
}'
get  /admin/api/cluster/assistant/run-local/<job_id>       # queued → running → ready|failed
```

- It re-proves that the model fits **this** Mac at that context with the same
  single-`budget_for` rule, replays the tuning preset for
  `(model, hardware_key([this Mac]))`, then loads through this Mac's own engine
  pool and verifies the pool reports `loaded`. Loopback only, this Mac's key, no
  SSH, never another Mac, never a silently smaller context.
- `409 local_run_in_progress` / `409 local_tune_in_progress`: one at a time.
- A too-large context is `context_too_large` / `adjust_model_or_context`; nothing
  fits is `insufficient_memory` / `choose_smaller_model`.
- `ready` runs a one-shot readiness canary first; the payload carries `endpoint`.

Tune the serving model **on this Mac** (write settings, unload, load `local_only`,
one real completion per config), then the winner stays loaded and is saved as the
preset for `(model, hardware_key([this Mac]))`:

```sh
post /admin/api/cluster/assistant/tune-local '{"job_id": "<run-local job_id>"}'
get  /admin/api/cluster/assistant/tune-local/<tune_job_id>
```

Stop the local run:

```sh
post /admin/api/cluster/assistant/run-local/<job_id>/stop '{}'
```

---

## 6. Stop, and resume after a reload

```sh
post /admin/api/cluster/assistant/stop '{"operation_id": "'$OP'"}'
```

The operation is the only input: it carries the coordinator, port and deployment
ID, so a stale measurement cannot block a stop and nothing is re-elected. The
route relays to the coordinator's own
`POST /admin/api/cluster/deployments/{id}/unload`. Idempotent. `state` stays
`ready` (the frozen machine has no `stopped` state) and `stopped: true` is the
authoritative flag; the endpoint must disappear from the card.

After a page reload the card resumes from:

```sh
get /admin/api/cluster/assistant/session
```

newest ready start (else applied apply) with its frozen plan inputs, coordinator,
deployment ID and `stopped` — no key, SSH target or secret. `Start again` re-posts
apply/start with those inputs; a drifted placement is still `plan_mismatch`.

A stale **cross-coordinator** deployment is never auto-unloaded. The supported
cleanup is the manual unload on the owning coordinator
(`POST /admin/api/cluster/deployments/{id}/unload`) plus the `diagnostics` action a
memory-pressure start reports.

---

## 7. macOS app test and DMG

### 7.1 Xcode tests — the locale caveat

```sh
cd apps/omlx-mac
xcodebuild test -scheme oMLX CODE_SIGNING_ALLOWED=NO -testLanguage en
```

`-testLanguage en` is **required on this German-locale Mac** and matches CI. Without
it, `IntegrationsScreenVMTests` fails on the host locale — that is not a merge
defect. Expected: `TEST SUCCEEDED`, 0 failures.

### 7.2 DMG

```sh
apps/omlx-mac/Scripts/build.sh release --slim     # Swift app + omlx source, no heavy layers
apps/omlx-mac/Scripts/package_dmg.sh              # unsigned DMG for inspection
```

Signed/notarized (team `YB894QRG4W`, one standard admin dialog for the helper):

```sh
SIGN_IDENTITY="Developer ID Application: Andreas Kolb (YB894QRG4W)" \
NOTARY_PROFILE=omlx-notary \
apps/omlx-mac/Scripts/package_dmg.sh
```

`package_dmg.sh` signs the helper first, then the app, then the `.dmg` (an unsigned
container cannot be assessed by Gatekeeper even after the app inside is signed).
Artifacts land in `build/`. Details: [`DISTRIBUTION.md`](DISTRIBUTION.md),
[`SIGNING.md`](SIGNING.md), [`SLIM-RUNTIME.md`](SLIM-RUNTIME.md),
[`HELPER.md`](HELPER.md).

### 7.3 Web assets, only if a template changed

```sh
cd apps/omlx-web && python build_css.py
```

Node for the web suite on this Mac:
`/Users/andreaskolb/.cache/uv/archive-v0/cjy55nIyXhJQdXu/nodejs/bin` (on `PATH`).

---

## 8. Known caveats for this run

1. **Peer build skew** (§2.2): peers answer `stage`/`plan` but not
   `assistant/operations/active` or `tuning-preset`. The cluster-wide lock and the
   tuning/run-local routes are only real **on this Mac**. Elect this Mac as
   coordinator for the run that must prove the lock.
2. **Coordinator-side MoE offload is not re-threaded** (as of this branch's
   `HEAD`). The planner prices MoE
   expert offload again (assistant path), but the coordinator's own `/plan` call
   sites (`omlx/cluster/routes.py` `plan_unequal_pipeline` / `plan_hybrid` /
   `_build_performance_plan`) and `deployment._assignment_from_dict` do not
   pass/decode `moe_offload_*`. A DeepSeek-MoE plan that *depends* on expert
   offload can be refused (`placement_recomputed`) or over-charged at apply. The
   engine still offloads at load. **Stay on GLM-5.3 (non-MoE-expert-offload) and
   Qwen3.8 VLM for this run**; the context-vs-experts `tradeoffs` can be read from
   the plan, just do not apply an experts-offloaded DeepSeek plan. If the tree is
   dirty with the MoE threading, check `git status`/`git log` first — that work
   would close this caveat, and the run should say which tree it ran.
3. **Xcode tests need `-testLanguage en`** (§7.1).
4. **Dimension mismatches are refusals, not fallbacks**: `plan_mismatch`,
   `placement_changed`, `node_not_paired`/`pair_mac`,
   `coordinator_unresolved`/`configure_endpoint`,
   `insufficient_disk`/`free_disk_space`, `topology_incomplete` (with the 2-Mac
   fallback offered only after explicit consent) each carry exactly one action.
   Never retry by starting this Mac as a substitute.

## 9. What to write down

Per scenario: measurement ID, plan signature, operation ID, coordinator node,
`rank_order`, `resource_policy`, `predicted.critical_stage_seconds` (or that it is
unmeasured), endpoint URL, the time from `start` to `ready`, `observed` if the
performance probe is on (`assistant_performance_probe`, default **off**), and
every `error.code` + `error.action` seen. Append to
[`PHASE-2B-RUN.md`](PHASE-2B-RUN.md).