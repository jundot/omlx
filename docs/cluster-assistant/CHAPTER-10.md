# Chapter 10 — DAU acceptance, diagnostics and the live proof

Status: **offline work done; the live block is the only open item (2026-11-16)**.
The Chapter-10 gate is closed offline; the only open gate item is live evidence —
the U1–U6 acceptance run A1–A7 on a wiped, identical-build cluster. This
repository had no earlier Chapter-10 specification; this document defines the
chapter from `PROJECT-BRIEF.md` Phase 4, §16.4 and §17.1, plus the recorded
decision that P2 stays after Chapter 10.

## 1. Scope

Chapter 10 makes the existing assistant acceptable for a non-technical user and
records the evidence. It does **not** add a second planner, app, wizard, chat or
tab. P0/P1 work stays in the existing Cluster tab and the existing oMLX admin
API.

P2 was opened by explicit decision on 2026-11-07 for the **macOS menu/tray
entry only**. The helper, telemetry, tensor parallelism and >4 Macs remain out
until their decisions below are answered.

## 2. What is already done

- Election, measurement, recommendation, plan, apply/start, Screen 3–6.
- Reload resume, the cluster-wide operation lock and stop/start-again.
- Hardware-only measurement and the measure-again-before-plan rule.
- Source-first `copy_model`, the last-copy guard and the free-space relay.
- The resource-policy solver, the compute probe and the post-ready
  `observed` block.
- The measured line topology and the consented 2-Mac fallback.
- The three offline DAU pieces: one recommended model, progress/error states and
  the English-only card.

## 3. Remaining offline work

### 3.1 Redacted support bundle — implemented 2026-11-07

`GET /admin/api/cluster/assistant/diagnostics?measurement_id=…&operation_id=…`
returns an allow-listed, read-only bundle built by
`omlx/cluster/assistant/diagnostics.py`:

- `app`: oMLX/MLX/Python/macOS versions;
- `measurement`: state, timestamp, model label, layer/weight totals, measured
  fabric, per-Mac capacity/admission/memory tier/link/disk/endpoint, and model
  labels with sizes;
- `operation`: kind, state, progress, coordinator, rank order, signatures,
  deployment, endpoint, observed rates, warnings and error;
- `plan`: the approved plan identity and resource policy;
- `topology`: measured line, links and missing pairs for the operation's rank
  order;
- `health`: operation state, endpoint readiness, deployment, stopped;
- `events`: measurement/operation warnings and the error;
- `redactions`: the explicit list of what was removed.

It never re-elects, probes, stages or launches. Unknown IDs return an app-only
bundle, not a 404, so the card can still copy diagnostics after a record was
pruned. The card fetches this bundle first and keeps its local text block as the
offline fallback.

Redaction is structural plus textual: only allow-listed fields are copied; model
paths become folder labels; free-text warnings lose SSH users, IPv4 addresses,
absolute paths and credential-shaped strings. No key, token, prompt or
completion is read.

### 3.2 Acceptance audit

The static/UI tests pin what can be checked without a cluster:

- the primary card has no IP/SSH/rank/layer/model-path input,
- the failure panel renders exactly one action from `error.action`,
- the English catalog is the only copy,
- no TP control exists,
- the endpoint stays hidden until `ready`,
- diagnostics contain no secret or full model path.

The live-only items are in §5.

### 3.3 Live runbook

Run the matrix in §5 as one block when the cluster is reachable. Record the
measurement, plan and operation IDs in
`docs/cluster-assistant/PHASE-2B-RUN.md`; keep the working results in the
unversioned verification note until the block is accepted.

### 3.4 Release checklist

- all local suites green and `ruff` clean on touched files,
- no key, cookie, SSH credential or model content in a file, fixture or log,
- P2 remains untouched,
- no push until the live block passes,
- `AGENTS.md` §10 and `PROJECT-BRIEF.md` §22 agree with the code.

## 4. Support-bundle example

```json
{
  "generated_at": "2026-11-07T20:00:00+00:00",
  "app": {"omlx": "…", "mlx": "…", "python": "3.12…", "macos": "…"},
  "measurement": {
    "measurement_id": "ms-…",
    "model": "qwen",
    "fabric_measured": true,
    "fabric_links": [["m1", "m5"]],
    "nodes": [{"node_id": "m5", "capacity_gib": 107.5, "models": [{"label": "qwen", "size_gib": 41.0}]}]
  },
  "operation": {"operation_id": "op-…", "state": "applied", "rank_order": ["m5", "m1"]},
  "topology": {"kind": "line", "measured": true, "complete": true, "missing_links": []},
  "health": {"state": "applied", "endpoint_ready": false},
  "events": [],
  "redactions": ["no API keys, tokens or passwords"]
}
```

The example is illustrative; no live value is committed.

## 5. Acceptance matrix

### 5.1 Offline-proven (tests)

- App origin does not determine rank 0; an infeasible origin is replaced.
- 2/3/4-Mac contiguous PP plans; rank 0 holds the tail; MTP is charged.
- The smallest runnable subset wins; excluded Macs are named.
- A 3/4-Mac plan needs a connected line; a missing pair is
  `topology_incomplete`; the 2-Mac fallback needs explicit consent and salts the
  signature.
- Source-first staging, missing-source refusal, last-copy refusal.
- Hardware-only measurement is refused by `plan`.
- One operation at a time; stop/start-again; reload resume.
- English-only copy, no TP UI, one action per failure, hidden endpoint.
- Diagnostics redaction.

### 5.2 Local single-node smoke (no cluster)

The local half of the assistant can be exercised against this Mac alone. It
proves measurement, recommendation, model-bound measurement, the one-Mac plan
and the apply refusal without staging or launching anything:

1. Run oMLX locally (port 8099) and use its admin API key read from the local
   settings; never print or copy the key.
2. `POST /assistant/measure` with one node:
   `{"nodes":[{"node_id":"local","ssh":"127.0.0.1"}],"roles":{"local":"headless"}}`.
   Poll `/assistant/status` until ready. This is a hardware-only measurement.
3. `POST /assistant/recommend` over local models under `~/Models`. A small
   pipeline-capable model is feasible; a model too large for the local
   admission is refused with `choose_smaller_model`.
4. Measure the recommended model again (model-bound), then `POST /assistant/plan`
   with one node. The plan is feasible with `rank_order: ["local"]`.
5. `POST /assistant/apply` with that plan must return `single_node_plan` and the
   one action `add_mac`; it must not stage, launch or invent a local-engine mode.
6. Build or fetch the support bundle for the real measurement/operation and
   check that the full model path, API key and SSH target are absent.

The current code accepts the browser's `node_selection` on apply and then
returns `single_node_plan`; an older running server can still answer `422` for
that field until it is restarted. Restarting the user's server is a separate,
explicit action.

This smoke is **not** a cluster test: it does not form a 2-Mac pipeline, stage a
model or start a rank. The live block in §5.3 remains the only proof of the
distributed path.

### 5.3 Live-only (deferred block)

1. 2-Mac baseline: measure → recommend → plan → apply → start → chat; endpoint
   ready only after health/smoke; probe off leaves `observed` null.
2. 3-Mac line: a model that does not fit two; minimal plan chooses three;
   apply/start, six progress rows, Stop, Start again.
3. 4-Mac line: `node_selection=all`, four ranks; a second origin gets
   `409 operation_in_progress`.
4. Any-to-any: originate from each Mac; the same elected coordinator/plan; no
   local fallback.
5. Copy flow: model only on one Mac; elected coordinator lacks it; apply stages
   from the holder; delete a duplicate (allowed), the last copy (refused) and
   the planned model (refused).
6. Dev model `dev-v41-15L` stage fragments: full-copy path, disk guard; never
   delete the last copy.
7. Probe on/off: next-plan compute profiles; post-ready prefill/decode/TTFT;
   in-app toggle and one Refresh.
8. Reload resume and the stop lock.
9. Error paths: coordinator offline, unpaired Mac, no disk, no source.
10. Incomplete line: pull one cable on a 4-Mac plan; `topology_incomplete` names
    the pair; “Run with 2 Macs” is refused until confirmed; the accepted plan
    signature records the consent.

## 6. P2 decisions (2026-11-07)

- **Start:** P2 starts with the macOS menu/tray entry; the helper follows.
- **Entry:** one menu item opens the **existing** Cluster tab in the existing
  web dashboard. No new window, app, wizard, chat or control plane; no
  status/stop in the tray in this slice.
- **Helper:** Developer ID signing plus network setup is the chosen scope. The
  signed SMJobBless helper is **implemented and live-blessed** on all four Macs
  (D6, 2026-11-15): one admin dialog, `PrivilegedHelper.status →
  {installed, version, wiredLimitCurrentMB, wiredLimitRequestedMB}` as the one
  truth, a **System helper** step in Welcome and a permanent Install/Reinstall
  row in Performance independent of the wired-limit warning. The **privileged
  network setup** (bridge/STP) remains open; it is deferred by decision.
- **Telemetry:** remains none. **TP, >4 Macs, HA, multi-tenancy:** remain out.

The menu item, the Welcome button and the deep link are implemented, and the
macOS app builds and tests green (2026-11-16: `xcodebuild test -scheme oMLX
CODE_SIGNING_ALLOWED=NO -testLanguage en` → 441 passed / 0 failed, 5 skipped),
including the D1 in-app uninstall (Settings → Advanced → Uninstall… with one
keep switch per scope). The read-only network preflight is implemented as
`GET /assistant/network-preflight` and a card panel; it shows exact commands and
executes nothing. Distribution is a direct **slim** DMG (79 MB, external Python
runtime) rather than the 1.6 GB embedded app; the 2026-11-16 artefact is
Developer-ID signed, notarized and stapled; see [`HELPER.md`](HELPER.md),
[`SLIM-RUNTIME.md`](SLIM-RUNTIME.md), [`DISTRIBUTION.md`](DISTRIBUTION.md) and
[`SIGNING.md`](SIGNING.md).

## 7. Exit criteria

- Offline: §3.1–3.4 done; suites green; docs reconciled.
- Live: §5.3 passes on 2–4 Apple Silicon Macs over the validation line; evidence
  in `PHASE-2B-RUN.md`. The U1–U6 acceptance gate (A1–A7 in
  [`ROADMAP-ACCEPTANCE-2026-11-16.md`](ROADMAP-ACCEPTANCE-2026-11-16.md) §2) is
  the remaining block and runs from a wiped four-Mac state with one identical
  build.
- P2 app/tray slice: source, catalog and Xcode tests in place and green.
- P2 helper: implemented (signed, one admin dialog, live-blessed); the
  privileged network apply is the only open P2 privilege item.

The next agent should start with the live block, not with a new feature. A
failure in the block is fixed in the existing assistant surface; it is not a
reason to add a second planner, app, wizard, chat or tab.