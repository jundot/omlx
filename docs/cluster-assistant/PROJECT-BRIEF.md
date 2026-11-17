# oMLX Cluster Assistant — Project Brief

| Field | Value |
|---|---|
| Status | Draft v0.1 |
| Date | 2026-11-03 |
| Purpose | Starting point for a new project. This file is intentionally self-contained and can be copied into the new repository. |
| Working title | oMLX Cluster Assistant |
| Product UI language | English only. No German localization in the MVP. |

---

## Contents

- [0. Executive summary](#0-executive-summary)
- [1. Why this project exists](#1-why-this-project-exists)
- [2. Product vision](#2-product-vision)
- [3. Product principles](#3-product-principles)
- [4. Scope](#4-scope)
- [5. User journey](#5-user-journey)
- [6. Architecture](#6-architecture)
- [7. Coordinator election (rank 0)](#7-coordinator-election-rank-0)
- [8. PP planning](#8-pp-planning)
- [9. Any-to-any install and start](#9-any-to-any-install-and-start)
- [10. Discovery, pairing, and network](#10-discovery-pairing-and-network)
- [11. Model catalogue and recommendation](#11-model-catalogue-and-recommendation)
- [12. UI/UX specification](#12-uiux-specification)
- [13. API and data contracts](#13-api-and-data-contracts)
- [14. Security and privacy](#14-security-and-privacy)
- [15. Observability and support](#15-observability-and-support)
- [16. Testing strategy](#16-testing-strategy)
- [17. Acceptance criteria](#17-acceptance-criteria)
- [18. Milestones](#18-milestones)
- [Chapter 5 — Apply / Start (implementation brief)](CHAPTER-5.md)
- [19. Proposed project structure](#19-proposed-project-structure)
- [20. Migration from LocalInferenz](#20-migration-from-localinferenz)
- [21. Risks and mitigations](#21-risks-and-mitigations)
- [22. Open decisions](#22-open-decisions)
- [23. Definition of done for the starting project](#23-definition-of-done-for-the-starting-project)
- [24. Glossary](#24-glossary)
- [Appendix A — Cold-start handoff for the next agent](#appendix-a--cold-start-handoff-for-the-next-agent)

---

## 0. Executive summary

Build a DAU-friendly cluster experience **inside oMLX**, not as a second application and not as a second control plane.

The user connects 2–4 Macs with Thunderbolt cables, opens the oMLX app, presses **Set up cluster**, and the system does the rest:

1. discovers the Macs,
2. pairs them,
3. installs/verifies the oMLX runtime on every node,
4. stages the model,
5. measures memory and performance,
6. elects the coordinator/rank 0,
7. plans a pipeline-parallel split,
8. starts the coordinator on the elected Mac,
9. verifies the endpoint with a real completion,
10. opens Chat or hands the endpoint to an agent.

The user never types an IP address, SSH command, model path, layer count, rank, or endpoint. The app origin is irrelevant: the coordinator/rank 0/endpoint may be on a different Mac than the one where the app was opened.

**MVP scope:** pipeline parallelism (PP) only, 2–4 Apple Silicon Macs over Thunderbolt, English UI, chat/agent smoke test. Tensor parallelism (TP) is explicitly out of scope and comes later.

**Non-goal:** rebuilding exo, or maintaining a parallel LocalInferenz control plane beside oMLX.

---

## 1. Why this project exists

The current LocalInferenz prototype proved several things:

- A measured planner is useful.
- A simple five-step setup is possible.
- Automatic coordinator placement is necessary.
- Remote start of a coordinator works.
- The DAU experience must hide ranks, layers, IPs, SSH keys, and model paths.

It also exposed a strategic problem:

> oMLX already contains a Cluster v2 dashboard, device discovery, pairing, plan approval, model staging, activation, chat, macOS app, and client integrations.

Therefore the new project must not duplicate those surfaces. It must extend the existing oMLX product with the missing pieces:

- automatic coordinator election,
- measured admission as a first-class planning input,
- any-to-any installation and remote start,
- a truly simple DAU mode,
- English-only product copy,
- PP-first behavior.

The old LocalInferenz app is a prototype and reference. It is not the target architecture.

---

## 2. Product vision

> **Plug in 2–4 Macs, open the app, choose nothing complicated, and get a working local AI endpoint.**

The cluster must be invisible until the user opens details.

The user should experience:

- one app,
- one cable/ring,
- one primary button,
- one recommended model,
- one endpoint,
- one chat,
- one agent integration.

The system should decide:

- which Mac is coordinator,
- which Mac is rank 0,
- which Mac holds which pipeline stage,
- how many layers each stage gets,
- which context/chunk/model variant fits,
- whether the transport is ring or RDMA,
- where the model is copied from,
- when the endpoint is ready.

---

## 3. Product principles

1. **oMLX owns the runtime and cluster lifecycle.**
   The assistant is a feature module, not a separate deployment engine.

2. **The app origin is irrelevant.**
   Starting the app on the smallest Mac must not make that Mac rank 0.

3. **Coordinator, rank 0, and endpoint are co-located in the MVP.**
   The elected coordinator runs the supervisor; rank 0 is local on that Mac; the endpoint is that Mac.

4. **Feasibility before performance.**
   If only one rank-0 candidate produces a legal split, it wins—even if it is not the obvious performance choice.

5. **Free memory is a placement signal, not a hard “largest number wins” rule.**
   Measured admission is a constraint and a strong preference. The final choice is the best feasible joint plan.

6. **PP only in the MVP.**
   No tensor-parallel UI, no TP placement, no TP promises. Design extension points, but do not ship TP behavior.

7. **No terminal in the primary path.**
   No IP, SSH, model path, layer cut, rank, JACCL, bridge, or wired-limit fields.

8. **One recommended model by default.**
   A model shelf is advanced; the primary path recommends one.

9. **Every error has exactly one actionable next step.**
   Raw logs and diagnostic details are behind “Details”.

10. **No second app.**
    The oMLX macOS app is the shell. The web admin panel is the UI. The assistant is a mode/page inside it.

11. **English only.**
    No German strings, no German locale work in the MVP.

12. **Do not rebuild what oMLX already has.**
    Discovery, pairing, staging, activation, chat, and integrations are reused.

---

## 4. Scope

### 4.1 In scope

- 2–4 Apple Silicon Macs.
- Thunderbolt direct connections; 3/4-node ring topology.
- Pipeline parallel inference.
- Automatic discovery and pairing.
- Automatic runtime installation/verification on all nodes.
- Automatic model staging from any complete source.
- Measured admission and performance profiles.
- Coordinator election (rank 0).
- Remote start of the coordinator from any origin.
- Endpoint on the coordinator.
- Chat smoke test and agent handoff.
- English-only simple UI.
- Recovery and diagnostics.
- Signed/notarized app/helper path (productization requirement, not necessarily first code milestone).

### 4.2 Out of scope for the MVP

- Tensor parallelism.
- Linux, Windows, CUDA, or non-Apple hardware.
- More than 4 Macs.
- General-purpose cluster federation.
- Training or fine-tuning.
- A second cluster runtime.
- A second discovery/pairing implementation.
- A second chat UI.
- A separate OpenCode/Codex client helper.
- German localization.
- High-availability coordinator failover.
- Public internet exposure of the inference endpoint.
- Automatic model fine-tuning or arbitrary remote code execution.

### 4.3 Later, only after PP is solid

- Tensor parallelism.
- Dynamic rebalancing at runtime.
- Coordinator failover/election.
- More than 4 Macs.
- Mixed Linux/CUDA nodes.
- Remote/cloud burst.
- Multi-user tenancy.

---

## 5. User journey

### Screen 1 — Welcome / Start

The user opens the oMLX app on any Mac.

Possible states:

- **Single Mac ready:** “This Mac can run a model alone.”
- **Connect Macs:** “Add another Mac to run bigger models.”
- **Existing cluster:** “Your cluster is running.”

Primary action:

- **Set up cluster** (or **Add a Mac**)

The screen must not ask for an IP or a role.

### Screen 2 — Connect Macs

The app shows:

- this Mac,
- discovered Macs with friendly names,
- cable/topology status,
- a simple chassis-aware illustration when a cable is missing.

Actions:

- **Connect** a discovered Mac.
- **Show pairing code** / **Enter code** if mDNS is blocked.
- **Details** for Thunderbolt ports, addresses, and diagnostics.

No “rank 0” selector. No “coordinator” selector. The system elects it.

### Screen 3 — Plan

The assistant measures and plans.

User sees:

- recommended model,
- number of Macs,
- context,
- estimated speed/quality,
- **Coordinator: Mac Studio (M5 Max, 128 GB)**,
- **Endpoint will be: 172.16.1.11:8099**,
- one primary action: **Set up and start**.

Details expand to show:

- stage table,
- layers per rank,
- memory/headroom,
- why the coordinator was chosen,
- model size/download/staging status,
- transport (ring/RDMA).

The primary path must never show raw rank/layer fields.

### Screen 4 — Progress

A single progress surface:

1. Checking Macs
2. Preparing runtime
3. Copying model
4. Planning
5. Starting coordinator
6. Loading model
7. Verifying endpoint
8. Ready

Each phase has a plain-language status and one retry action if it fails.

### Screen 5 — Ready

Shows:

- **Chat** button,
- endpoint,
- model,
- **Connect agent** button,
- “Open details” for diagnostics.

The screen is only shown after:

- supervisor is healthy,
- model is loaded,
- a real completion succeeded.

**Handoff (implemented 2026-11-07; `SCREEN-5-HANDOFF.md`).**

- `Open Chat` opens the **existing** `/chat` with the cluster’s serving model
  id: `/chat?model=<id>` when the coordinator is this Mac,
  `<endpoint.url>/chat?model=<id>` when it is another Mac. For a remote
  coordinator the browser talks to that endpoint directly, because the origin
  does not have that Mac’s key and must not get it; that Mac’s own page injects
  its own key into its own `localStorage`, and its admin session applies (a
  cookie is per host, so the user may sign in there once). The card says so in
  one English sentence.
- The serving id is resolved from the catalogue’s `model_path` → `id`. An
  unresolved id links without the `model` parameter; no id is invented.
- `Connect agent` shows `<endpoint>/v1`, the serving model id, and where the key
  lives on the coordinator (“Settings → API key”). It never shows the key.
- The chat page reads `model` once on load, selects a known id, leaves its
  normal default for an unknown one, adds no history entry and reads no
  credential. There is no `endpoint` parameter.
- A measured operation may carry one nullable `observed` block
  (`prefill_tok_per_s`, `decode_tok_per_s`, `ttft_ms`, `measured_at`), filled
  only **after** `ready` by one real completion on the elected coordinator.
  `predicted` is never rewritten; with the probe off (the default) `observed` is
  `null` and the card says nothing about speed.

### Screen 6 — Running / Manage

When the user returns:

- “Your cluster is running.”
- **Open chat**
- **Open agent**
- **Stop**
- **Details**

If the origin Mac is not the coordinator, this is fine. The UI connects to the coordinator endpoint.

**Implemented 2026-11-07 (`SCREEN-6-MANAGE.md`): state + stop/restart only.**

- Stop is `POST /admin/api/cluster/assistant/stop` with the ready start
  operation's ID; the backend relays to the elected coordinator's existing
  `POST /admin/api/cluster/deployments/{id}/unload` through the same local/SSH
  seam, using the deployment ID, coordinator and port recorded on the operation.
- The operation gains `deployment_id`, `coordinator_local` and `stopped`; its
  `state` stays `ready` (the frozen machine has no stopped value).
- A stopped card hides the endpoint and offers **Start again**, which re-applies
  the frozen plan inputs and starts. Re-planning, editing the model or
  context of a running cluster, and adding/removing a Mac are out of scope.
- After a reload the card **resumes** (Phase C1, recorded in
  `ROADMAP-TO-CHAPTER-10.md` §6 C1): `GET /assistant/session` returns the newest
  ready start (else applied apply) with its frozen plan inputs, coordinator,
  deployment ID and stopped flag — no key, SSH target or secret. The session
  strip shows Running/Stopped and offers Start again/Stop; apply re-proves the
  plan signature (`plan_mismatch` on drift). The existing deployments list
  remains a return surface, not the only one.

---

## 6. Architecture

```text
┌────────────────────────────────────────────────────────────────┐
│ oMLX macOS app                                                 │
│  - menu bar / supervisor bootstrap                             │
│  - Welcome / Cluster Assistant entry                           │
│  - opens admin panel or embedded web view                      │
└──────────────────────────────┬─────────────────────────────────┘
                               │ local HTTP
┌──────────────────────────────▼─────────────────────────────────┐
│ oMLX server / admin panel                                      │
│                                                                │
│  Existing oMLX cluster v2                                      │
│    - discovery / pairing                                       │
│    - device registry                                           │
│    - model catalogue / staging                                 │
│    - plan approval / deployment registry                       │
│    - runtime / telemetry                                       │
│                                                                │
│  New: Cluster Assistant module                                 │
│    - measured admission                                        │
│    - coordinator election                                      │
│    - PP-focused recommendation                                 │
│    - any-to-any start orchestration                            │
│    - DAU simple mode                                           │
└──────────────────────────────┬─────────────────────────────────┘
                               │ SSH / agent / existing oMLX launch
┌──────────────────────────────▼─────────────────────────────────┐
│ Elected coordinator Mac (rank 0)                               │
│  - oMLX supervisor                                             │
│  - HTTP endpoint                                               │
│  - rank 0 pipeline stage (tail / LM head / optional MTP head)  │
└──────────────────────────────┬─────────────────────────────────┘
                               │ pipeline / ring / optional RDMA
┌──────────────────────────────▼─────────────────────────────────┐
│ Worker Macs (rank 1..N)                                        │
│  - oMLX rank workers                                           │
│  - pipeline stages                                             │
└────────────────────────────────────────────────────────────────┘
```

### 6.1 Layer boundaries

| Layer | Owns |
|---|---|
| oMLX runtime | Inference, rank workers, pipeline transport, model loading, health, API |
| oMLX cluster v2 | Discovery, pairing, device registry, staging, deployment registry, existing wizard |
| Cluster Assistant | Measurement, coordinator election, PP recommendation, any-to-any orchestration, simple UI |
| macOS app/helper | App lifecycle, privileged network setup, login item, updates |
| Client integrations | Chat, OpenCode/Codex/Claude/etc. endpoint handoff |

**Rule:** if a feature is inference/runtime, it belongs to oMLX. If it is macOS/network/privilege, it belongs to the app/helper. If it is a decision or user guidance, it belongs to the Assistant.

---

## 7. Coordinator election (rank 0)

This is the core new feature.

### 7.1 Current behavior to replace

The current LocalInferenz prototype is local-first:

- ad-hoc `setup`/`deploy`: the Mac running the command becomes rank 0;
- saved profiles: `profile.nodes[0]` is rank 0;
- the layer split gets a fixed rank 0 and only permutes peers;
- rank 0 is always coordinator and endpoint.

This is not acceptable for the new product.

### 7.2 Target behavior

The assistant evaluates **every Mac as a possible rank-0 candidate**.

Rank 0 is also:

- coordinator,
- endpoint owner,
- supervisor host.

The app origin has no influence on this choice.

### 7.3 Inputs

For each node:

- measured admission:
  - p05 conservative available bytes,
  - p50/p95 for confidence,
  - reserve/role/workstation state,
- total capacity and free pressure,
- model present or downloadable,
- disk free,
- measured compute performance:
  - prefill weight bytes/s,
  - decode weight bytes/s,
  - optionally measured tokens/s,
- memory bandwidth if available,
- link matrix:
  - Thunderbolt/RDMA/TCP,
  - latency/bandwidth,
- endpoint host/port from that node's settings,
- node health/SSH/runtime parity,
- stage constraints:
  - fixed weights,
  - layer weights,
  - KV owner boundaries,
  - tail/head,
  - optional MTP head,
  - HTTP/coordinator overhead.

### 7.4 Algorithm

```text
candidates = all paired, healthy Macs with the model available/stageable

feasible = []
for candidate in candidates:
    order = [candidate] + order_rest_by_measured_rank(order_rest)
    for permutation in permutations(order[1:]):
        for legal_layer_cuts in enumerate_legal_cuts(model):
            plan = build_pp_plan(
                rank_order=[candidate] + permutation,
                cuts=legal_layer_cuts,
                admissions=measured_admissions,
                reserves=roles,
                rank0_extra=lm_head + mtp_head + http_overhead,
                context=context,
                chunk=chunk,
            )
            if plan.valid:
                feasible.append(plan)

if not feasible:
    return no_legal_plan(...)

best = choose_best(feasible)
return {
    coordinator_node: best.rank0,
    rank_order: best.order,
    endpoint: "http://<best.rank0>:8099",
    plan: best,
}
```

For 2–4 Macs this is cheap: at most 4 rank-0 candidates × 6 peer permutations × legal cuts.

### 7.5 Scoring

Hard order:

1. **Feasibility**
   - every stage fits admission,
   - rank 0 fits tail + LM head + optional MTP + coordinator overhead,
   - all legal PP boundaries respected.

2. **Performance**
   - minimize the predicted slowest pipeline stage,
   - maximized predicted prefill/decode throughput,
   - use measured compute/link profiles where available.

3. **Headroom**
   - maximize the minimum stage headroom,
   - then rank 0 headroom.

4. **Coordinator suitability**
   - reachable endpoint,
   - stable network,
   - enough free admission for the role,
   - disk/model availability.

5. **Free admission preference**
   - if plans are otherwise equal, prefer the candidate with more measured free admission for rank 0.

**Product rule:** rank 0 prefers the Mac with the most measured free admission, but a purely free-memory sort is not allowed to override feasibility or a clearly better plan.

### 7.6 The “M1 rank 0 does not fit, M5 rank 0 does” case

This is a valid and important case.

Example:

- M1 Studio: 64 GB.
- M5 Max: 128 GB.
- The model’s tail + LM head needs 70 GB.
- M1 as rank 0: no legal split.
- M5 as rank 0: legal split; M1 takes a lighter head stage.

Result:

- Coordinator/rank 0/endpoint: **M5 Max**.
- M1: rank 1.
- App origin: irrelevant; can still be the M1.

The assistant must choose the M5 because it is the only feasible candidate, not because of a performance preference.

### 7.7 No local fallback

If the elected coordinator cannot be started:

- stop and show the exact action,
- do **not** silently start the local Mac as rank 0,
- do **not** silently choose a smaller model/context if another candidate plan was feasible,
- allow retry on the same or next candidate.

### 7.8 Persistence

The chosen election is persisted in the oMLX cluster deployment/assistant state:

- coordinator node ID,
- rank order,
- plan signature,
- model paths per node,
- endpoint,
- measured profile timestamp,
- model/context/chunk,
- reason summary.

A saved plan can be restarted only if its signature matches the current hardware/model.

### 7.9 Re-election

Re-elect when:

- model changes,
- context/chunk changes,
- a node joins/leaves,
- measured admission drifts beyond threshold,
- the coordinator becomes unreachable before start,
- the user explicitly asks to replan.

Re-election is not automatic while a healthy cluster is serving unless the user requests it.

---

## 8. PP planning

### 8.1 PP only

The MVP planner and UI expose only pipeline parallelism.

- Disable/hide tensor-parallel controls.
- Reject TP in the assistant API with a clear message.
- Keep the planner interface extensible for future TP.
- No mixed PP/TP in the MVP.

### 8.2 PP model constraints

- Only models with a valid pipeline contract.
- Exact layer boundaries, not a percentage formula.
- Respect KV owner boundaries.
- Respect fixed weights, embeddings, LM head, MTP head.
- Model-specific KV geometry.
- Engram/SSD tables where applicable.
- Stage peak memory, not just resident weights.
- Rank 0 gets the tail and its extra head/coordinator bytes.

### 8.3 Stage balance

Pipeline throughput is the slowest stage:

```text
throughput ≈ min(stage_time)
```

The planner must:

- assign layer counts proportional to measured compute/memory bandwidth,
- account for fixed per-stage bytes,
- account for rank-0 head/HTTP overhead,
- avoid making a small Mac rank 0 when its tail cannot fit,
- avoid making a slow Mac the critical stage if another assignment is feasible.

**Fewest Macs that fit (2026-11-07).** The planner tries subsets from 2 Macs
upward and uses the smallest feasible subset; it must not add a third or fourth
Mac merely because it is paired (three pipeline hops instead of one). The
comparison is: measured critical stage, then headroom, then fewer hops, then
the coordinator preference. Without measured compute it prefers headroom and
records `unmeasured_compute` rather than inventing a speed order. The
measurement still covers every paired Mac; plan/apply/start bind to the chosen
subset and the plan signature includes it. An advanced “use all Macs” option is
explicit, not the default. A single-Mac plan is still not applicable (a
`ClusterDeployment` needs at least two hosts).

### 8.4 Planning inputs

- measured admission p05/p50/p95,
- model layout,
- context,
- prefill chunk,
- reserve/role,
- performance profiles,
- link matrix,
- endpoint host/port per node (resolved by measurement, passed as facts),
- rank-0 extra bytes,
- optional MTP,
- SSD/Engram metadata.

### 8.5 Planner output

```json
{
  "feasible": true,
  "measurement_id": "ms-20261103-...",
  "coordinator_node": "studio-m5",
  "endpoint": {
    "node": "studio-m5",
    "host": "172.16.1.11",
    "port": 8099,
    "url": "http://172.16.1.11:8099",
    "ready": false
  },
  "rank_order": ["studio-m5", "studio-m1"],
  "model": "Qwen3.8-Flash-Next-oQ5e-mtp",
  "context": 32768,
  "chunk": 2048,
  "groups": [24, 24],
  "stages": [
    {
      "rank": 1,
      "node": "studio-m1",
      "first_layer": 0,
      "last_layer": 24,
      "resident_gib": 48.9,
      "headroom_gib": 9.0,
      "critical": false
    },
    {
      "rank": 0,
      "node": "studio-m5",
      "first_layer": 24,
      "last_layer": 48,
      "resident_gib": 50.5,
      "headroom_gib": 13.2,
      "critical": true
    }
  ],
  "predicted": {
    "measured": true,
    "critical_stage_rank": 1,
    "critical_stage_seconds": 0.42,
    "stages": [
      {"rank": 1, "seconds": 0.42},
      {"rank": 0, "seconds": 0.38}
    ]
  },
  "observed": null,
  "reasons": [
    "M5 is the best feasible coordinator; it has the largest rank-0 headroom"
  ],
  "plan_signature": "…"
}
```

The exact schema should align with oMLX’s existing plan/deployment objects.

The `predicted` block deliberately contains **stage seconds only**. It must not
contain `prefill_tok_per_s`, `decode_tok_per_s`, or `ttft_ms` before those
numbers have been measured by a smoke request. Internal election scoring may
use a relative stage cost; the public plan must not present it as a token rate.

`predicted.measured` is false when the caller had no compute profile; the block
is then a worst-case estimate (or stage seconds are null), never a fake rate.

`observed` stays `null` until the smoke test has measured real values. After
the smoke test it may be filled with measured prefill/decode/TTFT and a
timestamp.

### 8.6 No legal plan

If no candidate fits:

- show which constraint failed,
- offer one action:
  - reduce context,
  - choose a smaller model,
  - free memory,
  - add another Mac,
  - switch transport (if relevant),
  - stop other workloads.
- Do not silently downgrade the model/context.

---

## 9. Any-to-any install and start

### 9.1 Roles

| Role | Meaning |
|---|---|
| **Origin** | Mac where the user opened the app |
| **Controller** | The app/UI driving setup |
| **Coordinator** | Elected rank 0; runs supervisor and endpoint |
| **Worker** | rank 1..N |
| **Source** | Any node with a complete model copy |

These roles are deliberately independent.

### 9.2 Flow

```text
Origin app
  → discover/pair all Macs
  → measure all Macs
  → elect coordinator
  → ensure oMLX runtime on every Mac (push from origin or download)
  → stage model from the fastest complete source
  → start supervisor on the coordinator (local or SSH/agent)
  → wait for health
  → load model
  → smoke completion
  → expose endpoint
  → open Chat / hand off to agent
```

### 9.3 Remote start

If the coordinator is not the origin:

- use the existing oMLX pairing/SSH keys,
- ensure the oMLX interpreter/console exists on the coordinator,
- start the supervisor detached on the coordinator,
- return the coordinator PID/state,
- never block the HTTP request on the long-running supervisor,
- no `watch` call in the request thread.

If the coordinator is the origin:

- start locally through the existing oMLX app/server path.

### 9.4 Endpoint

- Endpoint belongs to the coordinator.
- Bind to a reachable private interface (Thunderbolt bridge / LAN / Tailscale as configured), not localhost-only when remote clients are expected.
- API key auth is required.
- The origin may open the remote UI or use a local controller UI that talks to the coordinator.
- Do not proxy inference through the origin unless explicitly required for a client.
- Clients should connect directly to the coordinator endpoint.

### 9.5 Origin independence

- The origin can be the smallest Mac.
- The origin can shut down after start; the coordinator continues.
- The origin can reconnect later through discovery/endpoint.
- The coordinator choice is stored in the cluster state, not in the app process.

### 9.6 Model staging

- Model can come from any node with a complete checkpoint.
- Resume interrupted copies.
- Verify size/checksum.
- Check destination disk before copying.
- Do not assume the coordinator has the model.
- Do not assume the origin is the source.

---

## 10. Discovery, pairing, and network

### 10.1 Reuse oMLX

Use the existing oMLX cluster v2:

- Bonjour/mDNS discovery,
- pairing codes,
- device registry,
- SSH key management,
- host-key pinning,
- device removal,
- version mismatch reporting.

Do not build a second discovery/pairing system.

### 10.2 Pairing UX

- Show discovered Macs by name.
- One click to pair.
- Pairing code shown on the joining Mac, approved on the other.
- Manual “Add by code/IP” fallback only under Advanced.
- No SSH password prompts in the primary flow.
- No key files shown to the user.

### 10.3 Thunderbolt topology

- 2 Macs: one direct cable.
- 3 Macs: ring A–B, B–C, C–A.
- 4 Macs: ring A–B, B–C, C–D, D–A.
- No switch or dock.
- Chassis-aware port guidance.
- Detect occupied ports and wrong bridge membership.
- If 3/4 nodes are selected but the ring is incomplete, show the missing cable and offer “run with 2” only with explicit confirmation.

### 10.4 Transport

- Default: TCP ring; works on TB4/TB5.
- RDMA over Thunderbolt 5: optional, detected, never required.
- If RDMA is not enabled/available, fall back to ring automatically.
- Do not put Recovery-mode steps in the primary DAU flow.
- RDMA can be an Advanced “Turbo” path later.

### 10.5 Privileged helper

To be truly plug-and-play, the app needs a small signed/notarized helper for:

- Thunderbolt interface/network configuration,
- avoiding bridge conflicts,
- optional wired-memory limit,
- login start/stop if needed.

The user should see one macOS admin prompt, not a terminal.

The helper is part of the productization work, not a separate app.

---

## 11. Model catalogue and recommendation

### 11.1 Primary path

- Show one recommended model.
- Recommendation considers:
  - 2–4 Macs,
  - measured admission,
  - PP capability,
  - model size/quantization,
  - context,
  - expected speed/quality.
- Use friendly names, not raw Hugging Face IDs.
- Show download size and disk requirement before copying.

### 11.1a Recommendation vs. planning contract (resolved)

Recommendation and planning answer different questions:

- **Recommendation:** “Which model should we run on these Macs?”
- **Planning:** “Given this chosen model, how do we split it?”

The chosen contract is **Option A**:

- Recommendation reads only the **hardware part** of a stored measurement:
  admission, compute, links, endpoint, disk and model availability.
- Recommendation may evaluate **any candidate model** in the catalogue.
- Recommendation does not require the measurement’s model identity to match
  the candidate.
- `plan` keeps the strict Chapter 3 check: node set, model identity and `mtp`
  must match the measurement it uses.
- After the user chooses a model, the UI starts a **model-bound measurement**
  for that model and then calls `plan`.
- A hardware-only measurement (`model: null`) is allowed for recommendation;
  it cannot be used by `plan`.

UI order:

```text
measure(hardware-only)
  → recommend(measurement_id)
  → user chooses a model
  → measure(model-bound)
  → plan(measurement_id)
  → approve
  → apply/start
```

This keeps the plan’s safety property intact while allowing model selection
to happen before a model is known.

The contract is implemented in `omlx/cluster/assistant/recommendation.py` and
exposed by `POST /admin/api/cluster/assistant/recommend`:

- the request carries `measurement_id`, 1–32 `candidates` (`model_path`,
  optional `friendly_name`, optional `declared_context_tokens`), an optional
  target `context_tokens` and an `execution_profile`;
- freshness and the 15-minute TTL are the Chapter 3 rule
  (`state.measurement_refusal`); the measurement's model identity and MTP
  setting are **not** compared with the candidates. `plan` keeps that check;
- feasibility has one authority: `election.elect()` is asked per candidate, so
  admission, the rank-0 tail/head and the layer cut are never re-derived. The
  MVP is PP only, so a candidate mlx-lm cannot split into pipeline stages is
  refused with one action and never offered as tensor parallelism;
- the context is the largest ladder value at or below the target and the
  model's declared limit that still plans. No context is invented, and the
  ladder is `catalogue.py`'s, imported rather than copied;
- the answer names one `selected_model` at one `selected_context_tokens`, with
  the ordered candidate list, per-Mac staging bytes, headroom, per-stage
  seconds only when compute was measured, and exactly one action per refusal;
- no feasible candidate is `409 no_feasible_candidate` with the ranked near
  misses in the body; an unusable measurement is the same flat
  `409 stale_measurement` `plan` returns.

`measure` accepts a hardware-only measurement (`model: null`, Phase C3 in
`ROADMAP-TO-CHAPTER-10.md` §6): it is valid for recommendation and carries
`hardware_only: true`, and `plan` refuses it — a plan always needs the
model-bound measurement of the chosen model. Recommendation ignores the
identity a measurement does carry, which is what makes the field a formality
rather than a precondition.

### 11.2 Advanced

- Full model shelf.
- Quality floor.
- Context/chunk overrides.
- Model path/repo.
- PP-only filter.

### 11.3 Model metadata

- PP support.
- Exact layer count.
- Fixed/tail/head bytes.
- KV geometry.
- Engram/SSD table metadata.
- Context limit.
- License/source.

---

## 12. UI/UX specification

### 12.1 Language

- **English only.**
- No German strings.
- No new translation framework work in the MVP.
- Use the existing oMLX locale infrastructure only if required by the host UI.

### 12.2 Primary UI

Keep the primary flow to five states:

1. Welcome
2. Connect Macs
3. Plan
4. Progress
5. Ready

### 12.3 Hidden in Advanced

- IP addresses.
- SSH details.
- Ranks and layers.
- Headroom in GiB.
- Model repo/path.
- Chunk/context internals.
- RDMA/JACCL.
- Wired limit.
- Logs.

### 12.4 Error messages

| Problem | Message | Action |
|---|---|---|
| No cable | “No Thunderbolt cable detected.” | Show port illustration |
| Peer offline | “Mac Studio is not reachable.” | Retry / check cable |
| Version mismatch | “The Macs run different oMLX versions.” | Update all |
| Runtime missing | “oMLX is not ready on this Mac.” | Prepare automatically |
| Model missing | “The model is not on this Mac yet.” | Copy from source |
| Disk full | “Not enough disk space on Mac Studio.” | Free space / choose smaller model |
| No PP split | “These Macs cannot hold this model at 32k.” | Reduce context / smaller model / add Mac |
| rank 0 infeasible | “This Mac cannot hold the coordinator stage; using Mac Studio.” | Continue automatically |
| Coordinator start failed | “The coordinator did not start on Mac Studio.” | Retry / choose next candidate |
| Health timeout | “The cluster did not become ready.” | Retry / diagnostics |
| Auth failure | “The endpoint rejected the key.” | Repair key |

No raw stack traces in the primary path.

---

## 13. API and data contracts

### 13.1 Reuse existing oMLX endpoints

Use existing routes for:

- discovery/devices,
- pairing,
- peer probe,
- model catalogue,
- autoconfigure/plan,
- stage,
- deployments/activation,
- replan,
- diagnostics.

Do not create parallel deployment or discovery endpoints.

### 13.2 New assistant endpoints

Proposed:

```text
POST /admin/api/cluster/assistant/session
POST /admin/api/cluster/assistant/measure
POST /admin/api/cluster/assistant/recommend
POST /admin/api/cluster/assistant/plan
POST /admin/api/cluster/assistant/apply
POST /admin/api/cluster/assistant/start
GET  /admin/api/cluster/assistant/status
GET  /admin/api/cluster/assistant/diagnostics
```

Semantics:

- `session`: idempotent setup session; preferences.
- `measure`: start an asynchronous measurement job; returns a `measurement_id`
  and a job/status handle. It persists a `MeasurementSet` with per-node
  p05/p50/p95 admission, compute/link hints, model availability, endpoint
  host/port and a timestamp. It is read-only with respect to the cluster.
  The measurement layer reads the endpoint port from each node's settings;
  the pure `plan` call does not read settings itself. One job runs at a time;
  a second concurrent `measure` is refused (see §13.4).
  A hardware-only measurement (`model: null`) is allowed for recommendation;
  it carries no usable model identity and cannot be passed to `plan`.
- `recommend`: synchronous, pure recommendation over the hardware part of a
  ready measurement. It accepts a `measurement_id`, 1–32 candidate models
  (`model_path`, optional friendly name and declared context), an optional
  target `context_tokens` and an execution profile, and returns the ordered
  candidates with fit, PP support, the largest context that still plans, the
  per-Mac staging requirement and stage-second estimates. It may evaluate
  models whose identity differs from the measurement’s original model identity.
  It never starts a plan, download, stage or launch. No feasible candidate is
  `409 no_feasible_candidate` with the near misses and one action; a missing,
  stale or failed measurement is the same `409 stale_measurement` `plan`
  returns.
- `plan`: synchronous, pure planning call; **requires a model-bound
  `measurement_id`**. It must not probe the network or trigger measurement
  implicitly. A missing, stale, or mismatched measurement (node set, model
  identity or `mtp`) returns `409 stale_measurement` with one action.
- `apply`: stage/prepare on the elected coordinator; no inference start. The
  plan is not persisted, so `apply` takes the plan-time inputs again, re-elects
  from the stored measurement and refuses a `plan_signature` mismatch
  (`409 plan_mismatch`, one action `plan`). A single-Mac plan is
  `409 single_node_plan` with the one action `add_mac`, and an unpaired Mac is
  `409 node_not_paired` with `pair_mac` — never a guess and never a local
  fallback. Returns an operation handle.
- `start`: start the supervisor on the elected coordinator; returns an
  asynchronous operation. Only an applied plan may start (`409
  operation_not_applied`, one action `apply`), and a placement that changed
  since apply is refused (`placement_changed`, one action `plan`).
- `operations/{operation_id}`: one apply/start, with progress, the endpoint and
  one `error.action` on failure. One apply/start at a time; a second is
  `409 operation_in_progress` with the running `operation_id` and the action
  `status`.
- `status`: snapshot for SSE polling, including measurement/job state.
- `diagnostics`: redacted support bundle.

Remote start (recorded, Option A — see CHAPTER-5.md §2.1): the origin drives the
coordinator’s own admin routes over the existing SSH seam, using the key that
Mac already stores locally. Rank 0 stays local to the coordinator; the origin
never starts a rank, never stores a peer key, and never falls back to itself if
the coordinator refuses.

`endpoint.ready` (in the plan object and in an operation) is false until health
and one smoke completion have succeeded. A missing distributed performance
probe never blocks apply/start/ready and never becomes a rate.

All mutating routes are admin-authenticated.

The UI flow is therefore:

```text
measure(hardware) (job)
  → poll/stream until MeasurementSet is ready
  → recommend(measurement_id)
  → user chooses model
  → measure(model-bound) (job)
  → poll/stream until ready
  → plan(measurement_id)
  → review/approve
  → apply/start
```

The UI must not call `plan` first and hope it measures. Measurement and
planning are separate so a slow SSH/memory probe never looks like a hung plan.
Recommendation and planning are also separate so model selection never
weakens the plan’s strict model-identity check.

### 13.3 Plan object

Must include:

- `plan_signature`,
- `measurement_id`,
- `measurement_timestamp`,
- `coordinator_node`,
- `rank_order`,
- `endpoint`:
  - `node`,
  - `host`,
  - `port` (read from the target coordinator's settings; default 8099 only
    when settings are unavailable),
  - `url`,
  - `ready` (false until the supervisor/health check succeeds),
- `stages`,
- `feasible`,
- `reasons`,
- `warnings`,
- `predicted`:
  - `measured`,
  - `critical_stage_rank`,
  - `critical_stage_seconds`,
  - per-stage `seconds`,
- `observed`: `null` on the plan. Measured token rates live on the **operation**
  status after `ready`, never on the pure plan (`SCREEN-5-HANDOFF.md` §4).

`predicted` contains stage seconds only. Token rates, TTFT, and decode/prefill
throughput belong in the operation’s `observed` block after the post-ready
measurement; a plan request never reads it and is never rewritten by it.

Do not return `port: 0`. If the coordinator port cannot be resolved, return
`endpoint: null` and a reason/action; do not invent a clickable URL. The plan
may return a configured endpoint with `ready: false`; only the health check
turns it into a ready endpoint.

### 13.3a Recommendation object

`recommend` returns:

- `measurement_id` (the measurement used) and `measurement_timestamp`,
- `measurement_model` (the checkpoint the measurement covered; the candidates
  need not include it),
- `hardware_only` (true when recommendation used a model-less measurement —
  always false until `measure` accepts `model: null`),
- `target_context_tokens` (the caller's target, or null for "as far as memory
  allows"),
- `mtp` and `execution_profile` (the inputs the answer was computed with),
- `feasible`, `selected_model`, `selected_context_tokens`,
- `candidates`: ordered list of:
  - `model_id`, `label`, `model_path`,
  - `feasible` / `pp_supported` and `failure_kind`
    (`memory`, `pipeline_unsupported`, `model_unreadable`),
  - `pipeline_stages`, `nodes_required`, `rank0_node`, `rank_order`, `stages`,
  - `weight_bytes`, `download_bytes`, `download_gib`, and `nodes` with
    `model_present`, `stage_bytes`, `disk_free_bytes` (`null` = unknown) and
    `can_stage` per Mac,
  - `max_context_tokens`, `declared_context_tokens`, `context_limited`,
  - `headroom_bytes` (the tightest stage),
  - `critical_stage_rank`, `critical_stage_seconds`, `measured_speed`,
  - `plan_signature`, `reasons`, `warnings`, and one `action` when it cannot be
    chosen,
- `reasons`, `warnings`.

There is no `quality` field: oMLX has no measured quality metric, and a
recommendation that invented one would be the throughput promise of §17.2 in
another form. The ordering is the catalogue’s deliberate one — feasible models
largest first, near misses smallest first.

The recommendation object contains no token-rate promise. It is a model
selection result, not a launch plan. The chosen model must be measured again
and planned through `plan`.

### 13.4 Persistence

Use oMLX’s base path/registry where possible:

- cluster registry,
- deployment records,
- measured performance profiles,
- assistant session,
- coordinator/rank order,
- endpoint.
- API key: Keychain only.

Do not introduce a new `setup.json` control plane.

Assistant measurements (implemented):

- One JSON file: `<base_path>/cluster/assistant-measurements.json`, schema
  version 1, mode 0600, atomic replacement (temp file + `os.replace`), newest
  8 records kept.
- A measurement is **stale 15 minutes** after `measured_at`
  (`MEASUREMENT_TTL_SECONDS = 900`). Freshness is decided in one function
  (`state.measurement_refusal`), never at a call site.
- A plan is refused with `409 stale_measurement` when the measurement is
  missing, unknown, still running, failed, stale, or does not cover the same
  node set, model identity or `mtp` setting. `context_tokens` and
  `rank0_extra_bytes` stay plan-time inputs.
- One measurement job runs at a time. A second `measure` returns
  `409 measurement_in_progress` with the running `measurement_id` and the one
  action `status`; it is not joined, because joining would hand the caller a
  measurement of a different node set or model.
- A job interrupted by a server restart is recovered as `failed`
  (`measurement_interrupted`) rather than reported as still running.
- `compute` is only ever a measurement: without the distributed performance
  probe it stays unmeasured (`compute=None`) and the election scores that Mac
  as worst case.
- The distributed performance probe is **deferred to the start chapter**
  (resolved, not open). `run_cluster_performance_probe` needs an approved
  `ClusterDeployment`, so it cannot run before apply/start exists. Until then
  `compute` stays `None`, `predicted.measured` stays false in plans and
  `measured_speed` stays false in recommendations. Do not estimate a rate in
  its place, and do not wire launch code from the recommendation or plan
  surface to get one. The post-ready measurement is **not** this synthetic
  probe: it is one real completion whose timings oMLX computed, and it fills
  only the operation’s nullable `observed` (`SCREEN-5-HANDOFF.md` §5).
- Coordinator election and recommendation share exactly one budget builder
  (`election.budget_for`). Admission must not be converted to a `NodeBudget`
  anywhere else, or the two would eventually disagree about what a Mac holds.
- A recommendation reads only the hardware part of a measurement and never
  relaxes `plan`. Model selection therefore happens before a model is
  measured, and the chosen model is measured again (model-bound) before it is
  planned.

Assistant operations (implemented):

- One JSON file: `<base_path>/cluster/assistant-operations.json`, schema
  version 1, mode 0600, atomic replacement, newest 8 records kept.
- States are forward-only: `queued → applying → applied` for an apply,
  `queued → starting → loading → ready` for a start, and `failed` from any
  running state. `applied` and `ready` are terminal, so a late event cannot undo
  readiness or revive a failure. A late `applying` after `ready` is dropped.
- One apply/start at a time. A second request is refused with
  `409 operation_in_progress`, the running `operation_id` and the one action
  `status`; it is not joined.
- An operation interrupted by a server restart is recovered as `failed`
  (`operation_interrupted`) with the kind’s retry action, not reported as
  running.
- The record stores node IDs, the election `plan_signature`, the coordinator’s
  `placement_signature`, the plan-time inputs needed to re-elect, progress, one
  endpoint, warnings, the coordinator’s `deployment_id`, whether the coordinator
  was local, whether the deployment is `stopped`, and — only after `ready` — one
  nullable `observed` block. It stores no SSH target, address, API key, argv or
  model file list; paired targets are resolved from the device registry at
  request time.
- Operation status is separate from measurement status: a measurement can expire
  while a deployment runs, and a start can be ready while its measurement is
  stale.

---

## 14. Security and privacy

- All inference APIs require the oMLX API key.
- Admin routes require admin authentication.
- Pairing uses codes and pinned host keys.
- No secrets in logs, plans, or crash reports.
- Helper is signed/notarized.
- Endpoint binds only to the intended private interface.
- No public internet exposure in the MVP.
- Diagnostics redact keys, tokens, and model paths if sensitive.
- Update mechanism verifies signatures.
- No arbitrary `trust_remote_code` unless explicitly advanced and warned.

---

## 15. Observability and support

The primary UI needs only a few states:

- preparing,
- copying,
- planning,
- starting coordinator,
- loading,
- verifying,
- ready,
- failed.

Behind Details:

- per-node measurement,
- stage table,
- rank order,
- coordinator election reason,
- transport,
- logs,
- plan signature,
- diagnostic export.

Support bundle:

- app/engine versions,
- node hardware,
- topology,
- measurements,
- plan/result,
- health,
- redacted logs.
- No API keys.

---

## 16. Testing strategy

### 16.1 Unit tests

- coordinator election with asymmetric capacities,
- rank 0 infeasible but another candidate feasible,
- only one feasible candidate,
- no feasible candidate,
- free-memory preference vs performance,
- rank-0 extra bytes,
- legal PP cuts,
- KV owner boundaries,
- stage balance,
- no local-first fallback.

### 16.2 Integration tests

- 2 Macs: app on small Mac, coordinator on large Mac.
- 2 Macs: app on large Mac, coordinator on large Mac.
- 3 Macs: ring, coordinator not origin.
- 4 Macs: ring, coordinator not origin.
- coordinator remote start over SSH/agent.
- endpoint reachable from origin and another client.
- model staging from a non-coordinator source.
- runtime missing on coordinator.
- version mismatch.
- cable removed before start.
- memory drift between plan and start.
- coordinator start failure: no local fallback.
- smoke chat and agent endpoint.
- TP request rejected in MVP.

### 16.3 Hardware matrix

- M1 Max 32/64 GB.
- M2 Ultra 64/128 GB.
- M5 Max 128 GB.
- macOS 27.2+.
- TB4 and TB5.
- 2/3/4 nodes.
- Ring only; RDMA optional.

### 16.4 DAU usability test

Observe a non-technical user:

- open app,
- plug cables,
- pair,
- start,
- chat,
- connect agent.

Pass criteria:

- no terminal,
- no IP/SSH/model path,
- no rank/layer question,
- no unexplained error,
- one action per failure.

---

## 17. Acceptance criteria

### 17.1 MVP acceptance

- 2–4 Macs discovered and paired from any origin.
- Coordinator/rank 0 elected automatically.
- App origin does not determine rank 0.
- If M1 rank 0 is infeasible and M5 rank 0 is feasible, M5 is selected.
- Coordinator supervisor starts remotely when needed.
- Endpoint is on the coordinator.
- Model staged from any complete source.
- PP split is exact and respects model constraints.
- Chat smoke test succeeds.
- Agent endpoint works.
- No terminal in the primary flow.
- English UI only.
- TP controls are absent/disabled.
- Failure paths give one action and never silently fall back to local rank 0.

### 17.2 Performance targets

Do not promise absolute tokens/s. Define:

- predicted stage time from measured profiles,
- no avoidable critical-stage regression versus the best feasible candidate,
- coordinator election chooses the best feasible plan,
- measured TTFT/decode displayed after smoke,
- re-measure before start if admission drifted > threshold.

The distributed performance probe is **not a gate** for Chapter 5. Until an
approved deployment exists, `compute` stays `None`, `predicted.measured` is
`false`, and recommendation `measured_speed` is `false`. Do not estimate a
rate and do not delay apply/start for the probe. Run the probe only after the
approved deployment/start, and put measured values in `observed`.

### 17.3 Product targets

- Time from app open to plan: < 60 s (excluding model download).
- Time from start to ready: bounded by model load; visible progress.
- Zero manual network/SSH/model/rank inputs.
- One primary action per screen.
- Every failure has a retry or one repair action.

---

## 18. Milestones

### Phase 0 — Audit and contract

Deliverables:

- map every assistant feature to an existing oMLX endpoint,
- define coordinator-election data model,
- define PP plan schema,
- list what to delete/port from LocalInferenz,
- decide fork/upstream strategy.

Exit:

- no duplicate endpoint is planned,
- rank-0 election API agreed.

### Phase 1 — Election library and tests

Deliverables:

- `cluster/assistant/election.py`,
- measurement input model,
- candidate enumeration,
- feasibility checks,
- score/ordering,
- unit tests for all asymmetric cases.

Exit:

- all election tests pass,
- no UI yet.

### Phase 2a — Chapter 5: Apply/Start backend

Detailed brief: [`CHAPTER-5.md`](CHAPTER-5.md).

This is the first phase allowed to touch staging, deployment, launch,
supervisor and health.

Deliverables:

- approved plan validation (`measurement_id`, `plan_signature`, model, `mtp`,
  node set),
- apply/stage,
- local and remote coordinator start through existing oMLX launch paths,
- operation status,
- health/model-ready check,
- one smoke completion,
- endpoint `ready` only after health/smoke,
- no local fallback,
- tests with launch/deployment fakes.

Not in this phase:

- distributed performance probe before an approved deployment,
- model-less/hardware-only measurement,
- full Screen-3 UI,
- token-rate estimates,
- TP.

Exit:

- approved model-bound plan applies and starts on the elected coordinator,
- remote coordinator start works from a non-coordinator origin,
- endpoint is not ready before health/smoke,
- failure yields one action and never a local rank-0 fallback,
- backend tests pass without a live cluster.

### Phase 2b — 2-Mac PP end-to-end UI

Deliverables:

- minimal Screen-3 bindings for `selected_model` and
  `selected_context_tokens`,
- approve/apply/start calls,
- operation progress,
- endpoint shown only when ready,
- simple 2-Mac UI path.

Exit:

- app on M1, coordinator M5 works,
- app on M5, coordinator M5 works,
- no local fallback.

### Phase 3 — 3/4-Mac ring

Deliverables:

- ring topology checks,
- 3/4-node plans,
- coordinator election,
- staging,
- start,
- smoke.

Exit:

- 3/4-node cluster works without manual rank/IP input.

### Phase 4 — DAU productization

Deliverables:

- Assistant mode inside oMLX UI,
- English copy,
- one recommended model,
- progress/error states,
- app menu entry,
- diagnostics.

Exit:

- non-technical user completes setup unaided.

### Phase 5 — Distribution and helper

Deliverables:

- signed/notarized app and helper,
- one admin prompt,
- auto-update,
- network setup,
- optional RDMA.

Exit:

- clean install on fresh Macs works.

### Later — TP

Only after PP is stable:

- TP placement,
- TP/PP strategy selector,
- TP performance profiles.
- No TP in MVP.

---

## 19. Proposed project structure

```text
omlx/
  cluster/
    assistant/
      __init__.py
      election.py          # rank-0 candidates, feasibility, scoring
      measurement.py       # p05/p50/p95 admission, performance profiles
      recommendation.py    # model/context/chunk recommendation
      provisioning.py      # any-to-any runtime/model preparation
      start.py             # local/remote coordinator start
      routes.py            # admin API
      models.py            # request/response schemas
      state.py             # session/persistence
  apps/
    omlx-web/
      ...                  # Assistant mode in existing Cluster tab
    omlx-mac/
      ...                  # menu entry / Welcome entry
  tests/
    cluster_assistant/
      test_election.py
      test_election_infeasible.py
      test_any_to_any.py
      test_remote_start.py
      test_pp_plan.py
```

Do not create:

```text
localinferenz-console/     # no second control plane
localinferenz-app/         # no second app
localinferenz-wizard/      # no second wizard
```

---

## 20. Migration from LocalInferenz

### 20.1 Keep / port

- measured admission (`console/measure.py` concepts),
- exact PP planning and KV owner boundaries,
- rank-0 candidate election (new),
- remote start pattern from `console/profiles.py`,
- any-to-any provisioning,
- error translation and DAU insights,
- test scenarios.

### 20.2 Do not port

- second discovery/pairing,
- second wizard/state machine,
- second profile system,
- second chat UI,
- second OpenCode/Codex helper,
- second macOS app,
- German localization.

### 20.3 Repository strategy

Recommended:

- Start from an oMLX fork/branch.
- Add the assistant module and UI mode.
- Keep the LocalInferenz repo as a reference/prototype.
- Do not maintain both products.

If upstreaming is possible, upstream the assistant in small PRs:

1. election library + tests,
2. measurement profile input,
3. assistant API,
4. UI mode,
5. mac app entry.

If upstream is not possible, maintain a focused fork with a small patch surface.

### 20.4 License and upstream hygiene

- Keep oMLX’s Apache-2.0 license and notices.
- Keep assistant code in a clearly separated module so upstream rebases remain cheap.
- Do not mix proprietary branding/assets into the runtime module.
- Track every local oMLX patch in a patch series with a reason and a test.
- Re-run the full cluster test suite after every engine pin update.

---

## 21. Risks and mitigations

| Risk | Mitigation |
|---|---|
| oMLX moves fast; fork drift | Small module, reuse routes, regular rebase |
| Two wizards/control planes | Assistant mode inside existing Cluster tab |
| Remote start permissions | Use existing pairing/SSH keys; agent/helper; clear errors |
| Rank-0 election picks wrong node | Feasibility first; measured performance; tests |
| Free-memory-only selection hurts performance | Best feasible plan; free memory as preference/tie |
| PP split unstable on some models | PP-only catalogue; exact boundaries; smoke tests |
| Model load time | Progress; no fake ready; bounded timeouts |
| Memory drift | Re-measure before start; plan signature |
| Cable/topology confusion | Chassis illustrations; ring checks; one action |
| DAU support burden | Diagnostics bundle; plain error messages |
| TP scope creep | Explicitly out of scope; interface stub only |
| German/localization scope creep | English only; no locale work |

---

## 22. Open decisions

### Resolved before Phase 1

1. **Measurement flow:** asynchronous measurement job.
   - `measure` persists a `MeasurementSet` and returns `measurement_id` plus a
     job/status handle.
   - `plan` is synchronous, pure, and requires `measurement_id`.
   - `plan` never probes implicitly; missing/stale measurement is a 409 with
     one retry action.
   - UI order: measure → poll/stream → plan → approve → apply/start.

2. **Endpoint port:** read from the target coordinator's configured port.
   - Default 8099 only when the settings cannot be read.
   - Return `endpoint.host`, `endpoint.port`, `endpoint.url`, `endpoint.ready`.
   - Never return `port: 0`; if unresolved, return `endpoint: null` plus an
     action.
   - `ready` stays false until the supervisor/health check succeeds.

3. **Prediction shape:** stage seconds only before measurement.
   - No `prefill_tok_per_s`, `decode_tok_per_s`, or `ttft_ms` in the plan.
   - Use `predicted.measured`, `predicted.critical_stage_rank`,
     `predicted.critical_stage_seconds`, and per-stage `seconds`.
   - After the smoke test, attach measured values under `observed`.

### Resolved 2026-11-07

1. **Strict rank-0 rule:** best feasible plan, with max free admission as a
   strong preference. (Already frozen in `AGENTS.md` §2.)
2. **Product name/branding:** **oMLX Cluster Assistant**.
3. **Fork vs upstream, pin:** focused fork. The pin is `1e27576a`
   (`cluster-assistant-base`, tag `cluster-assistant-base-2026-11-07`): #5024 +
   #5033 + rank-0-over-SSH + launch reliability + the MoE fit hint. Do not
   rebase again without a decision.
4. **Coordinator/rank 0 coupling:** keep co-located in the MVP. (Already
   frozen.)
5. **RDMA default:** ring over TCP only; RDMA/JACCL is the later advanced path.
6. **Default context:** the recommendation decides; target 32k, no fixed value.
7. **Default model:** the recommendation decides; no fixed default model.
8. **3/4-node ring requirement:** a 3/4-Mac plan requires the full topology; an
   incomplete selected topology offers “run with 2 Macs” only with explicit
   confirmation, never a silent degrade. A 2-Mac plan is the normal minimal
   subset when it fits.
9. **Helper timing:** full signed/notarized helper is Phase 5 (after Chapter
   10); a too-low wired limit is a preflight refusal with one explicit action
   before the gate.
10. **Telemetry:** none; opt-in diagnostics/local logs only, never model or
    prompt content.

### Additional decisions from the Chapter-10 gate plan

- **Resource levers:** tier/context/offload/MoE changes are shown in the plan
  and applied only on **Start setup**; never silent, never overwrite an
  explicit user choice without showing it.
- **Admission:** p05 is the safety floor; p50/live admission plans by default.
  p95 is only tried for a huge model with an explicit risk confirmation.
- **Measured compute:** the synthetic distributed probe runs after apply
  (staged, before start), its profiles are stored on the origin, and the **next**
  measurement/plan uses them; it never rewrites the current `predicted` and
  never gates apply/start. A failure is one `compute_probe_failed` warning.
  Implemented 2026-11-07 (`AssistantPerformanceStore`, coordinator
  `/admin/api/cluster/performance-probe`).
- **TTFT/prefill:** one streaming measurement after ready supplies
  `time_to_first_token` / `prompt_tokens_per_second`; the non-streaming decode
  block stays. Off by default, toggled in-app; a failure is one warning.
  Implemented 2026-11-07.
- **Gate scope:** P0/P1 only; macOS app/tray, full helper, telemetry, TP and >4
  Macs are P2 after Chapter 10 (confirmed 2026-11-07).
- **Topology completeness (2026-11-07):** a complete fabric is a **connected
  line** — every consecutive pair in the chosen rank order has a measured link.
  A 3/4-Mac plan with an incomplete line is refused (`topology_incomplete`) and
  names the missing pairs; the 2-Mac plan is offered only after explicit
  confirmation, and that consent is part of the plan signature. A transport
  probe that did not answer (`fabric_measured=false`) keeps the documented
  nominal-ring fallback; an empty matrix is not “no links”.

### Additional decisions from Phase A (2026-11-07)

- **The resource policy is one object.** Context, table offload
  (`engram`/`ple`/`none`), MoE offload and resident fraction, memory-guard tier,
  microbatch and the admission percentile travel together through plan, apply
  and start; they are part of the plan signature. A drift is `plan_mismatch`.
- **Lever priority is quality first, then least-invasive.** At the smallest
  subset, the solver prefers the largest context that fits; at that context it
  prefers the saved lever, then enabling the checkpoint's table offload, then
  MoE 0.75/0.5/0.25, then a higher tier, then the p95 admission with explicit
  consent. Context reduction is only accepted when no lever keeps the target
  context. This deliberately keeps the target context ahead of a tier raise.
- **Admission:** `p05` is the floor; `p50`/live admission is the planning value;
  `p95` is only tried with `allow_optimistic_admission` and is reported as
  `optimistic_admission`. A raised tier uses oMLX's own `tier_reserve_bytes`
  arithmetic, capped by the measured p95.
- **Explicit "keep it resident" is written.** When the plan chooses no table
  offload for a checkpoint that has one, apply writes the switch `false` on the
  selected Macs. The plan showed the lever and Start setup is the confirmation;
  this is not a silent overwrite.
- **A too-low wired limit is a preflight refusal** with the action
  `raise_wired_limit` (R5, 2026-11-15): the refusal names the Mac and its MB
  value, and the step is Performance → “Apply with Admin…” in the app — one
  admin dialog through the signed helper. The browser has no privileged path and
  only names the step. A measured `0` is the macOS default nobody raised;
  `None` (not measured) never reads as too low.
- **Live Phase A evidence (2026-11-07).** `.12` was paired through the existing
  join/approve API; `.14` was synced to this branch and restarted. A four-Mac
  measurement of `GLM-5.3-Flash-oQ2e-mtp` (107.8 GiB) planned on `.12`+`.11`,
  named `.14`/`.13` as not needed, started and served a completion. A DeepSeek
  V4.1 Engram start (`.11`+`.12`, 4k context) and a routed-expert 25 % start
  both served completions. The Qwen4 PLE checkpoint's table is detected.
  **Updated 2026-11-16:** with upstream PR #5033 applied, Qwen4-Exp/Qwen3.8 is
  **pipeline-capable** on this branch and the assistant may plan it as a
  pipeline; the earlier “not pipeline-capable / `choose_pipeline_model`”
  statement described the pre-#5033 state and is withdrawn (R1 = PP only,
  TP = R2). A pipeline-capable 150 GiB-class checkpoint is not present on the
  validation cluster; the 4-Mac start of a >100 GiB model stays Phase D
  environment work (`.13` disk/admission). Evidence is in `PHASE-2B-RUN.md` §14
  and §27.

If a new product decision appears, ask once and record it here; do not silently
decide it.

---

## 23. Definition of done for the starting project

This document is successful when the new project has:

- a clear oMLX-based architecture,
- a coordinator-election design with tests,
- a PP-only planner contract,
- an any-to-any remote-start design,
- a DAU flow with no terminal/IP/SSH/rank inputs,
- an English-only UI contract,
- a migration list from LocalInferenz,
- explicit non-goals (TP, second app, second wizard),
- a phased plan ending in a working 2–4 Mac chat/agent cluster.

**First implementation task:** build the election library and its tests before touching the UI.

---

## 24. Glossary

| Term | Meaning |
|---|---|
| **PP** | Pipeline parallelism; layers split across stages/ranks |
| **TP** | Tensor parallelism; each node computes part of every layer |
| **rank 0** | Pipeline rank that holds the tail/LM head and runs the coordinator in the MVP |
| **Coordinator** | Process that runs the supervisor/API endpoint; co-located with rank 0 in the MVP |
| **Origin** | Mac where the user opened the app |
| **Controller** | UI/agent driving setup; may be the origin |
| **Elected coordinator** | Mac chosen by the assistant to host rank 0/endpoint |
| **Admission** | Measured memory ceiling a rank may use |
| **Headroom** | Admission minus planned resident/peak bytes |
| **Owner boundary** | Legal layer cut imposed by KV ownership |
| **Any-to-any** | Any Mac can originate, install, coordinate, or work |
| **DAU** | User without cluster/MLX/terminal knowledge |

---

## 25. Final statement

The new project is not “LocalInferenz 2” and not “exo for oMLX”.

It is:

> **The simple, automatic, PP-first cluster mode inside oMLX.**

The core differentiator is not another wizard. It is:

- automatic coordinator election,
- feasibility-first rank placement,
- measured admission,
- any-to-any install and start,
- an endpoint on the best feasible Mac,
- a DAU path that ends in Chat or an agent without terminal work.

---

## Appendix A — Cold-start handoff for the next agent

This appendix exists so a new agent can start without the prior conversation. It is not a replacement for reading the current source; it is the minimum context and the first concrete task.

### A.1 Decisions that are already made — do not re-open them

1. The product is **not** a new exo clone.
2. The product is **not** a second LocalInferenz app.
3. The target is the **existing oMLX app + web UI + cluster control plane**.
4. The assistant is a **feature/module inside oMLX**, not a sidecar product.
5. MVP is **PP only**. Do not add TP behavior now.
6. Product UI is **English only**. No German localization in the MVP.
7. Support **2–4 Apple Silicon Macs over Thunderbolt**.
8. The app origin must never determine rank 0.
9. Rank 0, coordinator, and endpoint are co-located in the MVP.
10. The chosen coordinator process runs on the elected Mac; rank 0 remains local there.
11. Feasibility wins over performance; a candidate with no legal split is discarded.
12. Free admission is a strong rank-0 preference, not the sole criterion.
13. If the elected coordinator fails, do not silently fall back to the local Mac.
14. The election library and its tests were the **first** milestone (long done).
    **Current state (2026-11-16):** every product surface exists and is tested;
    the single open item is the live acceptance run A1–A7, then the PR fold and
    push. Do not start a new feature — follow
    [`PLAN-REMAINING-2026-11-16.md`](PLAN-REMAINING-2026-11-16.md).

### A.2 Reference workspaces

Current reference environment on this machine:

| What | Path |
|---|---|
| LocalInferenz prototype | `~/Documents/PycharmProjects/localinferenz` |
| This brief | `~/Documents/PycharmProjects/localinferenz/PLAN-oMLX-Cluster-Assistant.md` |
| Installed oMLX checkout | `~/Library/Application Support/LocalInferenzMachine/src/omlx` |
| oMLX upstream | `https://github.com/jundot/omlx` |
| Project fork | `https://github.com/digitalfactory-ako/omlx` |
| Pinned oMLX commit | `upstream/main` `624cff0e` — this fork is replayed onto it (`rebase/upstream-main`, rebased 2026-11-14) |
| Historical pin | `1e27576af5e10c8f4459e0fdcb05c5cf59245d44` (tag `cluster-assistant-base-2026-11-07`, no longer used) |

Patch/PR status is tracked in `docs/cluster-assistant/UPSTREAM.md`. The base
includes merged #4985 and #4974 plus the local MoE/MTP series; it does **not**
include the full launch series, the rank-0 SSH draft, or the Qwen4 experiment
branch.

The new project should **not** be seeded from the LocalInferenz app. Start from an oMLX fork/branch or an oMLX worktree, then add the assistant module.

If the new project uses a newer oMLX pin, re-audit the routes and UI paths before copying assumptions. oMLX’s cluster surface moves quickly.

### A.3 oMLX code to read first

Read these before writing code:

| Area | Files |
|---|---|
| Cluster v2 wizard | `apps/omlx-web/omlx_web/templates/dashboard/_cluster_v2.html` |
| Cluster v2 JS | `apps/omlx-web/omlx_web/static/js/cluster_v2.js` |
| Cluster routes | `omlx/cluster/routes.py` |
| Pairing routes | `omlx/cluster/pairing_routes.py` |
| Discovery routes | `omlx/cluster/discovery_routes.py` |
| Deployment constraints | `omlx/cluster/deployment.py` |
| Planner / strategy | `omlx/cluster/planner.py` |
| Launch / remote rank start | `omlx/cluster/launch.py` |
| Staging / model copy | `omlx/cluster/staging.py`, `omlx/cluster/modelsync.py` |
| macOS app | `apps/omlx-mac/Sources/` |
| Cluster UI tests | `apps/omlx-web/tests/test_cluster_ui_integration_ui.py` |
| Plan approval tests | `tests/test_cluster_plan_approval.py` |
| Cluster UI backend tests | `tests/test_cluster_ui_integration.py` |

Existing oMLX surfaces that must be reused, not duplicated:

```text
/api/cluster/devices
/api/cluster/pair/*
/api/cluster/discovery/*
/admin/api/cluster/models
/admin/api/cluster/catalogue
/admin/api/cluster/peer-probe
/admin/api/cluster/autoconfigure
/admin/api/cluster/stage
/admin/api/cluster/deployments
/admin/api/cluster/replan
/admin/api/cluster/diagnostics
```

### A.4 LocalInferenz code worth harvesting

| Area | File | Use |
|---|---|---|
| Admission measurement | `console/measure.py` | p50/p95/p05, pressure, worker RSS |
| Joint planner | `console/planner.py` | model/context/chunk/Pareto ideas |
| PP split | `console/deploy.py` | exact cuts, KV owner boundaries, stage costs |
| Remote profile start | `console/profiles.py` | detached start on a coordinator via SSH |
| Model catalog | `catalog/` | model-specific PP/KV/Engram metadata |
| Tests | `tools/test_nnodes.py` | asymmetric split/election test scenarios |

Do **not** port:

- `console/setup.py` as a second wizard,
- `console/ui.py` as a second UI,
- the LocalInferenz app as a second app,
- the LocalInferenz chat/OpenCode helper,
- German product strings.

### A.5 Current oMLX constraint that matters

oMLX currently assumes:

```text
rank 0 == local coordinator == endpoint owner
hosts[0].ssh == 127.0.0.1
```

The new design does **not** remove that constraint. It satisfies it by starting the supervisor **on the elected coordinator Mac**. That Mac is local to its own rank 0.

Do not “fix” this by making a local coordinator control a remote rank 0. That is a larger and riskier refactor.

### A.6 Reference test cluster

The current validation environment (not production, may be offline):

| Node | Hardware | RAM | Role in tests |
|---|---|---|---|
| `172.16.1.11` | MacBook Pro M5 Max | 128 GiB | fast/large candidate |
| `172.16.1.13` | Mac Studio M2 Ultra | 64 GiB | medium candidate |
| `172.16.1.14` | Mac Studio M1 Max | 32 GiB | **reassigned 2026-11-16 — never touched** (own task, needs its own oMLX) |
| `172.16.1.12` | MacBook Pro M1 Max | 32 GiB | known 2-node peer, often offline |

The critical asymmetric test is:

> App on M1 or M2, M5 as elected coordinator; endpoint on M5; M1/M2 as workers.

Do not assume the cluster is online, and do not copy API keys or model data into the new repo.

### A.7 Build/test commands to discover

The new project should use oMLX’s own commands. Verify them in the target checkout rather than copying stale commands:

```sh
# oMLX runtime / tests
uv sync --extra cluster
uv run pytest tests/test_cluster_plan_approval.py
uv run pytest apps/omlx-web/tests/test_cluster_ui_integration_ui.py

# Web CSS rebuild (if templates change)
cd apps/omlx-web
python build_css.py
```

For reference, the LocalInferenz prototype used:

```sh
.venv/bin/python tools/test_nnodes.py
.venv/bin/python tools/test_catalog.py
.venv/bin/python tools/test_profiles.py
```

Do not treat those as the new project’s test suite. They are reference tests for election/planning behavior.

### A.8 First task for the new agent

Build the election library, not the UI.

Suggested module:

```text
omlx/cluster/assistant/
  election.py
  measurement.py
  models.py
  tests/test_election.py
```

Minimum types:

- `NodeFacts`: node id, capacity, admission p05/p50/p95, compute profile, link, model presence, disk.
- `PlanCandidate`: rank order, cuts, stage bytes, headroom, predicted critical stage, feasibility reason.
- `ElectionResult`: coordinator node, endpoint, rank order, chosen plan, alternatives, reason.

Minimum tests:

1. App origin on small Mac, M5 rank 0 fits → M5 selected.
2. M1 rank 0 cannot hold tail/head, M5 rank 0 fits → M5 selected.
3. Both candidates feasible → best plan wins; free admission is a preference/tie-break.
4. No candidate feasible → explicit error, no local fallback.
5. Rank-0 extra bytes (LM head/MTP/HTTP) change feasibility.
6. 3/4-node permutations produce valid rank orders.
7. TP request is rejected or hidden.

Exit criterion for the first task:

> A standalone election test proves that the app origin does not decide rank 0, and that an infeasible local rank 0 is replaced by a feasible remote rank 0.

### A.9 What the agent must not do

- Do not create a second wizard or control plane.
- Do not add German strings.
- Do not add TP behavior.
- Do not assume the local Mac is rank 0.
- Do not silently fall back to local rank 0.
- Do not change oMLX’s rank-0-local constraint; start the coordinator on the elected Mac.
- Do not duplicate oMLX discovery, pairing, staging, chat, or client integrations.
- Do not commit API keys, model files, or private logs.
- Do not promise absolute throughput before measuring.

### A.10 What the agent still needs from the environment

This document cannot supply:

- repository access/credentials,
- the final product name,
- the new repo path,
- the chosen oMLX pin for the new project,
- decisions in section 22,
- actual hardware availability.

If those are missing, the agent should ask for them once and then proceed with the documented defaults.