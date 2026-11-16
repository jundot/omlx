# Cluster Assistant — project notes

This directory holds the project-specific material for the oMLX Cluster Assistant.

## Order

1. `PROJECT-BRIEF.md` — full product and engineering brief.
2. `../../AGENTS.md` — operating rules for any agent working here.
3. `../../CLUSTER-ASSISTANT.md` — short repository orientation.
4. `CHAPTER-5.md` — apply/start implementation brief.
5. `SCREEN-3-MINIMAL.md` — minimal UI contract for the first end-to-end slice.
6. `ROADMAP-TO-CHAPTER-10.md` — the resource-planning gate plan (closed offline).
7. `CHAPTER-10.md` — DAU acceptance, the redacted support bundle and the live
   matrix.
8. **Current session:**
   - `PLAN-REMAINING-2026-11-16.md` — **the checklist for what is left** (live
     acceptance, deferred matrix, fold/push, cleanup) with a done-when per line.
   - `ROADMAP-ACCEPTANCE-2026-11-16.md` — the acceptance gate A1–A7 (W1–W6),
     the five recorded decisions and the PR path.
   - `ASSISTANT-UI-SPEC-2026-11-16.md` — the code-level contract of the one-flow
     Cluster tab (screens, markers, semantics, test matrix).
   - `ASSISTANT-UI-IMPLEMENTATION-PLAN-2026-11-16.md` — U1–U6 log and the
     recorded deviations.
   - `PR-READY-2026-11-16.md` — what is committed, the fold into the PR series,
     the DMG artefact and what is still open.
   - `READY-FOR-CLUSTER-TEST.md` — rollout order, per-branch suite numbers, DMG
     and the wipe/install steps.
9. `HELPER.md` — the privilege capability and its open points.
10. `DISTRIBUTION.md` — direct-DMG release plan and open credentials.
11. `SIGNING.md` — Developer ID and notarization setup checklist.
12. `SLIM-RUNTIME.md` — external Python runtime instead of the 1.6 GB embed.

## Scope in one sentence

A DAU-friendly, English-only, PP-first cluster mode inside oMLX for 2–4 Macs over Thunderbolt, where any Mac can originate setup and the coordinator/rank 0/endpoint is elected by a measured, feasibility-first plan.

## Current milestone

Chapter 10 — DAU acceptance, and it is down to one block: **the live acceptance
run**. The offline gate is closed: election, measurement, recommendation,
plan/apply/start, the Screen-3–6 surfaces, hardware-only measurement,
source-first copy, the resource-policy solver, the measured compute/TTFT probes,
the connected-line topology with the consented 2-Mac fallback, the reload resume
and the cluster-wide operation lock all exist and are fake-driven tested
(`tests/cluster_assistant` **658 passed**).

Since 2026-11-15/16 the acceptance package itself is implemented:

- **D1** in-app uninstall with one keep switch per scope (models and
  `cluster/identity.json` kept by default) — `caa3901a`, live-checked.
- **D6** the helper is a capability (`PrivilegedHelper.status`, Welcome step,
  permanent Performance row) — `a62c29d9`, live-checked.
- **D2/D3/D4** steps, back/`stale`, in-place retry, error placement and the
  one-flow Cluster tab (`Assistant` / `Manual setup`) — `0c6e7479`, U1–U6,
  `apps/omlx-web/tests` **578 passed**.
- Distributed inference is **on by default** (`a6758008`).

Still open: **A1–A7** on a wiped four-Mac cluster with one identical build
(`ROADMAP-ACCEPTANCE-2026-11-16.md` §2, `PR-READY-2026-11-16.md` §5), then the
fold into the PR series and the push.

Read:

- [`PHASE-2B-RUN.md`](PHASE-2B-RUN.md) — the live evidence, the pass/fail per
  Phase 2b exit criterion, and the findings (bugs, contract gaps,
  environment).
- [`PHASE-2B-RUNBOOK.md`](PHASE-2B-RUNBOOK.md) — the operator click sheet used
  for that run.
- [`SCREEN-4-PROGRESS.md`](SCREEN-4-PROGRESS.md) — the Screen-4 progress
  decision and its implementation record (§8): six observable rows mapped from
  the data the operation already reports, no backend change, no SSE.
- [`SCREEN-5-HANDOFF.md`](SCREEN-5-HANDOFF.md) — the Screen-5 decision and
  implementation record: the ready card opens the existing `/chat` with the
  serving model id, the agent card carries base URL/model id/key location, and
  the operation’s nullable `observed` block is filled only by a real post-ready
  completion (default off).
- [`SCREEN-6-MANAGE.md`](SCREEN-6-MANAGE.md) — the Screen-6 decision and
  implementation record: Stop relays to the coordinator’s own unload route and
  marks the operation `stopped`; Start again re-applies the frozen plan inputs.
- [`CHAPTER-10.md`](CHAPTER-10.md) — the current chapter: DAU acceptance, the
  support-bundle contract, the offline-proven list and the deferred live matrix.
- [`ROADMAP-TO-CHAPTER-10.md`](ROADMAP-TO-CHAPTER-10.md) — the closed gate plan
  (including the resource-planning defect against the LocalInferenz reference);
  only live evidence remains open.
- [`GATE-START-PROMPT.md`](GATE-START-PROMPT.md) — the copy-paste starting
  prompt for the agent that executes that gate plan.

Still later:

- tensor parallelism (TP): R2, after R1 ships PP only.

## Key non-goals

- exo clone
- second LocalInferenz app
- second oMLX wizard/control plane
- tensor parallelism in the MVP
- German product UI
- local-first rank 0

## Reference pin

```text
upstream/main = 624cff0e   (rebase/upstream-main is this fork replayed onto it)
```

The historical base `1e27576af5e10c8f4459e0fdcb05c5cf59245d44` (tag
`cluster-assistant-base-2026-11-07`) is no longer the pin; see
`REBASE-2026-11-14.md`. If you move the pin, re-audit the oMLX cluster routes
and UI before reusing assumptions from this brief.