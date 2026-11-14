# Cluster Assistant — project notes

This directory holds the project-specific material for the oMLX Cluster Assistant.

## Order

1. `PROJECT-BRIEF.md` — full product and engineering brief.
2. `../../AGENTS.md` — operating rules for any agent working here.
3. `../../CLUSTER-ASSISTANT.md` — short repository orientation.
4. `CHAPTER-5.md` — apply/start implementation brief.
5. `SCREEN-3-MINIMAL.md` — minimal UI contract for the first end-to-end slice.
6. `CHAPTER-10.md` — current chapter: DAU acceptance, the redacted support
   bundle and the deferred live matrix.
7. `HELPER.md` — the P2 privileged-helper first slice and its open points.
8. `DISTRIBUTION.md` — direct-DMG release plan and open credentials.
9. `SIGNING.md` — Developer ID and notarization setup checklist.
10. `SLIM-RUNTIME.md` — external Python runtime instead of the 1.6 GB embed.

## Scope in one sentence

A DAU-friendly, English-only, PP-first cluster mode inside oMLX for 2–4 Macs over Thunderbolt, where any Mac can originate setup and the coordinator/rank 0/endpoint is elected by a measured, feasibility-first plan.

## Current milestone

Chapter 10 — DAU acceptance. The Chapter-10 gate is closed offline: election,
measurement, recommendation, plan, apply/start, the Screen-3–6 surfaces,
hardware-only measurement, source-first copy, the resource-policy solver, the
measured compute/TTFT probes, the connected-line topology and the consented
2-Mac fallback all exist and are fake-driven tested. The redacted support bundle
is implemented. The only open gate item is the live evidence block, deferred
until the cluster is reachable.

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

- tensor parallelism (TP), which stays out of the MVP.

## Key non-goals

- exo clone
- second LocalInferenz app
- second oMLX wizard/control plane
- tensor parallelism in the MVP
- German product UI
- local-first rank 0

## Reference pin

```text
1e27576af5e10c8f4459e0fdcb05c5cf59245d44
```

If you move the pin, re-audit the oMLX cluster routes and UI before reusing assumptions from this brief.