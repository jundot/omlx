# Roadmap to acceptance, then to the PRs — 2026-11-15

This is the plan for the next sessions. It replaces guessing with a gate:
**acceptance first, PRs second.** Nothing here re-opens a frozen product decision
(brief §7–§10, AGENTS §1); where a choice is still open it is listed in §7 and
answered once.

Read with:

- `PACKAGE-2026-11-15-ACCEPTANCE.md` — the defect package (D1, D2, D3, D4, D5, D6)
  with symptoms, root causes, proposed shape and acceptance criteria.
- `PHASE-2B-RUN.md` §29 — the last live run (ready in ~50 s, one real completion
  at 18.76 tok/s, clean stop), and R23 in
  `UPSTREAM-PR-REVIEW-2026-11-15.md`.
- `READY-FOR-CLUSTER-TEST.md` — rollout order, per-branch suite numbers, DMG.

---

## 1. Where we actually are

**Proven live (2026-11-15):** measure → recommend → model-bound measure → plan →
apply → start → **ready** → real completion → stop, with the coordinator **not**
the origin (any-to-any), on a cleanly applied plan, after fixing the
plan-priced-tier defect (R23). The tier is now written on every selected Mac
before staging; the admission refusal names the tier.

**Proven locally:** the assistant suite is green (663 incl. UI tests), A3's full
suite is green (20 229 passed / 1005 skipped / 0 failed), R5 (`raise_wired_limit`)
and the rank-0 pin are implemented with tests, the notarized DMG builds through
one script that leaves no app copies behind.

**Not proven, and this is the point of this roadmap:**

| Gap | Why it blocks acceptance |
|---|---|
| The **manual** path never brought a model up | If only the assistant works, a refusal has no fallback and no cross-check |
| No **back** from a green step, no **retry** of a red one | One wrong choice means starting over; a failure is a dead end |
| Errors surface in places nobody looks (advanced JSON, toasts, wrong step) | A user cannot connect a failure to a cause |
| The tab layout reads in the wrong order | The user cannot tell which flow is the flow |
| **Helper install** is reachable only from the low-limit warning | The privilege capability is hidden exactly when it is not needed |
| **Uninstall** is a shell script, not a button | Support cannot ask a person to "uninstall and retry" |

**Build state (branch `rebase/upstream-main`, HEAD `1f62ae72`):** the tier fix,
the inspection fixes, R5, the pin (backend + card), the cleanup/install scripts and
the DMG wrapper are committed. The PR series (`feat/cluster-assistant-plan` →
`-run` → `-run-local`, heads `da13ab2e` / `6c37cacc` / `a574ba70`) does **not**
contain them yet — folding is a single deliberate step (§6), not per-commit.

**DMG in use:** `build/oMLX-0.8.0rc1.dmg` = `19a446d9…`, app binary
`01e7898c…`, installed on all four Macs, all four cleaned (0 processes,
`~/.omlx` = `cluster/identity.json`, 0 app copies, 0 LaunchServices records).

---

## 2. The acceptance gate (what "accepted" means)

A run counts as accepted only when all of these hold, from a **wiped** four-Mac
state (`scripts/cleanup-cluster-macs.sh --all --with-helper`), with the same build
on every Mac (`version` **and** app-binary hash):

- **A1 Manual:** one model brought up through the manual Cluster flow alone —
  devices → pair → measure → model → plan → apply → start → ready → one real
  completion → stop.
- **A2 Assistant:** the same model through the assistant card, with the endpoint
  only shown when `ready`.
- **A3 Correction:** a real failure (e.g. pin a Mac that cannot hold the tail, or
  a too-low tier) is shown **at the failing step**, corrected in place, and the
  step is retried successfully — without reloading the page.
- **A4 Back:** after a green plan, change the model or context, see the downstream
  steps become `stale`, and finish the flow.
- **A5 Capability:** the helper can be installed (or reinstalled) at any time from
  Welcome and from Performance, independent of the wired-limit warning; the wired
  limit's current/target is visible.
- **A6 Tear down:** uninstall from the app removes app, helper, runtime, settings
  and cluster state, keeps models and `cluster/identity.json`, and leaves 0
  LaunchServices records.
- **A7 No surprises:** no error text exists only in the advanced preview, and no
  warning is rendered as an error.

Everything below exists to make A1–A7 true.

---

## 3. Work packages, in order

### W1 — D1 Uninstall button (1 day)

Settings → Advanced → **Uninstall…**: one sheet, one keep switch per scope, byte
counts, models and `cluster/identity.json` kept by default, helper behind the one
admin dialog, LS cleanup (`lsregister -u` per removed path, `-gc`, Dock repaint),
app quits last. XCTests over an injected path/locator layer (nothing real deleted).
`scripts/cleanup-cluster-macs.sh` stays the lab equivalent.

**Done when:** A6 holds and the XCTest suite proves each default scope.

### W2 — D6 Helper as a capability (0.5–1 day)

`PrivilegedHelper.status → {installed, version, wiredLimitCurrentMB,
wiredLimitRequestedMB}` as the one truth; a **System helper** step in Welcome; a
permanent Helper row with Install/Reinstall in Performance; the wired-limit row
separate with Apply whenever current ≠ requested; idempotent, versioned blessing.

**Done when:** A5 holds and the five status states are tested.

**Implementation status (2026-11-15):** W1 and W2 are implemented on
`rebase/upstream-main` (commits `a62c29d9` D6, `6f0bf259` D1, `804ad2c1`
decisions). The five §7 decisions are answered and recorded in AGENTS §10.

- W2: one seam `PrivilegedHelper.status → {installed, version,
  wiredLimitCurrentMB, wiredLimitRequestedMB}`; `ensureInstalled`
  (bless/reuse/replace, injectable); a permanent **System Helper** row in
  Performance with Install/Reinstall independent of the warning, a separate
  wired-limit row with Apply whenever current ≠ requested, and a **System
  helper** step in Welcome. The five D6 states are tested without a prompt.
- W1: `Sources/AppView/Utils/Uninstall.swift` (locator/plan/runner, injectable)
  + `AdvancedScreen.swift` (Settings → Advanced → Uninstall… sheet). Keep switch
  per scope, per-directory model opt-in, byte summary, kept dirs named verbatim;
  models + `cluster/identity.json` kept by default; helper = one admin dialog,
  cancel leaves the rest done; LS hygiene then quit last. XCTests drive temp
  dirs + a recording seam, deleting nothing real.
- Suites: `tests/cluster_assistant` **658 passed**; macOS python packaging
  **27 passed**; full Xcode `xcodebuild test -scheme oMLX CODE_SIGNING_ALLOWED=NO
  -testLanguage en` **441 passed / 0 failed** (5 skipped); `make web` no-op.
- **Still open: the live A5/A6 proof** — it needs a person at each Mac (open the
  app, Local Network, the helper admin dialog, the runtime installer). The
  notarized DMG is rebuilt through `scripts/build-release-dmg.sh --install` and
  rolled out per `CLUSTER-TEST-RUNBOOK.md` §1.5; the five helper states and the
  uninstall scopes are then observed live and recorded in `PHASE-2B-RUN.md`.

### W3 — D4 Error placement (0.5–1 day)

One surface: inline at the failing step, `error.step` + `error.code` + one action,
focus moves there; never only in the advanced `<pre>`; warnings never red.

**Done when:** A7 holds and every backend failure code has a test that it renders
at its step.

**Implementation status (2026-11-16):** implemented as the assistant's `failed`
screen (`data-assistant-screen="failed"`). The failure owns the card, the
primary action is `error.action`, retry re-enters the failed step, the topology
consent and free-space picker render inside it, warnings are amber on the plan,
and the raw payload is collapsed but never the only copy. See
`ASSISTANT-UI-IMPLEMENTATION-PLAN-2026-11-16.md` §7.

### W4 — D2 Steps, back, retry (2–3 days, the big one)

Explicit steps (`measure`, `model`, `plan`, `apply`, `start`, `ready`) with
`done | current | failed | stale`; **back** legal while nothing runs, downstream
becomes `stale`; **Retry this step** with the user's inputs preserved and the
corrected field applied; `ready` keeps today's Screen-6 rules.

**Done when:** A3 and A4 hold, and the card tests cover: model change after a
green plan, corrected apply after a refused tier, back refused while `starting`.

**Implementation status (2026-11-16):** implemented as a client-side step model
on top of `assistantState()`; `choose` is the back screen, `assistantInputsStale()`
derives `stale` from the frozen plan inputs, `assistantBackAllowed()` refuses
while a step runs and at `ready`/`stopped`, and a failed `start` retries in
place with the applied `operation_id`. Fake-driven tests cover model change,
context-only change, stale → re-plan, back refusal and in-place retry.

### W5 — D3 One flow, one page (1–2 days)

Assistant is the flow, top to bottom; the manual wizard collapses behind
`Manual setup`; tabs (if any) follow the flow order. No second wizard.

**Done when:** a screenshot review shows each step once and the manual path one
click away.

**Implementation status (2026-11-16):** implemented. `clusterMode`
(`localStorage('clusterV2.mode')`, default `assistant`) gates today's wizard
blocks unchanged behind `Manual setup`; the assistant card sits above them and
renders exactly one `data-assistant-screen` per state. The manual path stays
visible in Assistant mode only when the assistant cannot render yet (fresh Mac).

### W6 — D5 Manual path proof + live acceptance run (0.5 day + cluster time)

Run §2's A1–A4 on the wiped cluster with the performance probe **on** (so TTFT and
prefill are measured, not derived), one model up manually and the same through the
assistant, plus a corrected-and-retried failure. Record IDs, plan split, endpoint,
rates and the stop in `PHASE-2B-RUN.md` §30.

**Done when:** §2's gate is met end to end, on identical builds.

---

## 4. Small defects already collected (fix inside the package)

- **Helper state invisible** — a raised limit hides the Apply button, which looks
  like a missing feature (W2).
- **Cleanup script left the root-owned helper** — fixed: it now reports it and
  removes it with `--with-helper`.
- **Build products became Launchpad icons** — fixed by
  `scripts/build-release-dmg.sh` (cleans and deregisters before/after packaging).
- **First run decided by `settings.json` only** — fixed: a missing runtime now
  routes to Welcome instead of "server did not start".
- **Slim launcher scripts broke the notarization seal** — fixed (R25).

---

## 5. Deferred (do not start before the gate)

3/4-Mac start, RDMA/JACCL, TP (R2), macOS tray status, privileged network setup
(bridge/STP), telemetry. Screen 6's "re-plan a running cluster" stays out.

---

## 6. After acceptance: the PR path

1. **Fold the branch into the series, once.** The commits on
   `rebase/upstream-main` since the series fork (tier fix, inspection fixes, R5,
   pin backend, pin card, scripts, docs) belong in `feat/cluster-assistant-run`
   (apply/start/plan surface) with `-run-local` rebased on top. Expect conflicts
   in `AGENTS.md`, `models.py`, `routes.py`, `cluster_v2.js` — resolve toward the
   branch's current content (main is newer).
2. **Re-run and record:** `tests/cluster_assistant` + `apps/omlx-web/tests` +
   `tests/` on A3, `make web` (must be a no-op), `xcodebuild test -testLanguage
   en` for the macOS PR.
3. **Update the PR bodies** with the new SHAs and the new counts
   (`PR-BODIES-2026-11-15.md`), and the action vocabulary (17 actions incl.
   `raise_wired_limit`) plus the pin in the body text.
4. **Push order:** F1, F2, F4, F5 → A1 → A2 (draft) → A3 (draft) → optionally F6.
   `#5024` and `#5033` stay as they are (`bbbf18d2`, `25da70de`) until upstream
   moves.
5. **Then** the release DMG for the four Macs (already the artefact of record)
   and the verification note.

---

## 7. Decisions needed before W1/W2 start (ask once, then record in AGENTS)

1. **Back scope:** change model/context after a green **applied** (recommended) or
   only after `plan`?
2. **Retry scope:** a failed **start** may retry in place against the same applied
   deployment (recommended) or must re-apply?
3. **Manual path:** keep the oMLX wizard behind `Manual setup` (recommended) or
   dissolve it into the assistant (larger, touches frozen UI decisions)?
4. **Models on uninstall:** never deleted by default, opt-in per model directory
   (recommended)?
5. **Helper install timing:** Welcome (recommended, one prompt while a person is
   already clicking) or first privileged action only?

---

## 8. Start here in the next session (cold-start handoff)

```sh
cd ~/Documents/PycharmProjects/omlx-cluster-assistant
git status                      # expect: clean, branch rebase/upstream-main
git log --oneline -8            # HEAD includes 1f62ae72 (D6 doc)
scripts/cleanup-cluster-macs.sh --all --with-helper   # wiped four-Mac state
uv run pytest tests/cluster_assistant -q               # expect 658 passed
```

Then: answer §7, build **W1 (D1 uninstall)** — seams first (path/locator +
`PrivilegedHelper.status`), then the sheet, then XCTests, then
`scripts/build-release-dmg.sh --install` and the same DMG on the peers.

**Do not:** run `build.sh` + `package_dmg.sh` by hand (launch `scripts/build-release-dmg.sh`),
delete `cluster/identity.json` without saying so, treat an unmeasured rate as a
rate, or let a step look green when its inputs changed.