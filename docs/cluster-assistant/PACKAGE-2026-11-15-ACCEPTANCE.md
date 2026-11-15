# Package 2026-11-15 — Acceptance gaps: uninstall, assistant workflow, navigation, errors

Status: **collecting (user request 2026-11-15)**. Build order is top-down; nothing
here re-opens a frozen product decision (brief §7–§10, AGENTS §1). Where a fix
would, it says so and asks once.

Why this package exists: the backend is green (two Macs start, serve and stop;
R5 and the rank-0 pin are in), but the **product** is not acceptable yet. The
user's own words:

> „zuerst unser deinstall button, dann ist der workflow vom assistent shit,
> manuell habe ich noch kein model zum laufen gebracht. Es gibt auch kein
> wirkliches zurück um schritt zu ändern die schon grün sind, oder noch
> schlimmer, was rot ist kann nicht wirklich korrigiert und nochmal durch
> geführt werden … Auch der workflow von oben nach unten in den tabs ist komisch
> … fehlermeldungen kommen auch da wo ein user sie nie erwarten würde"

Each item below is written so it can be closed with a test, not an opinion.

---

## D1. Uninstall — one button, explicit keep options

**Symptom.** Removing oMLX means a shell script today. A person cannot uninstall
the app they installed with one dialog, and the leftovers (helper, runtime,
`~/.omlx`, build copies, Launchpad icons) are exactly what a support case cannot
explain.

**Decision (already recorded in AGENTS §10).** The app uninstalls itself:
Settings → Advanced → **Uninstall…**, one sheet, every scope with its own keep
switch. Models are **never** removed unless the user opts in per directory.

**Scopes and defaults**

| Scope | Path | Default | Admin |
|---|---|---|---|
| App | `/Applications/oMLX.app` (move to Trash, then quit) | remove | no |
| Helper | `/Library/PrivilegedHelperTools/app.omlx.helper` + `/Library/LaunchDaemons/app.omlx.helper.plist`, `launchctl bootout` | remove | **yes, one dialog** |
| Runtime | `~/Library/Application Support/oMLX` (venv, logs, updater state) | remove | no |
| Settings | `~/.omlx/{settings.json,model_settings.json,assistant-*.json,logs}` | remove | no |
| Cluster state | `~/.omlx/cluster/*` | remove | no |
| Node identity | `~/.omlx/cluster/identity.json` | **keep** | no |
| Models | the model dirs the user chose (`~/Models`, …) | **keep**, opt-in per directory | no |
| Diagnostics | `~/Library/Logs/oMLX`, support bundles | remove | no |

**Behaviour**

- A summary list of what will be deleted with byte counts before anything is
  removed; a "what is kept" line that names the model directories verbatim.
- The helper step is the only admin prompt; if the user cancels it, the rest is
  still done and the sheet says the helper was left in place.
- The app quits last (moving itself to the Trash while running is the documented
  path; relaunching from the DMG restores).
- Launchpad/LS hygiene is part of it: after removal, unregister the removed
  bundle (`lsregister -u`), then `lsregister -gc -f -r`, then repaint the Dock.
  A missing icon is part of "uninstalled", not cosmetics.
- `scripts/cleanup-cluster-macs.sh` stays the lab/CI equivalent, same scope set.

**Helper state must be visible.** A person cannot tell whether the helper is
installed, which wired limit is in force, or why no button shows. The
Performance screen gets one line — `Helper: installed (vX) · wired limit
122 880 MB of 122 880 MB requested` — and the Apply button stays hidden only
when the kernel cap already meets the request. Today a raised limit hides the
button, which is correct but indistinguishable from "the feature is missing".

**Acceptance.** XCTest drives the model layer without deleting anything real
(paths injected): every default removes exactly its scope, models survive, the
identity survives, and a cancelled helper step leaves the rest done. On a real
Mac: uninstall → 0 app bundles, 0 LaunchServices records for oMLX, no helper, no
`~/.omlx` except `cluster/identity.json`, models untouched.

---

## D2. The assistant workflow cannot go back and cannot recover

**Symptoms (from the user, to be reproduced in the card).**
1. A **green** step cannot be changed: once measured/planned/applied, the only
   way to change the model or the context is to abandon the card.
2. A **red** step cannot be corrected and re-run: after a failure the card
   offers one action, and several of them leave the card in a state where the
   same failure repeats.
3. Manual flow: **no model was ever brought up by hand** end-to-end. Until that
   works, the assistant is the only path and the manual path is untested
   product.

**Root causes in the current design**

- `assistantState()` is a single forward chain
  (`idle → measuring → … → ready | failed`); it has no notion of a *step* that
  can be re-entered. The card derives everything from the state, so "go back"
  has nowhere to live.
- Apply/start are **operations** with terminal states (`applied`, `ready`,
  `failed`); the record is forward-only by design, and the card mirrors that.
  Re-planning a running cluster is explicitly out of scope (Screen 6), which is
  correct — but *not* the same as "you may not change the model before start".
- The failure path has one `error.action` and the handler for it decides where
  the user lands (`reopenAssistantPicker`, `openAssistantDiagnostics`,
  `openAssistantFreeSpace`). A corrected input then has to travel through the
  whole chain again, and some actions only reset the card instead of resuming
  the step that failed.

**Proposed shape (needs one decision, see §8)**

- **Steps, not a chain.** Give the assistant explicit steps with stable ids:
  `measure`, `model` (recommendation + model-bound measurement), `plan`,
  `apply`, `start`, `ready`. Every step has `done | current | failed | stale`.
- **Back is legal while nothing is running.** From any state before `start` the
  user may return to `model` or `plan` and change the model, context, tier,
  offload or the pinned coordinator; everything downstream becomes `stale`, not
  silently kept, and re-running a step re-derives the rest.
- **Retry is per step.** `error.step` names the step that failed; the card keeps
  the inputs the user already chose, lets them correct exactly the field the
  error is about, and offers **Retry this step** (not "continue" and not a jump
  to diagnostics). Diagnostics stays the secondary action.
- **Running cluster stays frozen.** `ready` keeps today's Screen-6 rules
  (stop / start again, no silent re-plan). "Back" is a pre-start affordance.
- **Manual flow is a first-class path again**, not a fallback: the existing
  manual wizard must be able to bring one model up, measured from a wiped state,
  with the same evidence as §29 of `PHASE-2B-RUN.md`. If it cannot, that is a
  bug in this package, not a documentation gap.

**Acceptance.** Card-level tests (fake seams): change the model after a green
plan → plan is `stale` and re-plans with the new model; a failed apply keeps the
chosen model/context and a corrected tier re-applies successfully; "back" from
`applied` is possible, from `starting` is refused with a reason. Live: one model
up through the **manual** path and one through the assistant, from a wiped state.

---

## D3. Top-to-bottom workflow across the tabs is confusing

**Symptom.** The Cluster tab carries the existing oMLX wizard chain *and* the
assistant card, each with its own notion of "next", and the user cannot tell
which one is the workflow, which one is a leftover, and in what order to read
them.

**Proposed shape.** One page, one vertical flow: the assistant is the primary
path (measure → model → plan → apply → start → ready) with the manual wizard
collapsed behind a single `Manual setup` disclosure for the cases the assistant
does not cover (pairing without a measurement, a Mac that is not answering, a
model that never fits). Tabs, if any remain, must be ordered by the flow and
never require reading them out of order. No second wizard, no second control
plane — this is a layout change inside the existing Cluster tab.

**Acceptance.** A screenshot review against the screen order: every step appears
once, top to bottom, and the manual path is reachable in one click without ever
being the only way forward.

---

## D4. Errors appear where a user would never expect them

**Symptoms.** Actionable failures surface in places a person does not look:
inside the advanced JSON preview, as a toast, at the bottom of a collapsed
`<details>`, or attached to a step other than the one that failed.

**Rules to implement**

- **One surface:** a failure is rendered inline at the step that failed, at the
  position the user's eye is (the step header), never only in the advanced
  block and never only as a toast.
- **Errors carry a step.** `error.step` + `error.code` + one action; the card
  scrolls/focuses that step when it appears.
- **No error in the advanced JSON.** The `<pre>` is a debugging aid; it may never
  be the only place a failure is visible.
- **Warnings are not errors.** `tier_raised`, `macs_not_needed`,
  `performance_unmeasured` stay informational and must not colour a step red.
- **Refusals explain themselves.** Every message names the Mac, the number and
  the one next step (R23/R5 already do; keep it as a rule).

**Acceptance.** UI tests assert that every backend failure code renders at the
failing step with its action, and that no error text exists only inside the
advanced preview.

---

## D5. Package order and effort

| # | Item | Depends on | Rough size |
|---|---|---|---|
| 1 | D1 uninstall (button, sheet, tests, DMG) | — | 1 day |
| 2 | D4 error placement (rules + rendering) | — | 0.5–1 day |
| 3 | D2 steps/back/retry (state machine + card) | D4 | 2–3 days |
| 4 | D3 layout (one flow, manual behind a disclosure) | D2 | 1–2 days |
| 5 | Manual path proof (live, wiped state) | D2/D3 | 0.5 day + cluster time |

Shipping gate for the package: D1–D4 green in the fake-driven suites, and a live
run from a wiped four-Mac state that shows (a) one model up manually, (b) the
same model up through the assistant, (c) a failure that is corrected and retried
in place, and (d) an uninstall that leaves models and identity behind.

---

## §8. Decisions this package needs (ask once)

1. **D2 back-scope.** Is "go back and change the model/context after a green
   plan" allowed up to `applied` (recommended: yes, downstream becomes `stale`),
   or only up to `plan`?
2. **D2 retry-scope.** May a failed `start` be retried **in place** with the same
   applied deployment after the cause is fixed (recommended: yes, re-validate
   the placement and start), or must it re-apply?
3. **D3 manual path.** Keep the existing oMLX wizard as a collapsed `Manual
   setup` (recommended), or fold its remaining steps into the assistant and
   delete the wizard? (The second is a larger change and touches frozen UI
   decisions.)
4. **D1 model deletion.** Confirm models are never deleted by the uninstall
   default (recommended), i.e. the option is per directory and off.