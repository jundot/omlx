# Cluster Assistant UI — implementation plan (2026-11-16)

Status: **plan accepted for this session; implementation starts with U1.**
This is the working plan for `ASSISTANT-UI-SPEC-2026-11-16.md` (U1–U6). It
adds no product decision: every rule comes from the spec, `AGENTS.md` §1/§2 and
the 2026-11-15 acceptance decisions.

---

## 0. Baseline (measured in this session)

- Branch `rebase/upstream-main`, working tree clean, HEAD `a23ef6b3`
  (the spec commit).
- Node was missing from `PATH`; a Node 22 binary was placed under
  `$TMPDIR/opencode/node` so the two Node-driven suites actually run instead of
  skipping (74 tests would otherwise skip silently).
- Baseline green: `test_cluster_v2_wizard.py` + `test_cluster_ui_integration_ui.py`
  + `test_cluster_dashboard.py` = **270 passed**.

Verified command:

```sh
export PATH="/private/var/folders/yy/b17vq8697xv9w_xxtfdk3lw40000gn/T/opencode/node/bin:$PATH"
uv run pytest apps/omlx-web/tests/ui/test_cluster_v2_wizard.py \
              apps/omlx-web/tests/test_cluster_ui_integration_ui.py \
              apps/omlx-web/tests/test_cluster_dashboard.py -q
```

---

## 1. What exists (from the code map)

- Template: `_cluster_v2.html` — header 12–17, wizard tiles 53–92, page banners
  97–160, manual state sections 163–833 and 1919–2708, assistant section
  839–1916 (card interior 849–1913, `<!-- /Assistant -->` 1914), toasts 2711.
- JS: one factory `clusterV2Wizard()` in `cluster_v2.js`.
  `assistantState()` :4511, `assistantRunning()` :4515, `assistantReady()` :4529,
  `assistantSurfaceVisible()` :4537, progress rows :174/:4912, error handler
  :7174, `retryAssistantCoordinator()` :7278, `reopenAssistantPicker()` :7312,
  free space :7350, diagnostics :7459, apply/start :6732/:6807, session/resume
  :7468–7581, stop/start-again :7682/:7725.
- No `clusterMode` and no `localStorage` in this component. `assistant.selectedModel`
  + `assistant.inputs` are the frozen plan inputs (`assistantPlanInputs()` :6585).
- Tests pin: one `<select>` in the card, no `<input>`/`x-model`, exactly one
  `data-cluster-v2-assistant-action`, `x-show` and `data-assistant-state` on the
  same start tag, 2 `setInterval(` in the assistant region, no `%` in the card,
  `ready < endpoint < chat < agent`, `stopped < start-again`, the search slice
  before the first `applying` marker, and the Node sandbox has **no**
  `localStorage`/`document`.

---

## 2. Ground rules for every package

1. Screens are a render-mode: `assistantScreen()` reads `assistantState()` and
   returns `start | check | plan | launch | ready | stopped | failed | choose`.
   Every screen div is `data-assistant-screen="<name>" x-show="assistantScreen() === '<name>'"`.
2. Screens are exclusive by construction (equality gating). A Node test asserts
   the mapping for every backend state, including `planning` with and without a
   plan and `ready` + `stopped`.
3. The existing `data-assistant-state` markers keep their exact tag shape. A
   screen wrapper may contain them, never split `x-show` from the marker.
4. `clusterMode` is a plain field (`'assistant' | 'manual'`, default assistant)
   with a guarded `localStorage('clusterV2.mode')` read in `init()` and a
   one-line setter. It is **not** a `$watch`, not a timer, not a wizard state.
5. Manual mode keeps today's markup byte-identical; only wrappers gate it. The
   assistant card stays visible in Manual mode (and stays first; the wizard
   blocks are what "today's tab" means here).
6. No new route, no new poll, no SSE, no number/`%` in the card, one primary
   action per failure, endpoint only under `assistantReady()`.
7. Every new English key goes to `en.json` first, then `make web` mirrors it.

---

## 3. Packages

### U1 — Mode split & skeleton (W5/D3)

- Template: segmented control `Assistant / Manual setup` with
  `data-cluster-v2-mode`, `-mode-assistant`, `-mode-manual`; header + control
  above the assistant section.
- Move the assistant section (839–1916) above the wizard tiles.
- Wrap tiles (53–92) and the manual state sections in
  `x-show="clusterMode === 'manual' || !assistantSurfaceVisible()"` so a fresh
  Mac that cannot show the assistant still gets pairing/install.
- `clusterMode` field + guarded restore + `setClusterMode()`; `start` screen with
  hero, paired-Mac chips (`assistantMacChips()`), entry button (marker kept),
  quiet manual link, and the resume strip (moved here).
- Tests: static markers + Node `setClusterMode` persistence fallback (no
  `localStorage` in the sandbox) + manual section wrappers.

### U2 — Screen render-mode (W5/D3)

- Eight screen divs, in this DOM order (existing index assertions survive):
  `start`, `check`, `plan`, `launch`, `ready`, `stopped`, `failed` and the
  client-only `choose` (between `plan` and `launch` is not allowed because the
  search slice must precede the first `applying` marker — `choose` sits after
  `failed`).
- `check` and `launch` own the stepper + the six `assistantProgressRows()`
  rows; no other panel inside them. The pre-plan `planning` spinner stays in
  `check`; `plan` renders only when `assistant.plan` exists.
- `plan` owns recommendation, kv/chips, tradeoffs, warning notes, Start setup,
  the quiet affordances and the collapsed details.
- `ready` owns endpoint/copy/chat/agent/refresh/stop, measured line, and the
  collapsed **Performance** section (probe toggle, tuning, tune-redeploy).
- `stopped` owns Start again (marker order preserved).
- One progress container per async screen is allowed; both render the same
  helper (single truth stays `assistantProgressRows()`).
- Tests: `assistantScreen()` mapping (Node), one screen div per name (static),
  no dual progress, Screen-4 row rules unchanged.

### U3 — Error screens (W3/D4)

- `failed` becomes the card: full-width red banner (title from
  `assistantFailureTitle()` by code, cause `assistantErrorMessage()`), **one**
  primary `data-cluster-v2-assistant-action` bound to `assistantActionHandler`,
  secondary Back / Plan again / Show details, and the branch affordances
  (coordinator instruction, topology consent, free-space picker) inside it.
- Warnings stay amber and never set `failed`; page banners stay page-level.
- Details: the full evidence block lives in `plan`; the failed screen adds a
  compact collapsed raw view via `assistantFailureDetails()` (no duplicate
  operation marker).
- Tests: every `error.action` renders at the failed screen; one action; no
  other screen's primary marker inside it.

### U4 — Back, choose, stale (W4/D2)

- `choose` screen: candidate buttons from `assistantCandidates()`, context
  ladder buttons (`assistantChooseContextOptions()`), current selection marked,
  `← Back to plan`, primary `Continue with {model}`.
- Confirm: model changed → `runAssistantModelMeasure` (model-bound, hardware
  measurement reused); context only → `runAssistantPlan`; nothing changed →
  close. `assistantChooseOpen` and a draft object keep the card's frozen inputs
  untouched until confirm.
- `assistantInputsStale()` compares `assistant.inputs` with the current
  selection; `assistantStepStale()` marks the plan + downstream rows stale.
  The plan screen shows the stale banner; Start setup re-plans first when
  stale.
- Back scope: `assistantBackAllowed()` is false while `assistantRunning()` or
  when ready/stopped. Back from a failure returns to the plan review
  (`assistantBackToPlan()`), never a new backend state.
- Tests: stale after model change, no re-measure on context change, back refused
  while `starting`, choose has no input/select.

### U5 — Retry in place (W4/D2)

- `retryAssistantStep()` re-enters the failed step: `stale_measurement` →
  re-measure; `plan`/`raise_wired_limit` → `runAssistantPlan()` with the same
  measurement/inputs; `apply` → apply; `start`/`retry_coordinator` with an
  applied operation → `runAssistantStart()` (same `applyOperationId`, no
  re-apply); near-miss actions open `choose` pre-filtered.
- The primary failure button keeps `assistantActionHandler(error.action)`; the
  secondary "Plan again"/"Retry start" call `retryAssistantStep()`.
- Tests: corrected failure retried in place (start body carries the same
  `operation_id`), no re-apply, placement refusal still authoritative.

### U6 — Ready/Stopped polish (W5/D3)

- Performance collapse, measured line only with `observed`, copy/agent links,
  stop/start-again, `assistantChips()` for rank/tier/MTP, fit headline from
  `assistantFitHeadline()` (min stage headroom), resume summary line.
- Screenshot review against the mock; `make web` no-op; full two suites green.

---

## 4. Subagent split (used in this session)

| # | Agent | Scope | Files |
|---|---|---|---|
| 1–4 | explore (done) | template map, JS map, test map, i18n/mock copy | read-only |
| 5 | i18n | add the frozen English key set, run `normalize_i18n.py`, verify mirror test | `i18n/*.json` |
| 6 | reviewer | independent review of the finished diff against the spec §0 rules + test contract | read-only |

The template and `cluster_v2.js` are edited only by the lead (single writer), so
no two agents ever touch the same file.

---

## 5. Test contract (what must stay true)

- One `<select>` (coordinator), no `<input>`, no `x-model` in the card.
- Exactly one `data-cluster-v2-assistant-action`.
- `x-show="assistantState() === '<state>'"` immediately precedes
  `data-assistant-state="<state>"` on the same tag.
- `setInterval(` count 2 in the assistant JS region; no `Date.now` in progress
  helpers.
- No `%`, no rate, no ETA in the card.
- `data-cluster-v2-assistant-progress` is a `<div>`; every spinner is
  `aria-hidden="true"`.
- Search slice stays before the first `data-assistant-state="applying"`.
- `ready < endpoint < chat < agent`, `stopped < start-again`.
- `data-cluster-v2-assistant>` and `<!-- /Assistant -->` keep their literal
  form (the static helper slices on them).
- i18n: every `cluster.assistant.*` literal exists in `en.json`; all locales
  mirror the key set and the English values.

---

## 6. Verification (after every package)

```sh
export PATH="$TMPDIR/opencode/node/bin:$PATH"   # or the path from §0
uv run pytest apps/omlx-web/tests/ui/test_cluster_v2_wizard.py \
              apps/omlx-web/tests/test_cluster_ui_integration_ui.py \
              apps/omlx-web/tests/test_cluster_dashboard.py -q
cd apps/omlx-web && python build_css.py
cd ../.. && make web        # second run must be a no-op
uv run pytest tests/cluster_assistant -q   # backend untouched, must stay green
```

Definition of done: the U1–U6 tests are green, the two suites and the dashboard
gates pass, `make web` is a no-op, the backend suite is unchanged, and the
reviewer finds no violation of spec §0.

---

## 7. Implementation log (2026-11-16)

Landed in one working session on `rebase/upstream-main` (uncommitted):

| Package | What landed |
|---|---|
| U1 | `clusterMode` (`assistant` default) + guarded `localStorage('clusterV2.mode')`; segmented control at the top of the tab; assistant section moved above the wizard tiles; manual tiles, activation bar and all manual state sections wrapped in `manualSectionsVisible()`; start screen with hero, paired-Mac chips, resume strip, entry and `Manual setup` link. |
| U2 | Eight `data-assistant-screen` divs (`start/check/plan/launch/ready/stopped/failed/choose`), gated by `assistantScreen()`; compact 6-chip stepper over `assistantProgressRows()`; check/launch own the card; plan owns recommendation/kv/chips/tradeoffs/warnings/Start/choose/details; ready owns endpoint + collapsed Performance; stopped owns Start again. |
| U3 | Failed screen owns the card: `assistantFailureTitle()`, one primary `error.action`, secondary retry/back/change, topology + free-space inside, collapsed raw payload, amber-only plan warnings. |
| U4 | `choose` screen (candidate + context buttons), `assistantInputsStale()`, `assistantRowStale()`, stale banner + chip, `assistantBackAllowed()`, `assistantBackToPlan()`; Start setup re-plans a stale plan. |
| U5 | `assistantFailureStep()` + `retryAssistantStep()`: measure/plan/apply/start re-entry; an applied deployment restarts in place with the same `operation_id`. |
| U6 | Performance collapse (probe + tuning + redeploy), fit headline from stage headroom, rank/tier/MTP chips, measured-ago chip from the backend's `measured_at`, copy/agent/stop/start-again unchanged. |

New tests: 6 static in `apps/omlx-web/tests/ui/test_cluster_v2_wizard.py`
(screens, mode split, choose, failure ownership, back/retry/stale, copy keys)
and 13 Node-driven in `apps/omlx-web/tests/test_cluster_ui_integration_ui.py`
(screen mapping, choose re-measure rules, stale through a real choose, back
refusal, in-place start retry, catalogue-before-measure, pairing mode switch,
diagnostics keeps the failure, one primary action, failed-screen branch
affordances, session endpoint `ready` gate, the full error-action matrix, and
warnings never colouring a row).

Verification actually run:

```text
apps/omlx-web/tests/ui/test_cluster_v2_wizard.py
apps/omlx-web/tests/test_cluster_ui_integration_ui.py
apps/omlx-web/tests/test_cluster_dashboard.py   → 289 passed
tests/cluster_assistant                          → 658 passed
node --check cluster_v2.js                       → clean
build_css.py + normalize_i18n.py                 → tailwind.css + 10 locales updated
```

Recorded deviations (all UI-only, no backend change):

1. The assistant card stays at the top in **both** modes; Manual mode keeps the
   wizard blocks byte-identical below it rather than restoring the old scroll
   position of the card. This is the "one flow, one page" reading of §6.
2. The probe toggle moved into the ready Performance section; it is no longer
   visible on decision screens (spec §2.4).
3. The freshness chip only exists when the backend supplied `measured_at`; no
   age is invented, and a model-bound re-measure clears the old timestamp.
4. There is no resting "applied" screen in the MVP flow (apply auto-starts), so
   "back up to a green applied" is exercised through the failed-start retry and
   the choose screen, which is where the frozen inputs are actually changeable.
5. `Run on this Mac` remains the existing local-run panel (gated by the
   recommendation's `single_node` answer), not a second link.
6. `choose` always finishes its job (re-measure → plan, or context-only plan), so
   the amber `stale` state is visible **while that job runs** (stepper rows +
   header chip) rather than as a resting screen. A resting stale plan would
   contradict §2.6; the comparison itself is complete (model, context, pin,
   node selection, MTP, execution profile, measurement id, 2-Mac consent).
7. The constellation search panel now lives on the `plan` screen, so it is not
   reachable while `planning` has no plan yet. Spec §2.2 asks exactly that; the
   earlier pre-plan entry point is gone on purpose.
8. A `pair_mac`/`add_mac` action switches to Manual setup before opening the
   existing pairing surface — that surface is the manual region and would
   otherwise be hidden in Assistant mode.