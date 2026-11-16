# Cluster Assistant UI — implementation spec and roadmap (2026-11-16)

Status: **ready for implementation.** This turns the approved mock
(`mockups/assistant-flow-klickstrecke.html`) and its design doc
(`UX-FLOW-REDESIGN-2026-11-15.md`) into code. It is the UI half of the
acceptance package: **W3 (D4 error placement) + W4 (D2 steps/back/retry) +
W5 (D3 one flow, one page)** in `ROADMAP-ACCEPTANCE-2026-11-16.md`.

Read order for the implementing agent:

1. `AGENTS.md` §1 (frozen), §2 (resolved), the acceptance decisions (§10).
2. `UX-FLOW-REDESIGN-2026-11-15.md` — the why and the screen-by-screen table.
3. The mock — the visual contract (English strings in it are the product copy).
4. This spec — the code-level contract and the build order.

---

## 0. One sentence, and the hard rules

**Goal:** the Cluster tab becomes one vertical flow — the assistant card renders
exactly one screen per backend state, errors own the card, back reaches a green
`applied`, and the manual wizard collapses behind a `Manual setup` mode.

Nothing here may:

- add a backend route, a backend state, SSE/streaming, or a second wizard;
- change `assistantState()`'s state set (`idle → measuring → recommending →
  measuring_model → planning → applying → starting → loading → ready`, plus
  `failed`) — screens are a **render-mode** on top of it;
- show an endpoint before `assistantState() === 'ready' && endpoint.ready`;
- promise a token rate (no number before `observed` exists);
- let browser JS launch, deploy, SSH, stage or supervise anything;
- render more than **one** primary action for a failure (`error.action`);
- add German strings (product UI is English only; the mock's German
  annotations are review notes, not copy).

---

## 1. Architecture (render-mode rules)

1. Everything stays inside `clusterV2Wizard()` and the existing Cluster tab.
   No new component, no new `WIZARD_STATES` entry, no second tab.
2. The assistant card gains one `<div data-assistant-screen="…">` per screen
   (values in §2). Each screen div is gated by `assistantState()` exactly like
   today's `data-assistant-state` blocks — and the existing
   `data-assistant-state` markers **stay** (the static contract tests keep
   working; the screen div is an additional, coarser gate).
3. Exactly one screen div is mounted per state — the card never stacks two
   screens. A static test asserts `card.count('data-assistant-screen=') ==
   expected mounted count` per state.
4. The mode split is one Alpine field on the wizard component:
   `clusterMode: 'assistant' | 'manual'` (default `assistant`, persisted in
   `localStorage('clusterV2.mode')`). Manual mode keeps today's wizard sections
   exactly as they are — byte-identical template blocks, only gated.
5. The 6-step stepper is a small render helper over `assistantProgressRows()`
   (cluster_v2.js:4912) — same rows, same `cluster.assistant.progress.*` keys,
   no second truth. It replaces the *assistant's* progress rows on Check and
   Launch; the wizard's 5-step tiles are hidden in Assistant mode and stay in
   Manual mode.
6. No new polling. The existing single-flight poll loops (measurement +
   operation, generation counter) drive everything; screens re-render from the
   same state they already update.

---

## 2. Screen inventory and state mapping

| Backend state / condition | Screen (`data-assistant-screen`) | Primary action | Mock ref |
|---|---|---|---|
| `idle` (no session) | `start` | Find and start a setup | S1 |
| `idle` with a ready/stopped session (`GET /assistant/session`) | `start` + **resume strip** | Stop / Start again | S1 strip, B5 |
| `measuring`, `recommending`, `measuring_model`, `planning` (no `assistant.plan`) | `check` | — (async) | S2, B4a |
| `planning` with `assistant.plan` | `plan` | Start setup | S3 |
| `applying`, `starting`, `loading` | `launch` | — (async) | S4 |
| `ready` && `endpoint.ready` | `ready` | Open Chat | S5 |
| `ready` && `stopped` | `stopped` | Start again | B5 |
| `failed` (any step) | `failed` | from `error.action` | B1–B3 |
| client-side back affordance from `plan`/applied | `choose` | Confirm selection | B4 |

`choose` is the only screen that is not driven by a backend state: it is a
client-side picker mounted by the "Change model or context" affordance (§3).

### 2.1 `start` (S1)

- **Resume strip first** (Phase C1): status chip (Running/Stopped), the frozen
  plan inputs in one line (model · Macs · context), the endpoint link (only
  when not stopped), `Stop`. Data: `GET /assistant/session` (already exists).
  A reload lands here with the strip — nothing else must be hunted.
- Hero: eyebrow `Assistant`, title "Set up this cluster", one-sentence sub.
- **Paired Macs as compact chips** (name · chip · RAM · green dot, plus
  combined-RAM and "measured N min ago" chips when a fresh measurement
  exists). The full device-card grid moves to Manual mode.
- Flowline: primary **Find and start a setup** (today's entry button,
  `data-cluster-v2-assistant-entry` marker stays), quiet link
  **Manual setup — full wizard, device cards and plan form** (sets
  `clusterMode = 'manual'`).

### 2.2 `check` and `launch` (S2/S4 — identical layout)

- Card = eyebrow + phase label + **stepper** + the six progress rows. **No
  other panel is mounted** (no plan, no tuning, no search — P3/P7).
- The active row's detail is the backend's own `progress.message` and
  `{n} of {m} Macs ready` — a row may never animate a phase the backend did
  not report, no percentages, no byte counts, no rates (Screen-4 rules).
- `launch` differs from `check` only in which rows are active (operation kind
  + `progress.phase`), exactly like today's `assistantProgressRows()`.

### 2.3 `plan` (S3 — one decision, one CTA)

- **Recommendation panel**: eyebrow `Recommended`, model name, fit headline
  ("Fits with 21 GiB headroom — no quality tradeoff needed."), `Fits` chip.
  kv grid: Macs (n of m paired), Context, Coordinator, Offload. Chips: rank
  order per rank (`rank 0 · <Mac> — tail + sampler`), `Tier <t> · <p> admission`,
  MTP state. Caption: **"Speed stays unknown until the running cluster has
  been measured. No rate is promised here."** — always, never a number.
- **Tradeoff block** (only when the plan carries `tradeoffs`): the frozen
  2026-11-14 context-vs-experts decision. Two buttons — the quality-first
  plan (default, already chosen) and the experts-resident `label: speed`
  variant. Choosing a variant swaps the card's `resource_policy` +
  `plan_signature` inputs for apply (the backend already takes them).
- Amber notes for plan warnings (`tier_raised`, `macs_not_needed`, …) — never
  red, never an error.
- Collapsed `<details>`: stages, rank order, plan signature, measurement id,
  warnings, and the quiet affordances **Copy diagnostics**, **Network
  preflight**, **Use all Macs**, **Pin rank 0…** (the coordinator `<select>`
  with marker `data-cluster-v2-assistant-coordinator` — the only `<select>`
  allowed in the card).
- Flowline: secondary **Change model or context** (→ `choose`, §3), quiet
  **Run on this Mac** (only when the recommendation's `single_node` answer
  names this Mac — decision E6; otherwise absent), primary **Start setup**
  (`runAssistantApply()` — today's wiring, marker
  `data-cluster-v2-assistant-start-setup` stays).

### 2.4 `ready` (S5) and `stopped` (B5)

- Endpoint block: link, copy, **Open Chat**, **Connect agent**, bounded
  **Refresh** (re-reads the operation once — SCREEN-5 rules). URL renders only
  under `assistantReady()` (`ready` && `endpoint.ready` && not stopped), never
  a port-0 URL.
- **Stop cluster** (danger outline, right) → `POST …/assistant/stop` with the
  ready start operation's ID (Screen 6).
- Measured line **only** when `observed` is present (prefill/decode/TTFT);
  otherwise the card says nothing about speed.
- `stopped`: the same screen without the endpoint block, plus **Start again**
  (re-applies the frozen plan inputs — signature re-proved, `plan_mismatch`
  possible — and lets the existing poll start it).

### 2.5 `failed` (B1–B3) — see §5

### 2.6 `choose` (B4)

- Reached only via "Change model or context" from `plan` (or an applied-before-
  start state, §3). Shows the recommendation's candidate list (models with
  sizes/contexts from the measured facts) + the context ladder for the chosen
  model, and the current selection marked.
- Confirming: if the **model** changed → model-bound re-measure
  (`measuring_model` screen) then plan; if only the **context** changed →
  plan only. Hardware measurement is reused (TTL rules are the backend's).
- Back-link to the plan review without changes. `x-show`-level state only; the
  card's frozen inputs object is updated on confirm, never before.

---

## 3. Step model, back, stale (D2 — W4)

- The card derives a **step model** from `assistantState()`, the operation
  kind/state and the failure point: steps `measure`, `model`, `plan`, `apply`,
  `start`, `ready`, each `done | current | failed | stale`. This is client-side
  derivation — **no backend state is added**; `stale` is a UI flag.
- **Stale marking:** changing the model or context (via `choose`) after a
  computed plan/applied marks every downstream step `stale` — visibly, never
  silently kept. The plan screen then shows a "inputs changed — re-plan"
  state, and Start setup re-plans first (the backend's strict signature match
  already enforces this; the UI just stops hiding it).
- **Back scope (answered decision):** back reaches up to a green **`applied`**.
  While anything is running (`applying/starting/loading` or a poll in flight),
  the affordance is refused with a one-line reason. A running (`ready`)
  cluster stays frozen — Screen-6 rules.
- **What is preserved on back:** both measurements (subject to the backend's
  15-minute TTL refusal, which surfaces as `stale_measurement` → re-measure),
  settings, the pin. **What re-runs:** exactly the affected step (§2.6).
- `failed start` retry stays in place against the **same applied deployment**
  (§4). Back from other failures preserves everything before the failed step.

## 4. Retry semantics (D2 — W4)

- The `failed` screen's primary action **re-enters at the failed step**, never
  at the top:
  - `raise_wired_limit`: the action names the step (app-side Performance →
    Apply with Admin…); when the user returns, the retry action ("Plan again")
    re-posts **plan** with the same measurement + inputs (hardware measurement
    still fresh → no re-measure, model-bound unchanged → no re-measure).
  - `retry_coordinator` re-posts the failed stage (start if an applied
    operation exists, else apply) with the same frozen plan.
  - `stale_measurement` re-measures; `insufficient_disk` shows the free-space
    picker only when the backend's automatic free-retry refused; the
    near-miss actions (`choose_smaller_model`, …) open `choose` pre-filtered.
- The mock's interstitials (B1b, B3b) are **demo visualizations**, not polling:
  the product never watches for the user to fix something outside the browser.
  The failed screen stays until the user clicks the retry action.
- A failed `start` retry validates the placement first (the backend's
  `placement_changed` refusal stays authoritative) and then starts in place —
  no forced re-apply (answered decision).

## 5. Error screens (D4 — W3)

1. A failure **becomes the card's content** (`data-assistant-screen="failed"`):
   red banner (title + plain-language cause naming the Mac/number involved) +
   **exactly one primary action** from `error.action` + secondary links
   (**Back** to the previous decision, **Details** collapsed with the raw
   payload + Copy diagnostics). No other panel is mounted.
2. Branch affordances render **inside** the failed screen, directly under
   their cause: the `topology_incomplete` 2-Mac consent (salting the plan
   signature, the existing mechanism), the free-space picker, the
   `choose` pre-filter.
3. No failure text may exist **only** in the advanced `<details>` or only as
   a toast. Page-level banners stay for page-level facts (version mismatch,
   discovery) and are not assistant errors.
4. Warnings (`tier_raised`, `macs_not_needed`, `performance_unmeasured`) stay
   amber and never colour a step red.
5. The local-run (`run-local`), tune and search flows follow the same rule
   when their flow owns the card.

## 6. Layout and mode split (D3 — W5)

- The assistant section moves **above** the device-card section in
  `_cluster_v2.html`; in Assistant mode the wizard tiles and the state
  sections are hidden. The Cluster tab then reads top-to-bottom:
  `Assistant card → (Manual setup: everything of today)`.
- The segmented control **Assistant / Manual setup** sits at the top of the
  tab (mock S1; one Alpine field, §1.4). Manual mode is today's tab unchanged.
- CSS: reuse the mock's tokens (white cards `border-neutral-200
  rounded-2xl`, `bg-neutral-900 rounded-full` primary buttons, amber/red/green
  banners, blue links, `text-caption` eyebrows) — they are the dashboard's
  existing language; no new stylesheet, extend `_cluster_v2`'s classes.

## 7. i18n

- Reuse every existing `cluster.assistant.*` key (actions, progress rows,
  error copy). New English keys (mirrored into all locales by
  `normalize_i18n.py` via `make web`; the catalog test asserts the mirror):
  screen titles/subs (`cluster.assistant.screen.start|check|plan|launch|
  ready|stopped|failed|choose.*`), mode labels (`cluster.assistant.mode.
  assistant|manual`), back/stale copy (`cluster.assistant.back.*`,
  `cluster.assistant.stale.*`), resume strip (`cluster.assistant.resume.*`).
  Copy source: the mock's English strings.
- No German. The mock's German side notes stay in the mock file only.

## 8. Tests

- **Static contract** (`apps/omlx-web/tests/ui/test_cluster_v2_wizard.py`,
  `_assistant_card()`): per state exactly the expected screen div is present;
  the card contains at most the one coordinator `<select>` and no `<input>`/
  `x-model` on the primary path; primary-action markers
  (`data-cluster-v2-assistant-entry`, `-start-setup`, `-stop`, …) survive.
- **Alpine integration** (`apps/omlx-web/tests/test_cluster_ui_integration_ui.py`,
  fake seams): every backend state renders its screen; a failure replaces the
  card content with one action; model change after a green plan → plan marked
  stale → Start setup re-plans; context-only change → no re-measure; back
  refused while `starting`; failed `start` retries in place against the
  applied operation; stop/start-again and the resume strip after a reload;
  endpoint visible only when ready.
- Every case is fake-driven (existing fixtures); a real Mac, model or endpoint
  never appears in the suite (AGENTS: live evidence lives in PHASE-2B-RUN.md).

## 9. Roadmap

Build order (each package lands green before the next; sizes are focused
days):

| # | Package | Maps to | Size | Done when |
|---|---|---|---|---|
| U1 | Mode split & skeleton: `clusterMode` segmented control, assistant section above the device cards, wizard tiles/state sections hidden in Assistant mode, resume strip on `start` (`/assistant/session`) | W5/D3 | 0.5 | Entry reachable without scrolling; Manual mode byte-identical; static + Alpine tests for the split and the strip |
| U2 | Screen render-mode: one `data-assistant-screen` div per state, exactly one mounted; stepper over `assistantProgressRows()`; Check/Launch own the card | W5/D3 | 1 | screen-per-state tests green; no dual progress anywhere; Screen-4 label rules unchanged |
| U3 | Error screens: `failed` owns the card, one primary action, branch affordances inside, warnings amber, nothing only in advanced/toast | W3/D4 | 0.5–1 | the failure-code matrix (every `error.action`) renders at its step; A7 holds |
| U4 | Back & stale: `choose` screen, downstream `stale`, model-bound re-measure only on model change, back up to `applied`, refused while running | W4/D2 | 1–1.5 | card tests: stale after model change, no re-measure on context change, back refused at `starting`; A4 holds |
| U5 | Retry in place: failed `start` retries against the applied deployment; every retry action re-enters its step | W4/D2 | 1 | card test: corrected failure retried in place, no re-apply; A3 holds |
| U6 | Ready/Stopped polish: Performance collapse, measured line only with `observed`, copy/agent links, stop/start-again | W5/D3 | 0.5–1 | screenshot review against the mock; hands over to W6 (the live acceptance run) |

Definition of done for the whole UI package: A3, A4 and A7 from the acceptance
gate hold in the fake-driven suites, `make web` is a no-op, and a screenshot
review shows each mock screen once, top to bottom, with the manual path one
click away.

## 10. Handoff notes for the implementing agent

- Start from `cluster_v2.js`: `assistantState()` (:4511), `assistantReady()`
  (:4529), `assistantProgressRows()` (:4912), the error-action switch (:7194
  ff.), `reopenAssistantPicker()` (:7312), `openAssistantFreeSpace()`
  (:7350), `openAssistantDiagnostics()` (:7459), `runAssistantApply()`
  (:6732), `runAssistantStart()` (:6807), `setAssistantCoordinator()` (:6637).
- The template is `templates/dashboard/_cluster_v2.html` (11
  `data-assistant-state` markers today — keep them all).
- After template changes: `cd apps/omlx-web && python build_css.py`, then
  `make web` (must be a no-op), then the wizard + UI suites.
- If a step seems to need a new backend state or route: stop and re-read §0.
  The render-mode is the whole trick.
