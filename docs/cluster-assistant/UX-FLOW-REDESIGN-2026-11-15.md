# Cluster Assistant — UX flow redesign (accepted, 2026-11-15)

Status: **accepted — the implementation spec exists. No production code
changed yet.** The mock and this document are the approved design; the
code-level contract and build order live in
[`ASSISTANT-UI-SPEC-2026-11-16.md`](ASSISTANT-UI-SPEC-2026-11-16.md) (W3/W4/W5
of the acceptance roadmap).

Deliverables:

- This document — workflow audit, potentials, concrete proposals.
- Clickable mock (Klickstrecke): `docs/cluster-assistant/mockups/assistant-flow-klickstrecke.html`
  (self-contained HTML, open directly in a browser; product UI English, review
  annotations German).

---

## Zusammenfassung (DE)

Der Cluster-Tab vereint heute zwei Produkte in einer Seite: den exo-artigen
5-Schritt-Wizard (Pairing, Checks, manuelles Plan-Formular) und den Assistant
(theoretisch der „one-action path“). Der Assistant-Einstieg liegt dabei unter
4–6 Bildschirmhöhen Wizard-Chrome, und die Assistant-Karte selbst ist ein
vertikaler Stack aus ~10 Panels (Entry, Progress, Plan, Local Run, Search,
Tuning, 5 verschiedene Fehlerboxen, Details), der während des Flusses weiter
nach unten wächst. Wer „nur schnell ein Modell starten“ will, scrollt oben
nach unten, ohne zu wissen, was der nächste Schritt ist.

Vorschlag (siehe Mock):

1. **Zwei Modi im selben Tab** — Assistant (default) und „Manual setup“
   (der bestehende Wizard, eingeklappt). Kein zweiter Tab/Wizard (frozen #13
   bleibt).
2. **Fokussierte Schritt-Bildschirme** — genau eine Entscheidung pro
   Bildschirm, ein primärer CTA, alles andere eingeklappt oder weg.
3. **Ein Fortschrittssystem** — ein schlanker 6er-Stepper aus den sechs
   beobachtbaren Screen-4-Phasen ersetzt die Doppeldarstellung
   (Wizard-Header + Assistant-Zeilen).
4. **Fehler sind Bildschirme, keine Boxen** — ein Fehlschlag übernimmt die
   Karte: Klartext, genau eine Aktion (`error.action`), Sekundär-Rückweg.
5. **Rückwärts ohne Neustart** — „Change model or context“ kehrt zur Modellauswahl
   zurück; die Hardware-Messung bleibt gültig, nur ein Modellwechsel misst
   modell-gebunden neu; ein Retry nach einer Settings-Änderung re-entryt am
   fehlschlagenden Schritt, nicht am Anfang.

---

## 1. Scope and constraints

- UI/UX only. **No backend change, no new route, no new state.** Every screen in
  the mock maps onto an existing assistant state (`SCREEN-3-MINIMAL.md` §5,
  `SCREEN-4-PROGRESS.md`).
- Frozen decisions honored (checked against `AGENTS.md` §1):
  - assistant lives **inside** the existing Cluster tab, no second tab/wizard
    (the „Manual setup“ mode *is* the existing wizard, not a new one);
  - every failure renders **exactly one action** from `error.action`;
  - never a local fallback for `retry_coordinator` / `pair_mac`;
  - endpoint URL only when `ready=true` and `endpoint.ready=true`;
  - English product strings only, no token-rate estimate, no SSE;
  - browser JS never launches, deploys or stages anything.
- The acceptance decisions of 2026-11-15 (`AGENTS.md` §10, D1–D6) are built in:
  - **D3 (manual path):** the existing wizard stays behind a collapsed
    „Manual setup“ disclosure and the assistant is the primary top-to-bottom
    flow — the mode split below is the decided shape, not a proposal anymore.
  - **D2 (back scope):** model/context/tier/offload/pin may be changed back up
    to a green `applied`; everything downstream becomes `stale`, never silently
    kept; a running (`ready`) cluster stays frozen (Screen-6 rules).
  - **D2 (retry scope):** a failed `start` is retried **in place** against the
    same applied deployment once the cause is fixed — re-validate the placement
    and start again, no forced re-apply.
- The design language is untouched: Inter, white cards
  `border-neutral-200 rounded-2xl`, `bg-neutral-900 rounded-full` primary
  buttons, amber/red/green banners, blue links, `text-caption` eyebrows.

---

## 2. Current-state audit

### 2.1 DOM order of the Cluster tab (top → bottom)

```text
1  Cluster header (title + subtitle)
2  Lifecycle buttons + banners (cleanup, loopback, version mismatch, wizard error)
3  5-step exo wizard tiles (Find devices → Pair → Check → Split the model → Activate)
4  Activation progress banner
5  State sections (empty / discovering / pairing / purpose / checks …)
6  Device card grid (N cards) + “Add Mac” panel + CUDA guide
7  ASSISTANT card                       ← ~1,080 lines of template
8  Manual plan form (model picker, context, layers)
9  Active deployment section
```

The assistant entry button sits *below* items 1–6. With 3–4 paired Macs that is
roughly 4–6 viewport heights of scrolling before the “one-action path” is
reachable.

### 2.2 Inside the assistant card

All branches mount in one card, stacked with `mt-3`, in this order: probe
toggle → resume strip → idle entry → six progress rows → per-state sentences →
plan summary → policy lines → tradeoffs → small-Mac notes → local-run panel
(+ tune-local) → constellation search panel → applying/starting/loading
sentences → ready endpoint (+ tuning, tune-redeploy) → failed box (+ topology
consent, free-space picker) → collapsed details (+ network preflight, use-all
Macs, coordinator pin).

### 2.3 Error rendering today

Five different red boxes exist (`assistant.error`, `localRun.error`,
`search.error`, `tune.error`, `tuneRedeploy.error`), each rendered *inline at
its panel's position* in the stack. Where a failure appears on screen therefore
depends on which panel failed — the user cannot predict it, and the one action
button sits below whatever the panel above showed. Branch affordances
(topology consent, free-space picker) open *below* the running flow.

### 2.4 Findings

| # | Finding | Effect |
|---|---|---|
| P1 | Assistant entry buried below wizard chrome and device cards | The “quick start” is the slowest path to reach |
| P2 | Two competing progress systems (wizard tiles + assistant rows) | During `measuring`, the header says “Plan”; nobody says which is true |
| P3 | One endless card: ~10 panels mounted simultaneously | Primary vs secondary actions indistinguishable; card grows downward |
| P4 | Errors are inline blobs at unpredictable positions (5 variants) | Failures easy to miss; action decoupled from cause |
| P5 | No back navigation inside the flow | Changing model/context means restarting from the entry |
| P6 | Equal-weight CTAs (Start setup, Run on this Mac, Search, tradeoffs) | No hierarchy on the decision screen |
| P7 | Progress rows stay mounted while result panels render below | The user follows the flow by scrolling (“von oben nach unten”) |

---

## 3. Design principles applied

- **Progressive disclosure** — one decision per screen; advanced evidence
  (stages, signature, warnings, tuning, search) collapsed by default.
- **One primary action per screen** — a single `bg-neutral-900` pill; everything
  else is a border-outline secondary or a quiet link.
- **Focused steps, no mixed modes** — async phases (measuring, applying,
  starting, loading) *own* the whole card: no plan panel, no tuning, no other
  panels visible while the flow runs.
- **Error as a state, not a fragment** — a failed step replaces the card
  content: plain-language title, cause, exactly one primary action, secondary
  back/details. Consistent *placement* (always the card), consistent *shape*
  (banner + action), consistent *vocabulary* (the frozen action set).
- **Back navigation with preserved state** — every post-entry screen has a back
  affordance; re-entry reuses the still-fresh measurement (15 min TTL) and only
  re-runs the step that actually depends on what changed.
- **Honest progress** — the stepper is exactly the six observable Screen-4
  phases; no percentage, no invented rate.

---

## 4. Proposed flow

```text
Start ──► Check ──► Plan review ──► Launch ──► Ready
 (idle)   (measuring/        (plan          (applying/     (endpoint,
           recommending/      set,          starting/      chat, agent,
           measuring_model/   tradeoffs,    loading)       stop)
           planning)          Start setup)
                ▲                │  ▲
                │  back ◄───────┘  │ change model/context
                └──────────────────┘  (re-measure model-bound only)
Fehler: jeder Schritt kann fehlschlagen → Fehlerbildschirm (eine Aktion)
        Retry re-entryt am fehlschlagenden Schritt.
```

### 4.1 Backend state → screen mapping (1:1, nothing invented)

| Assistant state (backend) | Screen | Primary action |
|---|---|---|
| `idle` | **Start** — hero + resume strip + Mac chips | Find and start a setup |
| `measuring`, `recommending`, `measuring_model`, `planning` (no plan yet) | **Check** — stepper + six rows | — (async) |
| `planning` with `assistant.plan` | **Plan review** — recommendation, plan summary, tradeoffs, details collapsed | Start setup |
| `applying`, `starting`, `loading` | **Launch** — stepper + six rows | — (async) |
| `ready` | **Ready** — endpoint, copy, Open Chat, Connect agent, Stop; Performance collapsed | Open Chat |
| `failed` (any step) | **Failed** — full-card error + one action + back | from `error.action` |
| `stopped` (ready + `stopped`) | Ready card without endpoint, Start again | Start again |

The wizard's own 5-step tiles are shown only in **Manual setup** mode; in
Assistant mode the slim 6-step stepper is the single progress system (P2, P7).

### 4.2 Screen-by-screen changes

| Screen | Concrete changes |
|---|---|
| Start | Entry moves to the **top** of the tab. Segmented control `Assistant / Manual setup` (default Assistant). Paired Macs become compact status chips (name, chip, RAM, green dot) instead of the full card grid; the full device cards remain in Manual setup. Resume strip (running/stopped, endpoint, Stop) is the first thing visible after a reload. |
| Check | The async phases own the card. Stepper row 1 active, six Screen-4 rows below, the assistant state sentence as the active row's detail. No other panel is mounted. |
| Plan review | One screen with the recommendation (model, context, fit) *and* the plan summary (Macs, coordinator, tier, offload, rank chips) *and* — when present — the context-vs-experts tradeoff as a decision block (stays plan-time per the 2026-11-14 decision). Primary: **Start setup**. Secondary: **Change model or context** (→ model list, see §6), quiet option **Run on this Mac** (only when the recommendation's `single_node` names this Mac, decision E6). Details (stages, rank order, signature, warnings, Use all Macs, coordinator pin, network preflight) stay collapsed. |
| Launch | Same layout as Check with later rows active; the row detail carries the backend's own `progress.message` and `{n} of {m} Macs ready`. Nothing else mounted. |
| Ready | Endpoint block with link/copy/Open Chat/Connect agent/Refresh, Stop cluster (red outline, right). Measured line only when `observed` is present. Tuning, tune-MTP, constellation search and the probe toggle collapse into one **Performance** section. |
| Failed | The card becomes the error screen (see §5). |

### 4.3 Error model (placement rules)

1. Assistant errors render **as the card's content**, never as an inline box
   appended below a running flow. Position is therefore predictable: always
   where the user is looking.
2. Shape: red banner (title + plain-language cause) + **exactly one primary
   action** from `error.action` + secondary links (back to the previous
   decision, Details). No other panels visible.
3. Branch affordances that need more UI (topology 2-Mac consent, free-space
   picker) render **inside the error screen**, directly under their cause —
   not below unrelated running panels.
4. Page-level banners stay for page-level facts (version mismatch, discovery
   errors) — they are not assistant errors.
5. The local-run / tune / search error boxes follow the same rule when their
  flow owns the card.

### 4.4 Retry after a settings change (concrete semantics)

Example — the plan refuses because a measured Mac sits at its GPU wired limit
(`raise_wired_limit`, R5):

```text
Plan review → Failed („No plan fits … Mac Studio (.13) is at its GPU wired
             limit (36,864 MB)“)
  primary:   Raise the GPU limit   → app-side Performance / Apply with Admin…
  secondary: smaller context · Change model
user raises the limit (one admin dialog, in the app, never the browser)
  → returns → „Plan again“ re-enters at the PLAN step:
     hardware measurement still fresh (15 min TTL) → no re-measure,
     model-bound measurement unchanged → no re-measure,
     only solve_cluster_plan runs again with the new ceiling
```

The same re-entry rule holds for every one-action failure: `retry_coordinator`
re-posts the failed stage (start if applied, else apply) with the same frozen
plan; `stale_measurement` re-measures; `insufficient_disk` (with the automatic
free-retry already in the backend) surfaces the free-space picker only when the
automatic retry refused.

---

## 5. Back navigation rules

Per acceptance decision D2: back-navigation reaches up to a green `applied`;
everything downstream of a change becomes **`stale`** and is shown as stale,
never silently kept. A running (`ready`) cluster stays frozen (Screen-6 rules).

| From | Back affordance | What is preserved | What re-runs |
|---|---|---|---|
| Plan review | „Change model or context“ | hardware measurement (TTL), settings | model-bound measure **only if** the model actually changed; plan |
| Plan review | „Change context“ | both measurements | plan only (context is a plan input) |
| Applied (before start) | „Change model or context“ | measurements, settings | plan + apply; the old applied operation is marked `stale`, never silently reused |
| Failed `start` (D2 retry scope) | primary retry action | the **applied** deployment | re-validate the placement, then `start` again in place — no forced re-apply |
| Failed other steps | secondary „Back“ | everything before the failed step | the failed step's request on retry |
| Ready | Stop → Start again | frozen plan inputs + signature | apply (signature re-proved) + start |
| any | „Start over“ (Details) | nothing | full flow |

Nothing here is new backend surface: `plan` is pure, `apply` already takes the
plan-time inputs again, and the strict node-set/model/`mtp` match stays intact.

---

## 6. Implementation notes (when this becomes code)

- Everything stays inside `clusterV2Wizard()` — no new component, no new
  `WIZARD_STATES` entry, `assistantState()` untouched. The screens are a
  render-mode on top of the existing state machine: one `<div
  data-assistant-screen="…">` per screen inside the assistant card, gated by
  `assistantState()` exactly like today's `data-assistant-state` blocks (the
  static contract test keeps working if the markers stay).
- The mode split is one Alpine boolean (`clusterMode: 'assistant' | 'manual'`);
  Manual mode simply keeps today's wizard sections visible.
- The 6-step stepper is a small render helper over `assistantProgressRows()`'s
  keys — same source, no second truth.
- Reordering: move the assistant section **above** the device-card section in
  `_cluster_v2.html` and hide the wizard tiles in Assistant mode.
- i18n: reuses existing `cluster.assistant.*` keys; the few new strings
  (screen titles, „Change model or context“, segmented-control labels) need new
  English keys mirrored by `normalize_i18n.py`.
- UI tests to extend: `test_cluster_ui_integration_ui.py` (screen-per-state
  assertions, back-affordance presence, error screen takes over the card).

---

## 7. Mock — how to use it

```sh
open docs/cluster-assistant/mockups/assistant-flow-klickstrecke.html
```

- Top bar: **branch navigator** (happy path steps S0–S5 and branches B1–B5) and
  a notes toggle.
- Right column: per-screen German annotations (what changed vs today, why).
- Dashed controls labelled „Demo“ are mock navigation only — they are not
  product UI.

Branches in the mock:

| ID | Branch | Demonstrates |
|---|---|---|
| S0 | Today (problem sketch) | the scroll-path problem |
| S1–S5 | Start → Check → Plan → Launch → Ready | happy path, one screen per step |
| B1 | Coordinator offline at Launch | error as a screen, one action, retry resumes the failed stage |
| B2 | `topology_incomplete` at Plan | explicit 2-Mac consent, signature salting |
| B3 | Wired limit refusal (settings changed) | re-entry at the failed step, no re-measure |
| B4 | Change model from Plan review | back navigation + model-bound re-measure |
| B5 | Stop / Start again | Screen 6 state + resume |
