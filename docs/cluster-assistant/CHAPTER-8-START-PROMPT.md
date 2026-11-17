# Session 8 — Screen 4: the progress surface

## CHAPTER 8 — Implement Screen 4 on the measured contract, then the free-space affordance

### WORKING DIRECTORY

```text
/Users/andreaskolb/Documents/PycharmProjects/omlx-cluster-assistant
```

BRANCH `feat/cluster-assistant`

Last commits:

```text
401b72b6  feat(cluster): small Macs are Headless, and a peer's model is usable
64dbff76  fix(cluster): cluster-wide operation lock and a reproducible election
bc263d38  fix(cluster): close five gaps recorded by the live run
7f4d6104  feat(cluster): first live run on real Macs, and the Screen-4 decision
ac059445  feat(cluster): minimal Screen-3 assistant card in the Cluster tab
```

Do not push. Do not assume any prior conversation: every decision is in the
repository. Chapters 1–7 are finished and the live run has already happened — do
not redo them and do not re-run Phase 2b acceptance. This chapter is **Screen 4**:
the progress surface between “Start setup” and “Ready”.

---

### READ FIRST, IN THIS ORDER

1. `AGENTS.md` — especially §2 “Resolved implementation decisions” (Screen-4
   Option A, one-tab/`assistantState()`, endpoint visibility, no SSE, no token
   rate, the small-Mac rule, the cluster-wide lock, the measurement-owned
   layout).
2. `docs/cluster-assistant/SCREEN-4-PROGRESS.md` — **the authoritative brief for
   this chapter**. Everything below is a summary of it.
3. `docs/cluster-assistant/PHASE-2B-RUN.md` — what the screens actually look like
   on hardware, the real operation states/timings, and the defects already
   fixed. §3.2 has a full three-Mac run; §5 explains the generation settings.
4. `docs/cluster-assistant/SCREEN-3-MINIMAL.md` §5/§7/§11 — the shipped card, its
   states, its one-action rule and what was deliberately deferred.
5. `docs/cluster-assistant/CLUSTER-ASSISTANT.md` (now stale in two small places —
   update them, they are part of the deliverables) and
   `docs/cluster-assistant/README.md`.

Then the code you will drive:

| Area | File |
|---|---|
| Assistant card markup | `apps/omlx-web/omlx_web/templates/dashboard/_cluster_v2.html` (starts at the marker `data-cluster-v2-assistant`, line ~675) |
| Assistant component | `apps/omlx-web/omlx_web/static/js/cluster_v2.js` (the block starting `// Assistant — the minimal Screen-3 slice`, line ~3974) |
| Strings | `apps/omlx-web/omlx_web/i18n/en.json` (77 `cluster.assistant.*` keys today) |
| Fixtures | `apps/omlx-web/tests/ui/fixtures/cluster_v2/assistant_*.json` (9 files) |
| Static UI contract tests | `apps/omlx-web/tests/ui/test_cluster_v2_wizard.py` |
| Behaviour tests (skip without node) | `apps/omlx-web/tests/test_cluster_ui_integration_ui.py` |
| Operation/measurement shapes | `omlx/cluster/assistant/models.py`, `operations.py`, `routes.py` |

---

### STATE OF THE REPO (verified at `401b72b6` — not guesses)

Tests: `tests/cluster_assistant` = **220 passed**; `tests/test_cluster_*.py` =
**1600 passed, 2 skipped**; `apps/omlx-web/tests` = **292 passed, 164 skipped**
(164 skip because `node` is absent; the static contract tests run everywhere).

Backend: **eight** assistant admin routes —

```text
POST /admin/api/cluster/assistant/measure
GET  /admin/api/cluster/assistant/status?measurement_id=…
POST /admin/api/cluster/assistant/recommend
POST /admin/api/cluster/assistant/plan
POST /admin/api/cluster/assistant/apply
POST /admin/api/cluster/assistant/start
GET  /admin/api/cluster/assistant/operations/active        ← coordinator lock
GET  /admin/api/cluster/assistant/operations/{operation_id}
```

`ASSISTANT_API` in the component still holds exactly the seven the browser may
call; `operations/active` is called server-to-server and is listed in
`test_cluster_seams_ui.py`'s `allowed_without_caller` with its reason.

The measured operation vocabulary (do not invent more):

```text
kind:    apply | start
state:   queued | applying | applied | starting | loading | ready | failed
progress.phase: staging | applied | starting | loading | ready | failed
progress: {phase, message, nodes_ready, nodes_total}
```

The assistant state machine in the card:

```text
idle → measuring → recommending → measuring_model → planning
     → applying → starting → loading → ready        (+ failed)
```

Ten `data-assistant-state="…"` markers exist in the template today.

---

### GOAL FOR THIS SESSION

1. Render the **eight Screen-4 labels** on the existing card from data the
   backend already reports (Option A, no backend change, no SSE).
2. Add the **free-space affordance** (the last run-proven gap): when a failure's
   action is `free_disk_space`, let the user delete an unused model on the Mac
   that is short of space, with a two-step confirm.
3. Keep every existing test green, and update the stale docs.

This session is NOT: Screen 5, Screen 6, SSE/streaming, the performance probe,
model-less measurement, tensor parallelism, a new tab/page/wizard, macOS-app or
tray work, or any §22 product decision (still postponed).

---

### PART 1 — Screen 4 (no backend change)

`SCREEN-4-PROGRESS.md` is the contract. The short version:

**Only six labels are independently observable.** Two pairs are combined because
the backend cannot separate them (runtime-prepare happens inside the staging job;
the readiness canary happens inside the blocking activation call). The order is
**corrected**: planning precedes staging.

| # | Label (en) | Source |
|---|---|---|
| 1 | Checking Macs | `assistantState() ∈ {'measuring', 'measuring_model'}` |
| 2 | Planning | `assistantState() ∈ {'recommending', 'planning'}` |
| 3 | Preparing runtime **and** Copying model | `kind='apply'`, `progress.phase='staging'` |
| 4 | Starting coordinator | `kind='start'`, `progress.phase='starting'` |
| 5 | Loading model **and** Verifying endpoint | `kind='start'`, `progress.phase='loading'` |
| 6 | Ready | `state='ready'` **and** `endpoint.ready === true` |

Rules that may not be broken:

- A row never animates without a backend value from that table. No timers, no
  optimistic advance.
- `Ready` requires `state === 'ready' && endpoint.ready === true`. `loading` is
  never “almost ready”.
- A failure keeps the last reported phase, shows `error.message`, and offers
  exactly **one** button labelled from `error.action` (SCREEN-3 §7).
- `nodes_ready / nodes_total` may be shown as “2 of 3 Macs”. **No percentage, no
  ETA, no byte progress, no token rate** — `predicted.measured` is false and
  `observed` is null everywhere.
- Unknown `progress.phase` (a newer coordinator): keep the previous row and show
  `progress.message` underneath. Never guess, never blank.
- One tab, one component, `assistantState()` unchanged in shape; no new wizard
  state; English only; new keys are English in every locale (the repo's
  `normalize_i18n.py` / identical-key-set test).

Acceptance: during a real apply/start the card walks
Checking → Planning → Preparing/Copying → Starting → Loading/Verifying → Ready
in that order, with the backend's own `progress.message` under each row, and the
endpoint still appears only at Ready.

---

### PART 2 — the free-space affordance (decision already taken)

**Decided 2026-11-06, Option A — do not re-ask.** Recorded in `AGENTS.md` and
`PHASE-2B-RUN.md` §9.

Today a copy that cannot fit is refused with `insufficient_disk` and the action
`free_disk_space`; the message names the Mac, the needed and the free GiB.
Deleting an unused model is the fix, and `DELETE /admin/api/hf/models/{name}`
already removes a model from a Mac and refreshes its pool.

Build:

1. **One new assistant route (the ninth):**
   `DELETE /admin/api/cluster/assistant/models/{node_id}/{model_id}`, relayed
   over the existing admin seam (`run_remote_python`, the same credential story
   as `operations/active`) to that Mac's own delete route. Works for the worker
   that is short of space, which is the case that fails.
2. **The measured model list of that Mac, with sizes.** Add the size to the
   inventory snapshot the measurement already stores (`NodeFacts.model_paths`
   today; a parallel `model_bytes` map or `{path, bytes}` rows — your call, keep
   it additive and fake-tested).
3. **The card**, on `free_disk_space`: the short Mac's list, one **Delete** per
   model, a second click to confirm (“Delete <model>?”), then `measure` again.
   Never automatic, never the model the cluster is about to run, never another
   Mac's list.
4. **Tests**: the route relays and refuses a worker it cannot reach; the card
   shows the action only for `free_disk_space`; the confirm is two-step; the
   router-list test and the seam/trace tests are updated (this route *is* called
   by the card, so it is a caller, not an allowlist entry).

### MUST NOT CHANGE

- The eight assistant routes' request/response shapes, the operation state
  machine, the error codes and the action strings (except the one recorded
  addition in Part 2 if you choose A).
- `selected_model` / `selected_context_tokens` binding, endpoint visibility,
  one-action-per-failure.
- `election.py`, `measurement.py`, `state.py`, `recommendation.py`,
  `omlx/cluster/routes.py`, `omlx/server.py` — unless a live defect proves it and
  you record why.
- The existing 8 wizard states and their guards, the plan/activate controls, the
  pairing flows, anything under `apps/omlx-mac`.
- English only. No German, no translation.

---

### IMPLEMENTATION RULES

- The UI renders what the backend says. It never computes a plan, rank order,
  coordinator, admission, phase or readiness.
- Exactly one action per failure, taken from `error.action`.
- No token rate anywhere, and no progress percentage.
- Keep the diff reviewable: the card, its strings, its tests/fixtures, the two
  docs; Part 2 adds one route + one relay helper if A is chosen.

---

### LIVE CHECK (the agent cannot click)

The validation cluster is 4 Macs (`.11` M5 Max 128 GB, `.14` and `.12` M1 Max
32 GB, `.13` M2 Ultra 64 GB) over the **Thunderbolt bridge**. It is not required
to be online; check before relying on it.

- Start each Mac's server **from a Terminal window on that Mac**, never over SSH
  (`PYTHONPATH=<checkout>:<checkout>/apps/omlx-web`, the installed oMLX 0.7.3
  interpreter, `--host 0.0.0.0 --port 8099`). An SSH-started server gets no
  macOS Local Network permission: no mDNS, no outbound LAN HTTP, “No oMLX node
  answered at that address yet”.
- Unload any running deployment before measuring: a loaded cluster drops the
  measured admission (97 → 30 GiB) and the plan then refuses for lack of memory.
- The user drives the browser; the agent reads the measurement/operation JSON and
  the server logs (`~/.omlx/logs/server.log`). Record evidence in the run
  document, never in the test suite.
- Admin keys live outside the repository; never paste one into a doc, a test or a
  commit.

---

### WORKFLOW

1. Read `SCREEN-4-PROGRESS.md` and the assistant block; trace the operation
   polling in `cluster_v2.js` before touching anything.
2. Part 1: markers, labels, sources, the unknown-phase fallback, tests and
   fixtures. Static tests first, then the live walk with the user.
3. Part 2: ask the A/B/C question, record it, then implement.
4. Rebuild CSS (`cd apps/omlx-web && python build_css.py`) and commit the output
   after template changes.
5. Verification pass:

```sh
uv run pytest tests/cluster_assistant -q            # 220 passed (or more)
uv run pytest tests/test_cluster_*.py -q            # 1600 passed, 2 skipped
uv run pytest apps/omlx-web/tests -q                # 292 passed, 164 skipped
uv run ruff check <touched python files>
```

6. Commit on `feat/cluster-assistant` with a focused message. Do not push.

---

### DEFINITION OF DONE

- The card walks the six observable Screen-4 labels in order during a real
  apply/start, from backend values only, with the backend's message under each.
- `Ready` is unreachable unless `state === 'ready'` and `endpoint.ready === true`;
  no rate, no percentage, no invented phase.
- A failed operation keeps the last phase and shows one action.
- Unknown phase and reload-mid-operation are handled as `SCREEN-4-PROGRESS.md`
  §5 requires, with tests.
- Part 2 is either implemented with the recorded decision, or recorded as a
  decision with the reason — not silently dropped.
- All pre-existing tests stay green; any new test is fake-driven.
- `SCREEN-3-MINIMAL.md`, `CLUSTER-ASSISTANT.md` and `README.md` no longer say the
  Screen-4 view is missing.

Report back: what the card shows per phase (screenshot or the state sequence),
the tests added, the Part 2 decision and where it is recorded, every file
changed, whether any backend file changed (and why), and what still blocks
Screen 5.

STOP after Screen 4 and the Part 2 decision. Do not implement Screen 5, Screen 6,
SSE/streaming, the performance probe, model-less measurement, chat, tray,
macOS-app work or tensor parallelism without a new instruction.