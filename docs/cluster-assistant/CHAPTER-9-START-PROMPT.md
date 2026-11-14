# Session 9 — Screen 5: the handoff, and the answer to the speed question

## CHAPTER 9 — Make the ready surface a real handoff, and settle `observed`

### WORKING DIRECTORY

```text
/Users/andreaskolb/Documents/PycharmProjects/omlx-cluster-assistant
```

BRANCH `feat/cluster-assistant`

Last commits:

```text
7c9cecb5  docs(cluster): live evidence for Screen 4 and the free-space relay
db1c6a03  feat(cluster): Screen 4 progress rows and the free-space delete relay
8b512591  docs(upstream): #5033 rebased and pushed (head 9bc47278, CLEAN)
02072073  docs(upstream): #5033 rebased, PLE planning gap fixed, real 2-Mac smoke passed
ce7d3cc1  docs(upstream): record the two open PRs and why they do not need a rebase
```

Do not push. Do not assume any prior conversation: every decision is in the
repository. Chapters 1–8 are finished, the live run has happened, and Screen 4
plus the free-space relay were implemented and proven on hardware on 2026-11-07.
Do not redo any of that. This chapter is **Screen 5 — Ready**: the handoff from a
ready cluster to chat and to an agent, plus the one open question that has been
sitting under it since Chapter 5: **what `observed` means and who may fill it.**

---

### READ FIRST, IN THIS ORDER

1. `AGENTS.md` §2 “Resolved implementation decisions” — the two items this
   chapter must honour:
   - **Screen 5 preselects in the existing `/chat`.**
     `/chat?model=<id>&endpoint=<url>` preselects the model and, for a remote
     endpoint, uses it as the API base. No new page, no second chat.
   - **The performance probe runs after `ready`, asynchronously, on the elected
     coordinator.** It may fill one nullable `observed` block on the operation
     (`prefill_tok_per_s`, `decode_tok_per_s`, `ttft_ms`, `measured_at`) and a
     performance profile on the coordinator. `predicted` is never rewritten, and
     with the probe off (the default) `observed` stays `null` — the UI then says
     nothing about speed.
2. `docs/cluster-assistant/PROJECT-BRIEF.md` §5 “Screen 5 — Ready” (the five
   things the screen shows and the three conditions before it may appear) and
   §12.4 (one action per failure). §22 stays postponed: do not decide product
   questions listed there.
3. `docs/cluster-assistant/PHASE-2B-RUN.md`:
   - §11 — the Screen-4 live walk: ids, the real phase sequence, the endpoint
     that answered, the completion. This is the state the handoff starts from.
   - §9.2 — the free-space relay, proven live on `.14`.
   - §6 — the blocker table. The two Screen-5 rows are what this chapter closes.
   - §5 — the generation findings: a thinking model with a small cap returns an
     empty answer. Do not re-derive this.
4. `docs/cluster-assistant/SCREEN-4-PROGRESS.md` §8 — what the card renders now,
   so the ready surface is not rebuilt.
5. `docs/cluster-assistant/SCREEN-3-MINIMAL.md` §11 — the deliberately deferred
   items, of which two are *this* chapter:
   `Connect agent` has no deep link and `Open Chat` navigates to `/chat` without
   a preselection.
6. `docs/cluster-assistant/CLUSTER-ASSISTANT.md`, `docs/cluster-assistant/README.md`
   — both are deliverables: they must no longer describe the Screen-5 handoff as
   missing when you are done.

Then the code you will drive:

| Area | File |
|---|---|
| Chat page (the handoff target) | `apps/omlx-web/omlx_web/templates/chat.html` — one inline Alpine component, ~8.5k lines; `currentModel`, `availableModels`, `resolvePreferredDefaultModel()` (~3379), model load via `fetch('/v1/models')` (~5176), completions via `fetch('/v1/chat/completions')` (~6746) |
| Chat route | `apps/omlx-web/omlx_web/routes.py::chat_page` (line ~145) — renders the template, injects the host API key for `localStorage` |
| Assistant card markup | `apps/omlx-web/omlx_web/templates/dashboard/_cluster_v2.html` — the ready block at `data-assistant-state="ready"` |
| Assistant component | `apps/omlx-web/omlx_web/static/js/cluster_v2.js` — `openAssistantChat()`, `toggleAssistantAgent()`, `copyAssistantEndpoint/Model()`, the block from `// Assistant — the minimal Screen-3 slice` |
| Strings | `apps/omlx-web/omlx_web/i18n/en.json` (91 `cluster.assistant.*` keys today) |
| Fixtures | `apps/omlx-web/tests/ui/fixtures/cluster_v2/assistant_*.json` (13 files) |
| Static UI contract tests | `apps/omlx-web/tests/ui/test_cluster_v2_wizard.py` |
| Behaviour tests (skip without node) | `apps/omlx-web/tests/test_cluster_ui_integration_ui.py` |
| Operation record / status | `omlx/cluster/assistant/operations.py`, `start.py` (`run_start_job`, `_readiness_proven`), `models.py` (`PlanObserved`, `AssistantOperationStatus`) |
| The existing probe | `omlx/cluster/launch.py::run_cluster_performance_probe` (line ~1414), `omlx/cluster/performance.py` (`NodePerformanceProfile`, `performance_profiles_from_records`), `omlx/cluster/performance_worker.py::main` |

---

### STATE OF THE REPO (verified at `7c9cecb5`)

Tests: `tests/cluster_assistant` = **234 passed**; `tests/test_cluster_*.py` =
**1600 passed, 2 skipped**; `apps/omlx-web/tests` = **306 passed, 170 skipped**
(the 170 skip because `node` is absent; with a Node 22 binary on PATH the same
suite is **476 passed, 0 skipped** — use that if you can, the Node-driven tests
are the ones that exercise the card).

Backend: **nine** assistant admin routes —

```text
POST   /admin/api/cluster/assistant/measure
GET    /admin/api/cluster/assistant/status?measurement_id=…
POST   /admin/api/cluster/assistant/recommend
POST   /admin/api/cluster/assistant/plan
POST   /admin/api/cluster/assistant/apply
POST   /admin/api/cluster/assistant/start
GET    /admin/api/cluster/assistant/operations/active        ← server-to-server lock
GET    /admin/api/cluster/assistant/operations/{operation_id}
DELETE /admin/api/cluster/assistant/models/{node_id}/{model_id}?measurement_id=…
```

`ASSISTANT_API` in the component holds the eight the browser calls;
`operations/active` is the one it may not. The measured operation vocabulary is
unchanged and must stay so:

```text
kind:    apply | start
state:   queued | applying | applied | starting | loading | ready | failed
progress.phase: staging | applied | starting | loading | ready | failed
progress: {phase, message, nodes_ready, nodes_total}
```

The ready surface that already exists (do not rebuild it): the assistant card's
`ready` block shows the model, the context, the endpoint **only** when
`assistantReady()` is true, Copy URL, `Open Chat` (navigates to a bare `/chat`),
`Connect agent` (toggles a card with the base URL and the model ID) and a
collapsed Details panel with stages, rank order, ids, warnings and the raw
operation body.

Facts discovered in the live run you must not re-derive:

- `/v1/models` on the endpoint requires the admin key (`401 API key required`
  without it). The coordinator's key lives on the coordinator; the origin never
  has it and never receives it (that is the relay credential story).
- **The assistant's model id is not the serving id.** The recommendation answers
  with `selected_model` = the planner path
  (`/Users/andreaskolb/Models/dev-1.5b`), while `/v1/models` advertises the
  catalogue id (`dev-1.5b`). The card's `modelOptions` carry **both** (`id` and
  `model_path`), so the serving id is resolvable without a backend change.
- The readiness canary is a real completion but a tiny one (live: 2 tokens,
  0.73 s). It is proof of readiness, **not** a measurement: never turn it into a
  rate.
- `.14` is at 97 % disk (27 GiB free); `.12` is discovered but unpaired; `.13`
  did not appear. A 2-Mac `.11`+`.14` cluster is the working shape.

---

### GOAL FOR THIS SESSION

1. Make Screen 5 a real handoff: the ready surface opens the **existing** chat
   with the model preselected, and `Connect agent` gives an agent everything it
   needs — except a secret.
2. **Answer the speed question.** Implement the recorded post-ready probe (one
   nullable `observed` block, off by default, `predicted` never rewritten) or, if
   you find a reason not to, record that decision and say why. It may not be
   silently dropped.
3. Keep every existing test green and update the stale docs.

This session is NOT: Screen 6 (stop/restart, “Running / Manage”), SSE/streaming,
tensor parallelism, model-less measurement, a new tab/page/wizard, chat UI
redesign, macOS-app or tray work, or any §22 product decision.

---

### PART 1 — Screen 5: the handoff (one tab, one component, no second chat)

The screen itself already exists; what is missing is that its two buttons do not
finish the job. Fix exactly that.

**The link contract.** `Open Chat` and `Connect agent` build one URL each, from
values the backend reported:

```text
Chat:   <base>/chat?model=<serving model id>
Agent:  <endpoint>/v1                      (base URL)
        <serving model id>                 (model id)
        where the API key lives on that Mac (a sentence, never the key)
```

`<base>` rules:

- coordinator **is this Mac** → relative `/chat?model=…` (the browser already
  talks to this server);
- coordinator **is another Mac** → absolute `<endpoint.url>/chat?model=…`, i.e.
  the coordinator's own page. The endpoint *is* the API base; the origin cannot
  proxy, because it does not have that Mac's key and must not get it. Record this
  in the brief: for a remote coordinator the browser talks to the endpoint
  directly, and that Mac's own admin session applies. Two consequences to say out
  loud in the handoff: the page injects **that** Mac's own key into its own
  `localStorage` (which is exactly why the origin needs none), and the user may
  have to sign in to that Mac's UI once, because a session cookie is per host.
  The card must not pretend otherwise; one English sentence under the button is
  enough.

`<serving model id>` rules:

- resolve it from the catalogue: `modelOptions.find(m => m.model_path === selectedModel)?.id`;
- if it cannot be resolved, or the endpoint would not list it, link **without**
  the parameter instead of linking to a broken selection. Never invent an id and
  never fall back to a different model silently — say which model is open in the
  chat if you add a line.

**`/chat` preselection.** The chat page has no query-string handling today
(`URLSearchParams`/`location.search` appear nowhere in the template). Add the
smallest change that works:

- read `model` once on load, and once `availableModels` is non-empty select it if
  it is present; an unknown value leaves the normal default selection untouched;
- do not touch the streaming, upload, audio, theme or settings code;
- no `endpoint` parameter is needed for the *local* case; if you accept one for
  diagnostics, it may only ever be used to display what the page is talking to, it
  must never redirect, fetch or carry a key — or reject the parameter outright
  and say so in the record;
- the preselection must survive a reload (it is in the URL) and must not add a
  history entry of its own.

**`Connect agent`.** Show, in the existing agent card: base URL, model id, and
one English sentence naming where to get the API key on the coordinator
(“Settings → API key on <coordinator Mac>”). No key, no token, no credential in
the DOM, the URL, a fixture, a log or a commit. Keep the copy buttons that exist.

**Ready-only.** Everything above appears only when
`assistantState() === 'ready'` **and** `endpoint.ready === true`; the endpoint
stays `null` otherwise and no port-0 URL is ever rendered. A failure keeps its
one action (SCREEN-3 §7). “Open details” is the existing Details panel — do not
add a second diagnostics surface.

**Tests.** Static: the ready block builds its links from `endpoint.url` and the
resolved serving id, both buttons are ready-gated, no key/token appears in the
card or the chat page diff, and the strings exist in every locale (the repo's
`normalize_i18n.py` mirrors English). Node: with a ready operation whose endpoint
is `http://172.16.1.11:8099`, `Open Chat` targets that base with the resolved id;
with a *remote* coordinator it targets the coordinator's own page; without a
resolved id it targets the page with no `model` parameter; and the chat page
selects a known `?model=` while an unknown one leaves the default alone.

---

### PART 2 — the speed question (`observed`) — ask once, then implement

The decision is already recorded in `AGENTS.md` §2; this chapter is where it
becomes code or is deliberately re-recorded. Constraints that are not negotiable:

- `predicted` is **never** rewritten; `predicted.measured` stays false for an
  unmeasured plan.
- `observed` is one **nullable** block, filled only by a real measurement. With
  the probe off (the default) it stays `null`, and the UI then says **nothing**
  about speed. No percentage, no ETA, no byte progress anywhere.
- The 2-token readiness canary is not a measurement.
- A probe failure is a recorded warning, never a reason to unready a cluster,
  stop a deployment, or fail an operation.
- The assistant request sets `auto_tune=False`; keep it that way unless the probe
  decision explicitly says otherwise. Keeping the probe out is what keeps the
  approved placement the launched one.

Decide and record **once**, before coding, in a new
`docs/cluster-assistant/SCREEN-5-HANDOFF.md` plus `PHASE-2B-RUN.md`:

- **Where `observed` lives.** The operation record already survives the request
  (`assistant-operations.json`); the plan does not (it is pure and recomputed).
  Adding a nullable `observed` to `AssistantOperationStatus` (and its record) is
  the additive contract change the AGENTS decision describes. Pin it with one
  test on the router/record shape.
- **What fills it.** The repo already has a real distributed probe
  (`launch.run_cluster_performance_probe` → `performance_worker` →
  `performance_profiles_from_records`). Reusing it is the honest source; a
  hand-rolled timing of the canary is not.
- **When it runs and who may ask.** After `ready`, asynchronously, on the elected
  coordinator, through the same local/remote seam apply/start use — never from
  the browser, and never as part of the blocking activation.
- **How it is switched on.** Default off. The recorded options are a server
  setting or an explicit per-operation flag; pick one, record it, and make the
  default unambiguous in the docs.
- **What the UI does with it.** If `observed` is present, Screen 5 may state the
  measured prefill/decode rate and when it was measured — labelled as measured
  and timestamped, never as a promise. If it is `null`, the card keeps the
  sentence it has today. Do not add a rate to the plan, the progress rows or the
  endpoint card beyond that one measured line.

If, after reading the code, you conclude the probe cannot be wired without a
product decision this chapter may not take, **stop and ask once**, then record
the answer in the same two documents. What you may not do is leave the item
untouched: §6 of the run document must end this chapter with the row either
closed or explicitly re-scoped, with the reason.

Tests: fake-driven only — `observed: null` for a normal start, a recorded
`observed` when a fake probe answers, a failing probe producing a warning and
still `ready`, `predicted` unchanged in both cases, and no rate anywhere in the
card unless `observed` is present.

---

### MUST NOT CHANGE

- The nine assistant routes' request/response shapes (except the one additive
  `observed` field Part 2 records), the operation state machine, the error codes
  and the action strings.
- `selected_model` / `selected_context_tokens` binding, endpoint visibility,
  one-action-per-failure, the Screen-4 rows and their sources.
- `election.py`, `measurement.py`, `recommendation.py`, `omlx/cluster/routes.py`,
  `omlx/server.py` — unless a live defect proves it and you record why.
- The chat page's streaming, upload, audio, theme and settings behaviour. The
  only chat change is the preselection (and nothing else).
- The existing 8 wizard states and their guards, the plan/activate controls, the
  pairing flows, anything under `apps/omlx-mac`.
- English only. No German, no translation. No API key, cookie or SSH credential
  in any file, fixture, test, log or commit.

---

### IMPLEMENTATION RULES

- The UI renders what the backend says. It never computes a plan, rank order,
  coordinator, admission, phase, readiness or rate.
- Browser JS never launches, deploys, SSHes, stages, supervises or probes. It
  calls the assistant API and navigates.
- Exactly one action per failure, taken from `error.action`.
- Keep the diff reviewable: the ready block, its strings, the chat preselection,
  the tests/fixtures, the two documents — plus one optional `observed` field and
  its producer if Part 2 implements it.

---

### LIVE CHECK (the agent cannot click)

The validation cluster is free as of 2026-11-07. It is a line of four Macs over
the Thunderbolt bridge:

| Node | Hardware | RAM | Last measured |
|---|---|---|---|
| `.11` | MacBook Pro M5 Max | 128 GiB | 53.9 GiB admission, runs this branch |
| `.14` | Mac Studio M1 Max | 32 GiB | 19.9 GiB admission, Headless, **97 % disk** |
| `.12` | Mac Studio M1 Max | 32 GiB | discovered but **unpaired** |
| `.13` | Mac Studio M2 Ultra | 64 GiB | did not appear on 2026-11-07 |

- Start each Mac's server from a **Terminal window on that Mac**, never over SSH
  (`PYTHONPATH=<checkout>:<checkout>/apps/omlx-web`, the installed oMLX 0.7.3
  interpreter, `--host 0.0.0.0 --port 8099`). An SSH-started server gets no macOS
  Local Network permission: no mDNS, no outbound LAN HTTP.
- A server picks up new code only on restart, and `.14`/`.12`/`.13` still run an
  older checkout (the ninth route answers 404 there — that is fine, the relay
  only needs their own `/admin/api/hf/models`). The *origin* you drive must run
  this branch.
- Unload any running deployment before measuring: a loaded cluster drops the
  measured admission (live: 97 → 30 GiB) and the plan refuses for lack of memory.
  `GET /admin/api/cluster/assistant/operations/active` must answer
  `{"operation": null}` before you start.
- `/v1/models` and `/v1/chat/completions` on the endpoint need the admin key;
  read it from that Mac's own `~/.omlx/settings.json` (`auth.api_key`). Keys live
  outside the repository: never paste one into a doc, a test, a fixture or a
  commit.
- The user drives the browser. You read the operation/measurement JSON and
  `~/.omlx/logs/server.log`, and you record evidence in
  `docs/cluster-assistant/PHASE-2B-RUN.md` (a new §12), never in the test suite.
- Never delete a model on hardware without asking first, and never stop or
  restart a deployment the user did not ask you to touch.

---

### WORKFLOW

1. Read Part 1's targets and `chat.html`'s model-load/selection path before
   touching anything; trace how `currentModel` is chosen today.
2. Part 1: link rules, chat preselection, the agent card, strings, fixtures,
   static tests, then the Node tests.
3. Part 2: ask the one question if it is still open, record it, then implement.
4. Rebuild CSS after template changes and commit the output:

   ```sh
   cd apps/omlx-web && python build_css.py
   ```

5. Verification pass:

   ```sh
   uv run pytest tests/cluster_assistant -q            # 234 passed (or more)
   uv run pytest tests/test_cluster_*.py -q            # 1600 passed, 2 skipped
   uv run pytest apps/omlx-web/tests -q                # 306 passed, 170 skipped
   # with Node on PATH, the same suite must be 476 passed, 0 skipped
   uv run ruff check <touched python files>
   ```

6. Commit on `feat/cluster-assistant` with a focused message. Do not push.

---

### DEFINITION OF DONE

- From a ready cluster, `Open Chat` opens the existing `/chat` with the cluster's
  model already selected — the coordinator's own page when the coordinator is
  another Mac — and `Connect agent` shows base URL, model id and where the key
  lives, with no secret anywhere.
- Both appear only when the operation is `ready` and `endpoint.ready` is true.
- The chat page's only new behaviour is that preselection; normal chat, an
  unknown `?model=`, and a reload all still work.
- The speed question is **closed or explicitly re-scoped with a reason**, in
  `SCREEN-5-HANDOFF.md` and `PHASE-2B-RUN.md`; `observed` is either implemented
  (nullable, off by default, `predicted` untouched, warning on failure) or
  deliberately recorded as not implemented. No rate is displayed unless it was
  measured.
- All pre-existing tests stay green; every new test is fake-driven.
- `SCREEN-3-MINIMAL.md` §11, `CLUSTER-ASSISTANT.md`, `README.md` and
  `PHASE-2B-RUN.md` §6 no longer describe the Screen-5 handoff as missing.

Report back: the exact links the card produces for a local and a remote
coordinator, what the chat page does with a known and an unknown `?model=`, the
Part 2 decision and where it is recorded, whether `observed` now exists and what
fills it, the tests added, every file changed, whether any backend file changed
(and why), and what still blocks Screen 6.

STOP after Screen 5 and the `observed` decision. Do not implement Screen 6,
SSE/streaming, the macOS app or tray, chat redesign, tensor parallelism,
model-less measurement, or any §22 product decision without a new instruction.