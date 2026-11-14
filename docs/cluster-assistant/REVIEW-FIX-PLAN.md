# Fix plan — review findings (2026-11-13)

Execution plan for [`REVIEW-2026-11-13.md`](REVIEW-2026-11-13.md).
Finding IDs (B…, U…, R…, D…) reference the review; this plan tracks progress
(checkboxes). No frozen decision is changed silently — open points are in §2.

---

## 1. Principles

1. **One slice = one commit** (or a few small commits), small enough for a
   Local History / `git diff` review. The commit message names the finding
   IDs.
2. **Red → green**: first a test that reproduces the bug (for B1, bring the
   fakes to the real API shape first and watch the test go red), then the
   fix.
3. **Fake-driven** (AGENTS.md): no new tests against real Macs/models. Live
   evidence only in `PHASE-2B-RUN.md`, and only after approval.
4. **Verification ladder per slice** (§4): target tests → assistant suite →
   UI suites (with Node) → periodically the full suite.
5. **UI slices are sequential** (they all touch `cluster_v2.js`/the
   template); backend slices can interleave with UI slices, but `start.py`
   is touched by several backend slices → sequential there too.
6. The review stays the evidence document and is not rewritten; this plan
   tracks status. At the end of a slice: tick the checkbox here.

---

## 2. Decision points

| # | Question | Recommendation | Status |
|---|---|---|---|
| E1 | **C1/C3 divergence**: align AGENTS.md/brief/SCREEN-3/SCREEN-6 + `test_cluster_ui_integration_ui.py` with the recorded roadmap decisions (reload resume C1, hardware-only entry measurement C3) — or revert the code? | **Align docs + tests** (the roadmap recorded the decisions as closed; the backend already supports both). | **Decided 2026-11-13: align. Done in slice 0.** |
| E2 | **D1 dotfiles**: untrack `.live_prefill_driver*.py` + `.local-3rank-2026-11-13.md` via `git rm --cached` + `.gitignore`. Rewrite history? | **Untrack only**, no history rewrite (the internal IPs remain in history, but the repo is private; a rewrite would be costlier and riskier). | **Decided 2026-11-13: untrack only. Done in slice 0.** |
| E3 | **B5 lock design**: create the operation record on the coordinator through the admin seam (option A) vs. a separate lock route on the coordinator (option B). | **Option A**: `_remote_ops` creates an assistant operation record on the coordinator at start and updates it; `operations/active` then works without a new route. Fits the frozen architecture (same seam, no new control plane). Short design note in CHAPTER-5.md before implementation. | Recommended; design note pending in slice 7. |
| E4 | **B10 probe weighting**: election ranks by measured critical stage today (frozen). Improve the probe's physical plausibility and/or use the probe only as a tie-break? | **Two steps**: first make the probe physically plausible (kernel/iterations) + document its latency nature; a weight change only with an explicit decision, since it touches the frozen ranking. | **Decided 2026-11-13: option B** — make the probe bandwidth-bound (live-validated via E5) AND add a plausibility guard: an implausible/missing probe is treated as unknown compute → the frozen headroom preference + `unmeasured_compute`. Trustworthy probes still rank by critical stage. |
| E5 | **Live gate**: bring the paused cluster back up at the end for a smoke + optional A/B run? | Yes, but only after P0/P1 and only with an explicit go; evidence into `PHASE-2B-RUN.md`. | Open — needs the user's go. |

---

## 3. Slices

Order = dependencies + risk. Effort rough: S < 1 h, M ≈ half a day,
L ≈ a day+ (agent work incl. tests).

| Slice | Findings | Effort | Depends on | Status |
|---|---|---|---|---|
| 0 Hygiene & doc alignment | D1, UPSTREAM.md, E1 | S | E1, E2 | ☑ `526c8257` |
| 1 UI quick fixes + test net | U1, U5, U3, U4, small-Mac constant, busy guards | M | — | ☑ `4ffce5d1` |
| 2 Measurement recovery & poll robustness | U2, U7, resume state, retry/backoff | M | 1 | ☑ `159f51b0` |
| 3 Chat profile & reload | B1, B2 | M | — | ☑ `e8bb3509` |
| 4 Persistence & operation ordering | B7, B6 | S–M | — | ☑ `2ef20f66` |
| 5 Settings fallback | B4 | M | — | ☑ `c5487f92` |
| 6 Always-mounted card + wizard order | B3, U6 | L | 1 (test net) | ☑ `2bffa7b9` |
| 7 Cluster-wide lock | B5 | L | E3, 4 | ☑ `549fe8ff` |
| 8 Event loop & SSH robustness | B8, B9, B13 | M | 3, 7 (start.py) | ☑ |
| 9 Harden the upstream patches | R1, R2, R3 | M | 0 (UPSTREAM.md) | ☑ `e7c033fe` |
| 10 Efficiency | B11, B12, solver pruning/bisect, polling/fsync, B10 | L (split) | 8, E4 | ◐ 10a/b/c/e done; 10d deferred |
| 11 Remaining mediums/lows + test gaps + CI Node | API/UI mediums, T1–T9, session runtime tests | M–L | 1–9 | ◐ UI + test + nodes_total done; rest deferred |
| Live gate | Smoke + optional A/B | S | P0/P1 done, E5 | ◐ probe (§17) + inert stack (§18) live validated; apply/start deployment pending (GUI session) |

### Slice 0 — Hygiene & doc alignment (S) ☑

- `git rm --cached .live_prefill_driver.py .live_prefill_driver3.py
  .local-3rank-2026-11-13.md` + `.gitignore` entries (files stay local).
- UPSTREAM.md: document the 7 undocumented changes; correct lines 119–121
  ("no staging semantics changed").
- Per E1: align AGENTS.md §2, PROJECT-BRIEF, SCREEN-3-MINIMAL §4.1 (new
  §4.2), SCREEN-6-MANAGE with C1/C3; update
  `test_cluster_ui_integration_ui.py:312` to the hardware-only entry
  measurement (WIZTEST was already C3).
- **Acceptance**: `git status` shows the dotfiles untracked/ignored; the docs
  no longer contradict the code; the C3 UI test is green. The second red
  test (U6) stays red until slice 6 — deliberately, as a regression marker.

### Slice 1 — UI quick fixes + test net (M) ☑

- **U1**: fix the quote at `_cluster_v2.html:1302`. New runtime test: compile
  every `x-text`/`:href`/`@click`/… binding in the template the way Alpine
  evaluates it (expressions vs. statement bodies) — catches the whole bug
  class; verified it rejects the original broken expression.
- **U5**: gate `assistantSessionEndpoint()` on `ready && !stopped`
  (reuse `assistantSessionRunning()`) + runtime test.
- **Small Mac**: `ASSISTANT_SMALL_MAC_RAM_GB` instead of
  `this.constructor.SMALL_MAC_RAM_GB` (`cluster_v2.js:4998`) + runtime test.
- **U3**: "Use all Macs" into a named method: busy/running guard, generation
  bump, hidden/disabled in `ready`/`stopped` + tests.
- **U4**: cap `assistantCandidates()` at 32 client-side (deterministic
  priority: chosen model first, then peer-held, then smallest) + 32/33
  boundary test; readable 422 message.
- Busy guards + `:disabled` on the plan/start/stop buttons (double-click
  protection).
- Deferred to slice 11: the UI runtime suite should **fail instead of skip**
  without Node (CI protection; needs an escape hatch for local dev without
  Node).
- **Acceptance**: both UI suites green with Node (PATH prefix §4), except the
  known U6 marker (stays red until slice 6); the new expression test green;
  the assistant suite unchanged green.

### Slice 2 — Measurement recovery & poll robustness (M)

- **U2**: `failAssistantFrom` adopts `payload.measurement_id`; the `measure`
  action resumes a named running measurement instead of resetting; runtime
  test mirroring the `operation_in_progress` resume (UITEST:953).
- Resume clears `assistant.error` and restores the state (`measuring` or the
  operation's state); handle a `running` snapshot explicitly.
- **U7**: a resumed apply auto-starts only when `plan && inputs` exist;
  otherwise rest on the applied/session surface + test.
- Pollers: count consecutive errors, bounded retry with backoff (e.g. 3),
  keep the timer alive on transient errors + tests.
- **Acceptance**: the reproduced 409 loop from the review is covered by a
  runtime test and green.

### Slice 3 — Chat profile & reload (M)

- **B1, red first**: bring the fakes (`assistant_harness.py:301`,
  `test_start_api.py:657/684`) to the real catalogue shape (sampling under
  `entry["settings"]`) → the "never overwrites" test goes red.
- The guard reads `entry.get("settings", {})` and keeps the **all-or-nothing**
  semantics (frozen: the profile is applied "when the model has none"; a
  user-set value skips the whole profile). `force_sampling` is checked truthy
  (its default is `False`, so `is not None` would block every model). A
  contract test against the real `list_models` shape (entries built from the
  real `ModelSettings` dataclass via `asdict`) pins the shape so fake drift
  cannot silently return.
- **B2**: `ops.reload()` only when the target model is not already listed
  (`_model_entry`); tests: model present → no reload, model absent → reload.
- **Acceptance**: `tests/cluster_assistant` green; the new contract tests
  prove the catalogue shape + merge behaviour; no unconditional reload left
  in apply/start.

### Slice 4 — Persistence & operation ordering (S–M)

- **B7**: `OperationStore.applied()` searches backwards (like
  `latest_session`); a test with two applies → the newest wins.
- **B6**: a save failure does not roll back the terminal state; the operation
  goes explicitly to `failed` with the action `diagnostics`; `OSError` is not
  swallowed silently; a test with an unwritable store directory.
- **Acceptance**: the assistant suite green incl. the new failure-path tests.

### Slice 5 — Settings fallback (M)

- **B4**: resolve the path from the server configuration (local) or pass it
  in the remote snippet payload instead of hardcoding `~/.omlx`; a read
  error → abort with a warning instead of `{'version':1,'models':{}}`; a
  unique temp name (e.g. pid/uuid).
- Tests: a moved base path, a corrupt file → refuse-not-wipe, the policy is
  really applied on the worker.
- **Acceptance**: the assistant suite green; no code path left that replaces
  an existing settings file on a read error.

### Slice 6 — Always-mounted card + wizard order (L, highest UI risk)

- **B3**: lift the assistant card (TPL 835–1367) out of the `plan` section
  into an always-mounted block; the `data-assistant-state` markers stay; no
  new `WIZARD_STATES`.
- **U6**: the `choosingModel` check before the `configuredDeployment()`
  shortcut (or gate the shortcut with `&& !this.choosingModel`) — restore the
  baseline order.
- **New visibility tests**: section rendering with a populated
  `deploymentsPayload` (the gap that hid B3): the card is visible in
  idle/ready/stopped despite an existing deployment; the picker opens on a
  model change.
- **Acceptance**: both UI suites green — incl. the second formerly red test
  (`test_model_change_keeps_other_saved_setup_and_opens_picker`); a manual
  browser smoke (local server, no cluster start) for idle → ready → stop.

### Slice 7 — Cluster-wide lock (L)

- A short design note in CHAPTER-5.md (E3, option A).
- **T1 first (red)**: a two-origin test — origin A starts remotely on
  coordinator C, origin B asks `operations/active` on C and must see the
  running operation; today null → red.
- `_remote_ops` creates/updates assistant operation records on the
  coordinator through the admin seam; the lifecycle (queued→…→ready/failed)
  is mirrored remotely; cleanup on abort.
- **Acceptance**: T1 green; a second origin gets `409 operation_in_progress`
  + action `status`; the assistant suite green.

### Slice 8 — Event loop & SSH robustness (M)

- **B8/B9**: `plan`, the apply re-election, `_topology_refusal`, the busy
  checks into `asyncio.to_thread` (pattern: `recommend`, routes.py:1403).
- **B13**: wire the SSH keepalive (`ssh_policy.py:55–62` exists);
  `_poll_staging` gets a deadline; an unknown local job → `failed` instead of
  stuck; tests with a fake clock.
- **B6 follow-up (from slice 4)**: the start job's `_advance`/`_fail` catch
  only `KeyError` (~start.py:1216/1228); handle `OSError` so a save failure
  surfaces as `failed` with action `diagnostics` instead of being lost. The
  store already keeps terminal states (slice 4); this wires the job-level
  error path.
- **Acceptance**: the assistant suite + the affected upstream cluster tests
  green; no synchronous solver/SSH call left in an async handler (grep
  check).

### Slice 9 — Harden the upstream patches (M)

- **R1**: the remote inventory snippet imports `mtp_compatible_for_model`
  guarded (try/except ImportError → fallback), so stock-oMLX peers keep their
  inventory + a test with a simulated ImportError.
- **R2**: `_local_interface_addresses` does not cache a failure/empty set
  permanently (invalidation or no caching on failure) + test.
- **R3**: document the pre-load patch side effects in
  `inspect_safetensors_layout` and pin them with a test (behaviour stays —
  frozen).
- Add the UPSTREAM.md entries (the rest from slice 0).
- **Acceptance**: the upstream + assistant suites green; UPSTREAM.md
  complete.

### Slice 10 — Efficiency (L, split into sub-slices)

Progress: **10a done** (`549fe8ff`, recommendation ranking), **10b done**
(`112dffb6`, microbatch). 10c delegated (solver cost). 10d waits for slice 8
(shared `start.py`/`routes.py`).
**10e (probe physical plausibility) is deferred to the live gate**: improving
the GEMV kernel/iterations needs a real measurement to confirm the rates are
plausible, and the cluster is paused; the election-weighting part additionally
needs decision E4. No token-rate estimates are added (frozen).

- **10a Recommendation ranking (B11)**: away from `weight_bytes`, towards
  "fits + usable context + estimated critical stage"; the A/B script shows
  Qwen3.8 above dev-v41-15L.
- **10b Microbatch (B12)**: thread the requested `microbatch_size` through
  `_policy_for`; a plan/launch consistency test.
- **10c Solver cost**: an early filter (summed admission), bisect the context
  ladder (fix the docstring), memoization; target worst case < 1 s (baseline:
  2260 ms / 26,880 calls). Rerun the A/B script.
- **10d Polling/fsync**: a visibility gate for the assistant pollers, a
  lightweight status projection instead of full inventories, a 2–5 s
  interval + backoff, no whole-store fsync per tick, `ops.models()` once per
  start, parallelize the policy writes; consider SSH ControlMaster.
- **10e Probe (B10, only per E4)**: make the kernel/iterations
  bandwidth-bound, document the latency nature; a weight change only with a
  decision.
- Optional: move the A/B script into the repo as a dev tool (`scripts/`) so
  the baseline stays reproducible (live evidence still only in
  PHASE-2B-RUN.md).
- **Acceptance**: the A/B numbers improved/equal, all suites green.

### Slice 11 — Remaining mediums/lows + test gaps (M–L)

- API: the one-job check into the store, `_redacted` real redaction, delete
  target resolution (loopback protection, TTL), remove or implement the
  `admission` field, align the status codes (400 vs. 409), quote the IDs,
  reject duplicate SSH targets, dynamic `nodes_total`, surface `load_error`,
  split `routes.py`.
- Launch: map remote refusals correctly (`plan`/`copy_model` instead of a
  blanket `retry_coordinator`), do not persist `observed` with `decode=None`,
  partial policy writes → action `apply`.
- UI: the clipboard failure path, refresh the details JSON after stop, route
  Refresh through `applyAssistantOperation`, redact the endpoint in the debug
  pane before ready, `aria-live`/`role="alert"`, move the hardcoded strings
  into the catalog, preserve the session's `node_selection` on resume + an
  honest label.
- Tests: session-card runtime coverage, the rest of T1–T9, Screen-4
  rate-freedom on the row messages too. Fix the load-sensitive flake
  `test_observed_api.py::test_streaming_fills_ttft_and_prefill_and_keeps_decode`:
  its `_wait` returns when the non-streaming probe fills `observed`, before the
  streaming merge lands, so under parallel load it asserts the intermediate
  prefill value (passes in isolation). Wait for the streaming merge specifically.
- CI: the UI runtime suite fails instead of skipping without Node (with an
  escape hatch for local dev).
- **Acceptance**: the review mediums/lows ticked or deliberately deferred
  (with a reason here in the plan).

### Live gate (after P0/P1, only with the E5 go)

1. Bring the cluster back up (order/STP per PHASE-2B-RUN §10).
2. Smoke: measure → recommend → plan → apply → start → ready → stop →
   Start again; check the chat-profile behaviour with a user setting (B1
   live).
3. Optional A/B: upstream vs. assistant start, compare the numbers.
4. Evidence (IDs, timings, observations) into `PHASE-2B-RUN.md`.

---

## 4. Verification ladder (per slice)

```sh
# 1. The slice's target tests
uv run pytest tests/cluster_assistant/<affected file> -q

# 2. The assistant suite
uv run pytest tests/cluster_assistant -q

# 3. The UI suites (Node must be on PATH; otherwise 105 tests skip silently)
export PATH="/private/var/folders/yy/b17vq8697xv9w_xxtfdk3lw40000gn/T/opencode/node/bin:$PATH"
uv run pytest apps/omlx-web/tests/ui/test_cluster_v2_wizard.py \
              apps/omlx-web/tests/test_cluster_ui_integration_ui.py -q

# 4. Periodically (after each backend slice): the upstream cluster tests
uv run pytest tests/test_cluster_plan_approval.py tests/test_cluster_ui_integration.py -q

# 5. Before the live gate: the full suite
uv run pytest tests/ -q --ignore=tests/cluster_assistant -p no:cacheprovider
```

Known pre-existing failure (not caused by us, not counted):
`tests/test_glm5_next_mtp.py::test_split_k_batched_decode_matches_rows_alone`.

CSS after template changes: `cd apps/omlx-web && python build_css.py`.

---

## 5. Risks and countermeasures

| Risk | Countermeasure |
|---|---|
| B3 (template restructure) breaks other wizard states | Slice 1 first (test net), then visibility tests with deployments, then the restructure; a manual browser smoke |
| The B1 fix corrects only the fakes, not reality | A contract test against the real `list_models` shape; red-first discipline |
| B5 mirrors operations remotely → new distributed state | A design note up front; T1 red/green; test the cleanup path; no new seam |
| Slice 10 changes election behaviour (frozen) | Decide E4 explicitly; only the probe quality without a weight change is decision-free |
| Parallel slices conflict in `cluster_v2.js`/`start.py` | UI sequential, `start.py` slices sequential (3 → 7 → 8) |
| The live gate disturbs the paused setup | Only after E5; order + STP per PHASE-2B-RUN §10 |

---

## 6. Tracking

- Tick the checkboxes in §3 per slice; the review document stays unchanged
  (evidence).
- Commit messages: `assistant: fix B1/B2 — chat profile guard + conditional
  reload` (always name the finding IDs).
- Deliberately deferred findings are noted here with a reason, so the review
  is not silently "forgotten".
---

## Real-measurement architecture (product direction 2026-11-13, built)

Decision: ranking rests on REAL model measurements, not the synthetic probe;
the optimization loop is fully automatic (budget 3). oMLX's planner keeps every
layer split; the assistant only chooses rank orders / subsets and measures.

| Component | File | Role |
|---|---|---|
| Calibration conversion | `calibration.py` | real dev-1.5b tok/s -> ComputeProfile (decode=bandwidth, prefill=compute) |
| Calibration runner | `calibration.py` `calibrate_node` | warmup+measured completion via the observe seam -> real per-node profile |
| Measurement preference | `measurement.py` | real calibration preferred over synthetic probe (fallback intact) |
| Candidate generation | `policy.py` `diverse_candidates` | best feasible plan per subset size (2..4), ranked, deployable |
| Best selection | `evaluation.py` | Pareto + priority over MEASURED constellations; diverse candidate pick |
| Search loop | `search.py` `run_search` | async: evaluate each -> pick best measured -> stop losers |
| Evaluate adapter | `search_wiring.py` | plan_apply -> run_apply_job -> run_start_job -> observe (real seams) |
| Search route | `routes.py` `POST /search` | wires the loop to the getters; API-first |
| Fine-tuning | `tuning.py` | chunk/sampler/MTP sweeps, golden-prompt quality, hardware-keyed presets |

The loop: rank (hardware estimate) -> oMLX plans+starts -> measure -> repeat over
worth-testing constellations -> pick best measured -> fine-tune. Spec-grounded
probe expectations are pinned in `test_probe_spec_expectations.py`; live evidence
in PHASE-2B-RUN.md §17.

Wired and tested (assistant + upstream + UI green):
- Calibration runner wired into the live measure job (real per-node profiles
  preferred over the synthetic probe, with fallback; `stream_completion` seam).
- `POST /assistant/tune` fine-tunes the winning constellation (chunk/sampler/MTP,
  golden-prompt quality) through the coordinator completion seam.
- Dashboard UI for the search loop ("Find the best setup", priority/max-count,
  results table; generation/busy guards; 23 i18n keys mirrored).
- Live calibration validated on .11 (PHASE-2B-RUN.md §19): real dev-1.5b decode
  100.5 GB/s, plausible.
- Single-node vs cluster benchmark (PHASE-2B-RUN.md §22): single M5 Max 254 tok/s
  vs 4-Mac cluster 53 on dev-1.5b — single-node wins for small models.
- Single-node recommendation (E6): the recommendation offers a `single_node`
  option when the selected model fits on one Mac at the target context (additive,
  backward-compatible; `mode` field; reuses budget_for).
- Async search job: POST /search returns a handle, the loop runs in the background
  with progress, GET /search/{job_id} polls; the UI polls (no more minutes-long
  blocking request).

Remaining: a full live cluster search smoke with the async job + single-node
option (E5), and the SMJobBless helper install click on the nodes (user GUI
action, optional for the cluster). The live search smoke ran successfully before
the async refactor (PHASE-2B-RUN.md §21); re-run it against the async job to
confirm the polling path end-to-end on hardware.
