# Rebase verification — 2026-11-07 (working note)

Status: working note, committed 2026-11-12 as live evidence. Base: `1e27576a`
(`cluster-assistant-base-2026-11-07`), branch `feat/cluster-assistant`.

## What was verified

- Rebase onto the complete cluster base; `tests/cluster_assistant` 271 passed,
  `tests/test_cluster_*.py` 1614 passed (2 skipped), web 317 passed
  (179 skipped), ruff clean.
- The branch was deployed to all four Macs at `~/omlx-cluster-assistant`
  (`.11` runs the working tree). `.12` and `.13` were paired; `.14` was already
  paired. All four run `0.7.4.dev1`.
- Each Studio starts the server from `~/start-omlx-deploy.command` in a
  Terminal. `~/.omlx/bin/omlx-cluster-python-branch` sets `PYTHONPATH` to the
  deployed branch; without it an SSH-launched worker imports the reference
  install (0.7.3) and the launch version check refuses it.

## Live matrix

| Model | Nodes | Result |
|---|---|---|
| Qwen3.8-Flash-Next-oQ5e (125.7 GiB, PLE) | 2 (`.12`+`.11`) | ready, PLE on SSD, 43 tokens, 11.67 tok/s |
| Qwen3.8-Flash-Next-oQ5e | 4 (all) | ready, PLE, 43 tokens, 11.61 tok/s |
| GLM-5.3-Flash-oQ2e (107.8 GiB) | 2 (`.11`+`.13`) | ready, 17 tokens, 18.65 tok/s |
| GLM-5.3-Flash-oQ2e | 3 (`.11`+`.12`+`.13`) | ready, 17 tokens, 4.96 tok/s |
| GLM-5.3-Flash-oQ2e | 4 (all) | ready, 21 tokens, 13.35 tok/s |
| dev-v41-15L (Engram, 4k) | 2 (`.12`+`.11`) | ready, 64 tokens, 22.20 tok/s |
| dev-v41-15L (Engram, 4k) | 4 (all) | ready, 64 tokens, 17.47 tok/s |
| dev-v41-15L (MoE 25 %, table resident) | 2 (`.11`+`.12`) | ready, 256 tokens, 31.63 tok/s |

The 3-node rate is lower because of the extra hops; all rates are the
endpoint's own `usage`, not estimates.

## Fixes found by the live run (committed as `4998eb7b`)

1. Rank 0 is spawned over SSH to the coordinator's **own LAN address**, not as a
   local child: macOS Tahoe fails the local child with
   `[ring] Couldn't connect (error: 60)`.
2. A worker that receives the model through staging is not yet in that Mac's
   model pool. The coordinator's levers are written before `prepare`; workers
   after staging, with a per-model settings-file fallback on `404`.
3. The measurement records each Mac's interpreter; the worker wrapper above is
   the deployed interpreter on the Studios.

## Known limits (not fixed here)

- `dev-v41-15L` is a stage-fragment set on `.13`/`.14` (14 / 5 of 23 shards).
  A different split (4-node MoE) needs a shard that is absent; the 4-node
  Engram split and the 2-node MoE split work. Leave the models as they are; a
  full copy is only needed for a re-split.
- `.13` reports a lower admission than its 64 GiB after a restart (roughly
  23–27 GiB); the remaining pressure is outside this application.
## Phase C1 / C3 implemented after the live matrix (unversioned note)

### C1 — reload resume and the stop lock

- `OperationRecord` now keeps `model_path` and `target_context_tokens`, so the
  persisted operation already carries the plan inputs a reloaded card needs.
- `GET /admin/api/cluster/assistant/session` returns the newest ready start (or
  applied plan): operation id, measurement id, plan signature, model/context,
  MTP, execution profile, microbatch, coordinator, rank order, resource policy,
  deployment id and endpoint. No key, SSH target or host is stored.
- A stop is refused with `409 operation_in_progress` while any local operation
  is running; for a remote coordinator the same question is asked through the
  existing admin seam. A not-ready running operation now answers
  `operation_in_progress`, not `operation_not_ready`.
- The card reads the session once on load and shows `Last deployment` with
  `Start again` (re-applies the stored inputs, then the existing poll starts)
  or `Stop` while it is running. A session older than the measurement TTL
  surfaces the normal `stale_measurement` / `measure` refusal.

### C3 — model-less (hardware-only) measurement

- `ModelIdentity` has a third identity: `hardware_only=True` (no path, no size,
  no MTP). The measurement stores `layout=null`, `model_present=true` (unknown,
  not false) and no model paths.
- `recommend` reads a hardware-only measurement; `plan` refuses it with
  `stale_measurement` / reason `model` and the one action `measure`.
- The card's entry measurement is now hardware-only. It recommends, then the
  existing model-bound measurement measures the chosen model before plan.
  This removes the “measure the wrong model first” step.

Tests added/updated: `test_session_api.py` (4), hardware-only measure and
identity tests, router/session allowlists, operations key check, stop lock.
Suite after these changes: 276 assistant, 1614 cluster, 317 web, ruff clean.

### C1/C3 live check (same session)

- **C1:** a fresh 2-node GLM start (`op-20261107T114813Z-start-0002ca3c`,
  deployment `GLM-5.3-Flash-oQ2e-mtp-df1ae7ead9be`) returned
  `GET /assistant/session` with `model_path`, `context_tokens: 8192`, `state:
  ready`, `stopped: false`, deployment id and endpoint. After stop the same
  session reported `stopped: true`. The cluster was left with no loaded model.
- **C3:** `ms-20261107T114748Z-0001418f` was measured with **no model**
  (`status.model: "hardware only (no model measured)"`). `recommend` returned
  `hardware_only: true`, `feasible: true` for GLM; `plan` refused with
  `stale_measurement`, reason `model`, action `measure` and “measure the chosen
  model before planning”. This is the intended Option A/C3 handoff.

Suites after C1/C3: 276 assistant, 1614 cluster, 317 web, ruff clean.

## Phase B live check (2026-11-07, unversioned note)

### B1 — compute probe after apply, used by the next plan

- Apply `op-20261107T121139Z-apply-000158f4` finished `applied` with **no**
  `compute_probe_failed`; the synthetic probe stored profiles for `.11` and
  `.14` in `~/.omlx/cluster/assistant-performance.json` (both
  `measured_at 12:11:42`).
- The next model-bound measurement `ms-20261107T121317Z-00011b9f` came back
  with **no warnings** (no `unmeasured_compute`) and the plan
  `c9bbd2e5eab9ee8e9636b2d7e72850d037a6cad8279667b1c2bdaf0ba3699387` reports
  predicted stage seconds (slowest 5.086 s) and elects on the fastest predicted
  critical stage. That is the measured-compute path, not a rate estimate.

### B2 — streaming TTFT/prefill after ready

- With `assistant_performance_probe` on, start
  `op-20261107T121343Z-start-00027112` (deployment
  `GLM-5.3-Flash-oQ2e-mtp-47321e7b7808`) produced
  `observed: {prefill_tok_per_s: 9.06, decode_tok_per_s: 21.43, ttft_ms: 110.0}`.
  The non-streaming completion supplied decode; the streaming final usage chunk
  supplied TTFT/prefill. `stream_options.include_usage` is required, otherwise
  the final chunk carries no usage.
- A streaming failure is one `performance_probe_failed` warning and keeps the
  decode block; it never unreadies the cluster.

### B3 — refresh and the in-app probe switch

- `GET /assistant/performance-probe` returned `{"enabled": false}`; POST true
  and false round-tripped and persisted in the server settings. The card has a
  `Performance probe` button and a `Refresh` action that re-reads the operation
  once (so a probe that lands after the ready poll is visible without reload).
- After the run the probe was turned back **off** and the cluster stopped;
  `.11` and `.14` report `loaded_count: 0`.

Suites after B1/B2/B3: 285 assistant, 1614 cluster, 317 web, ruff clean.


## Offline iteration 2026-11-07 — C2, D, DAU diagnostics

Cluster offline; no live test was run. Commit `66e31a8a`.

- C2: the `copy_model` action now runs the assistant flow; apply stages from a
  complete peer (`test_copy_model.py`: source host m1, no download; missing
  source -> 409 `model_source_missing` / `copy_model`). The free-space delete
  refuses the last measured copy (`last_cluster_copy`, action
  `free_disk_space`) and still refuses the model the cluster is about to run.
- D: `test_pp_plan.py` (3/4-node contiguous stages with rank 0 on the tail,
  full admission check, minimal subset + excluded names, MTP charge, 3-node
  permutations) and `test_any_to_any.py` (the same plan for all four origins, an
  infeasible origin is not rank 0, a 125 GiB model on four Macs still elects
  m5+m2 with m1/m4 excluded).
- DAU: `Copy diagnostics` in Details with measurement, plan, coordinator, rank
  order, excluded, policy, operation, deployment, endpoint, observed and
  warnings; no key, token, SSH target or full path.
- Test harness: the coordinator's cluster-wide active-operation read is faked to
  "unanswerable" by default; the suite is back to ~6 s instead of ~12 min (a
  real SSH timeout). Two lock tests set their own fake after `api()`.

Suites: 300 assistant, 1614 cluster, 319 web; ruff clean on touched files.

### Open questions for the resolve iteration

1. What is "complete topology" for a 3/4-Mac plan: every pair connected, a
   line, or a ring? The validation cluster runs as a line but the plan requests
   `backend="ring"`. Should an incomplete fabric refuse the 3/4 plan with
   `topology_incomplete` and offer the 2-Mac plan only after explicit
   confirmation? Current code has no fabric-completeness fact.
2. Do the remaining P2 items (macOS app/tray, signed helper, telemetry, TP,
   >4 Macs) stay out of this iteration and after Chapter 10?
3. Is the 2-Mac fallback confirmation recorded in the plan signature/policy so
   apply/start cannot silently run a different subset?


## Deferred live test matrix (run as one block when the cluster is back)

1. **2-Mac baseline** (`.11` + `.14`): measure -> recommend -> plan -> apply ->
   start, one chat completion, endpoint ready only after health/smoke; probe
   off, `observed` null.
2. **3-Mac line** (`.11` + `.13` + `.14`): a model that does not fit two Macs;
   minimal plan chooses three; apply/start, six progress rows, Stop, Start
   again, second chat completion.
3. **4-Mac line** (all four): `node_selection=all`, plan size four, apply/start;
   verify the line fabric and the rank order; a second origin gets
   `409 operation_in_progress` while it runs.
4. **Any-to-any**: originate from `.11`, `.13` and `.14`; the same elected
   coordinator/plan each time; an infeasible origin never becomes rank 0 and
   there is no local fallback.
5. **Copy flow**: model only on `.14`, elected coordinator `.11`; apply stages
   from `.14` (`source_host`), no download. Delete a duplicated model on `.13`
   (allowed); attempt the last copy (409 `last_cluster_copy`); attempt the
   planned model (409 `model_in_use`).
6. **Dev model** `dev-v41-15L` (stage-fragmented 23/14/5 shards): full copy
   needed; verify the disk guard and source-first staging; do not delete the
   last copy.
7. **Probe on/off**: apply with the compute probe; the next measurement/plan
   uses the profiles; start with `assistant_performance_probe` on and verify
   `observed` prefill/decode/TTFT; toggle off, Refresh once.
8. **Reload resume**: start, reload, `GET /assistant/session` returns the
   deployment; Stop; Start again; Stop while another operation runs -> 409.
9. **Error paths**: coordinator offline (`retry_coordinator`), unpaired Mac
   (`pair_mac`), no disk (`free_disk_space`), no source (`copy_model`), and an
   incomplete 3/4 fabric (the consented 2-Mac fallback, once defined).
10. **3/4-node consent**: disconnect one link, run a 4-Mac `all` plan; the plan
    must name the missing link and offer 2 Macs only after confirmation; the
    accepted plan signature records the fallback.


## Resolve iteration 2026-11-07 — topology line and the consented fallback

Answers: (1) a complete fabric is a **connected line**; (2) an incomplete line is
refused with `topology_incomplete` and the 2-Mac plan is offered only after
explicit confirmation, with that consent in the plan signature; (3) P2 stays out.

Implemented offline (cluster still unavailable):

- `omlx/cluster/assistant/topology.py` — missing consecutive rank pairs.
- The measurement stores `fabric_links` (node-ID pairs) and `fabric_measured`.
  An empty transport matrix is **not measured**: the nominal-ring warning stays.
- `elect`/`solve_cluster_plan` accept the measured links; a rank order is only
  feasible when every consecutive pair is linked. `max_size` caps the offered
  fallback at two Macs.
- `plan` returns `409 topology_incomplete` with `missing_links` and a full
  `fallback` plan when two Macs fit. `confirm_two_mac_fallback=true` returns the
  2-Mac plan with a salted signature and a `two_mac_fallback` warning; apply and
  start repeat the consent or fail `plan_mismatch`. A 2-Mac plan whose pair is
  unlinked is refused too (no fallback).
- `AssistantPlanResponse.topology` reports `measured/complete/missing_links/
  fallback_used`; the card shows the missing cable and a “Run with 2 Macs”
  button, and the diagnostics block includes the topology.
- Fixed a latent UI/backend mismatch found while wiring this: the browser
  spreads the frozen plan inputs into `apply`, but `AssistantApplyRequest`
  forbade `node_selection` (and now also receives
  `confirm_two_mac_fallback`); both are accepted, the approved rank order still
  decides the subset.
- Reload resume carries `two_mac_fallback` so Start again re-proves the same
  signature.

Suites: 314 assistant, 1614 cluster, 320 web; ruff clean on the assistant and
touched test files. The live cable-pull scenario is in the deferred matrix.


## Local single-node smoke 2026-11-07 (cluster offline)

The local oMLX server on `127.0.0.1:8099` was used read-only. No model was
staged, no rank was started, no server was restarted, and the admin key was
never printed or copied into a file.

- Hardware-only measurement `ms-20261107T193226Z-00024bc5` with node `local`
  (`ssh 127.0.0.1`) became ready; admission p05 ≈ 96.77 GiB; warning
  `unmeasured_compute` only.
- Recommendation over `/Users/andreaskolb/Models/dev-1.5b` and
  `.../GLM-5.3-Flash-oQ2e-mtp`: `dev-1.5b` feasible at 8192; GLM infeasible with
  action `choose_smaller_model`.
- Model-bound measurement `ms-20261107T193320Z-0003115a` for `dev-1.5b` became
  ready; plan `3133f42d0c6cc6f37d76da0f62fac0c74954d26d685c390b9096b91d3c7e8383`
  is feasible with `rank_order: ["local"]`.
- The **running** server (started before the `node_selection` apply fix) answered
  `422 extra_forbidden` for `node_selection`. The current code accepts it and a
  direct `plan_apply` on the real record refuses
  `single_node_plan` / `add_mac` before resolving any target:
  "this plan runs the model on one Mac, and the distributed cluster path needs
  at least two; add another Mac and measure again".
- `build_diagnostics` on the real live record `ms-20261107T121317Z-00011b9f`
  and operation `op-20261107T121343Z-start-00027112` produced a 5476-byte bundle:
  model label `GLM-5.3-Flash-oQ2e-mtp`, both nodes, operation `ready`, plan
  signature matches, and the full model path, API key and SSH user are absent.
  The record predates fabric links, so `topology.measured` is false, as expected.

The new `/assistant/diagnostics` route itself is not loaded in the running
server (it predates the commit); it is covered by the fake-driven API tests and
will be live once the server is restarted. The local single-node smoke is
documented in `CHAPTER-10.md` §5.2.


### Local recommendation matrix (target 32768)

Using the same hardware-only measurement `ms-20261107T193226Z-00024bc5` and all
seven local models, `recommend` selected `dev-v41-15L-real` at 4096 tokens with
`context_below_target`. Feasible: `dev-v41-15L-real`, `Qwen3.8-Flash-Next-oQ5e-mtp`,
`dev-v41-15L`, `dev-1.5b`. Refused with `choose_smaller_model`:
`GLM-5.3-Flash-oQ2e-mtp`, `GLM-5.3-Flash-MLX-4bit-MTP`,
`DeepSeek-V4.1-Flash-1017-oQ2e-ngram-mtp`.

Finding to keep in mind (not fixed, per the dev-model decision): the local
`dev-v41-15L` is a stage fragment, but `recommend` sees its local layout and
marks it feasible. A real apply still needs a complete source; the fragment
stays untouched.


### Local server restart and new routes (2026-11-07)

The local oMLX server was restarted from this repository with the same command
(`uv run omlx serve --host 0.0.0.0 --port 8099 --model-dir ~/Models`); the new
process is PID 99581 and was left running. Nothing was staged or launched.

- `GET /assistant/diagnostics` on the real old record/operation returned 200;
  the model label is `GLM-5.3-Flash-oQ2e-mtp`, the operation is `ready`, and the
  full model path is absent.
- Fresh hardware-only measurement `ms-20261107T194128Z-0001ae6f` became ready;
  recommendation over `dev-1.5b` selected it at 8192.
- Model-bound measurement `ms-20261107T194129Z-0002e8a8` became ready with all
  seven local models in the inventory; the one-node plan is feasible with
  `rank_order: ["local"]` and `topology.measured: false` (no peers/links).
- `POST /assistant/apply` with the browser's `node_selection` now returns
  `409 single_node_plan` / `add_mac` (no 422, no operation, no staging).
- `GET /assistant/diagnostics` for the new measurement returns 200 with model
  label `dev-1.5b`, one node, and no full model path or API key.

This closes the local live check of the new diagnostics route and the
`node_selection` apply fix. The distributed live block is still pending.


### P2 app/tray slice (2026-11-07)

Decision: P2 starts with the macOS menu/tray entry; it opens the existing
Cluster tab in the existing web dashboard, with no new window, app, wizard or
control plane, and no status/stop in the tray.

- `MenubarController` now has a `Cluster Assistant` menu item,
  `openClusterAssistant()`, and the deep link
  `/admin/dashboard?tab=cluster` through the existing `/admin/auto-login` URL
  builder.
- `menubar.item.cluster_assistant` is in `Localizable.xcstrings` (English);
  `LocalizationSmokeTests` includes it, and
  `MenubarControllerPortTests.testWebAdminURLDeepLinksTheClusterTab` pins the
  encoded redirect.
- `swiftc -parse` accepts the changed Swift files and the string catalog parses
  as JSON. `xcodebuild` is unavailable on this Mac (only Command Line Tools),
  so the Xcode build and the two unit tests are still open; run them on a Mac
  with Xcode.
- The helper is not started: the chosen scope is Developer ID signing plus
  network setup, but the signing identity/notarization credentials and the
  privilege model are still open.


### macOS app build and P2 first slice (2026-11-08)

Xcode 27.2 was installed; the earlier "no Xcode" blocker is gone.

- `xcodebuild -scheme oMLX -configuration Debug build CODE_SIGNING_ALLOWED=NO`
  succeeds.
- Focused tests pass: `PerformanceScreenVMTests`,
  `MenubarControllerPortTests`, `LocalizationSmokeTests`,
  `WelcomeViewModelTests` — 55 tests, 0 failures.
- The full app suite ran 389 tests with 7 failures in
  `ModelSettingsScreenVMTests` and `ThemeTests`, outside this surface; they are
  pre-existing/environment issues, not caused by the menu/Welcome/helper slice.
- The menu item, Welcome button, Developer ID team
  (`YB894QRG4W`) and the one-shot wired-limit admin action are in place.
- `security find-identity -v -p codesigning` reports **0 valid identities**;
  signing/notarization is still an external prerequisite.
- `package_dmg.sh` and `DISTRIBUTION.md` are in place; no final app bundle can be
  produced without the donor/venvstacks input, and no signed DMG without the
  certificate.
- The logs showing `visibility probe: hidden=true`, the recreated NSStatusItem
  and the Full Disk Access alert at 08:50/08:52 came from the **XCTest host**,
  not the production app. The probe now skips under XCTest
  (`XCTESTConfigurationFilePath`), so unrelated test runs no longer trigger it.
  The production menu item and its recovery path are unchanged.


### Full app bundle, DMG, network preflight (2026-11-08)

- Installed CPython 3.11 via `uv python install 3.11` because venvstacks
  cannot build `webrtcvad` with the repo's 3.12 venv.
- `PYTHON_BIN=<uv 3.11> apps/omlx-mac/Scripts/build.sh release` built the full
  staged app at `apps/omlx-mac/build/Stage/oMLX.app` (1.6 GB, Python 3.11 +
  framework-mlx-base + omlx/omlx_web).
- `apps/omlx-mac/Scripts/package_dmg.sh` produced
  `build/oMLX-0.7.4.dev1.dmg` (884 MB, **unsigned**, no notarization):
  SHA-256 `fbf96753ea99b3f91f2d144f3d3b36f7110929632650a0d76ce7df0589b315ed`.
  `hdiutil verify` is valid; the mounted image contains `oMLX.app` and the
  `/Applications` symlink.
- `security find-identity -v -p codesigning` still reports 0 identities. The
  Developer ID/notarization setup is recorded in `SIGNING.md`; signing and the
  SMJobBless install remain blocked on the certificate.
- Read-only network preflight: `GET /assistant/network-preflight` uses
  `transport.probe_host_interfaces`, returns interface evidence + exact
  bridge command templates, warns about missing Thunderbolt and ring/STP, and
  returns `executed: false`. The card has a Details panel and Copy commands;
  it never executes anything. 5 new builder/route tests.
- App-focused tests: 55 pass. Python: 325 assistant, 1614 cluster, 321 web;
  `ruff` clean.


### Network preflight live check (2026-11-08)

After restarting the local server on the new route, `GET
/assistant/network-preflight` returned `executed: false`,
`confirmation_required: true`, and the local evidence:

- bridges: `bridge0`; candidate Thunderbolt ports: `en1`, `en2`, `en6`;
- warning `existing_bridge: 127.0.0.1 already has bridge0; the preflight will
  not modify it`;
- commands: do not create a second bridge, list the candidate ports, and
  suggest adding only the peer port (`sudo ifconfig bridge0 addm en1`), then
  review the bridge members/address in System Settings.

The first live run before the fix produced the unsafe `create` +
`addm bridge0`; `3ca7a1e4` fixes it and the 6 preflight tests pin it. Nothing
was executed.


### Slim runtime / DMG (2026-11-08)

The 1.6 GB embedded runtime was rejected as bloatware. The new path:

- `provision_runtime.sh` uses the repo's `.python-version` (3.12.16), preferring
  Homebrew `python3.12`, otherwise a tagged `uv` CPython. It installs the locked
  dependencies with `uv sync --frozen --no-dev --no-install-project --extra
  cluster` into `~/Library/Application Support/oMLX/runtime/venv` (~1.0 GB,
  outside the app). The app source stays in `Contents/Resources`.
- `build.sh release --slim` stages the Swift app + `omlx`/`omlx_web` and the
  runtime wrappers, skipping `cpython-3.11` and `framework-mlx-base`.
- `PythonRuntime.resolve()` prefers the Application Support venv before the
  legacy embedded layout.
- Slim app: **146 MB**; slim DMG: **58 MB**, SHA-256
  `836f3f7fbe31fb9d74cc1516f35413ce39283f3704aa588a7533af3c97b644a1`;
  `hdiutil verify` valid and the mounted image contains `oMLX.app`.
- `oMLX.app/Contents/MacOS/omlx-cluster-python -c "import omlx"` uses the
  app's `Resources/omlx` from a non-repo cwd.
- The old 1.6 GB `Stage`, 1.5 GB `packaging/_export` and 884 MB DMG were
  removed from the working tree; they are reproducible build artifacts.
- Open: first-run GUI runtime installer, distribution of the lock/provisioner,
  signing/notarization, and runtime repair/update.


### Developer ID signed slim DMG (2026-11-08)

- Developer ID Application certificate imported from the portal; the public key
  matched the locally generated CSR/key. The Developer ID G2 intermediate from
  Apple was installed; `security verify-cert` is successful and
  `security find-identity -v -p codesigning` reports **1 valid identity**:
  `Developer ID Application: Andreas Kolb (YB894QRG4W)`.
- The keychain access was allowed; `codesign` no longer hangs.
- `apps/omlx-mac/build/SlimStage/oMLX.app` is signed with Developer ID,
  hardened runtime and the app entitlements; TeamIdentifier `YB894QRG4W`,
  timestamp present; `codesign --verify --deep --strict` passes.
- Signed DMG: `build/slim-signed/oMLX-0.7.4.dev1.dmg`, 59 MB, signed,
  SHA-256 `1c1c41f9cb72a9a21cbb126104bb1ce00ca4434cab100c7bec633f0f9e00ab99`.
  `hdiutil verify` passes; the inner app signature is the same identity.
- Gatekeeper currently returns `source=Unnotarized Developer ID`; notarization
  is the remaining step. No `omlx-notary` profile exists yet.


### Notary profile and submission (2026-11-08)

The `omlx-notary` keychain profile now exists; `xcrun notarytool history
--keychain-profile omlx-notary` returned `No submission history` (the expected
empty state). The signed slim DMG
`build/slim-signed/oMLX-0.7.4.dev1.dmg` (59 MB, SHA-256
`1c1c41f9cb72a9a21cbb126104bb1ce00ca4434cab100c7bec633f0f9e00ab99`) was
submitted with `notarytool submit --wait`. The submission result is pending;
stapling and `spctl` verification follow only after it is accepted.


### First-run runtime installer and SMJobBless helper (2026-11-08)

- `RuntimeInstaller` is embedded in the slim app with the pinned
  `uv` binary (27 MB), `pyproject.toml`, `uv.lock`, `.python-version` and the
  fallback provision script. The Welcome setup shows an explicit **Install
  Runtime…** card with progress/log; it runs `uv python install`, `uv venv`,
  and `uv sync --frozen --no-dev --no-install-project --extra cluster`. The
  first-run path with a fresh Application Support directory is built but not
  yet clicked through end to end.
- `apps/omlx-mac/Helper/` is a signed SMJobBless helper for
  `iogpu.wired_limit_mb` over XPC. `SMJobBless` uses one standard macOS admin
  authorization dialog; the one-shot AppleScript remains a fallback. Helper
  and app designated requirements pin `app.omlx`/`app.omlx.helper` to team
  `YB894QRG4W`. The helper install itself still needs a click on the signed
  app (the code compiles and both signatures verify).
- Final signed DMG: `build/slim-helper2/oMLX-0.7.4.dev1.dmg`, **72 MB**,
  SHA-256 `cc80d16e361c5adf2e2ab42eec206483e7e7f88730697e523efefb6d958c354b`.
  It contains the runtime installer and signed helper. Signed DMG verification
  and `hdiutil verify` pass.
- The final DMG was submitted to notarytool; the result is pending. Gatekeeper
  still reports Unnotarized Developer ID until Apple accepts and the image is
  stapled.


### Notary queue status (2026-11-08)

- `554bb403-f6e2-4489-b0ca-6a8095506483` — the **old 59 MB**
  `build/slim-signed` image (no helper/installer) was **Accepted**. It is not
  the final artifact and was not stapled.
- `ca2fd1f4-04c5-4323-8edd-6a77145b0edf` — `build/slim-final` (runtime
  installer, no helper) is still **In Progress**.
- `f9615eda-2935-47cd-bdec-6d065062e6e5` — the final 72 MB
  `build/slim-helper2` image (runtime installer + SMJobBless helper) is still
  **In Progress**.
- The final image remains signed and `hdiutil verify`-clean; staple and
  `spctl` wait only for the final submission. Apple's notary service is the
  current bottleneck, not the local build.


### Final notarization accepted (2026-11-08)

- Submission `f9615eda-2935-47cd-bdec-6d065062e6e5` for the final
  `build/slim-helper2/oMLX-0.7.4.dev1.dmg` reached **Accepted**.
- `xcrun stapler staple` and `stapler validate` succeeded.
- `spctl --assess --type open ...` returns **accepted**,
  `source=Notarized Developer ID`.
- Mounted image: inner `oMLX.app` `codesign --verify --deep --strict` passes,
  `spctl --type execute` accepts it as `Notarized Developer ID`; the embedded
  helper is Developer ID signed with team `YB894QRG4W`.
- Final SHA-256:
  `b44aaa686d36bfe7ab27cc84375170c64b0e4cd94f8dafd45bc4f35bbdeb03f6`.
- The old 59 MB `slim-signed` submission was accepted earlier but is obsolete
  and was not stapled/distributed.
