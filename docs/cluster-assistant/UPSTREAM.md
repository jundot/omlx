# Upstream and patch status

Status date: 2026-11-14

## 2026-11-14: rebased onto `upstream/main`

The fork is now based on `upstream/main` `624cff0e` (branch
`rebase/upstream-main`); `upstream/main` is an ancestor of HEAD. The pre-rebase
tree is untouched on `feat/cluster-assistant` (`28245931`) and
`backup/pre-rebase-2026-11-14`. Our net work from the old pin
`1e27576a..feat/cluster-assistant` was squash-reapplied (not a 154/158-commit
replay), so it can still be curated into upstream packages.

What the rebase changed in the fork:

- Upstream **removed routed-expert (MoE) offload from the planner**. It is
  restored on top of upstream's refactored pipeline (`plan_hybrid` /
  `plan_unequal_pipeline` / `_finish_pipeline_plan` / `_held_layer_bytes` +
  `PipelineAssignment` fields); a `()` default reproduces upstream exactly. The
  coordinator-side threading is the one documented follow-up.
- Upstream **added the Qwen4 PLE switch path**; `_saved_engram_ssd_offload`
  honours Engram or PLE (table-offload parity restored).
- Upstream **refactored the cluster wizard** device model; the merged
  `cluster_v2.js` keeps the assistant UI and ports the upstream helper stages
  (`splitDevices`, `deviceHelperOnly`, `pairingStepOne`, CUDA guide, helper-aware
  `restoreAccelerator`/`planNodes`/`inventoryHosts`/`versionMismatches`/
  `acceleratorRows`).
- `Localizable.xcstrings` is **upstream-preferred** (upstream base + feat-only
  keys) so the compiled catalogue stays complete.
- **Dropped** (upstream removed them): `qwen4_exp_mlx_lm` adapter, `qwen4`
  cluster/pipeline tests, the experimental doc.

Local evidence and the full conflict list: `REBASE-2026-11-14.md`.

## Pre-rebase status (2026-11-13)

Historical base: `1e27576af5e10c8f4459e0fdcb05c5cf59245d44`
(tag `cluster-assistant-base-2026-11-07`)

The project pin is unchanged. Only the PR branch was rebased on 2026-11-12:
`feat/qwen4-exp-cluster` now sits on `upstream/main` `c74f1b18` (2026-11-13)
instead of `2165e757` (which was 95 commits behind). See “#5033 after the
2026-11-12 rebase” below. The fork branches (`cluster-assistant-base`,
`feat/cluster-assistant`) deliberately stay on `1e27576a`.

The base is the **cluster-assistant-base** branch: `upstream/main` at the
2026-11-07 rebase point plus the tested fork patches:

- #5024 (`feat/cluster-moe-offload`, the old `cb32b021` pin),
- #5033 (`feat/qwen4-exp-cluster`, rebased head `793e2197`: Qwen4-Exp pipeline,
  PLE streaming, the plan-capability split),
- `fix/cluster-rank0-ssh` (`acdfda73`),
- `fix/cluster-distributed-launch-reliability` (the stale-rank reap, port range
  and launch error reporting),
- `feat/moe-offload-fit-hint` (`b6f0c6fd`).

Historical base: `cb32b021f54dd34a2afd4f48b88a30d06641972b` (#5024 head).
That commit is no longer the pin; the sections below that describe it are the
2026-11-03/06 history, not the current base.

Base of the original pin: `e47101f36c10e8b3b99f74d632f403cf01d1e313`

This file records what is actually present in this fork. It is deliberately
separate from the project brief because PR state changes.

## How this project was bootstrapped

It was cloned from the local oMLX development checkout:

```text
~/Documents/PycharmProjects/olmx
```

It was **not** cloned directly from GitHub. The clone was then checked out at
the pinned commit above and given these remotes:

```text
origin   https://github.com/digitalfactory-ako/omlx.git
upstream https://github.com/jundot/omlx.git
```

The local source was used because it contains the pinned engine state and the
local patch branches. Do not assume that `origin/main` equals this base.

## Included in `cb32b`

| Work | Commit | State |
|---|---|---|
| Engram SSD offload on pipeline ranks (#4985 / #4984) | `43393f76` | **merged upstream**; ancestor of `cb32b` |
| Launch idempotency, trimmed (#4974 / #4973) | `d8babb1d` | **merged upstream**; ancestor of `cb32b` |
| MoE expert offload in pipeline plans | `d594472f` | local/fork patch series; ancestor of `cb32b` |
| Skip non-owned layers when wrapping expert offload | `a0e917c9` | local/fork patch series; ancestor of `cb32b` |
| DeepSeek V4.1 MTP with expert offload | `d5649967` | local/fork patch series; ancestor of `cb32b` |
| Document MoE expert offload on pipeline ranks | `cb32b021` | local/fork patch series; ancestor of `cb32b` |

The MoE/MTP series is on:

```text
local/cluster-moe-offload
pr/feat-cluster-moe-offload
fork/feat/cluster-moe-offload
```

It is **not** in `origin/main` at the time of writing. It is included in this
project because the pinned engine needs it.

## Not included in `cb32b`

| Work | Commit / branch | State |
|---|---|---|
| Full launch-reliability series | `666cf36f` on `fix/cluster-distributed-launch-reliability` | local branch; not merged; only the trimmed `d8babb1d` is in base |
| Rank-0 SSH launch | `574f0460` on `fix/cluster-rank0-ssh` | local draft; **not filed**; not in base |
| Qwen4 experimental cluster branch | `1c008995` on `pr/feat-qwen4-exp-cluster` | separate experimental branch; not in base |
| LocalInferenz console/app/UI changes | `~/Documents/PycharmProjects/localinferenz` | not part of oMLX; harvest only specific planner/measurement ideas |
| LocalInferenz rank-0/UI patches | LocalInferenz `main` | not part of this oMLX base; the new coordinator election must be implemented here |

## What this means

- Merged upstream PRs (#4985, #4974) are already in the base.
- Our local MoE/MTP patch stack is in the base even though it is not upstream.
- The full launch series and rank-0 SSH draft are **not** in the base.
- The Qwen4 experiment branch is **not** in the base.
- The LocalInferenz app is **not** the new project and its patches are not
  automatically present here.

## Local patches added by this project

Every local change to an upstream file is listed here with its reason and the
test that keeps it honest.

| Patch | File | Why | Test |
|---|---|---|---|
| Coordinator-election library | `omlx/cluster/assistant/election.py` | New module; no upstream file touched | `tests/cluster_assistant/test_election.py` |
| Assistant route registration and getters | `omlx/server.py` in `_register_cluster_routes()` | Mount the assistant router with the same `Depends(require_admin)` and `Depends(require_distributed_inference_enabled)` as `cluster.routes`, so the assistant surface is never reachable when the cluster surface is off or the caller is not an admin. One `include_router` call plus one `set_assistant_getters(...)` call (settings manager → measurement store base path, server settings → this Mac's own endpoint port). No new app, no new control plane, no second patch site. | `tests/cluster_assistant/test_plan_api.py`, `tests/cluster_assistant/test_measure_api.py` |
| Public seams for prepare, stage and stage-status | `omlx/cluster/routes.py` (`prepare_cluster_deployment`, `start_cluster_staging`, `read_cluster_staging_job`) | `apply`/`start` must stage and launch through the *existing* pipeline: `POST /deployments` recomputes a plan and requires a signed `approved_placement`, and `POST /stage` owns the one staging runner. Instead of copying either, these three functions were extracted verbatim (the two routes now call them) so the Cluster Assistant reuses the same planner, the same placement signature and the same staging job. No behaviour change on the routes, no new endpoint, no second staging implementation. | `tests/cluster_assistant/test_apply_api.py` (`test_apply_stages_and_never_starts_inference`, `test_apply_through_the_ssh_seam_stages_then_stops`), `tests/cluster_assistant/test_start_api.py` (`test_the_local_coordinator_uses_the_existing_local_pipeline`), `tests/test_cluster_staging.py`, `tests/test_cluster_routes.py` |
| Cluster Assistant card (template) | `apps/omlx-web/omlx_web/templates/dashboard/_cluster_v2.html` | Screen-3 minimal slice: one card inside the existing plan section. It renders the assistant API's answer (recommend → plan → apply → start), pins every branch with `data-assistant-state`, hides `endpoint.url` unless `assistantReady()`, and shows exactly one failure action from `error.action`. No second tab/wizard and no launch/stage/deploy markup. | `apps/omlx-web/tests/ui/test_cluster_v2_wizard.py` (`test_assistant_states_are_guarded_and_separate_from_the_wizard`, `test_assistant_failure_renders_exactly_one_action_from_error_action`, `test_assistant_endpoint_is_unreachable_until_ready`, `test_assistant_primary_path_has_no_ip_ssh_rank_layer_or_model_path_field`) |
| Cluster Assistant component | `apps/omlx-web/omlx_web/static/js/cluster_v2.js` | `ASSISTANT_API` (the assistant routes and nothing else), `assistantState()`, one generation-guarded single-flight poll loop per job, the recorded entry decision (Phase C3: hardware-only measure → recommend → measure the chosen model model-bound → plan), frozen plan-time inputs shared by plan/apply, apply→applied→start, reload-resume via `GET /session` (Phase C1), and one-action dispatch. It never stages, deploys, launches or talks SSH. | `apps/omlx-web/tests/test_cluster_ui_integration_ui.py` (the `test_assistant_*` tests), `apps/omlx-web/tests/ui/test_cluster_v2_wizard.py` (`test_assistant_endpoints_are_only_in_the_api_map_and_allowlisted`, `test_assistant_component_never_stages_deploys_or_launches`) |
| Cluster Assistant strings | `apps/omlx-web/omlx_web/i18n/en.json` (and the existing locale files via `normalize_i18n.py`) | English-only product copy for the assistant card (`cluster.assistant.*`). No translation and no new locale: the repo's own normalizer mirrors the key set into each existing locale with the English value as fallback (`make web`). | `apps/omlx-web/tests/ui/test_cluster_v2_wizard.py` (`test_assistant_strings_exist_in_the_english_catalog`), `apps/omlx-web/tests/test_mcp_expose_tools_ui.py` (`test_locale_key_sets_identical`) |
| Discovery announcement carries the display identity | `omlx/cluster/discovery.py` (`encode_wassup`, `decode_wassup`, `_handle_hello`, `_handle_wassup`, `_announced_ipv4_addrs`) | Live defect from Phase 2b: without mDNS the HELLO/WASSUP fallback carried only the machine ID and port, so an unpaired Mac appeared as `11618ac4 · Apple silicon · Memory unknown · Unreachable` — Screen 2 requires friendly names, and the probe could not even reach the only (IPv6 link-local) address. The reply now also carries `friendly_name`, the same capability snapshot mDNS publishes as TXT, and the Mac's IPv4 addresses, which the receiver records and probes exactly like mDNS A records. All keys are optional, older decoders ignore them, and an announcement without them never erases what is already known. No new transport, no new endpoint, no change to the trust decision. | `tests/test_cluster_v2_discovery.py` (`test_wassup_carries_the_display_identity`, `test_wassup_names_the_peer_without_mdns`, `test_wassup_records_the_announced_ipv4_and_probes_it`, `test_wassup_without_a_name_does_not_erase_a_known_one`, `test_hello_from_same_cluster_sets_multicast_ok_and_replies_wassup`) |
| Cluster Assistant Welcome entry | `apps/omlx-mac/Sources/Welcome/WelcomeWindow.swift`, `Resources/Localizable.xcstrings`, `Tests/oMLXTests/LocalizationSmokeTests.swift` | The Welcome complete step gets a second button to the same existing Cluster tab, using the shared `/admin/auto-login` redirect builder. No new window or control plane. | `LocalizationSmokeTests`; the app builds with Xcode 27.2. |
| macOS wired-limit first slice | `apps/omlx-mac/Sources/AppView/ViewModels/PerformanceScreenVM.swift`, `apps/omlx-mac/Sources/AppView/Screens/PerformanceScreen.swift`, `Resources/Localizable.xcstrings`, `Tests/oMLXTests/PerformanceScreenVMTests.swift` | One standard macOS admin dialog raises the kernel `iogpu.wired_limit_mb`; the command is a fixed `sysctl` with a validated MiB integer, never a shell string from UI text. This is a one-shot privileged action; a persistent signed helper and automatic network setup remain open (`HELPER.md`). | `PerformanceScreenVMTests.testPrivilegedWiredLimitValidation`; `xcodebuild build/test` with `CODE_SIGNING_ALLOWED=NO`; no dialog is opened in tests. |
| Developer ID team | `apps/omlx-mac/oMLX.xcodeproj/project.pbxproj` | Set `DEVELOPMENT_TEAM = YB894QRG4W` for the app target per the P2 decision. No certificate or credential is committed; 0 identities were installed when this was recorded. | Xcode build with code signing disabled; signing/notarization remains an external prerequisite (`DISTRIBUTION.md`). |
| Read-only network preflight | `omlx/cluster/assistant/network_preflight.py`, `routes.py`, `models.py`, `apps/omlx-web/.../_cluster_v2.html`, `cluster_v2.js`, tests | New read-only assistant endpoint using the existing `transport.probe_host_interfaces`; returns interface evidence, exact bridge command templates and STP/ring warnings, with `executed: false`. The card displays/copies the commands; it never runs them. No privileged action and no second network control plane. | `tests/cluster_assistant/test_network_preflight.py`, `test_plan_api.py`/`test_recommend_api.py` route lists, UI allowlist/static test. |
| First-run runtime installer + SMJobBless helper | `Sources/Server/PythonRuntime.swift`, `Sources/Welcome/WelcomeWindow.swift`, `Sources/AppView/ViewModels/PerformanceScreenVM.swift`, `Helper/`, `Resources/Info.plist`, `Scripts/build.sh`, `Scripts/package_dmg.sh` | Welcome card installs the pinned external runtime with bundled `uv`/lock and explicit consent; the signed SMJobBless helper replaces the one-shot AppleScript for `iogpu.wired_limit_mb` with one admin prompt, falling back to the one-shot action. Helper/launchd plists and `SMPrivilegedExecutables`/`SMAuthorizedClients` pin team `YB894QRG4W`. | App build; helper/app `codesign` verification; `PerformanceScreenVMTests`; the SMJobBless install itself still needs a click on the signed app. |
| Slim external Python runtime | `apps/omlx-mac/Scripts/provision_runtime.sh`, `build.sh` (`--slim`), `Sources/Server/PythonRuntime.swift`, `SLIM-RUNTIME.md` | Python/MLX stay out of the DMG: the app ships the Swift shell + `omlx`/`omlx_web` source, resolves a pinned Application Support venv first, and `--slim` writes wrappers for it. The runtime is provisioned from Homebrew or a tagged uv CPython with locked deps. Measured: 58 MB DMG instead of 884 MB; 1.0 GB runtime outside the bundle. | `omlx-cluster-python -c "import omlx"` from the slim app; `xcodebuild` build; the app-focused tests. |
| Cluster Assistant macOS menu entry | `apps/omlx-mac/Sources/Menubar/MenubarController.swift`, `apps/omlx-mac/Resources/Localizable.xcstrings`, `apps/omlx-mac/Tests/oMLXTests/MenubarControllerPortTests.swift` | P2 entry decision: one menu item opens the existing web dashboard's Cluster tab (`/admin/dashboard?tab=cluster`) through the existing `/admin/auto-login` URL builder. No second window, app, wizard or control plane; no status/stop in the tray. | `MenubarControllerPortTests.testWebAdminURLDeepLinksTheClusterTab`, `LocalizationSmokeTests` (build/test blocked on this Mac: no Xcode, only Command Line Tools). |
| Cluster Assistant styles | `apps/omlx-web/omlx_web/static/css/dashboard.css`, `apps/omlx-web/omlx_web/static/css/tailwind.css` | Dark-mode readability for the assistant summary and the collapsed “Details” affordance, plus the rebuilt Tailwind output after the template/component change. Built with `python build_css.py`. | `apps/omlx-web/tests/ui/test_cluster_v2_wizard.py` (`test_dark_tensor_controls_use_explicit_high_contrast_palette` stays green; template tests exercise the new markup) |
| Assistant performance-probe setting | `omlx/settings.py` | New `ServerSettings.assistant_performance_probe` (default **false**) plus its `from_dict` read. The post-ready observed measurement is opt-in per Mac; it never rewrites `predicted` and never changes the approved placement. | `tests/cluster_assistant/test_start_api.py` (observed block on/off), `test_plan_api.py`/`test_recommend_api.py` (route lists) |
| Synthetic distributed performance probe route | `omlx/cluster/routes.py` (`POST /performance-probe`) | The assistant's after-apply compute probe runs through the coordinator's own admin route (existing synthetic probe machinery), not a second implementation. Profiles feed the **next** measurement/plan; `auto_tune` stays off and the probe never gates apply/start. | `tests/cluster_assistant/test_apply_api.py`, `test_plan_api.py` |
| Spec-correct compute probe + plausibility guard | `omlx/cluster/performance_worker.py`, `omlx/cluster/assistant/election.py` | The old decode probe was a `rows=1`, 2048-wide (8 MiB) GEMV — kernel-launch-bound, so its "GB/s" inverted the hardware ordering. Widening the GEMV was still kernel-limited (MLX's small-M efficiency placed M5 Max below M1 Max). The final probe measures **decode as a streaming bandwidth reduction** (`_bandwidth_rate`, decode is memory-bound) and **prefill as a compute-bound GEMM** (`_weight_rate`, rows=1024 × width=8192, prefill is compute-bound), matching published specs: decode orders M2 Ultra > M5 Max > M1 Max (800/614/400 GB/s), prefill orders M5 Max > M2 Ultra > M1 Max (Neural Accelerators). The election adds `ComputeProfile.plausible()` and treats an implausible rate as unknown compute in `budget_for` (headroom preference); the frozen "rank by measured critical stage" stays for trustworthy probes. | `tests/cluster_assistant/test_probe_spec_expectations.py` (spec-grounded, fake-driven), `test_election.py` (plausibility), `tests/test_cluster_performance.py`; live evidence PHASE-2B-RUN.md §17 |
| Staging: own LAN address is local; disk walk for absent paths | `omlx/cluster/staging.py` (`is_local_host`, `_local_interface_addresses`, `free_disk_bytes`) | A coordinator reaches its own rank 0 over its LAN address (Tahoe ring workaround); without treating the machine's interface addresses as local, staging SSHes to itself and a stale self `known_hosts` entry fails a 3/4-Mac apply. `free_disk_bytes` now quotes the path as one shell word and walks to the nearest existing parent, so a not-yet-staged checkpoint reports real free space instead of 0. The `lru_cache` failure-caching defect (REVIEW-2026-11-13.md R2) is fixed: a failed discovery read is no longer cached, so own-LAN detection retries. `run_remote_python` now enables the SSH keepalive (`ServerAliveInterval=15`/`CountMax=4`) so a coordinator that dies mid-call is detected in ~60 s instead of hanging to the full subprocess timeout (B13). | `tests/test_cluster_staging.py` (`test_is_local_host_accepts_the_macs_own_lan_address`, the retry test) |
| Cluster inventory: MTP fact + peer layout read | `omlx/cluster/model_inventory.py` (`mtp_compatible_for_model`, `remote_model_layout`, `mtp_compatible` in all three inventory builders) | The assistant plans Lightning MTP from the catalogue, so the cluster inventory must carry the same `mtp_compatible` fact the admin list computes (fail-closed false). A peer-only checkpoint's layout is read on the Mac that holds it (`layout_read_on_peer`). The mixed-version defect (REVIEW-2026-11-13.md R1) is fixed: the snippet imports `mtp_compatible_for_model` guarded, so a stock-oMLX peer keeps its whole inventory with `mtp_compatible: false` instead of losing it to an ImportError. | `tests/test_cluster_model_inventory.py`, `tests/test_cluster_remote_planning.py` |
| Planner layout: declared context, table kind, PLE parity, pre-load patches | `omlx/cluster/planner.py` (`ModelLayout.max_context_tokens`/`table_offload_kind`, `_qwen4_ple_layer_ids`, `_declared_context_tokens`, `inspect_safetensors_layout`) | The assistant never plans past the checkpoint's declared context and must name the SSD table switch it writes (`engram`/`ple`). Qwen4 PLE tensors are priced into the same per-layer SSD bytes as DeepSeek Engram. `inspect_safetensors_layout` applies oMLX's pre-load architecture patches before capability detection so a patched checkpoint is not refused as non-PP. New layout fields change plan hashes for table-capable checkpoints. | `tests/cluster_assistant/test_policy.py`, `test_policy_api.py`, `tests/test_cluster_plan_approval.py` |
| Devices payload: pending-pairing address shape | `omlx/cluster/discovery_routes.py` (`_address_rows`) | Pending pairing requests carry `list[str]` addresses while discovery rows carry `[{"ip","if_type"}]`; merging them without normalization silently empties the address list for awaiting-approval devices. | `tests/test_cluster_pairing.py`, `tests/test_cluster_pairing_session.py` |
| Chat handoff: `?model=` preselect | `apps/omlx-web/omlx_web/templates/chat.html` (`requestedModelFromUrl`, `applyRequestedModelFromUrl`) | Screen 5 preselects in the existing `/chat`: the parameter is read once on load, applied only when the model is available, never written into history, and no credential or `endpoint` parameter is read. No second chat page. | `apps/omlx-web/tests/ui/test_cluster_v2_wizard.py`, `apps/omlx-web/tests/test_cluster_ui_integration_ui.py` (Open Chat URL contract) |
| macOS app runtime hardening | `apps/omlx-mac/Sources/Menubar/MenubarVisibilityWatcher.swift`, `Sources/Server/ServerProcess.swift` | The menu-bar visibility probe is skipped under XCTest (no real menu bar in the test host; it triggered the Full Disk Access alert mid-run). `ServerProcess` creates the logs directory before opening the log file, so a fresh install does not fail server start silently. | App-focused Xcode tests; production behaviour unchanged |

Chapter 5 added **no** `omlx/server.py` change: the assistant router was already
mounted, and the operation store resolves its base path through the existing
`set_assistant_getters` settings-manager getter. Remote apply/start reach the
coordinator through the existing admin routes. The launch and deployment
modules kept their semantics; `omlx/cluster/staging.py` did change (own-LAN
local detection and the quoted disk-space walk, both listed in the patch table
above), so the earlier claim that no staging semantics changed is superseded.

The minimal Screen-3 UI (Chapter 6) is an upstream **web** patch: the template,
component, English catalog and CSS above are upstream files, patched in place
inside the existing Cluster tab. It added no route, no `server.py` change and no
backend change; `tests/cluster_assistant` is unchanged and green. The web seam
test (`apps/omlx-web/tests/test_cluster_seams_ui.py`) now resolves registered
routes from both the cluster router and the assistant router, so the
assistant URLs are verified against their real registration instead of being
reported as typos. The node-gated JS tests skip where `node` is absent; the
static contract tests in `tests/ui/test_cluster_v2_wizard.py` run everywhere.

## Upstream drift (checked 2026-11-06)

```text
our branch   feat/cluster-assistant @ c75bab47 (on the pin cb32b021, 2026-11-02)
upstream/main 2165e757 (2026-11-05, tag v0.7.4.dev1)
```

`git rev-list --left-right --count cb32b021...upstream/main` → **60 behind, 4
ahead**. The four are our fork-only MoE/MTP series (`d594472f`, `a0e917c9`,
`d5649967`, `cb32b021`).

Of the 60 upstream commits, **3 touch `omlx/cluster`** and **17 touch
`apps/omlx-web`** — the drift is mostly web UI, not the cluster engine.

A silent merge (`git merge-tree --write-tree upstream/main HEAD`) reports **two
textual conflicts** and auto-merges everything else:

| Conflict | Why | Handling |
|---|---|---|
| `omlx/cluster/discovery.py` | upstream added its own link classification (`_LINK_RANK` with the *inverse* ordering, `_local_ports()` from `networksetup`) where we added `_LINK_RANK`/`_best_link`/`_classify_link` | take upstream's link logic, keep only our announcement additions (`friendly_name`, caps, IPv4), which upstream does not have |
| `apps/omlx-web/omlx_web/static/css/tailwind.css` | generated | `python build_css.py` |

Checks that came back clean (the assistant itself ports):

- `ClusterDeploymentRequest`, `ClusterHostRequest`, `ClusterPlanNodeRequest` have
  identical fields;
- `ModelLayout` fields are identical, so our persisted
  `MeasurementRecord.layout` dictionaries stay valid;
- `run_remote_python` and every module the assistant imports still exist with the
  same signature;
- our election passes only stable `plan_hybrid` kwargs.

The one structural divergence is **MoE expert offload**: the pin has it in
`omlx/cluster` (6 files, including `plan_hybrid(moe_offload_layer_bytes=…)` and
`_moe_offload_for_plan` in `cluster/routes.py`); upstream/main has **zero**
`moe_offload` references under `omlx/cluster` (the engine still has MoE support
elsewhere). Rebasing therefore means either dropping our MoE planning patch
series or porting it, and our assistant never used those knobs.

Also worth cherry-picking independently of a rebase: `f6897ff4` “Prefill
Accelerator, a fast paired Mac speeds up prefill (#5065)” and `2165e757`
(turboquant/MTP verify windows) — both are cluster- or MTP-relevant to this
project.

Recommendation: decide the MoE question first (it is the only thing that pins
us), and if the answer is “drop it”, rebase as its own chapter: branch from
`upstream/main`, replay our commits, resolve the two conflicts as above,
re-apply the three extracted route functions (`prepare_cluster_deployment`,
`start_cluster_staging`, `read_cluster_staging_job`) onto upstream's
`cluster/routes.py`, rebuild CSS, run every suite, then one live 2-Mac
apply→start→ready smoke.

## Open pull requests (checked 2026-11-06)

| PR | Branch | State | Commits | Behind base | Merge | CI | Review |
|---|---|---|---|---|---|---|---|
| #5024 | `feat/cluster-moe-offload` | open, not draft | 4 | 60 | **CLEAN / mergeable** | `test (3.11)` SUCCESS | 0 reviews, 0 comments |
| #5033 | `feat/qwen4-exp-cluster` | open, not draft | 3 | 60 | **CLEAN / mergeable** | `test (3.11)` SUCCESS | 0 reviews, 0 comments |

Both are based on `e47101f3` and are 60 commits behind `upstream/main`, **but
neither needs a rebase to merge**:

- A silent merge of either branch onto today's `upstream/main` reports **zero
  conflicts**, and GitHub's `mergeStateStatus` is `CLEAN`.
- The PR CI check runs on the merge commit with the current base, so a green
  `test (3.11)` already covers the 60 upstream commits.
- Upstream has not touched a single cluster-engine file since the PRs' base:
  `omlx/cluster/{routes,planner,deployment,inference_worker,autoconfigure,
  progressive_loading}.py` and the cluster tests are all **unchanged upstream**
  (the “−127 lines” seen earlier when diffing from `cb32b021` is only our MoE
  additions being absent upstream).

One file is worth a rebase before merge: **#5033** edits
`omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py`
(+118/−32) while upstream rewrote the same file (+104/−88) in the last 60
commits. The merge is textually clean (different hunks) and unit tests pass, but
unit tests cannot prove that the PR's pipeline split still matches upstream's
refactored attention — that needs a rebase, a read of the merged code and a real
Qwen4 PP smoke.

### Rebase + live smoke of #5033 (done 2026-11-07)

`git rebase --onto upstream/main e47101f3` on the PR branch: **clean, 3/3
commits**, no conflicts. The merged `qwen4_exp/language.py` keeps upstream's
refactored attention (its `q_x_proj` split is unchanged) and only adds the
pipeline method, so the rebase did not revert upstream's work.

- Tests in the rebased tree: `tests/test_cluster_*.py` → **1600 passed, 2
  skipped**.
- The PLE fix now has its own regression test
  (`test_cluster_plan_follows_the_saved_qwen4_ple_ssd_offload_setting`), pushed
  as the fifth commit; the branch head is `793e2197`.
- **CI note**: the first rebased run failed one test,
  `tests/test_prefill_helper.py::test_manager_retries_a_lost_helper_but_not_a_stopped_one`
  (30.28 s, “state stayed ready”). That is upstream's Prefill Accelerator test,
  which this PR does not touch; it passes 3/3 on clean `upstream/main` here in
  2.5–2.9 s, so it is a timing-sensitive flake under the xdist CI load. The run
  cannot be re-run from the fork (needs write access on `jundot/omlx`), so a new
  commit was pushed to trigger a fresh run instead.
- **Pushed 2026-11-07**: `feat/qwen4-exp-cluster` is now the rebased head
  `9bc47278` (4 commits), PR #5033 reports **MERGEABLE / CLEAN** against
  `upstream/main`. The push needed the `workflow` token scope (the rebased head
  carries upstream's `.github/workflows/ci.yml`) and `gh` as the credential
  helper, because the keychain still held a repo-only token.
- **One real gap found and fixed**: the planner only honoured the DeepSeek
  Engram SSD setting, so a Qwen4-Exp plan charged the **37 GiB PLE table** as
  resident and no split fitted. `_saved_ssd_offload` now accepts either offload
  flag (commit `9bc47278` on `rebase/qwen4-exp`). With
  `qwen4_ple_ssd_offload: true` on both Macs the plan fits.
- **Live end-to-end on the line**: `autoconfigure` → 2-stage plan (`.14`
  layers 41–48, `.11` layers 0–41) → activation `2ae8ec9a95d3` →
  `Distributed engine ready: ranks=2 backend=ring` in ~14 s → `/v1/models`
  lists it → a real completion answered **“Earth”** (36 tokens, 3.6 s,
  10.1 tok/s). The worker log confirms PLE streaming:
  `Qwen4-Exp PLE forced to SSD … 135.0GB resident … mmap needs 94.7GB`.

Environment note for any future PR smoke: the preflight compares the local
*module* version with the peer's *installed metadata*, so an uninstalled checkout
needs either `python_executable` set to `~/.omlx/bin/omlx-cluster-python` in the
deployment hosts, or a `dist-info` next to the checkout (this smoke used both).
That is a property of running a checkout, not of the PR.

**Consequence for this project:** our pin `cb32b021` *is* the head of #5024. So
the “do we keep the MoE patch series?” question is not ours to resolve while
#5024 is open — if it merges, the four-commit divergence disappears and a rebase
onto `upstream/main` becomes routine. Do not rebase the feature branch before
that decision; #5024 is otherwise clean and only waiting for the maintainer.

## #5033 after the 2026-11-12 rebase

`jundot` reviewed #5033 on 2026-11-12 (`CHANGES_REQUESTED`, no inline
comments): the PLE fix in `9bc47278` works and now builds a 128 GiB + 32 GiB
split, but three things block the merge. The rebase was done first so the
homework lands on today's upstream instead of a five-day-old base.

```text
PR head before    793e2197, merge-base 2165e757 (2026-11-05), 95 behind
upstream/main     c74f1b18 (2026-11-13)
rebase            git rebase --onto upstream/main 2165e757  ->  5/5 clean
PR head after     b859fa56, 0 behind / 5 ahead
```

- **Zero conflicts.** The only auto-merged file is
  `omlx/cluster/inference_worker.py`: upstream added `_cache_only_prompt_prefill`,
  the CED/A8 prefill options and numerics-scoped prompt-cache directories
  (+120/−5), while the PR changes `_guard_effective_stage`. The prefill wrapper
  is installed only for `tensor_parallel_size > 1`, so the PP path is not
  affected, and `sampling_rank_only` still defaults to `True`.
- **Tests in the rebased tree**: `tests/test_cluster_*.py tests/test_qwen4_*.py`
  → **2417 passed, 190 skipped, 7 deselected**. Upstream added
  `tests/test_qwen4_qsa_prefill_memory.py`, which is directly relevant to the
  first review item.
- **Ruff**: with `--force-exclude` (the vendored `mlx_vlm` copies are excluded
  by `pyproject.toml`) four findings remain in `omlx/cluster/routes.py`,
  `omlx/utils/model_loading.py` and `tests/test_cluster_routes.py`. All four
  reproduce on clean `upstream/main`; the PR adds none.
- **Push** needs the `gh` credential helper for the `workflow` scope
  (`.github/workflows/ci.yml` is in the rebased tree):
  `git -c credential.helper= -c 'credential.https://github.com.helper=!gh auth git-credential' push --force-with-lease`.
  Backup branch: `backup/qwen4-exp-793e2197`.

### The maintainer's three blockers

1. **Prefill memory on the first-layer Mac.** The planner picks the split that
   ran out of memory: layers 0–8 (or 0–9) on the 32 GiB Mac with 13–15 GiB
   shown free, and the 4096-token throughput profile gets the same split. Cut 6
   was set by hand, so the plan is missing some memory the first-layer Mac uses
   during prefill. The question to answer with evidence: *what grew on the
   32 GiB Mac when cut 9 ran out of memory?*
2. **`lm_head` on rank 1.** `_stage_key_needed` skips
   `language_model.lm_head.*` for `rank != 0`, but `_FullModel` still
   constructs the head with random weights, so the skip saves no memory. With
   `sampling_rank_only=false` every rank samples its own token, so rank 1
   samples the random head and the output goes wrong with no error.
   `glm5_next` and DeepSeek V4.1 load `lm_head` on every rank. Note that
   `CLUSTER_UNLOADED_WEIGHT_PREFIXES` does **not** list `lm_head`, so the
   planner already charges it on every rank — loading it is a correctness fix,
   not a planning change.
3. **Drop the #5020 check.** Qwen4 counts as `honored`, so
   `_guard_effective_stage` returns before the `pipeline_support(model_path)`
   branch, and the planner already refuses a model with no pipeline support.
   Remove that branch, its tests and `pipeline_support()` (no other caller);
   keep `model_type_of`/`_pipeline_module` only if they still simplify
   `pipeline_assignment_is_honored`.

After the fixes the maintainer wants one more live run of the 128 + 32 GiB
setup with the split the planner picks, once with `sampling_rank_only` on and
once off.

## The three blockers: fixes and the live run (2026-11-12)

Branch `feat/qwen4-exp-cluster`, on top of the rebase commit `b859fa56`, pushed
as `01f8af0e`.

| Item | Commit | Change |
|---|---|---|
| 3 — drop the #5020 probe | `5c047daf` | `_guard_effective_stage` no longer takes `model_path`, the `pipeline_support(model_path) == "none"` refusal is gone with its test, and `pipeline_support()` is removed. `model_type_of`/`_pipeline_module` stay: `pipeline_assignment_is_honored` still uses them. |
| 2 — `lm_head` on every rank | `5c047daf` | `_stage_key_needed` returns `True` for `language_model.lm_head.*` on every rank. The old skip saved no memory (`load_model_weights(strict=False)` left a randomly initialised head) and made `sampling_rank_only=false` answer from that head; `CLUSTER_UNLOADED_WEIGHT_PREFIXES` never listed the head, so the planner already charged it everywhere. |
| 1 — prefill transient in the plan | `01f8af0e`, wording `f7953597` | `ModelLayout.prefill_transient_bytes_per_context_token`, priced once at 4096 chunk / 8192 context from `MemoryMonitor.estimate_prefill_peak_bytes` on the **generic SDPA path**, then scaled by the execution profile’s chunk in `plan_unequal_pipeline`/`plan_proportional_pipeline`. `_partition_layers` subtracts it from every node budget, `_finish_pipeline_plan` charges it in the fit, and `PipelineAssignment.prefill_transient_bytes` is part of `planned_weight_bytes`/`to_dict`, so operator headroom includes it. The generic price is not the number a profile model is admitted against (Qwen4 QSA, GLM-5 sparse MLA and DeepSeek V4.1 are priced lower), so the reservation is a conservative upper bound on admission — `f7953597` says that instead of claiming an identity. MLA checkpoints stay at 0 because the generic formula cannot price a latent cache; that is a known gap, not a claim that their prefill is free. Layouts with unreadable dims keep their exact bytes and hash. |

Tests: `tests/test_cluster_*.py` 1610 passed, 2 skipped;
`tests/test_qwen4_*.py tests/test_cluster_prefill_guard.py` 824 passed, 188
skipped; ruff clean on the touched files. CI on `01f8af0e`: `test (3.11)` and
`mac-app-build` both green (run 43289146310); `f7953597` is a comment/docstring
commit on the same tree and re-runs them.

The split at 32k/throughput changes from layers 0–7 to 0–4 on the 32 GiB Mac
(7.65 GB reserved instead of 0), and `interactive`/`balanced` reserve 1.78/3.56
GB for the same context instead of nothing.

### The live run the review asked for

`.11` (128 GiB) rank 0 layers 4–48, measured 80.2 GB; `.14` (32 GiB) rank 1
layers 0–4, measured 8.55 GB, prefill reserve 7.65 GB; plan hash `77d0796a…`.
Launched with the argv `build_mlx_launch_argv` produces (worker pair directly,
released through `dev-qwen4-prefill-serve.json` on both Macs).

| Measurement | `.14`, layers 0–4 | `.14`, layers 0–7 (pre-fix split) |
|---|---|---|
| footprint after load | 8.75 GB | 14.0 GB |
| peak during a 26.1k-token prefill | **25 GB** | **25 GB** |
| wall clock | 27.4 s | 32.4 s |

`iogpu.wired_limit_mb` is `0` (macOS default) on the 32 GiB Mac, so 25 GB is the
GPU ceiling both runs sit at, not their demand: cut 9 starts at ~17 GB, leaving
~8 GB where this prefill wants ≥16 GB — which is the reported OOM. The growth is
the per-chunk working set of the head stage, not the KV cache and not the
weights. It is not the expert dispatch either (1024 × top-10 × 640 × 2 B is 13 MB
of intermediate per layer), and — measured, see below — not the attention
transient.

Chunk sweep at the same 26.1k prompt and the same split (reservation is what the
plan charges at a 26k context):

| Chunk | Peak | Δ above load | Plan reserves |
|---|---|---|---|
| 1024 | 16 GB | +7.25 GB | 1.42 GiB |
| 2048 | 22 GB | +13.25 GB | 2.84 GiB |
| 4096 | 25 GB (ceiling) | ≥ +16.3 GB | 5.68 GiB |
| 8192 | 25 GB (ceiling) | ≥ +16.3 GB | 11.36 GiB |

Context sweep at a 1024 chunk: 4.6k → 14 GB (+5.25), 13.1k → 16 GB (+7.25),
26.1k → 16 GB (+7.25). The chunk sets the size and the context saturates above
~13k.

The growth is not proportional to attention layers. Same request, both stages;
the head stage has one full-attention layer, the tail stage eleven:

| Stage | Layers | Full-attention layers | Δ over load |
|---|---|---|---|
| `.14`, head | 0–4 | 1 | **+7.25 GB** (8.75 → 16 GB) |
| `.11`, tail | 4–48 | 11 | **+4.0 GB** (75 → 79 GB peak) |

The tail's 4 GB spread over 44 layers is 0.09 GB per layer — the attention/GDN
transient the estimator reasons about. The head grows 1.8× as much with 1/11 of
the attention layers, and it is the stage holding layer 1, the SSD-backed PLE
n-gram layer (128 shards, 38.4 GB). It is not the table being paged in: one
lookup over the prompt reads ~0.8 GB of file data (26k × 24 n-gram rows ×
~1.3 KB), an order of magnitude below the spike, and `vmmap --summary` shows
177 MB of mapped file against 9 GB of footprint. One fp32 score matrix for the
stage's single QSA layer is 453 MB, so attention cannot explain 7 GB.

`sampling_rank_only` was exercised on and off with the same split. Off, the
answer is coherent, which is the `lm_head` fix working on rank 1. Full numbers
and method: `PHASE-2B-RUN.md` §15.

Residual, reported rather than hidden. At a 26k context and a 4096-token chunk
three numbers disagree: the rank guard’s Qwen4 gathered profile says 1.34 GiB,
the plan’s generic price says 5.68 GiB, and `.14` measures ≥ 16.3 GB. The
guard is the looser one — it prices a gathered core of `indexer_budget` (2048)
keys for one layer. That is the hole cut 9 passed admission through: 0.34 GiB
priced against 7.25 GB measured at 26k/1024, a 21× miss.

Both numbers are attention terms and the growth is not attention, so the plan’s
5.68 GiB landing near a third of the peak is coincidence of magnitude, not the
right mechanism. What the planner needs is a per-chunk charge to the
**table-holding stage** (the layer in `ple_layer_ids`, Engram has the same
shape), which `ModelLayout` can already locate; the coefficient per chunk token
is not in the config and comes from this measurement, which is why this PR stops
at the correct scaling. It does not touch admission either; the generic price is
~3× the guard and ~3× under the peak, and it is what moved the split from 0–7 to
0–4. The 0–4 stage completes a 26k prefill at chunk 4096 (25 GB of a 26 GB
budget); a 9-layer stage would still OOM at that chunk and needs a smaller chunk
or context. Numbers and method: `PHASE-2B-RUN.md` §15–§15.2.

### The three-Mac proof (2026-11-13)

The split had only been run on two Macs. `.12` (M2 Ultra, 64 GiB) rank 0 layers
21–47, `.14` (M1 Max, 32 GiB) rank 1 layers 10–20, `.13` (M1 Max, 32 GiB) rank 2
layers 0–9 with the PLE table on SSD; plan hash `1a7e9e22…`, 1024-token chunks.

| Rank | Planned | Measured weights | Footprint after load |
|---|---|---|---|
| 0 `.12` | 46.8 GiB | 46.35 GiB | 47.00 GiB |
| 1 `.14` | 19.8 GiB | 19.62 GiB | 20.00 GiB |
| 2 `.13` | 18.1 GiB | 17.98 GiB | 19.00 GiB |

`rank_ready` ×3 about 20 s after launch, ring up (four connections `.12`↔`.13`,
one `.12`↔`.14`), and one real completion (5,868 prompt tokens at a 1024-token
chunk, 8 new tokens) in 31.03 s with peaks of 51.00 / 25.00 / 24.00 GiB. The
plan's weight term holds on three Macs; the 32 GiB stages finish at their Metal
ceiling, so the context/chunk envelope for N=3 is a measurement of its own and
not claimed here. Rank 0 has to be started from that Mac's own login session —
an ssh-spawned rank 0 never completes the ring handshake. Full numbers and
method: `PHASE-2B-RUN.md` §16.

## Update procedure

When fetching or rebasing:

1. Fetch `upstream`.
2. Check whether the local MoE/MTP commits landed upstream.
3. If yes, drop the local duplicate commits and rebase onto upstream.
4. If no, preserve the local patch series and re-run the cluster tests.
5. Re-audit the cluster routes/UI if the upstream pin moves.
6. Never assume a merged PR commit is present without checking ancestry.

Useful checks:

```sh
git merge-base --is-ancestor 43393f76 cb32b021
git merge-base --is-ancestor d8babb1d cb32b021
git merge-base --is-ancestor cb32b021 upstream/main
git log --oneline --decorate -8
```

`git merge-base --is-ancestor` exits 0 for YES and 1 for NO.