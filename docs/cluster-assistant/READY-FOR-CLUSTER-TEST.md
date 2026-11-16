# Ready for the cluster test — 2026-11-15

State of the fork after the full-suite pass, the PR review fixes and the local
end-to-end runs. Everything below is measured, not assumed. Read
[`CLUSTER-TEST-RUNBOOK.md`](CLUSTER-TEST-RUNBOOK.md) before touching the Macs.

## 0. Update 2026-11-16 — assistant UI (U1–U6) and a new DMG

- `rebase/upstream-main` @ **`0c6e7479`** (`feat(cluster): one assistant flow
  with a UI render-mode (U1-U6)`), `upstream/main` `624cff0e` still an ancestor.
- The Cluster tab is now one flow: one `data-assistant-screen` per assistant
  state, `Assistant / Manual setup` mode split, failure-owns-the-card,
  choose/stale/back and in-place start retry. No backend route, state, poll or
  SSE added. Spec `ASSISTANT-UI-SPEC-2026-11-16.md`, log
  `ASSISTANT-UI-IMPLEMENTATION-PLAN-2026-11-16.md`, PR handoff
  `PR-READY-2026-11-16.md`.
- Suites with this change: `apps/omlx-web/tests` **578 passed / 3 deselected**
  (focused three files 289), `tests/cluster_assistant` **658 passed**;
  `make web` is a no-op.
- New artefact: `build/oMLX-0.8.0rc1.dmg`, 79 MB, SHA256
  `9400d290041ce80aa6a3dc84bd919884ec9f38a7d99ffd76c58e37879942d36c`,
  Developer-ID signed, notarized and stapled (`hdiutil verify` VALID,
  `spctl` accepted / Notarized Developer ID), built by
  `scripts/build-release-dmg.sh` (no staged app copies left behind).
- Still open before the PRs: fold into `feat/cluster-assistant-run` (+
  `-run-local` rebase), the live A1–A7 acceptance run, and the screenshot
  review — all three are listed in `PR-READY-2026-11-16.md` §5.

## 1. Clean state

- Working tree clean, `rebase/upstream-main` @ `7f9c120f`, `upstream/main`
  (`624cff0e`) is an ancestor.
- **Cluster state wiped** (`CLUSTER-TEST-RUNBOOK.md` §0.5): no devices, no
  deployments, no measurements/operations/presets, 27 GB staging cache removed.
  Only `~/.omlx/cluster/identity.json` kept (this Mac's `e8a5fd55…`).
- Settings untouched: `~/.omlx/settings.json` (API key, model dirs, bind) and
  `~/.omlx/model_settings.json` (per-model preferences).
- Backup of the wiped state: `~/.omlx/backup-cluster-2026-11-15/`.
- No leftover processes: no test `pytest`/`xcodebuild`, no test `omlx-server`,
  port 8099 free, throwaway worktrees removed. Branches all remain.

## 2. Prepared branches (the offers)

| Branch | Commit | Base | Files | +/− | Full suite |
|---|---|---|---:|---|---|
| `fix/local-only-load` | `bda778e0` | `upstream/main` | 5 | +271/−22 | **20 154 passed, 0 failed** |
| `fix/test-infra-conftest` | `c819ec74` | `upstream/main` | 3 | +74/−48 | **20 147 passed, 0 failed** |
| `fix/macos-theme-locale-xctest` | `1c1674c2` | `upstream/main` | 3 | +29/−3 | **401 Xcode tests, 0 failures** |
| `fix/cluster-discovery-inventory` | `ef0b09fd` | `upstream/main` | 11 | +888/−53 | 19 640 passed; the 2 remaining were the missing native kernels in a worktree (§0.4), 98/98 green with them |
| `feat/macos-helper-dmg` | `3248f848` | `upstream/main` | 24 | +2472/−19 | 410 Xcode tests, 7 failures in the 3 tests the fixes PR repairs (**R18 dependency**) |
| `base/cluster-prereqs` (local only, never offered) | `23cc45b7` | `upstream/main` | 44 | +4047/−258 | 295 focused |
| `feat/cluster-assistant-plan` | `da13ab2e` | base | 64 | +22 893/−82 | **20 555 passed, 0 failed** |
| `feat/cluster-assistant-run` | `a9836644` | plan | 84 | +23 310/−571 | **20 933 passed, 0 failed** |
| `feat/cluster-assistant-run-local` | `f21a6e0f` | run | 13 | +4 268/−16 | **20 976 passed, 0 failed** |

`make web` (`build_css.py` + `normalize_i18n.py`) produces **no diff** on all three
assistant branches, as `docs/CONTRIBUTING.md` requires.

## 3. Upstream PRs

| PR | Head | Mergeable | CI |
|---|---|---|---|
| **#5024** carry MoE expert offload into pipeline ranks | `bbbf18d2` | `clean` | `test (3.11)` ✅ `mac-app-build` ✅ |
| **#5033** make Qwen4-Exp pipeline-parallel | `25da70de` | `clean` | `test (3.11)` ✅ `mac-app-build` ✅ |

Both heads are `upstream/main` + PR content only; their code is applied on our
branch (`054f92e4` #5033, `392aaccb` #5024 runtime).

## 4. Local app

- Combined suite on `rebase/upstream-main`: **20 981 passed / 0 failed**;
  macOS python 27; Xcode `TEST SUCCEEDED` (`-testLanguage en`).
- DMG: `build/oMLX-0.8.0rc1.dmg`, 79 MB, `hdiutil verify` VALID, Developer-ID
  signed **and notarized + stapled** (`spctl` → `accepted, source=Notarized
  Developer ID`), SHA256
  `857fdb690610195419fa30d1c9bb98e81f51c722775fe859620b82c69c1296be`. This
  rebuild contains the R23 tier fix (apply writes the approved memory-guard tier)
  and the R25 launcher-signing fix; the previous dry-run DMG was neither
  notarized nor fixed.
- Live 2-Mac cluster run (2026-11-15, any-to-any): `ready` on a remote
  coordinator, one real completion at 18.76 tok/s, clean stop — evidence, IDs and
  the plan split in `PHASE-2B-RUN.md` §29. That run still mixed builds (the
  coordinator ran the installed 0.8.0rc1, the origin this tree), so it is not the
  §1.5-clean repeat.
- Local end-to-end on `.11` with a wiped cluster state: hardware measure →
  recommend (`dev-1.5b` @ 8192, single-node, 96.1 GiB headroom) → model-bound
  measure → plan (`7179c6975b`) → `run-local` ready → real completion
  ("Hello world.", 35/4 tokens) → **`tune-local` completed** (preset
  `f1b39c284bddb746`) → completion after the unload/load cycle ("wave") → stop.
  `network-preflight` `executed:false`, `diagnostics` returns its bundle,
  `check_peer_versions.py` exit 0.

## 5. What the cluster test must cover

With two or more Macs available, from the wiped state:

1. **find devices → pair → measure → recommend → model-bound measure → plan →
   apply → start → ready → stop** — manually in the Cluster tab *and* through the
   assistant card (the two must agree).
2. **2-Mac plan** (minimal subset) and a **3/4-Mac line** with the topology rule.
3. **MoE expert offload runtime** (#5024): a DeepSeek-V4.1 checkpoint planned with
   offload must actually stream experts on the pipeline ranks (the pricing half is
   already on the branch).
4. **Qwen4-Exp/Qwen3.8 pipeline parallelism** (#5033): plan and start across Macs.
   R1 decision: PP only, no TP.
5. **Tuning on a real cluster**: `/tune` + `/tune-redeploy` + preset replay on the
   next Start.
6. **Peer inventory / layout read** (`layout_read_on_peer`) and the cluster-wide
   operation lock with a remote coordinator.
7. **Screenshots** for the PR bodies (R8): the plan card, the progress rows /
   ready+endpoint / stop, the "Run on this Mac" card, tuning result.
8. **Confirm or drop R16/R17** (`auto-login` sets no session cookie; the card can
   report `no_model`) — see the review report §11.

## 6. Push order (when the cluster test is done)

```sh
# phase 1 — fixes (independent, merge alone)
for b in fix/local-only-load fix/test-infra-conftest fix/macos-theme-locale-xctest; do
  git push origin "$b:$b"
done
# phase 2 — cluster core
git push origin fix/cluster-discovery-inventory:fix/cluster-discovery-inventory
# phase 3 — after #5024, #5033, #5024 and the cluster-core PR are merged:
#   rebase the assistant stack onto main, the base branch disappears
git rebase --onto origin/main base/cluster-prereqs feat/cluster-assistant-plan
git push origin feat/cluster-assistant-plan:feat/cluster-assistant-plan   # draft
# then run -> run-local the same way, drafts, merged bottom-up
# optional feature, only after the fixes PR is in:
git push origin feat/macos-helper-dmg:feat/macos-helper-dmg
```

PR texts ready to paste: [`PR-BODIES-2026-11-15.md`](PR-BODIES-2026-11-15.md).
Discussion draft (R12) if appetite is unclear:
[`DISCUSSION-PROPOSAL-2026-11-15.md`](DISCUSSION-PROPOSAL-2026-11-15.md).

## 7. Still open

- **Onboarding (noted 2026-11-15):** the helper + wired-limit step belongs in the
  Welcome/first-run screen next to the dependency installer, and the web
  *Settings → Performance* page must not look like the terminal command is the
  only way (the button is app-only). Recorded in `ROADMAP-TO-CHAPTER-10.md` §9.

- **Helper install is proven; the wired-limit preflight is built (R5).** The
  one-prompt SMJobBless install ran on all four Macs during the 0.8.0rc1 rollout
  (helper + LaunchDaemon present, wired limit raised to 122 880 / 60 293 /
  29 696 / 29 696 MB) — see `HELPER.md`. The plan-side half is implemented
  2026-11-15: `election.wired_limit_binds()` recognises the Mac whose GPU cap
  (not RAM or pressure) sets its ceiling, and a refusal on that Mac names it and
  its MB value with the action **`raise_wired_limit`** (Performance → “Apply with
  Admin…” in the app; the card only names the step). What remains is a real run
  on a Mac whose limit is still at the macOS default — the rollout machines all
  have theirs raised already.

- **R24 (origin hard-kill during a failed start)** is recorded, not explained:
  no traceback, no crash report, not reproducible. Capture the next occurrence
  with `log show` + `sample` (`UPSTREAM-PR-REVIEW-2026-11-15.md` §17).

- **R8** screenshots (needs the cluster and a browser session).
- **R16/R17** to confirm during the run.
- **R9** further split of the plan slice (measured: A2 changes the shared modules
  by only ~5 %, so it is not urgent).
- **R18** the macOS feature branch needs the fixes PR merged first.
- **Rank-0 pinning** is a product question, not a bug: oMLX's rank 0 is the
  sampler's process and must own the tail, so the tail must fit the Mac that
  serves the endpoint. On GLM-5.3 at 32k that made `.12` rank 0 because `.11`
  alone cannot hold the tail plus head. An explicit "coordinator on this Mac"
  preference (refused, never silently moved, when infeasible) is the way to give
  the user control; it is not implemented yet.

Review of all findings: [`UPSTREAM-PR-REVIEW-2026-11-15.md`](UPSTREAM-PR-REVIEW-2026-11-15.md).
## 8. Fresh rollout order (after the 2026-11-15 wipe)

Every Mac is bare (no app, no `~/.omlx`, no helper; models and this repo kept).
The notarized `build/oMLX-0.8.0rc1.dmg` is the artefact to install.

1. Copy + install the DMG on each Mac (§1.5 steps 2–3 of the runbook; SSH/scp is
   fine for the copy).
2. **Open the app at the Mac, interactively** — Local Network: allow. Then
   `/health` must answer on port 8099.
3. **Bless the helper once per Mac**: Performance → “Apply with Admin…”, accept the
   admin dialog, and confirm
   `/Library/PrivilegedHelperTools/app.omlx.helper` +
   `/Library/LaunchDaemons/app.omlx.helper.plist` exist and
   `launchctl print system/app.omlx.helper` lists the job. This also lifts the
   107.5 GB Metal cap on the 128 GB Mac.
4. Version gate: `GET /api/cluster/node_id` → same `version` on all four. The
   version string stays `0.8.0rc1` for this rebuild, so also compare the **app
   binary hash** on every Mac (`shasum -a 256
   /Applications/oMLX.app/Contents/MacOS/oMLX`) — a missed reinstall is invisible
   in the version gate alone. The DMG's own hash is
   `857fdb690610195419fa30d1c9bb98e81f51c722775fe859620b82c69c1296be`.
5. Then pair (§1) and run §3.

## 9. Current rollout artefact (2026-11-16, the acceptance build)

This supersedes §8's hashes and is the build the A1–A7 acceptance run must use.

| Item | Value |
|---|---|
| DMG | `build/oMLX-0.8.0rc1.dmg`, 79 MB, SHA256 `9400d290041ce80aa6a3dc84bd919884ec9f38a7d99ffd76c58e37879942d36c` |
| App binary inside the DMG | `760a1695ad5c17ca7473ffff574b9d251ee3b8aed872194ac4b2d89bf0afdfb3` |
| Version string | `0.8.0rc1` (unchanged — the hash is the real gate) |
| Signed / notarized | Developer-ID signed, notarized (`d3a1b6c9-…`, Accepted), stapled; `spctl` → `accepted, source=Notarized Developer ID` |
| Built from | `0c6e7479` (U1–U6 one-flow Cluster tab); `691aaa94` and later are docs only |
| Contents vs §8 | D1 uninstall, D6 helper capability, distributed inference on by default, **and the U1–U6 UI** |

**State check measured 2026-11-16:** this Mac (`.11`) still runs app binary
`09f51c2d…` (the pre-U1–U6 build), and `~/.omlx/cluster/` is **not wiped**
(`devices.json`, `deployments.json`, `assistant-measurements.json`,
`runtime/prompt-cache-ssd` present). So the acceptance run has **not** started:
reinstall the DMG on all four Macs, verify `760a1695…` everywhere, then wipe per
`CLUSTER-TEST-RUNBOOK.md` §0.5 and run A1–A7.

The checklist for the whole remaining work is
[`PLAN-REMAINING-2026-11-16.md`](PLAN-REMAINING-2026-11-16.md).
