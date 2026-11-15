# Ready for the cluster test — 2026-11-15

State of the fork after the full-suite pass, the PR review fixes and the local
end-to-end runs. Everything below is measured, not assumed. Read
[`CLUSTER-TEST-RUNBOOK.md`](CLUSTER-TEST-RUNBOOK.md) before touching the Macs.

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
- DMG: `build/oMLX-0.8.0rc1.dmg`, 78 MB, `hdiutil verify` VALID, Developer-ID
  signed, Gatekeeper correctly "Unnotarized Developer ID" (no notarization by
  design for the dry run).
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

- **R8** screenshots (needs the cluster and a browser session).
- **R16/R17** to confirm during the run.
- **R9** further split of the plan slice (measured: A2 changes the shared modules
  by only ~5 %, so it is not urgent).
- **R18** the macOS feature branch needs the fixes PR merged first.

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
4. Version gate: `GET /api/cluster/node_id` → same `version` on all four.
5. Then pair (§1) and run §3.
