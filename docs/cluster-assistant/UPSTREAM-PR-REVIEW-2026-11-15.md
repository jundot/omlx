# Upstream PR review — 2026-11-15

Critical review of everything prepared for `jundot/omlx`. This is a **fix list**,
not a summary: every finding has evidence, an impact, a concrete fix and a
verification step. Work it top-down; the pre-push gate at the end must pass
before any branch is pushed.

- Reviewer: agent session, 2026-11-15
- Scope: the 7 prepared PR branches, the optional macOS feature branch, the local
  integration base, and the two open upstream PRs (#5024, #5033).
- Method: `git diff upstream/main..<branch>` per branch, file-content inspection
  (`git show <branch>:<path>`), GitHub API for PR/CI state, and the upstream
  contribution rules (`.github/pull_request_template.md`, `docs/CONTRIBUTING.md`).
- **Nothing here is fixed yet**, except R15's first prevention step: the scratch
  patterns were added to `.git/info/exclude` (local, shared by all worktrees).

---

## 1. Inventory under review

| Branch | Commit | Base | Files | +/− | Kind |
|---|---|---|---:|---|---|
| `fix/local-only-load` | `bda778e0` | `upstream/main` | 5 | +271/−22 | fix |
| `fix/test-infra-conftest` | `c819ec74` | `upstream/main` | 3 | +74/−48 | fix (tests) |
| `fix/macos-theme-locale-xctest` | `fbde666e` | `upstream/main` | 4 | +32/−3 | fix |
| `fix/cluster-discovery-inventory` | `f62d50f0` | `upstream/main` | 11 | +888/−53 | fix/core |
| `feat/macos-helper-dmg` | `05e5dfe3` | `upstream/main` | 24 | +2348/−19 | feature (optional) |
| `base/cluster-prereqs` | `34b2fc6c` | `upstream/main` | 47 | +4471/−258 | local only, never offered |
| `feat/cluster-assistant-plan` | `5b3cafae` | a0 | 62 | +22 884/−79 | feature 1/3 |
| `feat/cluster-assistant-run` | `7e53f41b` | a1 | 83 | +23 315/−570 | feature 2/3 |
| `feat/cluster-assistant-run-local` | `417ef95d` | a2 | 13 | +4268/−16 | feature 3/3 |
| upstream #5024 | `bbbf18d2` | `upstream/main` | 16 | +619/−16 | open, clean, CI green |
| upstream #5033 | `25da70de` | `upstream/main` | 15 | +1689/−77 | open, clean, CI green |

Worktrees: `/tmp/pr-fixes`, `/tmp/pr-f4`, `/tmp/pr-f5`, `/tmp/pr-assistant`,
`/tmp/omlx-pr5024`. Branches live in the shared object database, so they survive
worktree removal.

---

## 2. Findings register

| ID | Severity | Scope | Finding | Status |
|---|---|---|---|---|
| R1 | **P0 blocker** | a0, A1, A2, A3 | Private scratch files committed (German live-run notes + drivers) | **fixed** |
| R2 | **P0 blocker** | feature | Signing identity hard-coded (team ID in project, entitlement, test) | **fixed** |
| R3 | **P0 blocker** | F4 | The headline "fresh-install logs dir" fix is redundant upstream | **fixed** |
| R4 | P1 | A2 (and A3 by inheritance) | Test calls `_saved_engram_ssd_offload`, a name that only exists via our fork alias | **fixed** |
| R5 | P1 | A1, A2, A3 | Code/comments reference fork-only docs (PHASE-2B-RUN, PROJECT-BRIEF, CHAPTER-5, SCREEN-*) | **fixed** |
| R6 | P1 | F5 | Private lab IPs (`172.16.1.x`) used as test fixtures | **fixed** |
| R7 | P1 | F4 | ThemeTests fix weakens the assertion for all macOS versions | **fixed** |
| R8 | P2 | all UI branches | PR template not satisfied (Validation, before/after screenshots, `make web`/`make app`) | bodies written; screenshots open |
| R9 | P2 | A1, A2 | Still very large; A1 ships partial modules that A2 replaces (review churn) | **accepted, measured** |
| R10 | P2 | F1 | Two concerns in one PR (`local_only` + VLM fallback guard) | **resolved: one concern** |
| R11 | P2 | feature | Docs live in `docs/cluster-assistant/` and describe our fork workflow | **fixed** |
| R12 | P2 | series | No plan B if maintainers decline #5024/#5033/F5 | **discussion draft ready** |
| R13 | P2 | all | Branch names `pr/…` are unusual for upstream | **fixed** (renamed) |
| R14 | P2 | A1 | Single 22.9k-line commit; per-commit review is impossible | **fixed** (3 commits) |
| R15 | P3 | process | Root cause of R1: `git add -A` on a foreign base whose `.gitignore` lacks our local rules | prevention applied (`info/exclude`); rest open |

Verified **not** a problem (do not "fix" these):

- **i18n English values in every locale are correct.** `apps/omlx-web/normalize_i18n.py`
  fills missing keys with the English fallback, and `base.html` defines
  `window.t = key => window._t[key] !== undefined ? window._t[key] : key` (falls back
  to the *key*, not to English). So keys must exist in every catalog; English values
  are exactly what upstream tooling produces. Only requirement: run `make web`.
- **F5's `is_local_host` tests are mocked** (`monkeypatch.setattr(... local_addr_dicts ...)`),
  so they are not machine-dependent — R6 is about neutrality, not flakiness.
- **No fork traces in F1/F2/F4** (checked across changed files only).
- **Assistant branches contain only assistant files** — no duplicated or reverted
  prerequisite code (verified against an allowlist).
- **#5024/#5033** are `mergeable: true` / `clean` with both CI checks green and no
  fork delta in their heads.

---

## 3. P0 — must fix before any push

### R1 · Private scratch files are committed

**Evidence**

```
git diff --name-only upstream/main..feat/cluster-assistant-plan | grep '^\.'
.live_prefill_driver.py
.live_prefill_driver3.py
.local-3rank-2026-11-13.md
```

Same three files in `feat/cluster-assistant-run` and `feat/cluster-assistant-run-local`. Introduced by
`34b2fc6c` (the a0 base) and inherited by all three assistant branches.

`.local-3rank-2026-11-13.md` is **German** and states verbatim: *"Diese Datei ist
absichtlich untracked und gehört nicht in die PR-Doku … bleibt bei uns."* It holds
live 3-rank cluster measurements. The two drivers are live-run scripts.

They are **not** tracked on `rebase/upstream-main` and **are** ignored there
(`.gitignore:137-138`), so the leak is confined to the PR branches.

**Impact**: private lab data and German text in a public PR; also violates the
English-only product rule. Pushing this would be a disclosure, not just a blemish.

**Fix**

```sh
cd /tmp/pr-assistant
git checkout feat/cluster-assistant-plan
git rm -q .live_prefill_driver.py .live_prefill_driver3.py .local-3rank-2026-11-13.md
git commit -q --amend --no-edit
git checkout feat/cluster-assistant-run && git rebase feat/cluster-assistant-plan
git checkout feat/cluster-assistant-run-local     && git rebase feat/cluster-assistant-run
# a0 carries them too; rebuild or strip it the same way (it is never pushed)
```

**Verify**

```sh
for b in base/cluster-prereqs feat/cluster-assistant-plan feat/cluster-assistant-run feat/cluster-assistant-run-local; do
  git diff --name-only upstream/main..$b | grep '^\.' && echo "LEAK in $b"
done
uv run pytest tests/cluster_assistant -q          # per assistant branch
```

**Prevention**: see R15.

---

### R2 · Feature PR is hard-wired to our signing identity

**Evidence**

| File | Line | Content |
|---|---|---|
| `apps/omlx-mac/oMLX.xcodeproj/project.pbxproj` | 1084, 1110 | `DEVELOPMENT_TEAM = YB894QRG4W;` |
| `apps/omlx-mac/Resources/Info.plist` | `SMPrivilegedExecutables` | `… certificate leaf[subject.OU] = "YB894QRG4W"` |
| `apps/omlx-mac/Tests/python/test_privileged_helper_packaging.py` | 29 | `TEAM = "YB894QRG4W"` (asserted) |
| `apps/omlx-mac/Scripts/package_dmg.sh` | 14 | usage example names our identity |

**Impact**: no other contributor can build, sign or test this. The packaging test
fails for everyone else. A maintainer cannot merge it as-is.

**Fix (proposal)**

1. Move the team to configuration: `DEVELOPMENT_TEAM = $(OMLX_TEAM_ID)` with an
   `xcconfig` (or build setting) that defaults to empty; document
   `OMLX_TEAM_ID=… xcodebuild …`.
2. Generate the `SMPrivilegedExecutables` requirement at build time from the same
   value (Info.plist preprocessing or a build phase), instead of a literal OU.
   Keep the *structure* of the requirement (Developer ID CA + leaf marker) — that
   part is the valuable fix and must stay.
3. Make the packaging test read the team from the environment/built artifact and
   `pytest.skip` when it is unset; assert the *relationship* (app and helper name
   the same team, Developer ID pinned) rather than a literal team.
4. Neutralise the script usage examples (`SIGN_IDENTITY="Developer ID Application: …"`).

**Verify**: `OMLX_TEAM_ID= xcodebuild build -scheme oMLX CODE_SIGNING_ALLOWED=NO`
succeeds; `uv run pytest apps/omlx-mac/Tests/python -q` passes with no team set;
with a team set, the requirement assertions still hold.

---

### R3 · F4's headline fix is redundant

**Evidence** — upstream `apps/omlx-mac/Sources/Server/ServerProcess.swift` already has:

```swift
private func doStart() throws {
    try ensureDir(basePath)
    try ensureDir(logURL.deletingLastPathComponent())   // line 331, upstream
```

and `ensureDir` is `createDirectory(at:withIntermediateDirectories:true)` (line 619).
Our diff adds a second, identical `createDirectory` call.

**Impact**: the claimed bug ("a fresh install has no logs directory, server start
fails silently") does not exist upstream. Offering it as a fix damages the
credibility of the whole series.

**Fix**

```sh
cd /tmp/pr-f4
git checkout fix/macos-theme-locale-xctest
git checkout upstream/main -- apps/omlx-mac/Sources/Server/ServerProcess.swift
git commit -q --amend -m "fix(macos): macOS 27 theme colour, locale percent separator, XCTest menubar probe"
```

F4 becomes **3 files, ~26 added lines**. Update `UPSTREAM-PACKAGES.md` (§2 P4-fixes,
§4 F4, §8 table) and the commit message accordingly.

**Verify**: `xcodebuild test -scheme oMLX CODE_SIGNING_ALLOWED=NO -testLanguage en
-only-testing:oMLXTests/ThemeTests -only-testing:oMLXTests/ModelSettingsScreenVMTests
-only-testing:oMLXTests/MenubarControllerPortTests`.

---

## 4. P1 — will break or annoy after the prerequisites merge

### R4 · Name drift: `_saved_engram_ssd_offload`

**Evidence**: `feat/cluster-assistant-run:tests/cluster_assistant/test_policy.py` lines
**383, 390, 397** call `cluster_routes._saved_engram_ssd_offload(...)`.

Upstream #5033 renames the function to `_saved_ssd_offload`. The old name survives
only because **our** a0 keeps an alias
(`_saved_engram_ssd_offload = _saved_ssd_offload`). That alias is fork-only and is
not part of any PR.

**Impact**: once #5033 merges and A2 is rebased onto `main`, the test raises
`AttributeError`. The branch is green today only because of a private shim.

**Fix**: use `_saved_ssd_offload` in the assistant test (3 call sites). Keep the
alias in `rebase/upstream-main` only if something else in the fork needs it;
otherwise drop it there too.

**Verify**: `uv run pytest tests/cluster_assistant/test_policy.py -q` on A2 and A3.

### R5 · References to fork-only documentation

**Evidence** (occurrences in changed files):

| Branch | PHASE-2B-RUN | PROJECT-BRIEF | CHAPTER-5 | SCREEN-* |
|---|---:|---:|---:|---:|
| A1 | 12 | 10 | – | 5 |
| A2 | 17 | 13 | 8 | 10 |
| A3 | 22 | 13 | 8 | 10 |

Top offenders in A2: `omlx/cluster/assistant/models.py` (7),
`apps/omlx-web/omlx_web/static/js/cluster_v2.js` (7),
`apps/omlx-web/tests/ui/test_cluster_v2_wizard.py` (6),
`apps/omlx-web/tests/test_cluster_ui_integration_ui.py` (5),
`recommendation.py` (4), `operations.py` (4), `start.py` (3).

**Impact**: a reviewer cannot resolve a single one of these references; comments
look like they point at missing design docs.

**Fix**: rewrite each comment/docstring to state the rule itself (one or two
sentences) or to reference the PR/issue number. Mechanical sweep:

```sh
git grep -nE "PHASE-2B-RUN|PROJECT-BRIEF|CHAPTER-5|SCREEN-[0-9]|ROADMAP-TO-CHAPTER-10" \
  feat/cluster-assistant-plan -- omlx apps tests
```

Keep the design docs in the fork; do not ship them.

**Verify**: the grep above returns nothing on each branch; tests still green.

### R6 · Private lab IPs in F5 tests

**Evidence**: `tests/test_cluster_v2_discovery.py` lines 254, 286, 386, 393 and
`tests/test_cluster_staging.py` lines 880, 884, 885, 901 use `172.16.1.12/13/14`.

**Impact**: not flaky (the interface list is mocked), but these are our lab
addresses; upstream fixtures should be neutral.

**Fix**: replace with documentation ranges (`192.0.2.11/12/13`, `198.51.100.x`).
Keep the semantics (one own address, one peer, one unrelated).

**Verify**: `uv run pytest tests/test_cluster_v2_discovery.py tests/test_cluster_staging.py -q`.

### R7 · ThemeTests fix weakens the assertion everywhere

**Evidence**: `XCTAssertGreaterThan(abs(actual.red - underPage.red), 0.25)` became
`XCTAssertGreaterThan(underPage.red, 0.9)`.

**Impact**: we lose the contrast regression check on macOS ≤ 26, where the two
colours really do differ.

**Fix (better)**: keep the strict assertion on older systems and relax only on 27:

```swift
if #available(macOS 27, *) {
    XCTAssertGreaterThan(underPage.red, 0.9)   // Tahoe converges the two colours
} else {
    XCTAssertGreaterThan(abs(actual.red - underPage.red), 0.25)
}
```

**Verify**: run ThemeTests on this Mac (27) and, if available, on a 26 runner;
otherwise state the limitation in the PR Validation section.

---

## 5. P2 — process and review quality

### R8 · PR template / CONTRIBUTING not satisfied
Upstream `.github/pull_request_template.md` requires **Summary**, **Validation**
(what ran, which Mac chip/RAM and model; before/after numbers for performance) and
a **Checklist** that includes *before and after screenshots for UI/UX changes*.
`docs/CONTRIBUTING.md` additionally requires `make web` after template/JS/i18n
changes and `make app` when the macOS app changes.

**Fix**: write one PR body per branch from the template; attach screenshots for
A1/A2/A3 and the feature branch; record validation hardware (e.g. M5 Max 128 GB,
M2 Ultra 64 GB) and the exact test commands/counts; run `make web` on each UI
branch and confirm the diff has no reordering churn.

### R9 · Size and partial modules
A1 = 62 files/+22 884 (11 909 code, 10 172 tests); A2 = 83/+23 315. A1 also ships
*subsets* of `provisioning.py` and `start.py` that A2 later replaces, so reviewers
see the same file rewritten twice.

**Fix options**: (a) split A1 into A1a (measure + recommend, backend + tests) and
A1b (plan + UI); (b) move the UI of each slice into its own PR; (c) extract the
shared seams A1 needs into a small module so A2 only adds, never rewrites.

### R10 · F1 mixes two concerns
`local_only` plus the `force_lm`/VLM fallback guard. Either split into two commits
with a clear rationale, or justify the coupling in the PR body (both stem from "a
cluster deployment leaking into a local load").

### R11 · Feature docs location
`docs/cluster-assistant/{HELPER,DISTRIBUTION,SIGNING,SLIM-RUNTIME}.md` describe our
fork workflow. Move to `docs/macos/…` and neutralise fork-specific paths before
offering the feature.

### R12 · No plan B for the series
A1–A3 are blocked on F5 + #5024 + #5033 merging. If maintainers decline any of
them, the assistant cannot be offered as prepared.

**Fix**: open a **discussion** first to gauge appetite; prepare a self-contained
A1 variant (minimal planner hook included) as a fallback; or offer the assistant
against our own fork as a visible stack and only upstream it once the base lands.

### R13 · Branch naming
`pr/…` is unusual. On push use `fix/local-only-load`, `fix/test-infra-conftest`,
`fix/macos-theme-locale-xctest`, `fix/cluster-discovery-inventory`,
`feat/cluster-assistant-plan`, `feat/cluster-assistant-run`,
`feat/cluster-assistant-run-local`, `feat/macos-helper-dmg`.

### R14 · Commit granularity
A1 is one 22.9k-line commit. Split it into backend / UI / tests commits so GitHub's
per-commit view is usable (the branch content stays identical).

---

## 6. P3 — process root cause (prevents R1 recurring)

### R15 · `git add -A` on a foreign base

**Root cause of R1**: the a0 base was built **in the main repo** after
`git checkout -B tmp/pre upstream/main`. That checkout replaced `.gitignore` with
upstream's version, which does **not** contain our local rules
(`.live_prefill_driver*.py`, `.local-*.md` — they live in our fork's `.gitignore`,
and F2 deliberately dropped that change). The scratch files were physically
present, suddenly unignored, and `git add -A` swept them into `34b2fc6c`.

**Fix / prevention**

1. Put local scratch patterns in `.git/info/exclude` (shared by all worktrees,
   never committed, independent of the checked-out `.gitignore`):
   ```sh
   printf '.live_prefill_driver*.py\n.local-*.md\n' >> .git/info/exclude
   ```
2. Never `git add -A` in the main repo while it is on a foreign base; stage
   explicit paths.
3. Build integration bases in a **worktree**, not in the main repo.
4. Add a pre-push gate (below) that rejects dotfiles and known scratch patterns.

---

## 7. Pre-push gate (must pass for every branch)

```sh
B=pr/<branch>
# 1 no private/scratch files
git diff --name-only upstream/main..$B | grep -E '^\.|\.local-|live_prefill' && exit 1
# 2 no fork-only doc references in shipped code
git grep -nE "PHASE-2B-RUN|PROJECT-BRIEF|CHAPTER-5|SCREEN-[0-9]|ROADMAP-TO-CHAPTER-10" $B -- omlx apps tests && exit 1
# 3 no personal identity / lab addresses
git grep -nE "YB894QRG4W|Andreas Kolb|andreaskolb|172\.16\.1\." $B -- . && exit 1
# 4 no prerequisite duplication (assistant branches)
git diff --name-only <base>..$B | grep -vE '^(omlx/cluster/assistant/|tests/cluster_assistant/|apps/omlx-web/|CLUSTER-ASSISTANT\.md$|omlx/server\.py$|omlx/settings\.py$|tests/test_bootstrap\.py$|tests/test_settings\.py$)' && exit 1
# 5 tests green on the branch itself
uv run pytest tests/cluster_assistant -q && (cd apps/omlx-web && PATH=<node>:$PATH uv run pytest tests -q)
# 6 UI branches: make web produced no churn
# 7 PR body follows .github/pull_request_template.md, screenshots attached
```

Run 1–4 for **every** branch, 5–7 for the ones that apply. Record the result per
branch in `UPSTREAM-PACKAGES.md` §8.

---

## 8. Suggested execution order

| Step | Findings | Effort | Notes |
|---|---|---|---|
| 1 | R1, R15 | ~20 min | strip the leak, add `info/exclude`, rebuild a0 |
| 2 | R3 | ~15 min | drop the redundant hunk, re-run mac tests, update docs |
| 3 | R4, R6 | ~20 min | rename 3 call sites, neutralise IPs, re-run tests |
| 4 | R7 | ~15 min | `#available` branch in ThemeTests |
| 5 | R5 | ~1–2 h | mechanical comment sweep across A1–A3 (largest P1) |
| 6 | R2 | ~2–3 h | parameterise the team ID; only needed if the feature is offered |
| 7 | R8, R13, R14 | ~1–2 h | PR bodies, screenshots, branch names, commit split |
| 8 | R9, R10, R11, R12 | planning | decide the further split and the discussion-first approach |

Steps 1–5 are prerequisites for pushing the **fix wave** (F1, F2, F4, F5).
Step 6 only blocks the optional feature PR. Steps 7–8 block the assistant series,
which is in any case waiting for #5024/#5033/F5 to merge.

---

## 9. What is already good (keep it)

- Both upstream PRs are `clean` with green CI and contain no fork delta.
- Fix branches are small, independent, and individually tested.
- Assistant branches contain only assistant files (allowlist-verified); no
  duplicated or reverted prerequisite code.
- The i18n approach matches upstream tooling (see §2).
- F5's host-detection tests are properly mocked.
- The MoE offload work is complete on both halves (pricing + runtime) and the
  `()` default reproduces upstream exactly.
- Every branch has a recorded test result; the local app is green end-to-end
  (20 752 passed) and the DMG builds, signs and verifies.
---

## 10. Fix log (2026-11-15)

| Finding | What changed | New commit |
|---|---|---|
| R1 | The three scratch files removed from a0 and all three assistant branches (rebuilt + rebased); `.git/info/exclude` now ignores them for every worktree | a0 `23cc45b7`, A1 `9a335e40`, A2 `ede28463`, A3 `00f53315` |
| R3 | ServerProcess hunk dropped (upstream already calls `ensureDir`); F4 is now 3 files | `fix/macos-theme-locale-xctest` `1c1674c2` |
| R7 | ThemeTests keeps the strict `>0.25` contrast below macOS 27 and only relaxes on 27+ (`#available`) | same commit |
| R4 | `_saved_ssd_offload` used in A2 and A3 (3 call sites each) | A2 `ede28463`, A3 `00f53315` |
| R5 | ~85 comment/docstring lines swept: file names replaced with self-contained noun phrases ("the live-run notes", "the design notes"), section markers dropped; `grep` for the old tokens returns nothing | A1 `9a335e40`, A2 `ede28463`, A3 `00f53315` |
| R6 | Lab addresses replaced with RFC 5737 ranges (`192.0.2.10/11/12`), semantics unchanged | `fix/cluster-discovery-inventory` `ef0b09fd` |
| R2 | Team id is `$(OMLX_TEAM_ID)`; the helper plists are rendered from the same marker at build time; the packaging test skips without a team and asserts the requirement structure | `feat/macos-helper-dmg` `3248f848` |
| R11 | Docs moved to `docs/macos/`, team and host references neutralised | same commit |
| R8 | PR bodies written from upstream's template, with real sizes and validation | `docs/cluster-assistant/PR-BODIES-2026-11-15.md` |

Verification after the fixes:

- `feat/cluster-assistant-plan`: 312 assistant + 473 web, green. `feat/cluster-assistant-run`: 615 + 547, green. `feat/cluster-assistant-run-local`: 646 + 559, green.
- `fix/cluster-discovery-inventory`: 200 focused cluster tests green.
- `fix/macos-theme-locale-xctest`: 52 macOS tests green, `TEST SUCCEEDED`.
- `feat/macos-helper-dmg`: build succeeds with and without `OMLX_TEAM_ID`; 36 macOS tests green; `apps/omlx-mac/Tests/python` 29 passed / 2 skipped without a team, 31 passed with one; `git grep YB894QRG4W` returns nothing.
- Assistant chain parents verified: A1→a0, A2→A1, A3→A2; every branch still contains only its own slice.

Test run after the fixes: combined `tests/` + `apps/omlx-web/tests` **20 981 passed, 0 failed**; macOS python 27 passed; `xcodebuild test … -testLanguage en` **TEST SUCCEEDED**; local end-to-end run on this Mac (measure → recommend → model-bound measure → plan → run-local → real completion → stop) succeeded — see `REBASE-2026-11-14.md`.

Resolved in this second pass:

- **R13** — branches renamed to upstream-conventional names: `fix/local-only-load`,
  `fix/test-infra-conftest`, `fix/macos-theme-locale-xctest`,
  `fix/cluster-discovery-inventory`, `feat/macos-helper-dmg`,
  `feat/cluster-assistant-plan`, `feat/cluster-assistant-run`,
  `feat/cluster-assistant-run-local`; the local base is `base/cluster-prereqs`.
- **R14** — A1 is now three commits (backend 15 files / 9 862 lines, card 24 files,
  tests 23 files); the tip tree is byte-identical to the single-commit version
  (verified by tree hash), so the test evidence still holds.
- **R10** — measured and resolved: the whole F1 diff is the `local_only` path. The
  VLM/`force_lm` code the subagent flagged is the *test vehicle* for the measured
  Qwen3.8 failure, not a second concern. No split.
- **R9** — measured: A2 changes the five shared modules with **+6 301 / −309**
  lines, i.e. about 5 % of that is replaced code; the rest is A2's own new code.
  A further split would double the stack depth for little review gain, so the
  3-commit A1 plus this measurement is the answer unless maintainers ask.
- **R12** — `DISCUSSION-PROPOSAL-2026-11-15.md` is ready to post: appetite
  question, scope, and what happens if only the five fixes are taken.

Only R8 (UI screenshots) is still blocked: it needs a real browser session and
must be done by a person.

## 11. Observations from the screenshot attempt (verify with the cluster)

Not defects of the prepared PRs, but found while trying to capture UI screenshots
on 2026-11-15; both need the cluster run to confirm:

- **R16 — `/admin/auto-login` sets no session cookie in this build.** Reproduced
  twice: the endpoint answers 302 to `/admin` with no `Set-Cookie`, while the same
  key authenticates every API route. Hypothesis (from the code's own comment
  about issue #2282): the CLI path imports `omlx.server` once for
  `init_server()`, and uvicorn then imports `omlx.server:app` as a second module
  instance, so the admin router's module-level `set_admin_getters(...)` lambdas
  point at the *other* `_server_state`, whose `global_settings` has no API key.
  That alias guard exists only for `python -m omlx.server`. Not isolated further;
  needs a run with a debug print in `auto_login`.
- **R17 — the assistant card can report `no_model`** ("No model is available on
  this Mac"). Measured on a fresh state: `GET /admin/api/models` returns **7**
  models immediately, while `GET /admin/api/cluster/catalogue` returns **0**
  entries. The card does not use that GET, it posts to
  `/admin/api/cluster/catalogue` for per-node fit (`cluster_v2.js:546`), so this
  is **not yet confirmed** as the cause — needs the UI session during the cluster
  run to see what candidates it actually sends.

### Local test run on a wiped cluster state (2026-11-15)

After the reset (§0.5 of the runbook), the whole single-Mac path was exercised
again through the API:

- devices: self only, 0 paired; `GET /admin/api/models` 7 entries; catalogue GET 0.
- hardware `measure` → ready; `recommend` → `dev-1.5b` @ 8192 with `single_node`
  feasible; model-bound `measure` → ready; `plan` → 1 node, signature `7179c6975b`.
- `run-local` → ready; real completion "Hello world." (3 completion tokens).
- **`tune-local`** → `completed`, preset `f1b39c284bddb746` written
  (`mtp: false`, the only config a model without an MTP head offers).
- completion **after** tuning → "wave", so the unload/load cycle left the model
  serving.
- `run-local/{id}/stop` → `stopped: true`; `network-preflight` → `executed: false`;
  `diagnostics` returned `app/events/generated_at/health/measurement/operation`.

## 12. R18 — the macOS feature branch needs the fixes PR

Measured with the full suite (`xcodebuild test -scheme oMLX CODE_SIGNING_ALLOWED=NO
-testLanguage en`, 2026-11-15): 410 tests, **7 assertion failures in three
tests**, all of them the defects the separate fixes PR repairs:

- `ThemeTests.testLightWindowBackgroundUsesStandardWindowColor`
- `ModelSettingsScreenVMTests.testK2AnePresetsKeepOneThirdAndExistingValues`
- `ModelSettingsScreenVMTests.testK2SharedPresetsKeepLoadedStringSelection`

The feature branch intentionally excludes those fixes, so this is a **dependency**,
not a regression: merge `fix/macos-theme-locale-xctest` first (or expect the three
on macOS 27). Recorded in the feature PR body.

## 13. R19/R20 — found by the full-suite run (fixed)

**R19 — the nested assistant `conftest.py` shadowed `tests/conftest.py`.**
The full suite on the plan slice failed 10 tests, 9 of them
`tests/test_scheduler.py::TestDetectNeedsThinkPrefix` with
`ImportError: cannot import name 'MockTokenizer' from 'conftest'`: the scheduler
helper imports `conftest` lazily, and once `tests/cluster_assistant/conftest.py`
is collected it occupies that module name. Fixed in the plan slice by importing
`MockTokenizer` from `_tokenizer_support` explicitly (3 call sites). This is the
rewrite the test-infra PR deliberately did not carry; it belongs to the slice that
introduces the nested conftest.

The tenth failure was `tests/test_cluster_seams.py` flagging three functions the
plan slice ships but cannot call yet (`measurement.with_calibration`,
`policy.policy_model_settings`, `routes.configure_assistant_seams`); added to
`allowed_uncalled` with the reason "ships with the run slice".

**R20 — the stylesheet was stale in the run slices.** `make web` changed
`static/css/tailwind.css` by one line in the run (and initially the run-local)
slice: the added markup used utilities that the copied stylesheet did not contain.
Rebuilt and committed in the run slice; the run-local slice inherits it.

Both are exactly the kind of cross-effect the focused runs could not see, which is
why the full suites per branch were worth running.

**R21 — dead helper hidden by an allowlist.** The run slice flagged
`tuning.preset_matches` as unreachable: a one-line predicate nothing called,
whose only coverage was its own test, and whose entry in `tests/test_cluster_seams.py`
existed only to silence the check. Removed from the slice and from
`rebase/upstream-main` (function, its three assertions, and the allowlist entry).
Deleting it is the clean answer the seam test itself suggests.
