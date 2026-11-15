lich # Remaining plan — .11-only, to the finish (2026-11-14)

Scope: everything that can be done on `.11` alone. **Excluded on purpose**
(separate, cluster work): cleaning up the nodes, rolling the latest version out to
`.12/.13/.14`, bringing the cluster up, the cluster test, and all live-cluster
evidence (2nd MTP model sweep, auto-free-disk live, 3/4-node live). The Qwen3.8
**PP** `rank 1` TypeError also needs a cluster and stays separate.

## Goal recap
A DAU-friendly, PP-only cluster experience inside oMLX (no second app/wizard),
elected from real measurements, and — when a model fits one Mac — a single-node
run on that Mac (E6). The PP core (measure → elect → plan → apply → start → ready
→ tune → stop/restart, resource policy, tuning presets, tradeoff UX) is done and
live-proven on `GLM-oQ2e`. What remains is model coverage (VLM), a usable
single-node run, and productization.

## Workstreams (parallelizable, file-disjoint)

| W | Point → outcome | Owner files | Depends on |
|---|---|---|---|
| **A** | Qwen3.8 on `.11`: force the engine load to be **local-only** (it currently auto-distributes across paired Macs), reconcile the `_single_node_fit` (84 GiB with PLE offload) vs the engine's per-rank refusal, diagnose the LLM fallback (`3748 params not in model`); add run-local **stop/unload** + **readiness canary** | `omlx/admin/routes.py`, `omlx/engine_pool.py`, `omlx/patches/**`, `omlx/cluster/assistant/{routes,start}.py`, `tests/cluster_assistant/**` | — |
| **B** | P2 productization audit: wired-limit helper (SMJobBless), first-run runtime installer, DMG/distribution — do what works without signing, document the signing/network-gated steps | `apps/omlx-mac/**`, `docs/…/{HELPER,DISTRIBUTION,SIGNING,SLIM-RUNTIME}.md` | — |
| **C** | run-local UI: Stop/Unload button, readiness display, endpoint/Open-Chat gating, tests; keep the tradeoff A/B | `apps/omlx-web/**` | A's stop route (contract fixed) |
| **Me** | Integration, full suite, docs reconcile (AGENTS/PHASE) | `AGENTS.md`, `docs/PHASE-2B-RUN.md` | A/B/C |
| **W5 (optional)** | glm5 split-K built for real: install the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`), `make kernels`, run the test for real | kernel build | — |

## Fixed contract A ↔ C
`POST /admin/api/cluster/assistant/run-local/{job_id}/stop` → `{job_id, stopped}`;
unloads this Mac's model for that run (or aborts it), idempotent. `ready` future
requires a readiness canary (one tiny completion; no rate claim).

## Findings feeding this plan
- **glm5 split-K**: kernel bitwise-correct; the test only fails without the native
  `dsa_topk_indices` extension (needs the Metal Toolchain). Skip-guard applied
  (`0470861d`); a real run is W5.
- **run-local** (E6) is built and fake-tested; its `ready` currently means
  "engine pool reports loaded", not "serves" — A adds the canary.
- **Qwen3.8** is a VLM; its single-node `/load` on `.11` went down the engine's
  distributed path and failed on memory, then the text fallback rejected VLM
  tensors. That is W-A, the top model-coverage blocker.

## Done when (for this plan)
- Qwen3.8 either loads single-node on `.11` (PLE offload, canary passes) or W-A
  ends with a precise, actionable diagnosis and the local-only/stop/canary code
  landed and tested — never a silent distributed or smaller-context fallback.
- run-local has stop/unload + canary + UI, all fake-tested; the tradeoff A/B and
  the tuning preset replay still work.
- P2 has a written status matrix; everything doable without signing is done.
- `tests/` is green (glm5 skipped; the scheduler isolation fix holds).
---

## The full arc to "done" (the end goal, recorded 2026-11-14)

The end goal is broader than the workstreams above. Engineering is **complete**
only when all of the following hold; keep this current so we never lose the last
two steps.

### Phase G — Engineering complete (zero gaps/bugs/todos)
1. **No open gaps, bugs or TODOs.** The inventory below is driven to zero, or
   each survivor is **explicitly decided out of scope** with a reason (never
   silently ignored).
2. **Everything tested locally on `.11`** (fake-driven where a real model/engine
   is not needed; real `.11` runs where it is). The full `tests/` suite is green.
3. **In principle ready to build the DMG** (`apps/omlx-mac/Scripts/build.sh` +
   `package_dmg.sh`; slim DMG, no embedded Python/MLX). "Ready" = code complete
   and tested; signing/notarization is a release step, not a code gap.
4. Cluster-only items stay **separate validation work**: node cleanup, rollout of
   the latest build to the nodes, cluster up/test, live-cluster evidence, and the
   Qwen3.8 PP `rank 1`. They must not be *code* gaps; a real code bug among them
   is fixed or filed as an oMLX issue.

### Phase H — Rebase onto upstream + cherry-pick
1. Move the pinned base (AGENTS §10: `1e27576a`, tag
   `cluster-assistant-base-2026-11-07`) to a newer upstream commit.
2. Reapply / cherry-pick our commits so the Cluster Assistant, the tuning/preset
   stack, run-local, the resource policy, the tradeoff options and every
   improvement still build and run; fix upstream drift.
3. Re-run the `.11` suite; update `docs/cluster-assistant/UPSTREAM.md` (clean
   upstream candidates, fork-only bits, conflicts).

### Phase I — Curated upstream offering (discussion -> issues -> PRs)
1. Split our diff into **independent packages, each with standalone value** where
   possible (e.g. a kernel/native-build fix, the tuning module, run-local, the
   resource-policy parity, docs). A package should be mergeable and useful on its
   own; document the dependency order where it is not.
2. Offer them to upstream cheapest-first: **discussion -> issues -> PRs**, starting
   with the independent, low-risk, broadly useful ones.
3. Track per-package status (draft / issue / PR / merged / fork-only) in
   `UPSTREAM.md`.

### Gap / bug / TODO inventory to drive to zero (Phase G)
Every row ends as **fixed**, **decided out of scope (reason)**, or **separate
cluster validation**.
- [x] Qwen3.8 single-node on `.11` (LIVE-VERIFIED 2026-11-14: loaded + served at
      40.3 tok/s, TTFT 0.34 s with PLE SSD offload): distributed-load-for-single-node, fit
      discrepancy, LLM fallback `3748 params not in model` (W-A). FIXED in code
      2026-11-14 (local_only load, `_single_node_fit` resident bytes, no false
      LLM fallback) + tested; the live `.11` run-local load is being verified.
- [ ] Qwen3.8 PP `rank 1` TypeError (`list indices ... NoneType`) — **still open,
      needs a cluster**: reproduce on the next 2-Mac run, then fix it or file it
      as an oMLX issue. The cluster test that covers it is scheduled
      (`READY-FOR-CLUSTER-TEST.md` §5 item 4, `CLUSTER-TEST-RUNBOOK.md` §3).
- [x] run-local stop/unload + readiness canary + UI (W-A/W-C), 2026-11-14:
      `POST …/run-local/{job_id}/stop`, one canary completion before ready, card
      Stop button; fake-tested.
- [x] Single-node tuning (2026-11-14): `POST /assistant/tune-local` runs the
      sampler + MTP on/off/depth locally, leaves the winner loaded and saves the
      (model, {this Mac}) preset; 15 backend + 2 UI tests. Original note: `/tune` and `/tune-redeploy` are cluster-only, so a
      ready `run-local` cannot be tuned (its preset is empty,
      `tuning_preset_skipped`). Add a local tune path (sampler + MTP on/off +
      depth via unload/load `local_only`) that saves the preset for
      `(model, hardware_key([this Mac]))`, so run-local replays it next time.
- [x] Stale-deployment cleanup — decided 2026-11-14: **no cross-coordinator
      auto-unload**. `local_only` already stops a model load from adopting a
      stale cluster deployment (the real bug). Auto-unloading another origin's or
      a user's serving deployment is unsafe, so the supported path is the manual
      unload (`/deployments/{id}/unload`, Screen 6) plus the `diagnostics` action
      on a memory-pressure start; the cluster stays the authority on its own
      registry.
- [~] Peer build parity: **in progress (2026-11-15)** — all three peers were
      reset (processes + cluster state, `CLUSTER-TEST-RUNBOOK.md` §0.5, the
      §1.5 rollout step), the **notarized `0.8.0rc1` DMG** was rolled out and
      `.12` already reports `version: 0.8.0rc1` on
      `GET /api/cluster/node_id` (`.11` does too), and the app is being started
      **interactively** at each node — the one-time helper admin dialog and the
      Local Network prompt are clicks at that Mac, never an SSH launch
      (`AGENTS.md` §11, runbook §1.5 step 4). Still open: `.13`/`.14` must answer
      the same version before the cluster test (`scripts/check_peer_versions.py`
      exit 0). Operational step, not a code gap.
- [x] TODO/FIXME/`NotImplemented` sweep (2026-11-14): **0** in
      `omlx/cluster/assistant/**` and `cluster_v2.js`. (Upstream oMLX's own TODOs
      are upstream's, not this fork's surface.)
- [x] glm5 split-K: not a bug — the kernel is bitwise-correct; the test only
      fails without the native `dsa_topk_indices` extension (needs the Metal
      Toolchain). Skip-guard added (`0470861d`). W5 (build for real) is optional.
- [~] Docs reconcile: **AGENTS, UPSTREAM.md, ROADMAP-TO-CHAPTER-10 and
      CLUSTER-TEST-RUNBOOK now agree with the tree** (AGENTS §2 decisions incl.
      Qwen4-Exp PP; new AGENTS §11 environment rules `3e2f5883`; runbook §1.5
      same-build rollout `e74ef88e`). **Still missing (two one-line fixes, not in
      this plan's files):** UPSTREAM.md lines 30–32 still say the
      `qwen4_exp_mlx_lm` adapter and the Qwen4 cluster/pipeline tests were “not
      replayed … no Qwen4-Exp pipeline on this branch”, although `054f92e4`
      applied #5033 (`omlx/patches/qwen4_exp_mlx_lm/model.py:46
      SUPPORTS_PIPELINE = True`, `tests/test_qwen4_pipeline.py`,
      `tests/test_qwen4_cluster_adapter.py`); and DISTRIBUTION.md does not yet
      record the notarized `0.8.0rc1` artifact (its table still stops at the
      2026-11-14 audit DMG).
- [x] Xcode host-dependent test failures fixed 2026-11-14 (macOS 27 narrow
      no-break space before %, and the under-page/window color assumption).
- Separate cluster validation (not code gaps): node cleanup, version rollout,
  cluster up/test, 2nd MTP model sweep, auto-free-disk live, 3/4-node live.

---

## Status 2026-11-15

Green tree, notarized app, peers rolling out. Numbers below are recorded
evidence, not re-assertions: `READY-FOR-CLUSTER-TEST.md` §2/§4,
`REBASE-2026-11-14.md` §“Final verification”, `UPSTREAM-PR-REVIEW-2026-11-15.md`
§13/§15.

### Suites

| Tree | Full suite (`tests/` + `apps/omlx-web/tests`) |
|---|---|
| `rebase/upstream-main` (this tree) | **20 981 passed, 0 failed** |
| `fix/local-only-load` (F1) | **20 154 passed, 0 failed** |
| `fix/test-infra-conftest` (F2) | **20 147 passed, 0 failed** |
| `fix/cluster-discovery-inventory` (F5) | 19 640 passed; the 2 remaining split-K tests need the native kernels a worktree does not build (§0.4) — 98/98 green with them |
| `feat/cluster-assistant-plan` (A1) | **20 555 passed, 0 failed** |
| `feat/cluster-assistant-run` (A2) | **20 933 passed, 0 failed** |
| `feat/cluster-assistant-run-local` (A3) | **20 976 passed, 0 failed** |
| `fix/macos-theme-locale-xctest` (macOS fix) | **401 Xcode tests, 0 failures** |
| `feat/macos-helper-dmg` (feature) | 410 Xcode tests, 7 failures in the 3 tests the fixes PR repairs — the known **R18** dependency |

`make web` (`build_css.py` + `normalize_i18n.py`) produces no diff on the three
assistant trees, as `docs/CONTRIBUTING.md` requires.

### App

`build/oMLX-0.8.0rc1.dmg` (82 630 736 bytes): `xcrun stapler validate` →
“worked”, `spctl -a -t open --context context:primary-signature` → **accepted,
`Notarized Developer ID`** (`Andreas Kolb, YB894QRG4W`), `hdiutil verify` →
VALID. The notarization blocker is fixed (`da8e30af`, R22).

### R19–R22 (`UPSTREAM-PR-REVIEW-2026-11-15.md` §13/§15)

- **R19** — the nested assistant `conftest.py` shadowed `tests/conftest.py`
  (whole-directory runs collected the wrong fixtures); fixed, focused runs had
  not seen it.
- **R20** — the stylesheet was stale in the run slices; `make web` is clean now.
- **R21** — dead `preset_matches` helper removed (`6eaf3f75`).
- **R22** — the packaged app did not sign its nested Mach-O kernels, so Apple
  rejected the build; fixed (`da8e30af`) and the `0.8.0rc1` image is notarized.

### Live cluster state

`.11` and `.12` answer `version: 0.8.0rc1` on `GET /api/cluster/node_id`;
`.13`/`.14` had not answered when this was written, so the same-build gate
(`CLUSTER-TEST-RUNBOOK.md` §1.5) is still open — that is the remaining item of
the peer-parity row above, and the precondition for the Qwen3.8 PP `rank 1`
cluster test.
