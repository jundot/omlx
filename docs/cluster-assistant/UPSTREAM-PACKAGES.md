# Upstream packages — how to offer this fork back (Phase I)

This is the **prep** for offering our work to upstream. It is a plan, not a
promise: after the rebase (Phase H) the diffs change, so sizes are from
`git diff --numstat cluster-assistant-base-2026-11-07..HEAD` (2026-11-14) and
every package is re-cut on the rebased branch.

Goal: **independent packages, each with standalone value**, offered cheapest-first
(**discussion → issues → PRs**). A package is worth opening alone only if it
helps an oMLX user who does not want the Cluster Assistant.

## The packages (by independence and risk)

| # | Package | Scope (rough) | Size | Independent? | Why it has standalone value |
|---|---|---|---|---|
| P1 | **`local_only` model load** | `omlx/engine_pool.py`, `omlx/admin/routes.py`, `tests/test_engine_pool.py`, `test_admin_load_errors.py`, `test_cluster_engine_pool.py` | ~245 + tests | **Yes** | `POST /api/models/{id}/load?local_only=true` loads a model on this Mac alone and never adopts a stale cluster deployment. A generic fix for anyone who wants a single-Mac load. |
| P2 | **Test-infra fixes** | `tests/conftest.py`, `tests/_tokenizer_support.py`, `tests/test_glm5_next_mtp.py` (skip), scheduler/settings tests | small | **Yes** | The `conftest` name-collision fix (`MockTokenizer` → uniquely named module) and the native-indexer skip make the suite stable; no product behaviour. |
| P3 | **Peer version check** | `scripts/check_peer_versions.py` | ~140 | **Yes** | A read-only tool that detects a mixed-version cluster (missing assistant routes on a peer). Useful even without the assistant. |
| P4 | **macOS helper / DMG fixes** | `apps/omlx-mac/Helper/**`, `Scripts/{build,package_dmg}.sh`, `PerformanceScreenVM.swift`, `PythonRuntime.swift`, mac tests, `docs/{HELPER,DISTRIBUTION,SIGNING,SLIM-RUNTIME}.md` | ~1736 | **Yes** | The SMJobBless helper could never install (launchd plist not embedded, trust requirements wrong) and the DMG was unsigned. Broadly useful to the app, independent of clustering. |
| P5 | **Cluster planner / resource-policy parity** | `omlx/cluster/planner.py`, election/admission/offload bits (non-assistant) + tests | ~632 + tests | Mostly | Measured admission, memory tier, table/Engram/PLE-aware planning. Useful to the cluster core; the assistant consumes it but does not own it. |
| P6 | **Cluster Assistant** | `omlx/cluster/assistant/**`, `apps/omlx-web` cluster UI (`cluster_v2.js`, `_cluster_v2.html`), `omlx/server.py`/`settings.py`, `tests/cluster_assistant/**`, `docs/cluster-assistant/**` | very large (~45k) | No (feature) | The product feature. Needs a **discussion + design issue first**, then a staged PR series (measurement → election/plan → apply/start → tuning/run-local → UI). Depends on P1 (local_only) and P5. |

Notes:
- **P1, P2, P3 are the easy first offers**: small, self-contained, low-risk.
- **P4** is app-maintainer territory; a clean bug-fix PR.
- **P5** is a real core improvement but may need a maintainer decision on the
  resource-policy model before a PR.
- **P6** is the reason for the fork. Open it as a **discussion** first ("run a
  model that does not fit one Mac, 2–4 Macs over Thunderbolt, in the existing
  app"), then an issue with the design (measurement, election, PP-only, plan/
  apply/start, tuning presets, run-local), then PRs in that order. Do not open a
  single 45k-line PR.

## Offering order (discussion → issues → PRs)

1. **Discuss** the broad idea (a short issue): cluster setup inside oMLX for
   models that do not fit one Mac, plus the standalone fixes we found along the
   way.
2. **Issues** for the independent packages P1–P4 (small, easy to accept).
3. **PRs** P1 → P2 → P3 → P4 (each references its issue; each merges alone).
4. **Issue + design** for P5 and P6; PR P5, then P6 in stages.
5. Track status per package below (draft / issue # / PR # / merged / fork-only).

## Status (fill in as we go)

| # | Status | Upstream issue | Upstream PR | Notes |
|---|---|---|---|---|
| P1 | draft | — | — | re-cut after rebase |
| P2 | draft | — | — | |
| P3 | draft | — | — | |
| P4 | draft | — | — | app maintainers |
| P5 | discuss | — | — | needs a model decision |
| P6 | discuss | — | — | staged series |

## Hard prerequisites

- **Phase H rebase first.** Cutting PRs off the old pin creates conflicts and
  mixes unrelated drift. Rebase to a recent upstream, re-run the `.11` suite,
  then re-cut these packages.
- Each package must build and pass its own tests **on the rebased tree**, alone.
- Never include fork-only product decisions as if they were upstream defaults;
  call them out in the issue and let maintainers decide.